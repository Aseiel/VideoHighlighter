"""music_analysis.py — turn a music file into a beat grid so cuts can land on it.

``modules/music_track.py`` can already lay a music bed under a finished reel,
but nothing about the cuts knows the music is there: clip boundaries fall
wherever the selector put them, and the result reads as a slideshow that
happens to have a soundtrack. The single thing that makes an edit feel
deliberate rather than arbitrary is the coincidence of picture change and
beat. This module produces the timing that buys it — where the beats are,
which of them start a bar, and where the music changes gear — and leaves the
choosing to the cutter.

What this is *not*: it is not a music recogniser and holds no opinion about the
material. It measures rhythm and energy, and everything it returns is a
timestamp or a level.

Two backends, one contract
==========================
The numpy path is the real implementation and the one that always exists:
ffmpeg (or the stdlib ``wave`` module) to decode, ``np.fft`` for the onset
envelope, autocorrelation for tempo, and the Ellis (2007) dynamic-programming
tracker for the beats. ``backend="librosa"`` swaps those stages for librosa's
when it is installed. Everything downstream — downbeats, sections, snapping,
serialisation — is shared, so both backends return the same shape of answer and
no caller ever has to branch on which one ran. librosa is never required; when
it is asked for and missing, the run degrades to numpy with a log line rather
than failing.

Why each stage is built the way it is
=====================================
**Decode.** A ``.wav`` that the stdlib ``wave`` module can already read is read
directly, skipping ffmpeg entirely — that is a process spawn and a full
re-encode saved on the format most music beds arrive in, and it keeps the
module usable where ffmpeg is missing.

**Onset envelope.** Spectral flux (frame-to-frame *increase* in log magnitude,
summed over bins) rather than raw loudness, because a beat is an attack, not a
level: a sustained chord is loud and is not an onset, while a snare inside a
loud mix is quiet in absolute terms and is unmistakable in flux.

**Tempo.** Autocorrelation of the onset envelope finds the lag at which the
music repeats, but that lag is genuinely ambiguous — half and double a tempo
correlate nearly as well as the tempo itself, which is the classic way a beat
grid ends up at 75 BPM on a 150 BPM track. A log-normal prior centred on
120 BPM leans the way a listener does, toward the reading closest to a
comfortable walking pace, and that is all it can do: two candidates an octave
apart are exactly equidistant from the centre when their geometric mean is
120 BPM — at 170 BPM — and above that the prior prefers the half outright.
Nor can the correlation itself overrule it there, because the margin between a
period and its double is a percent or two while a period that is not a whole
number of 23 ms frames measures several percent low. So the octave is settled
last, and by evidence rather than by preference: if the attack halfway between
two candidate beats is as strong as the beats themselves, it is a beat too and
the candidate spans two of them.

**Beats.** Peak-picking the envelope directly gives beats that wander: it takes
every loud syllable it sees and skips every beat the drummer implied but did
not play. The Ellis dynamic program instead scores a whole sequence at once —
onset strength where a beat lands, plus a penalty that grows with the log of
how far the gap between consecutive beats deviates from the estimated period —
so it rides through a fill or a soft bar and stays phase-locked.

**Sections.** Terciles of smoothed RMS, so the tiers are relative to *this*
track and mean the same thing on quiet and loud masters. Runs shorter than a
few seconds are merged away because a two-second "section" is a fill, and a
cutter that reacts to it produces exactly the twitchiness this is meant to
avoid.

Public API
==========

    analyze_music(path, backend="auto", meter=4, log_fn=print) -> MusicAnalysis
    snap_to_beat(t, analysis, mode="nearest") -> float
    snap_segments(segments, analysis, mode="nearest", min_duration=0.0) -> list
    beat_aligned_durations(analysis, bars=1) -> float
    save_analysis(analysis, path) -> str
    load_analysis(path) -> MusicAnalysis
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from modules.app_paths import ffmpeg_exe

# Analysis rate. 22050 Hz keeps everything below 11 kHz, which is all a
# percussive attack needs, and halves the FFT work against 44.1 kHz.
TARGET_SR = 22050

# 2048 samples (93 ms) resolves a kick from the bass note under it; 512 (23 ms)
# is finer than the ~40 ms at which two attacks stop being separately audible,
# so the grid is not the limiting factor in how tight a cut can be.
N_FFT = 2048
HOP = 512

# Tempo search range. Below 50 BPM the "beat" is slower than a cut wants to be;
# above 200 the tracker is picking subdivisions, not beats.
MIN_BPM = 50.0
MAX_BPM = 200.0

# The prior that leans the half/double ambiguity. 120 BPM is the centre of the
# range people tap to; one octave of width keeps it a lean, not a rule, so a
# genuinely slow or fast track still wins on evidence.
PRIOR_BPM = 120.0
PRIOR_OCTAVE_WIDTH = 1.0

# How strong an attack halfway between two candidate beats has to be, as a
# fraction of the attack on the beats, before it counts as a beat itself and
# the candidate period is halved. Attacks of equal weight measure 0.95 to 1.0
# here and a subdivision played at 70% of the beat measures about 0.80, so the
# threshold sits in the middle of a wide gap rather than on a cliff — which is
# what lets it override the prior without stealing the cases the prior exists
# for. Raising it toward 1.0 re-admits the half-tempo readings above 170 BPM;
# lowering it starts calling an ordinary hi-hat pattern the beat.
OCTAVE_EVIDENCE = 0.9

# Ellis' transition weight. Higher means the tracker defends the estimated
# period harder against a tempting onset off the grid; 100 is his published
# value and is balanced against an onset envelope scaled by its own deviation,
# which is why the normalisation in _local_score must not change alone.
TIGHTNESS = 100.0

# Section analysis: one RMS value per second, smoothed over three of them.
SECTION_WINDOW = 1.0
SECTION_SMOOTH = 3
MIN_SECTION = 4.0

ENERGY_TIERS = ("low", "mid", "high")

# Bumped when the JSON layout changes incompatibly, so a stale sidecar is
# rejected loudly instead of loading with fields silently missing.
SCHEMA_VERSION = 1

SNAP_MODES = ("nearest", "previous", "next")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class Section:
    """A contiguous stretch of one energy tier.

    ``energy`` is mean RMS normalised against the loudest window in the same
    track, so it is comparable within a track and deliberately not across
    tracks — absolute level is a mastering decision, not a musical one.
    ``label`` names the tier only; it describes where the level sits, never
    what is playing.
    """

    start: float
    end: float
    energy: float
    label: str

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class MusicAnalysis:
    """Everything the cutter needs to place a boundary musically.

    ``beats`` and ``downbeats`` are absolute seconds from the start of the
    music file, ``downbeats`` being a subset of ``beats``. ``bpm`` and
    ``beat_interval`` are exact reciprocals of each other; both are carried
    because callers want each and rounding them apart is how a grid drifts.

    A file with no detectable rhythm (silence, a few hundred samples, a spoken
    intro) returns ``bpm=0.0`` and empty beat lists rather than a plausible
    guess. Every snapping helper treats that as "no information here" and
    leaves times alone, so a caller that does not check still behaves
    correctly.
    """

    path: str
    duration: float
    sample_rate: int
    bpm: float
    beats: list[float]
    downbeats: list[float]
    beat_interval: float
    meter: int
    onset_envelope: list[float]
    onset_times: list[float]
    sections: list[Section]
    backend: str

    @property
    def has_beats(self) -> bool:
        # Two, not one: a single beat is a point, not a grid. Snapping against
        # it sends every cut in the reel to the same instant, which is a worse
        # answer than the no-rhythm path that leaves the times alone — so the
        # flag a caller gates on has to refuse it.
        return len(self.beats) >= 2 and self.beat_interval > 0.0


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------
def _read_pcm_wav(path: str) -> Optional[tuple]:
    """Mono float samples and the native rate from a PCM wav, or None.

    None means "not something the stdlib can read" — a compressed payload, a
    float format, an exotic sample width — and is the caller's signal to spend
    an ffmpeg process instead. It is never an error: the fallback is complete.
    """
    try:
        with wave.open(path, "rb") as wf:
            channels = max(1, wf.getnchannels())
            width = wf.getsampwidth()
            rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
    except (wave.Error, EOFError, OSError, ValueError):
        return None

    if rate <= 0:
        return None

    if width == 1:
        # 8-bit wav is unsigned by definition; the others are signed.
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        data = (data - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8)
        usable = (packed.size // 3) * 3
        triples = packed[:usable].reshape(-1, 3).astype(np.int32)
        data = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
        # Sign-extend the 24-bit value into int32 before scaling.
        data = np.where(data & 0x800000, data - (1 << 24), data)
        data = data.astype(np.float32) / float(1 << 23)
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / float(1 << 31)
    else:
        return None

    if channels > 1:
        usable = (data.size // channels) * channels
        data = data[:usable].reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), int(rate)


def _decode_with_ffmpeg(path: str, log_fn=print) -> tuple:
    """Decode anything ffmpeg understands to mono float samples at TARGET_SR.

    Raises RuntimeError when ffmpeg fails or writes something unreadable —
    silently returning an empty envelope would hand the cutter a grid of no
    beats and look like a track with no rhythm, which is a wrong answer rather
    than a missing one.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        proc = subprocess.run(
            [ffmpeg_exe(), "-y", "-i", str(path), "-vn",
             "-acodec", "pcm_s16le", "-ar", str(TARGET_SR), "-ac", "1",
             "-hide_banner", "-loglevel", "error", tmp],
            capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()[-6:]
            raise RuntimeError("ffmpeg could not decode the music file:\n"
                               + "\n".join(tail))
        decoded = _read_pcm_wav(tmp)
        if decoded is None:
            raise RuntimeError(f"ffmpeg produced an unreadable wav for {path!r}")
        log_fn(f"🎵 Decoded with ffmpeg at {TARGET_SR} Hz mono")
        return decoded
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _resample(y: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear resample, with a boxcar pre-filter when downsampling.

    Without the pre-filter, hi-hat and cymbal energy above the new Nyquist
    folds back down and lands in the onset envelope as flux no instrument
    played — spurious onsets that the tempo autocorrelation then treats as
    evidence. The kernel is a boxcar twice the decimation factor long, which
    is not an arbitrary smooth: a boxcar of length N nulls at multiples of
    ``src_rate / N``, so that length puts its first null exactly on the new
    Nyquist. Rejection above it is a sinc rolloff rather than a wall, and the
    passband droops about a percent — both far cheaper than taking on a
    resampler dependency for a signal that is about to be reduced to one
    number per 23 ms anyway.

    The filter is skipped for a signal shorter than its own kernel, because
    ``np.convolve(..., mode="same")`` returns ``max(len(y), taps)`` samples and
    would hand back more audio than it was given — a 1-sample 44.1 kHz file
    then resamples to 2 samples and reports four times its real duration.
    Nothing is lost by skipping it: fewer samples than taps is under a
    millisecond of audio, which resamples to at most one sample, and one sample
    has no band for an alias to land in.
    """
    if y.size == 0 or src_rate == dst_rate:
        return y
    ratio = float(dst_rate) / float(src_rate)
    if ratio < 1.0:
        taps = int(round(2.0 / ratio))
        if 1 < taps <= y.size:
            kernel = np.ones(taps, dtype=np.float32) / float(taps)
            y = np.convolve(y, kernel, mode="same").astype(np.float32)
    n_out = int(round(y.size * ratio))
    if n_out < 1:
        return np.zeros(0, dtype=np.float32)
    positions = np.arange(n_out, dtype=np.float64) / ratio
    return np.interp(positions, np.arange(y.size, dtype=np.float64),
                     y.astype(np.float64)).astype(np.float32)


def _load_audio(path: str, log_fn=print) -> tuple:
    """Mono samples at TARGET_SR, by whichever route is cheapest for this file."""
    decoded = None
    if os.path.splitext(str(path))[1].lower() == ".wav":
        decoded = _read_pcm_wav(str(path))
    if decoded is None:
        decoded = _decode_with_ffmpeg(str(path), log_fn=log_fn)
    y, native = decoded
    if native != TARGET_SR:
        y = _resample(y, native, TARGET_SR)
    return y, TARGET_SR


# ---------------------------------------------------------------------------
# Onset envelope
# ---------------------------------------------------------------------------
def _onset_envelope(y: np.ndarray, sr: int) -> tuple:
    """Spectral-flux onset envelope, normalised to a peak of 1, and its times.

    Frames are centred: the signal is padded by half a window so frame ``i``
    sits at ``i * HOP / sr``, which keeps the returned times comparable with
    the sample clock instead of running half a window late.

    The STFT runs in blocks with the previous block's last frame carried over.
    A five-minute track is ~13,000 frames of 1025 bins, and materialising that
    as one complex array costs a couple of hundred megabytes for no reason —
    flux only ever looks one frame back.
    """
    if y.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float64)

    pad = N_FFT // 2
    padded = np.pad(y.astype(np.float32), pad, mode="constant")
    if padded.size < N_FFT:
        padded = np.pad(padded, (0, N_FFT - padded.size), mode="constant")

    n_frames = 1 + (padded.size - N_FFT) // HOP
    window = np.hanning(N_FFT).astype(np.float32)
    frames_view = np.lib.stride_tricks.sliding_window_view(padded, N_FFT)

    flux = np.zeros(n_frames, dtype=np.float32)
    previous = None
    block = 256
    for start in range(0, n_frames, block):
        stop = min(start + block, n_frames)
        chunk = frames_view[start * HOP:stop * HOP:HOP]
        # log1p compresses the magnitude so a fortissimo attack and a quiet one
        # contribute on the same scale; a raw magnitude difference would let the
        # loudest bar own the whole envelope.
        mags = np.log1p(np.abs(np.fft.rfft(chunk * window, axis=1))).astype(np.float32)
        prepend = mags[:1] if previous is None else previous[None, :]
        diff = np.diff(mags, axis=0, prepend=prepend)
        flux[start:stop] = np.maximum(diff, 0.0).sum(axis=1)
        previous = mags[-1]

    peak = float(flux.max()) if flux.size else 0.0
    if peak > 0:
        flux /= peak
    times = np.arange(n_frames, dtype=np.float64) * HOP / float(sr)
    return flux, times


# ---------------------------------------------------------------------------
# Tempo
# ---------------------------------------------------------------------------
def _grid_peaks(padded: np.ndarray, size: int, offset: float,
                period: float) -> np.ndarray:
    """Attack strength at ``offset + k * period`` frames, for every k in range.

    Read through a one-frame maximum, because an attack lands somewhere inside
    a 23 ms frame and sampling a single index would make the answer depend on
    where the frame boundaries happen to fall — the same sub-frame accident
    that stops the autocorrelation from settling the octave on its own.
    ``padded`` is ``env`` with one zero frame each side, so that window is a
    plain slice at both ends instead of three bounds checks.
    """
    if period <= 0 or offset < 0.0 or offset > size - 1:
        return np.zeros(0, dtype=np.float64)
    count = int(math.floor((size - 1 - offset) / period)) + 1
    index = np.round(offset + np.arange(count, dtype=np.float64) * period)
    index = index.astype(np.int64)
    index = index[(index >= 0) & (index < size)]
    if index.size == 0:
        return np.zeros(0, dtype=np.float64)
    return np.maximum(np.maximum(padded[index], padded[index + 1]),
                      padded[index + 2])


def _interleaved_strength(env: np.ndarray, period: float) -> float:
    """Attack strength halfway between the beats, relative to the beats.

    The one measurement that separates a period from its double. Near 1.0 the
    midpoints carry the same attack the beats do, which is what a track whose
    beat is twice as fast looks like from the wrong octave; near 0.0 nothing
    happens between the beats and the period is already right.

    Each midpoint is measured against the *weaker* of the two beats it sits
    between, not against the average beat, and that is the whole reason this is
    usable on music rather than on click tracks. An accent pattern rides on top
    of the beat — a bar of 4/4 makes every fourth beat louder — so at twice the
    real period one candidate grid swallows all the accents and averages higher
    than the midpoints purely because of them. Pairing against the weaker
    neighbour removes that: an accent can only ever make a beat stronger, so
    the floor of a pair is what an unaccented beat looks like, which is exactly
    what a midpoint has to match to be a beat itself.

    The phase is searched rather than assumed, because at this stage the beats
    are only a period and not yet a grid.
    """
    step = int(round(period))
    if env.size < 4 or step < 2:
        return 0.0

    padded = np.concatenate(([0.0], env.astype(np.float64), [0.0]))
    on_beat = np.zeros(0, dtype=np.float64)
    phase, best = 0.0, 0.0
    for candidate in range(step):
        peaks = _grid_peaks(padded, env.size, float(candidate), period)
        value = float(peaks.mean()) if peaks.size else 0.0
        if value > best or on_beat.size == 0:
            on_beat, phase, best = peaks, float(candidate), value
    if on_beat.size < 2 or best <= 0.0:
        return 0.0

    midpoints = _grid_peaks(padded, env.size, phase + period / 2.0, period)
    pairs = min(on_beat.size - 1, midpoints.size)
    if pairs < 1:
        return 0.0
    floor = np.minimum(on_beat[:pairs], on_beat[1:pairs + 1])
    reference = float(floor.mean())
    if reference <= 0.0:
        return 0.0
    return float(midpoints[:pairs].mean()) / reference


def _faster_octave(env: np.ndarray, period: float, fps: float) -> float:
    """Halve ``period`` for as long as the midpoints between its beats are beats.

    Runs after the prior has chosen, because the prior is a preference and this
    is evidence. It only ever moves the answer one way — a period is halved,
    never doubled — since the ambiguity it exists to break is the one the prior
    gets wrong: above 170 BPM the prior favours the half of every tempo, so
    without this the top quarter of the search range is unreachable however
    plainly the beats are played.

    Halving is applied to the interpolated period rather than re-fitting the
    autocorrelation at the shorter lag, because halving a sub-frame estimate
    halves its error too — a fresh parabola would start again from a whole
    frame of quantisation.
    """
    shortest = fps * 60.0 / MAX_BPM
    while period / 2.0 >= shortest:
        if _interleaved_strength(env, period) < OCTAVE_EVIDENCE:
            break
        period /= 2.0
    return period


def _estimate_tempo(env: np.ndarray, sr: int) -> tuple:
    """(bpm, period_in_frames) from the onset envelope, or (0.0, 0.0).

    The mean is removed first: autocorrelation of a non-negative signal is
    dominated by its DC component, and every lag would score about the same.
    The peak is then refined by fitting a parabola through its two neighbours,
    because the lag grid is coarse where it hurts most — at 120 BPM one whole
    frame of lag is nearly 3 BPM, so the integer answer alone is never good
    enough to hold a grid together over a three-minute track.

    The octave the winning lag belongs to is decided afterwards and separately,
    by ``_faster_octave``; nothing before that point can be trusted with it.
    """
    fps = float(sr) / HOP
    if env.size < 4 or float(env.max()) <= 0.0:
        return 0.0, 0.0

    x = (env - env.mean()).astype(np.float64)
    n = 1 << int(math.ceil(math.log2(max(4, 2 * x.size))))
    spectrum = np.fft.rfft(x, n)
    acf = np.fft.irfft(spectrum * np.conj(spectrum), n)[:x.size]

    min_lag = max(1, int(math.floor(fps * 60.0 / MAX_BPM)))
    max_lag = min(x.size - 1, x.size // 2, int(math.ceil(fps * 60.0 / MIN_BPM)))
    if max_lag <= min_lag:
        return 0.0, 0.0

    # A raw autocorrelation slopes downward with lag — each one is summed over
    # fewer overlapping frames than the last — but capping the search at half
    # the envelope keeps that tilt under a couple of percent across a lag range
    # the prior spans a whole octave of. Correcting it was measured to change
    # the winning lag on nothing longer than two seconds, which is less than
    # two beats and has no tempo to find in the first place.
    lags = np.arange(min_lag, max_lag + 1, dtype=np.float64)
    prior_lag = fps * 60.0 / PRIOR_BPM
    prior = np.exp(-0.5 * (np.log2(lags / prior_lag) / PRIOR_OCTAVE_WIDTH) ** 2)
    scored = acf[min_lag:max_lag + 1] * prior
    if float(scored.max()) <= 0.0:
        return 0.0, 0.0

    best = int(np.argmax(scored))
    delta = 0.0
    if 0 < best < scored.size - 1:
        a, b, c = scored[best - 1], scored[best], scored[best + 1]
        denom = a - 2.0 * b + c
        if denom != 0:
            delta = float(np.clip(0.5 * (a - c) / denom, -0.5, 0.5))
    period = float(lags[best]) + delta
    if period <= 0:
        return 0.0, 0.0
    period = _faster_octave(env, period, fps)
    bpm = float(np.clip(60.0 * fps / period, MIN_BPM, MAX_BPM))
    return bpm, period


# ---------------------------------------------------------------------------
# Beats — Ellis (2007) dynamic programming
# ---------------------------------------------------------------------------
def _local_score(env: np.ndarray, period: float) -> np.ndarray:
    """Onset envelope scaled by its own deviation and smoothed at period/32.

    Scaling by the standard deviation is what makes TIGHTNESS a constant
    rather than a per-track tuning knob: without it the transition penalty
    would mean one thing on a sparse mix and another on a dense one.

    The smoothing is Ellis' and is kept for a less obvious reason than
    blurring. Its Gaussian is sub-frame at most tempi (sigma is 0.5 frames at
    160 BPM, 1.1 at 75) and on everything measured here — attacks from 1 ms to
    90 ms — removing it changed neither the beat count nor the interval
    jitter. What it does change is scale: the kernel is deliberately not
    normalised, and its gain runs from 1.3x at 160 BPM to 2.7x at 75, so the
    onset evidence carries more weight against the transition penalty on slow
    music than on fast. Deleting it as a no-op would silently retune TIGHTNESS
    by that factor, tempo-dependently.
    """
    if env.size == 0:
        return env.astype(np.float64)
    deviation = float(env.std()) if env.size > 1 else 0.0
    scaled = env.astype(np.float64) / (deviation if deviation > 0 else 1.0)
    if period <= 0:
        return scaled
    span = np.arange(-int(round(period)), int(round(period)) + 1, dtype=np.float64)
    if span.size < 3:
        return scaled
    kernel = np.exp(-0.5 * (span * 32.0 / period) ** 2)
    return np.convolve(scaled, kernel, mode="same")


def _track_beats(env: np.ndarray, period: float) -> np.ndarray:
    """Frame indices of the beats, from the cumulative-score DP plus backtrace.

    Every frame records the best predecessor within half to twice a period back
    and the running score of the sequence that ends there. Because the score is
    cumulative, the best sequence over the whole track is found once at the end
    rather than greedily frame by frame — that is precisely what lets the
    tracker keep the pulse through a bar nobody played on.
    """
    if env.size == 0 or period <= 0:
        return np.zeros(0, dtype=np.int64)

    local = _local_score(env, period)
    if local.size == 0 or float(np.max(local)) <= 0.0:
        return np.zeros(0, dtype=np.int64)

    lo = max(1, int(round(period / 2.0)))
    hi = max(lo + 1, int(round(2.0 * period)))
    offsets = np.arange(-hi, -lo + 1, dtype=np.int64)
    # Transition penalty, zero at exactly one period back and growing with the
    # square of the log ratio, so a half-period jump and a double-period jump
    # are punished equally. A linear penalty would make dropping beats cheaper
    # than inserting them and the grid would thin out over a long track.
    transition = -TIGHTNESS * (np.log(-offsets / period) ** 2)

    cumulative = np.zeros(local.size, dtype=np.float64)
    backlink = np.full(local.size, -1, dtype=np.int64)
    start_threshold = 0.01 * float(np.max(local))
    started = False

    for i in range(local.size):
        candidates = offsets + i
        valid = candidates >= 0
        if valid.any():
            scores = np.full(offsets.size, -np.inf)
            scores[valid] = cumulative[candidates[valid]] + transition[valid]
            best = int(np.argmax(scores))
            best_score = float(scores[best])
            best_index = int(candidates[best])
        else:
            best_score, best_index = 0.0, -1
        # Clamped at zero so starting a fresh chain is never worse than
        # continuing a bad one, and so cumulative stays non-negative — which
        # _last_beat's median threshold relies on to mean anything.
        cumulative[i] = local[i] + max(best_score, 0.0)
        if not started and local[i] < start_threshold:
            # Leading silence. Anchoring the chain here would lock the whole
            # grid to the noise floor before the music starts.
            backlink[i] = -1
        else:
            backlink[i] = best_index
            started = True

    tail = _last_beat(cumulative)
    if tail < 0:
        return np.zeros(0, dtype=np.int64)
    beats = []
    node = tail
    while node >= 0:
        beats.append(node)
        node = int(backlink[node])
    beats.reverse()
    return _trim_beats(local, np.asarray(beats, dtype=np.int64))


def _last_beat(cumulative: np.ndarray) -> int:
    """Where to start the backtrace: the last strong local maximum.

    Not simply the final frame — the tail of a track is a fade or a silence,
    and starting there hangs a phantom beat off the end of the grid.
    """
    if cumulative.size == 0:
        return -1
    if cumulative.size < 3:
        return int(np.argmax(cumulative))
    maxima = np.zeros(cumulative.size, dtype=bool)
    maxima[1:-1] = (cumulative[1:-1] > cumulative[:-2]) & (cumulative[1:-1] >= cumulative[2:])
    if not maxima.any():
        return int(np.argmax(cumulative))
    threshold = 0.5 * float(np.median(cumulative[maxima]))
    strong = np.flatnonzero(maxima & (cumulative >= threshold))
    if strong.size == 0:
        return int(np.flatnonzero(maxima)[-1])
    return int(strong[-1])


def _trim_beats(local: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """Drop leading and trailing beats that sit on no onset at all.

    The DP has to produce a beat every period, including across the silence
    before the first note and after the last. Only the ends are trimmed: a
    weak beat in the middle is a musical fact and removing it would leave a
    hole in the grid.
    """
    if beats.size == 0:
        return beats
    strength = local[beats]
    smoothed = np.convolve(strength, np.hanning(5), mode="same")
    threshold = 0.5 * float(np.sqrt(np.mean(smoothed ** 2)))
    keep = np.flatnonzero(smoothed > threshold)
    if keep.size == 0:
        return beats
    return beats[int(keep[0]):int(keep[-1]) + 1]


def _refine_bpm(beat_times: np.ndarray, fallback: float) -> float:
    """BPM from the tracked beats rather than from the autocorrelation peak.

    The lag grid is 23 ms wide; averaging the intervals the tracker actually
    produced averages that quantisation away and is roughly an order of
    magnitude more accurate. Intervals far from the median are excluded first,
    so one dropped beat in a quiet bar does not drag the answer toward half
    tempo.

    Under two beats there is no interval at all, and that is reported as no
    tempo rather than as the fallback. The autocorrelation always names some
    lag — it is an argmax over a range that excludes nothing — and hanging it
    off a grid of one beat produces the worst possible answer: a confident BPM,
    ``has_beats`` True, and a grid on which every cut in the reel snaps to the
    same instant. 0.0 collapses the whole analysis to the documented "no
    rhythm here" state, which every helper already handles by leaving times
    alone.
    """
    if beat_times.size < 2:
        return 0.0
    if beat_times.size < 4:
        return fallback
    gaps = np.diff(beat_times)
    gaps = gaps[gaps > 0]
    if gaps.size == 0:
        return fallback
    median = float(np.median(gaps))
    if median <= 0:
        return fallback
    consistent = gaps[np.abs(gaps - median) <= 0.1 * median]
    interval = float(consistent.mean()) if consistent.size else median
    if interval <= 0:
        return fallback
    return float(np.clip(60.0 / interval, MIN_BPM, MAX_BPM))


def _downbeat_phase(env: np.ndarray, beat_frames: np.ndarray, meter: int) -> int:
    """Which of the ``meter`` phases carries the most onset weight.

    The mean, not the sum: phase 0 always has as many or more beats than the
    others, and a sum would hand it the answer on any track short enough for
    that to matter.
    """
    if beat_frames.size == 0 or meter < 2:
        return 0
    strength = env[np.clip(beat_frames, 0, env.size - 1)]
    best_phase, best_value = 0, -np.inf
    for phase in range(meter):
        taken = strength[phase::meter]
        if taken.size == 0:
            continue
        value = float(taken.mean())
        if value > best_value:
            best_phase, best_value = phase, value
    return best_phase


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def _tier_labels(norm: np.ndarray) -> list:
    """Energy tier per window, by tercile of the track's own levels.

    A track whose levels are effectively constant has no terciles to speak of
    — the two boundaries collapse onto each other and every window would land
    in whichever tier the comparison happens to favour. That case falls back to
    fixed thirds of full scale, which puts silence in "low" and a wall of sound
    in "high" instead of both in the same arbitrary bucket.
    """
    if norm.size == 0:
        return []
    low, high = (float(v) for v in np.quantile(norm, [1.0 / 3.0, 2.0 / 3.0]))
    if high - low < 1e-6:
        low, high = 1.0 / 3.0, 2.0 / 3.0
    labels = []
    for value in norm:
        if value <= low:
            labels.append(ENERGY_TIERS[0])
        elif value <= high:
            labels.append(ENERGY_TIERS[1])
        else:
            labels.append(ENERGY_TIERS[2])
    return labels


def _sections(y: np.ndarray, sr: int, duration: float) -> list:
    """Energy sections over ~1 s windows, with short runs merged away.

    Nothing shorter than MIN_SECTION comes back unless the whole file is
    shorter than that, and the guarantee is in seconds of audio rather than in
    windows: the last window is zero-padded out to a full one, so a file whose
    length is not a whole number of seconds — which is every file that is not a
    click track — would otherwise get one window of credit for a few
    milliseconds of tail and end on a section under the minimum.
    """
    if y.size == 0 or duration <= 0:
        return []

    span = max(1, int(round(SECTION_WINDOW * sr)))
    count = max(1, int(math.ceil(y.size / span)))
    padded = np.pad(y.astype(np.float64), (0, count * span - y.size))
    rms = np.sqrt((padded.reshape(count, span) ** 2).mean(axis=1))
    if count >= SECTION_SMOOTH:
        kernel = np.ones(SECTION_SMOOTH) / float(SECTION_SMOOTH)
        rms = np.convolve(rms, kernel, mode="same")
    peak = float(rms.max())
    norm = rms / peak if peak > 0 else np.zeros_like(rms)

    labels = _tier_labels(norm)
    runs = []
    for index, label in enumerate(labels):
        if runs and runs[-1][2] == label:
            runs[-1][1] = index + 1
        else:
            runs.append([index, index + 1, label])

    window_seconds = span / float(sr)

    def run_seconds(run) -> float:
        """Real audio in a run: the padding in the final window is not music."""
        return min(run[1] * window_seconds, duration) - run[0] * window_seconds

    while len(runs) > 1:
        shortest = min(range(len(runs)), key=lambda i: run_seconds(runs[i]))
        if run_seconds(runs[shortest]) >= MIN_SECTION - 1e-9:
            break
        # Absorb into whichever neighbour is closer in energy, so a quiet fill
        # joins the quiet side of the boundary rather than the nearer one.
        before = runs[shortest - 1] if shortest > 0 else None
        after = runs[shortest + 1] if shortest + 1 < len(runs) else None
        mine = float(norm[runs[shortest][0]:runs[shortest][1]].mean())
        if before is None:
            target = shortest + 1
        elif after is None:
            target = shortest - 1
        else:
            gap_before = abs(mine - float(norm[before[0]:before[1]].mean()))
            gap_after = abs(mine - float(norm[after[0]:after[1]].mean()))
            target = shortest - 1 if gap_before <= gap_after else shortest + 1
        runs[target][0] = min(runs[target][0], runs[shortest][0])
        runs[target][1] = max(runs[target][1], runs[shortest][1])
        runs.pop(shortest)

    sections = []
    for start_index, stop_index, label in runs:
        start = start_index * window_seconds
        end = min(stop_index * window_seconds, duration)
        if end <= start:
            continue
        sections.append(Section(
            start=float(start),
            end=float(end),
            energy=float(np.clip(norm[start_index:stop_index].mean(), 0.0, 1.0)),
            label=label,
        ))
    if sections:
        sections[-1].end = float(max(sections[-1].end, duration))
    return sections


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
def _real_librosa():
    """The librosa module, but only when it is genuinely librosa.

    ``tests/conftest.py`` installs a ``MagicMock`` under the name so the suite
    runs without the 1.9 GB ML stack. A mock imports cleanly and then answers
    every call with another mock, which would sail through this module and
    produce an analysis made of nothing. A real module has a string
    ``__version__``; a mock's is a mock.
    """
    try:
        import librosa
    except Exception:
        return None
    if not isinstance(getattr(librosa, "__version__", None), str):
        return None
    return librosa


def _stage_numpy(path: str, log_fn) -> tuple:
    """Decode, onset envelope, tempo and beats — the always-available path."""
    y, sr = _load_audio(path, log_fn=log_fn)
    env, times = _onset_envelope(y, sr)
    bpm, period = _estimate_tempo(env, sr)
    beat_frames = _track_beats(env, period) if bpm > 0 else np.zeros(0, dtype=np.int64)
    return y, sr, env, times, beat_frames, bpm


def _stage_librosa(path: str, log_fn) -> tuple:
    """The same four stages through librosa, for callers that have it installed.

    Deliberately thin: downbeats, sections and everything the caller actually
    consumes are computed by the shared code below, so the two backends cannot
    drift apart in anything but the beat positions themselves.
    """
    librosa = _real_librosa()
    if librosa is None:
        raise ImportError("librosa is not installed")
    y, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
    y = np.ascontiguousarray(np.asarray(y, dtype=np.float32))
    env = np.asarray(
        librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP), dtype=np.float32)
    peak = float(env.max()) if env.size else 0.0
    if peak > 0:
        env = env / peak
    tempo, frames = librosa.beat.beat_track(
        onset_envelope=env, sr=sr, hop_length=HOP, units="frames")
    beat_frames = np.asarray(frames, dtype=np.int64).ravel()
    bpm = float(np.atleast_1d(np.asarray(tempo, dtype=np.float64))[0])
    times = np.arange(env.size, dtype=np.float64) * HOP / float(sr)
    return y, int(sr), env, times, beat_frames, bpm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def analyze_music(path: str, *, backend: str = "auto", meter: int = 4,
                  log_fn=print) -> MusicAnalysis:
    """Analyse ``path`` into a beat grid, downbeats and energy sections.

    ``backend`` is "auto" (librosa when genuinely importable, else numpy),
    "numpy", or "librosa". Asking for librosa without it installed, or a
    librosa stage that raises, logs and falls back to numpy — the analysis is
    the point, not which library produced it, and the ``backend`` field on the
    result records what actually ran.

    Raises ValueError for a missing file, an unknown backend or a meter below
    1, and RuntimeError when the audio cannot be decoded at all. A file that
    decodes but has no rhythm to find is not an error; see ``MusicAnalysis``.
    """
    if backend not in ("auto", "numpy", "librosa"):
        raise ValueError(f"unknown backend: {backend!r} (expected auto|numpy|librosa)")
    if not path or not os.path.exists(str(path)):
        raise ValueError(f"music file not found: {path!r}")
    meter = int(meter)
    if meter < 1:
        raise ValueError(f"meter must be at least 1, got {meter!r}")

    chosen = backend
    if chosen == "auto":
        chosen = "librosa" if _real_librosa() is not None else "numpy"

    staged = None
    if chosen == "librosa":
        try:
            staged = _stage_librosa(str(path), log_fn)
        except Exception as exc:  # ImportError, or anything librosa raises
            log_fn(f"⚠️ librosa backend unavailable ({exc}); using numpy")
            chosen = "numpy"
    if staged is None:
        chosen = "numpy"
        staged = _stage_numpy(str(path), log_fn)

    y, sr, env, times, beat_frames, bpm = staged
    duration = y.size / float(sr) if sr else 0.0

    beat_frames = beat_frames[(beat_frames >= 0) & (beat_frames < max(env.size, 1))]
    beat_times = beat_frames.astype(np.float64) * HOP / float(sr)
    bpm = _refine_bpm(beat_times, bpm) if beat_times.size else 0.0
    if bpm <= 0:
        beat_times = np.zeros(0, dtype=np.float64)
        beat_frames = np.zeros(0, dtype=np.int64)

    phase = _downbeat_phase(env, beat_frames, meter)
    downbeats = beat_times[phase::meter] if beat_times.size else np.zeros(0)

    analysis = MusicAnalysis(
        path=str(path),
        duration=float(duration),
        sample_rate=int(sr),
        bpm=float(bpm),
        beats=[float(t) for t in beat_times],
        downbeats=[float(t) for t in downbeats],
        beat_interval=float(60.0 / bpm) if bpm > 0 else 0.0,
        meter=meter,
        onset_envelope=[float(v) for v in env],
        onset_times=[float(t) for t in times],
        sections=_sections(y, sr, duration),
        backend=chosen,
    )
    if analysis.has_beats:
        log_fn(f"🎵 {os.path.basename(str(path))}: {analysis.bpm:.1f} BPM, "
               f"{len(analysis.beats)} beats, {len(analysis.downbeats)} downbeats, "
               f"{len(analysis.sections)} sections ({chosen})")
    else:
        log_fn(f"⚠️ {os.path.basename(str(path))}: no beat detected "
               f"({duration:.2f}s decoded); cuts will not be snapped")
    return analysis


def snap_to_beat(t: float, analysis: MusicAnalysis, *, mode: str = "nearest",
                 max_shift: float | None = None) -> float:
    """Move ``t`` onto the beat grid.

    ``nearest`` takes the closest beat, ``previous`` the last beat at or before
    ``t``, ``next`` the first at or after. With no beats at all, ``t`` is
    returned untouched.

    ``max_shift`` is a magnetic range: when the chosen beat is further than
    this many seconds away, ``t`` is left where it is. It defaults to ``None``,
    meaning no limit — the historical behaviour, where a time outside the grid
    clamps to its nearest end.

    That clamping is worth setting a range against, because outside the grid
    the distance is unbounded and nothing says so. A track with a 15 s ambient
    intro has no beats before 15 s, so a cut at 10 s does not nudge — it jumps
    forward five seconds onto the first beat. The same applies past the last
    beat: if the music is shorter than the reel, every later cut collapses onto
    the final beat. Inside the grid a nearest-snap is never more than half a
    beat away, so passing ``max_shift=analysis.beat_interval`` keeps every
    genuine snap and rejects only the two runaway cases.
    """
    if mode not in SNAP_MODES:
        raise ValueError(f"unknown snap mode: {mode!r} (expected one of {SNAP_MODES})")
    beats = analysis.beats
    if not beats:
        return float(t)

    grid = np.asarray(beats, dtype=np.float64)
    value = float(t)
    if mode == "previous":
        index = int(np.searchsorted(grid, value, side="right")) - 1
        snapped = float(grid[max(0, index)])
    elif mode == "next":
        index = int(np.searchsorted(grid, value, side="left"))
        snapped = float(grid[min(index, grid.size - 1)])
    else:
        index = int(np.searchsorted(grid, value, side="left"))
        if index <= 0:
            snapped = float(grid[0])
        elif index >= grid.size:
            snapped = float(grid[-1])
        else:
            before, after = grid[index - 1], grid[index]
            snapped = float(before if (value - before) <= (after - value) else after)

    if max_shift is not None and abs(snapped - value) > float(max_shift):
        return value
    return snapped


def snap_segments(segments: Sequence, analysis: MusicAnalysis, *,
                  mode: str = "nearest", min_duration: float = 0.0,
                  max_shift: float | None = None) -> list:
    """Snap every ``(start, end)`` pair onto the beat grid.

    Both ends are snapped independently, which can collapse a short segment
    onto a single beat; when that happens the end is pushed out to the next
    beat, and then further until the segment reaches ``min_duration``. The
    length guarantee is unconditional — if the grid runs out before the
    requirement is met, the end is placed off-grid rather than returning a clip
    too short to read. A caller asking for a minimum length means it.

    ``max_shift`` is passed to :func:`snap_to_beat`; see there for why bounding
    the move matters on a track whose beats do not span its whole length.
    Boundaries beyond the range keep their original time, so a segment that
    starts before the music does is trimmed to the beat only at the end that
    actually has one.

    Order and count are preserved and overlaps are not resolved: two segments
    that snap to the same beat stay two segments, because deciding what to drop
    belongs to whoever chose them.
    """
    if mode not in SNAP_MODES:
        raise ValueError(f"unknown snap mode: {mode!r} (expected one of {SNAP_MODES})")

    grid = np.asarray(analysis.beats, dtype=np.float64)
    floor = max(0.0, float(min_duration))
    out = []
    for segment in segments:
        start, end = (float(segment[0]), float(segment[1]))
        if grid.size:
            start = snap_to_beat(start, analysis, mode=mode, max_shift=max_shift)
            end = snap_to_beat(end, analysis, mode=mode, max_shift=max_shift)
        if end <= start:
            following = grid[grid > start]
            if following.size:
                end = float(following[0])
            elif analysis.beat_interval > 0:
                end = start + analysis.beat_interval
            else:
                end = max(float(segment[1]), start)
        if floor > 0 and (end - start) < floor:
            following = grid[grid >= start + floor]
            end = float(following[0]) if following.size else start + floor
        out.append((float(start), float(end)))
    return out


def beat_aligned_durations(analysis: MusicAnalysis, *, bars: int = 1) -> float:
    """Seconds spanned by ``bars`` bars at this track's tempo and meter.

    The natural clip length for a beat-matched edit: cut on a downbeat, run for
    a whole number of bars, and the next cut is a downbeat too. Returns 0.0
    when no tempo was found, which a caller must read as "use your own default"
    rather than "zero-length clip".
    """
    if analysis.beat_interval <= 0 or analysis.meter < 1:
        return 0.0
    return float(max(0.0, float(bars)) * analysis.meter * analysis.beat_interval)


def save_analysis(analysis: MusicAnalysis, path: str) -> str:
    """Write ``analysis`` to ``path`` as JSON and return ``path``.

    Floats are rounded — 6 decimals for times, 5 for the envelope — which is
    lossy at a level a hundred times finer than the 23 ms frame the numbers
    came from, and turns a multi-megabyte envelope dump into a file a person
    can open. Analysis is expensive enough (a full decode and STFT) that
    caching it beside the music is the whole reason this exists.
    """
    payload = {
        "schema": SCHEMA_VERSION,
        "path": analysis.path,
        "duration": round(float(analysis.duration), 6),
        "sample_rate": int(analysis.sample_rate),
        "bpm": round(float(analysis.bpm), 6),
        "beats": [round(float(v), 6) for v in analysis.beats],
        "downbeats": [round(float(v), 6) for v in analysis.downbeats],
        "beat_interval": round(float(analysis.beat_interval), 9),
        "meter": int(analysis.meter),
        "onset_envelope": [round(float(v), 5) for v in analysis.onset_envelope],
        "onset_times": [round(float(v), 6) for v in analysis.onset_times],
        "sections": [
            {
                "start": round(float(s.start), 6),
                "end": round(float(s.end), 6),
                "energy": round(float(s.energy), 6),
                "label": str(s.label),
            }
            for s in analysis.sections
        ],
        "backend": analysis.backend,
    }
    parent = os.path.dirname(os.path.abspath(str(path)))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
    return str(path)


def load_analysis(path: str) -> MusicAnalysis:
    """Read back what ``save_analysis`` wrote.

    Raises ValueError on anything that is not a current sidecar — wrong schema,
    missing field, unparseable. A cache that loads half a beat grid is worse
    than no cache: the cutter would place boundaries against a phantom tempo
    and nothing downstream could tell.
    """
    try:
        with open(str(path), "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read music analysis {path!r}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"music analysis {path!r} is not a JSON object")
    if int(data.get("schema", 0)) != SCHEMA_VERSION:
        raise ValueError(f"music analysis {path!r} has schema "
                         f"{data.get('schema')!r}, expected {SCHEMA_VERSION}")

    required = ("path", "duration", "sample_rate", "bpm", "beats", "downbeats",
                "beat_interval", "meter", "onset_envelope", "onset_times",
                "sections", "backend")
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"music analysis {path!r} is missing {', '.join(missing)}")

    try:
        sections = [
            Section(start=float(s["start"]), end=float(s["end"]),
                    energy=float(s["energy"]), label=str(s["label"]))
            for s in data["sections"]
        ]
        return MusicAnalysis(
            path=str(data["path"]),
            duration=float(data["duration"]),
            sample_rate=int(data["sample_rate"]),
            bpm=float(data["bpm"]),
            beats=[float(v) for v in data["beats"]],
            downbeats=[float(v) for v in data["downbeats"]],
            beat_interval=float(data["beat_interval"]),
            meter=int(data["meter"]),
            onset_envelope=[float(v) for v in data["onset_envelope"]],
            onset_times=[float(v) for v in data["onset_times"]],
            sections=sections,
            backend=str(data["backend"]),
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"music analysis {path!r} is malformed: {exc}") from exc
