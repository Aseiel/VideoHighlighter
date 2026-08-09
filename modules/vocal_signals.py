"""Per-second vocal brightness and onset sharpness, measured against a video's own voice.

``reaction_bursts`` answers "is this loud for here, and is it rhythmic". Both
questions turn out to be the wrong ones for telling *kinds* of vocalisation
apart. Measured against hand-labelled examples on one feature-length file, with
the loudness band and the vocal gate below held fixed so the comparison was
between vocal seconds rather than between voice and a door:

===================  =====  =========================================
feature              AUC    reading
===================  =====  =========================================
brightness           0.90   spectral centre of mass
onset                0.89   how fast the level rose into this second
high-band ratio      0.71   weak
roughness            0.69   CI includes chance
fundamental (F0)     0.69   CI includes chance
loudness             0.64   CI includes chance
``mod_ratio``        0.58   the shipped rhythm feature; a coin flip
===================  =====  =========================================

The two that survived are the two here. Loudness in particular is *not* one of
them: the loudest vocal second in that file belonged to the class the labeller
was trying to exclude, which is why nothing in this module thresholds on level.

**What this does not do.** It measures two properties and holds no opinion about
what they mean. Which kinds of vocalisation a video contains, what they should
be called, and where the boundary between them sits are all the user's to supply
-- from labelled examples at runtime, never from a constant in here. On the
material it was fitted to, one pair of classes separated cleanly and a second
pair did not separate at all, the labeller themselves hedging on five of
twenty-eight clips. A score is therefore what this returns: a threshold is a
judgement, and it belongs to whoever is making it.

**Absolute values do not transfer.** Brightness in hertz is a property of a
voice and a mastering chain as much as of what the voice is doing, so a constant
cut-off fitted on one file means nothing on the next -- the same lesson
``modules/loudness_bursts.py`` records about absolute dB thresholds. Both
features are therefore reported as robust z-scores *against the vocal seconds of
the same video*, so "bright for this voice" is what crosses a file boundary.

The vocal gate is worth having on its own account. Impacts, slaps and cloth are
broadband and flat, and they outrank real vocal events on every loudness-derived
measure in the app. It is a rejector rather than a selector, and deliberately
admits most of a file -- see its constants for why tightening it costs true
positives and gains nothing.
"""
from __future__ import annotations

import os
import tempfile
import wave
from typing import Callable, Optional

import numpy as np

from modules.reaction_bursts import (
    ENVELOPE_RATE,
    FRAME_HOP,
    FRAME_WIN,
    SAMPLE_RATE,
    SILENCE_DBFS,
    extract_audio,
    format_timestamp,
)

# Read the wav a minute at a time, as ``reaction_bursts`` does, so a two-hour
# file never has more than this in memory as samples.
BLOCK_SECONDS = 60

# --- the vocal gate ---------------------------------------------------------
# A rejector, not a selector, and set as permissively as that allows.
#
# Non-vocal material in the labelled set sat at median flatness 0.126 and
# high-band ratio 0.288, against 0.009 and 0.024 for vocal seconds -- an order
# of magnitude, which is what makes the gate worth having at all. But tightening
# it towards those vocal medians buys nothing: measured across the whole grid
# from 0.035/0.08 to 0.10/0.15, the number of non-vocal clips rejected never
# moved off 7 of 10. The three that survive are spectrally voice-like (flatness
# 0.007-0.020) and no setting of these two constants excludes them.
#
# So the limits cost true positives without gaining anything. At the tight end
# they threw away a confirmed vocal event that scored in the top three, which is
# the worst thing this gate can do -- the score below is the part that
# discriminates, and a second the gate drops never reaches it. These values are
# therefore the loosest that still reject what is rejectable: they keep all 31
# labelled vocal seconds and still drop 7 of 10 non-vocal ones. That they admit
# most of the file is not a failure; admitting is not selecting.
VOCAL_MAX_FLATNESS = 0.08
VOCAL_MAX_HIGH_RATIO = 0.15

# This one earns its place somewhere else entirely. It excludes nothing in the
# labelled set -- every clip there, of every class, measured above 0.31 -- so it
# is not helping to tell events apart. What it does is keep near-silence and
# room tone out of the *reference population* the z-scores below are computed
# against, which is 438 seconds of this file. Without it "bright for this voice"
# would partly mean "bright compared with this room", and the normalisation
# would drift with how much of a video is quiet rather than with the voice.
VOCAL_MIN_PERIODICITY = 0.30

# Energy above this is "high band" for the gate's purposes. Voice keeps most of
# its energy well below; a broadband transient does not.
HIGH_BAND_HZ = 2000.0

# Periodicity is searched over this range of fundamentals. Wide, because it is
# used only to ask "is this periodic at all" -- the fundamental itself measured
# AUC 0.69 with a confidence interval spanning chance, so nothing here uses it.
F0_MIN_HZ = 70.0
F0_MAX_HZ = 1200.0

# --- level peaks -------------------------------------------------------------
# The rule below is the same *shape* as the one behind the timeline's AUDIO
# WAVEFORM arrows -- normalise against the file's own 10th-97th percentile, take
# local maxima at or above the sensitivity, suppress within a second of a louder
# peak -- but it is emphatically NOT the same measurement, and an earlier version
# of this comment claimed it was.
#
# The difference is resolution and it is not small. `WaveformVisualizer` builds
# 1000 bins for a whole file, so on a 52-minute recording one bin spans three
# seconds; this peaks over a 10 Hz envelope. Measured on two files: 66 arrows
# against 790 peaks, and 61 against 338. Twelve times and five times as many.
#
# What follows from that is the thing to keep in mind when reading a density:
# with suppression at one second, a 20 s window can hold at most 20 peaks, so a
# density of 10 means *half the seconds carry a peak*. That is a measure of how
# much of the window sits above a high level -- sustained loudness -- and not of
# how spiky it is. A short burst in a quiet stretch scores low; a steady
# moderately-loud passage scores high while the display, averaging over three
# seconds, shows no arrows at all. Both have been reported as surprises, and
# both are this, working as built.
#
# It is kept at this resolution because it separates the labelled classes better
# than counting the drawn arrows does: AUC 0.84 against 0.65 on the same spans.
# The cost is that the display cannot be used to check it.
PEAK_SENSITIVITY = 0.75
PEAK_MIN_GAP_SECONDS = 1.0
PEAK_NORM_LOW_PCT = 10.0
PEAK_NORM_HIGH_PCT = 97.0
FINE_RATE = 10          # envelope bins per second for peak picking

# Window the peak *rate* is counted over. 20 s separated the labelled classes
# best of 10/15/20, and the ordering was stable across all three -- see
# ``peak_density_curve`` for what that separation actually was.
PEAK_DENSITY_WINDOW = 20

# The window "concentrated" is measured over. Ten seconds, chosen against 18
# hand-labelled episodes across seven recordings: 10 s recovered 16 of 16 on the
# four files whose labels were unambiguous, and widening to 20 or 30 s lost
# three or four of them by averaging a short episode into its quiet neighbours.
DENSITY_WINDOW_SECONDS = 10

# What counts as a loud second for the density's purposes, as a percentile of
# the file's own levels. Measured across 60/70/80 the recall did not move, which
# is the useful kind of insensitivity: the measurement is carried by how
# clustered the loud seconds are, not by where exactly the bar for "loud" sits.
DENSITY_LOUD_PERCENTILE = 80.0

# How far back an onset is measured from. Two seconds because that is the span
# over which the labelled classes separated (means 17 dB against 8 dB); a
# shorter window measures syllable edges, which every kind of vocalisation has.
ONSET_LOOKBACK_SECONDS = 2

# Floors on the robust spread, in each quantity's own units, so that a video
# whose voice barely varies cannot divide a rounding wobble out into an enormous
# z-score. Same guard and same reason as ``reaction_bursts.analyse``'s 0.5 dB,
# and the units matter as much as the guard: a bare epsilon here (which is what
# this was) leaves the arithmetic safe and the *answer* meaningless, reporting a
# z of a million for a tenth of a hertz of drift.
#
# The values are a judgement about when a difference stops being one. Brightness
# differences below a few hertz are not audible and not reliably measurable at
# this frame size; a level difference below half a decibel is not either.
BRIGHTNESS_MIN_SPREAD_HZ = 5.0
ONSET_MIN_SPREAD_DB = 0.5
MAD_TO_SIGMA = 1.4826

# Below this many gated seconds the video has not shown enough of its own voice
# for "bright for this voice" to mean anything, and the z-scores would be
# describing a handful of samples.
MIN_VOCAL_SECONDS = 30

ProgressFn = Optional[Callable[[float], None]]


class Cancelled(Exception):
    """Raised when a caller's cancel flag is set mid-analysis."""


def _tick(progress: ProgressFn, fraction: float) -> None:
    if progress:
        progress(max(0.0, min(1.0, float(fraction))))


def _check(cancel) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _periodicity(seg: np.ndarray) -> float:
    """Height of the autocorrelation peak in the voice range, 0..1.

    By FFT rather than ``np.correlate``: the direct method is quadratic in the
    window, which on a feature-length file is hours rather than seconds.
    """
    if seg.size == 0:
        return 0.0
    x = seg - seg.mean()
    energy = float(np.dot(x, x))
    if energy <= 1e-12:
        return 0.0
    n = 1 << int(np.ceil(np.log2(len(x) * 2)))
    spectrum = np.fft.rfft(x, n)
    ac = np.fft.irfft(spectrum * np.conj(spectrum), n)[:len(x)]
    lo = int(SAMPLE_RATE / F0_MAX_HZ)
    hi = min(len(ac), int(SAMPLE_RATE / F0_MIN_HZ))
    if hi <= lo:
        return 0.0
    return float(np.max(ac[lo:hi]) / energy)


def per_second_features(wav_path: str,
                        *,
                        progress: ProgressFn = None,
                        cancel=None,
                        progress_span: tuple = (0.0, 1.0)) -> dict:
    """Raw per-second measurements, before any normalisation.

    Streamed in blocks with an overlap carry so memory does not scale with the
    length of the video, and vectorised per block for the same reason
    ``reaction_bursts.frame_features`` is: a Python loop over a feature-length
    file's frames costs more than the ffmpeg decode that produced them.
    """
    hann = np.hanning(FRAME_WIN).astype(np.float32)
    freqs = np.fft.rfftfreq(FRAME_WIN, 1.0 / SAMPLE_RATE)
    high = freqs >= HIGH_BAND_HZ

    level: list = []
    brightness: list = []
    flatness: list = []
    high_ratio: list = []
    periodicity: list = []
    fine: list = []          # linear amplitude at FINE_RATE, for peak picking

    with wave.open(wav_path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise RuntimeError("expected 16-bit mono PCM from ffmpeg")
        total = max(1, wf.getnframes())
        done = 0
        lo_frac, hi_frac = progress_span
        carry = np.zeros(0, dtype=np.float32)

        while True:
            _check(cancel)
            raw = wf.readframes(SAMPLE_RATE * BLOCK_SECONDS)
            if not raw:
                break
            block = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            done += len(block)
            _tick(progress, lo_frac + (hi_frac - lo_frac) * (done / total))

            buf = np.concatenate((carry, block)) if carry.size else block
            whole = len(buf) // SAMPLE_RATE
            if whole == 0:
                carry = buf
                continue

            for s in range(whole):
                seg = buf[s * SAMPLE_RATE:(s + 1) * SAMPLE_RATE]
                rms = float(np.sqrt(np.mean(seg * seg)))
                level.append(max(20.0 * np.log10(max(rms, 1e-12)), SILENCE_DBFS))

                # Tenth-of-a-second amplitude, kept alongside the per-second
                # value: a peak is a local maximum, and at one sample per second
                # two peaks a second apart are one sample apart and cannot be
                # told from a plateau.
                blocks = seg.reshape(FINE_RATE, SAMPLE_RATE // FINE_RATE)
                fine.extend(np.sqrt(np.mean(blocks * blocks, axis=1)).tolist())

                count = (len(seg) - FRAME_WIN) // FRAME_HOP + 1
                frames = np.lib.stride_tricks.sliding_window_view(seg, FRAME_WIN)
                frames = frames[: count * FRAME_HOP : FRAME_HOP]
                power = np.abs(np.fft.rfft(frames * hann, axis=1)) ** 2
                totals = power.sum(axis=1) + 1e-20

                # Medians across the second, not means: one transient inside an
                # otherwise steady second should not carry the whole second's
                # description with it.
                brightness.append(float(np.median(
                    (power * freqs).sum(axis=1) / totals)))
                high_ratio.append(float(np.median(
                    power[:, high].sum(axis=1) / totals)))
                logs = np.log(power + 1e-20)
                brightness_floor = np.exp(logs.mean(axis=1))
                flatness.append(float(np.median(
                    brightness_floor / (power.mean(axis=1) + 1e-20))))
                periodicity.append(_periodicity(seg))

            carry = buf[whole * SAMPLE_RATE:]

    return {
        "level": np.asarray(level),
        "brightness": np.asarray(brightness),
        "flatness": np.asarray(flatness),
        "high_ratio": np.asarray(high_ratio),
        "periodicity": np.asarray(periodicity),
        "fine_energy": np.asarray(fine),
    }


def waveform_peaks(fine_energy: np.ndarray,
                   sensitivity: float = PEAK_SENSITIVITY) -> list:
    """Times of local maxima in the level, in seconds.

    Same rule as the timeline's arrows, at a far finer resolution -- see the
    constants. It finds several times as many peaks as the display draws, so
    these are not the arrows and a count here cannot be checked against them.
    """
    if not len(fine_energy):
        return []
    ordered = np.sort(fine_energy)
    lo = float(ordered[min(len(ordered) - 1,
                           int(len(ordered) * PEAK_NORM_LOW_PCT / 100.0))])
    hi = float(ordered[min(len(ordered) - 1,
                           int(len(ordered) * PEAK_NORM_HIGH_PCT / 100.0))])
    norm = np.clip((fine_energy - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    candidates = []
    for i, value in enumerate(norm):
        if value < sensitivity:
            continue
        before = norm[i - 1] if i > 0 else -1.0
        after = norm[i + 1] if i < len(norm) - 1 else -1.0
        if value >= before and value >= after:
            candidates.append((i / float(FINE_RATE), float(value)))

    # Loudest first, dropping anything too close to a peak already kept, so a
    # single loud region contributes one stop rather than a cluster of them.
    candidates.sort(key=lambda c: c[1], reverse=True)
    kept: list = []
    for t, _value in candidates:
        if all(abs(t - k) > PEAK_MIN_GAP_SECONDS for k in kept):
            kept.append(t)
    return sorted(kept)


def peak_density_curve(peaks, seconds: int,
                       window: int = PEAK_DENSITY_WINDOW) -> np.ndarray:
    """How many peaks fall within *window* seconds of each second.

    This is the measurement a person makes by eye when they say the arrows are
    dense here and sparse there.

    What it is worth, against 17 spans hand-marked as spontaneous and 5 marked
    as performed across seven independently mastered recordings, at the 10 Hz
    resolution this actually ships at: **AUC 0.84**. At 10 peaks per 20 s it
    keeps 15 of the 17 and rejects 4 of the 5.

    ============  ==========  ===============
    threshold     real kept   performed cut
    ============  ==========  ===============
    8             16/17       2/5
    10            15/17       4/5
    14            8/17        5/5
    ============  ==========  ===============

    **Two reasons not to trust that more than it deserves.**

    Five negatives. "4 of 5" is 80% with an error bar wide enough to cover 40%,
    and one performed span scored 13, above ten of the seventeen genuine ones.

    More seriously, the raw count beats the per-file percentile here (0.84
    against 0.75) and the reason is a *file-level* offset: the two recordings
    described as mostly performed carry 9.5 and 15.3 peaks a minute against
    6.8-7.7 for the rest. That difference may be performance, or it may be
    production style and compression, and this set cannot tell those apart.
    Every other measurement in this module is normalised per file precisely
    because absolute values usually turn out to be about the mastering. If this
    one stops working on a new recording, that is the first thing to suspect.

    So: a filter with a real error rate, not a decision.
    """
    n = max(0, int(seconds))
    counts = np.zeros(n)
    if not n:
        return counts
    marks = np.zeros(n)
    for t in peaks:
        index = int(t)
        if 0 <= index < n:
            marks[index] += 1.0

    # A running sum rather than ``np.convolve(..., "same")``: that returns
    # ``max(len(signal), len(kernel))`` samples, so on a clip shorter than the
    # window it hands back a curve LONGER than the audio, silently misaligned
    # against every other per-second signal a rule might test alongside it.
    window = max(1, int(window))
    half = window // 2
    cumulative = np.concatenate(([0.0], np.cumsum(marks)))
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + window - half, 0, n)
    return cumulative[hi] - cumulative[lo]


def density_curve(level: np.ndarray,
                  window: int = DENSITY_WINDOW_SECONDS,
                  loud_percentile: float = DENSITY_LOUD_PERCENTILE) -> np.ndarray:
    """How *concentrated* the loud seconds are around each second.

    Not "is this loud" -- that question was asked early in this work and
    answered badly, and the wrong lesson was drawn from it. Peak level does not
    separate the vocal classes here; what does is whether the loud seconds
    arrive in a sustained cluster rather than singly.

    The product of two things, because either alone is misleading: the fraction
    of the surrounding window that is loud for this file, and how far above the
    file's median that window sits. A quiet stretch of continuous murmur scores
    high on the first and low on the second; one isolated shout the reverse.

    Both halves are relative to the file's own distribution, so the result means
    the same thing across recordings mastered independently.
    """
    if not len(level):
        return np.zeros(0)
    hot = level > np.percentile(level, loud_percentile)
    median = float(np.median(level))
    # Clamped to the signal's own length for the reason spelled out in
    # ``peak_density_curve``: on a clip shorter than the window, "same" returns
    # the kernel's length instead and the curve stops lining up with the audio.
    window = max(1, min(int(window), len(level)))
    kernel = np.ones(window) / float(window)
    fraction = np.convolve(hot.astype(float), kernel, mode="same")[:len(level)]
    excess = np.convolve(np.maximum(level - median, 0.0), kernel,
                         mode="same")[:len(level)]
    return fraction * excess


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Each value's rank within the file, 0-100.

    So a rule can say ``min: 90`` and mean "the top tenth of this recording"
    on any recording. An absolute threshold on the raw density would need
    retuning per file, which is the failure mode this whole module is built to
    avoid.

    The cost is worth stating plainly: a percentile always has a top tenth, so a
    rule phrased this way finds candidates in a file that contains none. It
    ranks; it does not decide.
    """
    if not len(values):
        return np.zeros(0)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return 100.0 * ranks / max(len(values) - 1, 1)


def onset_curve(level: np.ndarray,
                lookback: int = ONSET_LOOKBACK_SECONDS) -> np.ndarray:
    """dB risen into each second, from the quietest of the preceding few.

    Against the *minimum* rather than the previous second's value: a rise that
    happens over two seconds is still a fast rise, and differencing against the
    immediate neighbour would score it as two small ones.
    """
    out = np.zeros(len(level))
    for i in range(1, len(level)):
        lo = max(0, i - lookback)
        out[i] = float(level[i] - level[lo:i].min())
    return out


def vocal_mask(features: dict) -> np.ndarray:
    """Which seconds carry voice rather than impacts, cloth or room tone."""
    return ((features["flatness"] <= VOCAL_MAX_FLATNESS)
            & (features["high_ratio"] <= VOCAL_MAX_HIGH_RATIO)
            & (features["periodicity"] > VOCAL_MIN_PERIODICITY))


def robust_z(values: np.ndarray,
             reference: np.ndarray,
             min_spread: float = ONSET_MIN_SPREAD_DB) -> np.ndarray:
    """Z-score against the median and scaled MAD of *reference*.

    Median and MAD rather than mean and standard deviation because the seconds
    being looked for are in the reference set -- they cannot be excluded without
    deciding in advance which they are -- and a mean would let them shift the
    centre they are supposed to stand out from.

    ``min_spread`` is in the units of *values*, and is a floor on what counts as
    variation rather than a guard against dividing by zero. See the constants.
    """
    if not len(reference):
        return np.zeros(len(values))
    median = float(np.median(reference))
    spread = MAD_TO_SIGMA * float(np.median(np.abs(reference - median)))
    return (values - median) / max(spread, float(min_spread))


def analyse(video_path: str,
            *,
            progress: ProgressFn = None,
            cancel=None) -> dict:
    """Measure one video. Everything returned is plain data.

    ``effort`` weights brightness and onset equally. They measured AUC 0.90 and
    0.89 on the labelled set, which is well inside each other's confidence
    interval, so a fitted weighting would be fitting to twenty-six points and
    would carry that sample's accidents into every other file.
    """
    wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        _tick(progress, 0.02)
        _check(cancel)
        extract_audio(video_path, wav_path)
        _tick(progress, 0.15)

        raw = per_second_features(wav_path, progress=progress, cancel=cancel,
                                  progress_span=(0.15, 0.95))
        if not len(raw["level"]):
            raise RuntimeError("no audio track found in this video")

        onset = onset_curve(raw["level"])
        vocal = vocal_mask(raw)
        enough = int(vocal.sum()) >= MIN_VOCAL_SECONDS

        # Normalised against the video's own vocal seconds -- not against every
        # second, which would mostly measure how much of the file is silence.
        if enough:
            zb = robust_z(raw["brightness"], raw["brightness"][vocal],
                          BRIGHTNESS_MIN_SPREAD_HZ)
            zo = robust_z(onset, onset[vocal], ONSET_MIN_SPREAD_DB)
        else:
            zb = np.zeros(len(raw["level"]))
            zo = np.zeros(len(raw["level"]))

        effort = (zb + zo) / 2.0
        # Outside the gate the score describes something that is not a voice, so
        # it is not reported as one.
        effort = np.where(vocal, effort, 0.0)

        # Density is deliberately *not* gated. The gate rejects broadband
        # transients frame by frame, and a sustained vocal passage routinely has
        # a slap or an impact inside it; zeroing those seconds would punch holes
        # in the very cluster this measures.
        density = density_curve(raw["level"])
        density_pct = percentile_rank(density)

        peaks = waveform_peaks(raw["fine_energy"])
        peak_density = peak_density_curve(peaks, len(raw["level"]))
        peak_density_pct = percentile_rank(peak_density)
        _tick(progress, 1.0)

        return {
            "video": os.path.abspath(video_path),
            "seconds": int(len(raw["level"])),
            "vocal_seconds": int(vocal.sum()),
            "peaks": [round(float(t), 2) for t in peaks],
            "enough_voice": bool(enough),
            "gate": {
                "max_flatness": VOCAL_MAX_FLATNESS,
                "max_high_ratio": VOCAL_MAX_HIGH_RATIO,
                "min_periodicity": VOCAL_MIN_PERIODICITY,
            },
            "curves": {
                "vocal": [bool(v) for v in vocal],
                "level": [round(float(v), 2) for v in raw["level"]],
                "brightness_hz": [round(float(v), 1) for v in raw["brightness"]],
                "onset_db": [round(float(v), 2) for v in onset],
                "brightness_z": [round(float(v), 3) for v in zb],
                "onset_z": [round(float(v), 3) for v in zo],
                "effort": [round(float(v), 3) for v in effort],
                "density": [round(float(v), 3) for v in density],
                "density_pct": [round(float(v), 1) for v in density_pct],
                "peak_density": [round(float(v), 1) for v in peak_density],
                "peak_density_pct": [round(float(v), 1) for v in peak_density_pct],
            },
        }
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


def top_moments(result: dict, *, count: int = 20, spacing: int = 5) -> list:
    """The highest-scoring vocal seconds, spaced out so they are distinct moments.

    For review, and for collecting the labels that set a threshold. No threshold
    is applied here -- see the module docstring for why one is not this file's
    to choose.
    """
    effort = np.asarray((result.get("curves") or {}).get("effort") or [],
                        dtype=float)
    vocal = (result.get("curves") or {}).get("vocal") or []
    if not len(effort):
        return []
    picked: list = []
    for sec in np.argsort(effort)[::-1]:
        sec = int(sec)
        if sec < len(vocal) and not vocal[sec]:
            continue
        if any(abs(sec - p["second"]) < spacing for p in picked):
            continue
        picked.append({
            "second": sec,
            "timestamp": format_timestamp(sec),
            "effort": round(float(effort[sec]), 3),
            "brightness_z": result["curves"]["brightness_z"][sec],
            "onset_z": result["curves"]["onset_z"][sec],
        })
        if len(picked) >= count:
            break
    return picked
