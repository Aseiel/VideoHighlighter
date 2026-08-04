"""Find where a video's audio jumps above its own local level, and how rhythmic it is.

This measures **audience reaction**, not laughter. Telling laughter from speech
by DSP alone is genuinely hard -- both are amplitude-modulated at roughly
4-5 Hz, which is the obvious feature and also the one that fails -- so nothing
here claims to classify anything. It reports two properties of the audio and
leaves the judgement to whoever reads them:

``burst``
    Loudness above the *local* baseline, as a robust z-score. An audience
    reaction is not absolutely loud, it is loud *for this part of this video*,
    which is why ``modules/audio_peaks.py``'s fixed ``-20 dBFS`` threshold
    cannot find one: that number is a property of the mastering level, not of
    the content.

    The scale matters as much as the reference. Per-second RMS on continuous
    speech swings ten decibels or more between syllables and pauses, so a fixed
    "4 dB above baseline" rule fires constantly on any dialogue-heavy video --
    measured, once, at one candidate every sixteen seconds across an hour. The
    threshold is therefore expressed in units of the video's own variability
    (median absolute deviation), so it self-calibrates instead of needing a
    per-video guess.

``modulation``
    How much of the amplitude envelope's 1-15 Hz energy falls in the 3.5-7.5 Hz
    syllabic band. Speech scores here too, so this ranks bursts against each
    other -- it is not a laughter test. Rhythmic group laughter tends to score
    higher than a shout or a door slam.

Both are medians and fractions rather than tuned magic numbers, so the output
means the same thing across videos mastered at different levels.

``analyse`` is the whole public surface. It decodes audio with ffmpeg, streams
it, and returns plain lists and dicts -- no numpy arrays escape, so the result
is JSON-serialisable and can cross a thread boundary into Qt untouched.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import wave
from typing import Callable, Optional, Sequence

import numpy as np

from modules.app_paths import ffmpeg_exe

# 16 kHz mono is plenty: everything measured here lives below 4 kHz, and it
# keeps a feature-length video's audio to a size that streams comfortably.
SAMPLE_RATE = 16000

# 25 ms analysis window every 10 ms. The hop is what matters -- it makes the
# amplitude envelope a 100 Hz signal, so the 3.5-7.5 Hz band being measured sits
# well inside Nyquist. A 100 ms hop (the obvious choice) would put it outside,
# and the band would alias into nonsense.
FRAME_WIN = 400
FRAME_HOP = 160
ENVELOPE_RATE = SAMPLE_RATE // FRAME_HOP  # 100 Hz

# Laughter and speech both concentrate here; below 300 Hz is rumble and room
# tone, above 3 kHz is mostly consonant hiss and noise.
BAND_LOW_HZ = 300.0
BAND_HIGH_HZ = 3000.0

# Syllabic rate. Laughter sits near the top of this range, speech near the
# bottom, and they overlap -- see the module docstring.
MOD_LOW_HZ = 3.5
MOD_HIGH_HZ = 7.5
MOD_REFERENCE_LOW_HZ = 1.0
MOD_REFERENCE_HIGH_HZ = 15.0
MOD_WINDOW_SECONDS = 2.0

# Half-width of the rolling window defining "normal around here". Two minutes of
# context: long enough that a single reaction cannot drag its own baseline up,
# short enough to track a change of scene or venue.
BASELINE_HALF_WIDTH = 60

# Scales median absolute deviation to be comparable with a standard deviation on
# normally distributed data, so the z-threshold reads on the familiar scale.
MAD_TO_SIGMA = 1.4826

# Digital silence has no dB value; this is the floor reported instead.
SILENCE_DBFS = -100.0

DEFAULT_Z = 3.0
DEFAULT_MIN_DURATION = 1.5

# Above this share of syllabic energy, a burst is rhythmic enough to be worth
# listening to before dismissing the video as having no laughter. A deliberately
# weak claim: pure 5 Hz modulation measures ~0.56, ordinary speech sits well
# below, and real laughter lands somewhere between with a lot of scatter.
MODULATION_NOTABLE = 0.25

# How many high-modulation seconds to surface regardless of loudness. A quiet
# chuckle never clears a loudness threshold, so a burst-only answer to "is there
# any laughter in here" would be begging the question.
DEFAULT_RHYTHMIC_COUNT = 15

# Keep surfaced rhythmic seconds this far apart, so the list is that many
# distinct moments rather than one moment sampled fifteen times.
RHYTHMIC_SPACING_SECONDS = 5

# Read the wav a minute at a time. The whole point is that a two-hour video
# never has more than this in memory as samples.
BLOCK_SECONDS = 60

ProgressFn = Optional[Callable[[float], None]]


class Cancelled(Exception):
    """Raised when a caller's cancel flag is set mid-analysis."""


def _tick(progress: ProgressFn, fraction: float) -> None:
    if progress:
        progress(max(0.0, min(1.0, float(fraction))))


def _check(cancel) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def extract_audio(video_path: str, wav_path: str) -> None:
    """Decode a video's audio to 16 kHz mono PCM."""
    subprocess.run(
        [
            ffmpeg_exe(), "-i", video_path, "-vn", "-acodec", "pcm_s16le",
            "-ar", str(SAMPLE_RATE), "-ac", "1", wav_path, "-y",
            "-hide_banner", "-loglevel", "error",
        ],
        check=True,
        capture_output=True,
    )


def frame_features(wav_path: str,
                   *,
                   progress: ProgressFn = None,
                   cancel=None,
                   progress_span: tuple = (0.0, 1.0)) -> tuple:
    """Per-frame full-band RMS and 300-3000 Hz envelope, both at 100 Hz.

    Streamed in blocks with an overlap carry, so memory does not scale with the
    length of the video. Framing and the FFT are vectorised per block; a Python
    loop over the ~700,000 frames in a two-hour video would take longer than the
    ffmpeg decode that produced them.
    """
    hann = np.hanning(FRAME_WIN).astype(np.float32)
    freqs = np.fft.rfftfreq(FRAME_WIN, 1.0 / SAMPLE_RATE)
    band = (freqs >= BAND_LOW_HZ) & (freqs <= BAND_HIGH_HZ)

    rms_frames: list = []
    env_frames: list = []
    carry = np.zeros(0, dtype=np.float32)
    lo_frac, hi_frac = progress_span

    with wave.open(wav_path, "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise RuntimeError("expected 16-bit mono PCM from ffmpeg")
        total = max(1, wf.getnframes())
        done = 0

        while True:
            _check(cancel)
            raw = wf.readframes(SAMPLE_RATE * BLOCK_SECONDS)
            if not raw:
                break
            block = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            done += len(block)
            _tick(progress, lo_frac + (hi_frac - lo_frac) * (done / total))

            buf = np.concatenate((carry, block)) if carry.size else block
            if buf.size < FRAME_WIN:
                carry = buf
                continue

            count = (buf.size - FRAME_WIN) // FRAME_HOP + 1
            frames = np.lib.stride_tricks.sliding_window_view(buf, FRAME_WIN)
            frames = frames[: count * FRAME_HOP : FRAME_HOP]

            rms_frames.append(np.sqrt(np.mean(frames * frames, axis=1)))
            spectrum = np.abs(np.fft.rfft(frames * hann, axis=1)) ** 2
            env_frames.append(np.sqrt(spectrum[:, band].sum(axis=1)))

            # Keep the tail a following frame would need to span.
            carry = buf[count * FRAME_HOP:]

    if not rms_frames:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(rms_frames), np.concatenate(env_frames)


def to_dbfs(values: np.ndarray) -> np.ndarray:
    """Linear amplitude -> dBFS, with a floor so silence is a number."""
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.maximum(values, 1e-12))
    return np.maximum(db, SILENCE_DBFS)


def per_second_loudness(rms: np.ndarray) -> np.ndarray:
    """Collapse 100 Hz frame RMS to one dBFS value per second.

    Energy-mean rather than peak: a reaction is *sustained* loudness, and a peak
    would let one door slam look like six seconds of applause.
    """
    usable = (len(rms) // ENVELOPE_RATE) * ENVELOPE_RATE
    if usable == 0:
        return np.zeros(0)
    blocks = rms[:usable].reshape(-1, ENVELOPE_RATE)
    return to_dbfs(np.sqrt(np.mean(blocks * blocks, axis=1)))


def local_statistics(loudness: np.ndarray,
                     half_width: int = BASELINE_HALF_WIDTH) -> tuple:
    """Rolling median and scaled median absolute deviation of loudness.

    Both are medians so that the reactions being looked for cannot inflate the
    reference they are supposed to stand out from -- a rolling *mean* would let
    a long stretch of applause raise its own bar.
    """
    if not len(loudness):
        return loudness, loudness
    baseline = np.empty_like(loudness)
    spread = np.empty_like(loudness)
    for i in range(len(loudness)):
        lo = max(0, i - half_width)
        hi = min(len(loudness), i + half_width + 1)
        window = loudness[lo:hi]
        median = np.median(window)
        baseline[i] = median
        spread[i] = MAD_TO_SIGMA * np.median(np.abs(window - median))
    return baseline, spread


def modulation_ratio(envelope: np.ndarray,
                     seconds: int,
                     *,
                     progress: ProgressFn = None,
                     cancel=None,
                     progress_span: tuple = (0.0, 1.0)) -> tuple:
    """Per second: the syllabic energy share, and the rate it peaks at.

    The envelope is windowed two seconds at a time, which buys 0.5 Hz
    resolution -- enough to separate the 3.5-7.5 Hz band from its neighbours.
    The share is reported as a fraction of nearby modulation energy rather than
    an absolute, so it does not simply restate loudness.

    The *rate* is measured across the whole 1-15 Hz reference range rather than
    inside the band, so a second whose rhythm falls outside the band still
    reports where it actually was. That matters more than it looks: this
    measurement finds rhythmic vocalisation in general, of which laughter is one
    kind, and the rate is the cheapest thing that distinguishes one kind from
    another -- a free discriminator from an FFT already computed.
    """
    width = int(MOD_WINDOW_SECONDS * ENVELOPE_RATE)
    hann = np.hanning(width)
    freqs = np.fft.rfftfreq(width, 1.0 / ENVELOPE_RATE)
    band = (freqs >= MOD_LOW_HZ) & (freqs <= MOD_HIGH_HZ)
    reference = (freqs >= MOD_REFERENCE_LOW_HZ) & (freqs <= MOD_REFERENCE_HIGH_HZ)

    ratios = np.zeros(max(0, seconds))
    rates = np.zeros(max(0, seconds))
    if not band.any() or not reference.any() or seconds <= 0:
        return ratios, rates
    reference_freqs = freqs[reference]
    lo_frac, hi_frac = progress_span

    for sec in range(seconds):
        if sec % 500 == 0:
            _check(cancel)
            _tick(progress, lo_frac + (hi_frac - lo_frac) * (sec / seconds))
        centre = sec * ENVELOPE_RATE
        lo = max(0, centre - width // 2)
        hi = lo + width
        if hi > len(envelope):
            break
        window = envelope[lo:hi]
        window = window - window.mean()
        if not np.any(window):
            continue
        power = np.abs(np.fft.rfft(window * hann)) ** 2
        total = power[reference].sum()
        if total <= 0:
            continue
        ratios[sec] = float(power[band].max() / total)
        rates[sec] = float(reference_freqs[int(np.argmax(power[reference]))])
    return ratios, rates


def format_timestamp(seconds: float) -> str:
    """Seconds -> ``M:SS`` or ``H:MM:SS`` once past an hour."""
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def find_candidates(z: np.ndarray,
                    burst_db: np.ndarray,
                    modulation: np.ndarray,
                    rates: np.ndarray,
                    *,
                    z_threshold: float,
                    min_duration: float,
                    rank_by: str = "modulation") -> list:
    """Runs of seconds that stayed loud relative to their own surroundings.

    Deliberately the simplest rule that could work: the output is meant to be
    judged by a person, and an elaborate rule here would only obscure whether
    the underlying *signal* is any good.

    ``rank_by`` orders the result and nothing else -- membership is decided by
    ``z_threshold`` and ``min_duration`` alone, whichever ordering is asked for.

    ``"modulation"`` (the default) answers "is any of this laughter", where the
    loudest burst is usually a door and the rhythmic one is the only one worth
    a listen. ``"z"`` answers "where is this video loudest for itself", and on
    material whose rhythm is speech the modulation ordering actively buries the
    answer -- measured on an hour of it, the six most syllabic seconds were all
    dialogue while every loud excursion sat at or below its local baseline.
    Neither ordering is right in general, which is why this is a parameter and
    not a new default.
    """
    above = np.append(z > z_threshold, False)
    candidates: list = []
    start: Optional[int] = None

    for sec, hot in enumerate(above):
        if hot and start is None:
            start = sec
        elif not hot and start is not None:
            duration = sec - start
            if duration >= min_duration:
                window = slice(start, sec)
                peak = start + int(np.argmax(z[window]))
                candidates.append({
                    "start": float(start),
                    "end": float(sec),
                    "duration": float(duration),
                    "peak_second": int(peak),
                    "timestamp": format_timestamp(peak),
                    "z": round(float(z[window].max()), 2),
                    "burst_db": round(float(burst_db[window].max()), 2),
                    "modulation": round(float(modulation[window].mean()), 3),
                    "peak_modulation": round(float(modulation[window].max()), 3),
                    # The rate at the second that was most rhythmic, not the
                    # mean -- averaging two different rhythms yields a rate
                    # neither of them had.
                    "mod_hz": round(float(
                        rates[start + int(np.argmax(modulation[window]))]), 2),
                })
            start = None

    if rank_by == "z":
        candidates.sort(key=lambda c: (c["z"], c["peak_modulation"]), reverse=True)
    else:
        # Most rhythmic first. When the question is "is there laughter here", the
        # loudest burst is usually a door or a music cue; the rhythmic one is the
        # only one worth the user's time.
        candidates.sort(key=lambda c: (c["peak_modulation"], c["z"]), reverse=True)
    return candidates


def rhythmic_seconds(modulation: np.ndarray,
                     z: np.ndarray,
                     rates: np.ndarray,
                     *,
                     count: int = DEFAULT_RHYTHMIC_COUNT,
                     spacing: int = RHYTHMIC_SPACING_SECONDS) -> list:
    """The most syllabic seconds in the video, loud or not.

    A quiet chuckle never clears a loudness threshold, so answering "is there
    any laughter in here" from bursts alone would beg the question. These are
    the seconds to actually listen to before concluding there is none.
    """
    if not len(modulation):
        return []
    order = np.argsort(modulation)[::-1]
    picked: list = []
    for sec in order:
        sec = int(sec)
        if modulation[sec] <= 0:
            break
        if any(abs(sec - p["second"]) < spacing for p in picked):
            continue
        picked.append({
            "second": sec,
            "timestamp": format_timestamp(sec),
            "modulation": round(float(modulation[sec]), 3),
            "mod_hz": round(float(rates[sec]) if sec < len(rates) else 0.0, 2),
            "z": round(float(z[sec]) if sec < len(z) else 0.0, 2),
        })
        if len(picked) >= count:
            break
    return picked


def verdict(candidates: Sequence, rhythmic: Sequence, seconds: int) -> dict:
    """A plain-language reading of whether this video has reaction structure.

    Deliberately conservative. The honest finding on content with no audience is
    "nothing here is rhythmic enough to be laughter", and saying that clearly is
    more useful than a ranked list that implies otherwise.
    """
    peak_mod = max((float(r["modulation"]) for r in rhythmic), default=0.0)
    notable = [c for c in candidates
               if float(c["peak_modulation"]) >= MODULATION_NOTABLE]
    per_hour = (len(candidates) / seconds * 3600.0) if seconds else 0.0

    if peak_mod < MODULATION_NOTABLE:
        headline = "No laughter-like rhythm anywhere in this video"
        detail = (
            f"The most syllabic second measures {peak_mod:.2f}, below the {MODULATION_NOTABLE:.2f} "
            "worth investigating. Sustained rhythmic laughter reads far higher. "
            "Whatever is loud here is loud without being rhythmic."
        )
    elif notable:
        headline = f"{len(notable)} burst(s) rhythmic enough to check"
        detail = (
            f"Peak modulation {peak_mod:.2f}. These are loud *and* syllabic, which is "
            "what group laughter looks like. Listen before trusting it — speech and "
            "applause can both land here."
        )
    else:
        headline = "Some rhythmic seconds, but none of them loud"
        detail = (
            f"Peak modulation {peak_mod:.2f}, but no rhythmic moment also rose above "
            "its local baseline. Quiet laughter would look like this; so would "
            "ordinary conversation."
        )

    # One candidate every few seconds is not a detector, it is a threshold sitting
    # inside normal variation. Say so rather than let a long list imply otherwise.
    noisy = per_hour > 120
    return {
        "headline": headline,
        "detail": detail,
        "peak_modulation": round(peak_mod, 3),
        "notable_count": len(notable),
        "candidates_per_hour": round(per_hour, 1),
        "noisy": bool(noisy),
    }


def analyse(video_path: str,
            *,
            z_threshold: float = DEFAULT_Z,
            min_duration: float = DEFAULT_MIN_DURATION,
            rhythmic_count: int = DEFAULT_RHYTHMIC_COUNT,
            progress: ProgressFn = None,
            cancel=None) -> dict:
    """Measure one video's reaction bursts. Everything returned is plain data."""
    wav_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    try:
        _tick(progress, 0.02)
        _check(cancel)
        extract_audio(video_path, wav_path)
        _tick(progress, 0.15)

        rms, envelope = frame_features(
            wav_path, progress=progress, cancel=cancel, progress_span=(0.15, 0.80))
        if not len(rms):
            raise RuntimeError("no audio track found in this video")

        loudness = per_second_loudness(rms)
        baseline, spread = local_statistics(loudness)
        burst_db = loudness - baseline
        # A perfectly steady stretch has zero spread; without a floor every
        # rounding wobble in it would divide out to an infinite z-score.
        z = burst_db / np.maximum(spread, 0.5)
        _tick(progress, 0.85)

        modulation, rates = modulation_ratio(
            envelope, len(loudness), progress=progress, cancel=cancel,
            progress_span=(0.85, 0.98))

        candidates = find_candidates(
            z, burst_db, modulation, rates,
            z_threshold=z_threshold, min_duration=min_duration)
        rhythmic = rhythmic_seconds(modulation, z, rates, count=rhythmic_count)
        _tick(progress, 1.0)

        return {
            "video": os.path.abspath(video_path),
            "seconds": int(len(loudness)),
            "params": {
                "z_threshold": float(z_threshold),
                "min_duration": float(min_duration),
            },
            "verdict": verdict(candidates, rhythmic, len(loudness)),
            "candidates": candidates,
            "rhythmic": rhythmic,
            # Curves, for a caller that wants to draw them. Plain lists so the
            # whole result stays JSON-serialisable.
            "curves": {
                "loudness": [round(float(v), 2) for v in loudness],
                "baseline": [round(float(v), 2) for v in baseline],
                "burst_db": [round(float(v), 2) for v in burst_db],
                "z": [round(float(v), 2) for v in z],
                "modulation": [round(float(v), 3) for v in modulation],
                "mod_hz": [round(float(v), 2) for v in rates],
            },
        }
    finally:
        try:
            os.remove(wav_path)
        except OSError:
            pass


def write_json(result: dict, path: str) -> None:
    """Persist a result, curves and all."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
