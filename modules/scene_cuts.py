"""Decide where the shot cuts are, from the frame differences already measured.

``detect_scenes_motion_optimized`` compares each sampled frame with the one
before it and averages the absolute difference over every pixel — one number per
frame pair, on a 0-255 scale. A cut was declared wherever that number exceeded a
fixed threshold of 70.

That threshold is unreachable on a great deal of ordinary footage, and the
reason is worth stating because it is not obvious. The measurement is a mean
over ~100,000 pixels, and averaging is brutal: a hard cut between two completely
different shots produces a *small* mean whenever the two happen to share overall
brightness. The number only grows large when total luminance swings — a dark
room to bright daylight. Saturated, high-contrast material clears 70 regularly;
a conventionally graded film never does. Measured on one 60-minute film, the
maximum difference anywhere in it was 31.7, so no cut could ever be declared and
the whole film came back as a single scene.

This is the same flaw ``modules/reaction_bursts.py`` documents in
``modules/audio_peaks.py``'s fixed -20 dBFS: an absolute threshold describes the
*mastering*, not the content, so one number cannot serve two videos.

The fix here is deliberately conservative, because the fixed threshold works
well on the material it was tuned for and changing that would be a regression
dressed as a fix. So the configured threshold is tried first, and only when it
has *evidently* failed — implausibly few cuts across the whole video — are the
cuts recomputed from the same differences against a threshold derived from that
video's own distribution. Footage where 70 works is untouched.

Nothing here decodes anything. The differences are computed by the detector's
existing pass, so recalibration costs one more sweep over an array of floats and
no second read of the video. That also makes the policy testable on synthetic
distributions in milliseconds — see ``tests/test_scene_cuts.py``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# Scales median absolute deviation onto the standard-deviation scale. Same
# constant, same reason, as modules/reaction_bursts.py.
MAD_TO_SIGMA = 1.4826

# How far above the video's own typical frame difference a cut has to sit, in
# robust deviations, when the threshold is derived rather than configured.
#
# Chosen against measurement, not taste. On a 5-minute stretch of a feature
# (median 14.2, MAD 4.8) the candidates gave:
#
#     z=3 -> 14.6 cuts/min   z=4 -> 4.8/min   z=5 -> 1.6/min   z=8 -> 0
#
# Feature films average roughly 7-15 cuts a minute, so z=3 sits at the typical
# rate and z=4 at the sparse end. Four is chosen anyway: a missed cut costs a
# chapter boundary that could have been finer, while a false one puts a
# boundary where nothing happened, and the second is the worse error for
# everything downstream of this.
DEFAULT_Z = 4.0

# Below this many cuts per minute across the whole video, detection is treated
# as having failed rather than as having found a slow edit. One cut every two
# minutes is far below any edited footage; a genuinely single-take recording
# also lands here, which is why `recalibrate` refuses when the distribution is
# too flat to distinguish a cut from sensor noise.
MIN_PLAUSIBLE_CUTS_PER_MINUTE = 0.5

# Fewer differences than this is not a distribution, and a median over it says
# nothing. Short clips keep the configured threshold.
MIN_SAMPLES = 120

# A derived threshold has to clear the video's own median by at least this
# factor. Without it, footage with almost no variation — a locked-off camera on
# a static subject — gets a threshold a hair above its noise floor and every
# sampled frame becomes a cut.
MIN_MEDIAN_RATIO = 1.25


def adaptive_threshold(diffs: Sequence[float], z: float = DEFAULT_Z) -> float:
    """Median plus ``z`` robust deviations of this video's own differences.

    Median and MAD rather than mean and standard deviation because the quantity
    is contaminated by what is being looked for: cuts are the high outliers, and
    a mean-based bar would be inflated by them enough to hide them.
    """
    values = np.asarray(diffs, dtype=np.float64)
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median))) * MAD_TO_SIGMA
    return median + z * max(mad, 1e-6)


def cuts_from_diffs(diffs: Sequence[float], threshold: float) -> np.ndarray:
    """Indices into ``diffs`` where a cut is declared."""
    values = np.asarray(diffs, dtype=np.float64)
    if values.size == 0:
        return np.zeros(0, dtype=int)
    return np.flatnonzero(values > float(threshold))


def suppress_adjacent(cuts: Sequence[int], diffs: Sequence[float],
                      min_gap: int) -> np.ndarray:
    """Collapse a run of neighbouring cuts to the single strongest one.

    A real cut is rarely one sampled frame. A dissolve spans several, and even a
    hard cut usually clears the threshold twice because the frames either side
    of it also differ from their neighbours. Left alone that produced scenes a
    tenth of a second long — measured on real footage, ``(42.7, 42.8)`` followed
    by ``(42.8, 42.9)`` — which are not shots and are noise in everything
    downstream.

    Strongest-first rather than first-past-the-post: the largest difference in a
    cluster is the frame the cut actually happened on, and taking the earliest
    instead puts the boundary a frame or two before it every time.
    """
    order = np.asarray(cuts, dtype=int)
    if order.size == 0 or min_gap <= 1:
        return np.sort(order)
    values = np.asarray(diffs, dtype=np.float64)
    kept: list[int] = []
    for i in sorted(order, key=lambda j: (-values[j], j)):
        if all(abs(i - j) >= min_gap for j in kept):
            kept.append(int(i))
    return np.array(sorted(kept), dtype=int)


def looks_undetected(cut_count: int, minutes: float) -> bool:
    """Is this result too sparse to be an edit rather than a failure?"""
    if minutes <= 0:
        return False
    return (cut_count / minutes) < MIN_PLAUSIBLE_CUTS_PER_MINUTE


def resolve(diffs: Sequence[float],
            minutes: float,
            threshold: float,
            z: float = DEFAULT_Z,
            min_gap: int = 1) -> tuple[np.ndarray, float, bool]:
    """``(cut_indices, threshold_used, recalibrated)``.

    Tries ``threshold`` first and keeps its answer whenever that answer is
    plausible. Recalibrates only when the configured threshold found almost
    nothing across the whole video *and* the distribution has enough spread for
    a derived threshold to mean something.

    ``min_gap`` is the shortest permitted shot, in samples — the caller knows
    the sampling rate and this module does not.
    """
    values = np.asarray(diffs, dtype=np.float64)
    cuts = suppress_adjacent(cuts_from_diffs(values, threshold), values, min_gap)
    if values.size < MIN_SAMPLES or not looks_undetected(len(cuts), minutes):
        return cuts, float(threshold), False

    derived = adaptive_threshold(values, z=z)
    median = float(np.median(values))

    # Refuse when the derived bar is not meaningfully above the video's own
    # normal: that is footage with no shot structure to find, and a threshold
    # sitting on the noise floor would invent one.
    if derived < median * MIN_MEDIAN_RATIO:
        return cuts, float(threshold), False

    # And refuse when it would only make things worse.
    derived_cuts = suppress_adjacent(cuts_from_diffs(values, derived), values, min_gap)
    if len(derived_cuts) <= len(cuts):
        return cuts, float(threshold), False

    return derived_cuts, float(derived), True


def scenes_from_cuts(cut_times: Sequence[float], duration: float) -> list:
    """``[(start, end), ...]`` covering ``[0, duration]`` with no gaps.

    Cut times landing outside the video, or duplicated by two adjacent sampled
    frames both clearing the threshold, would otherwise produce zero-length
    scenes that every consumer downstream has to special-case.
    """
    duration = float(duration)
    if duration <= 0:
        return []
    edges = sorted({round(float(t), 3) for t in cut_times if 0.0 < float(t) < duration})
    scenes = []
    start = 0.0
    for edge in edges:
        if edge > start:
            scenes.append((start, edge))
            start = edge
    scenes.append((start, duration))
    return scenes
