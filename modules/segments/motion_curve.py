"""How much motion there was, second by second, and how strong each peak was.

The motion pass keeps only the timestamps that crossed its thresholds. That is
enough to score with, but on the timeline it made every peak look the same: a
barely-there twitch and the biggest moment in the video were the same orange
box. So the pass also keeps how much changed between every pair of sampled
frames -- the mean change per pixel, *before* any threshold, so ordinary
movement draws as a continuous line across the whole video rather than as
zero between spikes -- binned, as the cache's `motion_curve`:

    {"step": 0.5, "values": [0.0, 0.031, ..., 1.0]}

One value per `step` seconds, the bin's *maximum* (a mean would average a
one-frame spike into its quiet neighbours, and the spike is the point),
divided by the video's own maximum so the file never depends on resolution or
the detector's area units.

A peak's strength is read off this curve rather than kept beside it. The
detector's own ratio is relative to the *scene's* average, so a small spike in
a still scene outranks a huge one in a busy scene — the opposite of "where is
the biggest moment in this video". Against the whole video's mean, and from
the one array, the ranking means what it says and there is no second list to
drift out of step with `motion_peaks`.
"""
from __future__ import annotations

import math

STEP = 0.5


def build_curve(samples, fps, duration=0.0, step=STEP, exclude_frames=()):
    """Bin per-frame motion levels into a cache-ready curve.

    `samples` yields tuples whose first two fields are (frame index, level).
    `exclude_frames` are dropped: the detector passes its scene cuts, where
    the whole picture changes at once and would otherwise draw the tallest
    "motion" in the video. Returns None when there is nothing to bin, so a
    cancelled or empty pass caches no curve rather than a flat one that claims
    the video is still.
    """
    if not fps or fps <= 0 or step <= 0:
        return None
    exclude = set(exclude_frames or ())
    rows = [(row[0] / fps, float(row[1])) for row in samples
            if row[0] not in exclude]
    if not rows:
        return None
    last = max(t for t, _ in rows)
    n_bins = int(math.ceil(max(float(duration or 0.0), last + 1e-9) / step))
    values = [0.0] * max(1, n_bins)
    for t, level in rows:
        i = min(len(values) - 1, int(t / step))
        if level > values[i]:
            values[i] = level
    top = max(values)
    if top > 0:
        values = [round(v / top, 3) for v in values]
    return {"step": float(step), "values": values}


def _values(curve):
    if not isinstance(curve, dict):
        return None, None
    values = curve.get("values") or []
    step = float(curve.get("step") or 0)
    if not values or step <= 0:
        return None, None
    return values, step


def mean_level(curve):
    """The video's average motion — what a peak's ×N is measured against."""
    values, _ = _values(curve)
    if not values:
        return 0.0
    return sum(values) / len(values)


def strength(curve, start, end=None):
    """How many times the video's average motion the strongest moment in
    [start, end] reached. None when there is no curve to read, or the video
    has no motion at all to compare against."""
    values, step = _values(curve)
    if not values:
        return None
    avg = mean_level(curve)
    if avg <= 0:
        return None
    end = start if end is None else end
    lo = max(0, int(start / step))
    hi = min(len(values) - 1, int(max(start, end - 1e-9) / step))
    if lo > hi:
        return None
    return max(values[lo:hi + 1]) / avg


def level_at(curve, t):
    """The curve's value at time `t` (0 outside it)."""
    values, step = _values(curve)
    if not values:
        return 0.0
    i = int(t / step)
    return values[i] if 0 <= i < len(values) else 0.0


def peak_time(curve, start, end):
    """The middle of the strongest bin in [start, end] -- where to mark a peak
    on the curve. `start` when there is no curve."""
    values, step = _values(curve)
    if not values:
        return start
    lo = max(0, int(start / step))
    hi = min(len(values) - 1, int(max(start, end - 1e-9) / step))
    if lo > hi:
        return start
    best = max(range(lo, hi + 1), key=lambda i: (values[i], -i))
    return (best + 0.5) * step


def ranks(strengths, top_n):
    """1-based rank for the `top_n` strongest entries, keyed by position.

    Entries without a strength are never ranked. Ties keep timeline order, so
    the earlier of two equal peaks gets the better number.
    """
    order = sorted((i for i, s in enumerate(strengths) if s is not None),
                   key=lambda i: (-strengths[i], i))
    return {i: rank for rank, i in enumerate(order[:max(0, top_n)], start=1)}
