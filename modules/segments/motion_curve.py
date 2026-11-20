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

import bisect
import math

STEP = 0.5

# A shot change is not movement. Leaving out the cut frame alone was not
# enough: the frames beside it still differ from each other by most of the
# picture, and 13 of the 15 highest values of one film's curve sat within
# half a second of a cut (2026-11-20). Its cuts faded over about a second,
# and with half a second the fade's tail still read as a peak. So nothing
# within a second of a cut counts - not on the curve, not as a peak, not as
# a sudden stop.
CUT_EDGE_SECONDS = 1.0

# A peak is the top of the curve, standing out from the seconds around it:
#  - the highest value within PEAK_WINDOW_SECONDS either side,
#  - at or above the video's own PEAK_PERCENTILE (judged against this video,
#    so resolution, mastering and an analysis copy do not move it),
#  - at least PEAK_PROMINENCE of its height above the lower of the two lows
#    within PEAK_BASE_SECONDS on either side (a bump, not a plateau's ripple).
PEAK_WINDOW_SECONDS = 2.0
PEAK_PERCENTILE = 90.0
PEAK_BASE_SECONDS = 5.0
PEAK_PROMINENCE = 0.3


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
    cuts = sorted(f / fps for f in exclude)
    rows = [(row[0] / fps, float(row[1])) for row in samples
            if row[0] not in exclude and not near_cut(row[0] / fps, cuts)]
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


def near_cut(t, cuts, edge=CUT_EDGE_SECONDS):
    """Whether time ``t`` is within ``edge`` seconds of a cut (``cuts`` sorted)."""
    if not cuts:
        return False
    i = bisect.bisect_left(cuts, t)
    return any(0 <= j < len(cuts) and abs(cuts[j] - t) <= edge for j in (i - 1, i))


def cut_times(scenes):
    """Cut times from a cache's ``scenes`` (each one's start but the first)."""
    out = []
    for s in scenes or []:
        start = s.get("start") if isinstance(s, dict) else (s[0] if s else None)
        if start:
            out.append(float(start))
    return sorted(out)


def find_peaks(curve, cuts=()):
    """Times of the curve's peaks (see the PEAK_* settings), never at a cut.

    Each peak is marked at the middle of its highest bin, so the mark sits at
    the top of the movement - not at its end, where the old rule (now
    :func:`stops`' job) put it.
    """
    values, step = _values(curve)
    if not values:
        return []
    cuts = sorted(cuts or ())
    v = [0.0 if near_cut((i + 0.5) * step, cuts) else float(x) for i, x in enumerate(values)]
    live = sorted(x for x in v if x > 0)
    if not live:
        return []
    floor = live[min(len(live) - 1, int(len(live) * PEAK_PERCENTILE / 100.0))]
    w = max(1, int(round(PEAK_WINDOW_SECONDS / step)))
    b = max(w, int(round(PEAK_BASE_SECONDS / step)))
    out = []
    i, n = 0, len(v)
    while i < n:
        x = v[i]
        if x > 0 and x >= floor and x == max(v[max(0, i - w):i + w + 1]):
            left = min(v[max(0, i - b):i] or [x])
            right = min(v[i + 1:i + b + 1] or [x])
            if x - max(left, right) >= PEAK_PROMINENCE * x:
                out.append(round((i + 0.5) * step, 3))
                i += w + 1          # one mark per bump; a flat top is not two peaks
                continue
        i += 1
    return out


def drop_near_cuts(times, cuts, edge=CUT_EDGE_SECONDS):
    """``times`` without the ones within ``edge`` of a cut."""
    cuts = sorted(cuts or ())
    return [t for t in times or [] if not near_cut(float(t), cuts, edge)]


def upgrade(cache):
    """A cache written before peaks meant peaks, read the new way, in place.

    Until 2026-11-20 ``motion_peaks`` held movement that ends in stillness
    (now ``motion_stops``, "Sudden stop"). A cache with a curve but no
    ``motion_stops`` is from then: its old peaks become stops (minus any on a
    cut), and its peaks are found on its own curve. A cache without a curve
    is left as it is - there is nothing to find peaks on. Idempotent.
    """
    if not isinstance(cache, dict) or "motion_stops" in cache:
        return cache
    curve = cache.get("motion_curve")
    if not isinstance(curve, dict) or not curve.get("values"):
        return cache
    cuts = cut_times(cache.get("scenes"))
    cache["motion_stops"] = drop_near_cuts(cache.get("motion_peaks") or [], cuts)
    cache["motion_peaks"] = find_peaks(curve, cuts)
    return cache


OUTLIER_PERCENTILE = 99.5
OUTLIER_FACTOR = 3.0


def display_top(values):
    """The level a curve is drawn at full height for: its highest value that
    is not a freak outlier.

    The curve is divided by the video's own maximum, so one freak bin flattens
    everything else. Seen 2026-11-20: a film whose bin at 0:02 (a flash from
    black) was 1.0 while every other bin was 0.05 or less, so the whole curve
    drew under 5 % of the row and looked like a video with no movement at
    all. A value more than ``OUTLIER_FACTOR`` times the 99.5th percentile is
    left out here and drawn clipped at full height; an ordinary video's
    biggest moment is well within that, so its curve draws as before.
    """
    vals = sorted(v for v in (values or []) if v > 0)
    if not vals:
        return 1.0
    p = vals[min(len(vals) - 1, int(len(vals) * OUTLIER_PERCENTILE / 100.0))]
    kept = [v for v in vals if v <= OUTLIER_FACTOR * p]
    return kept[-1] if kept else vals[-1]


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
