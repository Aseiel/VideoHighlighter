"""Compare the audio level measured during each labelled class of second.

Answers "was this video louder during one kind of labelled moment than another",
using whatever labels the caller supplies — detected classes, composed events,
anything keyed by second. This module knows none of their names and holds no
opinion about them; it counts seconds and compares distributions.

**Why the comparison is paired.** Level varies far more across a video than
between classes. Measured on an hour of material, the 60-second-smoothed level
moved 11 dB between passages while the difference between two classes was 1.7 dB
— so a class that happens to occur during loud passages will look "louder"
when what was actually measured is *where in the video it occurred*. Comparing
each stretch against nearby material of the other class holds the passage
roughly constant, which is the only version of this number worth printing.

**Why it reports its own resolution.** On that same hour the paired difference
was +1.33 dB across 10 pairs with a spread of 5.06 dB, which a sign test puts at
p=0.34 — the effect, if real, was about a third of the smallest difference the
material could resolve. A report that printed "louder during X" from that would
be inventing a finding. So :func:`summarise` computes the minimum detectable
difference alongside the observed one and says plainly when the second is
smaller than the first.

The per-event annotation is separate and needs no statistics at all: for a given
second, which classes were labelled there is a fact, and it is usually the thing
a reader actually wanted.
"""
from __future__ import annotations

from math import comb, sqrt
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

# A class needs at least this many labelled seconds before its median means
# anything. Below it the number is one or two moments wearing a statistic.
MIN_SECONDS = 8

# How far from a stretch to look for material of another class to compare it
# against. Two minutes: far enough to usually find some, close enough that the
# recording conditions have not changed.
PAIR_WINDOW = 120

# Seconds of gap that end a stretch.
STRETCH_GAP = 30

# Multiplier turning a spread into a minimum detectable difference at roughly
# the conventional two-sided 5% level for a paired comparison.
MDE_FACTOR = 2.8


def _stretches(seconds: Sequence[int], gap: int = STRETCH_GAP) -> list:
    """Contiguous runs of labelled seconds, split where the gap is large."""
    out: list = []
    start = prev = None
    for t in sorted(seconds):
        if prev is None or t - prev > gap:
            if start is not None:
                out.append((start, prev))
            start = t
        prev = t
    if start is not None:
        out.append((start, prev))
    return out


def _sign_test(diffs: Sequence[float]) -> float:
    """Two-sided exact sign test. Distribution-free, which suits 10 pairs."""
    n = sum(1 for d in diffs if d != 0)
    if not n:
        return 1.0
    k = sum(1 for d in diffs if d > 0)
    k = max(k, n - k)
    tail = sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def classes_at(labels_by_second: Mapping, second: int) -> list:
    """Which classes were labelled at one second. A fact, not an inference."""
    return sorted(str(name) for name in (labels_by_second.get(second) or ()))


def annotate(events: Iterable[Mapping], labels_by_second: Mapping) -> list:
    """Tag each event with the classes labelled at its peak second and across it.

    ``at_peak`` is what was labelled at the single loudest second; ``spanning``
    is everything labelled anywhere in the event. They differ when an event
    crosses a change, and the difference is worth showing rather than resolving.
    """
    out: list = []
    for e in events:
        peak = int(e.get("peak_second", e.get("start", 0)))
        spanning: set = set()
        for sec in range(int(e.get("start", peak)), int(e.get("end", peak)) + 1):
            spanning.update(classes_at(labels_by_second, sec))
        entry = dict(e)
        entry["classes_at_peak"] = classes_at(labels_by_second, peak)
        entry["classes_spanning"] = sorted(spanning)
        out.append(entry)
    return out


def peak_in_range(levels: Sequence[float],
                  labels_by_second: Mapping,
                  start: float,
                  end: float,
                  *,
                  video_median: Optional[float] = None) -> Optional[dict]:
    """The loudest second inside one range, and what was labelled there.

    Reported per clip because "the loudest moment in this clip was here, and
    this was on screen" is a fact about that clip, where the class medians are a
    fact about the whole video. A reader looking at one clip wants the first.

    ``video_median`` turns the level into something comparable -- +8 dB against
    the video's own middle says more than -13.5 dBFS does, since dBFS alone
    depends on how the file was mastered.
    """
    arr = np.asarray(levels, dtype=float)
    lo = max(0, int(start))
    hi = min(len(arr) - 1, int(np.ceil(end)))
    if not len(arr) or hi < lo:
        return None
    idx = lo + int(np.argmax(arr[lo:hi + 1]))
    out = {
        "second": idx,
        "timestamp": f"{idx // 60}:{idx % 60:02d}",
        "level_dbfs": round(float(arr[idx]), 2),
        "classes": classes_at(labels_by_second, idx),
    }
    if video_median is not None:
        out["vs_video_db"] = round(float(arr[idx]) - float(video_median), 2)
    return out


def _describe(levels: np.ndarray, seconds: Sequence[int]) -> dict:
    values = levels[np.asarray(sorted(seconds), dtype=int)]
    return {
        "seconds": int(len(values)),
        "median_dbfs": round(float(np.median(values)), 2),
        "p25_dbfs": round(float(np.percentile(values, 25)), 2),
        "p75_dbfs": round(float(np.percentile(values, 75)), 2),
    }


def _paired(levels: np.ndarray,
            a_secs: set,
            b_secs: set,
            *,
            pair_window: int) -> list:
    """Median level of each stretch of A minus nearby B, one entry per stretch."""
    diffs: list = []
    for start, end in _stretches(sorted(a_secs)):
        near = [t for t in b_secs
                if start - pair_window <= t <= end + pair_window
                and not (start <= t <= end)]
        inside = [t for t in range(start, end + 1) if t in a_secs]
        if len(inside) < 3 or len(near) < 3:
            continue
        diffs.append({
            "start": int(start),
            "end": int(end),
            "n_class": len(inside),
            "n_control": len(near),
            "difference_db": round(
                float(np.median(levels[np.array(inside)])
                      - np.median(levels[np.array(near)])), 2),
        })
    return diffs


def summarise(levels: Sequence[float],
              labels_by_second: Mapping,
              *,
              classes: Optional[Sequence[str]] = None,
              min_seconds: int = MIN_SECONDS,
              pair_window: int = PAIR_WINDOW) -> dict:
    """Per-class level, plus a paired comparison of the two best-represented.

    ``levels`` is one value per second (dBFS). ``labels_by_second`` maps a
    second to the class names labelled there. ``classes`` restricts the
    comparison; by default every class with enough seconds is described.
    """
    arr = np.asarray(levels, dtype=float)
    n = len(arr)
    if not n:
        return {"classes": [], "comparison": None, "loudest": None}

    by_class: dict = {}
    for sec, names in labels_by_second.items():
        sec = int(sec)
        if not 0 <= sec < n:
            continue
        for name in (names or ()):
            by_class.setdefault(str(name), set()).add(sec)

    if classes:
        wanted = {str(c) for c in classes}
        by_class = {k: v for k, v in by_class.items() if k in wanted}

    described = []
    for name, secs in by_class.items():
        if len(secs) < min_seconds:
            continue
        entry = {"name": name}
        entry.update(_describe(arr, secs))
        described.append(entry)
    described.sort(key=lambda d: d["median_dbfs"], reverse=True)

    comparison = None
    if len(described) >= 2:
        a, b = described[0]["name"], described[-1]["name"]
        a_secs, b_secs = by_class[a], by_class[b]
        pairs = _paired(arr, a_secs, b_secs - a_secs, pair_window=pair_window)
        diffs = [p["difference_db"] for p in pairs]
        if diffs:
            spread = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
            observed = float(np.median(diffs))
            mde = (MDE_FACTOR * spread / sqrt(len(diffs))) if spread else 0.0
            p = _sign_test(diffs)
            resolvable = bool(abs(observed) >= mde and p < 0.05)
            if resolvable:
                headline = (f"Louder during {a} than {b} "
                            f"by {abs(observed):.1f} dB")
                detail = (f"Measured across {len(diffs)} stretches compared with "
                          f"nearby {b} material, so the difference is not just "
                          f"where in the video each occurred. Sign test p={p:.3f}.")
            else:
                headline = f"No resolvable level difference between {a} and {b}"
                detail = (
                    f"The paired difference is {observed:+.2f} dB across "
                    f"{len(diffs)} stretches, but they scatter by {spread:.2f} dB, "
                    f"so the smallest difference this video could resolve is about "
                    f"{mde:.2f} dB. Sign test p={p:.3f}. Whatever the difference "
                    f"is, this much material cannot measure it — which is not the "
                    f"same as it being absent.")
            comparison = {
                "louder": a,
                "quieter": b,
                "pairs": len(diffs),
                "median_difference_db": round(observed, 2),
                "spread_db": round(spread, 2),
                "min_detectable_db": round(mde, 2),
                "p_value": round(p, 4),
                "resolvable": resolvable,
                "headline": headline,
                "detail": detail,
                "stretches": pairs,
            }

    # The single loudest labelled second, and what was labelled there. No
    # statistics involved and none needed: it is one measurement and one set of
    # labels, which is usually the thing a reader wanted before any aggregate.
    loudest = None
    labelled = sorted({s for secs in by_class.values() for s in secs})
    if labelled:
        idx = int(max(labelled, key=lambda s: arr[s]))
        loudest = {
            "second": idx,
            "timestamp": f"{idx // 60}:{idx % 60:02d}",
            "level_dbfs": round(float(arr[idx]), 2),
            "classes": classes_at(labels_by_second, idx),
        }

    return {"classes": described, "comparison": comparison, "loudest": loudest}
