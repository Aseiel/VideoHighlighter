"""Rank what was on screen against the rest of the same video.

The report can already say a clip outscored 97% of the video. That is a fact
about the weight table, and it answers "did the arithmetic pick this" rather
than "what is different about this moment". A reader who disagrees with a pick
is disagreeing about the second question, and nothing in the record spoke to it.

This module measures the *subjects* instead, and every number it produces is a
comparison against the same video's own distribution:

* **Frame share** — how much of the frame a detected class occupies at its
  largest here, against how large it gets in the video's other stretches.
* **Relative scale** — that same box divided by a co-present box of another
  class. Frame share alone cannot tell "the subject is larger" from "the camera
  is closer"; a ratio between two things in the *same frame* can, because moving
  the camera scales both. This is the only size claim here worth much, and it is
  reported with the reference class named so the reader can judge it.
* **Prevalence** — what share of the video's detected seconds carry this class
  at all. A class that shows up in 3% of the file is itself why a moment stands
  out, whatever it scored.
* **Persistence** — how much of the clip the class is actually present for,
  which is the difference between a subject and a single-frame flicker that a
  percentile would otherwise dress up as a finding.
* **Expression standing** — which built-in expression class dominates the clip,
  how strongly it read against the video's other stretches, and how far the
  clip's mix departs from the whole file's.

Three disciplines hold throughout.

**Like is compared with like.** A clip is described by its strongest moment, so
ranking that against the video's *individual seconds* is biased upward by
construction: any twelve-second clip contains a close-up, and every clip in a
report comes out in the eightieth percentile. Every ranking here is therefore
against windows of the clip's own length — "larger than in 90% of this video's
other twelve-second stretches" is a claim that means something, and the naive
version is a claim that every clip can make.

**Sample counts travel with every percentile.** Windows overlap, so the number
of them overstates how much independent evidence there is; the record reports
the count divided back down into whole stretches, and :data:`MIN_STRETCHES`
marks where a rank becomes reportable. Nothing is suppressed — the caller sees
the count and can decide — but the flag means the prose layer does not have to
re-derive the judgement.

**This module computes, it does not characterise.** No thresholds for "unusual",
no adjectives, no ordering by significance beyond a stable sort. What is worth
saying out loud is :mod:`modules.highlight_prose`'s decision, exactly as it is
for every other measurement in the report. Keeping the split means a threshold
lives in one place instead of two that drift.

Content is not this module's business: it compares a class against itself and
holds no vocabulary of its own. Whatever the detector was taught to find, the
same arithmetic applies.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

from modules.highlight_report import percentile_rank

# Below this many comparable stretches, a percentile is an artefact of the
# sample rather than a property of the video. Claims are still computed and
# returned — with the count beside them — but flagged so the sentence layer can
# decline to make them.
MIN_STRETCHES = 6

# How often two classes must have shared a frame before a ratio between them is
# ranked against anything. Same reasoning, applied to the pairing.
MIN_PAIR_SECONDS = 6

# How many classes to describe per clip. A clip with eleven detected classes has
# no story; the two or three most unusual do.
MAX_SUBJECTS = 3


def _area(box: Sequence[float]) -> float:
    """Fraction of the frame a normalised ``[x, y, w, h]`` box covers."""
    if box is None or len(box) < 4:
        return 0.0
    w, h = float(box[2]), float(box[3])
    return max(0.0, w) * max(0.0, h)


def _largest_per_class(boxes: Iterable[Mapping]) -> dict:
    """``{class: (area, confidence)}`` for one second, keeping the biggest box.

    One second can hold several boxes of a class. The largest is the one a
    viewer's eye is on and the one a size comparison is about, so a class is
    represented by its biggest instance rather than by a mean that no frame
    ever actually looked like.
    """
    best: dict = {}
    for box in boxes or ():
        name = str(box.get("name") or "")
        if not name:
            continue
        area = _area(box.get("box"))
        if area <= 0:
            continue
        confidence = float(box.get("confidence") or 0.0)
        current = best.get(name)
        if current is None or area > current[0]:
            best[name] = (area, confidence)
    return best


def _normalise_expressions(expressions: Optional[Mapping]) -> dict:
    """``{second: (label, confidence)}`` from either shape the app produces.

    ``face_scan`` stores ``{sec: {"label": ..., "confidence": ...}}`` and
    ``best_by_second`` hands out ``{sec: (label, confidence)}``. Accepting both
    means the caller passes whichever it happens to be holding instead of
    converting at the call site and getting it wrong once.
    """
    out: dict = {}
    for sec, value in (expressions or {}).items():
        if isinstance(value, Mapping):
            label = str(value.get("label") or "")
            confidence = float(value.get("confidence") or 0.0)
        elif isinstance(value, (tuple, list)) and len(value) >= 2:
            label, confidence = str(value[0] or ""), float(value[1] or 0.0)
        else:
            continue
        if label:
            out[int(sec)] = (label, confidence)
    return out


def build_distributions(bbox_cache: Optional[Iterable[Mapping]] = None,
                        expressions: Optional[Mapping] = None) -> dict:
    """Everything a clip will be compared against, indexed once for the video.

    What is stored is deliberately compact — the per-second detections, and
    counts. The per-second *series* a ranking needs are derived from these on
    demand and memoised, because materialising one for every ordered pair of
    classes up front is hundreds of megabytes on a long file and all but two of
    them are never looked at.

    ``bbox_cache`` is the detector's own cache, with boxes already normalised to
    0..1 — which is why nothing here needs the frame size.
    """
    by_second: dict = {}
    for record in bbox_cache or ():
        sec = int(float(record.get("timestamp", 0)))
        names = record.get("objects") or []
        boxes = record.get("bboxes") or []
        confs = record.get("confidences") or []
        bucket = by_second.setdefault(sec, [])
        for i, box in enumerate(boxes):
            bucket.append({
                "name": str(names[i]) if i < len(names) else "",
                "box": box,
                "confidence": confs[i] if i < len(confs) else 0.0,
            })

    largest: dict = {}
    seconds_with: dict = {}
    pair_seconds: dict = {}
    for sec, boxes in by_second.items():
        best = _largest_per_class(boxes)
        if not best:
            continue
        largest[sec] = best
        for name in best:
            seconds_with[name] = seconds_with.get(name, 0) + 1
        for name in best:
            for other in best:
                if other != name:
                    pair_seconds[(name, other)] = pair_seconds.get((name, other), 0) + 1

    expression_by_second = _normalise_expressions(expressions)
    expression_counts: dict = {}
    for _sec, (label, _confidence) in expression_by_second.items():
        expression_counts[label] = expression_counts.get(label, 0) + 1

    span = 0
    if largest:
        span = max(span, max(largest) + 1)
    if expression_by_second:
        span = max(span, max(expression_by_second) + 1)

    return {
        "largest": largest,
        "seconds_with": seconds_with,
        "pair_seconds": pair_seconds,
        "detected_seconds": len(largest),
        "span": span,
        "expressions": expression_by_second,
        "expression_counts": expression_counts,
        "expression_seconds": len(expression_by_second),
        # Memo for the derived per-second series. Populated on demand so a class
        # nobody asks about costs nothing.
        "_series": {},
    }


def _series(distributions: Mapping, key) -> np.ndarray:
    """A per-second array for one measurement, built once and remembered.

    ``key`` is ``("area", name)``, ``("ratio", name, reference)`` or
    ``("expression", label)``. Zero means "not measurable this second", which is
    also how the ranking spots the seconds it must not compare against.
    """
    cache = distributions.setdefault("_series", {})
    if key in cache:
        return cache[key]

    span = int(distributions.get("span") or 0)
    series = np.zeros(max(0, span), dtype=float)
    largest = distributions.get("largest") or {}

    if key[0] == "area":
        name = key[1]
        for sec, best in largest.items():
            if name in best and 0 <= sec < span:
                series[sec] = best[name][0]
    elif key[0] == "ratio":
        name, reference = key[1], key[2]
        for sec, best in largest.items():
            if name in best and reference in best and 0 <= sec < span:
                reference_area = best[reference][0]
                if reference_area > 0:
                    series[sec] = best[name][0] / reference_area
    elif key[0] == "expression":
        label = key[1]
        for sec, (found, confidence) in (distributions.get("expressions") or {}).items():
            if found == label and 0 <= sec < span:
                series[sec] = confidence

    cache[key] = series
    return series


def _rank_in_stretches(series: np.ndarray, length: int, value: float) -> tuple:
    """``(percentile, stretches)`` for ``value`` among same-length windows.

    The comparison a clip deserves. Its own figure is a maximum over ``length``
    seconds, so the distribution it is ranked against has to be maxima over
    ``length`` seconds too — otherwise a long clip beats a short one for no
    reason but its length, and every row of the report reads as exceptional.

    Windows slide one second at a time, so the rank does not depend on where a
    fixed grid happened to fall. They overlap heavily as a result, and counting
    them as evidence would badly overstate it — so the count returned is divided
    back down into whole non-overlapping stretches, which is the number a reader
    would recognise as "how many chances did this video have to do better".

    Windows in which the measurement never occurs are dropped. Ranking a clip's
    subject against stretches that do not contain it would put every appearance
    of a rare class at the top of its own distribution.
    """
    length = max(1, int(length))
    if series.size == 0:
        return 0.0, 0
    if length >= series.size:
        maxima = np.asarray([series.max()])
    elif length == 1:
        maxima = series
    else:
        maxima = np.lib.stride_tricks.sliding_window_view(series, length).max(axis=1)

    comparable = np.sort(maxima[maxima > 0])
    if comparable.size == 0:
        return 0.0, 0
    stretches = max(1, int(round(comparable.size / float(length))))
    return percentile_rank(comparable, float(value)), stretches


def _relative_for(distributions: Mapping, name: str, window: Sequence[int],
                  length: int) -> Optional[dict]:
    """``name`` at its largest *relative to something else in the same frame*.

    The reference class is whichever co-present class the video observed
    alongside ``name`` most often: a ratio is only as trustworthy as the
    distribution it is ranked against, and the most frequent pairing has the
    fullest one. Ties break on the name so a re-run cannot reorder the report.

    The second reported is the one with the highest *ratio*, not the biggest
    box. Choosing by box area would land on whichever second the camera happened
    to be nearest — which is the confound this entire measurement exists to
    remove, and taking it as the answer would quietly reintroduce it.
    """
    largest = distributions.get("largest") or {}
    pair_seconds = distributions.get("pair_seconds") or {}

    candidates: dict = {}
    for sec in window:
        best = largest.get(sec) or {}
        if name not in best:
            continue
        for other in best:
            if other == name:
                continue
            seen = int(pair_seconds.get((name, other), 0))
            if seen >= MIN_PAIR_SECONDS:
                candidates[other] = seen
    if not candidates:
        return None
    reference = max(sorted(candidates), key=lambda key: candidates[key])

    best_ratio = None
    for sec in window:
        best = largest.get(sec) or {}
        if name not in best or reference not in best:
            continue
        reference_area = best[reference][0]
        if reference_area <= 0:
            continue
        ratio = best[name][0] / reference_area
        if best_ratio is None or ratio > best_ratio[0]:
            best_ratio = (ratio, sec)
    if best_ratio is None:
        return None

    ratio, sec = best_ratio
    series = _series(distributions, ("ratio", name, reference))
    percentile, stretches = _rank_in_stretches(series, length, ratio)
    typical = series[series > 0]
    return {
        "reference": reference,
        "at": int(sec),
        "ratio": round(float(ratio), 2),
        # The same comparison in the unit people actually picture. A box with
        # twice the area is only about 1.4 times as long, and "2.0x" left
        # unqualified is read as the second thing every time — so both are
        # reported and the sentence names which is which. Ranking is unaffected:
        # a square root is monotonic, so the percentile is identical either way.
        "linear_ratio": round(float(math.sqrt(ratio)), 2),
        "percentile": percentile,
        "median": round(float(np.median(typical)), 2) if typical.size else 0.0,
        "seconds_together": int(pair_seconds.get((name, reference), 0)),
        "stretches": stretches,
        "stretch_seconds": int(length),
        "enough_samples": bool(stretches >= MIN_STRETCHES),
    }


def compare_segment(distributions: Mapping,
                    start: float,
                    end: float,
                    *,
                    max_subjects: int = MAX_SUBJECTS) -> dict:
    """How this clip's subjects and expression stand against the whole video."""
    lo, hi = int(start), max(int(start) + 1, int(np.ceil(end)))
    window = list(range(lo, hi))
    length = max(1, hi - lo)

    largest = distributions.get("largest") or {}
    seconds_with = distributions.get("seconds_with") or {}
    detected_seconds = int(distributions.get("detected_seconds") or 0)

    peak_instance: dict = {}
    seconds_present: dict = {}
    for sec in window:
        best = largest.get(sec) or {}
        for name, (area, confidence) in best.items():
            seconds_present[name] = seconds_present.get(name, 0) + 1
            current = peak_instance.get(name)
            if current is None or area > current[1]:
                peak_instance[name] = (sec, area, confidence)

    subjects = []
    for name, (sec, area, confidence) in peak_instance.items():
        percentile, stretches = _rank_in_stretches(
            _series(distributions, ("area", name)), length, area)
        entry = {
            "name": name,
            "at": int(sec),
            "frame_share": round(area * 100.0, 2),
            "frame_share_percentile": percentile,
            "stretches": stretches,
            "stretch_seconds": int(length),
            "enough_samples": bool(stretches >= MIN_STRETCHES),
            "detections": int(seconds_with.get(name, 0)),
            "clip_presence_pct": round(
                100.0 * seconds_present.get(name, 0) / length, 1),
            "confidence": round(float(confidence), 3),
        }
        if detected_seconds:
            entry["prevalence_pct"] = round(
                100.0 * seconds_with.get(name, 0) / detected_seconds, 1)

        # Proximity-invariant size, when something in the same frame can serve
        # as a yardstick. Measured over the whole clip in its own right, so it
        # can land on a different second from the frame-share figure above.
        relative = _relative_for(distributions, name, window, length)
        if relative:
            entry["relative"] = relative
        subjects.append(entry)

    # Most unusual first, by the strongest ranking the class has. Sorted rather
    # than filtered: the caller keeps its own counsel about what is worth
    # printing, and a stable order is what lets it take the first few.
    def _rank(entry: Mapping) -> tuple:
        relative = entry.get("relative") or {}
        return (max(float(entry.get("frame_share_percentile") or 0.0),
                    float(relative.get("percentile") or 0.0)),
                float(entry.get("frame_share") or 0.0),
                entry["name"])

    subjects.sort(key=_rank, reverse=True)

    out: dict = {}
    if subjects:
        out["subjects"] = subjects[:max_subjects]
    expression = _compare_expression(distributions, window, length)
    if expression:
        out["expression"] = expression
    return out


def _compare_expression(distributions: Mapping, window: Sequence[int],
                        length: int) -> dict:
    """Which expression the clip read as, and how far that is from the video's.

    Dominance is decided on summed confidence rather than a count of seconds.
    Two labels splitting a clip evenly is common, and a count breaks that tie by
    whichever the classifier happened to emit one more of; summing confidence
    breaks it by which reading was actually stronger.

    The classifier behind this has five coarse classes, no notion of intensity,
    and returns a label for every face it is handed. So the honest reading of
    everything here is "the classifier reported this label more strongly or more
    often than elsewhere in the file", and it is the caller's job to phrase it
    that way rather than as a claim about the person on screen.
    """
    by_second = distributions.get("expressions") or {}
    if not by_second:
        return {}

    clip_totals: dict = {}
    clip_counts: dict = {}
    clip_best: dict = {}
    for sec in window:
        found = by_second.get(int(sec))
        if not found:
            continue
        label, confidence = found
        clip_totals[label] = clip_totals.get(label, 0.0) + confidence
        clip_counts[label] = clip_counts.get(label, 0) + 1
        if confidence > clip_best.get(label, (0, 0.0))[1]:
            clip_best[label] = (int(sec), confidence)
    if not clip_totals:
        return {}

    label = max(clip_totals, key=lambda k: clip_totals[k])
    at, confidence = clip_best.get(label, (0, 0.0))
    clip_read = sum(clip_counts.values())

    video_counts = distributions.get("expression_counts") or {}
    video_read = int(distributions.get("expression_seconds") or 0)
    clip_share = 100.0 * clip_counts.get(label, 0) / max(1, clip_read)
    video_share = 100.0 * video_counts.get(label, 0) / max(1, video_read)

    percentile, stretches = _rank_in_stretches(
        _series(distributions, ("expression", label)), length, confidence)

    return {
        "label": label,
        "at": int(at),
        "confidence": round(float(confidence), 3),
        "confidence_percentile": percentile,
        "stretches": stretches,
        "stretch_seconds": int(length),
        "seconds_read": int(clip_read),
        "clip_share_pct": round(clip_share, 1),
        "video_share_pct": round(video_share, 1),
        # How much more of this clip is that label than of the video. The one
        # number that answers "is this different here", rather than "is this
        # what the video mostly is". A share, not a maximum, so unlike the
        # figures above it needs no windowing to be fair.
        "lift": round(clip_share / video_share, 2) if video_share > 0 else 0.0,
        "label_samples": int(video_counts.get(label, 0)),
        "enough_samples": bool(stretches >= MIN_STRETCHES),
        **({"video_dominant": max(video_counts, key=lambda k: video_counts[k]),
            "video_dominant_share_pct": round(
                100.0 * max(video_counts.values()) / max(1, video_read), 1)}
           if video_counts else {}),
    }
