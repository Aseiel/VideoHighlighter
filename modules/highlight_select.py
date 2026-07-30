"""Turn a per-second score curve into the segments that get cut.

This is the last step of the highlight pipeline and the one that decides what
the user actually sees, but it lived inline in ``pipeline.py`` surrounded by
video decoding, so it could only be exercised by running a real video end to
end. Nothing about the decision needs a video: it is a score array, a clip
length, and a duration budget. Pulling it out makes it testable with a numpy
array in milliseconds, and keeps the shared ``pipeline.py`` from growing.

The selection is greedy. Seconds are ranked by score (ties broken by detection
confidence), each one is expanded into a fixed-length window around itself,
windows that would overlap an already-taken one are skipped, and the loop stops
when the duration budget is spent. Because the ranking is global, the result
follows the score wherever it concentrates — if one stretch of the video scores
highest, the whole cut can come from that stretch.

``coverage`` is the control over that. The video is divided into as many buckets
as there are clips in the budget, and each bucket is given a ceiling on how much
of the cut it may supply. At ``0.0`` the ceiling is the whole budget, which is no
constraint at all and reproduces the pure ranking; at ``1.0`` it is one bucket's
fair share, so the cut is spread evenly and the whole span of the video is
represented. Values in between interpolate. Anything the capped pass leaves
unspent is filled by a second, uncapped pass, so a coverage setting can never
make the cut come up short of its target.

Because the choice is only arithmetic over an array, it can be re-made cheaply
after the fact. That is what ``exclude`` and :func:`swap_segment` are for: a user
who does not want a particular moment gets a different one for the cost of a
sort, with no detector re-run and no frame decoded.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np


def peak_confidence_by_sec(
    detections_by_sec: Mapping[int, Sequence[tuple[str, float]]],
) -> dict[int, float]:
    """Reduce ``{sec: [(name, confidence), ...]}`` to ``{sec: best confidence}``.

    Confidence only breaks ties between equally scoring seconds, so the highest
    one at a second is all the selection needs.
    """
    peaks = {}
    for sec, detections in detections_by_sec.items():
        if not detections:
            continue
        peaks[int(sec)] = max(confidence for _, confidence in detections)
    return peaks


def rank_seconds(
    score: np.ndarray,
    *,
    duration_mode: str,
    confidence_by_sec: Mapping[int, float] | None = None,
) -> np.ndarray:
    """Second indices, best candidate first.

    ``EXACT`` mode ranks every second, including zero-scoring ones, because the
    cut has to reach a fixed length even on a video where little was detected.
    ``MAX`` mode is free to return less than the budget, so it never considers a
    second that scored nothing.
    """
    if duration_mode == "EXACT":
        candidates = np.arange(len(score))
    else:
        candidates = np.where(score > 0)[0]

    confidence_by_sec = confidence_by_sec or {}
    confidences = np.array(
        [confidence_by_sec.get(int(sec), 0.0) for sec in candidates],
        dtype=float,
    )

    # lexsort's last key is the primary one: score first, confidence as the
    # tie-break. Negated because lexsort is ascending.
    order = np.lexsort((-confidences, -score[candidates]))
    return candidates[order]


def bucket_count(*, clip_time: int, target_duration: float) -> int:
    """How many buckets to spread the cut over: one per clip that fits."""
    if clip_time <= 0:
        return 1
    return max(1, int(target_duration // clip_time))


def bucket_ceiling(
    *, coverage: float, clip_time: int, target_duration: float
) -> float:
    """The most any one bucket may contribute to the cut.

    ``coverage`` is clamped to ``0.0..1.0`` and interpolates between the whole
    budget (no constraint) and a single bucket's fair share (an even spread).
    """
    coverage = min(1.0, max(0.0, float(coverage)))
    buckets = bucket_count(clip_time=clip_time, target_duration=target_duration)
    fair_share = target_duration / buckets
    return fair_share + (1.0 - coverage) * (target_duration - fair_share)


def select_fixed_window_segments(
    score: np.ndarray,
    *,
    video_duration: float,
    clip_time: int,
    target_duration: float,
    duration_mode: str = "MAX",
    confidence_by_sec: Mapping[int, float] | None = None,
    coverage: float = 0.0,
    exclude: Optional[Sequence[tuple]] = None,
) -> list[tuple[float, float]]:
    """Pick non-overlapping ``clip_time``-long windows until the budget is spent.

    Returns ``(start, end)`` pairs in chronological order. The final segment may
    be shorter than ``clip_time`` when only part of the budget is left, which is
    what makes ``EXACT`` mode land on its target exactly.

    ``coverage`` trades the best-scoring moments against representing the whole
    video; see the module docstring. ``0.0`` is the pure ranking.

    ``exclude`` marks time the selection may not touch — moments the user has
    already turned down, or the rest of a cut being kept while one clip is
    replaced. Excluded time is not part of the result and does not spend any of
    the budget, so banning a moment yields a different one rather than a shorter
    highlight.
    """
    ranked = rank_seconds(
        score, duration_mode=duration_mode, confidence_by_sec=confidence_by_sec
    )

    segments: list[tuple[float, float]] = []
    used_seconds: set[int] = set()
    for start, end in exclude or ():
        used_seconds.update(range(int(start), int(np.ceil(end))))

    buckets = bucket_count(clip_time=clip_time, target_duration=target_duration)
    ceiling = bucket_ceiling(
        coverage=coverage, clip_time=clip_time, target_duration=target_duration
    )
    spent_per_bucket = [0.0] * buckets

    def bucket_of(sec: int) -> int:
        if video_duration <= 0:
            return 0
        return min(buckets - 1, int(sec / video_duration * buckets))

    def fill(*, respect_ceiling: bool) -> None:
        for sec in ranked:
            sec = int(sec)
            if sec in used_seconds:
                continue

            bucket = bucket_of(sec)

            # Centre the window on the peak second, then slide it back inside
            # the video if centring pushed it past the end.
            start = max(0, sec - clip_time // 2)
            end = min(video_duration, start + clip_time)
            if end - start < clip_time and start > 0:
                start = max(0, end - clip_time)

            if any(s in used_seconds for s in range(int(start), int(end))):
                continue

            remaining = target_duration - sum(e - s for s, e in segments)
            if remaining <= 0:
                return
            if end - start > remaining:
                end = start + remaining

            if respect_ceiling and spent_per_bucket[bucket] + (end - start) > ceiling:
                continue

            segments.append((start, end))
            spent_per_bucket[bucket] += end - start
            used_seconds.update(range(int(start), int(end)))

            if sum(e - s for s, e in segments) >= target_duration:
                return

    fill(respect_ceiling=True)
    # Buckets can run out of candidates before the budget does — most obviously
    # in EXACT mode, which must reach its target however the score is shaped.
    # Whatever the capped pass could not place is placed by the ranking alone.
    if sum(e - s for s, e in segments) < target_duration:
        fill(respect_ceiling=False)

    segments.sort(key=lambda seg: seg[0])
    return segments


def swap_segment(
    score: np.ndarray,
    *,
    segments: Sequence[tuple],
    index: int,
    video_duration: float,
    clip_time: int,
    duration_mode: str = "MAX",
    confidence_by_sec: Mapping[int, float] | None = None,
    rejected: Optional[Sequence[tuple]] = None,
) -> Optional[list[tuple[float, float]]]:
    """Replace one clip with the best moment that did not make the cut.

    Returns the full segment list with ``segments[index]`` swapped out, or
    ``None`` when the video has nothing else to offer — the caller needs to tell
    "here is another one" apart from "there is no other one", and an unchanged
    list cannot say that.

    The replacement takes the same duration as the clip it replaces, so swapping
    never changes the length of the highlight. Everything else the user is
    keeping is passed as ``exclude``, along with ``rejected`` — the moments
    already turned down, which the caller accumulates so that swapping twice
    offers a third moment rather than the first one again.

    This costs a sort and a scan of an array that is already in memory. No
    detector runs, no frame is decoded: the score is the same score, and only the
    choice made from it changes.
    """
    if not 0 <= index < len(segments):
        raise IndexError(f"no segment at index {index}")

    keeping = [(float(s), float(e)) for i, (s, e) in enumerate(segments) if i != index]
    dropped = (float(segments[index][0]), float(segments[index][1]))

    replacement = select_fixed_window_segments(
        score,
        video_duration=video_duration,
        clip_time=clip_time,
        target_duration=dropped[1] - dropped[0],
        duration_mode=duration_mode,
        confidence_by_sec=confidence_by_sec,
        # The clip being replaced is banned along with everything kept, so the
        # "alternative" is never the thing the user just turned down.
        exclude=keeping + [dropped] + list(rejected or ()),
    )
    if not replacement:
        return None

    return sorted(keeping + replacement, key=lambda seg: seg[0])
