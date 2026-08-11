"""How unusual a combination of marks is, measured against the video's own rate.

Every other explanation in the report describes one moment. This one answers the
question that comes after: *is that combination actually unusual here, or does
this video do it all the time?*

It is worth having because the answer is frequently "all the time", and nothing
else in the report can say so. A clip where something arrives on screen,
movement bursts, the sound peaks and the expression reading turns looks like a
finding — four independent signals agreeing. If the video produces that same
combination in most of its stretches, it is not a finding about the clip, it is
a description of the footage, and the four signals agreeing tells you only that
they always agree here.

The measurement is a base rate and nothing more:

    of the video's comparable stretches, how many carry at least these marks?

That is the one form of "how likely is this" the material can support. It is a
frequency over stretches of this file, computed the same way for every clip, and
it says nothing whatever about *what* the combination means — no set of these
marks implies a cause, an intention or an experience, and the rarity of a
combination is not evidence for any particular reading of it. A rare combination
is a good place to look; that is the whole of the claim.

Stretches are non-overlapping and all the same length, because a rate over
windows that share seconds counts the same event several times, and a rate over
windows of different lengths compares a long stretch's chance of containing a
mark with a short one's.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

import numpy as np

# The stretch every rate is computed over. Long enough to hold a burst and what
# follows it, short enough that a feature-length file still divides into enough
# stretches for a rate to mean anything.
WINDOW_SECONDS = 30

# A second has to reach this percentile of the video's own levels to count as a
# loud one. The same bar `highlight_prose` uses to call a moment one of the
# video's loudest, so the word means the same thing in both places.
LOUD_PERCENTILE = 90.0

# Below this many stretches a share is not a rate. A video that divides into
# four windows can only ever report 0%, 25%, 50%, 75% or 100%, and every one of
# those reads as more precise than it is.
MIN_WINDOWS = 8

# What each mark is called in the record and in the prose. Kept here so the
# vocabulary cannot drift between the count and the sentence describing it.
MARK_NAMES = {
    "arrival": "something arriving on screen",
    "movement": "a burst of movement",
    "loud": "one of the video's loudest seconds",
    "reading": "the expression reading settling",
}


def marks_in_range(start: float,
                   end: float,
                   *,
                   motion_peaks: Optional[Sequence[float]] = None,
                   levels: Optional[Sequence[float]] = None,
                   loud_threshold: Optional[float] = None,
                   expressions: Optional[Mapping] = None,
                   object_detections: Optional[Mapping] = None,
                   composed_event_names: Optional[Iterable[str]] = None,
                   ) -> frozenset:
    """Which marks one stretch of the video carries.

    Deliberately the same four the report already marks per clip, found the same
    way — a rate computed off a different definition of "a burst of movement"
    than the one printed beside it would be a rate for a different question.
    """
    marks = set()

    if any(float(start) <= float(t) < float(end) for t in (motion_peaks or ())):
        marks.add("movement")

    if levels is not None and loud_threshold is not None:
        lo, hi = max(0, int(start)), min(len(levels), int(np.ceil(end)))
        # Strictly above, not at: on a file that sits at one level for most of
        # its length the 90th percentile *is* that level, and "at or above" then
        # marks every stretch in the video as one of its loudest.
        if hi > lo and max(levels[lo:hi]) > loud_threshold:
            marks.add("loud")

    if expressions:
        try:
            from modules.expression_arc import peak_in_range
            if peak_in_range(expressions, start, end):
                marks.add("reading")
        except Exception:                          # pragma: no cover - defensive
            pass

    if object_detections:
        try:
            from modules.highlight_report import event_onset
            if event_onset(object_detections, start, end, composed_event_names):
                marks.add("arrival")
        except Exception:                          # pragma: no cover - defensive
            pass

    return frozenset(marks)


def survey(video_duration: float,
           *,
           window: int = WINDOW_SECONDS,
           motion_peaks: Optional[Sequence[float]] = None,
           levels: Optional[Sequence[float]] = None,
           expressions: Optional[Mapping] = None,
           object_detections: Optional[Mapping] = None,
           composed_event_names: Optional[Iterable[str]] = None,
           ) -> dict:
    """Walk the whole video in equal stretches, recording what each one carries.

    The kept clips are included rather than excluded. They are part of the
    video, and a rate that removed them would answer "how unusual is this
    against the material nobody picked", which flatters every pick by
    construction.
    """
    duration = float(video_duration or 0.0)
    if duration <= 0 or window <= 0:
        return {}

    loud_threshold = None
    if levels is not None and len(levels):
        loud_threshold = float(np.percentile(np.asarray(levels, dtype=float),
                                             LOUD_PERCENTILE))

    stretches = []
    start = 0.0
    while start + window <= duration:
        stretches.append(marks_in_range(
            start, start + window,
            motion_peaks=motion_peaks, levels=levels,
            loud_threshold=loud_threshold, expressions=expressions,
            object_detections=object_detections,
            composed_event_names=composed_event_names))
        start += window

    return {
        "window": int(window),
        "count": len(stretches),
        "stretches": stretches,
        "loud_threshold": loud_threshold,
        "enough": bool(len(stretches) >= MIN_WINDOWS),
    }


def rate(found: Mapping, marks: Iterable[str]) -> dict:
    """How much of the video carries at least this set of marks.

    "At least" rather than "exactly": the question a reader has is whether this
    combination turns up elsewhere, and a stretch carrying these four marks and
    a fifth is another place it turned up.
    """
    marks = frozenset(m for m in (marks or ()) if m in MARK_NAMES)
    if not found or not found.get("enough") or not marks:
        return {}
    stretches = found.get("stretches") or []
    matching = sum(1 for s in stretches if marks <= s)
    total = len(stretches)
    if not total:
        return {}
    return {
        "marks": sorted(marks),
        "matching": matching,
        "windows": total,
        "window_seconds": int(found.get("window") or WINDOW_SECONDS),
        "pct": round(100.0 * matching / total, 1),
    }


def marks_of(entry: Mapping, loud_threshold: Optional[float] = None) -> list:
    """The marks one kept clip carries, read off what the report already found.

    Taken from the entry rather than re-measured so a clip is always counted
    under the same marks the page prints for it. Loudness is the one that needs
    the video's threshold: every clip has a loudest second, and only some of
    them are loud.
    """
    marks = []
    if (entry.get("event_onset") or {}).get("second") is not None:
        marks.append("arrival")
    if (entry.get("motion_peak") or {}).get("second") is not None:
        marks.append("movement")
    if (entry.get("expression_peak") or {}).get("second") is not None:
        marks.append("reading")
    level = (entry.get("loudest") or {}).get("level_dbfs")
    # Strictly above, for the reason `marks_in_range` uses the same test: a clip
    # must not be counted under a mark the stretches were not counted under.
    if (level is not None and loud_threshold is not None
            and float(level) > float(loud_threshold)):
        marks.append("loud")
    return marks
