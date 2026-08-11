"""Group loudness excursions into events worth a person's time.

``modules/reaction_bursts.py`` measures; this decides. The split matters: the
measurement is a property of the audio and means the same thing everywhere, but
every number below is a *policy* choice about what to surface, and policy that
suits one kind of material suits another badly. Keeping them apart is what lets
the thresholds move without the measurement moving.

Three things happen here that a raw candidate list does not do:

**Ranking by excursion rather than rhythm.** ``find_candidates`` orders by
syllabic modulation by default, which is right when the question is "is there
laughter here". On material whose rhythm is ordinary speech it is wrong in a way
that hides the answer, so this asks for ``rank_by="z"``. See that function's
docstring for the measurement behind the choice.

**An edge guard.** Opening and closing minutes carry titles, music beds and
encoding artifacts, all of which are loud against a quiet neighbourhood and none
of which are content. Measured on an hour of material, 18 of 48 raw candidates
fell in the first and last two minutes.

**Merging.** Candidates are per-run, and one event produces many of them -- the
same hour put 12 runs inside a four-minute stretch. Merging by a gap turns a
list of runs into a list of *moments*, which is the unit a person reviews and
the unit a clip gets cut around.

The defaults were fitted on two hours of material mastered 17 dB apart (median
loudness -37 and -20 dBFS) and gave 11 and 14 events per hour, of which the
person who supplied the material judged 10/11 and 12/12 to be real. They are
offered as a starting point, not a discovery: the honest reading is that a
z-score against a rolling median transfers across mastering where an absolute
dB threshold cannot, which is the same lesson ``modules/audio_peaks.py``'s
fixed -20 dBFS threshold teaches by failing.

**A known limitation, stated because it will be met.** The baseline is a median
over a rolling window, so a *sustained* loud passage raises its own reference
and its excursions shrink. Where more than about half of a window is loud, the
median becomes the loud level and events there go unreported. Widening the
window does not fix it -- the spread widens with it and the z-score shrinks
faster than the excursion grows (measured: at +/-300s, recall fell from 8/9 to
6/9). Adding an absolute criterion recovers such passages but costs far more
than it returns at any threshold that keeps the list reviewable. So: this finds
moments that stand out from their surroundings, and a passage that is loud
throughout is not one.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

from modules import reaction_bursts as rb

# Excursion, in robust standard deviations above the local median, that a second
# must reach. The shipped 3.0 answers a different question (sustained audience
# reaction) and returns nothing on material whose events are brief.
DEFAULT_Z = 2.0

# Seconds an excursion must hold. This is the parameter that matters most, and
# the one most easily set wrong: raising it to 1.5 -- the value that suits
# applause -- drops recall to between 0 and 5 of 9 at *every* threshold, because
# real events here last one or two seconds. A duration rule encodes an
# assumption about the shape of the thing being looked for.
DEFAULT_MIN_DURATION = 1.0

# How much of each end to ignore. Two minutes covers typical title and end
# material without reaching into content.
DEFAULT_EDGE_GUARD = 120.0

# Candidates closer than this join one event. 30s keeps distinct moments apart;
# 60s was measured to swallow four separately-identified moments into a single
# block, which is worse than useless because it hides them.
DEFAULT_MERGE_GAP = 30.0

ProgressFn = Optional[Callable[[float], None]]


def group(candidates: Sequence,
          *,
          total_seconds: float,
          edge_guard: float = DEFAULT_EDGE_GUARD,
          merge_gap: float = DEFAULT_MERGE_GAP) -> list:
    """Drop edge material, merge what is adjacent, return events by loudest first.

    Each event reports the peak second and the z-score of its loudest
    contributing run, plus how many runs it absorbed -- a moment built from
    twelve runs is a different thing from one built from a single second, and
    the report should be able to say so.
    """
    if not candidates:
        return []

    lo = float(edge_guard)
    hi = float(total_seconds) - float(edge_guard)
    inside = [c for c in candidates if lo <= float(c["peak_second"]) <= hi]
    inside.sort(key=lambda c: float(c["start"]))

    merged: list = []
    for c in inside:
        if merged and float(c["peak_second"]) - merged[-1]["end"] <= merge_gap:
            m = merged[-1]
            m["end"] = max(m["end"], float(c["end"]))
            m["runs"] += 1
            if float(c["z"]) > m["z"]:
                m["z"] = float(c["z"])
                m["peak_second"] = int(c["peak_second"])
                m["modulation"] = float(c["peak_modulation"])
                m["mod_hz"] = float(c["mod_hz"])
        else:
            merged.append({
                "start": float(c["start"]),
                "end": float(c["end"]),
                "peak_second": int(c["peak_second"]),
                "z": float(c["z"]),
                "modulation": float(c["peak_modulation"]),
                "mod_hz": float(c["mod_hz"]),
                "runs": 1,
            })

    for m in merged:
        m["duration"] = round(m["end"] - m["start"], 2)
        m["timestamp"] = rb.format_timestamp(m["peak_second"])
        m["z"] = round(m["z"], 2)

    merged.sort(key=lambda m: m["z"], reverse=True)
    return merged


def detect(video_path: str,
           *,
           z_threshold: float = DEFAULT_Z,
           min_duration: float = DEFAULT_MIN_DURATION,
           edge_guard: float = DEFAULT_EDGE_GUARD,
           merge_gap: float = DEFAULT_MERGE_GAP,
           include_levels: bool = False,
           progress: ProgressFn = None,
           cancel=None) -> dict:
    """Measure one video and return its loudness events.

    Returns plain data only -- no numpy escapes -- so the result caches and
    crosses a thread boundary untouched. Five of the six per-second curves are
    deliberately *not* carried: a caller who wants to draw them can ask
    ``reaction_bursts.analyse`` directly and pay the few seconds again.

    ``include_levels`` adds the one curve that cannot be re-derived cheaply from
    the events alone -- per-second loudness in dBFS -- because comparing the
    level measured during different labelled classes needs a value for every
    second, not only the ones that stood out. One array of floats, not six.
    """
    result = rb.analyse(video_path,
                        z_threshold=z_threshold,
                        min_duration=min_duration,
                        progress=progress,
                        cancel=cancel)
    curves = result["curves"]
    import numpy as np
    candidates = rb.find_candidates(
        np.array(curves["z"]),
        np.array(curves["burst_db"]),
        np.array(curves["modulation"]),
        np.array(curves["mod_hz"]),
        z_threshold=z_threshold,
        min_duration=min_duration,
        rank_by="z",
    )
    seconds = int(result["seconds"])
    events = group(candidates, total_seconds=seconds,
                   edge_guard=edge_guard, merge_gap=merge_gap)
    per_hour = (len(events) / seconds * 3600.0) if seconds else 0.0
    if include_levels:
        return {
            "seconds": seconds,
            "params": {
                "z_threshold": float(z_threshold),
                "min_duration": float(min_duration),
                "edge_guard": float(edge_guard),
                "merge_gap": float(merge_gap),
            },
            "events": events,
            "events_per_hour": round(per_hour, 1),
            "levels": [float(v) for v in curves["loudness"]],
        }
    return {
        "seconds": seconds,
        "params": {
            "z_threshold": float(z_threshold),
            "min_duration": float(min_duration),
            "edge_guard": float(edge_guard),
            "merge_gap": float(merge_gap),
        },
        "events": events,
        "events_per_hour": round(per_hour, 1),
    }


def event_seconds(events: Sequence) -> set:
    """Every whole second covered by an event, for scoring and signal counting."""
    out: set = set()
    for e in events:
        for sec in range(int(e["start"]), int(e["end"]) + 1):
            out.add(sec)
    return out
