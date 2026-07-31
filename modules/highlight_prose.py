"""Say what a moment was, in a sentence, from what was measured.

The report already knows a clip peaked at -1 dBFS in the 100th percentile with
two signals landing 0.3s apart. A person reading that has to do the translation
themselves, every row, and mostly does not — so a report full of true numbers
communicates less than one plain sentence would.

This is the translation, and it is deterministic. Every clause is produced by a
threshold over a measurement, so a sentence can be traced back to the figures
printed beside it and cannot say more than the data supports. No model is
involved: a language model asked to do this would produce better prose and
occasionally invent a reason, and the second thing costs more than the first
gains.

The hardest discipline here is refusing to flatter. Most moments in most videos
are not exceptional, and a report that calls every one of them outstanding is
worth exactly as much as one that says nothing. When the numbers are ordinary
the sentence says so — and when everything in a run scored alike, saying *that*
is the most useful sentence available, because it means the ranking had nothing
to work with.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

# Percentile at which a moment is genuinely unusual for its video rather than
# merely above average. Set high on purpose: the word has to keep its meaning.
EXCEPTIONAL = 90.0
NOTABLE = 70.0
# Wide on purpose. Most clips in most cuts are unremarkable relative to each
# other, and a narrow middle band pushes ordinary clips into "weaker", which
# reads as a criticism of a clip that is doing nothing wrong.
ORDINARY = 30.0

# A score shared by at least this much of the cut is a tie, not a rank. A
# majority, because the sentence says "most" and it has to be true.
TIED_SHARE = 0.5

# Loudness is judged against the file, not an absolute level — a quiet recording
# has loud moments too, and they are what its highlights are made of.
LOUD_PERCENTILE = 90.0

# Names as a reader would say them, not as the weight table spells them.
SIGNAL_PROSE = {
    "scene": "a shot change",
    "motion_event": "movement",
    "motion_peak": "a burst of movement",
    "audio": "a rise in sound",
    "keyword": "something said",
    "object": "what was on screen",
    "action": "a recognised action",
    "face": "a facial expression",
}


def _join(items: Sequence[str]) -> str:
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _standing(entry: Mapping, peer_scores: Optional[Sequence[float]]) -> str:
    """How this clip compares with the others in the same cut.

    Deliberately *not* its percentile against the whole video. Selection picks
    the highest-scoring seconds, so a kept clip is in the top percentile of its
    own file by construction — saying so is true, tautological, and reads as
    praise. What a reader cannot already infer is whether this clip is the
    strongest of the ones they got, or the one that scraped in.
    """
    if not peer_scores or len(peer_scores) < 3:
        return ""
    score = float(entry.get("score") or 0.0)
    ordered = sorted(peer_scores)
    if ordered[0] == ordered[-1]:
        return "Scored the same as every other clip here"

    # A clip tied with most of the cut has no rank worth reporting. Ranking it
    # anyway lands the whole tied group mid-scale and then calls them all
    # weaker — which is how fifteen of sixteen top-scoring clips came to be
    # described as the ones that scraped in. A tie is information about the
    # scoring, not about the clip, so say that instead.
    tied = sum(1 for s in ordered if s == score)
    if tied / len(ordered) >= TIED_SHARE:
        return "Scored the same as most of the other clips here"

    # Midrank, as elsewhere: counting only what is strictly below caps the best
    # of five clips at 80%, so the top of a short cut could never be called the
    # top of it.
    below = sum(1 for s in ordered if s < score)
    at_or_below = sum(1 for s in ordered if s <= score)
    share = (below + at_or_below) / 2.0 / len(ordered) * 100.0
    if share >= EXCEPTIONAL:
        return "The strongest clip in this highlight"
    if share >= NOTABLE:
        return "Among the stronger clips here"
    if share >= ORDINARY:
        return "Middling for this highlight"
    return "One of the weaker clips that still made the cut"


def _evidence(entry: Mapping) -> str:
    """What actually fired, named the way a person would name it.

    Taken from ``signals_present`` rather than the points, because a signal can
    contribute without scoring — an action suppressed by "require objects" was
    still detected, and a sentence that omits it describes the wrong moment.
    Points are the fallback for a record that predates the field.
    """
    present = set(entry.get("signals_present") or ())
    if not present:
        present = {key for key, value in (entry.get("breakdown") or {}).items()
                   if value > 0}
    ordered = [key for key in SIGNAL_PROSE if key in present]
    return _join([SIGNAL_PROSE[key] for key in ordered])


def _loudness(measured: Mapping) -> str:
    percentile = measured.get("loudness_percentile")
    dbfs = measured.get("loudness_dbfs")
    if percentile is None or dbfs is None:
        return ""
    if percentile >= LOUD_PERCENTILE:
        return f"one of its loudest points ({dbfs:.0f} dBFS)"
    return ""


def _agreement(entry: Mapping, measured: Mapping) -> str:
    """Signals arriving together is the difference between noise and an event.

    Phrased to follow the list of what fired, so the reader learns *which*
    signals agreed rather than only how many.
    """
    if len(entry.get("signals_present") or []) < 2:
        return ""
    if not measured.get("signals_coincide"):
        return "though not at the same instant"
    spread = measured.get("signal_spread_seconds", 0.0)
    if spread <= 0:
        return "landing on the same second"
    return f"landing within {spread:.0f}s of each other"


def describe(entry: Mapping,
             peer_scores: Optional[Sequence[float]] = None) -> str:
    """One sentence about why this clip is in the highlight.

    Every clause comes from a threshold over a measurement in ``entry``, so the
    sentence and the figures printed beside it can never disagree.

    ``peer_scores`` are the scores of the other kept clips; without them the
    sentence simply omits any ranking rather than falling back to the
    video-wide percentile, which is tautological for a selected clip.
    """
    measured = entry.get("measured") or {}
    standing = _standing(entry, peer_scores)
    evidence = _evidence(entry)
    agreement = _agreement(entry, measured)

    parts = []
    if evidence:
        # What fired always leads: "two signals agreed" is worth much less than
        # knowing it was sound and movement.
        only = ("" if len(entry.get("signals_present") or []) != 1
                else " alone")
        parts.append(f"{evidence}{only} {agreement}".strip()
                     if agreement else f"{evidence}{only}")
    loudness = _loudness(measured)
    if loudness:
        parts.append(loudness)

    if not standing:
        return f"Chosen on {_join(parts)}." if parts else ""
    if parts:
        return f"{standing} — {_join(parts)}."
    return f"{standing}."


def describe_all(report: Mapping) -> list:
    """A sentence per clip, each ranked against the others in the cut."""
    segments = list(report.get("segments") or [])
    peers = [float(s.get("score") or 0.0) for s in segments]
    return [describe(s, peers) for s in segments]


def summarise_run(report: Mapping) -> str:
    """One sentence about the highlight as a whole.

    Not "how many clips were exceptional" — every kept clip outscored the rest
    of the video, that is what being kept means. What is worth saying is
    whether the scores separated at all, and how much of the video the cut
    actually saw.
    """
    segments = list(report.get("segments") or [])
    if not segments:
        return "Nothing was selected."

    scores = [float(s.get("score") or 0.0) for s in segments]
    duration = float((report.get("video") or {}).get("duration") or 0.0)
    span = (max(s["end"] for s in segments) - min(s["start"] for s in segments)
            if duration else 0.0)

    if len(segments) > 2 and max(scores) == min(scores):
        return (f"All {len(segments)} clips scored identically, so nothing "
                "separated them — the order they were picked in was arbitrary.")

    parts = [f"{len(segments)} clips"]
    if max(scores) > 0:
        spread = (max(scores) - min(scores)) / max(scores) * 100.0
        parts.append(f"the best scoring {spread:.0f}% above the weakest"
                     if spread >= 5 else "all scoring within a few points")
    if duration and span:
        parts.append(f"drawn from {span / duration * 100:.0f}% of its length")
    return _join(parts).capitalize() + "."
