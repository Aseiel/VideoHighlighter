"""Say how each chapter differs from the rest of its own video.

:mod:`modules.highlight_compare` answers that question for a *clip*, and does it
by ranking the clip's peak against sliding windows of the clip's own length.
That method does not transfer to a chapter, and it is worth being precise about
why, because the obvious move is to reuse it.

A clip is ten or twenty seconds and is described by its strongest moment, so
"how does this maximum rank among other windows this long" is both answerable
and meaningful. A chapter is minutes. Windows of that length barely fit in the
video -- a ten-minute chapter of a ninety-minute film has eight non-overlapping
peers -- so a percentile computed against them is an artefact of the sample. And
a maximum is the wrong statistic anyway: a chapter is not characterised by its
loudest second, it is characterised by what is true *throughout* it.

So this module compares **shares and rates** instead:

* **Share of the cut.** What fraction of the kept clips came from this chapter,
  against what fraction of the runtime it occupies. A chapter holding 10% of the
  film and 40% of the highlights is the single most useful fact a chapter
  breakdown can carry, and it needs no detector at all.
* **Prevalence lift.** For each detected class, the share of this chapter's
  detected seconds carrying it, divided by that share across the whole video.
  This is a proportion over hundreds of seconds rather than a rank among eight
  peers, so it has the sample size the percentile lacked.
* **Expression mix.** The same ratio applied to the per-second expression read.
* **Pace and level.** Cutting rate and median loudness against the video's own,
  which are properties of how a stretch was shot and mixed rather than of what
  is in it.

Every ratio ships with the counts behind it. A lift of 4.0 computed from three
seconds is noise wearing a number's clothes, and the only defence is that the
reader can see the three. :data:`MIN_SECONDS_FOR_LIFT` marks where a ratio
becomes reportable; nothing is suppressed, but the flag means the prose layer
does not have to re-derive that judgement -- the same split
:mod:`modules.highlight_compare` keeps.

**This module computes, it does not characterise.** No adjectives, no "unusual",
no ordering beyond a stable sort by effect size. What is worth saying out loud
belongs to :mod:`modules.highlight_prose`, so a threshold lives in one place.

Content is not this module's business: it compares a class against itself and
holds no vocabulary of its own. Whatever the detector was taught to find, the
same arithmetic applies.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional, Sequence

import numpy as np

# Below this many seconds carrying a class inside a chapter, a prevalence ratio
# is not evidence about the chapter. Ratios are still computed and returned --
# with the count beside them -- but flagged so the sentence layer can decline.
MIN_SECONDS_FOR_LIFT = 8

# How far from parity a share has to sit before it is worth a reader's attention
# at all. Below this everything is "about the same as the rest of the video",
# which is the answer for most chapters and should read as one.
NOTABLE_LIFT = 1.4

# How many findings to keep per chapter. A chapter with eleven shifted classes
# has no story; the two or three furthest from parity do.
MAX_FINDINGS = 3

# A chapter contributing this many times its runtime share of the cut is where
# the highlights actually came from. Kept here rather than in the renderer so
# the text and HTML views cannot disagree about it.
CUT_SHARE_LIFT = 1.5


def _safe_ratio(part: float, whole: float) -> float:
    """``part / whole``, with zero standing for "no basis to compare"."""
    whole = float(whole)
    if whole <= 0:
        return 0.0
    return float(part) / whole


def assign_chapters(chapters: Sequence[Mapping],
                    segments: Sequence[Sequence[float]]) -> list[int]:
    """Which chapter each segment belongs to -> chapter *numbers*, 1-based.

    A clip is placed by its start. Clips do not respect chapter boundaries --
    a highlight can straddle one -- and splitting a clip across two chapters
    would double-count it in the share arithmetic that is the whole point of
    this module. Placing it whole, by where it begins, keeps every clip in
    exactly one chapter and the shares summing to 100%.

    Returns 0 for a segment that falls outside every chapter, which should not
    happen for a well-formed partition but is not worth crashing a report over.
    """
    out = []
    for seg in segments:
        start = float(seg[0])
        found = 0
        for ch in chapters:
            if float(ch["start"]) <= start < float(ch["end"]):
                found = int(ch["number"])
                break
        if not found and chapters and start >= float(chapters[-1]["end"]):
            found = int(chapters[-1]["number"])   # exactly on the final edge
        out.append(found)
    return out


def _prevalence_lifts(largest: Mapping, window: range,
                      seconds_with: Mapping, detected_seconds: int) -> list[dict]:
    """Per class: how much more (or less) of this chapter carries it.

    The denominator is *detected* seconds on both sides, not wall-clock seconds.
    A chapter that is half black frames would otherwise show every class as
    depressed, which is a fact about the detector's coverage rather than about
    the chapter's content.
    """
    chapter_seconds_with: dict = {}
    chapter_detected = 0
    for sec in window:
        best = largest.get(sec) or {}
        if best:
            chapter_detected += 1
        for name in best:
            chapter_seconds_with[name] = chapter_seconds_with.get(name, 0) + 1

    if not chapter_detected or not detected_seconds:
        return []

    findings = []
    for name, count in chapter_seconds_with.items():
        here = _safe_ratio(count, chapter_detected)
        overall = _safe_ratio(seconds_with.get(name, 0), detected_seconds)
        if overall <= 0:
            continue
        findings.append({
            "name": str(name),
            "seconds": int(count),
            "chapter_share_pct": round(100.0 * here, 1),
            "video_share_pct": round(100.0 * overall, 1),
            "lift": round(here / overall, 2),
            "enough_samples": bool(count >= MIN_SECONDS_FOR_LIFT),
        })

    # Furthest from parity first, in either direction: a class that vanishes in
    # a chapter is as much a description of it as one that takes it over. Ties
    # break on the name so a re-run cannot reorder the report.
    findings.sort(key=lambda f: (-abs(math.log(max(f["lift"], 1e-3))), f["name"]))
    return findings


def _expression_lifts(expressions: Mapping, window: range) -> list[dict]:
    """The same ratio over the per-second expression read."""
    if not expressions:
        return []
    video_counts: dict = {}
    for _sec, (label, _conf) in expressions.items():
        video_counts[label] = video_counts.get(label, 0) + 1
    video_total = sum(video_counts.values())
    if not video_total:
        return []

    chapter_counts: dict = {}
    for sec in window:
        found = expressions.get(sec)
        if found:
            chapter_counts[found[0]] = chapter_counts.get(found[0], 0) + 1
    chapter_total = sum(chapter_counts.values())
    if not chapter_total:
        return []

    findings = []
    for label, count in chapter_counts.items():
        here = _safe_ratio(count, chapter_total)
        overall = _safe_ratio(video_counts.get(label, 0), video_total)
        if overall <= 0:
            continue
        findings.append({
            "label": str(label),
            "seconds": int(count),
            "chapter_share_pct": round(100.0 * here, 1),
            "video_share_pct": round(100.0 * overall, 1),
            "lift": round(here / overall, 2),
            "enough_samples": bool(count >= MIN_SECONDS_FOR_LIFT),
        })
    findings.sort(key=lambda f: (-abs(math.log(max(f["lift"], 1e-3))), f["label"]))
    return findings


def summarise_chapters(chapters: Sequence[Mapping],
                       *,
                       score: Optional[np.ndarray] = None,
                       segments: Sequence[Sequence[float]] = (),
                       amps: Sequence[float] = (),
                       amps_per_second: float = 0.0,
                       distributions: Optional[Mapping] = None,
                       video_duration: float = 0.0,
                       signals: Optional[Mapping] = None,
                       cut_threshold: Optional[float] = None) -> list[dict]:
    """Measure every chapter against the whole video. Returns JSON-safe dicts.

    Each returned dict is the caller's chapter dict plus the comparisons. The
    input is not mutated -- the timeline holds the same objects, and a renderer
    quietly growing them is how two views end up disagreeing.
    """
    chapters = [dict(ch) for ch in (chapters or [])]
    if not chapters:
        return []

    duration = float(video_duration) or float(chapters[-1]["end"])
    score = np.asarray(score if score is not None else [], dtype=float)
    assignment = assign_chapters(chapters, segments)

    # Video-wide references, computed once.
    video_score_mean = float(score[score > 0].mean()) if (score.size and (score > 0).any()) else 0.0
    kept_seconds_total = sum(float(s[1]) - float(s[0]) for s in segments) or 0.0
    video_shots = sum(int(ch.get("shots") or 0) for ch in chapters)
    video_pace = _safe_ratio(video_shots, duration / 60.0) if duration else 0.0

    amps_arr = np.asarray([a for a in amps if a > 0], dtype=float) if len(amps) else np.zeros(0)
    video_amp_median = float(np.median(amps_arr)) if amps_arr.size else 0.0

    largest = (distributions or {}).get("largest") or {}
    seconds_with = (distributions or {}).get("seconds_with") or {}
    detected_seconds = int((distributions or {}).get("detected_seconds") or 0)
    expressions = (distributions or {}).get("expressions") or {}

    for ch in chapters:
        start, end = float(ch["start"]), float(ch["end"])
        seconds = max(0.0, end - start)
        window = range(int(start), int(math.ceil(end)))

        # --- share of the cut ------------------------------------------------
        mine = [i for i, n in enumerate(assignment) if n == int(ch["number"])]
        clip_seconds = sum(float(segments[i][1]) - float(segments[i][0]) for i in mine)
        runtime_share = _safe_ratio(seconds, duration)
        cut_share = _safe_ratio(clip_seconds, kept_seconds_total)
        ch["clips"] = len(mine)
        ch["clip_indices"] = [i + 1 for i in mine]     # 1-based, as the report numbers them
        ch["clip_seconds"] = round(clip_seconds, 1)
        ch["runtime_share_pct"] = round(100.0 * runtime_share, 1)
        ch["cut_share_pct"] = round(100.0 * cut_share, 1)
        # >1 means the chapter punched above its length. This is the number a
        # reader should look at first, which is why it is not buried in a list.
        ch["cut_share_lift"] = round(_safe_ratio(cut_share, runtime_share), 2)

        # --- score density ---------------------------------------------------
        if score.size:
            lo, hi = max(0, int(start)), min(len(score), int(math.ceil(end)))
            window_score = score[lo:hi]
            live = window_score[window_score > 0]
            ch["score_mean"] = round(float(live.mean()) if live.size else 0.0, 2)
            ch["score_lift"] = round(_safe_ratio(ch["score_mean"], video_score_mean), 2)
            # The best single second, and where. A chapter is selected on its
            # peaks, not its average, so the mean above cannot answer "why was
            # nothing taken from here" -- a stretch can be unremarkable overall
            # and still contain the second that should have been cut.
            if window_score.size:
                best = int(np.argmax(window_score))
                ch["score_peak"] = round(float(window_score[best]), 2)
                ch["score_peak_second"] = lo + best

        # --- which detectors fired here --------------------------------------
        # Named so an unselected chapter can say *which* weight to raise rather
        # than leaving the reader to guess from a total of zero.
        if signals:
            fired = []
            lo, hi = max(0, int(start)), int(math.ceil(end))
            for key, arr in signals.items():
                arr = np.asarray(arr, dtype=float)
                if arr.size and float(arr[lo:min(hi, len(arr))].sum()) > 0:
                    fired.append(key)
            ch["signals_present"] = sorted(fired)

        # --- pace ------------------------------------------------------------
        if seconds > 0:
            ch["shots_per_minute"] = round(_safe_ratio(int(ch.get("shots") or 0),
                                                       seconds / 60.0), 1)
            ch["pace_lift"] = round(_safe_ratio(ch["shots_per_minute"], video_pace), 2)

        # --- level -----------------------------------------------------------
        if amps_per_second and len(amps):
            lo = int(start * amps_per_second)
            hi = max(lo + 1, int(end * amps_per_second))
            here = np.asarray([a for a in amps[lo:hi] if a > 0], dtype=float)
            if here.size and video_amp_median > 0:
                median = float(np.median(here))
                ch["loudness_median"] = round(median, 4)
                # A ratio of amplitudes expressed in dB, so it reads on the
                # scale the rest of the report uses for level.
                ch["loudness_delta_db"] = round(
                    20.0 * math.log10(max(median, 1e-9) / video_amp_median), 1)

        # --- what is in it ---------------------------------------------------
        subjects = _prevalence_lifts(largest, window, seconds_with, detected_seconds)
        if subjects:
            ch["subjects"] = subjects[:MAX_FINDINGS]
        # Every class's share, not only the ones that cleared the lift bars.
        # The bars answer "is this chapter unusual for the video"; a *change*
        # between neighbours is a different question and needs the full set,
        # since a class can move sharply while staying near the video average.
        if subjects:
            ch["class_shares"] = {f["name"]: f["chapter_share_pct"]
                                  for f in subjects}
        expression = _expression_lifts(expressions, window)
        if expression:
            ch["expression"] = expression[:MAX_FINDINGS]

        # The bar this chapter had to clear. Carried on every row so a renderer
        # can explain one chapter without being handed the whole list.
        if cut_threshold is not None:
            ch["cut_threshold"] = round(float(cut_threshold), 2)

    _mark_changes(chapters)
    return chapters


# A class has to move at least this many percentage points between neighbouring
# chapters before the move is worth a sentence. Below it the "change" is mostly
# the detector's own second-to-second scatter.
CHANGE_POINTS = 12.0

# Above this share a class is doing enough of the chapter to be called its
# subject rather than something merely present in it.
DOMINANT_SHARE = 45.0


def _mark_changes(chapters: Sequence[Mapping]) -> None:
    """What each chapter changed relative to the one before it. Mutates in place.

    This is where a run of chapters becomes a sequence rather than a list. Each
    chapter compared against the *video* says how unusual it is; each compared
    against its *predecessor* says what happened at the boundary, and only the
    second reads as one thing following another.

    Nothing here interprets. A class rising from 4% to 28% of a chapter's
    detected seconds is a measurement; what it means is the reader's.
    """
    previous: Optional[Mapping] = None
    for ch in chapters:
        shares = ch.get("class_shares") or {}
        if previous is not None:
            before = previous.get("class_shares") or {}
            moves = []
            for name in set(shares) | set(before):
                was, now = float(before.get(name, 0.0)), float(shares.get(name, 0.0))
                if abs(now - was) >= CHANGE_POINTS:
                    moves.append({
                        "name": str(name),
                        "from_pct": round(was, 1),
                        "to_pct": round(now, 1),
                        "direction": "rose" if now > was else "fell",
                    })
            moves.sort(key=lambda m: -abs(m["to_pct"] - m["from_pct"]))
            # Two at most. A boundary where five things moved at once is a scene
            # change, and listing all five describes the edit rather than the
            # content -- the two largest movers are the beat worth reading.
            if moves:
                ch["changes"] = moves[:2]
        # The class carrying most of this chapter, when one does.
        if shares:
            name, pct = max(shares.items(), key=lambda kv: kv[1])
            if float(pct) >= DOMINANT_SHARE:
                ch["dominant"] = {"name": str(name), "share_pct": round(float(pct), 1)}
        previous = ch


def distinctive(chapter: Mapping) -> list[dict]:
    """The findings for one chapter that clear both the sample and effect bars.

    A convenience over the raw lists so the two renderers cannot apply different
    rules. Returns entries tagged with what they describe, most extreme first.
    """
    out = []
    for kind, key in (("subject", "subjects"), ("expression", "expression")):
        for finding in chapter.get(key) or []:
            if not finding.get("enough_samples"):
                continue
            lift = float(finding.get("lift") or 0.0)
            if lift <= 0 or (NOTABLE_LIFT > lift > 1.0 / NOTABLE_LIFT):
                continue
            out.append({**finding, "kind": kind,
                        "name": finding.get("name") or finding.get("label")})
    out.sort(key=lambda f: -abs(math.log(max(float(f["lift"]), 1e-3))))
    return out[:MAX_FINDINGS]
