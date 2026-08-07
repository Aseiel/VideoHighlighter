"""What the video talks about that this run *did* measure.

:mod:`modules.vocabulary_gap` computes half of a comparison and drops the other
half on the floor. It matches the words a stretch says against the names of the
classes the detector actually produced, and reports the **misses** — things
talked about that nothing was watching for. The **hits** are thrown away, and
the hits are the more useful half.

The case that makes this obvious: somebody on camera names a moment and says it
was the best one. If nothing in the run was watching for that thing, the gap
list is the honest answer — nobody can check it. But if the run *did* label it,
then this report is holding the answer and not saying it. It knows how many
seconds of the video carry that label, which stretch holds most of them, how
loud the video was there, what else was on screen at that moment, and whether
the cut took anything from it. A narrator handed only the transcript will
paraphrase the claim back as if repeating it were confirming it. Handed the
measurement, it can agree with the speaker, or disagree, and either is worth
more than the paraphrase.

That last part is the point. This module exists to make **disagreement
possible**, not to decorate claims that were going to be repeated anyway. A
speaker naming their favourite moment, over a label the run found in eleven
seconds at the quietest part of the file, is the most interesting row this
report can produce, and it is exactly the row a transcript alone cannot make.

Matching, and what it does not mean
-----------------------------------

A spoken line *names* a class when every token of the class name appears among
the line's tokens. Token equality and nothing cleverer — the same rule
:mod:`modules.vocabulary_gap` matches on, and for the same reason: a synonym
table is a vocabulary, this repo ships none, and one written for English would
be wrong the first time somebody transcribes anything else.

So a match means *the same words were used*, and it means nothing else. The
speaker may be using a word in a sense that has nothing to do with what the
detector was taught to find, and this module cannot tell. That is why every row
carries the line it was matched from: the reader — and the narrator, which is
told this in as many words — can see the sentence and decide whether the two
are about the same thing at all.

One-token class names are held to a higher bar: the token has to be one of the
words the stretch says far more often than the rest of the video
(:func:`modules.chapter_speech.distinctive_words`). Without that guard a class
named after an ordinary word fires on every chapter that happens to say it, and
a page of those teaches a reader to skip the section — which costs the rows that
mattered.

Where the evidence is measured from
-----------------------------------

The **whole video**, not the stretch the sentence was said in. People describe
in the last five minutes something that happened in the middle, and a row that
looked only at the current chapter would answer "your favourite moment never
happened" about a moment that plainly did. The chapter the claim was *made* in
decides where the row is filed; the chapter the label is *densest* in is part
of what the row says. :func:`modules.chapter_story._checks_here` files its rows
the same way and for the same reason.

Everything reported is arithmetic over arrays the report already holds. Nothing
here interprets, ranks a claim as true or false, or knows one class name from
another.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional, Sequence

# Rows kept per chapter. A stretch that names five measured classes is a
# stretch where somebody listed things, and five rows of arithmetic under one
# paragraph buries the paragraph. The two or three most specific are the ones a
# reader would have checked by hand.
MAX_PER_CHAPTER = 3

# Rows rendered on the page per chapter, below the cap above. The record keeps
# all of them because a later pass may want them; a chapter block is read, and
# two is what fits under one paragraph without becoming the block.
MAX_ON_PAGE = 2

# How far the loudest second of a class has to sit from the video's median
# before the distance is said out loud. Under a dB is inside the noise of the
# measurement — the finding :mod:`modules.level_by_class` was built around —
# and "0 dB above the median" invites a reader to weigh a number that means
# nothing. The second is still named; only the claim about it is dropped.
MIN_LEVEL_DELTA_DB = 1.0


def _timestamp(seconds) -> str:
    try:
        seconds = max(0, int(seconds))
    except (TypeError, ValueError):
        return "?"
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def names_in(line_words: set, named: Mapping, distinctive: set) -> list:
    """Which of ``named`` this line's words cover, most specific first.

    ``named`` maps a class name to its token set (:func:`vocabulary_gap.tokens_of`).
    A one-token name must also be distinctive of the stretch — see the header.
    """
    hits = []
    for name, tokens in named.items():
        if not tokens or not tokens <= line_words:
            continue
        if len(tokens) == 1 and not (tokens & distinctive):
            continue
        hits.append((name, tokens))
    # More tokens first: a two-word name matched in full is far less likely to
    # be a coincidence of vocabulary than a single common word.
    hits.sort(key=lambda h: (-len(h[1]), h[0]))
    return hits


def mentions(chapter: Mapping,
             names: Iterable[str],
             *,
             limit: int = MAX_PER_CHAPTER) -> list[dict]:
    """Lines in one chapter that name something the detector produced.

    One row per class at most, carrying the earliest line that named it — the
    first time a stretch raises something is where a reader looking for it will
    start playing.
    """
    from modules.vocabulary_gap import tokens_of
    from modules.chapter_speech import tokenize

    named = {}
    for name in (names or ()):
        tokens = tokens_of(name)
        if tokens:
            named[str(name)] = tokens
    if not named:
        return []

    distinctive = {str(found.get("word") or "")
                   for found in (chapter.get("speech_words") or [])}
    lines = list(chapter.get("dialogue") or chapter.get("quotes") or [])

    found: dict = {}
    for line in lines:
        words = set(tokenize(line.get("text")))
        if not words:
            continue
        for name, tokens in names_in(words, named, distinctive):
            if name in found:
                continue
            row = {"name": name,
                   "words": sorted(tokens),
                   "said_at": round(float(line.get("start") or 0.0), 2),
                   "timestamp": str(line.get("timestamp")
                                    or _timestamp(line.get("start") or 0)),
                   "quote": str(line.get("text") or "")}
            speaker = str(line.get("speaker") or "").strip()
            if speaker and speaker.upper() != "UNKNOWN":
                row["speaker"] = speaker
            found[name] = row
    ranked = sorted(found.values(),
                    key=lambda r: (-len(r["words"]), r["said_at"]))
    return ranked[:limit]


def _seconds_by_name(labels_by_second: Mapping) -> dict:
    """``{class name: sorted seconds it was labelled in}``.

    Tolerates both shapes the report passes around — a second mapping to a list
    of names, and one mapping to ``{name: (area, confidence)}`` — because both
    iterate to the same names and neither caller should have to normalise.
    """
    out: dict = {}
    for sec, found in (labels_by_second or {}).items():
        try:
            sec = int(sec)
        except (TypeError, ValueError):
            continue
        for name in (found or ()):
            out.setdefault(str(name), []).append(sec)
    for name in out:
        out[name] = sorted(set(out[name]))
    return out


def share_maps(chapters: Sequence[Mapping],
               labels_by_second: Optional[Mapping] = None) -> list:
    """Each chapter's ``{class: % of its detected seconds}``, or ``None``.

    Prefers the map :mod:`modules.chapter_compare` already put on the chapter,
    so this section and the "mostly X" line above it in the same block cannot
    quote different figures for the same stretch. That map is only built when a
    bbox cache reached the report, which is not the ordinary run — and the
    per-second labels are enough on their own, so they are the fallback.

    ``None`` for a chapter where nothing was detected at all. Absence of a
    class in a stretch full of detections is a fact about the stretch; absence
    in a stretch the detector saw nothing in is a fact about the detector, and
    printing the second as "0%" would pass one off as the other.
    """
    labels_by_second = labels_by_second or {}
    out: list = []
    for chapter in (chapters or []):
        shares = chapter.get("class_shares")
        if shares is not None:
            out.append(dict(shares))
            continue
        lo = int(float(chapter.get("start") or 0.0))
        hi = int(math.ceil(float(chapter.get("end") or 0.0)))
        detected, tally = 0, {}
        for sec in range(lo, hi):
            found = labels_by_second.get(sec)
            if not found:
                continue
            detected += 1
            for name in found:
                tally[str(name)] = tally.get(str(name), 0) + 1
        out.append({k: round(100.0 * v / detected, 1) for k, v in tally.items()}
                   if detected else None)
    return out


def _densest_chapter(name: str,
                     chapters: Sequence[Mapping],
                     shares: Sequence[Optional[Mapping]]) -> Optional[dict]:
    """The stretch holding the largest share of this class, if any does."""
    best = None
    for chapter, here in zip(chapters, shares):
        share = (here or {}).get(name)
        if share is None:
            continue
        share = float(share)
        if best is None or share > best["share_pct"]:
            best = {"number": int(chapter.get("number") or 0),
                    "timestamp": str(chapter.get("timestamp") or ""),
                    "share_pct": round(share, 1)}
    return best


def _level(name: str,
           seconds: Sequence[int],
           levels: Sequence[float],
           labels_by_second: Mapping,
           video_median: Optional[float]) -> Optional[dict]:
    """How loud the video was during this class, and at its loudest second.

    Two figures that answer different questions and must not be confused. The
    **loudest second** is a fact: one measurement, and whatever was labelled
    alongside it. The **median** is an aggregate, and it carries the confound
    :mod:`modules.level_by_class` exists to handle — level moves far more across
    a video than between classes, so a class concentrated in one loud passage
    reads as loud when what was measured is where it happened. It is reported
    because a reader weighing a claim wants it, and the prose layer says which
    of the two it is quoting.
    """
    inside = [s for s in seconds if 0 <= s < len(levels)]
    if not inside:
        return None
    values = sorted(float(levels[s]) for s in inside)
    middle = values[len(values) // 2] if len(values) % 2 else (
        (values[len(values) // 2 - 1] + values[len(values) // 2]) / 2.0)

    out = {"median_dbfs": round(middle, 2), "seconds": len(inside)}
    if video_median is not None:
        out["vs_video_db"] = round(middle - float(video_median), 2)

    peak = max(inside, key=lambda s: levels[s])
    loudest = {"second": int(peak),
               "timestamp": _timestamp(peak),
               "level_dbfs": round(float(levels[peak]), 2)}
    if video_median is not None:
        loudest["vs_video_db"] = round(float(levels[peak]) - float(video_median), 2)
    # What else was on screen at that second — the "and what was happening"
    # half of the question. A fact about one second, no statistics involved.
    alongside = sorted({str(n) for n in (labels_by_second.get(peak) or ())}
                       - {name})
    if alongside:
        loudest["with"] = alongside
    out["loudest"] = loudest
    return out


def _clips(seconds: Sequence[int], segments: Sequence[Sequence[float]]) -> list:
    """Which kept clips (1-based, as the report numbers them) overlap the class."""
    marked = set(int(s) for s in seconds)
    if not marked:
        return []
    hits = []
    for index, span in enumerate(segments or (), start=1):
        try:
            lo, hi = int(float(span[0])), int(math.ceil(float(span[1])))
        except (TypeError, ValueError, IndexError):
            continue
        if any(sec in marked for sec in range(lo, hi)):
            hits.append(index)
    return hits


def measure(name: str,
            *,
            seconds: Sequence[int],
            chapters: Sequence[Mapping] = (),
            chapter_shares: Sequence[Optional[Mapping]] = (),
            segments: Sequence[Sequence[float]] = (),
            levels: Sequence[float] = (),
            labels_by_second: Optional[Mapping] = None,
            video_median: Optional[float] = None,
            detected_seconds: int = 0) -> dict:
    """Everything this run measured about one class, video-wide.

    Deliberately reports a class the detector barely found. "It was labelled in
    nine seconds of the whole file" is not a failure of this function — beside a
    sentence calling it the best part of the video, it is the finding.
    """
    out: dict = {"seconds": len(seconds)}
    if detected_seconds > 0:
        out["video_share_pct"] = round(
            100.0 * len(seconds) / float(detected_seconds), 1)
    if not seconds:
        return out

    out["first"] = _timestamp(seconds[0])
    out["last"] = _timestamp(seconds[-1])

    densest = _densest_chapter(
        name, chapters,
        chapter_shares if chapter_shares else share_maps(chapters))
    if densest:
        out["densest_chapter"] = densest

    if levels:
        level = _level(name, seconds, levels, labels_by_second or {},
                       video_median)
        if level:
            out["level"] = level

    clips = _clips(seconds, segments)
    if clips:
        out["clips"] = clips
    return out


def attach(chapters: Sequence[Mapping],
           names: Iterable[str],
           *,
           labels_by_second: Optional[Mapping] = None,
           segments: Sequence[Sequence[float]] = (),
           levels: Sequence[float] = (),
           detected_seconds: int = 0,
           limit: int = MAX_PER_CHAPTER) -> list[dict]:
    """Chapters, each carrying the measurements for whatever it names.

    Returns new dicts. The input is not mutated for the same reason
    :func:`modules.chapter_compare.summarise_chapters` does not mutate: the
    timeline holds those objects, and a report quietly growing them is how two
    views end up disagreeing about one run.
    """
    rows = [dict(ch) for ch in (chapters or [])]
    names = [str(n) for n in (names or ())]
    if not rows or not names:
        return rows

    labels_by_second = labels_by_second or {}
    levels = list(levels or [])
    video_median = None
    if levels:
        ordered = sorted(float(v) for v in levels)
        mid = len(ordered) // 2
        video_median = (ordered[mid] if len(ordered) % 2
                        else (ordered[mid - 1] + ordered[mid]) / 2.0)

    by_name = _seconds_by_name(labels_by_second)
    if not detected_seconds:
        detected_seconds = len(labels_by_second)

    # One measurement per class however many chapters name it. The figures are
    # video-wide, so recomputing them per mention would spend the work to
    # produce the same numbers — and risk producing different ones.
    shares = share_maps(rows, labels_by_second)

    measured: dict = {}
    for chapter, here_shares in zip(rows, shares):
        found = mentions(chapter, names, limit=limit)
        if not found:
            continue
        out = []
        for row in found:
            name = row["name"]
            if name not in measured:
                measured[name] = measure(
                    name, seconds=by_name.get(name, []), chapters=rows,
                    chapter_shares=shares,
                    segments=segments, levels=levels,
                    labels_by_second=labels_by_second,
                    video_median=video_median,
                    detected_seconds=detected_seconds)
            row = dict(row)
            row.update(measured[name])
            # What the stretch the claim was made in holds of it. Usually zero,
            # and that is the useful case: it says the speaker was describing
            # something that happened elsewhere, which is what people do.
            #
            # A missing key is zero, not unknown -- the map lists every class
            # with a share in the chapter, so a name absent from it was absent
            # from the stretch. A chapter with no map at all is the case that
            # differs: nothing was detected there, and "0% of this stretch"
            # would pass a fact about the detector off as one about the content.
            if here_shares is not None:
                row["here_share_pct"] = round(
                    float(here_shares.get(name, 0.0)), 1)
            out.append(row)
        chapter["spoken_evidence"] = out
    return rows
