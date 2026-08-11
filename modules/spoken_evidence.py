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
speaker naming their favourite moment, over a label the run found in a handful
of seconds and put in none of the kept clips, is the most useful row this report
can produce, and it is exactly the row a transcript alone cannot make.

Matching: two tiers, and why one is not enough
----------------------------------------------

The obvious rule is that a line names a class when every token of the class name
appears in it. That rule was written first and it fired on **nothing**, on real
footage, against classes that were plainly the subject of the conversation.

People abbreviate and elide. A class name is a label somebody typed once; speech
shortens it, drops the second half, or contracts it to an initial that
tokenising discards for being one character long. Matching the full name only
finds the speaker who happened to say the label the way it was typed, which is
the rarest way of saying it. The rule was silently strict, and a section that
never appears looks exactly like a section that has nothing to report.

So there are two tiers, and both are token equality — no stemming, no synonyms,
no lexicon, for the reason :mod:`modules.vocabulary_gap` gives: a synonym table
is a vocabulary, this repo ships none, and one written for English would be
wrong the first time somebody transcribes anything else.

**Named in full.** Some line covers every token of a class name. Strong, quoted
as-is, and not required to be distinctive of anything — saying the whole label
is evidence enough on its own.

**Named by a word.** One of the words this stretch says far more often than the
rest of the video (:func:`modules.chapter_speech.distinctive_words`) is a token
of some class name. This is the tier that fires in practice, and the keyness
bar is what keeps it honest: a class named after an ordinary word cannot match
unless the stretch is *about* that word, so a common token does not paint the
whole report.

A word will often be a token of several class names at once, and that is
reported rather than resolved: the row names every class the word covers, each
with its own measurement. Which one the speaker meant is not decidable here, and
the comparison between them is usually the answer anyway — the same word
covering one class the run found everywhere and another it barely found at all
says more than either figure alone.

What a match does not mean
--------------------------

That the same words were used. Nothing else. The speaker may be using a word in
a sense with no relation to what the detector was taught to find, and this
module cannot tell. Every row therefore carries the line it was matched from,
and the narrator is told this in as many words, so a reader — or a model — can
see the sentence and decide whether the two are about the same thing at all.

Where the evidence is measured from
-----------------------------------

The **whole video**, not the stretch the sentence was said in. People describe
in the last five minutes something that happened in the middle, and a row that
looked only at the current chapter would answer "your favourite moment never
happened" about a moment that plainly did. The chapter the claim was *made* in
decides where the row is filed; the chapter the label is *densest* in is part
of what the row says. :func:`modules.chapter_story._checks_here` files its rows
the same way and for the same reason.

Two ways in
-----------

:func:`attach` runs inside :func:`modules.highlight_report.build_report`, off
the per-second labels, and measures everything.

:func:`from_report` derives what it can from a finished report on disk, and
exists because of how this feature is actually used. Analysis is expensive and
run once; the narration is re-run over the saved record afterwards, as often as
the user likes. Without a backfill this section could only ever appear on a
video re-analysed from scratch, which for the report it most wants to improve
means paying for the detectors again to add a paragraph. The record does not
carry per-second levels, so that path reports a class's loudest *kept clip*
instead of its loudest second, and says which of the two it has.

Everything reported is arithmetic over figures the report already holds.
Nothing here interprets, ranks a claim as true or false, or knows one class name
from another.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional, Sequence

# Things-said kept per chapter. A stretch naming five measured classes is a
# stretch where somebody listed things, and five blocks of arithmetic under one
# paragraph buries the paragraph.
MAX_PER_CHAPTER = 3

# Candidate classes carried on one row, when a word is a token of several names.
# Ranked by how much of each the run found, so the ones a reader would weigh
# first survive the cut.
MAX_CLASSES_PER_ROW = 3

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


def _named(names: Iterable[str]) -> dict:
    """``{class name: its token set}``, skipping names that tokenise to nothing."""
    from modules.vocabulary_gap import tokens_of

    out = {}
    for name in (names or ()):
        tokens = tokens_of(name)
        if tokens:
            out[str(name)] = tokens
    return out


def _line_row(line: Mapping, said: str, kind: str, classes: list) -> dict:
    row = {"said": said, "kind": kind, "classes": classes,
           "said_at": round(float(line.get("start") or 0.0), 2),
           "timestamp": str(line.get("timestamp")
                            or _timestamp(line.get("start") or 0)),
           "quote": str(line.get("text") or "")}
    speaker = str(line.get("speaker") or "").strip()
    if speaker and speaker.upper() != "UNKNOWN":
        row["speaker"] = speaker
    return row


def mentions(chapter: Mapping,
             names: Iterable[str],
             *,
             limit: int = MAX_PER_CHAPTER) -> list[dict]:
    """What this stretch says that something the run measured is named after.

    Rows are things *said*, not classes: one word can be a token of several
    class names, and splitting that into a row each would repeat the same
    sentence three times and lose the comparison between them.

    Full-name matches rank first because they are the stronger evidence, then
    distinctive words by how far above the video's own rate they were said.
    """
    from modules.chapter_speech import tokenize

    named = _named(names)
    if not named:
        return []
    lines = list(chapter.get("dialogue") or chapter.get("quotes") or [])

    # --- named in full ---------------------------------------------------
    # Two tokens or more. A one-word class name is "covered in full" by any
    # line that happens to say the word, which on real footage produced the
    # same row in six chapters of sixteen — no keyness, no quote worth
    # reading, and enough of them to teach a reader to skip the section. Those
    # names can still match, through the distinctive-word tier below, where the
    # stretch has to actually be about the word.
    exact: dict = {}
    for line in lines:
        words = set(tokenize(line.get("text")))
        if not words:
            continue
        for name, tokens in named.items():
            if len(tokens) > 1 and name not in exact and tokens <= words:
                exact[name] = line

    rows = []
    for name, line in exact.items():
        rows.append(_line_row(line, name, "name", [name]))

    # --- named by a distinctive word --------------------------------------
    # Skipping any class already matched in full: the stronger row for it
    # exists, and listing it again under a word it shares adds nothing.
    for found in (chapter.get("speech_words") or []):
        word = str(found.get("word") or "")
        if not word:
            continue
        covering = sorted(n for n, tokens in named.items()
                          if word in tokens and n not in exact)
        if not covering:
            continue
        line = _quote_containing(lines, word)
        if line is None:
            continue
        row = _line_row(line, word, "word", covering)
        row["count"] = int(found.get("count") or 0)
        row["times"] = round(float(found.get("times") or 0.0), 1)
        rows.append(row)

    rows.sort(key=lambda r: (r["kind"] != "name", -float(r.get("times") or 0.0),
                             r["said_at"]))
    return rows[:limit]


def _quote_containing(lines: Sequence[Mapping], word: str) -> Optional[Mapping]:
    """The first line in the stretch that actually contains the word.

    A row a reader cannot picture is one they cannot judge, and the whole
    defence against a coincidental token match is that the sentence is printed
    beside it.
    """
    from modules.chapter_speech import tokenize

    for line in lines:
        if word in tokenize(line.get("text")):
            return line
    return None


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
               "level_dbfs": round(float(levels[peak]), 2),
               "scope": "second"}
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

    Deliberately reports a class the detector barely found. "Labelled in nine
    seconds of the whole file" is not a failure of this function — beside a
    sentence calling it the best part of the video, it is the finding.
    """
    out: dict = {"name": name, "seconds": len(seconds)}
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

    out["clips"] = _clips(seconds, segments)
    return out


def _fill(rows: Sequence[Mapping],
          chapters: Sequence[Mapping],
          shares: Sequence[Optional[Mapping]],
          measured: Mapping,
          here_shares: Optional[Mapping]) -> list:
    """Put the measurement for each candidate class onto one chapter's rows."""
    out = []
    for row in rows:
        classes = []
        for name in row["classes"]:
            entry = dict(measured.get(name) or {"name": name})
            # What the stretch the claim was made in holds of it. Usually
            # nothing, and that is the useful case: it says the speaker was
            # describing something that happened elsewhere, which is what
            # people do.
            #
            # A missing key is zero, not unknown -- the map lists every class
            # with a share in the chapter, so a name absent from it was absent
            # from the stretch. A chapter with no map at all is the case that
            # differs: nothing was detected there, and "0% of this stretch"
            # would pass a fact about the detector off as one about the content.
            if here_shares is not None:
                entry["here_share_pct"] = round(
                    float(here_shares.get(name, 0.0)), 1)
            classes.append(entry)
        # Most-measured first: with one word covering several names, the reader
        # weighs the one the run actually found before the one it did not.
        classes.sort(key=lambda c: -int(c.get("seconds") or 0))
        row = dict(row)
        row["classes"] = classes[:MAX_CLASSES_PER_ROW]
        out.append(row)
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
    shares = share_maps(rows, labels_by_second)

    # One measurement per class however many chapters name it. The figures are
    # video-wide, so recomputing them per mention would spend the work to
    # produce the same numbers — and risk producing different ones.
    measured: dict = {}
    for chapter, here_shares in zip(rows, shares):
        found = mentions(chapter, names, limit=limit)
        if not found:
            continue
        for row in found:
            for name in row["classes"]:
                if name not in measured:
                    measured[name] = measure(
                        name, seconds=by_name.get(name, []), chapters=rows,
                        chapter_shares=shares, segments=segments,
                        levels=levels, labels_by_second=labels_by_second,
                        video_median=video_median,
                        detected_seconds=detected_seconds)
        chapter["spoken_evidence"] = _fill(found, rows, shares, measured,
                                           here_shares)
    return rows


# ---------------------------------------------------------------------------
# From a finished report, without the caches or a re-run
# ---------------------------------------------------------------------------
def _from_record(name: str, report: Mapping, shares: Sequence) -> dict:
    """One class, measured from whatever the saved record kept about it.

    Less than :func:`measure` has, and it says which parts it is missing rather
    than guessing at them. The two figures that survive intact are the ones the
    section is read for: how much of the video carries the class, and which
    stretch holds it.
    """
    chapters = report.get("chapters") or []
    out: dict = {"name": name, "from_record": True}

    # Total labelled seconds. `level_by_class` counted them already, but only
    # for classes it described — and it is handed the composed events alone
    # when a run has any, so absence there means "under its minimum" only for
    # a name that was in its scope.
    by_class = {str(row.get("name")): row
                for row in ((report.get("level_by_class") or {}).get("classes")
                            or [])}
    row = by_class.get(name)
    if row:
        out["seconds"] = int(row.get("seconds") or 0)
        out["level"] = {"median_dbfs": row.get("median_dbfs"),
                        "seconds": int(row.get("seconds") or 0)}
    else:
        # The per-clip comparison carries a video-wide count of its own.
        for segment in (report.get("segments") or []):
            for subject in (((segment.get("measured") or {}).get("comparison")
                             or {}).get("subjects") or []):
                if str(subject.get("name")) == name and subject.get("detections"):
                    out["seconds"] = int(subject["detections"])
                    break
            if "seconds" in out:
                break
    events = list((report.get("vocabulary") or {}).get("events") or [])
    if "seconds" not in out and by_class and (not events or name in events):
        # Labelled somewhere — it has chapter shares — but under the bar the
        # level summary reports at. Said as the bound it is, not as a count.
        from modules.level_by_class import MIN_SECONDS
        out["under_seconds"] = int(MIN_SECONDS)

    densest = _densest_chapter(name, chapters, shares)
    if densest:
        out["densest_chapter"] = densest

    # No per-second levels survive in the record, so the loudest second of the
    # class cannot be recovered. The loudest second of a *kept clip* carrying
    # it can, and it is marked as the narrower thing it is.
    loudest = None
    for segment in (report.get("segments") or []):
        found = segment.get("loudest") or {}
        if name not in [str(c) for c in (found.get("classes") or [])]:
            continue
        if loudest is None or float(found.get("level_dbfs") or -999) > \
                float(loudest.get("level_dbfs") or -999):
            loudest = dict(found)
    global_loudest = (report.get("level_by_class") or {}).get("loudest") or {}
    if name in [str(c) for c in (global_loudest.get("classes") or [])]:
        loudest, scope = dict(global_loudest), "video"
    else:
        scope = "clip"
    if loudest:
        alongside = sorted({str(c) for c in (loudest.get("classes") or ())}
                           - {name})
        entry = {"second": loudest.get("second"),
                 "timestamp": str(loudest.get("timestamp") or ""),
                 "level_dbfs": loudest.get("level_dbfs"),
                 "scope": scope}
        if loudest.get("vs_video_db") is not None:
            entry["vs_video_db"] = loudest["vs_video_db"]
        if alongside:
            entry["with"] = alongside
        out.setdefault("level", {})["loudest"] = entry

    clips = []
    for segment in (report.get("segments") or []):
        carried = ({str(o) for o in (segment.get("objects") or ())}
                   | {str(e) for e in (segment.get("events") or ())}
                   | {str(c) for c in ((segment.get("loudest") or {})
                                       .get("classes") or ())})
        if name in carried:
            clips.append(int(segment.get("index") or 0))
    out["clips"] = clips
    return out


def from_report(report: Mapping, *, limit: int = MAX_PER_CHAPTER) -> list[dict]:
    """Backfill the evidence onto a saved report's chapters. Returns new dicts.

    For reports written before this existed, and for the ordinary workflow of
    re-telling the narration over a record whose analysis was run hours ago.
    Everything it needs is already in the file.
    """
    chapters = [dict(ch) for ch in (report.get("chapters") or [])]
    vocabulary = report.get("vocabulary") or {}
    names = ([str(n) for n in (vocabulary.get("classes") or [])]
             + [str(n) for n in (vocabulary.get("events") or [])])
    if not chapters or not names:
        return chapters

    shares = share_maps(chapters)
    measured: dict = {}
    for chapter, here_shares in zip(chapters, shares):
        found = mentions(chapter, names, limit=limit)
        if not found:
            continue
        for row in found:
            for name in row["classes"]:
                if name not in measured:
                    measured[name] = _from_record(name, report, shares)
        chapter["spoken_evidence"] = _fill(found, chapters, shares, measured,
                                           here_shares)
    return chapters
