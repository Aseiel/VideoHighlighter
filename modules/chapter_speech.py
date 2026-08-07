"""What was said in each chapter, and what that says about the chapter.

``modules/chapters.py`` divides a video on how it *looks* and refuses to name
the result, for a stated reason: a vector is a point, not a name, and this repo
ships no vocabulary to match points against. Its docstring names the way out --
a caller that has a vocabulary can label the partition afterwards -- and a
transcript is exactly that, with the useful property that it came out of the
user's own footage rather than out of a list somebody shipped.

So this module is the labelling half. It takes the partition and the Whisper
segments and, per chapter, measures:

* how much of the stretch is speech at all, against the video's own rate;
* which words this stretch used that the other stretches did not;
* the lines carrying those words, timestamped so they can be played;
* who was speaking, when diarization ran.

Distinctiveness is inverse document frequency over the chapters, and the choice
matters more than it looks. A stop-word list would have to be per-language and
would be wrong the moment someone transcribes a language nobody thought about;
IDF needs no list, because a word used in every chapter carries no information
about any of them and falls out with a weight of zero. The video supplies its
own stop words. That also keeps this module honest about the repo's rule that
no vocabulary ships here: nothing below contains a word of anybody's content.

What this is not
----------------

It does not interpret. "These words are unusual for this stretch of this video"
is a measurement; "this is the scene where X happens" is a reading, and the
reading belongs to the person -- or, clearly labelled as such, to the narrator
in :mod:`modules.advisor`, which gets these quotes in its prompt and is the one
component allowed to speculate. The distinction is the whole reason a report
that quotes the footage is worth more than one that paraphrases it: a quote can
be checked against the video in one click, and a paraphrase cannot.

Speech is also not evidence of importance, and this module never treats it as
such. A stretch can be silent and be the point of the video. All the numbers
here describe the soundtrack; none of them rank it.
"""
from __future__ import annotations

import math
import re
from typing import Iterable, Mapping, Optional, Sequence

# Letters only, Unicode-aware, so a Polish or Japanese transcript tokenises the
# same way an English one does. Digits are dropped: "20" is a real word in a
# transcript and a useless one for telling two chapters apart, since numbers
# recur everywhere and carry no subject.
WORD_RE = re.compile(r"[^\W\d_]{2,}", re.UNICODE)

# A word has to be said this many times inside a chapter before it can stand as
# a description of it. One occurrence of a rare word is as likely to be Whisper
# guessing at a mumble as it is to be the subject, and a chapter titled from a
# mis-transcription is worse than one left unnamed.
MIN_WORD_COUNT = 2

# How many times more often a word must be said in a chapter than in the rest of
# the video before it describes that chapter. Three is low enough to catch a
# subject raised repeatedly in one stretch and high enough to reject the whole
# of ordinary speech, which sits within a factor of two of its own average
# almost everywhere. This is the number that stopped chapters being titled with
# words like "was" and "today".
MIN_KEYNESS = 3.0

# How many distinctive words are kept per chapter. Three is what a title-length
# phrase holds; past that the tail is words that only just cleared the bar.
TOP_WORDS = 3

# Below this many words a chapter has not said enough for "the words it used
# that others did not" to mean anything -- the IDF ranking of nine words is
# mostly noise. Such a chapter reports its speech share and no title.
MIN_WORDS_FOR_TITLE = 25

# Lines quoted per chapter. Enough to show a stretch's register and subject,
# few enough that the chapter list stays a list. A reader who wants all of it
# has the transcript panel and the SRT.
QUOTES_PER_CHAPTER = 3

# Quotes must be at least this far apart. Without it the three highest-scoring
# lines are routinely three consecutive lines of one exchange, which describes
# ten seconds of a nine-minute chapter and reads as if nothing else was said.
MIN_QUOTE_GAP = 20.0

# A quote longer than this is trimmed on a word boundary. Whisper occasionally
# emits a single segment covering a long uninterrupted stretch, and one such
# line can be taller than the chapter block it sits in.
MAX_QUOTE_CHARS = 180

# How far a chapter's speech share must sit from the video's own before the
# difference is worth a sentence. Speech detection is noisy at the edges of
# segments, and a chapter running a few points off the average is that noise.
SPEECH_LIFT = 1.6
SPEECH_DROP = 0.6

# Percentage points of speech share between neighbouring chapters that make a
# boundary worth remarking on. This is the "the talking stops here" line, and it
# has to clear enough to not fire on every boundary in a normal conversation.
SPEECH_CHANGE_POINTS = 25.0

# Below this share a stretch is not speech with pauses, it is silence with the
# occasional word, and saying "12% speech" invites a reader to picture dialogue.
NEARLY_SILENT_PCT = 5.0


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Lowercased word-ish tokens. No stemming, no stop list -- see the header."""
    return WORD_RE.findall(str(text or "").lower())


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Seconds two spans share."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def segments_in(transcript: Sequence[Mapping],
                start: float, end: float) -> list[Mapping]:
    """Transcript segments that overlap ``[start, end)`` at all.

    Overlap rather than containment: a line spoken across a chapter boundary
    belongs to both stretches it was audible in, and dropping it would silently
    lose exactly the lines that describe a transition.
    """
    return [s for s in (transcript or [])
            if _overlap(float(s.get("start") or 0.0),
                        float(s.get("end") or 0.0), start, end) > 0]


def speech_seconds(transcript: Sequence[Mapping],
                   start: float, end: float) -> float:
    """Seconds of ``[start, end)`` covered by speech, counting overlaps once.

    Whisper segments can overlap each other after chunk offsets are applied, so
    summing durations overstates the total; this merges first.
    """
    spans = sorted((max(start, float(s.get("start") or 0.0)),
                    min(end, float(s.get("end") or 0.0)))
                   for s in segments_in(transcript, start, end))
    total, cursor = 0.0, None
    for lo, hi in spans:
        if hi <= lo:
            continue
        if cursor is None or lo > cursor:
            total += hi - lo
            cursor = hi
        elif hi > cursor:
            total += hi - cursor
            cursor = hi
    return total


# ---------------------------------------------------------------------------
# Distinctiveness
# ---------------------------------------------------------------------------
def keyness(documents: Sequence[Sequence[str]]) -> list[dict]:
    """Per chapter, ``{word: how many times more often it is said here}``.

    Document frequency is not enough, and finding that out cost a report. On a
    sixteen-chapter video the first chapter was labelled *being, was, today* --
    all three perfectly ordinary words that simply did not happen to occur in
    all sixteen, so a binary "in how many documents" test scored them as rare
    while their high raw counts carried them to the top. Any tf-idf over a
    handful of long documents has that failure in it.

    The fix is to compare *rates* rather than presence. A function word occurs
    at about the same rate wherever people talk, so its rate here divided by its
    rate everywhere else is about one no matter how often it is said; a word
    that belongs to this stretch is many times more frequent here than
    elsewhere. That ratio is what this returns, and it needs no list of words to
    ignore -- the rest of the video is the list.
    """
    documents = [list(d) for d in documents]
    lengths = [len(d) for d in documents]
    if sum(lengths) == 0 or len([d for d in documents if d]) < 2:
        # One chapter has nothing to be distinctive *against*: every word is as
        # typical of the video as it is of the chapter. Returning empty maps
        # makes the caller skip the ranking instead of ranking noise.
        return [{} for _ in documents]

    totals: dict = {}
    for doc in documents:
        for word in doc:
            totals[word] = totals.get(word, 0) + 1
    grand = float(sum(lengths))

    out = []
    for doc, here_total in zip(documents, lengths):
        weights: dict = {}
        if not here_total:
            out.append(weights)
            continue
        counts: dict = {}
        for word in doc:
            counts[word] = counts.get(word, 0) + 1
        elsewhere_total = grand - here_total
        for word, count in counts.items():
            elsewhere = totals[word] - count
            # Add-half smoothing on the "elsewhere" side only. Without it a word
            # said nowhere else divides by zero; with it, one said nowhere else
            # scores in proportion to how much other material it is absent
            # from, which is the right ordering -- being the only chapter to use
            # a word means more across an hour than across a minute.
            rate_here = count / float(here_total)
            rate_there = (elsewhere + 0.5) / (elsewhere_total + 0.5)
            weights[word] = rate_here / rate_there if rate_there > 0 else 0.0
        out.append(weights)
    return out


def distinctive_words(words: Sequence[str], weights: Mapping,
                      limit: int = TOP_WORDS) -> list[dict]:
    """The words this chapter says far more often than the rest of the video.

    Ranked on the ratio alone. Weighting it by how much of the chapter the word
    makes up sounds more balanced and is not: term frequency is the thing that
    put *being, was, today* on a chapter in the first place, and multiplying by
    it hands the ranking straight back to whatever was said most. Measured on a
    real sixteen-chapter transcript, ratio alone returned *busy, draw, fantasy*
    for the same stretch.

    Count still decides between words that appear *only* here, because for those
    the ratio grows with the count — so the guard against a fluke is the
    threshold on both, not the sort.
    """
    if not words or not weights:
        return []
    counts: dict = {}
    for word in words:
        counts[word] = counts.get(word, 0) + 1
    total = float(len(words))
    scored = []
    for word, count in counts.items():
        ratio = float(weights.get(word, 0.0))
        if count < MIN_WORD_COUNT or ratio < MIN_KEYNESS:
            continue
        scored.append({"word": word,
                       "count": count,
                       "share_pct": round(100.0 * count / total, 2),
                       "times": round(ratio, 1)})
    scored.sort(key=lambda d: (-d["times"], -d["count"], d["word"]))
    return scored[:limit]


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------
def _trim(text: str) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= MAX_QUOTE_CHARS:
        return text
    cut = text[:MAX_QUOTE_CHARS].rsplit(" ", 1)[0]
    return (cut or text[:MAX_QUOTE_CHARS]).rstrip(",;:-") + "…"


def _timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def quotes_for(transcript: Sequence[Mapping], start: float, end: float,
               weights: Optional[Mapping] = None,
               limit: int = QUOTES_PER_CHAPTER,
               min_gap: float = MIN_QUOTE_GAP) -> list[dict]:
    """The lines worth showing from ``[start, end)``.

    Ranked by how much of the chapter's distinctive vocabulary each line
    carries, divided by the square root of its length so a long line cannot win
    by accumulating weak words. Picks are then spaced across the chapter: the
    three top-scoring lines are otherwise usually three consecutive lines of one
    exchange, which is a fair description of ten seconds and of nothing else.
    """
    here = segments_in(transcript, start, end)
    if not here:
        return []

    scored = []
    for index, seg in enumerate(here):
        words = tokenize(seg.get("text"))
        if not words:
            continue
        # Log of the rate ratio, so a line is scored by how unusual its words
        # are for the video rather than by how many it has. Words at or below
        # the video's own rate contribute nothing instead of counting against.
        weight = sum(max(0.0, math.log(max(float((weights or {}).get(w, 0.0)), 1e-9)))
                     for w in words)
        scored.append((weight / math.sqrt(len(words)), -index, seg))
    if not scored:
        return []
    scored.sort(key=lambda t: (-t[0], -t[1]))

    picked: list = []
    for _weight, _order, seg in scored:
        at = float(seg.get("start") or 0.0)
        if any(abs(at - float(p.get("start") or 0.0)) < min_gap for p in picked):
            continue
        picked.append(seg)
        if len(picked) >= limit:
            break
    # A chapter with a single long monologue would otherwise return nothing but
    # its first line; if spacing rejected everything, take the best line alone.
    if not picked:
        picked = [scored[0][2]]

    picked.sort(key=lambda s: float(s.get("start") or 0.0))
    out = []
    for seg in picked:
        at = float(seg.get("start") or 0.0)
        quote = {"start": round(at, 2),
                 "timestamp": _timestamp(at),
                 "text": _trim(seg.get("text"))}
        speaker = str(seg.get("speaker") or "").strip()
        if speaker and speaker.upper() != "UNKNOWN":
            quote["speaker"] = speaker
        out.append(quote)
    return out


# ---------------------------------------------------------------------------
# Speakers
# ---------------------------------------------------------------------------
def speakers_in(transcript: Sequence[Mapping],
                start: float, end: float) -> list[dict]:
    """Who spoke here and for how long, when diarization tagged the segments.

    Returns ``[]`` when nothing carries a speaker, which is the normal case --
    diarization is optional and off by default.
    """
    tally: dict = {}
    for seg in segments_in(transcript, start, end):
        name = str(seg.get("speaker") or "").strip()
        if not name or name.upper() == "UNKNOWN":
            continue
        row = tally.setdefault(name, {"speaker": name, "turns": 0,
                                      "seconds": 0.0, "words": 0})
        row["turns"] += 1
        row["seconds"] += _overlap(float(seg.get("start") or 0.0),
                                   float(seg.get("end") or 0.0), start, end)
        row["words"] += len(tokenize(seg.get("text")))
        gender = str(seg.get("gender") or "").strip().lower()
        # Carried only when the estimator was confident enough to commit; the
        # module reports "unknown" freely and a report should not print it as
        # if it were a finding.
        if gender and gender != "unknown":
            row["gender"] = gender
    rows = sorted(tally.values(), key=lambda r: -r["seconds"])
    for row in rows:
        row["seconds"] = round(row["seconds"], 1)
    return rows


# ---------------------------------------------------------------------------
# The whole video, and then each chapter against it
# ---------------------------------------------------------------------------
def video_speech(transcript: Sequence[Mapping],
                 video_duration: float = 0.0) -> dict:
    """The reference every chapter is measured against."""
    transcript = list(transcript or [])
    if not transcript:
        return {}
    words = sum(len(tokenize(s.get("text"))) for s in transcript)
    end = float(video_duration or 0.0) or max(
        float(s.get("end") or 0.0) for s in transcript)
    spoken = speech_seconds(transcript, 0.0, end)
    out = {
        "segments": len(transcript),
        "words": words,
        "speech_seconds": round(spoken, 1),
        "speech_share_pct": round(100.0 * spoken / end, 1) if end else 0.0,
    }
    if end:
        out["words_per_minute"] = round(words / (end / 60.0), 1)
    speakers = speakers_in(transcript, 0.0, end)
    if speakers:
        out["speakers"] = speakers
    return out


def summarise_speech(chapters: Sequence[Mapping],
                     transcript: Sequence[Mapping],
                     *,
                     video_duration: float = 0.0) -> list[dict]:
    """Each chapter plus what was said in it. Returns new dicts, JSON-safe.

    The input is not mutated, for the same reason
    :func:`modules.chapter_compare.summarise_chapters` does not: the timeline
    holds those objects, and a renderer quietly growing them is how two views
    end up disagreeing about the same run.

    Chapters with no speech are returned unchanged apart from a zero share, so
    "this stretch is silent" stays a visible fact rather than a missing key.
    """
    rows = [dict(ch) for ch in (chapters or [])]
    transcript = [s for s in (transcript or []) if str(s.get("text") or "").strip()]
    if not rows or not transcript:
        return rows

    reference = video_speech(transcript, video_duration)
    video_share = float(reference.get("speech_share_pct") or 0.0)

    per_chapter_words = []
    for ch in rows:
        start, end = float(ch["start"]), float(ch["end"])
        words = []
        for seg in segments_in(transcript, start, end):
            words.extend(tokenize(seg.get("text")))
        per_chapter_words.append(words)

    weights_per_chapter = keyness(per_chapter_words)

    for ch, words, weights in zip(rows, per_chapter_words, weights_per_chapter):
        start, end = float(ch["start"]), float(ch["end"])
        seconds = max(0.0, end - start)
        spoken = speech_seconds(transcript, start, end)
        share = (100.0 * spoken / seconds) if seconds else 0.0

        ch["speech_seconds"] = round(spoken, 1)
        ch["speech_share_pct"] = round(share, 1)
        ch["words"] = len(words)
        if seconds:
            ch["words_per_minute"] = round(len(words) / (seconds / 60.0), 1)
        if video_share > 0:
            # >1 means this stretch talks more than the video does on average.
            ch["speech_lift"] = round(share / video_share, 2)

        if not words:
            continue

        found = distinctive_words(words, weights)
        if found:
            ch["speech_words"] = found
            if len(words) >= MIN_WORDS_FOR_TITLE:
                # Not a title in the sense of a name somebody chose: it is the
                # vocabulary this stretch used and the others did not, which is
                # the strongest claim the measurement supports.
                ch["speech_title"] = ", ".join(f["word"] for f in found)

        quotes = quotes_for(transcript, start, end, weights)
        if quotes:
            ch["quotes"] = quotes
        speakers = speakers_in(transcript, start, end)
        if speakers:
            ch["speakers"] = speakers
        # Everything said in the stretch, in order. Carried in the record for
        # the same reason `score_per_second` is: it is what lets a later pass —
        # a narrator, a reader scrolling the page — work from the report without
        # the video, the cache or a re-run. Three quotes are enough to show what
        # a chapter sounded like and nowhere near enough to tell anyone what
        # happened in it.
        dialogue = []
        for seg in sorted(segments_in(transcript, start, end),
                          key=lambda s: float(s.get("start") or 0.0)):
            text = _trim(seg.get("text"))
            if not text:
                continue
            at = float(seg.get("start") or 0.0)
            line = {"start": round(at, 2), "timestamp": _timestamp(at),
                    "text": text}
            speaker = str(seg.get("speaker") or "").strip()
            if speaker and speaker.upper() != "UNKNOWN":
                line["speaker"] = speaker
            dialogue.append(line)
        if dialogue:
            ch["dialogue"] = dialogue

    _mark_speech_changes(rows)
    return rows


def _mark_speech_changes(chapters: Sequence[Mapping]) -> None:
    """Where the soundtrack changes character between neighbours. Mutates.

    A chapter compared with the video says how talky it is; a chapter compared
    with the one before it says the talking started or stopped *here*, and only
    the second reads as one thing following another. This is the same move
    :func:`modules.chapter_compare._mark_changes` makes on the visual classes.
    """
    previous = None
    for ch in chapters:
        share = float(ch.get("speech_share_pct") or 0.0)
        if previous is not None:
            delta = share - previous
            if abs(delta) >= SPEECH_CHANGE_POINTS:
                ch["speech_change"] = {
                    "direction": "rose" if delta > 0 else "fell",
                    "from_pct": round(previous, 1),
                    "to_pct": round(share, 1),
                }
        previous = share


# ---------------------------------------------------------------------------
# Clips
# ---------------------------------------------------------------------------
def clip_speech(transcript: Sequence[Mapping],
                start: float, end: float,
                limit: int = QUOTES_PER_CHAPTER) -> dict:
    """What was said during one kept clip.

    No IDF here: a 30-second clip has no vocabulary to be distinctive with, and
    ranking three lines against each other invents a difference. Every line is
    returned in order, trimmed, up to the limit -- for a clip that is the whole
    of what was said, which is the answerable question at this length.
    """
    here = segments_in(transcript, start, end)
    if not here:
        return {}
    lines = []
    for seg in sorted(here, key=lambda s: float(s.get("start") or 0.0))[:limit]:
        text = _trim(seg.get("text"))
        if not text:
            continue
        at = float(seg.get("start") or 0.0)
        line = {"start": round(at, 2), "timestamp": _timestamp(at), "text": text}
        speaker = str(seg.get("speaker") or "").strip()
        if speaker and speaker.upper() != "UNKNOWN":
            line["speaker"] = speaker
        lines.append(line)
    if not lines:
        return {}
    spoken = speech_seconds(transcript, start, end)
    seconds = max(0.0, float(end) - float(start))
    out = {"lines": lines,
           "speech_seconds": round(spoken, 1),
           "total": len(here)}
    if seconds:
        out["speech_share_pct"] = round(100.0 * spoken / seconds, 1)
    return out
