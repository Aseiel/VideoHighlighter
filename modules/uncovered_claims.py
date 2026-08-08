"""Lines this report already quotes that nothing in the run measured.

:mod:`modules.spoken_evidence` finds the sentences whose words a class name
covers, and answers them with figures. This is the other side of the same
comparison, and it exists because of what the two sides look like on a page: a
chapter block where somebody names something the detector found carries a
measurement, and the block right next to it, where somebody said something just
as substantial about something nothing was watching for, carries **the quote and
silence**. A reader cannot tell that second case from a claim that was checked
and passed. Both look like a sentence nobody argued with.

Why not :mod:`modules.vocabulary_gap`
-------------------------------------

That module answers a different question and answers it well: which *words* does
a stretch keep coming back to that no class covers. Its bar is frequency —
several occurrences, far above the video's own rate — and that bar is right for
what it does, because a subject a stretch is *about* is a subject it repeats.

The most informative thing anybody says in a video is often said exactly once.
Somebody states a preference, names what they were hoping for, says what they
liked most. It is one sentence, it never repeats, and it is the sentence the
whole report should be tested against — and a frequency detector cannot see it,
at any threshold, because there is nothing frequent about it. Measured on a
real transcript: the claim was one line, the words carrying it were said once,
and the eight gaps that *were* reported for the same file were filler and stray
nouns. This module is what makes that sentence visible.

The candidate set is what the report already prints
---------------------------------------------------

Not every line of dialogue — the chapter's kept quotes, from
:func:`modules.chapter_speech.quotes_for`. Three reasons, and the second one is
not obvious.

**They are already the substantial lines.** A quote is picked for carrying
vocabulary unusual for this video, which is as close to "contains an assertion"
as anything measurable here gets.

**They are already de-duplicated.** Whisper loops on hard audio and emits the
same sentence eight times in a row; ranked over raw dialogue, those eight
copies take the entire list. Quotes are spaced across the chapter, so the loop
contributes one line. This was not predicted — it was what the first pass over
real footage returned.

**They are on the page.** The claim this reports is one the reader is being
shown anyway. That makes the finding honest in a way a line dug out of the
transcript would not be: the report is not raising something obscure, it is
admitting that a sentence it chose to print has nothing beside it.

The bar, and what it excludes
-----------------------------

A line qualifies when **no** token of it is a token of any class or event name —
if one is, :mod:`modules.spoken_evidence` has the line and has figures for it,
and reporting it here as unmeasured would contradict the block above it.

Length does the rest. "Uh huh." is a chapter's top quote on real footage and
genuinely distinctive of it, and it is not a claim about anything. Ranking by
total unusual vocabulary rather than by density — the opposite of what
:func:`quotes_for` does — is what separates the two: a claim is a sentence, and
density rewards the shortest line that contains one rare word. The same crude
proxy as the seed in ``main.py``'s rule dialog, for the same reason and with
the same honesty about what it is.

No lexicon decides any of it, here or anywhere below. Nothing in this file
knows what a claim is *about*, and the ranking would behave the same on a
transcript in any language.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional, Sequence

# Tokens a line needs before it can be reported as a claim. Eight is where the
# backchannels stop on real footage -- "All right, all right, all right." is six
# and scores high on any vocabulary measure, because in a chapter that says it
# twice it genuinely is unusual for the video.
MIN_WORDS = 8

# Rows kept for the whole report. This is a list a person acts on one item from,
# and every row costs a paragraph of what-to-do-about-it, so the same reasoning
# as `vocabulary_gap.TOP_GAPS` applies twice over.
MAX_CLAIMS = 5

# And at most this many from any one stretch. Two claims from one conversation
# are usually the same claim said twice, and spending the whole list on one
# chapter hides that the rest of the video has its own.
MAX_PER_CHAPTER = 1


def _tokens(text: str) -> list:
    from modules.chapter_speech import tokenize
    return tokenize(text)


def _weights(chapters: Sequence[Mapping]) -> list:
    """Per chapter, how many times more often it says each word than the rest.

    Recomputed from the dialogue the chapters carry rather than taken from
    ``speech_words``, which keeps only the top three per chapter -- enough to
    title a stretch, nowhere near enough to score a sentence. The dialogue is
    every line of the stretch, so this is the same measurement
    :func:`modules.chapter_speech.summarise_speech` made from the transcript,
    and it means one implementation serves both a live report and a record
    read back from disk hours later.
    """
    from modules.chapter_speech import keyness

    return keyness([[word for line in (ch.get("dialogue") or ch.get("quotes") or [])
                     for word in _tokens(line.get("text"))]
                    for ch in (chapters or [])])


def unusual_words(text: str, weights: Mapping) -> list:
    """The words in a line that this stretch says more often than the video does.

    Above the video's own rate and nothing stronger. The threshold that decides
    whether a *word* is worth reporting belongs to whatever reports words; here
    the word is not the finding, the sentence is, and a sentence built of six
    mildly unusual words is as good a candidate as one built of two rare ones.
    """
    return [word for word in _tokens(text)
            if float((weights or {}).get(word, 0.0)) > 1.0]


def carrying_words(text: str, weights: Mapping, limit: int = 6) -> list:
    """The words that ranked the line, most unusual first.

    :func:`unusual_words` is in sentence order and keeps repeats, because the
    score counts a word said twice twice. This is for a reader asking *why was
    this line picked*, and the honest answer to that is the handful of words
    furthest from the video's own speech -- a list led by "and", "so", "your",
    each a hair above average, answers nothing.
    """
    ranked = sorted(set(unusual_words(text, weights)),
                    key=lambda w: (-float(weights.get(w, 0.0)), w))
    return ranked[:limit]


def score_line(text: str, weights: Mapping) -> float:
    """How much *different* vocabulary unusual for this video the line carries.

    Summed rather than averaged. :func:`modules.chapter_speech.quotes_for`
    divides by length, which is right for picking a line to *show* -- it prefers
    the pithiest phrasing of a subject. It is wrong for picking a line that
    *asserts* something, where the long sentence is the one making the case, and
    dividing hands the ranking to whichever two-word interjection happens to be
    unusual for its stretch.

    Each word counts once however often the line repeats it, which is the
    difference between an assertion and a chant. A line that says one rare word
    eight times is emphatic about nothing; a line that reaches for eight
    different unusual words is making a case. It also disarms the failure mode
    that dominates raw transcript ranking -- Whisper stuttering a phrase inside
    one segment on hard audio.
    """
    return sum(math.log(float(weights.get(word, 0.0)))
               for word in set(unusual_words(text, weights)))


def is_covered(text: str, covered: Iterable[str]) -> bool:
    """Whether any word of the line is a word of some class or event name."""
    covered = set(covered or ())
    return any(word in covered for word in _tokens(text))


def find(chapters: Sequence[Mapping],
         classes: Iterable[str] = (),
         events: Iterable[str] = (),
         *,
         min_words: int = MIN_WORDS,
         limit: int = MAX_CLAIMS,
         per_chapter: int = MAX_PER_CHAPTER) -> list:
    """Quoted lines nothing in this run measured, most substantial first.

    Every row carries the sentence, where it was said, and which of its words
    were unusual enough to rank it -- so a reader who thinks the line is not a
    claim at all can see exactly why it was picked, and disagree with the
    ranking rather than with the report.
    """
    from modules.vocabulary_gap import covered_words

    covered = covered_words(classes, events)
    weights = _weights(chapters)
    rows = []
    for chapter, here in zip(chapters or [], weights):
        found = []
        for line in (chapter.get("quotes") or []):
            text = str(line.get("text") or "")
            words = _tokens(text)
            if len(words) < min_words or is_covered(text, covered):
                continue
            row = {
                "quote": text,
                "said_at": round(float(line.get("start") or 0.0), 2),
                "timestamp": str(line.get("timestamp") or ""),
                "words": len(words),
                "unusual": carrying_words(text, here),
                "score": round(score_line(text, here), 1),
                "chapter": {"number": int(chapter.get("number") or 0),
                            "timestamp": str(chapter.get("timestamp") or "")},
            }
            speaker = str(line.get("speaker") or "").strip()
            if speaker and speaker.upper() != "UNKNOWN":
                row["speaker"] = speaker
            # Whether the detector produced anything at all in this stretch.
            # "Nothing was watching for this" and "plenty was watched here, just
            # not this" are different findings, and only the record can tell
            # them apart -- a chapter with no share map had no detections.
            shares = chapter.get("class_shares")
            if shares is not None:
                row["measured_here"] = sorted(shares)[:4]
            found.append(row)
        found.sort(key=lambda r: (-r["score"], r["said_at"]))
        rows.extend(found[:max(1, per_chapter)])
    rows.sort(key=lambda r: (-r["score"], r["said_at"]))
    return rows[:limit]


def summarise(report: Mapping,
              chapters: Optional[Sequence[Mapping]] = None) -> dict:
    """The report's section: what was said and never measured, and what would.

    Takes the finished record, so it serves both paths for free --
    :func:`modules.highlight_report.build_report` passes the chapters it has
    just built, and a re-narration hours later passes nothing and gets the same
    answer off the saved file. :mod:`modules.spoken_evidence` needs two separate
    implementations for that because its figures come from caches the record
    does not keep; every figure here is computed from the chapters themselves.

    The routes are attached once for the report rather than once per claim.
    They do not depend on what was said -- what a run is *able* to measure is a
    property of the run -- and repeating them under every row would read as five
    different recommendations.
    """
    from modules.detection_routes import pick

    chapters = chapters if chapters is not None else (report.get("chapters") or [])
    vocabulary = report.get("vocabulary") or {}
    claims = find(chapters,
                  vocabulary.get("classes") or [],
                  vocabulary.get("events") or [])
    if not claims:
        return {}
    return {"claims": claims, "routes": pick(report)}


def ensure(report: dict) -> dict:
    """Fill in the section on a record that does not carry it yet. Mutates.

    Reports are written once and read many times -- re-narrated, re-rendered,
    re-diagnosed, often hours later and always from the saved file. A section
    that only ever appeared on a video analysed from scratch would be missing
    from exactly the reports somebody is dissatisfied enough with to go back to,
    which is the failure :func:`modules.spoken_evidence.from_report` exists to
    avoid. Everything needed is in the record, so this costs a few milliseconds
    and can simply be called on the way in.

    An existing section is left alone. It was computed with the caches present
    and this one cannot improve on it.
    """
    if report.get("unmeasured"):
        return report["unmeasured"]
    try:
        report["unmeasured"] = summarise(report)
    except Exception as exc:                       # pragma: no cover - defensive
        print(f"⚠️ Unmeasured claims skipped: {exc}")
        report["unmeasured"] = {}
    return report["unmeasured"]
