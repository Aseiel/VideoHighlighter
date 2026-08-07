"""What the video talks about that nothing in the run could see.

The report can now quote what was said and list what was detected, and the
interesting thing about having both is the *difference* between them. A stretch
where somebody names something repeatedly, and no class the detector owns covers
it, is a hole in what this run was able to check — and it is a hole the record
can point at precisely, because both halves are already measured.

This matters more than it sounds. A narrator handed a transcript will describe
what people say happened; the measurements can only confirm the parts something
was watching for. Without this module the two are silently different sizes, and
a reader has no way to tell a claim the signals corroborate from one they merely
failed to contradict. Naming the gap is what turns "the model said so" into "the
model said so, and here is what would have to exist for the run to check it".

How the gap is found
--------------------

Two sets, both already computed. The spoken side is
:func:`modules.chapter_speech.distinctive_words` — words a stretch says far more
often than the rest of the video. The seen side is every class the detector
actually produced in this file, plus every composed event the rules can emit.
A word is *covered* when it appears among the tokens of some class or event
name, and *unseen* when it does not.

Matching is token equality and nothing cleverer. No stemming, no synonyms, no
lexicon: a synonym table is a vocabulary, this repo ships none, and one written
for English would be wrong the first time somebody transcribes anything else.
The consequence is that the gap over-reports -- a word can be covered in meaning
by a class it does not share a token with -- and that is the right direction to
be wrong in. Everything here is a *candidate for a person to look at*, and the
cost of an extra candidate is a glance, while the cost of a missed one is a
claim nobody realises went unchecked.

What this module will not do
----------------------------

It does not decide what any word means, propose what to detect, or rank one gap
as more worth closing than another beyond how distinctive the word was. Those
are judgements, and they belong to the user or -- with the user's approval
before anything is written -- to :mod:`modules.rule_proposal`. This file only
says: *this was talked about, and nothing here was watching for it.*
"""
from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence

# A word has to be at least this many times more frequent in its chapter than in
# the rest of the video before its absence from the class list is worth
# reporting. Higher than `chapter_speech.MIN_KEYNESS`, which decides what is
# distinctive enough to *describe* a chapter: this decides what is distinctive
# enough to be worth a person's time and possibly a re-run, which is a larger
# ask and deserves a higher bar.
MIN_KEYNESS_FOR_GAP = 6.0

# And it has to be said at least this often. A gap is a proposal to go and
# measure something; twice is not enough evidence that there is anything there.
MIN_COUNT_FOR_GAP = 3

# How many gaps are reported. This is a list a person reads and acts on one item
# from, so a long one is a worse answer than a short one.
TOP_GAPS = 8


def tokens_of(name: str) -> set:
    """The words inside a class or event name.

    ``double_vaginal`` covers a spoken "double" and a spoken "vaginal"; that is
    the whole of the matching rule. Splitting on non-letters means a name in any
    of the conventions this repo has met -- snake_case, kebab, spaced -- resolves
    the same way without anyone declaring which is in use.
    """
    from modules.chapter_speech import tokenize
    return set(tokenize(str(name or "").replace("_", " ").replace("-", " ")))


def covered_words(classes: Iterable[str], events: Iterable[str] = ()) -> set:
    """Every spoken word that some class or event name already accounts for."""
    out: set = set()
    for name in list(classes or []) + list(events or []):
        out |= tokens_of(name)
    return out


def observed_classes(object_detections: Optional[Mapping] = None,
                     bbox_cache: Optional[Iterable[Mapping]] = None,
                     composed_event_names: Iterable[str] = ()) -> list:
    """Class names the detector actually produced in this video.

    Observed rather than configured, deliberately. The configured list is what
    the model *could* emit; a rule built on a class this file never contains
    cannot fire however well it is written, and proposing one is how a user
    spends a re-run to learn nothing. Composed events are subtracted because
    they are conclusions the rules drew, not things a detector saw.
    """
    composed = {str(n) for n in (composed_event_names or ())}
    seen: set = set()
    for names in (object_detections or {}).values():
        seen |= {str(n) for n in (names or [])}
    for frame in (bbox_cache or []):
        seen |= {str(n) for n in (frame.get("objects") or [])}
    return sorted(seen - composed)


def find_gaps(chapters: Sequence[Mapping],
              classes: Iterable[str],
              events: Iterable[str] = (),
              *,
              min_keyness: float = MIN_KEYNESS_FOR_GAP,
              min_count: int = MIN_COUNT_FOR_GAP,
              limit: int = TOP_GAPS) -> list[dict]:
    """Distinctive spoken words no class or event name covers.

    One row per word, carrying where it was said and how far above the video's
    own rate it was said there -- so a reader can go and listen before deciding
    the gap is real, and so a proposal built on it can quote its evidence.
    """
    covered = covered_words(classes, events)
    rows: dict = {}
    for chapter in (chapters or []):
        for found in (chapter.get("speech_words") or []):
            word = str(found.get("word") or "")
            if not word or word in covered:
                continue
            if float(found.get("times") or 0.0) < min_keyness:
                continue
            if int(found.get("count") or 0) < min_count:
                continue
            row = rows.setdefault(word, {"word": word, "count": 0,
                                         "times": 0.0, "chapters": []})
            row["count"] += int(found.get("count") or 0)
            row["times"] = max(row["times"], float(found.get("times") or 0.0))
            row["chapters"].append({
                "number": int(chapter.get("number") or 0),
                "timestamp": str(chapter.get("timestamp") or ""),
                # One line from the stretch, so the word arrives with the
                # sentence it was said in rather than as a bare token. A gap a
                # reader cannot picture is one they cannot judge.
                "quote": _quote_containing(chapter, word),
            })
    ranked = sorted(rows.values(), key=lambda r: (-r["times"], -r["count"],
                                                  r["word"]))
    for row in ranked:
        row["times"] = round(row["times"], 1)
    return ranked[:limit]


def _quote_containing(chapter: Mapping, word: str) -> str:
    """A line from this chapter that actually contains the word, if one is kept."""
    from modules.chapter_speech import tokenize

    for line in ((chapter.get("quotes") or [])
                 + (chapter.get("dialogue") or [])):
        if word in tokenize(line.get("text")):
            return str(line.get("text") or "")
    return ""


def summarise(chapters: Sequence[Mapping],
              classes: Iterable[str],
              events: Iterable[str] = ()) -> dict:
    """The record's vocabulary section: what was seen, and what was only said."""
    classes = sorted(str(c) for c in (classes or []))
    events = sorted(str(e) for e in (events or []))
    gaps = find_gaps(chapters, classes, events)
    out = {"classes": classes, "events": events, "gaps": gaps}
    if gaps:
        # Carried so a renderer and a prompt can both state the ratio without
        # recomputing it, and disagree about it if they did.
        out["gap_count"] = len(gaps)
    return out
