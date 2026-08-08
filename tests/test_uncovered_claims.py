"""Tests for the lines a report quotes and has no measurement for.

The section exists because of a specific silence: a chapter block where
somebody says something substantial about a thing nothing was watching for
carries the quote and no figures, which on the page is indistinguishable from a
claim that was checked and held. So the tests are mostly about the ways it
could stop noticing that:

* a claim said exactly *once* has to surface, because that is the shape of the
  sentence this exists for — a frequency threshold cannot reach it at any
  setting, which is why :mod:`modules.vocabulary_gap` is not enough;
* a line some class already covers must *not* appear, or the section
  contradicts the measurements printed above it;
* an interjection must not outrank an assertion, which is the one thing
  inherited scoring would get backwards;
* the section has to appear on a record analysed hours earlier, because that is
  when people read these reports.

Fixture classes are named after workshop objects. Nothing in the module knows
what a class is called, and the repo carries no vocabulary of its own.
"""
from __future__ import annotations

import pytest

from modules import uncovered_claims


def _lines(*items):
    return [{"start": float(at), "timestamp": f"0:{int(at) // 60:02d}:{int(at) % 60:02d}",
             "text": text} for at, text in items]


def _chapter(number, start, end, dialogue, quotes=None, shares=None):
    row = {"number": number, "start": float(start), "end": float(end),
           "timestamp": f"0:{int(start) // 60:02d}:{int(start) % 60:02d}",
           "dialogue": list(dialogue),
           "quotes": list(quotes if quotes is not None else dialogue)}
    if shares is not None:
        row["class_shares"] = dict(shares)
    return row


# A stretch of ordinary talk, so the second chapter has something to be
# distinctive against. Keyness needs two documents or every word is as typical
# of the video as it is of the chapter. Every line is kept under the length bar
# so this chapter contributes no rows of its own and each test is about the
# stretch it is actually testing.
FILLER = _lines(
    (0, "right so we are back again"),
    (12, "yes the same as yesterday morning"),
    (24, "we will get on with it"),
    (36, "not much else to say"),
)


class TestFind:
    def test_a_claim_said_once_is_reported(self):
        # The whole reason this module exists. Somebody states a preference in
        # one sentence, never repeats it, and no frequency measure can see it.
        claim = _lines((60, "honestly the thing I care about most is a really "
                            "clean dovetail joint every single time"))
        chapters = [_chapter(1, 0, 60, FILLER),
                    _chapter(2, 60, 120, claim)]
        found = uncovered_claims.find(chapters, ["bench_vice"])
        assert [r["chapter"]["number"] for r in found] == [2]
        assert "dovetail" in found[0]["quote"]
        assert found[0]["words"] >= uncovered_claims.MIN_WORDS

    def test_a_line_a_class_already_covers_is_left_alone(self):
        # `spoken_evidence` has that line and has figures for it. Reporting it
        # here as unmeasured would contradict the block above it on the page.
        covered = _lines((60, "the bench vice is the thing I care about most "
                              "in this entire workshop honestly"))
        chapters = [_chapter(1, 0, 60, FILLER),
                    _chapter(2, 60, 120, covered)]
        assert uncovered_claims.find(chapters, ["bench_vice"]) == []
        # ...and it is only the class list that spares it.
        assert uncovered_claims.find(chapters, ["lathe"]) != []

    def test_a_short_interjection_is_not_a_claim(self):
        # "All right, all right, all right." is a real chapter's top quote on
        # real footage: genuinely distinctive, and not a claim about anything.
        chatter = _lines((60, "right, right, right."),
                         (80, "mm, quite."))
        chapters = [_chapter(1, 0, 60, FILLER),
                    _chapter(2, 60, 120, chatter)]
        assert uncovered_claims.find(chapters, ["bench_vice"]) == []

    def test_an_assertion_outranks_an_unusual_interjection(self):
        # The scoring inherited from `quotes_for` divides by length, which picks
        # the pithiest phrasing of a subject -- exactly wrong for finding the
        # sentence that makes a case.
        mixed = _lines(
            (60, "crikey crikey crikey crikey crikey crikey crikey crikey"),
            (90, "what I actually wanted from this whole session was a "
                 "properly flat reference surface to work against"))
        chapters = [_chapter(1, 0, 60, FILLER),
                    _chapter(2, 60, 120, mixed)]
        found = uncovered_claims.find(chapters, ["bench_vice"], per_chapter=2)
        assert "reference surface" in found[0]["quote"]

    def test_one_line_per_stretch_by_default(self):
        # Two claims from one conversation are usually the same claim twice.
        pair = _lines(
            (60, "the flattest reference surface is what I wanted from today"),
            (90, "a properly flat reference surface was the whole point of it"))
        chapters = [_chapter(1, 0, 60, FILLER),
                    _chapter(2, 60, 120, pair)]
        assert len(uncovered_claims.find(chapters, ["bench_vice"])) == 1
        assert len(uncovered_claims.find(chapters, ["bench_vice"],
                                         per_chapter=2)) == 2

    def test_the_report_wide_cap_holds(self):
        chapters = [_chapter(1, 0, 60, FILLER)]
        for number in range(2, 10):
            start = number * 60
            chapters.append(_chapter(
                number, start, start + 60,
                _lines((start, f"the {'ratchet spanner grinder chisel'.split()[number % 4]} "
                               f"is what mattered most to me on day {number} "
                               f"of this particular job"))))
        found = uncovered_claims.find(chapters, ["bench_vice"])
        assert len(found) == uncovered_claims.MAX_CLAIMS

    def test_the_row_says_whether_the_detector_was_busy_there(self):
        # "Nothing was watching" and "four things were, and none of them is
        # this" are different findings, and only the record tells them apart.
        claim = _lines((60, "what I wanted out of today was one properly "
                            "square corner and nothing else at all"))
        busy = [_chapter(1, 0, 60, FILLER),
                _chapter(2, 60, 120, claim, shares={"bench_vice": 12.0})]
        assert uncovered_claims.find(busy, ["bench_vice"])[0]["measured_here"] \
            == ["bench_vice"]

        blind = [_chapter(1, 0, 60, FILLER), _chapter(2, 60, 120, claim)]
        assert "measured_here" not in uncovered_claims.find(blind,
                                                            ["bench_vice"])[0]

    def test_the_words_that_ranked_the_line_come_with_it(self):
        # A reader who thinks the line is not a claim should be able to see why
        # it was picked and argue with the ranking, not with the report.
        claim = _lines((60, "the dovetail is the thing I care about most in "
                            "this workshop, honestly, every time"))
        chapters = [_chapter(1, 0, 60, FILLER),
                    _chapter(2, 60, 120, claim)]
        row = uncovered_claims.find(chapters, ["bench_vice"])[0]
        assert "dovetail" in row["unusual"]
        assert len(row["unusual"]) <= 6

    def test_the_carrying_words_are_the_most_unusual_ones(self):
        # Strongest first and de-duplicated, so the list answers "why this
        # line" instead of being led by whatever the sentence happens to open
        # with. Weights supplied directly: what is under test is the ordering,
        # not the measurement that produced it.
        weights = {"the": 1.4, "dovetail": 40.0, "joint": 12.0, "was": 0.4}
        assert uncovered_claims.carrying_words(
            "the dovetail joint was the dovetail joint", weights) == \
            ["dovetail", "joint", "the"]

    def test_a_speaker_is_carried_when_diarization_named_one(self):
        claim = _lines((60, "what I wanted out of today was one properly "
                            "square corner and nothing else"))
        claim[0]["speaker"] = "SPEAKER_01"
        chapters = [_chapter(1, 0, 60, FILLER),
                    _chapter(2, 60, 120, claim)]
        assert uncovered_claims.find(chapters, ["bench_vice"])[0]["speaker"] \
            == "SPEAKER_01"


class TestAgainstTheGap:
    def test_the_frequency_measure_cannot_see_what_this_finds(self):
        """Not a duplicate of `vocabulary_gap` -- the complement of it.

        Guards the reason this module was written. If a future change made the
        gap able to reach a one-off claim, this file would be doing work
        somebody else already does; while it cannot, the two are both needed.
        """
        from modules.vocabulary_gap import find_gaps

        claim = _lines((60, "the dovetail is the thing I care about most in "
                            "this entire workshop, honestly"))
        chapters = [_chapter(1, 0, 60, FILLER),
                    _chapter(2, 60, 120, claim)]
        # Said once, so it never becomes a distinctive word and the gap list is
        # empty -- with or without a `speech_words` entry to work from.
        assert find_gaps(chapters, ["bench_vice"]) == []
        assert uncovered_claims.find(chapters, ["bench_vice"]) != []


class TestSummarise:
    def _report(self):
        claim = _lines((60, "what I care about most is one properly square "
                            "corner and nothing else at all"))
        return {
            "chapters": [_chapter(1, 0, 60, FILLER),
                         _chapter(2, 60, 120, claim,
                                  shares={"bench_vice": 8.0})],
            "vocabulary": {"classes": ["bench_vice"], "events": []},
            "speech": {"words": 120},
            "settings": {"detector_activity": {"face": 400}},
        }

    def test_the_routes_are_attached_once_for_the_report(self):
        # They describe what the run can measure, not what was said, so a copy
        # under every claim would read as several different recommendations.
        out = uncovered_claims.summarise(self._report())
        assert out["claims"]
        assert out["routes"]["fastest"]["id"]
        assert "routes" not in out["claims"][0]

    def test_nothing_to_report_produces_no_section(self):
        report = self._report()
        report["vocabulary"]["classes"] = ["corner"]
        assert uncovered_claims.summarise(report) == {}

    def test_ensure_backfills_an_older_record_and_leaves_a_newer_one(self):
        report = self._report()
        uncovered_claims.ensure(report)
        assert report["unmeasured"]["claims"]

        already = {"unmeasured": {"claims": ["kept"]}}
        uncovered_claims.ensure(already)
        assert already["unmeasured"]["claims"] == ["kept"]


class TestTheFinding:
    def _report(self):
        report = {
            "chapters": [
                _chapter(1, 0, 60, FILLER),
                _chapter(2, 60, 120,
                         _lines((60, "what I care about most is one properly "
                                     "square corner and nothing else at all")),
                         shares={"bench_vice": 8.0})],
            "vocabulary": {"classes": ["bench_vice", "lathe"], "events": []},
            "speech": {"words": 120},
            "settings": {"detector_activity": {"face": 400}},
            "segments": [],
        }
        return report

    def test_it_fires_and_names_both_routes(self):
        from modules.highlight_advice import attach_advice

        report = self._report()
        attach_advice(report)
        found = [f for f in report["advice"] if f["id"] == "unmeasured_claim"]
        assert len(found) == 1
        remedy = found[0]["remedy"]
        assert "Fastest" in remedy and "Most reliable" in remedy
        # The quote itself, so the advice is checkable against the video.
        assert "square corner" in found[0]["detail"]

    def test_it_points_at_a_page_the_advisor_actually_has(self):
        # A finding whose topic names no file gives the model no material and
        # fails silently -- the answer just gets vaguer.
        from modules.advisor import knowledge_topics
        from modules.highlight_advice import diagnose

        report = self._report()
        uncovered_claims.ensure(report)
        topics = knowledge_topics()
        for finding in diagnose(report):
            assert finding.topic in topics, finding.id

    def test_no_claims_means_no_finding(self):
        from modules.highlight_advice import diagnose

        report = self._report()
        report["unmeasured"] = {}
        assert not [f for f in diagnose(report) if f.id == "unmeasured_claim"]
