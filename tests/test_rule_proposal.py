"""Tests for the gap, the proposed rule, and the loop that closes between them.

This is the one path in the app where a model's output becomes configuration,
so the tests are about refusal far more than about success.

The rejection that matters most is `test_rejects_a_class_this_video_never_had`.
A rule naming a class the detector never emitted parses, loads, and fires on
nothing — the only symptom is a re-run that costs minutes and changes nothing,
and by then the user has no reason to suspect the rule rather than the footage.
Everything else here is cheap to discover; that one is not.

Second in importance is that `apply` is the only function that writes, that it
backs up first, and that a corrupt write cannot leave a rules file that no
longer loads — because the failure mode of that is every composed event
silently vanishing from the next run.

Fixture class names are workshop objects. The engine and this module hold no
vocabulary of their own, which is exactly why they can be tested with any.
"""
from __future__ import annotations

import json

import pytest
import yaml

from modules import rule_proposal
from modules.rule_proposal import Proposal, apply, parse, propose
from modules.vocabulary_gap import (
    MIN_COUNT_FOR_GAP,
    MIN_KEYNESS_FOR_GAP,
    covered_words,
    find_gaps,
    observed_classes,
    summarise,
    tokens_of,
)

CLASSES = ["clamp", "bench", "board"]


class _FakeLLM:
    def __init__(self, reply):
        self.reply, self.prompt, self.system = reply, None, None

    def generate(self, prompt, system="", max_tokens=1024, **kw):
        self.prompt, self.system = prompt, system
        return self.reply


def _chapter(number=1, words=(), quotes=()):
    return {"number": number, "timestamp": "0:05:00", "start": 300.0,
            "end": 600.0, "title": f"Chapter {number}",
            "speech_words": [dict(w) for w in words],
            "quotes": [dict(q) for q in quotes]}


def _word(word, count=5, times=12.0):
    return {"word": word, "count": count, "times": times}


# ---------------------------------------------------------------------------
# The gap
# ---------------------------------------------------------------------------
class TestCoverage:
    def test_a_compound_class_name_covers_both_its_words(self):
        assert tokens_of("clamp_on_bench") == {"clamp", "on", "bench"}

    def test_naming_conventions_resolve_alike(self):
        assert tokens_of("clamp-on bench") == tokens_of("clamp_on_bench")

    def test_events_cover_words_too(self):
        # A composed event is a name the user chose; once it exists, the word
        # is no longer something nothing is watching for.
        assert "doubled" in covered_words(["clamp"], ["doubled_clamp"])


class TestFindGaps:
    def test_a_word_with_no_class_is_a_gap(self):
        gaps = find_gaps([_chapter(words=[_word("lathe")])], CLASSES)
        assert [g["word"] for g in gaps] == ["lathe"]

    def test_a_word_a_class_covers_is_not(self):
        assert find_gaps([_chapter(words=[_word("bench")])], CLASSES) == []

    def test_below_the_keyness_bar_is_not_a_gap(self):
        # The bar is higher than the one for describing a chapter: this asks a
        # person to consider a re-run, not to read a word.
        low = _word("lathe", times=MIN_KEYNESS_FOR_GAP - 1)
        assert find_gaps([_chapter(words=[low])], CLASSES) == []

    def test_said_too_few_times_is_not_a_gap(self):
        rare = _word("lathe", count=MIN_COUNT_FOR_GAP - 1)
        assert find_gaps([_chapter(words=[rare])], CLASSES) == []

    def test_a_gap_carries_a_line_it_was_said_in(self):
        chapter = _chapter(words=[_word("lathe")],
                           quotes=[{"start": 310.0, "text": "the lathe again"}])
        assert find_gaps([chapter], CLASSES)[0]["chapters"][0]["quote"] \
            == "the lathe again"

    def test_the_same_word_in_two_chapters_is_one_row(self):
        gaps = find_gaps([_chapter(1, [_word("lathe")]),
                          _chapter(2, [_word("lathe")])], CLASSES)
        assert len(gaps) == 1 and len(gaps[0]["chapters"]) == 2

    def test_composed_events_are_not_reported_as_detector_classes(self):
        seen = observed_classes({1: ["clamp", "doubled_clamp"]},
                                composed_event_names=["doubled_clamp"])
        assert seen == ["clamp"]

    def test_summarise_reports_nothing_when_everything_is_covered(self):
        out = summarise([_chapter(words=[_word("bench")])], CLASSES)
        assert out["gaps"] == [] and "gap_count" not in out


# ---------------------------------------------------------------------------
# Parsing a model's reply
# ---------------------------------------------------------------------------
class TestParse:
    def _reply(self, **over):
        body = {"name": "clamped_board", "label": "Clamped Board",
                "why": "two clamps on one board", "rules": [
                    {"source": "clamp", "region": "board",
                     "min_count": 2, "max_count": 999}]}
        body.update(over)
        return json.dumps(body)

    def test_a_clean_reply_becomes_a_proposal(self):
        p = parse(self._reply(), CLASSES)
        assert p.name == "clamped_board" and p.rules[0]["min_count"] == 2

    def test_rejects_a_class_this_video_never_had(self):
        # The expensive failure. A rule on an absent class loads fine and fires
        # on nothing, and the only symptom is a wasted re-run.
        bad = self._reply(rules=[{"source": "chisel", "region": "board",
                                  "min_count": 1, "max_count": 9}])
        assert parse(bad, CLASSES) is None

    def test_rejects_a_name_already_in_use(self):
        existing = [{"name": "clamped_board", "rules": []}]
        assert parse(self._reply(), CLASSES, existing) is None

    def test_rejects_an_unusable_name(self):
        assert parse(self._reply(name="Clamped Board!"), CLASSES) is None

    def test_rejects_counts_that_cannot_be_satisfied(self):
        bad = self._reply(rules=[{"source": "clamp", "region": "board",
                                  "min_count": 5, "max_count": 2}])
        assert parse(bad, CLASSES) is None

    def test_rejects_a_rule_with_no_conditions(self):
        assert parse(self._reply(rules=[]), CLASSES) is None

    def test_a_models_refusal_is_not_an_error(self):
        # "This cannot be expressed with these classes" is often the correct
        # answer, and must not look like a parse failure to the caller.
        assert parse('{"name": null, "why": "no class covers it"}',
                     CLASSES) is None

    def test_json_wrapped_in_chatter_is_still_read(self):
        wrapped = f"Sure! ```json\n{self._reply()}\n``` Hope that helps."
        assert parse(wrapped, CLASSES).name == "clamped_board"

    def test_a_reply_with_no_json_is_rejected(self):
        assert parse("I would add a rule for clamps.", CLASSES) is None


class TestPropose:
    def test_the_prompt_names_only_the_available_classes(self):
        llm = _FakeLLM('{"name": null, "why": "no"}')
        propose("two clamps at once", CLASSES, llm=llm)
        assert "clamp, bench, board" in llm.prompt or all(
            c in llm.prompt for c in CLASSES)
        assert "Reply with one JSON object" in llm.system

    def test_existing_rules_are_shown_so_it_does_not_duplicate_them(self):
        llm = _FakeLLM('{"name": null, "why": "no"}')
        propose("x", CLASSES, llm=llm,
                existing=[{"name": "clamped_board",
                           "rules": [{"source": "clamp", "region": "board",
                                      "min_count": 2, "max_count": 999}]}])
        assert "clamped_board" in llm.prompt

    def test_the_claim_is_carried_onto_the_proposal(self):
        llm = _FakeLLM(json.dumps({
            "name": "clamped_board", "label": "L", "why": "w",
            "rules": [{"source": "clamp", "region": "board",
                       "min_count": 2, "max_count": 999}]}))
        p = propose("I used two clamps", CLASSES, llm=llm, claim_at=42.0)
        assert p.claim == "I used two clamps" and p.claim_at == 42.0

    def test_no_model_means_no_proposal(self):
        assert propose("x", CLASSES, llm=None) is None


# ---------------------------------------------------------------------------
# Writing — the only function here that touches the user's file
# ---------------------------------------------------------------------------
def _proposal():
    return Proposal(name="clamped_board", label="Clamped Board",
                    why="two clamps on one board",
                    rules=[{"source": "clamp", "region": "board",
                            "min_count": 2, "max_count": 999}],
                    claim="I used two clamps", claim_at=42.0)


class TestApply:
    def _rules_file(self, tmp_path):
        path = tmp_path / "composition_rules.yaml"
        path.write_text(yaml.safe_dump({"events": [
            {"name": "on_bench", "label": "On Bench",
             "rules": [{"source": "clamp", "region": "bench",
                        "min_count": 1, "max_count": 1}],
             "window_secs": 0.75, "persist_secs": 0.5}]}), encoding="utf-8")
        return path

    def test_the_rule_is_appended_and_the_existing_ones_survive(self, tmp_path):
        path = self._rules_file(tmp_path)
        apply(str(path), _proposal())
        events = yaml.safe_load(path.read_text(encoding="utf-8"))["events"]
        assert [e["name"] for e in events] == ["on_bench", "clamped_board"]

    def test_the_file_is_backed_up_first(self, tmp_path):
        path = self._rules_file(tmp_path)
        apply(str(path), _proposal())
        assert (tmp_path / "composition_rules.yaml.bak").exists()

    def test_the_result_still_loads_in_the_real_engine(self, tmp_path):
        # The check that matters: a rules file this wrote must be one the
        # engine accepts, or the next run loses every composed event.
        from video_ai_editor.composition_engine import CompositionEngine

        path = self._rules_file(tmp_path)
        apply(str(path), _proposal())
        assert CompositionEngine(str(path)).event_names == [
            "on_bench", "clamped_board"]

    def test_a_duplicate_name_raises_rather_than_writing(self, tmp_path):
        path = self._rules_file(tmp_path)
        before = path.read_text(encoding="utf-8")
        with pytest.raises(ValueError):
            apply(str(path), Proposal(name="on_bench", label="x", why="",
                                      rules=[{"source": "clamp",
                                              "region": "bench",
                                              "min_count": 1,
                                              "max_count": 1}]))
        assert path.read_text(encoding="utf-8") == before

    def test_a_missing_file_is_created_rather_than_failing(self, tmp_path):
        path = tmp_path / "new_rules.yaml"
        apply(str(path), _proposal())
        assert yaml.safe_load(path.read_text(encoding="utf-8"))["events"]


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------
class TestChecks:
    def test_applying_records_what_the_rule_was_meant_to_test(self, tmp_path):
        rules = tmp_path / "rules.yaml"
        video = tmp_path / "a.mp4"
        video.write_bytes(b"")
        apply(str(rules), _proposal(), video_path=str(video))
        checks = rule_proposal.load_checks(str(video))
        assert checks[0]["rule"] == "clamped_board"
        assert checks[0]["claim"] == "I used two clamps"

    def test_the_note_survives_beside_the_video_not_in_the_report(self, tmp_path):
        video = tmp_path / "a.mp4"
        rule_proposal.record_check(str(video), _proposal())
        assert (tmp_path / "a_checks.json").exists()

    def test_no_notes_is_an_empty_list_not_an_error(self, tmp_path):
        assert rule_proposal.load_checks(str(tmp_path / "nothing.mp4")) == []

    def test_a_fired_rule_is_settled_as_fired(self):
        settled = rule_proposal.settle_checks(
            [{"rule": "clamped_board"}],
            {10: ["clamp", "clamped_board"], 12: ["clamped_board"]},
            ["clamped_board"])
        assert settled[0]["state"] == "fired"
        assert settled[0]["seconds"] == 2 and settled[0]["first"] == 10

    def test_a_loaded_rule_that_matched_nothing_is_never_fired(self):
        settled = rule_proposal.settle_checks(
            [{"rule": "clamped_board"}], {10: ["clamp"]}, ["clamped_board"])
        assert settled[0]["state"] == "never_fired"

    def test_a_rule_that_is_not_loaded_is_not_evidence(self):
        # The distinction the whole loop turns on: "looked and found nothing"
        # and "never looked" must never be shown as the same answer.
        settled = rule_proposal.settle_checks(
            [{"rule": "clamped_board"}], {10: ["clamp"]}, [])
        assert settled[0]["state"] == "not_in_rules"


class TestFindings:
    def _report(self, **over):
        report = {"vocabulary": {"classes": CLASSES, "events": [], "gaps": []},
                  "checks": [], "settings": {}, "segments": [],
                  "signal_totals": {}, "totals": {}}
        report.update(over)
        return report

    def test_a_gap_produces_a_finding_that_names_the_words(self):
        from modules.highlight_advice import diagnose

        gaps = [{"word": "lathe", "count": 5, "times": 12.0,
                 "chapters": [{"number": 2, "timestamp": "0:05:00",
                               "quote": "the lathe again"}]}]
        found = diagnose(self._report(
            vocabulary={"classes": CLASSES, "events": [], "gaps": gaps}))
        gap = next(f for f in found if f.id == "unchecked_vocabulary")
        assert "lathe" in gap.detail and gap.topic == "composition"

    def test_each_check_state_produces_its_own_finding(self):
        from modules.highlight_advice import diagnose

        for state, ident in (("fired", "check_fired:r"),
                             ("never_fired", "check_silent:r"),
                             ("not_in_rules", "check_missing:r")):
            found = diagnose(self._report(checks=[
                {"rule": "r", "state": state, "claim": "c",
                 "seconds": 3, "first": 10, "last": 20}]))
            assert any(f.id == ident for f in found), state

    def test_an_unloaded_rule_is_the_most_serious_of_the_three(self):
        from modules.highlight_advice import diagnose

        found = diagnose(self._report(checks=[
            {"rule": "r", "state": "not_in_rules", "claim": "c"}]))
        assert next(f for f in found if f.id == "check_missing:r").severity \
            == "high"

    def test_the_topic_every_one_of_these_points_at_exists(self):
        from modules.advisor import knowledge_topics

        assert "composition" in knowledge_topics()
