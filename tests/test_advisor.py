"""Tests for `modules.advisor` — findings joined to the pages that explain them.

The behaviour worth protecting is the boundary: everything works with no model,
the model is only ever given material that was computed rather than recalled,
and a model that misbehaves cannot take the run down with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from modules import advisor
from modules.highlight_advice import diagnose
from modules.highlight_report import build_report


def _report(n=600):
    keys = ("scene", "motion_event", "motion_peak", "audio",
            "keyword", "object", "action")
    sig = {k: np.zeros(n) for k in keys}
    sig["object"][100] = 10.0
    return build_report(
        video_path="a.mp4", video_duration=n, score=sum(sig.values()),
        signals=sig, segments=[(95, 105)],
        settings={"clip_time": 10, "duration_mode": "MAX", "object_points": 5})


class _FakeLLM:
    """Stands in for llm.llm_module.LLMModule — same generate() shape."""

    def __init__(self, reply="Raise a second signal's weight.", boom=False):
        self.reply, self.boom = reply, boom
        self.prompt = self.system = None

    def generate(self, prompt, system="", max_tokens=1024, **kw):
        if self.boom:
            raise RuntimeError("model not loaded")
        self.prompt, self.system = prompt, system
        return self.reply


class TestKnowledge:
    def test_the_shipped_pages_load(self):
        topics = advisor.knowledge_topics()
        assert {"weights", "coverage", "thresholds", "training"} <= set(topics)

    def test_the_index_is_not_served_as_a_topic(self):
        assert "README" not in advisor.knowledge_topics()

    def test_every_topic_a_finding_can_reference_actually_exists(self):
        """A finding pointing at a missing page would explain nothing."""
        from modules.highlight_advice import (
            _rule_single_signal, _rule_concentrated, _rule_dominant_tag,
            _rule_flat_score, _rule_boost_never_fired, _rule_near_miss_gap,
            _rule_short_of_target, _rule_silent_detector,
        )
        available = set(advisor.knowledge_topics())
        referenced = {"weights", "thresholds", "coverage", "variety"}
        assert referenced <= available

    def test_only_the_referenced_pages_are_selected(self):
        findings = diagnose(_report())
        picked = advisor.knowledge_for(findings)
        assert picked
        assert set(picked) <= {f.topic for f in findings}

    def test_a_missing_knowledge_folder_is_not_an_error(self, tmp_path):
        assert advisor.knowledge_topics(str(tmp_path / "nope")) == {}


class TestFormatting:
    def test_findings_render_as_text(self):
        text = advisor.format_findings(diagnose(_report()))
        assert "Only one kind of evidence" in text

    def test_no_findings_says_so_rather_than_returning_nothing(self):
        assert "No problems" in advisor.format_findings([])


class TestPrompt:
    def test_the_prompt_carries_the_findings_and_their_pages(self):
        rep = _report()
        prompt = advisor.build_prompt(rep, diagnose(rep))
        assert "Findings computed from this run" in prompt
        assert "Only one kind of evidence" in prompt
        assert "## weights" in prompt or "### weights" in prompt

    def test_the_prompt_states_the_run_it_is_about(self):
        rep = _report()
        prompt = advisor.build_prompt(rep, diagnose(rep))
        assert "1 clips" in prompt or "1 clip" in prompt
        assert "object_points=5" in prompt

    def test_a_question_replaces_the_default_task(self):
        rep = _report()
        prompt = advisor.build_prompt(rep, diagnose(rep), question="Why so short?")
        assert "Why so short?" in prompt

    def test_chapter_structure_reaches_the_model_as_sentences(self):
        """Sentences, not raw ratios — the model must not do the arithmetic."""
        rep = _report()
        rep["chapters"] = [
            {"number": 1, "start": 0.0, "end": 60.0, "duration": 60.0,
             "timestamp": "0:00:00", "title": "Chapter 1", "shots": 20,
             "pace": "steady", "method": "visual", "clips": 1,
             "runtime_share_pct": 25.0, "cut_share_pct": 100.0,
             "cut_share_lift": 4.0},
            {"number": 2, "start": 60.0, "end": 240.0, "duration": 180.0,
             "timestamp": "0:01:00", "title": "Chapter 2", "shots": 20,
             "pace": "steady", "method": "visual", "clips": 0,
             "runtime_share_pct": 75.0, "cut_share_pct": 0.0,
             "cut_share_lift": 0.0},
        ]
        prompt = advisor.build_prompt(rep, diagnose(rep))
        assert "How the video divides" in prompt
        assert "2 chapters" in prompt
        assert "Nothing from this chapter was selected." in prompt

    def test_a_report_without_chapters_adds_no_section(self):
        rep = _report()
        assert "How the video divides" not in advisor.build_prompt(rep, diagnose(rep))

    def test_the_system_prompt_forbids_inventing_numbers(self):
        assert "Never invent a number" in advisor.SYSTEM_PROMPT


class TestNarration:
    def test_no_model_means_no_narration_and_no_error(self):
        rep = _report()
        assert advisor.narrate(rep, diagnose(rep), llm=None) is None

    def test_a_model_is_given_the_system_prompt_and_the_findings(self):
        rep = _report()
        llm = _FakeLLM()
        advisor.narrate(rep, diagnose(rep), llm=llm)
        assert llm.system == advisor.SYSTEM_PROMPT
        assert "Findings computed from this run" in llm.prompt

    def test_a_failing_model_does_not_propagate(self):
        """A missing or broken model must not take the run down with it."""
        rep = _report()
        assert advisor.narrate(rep, diagnose(rep), llm=_FakeLLM(boom=True)) is None

    def test_an_empty_reply_is_treated_as_no_narration(self):
        rep = _report()
        assert advisor.narrate(rep, diagnose(rep), llm=_FakeLLM(reply="   ")) is None


class TestAdvise:
    def test_works_end_to_end_without_a_model(self):
        result = advisor.advise(_report())
        assert result["findings"]
        assert result["narration"] is None
        assert "weights" in result["topics"]

    def test_includes_the_narration_when_a_model_is_present(self):
        result = advisor.advise(_report(), llm=_FakeLLM(reply="Do this."))
        assert result["narration"] == "Do this."

    def test_findings_are_json_serialisable(self):
        import json
        json.dumps(advisor.advise(_report())["findings"])


class _QueryOnlyLLM:
    """Stands in for LLMModule, which exposes query() rather than generate()."""

    def __init__(self, reply="Enable a second signal."):
        self.reply = reply
        self.kwargs = None

    def query(self, user_message, **kwargs):
        self.kwargs = dict(kwargs, user_message=user_message)
        return self.reply


class _NeitherLLM:
    pass


class TestLLMInterfaces:
    """The advisor must work with what the app actually holds."""

    def test_a_query_only_model_is_supported(self):
        rep = _report()
        llm = _QueryOnlyLLM()
        assert advisor.narrate(rep, diagnose(rep), llm=llm) == "Enable a second signal."

    def test_query_is_given_the_system_prompt(self):
        rep = _report()
        llm = _QueryOnlyLLM()
        advisor.narrate(rep, diagnose(rep), llm=llm)
        assert llm.kwargs["system_prompt"] == advisor.SYSTEM_PROMPT

    def test_query_is_told_not_to_add_its_own_video_context(self):
        """The prompt is already complete; LLMModule must not prepend to it."""
        rep = _report()
        llm = _QueryOnlyLLM()
        advisor.narrate(rep, diagnose(rep), llm=llm)
        assert llm.kwargs["free_chat_mode"] is True

    def test_generate_is_preferred_when_both_exist(self):
        class Both:
            def generate(self, prompt, system="", max_tokens=1024):
                return "from generate"

            def query(self, user_message, **kw):
                return "from query"

        rep = _report()
        assert advisor.narrate(rep, diagnose(rep), llm=Both()) == "from generate"

    def test_an_object_with_neither_is_reported_not_crashed(self):
        rep = _report()
        assert advisor.narrate(rep, diagnose(rep), llm=_NeitherLLM()) is None


class TestSummaryTask:
    def test_the_default_ask_is_short(self):
        """This lands above findings that already say everything in full."""
        assert "3 sentences or fewer" in advisor.SUMMARY_TASK
        assert advisor.SUMMARY_TOKENS <= 250

    def test_the_default_task_is_used_when_no_question_is_given(self):
        rep = _report()
        assert advisor.SUMMARY_TASK in advisor.build_prompt(rep, diagnose(rep))


class TestSummariseReportFile:
    def _written(self, tmp_path):
        from modules.highlight_report import write_report
        json_path = tmp_path / "r.json"
        html_path = tmp_path / "r.html"
        write_report(_report(), str(html_path), str(json_path))
        return str(json_path), str(html_path)

    def test_the_summary_lands_in_both_the_record_and_the_page(self, tmp_path):
        import json
        json_path, html_path = self._written(tmp_path)
        text = advisor.summarise_report_file(
            json_path, llm=_FakeLLM(reply="Enable a second signal."))
        assert text == "Enable a second signal."

        record = json.loads(open(json_path, encoding="utf-8").read())
        assert record["advice_narration"] == "Enable a second signal."
        assert record["advice"], "findings must be stored alongside it"
        assert "Enable a second signal." in open(html_path, encoding="utf-8").read()

    def test_a_question_is_passed_through(self, tmp_path):
        json_path, _ = self._written(tmp_path)
        llm = _FakeLLM()
        advisor.summarise_report_file(json_path, llm=llm, question="Why short?")
        assert "Why short?" in llm.prompt

    def test_a_failed_generation_leaves_the_report_untouched(self, tmp_path):
        import json
        json_path, _ = self._written(tmp_path)
        before = open(json_path, encoding="utf-8").read()
        assert advisor.summarise_report_file(json_path, llm=_FakeLLM(boom=True)) is None
        assert open(json_path, encoding="utf-8").read() == before
        assert "advice_narration" not in json.loads(before)

    def test_a_missing_html_is_not_an_error(self, tmp_path):
        """The JSON is the record; the page beside it is optional."""
        import os
        json_path, html_path = self._written(tmp_path)
        os.remove(html_path)
        assert advisor.summarise_report_file(json_path, llm=_FakeLLM()) is not None


# --- reading the footage, as opposed to advising on the settings ------------

def _footage_report():
    return {
        "video": {"duration": 900.0},
        "totals": {"segments": 2, "duration": 60.0, "coverage_pct": 6.7},
        "settings": {},
        "segments": [
            {"index": 1, "range": "0:58 - 1:28", "breakdown": {"motion_peak": 5.0},
             "signals_present": ["motion_peak", "audio"],
             "measured": {"score_percentile": 90.0},
             "motion_peak": {"second": 60, "timestamp": "1:00", "count": 1},
             "loudest": {"second": 64, "timestamp": "1:04", "vs_video_db": 20.0,
                         "level_dbfs": -4.0, "classes": []},
             "expression_peak": {"second": 68, "timestamp": "1:08",
                                 "label": "surprise", "confidence": 0.8,
                                 "seconds": 4, "read_seconds": 12,
                                 "turned": True, "from_label": "neutral"},
             "combination": {"marks": ["loud", "movement", "reading"],
                             "matching": 2, "windows": 60,
                             "window_seconds": 30, "pct": 3.3}},
        ],
    }


def test_the_model_is_shown_what_the_run_found_in_the_footage():
    """Without it the model can discuss weights and never the moments."""
    from modules.advisor import build_prompt
    from modules.highlight_advice import diagnose
    report = _footage_report()
    prompt = build_prompt(report, diagnose(report))
    assert "## What the run found in the footage" in prompt
    assert "## The kept clips, one line each" in prompt
    assert "In clock order" in prompt


def test_the_clip_lines_carry_the_rarity_that_argues_with_them():
    from modules.advisor import build_prompt
    from modules.highlight_advice import diagnose
    report = _footage_report()
    prompt = build_prompt(report, diagnose(report))
    assert "is rare in this video" in prompt


def test_the_reading_prompt_permits_a_guess_and_requires_it_to_be_marked():
    from modules.advisor import READING_SYSTEM_PROMPT as rules
    assert "may suggest what a combination of signals could mean" in rules
    assert "Mark it as your reading" in rules
    # ...and still cannot introduce arithmetic of its own.
    assert "Never state a number that is not already in the material" in rules


def test_the_reading_prompt_carries_the_limit_of_the_expression_channel():
    from modules.advisor import READING_SYSTEM_PROMPT as rules
    assert "cannot tell a performed expression from a felt one" in rules


def test_advice_and_reading_are_kept_in_different_fields():
    """Two opposite questions; a reader must be able to tell which is which."""
    from modules.advisor import READING_SYSTEM_PROMPT, SYSTEM_PROMPT
    assert READING_SYSTEM_PROMPT != SYSTEM_PROMPT
    assert "never speculate" not in READING_SYSTEM_PROMPT.lower()


def test_a_gguf_path_reaches_the_backend_as_a_path():
    """The picker offers a .gguf; a name and a file are different arguments.

    Passing the path as `model` left llama-cpp with an empty `model_path` and a
    "GGUF model not found:" naming no file at all, so every local-model summary
    failed identically to a missing model.
    """
    import modules.advisor as advisor

    seen = {}

    class _Fake:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def load(self):
            return None

    import llm.llm_module as llm_module
    original = llm_module.LLMModule
    llm_module.LLMModule = _Fake
    try:
        advisor.load_llm("llama-cpp", "D:/models/a.gguf")
        assert seen.get("model_path") == "D:/models/a.gguf"
        assert "model" not in seen
        seen.clear()
        advisor.load_llm("ollama", "llama3")
        assert seen.get("model") == "llama3"
        assert "model_path" not in seen
    finally:
        llm_module.LLMModule = original


# --- fitting the prompt into a real context window --------------------------

def _blocks():
    return [(1, "## Footage\n" + "f" * 400),
            (2, "## Documentation\n" + "d" * 400),
            (3, "## Clips\n" + "c" * 400),
            (4, "## Chapters\n" + "h" * 400)]


def test_a_prompt_that_fits_keeps_everything():
    from modules.advisor import _fit
    said = _fit("HEAD", _blocks(), "## Task\nask", max_chars=10000)
    for heading in ("## Footage", "## Documentation", "## Clips", "## Chapters"):
        assert heading in said
    assert "Left out" not in said


def test_the_least_useful_section_goes_first():
    """llama.cpp refuses an over-long call outright, so this is a budget."""
    from modules.advisor import _fit
    said = _fit("HEAD", _blocks(), "## Task\nask", max_chars=1500)
    assert "## Footage" in said          # priority 1 survives
    assert "## Chapters" not in said     # priority 4 goes first
    assert len(said) <= 1500


def test_the_question_is_never_dropped():
    """A model given context and no question answers one it invented."""
    from modules.advisor import _fit
    said = _fit("HEAD", _blocks(), "## Task\nask", max_chars=1)
    assert said.endswith("## Task\nask")


def test_what_was_cut_is_named_in_the_prompt():
    """A summary written without the chapters must not describe a video without them."""
    from modules.advisor import _fit
    said = _fit("HEAD", _blocks(), "## Task\nask", max_chars=1500)
    assert "Left out of this prompt for length: Chapters" in said
    assert "do not describe them as missing from the run" in said


def test_a_real_report_fits_the_window_the_loader_asks_for():
    """The failure this replaces was 5391 tokens against a 4096-token window."""
    from modules.advisor import (DEFAULT_N_CTX, MAX_PROMPT_CHARS, SUMMARY_TOKENS,
                                 build_prompt)
    from modules.highlight_advice import diagnose
    report = _footage_report()
    # Twelve clips, scoring differently — a run the size that overran the window.
    report["segments"] = [
        dict(report["segments"][0], index=i, score=float(i),
             start=i * 60.0, end=i * 60.0 + 30.0)
        for i in range(1, 13)]
    prompt = build_prompt(report, diagnose(report))
    assert len(prompt) <= MAX_PROMPT_CHARS
    # Four characters to the token, plus the answer, inside the window.
    assert MAX_PROMPT_CHARS / 4 + SUMMARY_TOKENS < DEFAULT_N_CTX


def test_the_loader_asks_for_a_window_the_prompt_can_live_in():
    import inspect

    import modules.advisor as advisor
    assert "n_ctx=n_ctx" in inspect.getsource(advisor.load_llm)
    assert advisor.DEFAULT_N_CTX > 4096


def test_reading_is_given_room_that_advising_is_not():
    """At 0.3 and 200 tokens the same report produced the same paragraph."""
    from modules.advisor import (READING_TEMPERATURE, READING_TOKENS,
                                 SUMMARY_TOKENS)
    assert READING_TOKENS > SUMMARY_TOKENS
    assert READING_TEMPERATURE > 0.5


def test_the_temperature_reaches_the_model():
    from modules.advisor import _generate

    seen = {}

    class _Fake:
        def generate(self, prompt, system="", max_tokens=0, **kwargs):
            seen.update(kwargs)
            return "said"

    _generate(_Fake(), "p", "s", 100, 0.85)
    assert seen.get("temperature") == 0.85
    seen.clear()
    # Advising leaves it alone rather than pinning it to a number of its own.
    _generate(_Fake(), "p", "s", 100, None)
    assert "temperature" not in seen
