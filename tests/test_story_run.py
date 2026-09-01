"""Tests for narrating a report as part of the run, and for which brief goes out.

Two things are worth pinning here and neither is the prose:

* the brief a captioner receives is not the brief an instruction-follower
  receives. This is the bug that produced a page of recited rules instead of a
  report, and it is invisible in every other test — the fake model returns its
  canned reply whatever it is sent, so only the recorded ``system`` catches it;
* narration at the tail of a pipeline never costs the run. The cut and the
  report exist by the time it starts, so a model that cannot be reached, a pass
  that raises, or a cancelled run must all come back as a number and a log line.

No Qt here: this imports the pure half only, and the GUI's checkboxes are
somebody else's test.
"""
from __future__ import annotations

import pytest

from modules import chapter_story, clip_story, story_run
from modules.llm_models import is_captioner


class _Recorder:
    """Stands in for LLMModule, and keeps every system prompt it was sent."""

    def __init__(self, reply="A bench, and two people on it."):
        self.reply, self.systems = reply, []

    def generate(self, prompt, system="", max_tokens=1024, images=None, **kw):
        self.systems.append(system)
        return self.reply

    def accepts_images(self):
        return True


class TestWhichBrief:
    @pytest.mark.parametrize("name", ["joycaption:q4", "ollama/JoyCaption",
                                      "joy-caption-beta", "joy_caption"])
    def test_captioners_are_recognised_through_tags_and_prefixes(self, name):
        assert is_captioner(name)

    @pytest.mark.parametrize("name", ["qwen2.5vl:7b", "llava-llama3:8b", "",
                                      None])
    def test_everything_else_is_an_instruction_follower(self, name):
        assert not is_captioner(name)

    def test_a_captioner_is_not_sent_a_numbered_rulebook(self):
        # The whole point: "1." in the system prompt is what came back as the
        # answer, so its absence is the thing to assert.
        for module in (chapter_story, clip_story):
            flat = module.system_prompt_for("joycaption:q4")
            assert "Rules:" not in flat
            assert "\n1. " not in flat

    def test_an_instruction_follower_still_gets_the_rulebook(self):
        assert chapter_story.system_prompt_for("qwen2.5vl:7b") == \
            chapter_story.STORY_SYSTEM_PROMPT
        assert clip_story.system_prompt_for("qwen2.5vl:7b") == \
            clip_story.CLIP_SYSTEM_PROMPT

    def test_an_unnamed_model_keeps_the_behaviour_this_code_always_had(self):
        assert chapter_story.system_prompt_for(None) == \
            chapter_story.STORY_SYSTEM_PROMPT

    def test_the_flat_brief_keeps_what_the_rulebook_enforced(self):
        # Flattening is a change of packaging, not of constraints. The two that
        # would be missed silently: figures belong to the page, and a clip
        # paragraph must attribute the run's claims rather than adopt them.
        chapter = chapter_story.CAPTIONER_SYSTEM_PROMPT
        assert "number" in chapter and "already" in chapter
        clip = clip_story.CAPTIONER_SYSTEM_PROMPT
        assert "number" in clip
        assert "flagged" in clip

    def test_the_brief_reaches_the_model(self, monkeypatch):
        # Threading a prompt through four call sites is exactly the kind of
        # change that compiles and does nothing, so follow it to the call.
        llm = _Recorder()
        report = {"segments": [{"start": 1.0, "end": 3.0, "range": "0:01-0:03"}]}
        monkeypatch.setattr(clip_story, "_frames_delivered",
                            lambda _llm, _images: 1)
        clip_story.tell(report, llm=llm, frames_fn=lambda _s, _e: ["img"],
                        model_name="joycaption:q4", log_fn=lambda _m: None)
        assert llm.systems == [clip_story.CAPTIONER_SYSTEM_PROMPT]


class TestWhatTheRunAsksFor:
    def test_both_scales_are_narrated_by_default(self):
        # The default is the feature: a report that answers "what is in this
        # clip" but not "what was this stretch doing" can still be asked
        # something it has no answer for.
        assert story_run.wanted({}) == (True, True)

    def test_either_pass_can_be_turned_off_for_a_run(self):
        # The cost is a call per chapter plus one per clip, so switching one off
        # has to actually switch it off — not merely fail to switch it on.
        assert story_run.wanted({"narrate_chapters": False}) == (False, True)
        assert story_run.wanted({"narrate_clips": False}) == (True, False)
        assert story_run.wanted({"narrate_chapters": False,
                                 "narrate_clips": False}) == (False, False)

    def test_nothing_wanted_means_no_model_is_ever_built(self, monkeypatch):
        def _boom(*_a, **_kw):                      # pragma: no cover - guard
            raise AssertionError("loaded a model for a run that wanted none")

        monkeypatch.setattr("modules.advisor.load_llm", _boom)
        assert story_run.narrate_report_file(
            "nowhere.json",
            config={"narrate_clips": False, "narrate_chapters": False},
            log_fn=lambda _m: None) == {"chapters": 0, "clips": 0}


class TestNarrationNeverCostsTheRun:
    def test_an_unreachable_model_is_a_log_line_and_a_zero(self, monkeypatch):
        monkeypatch.setattr("modules.advisor.load_llm",
                            lambda *_a, **_kw: None)
        said = []
        assert story_run.narrate_report_file(
            "nowhere.json", config={}, log_fn=said.append) == \
            {"chapters": 0, "clips": 0}
        assert any("could not reach" in line.lower() for line in said)

    def test_a_model_that_raises_on_load_is_caught(self, monkeypatch):
        def _boom(*_a, **_kw):
            raise RuntimeError("no CUDA device")

        monkeypatch.setattr("modules.advisor.load_llm", _boom)
        said = []
        assert story_run.narrate_report_file(
            "nowhere.json", config={}, log_fn=said.append)["clips"] == 0
        assert any("no CUDA device" in line for line in said)

    def test_a_pass_that_raises_loses_its_paragraphs_and_nothing_else(
            self, monkeypatch):
        def _boom(*_a, **_kw):
            raise ValueError("the report moved")

        monkeypatch.setattr("modules.advisor.load_llm",
                            lambda *_a, **_kw: _Recorder())
        monkeypatch.setattr("modules.clip_story.tell_report_file", _boom)
        said = []
        assert story_run.narrate_report_file(
            "nowhere.json", config={"narrate_chapters": False},
            log_fn=said.append) == {"chapters": 0, "clips": 0}
        assert any("the report moved" in line for line in said)

    def test_a_blind_model_skips_the_clip_pass_rather_than_writing_from_figures(
            self, monkeypatch):
        # A model handed no pictures writes a fluent paragraph out of the
        # measurements and nothing on the page marks it as invented.
        class _Blind(_Recorder):
            def accepts_images(self):
                return False

        monkeypatch.setattr("modules.advisor.load_llm",
                            lambda *_a, **_kw: _Blind())
        monkeypatch.setattr(
            "modules.clip_story.tell_report_file",
            lambda *_a, **_kw: pytest.fail("read clips it could not see"))
        said = []
        assert story_run.narrate_report_file(
            "nowhere.json", config={"narrate_chapters": False},
            log_fn=said.append)["clips"] == 0
        assert any("cannot see" in line for line in said)

    def test_a_cancelled_run_narrates_nothing(self, monkeypatch):
        monkeypatch.setattr(
            "modules.advisor.load_llm",
            lambda *_a, **_kw: pytest.fail("built a model after cancel"))
        assert story_run.narrate_report_file(
            "nowhere.json", config={}, log_fn=lambda _m: None,
            cancel_fn=lambda: True) == {"chapters": 0, "clips": 0}
