"""Tests for the per-chapter narration — the boundary, not the prose.

Nothing here can check whether a paragraph is any good; a model wrote it. What
can be checked is everything around it, and that is where this feature can hurt
someone:

* a chapter is told from *its own* material and not the whole report's, so a
  paragraph cannot describe a stretch it was never shown;
* one failed or empty call costs one paragraph and not the run;
* what a model wrote is stored and rendered as a reading, with the model named,
  and never merged into the measured sentences beside it;
* the chapters are told in order, each handed the one before it, and a skipped
  chapter is not passed off as the predecessor of the next.

Fixture speech is mundane on purpose — a workshop, a kitchen. The narration
path does not care what was said, and the repo carries no content.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from modules import chapter_story
from modules.highlight_report import build_report


class _FakeLLM:
    """Stands in for llm.llm_module.LLMModule — same generate() shape."""

    def __init__(self, reply="They talked at a bench.", boom=False,
                 empty_after=None, sees_images=True):
        self.reply, self.boom = reply, boom
        self.empty_after, self.sees_images = empty_after, sees_images
        self.calls = []

    def generate(self, prompt, system="", max_tokens=1024, images=None, **kw):
        if images is not None and not self.sees_images:
            raise RuntimeError("this model has no projector")
        self.calls.append({"prompt": prompt, "system": system,
                           "images": images, "kw": kw})
        if self.boom:
            raise RuntimeError("model not loaded")
        if self.empty_after is not None and len(self.calls) > self.empty_after:
            return "  "
        return f"{self.reply} ({len(self.calls)})"


def _chapter(number, start, end):
    return {"number": number, "start": float(start), "end": float(end),
            "duration": float(end - start), "timestamp": "0:00:00",
            "title": f"Chapter {number}", "shots": 10, "pace": "steady"}


def _transcript():
    out = []
    for t in range(0, 290, 10):
        out.append({"start": float(t), "end": float(t) + 4.0,
                    "text": "pass me the chisel and the mallet"})
    for t in range(300, 590, 10):
        out.append({"start": float(t), "end": float(t) + 4.0,
                    "text": "the kettle is next to the mugs"})
    return out


def _report():
    score = np.zeros(600)
    score[100] = 10.0
    return build_report(
        video_path="a.mp4", video_duration=600, score=score,
        signals={"object": score}, segments=[(95, 125)],
        chapters=[_chapter(1, 0, 300), _chapter(2, 300, 600)],
        transcript=_transcript())


class TestPrompt:
    def test_a_chapter_is_told_from_its_own_speech(self):
        rep = _report()
        prompt = chapter_story.chapter_prompt(rep, rep["chapters"][0])
        assert "chisel" in prompt
        # The second chapter's words must not be in the first one's prompt, or
        # a paragraph can describe a stretch it was never shown.
        assert "kettle" not in prompt

    def test_the_measured_sentences_are_included(self):
        rep = _report()
        prompt = chapter_story.chapter_prompt(rep, rep["chapters"][0])
        assert "What was measured here" in prompt

    def test_a_silent_chapter_says_so_rather_than_omitting_the_section(self):
        rep = _report()
        bare = dict(rep["chapters"][0])
        bare.pop("dialogue", None)
        bare.pop("quotes", None)
        assert "Nothing was transcribed" in chapter_story.chapter_prompt(rep, bare)

    def test_the_previous_chapter_is_carried_forward_but_capped(self):
        rep = _report()
        prompt = chapter_story.chapter_prompt(rep, rep["chapters"][1],
                                              previous="X" * 5000)
        assert "What the previous stretch was" in prompt
        assert "X" * (chapter_story.CARRY_CHARS + 1) not in prompt

    def test_a_long_chapter_is_trimmed_from_the_middle(self):
        # Both ends survive: how a stretch opened and where it had got to are
        # what place it, and cutting the tail would describe every long chapter
        # by its first third.
        rep = _report()
        ch = dict(rep["chapters"][0])
        ch["dialogue"] = [{"start": float(i), "timestamp": "0:00:00",
                           "text": f"line {i} " + "padding " * 20}
                          for i in range(400)]
        prompt = chapter_story.chapter_prompt(rep, ch, max_chars=4000)
        assert "line 0 " in prompt
        assert "line 399 " in prompt
        assert "omitted for length" in prompt


class TestTell:
    def test_every_chapter_gets_a_paragraph(self):
        rep = _report()
        told = chapter_story.tell(rep, llm=_FakeLLM(), log_fn=lambda _m: None)
        assert all(ch["story"] for ch in told)

    def test_chapters_are_told_in_order_and_handed_the_previous_one(self):
        rep = _report()
        llm = _FakeLLM()
        chapter_story.tell(rep, llm=llm, log_fn=lambda _m: None)
        assert len(llm.calls) == 2
        assert "What the previous stretch was" not in llm.calls[0]["prompt"]
        # The first chapter's answer, verbatim, in the second chapter's prompt.
        assert "(1)" in llm.calls[1]["prompt"]

    def test_a_failing_call_costs_one_paragraph_not_the_run(self):
        rep = _report()

        class _FlakyLLM(_FakeLLM):
            def generate(self, prompt, system="", max_tokens=1024,
                         images=None, **kw):
                self.calls.append({"prompt": prompt})
                if len(self.calls) == 1:
                    raise RuntimeError("out of memory")
                return "The second one worked."

        told = chapter_story.tell(rep, llm=_FlakyLLM(), log_fn=lambda _m: None)
        assert "story" not in told[0]
        assert told[1]["story"] == "The second one worked."

    def test_a_skipped_chapter_is_not_passed_off_as_the_predecessor(self):
        rep = _report()
        llm = _FakeLLM(empty_after=0)
        told = chapter_story.tell(rep, llm=llm, log_fn=lambda _m: None)
        assert not any(ch.get("story") for ch in told)
        assert "What the previous stretch was" not in llm.calls[1]["prompt"]

    def test_no_model_means_the_chapters_come_back_untouched(self):
        rep = _report()
        told = chapter_story.tell(rep, llm=None, log_fn=lambda _m: None)
        assert told == rep["chapters"]

    def test_input_chapters_are_not_mutated(self):
        rep = _report()
        chapter_story.tell(rep, llm=_FakeLLM(), log_fn=lambda _m: None)
        assert "story" not in rep["chapters"][0]

    def test_cancelling_stops_the_walk(self):
        rep = _report()
        llm = _FakeLLM()
        chapter_story.tell(rep, llm=llm, log_fn=lambda _m: None,
                           cancel_fn=lambda: True)
        assert llm.calls == []

    def test_frames_are_offered_to_a_model_that_takes_them(self):
        rep = _report()
        llm = _FakeLLM()
        chapter_story.tell(rep, llm=llm, log_fn=lambda _m: None,
                           frames_fn=lambda _s, _e: ["ZmFrZQ=="])
        assert llm.calls[0]["images"] == ["ZmFrZQ=="]

    def test_a_model_that_rejects_images_still_gets_a_paragraph(self):
        # The projector mismatch is a configuration a user can reach in a few
        # clicks. Losing the pictures is acceptable; losing the telling is not.
        rep = _report()
        told = chapter_story.tell(rep, llm=_FakeLLM(sees_images=False),
                                  log_fn=lambda _m: None,
                                  frames_fn=lambda _s, _e: ["ZmFrZQ=="])
        assert all(ch["story"] for ch in told)

    def test_the_system_prompt_forbids_inventing_figures(self):
        rep = _report()
        llm = _FakeLLM()
        chapter_story.tell(rep, llm=llm, log_fn=lambda _m: None)
        assert "Never state a number" in llm.calls[0]["system"]


class TestOnDisk:
    def _written(self, tmp_path, **kw):
        rep = _report()
        json_path = tmp_path / "a_why.json"
        json_path.write_text(json.dumps(rep), encoding="utf-8")
        told = chapter_story.tell_report_file(
            str(json_path), llm=kw.pop("llm", _FakeLLM()),
            use_frames=False, log_fn=lambda _m: None, **kw)
        return told, json.loads(json_path.read_text(encoding="utf-8"))

    def test_the_record_keeps_the_paragraphs_and_names_the_model(self, tmp_path):
        told, rep = self._written(tmp_path, model_name="ollama/llama3")
        assert told == 2
        assert rep["chapters"][0]["story"]
        assert rep["chapter_story"]["model"] == "ollama/llama3"

    def test_the_page_says_a_model_wrote_them(self, tmp_path):
        from modules.highlight_report import render_html

        _told, rep = self._written(tmp_path, model_name="ollama/llama3")
        page = render_html(rep)
        assert rep["chapters"][0]["story"] in page
        assert "a reading, not a" in page
        assert "ollama/llama3" in page

    def test_nothing_told_leaves_the_record_alone(self, tmp_path):
        told, rep = self._written(tmp_path, llm=_FakeLLM(boom=True))
        assert told == 0
        assert "chapter_story" not in rep
        assert "story" not in rep["chapters"][0]

    def test_a_missing_video_only_costs_the_pictures(self):
        # The report is routinely opened away from its footage. That must cost
        # the frames and nothing else.
        assert chapter_story.frames_from_video("nowhere/at/all.mp4") is None
