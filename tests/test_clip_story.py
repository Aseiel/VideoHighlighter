"""Tests for the per-clip reading — the boundary, not the prose.

Nothing here can check whether a paragraph is any good; a model wrote it. What
can be checked is everything around it, and it is the same list the chapter
pass keeps, plus the one thing this pass exists for:

* a clip is read from *its own* frames and its own material, so a paragraph
  cannot describe a span it was never shown;
* the frames are sampled across the clip, not at the peak second, because the
  answer is supposed to be about what changes;
* one failed or empty call costs one paragraph and not the run;
* what a model wrote is stored and rendered as a reading, with the model named,
  and never merged into the measured sentences beside it;
* a report with no transcript still produces a full prompt — that is the case
  the pass was written for, and the one where an empty section silently
  becomes an empty answer.

Fixture speech is mundane on purpose — a workshop, a kitchen. The reading path
does not care what was said, and the repo carries no content.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from modules import clip_story
from modules.highlight_report import build_report, render_html


FRAMES = ["Zg==", "cg==", "YQ==", "bQ=="]


def _frames(_start, _end):
    return list(FRAMES)


class _FakeLLM:
    """A raw backend — exposes generate(), which takes a list of frames."""

    def __init__(self, reply="Two people at a bench.", boom=False,
                 empty_after=None):
        self.reply, self.boom = reply, boom
        self.empty_after = empty_after
        self.calls = []

    def generate(self, prompt, system="", max_tokens=1024, images=None, **kw):
        self.calls.append({"prompt": prompt, "system": system,
                           "images": images, "kw": kw})
        if self.boom:
            raise RuntimeError("model not loaded")
        if self.empty_after is not None and len(self.calls) > self.empty_after:
            return "  "
        return f"{self.reply} ({len(self.calls)})"


class _FakeModule:
    """What the GUI actually holds — `LLMModule`, which offers query() only.

    The distinction is the whole of `_frames_delivered` and half of
    `_read_one`: an object of this shape is why the chapter pass has never sent
    a picture from the app, and a fake that exposes generate() would prove
    nothing about the path the app takes.
    """

    def __init__(self, reply="Read across the clip."):
        self.reply = reply
        self.calls = []

    def query(self, prompt, system_prompt="", frame_base64=None, frames=None,
              **kw):
        self.calls.append({"prompt": prompt, "system": system_prompt,
                           "frame": frame_base64, "frames": frames, "kw": kw})
        return f"{self.reply} ({len(self.calls)})"


class _OldModule:
    """An `LLMModule` from before query() learned to take several frames.

    Handing this one ``frames=`` raises TypeError, which `tell` would catch and
    turn into a lost paragraph — a clip losing its reading because of a keyword
    argument rather than because of anything in the footage.
    """

    def __init__(self, reply="One frame's worth."):
        self.reply = reply
        self.calls = []

    def query(self, prompt, system_prompt="", frame_base64=None, **kw):
        self.calls.append({"prompt": prompt, "system": system_prompt,
                           "frame": frame_base64, "kw": kw})
        return f"{self.reply} ({len(self.calls)})"


def _transcript():
    out = []
    for t in range(100, 130, 10):
        out.append({"start": float(t), "end": float(t) + 4.0,
                    "text": "pass me the chisel and the mallet"})
    for t in range(300, 330, 10):
        out.append({"start": float(t), "end": float(t) + 4.0,
                    "text": "the kettle is next to the mugs"})
    return out


def _report(transcript=None):
    score = np.zeros(600)
    score[110] = 10.0
    score[310] = 12.0
    return build_report(
        video_path="a.mp4", video_duration=600, score=score,
        signals={"object": score}, segments=[(100, 130), (300, 330)],
        transcript=transcript)


class TestPrompt:
    def test_a_clip_is_read_from_its_own_speech(self):
        rep = _report(_transcript())
        prompt = clip_story.clip_prompt(rep, rep["segments"][0])
        assert "chisel" in prompt
        # The other clip's words must not be in this one's prompt, or a
        # paragraph can describe a span it was never shown.
        assert "kettle" not in prompt

    def test_the_measured_sentences_are_included(self):
        rep = _report()
        prompt = clip_story.clip_prompt(rep, rep["segments"][0])
        assert "What was measured here" in prompt

    def test_a_silent_clip_says_so_rather_than_omitting_the_section(self):
        # The whole reason this pass exists. A report with no transcript must
        # still tell the model there is nothing to account for, or it goes
        # looking for dialogue in the frames.
        rep = _report()
        prompt = clip_story.clip_prompt(rep, rep["segments"][0])
        assert "Nothing was transcribed as speech" in prompt
        assert "frames are all there is to go on" in prompt

    def test_the_previous_clip_is_carried_forward_but_capped(self):
        rep = _report()
        prompt = clip_story.clip_prompt(rep, rep["segments"][1],
                                        previous="X" * 5000)
        assert "What the previous clip was" in prompt
        assert "X" * (clip_story.CARRY_CHARS + 1) not in prompt

    def test_detector_labels_are_named_by_the_source_that_produced_them(self):
        rep = _report()
        entry = dict(rep["segments"][0])
        entry["objects"] = ["bench"]
        entry["events"] = ["bench_work"]
        entry["actions"] = [{"name": "sanding", "confidence": 0.02,
                             "tier": "reduced"}]
        prompt = clip_story.clip_prompt(rep, entry)
        assert "object detector labelled" in prompt
        assert "composition rules recognised" in prompt
        # The confidence travels with the action, so a 0.02 guess is not read
        # as an observation of equal standing to the detections above it.
        assert "0.02" in prompt

    def test_the_task_asks_what_the_clip_shows(self):
        rep = _report()
        assert clip_story.CLIP_TASK in clip_story.clip_prompt(
            rep, rep["segments"][0])

    def test_the_prompt_says_how_many_frames_the_model_actually_has(self):
        # Rule 4 asks what changes between the frames. Told it has four when it
        # was sent one, a model answers the question it was asked rather than
        # the one it can see — and invents the motion to do it.
        rep = _report()
        many = clip_story.clip_prompt(rep, rep["segments"][0], frame_count=4)
        one = clip_story.clip_prompt(rep, rep["segments"][0], frame_count=1)
        assert "4 frames from across this clip" in many
        assert "one frame, from the middle" in one


class TestNarrationNotes:
    """A narration already run over this video is free material, and only the
    notes landing inside the clip may be used.

    Skipped where per-frame narration does not exist: this file is shared
    verbatim between the two editions, and a test that cannot pass in one of
    them is worth less than one identical file that skips honestly there.
    """

    def test_notes_inside_the_clip_are_offered_and_others_are_not(self, tmp_path,
                                                                  monkeypatch):
        narration_notes = pytest.importorskip("modules.narration_notes")

        video = tmp_path / "a.mp4"
        video.write_bytes(b"")
        narration_notes.save(str(video), [
            {"start": 105.0, "end": 115.0, "text": "she reaches for the chisel"},
            {"start": 305.0, "end": 315.0, "text": "he fills the kettle"},
        ], log_fn=lambda _m: None)

        rep = _report()
        rep["video"]["path"] = str(video)
        prompt = clip_story.clip_prompt(rep, rep["segments"][0])
        assert "reaches for the chisel" in prompt
        assert "kettle" not in prompt

    def test_no_sidecar_means_no_section(self):
        rep = _report()
        assert "Notes written over this span" not in clip_story.clip_prompt(
            rep, rep["segments"][0])


class TestInterface:
    """Which call is made, and how many frames survive it.

    `LLMModule` — the object the GUI holds — offers `query`, not `generate`.
    The chapter pass reaches for `generate` inside a try/except and falls back
    to text on the AttributeError, so from the app it has never once sent a
    picture. Everything here exists so that cannot happen again quietly.
    """

    def test_a_raw_backend_is_sent_every_frame(self):
        rep = _report()
        llm = _FakeLLM()
        clip_story.tell(rep, llm=llm, log_fn=lambda _m: None, frames_fn=_frames)
        assert llm.calls[0]["images"] == FRAMES

    def test_an_llmmodule_is_sent_every_frame_through_query(self):
        rep = _report()
        llm = _FakeModule()
        read = clip_story.tell(rep, llm=llm, log_fn=lambda _m: None,
                               frames_fn=_frames)
        assert llm.calls[0]["frames"] == FRAMES
        assert "4 frames from across this clip" in llm.calls[0]["prompt"]
        assert all(e["story"] for e in read)

    def test_a_query_that_takes_one_frame_gets_the_middle_one(self):
        rep = _report()
        llm = _OldModule()
        read = clip_story.tell(rep, llm=llm, log_fn=lambda _m: None,
                               frames_fn=_frames)
        assert llm.calls[0]["frame"] == FRAMES[len(FRAMES) // 2]
        assert all(e["story"] for e in read)

    def test_a_one_frame_interface_is_told_it_has_one_frame_not_four(self):
        # Rule 4 asks what changes between the frames. A model sent one picture
        # and told it has four answers anyway, and invents the motion to do it.
        rep = _report()
        llm = _OldModule()
        clip_story.tell(rep, llm=llm, log_fn=lambda _m: None, frames_fn=_frames)
        assert "one frame, from the middle" in llm.calls[0]["prompt"]
        assert "4 frames" not in llm.calls[0]["prompt"]

    def test_an_object_offering_neither_interface_is_an_error_not_a_paragraph(self):
        rep = _report()
        logged = []
        read = clip_story.tell(rep, llm=object(), log_fn=logged.append,
                               frames_fn=_frames)
        assert not any(e.get("story") for e in read)

    def test_without_frames_nothing_is_written_at_all(self):
        # The paragraph's entire content is the picture. A text-only fallback
        # would produce a fluent restatement of the figures, indistinguishable
        # on the page from a paragraph written by a model that could see.
        rep = _report()
        llm = _FakeLLM()
        logged = []
        read = clip_story.tell(rep, llm=llm, log_fn=logged.append)
        assert llm.calls == []
        assert not any(e.get("story") for e in read)
        assert any("no frames to read" in m for m in logged)


class TestTell:
    def test_every_clip_gets_a_paragraph(self):
        rep = _report()
        read = clip_story.tell(rep, llm=_FakeLLM(), log_fn=lambda _m: None,
                               frames_fn=_frames)
        assert all(e["story"] for e in read)

    def test_clips_are_read_in_order_and_handed_the_previous_one(self):
        rep = _report()
        llm = _FakeLLM()
        clip_story.tell(rep, llm=llm, log_fn=lambda _m: None, frames_fn=_frames)
        assert len(llm.calls) == 2
        assert "What the previous clip was" not in llm.calls[0]["prompt"]
        assert "(1)" in llm.calls[1]["prompt"]

    def test_frames_are_sampled_across_the_clip_not_at_its_peak(self):
        # The thumbnail on the card is already the peak second. Sampling the
        # same instant again would make the paragraph a caption of a picture
        # the reader can see, instead of an account of what changes.
        rep = _report()
        spans = []

        def frames(start, end):
            spans.append((start, end))
            return list(FRAMES)

        clip_story.tell(rep, llm=_FakeLLM(), log_fn=lambda _m: None,
                        frames_fn=frames)
        assert spans == [(100.0, 130.0), (300.0, 330.0)]

    def test_a_failing_call_costs_one_paragraph_not_the_run(self):
        rep = _report()

        class _FlakyLLM(_FakeLLM):
            def generate(self, prompt, system="", max_tokens=1024,
                         images=None, **kw):
                self.calls.append({"prompt": prompt})
                if len(self.calls) == 1:
                    raise RuntimeError("out of memory")
                return "The second one worked."

        read = clip_story.tell(rep, llm=_FlakyLLM(), log_fn=lambda _m: None,
                               frames_fn=_frames)
        assert "story" not in read[0]
        assert read[1]["story"] == "The second one worked."

    def test_a_clip_whose_frames_fail_is_skipped_not_written_blind(self):
        rep = _report()
        llm = _FakeLLM()

        def frames(start, _end):
            return [] if start == 100.0 else list(FRAMES)

        read = clip_story.tell(rep, llm=llm, log_fn=lambda _m: None,
                               frames_fn=frames)
        assert "story" not in read[0]
        assert read[1]["story"]
        assert len(llm.calls) == 1

    def test_a_skipped_clip_is_not_passed_off_as_the_predecessor(self):
        rep = _report()
        llm = _FakeLLM(empty_after=0)
        read = clip_story.tell(rep, llm=llm, log_fn=lambda _m: None,
                               frames_fn=_frames)
        assert not any(e.get("story") for e in read)
        assert "What the previous clip was" not in llm.calls[1]["prompt"]

    def test_a_looping_clip_keeps_what_came_before_the_loop(self):
        rep = _report()
        sentence = "They stand at the bench and say very little to each other. "

        class _StuckLLM(_FakeLLM):
            def generate(self, prompt, **kw):
                self.calls.append({"prompt": prompt})
                return sentence * 5

        logged = []
        read = clip_story.tell(rep, llm=_StuckLLM(), log_fn=logged.append,
                               frames_fn=_frames)
        assert read[0]["story"] == sentence.strip()
        assert any("repeated itself" in m for m in logged)

    def test_no_model_means_the_clips_come_back_untouched(self):
        rep = _report()
        assert clip_story.tell(rep, llm=None, log_fn=lambda _m: None,
                               frames_fn=_frames) == rep["segments"]

    def test_input_segments_are_not_mutated(self):
        rep = _report()
        clip_story.tell(rep, llm=_FakeLLM(), log_fn=lambda _m: None,
                        frames_fn=_frames)
        assert "story" not in rep["segments"][0]

    def test_cancelling_stops_the_walk(self):
        rep = _report()
        llm = _FakeLLM()
        clip_story.tell(rep, llm=llm, log_fn=lambda _m: None,
                        frames_fn=_frames, cancel_fn=lambda: True)
        assert llm.calls == []

    def test_the_system_prompt_forbids_inventing_figures(self):
        rep = _report()
        llm = _FakeLLM()
        clip_story.tell(rep, llm=llm, log_fn=lambda _m: None, frames_fn=_frames)
        assert "Never state a number" in llm.calls[0]["system"]


class TestRender:
    def test_a_read_clip_is_marked_as_a_reading_on_the_page(self):
        rep = _report()
        rep["segments"][0]["story"] = "Two people stand at a bench."
        page = render_html(rep)
        assert "read from this clip" in page
        assert "Two people stand at a bench." in page

    def test_an_unread_report_gains_no_block(self):
        assert "read from this clip" not in render_html(_report())

    def test_the_reading_is_escaped_like_everything_else(self):
        rep = _report()
        rep["segments"][0]["story"] = "<script>alert(1)</script>"
        assert "<script>alert(1)</script>" not in render_html(rep)


class TestReportFile:
    def _write(self, tmp_path):
        from modules.highlight_report import write_report

        html_path, json_path = tmp_path / "r.html", tmp_path / "r.json"
        write_report(_report(), str(html_path), str(json_path))
        return html_path, json_path

    def test_the_record_and_the_page_are_both_updated(self, tmp_path):
        html_path, json_path = self._write(tmp_path)
        read = clip_story.tell_report_file(
            str(json_path), llm=_FakeLLM(), model_name="ollama/vision",
            frames_fn=_frames, log_fn=lambda _m: None)
        assert read == 2
        record = json.loads(json_path.read_text(encoding="utf-8"))
        assert all(e["story"] for e in record["segments"])
        # Re-rendered from the updated record rather than patched, so the two
        # cannot disagree about what a clip says.
        assert "read from this clip" in html_path.read_text(encoding="utf-8")

    def test_the_model_is_named_in_the_record(self, tmp_path):
        _html, json_path = self._write(tmp_path)
        clip_story.tell_report_file(
            str(json_path), llm=_FakeLLM(), model_name="ollama/vision",
            frames_fn=_frames, log_fn=lambda _m: None)
        stamp = json.loads(json_path.read_text(encoding="utf-8"))["clip_story"]
        assert stamp["model"] == "ollama/vision"
        assert stamp["clips"] == 2

    def test_a_run_that_reads_nothing_leaves_the_report_alone(self, tmp_path):
        _html, json_path = self._write(tmp_path)
        before = json_path.read_text(encoding="utf-8")
        read = clip_story.tell_report_file(
            str(json_path), llm=_FakeLLM(empty_after=0), frames_fn=_frames,
            log_fn=lambda _m: None)
        assert read == 0
        assert json_path.read_text(encoding="utf-8") == before

    def test_an_edit_made_while_the_pass_ran_survives_it(self, tmp_path):
        # The pass takes minutes and the record is live throughout — the app
        # writes the advisor's summary into the same file while a user waits.
        # Writing back the copy loaded at the start reverts their work, and
        # from the outside that looks exactly like the pass having done
        # nothing at all.
        _html, json_path = self._write(tmp_path)

        class _EditsTheFileMidRun(_FakeLLM):
            def generate(self, prompt, **kw):
                record = json.loads(json_path.read_text(encoding="utf-8"))
                record["advice_narration"] = "written while the pass ran"
                json_path.write_text(json.dumps(record), encoding="utf-8")
                return super().generate(prompt, **kw)

        clip_story.tell_report_file(
            str(json_path), llm=_EditsTheFileMidRun(), frames_fn=_frames,
            log_fn=lambda _m: None)
        record = json.loads(json_path.read_text(encoding="utf-8"))
        assert record["advice_narration"] == "written while the pass ran"
        assert all(e["story"] for e in record["segments"])

    def test_a_re_analysis_mid_run_is_refused_rather_than_merged(self, tmp_path):
        # Different clips on disk than the ones that were read: the readings
        # describe a cut the report no longer contains, so they are dropped
        # rather than pinned onto whatever now sits at those indices.
        _html, json_path = self._write(tmp_path)

        class _ReAnalyses(_FakeLLM):
            def generate(self, prompt, **kw):
                record = json.loads(json_path.read_text(encoding="utf-8"))
                for i, entry in enumerate(record["segments"], start=100):
                    entry["index"] = i
                json_path.write_text(json.dumps(record), encoding="utf-8")
                return super().generate(prompt, **kw)

        logged = []
        read = clip_story.tell_report_file(
            str(json_path), llm=_ReAnalyses(), frames_fn=_frames,
            log_fn=logged.append)
        assert read == 0
        assert any("re-analysed" in m for m in logged)
        record = json.loads(json_path.read_text(encoding="utf-8"))
        assert not any(e.get("story") for e in record["segments"])

    def test_a_missing_video_refuses_rather_than_writing_from_figures(self,
                                                                     tmp_path):
        # `_report()` names "a.mp4", which is not beside the record. The pass
        # must decline: a page full of paragraphs written without the frames
        # looks exactly like a page full of paragraphs written with them.
        _html, json_path = self._write(tmp_path)
        llm = _FakeLLM()
        logged = []
        read = clip_story.tell_report_file(str(json_path), llm=llm,
                                           log_fn=logged.append)
        assert read == 0
        assert llm.calls == []
        assert any("no frames to read" in m for m in logged)

    def test_the_players_survive_a_re_render(self, tmp_path):
        # Re-rendering without a media source would silently strip the per-clip
        # players a full write put on the page.
        from modules.highlight_report import write_report

        video = tmp_path / "a.mp4"
        video.write_bytes(b"")
        rep = _report()
        rep["video"]["path"] = str(video)
        html_path, json_path = tmp_path / "r.html", tmp_path / "r.json"
        write_report(rep, str(html_path), str(json_path))
        assert "<video" in html_path.read_text(encoding="utf-8")
        clip_story.tell_report_file(str(json_path), llm=_FakeLLM(),
                                    frames_fn=_frames, log_fn=lambda _m: None)
        assert "<video" in html_path.read_text(encoding="utf-8")
