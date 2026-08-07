"""Tests for corroborating what a stretch says against what the run measured.

The feature exists so a narration can *disagree* with a speaker, so most of
what is worth testing is the ways it could quietly stop being able to:

* the evidence is video-wide, not chapter-local — somebody describing in the
  last chapter something that happened in the middle is the ordinary case, and
  a row that only looked here would report it as never having happened;
* a class the detector barely found still produces a row, because "labelled in
  nine seconds" beside a sentence calling it the best part of the file is the
  finding, not a failure;
* a one-word class name does not fire on every chapter that says the word;
* the figures reach the page as well as the prompt, so the paragraph the model
  writes from them can be checked without trusting it.

Fixture classes are named after workshop objects. The matching path does not
care what a class is called, and the repo carries no vocabulary of its own.
"""
from __future__ import annotations

import numpy as np
import pytest

from modules import spoken_evidence
from modules.highlight_report import build_report


def _chapter(number, start, end, shares=None, words=()):
    row = {"number": number, "start": float(start), "end": float(end),
           "duration": float(end - start),
           "timestamp": f"{int(start) // 60}:{int(start) % 60:02d}",
           "title": f"Chapter {number}"}
    if shares:
        row["class_shares"] = dict(shares)
    if words:
        row["speech_words"] = [{"word": w, "count": 3, "times": 9.0}
                               for w in words]
    return row


def _said(*lines):
    return [{"start": float(at), "timestamp": "0:00:00", "text": text}
            for at, text in lines]


class TestMentions:
    def test_a_line_naming_a_class_in_full_is_matched(self):
        chapter = _chapter(1, 0, 60)
        chapter["dialogue"] = _said((10, "the bench vice held it steady"))
        found = spoken_evidence.mentions(chapter, ["bench_vice", "lathe"])
        assert [row["name"] for row in found] == ["bench_vice"]
        assert found[0]["quote"] == "the bench vice held it steady"

    def test_a_partly_named_class_is_not_matched(self):
        # Half a name is a coincidence of vocabulary, and a row built on one
        # sends the reader to check something nobody claimed.
        chapter = _chapter(1, 0, 60)
        chapter["dialogue"] = _said((10, "put it on the bench"))
        assert spoken_evidence.mentions(chapter, ["bench_vice"]) == []

    def test_a_one_word_class_needs_the_word_to_be_distinctive(self):
        plain = _chapter(1, 0, 60)
        plain["dialogue"] = _said((10, "hand me the mallet"))
        assert spoken_evidence.mentions(plain, ["mallet"]) == []

        marked = _chapter(1, 0, 60, words=["mallet"])
        marked["dialogue"] = _said((10, "hand me the mallet"))
        assert [r["name"] for r in spoken_evidence.mentions(marked, ["mallet"])] \
            == ["mallet"]

    def test_the_earliest_line_naming_a_class_is_the_one_kept(self):
        chapter = _chapter(1, 0, 60)
        chapter["dialogue"] = _said((10, "the bench vice is loose"),
                                    (40, "the bench vice again"))
        found = spoken_evidence.mentions(chapter, ["bench_vice"])
        assert len(found) == 1 and found[0]["said_at"] == 10.0

    def test_more_specific_names_rank_first(self):
        chapter = _chapter(1, 0, 60, words=["vice"])
        chapter["dialogue"] = _said((10, "the bench vice sits on the bench"))
        found = spoken_evidence.mentions(chapter, ["vice", "bench_vice"])
        assert [row["name"] for row in found] == ["bench_vice", "vice"]

    def test_quotes_stand_in_when_no_full_dialogue_was_kept(self):
        chapter = _chapter(1, 0, 60)
        chapter["quotes"] = _said((10, "the bench vice held it steady"))
        assert spoken_evidence.mentions(chapter, ["bench_vice"])

    def test_a_silent_chapter_names_nothing(self):
        assert spoken_evidence.mentions(_chapter(1, 0, 60), ["bench_vice"]) == []


class TestMeasure:
    def _labels(self):
        # bench_vice lives at 100-139 and nowhere else; a lathe runs alongside
        # it for part of that, so there is something to be "with".
        labels = {sec: ["bench_vice"] for sec in range(100, 140)}
        for sec in range(120, 130):
            labels[sec] = ["bench_vice", "lathe"]
        for sec in range(300, 320):
            labels[sec] = ["lathe"]
        return labels

    def test_it_reports_seconds_and_share_of_what_was_detected(self):
        out = spoken_evidence.measure(
            "bench_vice", seconds=sorted(self._labels()), detected_seconds=100)
        assert out["seconds"] == 60

    def test_a_class_found_in_seconds_still_produces_a_row(self):
        # The whole point beside a sentence calling it the best part of the
        # video. Suppressing it here would suppress the interesting case.
        out = spoken_evidence.measure("bench_vice", seconds=[10, 11, 12],
                                      detected_seconds=600)
        assert out["seconds"] == 3
        assert out["video_share_pct"] == pytest.approx(0.5)

    def test_the_densest_chapter_is_read_off_the_class_shares(self):
        chapters = [_chapter(1, 0, 200, shares={"bench_vice": 12.0}),
                    _chapter(2, 200, 400, shares={"bench_vice": 71.0})]
        out = spoken_evidence.measure("bench_vice", seconds=[100, 250],
                                      chapters=chapters)
        assert out["densest_chapter"]["number"] == 2
        assert out["densest_chapter"]["share_pct"] == pytest.approx(71.0)

    def test_the_loudest_second_carries_what_else_was_labelled_there(self):
        labels = self._labels()
        levels = [-30.0] * 400
        levels[125] = -6.0                       # inside the overlap
        out = spoken_evidence.measure(
            "bench_vice", seconds=sorted(s for s, n in labels.items()
                                         if "bench_vice" in n),
            levels=levels, labels_by_second=labels, video_median=-30.0)
        loudest = out["level"]["loudest"]
        assert loudest["second"] == 125
        assert loudest["vs_video_db"] == pytest.approx(24.0)
        assert loudest["with"] == ["lathe"]

    def test_the_class_itself_is_not_listed_as_alongside_itself(self):
        labels = {5: ["bench_vice"]}
        out = spoken_evidence.measure("bench_vice", seconds=[5],
                                      levels=[-20.0] * 10,
                                      labels_by_second=labels)
        assert "with" not in out["level"]["loudest"]

    def test_kept_clips_overlapping_the_class_are_named(self):
        out = spoken_evidence.measure("bench_vice", seconds=[100, 101, 102],
                                      segments=[(0, 30), (95, 125), (300, 330)])
        assert out["clips"] == [2]

    def test_no_levels_means_no_level_section_rather_than_a_guess(self):
        out = spoken_evidence.measure("bench_vice", seconds=[5, 6])
        assert "level" not in out


class TestAttach:
    def _chapters(self):
        first = _chapter(1, 0, 300, shares={"bench_vice": 64.0})
        last = _chapter(2, 300, 600, shares={"lathe": 80.0})
        last["dialogue"] = _said(
            (500, "the best bit was the bench vice, honestly"))
        return [first, last]

    def _attached(self, **kw):
        labels = {sec: ["bench_vice"] for sec in range(100, 200)}
        labels.update({sec: ["lathe"] for sec in range(300, 500)})
        args = {"labels_by_second": labels, "segments": [(120, 150)],
                "levels": [-30.0] * 600, "detected_seconds": 300}
        args.update(kw)
        return spoken_evidence.attach(self._chapters(),
                                      ["bench_vice", "lathe"], **args)

    def test_the_row_is_filed_where_the_claim_was_made(self):
        rows = self._attached()
        assert "spoken_evidence" not in rows[0]
        assert [r["name"] for r in rows[1]["spoken_evidence"]] == ["bench_vice"]

    def test_the_evidence_points_at_the_stretch_the_class_is_really_in(self):
        # Said in chapter 2, measured in chapter 1. A row that only looked at
        # the chapter it was filed under would report 0% and say nothing.
        row = self._attached()[1]["spoken_evidence"][0]
        assert row["densest_chapter"]["number"] == 1
        assert row["here_share_pct"] == pytest.approx(0.0)
        assert row["seconds"] == 100

    def test_the_input_chapters_are_not_mutated(self):
        chapters = self._chapters()
        spoken_evidence.attach(chapters, ["bench_vice"],
                               labels_by_second={5: ["bench_vice"]})
        assert all("spoken_evidence" not in ch for ch in chapters)

    def test_no_names_means_the_chapters_come_back_untouched(self):
        rows = spoken_evidence.attach(self._chapters(), [])
        assert all("spoken_evidence" not in ch for ch in rows)

    def test_a_class_mapping_to_boxes_reads_the_same_as_one_mapping_to_names(self):
        # The report passes both shapes around; neither caller should have to
        # normalise, and a silent mismatch would empty every row.
        boxes = {sec: {"bench_vice": (0.4, 0.9)} for sec in range(100, 200)}
        rows = self._attached(labels_by_second=boxes)
        assert rows[1]["spoken_evidence"][0]["seconds"] == 100


class TestThroughTheReport:
    """The wiring: a real report, a transcript, and a class named in it."""

    def _report(self):
        from modules.chapters import chapterize    # noqa: F401  (import guard)

        score = np.zeros(600)
        score[130] = 10.0
        transcript = [{"start": float(t), "end": float(t) + 4.0,
                       "text": "we set it in the bench vice and turned it"}
                      for t in range(0, 290, 10)]
        transcript += [{"start": float(t), "end": float(t) + 4.0,
                        "text": "the bench vice part was the best bit"}
                       for t in range(300, 590, 10)]
        labels = {sec: ["bench_vice"] for sec in range(100, 200)}
        return build_report(
            video_path="a.mp4", video_duration=600, score=score,
            signals={"object": score}, segments=[(120, 150)],
            object_detections=labels,
            loudness_levels=[-30.0] * 600,
            chapters=[_chapter(1, 0, 300), _chapter(2, 300, 600)],
            transcript=transcript)

    def test_the_record_carries_the_evidence(self):
        report = self._report()
        rows = [r for ch in report["chapters"]
                for r in (ch.get("spoken_evidence") or [])]
        assert rows and rows[0]["name"] == "bench_vice"
        assert rows[0]["seconds"] == 100

    def test_the_figures_are_printed_beside_the_paragraph_not_only_prompted(self):
        from modules.highlight_prose import describe_chapter

        report = self._report()
        told = [line for ch in report["chapters"]
                for line in describe_chapter(ch)
                if "bench_vice" in line]
        assert told, "the measured evidence never reached the page"
        assert any("Where bench_vice actually is" in line for line in told)

    def test_the_narrator_is_asked_to_weigh_it(self):
        from modules import chapter_story

        report = self._report()
        prompts = [chapter_story.chapter_prompt(report, ch)
                   for ch in report["chapters"]]
        named = [p for p in prompts if "this run measured for itself" in p]
        assert named
        assert "bears out what was said" in named[0]
        # And the caution that a shared word is not a shared meaning.
        assert "nothing more" in named[0]

    def test_a_run_without_a_transcript_is_the_report_it_always_was(self):
        score = np.zeros(600)
        report = build_report(
            video_path="a.mp4", video_duration=600, score=score,
            signals={"object": score}, segments=[(120, 150)],
            object_detections={sec: ["bench_vice"] for sec in range(100, 200)},
            chapters=[_chapter(1, 0, 300), _chapter(2, 300, 600)])
        assert all("spoken_evidence" not in ch for ch in report["chapters"])
