"""Tests for corroborating what a stretch says against what the run measured.

The feature exists so a narration can *disagree* with a speaker, so most of
what is worth testing is the ways it could quietly stop being able to:

* the evidence is video-wide, not chapter-local — somebody describing in the
  last chapter something that happened in the middle is the ordinary case, and
  a row that only looked here would report it as never having happened;
* a class the detector barely found still produces a row, because "labelled for
  under eight seconds" beside a sentence calling it the best part of the file
  is the finding, not a failure;
* matching is loose enough to fire on speech and tight enough not to fire on
  everything — both failures are silent, and the first one shipped;
* the figures reach the page as well as the prompt, so the paragraph the model
  writes from them can be checked without trusting it;
* a report whose analysis ran hours ago can still gain the section, because
  that is how the narration is actually used.

Fixture classes are named after workshop objects. The matching path does not
care what a class is called, and the repo carries no vocabulary of its own.
"""
from __future__ import annotations

import json

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
        assert [row["said"] for row in found] == ["bench_vice"]
        assert found[0]["kind"] == "name"
        assert found[0]["classes"] == ["bench_vice"]
        assert found[0]["quote"] == "the bench vice held it steady"

    def test_a_shortened_name_still_matches_through_its_distinctive_word(self):
        # The failure that shipped. People abbreviate: a transcript says a
        # contracted form of a two-word class name, so requiring the whole name
        # matches the one speaker who said the label the way it was typed.
        chapter = _chapter(1, 0, 60, words=["bench"])
        chapter["dialogue"] = _said((10, "we did the bench thing after lunch"))
        found = spoken_evidence.mentions(chapter, ["bench_vice"])
        assert [row["said"] for row in found] == ["bench"]
        assert found[0]["kind"] == "word"
        assert found[0]["classes"] == ["bench_vice"]

    def test_one_word_naming_several_classes_gives_one_row_listing_them(self):
        # Which one the speaker meant is not decidable here, and the comparison
        # between them is usually the answer anyway.
        chapter = _chapter(1, 0, 60, words=["bench"])
        chapter["dialogue"] = _said((10, "the bench again"))
        found = spoken_evidence.mentions(
            chapter, ["bench_vice", "bench_saw", "lathe"])
        assert len(found) == 1
        assert found[0]["classes"] == ["bench_saw", "bench_vice"]

    def test_a_one_word_class_needs_the_word_to_be_distinctive(self):
        # Without this the same row appeared in six chapters of sixteen on real
        # footage — no keyness, no quote worth reading, and enough of them to
        # teach a reader to skip the section.
        plain = _chapter(1, 0, 60)
        plain["dialogue"] = _said((10, "hand me the mallet"))
        assert spoken_evidence.mentions(plain, ["mallet"]) == []

        marked = _chapter(1, 0, 60, words=["mallet"])
        marked["dialogue"] = _said((10, "hand me the mallet"))
        assert [r["said"] for r in spoken_evidence.mentions(marked, ["mallet"])] \
            == ["mallet"]

    def test_a_class_named_in_full_is_not_repeated_under_a_word_it_shares(self):
        chapter = _chapter(1, 0, 60, words=["bench"])
        chapter["dialogue"] = _said((10, "the bench vice held it steady"))
        found = spoken_evidence.mentions(chapter, ["bench_vice"])
        assert len(found) == 1 and found[0]["kind"] == "name"

    def test_a_full_name_ranks_above_a_word(self):
        chapter = _chapter(1, 0, 60, words=["bench"])
        chapter["dialogue"] = _said((10, "the bench saw and the bench vice"))
        found = spoken_evidence.mentions(chapter, ["bench_vice", "bench_saw"])
        assert [r["kind"] for r in found] == ["name", "name"]

    def test_the_earliest_line_naming_a_class_is_the_one_kept(self):
        chapter = _chapter(1, 0, 60)
        chapter["dialogue"] = _said((10, "the bench vice is loose"),
                                    (40, "the bench vice again"))
        found = spoken_evidence.mentions(chapter, ["bench_vice"])
        assert len(found) == 1 and found[0]["said_at"] == 10.0

    def test_a_distinctive_word_never_actually_said_produces_no_row(self):
        # speech_words comes from the same transcript, but a row with no line
        # to quote is one nobody can check, so it is dropped rather than shown.
        chapter = _chapter(1, 0, 60, words=["bench"])
        chapter["dialogue"] = _said((10, "nothing relevant here"))
        assert spoken_evidence.mentions(chapter, ["bench_vice"]) == []

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
        assert loudest["scope"] == "second"
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

    def test_no_overlapping_clip_is_an_empty_list_not_a_missing_key(self):
        # "Nothing was cut from it" is the half of the answer a reader is most
        # likely to be checking, so it has to be sayable.
        out = spoken_evidence.measure("bench_vice", seconds=[100],
                                      segments=[(0, 30)])
        assert out["clips"] == []

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

    def _only(self, rows):
        return rows[1]["spoken_evidence"][0]["classes"][0]

    def test_the_row_is_filed_where_the_claim_was_made(self):
        rows = self._attached()
        assert "spoken_evidence" not in rows[0]
        assert self._only(rows)["name"] == "bench_vice"

    def test_the_evidence_points_at_the_stretch_the_class_is_really_in(self):
        # Said in chapter 2, measured in chapter 1. A row that only looked at
        # the chapter it was filed under would report 0% and say nothing.
        entry = self._only(self._attached())
        assert entry["densest_chapter"]["number"] == 1
        assert entry["here_share_pct"] == pytest.approx(0.0)
        assert entry["seconds"] == 100

    def test_candidates_are_ranked_by_how_much_of_each_was_found(self):
        chapters = self._chapters()
        chapters[1]["speech_words"] = [{"word": "bench", "count": 3,
                                        "times": 9.0}]
        chapters[1]["dialogue"] = _said((500, "the bench, mostly"))
        labels = {sec: ["bench_vice"] for sec in range(100, 200)}
        labels.update({sec: ["bench_saw"] for sec in range(200, 210)})
        rows = spoken_evidence.attach(chapters, ["bench_vice", "bench_saw"],
                                      labels_by_second=labels)
        found = rows[1]["spoken_evidence"][0]["classes"]
        assert [c["name"] for c in found] == ["bench_vice", "bench_saw"]

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
        assert self._only(rows)["seconds"] == 100


class TestFromRecord:
    """The backfill: a saved report, no caches, no video, no re-run."""

    def _record(self):
        return {
            "vocabulary": {"classes": ["lathe"], "events": ["bench_vice"]},
            "level_by_class": {
                "classes": [{"name": "lathe", "seconds": 300,
                             "median_dbfs": -22.0}],
                "loudest": {"second": 410, "timestamp": "6:50",
                            "level_dbfs": -8.0, "classes": ["lathe"]},
            },
            "segments": [
                {"index": 1, "objects": ["lathe"], "events": [],
                 "loudest": {"second": 410, "timestamp": "6:50",
                             "level_dbfs": -8.0, "classes": ["lathe"],
                             "vs_video_db": 12.0}},
                {"index": 2, "objects": [], "events": ["bench_vice"],
                 "loudest": {"second": 150, "timestamp": "2:30",
                             "level_dbfs": -19.0, "classes": ["bench_vice"],
                             "vs_video_db": 1.5}},
            ],
            "chapters": [
                _chapter(1, 0, 300, shares={"bench_vice": 5.0, "lathe": 60.0}),
                dict(_chapter(2, 300, 600, shares={"lathe": 80.0}, words=["best"]),
                     dialogue=_said((500, "the bench vice bit was the best"))),
            ],
        }

    def _row(self, record=None):
        chapters = spoken_evidence.from_report(record or self._record())
        return chapters[1]["spoken_evidence"][0]

    def test_the_section_appears_without_re_running_anything(self):
        assert self._row()["classes"][0]["name"] == "bench_vice"

    def test_a_class_under_the_level_summary_bar_reports_the_bound(self):
        # It has chapter shares, so it was labelled; it is missing from the
        # level summary, so it ran for less than that module's minimum. Said as
        # the bound it is rather than as a count nothing measured.
        from modules.level_by_class import MIN_SECONDS

        entry = self._row()["classes"][0]
        assert entry["under_seconds"] == MIN_SECONDS
        assert "seconds" not in entry

    def test_a_described_class_carries_its_counted_seconds(self):
        record = self._record()
        record["chapters"][1]["dialogue"] = _said((500, "the lathe was best"))
        record["chapters"][1]["speech_words"] = [{"word": "lathe", "count": 3,
                                                  "times": 9.0}]
        entry = self._row(record)["classes"][0]
        assert entry["name"] == "lathe" and entry["seconds"] == 300

    def test_the_loudest_second_is_marked_as_the_narrower_thing_it_is(self):
        # Per-second levels do not survive in the record, so what is available
        # is the loudest second of a kept clip. Reporting it as the class's own
        # loudest second would overstate what the file can support.
        entry = self._row()["classes"][0]
        assert entry["level"]["loudest"]["scope"] == "clip"
        assert entry["level"]["loudest"]["timestamp"] == "2:30"

    def test_the_video_wide_loudest_second_outranks_a_clip_one(self):
        record = self._record()
        record["chapters"][1]["dialogue"] = _said((500, "the lathe was best"))
        record["chapters"][1]["speech_words"] = [{"word": "lathe", "count": 3,
                                                  "times": 9.0}]
        entry = self._row(record)["classes"][0]
        assert entry["level"]["loudest"]["scope"] == "video"

    def test_the_densest_chapter_survives_the_round_trip(self):
        assert self._row()["classes"][0]["densest_chapter"]["number"] == 1

    def test_a_record_with_no_vocabulary_comes_back_untouched(self):
        record = self._record()
        record["vocabulary"] = {}
        assert all("spoken_evidence" not in ch
                   for ch in spoken_evidence.from_report(record))


class TestThroughTheReport:
    """The wiring: a real report, a transcript, and a class named in it."""

    def _report(self):
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
        assert rows
        assert rows[0]["classes"][0]["name"] == "bench_vice"
        assert rows[0]["classes"][0]["seconds"] == 100

    def test_the_figures_are_printed_beside_the_paragraph_not_only_prompted(self):
        from modules.highlight_prose import describe_chapter

        told = [line for ch in self._report()["chapters"]
                for line in describe_chapter(ch)
                if "bench_vice" in line]
        assert told, "the measured evidence never reached the page"
        assert any("labelled in 100s of the video" in line for line in told)

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

    def test_the_prompt_carries_the_evidence_once_not_twice(self):
        # Sending it in the measured facts *and* in its own section put every
        # figure in the prompt twice and tipped a local model into repeating
        # one sentence until it hit the token cap.
        from modules import chapter_story

        report = self._report()
        prompt = [chapter_story.chapter_prompt(report, ch)
                  for ch in report["chapters"]
                  if ch.get("spoken_evidence")][0]
        assert prompt.count("labelled in 100s of the video") == 1

    def test_the_page_still_shows_it_in_the_chapter_block(self):
        # The suppression is for the prompt alone. The figures have to stay
        # printed beside the paragraph, or a reader cannot check it.
        from modules.highlight_prose import describe_chapter

        chapter = [ch for ch in self._report()["chapters"]
                   if ch.get("spoken_evidence")][0]
        assert any("labelled in 100s" in line
                   for line in describe_chapter(chapter))
        assert not any("labelled in 100s" in line for line
                       in describe_chapter(chapter, spoken_evidence=False))

    def test_telling_an_older_report_backfills_it(self, tmp_path):
        # The path the feature is actually used on: analysis ran hours ago,
        # only the narration is being re-run. Without the backfill the section
        # could never appear on the report it most wants to improve.
        from modules import chapter_story

        report = self._report()
        for ch in report["chapters"]:
            ch.pop("spoken_evidence", None)
        path = tmp_path / "a_why.json"
        path.write_text(json.dumps(report), encoding="utf-8")

        class _LLM:
            def generate(self, prompt, **kw):
                self.prompt = prompt
                return "They worked at a bench."

        chapter_story.tell_report_file(str(path), llm=_LLM(), use_frames=False,
                                       log_fn=lambda _m: None)
        written = json.loads(path.read_text(encoding="utf-8"))
        assert any(ch.get("spoken_evidence") for ch in written["chapters"])

    def test_a_run_without_a_transcript_is_the_report_it_always_was(self):
        score = np.zeros(600)
        report = build_report(
            video_path="a.mp4", video_duration=600, score=score,
            signals={"object": score}, segments=[(120, 150)],
            object_detections={sec: ["bench_vice"] for sec in range(100, 200)},
            chapters=[_chapter(1, 0, 300), _chapter(2, 300, 600)])
        assert all("spoken_evidence" not in ch for ch in report["chapters"])
