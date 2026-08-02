"""Tests for the chapter breakdown — shares and rates, not window percentiles.

The arithmetic here is proportions, so every case is built from hand-counted
seconds: a chapter where a class occupies a known number of its detected
seconds, against a video where it occupies another known number. That makes the
expected lift a number you can work out on paper, which is the only way to tell
a broken denominator from a plausible-looking one.

The denominator is the thing most worth pinning down. `_prevalence_lifts`
divides by *detected* seconds on both sides, and
`test_lift_ignores_undetected_seconds` is what stops that quietly reverting to
wall-clock seconds — a change that leaves every test but that one passing while
making a chapter of black frames look like one where every class vanished.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from modules.chapter_compare import (
    CUT_SHARE_LIFT,
    MIN_SECONDS_FOR_LIFT,
    NOTABLE_LIFT,
    assign_chapters,
    distinctive,
    summarise_chapters,
)


def _chapter(number, start, end, shots=20):
    return {
        "number": number, "start": float(start), "end": float(end),
        "duration": float(end - start), "timestamp": "0:00:00",
        "title": f"Chapter {number}", "shots": shots, "pace": "steady",
        "boundary_score": 0.0, "method": "visual",
    }


def _distributions(largest=None, expressions=None):
    largest = largest or {}
    seconds_with: dict = {}
    for best in largest.values():
        for name in best:
            seconds_with[name] = seconds_with.get(name, 0) + 1
    return {
        "largest": largest,
        "seconds_with": seconds_with,
        "detected_seconds": len(largest),
        "expressions": expressions or {},
        "span": 600,
    }


# ---------------------------------------------------------------------------
# assign_chapters
# ---------------------------------------------------------------------------
class TestAssignChapters:
    def test_files_each_clip_by_its_start(self):
        chapters = [_chapter(1, 0, 100), _chapter(2, 100, 200)]
        segments = [(10.0, 20.0), (150.0, 160.0)]
        assert assign_chapters(chapters, segments) == [1, 2]

    def test_a_straddling_clip_goes_wholly_to_where_it_began(self):
        """Splitting it would double-count it in every share below."""
        chapters = [_chapter(1, 0, 100), _chapter(2, 100, 200)]
        assert assign_chapters(chapters, [(95.0, 130.0)]) == [1]

    def test_a_clip_on_the_final_edge_still_lands(self):
        chapters = [_chapter(1, 0, 100), _chapter(2, 100, 200)]
        assert assign_chapters(chapters, [(200.0, 205.0)]) == [2]

    def test_every_clip_is_placed_exactly_once(self):
        chapters = [_chapter(i, (i - 1) * 100, i * 100) for i in range(1, 6)]
        segments = [(float(t), float(t + 8)) for t in range(0, 490, 37)]
        placed = assign_chapters(chapters, segments)
        assert all(p != 0 for p in placed)
        assert len(placed) == len(segments)


# ---------------------------------------------------------------------------
# share of the cut
# ---------------------------------------------------------------------------
class TestCutShare:
    def test_lift_is_cut_share_over_runtime_share(self):
        # Chapter 1 is 25% of the runtime and supplies 100% of the cut.
        chapters = [_chapter(1, 0, 100), _chapter(2, 100, 400)]
        segments = [(10.0, 20.0), (30.0, 40.0)]

        rows = summarise_chapters(chapters, segments=segments, video_duration=400.0)

        assert rows[0]["runtime_share_pct"] == pytest.approx(25.0)
        assert rows[0]["cut_share_pct"] == pytest.approx(100.0)
        assert rows[0]["cut_share_lift"] == pytest.approx(4.0)
        assert rows[1]["clips"] == 0
        assert rows[1]["cut_share_lift"] == 0.0

    def test_even_contribution_is_parity(self):
        chapters = [_chapter(1, 0, 200), _chapter(2, 200, 400)]
        segments = [(10.0, 20.0), (210.0, 220.0)]

        rows = summarise_chapters(chapters, segments=segments, video_duration=400.0)
        assert rows[0]["cut_share_lift"] == pytest.approx(1.0)
        assert rows[1]["cut_share_lift"] == pytest.approx(1.0)

    def test_clip_indices_are_one_based_like_the_report(self):
        chapters = [_chapter(1, 0, 100), _chapter(2, 100, 200)]
        segments = [(10.0, 20.0), (110.0, 120.0), (30.0, 40.0)]

        rows = summarise_chapters(chapters, segments=segments, video_duration=200.0)
        assert rows[0]["clip_indices"] == [1, 3]
        assert rows[1]["clip_indices"] == [2]

    def test_the_input_is_not_mutated(self):
        """The timeline holds these objects; a renderer must not grow them."""
        chapters = [_chapter(1, 0, 100)]
        summarise_chapters(chapters, segments=[(1.0, 2.0)], video_duration=100.0)
        assert "cut_share_pct" not in chapters[0]


# ---------------------------------------------------------------------------
# prevalence
# ---------------------------------------------------------------------------
class TestPrevalenceLift:
    def test_a_class_concentrated_in_one_chapter_lifts(self):
        # 200 detected seconds; class "a" in all 100 of chapter 1 and in none
        # of chapter 2. Video share 50%, chapter-1 share 100% -> lift 2.0.
        largest = {}
        for sec in range(100):
            largest[sec] = {"a": (0.2, 0.9)}
        for sec in range(100, 200):
            largest[sec] = {"b": (0.2, 0.9)}

        rows = summarise_chapters(
            [_chapter(1, 0, 100), _chapter(2, 100, 200)],
            distributions=_distributions(largest), video_duration=200.0)

        first = {s["name"]: s for s in rows[0]["subjects"]}
        assert first["a"]["lift"] == pytest.approx(2.0)
        assert first["a"]["chapter_share_pct"] == pytest.approx(100.0)
        assert first["a"]["video_share_pct"] == pytest.approx(50.0)
        assert first["a"]["enough_samples"] is True

    def test_lift_ignores_undetected_seconds(self):
        """A chapter of blank frames must not depress every class in it.

        Chapter 2 has detections in only 10 of its 100 seconds, and every one of
        them carries "a". Its share of *detected* seconds is 100%, and that is
        the honest figure; dividing by wall-clock would report 10%.
        """
        largest = {sec: {"a": (0.2, 0.9)} for sec in range(100)}
        largest.update({sec: {"a": (0.2, 0.9)} for sec in range(100, 110)})

        rows = summarise_chapters(
            [_chapter(1, 0, 100), _chapter(2, 100, 200)],
            distributions=_distributions(largest), video_duration=200.0)

        second = {s["name"]: s for s in rows[1]["subjects"]}
        assert second["a"]["chapter_share_pct"] == pytest.approx(100.0)
        assert second["a"]["lift"] == pytest.approx(1.0)
        assert second["a"]["seconds"] == 10

    def test_a_thin_sample_is_flagged_not_hidden(self):
        largest = {sec: {"a": (0.2, 0.9)} for sec in range(200)}
        largest[0] = {"a": (0.2, 0.9), "rare": (0.1, 0.5)}

        rows = summarise_chapters(
            [_chapter(1, 0, 100), _chapter(2, 100, 200)],
            distributions=_distributions(largest), video_duration=200.0)

        rare = [s for s in rows[0]["subjects"] if s["name"] == "rare"]
        assert rare, "the thin finding was dropped instead of flagged"
        assert rare[0]["seconds"] < MIN_SECONDS_FOR_LIFT
        assert rare[0]["enough_samples"] is False

    def test_findings_are_ordered_by_distance_from_parity(self):
        largest = {}
        for sec in range(100):                       # chapter 1
            largest[sec] = {"steady": (0.2, 0.9), "spike": (0.2, 0.9)}
        for sec in range(100, 200):                  # chapter 2
            largest[sec] = {"steady": (0.2, 0.9)}

        rows = summarise_chapters(
            [_chapter(1, 0, 100), _chapter(2, 100, 200)],
            distributions=_distributions(largest), video_duration=200.0)

        assert rows[0]["subjects"][0]["name"] == "spike"

    def test_a_vanished_class_is_as_describing_as_a_dominant_one(self):
        largest = {}
        for sec in range(150):
            largest[sec] = {"a": (0.2, 0.9)}
        for sec in range(150, 200):
            largest[sec] = {"b": (0.2, 0.9)}

        rows = summarise_chapters(
            [_chapter(1, 0, 100), _chapter(2, 100, 200)],
            distributions=_distributions(largest), video_duration=200.0)

        names = [s["name"] for s in rows[1]["subjects"]]
        assert "b" in names
        lifts = {s["name"]: s["lift"] for s in rows[1]["subjects"]}
        assert lifts["a"] < 1.0 and lifts["b"] > 1.0


# ---------------------------------------------------------------------------
# expression, pace, level
# ---------------------------------------------------------------------------
class TestOtherSignals:
    def test_expression_mix_lifts(self):
        expressions = {sec: ("x", 0.8) for sec in range(100)}
        expressions.update({sec: ("y", 0.8) for sec in range(100, 200)})

        rows = summarise_chapters(
            [_chapter(1, 0, 100), _chapter(2, 100, 200)],
            distributions=_distributions(expressions=expressions),
            video_duration=200.0)

        first = {e["label"]: e for e in rows[0]["expression"]}
        assert first["x"]["lift"] == pytest.approx(2.0)

    def test_pace_lift_against_the_video_average(self):
        # 40 shots in 100 s vs 10 shots in 100 s -> video mean 25/100 s.
        rows = summarise_chapters(
            [_chapter(1, 0, 100, shots=40), _chapter(2, 100, 200, shots=10)],
            video_duration=200.0)

        assert rows[0]["shots_per_minute"] == pytest.approx(24.0)
        assert rows[0]["pace_lift"] == pytest.approx(1.6, abs=0.05)
        assert rows[1]["pace_lift"] == pytest.approx(0.4, abs=0.05)

    def test_loudness_delta_is_in_db(self):
        # Chapter 2 is twice the amplitude of chapter 1: ~+6 dB against a
        # median that sits between them.
        amps = [0.1] * 100 + [0.2] * 100
        rows = summarise_chapters(
            [_chapter(1, 0, 100), _chapter(2, 100, 200)],
            amps=amps, amps_per_second=1.0, video_duration=200.0)

        assert rows[0]["loudness_delta_db"] < 0
        assert rows[1]["loudness_delta_db"] > 0
        assert (rows[1]["loudness_delta_db"]
                - rows[0]["loudness_delta_db"]) == pytest.approx(6.0, abs=0.5)

    def test_score_lift_marks_where_the_points_were(self):
        score = np.zeros(200, dtype=float)
        score[:100] = 10.0
        score[100:] = 2.0

        rows = summarise_chapters(
            [_chapter(1, 0, 100), _chapter(2, 100, 200)],
            score=score, video_duration=200.0)

        assert rows[0]["score_lift"] > 1.0 > rows[1]["score_lift"]


# ---------------------------------------------------------------------------
# distinctive
# ---------------------------------------------------------------------------
class TestDistinctive:
    def test_drops_findings_that_are_near_parity(self):
        chapter = {"subjects": [
            {"name": "flat", "lift": 1.05, "seconds": 50, "enough_samples": True},
        ]}
        assert distinctive(chapter) == []

    def test_drops_findings_without_the_samples(self):
        chapter = {"subjects": [
            {"name": "thin", "lift": 9.0, "seconds": 2, "enough_samples": False},
        ]}
        assert distinctive(chapter) == []

    def test_keeps_both_directions(self):
        chapter = {"subjects": [
            {"name": "up", "lift": 3.0, "seconds": 50, "enough_samples": True},
            {"name": "down", "lift": 0.2, "seconds": 50, "enough_samples": True},
        ]}
        names = [f["name"] for f in distinctive(chapter)]
        assert set(names) == {"up", "down"}

    def test_tags_what_each_finding_describes(self):
        chapter = {
            "subjects": [{"name": "a", "lift": 3.0, "seconds": 50,
                          "enough_samples": True}],
            "expression": [{"label": "x", "lift": 0.2, "seconds": 50,
                            "enough_samples": True}],
        }
        kinds = {f["kind"]: f["name"] for f in distinctive(chapter)}
        assert kinds == {"subject": "a", "expression": "x"}


# ---------------------------------------------------------------------------
# shape
# ---------------------------------------------------------------------------
class TestShape:
    def test_empty_input_is_empty_output(self):
        assert summarise_chapters([]) == []

    def test_works_with_no_detectors_at_all(self):
        """The cut-share arithmetic must survive a run with nothing detected."""
        rows = summarise_chapters(
            [_chapter(1, 0, 100), _chapter(2, 100, 200)],
            segments=[(10.0, 20.0)], video_duration=200.0)
        assert rows[0]["clips"] == 1
        assert "subjects" not in rows[0]

    def test_result_is_json_serialisable(self):
        import json

        largest = {sec: {"a": (0.2, 0.9)} for sec in range(200)}
        rows = summarise_chapters(
            [_chapter(1, 0, 100), _chapter(2, 100, 200)],
            score=np.ones(200), segments=[(1.0, 9.0)],
            amps=[0.1] * 200, amps_per_second=1.0,
            distributions=_distributions(largest), video_duration=200.0)
        json.dumps(rows)
