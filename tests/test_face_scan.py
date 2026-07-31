"""Tests for `modules.face_scan` — one sweep, reused by everything downstream.

Decoding, detection and classification are injected, so none of this needs cv2,
a model or a video. The behaviour worth holding still is what an *absent* second
means: no readable face is not the same as a face showing nothing, and the
difference decides whether a highlight scores silence.
"""

from __future__ import annotations

import json

import numpy as np

from modules.face_emotions import EMOTION_LABELS
from modules.face_scan import (
    best_by_second,
    cache_path_for,
    label_counts,
    load,
    moments_for,
    save,
    scan,
)


def _frame(h=200, w=300):
    return np.full((h, w, 3), 90, dtype=np.uint8)


def _probs(label, value=0.9):
    row = np.zeros(len(EMOTION_LABELS), dtype=np.float32)
    row[EMOTION_LABELS.index(label)] = value
    return row


def _detect(n=1, score=0.9):
    def detect(frame):
        return [{"bbox": (10 + i * 70, 10, 70 + i * 70, 70), "det_score": score}
                for i in range(n)]
    return detect


def _classify(label="happy", value=0.9):
    def classify(crops):
        return np.stack([_probs(label, value) for _ in crops])
    return classify


class TestScan:
    def test_a_readable_expression_is_recorded_per_second(self):
        frames = [(float(t), _frame()) for t in (0, 1, 2)]
        seconds = scan(frames, detect_fn=_detect(), classify_fn=_classify("sad"))
        assert set(seconds) == {0, 1, 2}
        assert seconds[1]["label"] == "sad"
        assert seconds[1]["faces"] == 1

    def test_a_second_with_no_face_is_absent_not_neutral(self):
        """An absent second means 'no face', which must not score."""
        frames = [(0.0, _frame()), (1.0, _frame())]

        def detect(frame):
            detect.calls = getattr(detect, "calls", 0) + 1
            return _detect()(frame) if detect.calls == 1 else []

        seconds = scan(frames, detect_fn=detect, classify_fn=_classify())
        assert set(seconds) == {0}

    def test_a_face_below_the_confidence_floor_is_absent(self):
        frames = [(0.0, _frame())]
        seconds = scan(frames, detect_fn=_detect(),
                       classify_fn=_classify("happy", 0.2), min_confidence=0.5)
        assert seconds == {}

    def test_weak_detections_never_reach_the_classifier(self):
        seen = []

        def classify(crops):
            seen.append(len(crops))
            return np.stack([_probs("happy") for _ in crops])

        scan([(0.0, _frame())], detect_fn=_detect(score=0.1),
             classify_fn=classify)
        assert seen == [], "a guessed box must not teach the classifier anything"

    def test_the_clearest_face_in_a_frame_wins(self):
        frames = [(0.0, _frame(w=600))]

        def classify(crops):
            return np.stack([_probs("neutral", 0.6), _probs("anger", 0.95)])

        seconds = scan(frames, detect_fn=_detect(n=2), classify_fn=classify)
        assert seconds[0]["label"] == "anger"
        assert seconds[0]["faces"] == 2

    def test_a_crowd_is_capped(self):
        frames = [(0.0, _frame(w=2000))]
        seconds = scan(frames, detect_fn=_detect(n=9),
                       classify_fn=_classify(), max_faces_per_frame=3)
        assert seconds[0]["faces"] == 3

    def test_fractional_timestamps_land_on_whole_seconds(self):
        frames = [(4.7, _frame())]
        assert set(scan(frames, detect_fn=_detect(),
                        classify_fn=_classify())) == {4}


class TestQueries:
    def _seconds(self):
        return {
            10: {"label": "happy", "confidence": 0.95, "faces": 1},
            20: {"label": "happy", "confidence": 0.60, "faces": 1},
            30: {"label": "sad", "confidence": 0.80, "faces": 2},
        }

    def test_moments_for_a_label_come_back_strongest_first(self):
        assert moments_for(self._seconds(), "happy") == [(10, 0.95), (20, 0.60)]

    def test_a_confidence_floor_filters_them(self):
        assert moments_for(self._seconds(), "happy", min_confidence=0.9) == [(10, 0.95)]

    def test_labels_are_matched_case_insensitively(self):
        assert len(moments_for(self._seconds(), "HAPPY")) == 2

    def test_an_expression_that_never_appeared_is_empty_not_an_error(self):
        assert moments_for(self._seconds(), "surprise") == []

    def test_counts_cover_every_class_including_the_absent_ones(self):
        counts = label_counts(self._seconds())
        assert counts["happy"] == 2 and counts["sad"] == 1
        assert counts["anger"] == 0
        assert set(counts) == set(EMOTION_LABELS)

    def test_best_by_second_is_the_shape_the_signal_wants(self):
        assert best_by_second(self._seconds())[10] == ("happy", 0.95)

    def test_queries_on_an_empty_scan_do_not_raise(self):
        assert moments_for({}, "happy") == []
        assert best_by_second({}) == {}
        assert label_counts({})["happy"] == 0


class TestCache:
    def test_path_is_derived_from_the_video(self, tmp_path):
        path = cache_path_for("D:/clips/holiday.mp4", str(tmp_path))
        assert path.endswith("holiday_faces.json")

    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "s.json")
        seconds = {5: {"label": "happy", "confidence": 0.9, "faces": 1}}
        assert save(seconds, path, video_path="a.mp4") is True
        assert load(path) == seconds, "seconds must come back as ints"

    def test_loading_what_was_never_written_is_none(self, tmp_path):
        assert load(str(tmp_path / "absent.json")) is None

    def test_a_corrupt_scan_does_not_take_the_run_down(self, tmp_path):
        path = tmp_path / "s.json"
        path.write_text("{ truncated", encoding="utf-8")
        assert load(str(path)) is None

    def test_saving_leaves_no_temporary_behind(self, tmp_path):
        save({1: {"label": "sad", "confidence": 0.7, "faces": 1}},
             str(tmp_path / "s.json"))
        assert [p.name for p in tmp_path.iterdir()] == ["s.json"]

    def test_the_file_records_what_produced_it(self, tmp_path):
        path = tmp_path / "s.json"
        save({1: {"label": "sad", "confidence": 0.7, "faces": 1}}, str(path),
             video_path="a.mp4", interval=2.0)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["video"] == "a.mp4" and data["interval"] == 2.0
        assert data["schema"] == 1
