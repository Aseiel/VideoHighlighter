"""Tests for `modules.face_examples` — face categories taught from crops.

No cv2, no model, no video: detection and embedding are injected, which is the
reason those are callables. Frames are plain arrays and "embeddings" are chosen
so that similarity is something a reader can check by eye.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from modules.face_examples import (
    MIN_CROP_PIXELS,
    FaceCategory,
    FaceCategoryStore,
    FaceCrop,
    best_per_second,
    build_category,
    crop_face,
    l2_normalise,
    scan_frames,
    score_crops,
    to_signal,
)


def _frame(h=200, w=300, value=7):
    return np.full((h, w, 3), value, dtype=np.uint8)


def _vec(*values):
    return l2_normalise(np.asarray(values, dtype=np.float32))


class TestCropFace:
    def test_crops_the_box_with_padding(self):
        crop = crop_face(_frame(), (100, 50, 160, 110), pad=0.0)
        assert crop.shape[:2] == (60, 60)

    def test_padding_grows_the_region(self):
        tight = crop_face(_frame(), (100, 50, 160, 110), pad=0.0)
        padded = crop_face(_frame(), (100, 50, 160, 110), pad=0.5)
        assert padded.shape[0] > tight.shape[0]

    def test_padding_is_clamped_to_the_frame(self):
        crop = crop_face(_frame(h=100, w=100), (0, 0, 40, 40), pad=1.0)
        assert crop.shape[0] <= 100 and crop.shape[1] <= 100

    def test_a_face_too_small_to_read_is_rejected(self):
        assert crop_face(_frame(), (10, 10, 10 + MIN_CROP_PIXELS - 2,
                                    10 + MIN_CROP_PIXELS - 2), pad=0.0) is None

    def test_a_degenerate_box_is_rejected(self):
        assert crop_face(_frame(), (50, 50, 50, 50)) is None
        assert crop_face(_frame(), (80, 80, 40, 40)) is None

    def test_no_frame_is_not_a_crash(self):
        assert crop_face(None, (0, 0, 50, 50)) is None
        assert crop_face(np.zeros((0, 0, 3), dtype=np.uint8), (0, 0, 50, 50)) is None


class TestScanFrames:
    def _detect(self, n=1, score=0.9):
        def detect(frame):
            return [{"bbox": (10 + i * 60, 10, 70 + i * 60, 70),
                     "det_score": score} for i in range(n)]
        return detect

    def _embed(self, calls=None):
        def embed(crops):
            if calls is not None:
                calls.append(len(crops))
            return np.stack([_vec(1.0, 0.0, 0.0) for _ in crops])
        return embed

    def test_every_usable_face_is_found_and_embedded(self):
        frames = [(t, _frame()) for t in (0.0, 1.0, 2.0)]
        found = scan_frames(frames, detect_fn=self._detect(), embed_fn=self._embed())
        assert len(found) == 3
        assert all(c.embedding is not None for c in found)
        assert [c.timestamp for c in found] == [0.0, 1.0, 2.0]

    def test_low_confidence_detections_are_skipped(self):
        frames = [(0.0, _frame())]
        found = scan_frames(frames, detect_fn=self._detect(score=0.2),
                            embed_fn=self._embed())
        assert found == []

    def test_a_crowd_is_capped_so_background_faces_do_not_dominate(self):
        frames = [(0.0, _frame(w=1200))]
        found = scan_frames(frames, detect_fn=self._detect(n=10),
                            embed_fn=self._embed(), max_faces_per_frame=3)
        assert len(found) == 3

    def test_embedding_is_batched_rather_than_per_face(self):
        calls = []
        frames = [(float(t), _frame()) for t in range(10)]
        scan_frames(frames, detect_fn=self._detect(), embed_fn=self._embed(calls),
                    batch=4)
        assert calls == [4, 4, 2], "should batch, and flush the remainder"

    def test_no_faces_anywhere_is_not_an_error(self):
        found = scan_frames([(0.0, _frame())], detect_fn=lambda f: [],
                            embed_fn=self._embed())
        assert found == []

    def test_embeddings_come_back_unit_length(self):
        def embed(crops):
            return np.stack([np.asarray([3.0, 4.0, 0.0]) for _ in crops])
        found = scan_frames([(0.0, _frame())], detect_fn=self._detect(),
                            embed_fn=embed)
        assert round(float(np.linalg.norm(found[0].embedding)), 5) == 1.0


class TestCategory:
    def test_a_category_is_the_mean_of_its_examples(self):
        category = build_category("x", [_vec(1, 0, 0), _vec(0, 1, 0)])
        assert category.examples == 2
        assert round(float(np.linalg.norm(category.vector)), 5) == 1.0
        assert category.vector[0] == pytest.approx(category.vector[1])

    def test_a_category_needs_an_example(self):
        with pytest.raises(ValueError):
            build_category("x", [])

    def test_scoring_ranks_similar_crops_higher(self):
        category = build_category("x", [_vec(1, 0, 0)])
        crops = [FaceCrop(0.0, (0, 0, 1, 1), embedding=_vec(1, 0, 0)),
                 FaceCrop(1.0, (0, 0, 1, 1), embedding=_vec(0, 1, 0))]
        scores = score_crops(crops, category)
        assert scores[0] > scores[1]
        assert scores[0] == pytest.approx(1.0, abs=1e-5)

    def test_negatives_push_down_what_the_user_did_not_mean(self):
        category = build_category("x", [_vec(1, 1, 0)])
        crop = [FaceCrop(0.0, (0, 0, 1, 1), embedding=_vec(1, 1, 0))]
        plain = score_crops(crop, category)[0]
        with_negative = score_crops(crop, category, negatives=[_vec(1, 0, 0)])[0]
        assert with_negative < plain

    def test_scoring_nothing_returns_nothing(self):
        category = build_category("x", [_vec(1, 0, 0)])
        assert len(score_crops([], category)) == 0


class TestPerSecond:
    def test_the_strongest_face_in_a_second_wins(self):
        crops = [FaceCrop(4.2, (0, 0, 1, 1)), FaceCrop(4.8, (0, 0, 1, 1))]
        assert best_per_second(crops, [0.3, 0.7]) == {4: 0.7}

    def test_seconds_are_kept_apart(self):
        crops = [FaceCrop(1.0, (0, 0, 1, 1)), FaceCrop(9.0, (0, 0, 1, 1))]
        assert best_per_second(crops, [0.5, 0.9]) == {1: 0.5, 9: 0.9}


class TestSignal:
    def test_seconds_above_the_threshold_score(self):
        signal = to_signal({3: 0.9, 7: 0.2}, duration=10, threshold=0.5, points=6.0)
        assert signal[3] == 6.0
        assert signal[7] == 0.0

    def test_the_array_matches_the_video_length(self):
        assert len(to_signal({}, duration=120, threshold=0.5, points=1.0)) == 121

    def test_a_second_past_the_end_is_ignored_not_an_error(self):
        signal = to_signal({999: 0.9}, duration=10, threshold=0.5, points=6.0)
        assert not signal.any()

    def test_points_are_flat_not_scaled_by_similarity(self):
        """The rest of the weight table is in points; a fractional contributor
        would be impossible to reason about beside one that fires or does not."""
        signal = to_signal({1: 0.51, 2: 0.99}, duration=10, threshold=0.5,
                           points=6.0)
        assert signal[1] == signal[2] == 6.0


class TestStore:
    def _store(self, tmp_path):
        store = FaceCategoryStore(str(tmp_path / "cats.json"))
        store.add(build_category("one", [_vec(1, 0, 0)]))
        store.add(build_category("two", [_vec(0, 1, 0), _vec(0, 0, 1)]))
        return store

    def test_round_trip(self, tmp_path):
        self._store(tmp_path).save()
        reloaded = FaceCategoryStore(str(tmp_path / "cats.json"))
        assert reloaded.load() is True
        assert reloaded.names() == ["one", "two"]
        assert reloaded.get("two").examples == 2
        assert reloaded.get("one").vector[0] == pytest.approx(1.0, abs=1e-5)

    def test_loading_what_was_never_saved_is_false_not_an_error(self, tmp_path):
        assert FaceCategoryStore(str(tmp_path / "nope.json")).load() is False

    def test_a_corrupt_file_does_not_take_the_app_down(self, tmp_path):
        path = tmp_path / "cats.json"
        path.write_text("{not json", encoding="utf-8")
        assert FaceCategoryStore(str(path)).load() is False

    def test_saving_leaves_no_temporary_file_behind(self, tmp_path):
        self._store(tmp_path).save()
        assert [p.name for p in tmp_path.iterdir()] == ["cats.json"]

    def test_removal(self, tmp_path):
        store = self._store(tmp_path)
        assert store.remove("one") is True
        assert store.remove("one") is False
        assert store.names() == ["two"]

    def test_the_file_is_plain_json_a_user_can_read(self, tmp_path):
        self._store(tmp_path).save()
        data = json.loads((tmp_path / "cats.json").read_text(encoding="utf-8"))
        assert data["schema"] == 1
        assert {c["name"] for c in data["categories"]} == {"one", "two"}
