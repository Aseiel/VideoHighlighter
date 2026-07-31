"""Tests for `modules.face_crops` — finding faces and cutting them out.

Detection and embedding are injected, which is the reason those are callables:
no cv2, no model, no video. Frames are plain arrays.
"""

from __future__ import annotations

import numpy as np

from modules.face_crops import (
    MIN_CROP_PIXELS,
    FaceCrop,
    crop_face,
    l2_normalise,
    scan_frames,
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
