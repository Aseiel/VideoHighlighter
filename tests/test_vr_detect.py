"""Noticing that a frame holds two eyes, without being told.

The cost of the two possible mistakes is not symmetric. Missing side-by-side
footage leaves the filmstrip as it was — every thumbnail centre-cropped onto
the seam between the eyes, which is what prompted this. Claiming it on flat
footage silently hides half of every frame, and the person watching has no
reason to suspect the app of cropping. So the geometry gate is deliberately
narrow and the halves have to actually agree before anything is cropped.

The clips here are synthesised rather than sampled from real footage: the
formats worth pinning are the ones that fool a single test — a panorama, whose
halves are the same room but not the same picture, and a symmetric composition,
whose halves *are* each other, mirrored.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# conftest is loaded by pytest rather than imported, so its helpers are not on
# the path by default.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import real_opencv          # noqa: E402

from modules import vr_detect

cv2 = real_opencv()
if cv2 is None:
    pytest.skip("needs a real OpenCV to compare pixels", allow_module_level=True)


@pytest.fixture(autouse=True)
def _real_cv2_in_the_detector(monkeypatch):
    monkeypatch.setattr(vr_detect, "cv2", cv2)


def _write(path, frames, size):
    """Write `frames` (BGR arrays) as a short clip at `size` = (w, h)."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             10.0, size)
    assert writer.isOpened(), "no encoder available for the fixture"
    try:
        for frame in frames:
            for _ in range(10):          # a second each, so sampling has choices
                writer.write(frame)
    finally:
        writer.release()
    return str(path)


def _noise(w, h, seed):
    rng = np.random.default_rng(seed)
    coarse = rng.integers(0, 255, (h // 16, w // 16, 3), dtype=np.uint8)
    return cv2.resize(coarse, (w, h), interpolation=cv2.INTER_LINEAR)


def _stereo_pair(eye_w, eye_h, seed, shift=6):
    """One frame of side-by-side: the same picture twice, a parallax apart."""
    scene = _noise(eye_w + shift * 2, eye_h, seed)
    left = scene[:, :eye_w]
    right = scene[:, shift:eye_w + shift]
    return np.hstack([left, right])


@pytest.fixture(autouse=True)
def _fresh_results():
    vr_detect.forget()
    yield
    vr_detect.forget()


class TestWhatItCallsSideBySide:
    def test_two_eyes_of_one_scene(self, tmp_path):
        frames = [_stereo_pair(320, 320, seed) for seed in (1, 2, 3)]
        path = _write(tmp_path / "sbs.mp4", frames, (640, 320))

        layout = vr_detect.probe(path)

        assert layout.side_by_side, layout.reason
        assert layout.eye_aspect == pytest.approx(1.0)

    def test_two_unrelated_pictures_side_by_side(self, tmp_path):
        frames = [np.hstack([_noise(320, 320, seed), _noise(320, 320, seed + 50)])
                  for seed in (1, 2, 3)]
        path = _write(tmp_path / "pair.mp4", frames, (640, 320))

        assert not vr_detect.probe(path).side_by_side

    def test_a_panorama_is_not_two_eyes(self, tmp_path):
        # The format the geometry gate cannot rule out: a monoscopic 360° video
        # is 2:1 as well. Its halves are two parts of one continuous picture,
        # which is a different thing from two views of one scene.
        frames = [_noise(640, 320, seed) for seed in (1, 2, 3)]
        path = _write(tmp_path / "pano.mp4", frames, (640, 320))

        assert not vr_detect.probe(path).side_by_side

    def test_a_mirrored_composition_is_not_two_eyes(self, tmp_path):
        # Halves that match perfectly *because* the frame is symmetric —
        # mirrored letterbox furniture, a centred subject on a mirrored
        # backdrop. Correlation alone says stereo; the mirrored control is what
        # tells them apart.
        frames = []
        for seed in (1, 2, 3):
            half = _noise(320, 320, seed)
            frames.append(np.hstack([half, np.fliplr(half)]))
        path = _write(tmp_path / "mirror.mp4", frames, (640, 320))

        assert not vr_detect.probe(path).side_by_side


class TestTheGeometryGate:
    def test_a_16_9_video_is_never_even_sampled(self, tmp_path):
        # Nothing 16:9 can hold two eyes of any shape anyone shoots, so this
        # must cost no decode at all — it runs on every video that is opened.
        frames = [_noise(640, 360, seed) for seed in (1, 2)]
        path = _write(tmp_path / "flat.mp4", frames, (640, 360))
        sampled = []
        original = vr_detect._sample_halves
        vr_detect._sample_halves = lambda *a, **k: sampled.append(a) or None
        try:
            layout = vr_detect.probe(path)
        finally:
            vr_detect._sample_halves = original

        assert not layout.side_by_side
        assert sampled == [], "decoded a frame to rule out a 16:9 video"
        assert "narrow" in layout.reason

    def test_scope_format_halves_are_not_an_eye_shape(self, tmp_path):
        # 2.39:1 is wide enough to look suspicious and is nothing of the sort:
        # halved it is 1.19:1, a shape no camera produces. Ruled out on the
        # numbers, before any frame is compared.
        frames = [np.hstack([_noise(383, 320, 1), _noise(383, 320, 9)])]
        path = _write(tmp_path / "scope.mp4", frames, (766, 320))

        layout = vr_detect.probe(path)

        assert not layout.side_by_side
        assert "eye" in layout.reason

    @pytest.mark.parametrize("aspect", [1.0, 4 / 3, 16 / 9])
    def test_the_shapes_an_eye_comes_in(self, aspect):
        assert vr_detect._is_eye_shaped(aspect)

    @pytest.mark.parametrize("aspect", [1.19, 0.5, 2.0, 1.55])
    def test_shapes_it_does_not(self, aspect):
        assert not vr_detect._is_eye_shaped(aspect)


class TestItNeverGetsInTheWay:
    def test_a_missing_file_is_reported_as_a_plain_video(self, tmp_path):
        layout = vr_detect.probe(tmp_path / "not-here.mp4")
        assert not layout.side_by_side
        assert layout.eye_aspect == 0.0

    def test_no_path_at_all(self):
        assert not vr_detect.probe(None).side_by_side

    def test_a_second_probe_costs_nothing(self, tmp_path):
        frames = [_stereo_pair(320, 320, seed) for seed in (1, 2, 3)]
        path = _write(tmp_path / "sbs.mp4", frames, (640, 320))
        vr_detect.probe(path)

        calls = []
        original = vr_detect._read_metadata
        vr_detect._read_metadata = lambda p: calls.append(p) or original(p)
        try:
            assert vr_detect.probe(path).side_by_side
        finally:
            vr_detect._read_metadata = original

        assert calls == [], "re-probed a file it had already answered for"


class TestEmptyFramesAreNotEvidence:
    def test_a_black_video_is_not_two_matching_eyes(self, tmp_path):
        # Every half of a black frame matches every other, and a video that
        # fades to black in the middle would otherwise get a free vote.
        black = np.zeros((320, 640, 3), dtype=np.uint8)
        path = _write(tmp_path / "black.mp4", [black, black, black], (640, 320))

        layout = vr_detect.probe(path)

        assert not layout.side_by_side
        assert "no frame" in layout.reason or "different" in layout.reason
