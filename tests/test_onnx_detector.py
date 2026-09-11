"""The ONNX detector: boxes that land where the object is.

The arithmetic here replaces what Ultralytics does internally, so it is the
part that can be wrong in a way nothing else notices — a detection at the right
time with the wrong rectangle still writes a cache entry and still draws a box.
These tests drive real numbers through the decode path: no ONNX Runtime, no
OpenCV, no model file.

The letterbox tests need a real cv2 (the suite shims it away), so they skip
rather than assert against a mock.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

# conftest is loaded by pytest rather than imported, so its helpers are not on
# the path by default.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import real_opencv          # noqa: E402

from modules import onnx_detector         # noqa: E402

_real_cv2 = real_opencv()


@pytest.fixture
def real_cv2(monkeypatch):
    """Hand the lazy ``import cv2`` inside letterbox the real module.

    The suite shims OpenCV away, and a resize that returns a MagicMock makes
    every assertion about where a box landed vacuous.
    """
    if _real_cv2 is None:
        pytest.skip("needs a real OpenCV to resize a frame")
    monkeypatch.setitem(sys.modules, "cv2", _real_cv2)
    return _real_cv2


# A real export emits one row per anchor — 8400 of them at 640 — and all but a
# few are empty. Tests keep that shape (in miniature) rather than passing two
# rows, because the decoder reads the orientation off it: the anchors axis is
# the long one.
ANCHORS = 100


def _raw_output(boxes, num_classes=3, transposed=True, anchors=ANCHORS):
    """A model output holding `boxes` as (cx, cy, w, h, class_id, score).

    Built in the layout a YOLOv8/11 export actually emits — (1, 4 + nc,
    anchors) — so the orientation handling is exercised, not bypassed. The
    unused anchors score zero, which is what a real one does.
    """
    rows = np.zeros((max(anchors, len(boxes)), 4 + num_classes), dtype=np.float32)
    for i, (cx, cy, w, h, cls_id, score) in enumerate(boxes):
        rows[i, :4] = (cx, cy, w, h)
        rows[i, 4 + int(cls_id)] = score
    return np.expand_dims(rows.T if transposed else rows, axis=0)


class TestDecode:
    def test_a_centred_box_comes_back_in_frame_pixels(self):
        """640x640 model input, 1280x720 frame: scale 0.5, and the letterbox
        pads 140px top and bottom. A box at the canvas centre is the frame
        centre."""
        scale, pad_x, pad_y = 0.5, 0, 140
        raw = _raw_output([(320, 320, 100, 100, 1, 0.9)])

        boxes = onnx_detector.decode(raw, scale, pad_x, pad_y, (720, 1280, 3))

        assert len(boxes) == 1
        box = list(boxes)[0]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        assert (x1, x2) == (540.0, 740.0)          # 1280/2 ± 100
        assert (y1, y2) == (260.0, 460.0)          # 720/2 ± 100
        assert int(box.cls[0]) == 1
        assert float(box.conf[0]) == pytest.approx(0.9)

    def test_both_output_orientations_decode_the_same(self):
        args = (0.5, 0, 140, (720, 1280, 3))
        one = onnx_detector.decode(_raw_output([(320, 320, 100, 100, 1, 0.9)],
                                               transposed=True), *args)
        other = onnx_detector.decode(_raw_output([(320, 320, 100, 100, 1, 0.9)],
                                                 transposed=False), *args)

        assert (list(one)[0].xyxy[0].numpy() == list(other)[0].xyxy[0].numpy()).all()

    def test_low_confidence_rows_are_dropped(self):
        raw = _raw_output([(320, 320, 100, 100, 0, 0.10),
                           (100, 100, 20, 20, 2, 0.80)])

        boxes = onnx_detector.decode(raw, 1.0, 0, 0, (640, 640, 3), conf=0.25)

        assert [int(b.cls[0]) for b in boxes] == [2]

    def test_overlapping_boxes_collapse_to_the_best_one(self):
        raw = _raw_output([(320, 320, 100, 100, 0, 0.90),
                           (322, 321, 100, 100, 0, 0.80)])

        boxes = onnx_detector.decode(raw, 1.0, 0, 0, (640, 640, 3), iou=0.45)

        assert len(boxes) == 1
        assert float(list(boxes)[0].conf[0]) == pytest.approx(0.9)

    def test_distinct_boxes_both_survive(self):
        raw = _raw_output([(100, 100, 40, 40, 0, 0.90),
                           (500, 500, 40, 40, 1, 0.80)])

        boxes = onnx_detector.decode(raw, 1.0, 0, 0, (640, 640, 3))

        assert sorted(int(b.cls[0]) for b in boxes) == [0, 1]

    def test_a_box_predicted_off_frame_is_clamped(self):
        """Models do predict past the edge. A negative coordinate reaches
        OpenCV's rectangle call as a wrap-around, so it is clamped here."""
        raw = _raw_output([(10, 10, 200, 200, 0, 0.9)])

        boxes = onnx_detector.decode(raw, 1.0, 0, 0, (640, 640, 3))

        x1, y1, x2, y2 = list(boxes)[0].xyxy[0].numpy()
        assert (x1, y1) == (0.0, 0.0)
        assert x2 <= 640 and y2 <= 640

    def test_an_empty_frame_still_gives_something_to_iterate(self):
        """Call sites test `result.boxes is not None` and then loop."""
        boxes = onnx_detector.decode(_raw_output([]), 1.0, 0, 0, (640, 640, 3))

        assert boxes is not None
        assert list(boxes) == []


class TestResultShape:
    """The call sites were written against Ultralytics and are not changing."""

    def test_it_answers_the_ultralytics_idioms(self):
        raw = _raw_output([(320, 320, 100, 100, 2, 0.75)])
        boxes = onnx_detector.decode(raw, 1.0, 0, 0, (640, 640, 3))
        box = list(boxes)[0]

        assert isinstance(int(box.cls[0]), int)
        assert isinstance(float(box.conf[0]), float)
        assert box.xyxy[0].cpu().numpy().astype(int).tolist() == [270, 270, 370, 370]


class TestClassNames:
    def test_names_are_read_from_the_export(self):
        sess = types.SimpleNamespace(get_modelmeta=lambda: types.SimpleNamespace(
            custom_metadata_map={"names": "{0: 'person', 1: 'dog'}"}))

        assert onnx_detector._names_from_metadata(sess) == {0: "person", 1: "dog"}

    def test_a_list_of_names_works_too(self):
        sess = types.SimpleNamespace(get_modelmeta=lambda: types.SimpleNamespace(
            custom_metadata_map={"names": "['person', 'dog']"}))

        assert onnx_detector._names_from_metadata(sess) == {0: "person", 1: "dog"}

    def test_metadata_is_never_executed(self):
        """The file may have come from anywhere; a label list is not worth a
        code-execution hole."""
        sess = types.SimpleNamespace(get_modelmeta=lambda: types.SimpleNamespace(
            custom_metadata_map={"names": "__import__('os').system('echo no')"}))

        assert onnx_detector._names_from_metadata(sess) == {}

    def test_a_model_without_metadata_is_still_usable(self):
        sess = types.SimpleNamespace(get_modelmeta=lambda: types.SimpleNamespace(
            custom_metadata_map=None))

        assert onnx_detector._names_from_metadata(sess) == {}


class _FakeSession:
    """An ONNX session that returns one detection, whatever it is given."""

    def __init__(self, raw):
        self.raw = raw
        self.seen = []

    def get_inputs(self):
        return [types.SimpleNamespace(name="images")]

    def get_modelmeta(self):
        return types.SimpleNamespace(custom_metadata_map={"names": "{0: 'person'}"})

    def get_providers(self):
        return ["DmlExecutionProvider", "CPUExecutionProvider"]

    def run(self, output_names, feed):
        self.seen.append(feed)
        return [self.raw]


class TestTheDetector:
    def test_a_frame_goes_in_and_ultralytics_shaped_results_come_out(self, real_cv2):
        session = _FakeSession(_raw_output([(320, 320, 100, 100, 0, 0.9)]))
        detector = onnx_detector.OnnxDetector("fake.onnx", session=session)

        results = detector(np.zeros((720, 1280, 3), dtype=np.uint8), verbose=False)

        assert len(results) == 1
        assert detector.names[0] == "person"
        assert len(results[0].boxes) == 1

    def test_the_input_is_batched_float_rgb(self, real_cv2):
        session = _FakeSession(_raw_output([]))
        detector = onnx_detector.OnnxDetector("fake.onnx", session=session, imgsz=640)

        detector(np.zeros((720, 1280, 3), dtype=np.uint8))

        fed = session.seen[0]["images"]
        assert fed.shape == (1, 3, 640, 640)
        assert fed.dtype == np.float32
        assert fed.max() <= 1.0

    def test_it_reports_the_provider_it_actually_got(self):
        detector = onnx_detector.OnnxDetector.__new__(onnx_detector.OnnxDetector)
        detector.session = _FakeSession(_raw_output([]))

        assert detector.backend == "DmlExecutionProvider"


class TestLoad:
    def test_no_directml_means_no_detector_and_no_exception(self, monkeypatch):
        """Every caller's answer to 'no DirectML here' is the detector it
        already had, which is a fallback rather than an error."""
        monkeypatch.setattr(onnx_detector.ort_directml, "available", lambda: False)

        assert onnx_detector.load("anything.onnx") is None

    def test_a_missing_export_is_not_an_error_either(self, monkeypatch, tmp_path):
        monkeypatch.setattr(onnx_detector.ort_directml, "available", lambda: True)

        assert onnx_detector.load(tmp_path / "not-exported-yet.onnx") is None


class TestLetterbox:
    def test_the_aspect_ratio_survives_and_the_padding_is_centred(self, real_cv2):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        canvas, scale, pad_x, pad_y = onnx_detector.letterbox(frame, 640)

        assert canvas.shape == (640, 640, 3)
        assert scale == pytest.approx(0.5)
        assert (pad_x, pad_y) == (0, 140)

    def test_a_tall_frame_pads_sideways(self, real_cv2):
        frame = np.zeros((1280, 720, 3), dtype=np.uint8)

        canvas, scale, pad_x, pad_y = onnx_detector.letterbox(frame, 640)

        assert canvas.shape == (640, 640, 3)
        assert (pad_x, pad_y) == (140, 0)
