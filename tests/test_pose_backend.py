"""Tests for the RTMPose keypoint backend.

The geometry is tested without a model, because that is where the bugs live and
because CI has neither the IR nor openvino. The one test that needs the real
network skips itself when the IR has not been installed.

No Qt, no module-level openvino: conftest shims both, and a real import at
module scope would fail collection rather than skip.
"""
import numpy as np
import pytest

from modules.vision.pose_backend import (
    KEYPOINT_NAMES, _affine, _box_to_center_scale, _decode_simcc,
    find_default_rtmpose_ir,
)
from modules.vision.rtmpose_models import BBOX_PADDING, INPUT_SIZE, SIZES


def test_keypoint_layout_is_coco_17():
    """The cropper indexes keypoints positionally — [9, 10] wrists, [11, 12]
    hips, [15, 16] ankles. A model with a different layout returns numbers that
    are wrong rather than missing, so pin the order here."""
    assert len(KEYPOINT_NAMES) == 17
    assert KEYPOINT_NAMES[9:11] == ("left_wrist", "right_wrist")
    assert KEYPOINT_NAMES[11:13] == ("left_hip", "right_hip")
    assert KEYPOINT_NAMES[15:17] == ("left_ankle", "right_ankle")


def test_every_offered_size_is_the_same_input_resolution():
    """SIZES may only hold exports the shared INPUT_SIZE/SIMCC constants
    describe. A 384x288 export in this table would decode at the wrong scale."""
    assert set(SIZES) == {"s", "m", "l"}
    assert INPUT_SIZE == (192, 256)


def test_box_is_expanded_to_the_model_aspect_before_padding():
    """A wide box must grow in height, not be squashed — RTMPose was trained on
    boxes at its own 3:4 aspect, and feeding it a distorted person is the
    quietest way to lose accuracy."""
    (cx, cy), (w, h) = _box_to_center_scale((0, 0, 200, 100))

    assert (cx, cy) == (100.0, 50.0)          # centre is untouched
    assert w / h == pytest.approx(INPUT_SIZE[0] / INPUT_SIZE[1])
    assert w == pytest.approx(200 * BBOX_PADDING)   # width already dominant
    assert h > 100                                   # height grew to match


def test_tall_box_grows_in_width():
    _, (w, h) = _box_to_center_scale((0, 0, 100, 400))

    assert w / h == pytest.approx(INPUT_SIZE[0] / INPUT_SIZE[1])
    assert h == pytest.approx(400 * BBOX_PADDING)


def test_affine_inverse_is_exact():
    """The round trip that decides whether keypoints land on the person. An
    error here does not crash and does not look wrong — it offsets every
    keypoint by the crop origin, which nothing downstream can detect."""
    center, scale = _box_to_center_scale((120, 40, 320, 460))
    out_w, out_h = INPUT_SIZE
    fwd, (sx, sy, tx, ty) = _affine(center, scale, out_w, out_h)

    pts = np.array([[120.0, 40.0], [320.0, 460.0], [220.0, 250.0]])
    mapped = pts @ fwd[:, :2].T + fwd[:, 2]
    back = np.stack([(mapped[:, 0] - tx) / sx, (mapped[:, 1] - ty) / sy], axis=1)

    assert np.abs(back - pts).max() < 1e-4
    # The box centre lands in the middle of the patch, by construction.
    centre_mapped = np.array(center) @ fwd[:, :2].T + fwd[:, 2]
    assert centre_mapped == pytest.approx([out_w / 2, out_h / 2])


def test_simcc_decode_picks_the_argmax_and_the_weaker_axis_score():
    """Confidence is min(x_peak, y_peak): a keypoint is only as well localised
    as its worse axis, so a confident x with a flat y must not read as certain."""
    simcc_x = np.zeros((1, 2, 384), dtype=np.float32)
    simcc_y = np.zeros((1, 2, 512), dtype=np.float32)
    simcc_x[0, 0, 100] = 0.9
    simcc_y[0, 0, 200] = 0.4          # weaker axis decides
    simcc_x[0, 1, 50] = 0.8
    simcc_y[0, 1, 60] = 0.7

    coords, scores = _decode_simcc(simcc_x, simcc_y)

    assert coords[0, 0].tolist() == [50.0, 100.0]     # split ratio 2.0
    assert scores[0, 0] == pytest.approx(0.4)
    assert coords[0, 1].tolist() == [25.0, 30.0]
    assert scores[0, 1] == pytest.approx(0.7)


def test_simcc_decode_marks_empty_predictions():
    """An all-zero head means nothing was found; -1 keeps such a point out of
    every downstream mean and extent without a sentinel check at each site."""
    coords, scores = _decode_simcc(np.zeros((1, 1, 384), dtype=np.float32),
                                   np.zeros((1, 1, 512), dtype=np.float32))

    assert (coords[0, 0] == -1.0).all()
    assert scores[0, 0] == 0.0


def test_estimate_returns_one_entry_per_box_in_order():
    """Callers index the result against their own box list, so a box that could
    not be estimated must come back empty rather than be dropped."""
    ir = find_default_rtmpose_ir()
    if ir is None:
        pytest.skip("RTMPose IR not installed (tools/get_rtmpose_model.py)")

    import sys
    from unittest.mock import MagicMock

    if isinstance(sys.modules.get("openvino"), MagicMock):
        pytest.skip("openvino is shimmed in this run")

    from modules.vision.pose_backend import RTMPoseOpenVINOEstimator

    est = RTMPoseOpenVINOEstimator(ir, device="CPU")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = est.estimate(frame, [(10, 10, 200, 400), (300, 50, 500, 460)])

    assert len(out) == 2
    assert out[0].bbox == (10, 10, 200, 400)
    assert out[1].bbox == (300, 50, 500, 460)
