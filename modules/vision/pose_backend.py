"""Pose estimation backend — RTMPose (Apache-2.0) keypoint runtime.

WHY THIS EXISTS
    The counterpart to modules/vision/detection_backend.py. YOLOX answers "where are the
    people"; this answers "what is that person doing with their limbs". The app
    lost the second question when the AGPL YOLO package was dropped, and several
    features were written assuming both answers exist — see modules/crop/pose.py
    for what went quiet.

    Permissively licensed on both halves that matter: mmpose's code is Apache-2.0
    and the weights are OpenMMLab's own release, so a model trained or analysed
    with this does not inherit a copyleft obligation. That is the same bar
    detection_backend.py clears, and for the same reason.

TOP-DOWN, AND WHAT THAT COSTS
    RTMPose estimates keypoints for ONE person box at a time. It cannot discover
    a person the detector missed, so it is not a second opinion on *how many*
    people there are — it is a second opinion on whether a given box is a person,
    and on what that person is doing. Callers that want the first thing must
    lower the detector's own threshold and let this confirm, rather than expect
    keypoints to appear where no box was proposed.

SHAPE OF THE CONTRACT
    estimate(frame_bgr, boxes) returns one entry per input box, in the same
    order, each with pixel-space keypoints in the ORIGINAL frame's coordinates.
    Boxes that fall outside the frame come back with an empty keypoint array
    rather than being dropped, so the caller's indexing into `boxes` stays valid.

THE COORDINATE ROUND TRIP IS THE EASY THING TO GET WRONG
    Each box is expanded to the model's 3:4 aspect (padding 1.25), warped into a
    192x256 patch, and the SimCC output is decoded in that patch's pixel space —
    so every keypoint must be mapped back through the inverse of that warp. Skip
    it and the keypoints look plausible, sit inside the frame, and are wrong by
    exactly the crop offset, which no assertion catches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from modules.vision.rtmpose_models import (
    BBOX_PADDING, INPUT_SIZE, MEAN, MODEL_DIR, SIMCC_SPLIT_RATIO, SIZES, STD,
)

# COCO-17 layout, which is what the body7 exports emit. Named because positional
# indexing into keypoint arrays is where silent breakage lives.
KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)


@dataclass
class Pose:
    """One person's keypoints, in pixel coordinates of the ORIGINAL frame.

    `keypoints` is [K, 3] — x, y, score — matching the layout the rest of the
    app already reads (`kp[i][2] > threshold` for visibility).
    """
    keypoints: np.ndarray
    bbox: tuple
    num_visible: int = 0
    scores: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


class PoseEstimator(Protocol):
    """The contract callers code against, so the model can be swapped without
    touching a call site — the same arrangement detection_backend.Detector has."""

    def estimate(self, frame_bgr, boxes: Sequence[Sequence[float]]) -> list:
        ...


def find_default_rtmpose_ir(prefer: str = "large"):
    """The installed RTMPose IR, mirroring `find_default_yolox_ir`:
    `prefer="small"` picks the fastest installed size, anything else the most
    accurate. Returns None when nothing is installed — which is a normal state,
    not an error: pose is optional everywhere it is used."""
    order = list(SIZES)  # s, m, l - ascending accuracy
    present = [s for s in order if (MODEL_DIR / f"rtmpose_{s}.xml").exists()]
    if not present:
        return None
    chosen = present[0] if prefer == "small" else present[-1]
    return str(MODEL_DIR / f"rtmpose_{chosen}.xml")


def _box_to_center_scale(box, padding: float = BBOX_PADDING):
    """A box to the centre and the (w, h) region the affine warp samples.

    RTMPose was trained on boxes expanded to the model's aspect ratio and then
    padded; reproducing both is what makes the keypoints land. The aspect fix
    comes first — pad a wrong aspect and you have a bigger wrong aspect.
    """
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    cx, cy = x1 + w * 0.5, y1 + h * 0.5

    aspect = INPUT_SIZE[0] / INPUT_SIZE[1]     # 192 / 256
    if w > h * aspect:
        h = w / aspect
    else:
        w = h * aspect
    return (cx, cy), (w * padding, h * padding)


def _affine(center, scale, out_w: int, out_h: int):
    """The 2x3 warp from frame space into the model's patch, and its inverse.

    Axis-aligned, so it is a scale plus a translation and the inverse is exact —
    no cv2.invertAffineTransform round-off, and no rotation to reason about.
    """
    sx, sy = out_w / scale[0], out_h / scale[1]
    tx = out_w * 0.5 - center[0] * sx
    ty = out_h * 0.5 - center[1] * sy
    fwd = np.array([[sx, 0.0, tx], [0.0, sy, ty]], dtype=np.float32)
    return fwd, (sx, sy, tx, ty)


def _decode_simcc(simcc_x: np.ndarray, simcc_y: np.ndarray,
                  split_ratio: float = SIMCC_SPLIT_RATIO):
    """SimCC to (coords [N, K, 2] in patch pixels, scores [N, K]).

    SimCC predicts two 1-D distributions per keypoint, over x and over y at
    `split_ratio` sub-pixel bins. The location is the argmax of each; the
    confidence is the smaller of the two peaks, because a keypoint is only as
    well localised as its worse axis.
    """
    x_idx = simcc_x.argmax(axis=-1).astype(np.float32)
    y_idx = simcc_y.argmax(axis=-1).astype(np.float32)
    x_val = simcc_x.max(axis=-1)
    y_val = simcc_y.max(axis=-1)

    coords = np.stack([x_idx / split_ratio, y_idx / split_ratio], axis=-1)
    scores = np.minimum(x_val, y_val)
    # A non-positive peak on either axis means the head found nothing; -1 keeps
    # such a point out of every downstream mean/extent without a sentinel check.
    coords[scores <= 0] = -1.0
    return coords, scores


class RTMPoseOpenVINOEstimator:
    """RTMPose on OpenVINO — the same runtime YOLOX already uses, so pose costs
    no new dependency and lands on the same device (Arc included)."""

    def __init__(self, model_xml=None, device: str = "GPU",
                 keypoint_thr: float = 0.3, max_batch: int = 16):
        model_xml = model_xml or find_default_rtmpose_ir()
        if not model_xml:
            raise FileNotFoundError(
                "No RTMPose IR found. Run: python tools/get_rtmpose_model.py"
            )
        import openvino as ov  # lazy, as in detection_backend

        self.model_xml = str(model_xml)
        self.keypoint_thr = float(keypoint_thr)
        self.max_batch = int(max_batch)

        core = ov.Core()
        model = core.read_model(self.model_xml)
        try:
            self._net = core.compile_model(model, device)
        except Exception:
            # Same fallback the detector makes: a missing or busy GPU should
            # degrade to CPU rather than take the feature down.
            self._net = core.compile_model(model, "CPU")
        self._out_x = self._net.output(0)
        self._out_y = self._net.output(1)

    def estimate(self, frame_bgr, boxes) -> list:
        import cv2

        if frame_bgr is None or not len(boxes):
            return []

        out_w, out_h = INPUT_SIZE
        patches, inverses, kept = [], [], []
        for i, box in enumerate(boxes):
            center, scale = _box_to_center_scale(box)
            fwd, inv = _affine(center, scale, out_w, out_h)
            patch = cv2.warpAffine(frame_bgr, fwd, (out_w, out_h), flags=cv2.INTER_LINEAR)
            if patch is None or patch.size == 0:
                continue
            rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32)
            rgb = (rgb - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
            patches.append(rgb.transpose(2, 0, 1))
            inverses.append(inv)
            kept.append(i)

        results = [
            Pose(np.zeros((0, 3), dtype=np.float32), tuple(int(v) for v in b[:4]))
            for b in boxes
        ]
        if not patches:
            return results

        for start in range(0, len(patches), self.max_batch):
            chunk = np.stack(patches[start:start + self.max_batch])
            out = self._net(chunk)
            coords, scores = _decode_simcc(out[self._out_x], out[self._out_y])

            for j in range(coords.shape[0]):
                sx, sy, tx, ty = inverses[start + j]
                xy = coords[j].copy()
                visible = xy[:, 0] >= 0
                xy[:, 0] = (xy[:, 0] - tx) / sx
                xy[:, 1] = (xy[:, 1] - ty) / sy
                kp = np.concatenate([xy, scores[j][:, None]], axis=1).astype(np.float32)
                kp[~visible] = 0.0

                idx = kept[start + j]
                results[idx] = Pose(
                    keypoints=kp,
                    bbox=tuple(int(v) for v in boxes[idx][:4]),
                    num_visible=int(((kp[:, 2] > self.keypoint_thr) & visible).sum()),
                    scores=scores[j].astype(np.float32),
                )
        return results


def build_pose_estimator(device: str = "GPU", prefer: str = "large",
                         auto_install: bool = False):
    """The estimator, or None when pose is unavailable.

    None is a supported answer everywhere this is called — every caller already
    guards on it, because the app spent a long time with no pose model at all.
    Returning None rather than raising keeps that the normal path.
    """
    ir = find_default_rtmpose_ir(prefer=prefer)
    if ir is None and auto_install:
        try:
            from modules.vision import rtmpose_models
            rtmpose_models.install()
            ir = find_default_rtmpose_ir(prefer=prefer)
        except Exception as e:
            print(f"⚠️ RTMPose install failed: {e}")
            return None
    if ir is None:
        return None
    try:
        return RTMPoseOpenVINOEstimator(ir, device=device)
    except Exception as e:
        print(f"⚠️ RTMPose unavailable: {e}")
        return None
