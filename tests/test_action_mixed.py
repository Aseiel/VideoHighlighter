"""A trained model and actions by name in one run.

With a head in use, typed actions it does not have are scored by name beside
it, on the same windows (whole frames, the shared encoder), as the Intel/R3D
"mixed" mode ran a trained model next to the Kinetics ones. A blank field
stays the head's own actions. Synthetic vectors and a fake head: nothing here
needs the model.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest

sys.path.insert(0, __import__("os").path.dirname(__file__))
from conftest import real_opencv  # noqa: E402

from modules.vision import action_siglip as A  # noqa: E402
from modules.vision import frame_encoder as fe  # noqa: E402

NAMES = ["robot dancing", "swimming", "cooking", "dancing macarena"]
DIMS = 8


class _Head(A.ActionHead):
    """Always reports its first class; no ONNX session."""

    def __init__(self):
        self.folder = "mine"
        self.classes = ["class-a", "class-b"]
        self.frames = 4
        self.encoder_id = fe.ENCODER_ID
        self.own_encoder = None
        self.thresholds = [0.5, 0.5]
        self.pairs = []
        self.seen = 0

    def scores(self, features):
        self.seen += len(features)
        return np.tile(np.array([[0.9, 0.1]], np.float32), (len(features), 1))


class _Encoder:
    """Every frame looks like "swimming"."""
    encoder_id = fe.ENCODER_ID
    label = "test"

    def encode_bgr(self, frames):
        out = np.zeros((len(frames), DIMS), np.float32)
        out[:, NAMES.index("swimming")] = 1.0
        return out


class _NoPeople:
    def predict(self, *a, **k):
        return []


@pytest.fixture
def video(tmp_path, monkeypatch):
    cv2 = real_opencv()
    if cv2 is None:
        pytest.skip("needs OpenCV")
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for _ in range(60):                                      # 6 s
        writer.write(np.full((48, 64, 3), 200, np.uint8))
    writer.release()
    folder = tmp_path / "encoder"
    folder.mkdir()
    np.savez(folder / fe.ACTIONS_FILE, labels=np.array(NAMES),
             vectors=np.eye(DIMS, dtype=np.float16)[:len(NAMES)])
    (folder / fe.META_FILE).write_text(json.dumps({"logit_scale": 100.0}))
    monkeypatch.setattr(fe, "find_model_dir", lambda: str(folder))
    monkeypatch.setattr(fe, "is_installed", lambda: True)
    return path


def _run(video, typed, head=None):
    head = head or _Head()
    logged = []
    detections, boxes = A.run_action_detection_siglip(
        video, head=head, encoder=_Encoder(), detector=_NoPeople(),
        interesting_actions=typed, log=logged.append)
    return head, detections, boxes, logged


def test_a_typed_action_the_model_lacks_is_scored_by_name_beside_it(video):
    head, detections, boxes, logged = _run(video, ["class-a", "swimming"])
    assert {d[4] for d in detections} == {"class-a", "swimming"}
    ids = {d[4]: d[2] for d in detections}
    assert ids == {"class-a": 0, "swimming": len(head.classes)}   # ids never collide
    assert {b["action_name"] for b in boxes} == {"class-a", "swimming"}
    assert any("by name beside it" in m for m in logged)
    assert not any("Not in action head" in m for m in logged)


def test_a_blank_field_is_the_models_actions_only(video, monkeypatch):
    monkeypatch.setattr(A, "text_actions", lambda *a, **k: pytest.fail("scored by name"))
    _, detections, _, _ = _run(video, None)
    assert {d[4] for d in detections} == {"class-a"}


def test_the_models_own_actions_need_nothing_by_name(video, monkeypatch):
    monkeypatch.setattr(A, "text_actions", lambda *a, **k: pytest.fail("scored by name"))
    _, detections, _, _ = _run(video, ["class-a"])
    assert {d[4] for d in detections} == {"class-a"}


def test_without_the_encoder_the_extra_actions_are_named_and_skipped(video, monkeypatch):
    head = _Head()
    head.own_encoder = {"path": "vision.onnx"}               # a fine-tuned head
    head.encoder_id = "tuned"
    monkeypatch.setattr(fe, "is_installed", lambda: False)

    class Own(_Encoder):
        encoder_id = "tuned"

    logged = []
    detections, _ = A.run_action_detection_siglip(
        video, head=head, encoder=Own(), detector=_NoPeople(),
        interesting_actions=["class-a", "swimming"], log=logged.append)
    assert {d[4] for d in detections} == {"class-a"}
    assert any("Not in action head mine: swimming" in m for m in logged)


def test_a_fine_tuned_head_loads_the_shared_encoder_for_the_extras(video, monkeypatch):
    head = _Head()
    head.own_encoder = {"path": "vision.onnx"}
    head.encoder_id = "tuned"
    loaded = []
    monkeypatch.setattr(fe, "load", lambda *a, **k: loaded.append(1) or _Encoder())

    class Own(_Encoder):
        encoder_id = "tuned"

        def encode_bgr(self, frames):
            return np.zeros((len(frames), DIMS), np.float32)

    detections, _ = A.run_action_detection_siglip(
        video, head=head, encoder=Own(), detector=_NoPeople(),
        interesting_actions=["class-a", "swimming"], log=lambda m: None)
    assert loaded == [1]
    assert {d[4] for d in detections} == {"class-a", "swimming"}
