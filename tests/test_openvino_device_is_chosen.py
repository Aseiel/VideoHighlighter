"""The compute preference decides whether OpenVINO gets the card.

`detect_best_device`'s DirectML branches already said openvino_device="CPU" --
"there is no OpenVINO GPU here", which on AMD is simply true, because the GPU
plugin is Intel-only. Nothing acted on it: `load_models` asked OpenVINO for
AUTO, which takes an Intel GPU when one exists. On an Arc, "Compute: DirectML"
therefore put OpenVINO on the very card ONNX Runtime was driving, and the
person detector -- the one consumer that infers from a worker thread -- joined
it there.

That combination wedges the run for good. Measured on this machine, three runs
of three: the loop stops in `encoder wait (OpenVINO/GPU)` and never returns, no
exception, no traceback. Pin the detector to the processor and the same video
finishes; take DirectML away and it finishes; leave both and it hangs. Before
the YOLOX swap the detector was ultralytics on torch, so it never entered the
OpenVINO GPU plugin at all, which is why this is new.

Nothing here needs a GPU: the detector's OpenVINO backend is stubbed out.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from modules.system.device_utils import DeviceInfo

PIPELINE_PY = Path(__file__).resolve().parent.parent / "pipeline.py"
ACTION_PY = Path(__file__).resolve().parent.parent / "action_recognition.py"


def _info(**kw) -> DeviceInfo:
    base = dict(yolo_pt_device="cpu", yolo_ov_device="CPU", openvino_device="CPU",
                pytorch_device="cpu", motion_device="cpu", use_openvino_yolo=False,
                gpu_available=True, backend_name="test")
    base.update(kw)
    return DeviceInfo(**base)


@pytest.mark.parametrize("label,kw", [
    ("DirectML via ONNX Runtime", {"onnx_dml_torch": True}),
    ("DirectML via torch", {"dml_device": "privateuseone:0"}),
])
def test_directml_leaves_openvino_off_the_card(label, kw):
    """Both DirectML routes must keep OpenVINO on the processor."""
    assert _info(**kw).openvino_device == "CPU", label


def test_an_intel_gpu_still_gets_openvino():
    """The fix must not push OpenVINO off a card it is the right runtime for."""
    assert _info(openvino_device="GPU", use_openvino_yolo=True).openvino_device == "GPU"


@pytest.mark.parametrize("device", ["CPU", "GPU", "AUTO"])
def test_the_person_detector_uses_the_device_it_is_given(device, monkeypatch):
    """It used to default to AUTO and pick the GPU whatever the run decided."""
    from modules.vision import detection_backend as db
    import action_recognition as ar

    seen = {}

    class StubDetector:
        def __init__(self, xml, class_names=None, device="AUTO", score_thr=None):
            seen["device"] = device

    monkeypatch.setattr(db, "YoloxOpenVINODetector", StubDetector)
    monkeypatch.setattr(db, "find_default_yolox_ir", lambda prefer=None: __file__)

    detector = ar.ParallelYOLODetector(num_workers=1, skip_frames=4, device=device)
    try:
        assert seen["device"] == device
    finally:
        detector.shutdown()


def test_the_pipeline_hands_that_device_to_the_action_run():
    """A run that decides the device and then does not pass it decides nothing."""
    src = PIPELINE_PY.read_text(encoding="utf-8")
    assert re.search(r"openvino_device\s*=\s*getattr\(\s*_dev", src), \
        "pipeline no longer derives the OpenVINO device from the detected backend"
    call = src[src.index("run_action_detection("):]
    call = call[:call.index("\n                )")]
    assert "device=openvino_device" in call, \
        "run_action_detection is not told which OpenVINO device the run chose"


def test_the_action_run_builds_its_detector_with_that_device():
    """Forwarding works only if the call site actually forwards.

    The constructor test above passes a device in by hand, so it stays green
    even if run_action_detection goes back to letting the detector default to
    AUTO -- which is the exact regression that put it on the GPU.
    """
    src = ACTION_PY.read_text(encoding="utf-8")
    call = src[src.index("yolo_detector = ParallelYOLODetector("):]
    call = call[:call.index(")")]
    assert "device=device" in call,         "run_action_detection builds the person detector without the run's device"
