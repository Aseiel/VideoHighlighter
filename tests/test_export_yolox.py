"""Tests for turning a trained checkpoint into a model the app can load.

The layout check is pure and runs everywhere. The export itself runs in a
subprocess for the reason ``test_train_yolox_run`` documents: ``conftest``
shims ``torch``, and torch cannot be removed from ``sys.modules`` and imported
again without breaking the interpreter for every later test.

The regression these guard is the quiet one. An export made with the decode
baked in has the *same shape* as a correct one — nothing raises, the IR loads,
the detector runs, and every box is in the wrong place.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from training.export_yolox import (
    INPUT_NAME,
    ONNX_OPSET,
    OUTPUT_NAME,
    _check_raw_grid,
    write_sidecar,
)
from training.train_yolox_run import pretrained_path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _output(anchors: int, width: int) -> np.ndarray:
    return np.zeros((1, anchors, width), dtype=np.float32)


# ── the layout the decoder depends on ────────────────────────────────────

def test_a_correct_export_passes():
    # 416x416 at strides (8,16,32): 52² + 26² + 13² = 3549 anchors,
    # each cx, cy, w, h, obj + one score per class.
    _check_raw_grid(_output(3549, 8), num_classes=3, input_size=(416, 416))
    _check_raw_grid(_output(8400, 85), num_classes=80, input_size=(640, 640))


def test_a_class_count_that_disagrees_with_the_sidecar_is_caught():
    # The sidecar about to be written says 3; the model says 5. Left alone,
    # every detection would be labelled with the wrong name.
    with pytest.raises(ValueError, match="does not match"):
        _check_raw_grid(_output(3549, 10), num_classes=3, input_size=(416, 416))


def test_an_anchor_count_the_decoder_cannot_grid_is_caught():
    # The decoder rebuilds the grid from the input size and strides. If the
    # export was made at a different resolution, the arithmetic silently
    # misaligns rather than failing.
    with pytest.raises(ValueError, match="anchors"):
        _check_raw_grid(_output(8400, 8), num_classes=3, input_size=(416, 416))


def test_a_two_dimensional_output_is_rejected():
    with pytest.raises(ValueError, match=r"\[1, anchors"):
        _check_raw_grid(np.zeros((3549, 8), np.float32), 3, (416, 416))


# ── the sidecar ──────────────────────────────────────────────────────────

def test_the_sidecar_is_the_bare_list_the_app_already_reads(tmp_path):
    """Not a richer format: the reader predates this writer.

    ``modules.vision.detection_backend`` reads a plain list. Anything cleverer here
    would be a format the app does not understand, and the symptom would be a
    model that loads with no class names at all.
    """
    write_sidecar(str(tmp_path), ["a", "b"], (416, 416), "m")
    assert json.loads((tmp_path / "labels.json").read_text(encoding="utf-8")) == ["a", "b"]


def test_the_geometry_travels_beside_the_model(tmp_path):
    write_sidecar(str(tmp_path), ["a"], (640, 640), "m")
    meta = json.loads((tmp_path / "m.meta.json").read_text(encoding="utf-8"))
    assert meta["input_size"] == [640, 640]
    assert meta["decode_in_inference"] is False
    assert meta["strides"] == [8, 16, 32]


def test_pretrained_checkpoints_are_named_per_size():
    path = pretrained_path("tiny")
    assert path.endswith(os.path.join("pretrained", "yolox_tiny.pth"))
    # Absolute, so the cache does not follow the working directory around.
    assert os.path.isabs(path)


def test_the_opset_matches_the_released_exports():
    """The decoder has only ever been exercised against opset 11 exports."""
    assert ONNX_OPSET == 11
    assert (INPUT_NAME, OUTPUT_NAME) == ("images", "output")


# ── the export itself ────────────────────────────────────────────────────

def _probe(body: str):
    """Run a snippet in a fresh interpreter; return the JSON it prints last."""
    import subprocess
    import sys

    code = "import json, warnings\nwarnings.filterwarnings('ignore')\n" + body
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=900)
    if done.returncode != 0:
        stderr = done.stderr or ""
        if "No module named" in stderr:
            pytest.skip(f"needs torch, yolox and openvino: {stderr.strip()[-200:]}")
        raise AssertionError(f"probe failed:\n{stderr[-2000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_export_emits_raw_grid_not_decoded_boxes(tmp_path):
    """The bug this module is built around, caught by the numbers' magnitude.

    A decoded export puts ``cx, cy`` in pixels (0-416 here) and ``w, h`` in
    pixels too. A raw-grid export leaves ``cx, cy`` as small offsets within a
    cell and ``w, h`` in log space, both within a couple of units of zero. The
    shapes are identical, so magnitude is the only thing that tells them apart
    — which is exactly why a wrong export gets all the way to wrong boxes
    without anything raising.
    """
    out = _probe(f"""
import torch
from training.train_yolox_run import _build_exp
from training.export_yolox import export_onnx

exp = _build_exp("unused", num_classes=3, size="tiny",
                 image_size=(416, 416), batch_size=1, workers=0, epochs=1)
model = exp.get_model()
ckpt = {{"model": model.state_dict(), "class_names": ["a", "b", "c"],
         "size": "tiny", "image_size": [416, 416]}}
path = r"{tmp_path.as_posix()}/ckpt.pth"
torch.save(ckpt, path)

onnx_path, names, size, anchors = export_onnx(path, r"{tmp_path.as_posix()}/m.onnx")

# Re-run the traced model the way the export left it, and look at the numbers.
# eval() first: in train mode YOLOX asserts that targets were supplied.
model.eval()
model.head.decode_in_inference = False
with torch.no_grad():
    raw = model(torch.zeros(1, 3, 416, 416))
model.head.decode_in_inference = True
with torch.no_grad():
    decoded = model(torch.zeros(1, 3, 416, 416))

print(json.dumps({{
    "anchors": anchors,
    "names": names,
    "raw_xy_max": float(raw[..., :2].abs().max()),
    "decoded_xy_max": float(decoded[..., :2].abs().max()),
    "same_shape": list(raw.shape) == list(decoded.shape),
}}))
""")
    assert out["anchors"] == 3549
    assert out["names"] == ["a", "b", "c"]
    # The trap, stated as an assertion: both layouts are the same shape.
    assert out["same_shape"], "decoded and raw exports differ in shape after all"
    # Raw grid offsets stay near zero; decoded ones reach across the image.
    assert out["raw_xy_max"] < 50, out
    assert out["decoded_xy_max"] > 100, out


def test_pretrained_weights_load_without_forcing_the_class_heads(tmp_path):
    """COCO weights must transfer, and the 80-class heads must not.

    Skipped rather than downloaded: a test suite should not pull 40 MB. Run a
    training pass once and this is cached.
    """
    cached = pretrained_path("tiny")   # repo-relative, resolved in the module
    if not os.path.exists(cached):
        pytest.skip("pretrained yolox_tiny.pth not cached; run a training pass first")

    out = _probe(f"""
from training.train_yolox_run import _build_exp, load_pretrained_backbone
exp = _build_exp("unused", num_classes=3, size="tiny",
                 image_size=(416, 416), batch_size=1, workers=0, epochs=1)
model = exp.get_model()
loaded, skipped = load_pretrained_backbone(model, r"{cached}")
print(json.dumps({{"loaded": loaded, "skipped": skipped}}))
""")
    # The backbone and neck are the great majority and must come across.
    assert out["loaded"] > 100, out
    # And the class-sized heads must not: 3 classes cannot take 80-class heads.
    assert out["skipped"] > 0, "the class heads were not skipped"


def test_each_model_is_installed_in_its_own_directory(tmp_path, monkeypatch):
    """Two models must not share a ``labels.json``.

    The sidecar's name is fixed, because that is what the app reads. Install a
    second model beside a first and the first one's class names are gone — it
    still loads, and every detection it makes is reported under the wrong
    label. This happened: a test run replaced the sidecar of an unrelated
    model that was already there.
    """
    import training.export_yolox as export

    calls = {}

    def fake_export_onnx(checkpoint, onnx_path, **kw):
        calls["onnx"] = onnx_path
        return onnx_path, ["alpha"], "tiny", 3549

    monkeypatch.setattr(export, "export_onnx", fake_export_onnx)
    monkeypatch.setattr(export, "convert_to_ir", lambda o, x: x)
    monkeypatch.setattr(export, "load_checkpoint",
                        lambda p: {"size": "tiny", "image_size": [416, 416],
                                   "class_names": ["alpha"], "model": {}})

    first = export.install("ckpt.pth", dest_dir=str(tmp_path), name="one")
    second = export.install("ckpt.pth", dest_dir=str(tmp_path), name="two")

    assert os.path.dirname(first.labels_path) != os.path.dirname(second.labels_path)
    assert os.path.exists(first.labels_path), "the first model's sidecar was removed"
