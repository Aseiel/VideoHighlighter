"""A fine-tuned action head that brings its own image tower.

The head's folder holds ``vision.onnx`` and ``head.json`` an ``own_encoder``
block; the app runs that tower instead of the shared encoder, on the same
routes, and refuses a folder whose tower it cannot trust. Search and actions
by name never see it. Classes here are made up: the mechanism has no opinion
about what a user teaches.
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import real_opencv  # noqa: E402

from modules.system import compute_backend as cb  # noqa: E402
from modules.vision import action_siglip as A  # noqa: E402
from modules.vision import frame_encoder as fe  # noqa: E402

CLASSES = ["class-a", "class-b"]
OWN_ID = "test-tower-finetuned"


# ── tiny real models ─────────────────────────────────────────────────────────

def _tower_weights():
    # Positive, so a bright frame gives a positive vector and the test head's
    # first class a high score.
    return np.abs(np.random.default_rng(1).normal(size=(3, fe.DIMS))).astype(np.float32)


def _tower_vector(pixels):
    """What the tiny tower computes: mean colour per channel times W."""
    return pixels.mean(axis=(2, 3)) @ _tower_weights()


def write_tower(path):
    """pixel_values [N, 3, 256, 256] -> [N, 768], with a weight large enough
    to be stored as fp16 by the install tool."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["pixel_values"], ["m"], axes=[2, 3], keepdims=0),
         helper.make_node("MatMul", ["m", "W"], ["image_embeds"])],
        "tower",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT,
                                       ["n", 3, fe.INPUT_SIZE, fe.INPUT_SIZE])],
        [helper.make_tensor_value_info("image_embeds", TensorProto.FLOAT, ["n", fe.DIMS])],
        [numpy_helper.from_array(_tower_weights(), "W")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


def write_head_onnx(path, frames=4):
    """features [N, frames, 768] -> logits [N, 2]: the frames' mean times W,
    so class-a scores high on any input with a positive first number."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper

    w = np.zeros((fe.DIMS, len(CLASSES)), np.float32)
    w[:, 0] = 1.0 / fe.DIMS
    w[:, 1] = -1.0 / fe.DIMS
    graph = helper.make_graph(
        [helper.make_node("ReduceMean", ["features"], ["m"], axes=[1], keepdims=0),
         helper.make_node("MatMul", ["m", "W"], ["logits"])],
        "head",
        [helper.make_tensor_value_info("features", TensorProto.FLOAT, ["n", frames, fe.DIMS])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["n", len(CLASSES)])],
        [numpy_helper.from_array(w, "W")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


def head_meta(**over):
    meta = {"format": 1, "kind": A.HEAD_KIND, "encoder": OWN_ID, "frames": 4,
            "classes": list(CLASSES), "trust_thresholds": [0.5, 0.5]}
    meta.update(over)
    return meta


def own_block(probe=None, **over):
    block = {"file": "vision.onnx", "preprocess": fe.ENCODER_ID, "dims": fe.DIMS,
             "probe": [float(v) for v in (probe if probe is not None else np.ones(fe.DIMS))]}
    block.update(over)
    return block


def make_head(folder, meta, tower=b"onnx", head=b""):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / A.HEAD_META).write_text(json.dumps(meta), encoding="utf-8")
    if callable(head):
        head(folder / A.HEAD_MODEL)
    else:
        (folder / A.HEAD_MODEL).write_bytes(head)
    if callable(tower):
        tower(folder / "vision.onnx")
    elif tower is not None:
        (folder / "vision.onnx").write_bytes(tower)
    return str(folder)


# ── what a folder must say ───────────────────────────────────────────────────

def test_a_head_on_the_shared_encoder_has_no_own_encoder(tmp_path):
    folder = make_head(tmp_path / "h", head_meta(encoder=fe.ENCODER_ID), tower=None)
    assert A.own_encoder(folder, A.read_head_meta(folder)) is None


def test_a_valid_block_points_at_the_tower(tmp_path):
    folder = make_head(tmp_path / "h", head_meta(own_encoder=own_block()))
    own = A.own_encoder(folder, A.read_head_meta(folder))
    assert own["path"] == os.path.join(folder, "vision.onnx")
    assert len(own["probe"]) == fe.DIMS


@pytest.mark.parametrize("meta,tower,words", [
    (head_meta(own_encoder=own_block(probe=[])), b"x", "probe"),
    (head_meta(own_encoder=own_block(preprocess="another-encoder")), b"x", "input"),
    (head_meta(encoder=fe.ENCODER_ID, own_encoder=own_block()), b"x", "id of its own"),
    (head_meta(encoder=None, own_encoder=own_block()), b"x", "id of its own"),
    (head_meta(own_encoder=own_block(dims=512)), b"x", "768"),
    (head_meta(own_encoder=own_block(file="../vision.onnx")), b"x", "file name"),
    (head_meta(own_encoder=own_block()), None, "missing"),
    (head_meta(own_encoder="vision.onnx"), b"x", "object"),
])
def test_a_tower_that_cannot_be_trusted_is_refused(tmp_path, meta, tower, words):
    folder = make_head(tmp_path / "h", meta, tower=tower)
    with pytest.raises(ValueError, match=words):
        A.read_head_meta(folder)


def test_heads_with_their_own_encoder_are_found(tmp_path, monkeypatch):
    shared = make_head(tmp_path / "shared", head_meta(encoder=fe.ENCODER_ID), tower=None)
    tuned = make_head(tmp_path / "tuned", head_meta(own_encoder=own_block()))
    make_head(tmp_path / "other", head_meta(encoder="enc-b"), tower=None)
    make_head(tmp_path / "broken", head_meta(own_encoder=own_block(probe=[])))
    monkeypatch.setattr(A, "_head_dirs", lambda: sorted(str(p) for p in tmp_path.iterdir()))
    assert A.find_heads(fe.ENCODER_ID) == [shared, tuned]


def test_a_tuned_head_is_available_without_the_shared_encoder(tmp_path, monkeypatch):
    make_head(tmp_path / "tuned", head_meta(own_encoder=own_block()))
    monkeypatch.setattr(A, "_head_dirs", lambda: [str(tmp_path / "tuned")])
    monkeypatch.setattr(fe, "is_installed", lambda: False)
    assert A.available()


# ── which encoder a head reads ───────────────────────────────────────────────

class _Head:
    def __init__(self, own=None, encoder_id=OWN_ID):
        self.own_encoder, self.encoder_id = own, encoder_id


def test_a_tuned_head_reads_its_own_tower(monkeypatch):
    calls = []
    monkeypatch.setattr(fe, "load_tower", lambda *a, **k: calls.append(("tower", a)) or "T")
    monkeypatch.setattr(fe, "load", lambda *a, **k: calls.append(("shared", a)) or "S")
    own = {"path": "x/vision.onnx", "probe": [1.0] * fe.DIMS}
    assert A.encoder_for(_Head(own)) == "T"
    assert calls == [("tower", ("x/vision.onnx", own["probe"], OWN_ID))]


def test_a_frozen_head_reads_the_shared_encoder(monkeypatch):
    monkeypatch.setattr(fe, "load_tower", lambda *a, **k: pytest.fail("not this one"))
    monkeypatch.setattr(fe, "load", lambda *a, **k: "S")
    assert A.encoder_for(_Head(None, fe.ENCODER_ID)) == "S"


# ── loading a tower on the routes ────────────────────────────────────────────

@pytest.fixture
def no_openvino(monkeypatch):
    """The suite shims OpenVINO with a mock; make it absent instead, so the
    processor route here is ONNX Runtime."""
    monkeypatch.setitem(sys.modules, "openvino", None)


def test_load_tower_runs_on_the_routes_with_its_own_id(tmp_path, no_openvino):
    pytest.importorskip("onnxruntime")
    path = tmp_path / "vision.onnx"
    write_tower(path)
    probe = _tower_vector(fe.probe_pixels())[0]
    enc = fe.load_tower(str(path), probe, OWN_ID, backend=cb.CPU, log=lambda m: None)
    assert enc is not None and enc.route == fe.ONNX_CPU
    assert enc.encoder_id == OWN_ID
    assert fe.FrameEncoder.encoder_id == fe.ENCODER_ID     # the class is untouched
    pixels = np.repeat(fe.probe_pixels(), 2, axis=0)
    assert np.allclose(enc.encode_pixels(pixels), _tower_vector(pixels), atol=1e-4)


def test_load_tower_refuses_a_tower_that_misses_its_probe(tmp_path, no_openvino):
    pytest.importorskip("onnxruntime")
    path = tmp_path / "vision.onnx"
    write_tower(path)
    wrong = np.random.default_rng(9).normal(size=fe.DIMS)
    logged = []
    assert fe.load_tower(str(path), wrong, OWN_ID, backend=cb.CPU, log=logged.append) is None
    assert "no route" in logged[-1]


def test_load_tower_without_its_file(tmp_path):
    logged = []
    assert fe.load_tower(str(tmp_path / "gone.onnx"), [0.0] * fe.DIMS, OWN_ID,
                         log=logged.append) is None
    assert "missing" in logged[-1]


# ── the run ──────────────────────────────────────────────────────────────────

class _NoPeople:
    def predict(self, *a, **k):
        return []


def test_the_run_uses_the_heads_tower_and_never_the_shared_encoder(tmp_path, monkeypatch,
                                                                   no_openvino):
    pytest.importorskip("onnxruntime")
    cv2 = real_opencv()
    if cv2 is None:
        pytest.skip("needs OpenCV")
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    video = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for _ in range(60):                                      # 6 s, bright frames
        writer.write(np.full((48, 64, 3), 200, np.uint8))
    writer.release()

    probe = _tower_vector(fe.probe_pixels())[0]
    folder = make_head(tmp_path / "tuned", head_meta(own_encoder=own_block(probe)),
                       tower=write_tower, head=write_head_onnx)
    monkeypatch.setattr(A, "_head_dirs", lambda: [folder])
    monkeypatch.setattr(fe, "is_installed", lambda: False)
    monkeypatch.setattr(fe, "load", lambda *a, **k: pytest.fail("the shared encoder was loaded"))
    monkeypatch.setattr("modules.system.compute_backend.configured", lambda: cb.CPU)
    logged = []
    detections, boxes = A.run_action_detection_siglip(video, detector=_NoPeople(),
                                                      log=logged.append)
    assert detections and {d[4] for d in detections} == {"class-a"}
    assert any(f"the head's own {OWN_ID}" in m for m in logged)


def test_a_tuned_head_refuses_the_shared_encoder_handed_to_it(tmp_path, monkeypatch):
    folder = make_head(tmp_path / "tuned", head_meta(own_encoder=own_block()))

    class Shared:
        encoder_id = fe.ENCODER_ID
        label = "test"

    head = _Head({"path": os.path.join(folder, "vision.onnx")})
    head.name, head.frames = "tuned", 4
    logged = []
    assert A.run_action_detection_siglip("x.mp4", head=head, encoder=Shared(),
                                         log=logged.append) == ([], [])
    assert "trained on" in logged[-1]


# ── the install tool ─────────────────────────────────────────────────────────

def _export(tmp_path, **meta):
    folder = tmp_path / "export"
    make_head(folder, head_meta(**meta), tower=write_tower, head=write_head_onnx)
    return str(folder)


def test_install_writes_a_folder_the_app_accepts(tmp_path, no_openvino):
    pytest.importorskip("onnxruntime")
    from tools import install_action_model as tool

    dest = tool.install(_export(tmp_path), str(tmp_path / "models"), "tuned", log=lambda m: None)
    assert sorted(os.listdir(dest)) == [A.HEAD_META, A.HEAD_MODEL, "vision.onnx"]
    meta = A.read_head_meta(dest)
    own = A.own_encoder(dest, meta)
    assert own["preprocess"] == fe.ENCODER_ID
    assert fe._cosine(own["probe"], _tower_vector(fe.probe_pixels())[0]) > 0.99999
    assert meta["classes"] == CLASSES
    # the fp16-stored tower still reproduces the probe on a route
    assert fe.load_tower(own["path"], own["probe"], OWN_ID, backend=cb.CPU,
                         log=lambda m: None) is not None


def test_install_refuses_a_head_on_the_shared_id(tmp_path, no_openvino):
    pytest.importorskip("onnxruntime")
    from tools import install_action_model as tool

    with pytest.raises(ValueError, match="its own"):
        tool.install(_export(tmp_path, encoder=fe.ENCODER_ID), str(tmp_path / "m"), "x",
                     log=lambda m: None)
    assert not os.path.exists(tmp_path / "m" / "x")


def test_install_keeps_an_installed_folder_unless_forced(tmp_path, no_openvino):
    pytest.importorskip("onnxruntime")
    from tools import install_action_model as tool

    export = _export(tmp_path)
    tool.install(export, str(tmp_path / "m"), "x", log=lambda m: None)
    with pytest.raises(ValueError, match="exists"):
        tool.install(export, str(tmp_path / "m"), "x", log=lambda m: None)
    assert tool.install(export, str(tmp_path / "m"), "x", force=True, log=lambda m: None)
