"""The frame encoder: one encoder for every taught action model, on any GPU.

Nothing here needs the model, a GPU or either runtime, except the last test,
which runs the real model when ``VH_FRAME_ENCODER_DIR`` points at an export
(``python -m tools.export_frame_encoder``). The rest pins the decisions the
module makes: the exact preprocessing (part of the encoder's identity), which
route each backend choice tries in which order, refusing a route that
returns wrong or non-finite vectors, and batching that keeps order.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import real_opencv  # noqa: E402

from modules.system import compute_backend as cb  # noqa: E402
from modules.vision import frame_encoder as fe  # noqa: E402


@pytest.fixture
def cv2(monkeypatch):
    """The real OpenCV for the module under test: preprocessing is about
    actual pixels, which the suite's shim cannot produce."""
    real = real_opencv()
    if real is None:
        pytest.skip("needs OpenCV")
    monkeypatch.setitem(sys.modules, "cv2", real)
    return real


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def test_preprocess_shape_range_and_rgb_order(cv2):
    red_bgr = np.zeros((100, 300, 3), np.uint8)
    red_bgr[:, :, 2] = 255
    out = fe.preprocess([red_bgr])
    assert out.shape == (1, 3, fe.INPUT_SIZE, fe.INPUT_SIZE)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0, 0], 1.0)     # R first
    np.testing.assert_allclose(out[0, 1:], -1.0)


def test_preprocess_is_the_measured_pipeline(cv2):
    """Area shrink to a short side of 384, then a bilinear squash to 256x256:
    the pipeline the measured accuracy came from."""
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    small = cv2.resize(frame, (int(1920 * 384 / 1080), 384), interpolation=cv2.INTER_AREA)
    sq = cv2.resize(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), (256, 256),
                    interpolation=cv2.INTER_LINEAR)
    expected = (sq.astype(np.float32) / 255 - 0.5) / 0.5
    np.testing.assert_allclose(fe.preprocess([frame])[0], expected.transpose(2, 0, 1), atol=1e-6)


def test_preprocess_never_enlarges_before_the_squash(cv2):
    frame = np.full((120, 160, 3), 128, np.uint8)
    np.testing.assert_allclose(fe.preprocess([frame]), 128 / 127.5 - 1, atol=1e-6)


def test_preprocess_accepts_grey_and_bgra(cv2):
    assert fe.preprocess([np.zeros((50, 60), np.uint8)]).shape == (1, 3, 256, 256)
    assert fe.preprocess([np.zeros((50, 60, 4), np.uint8)]).shape == (1, 3, 256, 256)


def test_probe_pixels_are_fixed():
    a, b = fe.probe_pixels(), fe.probe_pixels()
    assert a.shape == (1, 3, fe.INPUT_SIZE, fe.INPUT_SIZE)
    assert a.dtype == np.float32
    np.testing.assert_array_equal(a, b)
    assert a.min() >= -1 and a.max() <= 1 and a.std() > 0.3


# ---------------------------------------------------------------------------
# Route order
# ---------------------------------------------------------------------------

CPU = [fe.OPENVINO_CPU, fe.ONNX_CPU]


@pytest.mark.parametrize("backend", [None, cb.AUTO])
@pytest.mark.parametrize("intel_gpu, discrete, cuda, first", [
    (True, True, False, fe.OPENVINO_GPU),    # Arc
    (True, True, True, fe.OPENVINO_GPU),     # Arc next to an NVIDIA card
    (True, False, False, fe.OPENVINO_GPU),   # integrated Intel only
    (True, False, True, fe.ONNX_GPU),        # NVIDIA next to an integrated Intel
    (False, False, True, fe.ONNX_GPU),       # NVIDIA
    (False, False, False, fe.ONNX_GPU),      # AMD, any DX12 card, a Mac
])
def test_automatic_order(backend, intel_gpu, discrete, cuda, first):
    order = fe.route_order(backend, intel_gpu=intel_gpu, intel_discrete=discrete, cuda=cuda)
    assert order[0] == first
    assert sorted(order[:2]) == sorted([fe.OPENVINO_GPU, fe.ONNX_GPU])
    assert order[2:] == CPU


@pytest.mark.parametrize("backend, first", [
    (cb.INTEL, [fe.OPENVINO_GPU]),
    (cb.CUDA, [fe.ONNX_GPU]),          # no CUDA provider in the build: DirectML
    (cb.DIRECTML, [fe.ONNX_GPU]),
    (cb.APPLE, [fe.ONNX_GPU]),
    (cb.CPU, []),
    ("nvidia", [fe.ONNX_GPU]),         # the aliases compute_backend accepts
])
def test_named_backend_goes_first_and_the_processor_closes(backend, first):
    order = fe.route_order(backend, intel_gpu=True, intel_discrete=True, cuda=True)
    assert order == first + CPU


# ---------------------------------------------------------------------------
# Loading: each route must prove itself
# ---------------------------------------------------------------------------

class FakeRunner:
    def __init__(self, vector, calls=None):
        self.vector = np.asarray(vector, np.float32)
        self.calls = calls if calls is not None else []

    def run(self, pixels):
        self.calls.append(len(pixels))
        return np.tile(self.vector, (len(pixels), 1))


@pytest.fixture
def model_dir(tmp_path):
    probe = np.random.default_rng(0).normal(size=fe.DIMS)
    (tmp_path / fe.MODEL_FILE).write_bytes(b"onnx")
    (tmp_path / fe.META_FILE).write_text(json.dumps({
        "format": fe.FORMAT, "id": fe.ENCODER_ID, "dims": fe.DIMS,
        "probe": probe.tolist()}), encoding="utf-8")
    return tmp_path, probe


@pytest.fixture
def machine(monkeypatch):
    """Pin the hardware: an Arc card, no NVIDIA."""
    monkeypatch.setattr(fe, "_openvino_gpu", lambda: ("GPU", True))
    monkeypatch.setattr(fe, "_cuda_present", lambda: False)
    monkeypatch.setattr(fe, "route_label", lambda route: route)


def _routes(monkeypatch, answers):
    """answers: route -> a vector, or an exception to raise when opened."""
    tried = []

    def open_route(route, path, device):
        tried.append(route)
        answer = answers.get(route, RuntimeError("not here"))
        if isinstance(answer, Exception):
            raise answer
        return FakeRunner(answer)

    monkeypatch.setattr(fe, "_open_route", open_route)
    return tried


def test_load_takes_the_first_route_that_matches(monkeypatch, model_dir, machine):
    folder, probe = model_dir
    tried = _routes(monkeypatch, {fe.OPENVINO_GPU: probe, fe.OPENVINO_CPU: probe})
    enc = fe.load(cb.AUTO, log=lambda m: None, model_dir=str(folder))
    assert enc.route == fe.OPENVINO_GPU and enc.on_gpu
    assert enc.batch == fe.GPU_BATCH
    assert tried == [fe.OPENVINO_GPU]


@pytest.mark.parametrize("bad", ["nan", "garbage", "shape", "raises"])
def test_load_skips_a_route_that_is_wrong(monkeypatch, model_dir, machine, bad):
    folder, probe = model_dir
    wrong = {"nan": np.full(fe.DIMS, np.nan),
             "garbage": np.random.default_rng(5).normal(size=fe.DIMS),
             "shape": probe[:100],
             "raises": RuntimeError("driver said no")}[bad]
    tried = _routes(monkeypatch, {fe.OPENVINO_GPU: wrong, fe.ONNX_GPU: probe})
    enc = fe.load(cb.AUTO, log=lambda m: None, model_dir=str(folder))
    assert enc is not None and enc.route == fe.ONNX_GPU
    assert tried == [fe.OPENVINO_GPU, fe.ONNX_GPU]


def test_load_reaches_the_processor_and_batches_smaller(monkeypatch, model_dir, machine):
    folder, probe = model_dir
    _routes(monkeypatch, {fe.ONNX_CPU: probe})
    enc = fe.load(cb.AUTO, log=lambda m: None, model_dir=str(folder))
    assert enc.route == fe.ONNX_CPU and not enc.on_gpu
    assert enc.batch == fe.CPU_BATCH


def test_load_returns_none_when_nothing_runs(monkeypatch, model_dir, machine):
    folder, _ = model_dir
    _routes(monkeypatch, {})
    logged = []
    assert fe.load(cb.AUTO, log=logged.append, model_dir=str(folder)) is None
    assert "no route" in logged[-1]


def test_load_reads_the_users_backend_setting(monkeypatch, model_dir, machine):
    folder, probe = model_dir
    monkeypatch.setenv(cb.ENV_VAR, cb.CPU)
    tried = _routes(monkeypatch, {fe.OPENVINO_GPU: probe, fe.OPENVINO_CPU: probe})
    assert fe.load(log=lambda m: None, model_dir=str(folder)).route == fe.OPENVINO_CPU
    assert tried == [fe.OPENVINO_CPU]


def test_missing_model_is_none_with_a_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(fe, "model_dir_candidates", lambda: [str(tmp_path / "nowhere")])
    logged = []
    assert fe.load(cb.AUTO, log=logged.append) is None
    assert "not installed" in logged[-1]


@pytest.mark.parametrize("change, words", [
    ({"id": "another-encoder"}, "this app uses"),
    ({"format": fe.FORMAT + 1}, "newer"),
    ({"dims": 512}, "768"),
])
def test_meta_for_another_encoder_is_refused(model_dir, change, words):
    folder, _ = model_dir
    meta = json.loads((folder / fe.META_FILE).read_text(encoding="utf-8"))
    meta.update(change)
    (folder / fe.META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match=words):
        fe.read_meta(str(folder))
    assert fe.load(cb.AUTO, log=lambda m: None, model_dir=str(folder)) is None


def test_find_model_dir_honours_the_override(monkeypatch, model_dir):
    folder, _ = model_dir
    monkeypatch.setenv(fe.DIR_ENV, str(folder))
    assert fe.model_dir_candidates()[0] == str(folder)
    assert fe.find_model_dir() == str(folder)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

class CountingRunner:
    """Returns each frame's first pixel value as its vector, so order shows."""

    def __init__(self):
        self.calls = []

    def run(self, pixels):
        self.calls.append(len(pixels))
        return np.repeat(pixels[:, :1, 0, 0], fe.DIMS, axis=1)


def test_encode_batches_and_keeps_order(cv2):
    runner = CountingRunner()
    enc = fe.FrameEncoder(runner, fe.OPENVINO_CPU, "x", batch=3)
    frames = [np.full((40, 40, 3), v, np.uint8) for v in range(0, 70, 10)]
    out = enc.encode_bgr(frames)
    assert runner.calls == [3, 3, 1]
    assert out.shape == (7, fe.DIMS) and out.dtype == np.float32
    np.testing.assert_allclose(out[:, 0], [v / 127.5 - 1 for v in range(0, 70, 10)], atol=1e-6)


def test_encode_nothing():
    enc = fe.FrameEncoder(CountingRunner(), fe.OPENVINO_CPU, "x")
    assert enc.encode_bgr([]).shape == (0, fe.DIMS)
    assert enc.encode_pixels(np.zeros((0, 3, 256, 256), np.float32)).shape == (0, fe.DIMS)


# ---------------------------------------------------------------------------
# The real model, when there is one
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.environ.get(fe.DIR_ENV), reason=f"set {fe.DIR_ENV} to an export")
def test_real_model_on_every_route_here():
    """Every route the test process can reach reproduces the export's
    reference vector, and the rest are skipped by load(), not failed. (The
    suite shims OpenVINO, so under pytest this is ONNX Runtime;
    tools/export_frame_encoder.py checks OpenVINO at export.)"""
    meta = fe.read_meta(os.environ[fe.DIR_ENV])
    for backend in (cb.AUTO, cb.INTEL, cb.DIRECTML, cb.CPU):
        enc = fe.load(backend, log=print)
        assert enc is not None
        v = enc.encode_pixels(np.repeat(fe.probe_pixels(), 3, axis=0))
        assert v.shape == (3, fe.DIMS)
        assert min(fe._cosine(x, meta["probe"]) for x in v) > 0.999
