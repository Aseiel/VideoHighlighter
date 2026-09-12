"""R3D through ONNX Runtime: the path that gives a packaged build its GPU back.

`torch-directml` cannot be bundled — it pins an exact torch, and the release
build ships the CUDA one — so on a DX12 card the exe had action recognition on
the processor with no way for the user to change it. `modules/r3d_onnx.py`
routes the model through the runtime that *is* in the bundle.

Nothing here needs an AMD card, ONNX Runtime, or a real R3D. What the module
decides is which file to write, when a cached one has gone stale, and when to
refuse the GPU and hand the caller back its torch model — and all four are
decisions about strings and return values.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

from modules import r3d_onnx


@pytest.fixture(autouse=True)
def cache_in_tmp(monkeypatch, tmp_path):
    """Exports land in a temp dir, never in the developer's user-data folder."""
    monkeypatch.setattr(r3d_onnx.app_paths, "user_data_dir", lambda: str(tmp_path))
    return tmp_path


def _set_ort(monkeypatch, *, available=True, provider=None, session=None):
    """Pin what ONNX Runtime answers, rather than asking this machine."""
    ort = r3d_onnx.ort_directml
    monkeypatch.setattr(ort, "available", lambda: available)
    if session is not None:
        monkeypatch.setattr(ort, "session", lambda path, **kw: session)
    monkeypatch.setattr(
        ort, "session_backend",
        lambda sess: provider if provider is not None else ort.CPU_PROVIDER)


class _FakeSession:
    """The three things :class:`OnnxR3D` asks of an InferenceSession."""

    def __init__(self, logits=None, input_name="input"):
        self.logits = np.zeros((1, 400), dtype=np.float32) if logits is None else logits
        self.input_name = input_name
        self.fed = []

    def get_inputs(self):
        return [types.SimpleNamespace(name=self.input_name)]

    def run(self, _outputs, feeds):
        self.fed.append(feeds)
        return [self.logits]


def _fake_torch(monkeypatch, writes=True):
    """A torch whose `onnx.export` writes a file, or writes part of one and
    then raises — which is how the real exporter fails partway through a graph.
    """
    def export(model, args, path, **kwargs):
        with open(path, "wb") as fh:
            fh.write(b"onnx")
        if not writes:
            raise RuntimeError("Unsupported: aten::conv3d")

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    module = types.ModuleType("torch")
    module.zeros = lambda *shape: object()
    module.no_grad = _NoGrad
    module.onnx = types.SimpleNamespace(export=export)
    monkeypatch.setitem(sys.modules, "torch", module)
    return module


# ---------------------------------------------------------------------------
# Which file an export belongs in
# ---------------------------------------------------------------------------

def test_class_count_is_in_the_name():
    """The class count changes the final layer, so it changes the graph. A
    cache keyed only on the model name would hand a 31-class export to a
    400-class run and the logits would be silently the wrong length."""
    assert r3d_onnx.export_path("r3d_18", 400).endswith("r3d_18-400c.onnx")
    assert r3d_onnx.export_path("r3d_18", 31).endswith("r3d_18-31c.onnx")


def test_two_custom_models_do_not_collide():
    """Two imported models with the same variant and the same class count are
    still two different models."""
    a = r3d_onnx.export_path("r3d_18", 31, custom_weights="/x/dance.pth")
    b = r3d_onnx.export_path("r3d_18", 31, custom_weights="/x/sports.pth")
    assert a != b
    assert os.path.basename(a) == "r3d_18-dance-31c.onnx"


def test_a_missing_cache_dir_is_not_a_lost_gpu(monkeypatch, tmp_path):
    """`makedirs` failing costs the subdirectory, not the feature."""
    monkeypatch.setattr(r3d_onnx.os, "makedirs",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    assert r3d_onnx.cache_dir() == str(tmp_path)


# ---------------------------------------------------------------------------
# When a cached export is no longer the model
# ---------------------------------------------------------------------------

def test_retraining_invalidates_the_export(tmp_path):
    """`import_r3d_action_model` overwrites r3d_finetuned.pth in place, so a
    user who retrains keeps the same filename. Without the mtime check they
    would keep running the previous model for ever, with nothing reporting it."""
    onnx = tmp_path / "m.onnx"
    weights = tmp_path / "w.pth"
    onnx.write_bytes(b"old")
    weights.write_bytes(b"new")
    os.utime(str(onnx), (1_000_000, 1_000_000))
    os.utime(str(weights), (2_000_000, 2_000_000))

    assert r3d_onnx._is_stale(str(onnx), str(weights)) is True


def test_built_in_weights_are_never_stale(tmp_path):
    """torchvision's weights are immutable for a given model name."""
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"x")
    assert r3d_onnx._is_stale(str(onnx), None) is False


def test_an_up_to_date_export_is_reused(monkeypatch, cache_in_tmp):
    """The export costs real seconds, so a second run must not repeat it."""
    _fake_torch(monkeypatch)
    path = r3d_onnx.export_path("r3d_18", 400)
    with open(path, "wb") as fh:
        fh.write(b"cached")

    calls = []
    monkeypatch.setattr(sys.modules["torch"].onnx, "export",
                        lambda *a, **k: calls.append(1))

    assert r3d_onnx.ensure_export(object(), "r3d_18", 400, log=lambda *a: None) == path
    assert calls == []


# ---------------------------------------------------------------------------
# Export failures are fallbacks, not faults
# ---------------------------------------------------------------------------

def test_a_failed_export_leaves_no_half_written_file(monkeypatch, cache_in_tmp):
    """A partial .onnx would be picked up as valid on the next run — and then
    fail at load, on a machine where the real fault happened a week ago."""
    _fake_torch(monkeypatch, writes=False)
    path = r3d_onnx.export_path("r3d_18", 400)

    logged = []
    assert r3d_onnx.ensure_export(object(), "r3d_18", 400,
                                  log=logged.append) is None
    assert not os.path.exists(path)
    assert any("export failed" in line for line in logged)


def test_no_onnx_runtime_means_keep_torch(monkeypatch):
    _set_ort(monkeypatch, available=False)
    assert r3d_onnx.load(object(), "r3d_18", 400, log=lambda *a: None) is None


# ---------------------------------------------------------------------------
# Refusing a GPU that is not one
# ---------------------------------------------------------------------------

def test_a_cpu_provider_session_is_declined(monkeypatch, cache_in_tmp):
    """ONNX Runtime drops a provider it cannot initialise without saying so. A
    session that quietly landed on the processor is not faster than the torch
    model it would displace, and torch is the better-tested of the two."""
    _fake_torch(monkeypatch)
    session = _FakeSession()
    _set_ort(monkeypatch, available=True, session=session,
             provider=r3d_onnx.ort_directml.CPU_PROVIDER)

    logged = []
    assert r3d_onnx.load(object(), "r3d_18", 400, log=logged.append) is None
    assert any("fell back to the processor" in line for line in logged)


def test_a_directml_session_is_taken(monkeypatch, cache_in_tmp):
    _fake_torch(monkeypatch)
    session = _FakeSession()
    _set_ort(monkeypatch, available=True, session=session,
             provider=r3d_onnx.ort_directml.PROVIDER)

    runner = r3d_onnx.load(object(), "r3d_18", 400, log=lambda *a: None)
    assert runner is not None
    assert runner.on_gpu is True
    # The dummy pass in load() is the point: a 3D CNN that DirectML cannot run
    # has to fail here, not an hour into a job.
    assert session.fed and session.fed[0]["input"].shape == (
        1, 3, r3d_onnx.CLIP_LENGTH, r3d_onnx.INPUT_SIZE, r3d_onnx.INPUT_SIZE)


def test_a_session_that_will_not_start_is_a_fallback(monkeypatch, cache_in_tmp):
    _fake_torch(monkeypatch)

    class _Boom:
        def get_inputs(self):
            raise RuntimeError("DML device removed")

    _set_ort(monkeypatch, available=True, session=_Boom(),
             provider=r3d_onnx.ort_directml.PROVIDER)
    logged = []
    assert r3d_onnx.load(object(), "r3d_18", 400, log=logged.append) is None
    assert any("failed to start" in line for line in logged)


# ---------------------------------------------------------------------------
# The runner itself
# ---------------------------------------------------------------------------

def test_predict_returns_flat_float32_logits():
    """The torch path returns `output.cpu().float().numpy().flatten()`, and every
    caller reads it as a flat array. The swap is only invisible if this matches."""
    session = _FakeSession(logits=np.arange(12, dtype=np.float64).reshape(1, 12))
    runner = r3d_onnx.OnnxR3D("model.onnx", session=session)

    out = runner.predict(np.zeros((1, 3, 16, 112, 112), dtype=np.float32))

    assert out.shape == (12,)
    assert out.dtype == np.float32


def test_the_input_is_fed_under_the_graphs_own_name():
    """Reading the name off the graph rather than assuming "input" is what lets
    an export made by a different torch version still load."""
    session = _FakeSession(input_name="clip")
    runner = r3d_onnx.OnnxR3D("model.onnx", session=session)
    runner.predict(np.zeros((1, 3, 16, 112, 112), dtype=np.float32))
    assert "clip" in session.fed[0]
