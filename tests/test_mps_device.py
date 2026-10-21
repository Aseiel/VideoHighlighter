"""torch on the Apple GPU (Metal): who may use it, and the two models that do.

CLIP and Whisper run through torch, and on a Mac they ran on the processor
although torch could reach the GPU: nothing here asked for "mps". These pin
the decision (macOS, Metal present, a backend that allows it) and how each
model takes it — and gives it back when Metal cannot run something.
"""
from __future__ import annotations

import sys
import types

import pytest

from modules.system import compute_backend, mps_device


@pytest.fixture()
def mac(monkeypatch):
    """A Mac whose torch reaches Metal, on the automatic backend."""
    torch = types.SimpleNamespace(
        backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.delenv(mps_device.MODE_ENV, raising=False)
    monkeypatch.setattr(compute_backend, "configured", lambda: compute_backend.AUTO)
    return torch


class TestWhoMayUseMetal:
    def test_a_mac_with_metal_gets_it(self, mac):
        assert mps_device.device() == "mps"

    def test_not_off_macos(self, mac, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        assert mps_device.device() is None

    def test_not_without_metal(self, mac):
        mac.backends.mps.is_available = lambda: False
        assert mps_device.device() is None
        assert "no Metal" in mps_device.unavailable_reason()

    def test_processor_only_means_torch_too(self, mac, monkeypatch):
        monkeypatch.setattr(compute_backend, "configured", lambda: compute_backend.CPU)
        assert mps_device.device() is None

    def test_the_apple_backend_allows_it(self, mac, monkeypatch):
        monkeypatch.setattr(compute_backend, "configured", lambda: compute_backend.APPLE)
        assert mps_device.device() == "mps"

    def test_a_switch_turns_it_off_alone(self, mac, monkeypatch):
        monkeypatch.setenv(mps_device.MODE_ENV, "off")
        assert mps_device.device() is None

    def test_cpu_fallback_is_set_on_a_mac_only_once_and_never_overridden(self, monkeypatch):
        monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
        monkeypatch.setattr(sys, "platform", "win32")
        mps_device.enable_cpu_fallback()
        assert "PYTORCH_ENABLE_MPS_FALLBACK" not in __import__("os").environ
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
        mps_device.enable_cpu_fallback()
        assert __import__("os").environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"


class TestClip:
    def test_automatic_picks_metal_on_a_mac(self, mac, monkeypatch):
        from llm import clip_prefilter as cp
        monkeypatch.setattr(cp, "cuda_device", lambda: None)
        monkeypatch.setattr(cp, "directml_device", lambda: None)
        assert cp.resolve_device("AUTO") == ("torch", "mps")

    def test_cuda_still_comes_first(self, mac, monkeypatch):
        from llm import clip_prefilter as cp
        monkeypatch.setattr(cp, "cuda_device", lambda: "cuda:0")
        assert cp.resolve_device("AUTO") == ("torch", "cuda:0")

    def test_asking_for_metal_without_it_is_the_processor(self, mac, monkeypatch):
        from llm import clip_prefilter as cp
        monkeypatch.setenv(mps_device.MODE_ENV, "off")
        assert cp.resolve_device("mps") == ("openvino", "CPU")

    def test_a_build_with_torch_weights_only_needs_no_optimum(self, monkeypatch, tmp_path):
        from llm import clip_prefilter as cp
        monkeypatch.setattr(cp, "_bundled_ov_dir", lambda: None)
        monkeypatch.setattr(cp, "_bundled_torch_dir", lambda: str(tmp_path))
        monkeypatch.setattr(cp, "resolve_device", lambda d: ("openvino", "CPU"))
        monkeypatch.setitem(sys.modules, "optimum.intel", None)   # import fails
        assert cp.ClipFramePrefilter.import_error("CPU") is None


class _Model:
    def __init__(self, fail_to=None):
        self._buffers = {"alignment_heads": "sparse-on-cpu"}
        self.fail_to = fail_to
        self.device = "cpu"
        self.buffers_moved = None

    def to(self, device):
        if self.fail_to == device:
            raise RuntimeError("no sparse tensors on Metal")
        self.device = device
        self.buffers_moved = dict(self._buffers)
        return self


class TestWhisper:
    def test_metal_keeps_the_sparse_buffer_on_the_processor(self, monkeypatch):
        from modules.audio import transcript as tr
        monkeypatch.setattr(mps_device, "is_mps", lambda d: d == "mps")
        loads = []
        monkeypatch.setattr(tr.whisper, "load_model",
                            lambda name, device: loads.append(device) or _Model())
        model, device = tr.load_whisper("base", "mps")
        assert (device, model.device, loads) == ("mps", "mps", ["cpu"])
        assert model.buffers_moved == {}          # moved without it...
        assert model._buffers["alignment_heads"] == "sparse-on-cpu"   # ...then put back

    def test_a_failed_move_is_the_processor_not_an_error(self, monkeypatch):
        from modules.audio import transcript as tr
        lines = []
        monkeypatch.setattr(tr.whisper, "load_model",
                            lambda name, device: _Model(fail_to="mps"))
        model, device = tr.load_whisper("base", "mps", lines.append)
        assert device == "cpu"
        assert "Apple GPU" in lines[0]

    def test_other_devices_load_as_before(self, monkeypatch):
        from modules.audio import transcript as tr
        loads = []
        monkeypatch.setattr(tr.whisper, "load_model",
                            lambda name, device: loads.append(device) or _Model())
        assert tr.load_whisper("base", "cuda")[1] == "cuda"
        assert loads == ["cuda"]

    def test_metal_decodes_in_full_precision_unless_asked(self, monkeypatch):
        from modules.audio import transcript as tr
        monkeypatch.delenv("VH_WHISPER_MPS_FP16", raising=False)
        assert tr._mps_fp16() is False
        monkeypatch.setenv("VH_WHISPER_MPS_FP16", "1")
        assert tr._mps_fp16() is True
