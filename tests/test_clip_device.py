"""Tests for CLIP's backend/device resolution.

No CLIP and no GPU here: `resolve_device` is pure logic over a single probe,
and `llm.clip_prefilter` imports nothing heavier than stdlib at module scope.
So the rule that decides whether a machine gets its GPU at all is testable in
milliseconds, on any machine, including CI with no accelerator.

The probe is monkeypatched rather than mocked at the torch level: `cuda_device`
is the module's one contract with the hardware, so faking it covers an NVIDIA
box from a machine that has never seen one.
"""

from __future__ import annotations

import pytest

import llm.clip_prefilter as cp
from llm.clip_prefilter import resolve_device


@pytest.fixture
def nvidia(monkeypatch):
    """A machine with an NVIDIA GPU."""
    monkeypatch.setattr(cp, "cuda_device", lambda: "cuda:0")


@pytest.fixture
def no_nvidia(monkeypatch):
    """A machine without one (Intel box, or a torch with no CUDA compiled in)."""
    monkeypatch.setattr(cp, "cuda_device", lambda: None)


# -- the reason this module exists ------------------------------------------

@pytest.mark.parametrize("requested", ["AUTO", "GPU", "gpu", "auto", " GPU "])
def test_auto_prefers_cuda_when_present(nvidia, requested):
    """The regression this whole backend exists for: "GPU" is what every caller
    passes, and OpenVINO's "GPU" means *Intel*, so on an NVIDIA box it used to
    land on the CPU."""
    assert resolve_device(requested) == ("torch", "cuda:0")


@pytest.mark.parametrize("requested", ["AUTO", "GPU", "gpu", "auto", " GPU "])
def test_auto_falls_back_to_openvino_gpu(no_nvidia, requested):
    assert resolve_device(requested) == ("openvino", "GPU")


# -- explicit requests ------------------------------------------------------

def test_explicit_cuda(nvidia):
    assert resolve_device("CUDA") == ("torch", "cuda:0")
    assert resolve_device("cuda") == ("torch", "cuda:0")


def test_explicit_ordinal_is_honoured(nvidia):
    """cuda:1 must reach the second GPU, not get flattened to the probe's cuda:0."""
    assert resolve_device("cuda:1") == ("torch", "cuda:1")
    assert resolve_device("CUDA:2") == ("torch", "cuda:2")


def test_cuda_without_a_cuda_gpu_falls_back(no_nvidia):
    """Asking for CUDA on a machine without it degrades rather than raising."""
    assert resolve_device("CUDA") == ("openvino", "GPU")


def test_explicit_cpu_is_never_hijacked(nvidia):
    """Someone who says CPU means CPU, even sitting on an NVIDIA GPU — the
    escape hatch when a backend misbehaves."""
    assert resolve_device("CPU") == ("openvino", "CPU")


@pytest.mark.parametrize("requested", ["CPU", "GPU.1", "NPU", "AUTO:GPU,CPU", "HETERO:GPU,CPU"])
def test_openvino_device_strings_pass_through_untouched(nvidia, requested):
    """OpenVINO has its own device grammar; anything we don't claim is its own."""
    assert resolve_device(requested) == ("openvino", requested)


@pytest.mark.parametrize("requested", ["", None])
def test_empty_request_is_auto(no_nvidia, requested):
    assert resolve_device(requested) == ("openvino", "GPU")


# -- the probe itself -------------------------------------------------------
#
# These install an explicit fake torch rather than leaning on the ambient one:
# conftest shims torch with a MagicMock when it isn't already imported, so a real
# torch (dev box) and a mock (CI) would otherwise take different paths here.

def _fake_torch(monkeypatch, **cuda_attrs):
    import sys

    cuda = type("cuda", (), {k: staticmethod(v) for k, v in cuda_attrs.items()})
    torch = type("torch", (), {"cuda": cuda})
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def test_cuda_probe_finds_a_gpu(monkeypatch):
    _fake_torch(monkeypatch, is_available=lambda: True, device_count=lambda: 1)
    assert cp.cuda_device() == "cuda:0"


def test_cuda_probe_returns_none_without_cuda(monkeypatch):
    """The ordinary Intel/CPU box: a torch that simply has no CUDA."""
    _fake_torch(monkeypatch, is_available=lambda: False, device_count=lambda: 0)
    assert cp.cuda_device() is None


def test_cuda_probe_survives_a_broken_torch(monkeypatch):
    """A driver/library mismatch raises from is_available(). That means "no
    CUDA", not a crashed search."""
    def boom():
        raise RuntimeError("driver/library version mismatch")

    _fake_torch(monkeypatch, is_available=boom, device_count=lambda: 1)
    assert cp.cuda_device() is None


def test_cuda_probe_survives_no_torch_at_all(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def spy(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", spy)
    assert cp.cuda_device() is None


def test_device_name_never_raises(monkeypatch):
    """It's a log string; a failed lookup must not cost us the GPU."""
    def boom(_device):
        raise RuntimeError("invalid device ordinal")

    _fake_torch(monkeypatch, get_device_name=boom)
    assert cp._device_name("cuda:9") == "cuda:9"


def test_device_name_reports_the_gpu(monkeypatch):
    _fake_torch(monkeypatch, get_device_name=lambda d: "NVIDIA GeForce RTX 4090")
    assert cp._device_name("cuda:0") == "NVIDIA GeForce RTX 4090"


# -- dependency reporting ---------------------------------------------------

@pytest.fixture
def import_spy(monkeypatch):
    """Record every module import_error() probes, with optimum-intel absent."""
    import builtins

    real_import = builtins.__import__
    seen: list[str] = []

    def spy(name, *args, **kwargs):
        seen.append(name)
        if name == "optimum.intel":
            raise ImportError("optimum-intel is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", spy)
    return seen


def test_torch_backend_does_not_require_optimum(nvidia, import_spy):
    """A CUDA-only install without optimum-intel must not be told CLIP is
    unavailable — the torch backend never touches optimum."""
    err = cp.ClipFramePrefilter.import_error("AUTO")

    assert err is None, f"torch backend wrongly blocked: {err}"
    assert "optimum.intel" not in import_spy
    assert "torch" in import_spy


def test_openvino_backend_still_reports_missing_optimum(no_nvidia, import_spy):
    """The converse: without CUDA, optimum is genuinely required, and its
    absence must be named rather than swallowed."""
    err = cp.ClipFramePrefilter.import_error("AUTO")

    assert err is not None and "optimum.intel" in err
