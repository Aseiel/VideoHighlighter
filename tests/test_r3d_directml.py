"""R3D on DirectML: the device it lands on, and what happens when it can't run.

R3D is a 3D CNN, and 3D convolution is the least certain part of DirectML's
operator coverage. Nobody here has an AMD card, so "does conv3d work on
DirectML" is not a question this file can answer and does not try to. What it
pins is the thing that decides how bad the answer is if it turns out to be no:

  * the device string reaching the wrapper at all (it used to be discarded — the
    old `torch.device(d if torch.cuda.is_available() else 'cpu')` collapsed every
    non-NVIDIA machine to the processor, silently), and
  * the warm-up demoting the model to the CPU instead of letting an
    "operator is not currently implemented" out into a run.

The second is why enabling this is defensible at all: the worst case is the
behaviour the app already had, plus one line of explanation.

`action_recognition` imports torch and torchvision at module scope, so this
file leans on conftest's shims and never touches a real model.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest

from modules import directml_device as dml
from tests.test_directml_device import FakeTorchDirectML


@pytest.fixture(autouse=True)
def clean_module_state(monkeypatch):
    monkeypatch.delenv(dml.MODE_ENV, raising=False)
    dml.set_mode(None)
    yield
    # set_mode(None) nulls the probe cache. Deliberately NOT probe(refresh=True):
    # fixture teardown runs before monkeypatch unwinds, so re-probing here would
    # cache the *fake* AMD box and hand it to every later test in the session.
    dml.set_mode(None)


@pytest.fixture
def amd_box(monkeypatch):
    fake = FakeTorchDirectML(names=("AMD Radeon RX 570",))
    monkeypatch.setattr(dml, "_import_torch_directml", lambda: fake)
    # The wrapper imports the package itself, for the backend registration.
    monkeypatch.setitem(sys.modules, "torch_directml", types.ModuleType("torch_directml"))
    dml.probe(refresh=True)
    return fake


@pytest.fixture
def no_directml(monkeypatch):
    def boom():
        raise ImportError("No module named 'torch_directml'")
    monkeypatch.setattr(dml, "_import_torch_directml", boom)
    monkeypatch.delitem(sys.modules, "torch_directml", raising=False)
    dml.probe(refresh=True)


@pytest.fixture
def ar(monkeypatch):
    """`action_recognition` with a torch whose devices we control.

    `torch.device` has to be real enough to carry `.type`, because that is what
    the fp16 decision and the cleanup path read.

    The module has to be *borrowed* rather than simply imported.
    `tests/test_pipeline_legacy_imports.py` installs a MagicMock under
    "action_recognition" with `sys.modules.setdefault` and never takes it out
    again, so every test file that sorts after it inherits the shim — this one
    included, at which point `ar._resolve_r3d_device(...)` returns a MagicMock
    and every assertion below compares two pieces of nonsense. So: drop the
    shim, import the real module, and hand the shim straight back on teardown,
    the way conftest's `real_opencv()` does for cv2.
    """
    import importlib

    if isinstance(sys.modules.get("action_recognition"), MagicMock):
        monkeypatch.delitem(sys.modules, "action_recognition")
    module = importlib.import_module("action_recognition")

    class Device:
        def __init__(self, spec):
            self.spec = str(spec)
            if ":" in self.spec and not self.spec.split(":")[0].isdigit():
                self.type = self.spec.split(":")[0]
            else:
                self.type = self.spec

        def __eq__(self, other):
            return str(other) == self.spec

        def __repr__(self):
            return self.spec

    stub = types.SimpleNamespace(
        device=Device,
        cuda=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setattr(module, "torch", stub)
    return module


# ---------------------------------------------------------------------------
# The line that made an AMD card impossible
# ---------------------------------------------------------------------------

def test_a_directml_device_survives_to_the_model(ar, amd_box):
    """The regression this whole change exists for. The previous form of
    `_resolve_r3d_device` read `torch.device(d if torch.cuda.is_available() else
    'cpu')`, so on any non-NVIDIA machine the requested device was thrown away
    without a word."""
    assert str(ar._resolve_r3d_device("privateuseone:0")) == "privateuseone:0"
    assert str(ar._resolve_r3d_device("dml")) == "privateuseone:0"


def test_cpu_is_still_the_cpu(ar, amd_box):
    """"R3D + CPU (PyTorch, slow)" must mean the CPU on an AMD box too — that
    label is the reason the device is passed explicitly rather than taken from
    whatever the machine reports."""
    assert str(ar._resolve_r3d_device("cpu")) == "cpu"


def test_cuda_without_cuda_still_degrades(ar, amd_box):
    assert str(ar._resolve_r3d_device("cuda")) == "cpu"


def test_directml_without_the_package_degrades_to_cpu(ar, no_directml):
    assert str(ar._resolve_r3d_device("dml")) == "cpu"


def test_an_unparseable_device_is_not_a_crash(ar, amd_box, monkeypatch):
    """A stale config or a typo'd CLI flag must cost the GPU, not the run."""
    real_device = ar.torch.device

    def picky(spec):
        if spec == "cpu":
            return real_device("cpu")
        raise RuntimeError(f"Expected one of cpu, cuda, ... device type at start "
                           f"of device string: {spec}")

    monkeypatch.setattr(ar.torch, "device", picky)
    assert str(ar._resolve_r3d_device("nonsense")) == "cpu"


# ---------------------------------------------------------------------------
# The warm-up, which is what makes enabling this safe
# ---------------------------------------------------------------------------

class _Wrapper:
    """The parts of R3DModelWrapper the warm-up touches, without a model."""

    def __init__(self, ar_module, device, fails_on_directml=True):
        self._ar = ar_module
        self.model_name = "r3d_18"
        self.device = ar_module.torch.device(device)
        self.half = False
        self.fails_on_directml = fails_on_directml
        self.placements = []
        self.forward_calls = []

    _place_on_device = lambda self: self.placements.append(str(self.device))  # noqa: E731

    def _forward_dummy(self):
        self.forward_calls.append(str(self.device))
        if self.fails_on_directml and self._ar._is_directml_device(self.device):
            raise RuntimeError("the operator aten::conv3d is not currently "
                               "implemented for the DirectML backend")
        return object()


def _warmup(ar_module, wrapper):
    return ar_module.R3DModelWrapper._warmup(wrapper)


def test_an_unimplemented_operator_demotes_to_cpu(ar, amd_box):
    """The failure surfaces at load, on a dummy clip of the real shape, instead
    of an hour into a job on the first real one."""
    w = _Wrapper(ar, "privateuseone:0")
    _warmup(ar, w)
    assert str(w.device) == "cpu"
    assert w.half is False
    assert w.forward_calls == ["privateuseone:0", "cpu"]
    assert w.placements == ["cpu"], "the model must actually be moved, not just relabelled"


def test_a_working_directml_stays_on_directml(ar, amd_box):
    w = _Wrapper(ar, "privateuseone:0", fails_on_directml=False)
    _warmup(ar, w)
    assert str(w.device) == "privateuseone:0"
    assert w.placements == [], "a working device must not be moved"


def test_a_cuda_failure_is_still_a_failure(ar, amd_box):
    """A broken warm-up on CUDA is a fault to be seen. Swallowing it here would
    turn every real bug into a silent slow run."""
    class Boom(_Wrapper):
        def _forward_dummy(self):
            raise RuntimeError("CUDA error: device-side assert triggered")

    with pytest.raises(RuntimeError, match="device-side assert"):
        _warmup(ar, Boom(ar, "cuda"))


# ---------------------------------------------------------------------------
# Which backend the two entry points choose
# ---------------------------------------------------------------------------

def test_on_demand_runs_send_r3d_to_directml(monkeypatch, amd_box):
    """The viewer's runs and a full pipeline run must agree, or a cache built by
    one is a cache the other would not have produced."""
    from modules import analysis_ondemand as ao
    from modules import device_utils as du
    monkeypatch.setattr(du, "_TORCH_AVAILABLE", False)

    enable, half, device = ao._r3d_flags("auto", log=lambda *a, **k: None)
    assert (enable, half, device) == (True, False, "privateuseone:0")


def test_on_demand_auto_without_directml_is_unchanged(monkeypatch, no_directml):
    from modules import analysis_ondemand as ao
    from modules import device_utils as du
    monkeypatch.setattr(du, "_TORCH_AVAILABLE", False)

    assert ao._r3d_flags("auto", log=lambda *a, **k: None) == (False, False, None)


@pytest.mark.parametrize("backend,expected", [
    ("openvino", (False, False, None)),
    ("r3d_cuda", (True, True, "cuda")),
    ("r3d_cpu", (True, False, "cpu")),
])
def test_explicit_backend_choices_name_their_device(backend, expected, amd_box):
    """Each choice pins its own device, so "R3D + CPU" cannot silently become
    DirectML on an AMD machine just because one is present."""
    from modules import analysis_ondemand as ao
    assert ao._r3d_flags(backend, log=lambda *a, **k: None) == expected
