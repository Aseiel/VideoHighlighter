"""Tests for the experimental DirectML backend.

No AMD card and no torch-directml here, and that is the point: the whole module
is written around a single seam (`_import_torch_directml`), so an RX 570 can be
simulated from any machine and the routing that decides whether somebody gets
their GPU at all is testable in milliseconds — including on CI, which has
neither a GPU nor the package.

The stub below is deliberately faithful to the two things that actually vary
between torch-directml releases: the backend name (`dml` in 0.1.x,
`privateuseone` in 0.2.x) and whether `device_count`/`device_name` exist at all.
Those are the parts of the real package this code is allowed to depend on.
"""

from __future__ import annotations

import sys

import pytest

from modules import directml_device as dml


class FakeTorchDirectML:
    """A stand-in for the `torch_directml` package."""

    __version__ = "0.2.5.dev240914"

    def __init__(self, names=("AMD Radeon RX 570",), backend="privateuseone",
                 available=True):
        self._names = list(names)
        self._backend = backend
        self._available = available

    def is_available(self):
        return self._available

    def device_count(self):
        return len(self._names)

    def device_name(self, index):
        return self._names[index]

    def device(self, index=0):
        # The real one returns a torch.device whose str() is "backend:index".
        return f"{self._backend}:{index}"


@pytest.fixture(autouse=True)
def clean_module_state(monkeypatch):
    """Reset the process-wide mode override and probe cache around every test.

    Both are module-level by design (the probe is far too expensive to repeat
    and its answer cannot change mid-process), which makes leaking them between
    tests the obvious way for this file to start lying.
    """
    monkeypatch.delenv(dml.MODE_ENV, raising=False)
    monkeypatch.delenv(dml.FP16_ENV, raising=False)
    dml.set_mode(None)
    yield
    dml.set_mode(None)


@pytest.fixture
def amd_box(monkeypatch):
    """A machine with a DirectML-capable AMD card."""
    def install(**kwargs):
        fake = FakeTorchDirectML(**kwargs)
        monkeypatch.setattr(dml, "_import_torch_directml", lambda: fake)
        dml.probe(refresh=True)
        return fake
    return install


@pytest.fixture
def no_directml(monkeypatch):
    """A machine where torch-directml is not installed."""
    def boom():
        raise ImportError("No module named 'torch_directml'")
    monkeypatch.setattr(dml, "_import_torch_directml", boom)
    dml.probe(refresh=True)


# -- the opt-in --------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("", dml.MODE_AUTO),
    ("auto", dml.MODE_AUTO),
    ("0", dml.MODE_OFF),
    ("off", dml.MODE_OFF),
    ("false", dml.MODE_OFF),
    ("1", dml.MODE_FORCE),
    ("force", dml.MODE_FORCE),
    ("  FORCE  ", dml.MODE_FORCE),
])
def test_mode_reads_the_environment(monkeypatch, value, expected):
    monkeypatch.setenv(dml.MODE_ENV, value)
    assert dml.mode() == expected


def test_an_unrecognised_mode_is_the_default_not_an_error(monkeypatch):
    """A typo in an environment variable must cost the default behaviour, not
    a crash on a code path that runs before the UI exists."""
    monkeypatch.setenv(dml.MODE_ENV, "yes-please")
    assert dml.mode() == dml.MODE_AUTO


def test_set_mode_overrides_the_environment(monkeypatch):
    monkeypatch.setenv(dml.MODE_ENV, "force")
    dml.set_mode("off")
    assert dml.mode() == dml.MODE_OFF
    dml.set_mode(None)
    assert dml.mode() == dml.MODE_FORCE


def test_switching_off_takes_effect_after_a_probe(amd_box):
    """The regression a cached probe invites: a settings toggle that reports
    success while the old answer is still being handed out."""
    amd_box()
    assert dml.is_available()
    dml.set_mode("off")
    assert not dml.is_available()
    assert "disabled" in dml.unavailable_reason()


def test_off_means_the_package_is_never_imported(monkeypatch):
    """Not just a policy check: importing torch_directml replaces torch's
    device registrations, so "off" has to mean the import does not happen."""
    def boom():
        raise AssertionError("torch_directml must not be imported when off")
    monkeypatch.setattr(dml, "_import_torch_directml", boom)
    dml.set_mode("off")
    assert dml.probe(refresh=True).available is False


# -- the probe ---------------------------------------------------------------

def test_a_missing_package_is_a_reason_not_an_exception(no_directml):
    probe = dml.probe()
    assert probe.available is False
    assert "not installed" in probe.reason
    assert dml.device_string() is None
    assert dml.adapter_names() == []


def test_a_packaged_build_is_told_what_it_actually_has(no_directml, monkeypatch):
    """Inside an exe there is no pip, so "not installed" reads as an instruction
    the reader cannot follow — and it is not the whole truth either, because
    detection still reaches the card through ONNX Runtime."""
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    reason = dml.probe(refresh=True).reason

    assert "not installed" not in reason
    assert "ONNX Runtime" in reason
    assert "docs/AMD-GPU.md" in reason


def test_an_installed_package_with_no_device_says_so_differently(amd_box):
    """The two ways this fails need different answers from the user — one is a
    pip install, the other is a driver or a card — so they must not collapse
    into one message."""
    amd_box(available=False)
    reason = dml.unavailable_reason()
    assert "no DirectML device" in reason
    assert "not installed" not in reason


def test_a_probe_that_raises_is_still_an_answer(monkeypatch):
    class Exploding:
        def is_available(self):
            raise RuntimeError("d3d12 device removed")
    monkeypatch.setattr(dml, "_import_torch_directml", lambda: Exploding())
    probe = dml.probe(refresh=True)
    assert probe.available is False
    assert "d3d12 device removed" in probe.reason


def test_adapters_are_named(amd_box):
    amd_box(names=("AMD Radeon RX 570", "Microsoft Basic Render Driver"))
    assert dml.adapter_names() == ["AMD Radeon RX 570",
                                   "Microsoft Basic Render Driver"]
    assert "RX 570" in dml.describe()
    assert "0.2.5" in dml.describe()


def test_a_nameless_adapter_does_not_lose_the_device(monkeypatch):
    class Nameless(FakeTorchDirectML):
        def device_name(self, index):
            raise RuntimeError("adapter has no description")
    monkeypatch.setattr(dml, "_import_torch_directml", lambda: Nameless())
    assert dml.probe(refresh=True).available is True
    assert dml.adapter_names() == ["DirectML device 0"]


# -- device strings ----------------------------------------------------------

def test_the_backend_name_comes_from_the_package_not_a_literal(amd_box):
    """torch-directml 0.1.x called the backend "dml" and 0.2.x calls it
    "privateuseone". Hardcoding either one means a device string torch rejects
    on the other, which surfaces as an unexplained failure inside a model load.
    """
    amd_box(backend="dml")
    assert dml.device_string(0) == "dml:0"
    amd_box(backend="privateuseone")
    assert dml.device_string(0) == "privateuseone:0"


@pytest.mark.parametrize("spec", ["dml", "DML", "directml", "DirectML",
                                  "privateuseone", "privateuseone:0", " dml "])
def test_every_spelling_a_user_might_type_is_recognised(spec):
    assert dml.is_directml(spec) is True


@pytest.mark.parametrize("spec", ["cuda", "cuda:0", "xpu", "cpu", "GPU", "", None])
def test_other_devices_are_not_mistaken_for_directml(spec):
    assert dml.is_directml(spec) is False


def test_normalize_maps_a_spelling_onto_what_torch_accepts(amd_box):
    amd_box(names=("AMD Radeon RX 570", "Intel(R) UHD Graphics"))
    assert dml.normalize("dml") == "privateuseone:0"
    assert dml.normalize("DirectML:1") == "privateuseone:1"
    assert dml.normalize("cuda:0") is None


def test_an_ordinal_past_the_last_adapter_is_none_not_a_bad_string(amd_box):
    """A string that fails later, inside a model load, produces a message about
    a device name rather than about the ordinal that was wrong."""
    amd_box(names=("AMD Radeon RX 570",))
    assert dml.normalize("dml:3") is None


def test_normalize_returns_none_when_there_is_no_directml(no_directml):
    assert dml.normalize("dml") is None


def test_a_garbled_ordinal_falls_back_to_the_first_device(amd_box):
    amd_box()
    assert dml.normalize("dml:banana") == "privateuseone:0"


# -- picking the right adapter -----------------------------------------------
#
# DirectML enumerates every DX12 adapter, and some of them are not graphics
# cards. Found by tools/simulate_directml.py's "software-renderer" scenario:
# the app was taking index 0 unconditionally, which on such a machine means
# running every model on a CPU implementation of D3D12 — slower than the CPU
# path it replaced, and silent.

def test_a_software_renderer_is_skipped_for_the_real_card(amd_box):
    amd_box(names=("Microsoft Basic Render Driver", "AMD Radeon RX 570"))
    assert dml.probe().preferred_index == 1
    assert dml.device_string() == "privateuseone:1"
    assert dml.normalize("dml") == "privateuseone:1"


def test_an_explicit_ordinal_still_wins(amd_box):
    """Skipping is a default, not a policy — somebody deliberately naming
    adapter 0 gets adapter 0."""
    amd_box(names=("Microsoft Basic Render Driver", "AMD Radeon RX 570"))
    assert dml.normalize("dml:0") == "privateuseone:0"


@pytest.mark.parametrize("name", [
    "Microsoft Basic Render Driver",
    "Microsoft Basic Display Adapter",
    "WARP Software Adapter",
])
def test_a_software_only_machine_reports_no_directml(amd_box, name):
    """Better than reporting a GPU that is slower than not having one."""
    amd_box(names=(name,))
    assert dml.is_available() is False
    assert "software renderer" in dml.unavailable_reason()
    assert dml.MODE_ENV in dml.unavailable_reason()


def test_force_still_allows_a_software_only_machine(monkeypatch, amd_box):
    """Testing the plumbing on a box with no real adapter is legitimate."""
    monkeypatch.setenv(dml.MODE_ENV, "force")
    dml.set_mode(None)
    amd_box(names=("Microsoft Basic Render Driver",))
    assert dml.is_available() is True
    assert dml.device_string() == "privateuseone:0"


def test_describe_marks_the_adapter_in_use(amd_box):
    amd_box(names=("Microsoft Basic Render Driver", "AMD Radeon RX 570"))
    described = dml.describe()
    assert "AMD Radeon RX 570 [using]" in described
    assert "Microsoft Basic Render Driver [using]" not in described


# -- backend registration ----------------------------------------------------

def test_ensure_backend_passes_non_directml_devices_through(no_directml):
    """Callers invoke this unconditionally before a `.to()`, so a CUDA device
    must not be reported as a failure just because DirectML is absent."""
    assert dml.ensure_backend("cuda:0") is True
    assert dml.ensure_backend("cpu") is True
    assert dml.ensure_backend("privateuseone:0") is False


def test_ensure_backend_is_true_on_a_working_box(amd_box):
    amd_box()
    assert dml.ensure_backend("privateuseone:0") is True
    assert dml.ensure_backend() is True


# -- precision ---------------------------------------------------------------

def test_fp16_is_off_unless_asked_for(monkeypatch):
    assert dml.prefer_float16() is False
    monkeypatch.setenv(dml.FP16_ENV, "1")
    assert dml.prefer_float16() is True
