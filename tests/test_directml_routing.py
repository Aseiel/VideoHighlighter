"""Where a DirectML device is allowed to win, and where it must not.

`test_directml_device.py` covers the probe. This covers the independent
`resolve_device` implementations that consume it, because the value of an
experimental backend is decided entirely by its position in each of those
orderings:

  * too eager, and it displaces CUDA or Arc on machines that had a faster path,
    turning a working install into a slow one for no reason;
  * too timid, and the AMD box it exists for still runs everything on the CPU.

The probes are separate on purpose (each module documents why it does not
import `modules.device_utils`), which is exactly the arrangement where one of
them silently drifts. So each is pinned here.

`describe_devices()` is not covered: it exists only in the Pro edition, for a
training panel this one does not have.
"""

from __future__ import annotations

import sys
import types

import pytest

from modules import directml_device as dml
from tests.test_directml_device import FakeTorchDirectML


@pytest.fixture(autouse=True)
def clean_module_state(monkeypatch):
    monkeypatch.delenv(dml.MODE_ENV, raising=False)
    dml.set_mode(None)
    yield
    dml.set_mode(None)


@pytest.fixture
def amd_box(monkeypatch):
    """A machine whose only accelerator is a DirectML-capable AMD card, with
    torch able to address it — a source install, never the exe."""
    fake = FakeTorchDirectML(names=("AMD Radeon RX 570",))
    monkeypatch.setattr(dml, "_import_torch_directml", lambda: fake)
    dml.probe(refresh=True)
    _set_ort_dml(monkeypatch, True)
    return fake


def _set_ort_dml(monkeypatch, available):
    """Pin what ONNX Runtime answers instead of asking this machine.

    Two runtimes can each provide DirectML and the ordering between them is the
    thing under test, so leaving one of them to be decided by whether the
    developer happens to have `onnxruntime-directml` installed would make these
    assertions about the box rather than about the code.
    """
    from modules import ort_directml
    # `dml.enabled()` is in the stub because it is in the real thing:
    # ort_directml.probe() consults it, so that VH_DIRECTML=off turns off
    # DirectML whichever runtime would have supplied it. A stub that ignored the
    # switch would hide a regression in exactly the setting users reach for when
    # the backend misbehaves.
    monkeypatch.setattr(ort_directml, "available",
                        lambda: available and dml.enabled())
    monkeypatch.setattr(
        ort_directml, "probe",
        lambda refresh=False: types.SimpleNamespace(
            available=available and dml.enabled(), version="1.24.4",
            reason=None if available else "onnxruntime is not installed",
            providers=()))


@pytest.fixture
def no_directml(monkeypatch):
    """Neither runtime has it: the machine genuinely has no DirectML."""
    def boom():
        raise ImportError("No module named 'torch_directml'")
    monkeypatch.setattr(dml, "_import_torch_directml", boom)
    dml.probe(refresh=True)
    _set_ort_dml(monkeypatch, False)


@pytest.fixture
def onnx_dml_box(monkeypatch):
    """The packaged build on a DX12 card: ONNX Runtime has DirectML and torch
    does not, because `torch-directml` pins an exact torch and so cannot be
    bundled beside the CUDA one this build ships."""
    def boom():
        raise ImportError("No module named 'torch_directml'")
    monkeypatch.setattr(dml, "_import_torch_directml", boom)
    dml.probe(refresh=True)
    _set_ort_dml(monkeypatch, True)


@pytest.fixture
def fake_torch(monkeypatch):
    """Install a stub `torch` in sys.modules with the accelerators we choose.

    The suite's conftest shims torch with a MagicMock, on which
    `torch.cuda.is_available()` is *truthy* — so a test that did not do this
    would silently assert against a machine that looks like it has an NVIDIA
    card, whichever branch it meant to exercise.
    """
    def install(cuda=False, xpu=False):
        stub = types.ModuleType("torch")
        stub.cuda = types.SimpleNamespace(is_available=lambda: cuda,
                                          device_count=lambda: 1 if cuda else 0)
        if xpu:
            stub.xpu = types.SimpleNamespace(is_available=lambda: True,
                                             device_count=lambda: 1)
        monkeypatch.setitem(sys.modules, "torch", stub)
        return stub
    return install


# ---------------------------------------------------------------------------
# modules/device_utils.py — the pipeline-wide answer
# ---------------------------------------------------------------------------

def _detect(monkeypatch, torch_available=False):
    from modules import device_utils as du
    monkeypatch.setattr(du, "_TORCH_AVAILABLE", torch_available)
    return du.detect_best_device(log_fn=lambda *a, **k: None)


def test_directml_rescues_a_machine_that_would_otherwise_be_cpu(monkeypatch, amd_box):
    info = _detect(monkeypatch)
    assert info.gpu_available is True
    assert info.backend_name == "DirectML (AMD/DX12)"
    assert info.dml_device == "privateuseone:0"


def test_no_directml_still_lands_on_cpu(monkeypatch, no_directml):
    info = _detect(monkeypatch)
    assert info.gpu_available is False
    assert info.backend_name == "CPU"
    assert info.dml_device is None


def test_directml_carries_the_pytorch_slot_so_r3d_can_use_it(monkeypatch, amd_box):
    """`pytorch_device` is what routes the R3D action model, so on an AMD box it
    has to be the DirectML device.

    R3D is a 3D CNN and 3D convolution is the least certain corner of DirectML's
    operator coverage — but the guard for that belongs at load, where
    `R3DModelWrapper._warmup()` runs a real forward pass and demotes itself to
    the CPU on failure. Withholding the device instead (which is what this did
    at first) makes the safe case unreachable as well as the unsafe one.
    """
    info = _detect(monkeypatch)
    assert info.pytorch_device == "privateuseone:0"
    assert info.dml_device == "privateuseone:0"


def test_detection_and_motion_stay_off_directml(monkeypatch, amd_box):
    """Detection runs through Ultralytics, which has no DirectML backend, so it
    stays where it was. Motion detection is unchanged for the same reason it
    always was."""
    info = _detect(monkeypatch)
    assert info.motion_device == "cpu"
    assert info.yolo_pt_device == "cpu"
    assert info.openvino_device == "CPU"


def test_every_cuda_test_in_the_app_still_answers_no(monkeypatch, amd_box):
    """`pytorch_device` now holds a non-"cpu" string on a machine with no CUDA.
    Several call sites gate on `== "cuda"` (pipeline's auto backend, the viewer's
    on-demand runs, main's custom-model pick); all of them must keep answering
    no, or DirectML would be mistaken for an NVIDIA card."""
    info = _detect(monkeypatch)
    assert info.pytorch_device != "cuda"
    assert not info.pytorch_device.startswith("cuda")


def test_directml_never_asks_openvino_for_a_gpu(monkeypatch, amd_box):
    """OpenVINO's GPU plugin is Intel-only. Requesting "GPU" on an AMD box buys
    a failed plugin load, not acceleration."""
    assert _detect(monkeypatch).openvino_device == "CPU"


def test_cuda_still_wins_by_default(monkeypatch, amd_box):
    """The regression that would make this feature a net loss: DirectML runs on
    any DX12 card, NVIDIA included, so an over-eager probe would demote a
    working CUDA install to a slower backend."""
    from modules import device_utils as du
    stub = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True, device_count=lambda: 1,
            get_device_name=lambda i: "NVIDIA GeForce RTX 4070",
            get_device_properties=lambda i: types.SimpleNamespace(
                total_mem=12 * 1024 ** 3)))
    monkeypatch.setattr(du, "_TORCH_AVAILABLE", True)
    monkeypatch.setattr(du, "torch", stub)
    info = du.detect_best_device(log_fn=lambda *a, **k: None)
    assert info.backend_name == "CUDA"


def test_force_puts_directml_ahead_of_cuda(monkeypatch, amd_box):
    """The only way to exercise this path on a machine that has something
    better — which is every development machine here."""
    monkeypatch.setenv(dml.MODE_ENV, "force")
    dml.set_mode(None)
    assert _detect(monkeypatch, torch_available=True).backend_name == "DirectML (AMD/DX12)"


def test_off_leaves_an_amd_box_exactly_as_it_was(monkeypatch, amd_box):
    monkeypatch.setenv(dml.MODE_ENV, "off")
    dml.set_mode(None)
    info = _detect(monkeypatch)
    assert info.backend_name == "CPU"
    assert info.dml_device is None


def test_the_detector_never_receives_a_directml_device(monkeypatch, amd_box):
    """`resolve_yolo_device` is the *detector's* device, and detection has no
    DirectML path — YOLO runs through Ultralytics.

    Ultralytics rejects "privateuseone:0" outright, so passing one on would
    trade a slow run for a failed one. The function exists to guarantee the
    value it returns is safe to use.
    """
    from modules import device_utils as du
    assert du.resolve_yolo_device("dml") == "cpu"
    assert du.resolve_yolo_device("privateuseone:0") == "cpu"


def test_a_directml_string_on_a_machine_without_one_degrades_to_cpu(
        monkeypatch, no_directml):
    from modules import device_utils as du
    assert du.resolve_yolo_device("dml") == "cpu"


# ---------------------------------------------------------------------------
# llm/clip_prefilter.py — visual search
# ---------------------------------------------------------------------------

def test_clip_takes_directml_over_openvino_on_an_amd_box(monkeypatch, amd_box):
    """OpenVINO's "GPU" means Intel, so on AMD the OpenVINO branch is the CPU
    wearing a GPU label. DirectML is competing with the processor here."""
    import llm.clip_prefilter as cp
    monkeypatch.setattr(cp, "cuda_device", lambda: None)
    assert cp.resolve_device("AUTO") == ("torch", "privateuseone:0")


def test_clip_still_prefers_cuda(monkeypatch, amd_box):
    import llm.clip_prefilter as cp
    monkeypatch.setattr(cp, "cuda_device", lambda: "cuda:0")
    assert cp.resolve_device("AUTO") == ("torch", "cuda:0")


def test_clip_falls_back_to_openvino_when_directml_is_absent(
        monkeypatch, no_directml):
    """The pre-existing Intel and CPU behaviour, unchanged."""
    import llm.clip_prefilter as cp
    monkeypatch.setattr(cp, "cuda_device", lambda: None)
    assert cp.resolve_device("AUTO") == ("openvino", "GPU")


def test_clip_honours_an_explicit_directml_request(monkeypatch, amd_box):
    import llm.clip_prefilter as cp
    monkeypatch.setattr(cp, "cuda_device", lambda: None)
    assert cp.resolve_device("dml") == ("torch", "privateuseone:0")


def test_clip_does_not_strand_an_explicit_request_on_a_box_without_directml(
        monkeypatch, no_directml):
    import llm.clip_prefilter as cp
    monkeypatch.setattr(cp, "cuda_device", lambda: None)
    assert cp.resolve_device("dml") == ("openvino", "GPU")


def test_clip_explicit_openvino_devices_are_untouched(monkeypatch, amd_box):
    import llm.clip_prefilter as cp
    assert cp.resolve_device("CPU") == ("openvino", "CPU")
    assert cp.resolve_device("GPU.1") == ("openvino", "GPU.1")


# ---------------------------------------------------------------------------
# modules/encoder_select.py — the win that lands without any model
# ---------------------------------------------------------------------------

@pytest.fixture
def vendor_for(monkeypatch):
    """preferred_gpu_vendor() against a chosen DeviceInfo."""
    from modules import device_utils as du
    from modules import encoder_select as es

    def ask(backend_name, dml_device=None):
        monkeypatch.setattr(es, "_vendor_cache", es._UNSET)
        monkeypatch.setattr(du, "detect_best_device", lambda **kw: du.DeviceInfo(
            backend_name=backend_name, dml_device=dml_device))
        return es.preferred_gpu_vendor()
    return ask


def test_an_amd_card_now_gets_the_amf_encoders(vendor_for, amd_box):
    """Worth having on its own: this is hardware video encoding, which works on
    an AMD card whether or not a single model ever runs on DirectML."""
    assert vendor_for("DirectML (AMD/DX12)", "privateuseone:0") == "amd"


def test_forcing_directml_on_an_nvidia_box_does_not_cost_it_nvenc(
        monkeypatch, vendor_for):
    """The backend label is a constant — DirectML runs on any DX12 card — so the
    vendor has to come from the adapter name, or testing DirectML on an NVIDIA
    machine would quietly swap nvenc for an AMF encoder that isn't there."""
    fake = FakeTorchDirectML(names=("NVIDIA GeForce RTX 4070",))
    monkeypatch.setattr(dml, "_import_torch_directml", lambda: fake)
    dml.probe(refresh=True)
    assert vendor_for("DirectML (AMD/DX12)", "privateuseone:0") == "nvidia"


def test_existing_vendors_are_unchanged(vendor_for):
    assert vendor_for("CUDA") == "nvidia"
    assert vendor_for("Intel GPU (OpenVINO)") == "intel"
    assert vendor_for("CPU") is None

# ---------------------------------------------------------------------------
# What the log says about a DX12 card, which is all most users ever see of this
# ---------------------------------------------------------------------------

def _lines(monkeypatch, torch_available=False):
    from modules import device_utils as du
    monkeypatch.setattr(du, "_TORCH_AVAILABLE", torch_available)
    out = []
    du.detect_best_device(log_fn=lambda *a, **k: out.append(" ".join(str(x) for x in a)))
    return out


def test_one_card_gets_one_explanation(monkeypatch, onnx_dml_box):
    """The packaged build used to print two things about the same GPU: a note
    that torch-directml is absent, and then the ONNX Runtime line announcing the
    card. The first read as a failure on a machine that was about to be told it
    had a working GPU, and it went on to describe what the second line was there
    to say. Only the announcement survives."""
    lines = _lines(monkeypatch)

    assert any("DirectML via ONNX Runtime" in line for line in lines)
    assert not any("torch-directml" in line for line in lines)


def test_the_onnx_line_names_what_runs_on_the_gpu(monkeypatch, onnx_dml_box):
    """It used to say "object detection only". Action recognition goes through
    the same runtime now (`modules/r3d_onnx.py`), and a user reading the old
    line would have no reason to expect it."""
    lines = _lines(monkeypatch)
    detail = " ".join(lines)

    assert "object detection and action recognition" in detail


def test_a_machine_with_no_directml_at_all_is_still_told_why(monkeypatch, no_directml):
    """Suppressing the reason must not mean never showing it. With neither
    runtime present there is no announcement coming, and the reason is the only
    thing that distinguishes "no DX12 card" from "wrong build installed"."""
    lines = _lines(monkeypatch)

    assert any("DirectML unavailable" in line for line in lines)


def test_the_onnx_box_grants_torch_models_the_gpu(monkeypatch, onnx_dml_box):
    """`onnx_dml_torch` is what lets R3D export itself and move. Without it the
    action model stays on the processor, which is where it was stuck."""
    info = _detect(monkeypatch)

    assert info.onnx_dml_torch is True
    assert info.onnx_dml_yolo is True
    assert info.pytorch_device == "cpu"   # torch itself really has no GPU here
    assert info.backend_name == "DirectML (ONNX Runtime)"


def test_torch_directml_keeps_the_card_to_itself(monkeypatch, amd_box):
    """Where torch can drive the adapter it already has R3D. Handing the same
    model to ONNX Runtime as well would put two sessions on one card, which is
    contention rather than acceleration."""
    info = _detect(monkeypatch)

    assert info.dml_device == "privateuseone:0"
    assert info.onnx_dml_torch is False
