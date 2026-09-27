"""The Mac's GPU through ONNX Runtime's Core ML provider.

On macOS the ONNX Runtime probe hands out Core ML where it hands out DirectML
on Windows, and the device probe turns that into the same two flags the AMD
path set — so detection and R3D move with no code of their own. What must hold:

* on a Mac, Core ML is what is asked for, with its own switch and options;
* everywhere else, nothing changed — DirectML is still the provider;
* a Mac's session on Core ML counts as a GPU, and one that landed on the CPU
  does not;
* a Mac never hears about DirectML.

No Mac and no ONNX Runtime are needed: the platform is patched, and a fake
module in `sys.modules` is exactly what the probe reads.
"""

from __future__ import annotations

import sys
import types

import pytest

from modules.system import device_utils
from modules.system import directml_device
from modules.system import ort_coreml
from modules.system import ort_directml

COREML = "CoreMLExecutionProvider"
CPU = "CPUExecutionProvider"


class _FakeOrt(types.ModuleType):
    def __init__(self, providers, version="1.24.4"):
        super().__init__("onnxruntime")
        self.__version__ = version
        self._providers = list(providers)

    def get_available_providers(self):
        return list(self._providers)

    def InferenceSession(self, path, providers=None, **kwargs):   # noqa: N802
        return types.SimpleNamespace(
            path=path, providers=providers,
            get_providers=lambda: [p[0] if isinstance(p, tuple) else p
                                   for p in (providers or [])])


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in (ort_coreml.MODE_ENV, ort_coreml.UNITS_ENV,
                directml_device.MODE_ENV, "VH_BACKEND"):
        monkeypatch.delenv(env, raising=False)
    directml_device.set_mode(None)
    ort_directml.reset_probe_cache()
    yield
    directml_device.set_mode(None)
    ort_directml.reset_probe_cache()


def _mac(monkeypatch, providers=(COREML, CPU)):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOrt(providers))
    ort_directml.reset_probe_cache()


class TestTheProbeOnAMac:
    def test_core_ml_is_what_a_mac_asks_for(self, monkeypatch):
        _mac(monkeypatch)

        probe = ort_directml.probe()

        assert probe.available
        assert probe.provider == COREML

    def test_the_session_gets_core_ml_with_its_options(self, monkeypatch):
        _mac(monkeypatch)

        providers = ort_directml.providers()

        assert providers[0][0] == COREML
        assert providers[0][1]["ModelFormat"] == "MLProgram"
        assert providers[0][1]["MLComputeUnits"] == "ALL"
        assert providers[-1] == CPU

    def test_the_plain_build_without_core_ml_says_so(self, monkeypatch):
        _mac(monkeypatch, providers=(CPU,))

        reason = ort_directml.unavailable_reason()

        assert not ort_directml.available()
        assert "Core ML" in reason
        assert "directml" not in reason.lower()

    def test_its_own_switch_turns_it_off(self, monkeypatch):
        _mac(monkeypatch)
        monkeypatch.setenv(ort_coreml.MODE_ENV, "off")

        assert not ort_directml.available()
        assert ort_coreml.MODE_ENV in ort_directml.unavailable_reason()
        assert ort_directml.providers() == [CPU]

    def test_directml_off_does_not_turn_off_the_mac(self, monkeypatch):
        """DirectML's switch is about an API a Mac does not have."""
        _mac(monkeypatch)
        directml_device.set_mode(directml_device.MODE_OFF)

        assert ort_directml.available()

    @pytest.mark.parametrize("written, units", [
        ("gpu", "CPUAndGPU"), ("metal", "CPUAndGPU"), ("ane", "CPUAndNeuralEngine"),
        ("cpu", "CPUOnly"), ("ALL", "ALL"), ("nonsense", "ALL"),
    ])
    def test_the_compute_units_can_be_chosen(self, monkeypatch, written, units):
        _mac(monkeypatch)
        monkeypatch.setenv(ort_coreml.UNITS_ENV, written)

        assert ort_directml.providers()[0][1]["MLComputeUnits"] == units


class TestEverywhereElseNothingChanged:
    def test_the_mac_backend_asked_for_on_a_pc_is_not_tried(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(device_utils, "_TORCH_AVAILABLE", False)
        monkeypatch.setattr(device_utils, "_dml", None)
        monkeypatch.setattr(device_utils, "_ort_dml", None)
        monkeypatch.setattr(device_utils, "_openvino_info", lambda log_fn=print: None)
        tried = []
        monkeypatch.setattr(device_utils, "_apple_info",
                            lambda log_fn=print: tried.append("apple"))
        lines = []

        info = device_utils.detect_best_device(log_fn=lines.append, prefer="apple")

        assert info.backend_name == "CPU"
        assert tried == []
        assert any("does not exist on this platform" in line for line in lines)

    def test_windows_still_asks_for_directml(self, monkeypatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setitem(sys.modules, "onnxruntime",
                            _FakeOrt(("DmlExecutionProvider", CPU)))
        ort_directml.reset_probe_cache()

        assert ort_directml.probe().provider == "DmlExecutionProvider"
        assert ort_directml.providers()[0][0] == "DmlExecutionProvider"

    def test_core_ml_on_windows_is_not_taken(self, monkeypatch):
        """A provider list is per build; a Windows wheel reporting Core ML
        would be a strange wheel, not a reason to use it."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOrt((COREML, CPU)))
        ort_directml.reset_probe_cache()

        assert not ort_directml.available()


class TestWhatCountsAsTheGpu:
    @pytest.mark.parametrize("name, gpu", [
        (COREML, True), ("DmlExecutionProvider", True), (CPU, False), ("unknown", False),
    ])
    def test_a_session_on_core_ml_is_on_the_gpu(self, name, gpu):
        assert ort_directml.is_gpu_provider(name) is gpu

    def test_r3d_on_core_ml_counts_as_moved(self, monkeypatch):
        from modules.vision import r3d_onnx
        session = types.SimpleNamespace(
            get_inputs=lambda: [types.SimpleNamespace(name="input")],
            get_providers=lambda: [COREML, CPU])

        assert r3d_onnx.OnnxR3D("r3d.onnx", session=session).on_gpu is True


class TestTheDeviceProbeOnAMac:
    @pytest.fixture(autouse=True)
    def _no_other_gpu(self, monkeypatch):
        monkeypatch.setattr(device_utils, "_TORCH_AVAILABLE", False)
        monkeypatch.setattr(device_utils, "_ort_dml", ort_directml)
        monkeypatch.setattr(device_utils, "_openvino_info", lambda log_fn=print: None)
        monkeypatch.setattr(device_utils, "_apple_chip_name", lambda: "Apple M2")

    def test_the_mac_gets_the_same_two_models_moved(self, monkeypatch):
        _mac(monkeypatch)

        info = device_utils.detect_best_device(log_fn=lambda *_: None)

        assert info.backend_name == "Apple GPU (Core ML)"
        assert info.gpu_available is True
        assert info.onnx_dml_yolo is True
        assert info.onnx_dml_torch is True
        # torch has no consumer taught "mps" yet, and OpenVINO has no GPU
        # plugin on macOS.
        assert info.pytorch_device == "cpu"
        assert info.openvino_device == "CPU"
        assert info.dml_device is None

    def test_without_core_ml_it_is_the_cpu_and_says_why(self, monkeypatch):
        _mac(monkeypatch, providers=(CPU,))
        lines = []

        info = device_utils.detect_best_device(log_fn=lines.append)

        assert info.backend_name == "CPU"
        assert info.onnx_dml_yolo is False
        assert any("Core ML" in line for line in lines)

    def test_a_mac_never_hears_about_directml(self, monkeypatch):
        """The line that started this: 'this build ships the CUDA one' on a
        build that ships no CUDA, about an API the Mac does not have."""
        _mac(monkeypatch, providers=(CPU,))

        class _TorchDml:
            MODE_ENV = "VH_DIRECTML"

            def forced(self):
                return False

            def enabled(self):
                return True

            def probe(self):
                return types.SimpleNamespace(available=False)

            def unavailable_reason(self):
                return "a packaged build cannot carry torch-directml"

        monkeypatch.setattr(device_utils, "_dml", _TorchDml())
        lines = []

        device_utils.detect_best_device(log_fn=lines.append)

        assert not any("DirectML" in line for line in lines)

    def test_choosing_the_processor_keeps_it_off_the_gpu(self, monkeypatch):
        _mac(monkeypatch)

        info = device_utils.detect_best_device(log_fn=lambda *_: None, prefer="cpu")

        assert info.onnx_dml_yolo is False
        assert info.onnx_dml_torch is False

    def test_choosing_apple_by_name(self, monkeypatch):
        _mac(monkeypatch)

        info = device_utils.detect_best_device(log_fn=lambda *_: None, prefer="apple")

        assert info.backend_name == "Apple GPU (Core ML)"

    @pytest.mark.parametrize("pc_backend", ["cuda", "intel", "directml"])
    def test_a_pc_backend_asked_for_on_a_mac_is_not_tried(self, monkeypatch,
                                                         pc_backend):
        _mac(monkeypatch)
        tried = []
        monkeypatch.setattr(device_utils, "_cuda_info",
                            lambda log_fn=print: tried.append("cuda"))
        monkeypatch.setattr(device_utils, "_xpu_info",
                            lambda log_fn=print: tried.append("xpu"))
        lines = []

        info = device_utils.detect_best_device(log_fn=lines.append,
                                               prefer=pc_backend)

        assert info.backend_name == "Apple GPU (Core ML)"
        assert tried == []
        assert any("does not exist on this platform" in line for line in lines)

    def test_automatic_on_a_mac_asks_nothing_but_apple(self, monkeypatch):
        _mac(monkeypatch, providers=(CPU,))
        tried = []
        for probe in ("_cuda_info", "_xpu_info", "_openvino_info",
                      "_any_directml_info"):
            monkeypatch.setattr(device_utils, probe,
                                lambda log_fn=print, _n=probe: tried.append(_n))

        info = device_utils.detect_best_device(log_fn=lambda *_: None)

        assert info.backend_name == "CPU"
        assert tried == []

    def test_the_gpu_is_listed_by_name(self, monkeypatch):
        _mac(monkeypatch)
        import builtins
        real_import = builtins.__import__

        def no_openvino(name, *args, **kwargs):
            if name == "openvino":
                raise ImportError("no openvino in this test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", no_openvino)

        listed = device_utils.describe_devices()

        assert listed == ["Apple M2 GPU - Core ML (ONNX Runtime)"]
