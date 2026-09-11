"""Who gets the GPU when only ONNX Runtime can reach it.

The point of this branch is the packaged build: `torch-directml` can never be
bundled (it pins an exact torch), so on an AMD machine the exe has no torch
device at all. ONNX Runtime does, and detection is the stage worth moving.

What must stay true is the order. A DirectML provider being *present* is not a
reason to take work away from CUDA or from an Intel GPU — ONNX Runtime is
installed on every Windows build, including the ones with something faster.
"""

from __future__ import annotations

import sys
import types

import pytest

from modules import device_utils


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    """No torch, no OpenVINO, no DirectML unless a test asks for one."""
    monkeypatch.setattr(device_utils, "_TORCH_AVAILABLE", False)
    monkeypatch.setattr(device_utils, "_dml", None)
    monkeypatch.setattr(device_utils, "_ort_dml", None)


class _FakeOrtDml:
    def __init__(self, ok=True, version="1.24.4"):
        self._ok = ok
        self.version = version

    def available(self):
        return self._ok

    def probe(self):
        return type("P", (), {"version": self.version, "available": self._ok})()


def _no_openvino(monkeypatch):
    """OpenVINO's GPU plugin is Intel-only, so an AMD box gets nothing here."""
    import builtins
    real_import = builtins.__import__

    def fake(name, *args, **kwargs):
        if name == "openvino":
            raise ImportError("no openvino in this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake)


class TestDetectionOnlyDirectML:
    def test_an_onnx_only_gpu_is_found(self, monkeypatch):
        monkeypatch.setattr(device_utils, "_ort_dml", _FakeOrtDml())
        _no_openvino(monkeypatch)

        info = device_utils.detect_best_device(log_fn=lambda *_: None)

        assert info.onnx_dml_yolo is True
        assert info.gpu_available is True
        assert info.backend_name == "DirectML (ONNX Runtime)"

    def test_torch_models_stay_on_the_processor(self, monkeypatch):
        """Nothing here gives torch a device — that is the honest answer when
        there is no torch build for the card."""
        monkeypatch.setattr(device_utils, "_ort_dml", _FakeOrtDml())
        _no_openvino(monkeypatch)

        info = device_utils.detect_best_device(log_fn=lambda *_: None)

        assert info.pytorch_device == "cpu"
        assert info.yolo_pt_device == "cpu"
        assert info.openvino_device == "CPU"
        assert info.dml_device is None

    def test_without_the_provider_it_is_the_cpu_as_before(self, monkeypatch):
        monkeypatch.setattr(device_utils, "_ort_dml", _FakeOrtDml(ok=False))
        _no_openvino(monkeypatch)

        info = device_utils.detect_best_device(log_fn=lambda *_: None)

        assert info.backend_name == "CPU"
        assert info.onnx_dml_yolo is False
        assert info.gpu_available is False

    def test_a_build_without_the_module_still_resolves(self, monkeypatch):
        """The module is absent from a checkout that predates it, and from any
        build that did not bundle it."""
        monkeypatch.setattr(device_utils, "_ort_dml", None)
        _no_openvino(monkeypatch)

        info = device_utils.detect_best_device(log_fn=lambda *_: None)

        assert info.backend_name == "CPU"
        assert info.onnx_dml_yolo is False


class TestItNeverOutranksSomethingFaster:
    def test_cuda_still_wins(self, monkeypatch):
        """ONNX Runtime ships on every Windows build, NVIDIA ones included."""
        class _Cuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def device_count():
                return 1

            @staticmethod
            def get_device_name(i):
                return "RTX 4080"

            @staticmethod
            def get_device_properties(i):
                return type("P", (), {"total_mem": 16 * 1024 ** 3})()

        torch = type("T", (), {"cuda": _Cuda})
        monkeypatch.setattr(device_utils, "_TORCH_AVAILABLE", True)
        monkeypatch.setattr(device_utils, "torch", torch, raising=False)
        monkeypatch.setattr(device_utils, "_ort_dml", _FakeOrtDml())

        info = device_utils.detect_best_device(log_fn=lambda *_: None)

        assert info.backend_name == "CUDA"
        assert info.onnx_dml_yolo is False

    def test_an_intel_gpu_still_wins(self, monkeypatch):
        """OpenVINO drives the Arc in the packaged build, and DirectML would be
        a slower route to the same card."""
        fake_openvino = types.ModuleType("openvino")
        fake_openvino.Core = lambda: types.SimpleNamespace(
            available_devices=["CPU", "GPU"])
        monkeypatch.setitem(sys.modules, "openvino", fake_openvino)
        monkeypatch.setattr(device_utils, "_ort_dml", _FakeOrtDml())

        info = device_utils.detect_best_device(log_fn=lambda *_: None)

        assert info.backend_name == "Intel GPU (OpenVINO)"
        assert info.onnx_dml_yolo is False


class TestDefaultsForOlderCallers:
    def test_every_branch_answers_the_new_field(self, monkeypatch):
        """__slots__ leaves an unset attribute missing rather than False, which
        is why DeviceInfo has a defaults table — a branch that predates the
        field must not raise on it."""
        info = device_utils.DeviceInfo(
            yolo_pt_device="cpu", yolo_ov_device="cpu", openvino_device="CPU",
            pytorch_device="cpu", motion_device="cpu", use_openvino_yolo=True,
            gpu_available=False, backend_name="CPU")

        assert info.onnx_dml_yolo is False
