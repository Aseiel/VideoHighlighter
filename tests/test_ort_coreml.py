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

import os
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
    for env in (ort_coreml.MODE_ENV, ort_coreml.UNITS_ENV, ort_coreml.FORMAT_ENV,
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
        # CLIP and Whisper ask modules.system.mps_device for Metal themselves,
        # the other torch consumers stay here; OpenVINO has no GPU plugin on
        # macOS.
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


def _conv_model(path, **attrs):
    """One Conv, written to ``path`` — enough to stand for SigLIP2's patch
    embedding (auto_pad="VALID") or YOLOX's (explicit pads)."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    w = numpy_helper.from_array(np.ones((2, 3, 4, 4), np.float32), "w")
    node = helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[4, 4],
                            strides=[4, 4], **attrs)
    graph = helper.make_graph(
        [node], "g",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1, 2, 2, 2])],
        [w])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
              str(path))
    return str(path)


class TestModelsCoreMLCannotCompile:
    """The first mac bench: the frame encoder failed to compile on Core ML
    ("Required param 'pad' is missing") and every action ran on the processor."""

    def test_a_valid_padding_is_written_out(self, tmp_path):
        onnx = pytest.importorskip("onnx")
        out = ort_coreml.coreml_ready_model(_conv_model(tmp_path / "m.onnx", auto_pad="VALID"))
        assert isinstance(out, bytes)
        node = onnx.load_from_string(out).graph.node[0]
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        assert "auto_pad" not in attrs
        assert attrs["pads"] == [0, 0, 0, 0]

    def test_explicit_padding_is_left_as_it_is(self, tmp_path):
        pytest.importorskip("onnx")
        path = _conv_model(tmp_path / "m.onnx", pads=[0, 0, 0, 0])
        assert ort_coreml.coreml_ready_model(path) == path

    def test_only_core_ml_sessions_get_the_rewrite(self, monkeypatch, tmp_path):
        pytest.importorskip("onnx")
        path = _conv_model(tmp_path / "m.onnx", auto_pad="VALID")
        _mac(monkeypatch)
        monkeypatch.setattr(ort_coreml, "cache_dir", lambda: None)
        assert isinstance(ort_directml.session(path).path, bytes)
        cpu = ort_directml.session(path, providers_override=[CPU])
        assert cpu.path == path

    def test_the_older_model_format_can_be_asked_for(self, monkeypatch):
        assert ort_coreml.provider_options()["ModelFormat"] == "MLProgram"
        monkeypatch.setenv(ort_coreml.FORMAT_ENV, "neuralnetwork")
        assert ort_coreml.provider_options()["ModelFormat"] == "NeuralNetwork"


def _dynamic_batch_model(path, weight=1.0):
    """A Conv with a free batch size, like the frame encoder's input."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    w = numpy_helper.from_array(np.full((2, 3, 4, 4), weight, np.float32), "w")
    node = helper.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[4, 4],
                            strides=[4, 4], pads=[0, 0, 0, 0])
    graph = helper.make_graph(
        [node], "g",
        [helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, ["n", 3, 8, 8])],
        [helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, ["n", 2, 2, 2])],
        [w])
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
              str(path))
    return str(path)


class TestFixedBatchAndItsCache:
    """The fourth mac bench: the frame encoder compiled on Core ML only with its
    batch size fixed, and the compile took 130 s, so it is kept, by content."""

    def test_the_batch_is_fixed_and_the_model_says_what_it_is(self, tmp_path):
        onnx = pytest.importorskip("onnx")
        out = ort_coreml.coreml_ready_model(_dynamic_batch_model(tmp_path / "m.onnx"), batch=32)
        model = onnx.load_from_string(out)
        assert model.graph.input[0].type.tensor_type.shape.dim[0].dim_value == 32
        assert model.graph.output[0].type.tensor_type.shape.dim[0].dim_value == 32
        keys = {p.key: p.value for p in model.metadata_props}
        assert len(keys[ort_coreml.CACHE_KEY]) == 32

    def test_the_cache_key_follows_the_content_not_the_path(self, tmp_path):
        onnx = pytest.importorskip("onnx")

        def key(path):
            model = onnx.load_from_string(ort_coreml.coreml_ready_model(path, batch=8))
            return {p.key: p.value for p in model.metadata_props}[ort_coreml.CACHE_KEY]
        a = key(_dynamic_batch_model(tmp_path / "a.onnx"))
        same = key(_dynamic_batch_model(tmp_path / "b.onnx"))
        # Retrained in place: same path, other weights.
        changed = key(_dynamic_batch_model(tmp_path / "a.onnx", weight=2.0))
        assert a == same
        assert a != changed

    def test_only_rewritten_models_are_cached(self, monkeypatch, tmp_path):
        pytest.importorskip("onnx")
        _mac(monkeypatch)
        monkeypatch.setattr(ort_coreml, "cache_dir", lambda: str(tmp_path / "cache"))

        def coreml_options(sess):
            return next(p[1] for p in sess.providers if isinstance(p, tuple))
        path = _dynamic_batch_model(tmp_path / "m.onnx")
        fixed = ort_directml.session(path, coreml_batch=32)
        as_is = ort_directml.session(path)
        os.makedirs(tmp_path / "cache")
        fixed = ort_directml.session(path, coreml_batch=32)
        assert coreml_options(fixed)["ModelCacheDirectory"] == str(tmp_path / "cache")
        assert "ModelCacheDirectory" not in coreml_options(as_is)
        assert as_is.path == path
        # Core ML caches only a model on disk: the rewritten one is saved
        # under the hash of its bytes, once, and opened from there.
        assert os.path.dirname(fixed.path) == str(tmp_path / "cache")
        again = ort_directml.session(path, coreml_batch=32)
        assert again.path == fixed.path
        assert len(os.listdir(tmp_path / "cache")) == 1
