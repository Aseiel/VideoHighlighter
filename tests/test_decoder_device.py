"""Where the action decoders run when another runtime already holds the GPU.

A 0.12.0 run stalled for good inside the Intel decoder's `infer()`. The stall
watchdog caught the stack: the pipeline thread sat in
`openvino/_ov_api.py -> infer` and never came out, while R3D ran through ONNX
Runtime's DirectML provider on the same Intel Arc A750. No error, no traceback
-- an inference that never returns is not an exception, and the progress bar
simply stopped.

`device_utils` already keeps two runtimes off one adapter in the other
direction (`onnx_dml_torch=False` where torch owns the GPU). This is the same
rule applied the way round nobody had needed yet, and it is free: measured on
that machine the decoder is *faster* on the processor (0.67 ms against 1.14 ms
per window) while the encoder is 8x faster on the card, so only the decoders
move.
"""

from __future__ import annotations

import pytest

import action_recognition as ar

ARC = "Intel(R) Arc(TM) A750 Graphics (dGPU)"


class FakeCompiled:
    def __init__(self, device):
        self.device = device

    def input(self, index=0):
        return f"in{index}"

    def output(self, index=0):
        return f"out{index}"


class FakeCore:
    """Enough OpenVINO to run load_models' device resolution, and no more."""

    def __init__(self, devices=("CPU", "GPU"), gpu_name=ARC):
        self.available_devices = list(devices)
        self._gpu_name = gpu_name
        self.compiled: list[tuple[str, str]] = []  # (model path, device)

    def get_property(self, device, key):
        return self._gpu_name

    def set_property(self, device, props):
        pass

    def read_model(self, model=None, weights=None):
        return str(model)

    def compile_model(self, model=None, device_name=None):
        self.compiled.append((str(model), device_name))
        return FakeCompiled(device_name)


def _load(monkeypatch, **kwargs):
    core = FakeCore()
    monkeypatch.setattr(ar, "Core", lambda: core)
    ar.load_models(enable_r3d=False, action_models="intel_only", **kwargs)
    return {role: device for path, device in core.compiled
            for role in ("encoder", "decoder") if role in path}


def test_decoders_move_to_the_cpu_when_directml_has_the_card(monkeypatch):
    where = _load(monkeypatch, r3d_onnx_dml=True)
    assert where["encoder"] == "GPU", "the encoder earns the card and keeps it"
    assert where["decoder"] == "CPU"


def test_both_stay_on_the_gpu_when_nothing_else_is_using_it(monkeypatch):
    where = _load(monkeypatch, r3d_onnx_dml=False)
    assert where["encoder"] == "GPU"
    assert where["decoder"] == "GPU"


def test_an_explicit_cpu_run_is_left_alone(monkeypatch):
    where = _load(monkeypatch, device="CPU", r3d_onnx_dml=True)
    assert where == {"encoder": "CPU", "decoder": "CPU"}


def test_a_non_intel_gpu_is_still_refused(monkeypatch):
    """AUTO only ever takes Intel graphics; DirectML does not change that."""
    core = FakeCore(gpu_name="NVIDIA GeForce GTX 1060")
    monkeypatch.setattr(ar, "Core", lambda: core)
    ar.load_models(enable_r3d=False, action_models="intel_only", r3d_onnx_dml=True)
    assert {device for _path, device in core.compiled} == {"CPU"}
