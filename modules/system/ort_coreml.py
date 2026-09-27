"""ONNX Runtime's Core ML execution provider — the Mac's GPU for the models
that already run through ONNX Runtime.

A Mac has no CUDA, and OpenVINO on macOS has a CPU plugin only, so until now a
Mac ran every model on the processor. The plain ``onnxruntime`` wheel for macOS
ships Apple's Core ML provider, and Core ML is what runs a model on the Apple
GPU (through Metal) or the Neural Engine. The two models that already have an
ONNX path for AMD cards — YOLOX detection and R3D action recognition — can take
it unchanged: :mod:`modules.system.ort_directml` hands out this provider instead
of DirectML when the process is on macOS, and everything downstream of it is
the same code.

What this module is
-------------------
The Mac-specific settings and nothing else: the switch, which compute units
Core ML may use, and the provider options built from them. The probe and the
session factory stay in ``ort_directml`` so a caller asks one module "can ONNX
Runtime reach the GPU here" on every platform.

Switches, both read on every call so a worker process decides for itself:

* ``VH_COREML=off`` — keep ONNX Runtime on the processor. For comparing a run
  with and without it, and for backing out if Core ML misbehaves on a machine.
* ``VH_COREML_UNITS`` — which silicon Core ML may use: ``all`` (the default:
  Core ML picks between GPU, Neural Engine and CPU per layer), ``gpu`` (GPU and
  CPU only, i.e. Metal), ``ane`` (Neural Engine and CPU), ``cpu``.

No model cache directory is set, deliberately. Core ML compiles a model when a
session opens, and ONNX Runtime can keep the compiled result — but it keys that
cache by the model's *path*, and the R3D export is rewritten in place when the
weights behind it change. A stale compiled model would run the old weights
without a word. Compiling each time costs a few seconds per session; wrong
answers cost more.
"""

from __future__ import annotations

import os

# ORT's name for the provider, as get_available_providers() reports it.
PROVIDER = "CoreMLExecutionProvider"

MODE_ENV = "VH_COREML"
UNITS_ENV = "VH_COREML_UNITS"

_OFF = {"off", "0", "false", "no", "disable", "disabled"}

# What a user may write → Core ML's MLComputeUnits value.
UNITS = {
    "all": "ALL",
    "auto": "ALL",
    "gpu": "CPUAndGPU",
    "metal": "CPUAndGPU",
    "ane": "CPUAndNeuralEngine",
    "npu": "CPUAndNeuralEngine",
    "neural": "CPUAndNeuralEngine",
    "cpu": "CPUOnly",
}
DEFAULT_UNITS = "ALL"


def enabled() -> bool:
    """False when the user turned Core ML off."""
    return str(os.environ.get(MODE_ENV, "")).strip().lower() not in _OFF


def compute_units() -> str:
    """The MLComputeUnits value for this run. An unknown value means the
    default rather than a failed session: a typo should cost a line, not a
    run."""
    raw = str(os.environ.get(UNITS_ENV, "")).strip().lower()
    if not raw:
        return DEFAULT_UNITS
    units = UNITS.get(raw)
    if units is None:
        print(f"⚠️ {UNITS_ENV}={raw!r} is not one of "
              f"{', '.join(sorted(set(UNITS) - {'auto', 'metal', 'npu', 'neural'}))}"
              f" — using {DEFAULT_UNITS}")
        return DEFAULT_UNITS
    return units


def provider_options() -> dict:
    """Options for the Core ML provider entry in a session's provider list.

    ``MLProgram`` rather than the older ``NeuralNetwork`` format: it covers
    more operators, so fewer nodes fall back to the processor, and it is the
    format Core ML's GPU path is built around. It needs macOS 12, which every
    Apple-silicon Mac can run.
    """
    return {
        "ModelFormat": "MLProgram",
        "MLComputeUnits": compute_units(),
    }


def describe_units(units: str | None = None) -> str:
    """Where Core ML may run a model, in words, for the log."""
    return {
        "ALL": "GPU / Neural Engine",
        "CPUAndGPU": "GPU (Metal)",
        "CPUAndNeuralEngine": "Neural Engine",
        "CPUOnly": "processor, by choice",
    }.get(units or compute_units(), units or compute_units())
