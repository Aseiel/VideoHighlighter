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
* ``VH_COREML_FORMAT`` — ``mlprogram`` (the default) or ``neuralnetwork``, the
  older format, converted by different code: the way out if a model will not
  compile as an ML Program.

Core ML compiles a model when a session opens, and ONNX Runtime can keep the
compiled result — but by default it keys that cache by the model's *path*, and
a model rewritten in place (a retrained custom detector) would then run its old
weights without a word. So a model opened from its file is never cached; YOLOX
compiles in under a second anyway. A model this module rewrites
(:func:`coreml_ready_model`) is different: it is handed over as bytes carrying
``COREML_CACHE_KEY``, a hash of those bytes, so a changed model can only miss
the cache. That is what makes the frame encoder usable here: its first compile
took 130 s on the M1 runner.
"""

from __future__ import annotations

import os

# ORT's name for the provider, as get_available_providers() reports it.
PROVIDER = "CoreMLExecutionProvider"

MODE_ENV = "VH_COREML"
UNITS_ENV = "VH_COREML_UNITS"
FORMAT_ENV = "VH_COREML_FORMAT"

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

CACHE_KEY = "COREML_CACHE_KEY"     # ONNX Runtime reads it from the model's metadata


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
    fmt = str(os.environ.get(FORMAT_ENV, "")).strip().lower()
    return {
        "ModelFormat": "NeuralNetwork" if fmt in ("neuralnetwork", "nn") else "MLProgram",
        "MLComputeUnits": compute_units(),
    }


def cache_dir():
    """Where compiled Core ML models are kept, or None when there is nowhere."""
    try:
        from modules.system import app_paths
        path = os.path.join(app_paths.user_data_dir(), "coreml-cache")
        os.makedirs(path, exist_ok=True)
        return path
    except Exception:  # noqa: BLE001 - no cache only costs time
        return None


def coreml_ready_model(model_path, batch=None):
    """``model_path`` as Core ML can compile it: the path itself when nothing
    needs changing, else the model's bytes with the change made, carrying a
    hash of themselves as their cache key.

    ``batch`` fixes a symbolic batch dimension to that size. With it left
    free, Core ML compiled the frame encoder's ML Program to nothing ("Failed
    to build the model execution plan", -14); fixed at 32 it compiled and ran
    at half the processor's time per frame. The caller then pads its batches.

    ONNX Runtime turns a Conv's ``auto_pad="VALID"`` into a "custom" padding
    with no pad values, and Core ML then refuses the whole model ("Required
    param 'pad' is missing"). SigLIP2's patch embedding, and so the frame
    encoder and every action model built on it, is exactly such a Conv, which
    kept them on the processor on every Mac. VALID means no padding, so the
    explicit ``pads=0`` is the same layer. Fixed here rather than only in the
    export because models people already made carry the old attribute.
    """
    try:
        import onnx
        model = onnx.load(str(model_path))
    except Exception:  # noqa: BLE001 - unreadable here; let the session say why
        return model_path
    changed = 0
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        pad = next((a for a in node.attribute if a.name == "auto_pad"), None)
        if pad is None or pad.s not in (b"VALID",):
            continue
        kernel = next((a for a in node.attribute if a.name == "kernel_shape"), None)
        if kernel is None or any(a.name == "pads" for a in node.attribute):
            continue
        node.attribute.remove(pad)
        node.attribute.append(onnx.helper.make_attribute("pads", [0] * (2 * len(kernel.ints))))
        changed += 1
    if changed:
        print(f"ℹ️ Core ML: {changed} Conv padding(s) written out in "
              f"{os.path.basename(str(model_path))}")
    fixed = 0
    if batch:
        for value in list(model.graph.input) + list(model.graph.output):
            dims = value.type.tensor_type.shape.dim
            if dims and not dims[0].HasField("dim_value"):
                dims[0].ClearField("dim_param")
                dims[0].dim_value = int(batch)
                fixed += 1
    if not changed and not fixed:
        return model_path
    import hashlib
    for prop in [p for p in model.metadata_props if p.key == CACHE_KEY]:
        model.metadata_props.remove(prop)
    key = hashlib.sha256(model.SerializeToString()).hexdigest()[:32]
    model.metadata_props.add(key=CACHE_KEY, value=key)
    return model.SerializeToString()


def describe_units(units: str | None = None) -> str:
    """Where Core ML may run a model, in words, for the log."""
    return {
        "ALL": "GPU / Neural Engine",
        "CPUAndGPU": "GPU (Metal)",
        "CPUAndNeuralEngine": "Neural Engine",
        "CPUOnly": "processor, by choice",
    }.get(units or compute_units(), units or compute_units())
