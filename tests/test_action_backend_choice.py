"""Every Action Recognition backend the dropdown offers has to mean something.

The dropdown named three backends while the code could reach four: R3D has run
through ONNX Runtime on DirectML since the packaged build learned to, but the
only way to ask for it was to set the *Compute* preference, which steers the
whole run rather than this one choice. A DX12 card therefore got R3D on the GPU
as a side effect, or not at all.

Two things are pinned here. Each explicit choice maps to the flags its label
promises -- "R3D + CPU (PyTorch, slow)" must not quietly become DirectML on a
machine that has a DX12 card, which is why r3d_device is set outright rather
than detected. And every value the dropdown can produce is a value the pipeline
handles: an entry wired to nothing silently does whatever "auto" felt like.

main.py is read as text, never imported: it pulls in Qt at module scope and CI
has no PySide6.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pipeline import ACTION_BACKEND_SETTINGS, action_backend_summary

MAIN_PY = Path(__file__).resolve().parent.parent / "main.py"


@pytest.mark.parametrize("backend,expected", [
    # backend     enable_r3d, r3d_half, r3d_device, r3d_onnx_dml
    ("openvino", (False, False, None, False)),
    ("r3d_cuda", (True, True, "cuda", False)),
    ("r3d_cpu", (True, False, "cpu", False)),
    ("r3d_dml", (True, False, "cpu", True)),
])
def test_each_backend_means_what_its_label_says(backend, expected):
    assert ACTION_BACKEND_SETTINGS[backend] == expected


def test_only_the_directml_choice_asks_for_onnx_runtime():
    """The flag that moves the forward pass off torch belongs to one choice."""
    asking = {b for b, s in ACTION_BACKEND_SETTINGS.items() if s[3]}
    assert asking == {"r3d_dml"}


def test_fp16_is_only_requested_on_cuda():
    """FP16 is uneven on DirectML and pointless on the processor."""
    half = {b for b, s in ACTION_BACKEND_SETTINGS.items() if s[1]}
    assert half == {"r3d_cuda"}


def _dropdown_values() -> list:
    src = MAIN_PY.read_text(encoding="utf-8")
    start = src.index("self.action_backend_combo = QComboBox()")
    end = src.index("current_backend", start)
    return re.findall(r'addItem\(\s*(?:"[^"]*"|\n\s*"[^"]*")\s*,\s*"([a-z0-9_]+)"\)',
                      src[start:end])


def test_the_dropdown_offers_the_backends_we_expect():
    assert _dropdown_values() == [
        "auto", "siglip2", "openvino", "r3d_cuda", "r3d_dml", "r3d_cpu",
    ]


def test_the_pipeline_branches_on_siglip2_before_the_old_backends():
    src = (MAIN_PY.parent / "pipeline.py").read_text(encoding="utf-8")
    branch = src.index('use_siglip = action_backend == "siglip2"')
    assert branch < src.index("_explicit = ACTION_BACKEND_SETTINGS.get(action_backend)")


def test_every_dropdown_value_is_one_the_pipeline_handles():
    offered = set(_dropdown_values())
    # "auto" is deliberately absent from the table: it probes the machine, so
    # it lives at the call site with the detection it depends on. "siglip2" is
    # not an Intel/R3D setting at all; the call site branches on it first.
    unhandled = offered - set(ACTION_BACKEND_SETTINGS) - {"auto", "siglip2"}
    assert not unhandled, f"dropdown offers {unhandled}, pipeline handles none of it"


@pytest.mark.parametrize("args,expected", [
    # enable_r3d, r3d_model, r3d_device, r3d_onnx_dml, openvino_device
    ((False, "r3d_18", None, False, "CPU"), "OpenVINO on CPU"),
    ((False, "r3d_18", None, False, "GPU"), "OpenVINO on Intel GPU"),
    ((False, "r3d_18", None, False, "GPU.1"), "OpenVINO on Intel GPU"),
    ((False, "r3d_18", None, False, "AUTO"), "OpenVINO on the device OpenVINO picks"),
    ((True, "r3d_18", "cuda", False, "AUTO"), "R3D-18 on CUDA"),
    ((True, "r2plus1d_18", "cpu", False, "CPU"), "R(2+1)D-18 on CPU (PyTorch)"),
    ((True, "r3d_18", "privateuseone:0", False, "CPU"), "R3D-18 on DirectML"),
    ((True, "mc3_18", "cpu", True, "CPU"),
     "MC3-18 on DirectML (ONNX Runtime; the processor if that cannot run it)"),
])
def test_the_log_names_what_runs_and_where(args, expected):
    """One readable line in the log; the raw flags go to the debug log."""
    assert action_backend_summary(*args) == expected


@pytest.mark.parametrize("platform,api", [("win32", "DirectML"), ("darwin", "Core ML")])
def test_the_onnx_runtime_route_is_named_for_the_platform(monkeypatch, platform, api):
    """One flag, two APIs: ONNX Runtime drives DirectML on Windows and Core ML
    on a Mac (modules/system/ort_coreml.py)."""
    monkeypatch.setattr("sys.platform", platform)
    assert action_backend_summary(True, "r3d_18", "cpu", True, "CPU") ==         f"R3D-18 on {api} (ONNX Runtime; the processor if that cannot run it)"


def test_an_automatic_choice_says_so():
    assert action_backend_summary(False, "r3d_18", None, False, "CPU", auto=True) ==         "OpenVINO on CPU (chosen automatically)"
