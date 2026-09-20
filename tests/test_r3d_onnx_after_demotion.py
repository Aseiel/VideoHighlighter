"""A card torch-directml cannot use is not a card the run should give up on.

torch-directml refuses a 5D tensor outright: nn.Conv3d raises "input must be
4-dimensional", and R3D is nothing but 3D convolution, so the warm-up demotes
the model. That demotion went straight to the processor and took a working
GPU with it.

ONNX Runtime's DirectML provider is a different operator implementation on the
same API, and it runs this model -- the exported r3d_18 carries 20 Conv nodes,
every one of them 3D, measured here at 27.9 ms a window on the DML provider.
`_try_onnx()` was written for exactly this moment and says so in its docstring;
the auto branch simply never passed allow_onnx_dml, so it returned None before
looking.

"R3D + CPU (PyTorch, slow)" must keep meaning the processor, so the permission
is per-choice rather than global -- that choice is checked here too.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import action_recognition as ar
from pipeline import ACTION_BACKEND_SETTINGS

PIPELINE_PY = Path(__file__).resolve().parent.parent / "pipeline.py"


class _Demoted:
    """A wrapper as the warm-up leaves it: on the CPU, card still present."""

    model = object()
    model_name = "r3d_18"
    num_classes = 400

    class _Cpu:
        # Not torch.device("cpu"): conftest shims torch, so a real device
        # object would hand back a MagicMock whose .type is not the string
        # _try_onnx compares against, and the gate would close for the wrong
        # reason while the test still looked meaningful.
        type = "cpu"

    def __init__(self, allow: bool):
        self.allow_onnx_dml = allow
        self.device = self._Cpu()


def test_a_demoted_model_is_offered_to_onnx_runtime(monkeypatch):
    from modules.vision import r3d_onnx

    sentinel = object()
    monkeypatch.setattr(r3d_onnx, "load", lambda *a, **kw: sentinel)
    assert ar.R3DModelWrapper._try_onnx(_Demoted(allow=True)) is sentinel


def test_the_cpu_choice_stays_on_the_cpu(monkeypatch):
    """Otherwise "R3D + CPU (PyTorch, slow)" would quietly become DirectML."""
    from modules.vision import r3d_onnx

    monkeypatch.setattr(r3d_onnx, "load",
                        lambda *a, **kw: pytest.fail("asked ONNX Runtime anyway"))
    assert ar.R3DModelWrapper._try_onnx(_Demoted(allow=False)) is None
    assert ACTION_BACKEND_SETTINGS["r3d_cpu"][3] is False


def test_the_torch_directml_branch_grants_that_permission():
    """The branch that hits the Conv3d wall is the one that needs it."""
    src = PIPELINE_PY.read_text(encoding="utf-8")
    branch = src[src.index("elif _dev.dml_device:"):]
    branch = branch[:branch.index("elif getattr(_dev,")]
    assert "r3d_onnx_dml = True" in branch, \
        "a torch-directml demotion still goes straight to the CPU"
