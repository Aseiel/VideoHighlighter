"""Run the R3D action-recognition model through ONNX Runtime, so a DX12 card
that torch cannot reach still runs it.

Why this exists
---------------
:mod:`modules.directml_device` puts R3D on an AMD card through ``torch-directml``,
and that works -- from source. It cannot work in the packaged exe:
``torch-directml`` pins an exact torch (2.4.1) and pip satisfies that by
*replacing* whatever torch is installed, so a build carrying it could not also
carry the CUDA torch the NVIDIA path needs. One process, one torch. That left
every packaged AMD user with action recognition on the processor.

``onnxruntime-directml`` has no torch dependency at all and is already in the
bundle, driving object detection (see :mod:`modules.onnx_detector`). The model
format is the common denominator, not the framework: export R3D once, and the
same runtime that accelerates detection accelerates this too.

Why the export is made here rather than shipped
-----------------------------------------------
The graph depends on the weights, and the weights are not fixed. A user can
import their own fine-tuned R3D with its own class count
(``app_paths.import_r3d_action_model``), and the built-in comes from
torchvision's hub cache rather than from this repo. So the export is made from
the model object the caller already built, cached beside the app's other user
data, and reused. Same arrangement as :mod:`modules.yolo_onnx`, for the same
reason.

Nothing here raises. Every caller already holds a working torch model, so the
answer to any failure is "use the one you have" -- a fallback, not an error.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from modules import app_paths, ort_directml

# Static input shape. The export bakes it in deliberately: every call site feeds
# exactly one clip at the model's native size, and dynamic axes cost speed on
# DirectML while buying nothing here. Mirrors modules/yolo_onnx.py.
CLIP_LENGTH = 16
INPUT_SIZE = 112

# 17 covers every operator a 3D ResNet uses and is what current ONNX Runtime
# builds are best tested against. The LSTM decoder in convert_pth_to_openvino.py
# stays on 13 because its graph was validated there; the two need not agree.
OPSET = 17

# Subdirectory of the user-data dir. Exports are derived files: a user who
# deletes the folder loses nothing but the minute it takes to rebuild them.
CACHE_SUBDIR = "onnx-cache"


def cache_dir() -> str:
    """Where exports live, created if needed.

    Falls back to the user-data dir itself when the subdirectory cannot be
    made, because a missing cache folder is not a reason to lose the GPU.
    """
    base = app_paths.user_data_dir()
    target = os.path.join(base, CACHE_SUBDIR)
    try:
        os.makedirs(target, exist_ok=True)
        return target
    except OSError:
        return base


def export_path(model_name, num_classes, custom_weights=None) -> str:
    """Where the ONNX form of this particular model belongs.

    The class count is in the name because it changes the graph's final layer,
    and a custom model's stem is too, so two imported models cannot collide.
    """
    stem = str(model_name)
    if custom_weights:
        base = os.path.basename(str(custom_weights))
        stem = f"{stem}-{os.path.splitext(base)[0]}"
    return os.path.join(cache_dir(), f"{stem}-{int(num_classes)}c.onnx")


def _is_stale(onnx_path, custom_weights) -> bool:
    """True when a cached export no longer matches the weights it came from.

    Only custom weights can change under a fixed name: ``import_r3d_action_model``
    overwrites ``r3d_finetuned.pth`` in place, so a user who retrains and
    re-imports would otherwise keep running the previous model for ever, with
    nothing anywhere reporting a fault. torchvision's built-in weights are
    immutable for a given model name and need no check.
    """
    if not custom_weights or not os.path.exists(str(custom_weights)):
        return False
    try:
        return os.path.getmtime(str(custom_weights)) > os.path.getmtime(onnx_path)
    except OSError:
        return True


def ensure_export(torch_model, model_name, num_classes, custom_weights=None,
                  log=print) -> Optional[str]:
    """The ONNX export for ``torch_model``, making it first if it is missing.

    ``torch_model`` must be on the CPU in fp32, which is exactly the state the
    caller is in when it reaches for this: the whole point is that torch found
    no GPU. Returns None when the export cannot be made, and the caller's answer
    to that is the torch model it already has.
    """
    path = export_path(model_name, num_classes, custom_weights)
    if os.path.exists(path) and not _is_stale(path, custom_weights):
        return path

    try:
        import torch

        log(f"⏳ Exporting {model_name} to ONNX for DirectML (one-off)…")
        dummy = torch.zeros(1, 3, CLIP_LENGTH, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            torch.onnx.export(
                torch_model,
                dummy,
                path,
                input_names=["input"],
                output_names=["logits"],
                opset_version=OPSET,
                do_constant_folding=True,
            )
    except Exception as e:  # noqa: BLE001 - a failed export falls back, not crashes
        log(f"⚠️ R3D ONNX export failed ({type(e).__name__}: {e}) — "
            f"action recognition stays on the CPU")
        # A half-written file would be loaded as valid next time.
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        return None

    return path if os.path.exists(path) else None


class OnnxR3D:
    """An R3D export, called the way :class:`R3DModelWrapper` calls its model.

    ``predict(clip)`` takes the same ``(1, 3, T, 112, 112)`` float32 block the
    torch path builds and returns the same flat logits, so the wrapper swaps one
    for the other and changes nothing else.
    """

    __slots__ = ("model_path", "session", "_input_name")

    def __init__(self, model_path, session=None):
        self.model_path = str(model_path)
        self.session = (session if session is not None
                        else ort_directml.session(model_path))
        self._input_name = self.session.get_inputs()[0].name

    @property
    def backend(self) -> str:
        """Which provider the session actually got -- asked, not assumed.

        ONNX Runtime silently drops a provider it cannot initialise, so a run
        that quietly landed back on the processor has to be able to say so.
        """
        return ort_directml.session_backend(self.session)

    @property
    def on_gpu(self) -> bool:
        """True only when DirectML really took the session.

        The distinction matters: an ORT session on the CPU provider is not
        faster than the torch model it displaced, and treating it as a GPU would
        put a device in the run's timing summary that never did any work.
        """
        return self.backend == ort_directml.PROVIDER

    def predict(self, clip) -> np.ndarray:
        """Flat logits for one preprocessed clip."""
        block = np.ascontiguousarray(clip, dtype=np.float32)
        outputs = self.session.run(None, {self._input_name: block})
        return np.asarray(outputs[0]).astype(np.float32).flatten()

    def close(self):
        self.session = None


def load(torch_model, model_name, num_classes, custom_weights=None,
         log=print) -> Optional[OnnxR3D]:
    """An :class:`OnnxR3D` for this model, or None to keep using torch.

    The dummy run at the end is load-bearing rather than a warm-up. DirectML
    implements a *subset* of the operator set, R3D is a 3D CNN, and 3D
    convolution is the least certain corner of that subset -- the same risk
    ``R3DModelWrapper._warmup`` guards on the torch side. Finding out here costs
    one forward pass; finding out later costs an hour into a job.
    """
    if not ort_directml.available():
        return None

    path = ensure_export(torch_model, model_name, num_classes,
                         custom_weights=custom_weights, log=log)
    if path is None:
        return None

    try:
        runner = OnnxR3D(path)
        runner.predict(np.zeros((1, 3, CLIP_LENGTH, INPUT_SIZE, INPUT_SIZE),
                                dtype=np.float32))
    except Exception as e:  # noqa: BLE001 - an unusable session is a fallback
        log(f"⚠️ R3D on ONNX Runtime failed to start "
            f"({type(e).__name__}: {e}) — staying on the CPU")
        return None

    if not runner.on_gpu:
        # ORT took the session but put it on the CPU provider. That is not an
        # error and not a win either, and the torch model is the better-tested
        # of the two CPU paths.
        log("ℹ️ R3D: ONNX Runtime fell back to the processor — "
            "keeping torch")
        runner.close()
        return None

    return runner
