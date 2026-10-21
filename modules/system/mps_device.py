"""torch on the Apple GPU (Metal Performance Shaders) — for the models that run
through torch rather than ONNX Runtime.

On a Mac, ONNX Runtime reaches the GPU through Core ML
(:mod:`modules.system.ort_coreml`), which covers YOLOX and the frame encoder.
CLIP and Whisper run through torch, and torch reaches the same GPU through
Metal as the ``"mps"`` device. This module answers one question for them: may
this process use it?

* Only on macOS, only when torch says MPS is there.
* Only when the compute backend is Automatic or Apple. "Processor only" means
  the processor for every model, torch's included.
* ``VH_MPS=off`` keeps torch on the processor while Core ML stays on — for
  comparing a run with and without it, and for backing out if a model
  misbehaves on Metal.

Never imports torch itself unless asked to probe: callers here have already
imported it, and importing it costs seconds.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

DEVICE = "mps"
MODE_ENV = "VH_MPS"

_OFF = {"off", "0", "false", "no", "disable", "disabled"}


def enabled() -> bool:
    """False when the user turned torch's Metal device off, by switch or by
    choosing the processor as the backend."""
    if str(os.environ.get(MODE_ENV, "")).strip().lower() in _OFF:
        return False
    try:
        from modules.system import compute_backend as cb
        return cb.configured() in (cb.AUTO, cb.APPLE)
    except Exception:  # noqa: BLE001 - no setting is the automatic one
        return True


def unavailable_reason() -> Optional[str]:
    """Why torch may not use the Apple GPU here, or None when it may."""
    if sys.platform != "darwin":
        return "not a Mac"
    if not enabled():
        return f"turned off ({MODE_ENV}=off or the processor backend)"
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return f"torch does not import ({type(e).__name__})"
    try:
        if not torch.backends.mps.is_available():
            return "torch reports no Metal device"
    except Exception as e:  # noqa: BLE001
        return f"Metal probe failed ({type(e).__name__}: {e})"
    return None


def device() -> Optional[str]:
    """``"mps"`` when torch may run a model on the Apple GPU, else None.
    Never raises."""
    return DEVICE if unavailable_reason() is None else None


def is_mps(device_str) -> bool:
    return str(device_str or "").lower().startswith(DEVICE)


def enable_cpu_fallback() -> None:
    """Let an operator Metal lacks run on the processor instead of raising.

    Read by torch when it meets such an operator, so it must be in the
    environment before the first model runs; ``setdefault`` leaves a user's
    own choice alone.
    """
    if sys.platform == "darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
