"""Get a YOLO ONNX export, so :mod:`modules.onnx_detector` has something to run.

Split from the detector on purpose. The runner is framework-free and ports to
the Pro edition as-is; this file is the free edition's half, because exporting
is where Ultralytics is unavoidable and Ultralytics is exactly what Pro cannot
link against.

The export is made once, next to the weights it came from, and reused. It is
only ever made when something is actually going to run it — an export costs
tens of seconds and an NVIDIA or Intel machine has no use for the file.
"""

from __future__ import annotations

import os
from typing import Optional

from modules import onnx_detector

# Matches the imgsz the detector runs at. The export bakes the input size in
# (dynamic axes are slower on DirectML and buy nothing here, since every call
# site already asks for 640), so the two must agree.
EXPORT_IMGSZ = onnx_detector.DEFAULT_IMGSZ


def export_path(pt_path) -> str:
    """Where the ONNX form of ``pt_path`` lives."""
    stem, _ = os.path.splitext(str(pt_path))
    return stem + ".onnx"


def ensure_export(pt_path, imgsz=EXPORT_IMGSZ, log=print) -> Optional[str]:
    """The ONNX export for ``pt_path``, making it first if it is missing.

    Returns None when the export cannot be made — a machine with no write
    access beside the weights, or an Ultralytics that refuses the format. The
    caller's answer to that is the detector it already had.
    """
    onnx_path = export_path(pt_path)
    if os.path.exists(onnx_path):
        return onnx_path
    if not os.path.exists(str(pt_path)):
        # Ultralytics would fetch the weights, but an export is not the moment
        # to start a download the user did not ask for.
        return None

    try:
        from ultralytics import YOLO
        log(f"⏳ Exporting {os.path.basename(str(pt_path))} to ONNX for DirectML…")
        YOLO(str(pt_path)).export(
            format="onnx",
            imgsz=int(imgsz),
            dynamic=False,
            # The graph simplifier is a separate package that Ultralytics
            # installs on demand, and a frozen build has no pip to do it with.
            simplify=False,
            verbose=False,
        )
    except Exception as e:  # noqa: BLE001 - a failed export falls back, not crashes
        log(f"⚠️ ONNX export failed ({type(e).__name__}: {e}) — detection stays on the CPU")
        return None

    return onnx_path if os.path.exists(onnx_path) else None


def load_detector(pt_path, log=print, **kwargs):
    """An :class:`~modules.onnx_detector.OnnxDetector` for ``pt_path``, or None.

    None means "use what you were going to use", which is why nothing here
    raises: every caller already has a working detector for this machine.
    """
    onnx_path = ensure_export(pt_path, log=log)
    if onnx_path is None:
        return None
    detector = onnx_detector.load(onnx_path, **kwargs)
    if detector is None:
        return None
    log(f"✅ Object detector: ONNX on {detector.backend} ({os.path.basename(onnx_path)})")
    return detector
