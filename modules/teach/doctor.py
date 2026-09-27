"""Is this machine ready to teach a model? Checked in seconds, before anything long.

Each check is a small function that returns what it found and, when
something is missing, what to do about it. Nothing here loads a model or
reads a video, so ``doctor`` is instant and safe to run first. ``quick``
runs it and stops on anything *required* for the steps it's about to run,
rather than failing an hour in.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass
from typing import Callable, Optional

REQUIRED = "required"      # the pipeline cannot run without it
TRAINING = "training"      # only `train` needs it
OPTIONAL = "optional"      # a step works without it, worse or slower

# Free space below this in the project folder: cutting samples will fail.
# Five-second samples of an hour of 1080p are a few GB.
MIN_FREE_GB = 5.0


@dataclass
class Check:
    name: str
    ok: bool
    level: str
    detail: str = ""
    fix: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def _opencv() -> Check:
    try:
        import cv2
        version = getattr(cv2, "__version__", "")
        if not hasattr(cv2, "VideoCapture") or not version:
            raise ImportError("not a usable OpenCV")
        return Check("opencv", True, REQUIRED, f"OpenCV {version}")
    except Exception as exc:
        return Check("opencv", False, REQUIRED, str(exc),
                     "pip install opencv-python (or run from the installed app)")


def _ffmpeg() -> Check:
    from modules.teach.cut import ffmpeg_exe
    exe = ffmpeg_exe()
    found = os.path.isfile(exe) or shutil.which(exe)
    if found:
        return Check("ffmpeg", True, REQUIRED, exe)
    return Check("ffmpeg", False, REQUIRED, "no ffmpeg found",
                 "pip install imageio-ffmpeg, or put ffmpeg on PATH")


def _clip() -> Check:
    try:
        from llm.clip_prefilter import ClipFramePrefilter
        problem = ClipFramePrefilter.import_error()
    except Exception as exc:          # noqa: BLE001 - report anything
        problem = f"{type(exc).__name__}: {exc}"
    if problem:
        return Check("clip", False, REQUIRED, problem,
                     "Install the CLIP pack from the app (visual search), or "
                     "pip install transformers torch")
    return Check("clip", True, REQUIRED, "CLIP stack imports")


def _torch_device() -> Check:
    try:
        from training.train_yolox_run import resolve_device
        device = resolve_device("AUTO")
    except Exception as exc:          # noqa: BLE001
        return Check("training device", False, TRAINING, str(exc),
                     "Install PyTorch (the NVIDIA pack in the app, or pip install torch)")
    if device == "cpu":
        return Check("training device", True, TRAINING,
                     "CPU only: training works but takes hours rather than minutes",
                     "An NVIDIA card with the NVIDIA pack, or an Intel Arc, trains "
                     "far faster")
    return Check("training device", True, TRAINING, device)


def _detector() -> Check:
    try:
        from modules.vision.detection_backend import find_default_yolox_ir
        found = find_default_yolox_ir()
    except Exception as exc:          # noqa: BLE001
        found, why = None, str(exc)
    else:
        why = "no YOLOX model installed"
    if found:
        return Check("object detector", True, OPTIONAL, str(found))
    return Check("object detector", False, OPTIONAL,
                 f"{why}: object projects cannot propose boxes, and focus cannot "
                 "find people", "python tools/get_yolox_model.py")


def _pose() -> Check:
    try:
        from modules.vision.pose_backend import find_default_rtmpose_ir
        found = find_default_rtmpose_ir()
    except Exception as exc:          # noqa: BLE001
        return Check("pose model", False, OPTIONAL, str(exc))
    if found:
        return Check("pose model", True, OPTIONAL, str(found))
    return Check("pose model", False, OPTIONAL,
                 "not installed yet: it's fetched the first time focus needs it")


def _window() -> Check:
    try:
        import importlib
        importlib.import_module("PySide6.QtWidgets")
        return Check("review window", True, OPTIONAL, "PySide6 available")
    except Exception as exc:          # noqa: BLE001
        return Check("review window", False, OPTIONAL, str(exc),
                     "Use `review` (contact sheet image) and `verdict` instead")


def _disk(root: str) -> Check:
    probe = root
    while probe and not os.path.exists(probe):
        probe = os.path.dirname(probe)
    try:
        free = shutil.disk_usage(probe or ".").free / (1024 ** 3)
    except OSError as exc:
        return Check("disk space", False, REQUIRED, str(exc))
    if free < MIN_FREE_GB:
        return Check("disk space", False, REQUIRED,
                     f"{free:.1f} GB free where the project lives",
                     "Free up space or put the project elsewhere (--project <folder>); "
                     "cut samples need a few GB per hour of footage")
    return Check("disk space", True, REQUIRED, f"{free:.0f} GB free")


CHECKS: tuple = (_opencv, _ffmpeg, _clip, _torch_device, _detector, _pose, _window)


def run(root: str, checks: Optional[tuple] = None) -> dict:
    results = [c() for c in (checks if checks is not None else CHECKS)] + [_disk(root)]
    blocking = [r.name for r in results if not r.ok and r.level == REQUIRED]
    return {"ready": not blocking, "blocking": blocking,
            "checks": [r.to_json() for r in results]}


def require(root: str, run_checks: Optional[Callable] = None) -> None:
    """Raise with every fix listed if anything required is missing."""
    report = (run_checks or run)(root)
    if report["ready"]:
        return
    lines = [f"{c['name']}: {c['detail']} -> {c['fix']}" for c in report["checks"]
             if not c["ok"] and c["level"] == REQUIRED]
    raise RuntimeError("not ready to teach: " + " | ".join(lines))
