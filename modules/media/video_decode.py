"""Reading a few frames out of many, on the graphics card's video decoder.

Every graphics card, integrated ones included, has a video decoder of its own,
separate from the processor. Whether using it pays depends on how many of the
decoded frames come back to Python:

* **Every frame** (detection, actions): it does not pay. OpenCV copies each
  frame back from the card and converts it to BGR on one thread, and that copy
  costs more than decoding on the processor did. Measured 2026-11-05 on this
  machine: about half the processor's speed from 480p to 8K.
* **One frame in many** (the CLIP index samples one a second): it pays a lot.
  Frames that are not wanted are only ``grab()``-ed and never leave the card,
  so the copy happens once a second instead of sixty times. Measured
  2026-11-06 on 5.7K HEVC: 340 frames/s on an Arc A750 against 92 on the
  Ryzen 5 5600 beside it.

So this is for sparse readers only, and it is a setting, ``compute.video_decode``
in ``config.yaml``: ``gpu`` (the default) or ``cpu``. Like ``compute.backend``
it is published into the environment at startup, so worker processes follow it.

**What it changes.** Decoding itself is exact - the codec standard fixes
every pixel - but the card's frames are converted to BGR by a different routine,
which comes out about one level in 255 off FFmpeg's. A CLIP vector of the same
frame moves by about as much as it does between two neighbouring frames:
every frame still finds itself first, and roughly one in ten of a frame's
ten nearest neighbours trades places. That is why it stays out of every-frame
detection, where the same shift cost YOLOX 3-4% of its boxes.

**Never silently.** A card that cannot decode the file (H.264 wider than
4096, say) falls back to the processor, and the caller is told which decoder
it got, to log.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

CONFIG_SECTION = "compute"
CONFIG_KEY = "video_decode"
ENV_VAR = "VH_VIDEO_DECODE"

GPU = "gpu"
CPU = "cpu"
DEFAULT = GPU

_ALIASES = {
    "gpu": GPU, "hw": GPU, "hardware": GPU, "auto": GPU, "": GPU,
    "cpu": CPU, "sw": CPU, "software": CPU, "processor": CPU, "none": CPU,
}


def normalise(value) -> Optional[str]:
    """``gpu`` / ``cpu`` for anything that means one of them, else ``None``."""
    if value is None:
        return None
    return _ALIASES.get(str(value).strip().lower())


def from_config(config: dict | None) -> Optional[str]:
    section = (config or {}).get(CONFIG_SECTION) or {}
    return normalise(section.get(CONFIG_KEY)) if isinstance(section, dict) else None


def configured() -> str:
    """The choice in force for this process: the environment, else the default."""
    return normalise(os.environ.get(ENV_VAR)) or DEFAULT


def apply(config: dict | None) -> Optional[str]:
    """Publish the saved choice into the environment, for this process and the
    ones it starts. A variable already exported is left alone: whoever set it
    before launching is testing something."""
    if os.environ.get(ENV_VAR):
        return None
    choice = from_config(config)
    if choice is not None:
        os.environ[ENV_VAR] = choice
    return choice


def open_sparse(path: str, choice: Optional[str] = None) -> Tuple[object, str]:
    """A ``cv2.VideoCapture`` for reading a few frames out of many.

    Returns ``(capture, decoder)`` where ``decoder`` is ``"gpu"``, ``"cpu"``,
    or ``"cpu (no GPU decoder for this file)"`` when the card was asked for and
    declined. Read wanted frames with ``read()`` and step over the rest with
    ``grab()``; with the GPU decoder that is the whole point.
    """
    import cv2

    choice = normalise(choice) or configured()
    if choice == GPU:
        try:
            cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG,
                                   [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])
            if cap.isOpened() and (cap.get(cv2.CAP_PROP_HW_ACCELERATION) or 0) > 0:
                return cap, GPU
            cap.release()
        except (cv2.error, AttributeError):   # an OpenCV without the option
            pass
        return cv2.VideoCapture(path), "cpu (no GPU decoder for this file)"
    return cv2.VideoCapture(path), CPU
