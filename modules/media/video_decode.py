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

**OpenCV cannot be asked whether it worked.** Its ``CAP_PROP_HW_ACCELERATION``
reports the decoder that was *requested*: when FFmpeg's hardware set-up then
fails ("hwaccel initialisation returned error"), FFmpeg decodes in software and
the property still says D3D11. Seen 2026-11-20 on a 4320x2160 H.264 film: the
index log said "decoded on gpu" while the card sat at 1 % and a processor core
at 100 %. So the file is checked first (``gpu_refusal``): a codec FFmpeg has no
hardware decoder for on Windows, or H.264 larger than 4096 on a side - the
limit of the H.264 decoders in Intel Arc and NVIDIA cards - goes straight to
the processor, and the log names the reason.
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

# What FFmpeg's D3D11 hardware decoding takes, as OpenCV names it (FOURCC).
H264_MAX_SIDE = 4096
_H264 = {"h264", "avc1", "avc3", "x264", "davc"}
_GPU_CODECS = _H264 | {"hevc", "hev1", "hvc1", "h265", "x265", "vp09", "vp9", "av01", "av1",
                       "mpg2", "mp2v", "mpeg", "vc-1", "vc1", "wvc1", "wmv3"}


def fourcc_name(value) -> str:
    """OpenCV's ``CAP_PROP_FOURCC`` (a float holding four bytes) as text."""
    n = int(value or 0)
    return "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")


def gpu_refusal(codec: str, width: float, height: float) -> Optional[str]:
    """Why the cards' video decoders will not take this video, or None when
    they should. An unknown (empty) codec is not refused."""
    c = (codec or "").strip().lower()
    if not c:
        return None
    if c not in _GPU_CODECS:
        return f"no GPU decoder for {codec.strip()}"
    w, h = int(width or 0), int(height or 0)
    if c in _H264 and max(w, h) > H264_MAX_SIDE:
        return f"H.264 {w}x{h} is larger than GPU decoders take ({H264_MAX_SIDE})"
    return None

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
                why = gpu_refusal(fourcc_name(cap.get(cv2.CAP_PROP_FOURCC)),
                                  cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                                  cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if why is None:
                    return cap, GPU
                cap.release()
                return cv2.VideoCapture(path), f"cpu ({why})"
            cap.release()
        except (cv2.error, AttributeError):   # an OpenCV without the option
            pass
        return cv2.VideoCapture(path), "cpu (no GPU decoder for this file)"
    return cv2.VideoCapture(path), CPU
