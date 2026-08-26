"""
Shared pytest fixtures + heavy-dependency import shims.

Why shims?
==========
The production code imports heavy ML libraries (torch, opencv, whisper,
ultralytics, openvino, googletrans) at module-load time. We deliberately want
the test suite to run **without** those installed so a CI job can validate
pure logic (forbidden-range math, clustering, SRT formatting) in seconds rather
than minutes, and so a contributor can run `pytest` after `pip install -r
requirements-dev.txt` (a 5 MB install) instead of `requirements.txt` (1.9 GB).

The shims here replace those heavy deps with `MagicMock` *only if* they are not
already importable. If a contributor has a full dev env installed, the real
libraries are used.

What is NOT shimmed
===================
`yaml` is deliberately NOT shimmed either. It is a ~250 KB parser rather than a
heavy ML library, so it costs nothing to install, and mocking it is actively
wrong: `MagicMock.safe_load()` returns a MagicMock instead of a dict, so
anything driven by a YAML file (the composition rules engine) silently parses
to nothing and its tests assert against an empty rule set that looks valid. It
is in `requirements-dev.txt` for the same reason a real parser always is.

`numpy` is a real dependency of the tests (the pipeline math operates on numpy
arrays) and is installed alongside pytest. `collections`, `os`, `re`, etc. are
stdlib.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Heavy-dependency shims
# ---------------------------------------------------------------------------
# Each of these is imported at module-load time by code we want to exercise.
# Order matters only for submodules: shim the parent first so attribute access
# returns a fresh MagicMock that we can further specialize if needed.

_HEAVY_DEPS = [
    "cv2",
    "torch",
    "torchaudio",
    "torchvision",
    "torch.nn",
    "torch.nn.functional",
    "whisper",
    "googletrans",
    "ultralytics",
    "openvino",
    "openvino.runtime",
    "pytorchvideo",
    "ffmpeg",
    "moviepy",
    "moviepy.editor",
    "tqdm",
    "resemblyzer",
    "librosa",
    "webrtcvad",
    "sklearn",
    "sklearn.cluster",
    "transformers",
    "sentencepiece",
]

for _name in _HEAVY_DEPS:
    if _name not in sys.modules:
        sys.modules[_name] = MagicMock()


def real_opencv():
    """The installed OpenCV, or None — for tests that need actual pixels.

    A MagicMock cannot encode a frame or measure one: under the shim above,
    every video a test writes is 1×1 and every comparison of one is vacuous.
    Tests about what real pixels do borrow the real module through this and
    hand the shim straight back, because the rest of the suite is written
    against the shim and must keep getting it.
    """
    shim = sys.modules.pop("cv2", None)
    if shim is not None and not isinstance(shim, MagicMock):
        sys.modules["cv2"] = shim
        return shim
    try:
        import cv2 as real
        return real
    except Exception:
        return None
    finally:
        if shim is not None:
            sys.modules["cv2"] = shim
