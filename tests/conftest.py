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
    "yaml",
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
