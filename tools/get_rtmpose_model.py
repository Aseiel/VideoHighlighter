"""Download the official RTMPose ONNX models (Apache-2.0) and convert them to IR.

Usage:
    python tools/get_rtmpose_model.py          # default: m (75.3 AP)
    python tools/get_rtmpose_model.py l        # most accurate
    python tools/get_rtmpose_model.py s m l

Output goes to models/rtmpose/rtmpose_<size>.xml/.bin — the location
modules.vision.pose_backend.find_default_rtmpose_ir() searches. The work itself lives
in modules/vision/rtmpose_models.py, which the app also calls on first run.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.vision import rtmpose_models  # noqa: E402


def main(argv: list[str]) -> int:
    sizes = [s.lower() for s in argv] or list(rtmpose_models.DEFAULT_SIZES)
    try:
        rtmpose_models.install(sizes)
    except ValueError as e:
        print(e)
        return 1
    print(f"\nDone. IR files in: {rtmpose_models.MODEL_DIR}")
    print("The app discovers these automatically "
          "(modules.vision.pose_backend.find_default_rtmpose_ir).")
    return 0


if __name__ == "__main__":
    from modules.system.debug_console import force_utf8_stdio
    force_utf8_stdio()
    raise SystemExit(main(sys.argv[1:]))
