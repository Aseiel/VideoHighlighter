"""Download the official YOLOX ONNX models (Apache-2.0) and convert them to IR.

Usage:
    python tools/get_yolox_model.py            # default: tiny + s
    python tools/get_yolox_model.py s          # just one size
    python tools/get_yolox_model.py nano tiny s m l x

Output goes to models/yolox/yolox_<size>.xml/.bin — the location
modules.detection_backend.find_default_yolox_ir() searches. The work itself
lives in modules/yolox_models.py, which the app also calls on first run.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import yolox_models  # noqa: E402


def main(argv: list[str]) -> int:
    sizes = [s.lower() for s in argv] or list(yolox_models.DEFAULT_SIZES)
    try:
        yolox_models.install(sizes)
    except ValueError as e:
        print(e)
        return 1
    print(f"\nDone. IR files in: {yolox_models.MODEL_DIR}")
    print("The app discovers these automatically "
          "(modules.detection_backend.find_default_yolox_ir).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
