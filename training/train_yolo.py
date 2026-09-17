"""Deprecated entry point — forwards to ``train_yolox_dataset.py``.

Training uses YOLOX (Apache-2.0), so a model made here can be used and shared
by anyone, in any build.
"""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("train_yolox_dataset.py")), run_name="__main__")
