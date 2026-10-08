"""Install a fine-tuned action model (its own image tower + head) for the app.

    python -m tools.install_action_model <export folder> [--name N] [--dest D] [--force]

The export folder is what a fine-tuning run writes: ``vision.onnx`` (the
fine-tuned SigLIP2 image tower, any weight precision), ``head.onnx`` and
``head.json`` (kind "action-head", an encoder id of its own). The tower must
have been trained on the app's input, ``frame_encoder.preprocess``.

Writes ``<dest>/<name>/`` (default: the app's managed action models folder,
the export folder's name) in the layout ``action_siglip.own_encoder`` reads:

  vision.onnx  the tower with its weights stored as fp16 and widened to fp32
               at load, the same storage as the shared encoder (half the size,
               computes as fp32; modules/vision/onnx_weights.py)
  head.onnx    copied as it is
  head.json    the export's, plus an ``own_encoder`` block: the file, whose
               input it takes, and the probe vector every route must reproduce

The probe is computed on the export's tower with ONNX Runtime on the processor
(the reference: the fine-tuning script checks that file against PyTorch). The
fp16 tower is then checked against it on every runtime present here, and
nothing is installed unless all are faithful. Training material never passes
through here: the export holds weights and class names only.

The work is ``modules.vision.action_models.install_export``, which the app's
Advanced > Action Recognition > Import also uses. Needs onnx and onnxruntime.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.vision.action_models import (  # noqa: E402,F401  (the tool's API)
    MIN_COSINE, TOWER, check_input, install_export as install)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("export", help="folder with vision.onnx, head.onnx and head.json")
    ap.add_argument("--name", help="installed folder name (default: the export folder's)")
    ap.add_argument("--dest", help="where to install (default: the app's action models folder)")
    ap.add_argument("--force", action="store_true", help="replace an installed folder of that name")
    args = ap.parse_args(argv)
    if args.dest:
        dest_root = args.dest
    else:
        from modules.system import app_paths
        dest_root = app_paths.action_models_dir()
    os.makedirs(dest_root, exist_ok=True)
    name = args.name or os.path.basename(os.path.normpath(args.export))
    try:
        dest = install(args.export, dest_root, name, force=args.force)
    except ValueError as e:
        print(f"[install_action_model] not installed: {e}")
        return 1
    print(f"[install_action_model] installed: {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
