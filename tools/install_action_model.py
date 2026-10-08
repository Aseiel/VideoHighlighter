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
               computes as fp32; tools/export_frame_encoder.store_weights_fp16)
  head.onnx    copied as it is
  head.json    the export's, plus an ``own_encoder`` block: the file, whose
               input it takes, and the probe vector every route must reproduce

The probe is computed on the export's tower with ONNX Runtime on the processor
(the reference: the fine-tuning script checks that file against PyTorch). The
fp16 tower is then checked against it on every runtime present here, and
nothing is installed unless all are faithful. Training material never passes
through here: the export holds weights and class names only.

Needs onnx and onnxruntime (both dev tools; the app needs neither of these
steps).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from modules.vision import action_siglip as A  # noqa: E402
from modules.vision import frame_encoder as fe  # noqa: E402

TOWER = "vision.onnx"
MIN_COSINE = 0.9999


def _worst_cosine(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)).min())


def check_input() -> np.ndarray:
    """The probe first, then three fixed random frames."""
    rng = np.random.default_rng(0)
    return np.concatenate([fe.probe_pixels(), rng.uniform(
        -1, 1, (3, 3, fe.INPUT_SIZE, fe.INPUT_SIZE)).astype(np.float32)])


def _ort(path: str, pixels: np.ndarray) -> np.ndarray:
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return np.asarray(sess.run(None, {sess.get_inputs()[0].name: pixels})[0], np.float32)


def install(export: str, dest_root: str, name: str, force: bool = False,
            log=print) -> str:
    """Write the installed folder and return its path. Raises ValueError with
    a sentence when the export cannot be installed."""
    import onnx
    from tools.export_frame_encoder import store_weights_fp16

    with open(os.path.join(export, A.HEAD_META), encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("kind") != A.HEAD_KIND:
        raise ValueError(f"{A.HEAD_META} is not an action head (kind={meta.get('kind')!r})")
    for f in (A.HEAD_MODEL, TOWER):
        if not os.path.isfile(os.path.join(export, f)):
            raise ValueError(f"{f} is missing from {export}")
    if not meta.get("encoder") or meta["encoder"] == fe.ENCODER_ID:
        raise ValueError(f"the head's encoder id must be its own, not {meta.get('encoder')!r}: "
                         f"its vectors are not the shared encoder's")

    dest = os.path.join(dest_root, name)
    if os.path.exists(dest) and not force:
        raise ValueError(f"{dest} exists (--force replaces it)")
    pixels = check_input()
    reference = _ort(os.path.join(export, TOWER), pixels)
    if reference.shape != (len(pixels), fe.DIMS):
        raise ValueError(f"the tower returns {reference.shape}, the app expects "
                         f"(N, {fe.DIMS}) for [N, 3, {fe.INPUT_SIZE}, {fe.INPUT_SIZE}]")

    staging = dest + ".installing"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    try:
        tower = os.path.join(staging, TOWER)
        converted, n = store_weights_fp16(onnx.load(os.path.join(export, TOWER)))
        onnx.save(converted, tower)
        log(f"[install_action_model] {n} weight tensors stored as fp16, "
            f"{os.path.getsize(os.path.join(export, TOWER)) / 1e6:.0f} -> "
            f"{os.path.getsize(tower) / 1e6:.0f} MB")

        results = {"ONNX Runtime CPU": _ort(tower, pixels)}
        try:
            import openvino as ov
            net = ov.Core().compile_model(tower, "CPU", fe.openvino_config("CPU"))
            results["OpenVINO CPU"] = np.asarray(net(pixels)[0], np.float32)
        except ImportError:
            log("  OpenVINO not installed, not checked")
        for where, out in results.items():
            cos = _worst_cosine(out, reference) if np.isfinite(out).all() else float("nan")
            log(f"  {where:17} worst cosine {cos:.7f} against the export")
            if not cos >= MIN_COSINE:
                raise ValueError(f"the fp16 tower is not faithful on {where} (cosine {cos:.6f})")

        shutil.copy2(os.path.join(export, A.HEAD_MODEL), os.path.join(staging, A.HEAD_MODEL))
        meta[A.OWN_ENCODER] = {"file": TOWER, "preprocess": fe.ENCODER_ID, "dims": fe.DIMS,
                               "probe": [float(v) for v in reference[0]]}
        with open(os.path.join(staging, A.HEAD_META), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1, ensure_ascii=False)
        A.read_head_meta(staging)          # the app's own check, before it goes live
        if os.path.exists(dest):
            shutil.rmtree(dest)
        os.replace(staging, dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return dest


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
