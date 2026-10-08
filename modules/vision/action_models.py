"""Trained action models on this computer: which there are, and importing one.

An action model is a folder in the app's action models folder
(``app_paths.action_models_dir()``), read by ``action_siglip``:

  head.onnx    the head, scoring a clip's frame vectors
  head.json    its classes, thresholds and the encoder it was trained on
  vision.onnx  only for a fine-tuned model: its own copy of the image tower,
               declared by an ``own_encoder`` block in head.json

Train > Actions and the teach loop write such folders themselves. Importing
brings in one made elsewhere - by someone else, on another computer, or by a
fine-tuning run - from any of three shapes:

* a model on the shared encoder (head.onnx + head.json): copied;
* a fine-tuned model already in the app's layout (with own_encoder): copied;
* a fine-tuning export (vision.onnx + head.onnx + head.json with an encoder
  id of its own and no own_encoder block yet): installed by
  :func:`install_export`, which stores the tower's weights as fp16 (half the
  size, computes as fp32), records the probe vector every route must
  reproduce, and checks the result on every runtime present.

Only those files are copied, never anything else in the folder: a model
carries the model, not the material it was trained on. Every import is
checked with the app's own reader (``action_siglip.read_head_meta``) in a
staging folder before it goes live, so a broken folder never becomes the
model a run picks up.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from typing import Callable, List, Optional

import numpy as np

from modules.vision import action_siglip as A
from modules.vision import frame_encoder as fe

TOWER = "vision.onnx"
MIN_COSINE = 0.9999

LogFn = Callable[[str], None]


def installed() -> List[dict]:
    """Every model a run could use, newest first: ``{"name", "folder",
    "classes", "fine_tuned"}``."""
    out = []
    for folder in A.find_heads(fe.ENCODER_ID):
        try:
            meta = A.read_head_meta(folder)
        except Exception:  # noqa: BLE001 - find_heads already read it; gone since
            continue
        out.append({"name": os.path.basename(os.path.normpath(folder)), "folder": folder,
                    "classes": list(meta.get("classes", [])),
                    "fine_tuned": A.OWN_ENCODER in meta})
    return out


def _models_root() -> str:
    from modules.system import app_paths
    return app_paths.action_models_dir()


def free_name(root: str, wanted: str) -> str:
    """``wanted`` made into a folder name that is not taken in ``root``
    (``name``, ``name-2``, ...). Never the word that means "no model"."""
    base = re.sub(r"[^\w.-]+", "-", wanted).strip("-.") or "action-model"
    if base.lower() == A.NO_MODEL:
        base += "-model"
    name, n = base, 2
    while os.path.exists(os.path.join(root, name)):
        name, n = f"{base}-{n}", n + 1
    return name


def _read_meta(folder: str) -> dict:
    path = os.path.join(folder, A.HEAD_META)
    if not os.path.isfile(path):
        raise ValueError(f"there is no {A.HEAD_META} in {folder}")
    with open(path, encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("kind") != A.HEAD_KIND:
        raise ValueError(f"{A.HEAD_META} is not an action model (kind={meta.get('kind')!r})")
    if not os.path.isfile(os.path.join(folder, A.HEAD_MODEL)):
        raise ValueError(f"{A.HEAD_MODEL} is missing from {folder}")
    return meta


def import_model(src: str, dest_root: Optional[str] = None, name: Optional[str] = None,
                 log: LogFn = print) -> str:
    """Bring the model in folder ``src`` into the action models folder and
    return where it went. Raises ValueError with a sentence when it cannot be
    used here; nothing is written then."""
    src = os.path.abspath(src)
    dest_root = os.path.abspath(dest_root or _models_root())
    if os.path.normcase(os.path.dirname(src)) == os.path.normcase(dest_root):
        raise ValueError(f"{os.path.basename(src)} is already installed")
    meta = _read_meta(src)
    os.makedirs(dest_root, exist_ok=True)
    name = free_name(dest_root, name or os.path.basename(src))

    if A.OWN_ENCODER in meta:
        A.read_head_meta(src)                     # the block, its tower and probe
        files = [A.HEAD_MODEL, A.HEAD_META, meta[A.OWN_ENCODER]["file"]]
    elif meta.get("encoder") == fe.ENCODER_ID:
        files = [A.HEAD_MODEL, A.HEAD_META]
    elif os.path.isfile(os.path.join(src, TOWER)):
        return install_export(src, dest_root, name, log=log)
    else:
        raise ValueError(f"it was trained on {meta.get('encoder')!r}, an encoder this app "
                         f"does not have ({fe.ENCODER_ID}), and brings none of its own")

    dest = os.path.join(dest_root, name)
    staging = dest + ".importing"
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging)
    try:
        for f in files:
            shutil.copy2(os.path.join(src, f), os.path.join(staging, f))
        A.read_head_meta(staging)                 # the app's own check, before it goes live
        os.replace(staging, dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    log(f"[action_models] imported {name} ({len(meta.get('classes', []))} actions)")
    return dest


# ── a fine-tuning export ─────────────────────────────────────────────────────

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


def install_export(export: str, dest_root: str, name: str, force: bool = False,
                   log: LogFn = print) -> str:
    """Install a fine-tuning export (``vision.onnx`` at any weight precision,
    ``head.onnx``, ``head.json`` with an encoder id of its own) as
    ``<dest_root>/<name>`` and return that path. Raises ValueError with a
    sentence when the export cannot be installed.

    The probe is computed on the export's tower with ONNX Runtime on the
    processor (the reference: the fine-tuning run checks that file against
    PyTorch). The fp16 tower is then checked against it on every runtime
    present here, and nothing is installed unless all are faithful."""
    import onnx

    from modules.vision.onnx_weights import store_weights_fp16

    meta = _read_meta(export)
    if not os.path.isfile(os.path.join(export, TOWER)):
        raise ValueError(f"{TOWER} is missing from {export}")
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
        log(f"[action_models] {n} weight tensors stored as fp16, "
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
