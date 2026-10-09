"""Writing a fine-tuned action model: its own image tower, its head, head.json.

The folder is the one the app loads (``action_siglip.own_encoder``), made in
two steps so nothing half-checked ever goes live:

1. **Export** next to the destination: the trained tower as fp32
   ``vision.onnx`` (``pixel_values [N, 3, 256, 256] -> image_embeds [N, dims]``,
   as tools/export_frame_encoder.py exports the shared one), checked on ONNX
   Runtime against PyTorch; ``head.onnx``, checked against PyTorch on the
   tower's own vectors; and ``head.json`` with an encoder id of its own.
2. **Install** with ``action_models.install_export``, the same code the app's
   Import uses: weights stored as fp16 (half the size, computes as fp32), the
   probe vector every route must reproduce, a check on every runtime present,
   and the app's own reader, all in a staging folder that replaces the old
   model only when everything passed.

The fp32 export is deleted afterwards; only the installed folder is kept.
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Callable

import numpy as np
import torch

from model_training.action_head import head as H

FINETUNED_SUFFIX = "-finetuned"
OPSET = 17
MIN_COSINE = 0.9999          # ONNX Runtime fp32 vs PyTorch
MAX_HEAD_DRIFT = 1e-4


def finetuned_id(shared_id: str) -> str:
    """The encoder id a fine-tuned head records: never the shared one, since
    its vectors are not the shared encoder's."""
    return shared_id + FINETUNED_SUFFIX


def _worst_cosine(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(((a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)).min())


def export(tower_vm, head, folder: str, meta: dict, check_vectors: np.ndarray,
           log: Callable[[str], None] = print) -> None:
    """Write fp32 ``vision.onnx``, ``head.onnx`` and ``head.json`` into
    ``folder``. ``check_vectors`` [N, frames, dims] are the tower's own
    vectors for a few clips, to check the head on. Raises ValueError when an
    export disagrees with PyTorch."""
    import onnxruntime as ort

    from model_training.action_head.finetune import Vision
    from modules.vision.action_models import TOWER, check_input
    from modules.vision.action_siglip import HEAD_META, HEAD_MODEL

    os.makedirs(folder, exist_ok=True)
    vision = Vision(tower_vm.float().cpu().eval()).eval()
    for p in vision.parameters():
        p.requires_grad_(False)
    pixels = check_input()
    with torch.no_grad():
        reference = vision(torch.from_numpy(pixels)).numpy()
    path = os.path.join(folder, TOWER)
    torch.onnx.export(vision, (torch.from_numpy(pixels[:1]),), path,
                      input_names=["pixel_values"], output_names=["image_embeds"],
                      opset_version=OPSET, dynamo=False, do_constant_folding=True,
                      dynamic_axes={"pixel_values": {0: "n"}, "image_embeds": {0: "n"}})
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    cos = _worst_cosine(sess.run(None, {"pixel_values": pixels})[0], reference)
    del sess
    log(f"  vision.onnx (fp32) against PyTorch: worst cosine {cos:.7f}")
    if not cos >= MIN_COSINE:
        raise ValueError(f"the exported image model disagrees with the trained one "
                         f"(cosine {cos:.6f})")

    head = head.float().cpu().eval()
    head_path = os.path.join(folder, HEAD_MODEL)
    H.export_onnx(head, head_path, int(meta["frames"]))
    x = np.asarray(check_vectors, np.float32)
    drift = float(np.abs(H.onnx_proba(H.load_onnx_session(head_path), x)
                         - H.predict_proba(head, x)).max())
    log(f"  head.onnx against PyTorch: max drift {drift:.1e}")
    if drift > MAX_HEAD_DRIFT:
        raise ValueError(f"the exported head disagrees with the trained one (max {drift:.2e})")
    with open(os.path.join(folder, HEAD_META), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)


def save(tower_vm, head, out: str, meta: dict, check_vectors: np.ndarray,
         log: Callable[[str], None] = print) -> str:
    """Export, then install as ``out`` (replacing a model already there only
    once the new one passed every check). Returns ``out``."""
    from modules.vision.action_models import install_export

    out = os.path.abspath(out)
    root, name = os.path.dirname(out), os.path.basename(out)
    os.makedirs(root, exist_ok=True)
    staging = out + ".export"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        export(tower_vm, head, staging, meta, check_vectors, log)
        install_export(staging, root, name, force=True, log=log)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return out
