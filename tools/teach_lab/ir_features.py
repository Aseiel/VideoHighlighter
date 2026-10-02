"""Per-frame features from an exported SigLIP2 IR, run as the app would run it.

    python tools/teach_lab/ir_features.py <out.npz> --ir <vision.xml> --dataset <root> [--split source_split.json]
        [--device GPU] [--key siglip]

8 frames per clip, evenly spread, decoded at short side 256, letterboxed to the
IR's input size, OpenVINO on --device. Kept per frame (8 x D) under --key, so
compare_split.py / search_eval.py read it like the PyTorch features. Resumable.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from pose_regions import letterbox  # noqa: E402
from video_features import evenly, read_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--ir", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--key", default="siglip")
    args = ap.parse_args()
    sys.modules.setdefault("sentence_transformers", None)
    import openvino as ov
    from ds_features import group_of
    from modules.teach.benchmark import read_dataset

    clips = read_dataset(args.dataset)["clips"]
    if args.split:
        sp = json.load(open(args.split, encoding="utf-8"))
        want = set(sp["train_videos"]) | set(sp["val_videos"])
        clips = [c for c in clips if group_of(os.path.basename(c.path)) in want]
    items = [(os.path.relpath(c.path, args.dataset), c.path) for c in clips]
    names, vecs = [], []
    if os.path.exists(args.out):
        z = np.load(args.out, allow_pickle=True)
        names, vecs = list(z["names"]), list(z[args.key])
    done = set(names)
    todo = [(n, p) for n, p in items if n not in done]
    core = ov.Core()
    model = core.read_model(args.ir)
    size = model.input(0).get_partial_shape()[2].get_length()
    model.reshape([8, 3, size, size])
    net = core.compile_model(model, args.device)
    req = net.create_infer_request()
    print(f"ir_features: {len(items)} clips, {len(todo)} to read, {os.path.basename(args.ir)} on {args.device}",
          flush=True)
    t0, t_model = time.time(), 0.0
    for i, (n, p) in enumerate(todo, 1):
        try:
            frames = read_all(p, short=size)
            if len(frames) < 8:
                raise ValueError(f"{len(frames)} frames")
            x = np.stack([letterbox(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), size) for f in evenly(frames, 8)])
            x = ((x.astype(np.float32) / 255 - 0.5) / 0.5).transpose(0, 3, 1, 2)
            tm = time.perf_counter()
            v = req.infer([x])[0].copy()
            t_model += time.perf_counter() - tm
        except Exception as exc:  # noqa: BLE001
            print(f"ir_features: {n}: {type(exc).__name__}: {exc}", flush=True)
            continue
        names.append(n)
        vecs.append(v.astype(np.float16))
        if i % 200 == 0 or i == len(todo):
            print(f"ir_features: {i}/{len(todo)} ({(time.time() - t0) / i:.2f}s each, "
                  f"model {t_model / i * 1000:.1f} ms per 8 frames)", flush=True)
            np.savez(args.out, names=np.array(names), **{args.key: np.stack(vecs)})


if __name__ == "__main__":
    sys.exit(main())
