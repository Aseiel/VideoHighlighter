"""SigLIP2 base/16 @256 on 8 frames per clip, kept per frame (key "siglip").

    python tools/teach_lab/siglip_base_features.py <out.npz> --dataset <dataset root>

Same clips, frames and keys as big_features.py, so compare_split.py and
eval_big.py can read it with --blocks siglip. Resumable.
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from big_features import Siglip  # noqa: E402
from video_features import evenly, read_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--name", default="google/siglip2-base-patch16-256")
    args = ap.parse_args()
    sys.modules.setdefault("sentence_transformers", None)
    from modules.teach.benchmark import read_dataset
    items = [(os.path.relpath(c.path, args.dataset), c.path) for c in read_dataset(args.dataset)["clips"]]
    names, vecs = [], []
    if os.path.exists(args.out):
        z = np.load(args.out, allow_pickle=True)
        names, vecs = list(z["names"]), list(z["siglip"])
    done = set(names)
    todo = [(n, p) for n, p in items if n not in done]
    print(f"siglip_base: {len(items)} clips, {len(todo)} to read", flush=True)
    sig = Siglip(args.name)
    t0 = time.time()
    for i, (n, p) in enumerate(todo, 1):
        try:
            frames = read_all(p, short=256)
            if len(frames) < 8:
                raise ValueError(f"{len(frames)} frames")
            v = sig([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in evenly(frames, 8)])
        except Exception as exc:  # noqa: BLE001
            print(f"siglip_base: {n}: {type(exc).__name__}: {exc}", flush=True)
            continue
        names.append(n)
        vecs.append(v.astype(np.float16))
        if i % 200 == 0 or i == len(todo):
            print(f"siglip_base: {i}/{len(todo)} ({(time.time() - t0) / i:.2f}s each)", flush=True)
            np.savez(args.out, names=np.array(names), siglip=np.stack(vecs))


if __name__ == "__main__":
    sys.exit(main())
