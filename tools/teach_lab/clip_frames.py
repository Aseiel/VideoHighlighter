"""The app's CLIP ViT-B/32 on 8 frames per clip, kept per frame (key "clip").

    python tools/teach_lab/clip_frames.py <out.npz> --dataset <dataset root>

Same clips and frames as siglip_base_features.py (decoded at short side 256,
8 evenly spread), through the app's own CLIP loader (llm.clip_index, bundled
OpenVINO IR), so visual search's model and SigLIP2 base can be compared on
identical input. Unit vectors. Resumable.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from video_features import evenly, read_all  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    sys.modules.setdefault("sentence_transformers", None)
    from llm.clip_index import ClipEmbedder
    from modules.teach.benchmark import read_dataset
    items = [(os.path.relpath(c.path, args.dataset), c.path) for c in read_dataset(args.dataset)["clips"]]
    names, vecs = [], []
    if os.path.exists(args.out):
        z = np.load(args.out, allow_pickle=True)
        names, vecs = list(z["names"]), list(z["clip"])
    done = set(names)
    todo = [(n, p) for n, p in items if n not in done]
    print(f"clip_frames: {len(items)} clips, {len(todo)} to read", flush=True)
    clip = ClipEmbedder(device="GPU")
    clip.load()
    t0 = time.time()
    for i, (n, p) in enumerate(todo, 1):
        try:
            frames = read_all(p, short=256)
            if len(frames) < 8:
                raise ValueError(f"{len(frames)} frames")
            v = clip.embed_frames_bgr(evenly(frames, 8))
        except Exception as exc:  # noqa: BLE001
            print(f"clip_frames: {n}: {type(exc).__name__}: {exc}", flush=True)
            continue
        names.append(n)
        vecs.append(v.astype(np.float16))
        if i % 200 == 0 or i == len(todo):
            print(f"clip_frames: {i}/{len(todo)} ({(time.time() - t0) / i:.2f}s each)", flush=True)
            np.savez(args.out, names=np.array(names), clip=np.stack(vecs))


if __name__ == "__main__":
    sys.exit(main())
