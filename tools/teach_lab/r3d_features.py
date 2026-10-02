"""Frozen torchvision r3d_18 (Kinetics-400) features, the same way video_features.py
takes R(2+1)D-18: penultimate 512-d, two 16-frame windows (stride 2) at 1/3 and
2/3 of the clip, averaged. Key "r3d".

    python tools/teach_lab/r3d_features.py <out.npz> --dataset <dataset root>

Resumable. compare_split.py reads it with --vid <out.npz> --blocks r3d.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import video_features as V  # noqa: E402


class R3D(V.R21D):
    def __init__(self):
        import torch
        from torchvision.models.video import R3D_18_Weights, r3d_18
        self.torch = torch
        self.device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
        m = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        m.fc = torch.nn.Identity()
        self.model = m.eval().to(self.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    sys.modules.setdefault("sentence_transformers", None)
    from modules.teach.benchmark import read_dataset
    items = [(os.path.relpath(c.path, args.dataset), c.path) for c in read_dataset(args.dataset)["clips"]]
    names, vecs = [], []
    if os.path.exists(args.out):
        z = np.load(args.out, allow_pickle=True)
        names, vecs = list(z["names"]), list(z["r3d"])
    done = set(names)
    todo = [(n, p) for n, p in items if n not in done]
    print(f"r3d_features: {len(items)} clips, {len(todo)} to read", flush=True)
    model = R3D()
    t0 = time.time()
    for i, (n, p) in enumerate(todo, 1):
        try:
            frames = V.read_all(p)
            if len(frames) < 8:
                raise ValueError(f"{len(frames)} frames")
            v = model(frames)
        except Exception as exc:  # noqa: BLE001
            print(f"r3d_features: {n}: {type(exc).__name__}: {exc}", flush=True)
            continue
        names.append(n)
        vecs.append(v)
        if i % 200 == 0 or i == len(todo):
            print(f"r3d_features: {i}/{len(todo)} ({(time.time() - t0) / i:.2f}s each)", flush=True)
            np.savez(args.out, names=np.array(names), r3d=np.stack(vecs))


if __name__ == "__main__":
    sys.exit(main())
