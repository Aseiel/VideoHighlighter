"""Stronger frozen features, kept per frame so a small head can be trained on them.

    python tools/teach_lab/big_features.py <out.npz> --dataset <dataset root>
    python tools/teach_lab/big_features.py <out.npz> --folder <folder of clips>

Per clip (decoded once):
  siglip    SigLIP2 so400m/14 @384 (Apache-2.0), pooled image vector, 8 frames  -> 8 x 1152
  dino      DINOv2-L/14 (Apache-2.0), CLS + mean patch token, 8 frames         -> 8 x 2048
  vjepa     V-JEPA 2 ViT-L (MIT), 16 frames, tokens averaged per time step     -> 8 x 1024

Resumable: names already in <out.npz> are skipped. Keys match video_features.py.
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from video_features import evenly, read_all  # noqa: E402

DEV = "xpu" if torch.xpu.is_available() else "cpu"
DT = torch.float16
KEYS = ("names", "siglip", "dino", "vjepa")


class Siglip:
    def __init__(self, name="google/siglip2-so400m-patch14-384"):
        from transformers import AutoImageProcessor, AutoModel
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, torch_dtype=DT).vision_model.eval().to(DEV)

    @torch.no_grad()
    def __call__(self, rgb):
        px = self.proc(images=rgb, return_tensors="pt")["pixel_values"].to(DEV, DT)
        return self.model(pixel_values=px).pooler_output.float().cpu().numpy()


class Dino:
    def __init__(self, name="facebook/dinov2-large"):
        from transformers import AutoImageProcessor, AutoModel
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, torch_dtype=DT).eval().to(DEV)

    @torch.no_grad()
    def __call__(self, rgb):
        px = self.proc(images=rgb, return_tensors="pt")["pixel_values"].to(DEV, DT)
        h = self.model(pixel_values=px).last_hidden_state.float()
        return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).cpu().numpy()


class VJepa:
    def __init__(self, name="facebook/vjepa2-vitl-fpc64-256"):
        from transformers import AutoModel, AutoVideoProcessor
        self.proc = AutoVideoProcessor.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, torch_dtype=DT).eval().to(DEV)

    @torch.no_grad()
    def __call__(self, rgb16):
        vid = np.stack(rgb16)                                   # T, H, W, C
        px = self.proc(torch.from_numpy(vid).permute(0, 3, 1, 2), return_tensors="pt")
        px = px["pixel_values_videos"].to(DEV, DT)
        tok = self.model.get_vision_features(px).float()[0]     # (T/2 * h * w), D
        t = len(rgb16) // 2
        return tok.reshape(t, -1, tok.shape[-1]).mean(1).cpu().numpy()   # 8 x D


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dataset")
    ap.add_argument("--folder")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many (timing)")
    args = ap.parse_args()
    # These tools never use sentence-transformers; optimum.intel imports it
    sys.modules.setdefault("sentence_transformers", None)

    if args.dataset:
        from modules.teach.benchmark import read_dataset
        items = [(os.path.relpath(c.path, args.dataset), c.path) for c in read_dataset(args.dataset)["clips"]]
    else:
        items = [(os.path.basename(p), p) for p in sorted(glob.glob(os.path.join(args.folder, "*.mp4")))]

    rows = {k: [] for k in KEYS}
    if os.path.exists(args.out):
        z = np.load(args.out, allow_pickle=True)
        rows = {k: list(z[k]) for k in KEYS}
    done = set(rows["names"])
    todo = [(n, p) for n, p in items if n not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"big_features: {len(items)} clips, {len(todo)} to read, device {DEV}", flush=True)
    if not todo:
        return 0
    sig, dino, vj = Siglip(), Dino(), VJepa()

    def save():
        np.savez(args.out, names=np.array(rows["names"]), **{k: np.stack(rows[k]) for k in KEYS[1:]})

    t0 = time.time()
    for i, (name, path) in enumerate(todo, 1):
        try:
            frames = read_all(path, short=384)
            if len(frames) < 8:
                raise ValueError(f"{len(frames)} frames")
            rgb8 = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in evenly(frames, 8)]
            rgb16 = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in evenly(frames, 16)]
            out = (name, sig(rgb8), dino(rgb8), vj(rgb16))
        except Exception as exc:  # noqa: BLE001
            print(f"big_features: {name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        for k, v in zip(KEYS, out):
            rows[k].append(v.astype(np.float16) if k != "names" else v)
        if i % 100 == 0 or i == len(todo):
            print(f"big_features: {i}/{len(todo)} ({(time.time() - t0) / i:.2f}s each)", flush=True)
            save()
    return 0


if __name__ == "__main__":
    sys.exit(main())

