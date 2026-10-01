"""Motion-aware features for clips: what CLIP on 4 stills cannot see.

    python tools/teach_lab/video_features.py <out.npz> --dataset <dataset root>
    python tools/teach_lab/video_features.py <out.npz> --folder <folder of clips>

Per clip (each clip decoded once):
  intel     Intel action-recognition-0001 encoder on 16 frames: mean + std of
            the 512-d frame vectors (1024), and the decoder's 400 Kinetics logits
  r21d      torchvision R(2+1)D-18, Kinetics-400 weights, penultimate 512-d,
            averaged over two 16-frame windows (stride 2) at 1/3 and 2/3
  clip8     CLIP on 8 frames: mean and max of the unit vectors (1024)

Resumable: names already in <out.npz> are skipped. Keys are paths relative to
the dataset root, or bare file names for --folder.
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
INTEL = os.path.join(ROOT, "models", "intel_action")
KIN_MEAN = np.array([0.43216, 0.394666, 0.37645], np.float32)
KIN_STD = np.array([0.22803, 0.22145, 0.216989], np.float32)


def read_all(path, short=256, limit=200):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while len(frames) < limit:
            ok, f = cap.read()
            if not ok:
                break
            h, w = f.shape[:2]
            s = short / min(h, w)
            if s < 1:
                f = cv2.resize(f, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            frames.append(f)
    finally:
        cap.release()
    return frames


def evenly(frames, n):
    if not frames:
        return []
    return [frames[int((k + 0.5) * len(frames) / n)] for k in range(n)]


def letterbox(frame, size=224):
    h, w = frame.shape[:2]
    s = min(size / w, size / h)
    r = cv2.resize(frame, (int(w * s), int(h * s)))
    out = np.zeros((size, size, 3), np.uint8)
    y, x = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


class Intel:
    def __init__(self, device="GPU"):
        import openvino as ov
        core = ov.Core()
        self.enc = core.compile_model(core.read_model(
            os.path.join(INTEL, "encoder", "FP32", "action-recognition-0001-encoder.xml")), device)
        self.dec = core.compile_model(core.read_model(
            os.path.join(INTEL, "decoder", "FP32", "action-recognition-0001-decoder.xml")), device)

    def __call__(self, frames):
        vecs = []
        for f in evenly(frames, 16):
            inp = letterbox(f).transpose(2, 0, 1)[None].astype(np.float32)
            vecs.append(np.asarray(self.enc([inp])[0]).reshape(-1))
        v = np.stack(vecs)
        logits = np.asarray(self.dec([v[None]])[0]).reshape(-1)
        return np.concatenate([v.mean(0), v.std(0)]).astype(np.float32), logits.astype(np.float32)


class R21D:
    def __init__(self):
        import torch
        from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
        self.torch = torch
        self.device = "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu"
        m = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        m.fc = torch.nn.Identity()
        self.model = m.eval().to(self.device)

    def _window(self, frames, centre):
        idx = [min(max(centre - 16 + 2 * k, 0), len(frames) - 1) for k in range(16)]
        clip = []
        for i in idx:
            f = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            h, w = f.shape[:2]
            s = 128 / min(h, w)
            f = cv2.resize(f, (max(112, int(round(w * s))), max(112, int(round(h * s)))))
            h, w = f.shape[:2]
            y, x = (h - 112) // 2, (w - 112) // 2
            clip.append(f[y:y + 112, x:x + 112])
        a = (np.stack(clip).astype(np.float32) / 255.0 - KIN_MEAN) / KIN_STD
        return a.transpose(3, 0, 1, 2)          # C, T, H, W

    def __call__(self, frames):
        n = len(frames)
        batch = np.stack([self._window(frames, n // 3), self._window(frames, 2 * n // 3)])
        with self.torch.no_grad():
            out = self.model(self.torch.from_numpy(batch).to(self.device))
        return out.float().cpu().numpy().mean(0).astype(np.float32)


class Clip8:
    def __init__(self):
        from modules.teach.embed import ClipBackend
        self.clip = ClipBackend()

    def __call__(self, frames):
        from modules.teach.embed import unit
        v = unit(self.clip.images(evenly(frames, 8)))
        return np.concatenate([unit(v.mean(0)), v.max(0)]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dataset")
    ap.add_argument("--folder")
    args = ap.parse_args()
    # These tools never use sentence-transformers; optimum.intel imports it
    # whenever it is installed, and a broken install (e.g. a torchcodec that
    # cannot load) would take CLIP down with it.
    sys.modules.setdefault("sentence_transformers", None)

    if args.dataset:
        from modules.teach.benchmark import read_dataset
        items = [(os.path.relpath(c.path, args.dataset), c.path) for c in read_dataset(args.dataset)["clips"]]
    else:
        items = [(os.path.basename(p), p) for p in sorted(glob.glob(os.path.join(args.folder, "*.mp4")))]

    keys = ("names", "intel", "kinetics", "r21d", "clip8")
    rows = {k: [] for k in keys}
    if os.path.exists(args.out):
        z = np.load(args.out, allow_pickle=True)
        rows = {k: list(z[k]) for k in keys}
    done = set(rows["names"])
    todo = [(n, p) for n, p in items if n not in done]
    print(f"video_features: {len(items)} clips, {len(todo)} to read", flush=True)
    if not todo:
        return 0
    intel, r21d, clip8 = Intel(), R21D(), Clip8()
    print(f"video_features: R(2+1)D on {r21d.device}", flush=True)
    t0 = time.time()
    for i, (name, path) in enumerate(todo, 1):
        try:
            frames = read_all(path)
            if len(frames) < 8:
                raise ValueError(f"{len(frames)} frames")
            iv, kl = intel(frames)
            rv = r21d(frames)
            cv = clip8(frames)
        except Exception as exc:  # noqa: BLE001
            print(f"video_features: {name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        for k, v in zip(keys, (name, iv, kl, rv, cv)):
            rows[k].append(v)
        if i % 100 == 0 or i == len(todo):
            print(f"video_features: {i}/{len(todo)} ({(time.time() - t0) / i:.2f}s each)", flush=True)
            np.savez(args.out, names=np.array(rows["names"]),
                     **{k: np.stack(rows[k]) for k in keys[1:]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
