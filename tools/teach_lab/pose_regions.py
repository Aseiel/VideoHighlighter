"""SigLIP2 base on pose-guided action regions vs the cropper's clip whole.

    python tools/teach_lab/pose_regions.py <out.npz> --dataset <dataset root> [--split source_split.json]

Per clip, 8 frames evenly spread (decoded at short side 384). In each frame:
YOLOX proposes people, RTMPose keeps the up-to-two largest proposals that hold
a body (as the cropper does), and the frame gets two regions:

  region   AdaptiveActionDetector's idea (model_training/shared/detection.py):
           which body part moved between sampled frames picks upper / lower /
           full body, and the box is the padded extent of those keypoints.
           No body: the people's boxes; no people: the whole frame.
  people   the kept people's boxes merged, 10 % padding (no pose logic).

SigLIP2 base/16 @256 vision tower (pooled vector) on four views of every frame,
each kept per frame (8 x 768) so compare_split.py --blocks can read them:

  whole_sq   whole frame squashed to 256x256 (what the HF processor does --
             the 0.555 baseline)
  whole_lb   whole frame letterboxed (proportions kept, padded)
  region_lb  action region letterboxed
  people_lb  people box letterboxed
  region_sq  action region squashed to 256x256
  people_sq  people box squashed

Also stored: focus (0 none, 1 upper, 2 lower, 3 full), region area as a
fraction of the frame, bodies found, and both boxes per frame (8 x 2 x 4,
x1 y1 x2 y2 as fractions of the frame). Resumable.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from video_features import evenly, read_all  # noqa: E402

DEV = "xpu" if torch.xpu.is_available() else "cpu"
VIEWS = ("whole_sq", "whole_lb", "region_lb", "people_lb", "region_sq", "people_sq")
FOCUS = {"upper_body": 1, "lower_body": 2, "full_body": 3}
KP_CONF, MIN_KEYPOINTS = 0.3, 3
MODELS = os.path.join(ROOT, "models")


def letterbox(rgb, size=256):
    h, w = rgb.shape[:2]
    s = min(size / w, size / h)
    r = cv2.resize(rgb, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size, 3), np.uint8)
    y, x = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def squash(rgb, size=256):
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)


class SiglipBase:
    def __init__(self, name="google/siglip2-base-patch16-256"):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(name, torch_dtype=torch.float16).vision_model.eval().to(DEV)

    @torch.no_grad()
    def __call__(self, rgb256):
        x = torch.from_numpy(np.stack(rgb256)).permute(0, 3, 1, 2).float().div(255).sub(0.5).div(0.5)
        return self.model(pixel_values=x.to(DEV, torch.float16)).pooler_output.float().cpu().numpy()


def merge(boxes, w, h, pad=0.1):
    if not boxes:
        return None
    b = np.array(boxes, np.float32)
    x1, y1 = b[:, 0].min(), b[:, 1].min()
    x2, y2 = b[:, 2].max(), b[:, 3].max()
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    return (int(max(0, x1 - dx)), int(max(0, y1 - dy)), int(min(w, x2 + dx)), int(min(h, y2 + dy)))


def cut(rgb, box):
    if box is None:
        return rgb
    x1, y1, x2, y2 = box
    if x2 - x1 < 8 or y2 - y1 < 8:
        return rgb
    return rgb[y1:y2, x1:x2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", help="only clips from these source videos (train + val)")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many (testing)")
    args = ap.parse_args()
    sys.modules.setdefault("sentence_transformers", None)
    from model_training.shared.detection import AdaptiveActionDetector
    from modules.teach.benchmark import read_dataset
    from modules.vision.detection_backend import YoloxOpenVINODetector
    from modules.vision.pose_backend import RTMPoseOpenVINOEstimator
    from ds_features import group_of

    clips = read_dataset(args.dataset)["clips"]
    if args.split:
        sp = json.load(open(args.split, encoding="utf-8"))
        want = set(sp["train_videos"]) | set(sp["val_videos"])
        clips = [c for c in clips if group_of(os.path.basename(c.path)) in want]
    items = [(os.path.relpath(c.path, args.dataset), c.path) for c in clips]

    keys = ("names",) + VIEWS + ("focus", "area", "bodies", "boxes")
    rows = {k: [] for k in keys}
    if os.path.exists(args.out):
        z = np.load(args.out, allow_pickle=True)
        rows = {k: list(z[k]) for k in keys}
    done = set(rows["names"])
    todo = [(n, p) for n, p in items if n not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"pose_regions: {len(items)} clips, {len(todo)} to read, SigLIP on {DEV}", flush=True)
    if not todo:
        return 0

    det = YoloxOpenVINODetector(os.path.join(MODELS, "yolox", "yolox_s.xml"), class_names=["person"],
                                device="GPU", score_thr=0.05)
    est = RTMPoseOpenVINOEstimator(os.path.join(MODELS, "rtmpose", "rtmpose_s.xml"), device="CPU")
    # RTMPose at the Arc GPU's default f16 returns NaN keypoints (OpenVINO 2026.4,
    # every box tried); f32 on the GPU matches the CPU.
    import openvino as ov
    net = ov.Core().compile_model(est.model_xml, "GPU", {"INFERENCE_PRECISION_HINT": "f32"})
    est._net, est._out_x, est._out_y = net, net.output(0), net.output(1)
    sig = SiglipBase()

    def save():
        np.savez(args.out, names=np.array(rows["names"]),
                 **{k: np.stack(rows[k]) for k in keys[1:]})

    t0 = time.time()
    for i, (name, path) in enumerate(todo, 1):
        try:
            frames = read_all(path, short=384)
            if len(frames) < 8:
                raise ValueError(f"{len(frames)} frames")
            aad = AdaptiveActionDetector()
            views = {k: [] for k in VIEWS}
            focus, area, bodies, boxes_out = [], [], [], []
            for f in evenly(frames, 8):
                h, w = f.shape[:2]
                cands = [d for d in det.detect(f) if d.class_id == 0]
                cands.sort(key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1), reverse=True)
                cands = cands[:5]
                found = est.estimate(f, [(d.x1, d.y1, d.x2, d.y2) for d in cands]) if cands else []
                kept = [(d, p) for d, p in zip(cands, found)
                        if int((p.keypoints[:, 2] > KP_CONF).sum()) >= MIN_KEYPOINTS][:2]
                boxes = [(d.x1, d.y1, d.x2, d.y2) for d, _ in kept] or \
                        [(d.x1, d.y1, d.x2, d.y2) for d in cands[:2] if d.confidence >= 0.3]
                people = merge(boxes, w, h)
                poses = [p.keypoints for _, p in kept]
                region, fc = None, 0
                if poses:
                    vis = aad._check_body_visibility(poses)
                    motion = aad._analyze_motion_regions(poses)
                    fname = aad._determine_focus_region(poses, motion, boxes, vis)
                    region = aad._adaptive_crop(fname, poses, w, h)
                    fc = FOCUS.get(fname, 0)
                else:
                    aad.prev_poses = None
                if region is None:
                    region = people
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                views["whole_sq"].append(squash(rgb))
                views["whole_lb"].append(letterbox(rgb))
                views["region_lb"].append(letterbox(cut(rgb, region)))
                views["people_lb"].append(letterbox(cut(rgb, people)))
                views["region_sq"].append(squash(cut(rgb, region)))
                views["people_sq"].append(squash(cut(rgb, people)))
                full = (0, 0, w, h)
                boxes_out.append(np.array([(region or full), (people or full)], np.float32) / [w, h, w, h])
                focus.append(fc)
                area.append(1.0 if region is None else (region[2] - region[0]) * (region[3] - region[1]) / (w * h))
                bodies.append(len(poses))
            vec = sig([im for k in VIEWS for im in views[k]]).reshape(len(VIEWS), 8, -1)
        except Exception as exc:  # noqa: BLE001
            print(f"pose_regions: {name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        rows["names"].append(name)
        for k, v in zip(VIEWS, vec):
            rows[k].append(v.astype(np.float16))
        rows["focus"].append(np.array(focus, np.int8))
        rows["area"].append(np.array(area, np.float32))
        rows["bodies"].append(np.array(bodies, np.int8))
        rows["boxes"].append(np.stack(boxes_out))
        if i % 100 == 0 or i == len(todo):
            print(f"pose_regions: {i}/{len(todo)} ({(time.time() - t0) / i:.2f}s each)", flush=True)
            save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
