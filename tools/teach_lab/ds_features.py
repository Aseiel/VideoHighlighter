"""CLIP + pose + motion features of every clip in a hand-sorted dataset,
with its classes, split and source video -- the answer key for grouping.

    python tools/teach_lab/ds_features.py <dataset> <out.npz> [--group REGEX]

Same features as discover.py computes for crops, so a grouping or classifier
can be measured on the dataset and then used on new footage unchanged.
Resumable: clips already in <out.npz> are skipped.
"""
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import discover as D  # noqa: E402

# The video a clip was cut from: the name before the cutter's suffix. The
# reader's default (a leading number) misses titled names, and then every
# clip counts as its own video and "held-out videos" leak.
GROUP = re.compile(r"^(.*?)(?:_highlight|_temp)")


def group_of(name: str) -> str:
    m = GROUP.search(name)
    return m.group(1) if m else os.path.splitext(name)[0]


def main():
    global GROUP
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out")
    ap.add_argument("--group", help="regex whose first group is the source video")
    a = ap.parse_args()
    root, out = a.root, a.out
    if a.group:
        GROUP = re.compile(a.group)
    # These tools never use sentence-transformers; optimum.intel imports it
    # whenever it is installed, and a broken install (e.g. a torchcodec that
    # cannot load) would take CLIP down with it.
    sys.modules.setdefault("sentence_transformers", None)
    from modules.teach.benchmark import read_dataset

    data = read_dataset(root)
    clips = data["clips"]
    done = {}
    if os.path.exists(out):
        z = np.load(out, allow_pickle=True)
        done = {p: i for i, p in enumerate(z["paths"])}
        rows = {k: list(z[k]) for k in ("paths", "labels", "split", "video", "clip", "pose", "motion")}
    else:
        rows = {k: [] for k in ("paths", "labels", "split", "video", "clip", "pose", "motion")}
    todo = [c for c in clips if os.path.relpath(c.path, root) not in done]
    print(f"ds_features: {len(clips)} clips, {len(todo)} to read", flush=True)
    if not todo:
        return

    from modules.teach.embed import ClipBackend, unit
    from modules.vision.detection_backend import YoloxOpenVINODetector, find_default_yolox_ir
    from modules.vision.pose_backend import build_pose_estimator

    clip = ClipBackend()
    det = YoloxOpenVINODetector(find_default_yolox_ir(prefer="large"), class_names=["person"],
                                device="GPU", score_thr=0.05)
    est = build_pose_estimator(device="GPU")
    t0 = time.time()
    for i, c in enumerate(todo, 1):
        try:
            cf = D.frames_at(c.path, D.CLIP_FRAMES)
            cv = unit(unit(clip.images(cf)).mean(0)) if cf else np.zeros(512, np.float32)
            pv = D.pose_features(D.frames_at(c.path, D.POSE_FRAMES), det, est)
            mv = D.motion_features(D.frames_at(c.path, D.MOTION_FRAMES, consecutive=True, size=(64, 64)))
        except Exception as exc:  # noqa: BLE001
            print(f"ds_features: {c.path}: {type(exc).__name__}: {exc}", flush=True)
            continue
        rows["paths"].append(os.path.relpath(c.path, root))
        rows["labels"].append("|".join(c.labels))
        rows["split"].append(c.split)
        rows["video"].append(group_of(os.path.basename(c.path)))
        rows["clip"].append(cv)
        rows["pose"].append(pv)
        rows["motion"].append(mv)
        if i % 100 == 0 or i == len(todo):
            print(f"ds_features: {i}/{len(todo)} ({(time.time() - t0) / i:.2f}s each)", flush=True)
            np.savez(out, **{k: np.array(v) if k in ("paths", "labels", "split", "video")
                             else np.stack(v) for k, v in rows.items()})


if __name__ == "__main__":
    main()
