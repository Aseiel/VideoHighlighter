"""INT8 versions of the exported SigLIP2 towers (NNCF), checked on real frames.

    python tools/teach_lab/quantize_siglip.py <export dir> --dataset <dataset root> [--calib 300] [--check 200]

  vision_int8.xml   image tower, post-training INT8 (weights and activations),
                    calibrated on --calib frames from the dataset's clips
  text_int8w.xml    text tower, INT8 weights only (it is mostly the 256k-token
                    embedding table)

Check: worst and mean cosine against the fp32 PyTorch-equivalent IR on --check
frames from other clips than the calibration ones.
"""
import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_regions import letterbox  # noqa: E402
from video_features import evenly, read_all  # noqa: E402


def frames(paths, per_clip=2, size=256):
    out = []
    for p in paths:
        for f in evenly(read_all(p, short=size), per_clip):
            out.append(((letterbox(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), size).astype(np.float32) / 255 - 0.5)
                        / 0.5).transpose(2, 0, 1))
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--calib", type=int, default=300)
    ap.add_argument("--check", type=int, default=200)
    args = ap.parse_args()
    import nncf
    import openvino as ov
    core = ov.Core()
    clips = sorted(glob.glob(os.path.join(args.dataset, "**", "*.mp4"), recursive=True))
    random.Random(0).shuffle(clips)
    calib = frames(clips[:args.calib // 2])
    check = frames(clips[args.calib // 2: args.calib // 2 + args.check // 2])
    print(f"calibration {len(calib)} frames, check {len(check)} frames", flush=True)

    vis = core.read_model(os.path.join(args.dir, "vision.xml"))
    q = nncf.quantize(vis, nncf.Dataset(list(calib[:, None])), model_type=nncf.ModelType.TRANSFORMER,
                      subset_size=len(calib))
    ov.save_model(q, os.path.join(args.dir, "vision_int8.xml"))
    txt = core.read_model(os.path.join(args.dir, "text.xml"))
    ov.save_model(nncf.compress_weights(txt, mode=nncf.CompressWeightsMode.INT8_ASYM),
                  os.path.join(args.dir, "text_int8w.xml"))

    ref = core.compile_model(os.path.join(args.dir, "vision.xml"), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    r = np.concatenate([ref(check[i:i + 16])[0] for i in range(0, len(check), 16)])
    for dev in ("CPU", "GPU"):
        net = core.compile_model(os.path.join(args.dir, "vision_int8.xml"), dev)
        o = np.concatenate([net(check[i:i + 16])[0] for i in range(0, len(check), 16)])
        c = (o * r).sum(1) / np.linalg.norm(o, axis=1) / np.linalg.norm(r, axis=1)
        print(f"vision_int8 on {dev} vs fp32: cosine worst {c.min():.4f} mean {c.mean():.4f}")
    for f in ("vision_int8.bin", "text_int8w.bin"):
        print(f"  {f:18} {os.path.getsize(os.path.join(args.dir, f)) / 1e6:8.1f} MB")


if __name__ == "__main__":
    sys.exit(main())
