"""Score the eval_big head on the action trainer's own source-video split.

    python tools/teach_lab/compare_split.py <ds_features.npz> <ds_big.npz> <source_split.json>
        [--blocks siglip,dino,vjepa] [--vid <ds video_features.npz>] [--min-train 5] [--seeds 3] [--quiet]

--blocks picks the frozen encoders: per-frame arrays in <ds_big.npz> (N x 8 x D,
any key, e.g. pose_regions.py's views), flat ones (intel, r21d, clip8) from
--vid, or flat ones from <ds_features.npz> (pose). A flat block next to
per-frame ones is repeated per frame. source_split.json is what the action
trainer writes with --split-by-source. --seeds trains that many heads; one
head's accuracy moves 1-4 points with the seed on 445 clips.

Trains on clips from the split's train videos, scores the clips from its val
videos -- the same 445-ish clips the trainer validates on -- so the numbers
sit next to the trainer's validation accuracy like for like.
"""
import argparse
import json
import sys
from collections import Counter

import numpy as np
import torch

import eval_big as B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds"); ap.add_argument("big"); ap.add_argument("split")
    ap.add_argument("--min-train", type=int, default=5)
    ap.add_argument("--balance", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, default=1, help="train this many heads; report each and their mean")
    ap.add_argument("--quiet", action="store_true", help="no per-class lines")
    ap.add_argument("--blocks", default="siglip,dino,vjepa",
                    help="per-frame blocks from --big, or flat ones from --vid (e.g. intel)")
    ap.add_argument("--vid", help="video_features.py output, for flat blocks like intel/r21d/clip8")
    args = ap.parse_args()
    z = np.load(args.ds, allow_pickle=True)
    b = np.load(args.big, allow_pickle=True)
    v = np.load(args.vid, allow_pickle=True) if args.vid else None
    wv = {n: i for i, n in enumerate(v["names"])} if v is not None else {}
    sp = json.load(open(args.split, encoding="utf-8"))
    tv, vv = set(sp["train_videos"]), set(sp["val_videos"])
    wb = {n: i for i, n in enumerate(b["names"])}
    single = [i for i, (p, l) in enumerate(zip(z["paths"], z["labels"]))
              if "|" not in l and p in wb and (v is None or p in wv)]
    y = z["labels"][single].astype(str)
    vid = z["video"][single].astype(str)
    is_tr = np.isin(vid, list(tv))
    is_va = np.isin(vid, list(vv))
    counts = Counter(y[is_tr])
    keep = np.array([counts[c] >= args.min_train for c in y])
    tr = np.where(is_tr & keep)[0]
    va = np.where(is_va & keep)[0]
    rows = [wb[p] for p in z["paths"][single]]
    vrows = [wv[p] for p in z["paths"][single]] if v is not None else None
    def block(k):
        if k in b.files:                                   # per frame, N x 8 x D
            return b[k][rows].astype(np.float32)
        if v is not None and k in v.files:                 # flat, per clip
            return v[k][vrows].astype(np.float32)[:, None, :]
        return z[k][single].astype(np.float32)[:, None, :]  # flat, from --ds (e.g. pose)
    seqs = [block(k) for k in args.blocks.split(",")]
    t = max(s.shape[1] for s in seqs)                     # flat next to per-frame: repeat it per frame
    seqs = [np.repeat(s, t, 1) if s.shape[1] == 1 else s for s in seqs]
    classes = np.array(sorted(set(y[tr])))
    va = va[np.isin(y[va], classes)]
    yi = np.searchsorted(classes, y)
    xs = B.standardise(seqs, tr)
    accs, logits = [], 0
    for seed in range(args.seeds):
        m = B.train_head([x[tr] for x in xs], yi[tr], vid[tr], len(classes), con=0, balance=args.balance,
                         seed=seed)
        with torch.no_grad():
            lg = torch.softmax(m([torch.as_tensor(x[va]) for x in xs])[1], 1).numpy()
        accs.append(np.mean(lg.argmax(1) == yi[va]))
        logits = logits + lg
    pred = logits.argmax(1)
    top3 = np.argsort(-logits, 1)[:, :3]
    acc = np.mean(pred == yi[va])
    per = {c: float(np.mean(pred[yi[va] == k] == k)) for k, c in enumerate(classes) if np.any(yi[va] == k)}
    print(f"[{args.blocks}] head on the trainer's split: {len(tr)} train clips, {len(va)} val clips from {len(vv)} videos, "
          f"{len(classes)} classes")
    if args.seeds > 1:
        print(f"  per seed {' '.join(f'{a:.3f}' for a in accs)}  mean {np.mean(accs):.3f}  "
              f"(the lines below are the {args.seeds} seeds' averaged probabilities)")
    print(f"  accuracy {acc:.3f}  top-3 {np.mean([t in r for t, r in zip(yi[va], top3)]):.3f}  "
          f"mean per-class {np.mean(list(per.values())):.3f}")
    for c, a in ([] if args.quiet else sorted(per.items(), key=lambda kv: -np.sum(y[va] == kv[0]))):
        print(f"  {c:22} {np.sum(y[va] == c):4}  {a:.2f}")


if __name__ == "__main__":
    sys.exit(main())
