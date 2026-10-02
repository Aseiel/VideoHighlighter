"""Which classes can be trusted? Per class, on held-out videos, with the eval_big head.

    python tools/teach_lab/class_report.py <ds_features.npz> <ds_video_features.npz> <ds_big.npz> <crops_big.npz> [--min 8]

Per class: clips, how many source videos they come from, held-out recall and
precision, what it is mistaken for -- and how many of the new video's crops the
head (trained on everything) puts in it, at what confidence.
"""
import argparse
import sys
from collections import Counter

import numpy as np
import torch

import eval_big as B
import eval_grouping as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds"); ap.add_argument("vid"); ap.add_argument("big"); ap.add_argument("crops")
    ap.add_argument("--min", type=int, default=8)
    args = ap.parse_args()
    seq, _, y, video = B.load(args.ds, args.vid, args.big, min_clips=args.min)
    seqs = [seq[k] for k in B.BIG]
    classes = np.array(sorted(set(y)))
    yi = np.searchsorted(classes, y)
    splits = E.folds(y, video, 5)
    pred = np.empty(len(y), dtype=object)
    for tr, te in splits:
        xs = B.standardise(seqs, tr)
        m = B.train_head([x[tr] for x in xs], yi[tr], video[tr], len(classes), con=0)
        with torch.no_grad():
            _, logits = m([torch.as_tensor(x[te]) for x in xs])
        pred[te] = classes[logits.argmax(1).numpy()]
    print(f"held-out acc {np.mean(pred == y):.3f} over {len(y)} clips, {len(classes)} classes\n")

    # the head on everything, applied to the new video's crops
    all_idx = np.arange(len(y))
    xs = B.standardise(seqs, all_idx)
    stats = [(s.reshape(-1, s.shape[-1]).mean(0), s.reshape(-1, s.shape[-1]).std(0) + 1e-5) for s in seqs]
    m = B.train_head(xs, yi, video, len(classes), con=0)
    c = np.load(args.crops, allow_pickle=True)
    cx = [((c[k].astype(np.float32) - mu) / sd).astype(np.float32) for k, (mu, sd) in zip(B.BIG, stats)]
    with torch.no_grad():
        p = torch.softmax(m([torch.as_tensor(x) for x in cx])[1], 1).numpy()
    cpred = classes[p.argmax(1)]
    cconf = p.max(1)

    print(f"{'class':22} {'clips':>5} {'videos':>6} {'top video':>9} {'recall':>6} {'prec':>5}  "
          f"{'new: crops':>10} {'conf>.5':>7}  mistaken for")
    for cl in sorted(classes, key=lambda c: -np.sum(y == c)):
        mk = y == cl
        vids = Counter(video[mk])
        rec = np.mean(pred[mk] == cl)
        pp = pred == cl
        prec = np.mean(y[pp] == cl) if pp.any() else float("nan")
        wrong = Counter(pred[mk & (pred != y)]).most_common(2)
        nc = cpred == cl
        print(f"{cl:22} {mk.sum():5} {len(vids):6} {vids.most_common(1)[0][1] / mk.sum():9.0%} "
              f"{rec:6.2f} {prec:5.2f}  {nc.sum():10} {np.sum(nc & (cconf > .5)):7}  "
              + ", ".join(f"{k} {v}" for k, v in wrong))


if __name__ == "__main__":
    sys.exit(main())
