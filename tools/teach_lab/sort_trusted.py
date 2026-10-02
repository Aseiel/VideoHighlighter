"""Sort new footage only into classes that earned it on held-out videos.

    python tools/teach_lab/sort_trusted.py <ds_features.npz> <ds_video_features.npz> <ds_big.npz>
        <crops_big.npz> <crops folder> <out folder> [--precision 0.7] [--balance 1,0.5,0]

1. For each class-weight setting (--balance: 1 balanced, 0 plain), the eval_big
   head is scored on held-out videos (GroupKFold 5).
2. Per class, the lowest confidence at which its predictions are right at
   least --precision of the time (with 3+ predictions above it). A class that
   never gets there is not trusted: it is only ever a suggestion.
3. The setting with the most clips sorted at that precision is trained on the
   whole dataset and sorts the crops: <out>/<class>/ for a trusted class above
   its threshold, <out>/_unsure/ for the rest. sorted.csv has every crop's
   top three guesses either way; summary.json the thresholds.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

import eval_big as B
import eval_grouping as E


def heldout_proba(seqs, yi, video, splits, n_cls, balance):
    p = np.zeros((len(yi), n_cls), np.float32)
    for tr, te in splits:
        xs = B.standardise(seqs, tr)
        m = B.train_head([x[tr] for x in xs], yi[tr], video[tr], n_cls, con=0, balance=balance)
        with torch.no_grad():
            p[te] = torch.softmax(m([torch.as_tensor(x[te]) for x in xs])[1], 1).numpy()
    return p


def wilson_low(k, n, z=1.28):
    """Lower end of the 80% interval for k right out of n: what n examples can vouch for."""
    p = k / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def thresholds(p, yi, target, min_n=3):
    """A class's threshold: the lowest confidence where the Wilson lower bound of
    its held-out precision still reaches ``target``. A handful of lucky hits
    cannot vouch for a class; dozens can."""
    pred, conf = p.argmax(1), p.max(1)
    out = {}
    for c in range(p.shape[1]):
        mk = pred == c
        if mk.sum() < min_n:
            out[c] = None
            continue
        order = np.argsort(-conf[mk])
        right = (yi[mk][order] == c).astype(float)
        n = np.arange(1, len(right) + 1)
        low = wilson_low(np.cumsum(right), n)
        ok = np.where((low >= target) & (n >= min_n))[0]
        out[c] = float(conf[mk][order][ok[-1]]) if len(ok) else None
    return out


def gated(p, th):
    pred, conf = p.argmax(1), p.max(1)
    return np.array([th[c] is not None and cf >= th[c] for c, cf in zip(pred, conf)])


def main():
    ap = argparse.ArgumentParser()
    for a in ("ds", "vid", "big", "crops_big", "crops", "out"):
        ap.add_argument(a)
    ap.add_argument("--precision", type=float, default=0.7)
    ap.add_argument("--balance", default="1,0.5,0")
    ap.add_argument("--min", type=int, default=8)
    args = ap.parse_args()

    seq, _, y, video = B.load(args.ds, args.vid, args.big, min_clips=args.min)
    seqs = [seq[k] for k in B.BIG]
    classes = np.array(sorted(set(y)))
    yi = np.searchsorted(classes, y)
    splits = E.folds(y, video, 5)

    best = None
    for bal in [float(b) for b in args.balance.split(",")]:
        p = heldout_proba(seqs, yi, video, splits, len(classes), bal)
        th = thresholds(p, yi, args.precision)
        g = gated(p, th)
        pred = p.argmax(1)
        prec = float(np.mean(pred[g] == yi[g])) if g.any() else 0.0
        trusted = sum(t is not None for t in th.values())
        print(f"balance {bal:3}: acc {np.mean(pred == yi):.3f}  sorted {g.mean():.0%} of held-out clips "
              f"at {prec:.0%} right; {trusted}/{len(classes)} classes trusted", flush=True)
        if best is None or g.mean() > best[1]:
            best = (bal, float(g.mean()), th, prec)
    bal, coverage, th, prec = best
    print(f"\nusing balance {bal}\n\n{'class':22} {'threshold':>9}")
    for c in np.argsort(classes):
        print(f"{classes[c]:22} {'not trusted' if th[c] is None else f'{th[c]:.2f}':>11}")

    stats = [(s.reshape(-1, s.shape[-1]).mean(0), s.reshape(-1, s.shape[-1]).std(0) + 1e-5) for s in seqs]
    xs = [((s - mu) / sd).astype(np.float32) for s, (mu, sd) in zip(seqs, stats)]
    m = B.train_head(xs, yi, video, len(classes), con=0, balance=bal)
    c = np.load(args.crops_big, allow_pickle=True)
    names = [str(n) for n in c["names"]]
    cx = [((c[k].astype(np.float32) - mu) / sd).astype(np.float32) for k, (mu, sd) in zip(B.BIG, stats)]
    with torch.no_grad():
        p = torch.softmax(m([torch.as_tensor(x) for x in cx])[1], 1).numpy()
    g = gated(p, th)
    pred = p.argmax(1)

    os.makedirs(args.out, exist_ok=True)
    rows, counts, suggested = [], Counter(), Counter()
    for i, n in enumerate(names):
        top = np.argsort(-p[i])[:3]
        folder = classes[pred[i]] if g[i] else "_unsure"
        counts[folder] += 1
        if not g[i]:
            suggested[classes[pred[i]]] += 1
        os.makedirs(os.path.join(args.out, folder), exist_ok=True)
        try:
            os.link(os.path.join(args.crops, n), os.path.join(args.out, folder, n))
        except OSError:
            pass
        rows.append({"crop": n, "folder": folder,
                     **{f"guess{k + 1}": classes[t] for k, t in enumerate(top)},
                     **{f"p{k + 1}": round(float(p[i, t]), 3) for k, t in enumerate(top)}})
    with open(os.path.join(args.out, "sorted.csv"), "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    summary = {"precision_target": args.precision, "balance": bal, "heldout_coverage": coverage,
               "heldout_precision": prec,
               "thresholds": {classes[k]: v for k, v in th.items()},
               "sorted": dict(counts.most_common()), "unsure_best_guess": dict(suggested.most_common())}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nnew video: {g.mean():.0%} of {len(names)} crops sorted")
    for k, v in counts.most_common():
        print(f"  {k:22} {v}")
    print("unsure, by best guess: " + ", ".join(f"{k} {v}" for k, v in suggested.most_common(8)))


if __name__ == "__main__":
    sys.exit(main())
