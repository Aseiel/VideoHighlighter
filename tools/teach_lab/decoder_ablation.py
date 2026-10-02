"""Same Intel features the trainer used, different things on top.

    python tools/teach_lab/decoder_ablation.py <checkpoint_dir of an intel trainer run>

Reads the trainer's own feature caches (train + val, its crop and its frame
sampling, its labels), so the input is identical to what its decoder saw. Only
the part trained on top changes:

  head/frames   eval_big's attention head over the 16 per-frame vectors
  head/mean+std the same head on one mean+std vector per clip (as video_features.py pools)
  logreg        a linear layer on mean+std
each with class weights balanced (power 1, the trainer's) and softened (0.5).
"""
import glob
import os
import sys
import warnings

import numpy as np
import torch

import eval_big as B

warnings.filterwarnings("ignore")


def load(ckpt_dir):
    caches = [torch.load(f, weights_only=False) for f in glob.glob(os.path.join(ckpt_dir, "feature_cache_*.pt"))]
    caches.sort(key=lambda d: -d["num_samples"])            # the train cache is the bigger one
    tr, va = caches
    return (tr["features"].float().numpy(), tr["labels"].numpy(),
            va["features"].float().numpy(), va["labels"].numpy())


def head(xtr, ytr, xva, yva, balance, seeds=(0, 1, 2)):
    n_cls = int(max(ytr.max(), yva.max())) + 1
    mu = xtr.reshape(-1, xtr.shape[-1]).mean(0)
    sd = xtr.reshape(-1, xtr.shape[-1]).std(0) + 1e-5
    a, b = ((xtr - mu) / sd).astype(np.float32), ((xva - mu) / sd).astype(np.float32)
    accs = []
    for s in seeds:
        # one pseudo-video per clip: the contrastive term is off (con=0), so it is unused
        m = B.train_head([a], ytr, np.arange(len(ytr)).astype(str), n_cls, con=0, balance=balance, seed=s)
        with torch.no_grad():
            pred = m([torch.as_tensor(b)])[1].argmax(1).numpy()
        accs.append(np.mean(pred == yva))
    return float(np.mean(accs)), float(np.std(accs))


def logreg(xtr, ytr, xva, yva, balance):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(xtr)
    cw = None
    if balance:
        cnt = np.bincount(ytr)
        cw = {c: (len(ytr) / (len(cnt) * n)) ** balance for c, n in enumerate(cnt) if n}
    m = LogisticRegression(C=1, max_iter=3000, class_weight=cw).fit(sc.transform(xtr), ytr)
    return float(np.mean(m.predict(sc.transform(xva)) == yva))


def main():
    xtr, ytr, xva, yva = load(sys.argv[1])
    print(f"trainer's features: train {xtr.shape}, val {xva.shape}\n")
    pool = lambda x: np.concatenate([x.mean(1), x.std(1)], 1)[:, None, :]
    for bal in (1.0, 0.5):
        m, s = head(xtr, ytr, xva, yva, bal)
        print(f"head over 16 frames,     class weights ^{bal}: {m:.3f} (+-{s:.3f} over 3 seeds)", flush=True)
        m, s = head(pool(xtr), ytr, pool(xva), yva, bal)
        print(f"head on mean+std,        class weights ^{bal}: {m:.3f} (+-{s:.3f})", flush=True)
        print(f"logreg on mean+std,      class weights ^{bal}: "
              f"{logreg(pool(xtr)[:, 0], ytr, pool(xva)[:, 0], yva, bal):.3f}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
