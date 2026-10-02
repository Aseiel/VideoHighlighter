"""Do stronger backbones and a similarity learned from the sorted dataset group
footage the way it was sorted?

    python tools/teach_lab/eval_big.py <ds_features.npz> <ds_video_features.npz> <ds_big.npz> [--out res.json]

All scores hold out whole source videos (GroupKFold, 5 folds).

  A  linear probe per backbone (mean over frames) vs the current mix
  B  learned similarity: a small head over per-frame features, trained with
     cross-entropy + a contrastive term whose positives are the same class in
     a DIFFERENT source video (so "same scene" stops counting as similar).
     Scored as a classifier and as unsupervised grouping of the held-out videos
     (KMeans, k = classes): purity, NMI with class, NMI with source video.
  C  unseen classes: the head is trained without 6 classes; their held-out
     clips are grouped in its space -- does the learned notion of similarity
     carry over to categories it was never shown?
"""
import argparse
import json
import sys
import warnings
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import eval_grouping as E

warnings.filterwarnings("ignore")
torch.manual_seed(0)
BIG = ("siglip", "dino", "vjepa")


def load(ds, vid, big, min_clips=20):
    z = np.load(ds, allow_pickle=True)
    b = np.load(big, allow_pickle=True)
    v = np.load(vid, allow_pickle=True)
    wb = {n: i for i, n in enumerate(b["names"])}
    wv = {n: i for i, n in enumerate(v["names"])}
    single = np.array(["|" not in l for l in z["labels"]]) & np.isin(z["split"], ["train", "val"])
    have = np.array([p in wb and p in wv for p in z["paths"]])
    idx = np.where(single & have)[0]
    y = z["labels"][idx].astype(str)
    counts = Counter(y)
    keep = np.array([counts[c] >= min_clips for c in y])
    idx, y = idx[keep], y[keep]
    rb = [wb[p] for p in z["paths"][idx]]
    rv = [wv[p] for p in z["paths"][idx]]
    seq = {k: b[k][rb].astype(np.float32) for k in BIG}                       # N, 8, D
    flat = {k: s.mean(1) for k, s in seq.items()}
    for k in ("clip8", "r21d", "intel"):
        flat[k] = v[k][rv].astype(np.float32)
    flat["clip"] = z["clip"][idx].astype(np.float32)
    return seq, flat, y, z["video"][idx].astype(str)


# ---------------------------------------------------------------- learned head
class Head(nn.Module):
    def __init__(self, dims, n_cls, hid=384, emb=128, p=0.3):
        super().__init__()
        self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Dropout(p), nn.Linear(d, hid)) for d in dims])
        self.mix = nn.Sequential(nn.GELU(), nn.Linear(hid * len(dims), hid), nn.GELU())
        self.att = nn.Linear(hid, 1)
        self.out = nn.Linear(hid, emb)
        self.cls = nn.Parameter(torch.randn(n_cls, emb) * 0.02)

    def embed(self, xs):
        h = self.mix(torch.cat([p(x) for p, x in zip(self.proj, xs)], -1))   # B, T, hid
        w = torch.softmax(self.att(h), 1)
        return F.normalize(self.out((w * h).sum(1)), dim=-1)

    def forward(self, xs):
        e = self.embed(xs)
        return e, 16.0 * e @ F.normalize(self.cls, dim=-1).T


def supcon_cross_video(e, y, v, t=0.1):
    """Positives: same class, other source video. Same-video same-class pairs are
    left out entirely (neither pulled nor pushed)."""
    sim = e @ e.T / t
    same_c = y[:, None] == y[None, :]
    same_v = v[:, None] == v[None, :]
    pos = same_c & ~same_v
    valid = ~(same_c & same_v)                 # drops the diagonal too
    sim = sim.masked_fill(~valid, -1e9)
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    n = pos.sum(1)
    has = n > 0
    if not has.any():
        return sim.new_zeros(())
    return -((logp * pos).sum(1)[has] / n[has]).mean()


def train_head(xs_tr, y_tr, v_tr, n_cls, steps=1500, con=0.5, seed=0, bs=128, balance=1.0):
    """balance: class weight = (n / count) ** balance -- 1 balanced, 0 plain."""
    torch.manual_seed(seed)
    dims = [x.shape[-1] for x in xs_tr]
    m = Head(dims, n_cls)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.05)
    yt = torch.as_tensor(y_tr)
    _, vt = np.unique(v_tr, return_inverse=True)
    vt = torch.as_tensor(vt)
    w = torch.as_tensor((len(y_tr) / (n_cls * np.bincount(y_tr, minlength=n_cls).clip(1))) ** balance,
                        dtype=torch.float32)
    xt = [torch.as_tensor(x) for x in xs_tr]
    n = len(y_tr)
    per_epoch = (n + bs - 1) // bs
    epochs = max(1, steps // per_epoch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * per_epoch)
    for _ in range(epochs):
        m.train()
        for b in torch.randperm(n).split(bs):
            xb = [x[b] for x in xt]
            keep = torch.rand(xb[0].shape[1]) > 0.25          # drop frames
            if keep.sum() >= 2:
                xb = [x[:, keep] for x in xb]
            e, logits = m(xb)
            loss = F.cross_entropy(logits, yt[b], weight=w, label_smoothing=0.1)
            if con:
                loss = loss + con * supcon_cross_video(e, yt[b], vt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return m.eval()


def standardise(seqs, tr):
    out = []
    for s in seqs:
        mu = s[tr].reshape(-1, s.shape[-1]).mean(0)
        sd = s[tr].reshape(-1, s.shape[-1]).std(0) + 1e-5
        out.append(((s - mu) / sd).astype(np.float32))
    return out


def group_scores(z, y, v, k):
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score as nmi
    g = KMeans(k, n_init=10, random_state=0).fit_predict(z)
    return E.purity(y, g), nmi(y, g), nmi(v, g)


def eval_head(seqs, y, video, splits, con, label):
    classes = np.array(sorted(set(y)))
    yi = np.searchsorted(classes, y)
    pred = np.empty(len(y), dtype=object)
    top3 = [None] * len(y)
    gp, gn, gv = [], [], []
    emb = np.zeros((len(y), 128), np.float32)
    for tr, te in splits:
        xs = standardise(seqs, tr)
        m = train_head([x[tr] for x in xs], yi[tr], video[tr], len(classes), con=con)
        with torch.no_grad():
            e, logits = m([torch.as_tensor(x[te]) for x in xs])
        order = logits.argsort(1, descending=True).numpy()
        pred[te] = classes[order[:, 0]]
        for i, row in zip(te, order[:, :3]):
            top3[i] = set(classes[row])
        emb[te] = e.numpy()
        p, n, vv = group_scores(e.numpy(), y[te], video[te], len(classes))
        gp.append(p); gn.append(n); gv.append(vv)
    s = E.scores(y, pred, top3)
    s.update(purity=float(np.mean(gp)), nmi=float(np.mean(gn)), nmi_video=float(np.mean(gv)))
    print(f"   {label:34} acc {s['acc']:.3f} bal {s['bal']:.3f} top3 {s['top3']:.3f} | "
          f"groups purity {s['purity']:.2f} NMI class {s['nmi']:.2f} video {s['nmi_video']:.2f}", flush=True)
    return s, emb


def unseen_classes(seqs, y, video, n_out=6, seeds=(0, 1, 2)):
    """Train on all but n_out mid-sized classes (all their clips removed), then
    group the held-out classes' clips from videos the head never saw."""
    from sklearn.decomposition import PCA
    from sklearn.model_selection import GroupKFold
    counts = Counter(y)
    pool = [c for c, _ in counts.most_common()][2:]          # skip the two biggest
    res = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        out = set(rng.choice(pool, n_out, replace=False))
        seen = np.array([c not in out for c in y])
        # hold out a third of the videos as "new footage"
        vids = np.unique(video)
        new_v = set(rng.choice(vids, len(vids) // 3, replace=False))
        is_new = np.array([v in new_v for v in video])
        tr = np.where(seen & ~is_new)[0]
        te = np.where(~seen & is_new)[0]
        if len(set(y[te])) < 3:
            continue
        classes = np.array(sorted(set(y[tr])))
        xs = standardise(seqs, tr)
        m = train_head([x[tr] for x in xs], np.searchsorted(classes, y[tr]), video[tr], len(classes))
        with torch.no_grad():
            e = m.embed([torch.as_tensor(x[te]) for x in xs]).numpy()
        k = len(set(y[te]))
        raw = np.concatenate([x.mean(1) for x in xs], 1)
        rawz = PCA(32, random_state=0).fit(raw[tr]).transform(raw[te])
        res.append({"classes": sorted(out & set(y[te])), "clips": int(len(te)),
                    "learned": group_scores(e, y[te], video[te], k),
                    "raw": group_scores(rawz, y[te], video[te], k)})
        r = res[-1]
        print(f"   seed {seed}: {k} unseen classes, {len(te)} clips | learned purity {r['learned'][0]:.2f} "
              f"NMI {r['learned'][1]:.2f} (video {r['learned'][2]:.2f}) | raw purity {r['raw'][0]:.2f} "
              f"NMI {r['raw'][1]:.2f} (video {r['raw'][2]:.2f})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds"); ap.add_argument("vid"); ap.add_argument("big")
    ap.add_argument("--out")
    ap.add_argument("--skip", default="", help="comma list of A,B,C to skip")
    args = ap.parse_args()
    seq, flat, y, video = load(args.ds, args.vid, args.big)
    splits = E.folds(y, video, 5)
    k = len(set(y))
    print(f"{len(y)} clips, {k} classes, {len(set(video))} videos\n")
    result = {"clips": len(y), "classes": k}

    if "A" not in args.skip:
        print("A. linear probe (logreg C=1 balanced), held-out videos; raw grouping purity/NMI class/NMI video")
        result["A"] = {}
        for name, w in {"current clip8+r21d+intel": {"clip8": 1, "r21d": 1, "intel": 1},
                        "siglip": {"siglip": 1}, "dino": {"dino": 1}, "vjepa": {"vjepa": 1},
                        "siglip+vjepa": {"siglip": 1, "vjepa": 1},
                        "siglip+dino+vjepa": {"siglip": 1, "dino": 1, "vjepa": 1},
                        "all six": {"siglip": 1, "dino": 1, "vjepa": 1, "clip8": 1, "r21d": 1, "intel": 1}}.items():
            s = E.logreg(flat, w, y, splits)
            x = E.mix(flat, w)
            (nc, nv), _ = __import__("diagnose_grouping").kmeans_nmi(x, [y, video], k)
            result["A"][name] = {**s, "nmi": nc, "nmi_video": nv}
            print(f"   {name:26} acc {s['acc']:.3f} bal {s['bal']:.3f} top3 {s['top3']:.3f} | "
                  f"KMeans NMI class {nc:.2f} video {nv:.2f}", flush=True)

    seqs = [seq[k_] for k_ in BIG]
    if "B" not in args.skip:
        print("\nB. learned head over per-frame siglip+dino+vjepa, held-out videos")
        result["B"] = {}
        result["B"]["ce only"], _ = eval_head(seqs, y, video, splits, 0.0, "cross-entropy only")
        result["B"]["ce+cross-video contrastive"], emb = eval_head(seqs, y, video, splits, 0.5,
                                                                   "+ cross-video contrastive")
    if "C" not in args.skip:
        print("\nC. classes the head never saw, from videos it never saw")
        result["C"] = unseen_classes(seqs, y, video)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    sys.exit(main())
