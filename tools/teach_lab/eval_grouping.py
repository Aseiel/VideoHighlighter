"""How well do the grouping features sort the way a person sorted?

    python tools/teach_lab/eval_grouping.py <ds_features.npz> [--extra <ds video_features.npz>]
        [--min 20] [--folds 5] [--mixes mixes.json]

Uses the hand-sorted dataset as the answer key. Every score holds out whole
source videos (GroupKFold on the video a clip came from), so nothing is
matched against footage from its own scene.

  cluster   KMeans with one group per class, no labels used: purity / NMI / ARI
  knn       the nearest labelled clips (cosine) vote
  proto     teach's way: 3 CLIP centres per class, nearest centre wins
  logreg    a linear layer trained on the features
"""
import argparse
import itertools
import json
import sys
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")


EXTRA_BLOCKS = ("intel", "kinetics", "r21d", "clip8")


def load(path, min_clips, extra=None):
    """Single-class train/val clips of classes with ``min_clips`` or more.

    ``extra``: a video_features.npz, joined on the clip's path; clips it lacks
    are dropped so every block covers the same clips.
    """
    z = np.load(path, allow_pickle=True)
    single = np.array(["|" not in l for l in z["labels"]]) & np.isin(z["split"], ["train", "val"])
    ex = None
    if extra:
        ex = np.load(extra, allow_pickle=True)
        where = {n: i for i, n in enumerate(ex["names"])}
        single &= np.array([p in where for p in z["paths"]])
    labels = z["labels"][single]
    counts = Counter(labels)
    keep = np.array([counts[l] >= min_clips for l in labels])
    idx = np.where(single)[0][keep]
    small = {k: v for k, v in counts.items() if v < min_clips}
    blocks = {k: z[k][idx].astype(np.float32) for k in ("clip", "pose", "motion")}
    if ex is not None:
        rows = [where[p] for p in z["paths"][idx]]
        for k in EXTRA_BLOCKS:
            blocks[k] = ex[k][rows].astype(np.float32)
    return blocks, z["labels"][idx], z["video"][idx], small


def mix(blocks, weights, fit_idx=None):
    from sklearn.preprocessing import StandardScaler
    parts = []
    for k, w in weights.items():
        if w <= 0:
            continue
        x = blocks[k]
        sc = StandardScaler().fit(x if fit_idx is None else x[fit_idx])
        x = sc.transform(x) / np.sqrt(x.shape[1])
        parts.append(x * np.sqrt(w))
    return np.concatenate(parts, axis=1)


def purity(y, groups):
    total = 0
    for g in set(groups):
        total += Counter(y[groups == g]).most_common(1)[0][1]
    return total / len(y)


def cluster_scores(x, y, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    k = len(set(y))
    x = PCA(n_components=min(32, x.shape[1]), random_state=seed).fit_transform(x)
    g = KMeans(k, n_init=10, random_state=seed).fit_predict(x)
    return {"purity": purity(y, g), "nmi": normalized_mutual_info_score(y, g),
            "ari": adjusted_rand_score(y, g)}


def folds(y, video, n):
    from sklearn.model_selection import GroupKFold
    return list(GroupKFold(n_splits=n).split(np.zeros(len(y)), y, video))


def scores(y, pred, top3=None):
    from sklearn.metrics import balanced_accuracy_score
    out = {"acc": float(np.mean(pred == y)), "bal": float(balanced_accuracy_score(y, pred))}
    if top3 is not None:
        out["top3"] = float(np.mean([t in row for t, row in zip(y, top3)]))
    return out


def knn(x, y, splits, k=10):
    pred = np.empty(len(y), dtype=object)
    xn = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    for tr, te in splits:
        sim = xn[te] @ xn[tr].T
        nn = np.argsort(-sim, axis=1)[:, :k]
        ytr = y[tr]
        for r, (i, row) in enumerate(zip(te, nn)):
            votes = Counter()
            for j, s in zip(row, sim[r][row]):
                votes[ytr[j]] += float(s)
            pred[i] = votes.most_common(1)[0][0]
    return scores(y, pred)


def proto(clipv, y, splits, per_class=3):
    from sklearn.cluster import KMeans
    pred = np.empty(len(y), dtype=object)
    for tr, te in splits:
        cents, names = [], []
        for c in sorted(set(y[tr])):
            xc = clipv[tr][y[tr] == c]
            k = min(per_class, len(xc))
            cc = KMeans(k, n_init=4, random_state=0).fit(xc).cluster_centers_ if k > 1 else xc[:1]
            cents.extend(cc / np.linalg.norm(cc, axis=1, keepdims=True))
            names.extend([c] * len(cc))
        sim = clipv[te] @ np.stack(cents).T
        pred[te] = np.array(names, dtype=object)[sim.argmax(1)]
    return scores(y, pred)


def logreg(blocks, weights, y, splits, C=1.0, return_proba=False):
    from sklearn.linear_model import LogisticRegression
    pred = np.empty(len(y), dtype=object)
    top3 = [None] * len(y)
    conf = np.zeros(len(y))
    for tr, te in splits:
        x = mix(blocks, weights, fit_idx=tr)
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
        m.fit(x[tr], y[tr])
        p = m.predict_proba(x[te])
        order = np.argsort(-p, axis=1)
        pred[te] = m.classes_[order[:, 0]]
        conf[te] = p[np.arange(len(te)), order[:, 0]]
        for i, row in zip(te, order[:, :3]):
            top3[i] = set(m.classes_[row])
    out = scores(y, pred, top3)
    if return_proba:
        return out, pred, conf
    return out


def learned_cluster(blocks, weights, y, splits, C=1.0):
    """Group held-out videos without their labels, in the space a linear layer
    learned from the other videos (its per-class scores). Purity/NMI per fold,
    averaged -- how well discovery on new footage would follow the sorting."""
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import normalized_mutual_info_score
    from sklearn.decomposition import PCA
    pur, nmi, raw_pur, raw_nmi = [], [], [], []
    for tr, te in splits:
        x = mix(blocks, weights, fit_idx=tr)
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(x[tr], y[tr])
        z = m.decision_function(x[te])
        k = min(len(set(y)), len(te) - 1)
        g = KMeans(k, n_init=10, random_state=0).fit_predict(z)
        pur.append(purity(y[te], g))
        nmi.append(normalized_mutual_info_score(y[te], g))
        # the same held-out clips grouped in the raw features, for a like-for-like baseline
        xr = PCA(n_components=min(32, x.shape[1]), random_state=0).fit(x[tr]).transform(x[te])
        gr = KMeans(k, n_init=10, random_state=0).fit_predict(xr)
        raw_pur.append(purity(y[te], gr))
        raw_nmi.append(normalized_mutual_info_score(y[te], gr))
    return {"purity": float(np.mean(pur)), "nmi": float(np.mean(nmi)),
            "raw_purity": float(np.mean(raw_pur)), "raw_nmi": float(np.mean(raw_nmi))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("--min", type=int, default=20)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out")
    ap.add_argument("--extra", help="video_features.npz of the same dataset")
    ap.add_argument("--mixes", help="JSON {name: {block: weight}} to score instead of the defaults")
    args = ap.parse_args()

    blocks, y, video, small = load(args.features, args.min, args.extra)
    splits = folds(y, video, args.folds)
    print(f"clips {len(y)}, classes {len(set(y))}, videos {len(set(video))}; "
          f"left out (<{args.min}): {len(small)} classes, {sum(small.values())} clips", flush=True)
    result = {"clips": len(y), "classes": len(set(y)), "videos": len(set(video)), "left_out": small,
              "chance_acc": max(Counter(y).values()) / len(y), "rows": []}

    mixes = {
        "clip": {"clip": 1, "pose": 0, "motion": 0},
        "pose": {"clip": 0, "pose": 1, "motion": 0},
        "motion": {"clip": 0, "pose": 0, "motion": 1},
        "clip+pose": {"clip": 1, "pose": .8, "motion": 0},
        "all": {"clip": 1, "pose": .8, "motion": .4},
        "movement": {"clip": 0, "pose": 1, "motion": .5},
    }
    if args.mixes:
        mixes = json.loads(open(args.mixes, encoding="utf-8").read()) \
            if args.mixes.endswith(".json") else json.loads(args.mixes)
    print(f"{'mix':14} {'raw groups pur/NMI':>18}  {'learned groups pur/NMI':>22}  {'knn acc':>7}  {'logreg acc/bal/top3':>20}")
    for name, w in mixes.items():
        x = mix(blocks, w)
        cs = cluster_scores(x, y)
        kn = knn(x, y, splits)
        lr = logreg(blocks, w, y, splits)
        lc = learned_cluster(blocks, w, y, splits)
        row = {"mix": name, "weights": w, "cluster": cs, "learned_cluster": lc, "knn": kn, "logreg": lr}
        result["rows"].append(row)
        print(f"{name:14} {lc['raw_purity']:.2f}/{lc['raw_nmi']:.2f}{'':>9}  {lc['purity']:.2f}/{lc['nmi']:.2f}{'':>13}  "
              f"{kn['acc']:.2f}     {lr['acc']:.2f}/{lr['bal']:.2f}/{lr['top3']:.2f}", flush=True)
    pr = proto(blocks["clip"], y, splits)
    result["proto_clip_3"] = pr
    print(f"teach prototypes (CLIP, 3/class): acc {pr['acc']:.2f} bal {pr['bal']:.2f}")
    print(f"majority-class baseline: acc {result['chance_acc']:.2f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1, default=float)


if __name__ == "__main__":
    sys.exit(main())
