"""Classifier settings and per-class errors for one feature mix, videos held out.

    python tools/teach_lab/tune.py <ds_features.npz> <ds video_features.npz> ["name,name"]
"""
import os
import sys
import warnings
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_grouping as E  # noqa: E402

warnings.filterwarnings("ignore")
MIX = {"clip8": 1, "r21d": 1, "intel": 1}


def run(model_factory, blocks, y, splits, pca=None):
    from sklearn.decomposition import PCA
    pred = np.empty(len(y), dtype=y.dtype)
    top3 = [None] * len(y)
    for tr, te in splits:
        x = E.mix(blocks, MIX, fit_idx=tr)
        if pca:
            p = PCA(n_components=pca, random_state=0).fit(x[tr])
            x = p.transform(x)
        m = model_factory().fit(x[tr], y[tr])
        pr = m.predict_proba(x[te])
        order = np.argsort(-pr, axis=1)
        pred[te] = m.classes_[order[:, 0]]
        for i, row in zip(te, order[:, :3]):
            top3[i] = set(m.classes_[row])
    return E.scores(y, pred, top3), pred


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    blocks, y, video, _ = E.load(sys.argv[1], 20, sys.argv[2])
    y = y.astype(str)
    splits = E.folds(y, video, 5)
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
    candidates = {
        "logreg C=0.1": (lambda: LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"), None),
        "logreg C=0.3": (lambda: LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"), None),
        "logreg C=1": (lambda: LogisticRegression(C=1, max_iter=3000, class_weight="balanced"), None),
        "logreg C=3": (lambda: LogisticRegression(C=3, max_iter=3000, class_weight="balanced"), None),
        "logreg C=0.3 unweighted": (lambda: LogisticRegression(C=0.3, max_iter=3000), None),
        "svm rbf C=3": (lambda: SVC(C=3, probability=True, class_weight="balanced", random_state=0), 128),
        "mlp 256": (lambda: MLPClassifier((256,), alpha=1e-1, max_iter=200, random_state=0), None),
    }
    best, best_pred = None, None
    for name, (factory, pca) in candidates.items():
        if only and name not in only:
            continue
        s, pred = run(factory, blocks, y, splits, pca)
        print(f"{name:26} acc {s['acc']:.3f}  bal {s['bal']:.3f}  top3 {s['top3']:.3f}", flush=True)
        if best is None or s["acc"] + s["bal"] > best[1]["acc"] + best[1]["bal"]:
            best, best_pred = (name, s), pred
    print(f"\nbest: {best[0]}")
    counts = Counter(y)
    print(f"\n{'class':22} {'clips':>5} {'recall':>6}  most often taken for")
    for c in sorted(counts, key=lambda c: -counts[c]):
        m = y == c
        rec = float(np.mean(best_pred[m] == c))
        wrong = Counter(best_pred[m & (best_pred != y)]).most_common(2)
        print(f"{c:22} {counts[c]:5} {rec:6.2f}  " + ", ".join(f"{k} {v}" for k, v in wrong))
    conf = Counter((a, b) for a, b in zip(y, best_pred) if a != b)
    pairs = Counter()
    for (a, b), n in conf.items():
        pairs[tuple(sorted((a, b)))] += n
    print("\nmost confused pairs (both directions):")
    for (a, b), n in pairs.most_common(10):
        print(f"  {a} <-> {b}: {n}")


if __name__ == "__main__":
    main()
