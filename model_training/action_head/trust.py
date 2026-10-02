"""How far a taught head can be trusted, from its held-out predictions.

numpy only: the trainer computes this from out-of-fold probabilities, and the
app will read the thresholds back without needing torch.
"""
from __future__ import annotations

import numpy as np


def wilson_low(k, n, z: float = 1.28):
    """Lower end of the 80 % interval for ``k`` right out of ``n``."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def trust_thresholds(proba: np.ndarray, y: np.ndarray, target: float = 0.7,
                     min_n: int = 3) -> list:
    """Per class, the lowest confidence above which its held-out predictions are
    right often enough: the Wilson lower bound of their precision reaches
    ``target``. None when it never does: that class is only ever a suggestion.
    A handful of lucky hits cannot vouch for a class; dozens can."""
    pred, conf = proba.argmax(1), proba.max(1)
    out: list = []
    for c in range(proba.shape[1]):
        mine = pred == c
        if mine.sum() < min_n:
            out.append(None)
            continue
        order = np.argsort(-conf[mine], kind="stable")
        right = (np.asarray(y)[mine][order] == c).astype(float)
        n = np.arange(1, len(right) + 1)
        good = np.where((wilson_low(np.cumsum(right), n) >= target) & (n >= min_n))[0]
        out.append(float(conf[mine][order][good[-1]]) if len(good) else None)
    return out


def trusted_mask(proba: np.ndarray, thresholds: list) -> np.ndarray:
    pred, conf = proba.argmax(1), proba.max(1)
    return np.array([thresholds[c] is not None and cf >= thresholds[c]
                     for c, cf in zip(pred, conf)], bool)


def top_k_hit(proba: np.ndarray, y: np.ndarray, k: int) -> float:
    if len(y) == 0:
        return 0.0
    top = np.argsort(-proba, 1)[:, :k]
    return float(np.mean([t in row for t, row in zip(y, top)]))


def balanced_accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    per = [np.mean(pred[y == c] == c) for c in np.unique(y)]
    return float(np.mean(per)) if per else 0.0
