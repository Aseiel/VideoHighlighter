"""A linear layer trained on the samples a person sorted. Pure numpy.

``scoring`` compares each sample with one or a few centres per class. That is
all there is to go on when a class has a name and a handful of examples, but a
hand-sorted dataset has hundreds per class, and centres throw most of that
away: which directions in CLIP space tell two neighbouring classes apart is
something only training on both learns. Measured on one such dataset with
whole videos held out, the same CLIP vectors sorted 51% right this way against
37% by nearest centre.

The layer is multinomial logistic regression on standardised vectors, with
classes weighted by their size (a class of 300 does not drown one of 25) and an
L2 penalty. Its outputs are probabilities, and ``threshold_for`` turns "how
sure" into "how often right": the lowest probability at which held-out
predictions reach a precision, measured on samples grouped by the video they
came from so a video never vouches for itself.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

# L2 on the weights: what keeps 512 inputs and a few hundred examples from
# fitting noise. Chosen on a hand-sorted dataset with videos held out: 0.1 gave
# the best balanced accuracy (0.49; 0.01 and 0.3 both did worse).
L2 = 0.1
STEPS = 300
LEARNING_RATE = 0.05
# Held-out folds when calibrating the threshold; fewer videos, fewer folds.
FOLDS = 5


@dataclass
class LinearModel:
    classes: list
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray        # [features, classes]
    bias: np.ndarray           # [classes]

    def proba(self, x) -> np.ndarray:
        z = ((np.asarray(x, dtype=np.float64) - self.mean) / self.scale) @ self.weights + self.bias
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)


def fit(x, labels: Sequence[str], *, l2: float = L2, steps: int = STEPS,
        learning_rate: float = LEARNING_RATE) -> LinearModel:
    """Train on rows ``x`` and their labels. Deterministic."""
    x = np.asarray(x, dtype=np.float64)
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise ValueError("a linear layer needs two classes or more")
    index = {c: i for i, c in enumerate(classes)}
    y = np.array([index[l] for l in labels])
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    xs = (x - mean) / scale
    n, d = xs.shape
    k = len(classes)
    onehot = np.zeros((n, k))
    onehot[np.arange(n), y] = 1.0
    counts = onehot.sum(axis=0)
    sample_weight = (n / (k * counts))[y]
    sample_weight /= sample_weight.sum()

    w = np.zeros((d, k))
    b = np.zeros(k)
    # Adam: plain gradient descent on standardised inputs is slow to settle.
    mw, vw, mb, vb = (np.zeros_like(w), np.zeros_like(w), np.zeros_like(b), np.zeros_like(b))
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        z = xs @ w + b
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        g = (p - onehot) * sample_weight[:, None]
        gw = xs.T @ g + l2 * w
        gb = g.sum(axis=0)
        mw = beta1 * mw + (1 - beta1) * gw
        vw = beta2 * vw + (1 - beta2) * gw ** 2
        mb = beta1 * mb + (1 - beta1) * gb
        vb = beta2 * vb + (1 - beta2) * gb ** 2
        step = learning_rate * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
        w -= step * mw / (np.sqrt(vw) + eps)
        b -= step * mb / (np.sqrt(vb) + eps)
    return LinearModel(classes, mean, scale, w, b)


def _folds(groups: Sequence[str], n: int) -> list:
    """Groups dealt round-robin by size into ``n`` folds; each fold's samples."""
    sizes = {}
    for g in groups:
        sizes[g] = sizes.get(g, 0) + 1
    order = sorted(sizes, key=lambda g: (-sizes[g], g))
    fold_of = {g: i % n for i, g in enumerate(order)}
    return [np.array([i for i, g in enumerate(groups) if fold_of[g] == f]) for f in range(n)]


def held_out(x, labels: Sequence[str], groups: Sequence[str], **fit_args) -> Optional[tuple]:
    """``(top probability, right)`` for every sample, each predicted by a layer
    that never saw its group (video). ``None`` when there are too few groups to
    hold any out, or a fold leaves fewer than two classes to train on."""
    x = np.asarray(x, dtype=np.float64)
    labels = list(labels)
    n_groups = len(set(groups))
    if n_groups < 2:
        return None
    conf, right = [], []
    for test in _folds(list(groups), min(FOLDS, n_groups)):
        train = np.setdiff1d(np.arange(len(labels)), test)
        if not len(test) or len({labels[i] for i in train}) < 2:
            return None
        model = fit(x[train], [labels[i] for i in train], **fit_args)
        p = model.proba(x[test])
        top = p.argmax(axis=1)
        conf.extend(p[np.arange(len(test)), top])
        right.extend(model.classes[j] == labels[i] for j, i in zip(top, test))
    return np.array(conf), np.array(right, dtype=float)


def threshold_at(conf, right, precision: float) -> tuple:
    """``(threshold, share)``: the lowest top probability at which held-out
    predictions are right at least ``precision`` of the time, and the share of
    samples at or above it. ``(1.0, 0.0)`` when no threshold reaches it."""
    order = np.argsort(-np.asarray(conf))
    running = np.cumsum(np.asarray(right)[order]) / np.arange(1, len(order) + 1)
    reached = np.where(running >= precision)[0]
    if not len(reached):
        return 1.0, 0.0
    last = int(reached.max())
    return float(np.asarray(conf)[order][last]), (last + 1) / len(order)


def threshold_for(x, labels: Sequence[str], groups: Sequence[str], precision: float,
                  **fit_args) -> Optional[dict]:
    """``threshold_at`` for ``precision``, on held-out videos; ``None`` as
    ``held_out``."""
    measured = held_out(x, labels, groups, **fit_args)
    if measured is None:
        return None
    conf, right = measured
    threshold, share = threshold_at(conf, right, precision)
    return {"threshold": threshold, "coverage": round(share, 3),
            "heldout_accuracy": round(float(right.mean()), 3), "groups": len(set(groups))}
