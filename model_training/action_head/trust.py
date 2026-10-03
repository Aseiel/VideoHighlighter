"""How far a taught head can be trusted, from its held-out scores.

Each action has its own score, so each action gets its own threshold: the
lowest score above which "this action is present" was right often enough on
clips from videos the head never saw. The test is strict on purpose: the
lower end of the 80 % Wilson interval of that precision must reach the
target, so a handful of lucky hits cannot vouch for an action, and dozens can.
And the hits must come from at least ``min_videos`` source videos: clips of
one video are near-copies of each other, so eight right answers from two
videos are two pieces of evidence, not eight. (Measured: a rare action
vouched for by a few videos sorted another video's clips into itself.) An
action that never gets there has no threshold: it is only ever a suggestion.

A taught pair of actions gets a threshold of its own, on the lower of its two
scores: a clip showing two actions rarely scores the second as high as a
clip showing it alone, so the single-action thresholds would almost never
find both.

numpy only: the trainer computes this from out-of-fold scores, and the app
reads the thresholds back without needing torch.
"""
from __future__ import annotations

import numpy as np


def wilson_low(k, n, z: float = 1.28):
    """Lower end of the 80 % interval for ``k`` right out of ``n``."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    d = 1 + z * z / n
    return (p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d


def lowest_trusted(score: np.ndarray, right: np.ndarray, target: float, min_n: int = 3,
                   groups=None, min_videos: int = 3):
    """The lowest ``score`` above which ``right`` holds often enough
    (Wilson lower bound >= ``target``) with hits from ``min_videos`` or more
    groups, or None."""
    order = np.argsort(-np.asarray(score), kind="stable")
    hit = np.asarray(right, bool)[order]
    n = np.arange(1, len(hit) + 1)
    ok = (wilson_low(np.cumsum(hit), n) >= target) & (n >= min_n)
    if groups is not None and min_videos > 1:
        seen, videos = set(), np.zeros(len(hit), int)
        for i, (h, g) in enumerate(zip(hit, np.asarray(groups)[order])):
            if h:
                seen.add(g)
            videos[i] = len(seen)
        ok &= videos >= min_videos
    good = np.where(ok)[0]
    return float(np.asarray(score)[order[good[-1]]]) if len(good) else None


def trust_thresholds(scores: np.ndarray, targets: np.ndarray, target: float = 0.7,
                     min_n: int = 3, groups=None, min_videos: int = 3) -> list:
    """Per action, the lowest score at which clips scoring that high or more
    really show it often enough, or None. ``scores`` and ``targets`` are
    [N, classes], targets multi-hot; ``groups`` the clips' source videos."""
    targets = np.asarray(targets) > 0
    return [lowest_trusted(scores[:, c], targets[:, c], target, min_n, groups, min_videos)
            for c in range(scores.shape[1])]


def taught_pairs(targets: np.ndarray, min_clips: int = 5) -> list:
    """The pairs of actions shown together in ``min_clips`` or more clips."""
    from collections import Counter
    counts = Counter(tuple(np.flatnonzero(row)) for row in np.asarray(targets) > 0
                     if row.sum() == 2)
    return sorted(p for p, n in counts.items() if n >= min_clips)


def pair_thresholds(scores: np.ndarray, targets: np.ndarray, pairs: list,
                    target: float = 0.7, min_n: int = 3, groups=None,
                    min_videos: int = 3) -> list:
    """Per taught pair ``(a, b)``, the lowest value of ``min(score a, score b)``
    above which clips really show both often enough, or None."""
    targets = np.asarray(targets) > 0
    return [lowest_trusted(np.minimum(scores[:, a], scores[:, b]), targets[:, a] & targets[:, b],
                           target, min_n, groups, min_videos) for a, b in pairs]


def detected(scores: np.ndarray, thresholds: list, pairs: list = (),
             pair_th: list = ()) -> np.ndarray:
    """[N, classes] bool: the actions each clip is trusted to show. A trusted
    pair whose lower score reaches its threshold adds both of its actions."""
    scores = np.asarray(scores)
    th = np.array([np.inf if t is None else t for t in thresholds], float)
    out = scores >= th
    for (a, b), t in zip(pairs, pair_th):
        if t is not None:
            both = np.minimum(scores[:, a], scores[:, b]) >= t
            out[both, a] = True
            out[both, b] = True
    return out


def top_k_hit(scores: np.ndarray, y: np.ndarray, k: int) -> float:
    if len(y) == 0:
        return 0.0
    top = np.argsort(-scores, 1)[:, :k]
    return float(np.mean([t in row for t, row in zip(y, top)]))


def balanced_accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    per = [np.mean(pred[y == c] == c) for c in np.unique(y)]
    return float(np.mean(per)) if per else 0.0


def score_clips(scores: np.ndarray, label_sets: list, thresholds: list,
                pairs: list = (), pair_th: list = ()) -> dict:
    """Held-out numbers for a set of clips, split by how many actions they show.

    ``label_sets``: per clip, the set of class indices it shows. Single-action
    clips: accuracy of the top score, top-3, balanced accuracy, and how many
    have their top action trusted (and how often that is right). Two-action
    clips: top score one of the two, both in the top two, both in the top
    five, and both detected (both over their thresholds).
    """
    out: dict = {}
    scores = np.asarray(scores)
    found = (detected(scores, thresholds, pairs, pair_th) if len(scores)
             else np.zeros((0, 0), bool))
    single = np.array([len(s) == 1 for s in label_sets], bool)
    if single.any():
        p = scores[single]
        y = np.array([next(iter(s)) for s, one in zip(label_sets, single) if one])
        pred = p.argmax(1)
        trusted = found[single][np.arange(len(pred)), pred]
        out["single"] = {
            "clips": int(single.sum()),
            "accuracy": round(float(np.mean(pred == y)), 4),
            "balanced_accuracy": round(balanced_accuracy(pred, y), 4),
            "top3": round(top_k_hit(p, y, 3), 4),
            "trusted_share": round(float(trusted.mean()), 4),
            "trusted_precision": (round(float(np.mean(pred[trusted] == y[trusted])), 4)
                                  if trusted.any() else None),
        }
    two = np.array([len(s) == 2 for s in label_sets], bool)
    if two.any():
        p = scores[two]
        pairs = [s for s, t in zip(label_sets, two) if t]
        order = np.argsort(-p, 1)
        f = found[two]
        out["two_actions"] = {
            "clips": int(two.sum()),
            "top1_is_one_of_them": round(float(np.mean([o[0] in s for o, s in zip(order, pairs)])), 4),
            "both_in_top2": round(float(np.mean([set(o[:2]) == s for o, s in zip(order, pairs)])), 4),
            "both_in_top5": round(float(np.mean([s <= set(o[:5]) for o, s in zip(order, pairs)])), 4),
            "both_detected": round(float(np.mean([all(r[list(s)]) for r, s in zip(f, pairs)])), 4),
            "one_detected": round(float(np.mean([r[list(s)].sum() == 1 for r, s in zip(f, pairs)])), 4),
            "wrong_detected": round(float(np.mean([bool((r & ~np.isin(np.arange(len(r)), list(s))).any())
                                                    for r, s in zip(f, pairs)])), 4),
        }
    return out
