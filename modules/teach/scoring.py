"""Which class, if any, does each sample show? Pure numpy, no CLIP here.

Each class gets a **prototype**: one unit vector in CLIP space.

* From **examples** when it has any: the mean of the example samples' vectors,
  plus every sample already accepted as that class. Each review batch
  therefore sharpens the next sort.
* From **words** otherwise: the class name (and its description) as CLIP text.
  Weaker, because CLIP's image-to-text cosines are compressed into a narrow
  band, but enough to rank a video's samples for the first review.

Raw cosines are not comparable between classes (text prototypes sit around
0.2-0.3, example prototypes around 0.6-0.9), so each class's scores are
**calibrated** against the project's own samples::

    calibrated = (cosine - background) / (anchor - background)

``background`` is what "not this" looks like in this footage (``background``
below): the median of the samples that are *not* the thing. When the scores
split cleanly into a low and a high group, that is the low group's median,
whichever group is larger. Footage chosen *because* it shows the thing can show
it in most samples, and a plain median then sits among the real examples and
calibrates them to zero. When they do not split, the thing is rare or absent,
and the plain median is ordinary footage. ``anchor`` is what "this" looks like:
the examples' mean cosine to their own prototype when there are examples, else
the 97th percentile of the samples. So 0 is ordinary footage and 1 is "looks
like the examples", on the same scale for every class.

A sample is **proposed** as the best class when its calibrated score clears
``gate`` *and* leads the runner-up by ``margin`` (two classes that both fit is
a question for a person, not a guess). When every class stays under ``floor``
it is proposed as ``NONE`` — a candidate negative. Everything in between is
``UNSURE``.

Samples marked negative form one more prototype, ``NONE``, which competes with
the classes: once a person has said what "none of these" looks like here, a
sample closer to that than to anything else is not proposed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from modules.teach.project import NONE, UNSURE

# The upper end of "ordinary footage" for a text-only class, as a percentile.
TEXT_ANCHOR_PERCENTILE = 97.0
# Keeps a class whose samples all score alike from dividing by nothing.
MIN_SPREAD = 0.02
# Cap on how many accepted samples feed a prototype: past this, more of the
# same moves the mean by nothing and costs a stack of vectors.
MAX_EXAMPLES = 200
# How cleanly the scores must split in two (Otsu's between-group share of the
# variance) before the low group alone is taken as ordinary footage. One bell
# curve splits at 2/pi (0.64) and a flat spread at 0.75, so above both.
BIMODAL_SPLIT = 0.8


def background(column) -> float:
    """What ordinary footage scores against one prototype (see above)."""
    values = np.sort(np.asarray(column, dtype=np.float64))
    n = len(values)
    if n < 4:
        return float(np.median(values))
    total_var = float(values.var())
    if total_var <= 1e-12:
        return float(values[0])
    # Otsu on one dimension: the split that maximises between-group variance.
    prefix = np.cumsum(values)
    mean = prefix[-1] / n
    best_share, best_k = 0.0, 0
    for k in range(1, n):
        w0, w1 = k / n, (n - k) / n
        m0, m1 = prefix[k - 1] / k, (prefix[-1] - prefix[k - 1]) / (n - k)
        share = (w0 * (m0 - mean) ** 2 + w1 * (m1 - mean) ** 2) / total_var
        if share > best_share:
            best_share, best_k = share, k
    if best_share >= BIMODAL_SPLIT:
        return float(np.median(values[:best_k]))
    return float(np.median(values))


@dataclass
class Prototype:
    name: str
    vector: np.ndarray
    kind: str                 # "examples" or "text"
    n_examples: int = 0
    anchor: Optional[float] = None
    # Several centres instead of one mean (``centers`` > 1): a sample scores
    # its likeness to the nearest one. A class shown in a few different ways
    # has a mean that sits between them and looks like none of them.
    centers: Optional[np.ndarray] = None

    def cosines(self, unit_rows: np.ndarray) -> np.ndarray:
        if self.centers is not None:
            return (unit_rows @ self.centers.T).max(axis=1)
        return unit_rows @ self.vector


def _unit(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)


# A centre needs this many examples of its own to be more than one clip's
# quirks; with fewer, a class keeps fewer centres.
MIN_PER_CENTER = 3


def spherical_kmeans(stack: np.ndarray, k: int, iterations: int = 25,
                     seed: int = 0) -> np.ndarray:
    """``k`` unit centres for unit rows, by cosine. Deterministic for a seed."""
    rng = np.random.default_rng(seed)
    centers = [stack[int(rng.integers(len(stack)))]]
    for _ in range(1, k):                       # k-means++: far from those chosen
        nearest = (stack @ np.stack(centers).T).max(axis=1)
        weights = np.clip(1.0 - nearest, 0.0, None) ** 2
        if weights.sum() <= 1e-12:
            break
        centers.append(stack[int(rng.choice(len(stack), p=weights / weights.sum()))])
    centers = np.stack(centers)
    for _ in range(iterations):
        assign = (stack @ centers.T).argmax(axis=1)
        moved = np.stack([_unit(stack[assign == c].mean(axis=0)) if np.any(assign == c)
                          else centers[c] for c in range(len(centers))])
        if np.allclose(moved, centers, atol=1e-6):
            break
        centers = moved
    return centers


def build_prototype(name: str, example_vectors: Sequence = (),
                    text_vectors: Sequence = (), centers: int = 1) -> Optional[Prototype]:
    examples = [v for v in example_vectors if v is not None][:MAX_EXAMPLES]
    if examples:
        stack = _unit(np.stack(examples))
        vector = _unit(stack.mean(axis=0))
        k = min(int(centers), len(stack) // MIN_PER_CENTER)
        if k > 1:
            found = spherical_kmeans(stack, k)
            proto = Prototype(name, vector, "examples", len(stack), None, found)
            proto.anchor = float(proto.cosines(stack).mean())
            return proto
        anchor = float((stack @ vector).mean()) if len(stack) >= 2 else None
        return Prototype(name, vector, "examples", len(stack), anchor)
    if len(text_vectors):
        vector = _unit(_unit(np.stack(list(text_vectors))).mean(axis=0))
        return Prototype(name, vector, "text", 0, None)
    return None


def calibrate(raw: np.ndarray, prototypes: Sequence[Prototype]) -> np.ndarray:
    """Raw cosines ``[n_samples, n_classes]`` -> calibrated scores."""
    out = np.zeros_like(raw, dtype=np.float32)
    for c, proto in enumerate(prototypes):
        column = raw[:, c]
        floor = background(column)
        anchor = proto.anchor
        if anchor is None:
            anchor = float(np.percentile(column, TEXT_ANCHOR_PERCENTILE))
        out[:, c] = (column - floor) / max(anchor - floor, MIN_SPREAD)
    return out


def propose(row: np.ndarray, names: Sequence[str], gate: float, margin: float,
            floor: float, none_index: Optional[int] = None) -> tuple:
    """``(proposal, lead)`` for one sample's calibrated scores."""
    order = np.argsort(-row)
    best = int(order[0])
    runner = float(row[order[1]]) if len(order) > 1 else float("-inf")
    lead = float(row[best] - runner) if np.isfinite(runner) else float(row[best])

    if none_index is not None and best == none_index:
        return NONE, lead
    class_scores = [float(row[i]) for i in range(len(names)) if i != none_index]
    if not class_scores or max(class_scores) < floor:
        return NONE, lead
    if row[best] >= gate and lead >= margin:
        return names[best], lead
    return UNSURE, lead


def score_samples(sample_ids: Sequence[str], sample_matrix: np.ndarray,
                  prototypes: Sequence[Prototype], *, gate: float, margin: float,
                  floor: float) -> dict:
    """``{sample id: (scores dict, proposal, lead)}`` for every sample.

    ``prototypes`` may end with one named ``NONE`` (from negatives); it takes
    part in the ranking but is not reported as a class score.
    """
    if not len(sample_ids) or not prototypes:
        return {}
    names = [p.name for p in prototypes]
    none_index = names.index(NONE) if NONE in names else None
    rows = _unit(sample_matrix)
    raw = np.stack([p.cosines(rows) for p in prototypes], axis=1)
    calibrated = calibrate(raw, prototypes)
    out = {}
    for i, sid in enumerate(sample_ids):
        row = calibrated[i]
        proposal, lead = propose(row, names, gate, margin, floor, none_index)
        scores = {names[c]: round(float(row[c]), 4) for c in range(len(names))
                  if c != none_index}
        out[sid] = (scores, proposal, round(lead, 4))
    return out
