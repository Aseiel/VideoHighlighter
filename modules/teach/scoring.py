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

``background`` is a low percentile (``BACKGROUND_PERCENTILE``) of all
samples' cosines to the prototype: what "not this" looks like in this footage.
Not the median: footage chosen *because* it shows the thing can show it in
more than half its samples, and a median then sits among the real examples and
calibrates them to zero, so nothing is ever proposed. The 20th percentile holds
until the thing fills four fifths of the footage. ``anchor`` is what "this" looks like: the examples' mean cosine to their
own prototype when there are examples, else the 97th percentile of the
samples. So 0 is ordinary footage and 1 is "looks like the examples", on the
same scale for every class.

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
# Where "not this" is read from, as a percentile of the footage (see above).
BACKGROUND_PERCENTILE = 20.0
# Keeps a class whose samples all score alike from dividing by nothing.
MIN_SPREAD = 0.02
# Cap on how many accepted samples feed a prototype: past this, more of the
# same moves the mean by nothing and costs a stack of vectors.
MAX_EXAMPLES = 200


@dataclass
class Prototype:
    name: str
    vector: np.ndarray
    kind: str                 # "examples" or "text"
    n_examples: int = 0
    anchor: Optional[float] = None


def _unit(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)


def build_prototype(name: str, example_vectors: Sequence = (),
                    text_vectors: Sequence = ()) -> Optional[Prototype]:
    examples = [v for v in example_vectors if v is not None][:MAX_EXAMPLES]
    if examples:
        stack = _unit(np.stack(examples))
        vector = _unit(stack.mean(axis=0))
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
        background = float(np.percentile(column, BACKGROUND_PERCENTILE))
        anchor = proto.anchor
        if anchor is None:
            anchor = float(np.percentile(column, TEXT_ANCHOR_PERCENTILE))
        out[:, c] = (column - background) / max(anchor - background, MIN_SPREAD)
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
    raw = _unit(sample_matrix) @ np.stack([p.vector for p in prototypes]).T
    calibrated = calibrate(raw, prototypes)
    out = {}
    for i, sid in enumerate(sample_ids):
        row = calibrated[i]
        proposal, lead = propose(row, names, gate, margin, floor, none_index)
        scores = {names[c]: round(float(row[c]), 4) for c in range(len(names))
                  if c != none_index}
        out[sid] = (scores, proposal, round(lead, 4))
    return out
