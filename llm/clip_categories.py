"""Custom categories taught from example frames.

The label-based backends can only report what their vocabulary already contains
— Kinetics-400's activity list, COCO's 80 objects. Anything outside it has no
output, at any threshold. This module removes the vocabulary: the user points at
a few frames that show what they mean, and that becomes a matchable category. No
dataset, no training run, no GPU.

It works because `llm.clip_index` has already reduced each frame to a point in
CLIP space. A category is then just another point — the average of its examples
— and scoring is a dot product against frames that are already embedded. Adding
a category costs milliseconds, and re-scoring an indexed video is instant.

The mechanism is content-neutral: it matches whatever it is shown, and the
categories are the user's own data, defined at runtime and stored outside the
repo (see CLAUDE.md).

Shared by both editions and kept byte-identical: a category is what a
community model starts as, and anyone should be able to teach one.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from llm.clip_index import ClipFrameIndex, l2_normalize

# How many frames to draw as the "everything else" contrast set when the caller
# doesn't supply one. Enough to characterise the video's look, small enough to
# stay instant.
DEFAULT_BACKGROUND_SAMPLES = 64


@dataclass
class CustomCategory:
    """A user-taught category: one unit-norm vector plus its provenance."""

    name: str
    vector: np.ndarray
    n_examples: int = 0
    model_id: str = ""
    created: float = field(default_factory=time.time)
    # Mean cosine of the taught examples to their own averaged vector: how
    # tightly they cluster, and thus roughly the cosine a genuine live match
    # should reach. 0.0 means "uncalibrated" (only one example, or a category
    # from before this field existed). Used live as a per-category absolute
    # gate so a merely-most-similar region in an empty scene isn't reported as
    # a match. Set by the live incremental teacher (video_ai_editor.live_category).
    self_sim: float = 0.0

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "vector": [float(x) for x in self.vector],
            "n_examples": self.n_examples,
            "model_id": self.model_id,
            "created": self.created,
            "self_sim": self.self_sim,
        }

    @classmethod
    def from_json(cls, d: dict) -> "CustomCategory":
        return cls(
            name=d["name"],
            vector=np.asarray(d["vector"], dtype=np.float32),
            n_examples=int(d.get("n_examples", 0)),
            model_id=d.get("model_id", ""),
            created=float(d.get("created", 0.0)),
            self_sim=float(d.get("self_sim", 0.0)),
        )


def learn_category(index: ClipFrameIndex, name: str,
                   example_indices: Sequence[int]) -> CustomCategory:
    """Turn the user's example frames into a category.

    `example_indices` index into `index.embeddings` (i.e. sampled frames, not
    raw video frames). Averaging is deliberate: with a handful of examples it
    beats anything fancier, and the result is one vector to store and score.
    """
    idx = list(example_indices)
    if not idx:
        raise ValueError("Need at least one example frame.")
    vector = l2_normalize(index.embeddings[idx].mean(axis=0))
    return CustomCategory(
        name=name,
        vector=vector,
        n_examples=len(idx),
        model_id=str(index.meta.get("model", "")),
    )


def background_vector(index: ClipFrameIndex,
                      exclude: Optional[Sequence[int]] = None,
                      samples: int = DEFAULT_BACKGROUND_SAMPLES,
                      seed: int = 0) -> np.ndarray:
    """One unit-norm vector standing for "what this footage normally looks like".

    Two constraints force this shape:

    Modality — it must be built from *image* embeddings, not text prompts. A
    category vector is image-derived, and CLIP's modality gap means it beats any
    text negative automatically, scoring every frame ~1.0 (see
    ClipFrameIndex.query_vector).

    Count — it must be *one* vector, not a bag of sampled frames. Softmax over
    [category, *n_negatives] makes the score depend on how many negatives were
    passed, and image<->image cosines are all high and tightly packed, so a few
    near-duplicate frames in the bag drag a perfect match below 0.5. Measured on
    a 40-frame index with 38 negatives, frames that visibly *were* the target
    scored under 0.5. Averaging first collapses that to a stable two-way
    comparison: category vs typical frame.

    Caveat: if the target fills most of the video, the average partly *is* the
    target and scores compress. Pass `exclude`, or supply your own vector.
    """
    n = len(index)
    dim = index.embeddings.shape[1] if n else 512
    if n == 0:
        return np.zeros(dim, dtype=np.float32)
    pool = np.setdiff1d(np.arange(n), np.asarray(list(exclude or []), dtype=int))
    if len(pool) == 0:
        return np.zeros(dim, dtype=np.float32)
    take = min(samples, len(pool))
    rng = np.random.default_rng(seed)   # seeded: same footage -> same scores
    picked = index.embeddings[rng.choice(pool, size=take, replace=False)]
    return l2_normalize(picked.mean(axis=0))


def score_category(index: ClipFrameIndex, category: CustomCategory,
                   background: Optional[np.ndarray] = None,
                   exclude_from_background: Optional[Sequence[int]] = None) -> np.ndarray:
    """Score every indexed frame against a category -> [n] in [0,1].

    Reads as "more like the user's examples than like a typical frame of this
    video", which is both calibratable and what the user actually means. The
    contrast frame is derived from the video itself unless `background` is given.
    """
    if background is None:
        background = background_vector(index, exclude=exclude_from_background)
    scale = float(index.meta.get("logit_scale", 100.0))
    return index.query_vector(
        category.vector, negatives=background.reshape(1, -1), logit_scale=scale,
    )


class CategoryStore:
    """Categories on disk, as user data (never in the repo).

    JSON: a category is ~512 floats, and keeping it readable means a user can
    delete or hand-edit their own data without the app.
    """

    def __init__(self, path: str):
        self.path = path
        self.categories: dict[str, CustomCategory] = {}

    def load(self) -> "CategoryStore":
        if not os.path.exists(self.path):
            return self
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for d in data.get("categories", []):
                cat = CustomCategory.from_json(d)
                self.categories[cat.name] = cat
        except Exception as e:
            print(f"⚠️  Custom categories: unreadable store ({e}); starting empty")
        return self

    def save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        payload = {"categories": [c.to_json() for c in self.categories.values()]}
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, self.path)   # atomic: a crash can't truncate the store

    def add(self, category: CustomCategory) -> None:
        self.categories[category.name] = category

    def remove(self, name: str) -> bool:
        return self.categories.pop(name, None) is not None

    def names(self) -> list[str]:
        return sorted(self.categories)

    def stale(self, model_id: str) -> list[str]:
        """Categories learned under a different CLIP than the one now loaded.

        Their vectors live in that model's space and are meaningless in another,
        so the GUI should offer to relearn rather than score them.
        """
        return sorted(n for n, c in self.categories.items()
                      if c.model_id and c.model_id != model_id)
