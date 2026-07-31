"""Face categories taught from example crops.

A detector reports what its vocabulary contains. Faces carry far more than
identity — where someone is looking, how lit they are, what they are doing with
their features — and no label-based model has a class for most of it. Training
one needs a dataset nobody has.

This takes the mechanism already validated for regions and points it at faces:
the user picks a handful of face crops that show what they mean, those crops are
embedded, and the average becomes a matchable category. Adding one costs
milliseconds and no GPU, and re-scoring an already-scanned video is a dot
product.

**The embedding must come from a general visual model, not the face
recogniser.** ``FaceIdentityBank`` hands out SFace vectors, which are trained to
be *invariant* to everything except who the person is — that is what makes
recognition work across expressions. Averaging those would build a prototype of
a person, and it would look like it was working: confident, stable, and about
the wrong thing entirely. The crops go to CLIP instead, which is why this module
takes an embedder rather than reusing the vectors the detector already returns.

The mechanism is content-neutral. It matches whatever it is shown; the
categories, their names and their examples are the user's own data, stored
outside the repository under ``cache/``.

Nothing here imports cv2 or a model: detection and embedding arrive as callables
so the logic can be tested with arrays, and so the caller decides what runs on
the GPU.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

# Faces are detected on the frame as decoded; a crop tight to the detector's box
# loses the forehead and chin, which is where much of the signal is. Pad by a
# fraction of the box rather than a pixel count, so it scales with face size.
DEFAULT_PAD = 0.2

# Below this the detector is guessing, and a guessed crop teaches a prototype
# something that is not a face.
MIN_DETECTION_SCORE = 0.6

# A crop smaller than this carries no usable detail once resized for the
# embedder — it contributes noise to the average.
MIN_CROP_PIXELS = 24


@dataclass
class FaceCrop:
    """One face found at one moment, with wherever it ended up in vector space."""
    timestamp: float
    bbox: tuple
    det_score: float = 0.0
    embedding: Optional[np.ndarray] = None


@dataclass
class FaceCategory:
    """A matchable category: the mean of what the user pointed at."""
    name: str
    vector: np.ndarray
    examples: int = 0
    created: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "vector": [round(float(v), 6) for v in self.vector],
            "examples": int(self.examples),
            "created": float(self.created),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> "FaceCategory":
        return cls(
            name=str(data.get("name") or ""),
            vector=np.asarray(data.get("vector") or [], dtype=np.float32),
            examples=int(data.get("examples") or 0),
            created=float(data.get("created") or 0.0),
        )


def l2_normalise(vector: np.ndarray) -> np.ndarray:
    """Unit length, so a dot product is a cosine similarity."""
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else vector


def crop_face(frame_bgr: np.ndarray, bbox, pad: float = DEFAULT_PAD):
    """The padded face region, or ``None`` if it lands outside or is too small.

    Pure array slicing — no image library — so this is testable without a video
    and cheap enough to run on every detected face.
    """
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        return None
    height, width = frame_bgr.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    if x2 <= x1 or y2 <= y1:
        return None

    margin_x = (x2 - x1) * pad
    margin_y = (y2 - y1) * pad
    x1 = int(max(0, x1 - margin_x))
    y1 = int(max(0, y1 - margin_y))
    x2 = int(min(width, x2 + margin_x))
    y2 = int(min(height, y2 + margin_y))

    if x2 - x1 < MIN_CROP_PIXELS or y2 - y1 < MIN_CROP_PIXELS:
        return None
    return frame_bgr[y1:y2, x1:x2]


def scan_frames(frames: Iterable,
                *,
                detect_fn: Callable,
                embed_fn: Callable,
                pad: float = DEFAULT_PAD,
                min_det_score: float = MIN_DETECTION_SCORE,
                max_faces_per_frame: int = 4,
                batch: int = 32) -> list:
    """Find and embed every usable face in ``(timestamp, frame_bgr)`` pairs.

    ``detect_fn(frame_bgr)`` returns dicts with ``bbox`` and ``det_score`` —
    ``FaceIdentityBank.detect_faces`` satisfies this. ``embed_fn(crops)`` takes a
    list of BGR crops and returns one row per crop; ``ClipEmbedder`` satisfies
    it. Embedding is batched because that is where the model time goes.

    Faces are capped per frame: a crowd scene would otherwise contribute dozens
    of tiny background faces, none of which the user meant.
    """
    found: list = []
    pending: list = []
    pending_crops: list = []

    def flush():
        if not pending_crops:
            return
        vectors = np.asarray(embed_fn(pending_crops), dtype=np.float32)
        for crop, vector in zip(pending, vectors):
            crop.embedding = l2_normalise(vector)
            found.append(crop)
        pending.clear()
        pending_crops.clear()

    for timestamp, frame in frames:
        faces = detect_fn(frame) or []
        faces = sorted(faces, key=lambda f: -float(f.get("det_score") or 0.0))
        for face in faces[:max_faces_per_frame]:
            if float(face.get("det_score") or 0.0) < min_det_score:
                continue
            crop = crop_face(frame, face.get("bbox") or (0, 0, 0, 0), pad)
            if crop is None:
                continue
            pending.append(FaceCrop(timestamp=float(timestamp),
                                    bbox=tuple(face.get("bbox")),
                                    det_score=float(face.get("det_score") or 0.0)))
            pending_crops.append(crop)
            if len(pending_crops) >= batch:
                flush()
    flush()
    return found


def build_category(name: str, embeddings: Sequence) -> FaceCategory:
    """Average the examples into one point. That average *is* the category."""
    rows = [l2_normalise(e) for e in embeddings if e is not None and len(e)]
    if not rows:
        raise ValueError("a category needs at least one example")
    return FaceCategory(name=str(name),
                        vector=l2_normalise(np.mean(np.stack(rows), axis=0)),
                        examples=len(rows))


def score_crops(crops: Sequence[FaceCrop],
                category: FaceCategory,
                negatives: Optional[Sequence] = None) -> np.ndarray:
    """Cosine similarity of every crop to the category.

    ``negatives`` are examples of what the user does *not* mean; their mean is
    subtracted from each score. Two categories that share a background or a
    person are otherwise hard to separate, and saying "not like these" is much
    less work for the user than finding more positives.
    """
    if not crops:
        return np.zeros(0, dtype=np.float32)
    matrix = np.stack([
        c.embedding if c.embedding is not None else np.zeros_like(category.vector)
        for c in crops
    ])
    scores = matrix @ category.vector
    if negatives is not None and len(negatives):
        against = l2_normalise(np.mean(np.stack(
            [l2_normalise(n) for n in negatives]), axis=0))
        scores = scores - (matrix @ against)
    return scores.astype(np.float32)


def best_per_second(crops: Sequence[FaceCrop], scores: Sequence[float]) -> dict:
    """``{second: best score}`` — several faces can share a second."""
    out: dict = {}
    for crop, score in zip(crops, scores):
        sec = int(crop.timestamp)
        if score > out.get(sec, -np.inf):
            out[sec] = float(score)
    return out


def to_signal(best: Mapping[int, float],
              duration: float,
              *,
              threshold: float,
              points: float) -> np.ndarray:
    """A per-second points array the scoring step can add like any other signal.

    Flat points above a threshold rather than scaled similarity: the rest of the
    weight table is in points, and a signal that quietly contributed a fraction
    of its weight everywhere would be impossible to reason about next to one
    that either fires or does not.
    """
    signal = np.zeros(int(duration) + 1, dtype=float)
    for sec, score in best.items():
        if 0 <= sec < len(signal) and score >= threshold:
            signal[sec] = points
    return signal


class FaceCategoryStore:
    """The user's categories on disk. Their data, kept out of the repository."""

    def __init__(self, path: str = "./cache/face_categories.json"):
        self.path = path
        self.categories: dict = {}

    def add(self, category: FaceCategory) -> None:
        self.categories[category.name] = category

    def remove(self, name: str) -> bool:
        return self.categories.pop(name, None) is not None

    def get(self, name: str) -> Optional[FaceCategory]:
        return self.categories.get(name)

    def names(self) -> list:
        return sorted(self.categories)

    def save(self, path: Optional[str] = None) -> bool:
        target = path or self.path
        try:
            directory = os.path.dirname(target)
            if directory:
                os.makedirs(directory, exist_ok=True)
            payload = {"schema": 1,
                       "categories": [c.as_dict() for c in self.categories.values()]}
            # Write beside and replace, so an interrupted save cannot leave the
            # user with a truncated file where their examples used to be.
            temporary = target + ".tmp"
            with open(temporary, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1)
            os.replace(temporary, target)
            return True
        except OSError as exc:
            print(f"⚠️ Could not save face categories: {exc}")
            return False

    def load(self, path: Optional[str] = None) -> bool:
        target = path or self.path
        if not os.path.exists(target):
            return False
        try:
            with open(target, encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, ValueError) as exc:
            print(f"⚠️ Could not read face categories: {exc}")
            return False
        self.categories = {}
        for record in payload.get("categories") or []:
            category = FaceCategory.from_dict(record)
            if category.name and category.vector.size:
                self.categories[category.name] = category
        return True
