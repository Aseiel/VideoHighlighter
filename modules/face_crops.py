"""Find faces in frames and cut them out, once, for whoever wants them.

The expensive part of looking at faces is decoding and detection; what happens
to a crop afterwards is cheap and varies. So the sweep lives here on its own,
and the consumers stay separate: the built-in expression classes read these
crops, and so does the taught-category layer, without either needing the other.

Detection and embedding arrive as callables. That keeps the module testable
with plain arrays — no cv2, no model, no video file — and leaves the caller
deciding what runs on the GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import numpy as np

# Faces are detected on the frame as decoded; a crop tight to the detector's box
# loses the forehead and chin, which is where much of the signal is. Pad by a
# fraction of the box rather than a pixel count, so it scales with face size.
DEFAULT_PAD = 0.2

# Below this the detector is guessing, and a guessed crop is not a face.
MIN_DETECTION_SCORE = 0.6

# A crop smaller than this carries no usable detail once resized for a model —
# it contributes noise to whatever consumes it.
MIN_CROP_PIXELS = 24


@dataclass
class FaceCrop:
    """One face found at one moment, with wherever it ended up in vector space."""
    timestamp: float
    bbox: tuple
    det_score: float = 0.0
    embedding: Optional[np.ndarray] = None


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
    list of BGR crops and returns one row per crop. Embedding is batched because
    that is where the model time goes.

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
