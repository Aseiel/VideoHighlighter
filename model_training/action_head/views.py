"""What the head sees of each frame: the whole frame, or the whole frame and tiles of it.

The head pools its inputs with learned weights (``head.ActionHead``), so it
takes any number of vectors per clip and learns which one shows the action.
Giving it parts of each frame as well as the whole lets it look closer than
the encoder's square input allows: an action that fills a small part of the
picture is a small part of a whole-frame vector, and most of a tile's.

``whole+tiles`` adds, for every frame, its four quarters and its centre at 2x
(the middle half in each direction). Measured 2026-11-19 on a 2,430-clip,
38-class dataset, scored on videos held out of training, same clips, folds and
head: whole frames 52.8 % top-1, whole frames + tiles 57.2 %, higher in each of
3 seeds; 22 classes gained, 3 small ones lost. The cost is six encoder passes
per frame instead of one, in training and wherever the head is used.

The views are the same everywhere a head is trained or used, so they are
defined once, here, and recorded in ``head.json`` (``"views"``); a head
without the field is ``whole``, which is what every head was before it.

Order of the vectors for one clip of ``k`` frames: the ``k`` whole frames, then
for each frame in turn its top-left, top-right, bottom-left and bottom-right
quarters and its centre. A tile is never resized here; the encoder's own
preprocessing fits it as it fits a whole frame.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

WHOLE = "whole"
TILES = "whole+tiles"
VIEWS = (WHOLE, TILES)

TILES_PER_FRAME = 5


def check(views: str) -> str:
    if views not in VIEWS:
        raise ValueError(f"views must be one of {VIEWS}, not {views!r}")
    return views


def per_frame(views: str) -> int:
    """Vectors each frame contributes."""
    return 1 if check(views) == WHOLE else 1 + TILES_PER_FRAME


def tokens(frames: int, views: str) -> int:
    """Vectors per clip of ``frames`` frames."""
    return int(frames) * per_frame(views)


def tiles(frame: np.ndarray) -> List[np.ndarray]:
    """The four quarters and the centre at 2x of one frame."""
    h, w = frame.shape[:2]
    parts = (frame[:h // 2, :w // 2], frame[:h // 2, w // 2:],
             frame[h // 2:, :w // 2], frame[h // 2:, w // 2:],
             frame[h // 4:3 * h // 4, w // 4:3 * w // 4])
    return [np.ascontiguousarray(p) for p in parts]


def expand(frames: Sequence[np.ndarray], views: str) -> List[np.ndarray]:
    """The images to encode for one clip, in the order the head reads them."""
    frames = list(frames)
    if check(views) == WHOLE:
        return frames
    return frames + [t for f in frames for t in tiles(f)]


def of(meta_or_scorer) -> str:
    """The views a head was trained with, from its ``head.json`` dict or a
    loaded head; ``whole`` when it does not say."""
    if isinstance(meta_or_scorer, dict):
        views = meta_or_scorer.get("views") or WHOLE
    else:
        views = getattr(meta_or_scorer, "views", None) or WHOLE
    return check(views)
