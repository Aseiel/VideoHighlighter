"""Scoring a frame against taught categories, by regions rather than whole.

This is the scorer the live overlay arrived at, lifted out of its Qt worker so
the offline passes can use it too. Nothing here imports Qt, opens a video, or
loads a model: it takes embeddings and returns numbers, which is what makes the
calibration testable without a GPU or a display.

Why regions
-----------

Teaching embeds a *crop*. Scoring a whole frame against that crop-space
prototype is a domain mismatch, and it guts the score for anything that does
not fill the picture — a small object barely moves a whole-frame vector.
Embedding an overlapping grid of sub-regions and keeping the best one puts the
query back in the same space as the taught crop, and the winning region doubles
as a coarse "where it is" box.

Why not softmax against a background embedding
----------------------------------------------

Because it was tried, measured, and abandoned. Tiles of one frame sit at ~0.9
cosine to each other (same scene, same light) while a taught crop reaches only
~0.7 to any of them, so an image-derived background vector beats the category
on every tile and CLIP's ×100 temperature collapses the softmax to ~0%.

Comparing cosines *across tiles of the same frame* cancels that shared-scene
bias: two tiles of one frame are alike for reasons that have nothing to do with
the category, and the comparison divides those reasons out. What survives is
category affinity.

The trade-off is deliberate: this rewards a *localised* subject, which is what
taught categories are for. Something spread uniformly over the whole frame does
not stand out from that frame's other regions and reads low.

The two gates
-------------

A region has to be **both** genuinely category-like in absolute terms **and**
stand out from a typical region of the same frame. The score is the lower of
the two, because each gate covers the other's blind spot:

* absolute alone cannot reject a diffuse category whose vector matches the
  whole scene — its best region still reads ~0.88 when the subject is absent;
* standout alone cannot reject the most-similar region of an empty frame, since
  there is always one, sitting ~0.13 above typical.

The conjunction is what lets this answer *"not here"* — the thing the
background-vector scoring in :mod:`llm.clip_categories` structurally cannot do,
and the reason a whole-file scan built on that one reports confident matches in
videos that do not contain the subject at all.

Constants are fitted to measured live data: an absent subject peaks around 0.76
absolute with a ~0.10 margin, a present one around 0.90 with ~0.16.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, NamedTuple, Sequence

import numpy as np

from llm.clip_categories import CustomCategory

# Grid geometry. frac=0.5 with grid=3 gives 3x3 half-size regions overlapping
# their neighbours by 50%, so a subject straddling one seam still lands whole
# inside some region.
DEFAULT_GRID = 3
DEFAULT_TILE_FRAC = 0.5


@dataclass(frozen=True)
class Gates:
    """The calibration. Frozen so a caller cannot mutate a shared instance.

    ``LIVE_GATES`` is the measured default and what the overlay uses. ``mining()``
    loosens it for label collection, where the trade-off runs the other way: a
    candidate a person rejects costs one click, while one never proposed costs
    a class the model never learns. Precision there is the review step's job,
    not the scorer's.
    """

    # best_cos at which the absolute gate reads 0.5, and how sharply it turns.
    abs_mid: float = 0.80
    abs_w: float = 0.03
    # best-minus-typical margin at which the standout gate reads 0.5.
    standout_mid: float = 0.13
    standout_w: float = 0.025
    # A calibrated category (>=2 examples) sets its own absolute midpoint at
    # self_sim minus this, rather than using abs_mid.
    calibration_margin: float = 0.10
    # Which percentile of a frame's regions counts as "typical". Below the
    # median on purpose: with 9 regions and a subject present in one or two,
    # the median is already dragged upward by them.
    baseline_pctl: float = 40.0

    def mining(self) -> "Gates":
        """The same shape, shifted to favour recall."""
        return Gates(
            abs_mid=self.abs_mid - 0.04,
            abs_w=self.abs_w,
            standout_mid=self.standout_mid - 0.03,
            standout_w=self.standout_w,
            calibration_margin=self.calibration_margin + 0.04,
            baseline_pctl=self.baseline_pctl,
        )


LIVE_GATES = Gates()


class RegionScore(NamedTuple):
    """One category's answer about one frame.

    The intermediate terms are kept rather than discarded because every
    question asked of this scorer later — why did this fire, why did it not,
    should the gate move — is answered by them and cannot be reconstructed from
    the score alone.
    """

    score: float                      # in [0,1], min of the two gates
    box: tuple                        # (x1, y1, x2, y2) of the winning region
    best_cos: float                   # cosine of the winning region
    typical_cos: float                # the frame's baseline percentile
    margin: float                     # best_cos - typical_cos
    index: int                        # which region won

    @property
    def present(self) -> bool:
        """Whether this reads as a match at the conventional halfway point.

        Callers wanting a different cut should compare ``score`` themselves —
        this is a convenience for the common case, not a policy.
        """
        return self.score >= 0.5


def tile_rects(w: int, h: int, grid: int = DEFAULT_GRID,
               frac: float = DEFAULT_TILE_FRAC) -> list:
    """An overlapping grid of candidate regions covering a w×h frame.

    Returns ``(x1, y1, x2, y2)`` in frame pixels, row-major from the top left.
    """
    tw, th = max(8, int(w * frac)), max(8, int(h * frac))
    xs = _axis_positions(w, tw, grid)
    ys = _axis_positions(h, th, grid)
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


def _axis_positions(extent: int, tile: int, grid: int) -> list:
    if tile >= extent:
        return [0]
    if grid <= 1:
        return [(extent - tile) // 2]
    step = (extent - tile) / (grid - 1)
    return [int(round(i * step)) for i in range(grid)]


def crop_tiles(frame_bgr: np.ndarray, tiles: Sequence) -> list:
    """The frame's regions as separate arrays, ready to embed."""
    return [frame_bgr[y1:y2, x1:x2] for (x1, y1, x2, y2) in tiles]


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def score_regions(tile_embeddings: np.ndarray, category: CustomCategory,
                  tiles: Sequence, gates: Gates = LIVE_GATES) -> RegionScore:
    """Score one category against one frame's already-embedded regions.

    ``tile_embeddings`` is ``[n_regions, dim]`` and unit-norm, as
    ``ClipEmbedder.embed_frames_bgr`` returns it, so the dot product against the
    category's own unit-norm vector is a cosine.

    This function is the whole calibration. It does no I/O and holds no state,
    which is deliberate: the numbers here decide whether the feature works, and
    they are only arguable if they can be measured on fixed inputs.
    """
    if tile_embeddings.ndim != 2 or len(tile_embeddings) == 0:
        raise ValueError("tile_embeddings must be [n_regions, dim] with n_regions >= 1")
    if len(tiles) != len(tile_embeddings):
        raise ValueError(
            f"got {len(tiles)} regions but {len(tile_embeddings)} embeddings")

    cos = tile_embeddings @ category.vector          # [n_regions]
    best = int(np.argmax(cos))
    best_cos = float(cos[best])
    typical = float(np.percentile(cos, gates.baseline_pctl))
    margin = best_cos - typical

    # A category that has been taught more than once knows how tightly its own
    # examples cluster, which is a better absolute expectation than any global
    # constant — a broad category should not be held to a narrow one's bar.
    if category.self_sim > 0.0:
        abs_mid = category.self_sim - gates.calibration_margin
    else:
        abs_mid = gates.abs_mid

    abs_gate = _sigmoid((best_cos - abs_mid) / gates.abs_w)
    standout_gate = _sigmoid((margin - gates.standout_mid) / gates.standout_w)

    return RegionScore(
        score=float(min(abs_gate, standout_gate)),
        box=tuple(tiles[best]),
        best_cos=best_cos,
        typical_cos=typical,
        margin=margin,
        index=best,
    )


def score_all(tile_embeddings: np.ndarray,
              categories: Iterable[CustomCategory],
              tiles: Sequence, gates: Gates = LIVE_GATES) -> dict:
    """:func:`score_regions` for several categories over one frame's regions.

    One embedding pass serves every category, which is why this takes them
    together: the regions are the expensive part and they do not depend on what
    is being looked for.
    """
    return {cat.name: score_regions(tile_embeddings, cat, tiles, gates)
            for cat in categories}


def score_frame(frame_bgr: np.ndarray, categories: Iterable[CustomCategory],
                embedder, grid: int = DEFAULT_GRID,
                frac: float = DEFAULT_TILE_FRAC,
                gates: Gates = LIVE_GATES) -> dict:
    """Tile a frame, embed the regions, and score every category against them.

    ``embedder`` is anything with ``embed_frames_bgr(list_of_bgr) -> [n, dim]``
    — :class:`llm.clip_index.ClipEmbedder` in the app, a stand-in in tests. It
    is a parameter rather than an import so this module never loads a model and
    stays importable where torch is not.
    """
    categories = list(categories)
    if not categories:
        return {}
    h, w = frame_bgr.shape[:2]
    tiles = tile_rects(w, h, grid, frac)
    tile_embeddings = embedder.embed_frames_bgr(crop_tiles(frame_bgr, tiles))
    return score_all(tile_embeddings, categories, tiles, gates)


def explain(name: str, result: RegionScore) -> str:
    """One line saying why a score came out as it did.

    For the debug log and for the report. The score alone is not reviewable —
    a category that never fires and one that fires everywhere look the same
    from the outside, and these terms are what tell them apart.
    """
    return (f"{name}: best={result.best_cos:.3f} typical={result.typical_cos:.3f} "
            f"margin={result.margin:.3f} -> {result.score:.2f}")
