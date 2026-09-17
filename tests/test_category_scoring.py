"""Tests for region scoring against taught categories.

Synthetic embeddings with known geometry, no CLIP and no Qt — the point of
lifting this scorer out of the live worker was that its calibration could be
measured on fixed inputs, so these tests are the reason the module exists.

Every embedding here is built by :func:`_rows`, which produces unit-norm
vectors with an exact cosine to the category vector. That makes each case a
statement about the gates rather than about CLIP.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm.clip_categories import CustomCategory
from llm.category_scoring import (
    Gates,
    LIVE_GATES,
    crop_tiles,
    explain,
    score_all,
    score_frame,
    score_regions,
    tile_rects,
)

DIM = 4


def _category(name: str = "taught", self_sim: float = 0.0) -> CustomCategory:
    """A category whose vector is the first basis direction."""
    vector = np.zeros(DIM, dtype=np.float32)
    vector[0] = 1.0
    return CustomCategory(name=name, vector=vector, n_examples=3,
                          model_id="test", self_sim=self_sim)


def _rows(*cosines: float) -> np.ndarray:
    """Unit-norm rows whose cosine to the category vector is exactly given.

    The orthogonal remainder is spread over a second axis, so the rows differ
    from each other in a way the scorer must ignore.
    """
    out = np.zeros((len(cosines), DIM), dtype=np.float32)
    for i, c in enumerate(cosines):
        out[i, 0] = c
        out[i, 1] = float(np.sqrt(max(0.0, 1.0 - c * c)))
    return out


def _tiles(n: int) -> list:
    """Placeholder geometry — score_regions only indexes into this."""
    return [(i, 0, i + 10, 10) for i in range(n)]


# ── geometry ─────────────────────────────────────────────────────────────

def test_tile_rects_is_an_overlapping_grid_covering_the_frame():
    tiles = tile_rects(100, 100, grid=3, frac=0.5)
    assert len(tiles) == 9
    # Half-size regions...
    assert all((x2 - x1, y2 - y1) == (50, 50) for x1, y1, x2, y2 in tiles)
    # ...anchored at the corners, so the frame is covered edge to edge.
    assert tiles[0] == (0, 0, 50, 50)
    assert tiles[-1] == (50, 50, 100, 100)


def test_tile_rects_overlap_so_a_subject_on_a_seam_lands_whole_somewhere():
    tiles = tile_rects(100, 100, grid=3, frac=0.5)
    xs = sorted({x1 for x1, _, _, _ in tiles})
    # Neighbouring positions are closer together than a region is wide.
    assert all(b - a < 50 for a, b in zip(xs, xs[1:]))


def test_tile_rects_degenerates_safely_on_tiny_frames():
    # A region as large as the frame cannot be tiled; one region is the answer,
    # not a crash or an empty list.
    assert tile_rects(10, 10, grid=3, frac=2.0) == [(0, 0, 20, 20)]
    assert len(tile_rects(100, 100, grid=1, frac=0.5)) == 1


def test_crop_tiles_returns_the_regions_it_was_asked_for():
    frame = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    crops = crop_tiles(frame, [(0, 0, 50, 50), (50, 50, 100, 100)])
    assert [c.shape for c in crops] == [(50, 50, 3), (50, 50, 3)]
    assert np.array_equal(crops[0], frame[0:50, 0:50])


# ── the two gates ────────────────────────────────────────────────────────

def test_a_present_subject_scores_high():
    # One region clearly the category, the rest ordinary scene: the measured
    # signature of a present subject (~0.90 absolute, ~0.16 margin).
    embeddings = _rows(0.74, 0.74, 0.90, 0.74, 0.74, 0.73, 0.74, 0.75, 0.74)
    result = score_regions(embeddings, _category(), _tiles(9))
    # ~0.77 at this operating point. Worth knowing which gate decides: the
    # absolute one is saturated here (0.97) and the standout one is not, so a
    # present subject's score is set by how much it stands out, not by how
    # category-like it is. Moving standout_mid moves this number; moving
    # abs_mid does not.
    assert result.score > 0.7
    assert result.present
    assert result.index == 2
    assert result.box == (2, 0, 12, 10)


def test_an_absent_subject_scores_low_even_though_some_region_wins():
    # There is always a most-similar region. Without the standout gate this
    # would be reported as a match, which is exactly the failure the
    # background-vector scorer cannot avoid.
    embeddings = _rows(0.74, 0.76, 0.75, 0.74, 0.73, 0.75, 0.74, 0.74, 0.75)
    result = score_regions(embeddings, _category(), _tiles(9))
    assert result.score < 0.1
    assert not result.present


def test_a_diffuse_match_across_the_whole_frame_is_rejected():
    # High absolute cosine everywhere: the absolute gate alone would pass this.
    # Nothing stands out, so nothing is localised, so it is not a detection.
    embeddings = _rows(*([0.88] * 9))
    result = score_regions(embeddings, _category(), _tiles(9))
    assert result.best_cos == pytest.approx(0.88, abs=1e-3)
    assert result.score < 0.1


def test_a_standing_out_but_weak_region_is_rejected():
    # Big margin, low absolute: the standout gate alone would pass this.
    embeddings = _rows(0.30, 0.30, 0.55, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30)
    result = score_regions(embeddings, _category(), _tiles(9))
    assert result.margin > LIVE_GATES.standout_mid
    assert result.score < 0.1


def test_score_is_the_lower_of_the_two_gates():
    # Neither gate alone decides: whichever is less confident sets the score.
    embeddings = _rows(0.74, 0.74, 0.90, 0.74, 0.74, 0.73, 0.74, 0.75, 0.74)
    result = score_regions(embeddings, _category(), _tiles(9))
    generous = Gates(abs_mid=0.0, standout_mid=10.0)   # abs passes, standout fails
    stifled = score_regions(embeddings, _category(), _tiles(9), generous)
    assert stifled.score < 0.01


# ── per-category calibration ─────────────────────────────────────────────

def test_a_calibrated_category_sets_its_own_absolute_bar():
    # A broad category (examples agreeing only at 0.80) should not be held to
    # the global bar meant for a tight one.
    embeddings = _rows(0.55, 0.55, 0.72, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55)
    uncalibrated = score_regions(embeddings, _category(self_sim=0.0), _tiles(9))
    calibrated = score_regions(embeddings, _category(self_sim=0.80), _tiles(9))
    assert calibrated.score > uncalibrated.score
    # The geometry is untouched by calibration — only the verdict moves.
    assert calibrated.best_cos == uncalibrated.best_cos
    assert calibrated.index == uncalibrated.index


# ── mining gates ─────────────────────────────────────────────────────────

def test_mining_gates_never_score_below_the_live_ones():
    mining = LIVE_GATES.mining()
    for best in (0.60, 0.74, 0.80, 0.90):
        embeddings = _rows(*([0.70] * 8 + [best]))
        live = score_regions(embeddings, _category(), _tiles(9), LIVE_GATES)
        mined = score_regions(embeddings, _category(), _tiles(9), mining)
        assert mined.score >= live.score - 1e-9


def test_mining_gates_admit_a_borderline_candidate_the_overlay_would_drop():
    # The whole point of a separate mining calibration: a candidate the live
    # overlay declines to draw is still worth putting in front of a reviewer.
    embeddings = _rows(0.70, 0.70, 0.82, 0.70, 0.70, 0.70, 0.70, 0.70, 0.71)
    live = score_regions(embeddings, _category(), _tiles(9), LIVE_GATES)
    mined = score_regions(embeddings, _category(), _tiles(9), LIVE_GATES.mining())
    assert not live.present
    assert mined.present


def test_gates_are_frozen_so_a_shared_instance_cannot_be_edited():
    with pytest.raises(Exception):
        LIVE_GATES.abs_mid = 0.1


# ── several categories at once ───────────────────────────────────────────

def test_score_all_answers_for_every_category_from_one_embedding_pass():
    embeddings = _rows(0.74, 0.74, 0.90, 0.74, 0.74, 0.73, 0.74, 0.75, 0.74)
    results = score_all(embeddings, [_category("a"), _category("b")], _tiles(9))
    assert set(results) == {"a", "b"}
    assert results["a"].score == results["b"].score   # same vector, same answer


def test_score_frame_tiles_embeds_and_scores():
    """The plumbing, with a stand-in for CLIP.

    Verifies the one thing the pure functions cannot: that a frame is cut into
    the regions the scorer is then told about, in the same order.
    """
    captured = {}

    class StubEmbedder:
        def embed_frames_bgr(self, crops):
            captured["n"] = len(crops)
            captured["shapes"] = [c.shape for c in crops]
            return _rows(*([0.74] * 8 + [0.90]))

    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    results = score_frame(frame, [_category()], StubEmbedder())

    assert captured["n"] == 9
    assert captured["shapes"] == [(60, 60, 3)] * 9
    assert results["taught"].present
    # The winning region is the last one, so the box must be the frame's corner.
    assert results["taught"].box == (60, 60, 120, 120)


def test_score_frame_with_no_categories_does_not_load_or_embed_anything():
    class ExplodingEmbedder:
        def embed_frames_bgr(self, crops):
            raise AssertionError("should not embed when nothing is taught")

    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    assert score_frame(frame, [], ExplodingEmbedder()) == {}


# ── contract ─────────────────────────────────────────────────────────────

def test_mismatched_regions_and_embeddings_are_rejected_loudly():
    embeddings = _rows(0.8, 0.7, 0.7)
    with pytest.raises(ValueError, match="regions"):
        score_regions(embeddings, _category(), _tiles(9))


def test_empty_embeddings_are_rejected():
    with pytest.raises(ValueError):
        score_regions(np.zeros((0, DIM), dtype=np.float32), _category(), [])


def test_explain_reports_the_terms_the_score_hides():
    embeddings = _rows(0.74, 0.74, 0.90, 0.74, 0.74, 0.73, 0.74, 0.75, 0.74)
    line = explain("taught", score_regions(embeddings, _category(), _tiles(9)))
    assert "taught" in line
    assert "best=0.900" in line
    assert "margin=" in line
