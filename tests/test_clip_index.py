"""Tests for the CLIP frame index's scoring and persistence.

No CLIP here: `llm.clip_index` only imports the model inside functions, and
everything below the embeddings — softmax calibration, ranking, caching — is
numpy. So the layer that actually decides what the user sees is testable in
milliseconds, on synthetic embeddings with known geometry.

The `logit_scale=2.0` in the scoring tests is deliberate: CLIP's real 100.0
saturates softmax to exactly 0/1 on clean synthetic vectors, which would hide
the very dilution effects these tests exist to pin down.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from llm.clip_index import (
    INDEX_FORMAT_VERSION,
    ClipFrameIndex,
    l2_normalize,
    score_embeddings,
)


DIM = 8


def _unit(*components) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for i, c in enumerate(components):
        v[i] = c
    return l2_normalize(v)


@pytest.fixture
def index() -> ClipFrameIndex:
    """Four frames: two on the +x axis, two on the +y axis."""
    emb = np.vstack([_unit(1, 0), _unit(0.9, 0.1), _unit(0, 1), _unit(0.1, 0.9)])
    return ClipFrameIndex(np.array([0.0, 1.0, 2.0, 3.0]), emb,
                          {"logit_scale": 100.0, "model": "test/model"})


class TestQueryVector:
    def test_cosine_without_negatives_ranks_by_similarity(self, index):
        scores = index.query_vector(_unit(1, 0))
        assert scores[0] > scores[1] > scores[3] > scores[2]

    def test_cosine_without_negatives_stays_in_unit_range(self, index):
        # Rescaled from [-1,1]; an orthogonal query must not go negative.
        scores = index.query_vector(_unit(0, 0, 1))
        assert np.all((scores >= 0.0) & (scores <= 1.0))

    def test_softmax_against_negative_separates_classes(self, index):
        scores = index.query_vector(_unit(1, 0), negatives=_unit(0, 1).reshape(1, -1),
                                    logit_scale=2.0)
        assert scores[0] > 0.5 and scores[1] > 0.5      # +x frames
        assert scores[2] < 0.5 and scores[3] < 0.5      # +y frames

    def test_normalizes_a_non_unit_query(self, index):
        long_vec = _unit(1, 0) * 17.0
        assert np.allclose(index.query_vector(long_vec), index.query_vector(_unit(1, 0)))

    def test_more_negatives_dilute_the_same_match(self, index):
        """Why score_category contrasts against ONE averaged vector.

        Softmax over [positive, *negatives] divides by a sum that grows with the
        negative count, so passing more of them lowers an unchanged match. This
        is what dropped real hits below 0.5 with a bag of sampled frames.
        """
        q, neg = _unit(1, 0), _unit(0, 1)
        one = index.query_vector(q, negatives=neg.reshape(1, -1), logit_scale=2.0)
        many = index.query_vector(q, negatives=np.vstack([neg] * 20), logit_scale=2.0)
        assert many[0] < one[0]

    def test_softmax_is_stable_for_extreme_scale(self, index):
        # Large logits must not overflow into nan via exp().
        scores = index.query_vector(_unit(1, 0), negatives=_unit(0, 1).reshape(1, -1),
                                    logit_scale=10_000.0)
        assert np.all(np.isfinite(scores))


class TestMemo:
    """lookup/extend are what let a scan reuse earlier work and fill only gaps."""

    def test_lookup_splits_known_from_unknown(self, index):
        rows, positions, missing = index.lookup([1.0, 9.0, 2.0])
        assert missing == [9.0]
        assert [index.timestamps[r] for r in rows] == [1.0, 2.0]
        assert positions == [0, 2]      # indexes back into the query list

    def test_lookup_tolerates_float_drift(self, index):
        # 3*0.1 != 0.3 in binary; ms-resolution keys must still hit.
        rows, _, missing = index.lookup([0.1 + 0.1 + 0.1 + 2.7])
        assert not missing and len(rows) == 1

    def test_extend_adds_only_new_timestamps(self, index):
        added = index.extend([9.0, 10.0], np.vstack([_unit(0, 0, 1), _unit(0, 0, 0, 1)]))
        assert added == 2 and len(index) == 6
        assert not index.lookup([9.0, 10.0])[2]

    def test_extend_ignores_duplicates(self, index):
        # Replaying a scan must be free, not corrupting.
        before = len(index)
        assert index.extend([0.0, 1.0], index.embeddings[:2]) == 0
        assert len(index) == before

    def test_extend_dedupes_within_one_call(self, index):
        assert index.extend([9.0, 9.0], np.vstack([_unit(0, 0, 1)] * 2)) == 1

    def test_extend_rejects_mismatched_lengths(self, index):
        with pytest.raises(ValueError):
            index.extend([9.0, 10.0], _unit(0, 0, 1).reshape(1, -1))

    def test_extend_onto_empty_index(self):
        empty = ClipFrameIndex(np.zeros(0), np.zeros((0, DIM), dtype=np.float32))
        assert empty.extend([1.0], _unit(1, 0).reshape(1, -1)) == 1
        assert len(empty) == 1

    def test_extended_rows_are_queryable(self, index):
        index.extend([9.0], _unit(1, 0).reshape(1, -1))
        rows, _, _ = index.lookup([9.0])
        assert index.query_vector(_unit(1, 0))[rows[0]] > 0.99

    def test_round_trip_preserves_the_memo(self, index, tmp_path):
        index.extend([9.0], _unit(0, 0, 1).reshape(1, -1))
        path = str(tmp_path / "m.npz")
        index.save(path)
        assert not ClipFrameIndex.load(path).lookup([9.0])[2]   # still known


class TestScoreEmbeddings:
    def test_matches_query_vector(self, index):
        labels = np.vstack([_unit(1, 0), _unit(0, 1)])
        direct = score_embeddings(index.embeddings, labels, logit_scale=2.0)
        via = index.query_vector(_unit(1, 0), negatives=_unit(0, 1).reshape(1, -1),
                                 logit_scale=2.0)
        assert np.allclose(direct, via)


class TestTopK:
    def test_returns_best_first_with_timestamps(self, index):
        top = index.top_k(index.query_vector(_unit(1, 0)), k=2)
        assert [ts for ts, _ in top] == [0.0, 1.0]
        assert top[0][1] >= top[1][1]

    def test_k_larger_than_index_returns_everything(self, index):
        assert len(index.top_k(index.query_vector(_unit(1, 0)), k=99)) == len(index)


class TestPersistence:
    def test_round_trip_preserves_ranking(self, index, tmp_path):
        path = str(tmp_path / "idx.npz")
        index.save(path)
        loaded = ClipFrameIndex.load(path)
        assert len(loaded) == len(index)
        assert np.allclose(loaded.timestamps, index.timestamps)
        # fp16 on disk: values shift slightly, order must not.
        assert np.allclose(loaded.embeddings, index.embeddings, atol=1e-3)
        assert loaded.meta["model"] == "test/model"

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ValueError):
            ClipFrameIndex(np.array([0.0, 1.0]), np.zeros((3, DIM), dtype=np.float32))

    def test_matches_detects_changed_video(self, index, tmp_path):
        video = tmp_path / "v.mp4"
        video.write_bytes(b"x" * 100)
        st = os.stat(video)
        index.meta = {"format": INDEX_FORMAT_VERSION, "video": "v.mp4", "size": st.st_size,
                      "mtime": int(st.st_mtime), "interval": 1.0, "model": "m"}
        assert index.matches(str(video), "m", 1.0)
        assert not index.matches(str(video), "m", 0.5)      # different sampling
        assert not index.matches(str(video), "other", 1.0)  # different model
        assert index.matches(str(video), "m")               # memo: interval unchecked
        video.write_bytes(b"x" * 200)                       # re-encoded
        assert not index.matches(str(video), "m", 1.0)
