"""Tests for categories taught from example frames.

Synthetic embeddings with known geometry, no CLIP — see test_clip_index.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from llm.clip_index import ClipFrameIndex, l2_normalize
from llm.clip_categories import (
    CategoryStore,
    CustomCategory,
    background_vector,
    learn_category,
    score_category,
)


DIM = 8


def _unit(*components) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for i, c in enumerate(components):
        v[i] = c
    return l2_normalize(v)


@pytest.fixture
def index() -> ClipFrameIndex:
    """Ten frames: 0-2 are the 'target' look, 3-9 are background."""
    target = [_unit(1, 0), _unit(0.95, 0.05), _unit(0.9, 0.1)]
    background = [_unit(0.05 * i, 1) for i in range(7)]
    emb = np.vstack(target + background)
    return ClipFrameIndex(np.arange(10, dtype=np.float32), emb,
                          {"logit_scale": 2.0, "model": "test/model"})


class TestLearnCategory:
    def test_prototype_is_unit_norm(self, index):
        cat = learn_category(index, "thing", [0, 1])
        assert np.isclose(np.linalg.norm(cat.vector), 1.0)

    def test_prototype_sits_between_its_examples(self, index):
        cat = learn_category(index, "thing", [0, 2])
        assert cat.vector @ index.embeddings[0] > 0.9
        assert cat.vector @ index.embeddings[2] > 0.9

    def test_records_provenance(self, index):
        cat = learn_category(index, "thing", [0, 1, 2])
        assert cat.n_examples == 3
        assert cat.model_id == "test/model"   # needed to spot a CLIP swap later

    def test_no_examples_rejected(self, index):
        with pytest.raises(ValueError):
            learn_category(index, "thing", [])

    def test_single_example_is_that_frame(self, index):
        cat = learn_category(index, "thing", [0])
        assert np.allclose(cat.vector, index.embeddings[0], atol=1e-6)


class TestBackgroundVector:
    def test_is_one_unit_vector_not_a_bag(self, index):
        bg = background_vector(index)
        assert bg.shape == (DIM,)
        assert np.isclose(np.linalg.norm(bg), 1.0)

    def test_excludes_requested_frames(self, index):
        # Excluding every background frame leaves only target frames to average,
        # so the "background" must swing toward the target axis.
        bg = background_vector(index, exclude=list(range(3, 10)))
        assert bg[0] > bg[1]

    def test_deterministic(self, index):
        assert np.allclose(background_vector(index), background_vector(index))

    def test_empty_index_returns_zeros(self):
        empty = ClipFrameIndex(np.array([]), np.zeros((0, DIM), dtype=np.float32))
        assert not np.any(background_vector(empty))


class TestScoreCategory:
    def test_separates_target_from_background(self, index):
        cat = learn_category(index, "thing", [0, 1])
        scores = score_category(index, cat, exclude_from_background=[0, 1])
        assert np.all(scores[:3] > 0.5), "the target frames must score high"
        assert np.all(scores[3:] < 0.5), "background frames must score low"

    def test_generalizes_beyond_the_examples(self, index):
        """Frame 2 is never shown as an example but shares the target's look."""
        cat = learn_category(index, "thing", [0, 1])
        assert score_category(index, cat, exclude_from_background=[0, 1])[2] > 0.5

    def test_explicit_background_overrides_sampling(self, index):
        cat = learn_category(index, "thing", [0, 1])
        scores = score_category(index, cat, background=_unit(1, 0))
        # Contrasted against itself, nothing can stand out.
        assert np.all(scores < 0.6)


class TestCategoryStore:
    def test_round_trip(self, index, tmp_path):
        path = str(tmp_path / "cats.json")
        cat = learn_category(index, "thing", [0, 1])
        store = CategoryStore(path)
        store.add(cat)
        store.save()

        reloaded = CategoryStore(path).load()
        assert reloaded.names() == ["thing"]
        assert np.allclose(reloaded.categories["thing"].vector, cat.vector, atol=1e-6)
        assert reloaded.categories["thing"].n_examples == 2

    def test_missing_file_loads_empty(self, tmp_path):
        assert CategoryStore(str(tmp_path / "nope.json")).load().names() == []

    def test_corrupt_file_does_not_raise(self, tmp_path):
        path = tmp_path / "cats.json"
        path.write_text("{ not json", encoding="utf-8")
        assert CategoryStore(str(path)).load().names() == []

    def test_add_replaces_same_name(self, index, tmp_path):
        store = CategoryStore(str(tmp_path / "c.json"))
        store.add(learn_category(index, "thing", [0]))
        store.add(learn_category(index, "thing", [3]))
        assert len(store.categories) == 1
        assert store.categories["thing"].n_examples == 1

    def test_remove(self, index, tmp_path):
        store = CategoryStore(str(tmp_path / "c.json"))
        store.add(learn_category(index, "thing", [0]))
        assert store.remove("thing") is True
        assert store.remove("thing") is False

    def test_stale_flags_categories_from_another_model(self, index, tmp_path):
        store = CategoryStore(str(tmp_path / "c.json"))
        store.add(learn_category(index, "thing", [0]))
        assert store.stale("test/model") == []
        assert store.stale("other/model") == ["thing"]

    def test_stale_ignores_categories_without_provenance(self, tmp_path):
        store = CategoryStore(str(tmp_path / "c.json"))
        store.add(CustomCategory(name="old", vector=_unit(1, 0), model_id=""))
        assert store.stale("any/model") == []
