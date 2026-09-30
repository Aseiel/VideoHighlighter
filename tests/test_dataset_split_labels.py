"""The train/val auto-split must keep every clip under the folder it came from.

Each split numbers its classes from its own folder list, so when ``val`` lacks
a folder ``train`` has, the same index names two different classes. The
auto-split mixes the two splits; reading a val sample's index against train's
list relabelled clips silently (and dropped others), which corrupted both the
training labels and the validation score that picks the best epoch.
"""
from __future__ import annotations

import os
import random
import sys
import types

import pytest


class _Dataset:
    pass


def _train_test_split(items, test_size, random_state):
    items = list(items)
    random.Random(random_state).shuffle(items)
    n = max(1, round(len(items) * test_size))
    return items[n:], items[:n]


@pytest.fixture
def dataset_module(monkeypatch):
    # torch and sklearn are shimmed as MagicMocks in conftest; this code path
    # needs a real base class and a real splitter, nothing else from them.
    data = types.ModuleType("torch.utils.data")
    data.Dataset = _Dataset
    utils = types.ModuleType("torch.utils")
    utils.data = data
    selection = types.ModuleType("sklearn.model_selection")
    selection.train_test_split = _train_test_split
    monkeypatch.setitem(sys.modules, "torch.utils", utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", data)
    monkeypatch.setitem(sys.modules, "sklearn.model_selection", selection)
    monkeypatch.delitem(sys.modules, "model_training.shared.dataset", raising=False)
    monkeypatch.delitem(sys.modules, "model_training.shared", raising=False)
    import model_training.shared.dataset as dataset
    return dataset


def _make(root, layout):
    for split, classes in layout.items():
        for name, count in classes.items():
            folder = os.path.join(root, split, name)
            os.makedirs(folder)
            for i in range(count):
                open(os.path.join(folder, f"{split}_{name}_{i}.mp4"), "w").close()


def _split(dataset, root):
    config = {"min_train_per_action": 5, "min_val_per_action": 2}
    train = dataset.VideoDataset(os.path.join(root, "train"), config)
    val = dataset.VideoDataset(os.path.join(root, "val"), config)
    ok, actions, new_train, new_val = dataset.validate_and_split_dataset(train, val, config)
    assert ok
    dataset.apply_dataset_split(train, val, actions, new_train, new_val)
    return train, val


def _mislabelled(ds):
    return [(os.path.basename(path), ds.idx_to_label[label])
            for path, label in ds.samples
            if os.path.basename(os.path.dirname(path)) != ds.idx_to_label[label]]


def test_val_missing_a_class_keeps_every_label(dataset_module, tmp_path):
    # val has no "b": its "c" is index 1, which is train's "b".
    layout = {"train": {"a": 10, "b": 10, "c": 10}, "val": {"a": 1, "c": 1}}
    _make(str(tmp_path), layout)

    train, val = _split(dataset_module, str(tmp_path))

    assert _mislabelled(train) == []
    assert _mislabelled(val) == []
    assert len(train.samples) + len(val.samples) == 32


def test_val_only_class_is_left_out(dataset_module, tmp_path):
    # A folder only val has (a typo, say) is not a class train can learn.
    layout = {"train": {"a": 10, "b": 10}, "val": {"a": 3, "b": 3, "a typo": 2}}
    _make(str(tmp_path), layout)

    train, val = _split(dataset_module, str(tmp_path))

    assert train.labels == ["a", "b"]
    assert _mislabelled(train) == [] and _mislabelled(val) == []
    assert not any("a typo" in path for path, _ in train.samples + val.samples)
