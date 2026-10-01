"""The linear layer trained on what a person sorted (modules.teach.linear)."""
from __future__ import annotations

import numpy as np
import pytest

from modules.teach import linear, sort
from modules.teach.project import ACCEPTED, ACTIONS, Project, Sample


def _blobs(per_class, centres, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    x, y, g = [], [], []
    for name, centre in centres.items():
        for i in range(per_class[name]):
            x.append(np.asarray(centre, float) + noise * rng.normal(size=len(centre)))
            y.append(name)
            g.append(f"video{i % 4}")
    return np.array(x), y, g


def test_it_tells_classes_apart_and_gives_probabilities():
    x, y, _ = _blobs({"alpha move": 40, "beta move": 40, "gamma move": 40},
                     {"alpha move": [2, 0, 0], "beta move": [0, 2, 0], "gamma move": [0, 0, 2]})
    model = linear.fit(x, y)
    p = model.proba(x)
    assert model.classes == ["alpha move", "beta move", "gamma move"]
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.mean(np.array(model.classes)[p.argmax(axis=1)] == np.array(y)) > 0.95


def test_a_small_class_is_not_drowned_by_a_large_one():
    # Overlapping classes, 10x apart in size: weighted by size, the small one
    # is still predicted for most of its own samples.
    x, y, _ = _blobs({"alpha move": 300, "beta move": 30},
                     {"alpha move": [0.0, 0.0], "beta move": [1.2, 1.2]}, noise=0.8)
    model = linear.fit(x, y)
    pred = np.array(model.classes)[model.proba(x).argmax(axis=1)]
    small = np.array(y) == "beta move"
    assert np.mean(pred[small] == "beta move") > 0.6


def test_threshold_holds_out_whole_videos():
    x, y, g = _blobs({"alpha move": 40, "beta move": 40},
                     {"alpha move": [1.5, 0], "beta move": [0, 1.5]}, noise=0.8)
    out = linear.threshold_for(x, y, g, precision=0.9)
    assert out["groups"] == 4
    assert 0.5 <= out["threshold"] <= 1.0
    assert 0 < out["coverage"] <= 1.0
    # one video: nothing to hold out, so no threshold rather than a made-up one
    assert linear.threshold_for(x, y, ["v"] * len(y), precision=0.9) is None


def test_one_class_is_not_enough():
    with pytest.raises(ValueError):
        linear.fit(np.zeros((3, 2)), ["alpha move"] * 3)


def _project_with(tmp_path, counts):
    project = Project.create(str(tmp_path / "p"), ACTIONS)
    project.settings.scorer = "linear"
    vectors = {}
    for k, (name, n) in enumerate(counts.items()):
        project.add_class(name)
        for i in range(n):
            sid = f"s{k}_{i}"
            project.samples.append(Sample(id=sid, source=f"video{i % 3}", path=f"/x/{sid}.mp4",
                                          start=0.0, duration=0.0))
            project.decide(project.get_sample(sid), ACCEPTED, name, by="example")
            project.get_class(name).examples.append(sid)
            v = np.zeros(8, np.float32)
            v[k] = 1.0
            vectors[sid] = v
    return project, vectors


def test_a_class_with_too_few_examples_is_left_out_of_the_layer(tmp_path):
    project, vectors = _project_with(tmp_path, {"alpha move": 8, "beta move": 8, "gamma move": 2})
    rows, labels, groups, left_out = sort.linear_training_set(project, vectors)
    assert left_out == {"gamma move": 2}
    assert sorted(set(labels)) == ["alpha move", "beta move"]
    assert len(rows) == len(labels) == len(groups) == 16


def test_too_few_classes_with_enough_examples_falls_back_to_prototypes(tmp_path):
    project, vectors = _project_with(tmp_path, {"alpha move": 8, "beta move": 2})
    assert sort.score_linear(project, vectors, list(vectors)) is None
