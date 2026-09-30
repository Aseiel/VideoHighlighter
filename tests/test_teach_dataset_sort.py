"""Sorting new footage by a dataset sorted by hand (modules.teach.dataset_sort)."""
from __future__ import annotations

import csv
import os

import numpy as np

from modules.teach import benchmark, cli, dataset_sort
from modules.teach.project import ACCEPTED, ACTIONS, PENDING, Project, Sample

from tests.test_teach_benchmark import AXES, FakeEmbedder, _make


def _clips(root, layout):
    _make(root, layout)
    return benchmark.read_dataset(root)["clips"]


def test_dataset_becomes_classes_and_examples_in_place(tmp_path):
    clips = _clips(str(tmp_path / "ds"), {
        "train": {"alpha move": 5, "Beta Move": 3},
        "val": {"alpha move": 2},
        "test": {"alpha move_beta move": 4}})
    project = Project.create(str(tmp_path / "p"), ACTIONS)

    out = dataset_sort.add_dataset(project, clips, max_examples=6)

    # Folder names as they are (a model trained on them uses the same), train
    # and val both, test left alone, capped per class.
    assert project.class_names() == ["Beta Move", "alpha move"]
    assert out["examples_added"] == {"Beta Move": 3, "alpha move": 6}
    for sample in project.samples:
        assert sample.verdict == ACCEPTED and sample.is_human
        assert os.path.dirname(os.path.dirname(sample.path)).endswith(("train", "val"))
    assert not (tmp_path / "p" / "samples").exists()      # nothing copied

    again = dataset_sort.add_dataset(project, clips, max_examples=6)
    assert again["examples_added"] == {"Beta Move": 0, "alpha move": 0}
    assert len(project.samples) == 9


def _film(project, tmp_path, truths):
    """A cut video: one sample per entry, frames tagged with its axis."""
    source = project.add_source(str(tmp_path / "film.mp4"))
    source.cut = True
    for i, axis in enumerate(truths):
        path = tmp_path / f"film_{i}.mp4"
        path.write_bytes(b"x")
        project.samples.append(Sample(id=f"{source.id}__{i * 5000:08d}", source=source.id,
                                      path=str(path), start=i * 5.0, duration=5.0))
    return source


def test_film_is_sorted_by_the_examples_and_laid_out(tmp_path):
    from modules.teach.sort import sort_project

    clips = _clips(str(tmp_path / "ds"), {"train": {n: 12 for n in AXES}})
    project = Project.create(str(tmp_path / "p"), ACTIONS)
    dataset_sort.add_dataset(project, clips)
    source = _film(project, tmp_path, [0, 0, 1, 2, 1, 0])
    axis_of = {c.path: AXES[c.labels[0]] for c in clips}
    axis_of.update({s.path: [0, 0, 1, 2, 1, 0][i] for i, s in enumerate(
        s for s in project.samples if s.source == source.id)})

    def read(path, count, **_):
        return [("truth", axis_of[path])] * count

    sort_project(project, FakeEmbedder(), frame_reader=read)
    film = [s for s in project.samples if s.source == source.id]
    assert [s.label or s.proposed for s in film] == [
        "alpha move", "alpha move", "beta move", "gamma move", "beta move", "alpha move"]
    assert all(s.is_auto for s in film)          # clear matches need nobody

    out = dataset_sort.lay_out(project)
    assert list(out) == [source.id]
    video = out[source.id]
    assert video["by_class"] == {"alpha move": 3, "beta move": 2, "gamma move": 1}
    assert video["decided"] == {"auto": 6}
    folder = video["folder"]
    names = sorted(os.listdir(os.path.join(folder, "alpha move")))
    assert [n.split("_")[0] for n in names] == ["00h00m00s", "00h00m05s", "00h00m25s"]
    assert all(n.endswith("_auto.mp4") for n in names)
    with open(os.path.join(folder, dataset_sort.TIMELINE), newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["at"] for r in rows][:2] == ["00h00m00s", "00h00m05s"]
    assert rows[3]["class"] == "gamma move" and rows[3]["how"] == "auto"


def test_layout_follows_later_verdicts(tmp_path):
    project = Project.create(str(tmp_path / "p"), ACTIONS)
    project.add_class("alpha move")
    source = _film(project, tmp_path, [0, 0])
    first, second = [s for s in project.samples if s.source == source.id]
    first.proposed, first.scores = "_unsure", {"alpha move": 0.4}
    second.proposed, second.scores = "alpha move", {"alpha move": 0.7}

    out = dataset_sort.lay_out(project)[source.id]
    assert out["by_class"] == {"_unsure": 1, "alpha move": 1}
    assert dataset_sort.pending_of(project, [source.id]) == 2

    project.decide(first, ACCEPTED, "alpha move", by="sheet:1")
    out = dataset_sort.lay_out(project)[source.id]
    assert out["by_class"] == {"alpha move": 2}
    assert out["decided"] == {"checked": 1, "guess": 1}
    assert not os.path.exists(os.path.join(out["folder"], "_unsure"))


def test_from_dataset_command_runs_end_to_end(tmp_path, monkeypatch):
    _make(str(tmp_path / "ds"), {"train": {n: 12 for n in AXES}})
    (tmp_path / "film.mp4").write_bytes(b"x")
    truths = [2, 2, 0, 1]

    def fake_cut(project, progress=None, **_):
        made = 0
        for source in project.sources:
            if source.cut:
                continue
            for i in range(len(truths)):
                path = tmp_path / f"cut_{i}.mp4"
                path.write_bytes(b"c")
                project.samples.append(Sample(id=f"{source.id}__{i}", source=source.id,
                                              path=str(path), start=i * 5.0, duration=5.0))
                made += 1
            source.cut = True
        project.save()
        return {"made": made, "failed": []}

    def read(path, count, **_):
        name = os.path.basename(path)
        if name.startswith("cut_"):
            return [("truth", truths[int(name[4:-4])])] * count
        return [("truth", AXES[os.path.basename(os.path.dirname(path))])] * count

    from modules.teach import cut, embed
    monkeypatch.setattr(cut, "cut_project", fake_cut)
    monkeypatch.setattr(embed, "read_frames", read)
    monkeypatch.setattr(cli, "make_embedder", FakeEmbedder)

    code, out = cli.run(["--project", str(tmp_path / "p"), "from-dataset",
                         str(tmp_path / "ds"), "--videos", str(tmp_path / "film.mp4")])

    assert code == 0, out
    video = next(iter(out["videos"].values()))
    assert video["by_class"] == {"gamma move": 2, "alpha move": 1, "beta move": 1}
    assert out["still_to_check"] == 0

    code, again = cli.run(["--project", str(tmp_path / "p"), "by-class"])
    assert code == 0 and next(iter(again["videos"].values()))["samples"] == 4
