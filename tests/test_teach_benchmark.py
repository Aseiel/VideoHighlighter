"""Measuring the teaching loop against a dataset sorted by hand (modules.teach.benchmark)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from modules.teach import benchmark, cli

DIM = 16


def _direction(k):
    v = np.zeros(DIM, np.float32)
    v[k] = 1.0
    return v


class FakeEmbedder:
    """A frame ``("truth", k)`` embeds near axis ``k``; texts are never needed."""

    model_id = "fake"

    def __init__(self, noise=0.05, seed=0):
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def images(self, frames):
        out = [_direction(f[1]) + 0.3 * _direction(DIM - 1)
               + self.noise * self.rng.normal(size=DIM) for f in frames]
        v = np.array(out, np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def texts(self, texts):
        raise AssertionError("every class has examples; words are not needed")


def _make(root, layout, content=None):
    """``{split: {folder: count}}`` of empty (or given) clips, named like the app's."""
    for split, folders in layout.items():
        for folder, count in folders.items():
            path = os.path.join(root, split, folder)
            os.makedirs(path, exist_ok=True)
            for i in range(count):
                name = f"{i % 4 + 1}_temp_clip_{i}.mp4"
                with open(os.path.join(path, name), "wb") as handle:
                    handle.write((content or f"{split}/{folder}/{i}").encode())


def _reader(axis_of_folder):
    """frame_reader: a clip's frames carry the axis of the folder it sits in."""
    def read(path, count, **_):
        folder = os.path.basename(os.path.dirname(path))
        return [("truth", axis_of_folder[folder])] * count
    return read


# --- reading ---------------------------------------------------------------

def test_labels_split_pairs_and_follow_aliases():
    assert benchmark.labels_of("alpha move_beta move") == ("alpha move", "beta move")
    aliases = {"alfa move": "alpha move", "fine beta": "beta move", "gone": "",
               "a b whole": "alpha move_beta move"}
    assert benchmark.labels_of("alfa move", aliases) == ("alpha move",)
    assert benchmark.labels_of("fine beta_alpha move", aliases) == ("beta move", "alpha move")
    assert benchmark.labels_of("gone", aliases) == ()
    assert benchmark.labels_of("a b whole", aliases) == ("alpha move", "beta move")


def test_group_is_the_name_before_the_cutters_suffix():
    assert benchmark.group_of("14_highlight_temp_clip_8.mp4") == "14"
    assert benchmark.group_of("29_temp_clip_7_2.mp4") == "29"
    assert benchmark.group_of("5_highlight_new_database_temp_clip_6.mp4.mp4") == "5"
    # A titled video: the old default (a leading number) made each of these
    # clips a video of its own, and held-out videos then held nothing out.
    assert benchmark.group_of("Some Title Here_temp_trimmed_temp_clip_31_cropped_left.mp4") \
        == "Some Title Here"
    assert benchmark.group_of("Some Title Here_temp_trimmed_temp_clip_4.mp4") == "Some Title Here"
    assert benchmark.group_of("clip.mp4") == "clip"
    assert benchmark.group_of("clip_7.mp4", r"^(clip)_") == "clip"


def test_report_warns_when_most_names_hold_no_video(tmp_path):
    folder = tmp_path / "train" / "alpha move"
    folder.mkdir(parents=True)
    for i in range(4):
        (folder / f"take {i}.mp4").write_bytes(b"x")
    (folder / "1_temp_clip_0.mp4").write_bytes(b"y")

    out = benchmark.report(benchmark.read_dataset(str(tmp_path)))

    assert out["clips_without_a_video"] == 4
    assert "held-out scores leak" in out["warning"]


def test_small_classes_are_left_out():
    clips = [benchmark.Clip(f"/d/{c}/{i}.mp4", "train", c, (c,), str(i % 3))
             for c, n in (("alpha move", 25), ("beta move", 3)) for i in range(n)]
    clips.append(benchmark.Clip("/d/t.mp4", "test", "alpha move_beta move",
                                ("alpha move", "beta move"), "9"))

    kept, left_out = benchmark.big_enough(clips, 20)
    assert {c.labels[0] for c in kept} == {"alpha move"} and len(kept) == 25
    assert left_out == {"beta move": 3}
    kept, left_out = benchmark.big_enough(clips, 0)
    assert len(kept) == 28 and left_out == {}


def test_read_skips_underscore_and_videoless_folders(tmp_path):
    _make(str(tmp_path), {"train": {"alpha move": 3, "_backup": 2},
                          "test": {"alpha move_beta move": 1}})
    notes = tmp_path / "train" / "labels" / "alpha move"
    notes.mkdir(parents=True)
    (notes / "a.txt").write_text("0 0.5 0.5 0.1 0.1")

    data = benchmark.read_dataset(str(tmp_path))

    assert [c.labels for c in data["clips"]] == [("alpha move",)] * 3 + [
        ("alpha move", "beta move")]
    why = {s["folder"]: s["why"] for s in data["skipped"]}
    assert why == {"train/_backup": "starts with _", "train/labels": "no video directly inside"}


def test_report_flags_what_would_skew_a_measurement(tmp_path):
    _make(str(tmp_path), {"train": {"alpha move": 25, "beta move": 3},
                          "val": {"alpha move": 2, "alfa typo": 1},
                          "test": {"alpha move_gamma move": 2}})
    # One clip copied into another class's folder.
    src = tmp_path / "train" / "alpha move" / "1_temp_clip_0.mp4"
    (tmp_path / "train" / "beta move" / "copy (1).mp4").write_bytes(src.read_bytes())

    out = benchmark.report(benchmark.read_dataset(str(tmp_path)), str(tmp_path))

    assert out["below_minimum"] == {"beta move": 4}
    assert out["not_in_train"] == {"val": ["alfa typo"], "test": ["gamma move"]}
    assert out["missing_from_val"] == ["beta move"]
    assert out["multi_class_clips"]["test"] == 2
    assert out["duplicates"]["groups"] == 1 and out["duplicates"]["across_classes"] == 1
    assert out["videos_in_several_splits"]            # clip names share video numbers


def test_import_command_reads_only(tmp_path, capsys):
    _make(str(tmp_path / "ds"), {"train": {"alpha move": 2}})
    aliases = tmp_path / "aliases.json"
    aliases.write_text(json.dumps({"alpha move": "alpha renamed"}))

    code, out = cli.run(["--project", str(tmp_path / "p"), "import", str(tmp_path / "ds"),
                         "--aliases", str(aliases)])

    assert code == 0 and out["counts"]["train"] == {"alpha renamed": 2}
    assert not (tmp_path / "p").exists()


# --- the loop, answered by the folders -------------------------------------

LAYOUT = {"train": {"alpha move": 30, "beta move": 30, "gamma move": 30},
          "val": {"alpha move": 4, "beta move": 4}}
AXES = {"alpha move": 0, "beta move": 1, "gamma move": 2}


def test_simulation_auto_accepts_clear_classes_and_counts_the_looks(tmp_path):
    _make(str(tmp_path / "ds"), LAYOUT)
    clips = benchmark.read_dataset(str(tmp_path / "ds"))["clips"]

    out = benchmark.simulate(clips, FakeEmbedder(), str(tmp_path / "work"),
                             frame_reader=_reader(AXES))

    assert out["clips"] == 98 and out["left_undecided"] == 0
    # Well-separated classes: most go in unchecked, and none wrongly.
    assert out["auto_accepted_unchecked"] > 30
    assert out["auto_wrong"] == 0
    assert out["looked_at"] + out["auto_accepted_unchecked"] >= out["clips"]
    assert out["looked_at_share"] < 0.7
    assert out["first_sort"]["proposed_right"] == out["first_sort"]["proposed"]
    assert set(out["per_class"]) == set(AXES)
    assert out["curve"] and out["curve"][-1]["pending"] == 0


def test_simulation_catches_a_class_that_looks_like_another(tmp_path):
    # "gamma" clips look exactly like "alpha" ones: nothing can tell them
    # apart, so the loop must not wave them through as either.
    _make(str(tmp_path / "ds"), LAYOUT)
    clips = benchmark.read_dataset(str(tmp_path / "ds"))["clips"]
    axes = dict(AXES, **{"gamma move": 0})

    out = benchmark.simulate(clips, FakeEmbedder(), str(tmp_path / "work"),
                             frame_reader=_reader(axes))

    assert out["left_undecided"] == 0 and out["auto_wrong"] == 0
    for name in ("alpha move", "gamma move"):
        assert out["per_class"][name]["auto_accepted"] == 0
    assert out["per_class"]["beta move"]["auto_accepted"] > 0
    assert out["looked_at_share"] > 0.7


def test_simulation_reuses_vectors_and_never_clears_a_foreign_folder(tmp_path):
    _make(str(tmp_path / "ds"), LAYOUT)
    clips = benchmark.read_dataset(str(tmp_path / "ds"))["clips"]
    work = tmp_path / "work"
    benchmark.simulate(clips, FakeEmbedder(), str(work), max_sheets=1,
                       frame_reader=_reader(AXES))

    def no_frames(path, count, **_):
        raise AssertionError("vectors should come from the cache")

    again = benchmark.simulate(clips, FakeEmbedder(), str(work), max_sheets=1,
                               frame_reader=no_frames)
    assert again["clips"] == 98

    foreign = tmp_path / "mine"
    (foreign / benchmark.SIMULATION_DIR).mkdir(parents=True)
    (foreign / benchmark.SIMULATION_DIR / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        benchmark.simulate(clips, FakeEmbedder(), str(foreign), frame_reader=_reader(AXES))
    assert (foreign / benchmark.SIMULATION_DIR / "keep.txt").exists()


# --- a model on val and test ------------------------------------------------

def test_model_test_scores_single_and_paired_clips(tmp_path):
    _make(str(tmp_path / "ds"), {"val": {"alpha move": 2, "beta move": 2},
                                 "test": {"alpha move_beta move": 2,
                                          "alpha move_gamma move": 1,
                                          "delta move": 1}})
    clips = benchmark.read_dataset(str(tmp_path / "ds"))["clips"]

    def scorer(path):
        folder = os.path.basename(os.path.dirname(path))
        if folder == "beta move":           # always mistaken for alpha
            return {"alpha move": 0.6, "beta move": 0.3, "gamma move": 0.1}
        if folder == "alpha move_gamma move":
            return {"alpha move": 0.5, "beta move": 0.3, "gamma move": 0.2}
        return {"alpha move": 0.5, "beta move": 0.4, "gamma move": 0.1}

    out = benchmark.model_test(clips, scorer)

    assert out["val"]["accuracy"] == 0.5 and out["val"]["balanced_accuracy"] == 0.5
    assert out["confusions"] == {"beta move -> alpha move": 2}
    multi = out["test_multi"]
    assert multi["clips"] == 3
    assert multi["per_combination"]["alpha move + beta move"]["all_on_top"] == 2
    assert multi["per_combination"]["alpha move + gamma move"] == {
        "clips": 1, "all_on_top": 0, "top_is_one": 1, "all_in_top3": 1}
    assert out["classes_the_model_lacks"] == {"delta move": 1}


def test_evaluate_command_saves_its_answer(tmp_path, monkeypatch):
    _make(str(tmp_path / "ds"), LAYOUT)
    monkeypatch.setattr(cli, "make_embedder", lambda *_: FakeEmbedder())
    from modules.teach import embed
    monkeypatch.setattr(embed, "read_frames", _reader(AXES))

    code, out = cli.run(["--project", str(tmp_path / "bench"), "evaluate",
                         str(tmp_path / "ds"), "--max-sheets", "2"])

    assert code == 0, out
    assert out["simulation"]["sheets"] <= 2
    with open(out["saved"], encoding="utf-8") as handle:
        assert json.load(handle)["simulation"]["clips"] == 98


# --- new footage, sorted by everything else --------------------------------

def test_sort_test_holds_out_whole_videos(tmp_path):
    _make(str(tmp_path / "ds"), LAYOUT)       # clip names carry videos 1-4
    clips = benchmark.read_dataset(str(tmp_path / "ds"))["clips"]
    seen = []

    def read(path, count, **_):
        seen.append(path)
        return _reader(AXES)(path, count)

    out = benchmark.sort_test(clips, FakeEmbedder(), str(tmp_path / "work"),
                              holdout=0.25, frame_reader=read)

    assert out["held_out_videos"] == 1
    # Every held-out clip is from one video, and none of them is an example.
    assert 0 < out["clips"] < 98 and out["examples"] + out["clips"] == 98
    assert out["best_guess_right"] == 1.0 and out["auto_wrong"] == 0
    assert out["auto_accepted"] + out["left_to_check"] == out["clips"]
    assert out["classes_without_examples"] == []


def test_sort_test_needs_more_than_one_video(tmp_path):
    _make(str(tmp_path / "ds"), {"train": {"alpha move": 4}})
    clips = benchmark.read_dataset(str(tmp_path / "ds"), group_pattern="^(x)")["clips"]
    clips = [benchmark.Clip(c.path, c.split, c.folder, c.labels, "one") for c in clips]
    with pytest.raises(ValueError, match="one video"):
        benchmark.sort_test(clips, FakeEmbedder(), str(tmp_path / "w"))


def test_files_named_past_their_extension_are_reported(tmp_path):
    _make(str(tmp_path / "ds"), {"train": {"alpha move": 2}})
    (tmp_path / "ds" / "train" / "alpha move" / "7_temp_clip_1.mp4 (1)").write_bytes(b"x")
    (tmp_path / "ds" / "train" / "alpha move" / "notes.txt").write_text("x")

    out = benchmark.report(benchmark.read_dataset(str(tmp_path / "ds")))

    assert out["misnamed"] == {"files": 1,
                               "examples": ["train/alpha move/7_temp_clip_1.mp4 (1)"]}
    assert out["counts"]["train"] == {"alpha move": 2}


# --- several centres per class ----------------------------------------------

def test_a_class_shown_two_ways_keeps_both(tmp_path):
    from modules.teach import scoring

    rng = np.random.default_rng(0)
    looks = [_direction(0), _direction(1)]
    examples = [looks[i % 2] + 0.05 * rng.normal(size=DIM) for i in range(12)]

    one = scoring.build_prototype("alpha move", examples)
    two = scoring.build_prototype("alpha move", examples, centers=2)

    assert one.centers is None and two.centers.shape == (2, DIM)
    probe = scoring._unit(np.stack([looks[0], looks[1]]))
    # The mean sits between the two looks; the centres sit on them.
    assert np.all(two.cosines(probe) > 0.95) and np.all(one.cosines(probe) < 0.8)
    assert two.anchor > one.anchor
    # Too few examples for a second centre: one mean, as before.
    assert scoring.build_prototype("alpha move", examples[:5], centers=2).centers is None
