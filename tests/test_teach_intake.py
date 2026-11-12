"""Adding a reviewed video to a dataset, and asking whether it helped.

Pinned here: the added crops are one source video; a class the dataset has
plenty of gets nothing and a thin one gets at most the cap, spread over the
video; the plan changes nothing; what was added is recorded and ``undo``
removes exactly that; a name the dataset already uses is refused. And the
comparison scores only the dataset's other videos, calls a change only when
every seed agrees, and says "worse" when the added clips teach the wrong
thing.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from model_training.action_head import compare as C
from modules.teach import benchmark, cli, intake


def _clip(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(os.path.basename(path).encode())


@pytest.fixture
def folders(tmp_path):
    """A dataset with a big class and a thin one, and a reviewed video with
    plenty of both plus a class the dataset has never seen."""
    dataset, review = str(tmp_path / "dataset"), str(tmp_path / "review")
    for i in range(12):
        _clip(os.path.join(dataset, "train", "big", f"film{i % 4}_temp_clip_{i}.mp4"))
    for i in range(3):
        _clip(os.path.join(dataset, "train", "thin", f"film{i}_temp_clip_t{i}.mp4"))
    _clip(os.path.join(dataset, "val", "big", "film9_temp_clip_0.mp4"))
    for i in range(20):
        _clip(os.path.join(review, "big", f"src__{i:08d}_crop.mp4"))
        _clip(os.path.join(review, "thin", f"src__{i:08d}_crop.mp4"))
    for i in range(4):
        _clip(os.path.join(review, "thin", "_overlap", f"src__{100 + i:08d}_crop.mp4"))
        _clip(os.path.join(review, "fresh", f"src__{i:08d}_crop.mp4"))
    _clip(os.path.join(review, "_reject", "src__00000000_crop.mp4"))
    return dataset, review


def _rows(planned):
    return {row["folder"]: row for row in planned["folders"]}


def test_plan_takes_thin_classes_capped_and_spread(folders):
    dataset, review = folders
    planned = intake.plan(dataset, review, "newfilm", thin_under=10, cap=6)
    rows = _rows(planned)
    assert rows["big"]["take"] == 0 and rows["big"]["why"] == intake.ENOUGH
    assert rows["thin"]["found"] == 24 and rows["thin"]["take"] == 6
    assert rows["fresh"]["take"] == 4                    # a class it has none of is thin
    assert "_reject" not in rows
    # Spread over the video, not its first six crops.
    times = [int(os.path.basename(p).split("__")[1][:8]) for p in rows["thin"]["paths"]]
    assert times == sorted(times) and times[-1] >= 12
    assert planned["take"] == 10 and planned["new_classes"] == ["fresh"]
    assert any("Only in this video" in w for w in planned["warnings"])
    # Reads only.
    assert not os.path.exists(os.path.join(dataset, intake.MANIFEST_DIR))
    assert len(benchmark.read_dataset(dataset)["clips"]) == 16


def test_no_limits_is_allowed_and_flagged(folders):
    dataset, review = folders
    planned = intake.plan(dataset, review, "newfilm", thin_under=0, cap=0)
    assert planned["take"] == planned["found"] == 48
    assert planned["share_of_train"] > 0.7
    assert any("No limit" in w for w in planned["warnings"])
    assert any("of the training clips" in w for w in planned["warnings"])


def test_apply_adds_one_source_video_and_undo_removes_exactly_it(folders):
    dataset, review = folders
    before = sorted(c.path for c in benchmark.read_dataset(dataset)["clips"])
    planned = intake.plan(dataset, review, "newfilm", thin_under=10, cap=6)
    done = intake.apply(planned)
    assert done["added"] == 10
    clips = benchmark.read_dataset(dataset)["clips"]
    new = [c for c in clips if c.path not in before]
    assert len(new) == 10 and {c.group for c in new} == {"newfilm"}
    assert {c.split for c in new} == {"train"}
    assert intake.added_videos(dataset) == ["newfilm"]
    # The review folder is untouched, and a repeat adds nothing.
    assert len(os.listdir(os.path.join(review, "thin"))) == 21
    again = intake.apply(intake.plan(dataset, review, "newfilm", thin_under=10, cap=6))
    assert again["added"] == 0 and again["in_dataset"] == 10
    gone = intake.undo(dataset, "newfilm")
    assert gone["removed"] == 10
    assert sorted(c.path for c in benchmark.read_dataset(dataset)["clips"]) == before
    assert not os.path.isdir(os.path.join(dataset, "train", "fresh"))
    assert intake.added_videos(dataset) == []


def test_a_repeat_does_not_count_its_own_crops_as_the_dataset(folders):
    dataset, review = folders
    intake.apply(intake.plan(dataset, review, "newfilm", thin_under=5, cap=6))
    # "thin" now holds 3 + 6 clips; its own 6 must not make it look full.
    rows = _rows(intake.plan(dataset, review, "newfilm", thin_under=5, cap=6))
    assert rows["thin"]["take"] == 6 and rows["thin"]["dataset_clips"] == {"thin": 3}


def test_names_that_would_break_the_source_video_rule_are_refused(folders):
    dataset, review = folders
    with pytest.raises(ValueError, match="already has"):
        intake.plan(dataset, review, "film1")
    with pytest.raises(ValueError, match="cannot name"):
        intake.plan(dataset, review, "my_highlight_reel")
    with pytest.raises(FileNotFoundError):
        intake.undo(dataset, "never-added")


def test_the_command_plans_by_default_and_applies_when_told(folders, tmp_path):
    dataset, review = folders
    base = ["--project", str(tmp_path / "p"), "add-reviewed", dataset]
    code, result = cli.run(base + [review, "--video", "newfilm", "--thin-under", "10",
                                   "--cap", "6"])
    assert code == 0 and result["take"] == 10 and "applied" not in result
    assert all("paths" not in row for row in result["folders"])
    assert intake.added_videos(dataset) == []
    code, result = cli.run(base + [review, "--video", "newfilm", "--thin-under", "10",
                                   "--cap", "6", "--apply"])
    assert code == 0 and result["applied"]["added"] == 10
    code, result = cli.run(base + ["--video", "newfilm", "--undo"])
    assert code == 0 and result["removed"] == 10
    code, result = cli.run(base + [review, "--video", "film1"])
    assert code == 2 and "already has" in result["error"]


# ---------------------------------------------------------------------------
# Did it help?
# ---------------------------------------------------------------------------

def _centroid_train(x, targets, n_classes, steps=0, seed=0):
    flat = x.mean(1)
    return np.stack([flat[targets[:, k] > 0].mean(0) if (targets[:, k] > 0).any()
                     else np.full(flat.shape[1], 1e3) for k in range(n_classes)])


def _centroid_predict(model, x):
    return -((x.mean(1)[:, None, :] - model[None]) ** 2).sum(-1)


def _by_video(strat, groups, folds, seed):
    names = sorted(set(groups.tolist()))
    rng = np.random.default_rng(seed)
    rng.shuffle(names)
    out = []
    for f in range(folds):
        held = np.isin(groups, names[f::folds])
        out.append((np.flatnonzero(~held), np.flatnonzero(held)))
    return out


def _problem(added_points, added_class, n_videos=10, per=12):
    """Two classes on a line (0 at -1, 1 at +1), ``n_videos`` videos, plus an
    added video whose clips sit at ``added_points`` and are filed as
    ``added_class``."""
    rng = np.random.default_rng(0)
    xs, ys, gs = [], [], []
    for v in range(n_videos):
        for i in range(per):
            k = i % 2
            xs.append([(-1.0, 1.0)[k] + rng.normal(0, 0.45), rng.normal(0, 0.1)])
            ys.append(k)
            gs.append(f"film{v}")
    for point in added_points:
        xs.append([point, 0.0])
        ys.append(added_class)
        gs.append("newfilm")
    x = np.repeat(np.array(xs, np.float32)[:, None, :], 4, axis=1)
    targets = np.eye(2, dtype=np.float32)[ys]
    groups = np.array(gs)
    return x, targets, groups, groups == "newfilm"


def _run(x, targets, groups, added):
    return C.compare_features(x, targets, groups, added, ["a", "b"], seeds=3, folds=5,
                              train_fn=_centroid_train, predict_fn=_centroid_predict,
                              folds_fn=_by_video, log=lambda _line: None)


def test_clips_that_teach_the_wrong_thing_measure_worse():
    # 300 clips filed as class b that sit deep in class a's side: b's centre
    # is pulled across, and the other videos pay for it.
    result = _run(*_problem([-3.0] * 300, added_class=1))
    assert result["verdict"] == C.WORSE and result["keep"] is False
    assert all(d < 0 for d in result["top1_change_per_seed"])
    assert result["clips"] == {"dataset": 120, "added": 300, "added_share": 0.7143}
    # Scored on the dataset's own clips only, every one exactly once.
    assert sum(r["held_out_clips"] for r in result["per_class"]) == 120
    assert result["summary"].startswith("Worse")


def test_a_few_ordinary_clips_are_no_clear_change():
    result = _run(*_problem([1.0] * 5, added_class=1))
    assert result["verdict"] == C.SAME and result["keep"] is True
    row = next(r for r in result["per_class"] if r["class"] == "b")
    assert row["added_clips"] == 5 and row["dataset_videos"] == 10


def test_a_change_needs_every_seed_and_a_real_size():
    assert C._direction([0.02, 0.03, 0.02], 0.01) == C.BETTER
    assert C._direction([-0.02, -0.03, -0.02], 0.01) == C.WORSE
    assert C._direction([0.04, -0.01, 0.04], 0.01) == C.SAME      # one seed disagrees
    assert C._direction([0.002, 0.003, 0.001], 0.01) == C.SAME    # too small to be one
    assert C._direction([-0.001, -0.009, -0.024], 0.01) == C.SAME  # one seed did it all


def test_nothing_to_compare_is_said():
    x, targets, groups, added = _problem([1.0] * 5, added_class=1)
    with pytest.raises(ValueError, match="none of the dataset"):
        _run(x, targets, groups, np.zeros(len(x), bool))


# ---------------------------------------------------------------------------
# Parts, dropping, health, and what the MCP tools send
# ---------------------------------------------------------------------------

def test_parts_are_scored_on_their_own_and_the_bad_one_is_named():
    x, targets, groups, added = _problem([-3.0] * 300 + [1.0] * 20, added_class=1)
    idx = np.flatnonzero(added)
    bad, fine = np.zeros(len(x), bool), np.zeros(len(x), bool)
    bad[idx[:300]], fine[idx[300:]] = True, True
    result = C.compare_features(x, targets, groups, added, ["a", "b"], seeds=3, folds=5,
                                parts={"bad": bad, "fine": fine}, train_fn=_centroid_train,
                                predict_fn=_centroid_predict, folds_fn=_by_video,
                                log=lambda _line: None)
    assert result["drop"] == ["bad"]
    by = {row["part"]: row for row in result["parts"]}
    assert by["bad"]["verdict"] == C.WORSE and by["bad"]["clips"] == 300
    assert by["fine"]["keep"] is True and "per_class" not in by["fine"]
    assert "bad" in result["summary"]


def test_drop_takes_out_one_folder_and_keeps_the_record(folders):
    dataset, review = folders
    intake.apply(intake.plan(dataset, review, "newfilm", thin_under=10, cap=6))
    out = intake.drop(dataset, "newfilm", ["fresh", "nowhere"])
    assert out["removed"] == 4 and out["in_dataset"] == 6 and out["not_found"] == ["nowhere"]
    assert not os.path.isdir(os.path.join(dataset, "train", "fresh"))
    record = intake.read_manifest(dataset, "newfilm")
    assert record["dropped"]["fresh"]["clips"] == 4 and len(record["added"]) == 6
    # Left out on request the next time, and undo still removes the rest.
    rows = _rows(intake.plan(dataset, review, "newfilm", thin_under=10, cap=6,
                             exclude=["fresh"]))
    assert rows["fresh"]["why"] == intake.EXCLUDED and rows["thin"]["take"] == 6
    assert intake.undo(dataset, "newfilm")["removed"] == 6


def test_dataset_health_names_what_is_thin_and_what_leaks(folders):
    from modules.teach import dataset_health as DH

    dataset, _review = folders
    _clip(os.path.join(dataset, "train", "big", "loose clip.mp4"))
    _clip(os.path.join(dataset, "val", "thin", "film0_temp_clip_v.mp4"))
    for i in range(5):
        _clip(os.path.join(dataset, "train", "solo", f"only_temp_clip_{i}.mp4"))
    report = DH.check(dataset)
    found = {f["what"]: f for f in report["findings"]}
    assert found["few_videos"]["items"] == ["solo"]
    assert found["one_video_floods_class"]["count"] == 1
    assert found["no_source_video"]["count"] == 1
    assert found["val_shares_videos_with_train"]["items"] == ["film0"]
    assert report["wanted"][0]["class"] == "solo" and report["wanted"][0]["videos_needed"] == 2
    assert report["level"] == DH.WARN


def test_near_copies_reports_only_pairs_that_matter():
    from modules.teach import dataset_health as DH

    def clip(name, folder, group):
        return benchmark.Clip(f"/d/{folder}/{name}.mp4", "train", folder, (folder,), group)

    clips = [clip("a1", "x", "v1"), clip("a2", "x", "v1"), clip("b1", "y", "v2"),
             clip("c1", "x", "v3"), clip("d1", "x", "v4")]
    v = np.array([[1, 0, 0], [1, 0.01, 0], [1, 0, 0.01], [1, 0.01, 0.01], [0, 1, 0]], float)
    out = DH.near_copies(v, clips)
    assert out["different_classes"]["count"] == 3        # b1 against a1, a2 and c1
    assert out["different_videos"]["count"] == 2         # c1 against a1 and a2
    assert all("d1" not in p["a"] + p["b"] for p in out["different_videos"]["pairs"])


def test_mcp_tools_send_codes_not_class_names(folders, monkeypatch):
    pytest.importorskip("mcp")
    from mcp_server import dataset_tools as DT

    dataset, review = folders
    monkeypatch.setattr(DT, "sends_names", lambda: False)
    health = DT.dataset_health(dataset)
    planned = DT.plan_reviewed_video(dataset, review, "newfilm", thin_under=10, cap=6)
    added = DT.add_reviewed_video(dataset, review, "newfilm", thin_under=10, cap=6)
    text = repr([health, planned, added])
    for name in ("big", "thin", "fresh"):
        assert f"'{name}'" not in text and f" {name}" not in text
    assert added["applied"]["added"] == 10 and "review" not in planned
    codes = __import__("json").load(open(planned["class_codes_file"], encoding="utf-8"))
    assert set(codes) == {"big", "thin", "fresh"}
    # The client acts by code; the tool finds the folder.
    out = DT.drop_reviewed_folders(dataset, "newfilm", [codes["fresh"]])
    assert out["removed"] == 4 and out["by_folder"] == {codes["fresh"]: 4}
    assert intake.added_videos(dataset) == ["newfilm"]    # the code table is not a video
    assert DT.undo_reviewed_video(dataset, "newfilm")["removed"] == 6
    # And with names turned on they are sent as they are.
    monkeypatch.setattr(DT, "sends_names", lambda: True)
    rows = DT.plan_reviewed_video(dataset, review, "newfilm", thin_under=10, cap=6)["folders"]
    assert {"big", "thin", "fresh"} == {r["folder"] for r in rows}


def test_the_window_previews_adds_and_takes_out(folders):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    import time

    from modules.teach.reviewed_dialog import ReviewedVideoDialog

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    dataset, review = folders
    dialog = ReviewedVideoDialog()

    def wait():
        deadline = time.time() + 10
        while dialog.runner.busy and time.time() < deadline:
            app.processEvents()
            time.sleep(0.01)
        app.processEvents()

    dialog.plan(False)                                    # nothing chosen yet
    assert "Choose" in dialog.status.text() and not dialog.runner.busy
    dialog.dataset.edit.setText(dataset)
    dialog.review.edit.setText(review)
    dialog.video.setText("newfilm")
    dialog.thin.setValue(10)
    dialog.cap.setValue(6)
    dialog.plan(False)
    wait()
    assert dialog.plan_table.rowCount() == 3 and dialog.add_btn.isEnabled()
    # The first class it takes is shown, crop by crop; another row shows its own.
    rows = dialog.plan_rows
    assert len(dialog.tiles.thumbs) == len(dialog.paths[rows[dialog.plan_table.currentRow()]][0])
    dialog.plan_table.selectRow(rows.index("thin"))
    app.processEvents()
    assert len(dialog.tiles.thumbs) == 6 and "6 of 24" in dialog.samples_title.text()
    dialog.plan_table.selectRow(rows.index("big"))
    app.processEvents()
    assert not dialog.tiles.thumbs and "dataset has enough" in dialog.samples_title.text()
    assert "Nothing has been changed" in dialog.status.text()
    assert intake.added_videos(dataset) == []
    dialog.plan(True)
    wait()
    assert "Added 10 crops" in dialog.status.text() and not dialog.add_btn.isEnabled()
    # A comparison that names a folder to drop arms the button; dropping uses it.
    dialog.show_comparison({
        "summary": "Worse.", "keep": False, "per_class": [], "drop_folders": ["fresh"],
        "parts": [{"part": "p", "clips": 4, "top1_change": -0.02, "verdict": "worse",
                   "keep": False}],
        "top1_without": {"mean": 0.5}, "top1_with": {"mean": 0.48}, "how": {"seeds": 5}})
    assert dialog.drop_btn.isEnabled() and dialog.parts_table.rowCount() == 1
    dialog.drop()
    wait()
    assert "Took out 4 crops" in dialog.status.text() and not dialog.drop_btn.isEnabled()
    dialog.health()
    wait()
    assert dialog.health_table.rowCount() >= 1
    dialog.undo()
    wait()
    assert intake.added_videos(dataset) == []
    dialog.runner.wait()
