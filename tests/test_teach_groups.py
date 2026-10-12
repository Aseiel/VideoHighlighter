"""Footage first, names after: grouping, naming groups, health and sorting accuracy.

Same fakes as ``test_teach.py``: a sample's "frames" say which thing it truly
shows, so groups and guesses can be checked against known answers.
"""
from __future__ import annotations

import os

import pytest

from modules.teach import cli, group, health, review, sort, status
from modules.teach.project import (
    ACCEPTED, ACTIONS, MIXED, NAMED, NEGATIVE, NONE, NOTHING, OPEN, PENDING, SPLIT,
    Project,
)
from tests.test_teach import FakeEmbedder, _add_samples, _reader


@pytest.fixture
def bare(tmp_path):
    """A project with footage and no classes: the group-first start."""
    p = Project.create(str(tmp_path / "proj"), ACTIONS)
    truth = _add_samples(p, [0] * 14 + [1] * 10 + [-1] * 8)
    return p, truth


def _group(p, truth, **kw):
    return group.group_samples(p, FakeEmbedder(), frame_reader=_reader(truth), **kw)


def _truth_of(p, truth, sid):
    return truth[os.path.basename(p.get_sample(sid).path)]


def test_status_asks_to_group_then_to_name_when_nothing_is_named(bare):
    p, truth = bare
    assert status.next_step(p)["args"] == ["group"]
    _group(p, truth, k=3)
    step = status.next_step(p)
    assert step["who"] == "judge" and step["args"][0] == "name-group"


def test_groups_follow_what_samples_show_tightest_and_most_typical_first(bare):
    p, truth = bare
    result = _group(p, truth, k=3)
    assert result["grouped"] == 32 and len(p.groups) == 3
    for g in p.groups:
        kinds = {_truth_of(p, truth, sid) for sid in g.members}
        assert len(kinds) == 1                      # each group is one thing
    assert [g.tightness for g in p.groups] == sorted(
        (g.tightness for g in p.groups), reverse=True)
    assert all(s.group > 0 for s in p.samples)


def test_tiny_outlier_groups_come_last(tmp_path):
    p = Project.create(str(tmp_path / "proj"), ACTIONS)
    truth = _add_samples(p, [0] * 10 + [1])
    group.group_samples(p, FakeEmbedder(), k=2, frame_reader=_reader(truth))
    assert [len(g.members) for g in p.groups] == [10, 1]


def test_naming_decides_only_the_shown_tiles_and_not_the_wrong_ones(bare):
    p, truth = bare
    _group(p, truth, k=3)
    g = next(g for g in p.groups if _truth_of(p, truth, g.members[0]) == 0)
    out = group.name_group(p, g.id, name="alpha move", wrong="2", shown=5)
    assert out["class_added"] == "alpha move" and out["decided"] == 4
    shown = [p.get_sample(sid) for sid in g.shown]
    assert [s.verdict for s in shown] == [ACCEPTED, PENDING, ACCEPTED, ACCEPTED, ACCEPTED]
    assert shown[0].is_human and shown[0].shown_guess == "alpha move"
    assert shown[0].guess_from == "group"
    # Members not shown are left for sort and review.
    unseen = [p.get_sample(sid) for sid in g.members if sid not in g.shown]
    assert unseen and all(s.verdict == PENDING for s in unseen)
    assert g.state == NAMED and g.wrong == [g.shown[1]]


def test_nothing_makes_negatives_and_a_background_class(bare):
    p, truth = bare
    _group(p, truth, k=3)
    g = next(g for g in p.groups if _truth_of(p, truth, g.members[0]) == -1)
    out = group.name_group(p, g.id, nothing=True)
    assert out["background_class"] == group.BACKGROUND
    assert p.settings.background_class == group.BACKGROUND
    assert all(p.get_sample(sid).verdict == NEGATIVE for sid in g.shown)
    assert g.state == NOTHING


def test_reopen_undoes_and_drops_a_class_nothing_else_holds(bare):
    p, truth = bare
    _group(p, truth, k=3)
    g = p.groups[0]
    group.name_group(p, g.id, name="alpha move")
    out = group.name_group(p, g.id, reopen=True)
    assert out["undone"] > 0 and out["class_removed"] is True
    assert p.class_names() == [] and g.state == OPEN
    assert all(p.get_sample(sid).verdict == PENDING for sid in g.members)


def test_renaming_a_group_moves_its_samples(bare):
    p, truth = bare
    _group(p, truth, k=3)
    g = p.groups[0]
    group.name_group(p, g.id, name="alpha move")
    group.name_group(p, g.id, name="beta move")
    assert p.class_names() == ["beta move"]
    assert {p.get_sample(sid).label for sid in g.shown} == {"beta move"}


def test_mixed_groups_split_into_smaller_ones(bare):
    p, truth = bare
    _group(p, truth, k=1)
    only = p.groups[0]
    group.name_group(p, only.id, mixed=True)
    assert only.state == MIXED
    out = _group(p, truth, split=only.id, k=3)
    assert out["split"] == only.id and only.state == SPLIT
    children = [g for g in p.groups if g.parent == only.id]
    assert len(children) == 3
    with pytest.raises(ValueError):
        group.name_group(p, only.id, name="alpha move")


def test_regrouping_keeps_decided_groups_and_replaces_open_ones(bare):
    p, truth = bare
    _group(p, truth, k=3)
    named = p.groups[0]
    group.name_group(p, named.id, name="alpha move", shown=len(named.members))
    _group(p, truth, k=2)
    assert named in p.groups and named.state == NAMED
    assert [g.state for g in p.groups if g is not named] == [OPEN, OPEN]
    regrouped = {sid for g in p.groups if g is not named for sid in g.members}
    assert not regrouped & set(named.members)


def test_all_groups_named_then_sort_and_review_take_over(bare):
    p, truth = bare
    _group(p, truth, k=3)
    for g in p.groups:
        kind = _truth_of(p, truth, g.members[0])
        if kind < 0:
            group.name_group(p, g.id, nothing=True, shown=4)
        else:
            group.name_group(p, g.id, name=["alpha move", "beta move"][kind], shown=4)
    assert status.next_step(p)["args"] == ["sort"]
    sort.sort_project(p, FakeEmbedder(), frame_reader=_reader(truth))
    assert status.next_step(p)["args"][0] == "review"


def test_review_records_the_guess_and_sorting_is_measured(bare):
    p, truth = bare
    _group(p, truth, k=3)
    for g in p.groups:
        kind = _truth_of(p, truth, g.members[0])
        if kind >= 0:
            group.name_group(p, g.id, name=["alpha move", "beta move"][kind],
                             wrong="1", shown=4)
        else:
            group.name_group(p, g.id, nothing=True, shown=4)
    p.settings.auto_accept = False
    sort.sort_project(p, FakeEmbedder(), frame_reader=_reader(truth))
    sheet = review.next_sheet(p, size=12, frame_reader=lambda *a, **k: [],
                              renderer=lambda *a, **k: None)
    names = ["alpha move", "beta move"]
    relabel, accept = [], []
    for item in sheet["items"]:
        real = truth[os.path.basename(p.get_sample(item["sample"]).path)]
        actual = names[real] if real >= 0 else NONE
        if item["proposed"] == actual:
            accept.append(str(item["n"]))
        elif actual != NONE:
            relabel.append(f"{item['n']}={actual}")
    # One deliberate "the sort was wrong": the first tile is said to be the
    # other class.
    first = sheet["items"][0]
    other = [n for n in names if n != first["proposed"]][0]
    accept = [a for a in accept if a != "1"]
    relabel = [r for r in relabel if not r.startswith("1=")] + [f"1={other}"]
    out = review.apply_verdicts(p, sheet["sheet"], accept=",".join(accept),
                                relabel=relabel)
    assert not out["errors"]
    s = p.get_sample(first["sample"])
    assert s.shown_guess == first["proposed"] and s.guess_from == "sort"

    found = health.sorting(p)
    # Two named groups had tile 1 marked wrong: 10 of 12 shown were right.
    assert found["groups"]["wrong"] == 2 and found["groups"]["shown"] == 12
    assert found["by_source"]["group"]["accuracy"] == round(10 / 12, 3)
    sorted_ = found["by_source"]["sort"]
    assert sorted_["wrong"] >= 1 and sorted_["accuracy"] < 1.0
    assert found["by_class"][first["proposed"]]["really"].get(other, 0) >= 1


def test_health_one_video_needs_more_videos_and_open_groups_block(bare):
    p, truth = bare
    _group(p, truth, k=3)
    report = health.report(p)
    assert report["verdict"] == health.NOT_READY
    assert any(c["name"] == "groups named" and c["level"] == health.BAD
               for c in report["checks"])
    for g in p.groups:
        kind = _truth_of(p, truth, g.members[0])
        if kind < 0:
            group.name_group(p, g.id, nothing=True, shown=len(g.members))
        else:
            group.name_group(p, g.id, name=["alpha move", "beta move"][kind],
                             shown=len(g.members))
    report = health.report(p)
    assert report["verdict"] == health.NEEDS_VIDEOS
    videos = [c for c in report["checks"] if c["name"] == "videos"]
    assert videos and all(c["level"] == health.WARN for c in videos)
    assert "2 more videos" in videos[0]["fix"]
    sizes = {c["class"]: c["level"] for c in report["checks"] if c["name"] == "size"}
    assert sizes == {"alpha move": health.WARN, "beta move": health.WARN}


def test_cli_group_name_health_and_projects(tmp_path, monkeypatch):
    root = str(tmp_path / "projects" / "mine")
    p = Project.create(root, ACTIONS)
    truth = _add_samples(p, [0] * 8 + [1] * 8)
    monkeypatch.setattr(cli, "make_embedder", lambda name="clip": FakeEmbedder())
    from modules.teach import embed
    monkeypatch.setattr(embed, "read_frames", _reader(truth))

    code, out = cli.run(["--project", root, "group", "--k", "2"])
    assert code == 0 and len(out["groups"]) == 2
    assert out["next"]["args"][0] == "name-group"
    first = out["groups"][0]["id"]
    code, out = cli.run(["--project", root, "name-group", str(first), "--name",
                         "alpha move", "--wrong", "2"])
    assert code == 0 and out["decided"] >= 1
    code, out = cli.run(["--project", root, "name-group", str(first)])
    assert code == 2                                # one of --name/--nothing/...
    code, out = cli.run(["--project", root, "health"])
    assert code == 0 and out["verdict"] == health.NOT_READY and "next" in out
    code, out = cli.run(["--project", str(tmp_path / "projects"), "projects"])
    assert code == 0 and [x["name"] for x in out["projects"]] == ["mine"]
    assert out["projects"][0]["health"] == health.NOT_READY


# --- crops as samples -----------------------------------------------------------

def _two_people_cropper(p):
    """Writes a left and a right crop for every sample in the inbox."""
    def cropper(inbox, out):
        for name in os.listdir(inbox):
            stem = name[:-4]
            for side in ("left", "right"):
                open(os.path.join(out, f"{stem}_cropped_{side}.mp4"), "wb").close()
    return cropper


def test_each_crop_becomes_its_own_sample_and_open_groups_are_regrouped(bare):
    from modules.teach import focus
    p, truth = bare
    _group(p, truth, k=3)
    named = p.groups[0]
    decided = p.get_sample(named.members[0])
    group.name_group(p, named.id, name="alpha move", shown=1)
    p.settings.focus = True
    out = focus.focus_project(p, cropper=_two_people_cropper(p))
    assert out["split"] == 31 and out["crops"] == 62
    # The decided sample keeps its crops attached; the rest are replaced.
    assert decided in p.samples and len(decided.focus_paths) == 2
    crops = [s for s in p.samples if s.crop_of]
    assert len(crops) == 62 and all(s.focus_tried and s.verdict == PENDING for s in crops)
    assert crops[0].id.endswith("_cropped_left") and crops[0].path.endswith(".mp4")
    # Open groups went with their whole samples; the named one stayed.
    assert [g.id for g in p.groups] == [named.id]
    again = Project.load(p.root)
    assert len(again.replaced) == 31 and len(again.samples) == 63
    assert status.next_step(again)["args"][0] == "sort"      # a class exists now


def test_with_nothing_named_crops_are_grouped_next(bare):
    from modules.teach import focus
    p, truth = bare
    _group(p, truth, k=3)
    p.settings.focus = True
    assert status.next_step(p)["args"] == ["focus"]
    focus.focus_project(p, cropper=_two_people_cropper(p))
    assert p.groups == [] and status.next_step(p)["args"] == ["group"]
    crop_truth = {os.path.basename(s.path): truth[s.crop_of + ".mp4"] for s in p.samples}
    group.group_samples(p, FakeEmbedder(), k=3, frame_reader=_reader(crop_truth))
    assert sum(len(g.members) for g in p.groups) == 64


def test_a_shot_kept_whole_gets_its_own_id(bare):
    from modules.teach import focus
    p, truth = bare
    p.settings.focus = True

    def cropper(inbox, out):
        for name in os.listdir(inbox):
            open(os.path.join(out, name), "wb").close()      # one person: copied
    focus.focus_project(p, cropper=cropper)
    assert all(s.id == s.crop_of + "_whole" for s in p.samples)
    assert {s.id for s in p.replaced}.isdisjoint(s.id for s in p.samples)


# --- moving by hand -------------------------------------------------------------

def test_move_to_a_new_class_decides_and_records_the_guess(bare):
    p, truth = bare
    ids = [s.id for s in p.samples[:3]]
    p.samples[0].proposed = NONE
    out = group.move_samples(p, ids, to_class="beta move")
    assert out["class_added"] == "beta move" and out["moved"] == 3
    s = p.get_sample(ids[0])
    assert s.verdict == ACCEPTED and s.label == "beta move" and s.is_human
    assert s.shown_guess == NONE and s.guess_from == "sort"
    assert health.sorting(p)["by_source"]["sort"]["wrong"] == 1


def test_moving_a_group_decided_sample_counts_against_the_group(bare):
    p, truth = bare
    _group(p, truth, k=3)
    g = p.groups[0]
    group.name_group(p, g.id, name="alpha move", shown=4)
    group.move_samples(p, [g.shown[0]], reject=True)
    assert g.wrong == [g.shown[0]]
    assert health.sorting(p)["groups"]["wrong"] == 1
    group.move_samples(p, [g.shown[0]], undecided=True)
    assert p.get_sample(g.shown[0]).verdict == PENDING


def test_move_between_groups_and_into_a_new_one(bare):
    p, truth = bare
    _group(p, truth, k=3)
    a, b = p.groups[0], p.groups[1]
    moving = a.members[:2]
    group.move_samples(p, moving, to_group=b.id)
    assert not set(moving) & set(a.members) and set(moving) <= set(b.members)
    out = group.move_samples(p, moving, new_group=True)
    fresh = p.get_group(out["group_added"])
    assert fresh.members == moving and fresh.state == OPEN
    assert all(p.get_sample(i).group == fresh.id for i in moving)


def test_cli_move_by_group_tiles(tmp_path, monkeypatch):
    root = str(tmp_path / "mine")
    p = Project.create(root, ACTIONS)
    truth = _add_samples(p, [0] * 6 + [1] * 6)
    monkeypatch.setattr(cli, "make_embedder", lambda name="clip": FakeEmbedder())
    from modules.teach import embed
    monkeypatch.setattr(embed, "read_frames", _reader(truth))
    code, out = cli.run(["--project", root, "group", "--k", "2"])
    first = out["groups"][0]["id"]
    code, out = cli.run(["--project", root, "move", "--group", str(first), "--tiles",
                         "1-2", "--nothing"])
    assert code == 0 and out["moved"] == 2 and out["verdict"] == NEGATIVE
    code, out = cli.run(["--project", root, "move", "--samples", "nope", "--reject"])
    assert code == 2
