"""Tests for modules/teach: the pipeline from "find this" to a trained model.

No CLIP, no video decoding, no GPU. Samples are stand-ins whose "frames" say
which class they truly show, and a fake embedder turns that into a vector near
that class's direction, so sorting, reviewing, splitting and the status
machine are exercised on known answers. ``test_teach_e2e.py`` runs the real
decode and ffmpeg path wherever OpenCV and imageio-ffmpeg are installed.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from modules.teach import (
    boxes, build, cli, cut, naming, review, scoring, sort, status, train,
)
from modules.teach.project import (
    ACCEPTED, ACTIONS, NEGATIVE, NONE, OBJECTS, PENDING, REJECTED, UNSURE, VAL,
    Project, Sample,
)

DIM = 16


# --- fakes --------------------------------------------------------------------

def _direction(k: int) -> np.ndarray:
    v = np.zeros(DIM, np.float32)
    v[k] = 1.0
    return v


class FakeEmbedder:
    """Image "frames" are ``("truth", k)`` tuples -> a vector near axis ``k``.
    ``k = -1`` is background: near a shared background axis."""

    model_id = "fake"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def images(self, frames):
        out = []
        for frame in frames:
            k = frame[1] if isinstance(frame, tuple) else -1
            base = _direction(DIM - 1) if k < 0 else _direction(k) + 0.3 * _direction(DIM - 1)
            out.append(base + 0.05 * self.rng.normal(size=DIM))
        v = np.array(out, np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def texts(self, texts):
        # Words point loosely at the class they name (index = order of creation
        # in these tests), which is roughly how weak CLIP text is.
        out = []
        for t in texts:
            k = 0 if "alpha" in t else 1 if "beta" in t else 2
            out.append(_direction(k) + 0.8 * _direction(DIM - 1)
                       + 0.1 * self.rng.normal(size=DIM))
        v = np.array(out, np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def _reader(truth: dict):
    """frame_reader: each sample path yields frames tagged with its truth."""
    def read(path, count, **_):
        return [("truth", truth.get(os.path.basename(path), -1))] * count
    return read


@pytest.fixture
def project(tmp_path):
    p = Project.create(str(tmp_path / "proj"), ACTIONS)
    p.add_class("alpha move", "the first thing")
    p.add_class("beta move", "the second thing")
    p.save()
    return p


def _add_samples(p, layout):
    """layout: list of truth indices per 5 s sample of one source."""
    source = p.add_source(os.path.join(p.root, "v.mp4"))
    source.cut = True
    truth = {}
    os.makedirs(p.path("samples"), exist_ok=True)
    for i, k in enumerate(layout):
        sid = cut.sample_id(source.id, i * 5.0)
        path = p.path("samples", sid + ".mp4")
        with open(path, "wb") as handle:
            handle.write(b"clip")
        p.samples.append(Sample(id=sid, source=source.id, path=path, start=i * 5.0,
                                duration=5.0))
        truth[sid + ".mp4"] = k
    p.save()
    return truth


# --- naming -------------------------------------------------------------------

def test_names_are_normalised_to_the_stock_style():
    assert naming.normalize_name("  Kick_Flip-Fast ") == "kick flip fast"


@pytest.mark.parametrize("name, code", [
    ("", "empty"), ("Upper", "format"), ("_mine", "reserved"),
    ("a/b", "path"), ("123", "numeric"), ("x" * 41, "long"),
])
def test_names_that_cannot_be_a_class_are_blocked(name, code):
    problems = naming.check_name(name, [])
    assert any(p.code == code and p.blocking for p in problems)


def test_advice_does_not_block():
    problems = naming.check_name("thing", [], vocabulary=[])
    assert [p.code for p in problems] == ["vague"]
    assert not any(p.blocking for p in problems)


def test_duplicates_and_near_duplicates():
    assert any(p.code == "duplicate" and p.blocking
               for p in naming.check_name("flip", ["flip"], vocabulary=[]))
    near = naming.check_name("flips", ["flip"], vocabulary=[])
    assert [p.code for p in near] == ["near_duplicate"]


def test_objects_are_singular_and_stock_names_are_recognised():
    assert any(p.code == "plural" for p in naming.check_name("boxes", [], "objects",
                                                             vocabulary=[]))
    assert any(p.code == "stock" for p in naming.check_name("person", [], "objects"))


def test_the_stock_vocabularies_load():
    assert "person" in naming.load_vocabulary("objects")
    assert len(naming.load_vocabulary("actions")) == 400


def test_suggestions_rank_the_label_that_describes_the_examples():
    rng = np.random.default_rng(1)
    labels = [f"label {i}" for i in range(40)]
    label_vectors = rng.normal(size=(40, DIM)).astype(np.float32)
    examples = np.stack([label_vectors[7] + 0.1 * rng.normal(size=DIM) for _ in range(5)])
    result = naming.suggest_names(examples, labels, label_vectors, top_k=3)
    assert result["suggestions"][0]["name"] == "label 7"
    assert result["suggestions"][0]["fit"] == "good"
    assert result["consistency"] > 0.9
    assert result["split"] is None


def test_examples_of_two_things_are_flagged_for_splitting():
    examples = np.stack([_direction(0)] * 3 + [_direction(1)] * 3) + 0.01
    groups = naming.split_hint(examples)
    assert sorted(map(sorted, groups)) == [[0, 1, 2], [3, 4, 5]]


# --- scoring ------------------------------------------------------------------

def test_example_prototypes_propose_and_negatives_compete():
    emb = FakeEmbedder()
    a = scoring.build_prototype("a", emb.images([("t", 0)] * 3))
    b = scoring.build_prototype("b", emb.images([("t", 1)] * 3))
    samples = emb.images([("t", 0), ("t", 1), ("t", -1), ("t", -1), ("t", -1), ("t", -1)])
    out = scoring.score_samples(list("uvwxyz"), samples, [a, b], gate=0.5,
                                margin=0.1, floor=0.2)
    assert out["u"][1] == "a" and out["v"][1] == "b"
    assert out["w"][1] == NONE

    none = scoring.build_prototype(NONE, emb.images([("t", -1)] * 3))
    out = scoring.score_samples(list("uvwxyz"), samples, [a, b, none], gate=0.5,
                                margin=0.1, floor=0.2)
    assert out["w"][1] == NONE
    assert NONE not in out["u"][0]          # not reported as a class score


def test_two_classes_that_both_fit_are_unsure():
    row = np.array([0.9, 0.85], np.float32)
    assert scoring.propose(row, ["a", "b"], gate=0.5, margin=0.1, floor=0.2)[0] == UNSURE


# --- project ------------------------------------------------------------------

def test_project_round_trips_and_renames_everywhere(project):
    _add_samples(project, [0, 1])
    s = project.samples[0]
    project.decide(s, ACCEPTED, "alpha move")
    s.scores = {"alpha move": 1.0}
    project.rename_class("alpha move", "alpha step")
    project.save()
    again = Project.load(project.root)
    assert again.class_names() == ["alpha step", "beta move"]
    assert again.samples[0].label == "alpha step"
    assert "alpha step" in again.samples[0].scores


def test_accepting_needs_a_real_class(project):
    _add_samples(project, [0])
    with pytest.raises(ValueError):
        project.decide(project.samples[0], ACCEPTED, "nope")


# --- cut ----------------------------------------------------------------------

def test_segments_cover_the_video_and_drop_a_short_tail():
    assert cut.plan_segments(17.0, 5.0) == [(0.0, 5.0), (5.0, 5.0), (10.0, 5.0)]
    assert cut.plan_segments(18.0, 5.0)[-1] == (15.0, 3.0)
    assert len(cut.plan_segments(10.0, 5.0, stride=2.5)) == 3


def test_cutting_is_resumable_and_records_failures(project, tmp_path):
    src = tmp_path / "long.mp4"
    src.write_bytes(b"x")
    project.add_source(str(src))
    calls = []

    class R:
        returncode = 0
        stderr = ""

    def run(cmd, **kw):
        calls.append(cmd)
        if "00005000" in cmd[-1]:
            r = R()
            r.returncode, r.stderr = 1, "boom"
            return r
        open(cmd[-1], "wb").close()
        return R()

    result = cut.cut_project(project, run=run, duration_of=lambda p: 15.0)
    assert result["samples_made"] == 2 and len(result["failed"]) == 1
    assert not project.sources[0].cut            # retried next time

    result = cut.cut_project(project, run=lambda cmd, **kw: (open(cmd[-1], "wb").close(), R())[1],
                             duration_of=lambda p: 15.0)
    assert result["samples_made"] == 1 and project.sources[0].cut
    assert len(project.samples) == 3


def test_cuts_are_frame_accurate_reencodes():
    cmd = cut.cut_command("in.mp4", "out.mp4", 12.5, 5.0)
    assert cmd[cmd.index("-ss") + 1] == "12.500" and "libx264" in cmd
    assert cmd.index("-ss") < cmd.index("-i")


# --- sort and folders -----------------------------------------------------------

def test_sorting_proposes_from_words_then_sharpens_with_examples(project):
    truth = _add_samples(project, [0, 1, -1, 0, -1, 1, -1, -1, 0, -1])
    emb = FakeEmbedder()
    result = sort.sort_project(project, emb, frame_reader=_reader(truth))
    assert result["prototypes"]["alpha move"]["from"] == "text"

    project.decide(project.samples[0], ACCEPTED, "alpha move")
    project.decide(project.samples[1], ACCEPTED, "beta move")
    project.decide(project.samples[2], NEGATIVE)
    result = sort.sort_project(project, emb, frame_reader=_reader(truth))
    assert result["prototypes"]["alpha move"]["from"] == "examples"
    assert NONE in result["prototypes"]
    by_id = {s.id: s for s in project.samples}
    for sid_file, k in truth.items():
        s = by_id[sid_file[:-4]]
        if s.verdict == PENDING:
            expected = {0: "alpha move", 1: "beta move", -1: NONE}[k]
            assert s.proposed == expected, (s.id, s.scores)


def test_folders_moved_by_hand_become_verdicts(project):
    truth = _add_samples(project, [0, 1, -1, 0])
    sort.sort_project(project, FakeEmbedder(), frame_reader=_reader(truth))
    for s in project.samples:
        s.proposed = "alpha move"          # everything guessed as alpha
    sort.lay_out_folders(project)
    root = project.path("sorted")
    ids = [s.id for s in project.samples]
    os.replace(os.path.join(root, "alpha move", ids[1] + ".mp4"),
               os.path.join(root, "beta move", ids[1] + ".mp4"))
    os.replace(os.path.join(root, "alpha move", ids[2] + ".mp4"),
               os.path.join(root, NONE, ids[2] + ".mp4"))

    result = review.from_folders(project)            # nothing confirmed yet
    assert result["decisions"] == {"beta move": 1, NEGATIVE: 1}
    assert project.get_sample(ids[0]).verdict == PENDING

    result = review.from_folders(project, ["alpha move"])
    assert project.get_sample(ids[0]).label == "alpha move"
    assert project.get_sample(ids[3]).label == "alpha move"


# --- review ---------------------------------------------------------------------

def _scored(project, rows):
    """rows: (proposed, margin, model_proposed)"""
    _add_samples(project, [0] * len(rows))
    for s, (proposed, margin, model) in zip(project.samples, rows):
        s.proposed, s.margin, s.model_proposed = proposed, margin, model
        s.scores = {"alpha move": 0.6, "beta move": 0.4}
    project.save()


def test_disagreements_come_first_and_confident_guesses_are_never_skipped(project):
    rows = [("alpha move", 0.9, "")] * 20 + [("beta move", 0.2, "alpha move")] \
        + [(UNSURE, 0.0, "")] * 10 + [(NONE, 0.5, "")] * 5
    _scored(project, rows)
    batch = review.pick_batch(project, size=10)
    assert len(batch) == 10
    assert batch[0].model_proposed == "alpha move"
    assert any(s.proposed == NONE for s in batch)
    assert any(s.proposed == "alpha move" and s.margin == 0.9 for s in batch)


def test_a_sheet_and_its_verdicts(project):
    _scored(project, [("alpha move", 0.5, "")] * 3 + [(UNSURE, 0.0, "")] * 2
            + [(NONE, 0.3, "")])
    drawn = {}
    record = review.next_sheet(project, size=6, frame_reader=lambda *a, **k: [],
                               renderer=lambda tiles, caps, cols, path, header="":
                               drawn.setdefault("caps", caps))
    assert record["sheet"] == 1 and len(drawn["caps"]) == 6
    by_proposal = {}
    for item in record["items"]:
        by_proposal.setdefault(item["proposed"], []).append(item["n"])
    unsure = by_proposal[UNSURE]

    # Accepting an unsure guess is refused: it needs a class.
    bad = review.apply_verdicts(project, 1, accept=str(unsure[0]))
    assert bad["applied"] == 0 and bad["errors"]
    bad = review.apply_verdicts(project, 1, accept="1", reject="1")
    assert "two verdicts" in bad["errors"][0]

    alpha = by_proposal["alpha move"]
    result = review.apply_verdicts(project, 1, reject=str(alpha[0]),
                                   relabel=[f"{unsure[0]}=beta move"], accept_rest=True)
    assert result["errors"] == []
    verdicts = {s.id: (s.verdict, s.label) for s in project.samples}
    items = {i["n"]: i["sample"] for i in record["items"]}
    assert verdicts[items[alpha[0]]] == (REJECTED, "")
    assert verdicts[items[alpha[1]]] == (ACCEPTED, "alpha move")
    assert verdicts[items[unsure[0]]] == (ACCEPTED, "beta move")
    assert verdicts[items[unsure[1]]] == (PENDING, "")        # rest: unsure stays
    assert verdicts[items[by_proposal[NONE][0]]] == (NEGATIVE, "")


def test_ranges_parse():
    assert review.parse_numbers("1-3, 7 9") == [1, 2, 3, 7, 9]


# --- build ----------------------------------------------------------------------

def test_the_held_out_set_is_frozen_and_every_class_gets_enough(project):
    truth = _add_samples(project, [0] * 30 + [1] * 12)
    for s in project.samples:
        k = truth[s.id + ".mp4"]
        project.decide(s, ACCEPTED, "alpha move" if k == 0 else "beta move")
    build.assign_splits(project)
    val = {s.id for s in project.samples if s.split == VAL}
    for name in project.class_names():
        members = project.accepted(name)
        n_val = sum(1 for s in members if s.split == VAL)
        assert n_val >= max(2, 0.2 * len(members))

    # New footage later: the old split does not move.
    _add_samples(project, [0] * 10)
    for s in project.samples:
        if not s.is_decided:
            project.decide(s, ACCEPTED, "alpha move")
    build.assign_splits(project)
    assert val <= {s.id for s in project.samples if s.split == VAL}


def test_neighbouring_samples_land_on_the_same_side(project):
    _add_samples(project, [0] * 20)
    for s in project.samples:
        project.decide(s, ACCEPTED, "alpha move")
    build.assign_splits(project)
    splits = [s.split for s in sorted(project.samples, key=lambda s: s.start)]
    changes = sum(1 for a, b in zip(splits, splits[1:]) if a != b)
    assert changes <= 2          # contiguous runs, not a salt-and-pepper split


def test_build_lays_out_what_the_trainers_read(project):
    _add_samples(project, [0] * 6 + [1] * 6)
    for s in project.samples:
        project.decide(s, ACCEPTED, "alpha move" if s.start < 30 else "beta move")
    result = build.build(project)
    root = project.path("dataset")
    for split in ("train", "val"):
        for name in project.class_names():
            assert os.listdir(os.path.join(root, split, name))
    assert build.built_signature(project) == result["signature"]
    project.decide(project.samples[0], REJECTED)
    assert build.dataset_signature(project) != result["signature"]


# --- train ----------------------------------------------------------------------

def _ready(project):
    _add_samples(project, [0] * 6 + [1] * 6)
    for s in project.samples:
        project.decide(s, ACCEPTED, "alpha move" if s.start < 30 else "beta move")
    build.build(project)


def _fake_trainer(score):
    def run(cmd, **kw):
        out = cmd[cmd.index("--metrics-out") + 1]
        with open(out, "w") as fh:
            json.dump({"balanced_accuracy": score, "weights": "w.pth",
                       "mapping": "w_mapping.json",
                       "per_class_accuracy": {"alpha move": score}}, fh)

        class R:
            returncode = 0
        return R()
    return run


def test_a_round_is_installed_only_if_it_beats_the_last(project, monkeypatch):
    _ready(project)
    installs = []
    monkeypatch.setattr(train, "install",
                        lambda p, record: installs.append(record["round"]) or
                        record.update(installed=True) or {"slot": "test"})

    first = train.train_round(project, epochs=1, run=_fake_trainer(0.6))
    assert first["installed"] and installs == [1]
    worse = train.train_round(project, epochs=1, run=_fake_trainer(0.5))
    assert not worse["installed"] and worse["kept_previous"] == 1
    better = train.train_round(project, epochs=1, run=_fake_trainer(0.8))
    assert better["installed"] and installs == [1, 3]
    assert project.rounds[-1]["dataset"] == build.built_signature(project)


def test_a_failed_training_run_says_where_the_log_is(project):
    _ready(project)

    class R:
        returncode = 3

    with pytest.raises(RuntimeError, match="train.log"):
        train.train_round(project, epochs=1, run=lambda cmd, **kw: R())


def test_actions_train_into_the_round_folder_not_over_the_installed_model():
    cmd = train.actions_command("/p/dataset", "/p/runs/001", 5)
    assert cmd[cmd.index("--model-save-path") + 1].startswith("/p/runs/001")
    assert "--metrics-out" in cmd


def test_better_means_higher_accuracy_or_lower_loss():
    assert train.is_better({"balanced_accuracy": 0.7}, {"balanced_accuracy": 0.6})
    assert not train.is_better({"best_val_loss": 2.0}, {"best_val_loss": 1.5})
    assert train.is_better({"best_val_loss": 1.0}, None)


# --- status: the path an agent follows ------------------------------------------------

def test_status_walks_the_whole_way(tmp_path, monkeypatch):
    p = Project.create(str(tmp_path / "p"), ACTIONS)
    step = lambda: status.next_step(Project.load(p.root))   # noqa: E731
    assert "add-class" in step()["command"]
    p.add_class("alpha move")
    p.add_class("beta move")
    p.save()
    assert "add-video" in step()["command"]
    p.add_source(str(tmp_path / "v.mp4"))
    p.save()
    assert step()["command"].endswith(" cut")

    p = Project.load(p.root)
    p.sources = []
    for spec in p.classes:
        spec.target = 20
    p.save()
    truth = _add_samples(p, [0] * 30 + [1] * 30 + [-1] * 10)
    assert step()["command"].endswith(" sort")
    sort.sort_project(p, FakeEmbedder(), frame_reader=_reader(truth))
    assert step()["who"] == "judge" and step()["command"].endswith(" review")

    for s in p.samples:
        if s.proposed in p.class_names():
            p.decide(s, ACCEPTED, s.proposed)
    p.save()
    assert step()["command"].endswith(" build")
    build.build(p)
    assert step()["command"].endswith(" train")
    monkeypatch.setattr(train, "install", lambda proj, record: {"slot": "test"})
    train.train_round(p, epochs=1, run=_fake_trainer(0.7))
    assert "add-video" in step()["command"]


# --- boxes (object projects) --------------------------------------------------------------

class _Det:
    def __init__(self, name, box, conf=0.9):
        self.class_name, self.confidence = name, conf
        self.class_id = 0
        self.x1, self.y1, self.x2, self.y2 = box


class FakeDetector:
    def detect(self, frame):
        return [_Det("person", (10, 10, 40, 60)), _Det("cup", (60, 20, 90, 50), 0.4)]


class CropEmbedder(FakeEmbedder):
    """Crops from x >= 50 look like class 0 ("alpha"); the rest like background."""

    def images(self, frames):
        tagged = [("t", 0) if isinstance(f, np.ndarray) and f.mean() > 100 else ("t", -1)
                  for f in frames]
        return super().images(tagged)


def _object_project(tmp_path):
    p = Project.create(str(tmp_path / "obj"), OBJECTS)
    p.add_class("alpha widget")
    p.add_class("person")
    _add_samples(p, [0, 0, -1])
    p.decide(p.samples[0], ACCEPTED, "alpha widget")
    p.decide(p.samples[1], ACCEPTED, "person")
    p.decide(p.samples[2], NEGATIVE)
    p.settings.boxes_per_sample = 1
    p.save()
    return p


def _frame(*_):
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[:, 50:] = 200          # the right half is the "alpha widget"
    return frame


def test_boxes_are_proposed_from_the_stock_detector_and_clip(tmp_path):
    p = _object_project(tmp_path)
    result = boxes.propose(p, FakeDetector(), CropEmbedder(), read_at=_frame)
    assert result["proposed"] == 2 and result["auto_accepted"] == 1
    labels = boxes.store(p)
    by_class = {b.class_name: b for b in labels.boxes if b.class_name}
    # A stock class takes the detector's own box, and at that confidence
    # needs no review...
    assert by_class["person"].box == pytest.approx((0.1, 0.1, 0.3, 0.5))
    assert by_class["person"].verdict == ACCEPTED
    assert by_class["alpha widget"].verdict == PENDING
    # ...anything else, the detected region that looks like it.
    assert by_class["alpha widget"].box == pytest.approx((0.6, 0.2, 0.3, 0.3))
    assert len(labels.negatives()) == 1

    again = boxes.propose(p, FakeDetector(), CropEmbedder(), read_at=_frame)
    assert again["proposed"] == 0                  # nothing proposed twice


def test_box_verdicts_and_the_labeller_worklist(tmp_path):
    p = _object_project(tmp_path)
    boxes.propose(p, FakeDetector(), CropEmbedder(), read_at=_frame)
    record = boxes.next_sheet(p, read_at=lambda *a: None,
                              renderer=lambda *a, **k: None)
    assert [i["class_name"] for i in record["items"]] == ["alpha widget"]
    result = boxes.apply_verdicts(p, record["sheet"], reject="1")
    assert result["applied"] == 1 and result["errors"] == []
    todo = boxes.labeler_worklist(p)
    assert [t["class"] for t in todo] == ["alpha widget"]


def test_labeller_exports_attach_to_the_project_samples(tmp_path):
    p = _object_project(tmp_path)
    sample = p.samples[0]
    export = tmp_path / "export.json"
    export.write_text(json.dumps({
        "video": os.path.join("C:/elsewhere", os.path.basename(sample.path)),
        "fps": 10, "frame_width": 100, "frame_height": 100,
        "keyframes": [{"frame_number": 5, "points": {"alpha widget": [50, 50],
                                                      "not a class": [1, 1]}}]}))
    result = boxes.import_labeler(p, [str(export)], accept=True)
    assert result == {"imported": 1, "skipped_unknown_classes": ["not a class"]}
    box = boxes.store(p).accepted()[0]
    assert box.video == sample.path and box.time == pytest.approx(0.5)


# --- the CLI --------------------------------------------------------------------------------

def test_cli_answers_in_json_and_names_the_next_step(tmp_path, capsys):
    root = str(tmp_path / "cli")
    assert cli.main(["--project", root, "init", "--task", "actions"]) == 0
    capsys.readouterr()
    assert cli.main(["--project", root, "add-class", "Alpha_Move"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["added"] == "alpha move"
    assert "add-video" in out["next"]["command"]

    assert cli.main(["--project", root, "add-class", "alpha move"]) == 2
    assert "already" in json.loads(capsys.readouterr().out)["error"]

    assert cli.main(["--project", str(tmp_path / "missing"), "status"]) == 2
    assert "init" in json.loads(capsys.readouterr().out)["error"]


def test_cli_bad_arguments_are_an_answer_not_an_exit(tmp_path):
    code, result = cli.run(["--project", str(tmp_path / "p"), "sort", "--no-such-flag"])
    assert code == 2 and "unrecognized arguments: --no-such-flag" in result["error"]


def test_cli_settings_are_typed(tmp_path, capsys):
    root = str(tmp_path / "cli")
    cli.main(["--project", root, "init", "--task", "objects"])
    capsys.readouterr()
    assert cli.main(["--project", root, "set", "gate=0.4", "focus=yes"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["changed"] == {"gate": 0.4, "focus": True}
    assert cli.main(["--project", root, "set", "nonsense=1"]) == 2


def test_a_sample_the_cropper_or_the_proposer_found_nothing_in_is_not_asked_about_forever(tmp_path):
    from modules.teach import focus

    p = Project.create(str(tmp_path / "f"), ACTIONS)
    p.add_class("alpha move")
    p.settings.focus = True
    _add_samples(p, [0, 0])
    example = tmp_path / "example.mp4"
    example.write_bytes(b"clip")
    p.add_source(str(example)).cut = True
    p.samples.append(Sample(id="v002__example", source="v002", path=str(example),
                            start=0.0, duration=5.0))
    p.save()
    seen = []

    def cropper(inbox, out):
        names = sorted(os.listdir(inbox))
        seen.extend(names)
        # Crops the first sample only; finds nobody in the rest.
        open(os.path.join(out, names[0][:-4] + "_cropped_left.mp4"), "wb").close()

    result = focus.focus_project(p, cropper=cropper)
    assert result["cropped"] == 1 and result["nothing_found"] == 2
    assert "v002__example.mp4" in seen                  # examples are cropped too
    assert not status.next_step(p)["command"].endswith(" focus")
    assert focus.focus_project(p, cropper=lambda *a: seen.append("again")) == {
        "cropped": 0, "nothing_found": 0, "samples_total": 3}

    o = _object_project(tmp_path)

    class Blind:
        def detect(self, frame):
            return []

    boxes.propose(o, Blind(), CropEmbedder(), read_at=_frame)
    assert "propose" not in status.next_step(Project.load(o.root))["command"]


# --- auto-accept -------------------------------------------------------------------

from modules.teach import autolabel  # noqa: E402
from modules.teach.project import AUTO  # noqa: E402


def _confident(project, n_alpha=20, checked=5):
    """n_alpha pending samples confidently guessed alpha; ``checked`` of them
    accepted by a person first."""
    _add_samples(project, [0] * (n_alpha + checked))
    for i, s in enumerate(project.samples):
        s.scores = {"alpha move": 1.2, "beta move": 0.1}
        s.proposed, s.margin = "alpha move", 1.1
        if i < checked:
            project.decide(s, ACCEPTED, "alpha move", by="sheet:1")
    project.save()


def test_nothing_is_auto_accepted_before_a_class_has_checked_samples(project):
    _confident(project, checked=4)
    result = autolabel.apply(project)
    assert result["accepted"] == {}
    assert "4 of 5" in result["classes"]["alpha move"]["why"]


def test_confident_guesses_are_accepted_once_a_class_is_checked(project):
    _confident(project, checked=5)
    result = autolabel.apply(project)
    assert result["accepted"] == {"alpha move": 20}
    auto = [s for s in project.samples if s.is_auto]
    assert len(auto) == 20 and all(s.auto_label == "alpha move" for s in auto)


def test_a_narrow_or_disputed_guess_is_left_for_review(project):
    _confident(project, checked=5)
    project.samples[10].margin = 0.05
    project.samples[11].model_proposed = "beta move"
    autolabel.apply(project)
    assert project.samples[10].verdict == PENDING
    assert project.samples[11].verdict == PENDING


def test_spot_checks_that_overturn_too_much_switch_it_off_and_take_it_back(project):
    _confident(project, checked=5)
    autolabel.apply(project)
    auto = [s for s in project.samples if s.is_auto]
    # A person checks five and overturns two: 40% > 20%.
    for s in auto[:3]:
        project.decide(s, ACCEPTED, "alpha move", by="sheet:2")
    for s in auto[3:5]:
        project.decide(s, ACCEPTED, "beta move", by="sheet:2")
    result = autolabel.apply(project)
    assert result["reverted"] == 15
    assert not result["classes"]["alpha move"]["on"]
    assert not any(s.is_auto for s in project.samples)
    # ... and it stays off: the next sort decides nothing for that class.
    assert autolabel.apply(project)["accepted"] == {}


def test_review_sheets_carry_spot_checks_and_they_are_scored(project):
    _confident(project, checked=5)
    autolabel.apply(project)
    assert autolabel.audits_needed(project, "alpha move") == 3
    record = review.next_sheet(project, size=10, frame_reader=lambda *a, **k: [],
                               renderer=lambda *a, **k: None)
    audit = [i for i in record["items"] if "spot check" in i["caption"]]
    assert audit and all(i["proposed"] == "alpha move" for i in audit)
    review.apply_verdicts(project, record["sheet"], accept_rest=True)
    checks, overturned = autolabel.spot_checks(project, "alpha move")
    assert checks == len(audit) and overturned == 0


def test_status_asks_for_spot_checks_before_training_on_auto_labels(project):
    _confident(project, n_alpha=25, checked=25)
    for s in project.samples[:25]:
        s.verdict, s.decided_by = PENDING, ""
    project.samples = project.samples[:25]
    for s in project.samples[:5]:
        project.decide(s, ACCEPTED, "alpha move", by="sheet:1")
    project.classes = project.classes[:1]
    project.classes[0].target = 20
    autolabel.apply(project)
    step = status.next_step(project)
    assert step["who"] == "judge" and "Spot-check" in step["why"]


def test_only_checked_samples_are_ever_held_out(project):
    _confident(project, n_alpha=30, checked=10)
    autolabel.apply(project)
    build.assign_splits(project)
    val = [s for s in project.samples if s.split == VAL]
    assert val and all(s.is_human for s in val)
    skipped = [s for s in project.samples if s.split == build.SKIP]
    assert all(s.is_auto for s in skipped)
    # A skipped neighbour a person later checks joins validation.
    if skipped:
        project.decide(skipped[0], ACCEPTED, "alpha move", by="sheet:3")
        build.assign_splits(project)
        assert skipped[0].split == VAL


def test_auto_prototypes_ignore_auto_accepted_samples(project):
    truth = _add_samples(project, [0, 0, 1, 1])
    project.decide(project.samples[0], ACCEPTED, "alpha move", by="sheet:1")
    project.decide(project.samples[1], ACCEPTED, "alpha move", by=AUTO)
    vectors = {s.id: _direction(i) for i, s in enumerate(project.samples)}
    protos = {p.name: p for p in sort.build_prototypes(project, vectors, FakeEmbedder())}
    assert protos["alpha move"].n_examples == 1
    assert truth  # (layout used)


# --- auto / quick -------------------------------------------------------------------

def test_auto_runs_unattended_steps_and_stops_where_someone_must_look(tmp_path, monkeypatch):
    p = Project.create(str(tmp_path / "a"), ACTIONS)
    p.add_class("alpha move")
    p.save()
    truth = _add_samples(p, [0] * 6 + [-1] * 4)
    for s in p.sources:
        s.cut = False
    p.save()
    ran = []

    def fake_cut(proj, **kw):
        for s in proj.sources:
            s.cut = True
        proj.save()
        ran.append("cut")
        return {"samples_made": 0}

    monkeypatch.setattr(cut, "cut_project", fake_cut)
    monkeypatch.setattr(cli, "make_embedder", lambda *_: FakeEmbedder())
    import modules.teach.embed as embed_mod
    monkeypatch.setattr(embed_mod, "read_frames", _reader(truth))

    result = cli.run_auto(p.root)
    assert [r["args"][0] for r in result["ran"]] == ["cut", "sort"]
    assert result["stopped_at"]["who"] == "judge"


def test_quick_builds_a_project_from_an_examples_folder(tmp_path, monkeypatch):
    examples = tmp_path / "examples"
    for folder in ("Alpha_Move", "beta move", "_ignored"):
        (examples / folder).mkdir(parents=True)
        (examples / folder / "one.mp4").write_bytes(b"x")
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "long.mp4").write_bytes(b"x")
    monkeypatch.setattr(cut, "probe_duration", lambda path: 5.0)
    monkeypatch.setattr(cli, "run_auto", lambda root, train=False: {"ran": [], "stopped_at": {}})
    from modules.teach import doctor
    monkeypatch.setattr(doctor, "require", lambda root: None)

    code, result = cli.run(["--project", str(tmp_path / "q"), "quick", "--task", "actions",
                            "--examples", str(examples), "--videos", str(videos)])
    assert code == 0, result
    assert result["classes"] == ["alpha move", "beta move"]
    assert result["examples_added"] == {"alpha move": 1, "beta move": 1}
    project = Project.load(str(tmp_path / "q"))
    assert [s.verdict for s in project.samples] == [ACCEPTED, ACCEPTED]
    assert len(project.sources) == 3            # two example clips + one video

    bad = tmp_path / "bad"
    (bad / "123").mkdir(parents=True)
    code, result = cli.run(["--project", str(tmp_path / "q2"), "quick", "--task", "actions",
                            "--examples", str(bad)])
    assert code == 2 and "rename" in result["error"]


def test_a_thing_no_detector_knows_is_found_by_scanning_regions(tmp_path):
    p = _object_project(tmp_path)

    class Blind:
        def detect(self, frame):
            return []

    def frame(*_):
        f = np.zeros((90, 160, 3), np.uint8)
        f[30:70, 100:150] = 200          # the thing: bright, right of centre
        return f

    class Brightness(FakeEmbedder):
        """Likeness to the class = how much of the crop is the bright thing."""

        def images(self, frames):
            out = []
            for f in frames:
                share = float((f > 100).mean()) if isinstance(f, np.ndarray) else 0.0
                out.append(share * _direction(0) + (1 - share) * _direction(DIM - 1))
            v = np.array(out, np.float32)
            return v / np.linalg.norm(v, axis=1, keepdims=True)

    result = boxes.propose(p, Blind(), Brightness(), read_at=frame)
    found = [b for b in boxes.store(p).boxes if b.class_name == "alpha widget"]
    assert result["proposed"] >= 1 and found
    x, y, w, h = found[0].box
    assert found[0].source == "category" and found[0].verdict == PENDING
    # Around the thing (x 0.62-0.94, y 0.33-0.78), not a fixed half-frame.
    assert x >= 0.45 and x + w <= 1.0 and w < 0.5



# --- doctor ------------------------------------------------------------------------

def test_doctor_separates_what_blocks_from_what_only_slows(tmp_path):
    from modules.teach import doctor

    checks = (lambda: doctor.Check("ffmpeg", True, doctor.REQUIRED, "found"),
              lambda: doctor.Check("clip", False, doctor.REQUIRED, "missing", "install it"),
              lambda: doctor.Check("pose model", False, doctor.OPTIONAL, "later"))
    report = doctor.run(str(tmp_path / "p"), checks=checks)
    assert not report["ready"] and report["blocking"] == ["clip"]
    assert {c["name"] for c in report["checks"]} >= {"ffmpeg", "clip", "pose model",
                                                     "disk space"}


def test_quick_stops_before_creating_anything_when_not_ready(tmp_path, monkeypatch):
    from modules.teach import doctor

    monkeypatch.setattr(doctor, "run", lambda root: {
        "ready": False, "blocking": ["clip"],
        "checks": [{"name": "clip", "ok": False, "level": doctor.REQUIRED,
                    "detail": "missing", "fix": "install the CLIP pack"}]})
    root = tmp_path / "q"
    code, result = cli.run(["--project", str(root), "quick", "--task", "actions"])
    assert code == 2 and "install the CLIP pack" in result["error"]
    assert not root.exists()


def test_an_unreadable_clip_does_not_keep_sort_the_next_step_forever(project):
    truth = _add_samples(project, [0, 1, 0, 1])
    broken = project.samples[2].id + ".mp4"

    def reader(path, count, **_):
        return [] if os.path.basename(path) == broken else _reader(truth)(path, count)

    result = sort.sort_project(project, FakeEmbedder(), frame_reader=reader)
    assert result["unreadable"] == 1
    assert project.samples[2].unreadable
    assert not status.next_step(Project.load(project.root))["command"].endswith(" sort")


def test_vectors_from_another_model_are_not_reused(tmp_path):
    from modules.teach.embed import VectorCache

    first = VectorCache(str(tmp_path), "model-a")
    first.put("s1", np.ones(4, np.float32))
    first.save()
    assert VectorCache(str(tmp_path), "model-a").get("s1") is not None
    assert VectorCache(str(tmp_path), "model-b").get("s1") is None
    assert VectorCache(str(tmp_path), "").get("s1") is None


def test_renaming_a_class_reaches_every_record_of_it(tmp_path):
    from modules.vision.label_store import LabelledBox, LabelStore

    p = _object_project(tmp_path)
    s = p.samples[0]
    s.auto_label, s.model_proposed = "alpha widget", "alpha widget"
    labels = LabelStore(p.path("labels.json"))
    labels.add(LabelledBox(video=s.path, time=0.5, class_name="alpha widget",
                           box=(0.1, 0.1, 0.2, 0.2), verdict=ACCEPTED))
    labels.save()
    p.rename_class("alpha widget", "beta widget")
    assert (s.label, s.auto_label, s.model_proposed) == ("beta widget",) * 3
    assert [b.class_name for b in LabelStore(p.path("labels.json")).load().boxes] == [
        "beta widget"]


def test_negatives_train_as_a_background_class_only_when_one_is_named(project):
    _add_samples(project, [0] * 6 + [-1] * 6)
    for s in project.samples[:6]:
        project.decide(s, ACCEPTED, "alpha move", by="sheet:1")
    for s in project.samples[6:]:
        project.decide(s, NEGATIVE, by="sheet:1")
    before = build.build(project)["signature"]
    assert not os.path.exists(project.path("dataset", "train", "background"))

    # Marking yet another negative trains nothing new, so it needs no rebuild.
    extra = _add_samples(project, [-1])
    project.decide(project.samples[-1], NEGATIVE, by="sheet:2")
    assert build.dataset_signature(project) == before and extra

    project.settings.background_class = "background"
    result = build.build(project)
    assert result["signature"] != before
    names = {split: os.listdir(project.path("dataset", split)) for split in ("train", "val")}
    assert all("background" in n for n in names.values())

    project.settings.background_class = "alpha move"      # a class already
    with pytest.raises(ValueError, match="background_class"):
        build.build(project)


def test_accepting_a_none_of_these_guess_confirms_it(project):
    _scored(project, [("alpha move", 0.5, "")] * 2 + [(NONE, 0.3, "")])
    record = review.next_sheet(project, size=3, frame_reader=lambda *a, **k: [],
                               renderer=lambda *a, **k: None)
    result = review.apply_verdicts(project, record["sheet"], accept="1-3")
    assert result["errors"] == []
    assert sorted(s.verdict for s in project.samples) == [ACCEPTED, ACCEPTED, NEGATIVE]


# --- round 2: the project's own model proposes too ----------------------------------

def test_a_trained_round_proposes_and_disagreements_are_reviewed_first(project, tmp_path):
    weights = tmp_path / "r.pth"
    weights.write_bytes(b"w")
    mapping = tmp_path / "r_mapping.json"
    mapping.write_text(json.dumps({"idx_to_label": {"0": "alpha move", "1": "beta move"},
                                   "metadata": {"model_variant": "mc3_18"}}))
    made = {}

    class Wrapper:
        def __init__(self, **kw):
            made.update(kw)

        def predict_from_frames(self, frames):
            # Says "beta" for everything: disagrees with CLIP on alpha samples.
            return np.array([0.0, 3.0])

    classify = sort.r3d_classifier(str(weights), str(mapping), wrapper_factory=Wrapper,
                                   frame_reader=lambda path, n: [np.zeros((4, 4, 3))] * n)
    assert made["model_name"] == "mc3_18" and made["custom_num_classes"] == 2
    label, confidence = classify("x.mp4")
    assert label == "beta move" and confidence > 0.9

    truth = _add_samples(project, [0, 0, 1, -1])
    sort.sort_project(project, FakeEmbedder(), frame_reader=_reader(truth),
                      model_classifier=classify)
    batch = review.pick_batch(project, size=4)
    assert batch[0].model_proposed == "beta move" and batch[0].proposed == "alpha move"


def test_round_model_reads_the_mapping_train_py_writes(tmp_path):
    # Top-level variant, and a production mapping that dropped class 1: the
    # head still has three outputs.
    weights = tmp_path / "r.pth"
    weights.write_bytes(b"w")
    mapping = tmp_path / "r_mapping.json"
    mapping.write_text(json.dumps({"idx_to_label": {"0": "alpha move", "2": "gamma move"},
                                   "num_classes_total": 3, "model_variant": "r2plus1d_18"}))
    made = {}

    class Wrapper:
        def __init__(self, **kw):
            made.update(kw)

        def predict_from_frames(self, frames):
            return np.array([0.0, 0.0, 4.0])

    classify = sort.r3d_classifier(str(weights), str(mapping), wrapper_factory=Wrapper,
                                   frame_reader=lambda path, n: [np.zeros((4, 4, 3))] * n)
    assert made["model_name"] == "r2plus1d_18" and made["custom_num_classes"] == 3
    assert classify("x.mp4")[0] == "gamma move"


def test_only_an_installed_round_proposes(project, tmp_path):
    assert sort.round_classifier(project) is None
    project.rounds.append({"round": 1, "installed": False,
                           "metrics": {"weights": "x", "mapping": "y"}})
    assert sort.round_classifier(project) is None


def test_a_frame_whose_box_was_rejected_is_proposed_again_elsewhere(tmp_path):
    from modules.vision.label_store import LabelledBox, REJECTED as BOX_REJECTED

    p = _object_project(tmp_path)
    p.samples[1].verdict, p.samples[1].label = "rejected", ""     # only alpha matters
    p.save()
    sample = p.samples[0]
    moment = boxes.frame_times(sample.duration, 1)[0]
    labels = boxes.store(p)
    # Three accepted boxes elsewhere teach what the class looks like...
    for t in (10.0, 11.0, 12.0):
        labels.add(LabelledBox(video=sample.path, time=t, class_name="alpha widget",
                               box=(0.6, 0.2, 0.3, 0.3), verdict=ACCEPTED))
    # ...and a person already said the obvious box on this frame is wrong.
    labels.add(LabelledBox(video=sample.path, time=moment, class_name="alpha widget",
                           box=(0.6, 0.2, 0.3, 0.3), verdict=BOX_REJECTED))
    labels.save()

    result = boxes.propose(p, FakeDetector(), CropEmbedder(), read_at=_frame)
    assert result["proposed"] == 1
    new = [b for b in boxes.store(p).pending() if b.time == moment]
    assert new and boxes._iou(new[0].box, (0.6, 0.2, 0.3, 0.3)) <= 0.5

    # Rejected twice: left to the labeller, not asked about again.
    labels = boxes.store(p)
    for b in labels.boxes:
        if b.verdict == "pending":
            b.verdict = BOX_REJECTED
    labels.save()
    assert boxes.propose(p, FakeDetector(), CropEmbedder(), read_at=_frame)["proposed"] == 0


def test_rejected_frames_are_retried_by_status_and_a_fruitless_retry_ends_it(tmp_path):
    from modules.teach import status
    from modules.vision.label_store import LabelledBox, REJECTED as BOX_REJECTED

    p = _object_project(tmp_path)
    p.samples[1].verdict, p.samples[1].label = "rejected", ""
    for s in p.samples:
        s.scores = {"alpha widget": 1.0}
        s.boxes_tried = True
    p.save()
    sample = p.samples[0]
    moment = boxes.frame_times(sample.duration, 1)[0]
    labels = boxes.store(p)
    labels.add(LabelledBox(video=sample.path, time=moment, class_name="alpha widget",
                           box=(0.6, 0.2, 0.3, 0.3), verdict=BOX_REJECTED))
    labels.save()
    # Named only in words, a retry would guess the same box again: not offered.
    assert boxes.retryable(p) == []

    labels = boxes.store(p)
    for t in (10.0, 11.0, 12.0):
        labels.add(LabelledBox(video=sample.path, time=t, class_name="alpha widget",
                               box=(0.6, 0.2, 0.3, 0.3), verdict=ACCEPTED))
    labels.save()
    assert boxes.retryable(p) == [(sample.path, moment)]
    assert status.next_step(p)["args"] == ["boxes", "propose"]

    class Nothing:
        def detect(self, frame):
            return []

    blank = lambda *_: np.zeros((100, 100, 3), np.uint8)     # noqa: E731
    result = boxes.propose(p, Nothing(), CropEmbedder(), read_at=blank)
    assert result["proposed"] == 0
    assert boxes.retryable(p) == []
    assert status.next_step(p)["args"] != ["boxes", "propose"]


# --- sharing a taught detector ---------------------------------------------------------

def test_the_installed_detector_is_drafted_for_the_hub(tmp_path):
    from modules.teach import share
    from modules.vision.label_store import LabelledBox, LabelStore

    p = _object_project(tmp_path)
    with pytest.raises(share.NotShareable, match="train a round"):
        share.share_draft(p)

    model_dir = tmp_path / "models" / "teach_obj"
    model_dir.mkdir(parents=True)
    (model_dir / "m.onnx").write_bytes(b"onnx")
    (model_dir / "labels.json").write_text(json.dumps(["alpha widget", "person"]))
    p.rounds.append({"round": 1, "installed": True,
                     "install": {"xml": str(model_dir / "m.xml")}})
    labels = LabelStore(p.path("labels.json"))
    for t in (1.0, 2.0):
        labels.add(LabelledBox(video=p.samples[0].path, time=t, class_name="alpha widget",
                               box=(0.1, 0.1, 0.2, 0.2), verdict=ACCEPTED))
    labels.save()

    onnx, draft = share.share_draft(p)
    assert onnx == str(model_dir / "m.onnx")
    assert draft.labels == ["alpha widget", "person"]
    assert draft.metrics == {"rounds": 1, "train_frames": 2, "videos": 1}
    assert draft.name == "" and draft.description == ""       # the person's to write
    assert draft.problems(require_compliance=False)            # ...so not ready yet

    p.task = ACTIONS
    with pytest.raises(share.NotShareable, match="Only object"):
        share.share_draft(p)


def test_sharing_describes_the_installed_round_not_the_latest(tmp_path, monkeypatch):
    from modules.teach import share
    from modules.teach.project import slugify

    p = _object_project(tmp_path)
    # Installed before rounds recorded where: found where install puts it.
    monkeypatch.chdir(tmp_path)
    name = f"teach_{slugify(p.name)}"
    model_dir = tmp_path / "models" / "custom" / name
    model_dir.mkdir(parents=True)
    (model_dir / f"{name}.onnx").write_bytes(b"onnx")
    (model_dir / "labels.json").write_text(json.dumps(["alpha widget"]))
    p.rounds.append({"round": 1, "installed": True, "trained_frames": 5, "videos": 2})
    # A later round that did not beat it, trained on more.
    p.rounds.append({"round": 2, "installed": False, "trained_frames": 9, "videos": 3})

    onnx, draft = share.share_draft(p)
    assert onnx == str(model_dir / f"{name}.onnx")
    assert draft.metrics == {"rounds": 1, "train_frames": 5, "videos": 2}


@pytest.mark.parametrize("share", [0.1, 0.3, 0.6, 0.8])
def test_ordinary_footage_calibrates_to_zero_however_common_the_thing_is(share):
    rng = np.random.default_rng(0)
    n = 200
    k = int(n * share)
    ordinary = rng.normal(0.5, 0.03, n - k)
    thing = rng.normal(0.9, 0.03, k)
    column = np.concatenate([ordinary, thing])
    floor = scoring.background(column)
    assert abs(floor - 0.5) < 0.02, (share, floor)


def test_footage_without_the_thing_is_its_own_ordinary():
    column = np.random.default_rng(1).normal(0.5, 0.03, 200)
    assert abs(scoring.background(column) - np.median(column)) < 1e-9
