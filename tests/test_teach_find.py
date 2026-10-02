"""From one drawn box to a trained-on set, asking only about what is in doubt.

A synthetic world: frames are 100x100, and a sample that shows the thing has
a bright patch where the stock detector also finds a region. The fake CLIP
sees how much of a crop is the patch, so the region search, the calibration
and the learnt cutoff all run on real arithmetic.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from modules.teach import boxes, cli, cutoff, find, status
from modules.teach.project import ACCEPTED, AUTO_BOX, OBJECTS, PENDING, Project, Sample
from modules.teach.seed import parse_box, seed

DIM = 16
PATCH = (60, 20, 90, 50)            # x1, y1, x2, y2 in pixels
PATCH_BOX = (0.6, 0.2, 0.3, 0.3)


def _unit(v):
    return v / np.linalg.norm(v)


THING = _unit(np.eye(DIM)[0])
BACKGROUND = _unit(np.eye(DIM)[DIM - 1])


class PatchEmbedder:
    model_id = "fake-patch"

    def __init__(self):
        self.rng = np.random.default_rng(0)

    def images(self, crops):
        out = []
        for crop in crops:
            share = float((np.asarray(crop) > 100).mean()) if np.asarray(crop).size else 0.0
            out.append(share * THING + (1 - share) * BACKGROUND
                       + 0.01 * self.rng.normal(size=DIM))
        v = np.array(out, np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def texts(self, texts):
        return np.array([_unit(THING + BACKGROUND)] * len(texts), np.float32)


class _Det:
    def __init__(self, box):
        self.class_name, self.confidence, self.class_id = "cup", 0.4, 0
        self.x1, self.y1, self.x2, self.y2 = box


class PatchDetector:
    """Always finds the region the patch would be in, patch or not."""

    def detect(self, frame):
        return [_Det(PATCH), _Det((5, 5, 35, 45))]


def _frame(has_thing: bool):
    frame = np.zeros((100, 100, 3), np.uint8)
    if has_thing:
        x1, y1, x2, y2 = PATCH
        frame[y1:y2, x1:x2] = 220
    return frame


@pytest.fixture
def world(tmp_path):
    truth = {}

    def add(project, layout, name):
        video = str(tmp_path / f"{name}.mp4")
        with open(video, "wb") as handle:
            handle.write(b"video")
        source = project.add_source(video)
        source.cut = True
        os.makedirs(project.path("samples"), exist_ok=True)
        for i, has in enumerate(layout):
            sid = f"{source.id}__{i:08d}"
            path = project.path("samples", sid + ".mp4")
            with open(path, "wb") as handle:
                handle.write(b"clip")
            project.samples.append(Sample(id=sid, source=source.id, path=path,
                                          start=i * 5.0, duration=5.0))
            truth[path] = has
        truth[source.path] = True           # the seed frame shows it
        project.save()
        return video

    def read_at(path, moment):
        return _frame(truth.get(path, False))

    return {"add": add, "read_at": read_at, "truth": truth, "root": str(tmp_path / "p")}


@pytest.fixture
def real_cv2(monkeypatch):
    """Review sheets draw the box on the frame: that needs real pixels."""
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from conftest import real_opencv

    cv2 = real_opencv()
    if cv2 is None:
        pytest.skip("drawing a box needs OpenCV")
    monkeypatch.setitem(sys.modules, "cv2", cv2)


def _answer_all(project, read_at, right):
    """Answer every pending box: accept where ``right(box)``, else reject."""
    while True:
        record = boxes.next_sheet(project, size=50, read_at=read_at,
                                  renderer=lambda *a, **k: None)
        if not record:
            return
        labels = {(b.video, round(b.time, 3)): b for b in boxes.store(project).pending()}
        yes = [str(i["n"]) for i in record["items"]
               if right(labels[(i["video"], round(i["time"], 3))])]
        no = [str(i["n"]) for i in record["items"] if str(i["n"]) not in yes]
        boxes.apply_verdicts(project, record["sheet"], accept=",".join(yes),
                             reject=",".join(no))


def test_a_box_must_be_inside_the_frame():
    assert parse_box("0.1,0.2,0.3,0.4") == (0.1, 0.2, 0.3, 0.4)
    for bad in ("0.1,0.2,0.3", "0.8,0.2,0.3,0.3", "0.1,0.2,0,0.3", "-0.1,0,0.2,0.2"):
        with pytest.raises(ValueError):
            parse_box(bad)


def test_the_cutoff_is_the_lowest_score_that_was_right_often_enough():
    assert cutoff.learn([(0.9, True)] * (cutoff.MIN_DECIDED - 1), 0.1) is None
    scores = [round(1.0 - 0.05 * i, 2) for i in range(12)]          # 1.0 .. 0.45
    decided = [(s, s >= 0.7) for s in scores]
    assert cutoff.learn(decided, 0.1) == 0.7
    # Wrong at the top: nothing is safe to accept unasked.
    assert cutoff.learn([(1.0, False)] * 3 + [(0.9, True)] * 9, 0.1) is None
    # Equal scores are one step: a cutoff cannot split them.
    assert cutoff.learn([(0.8, True)] * 9 + [(0.8, False)] * 3, 0.1) is None


def test_one_drawn_box_finds_the_thing_and_asks_only_about_it(world):
    p = Project.create(world["root"], OBJECTS)
    video = world["add"](p, [True] * 10 + [False] * 20, "first")
    result = seed(p.root, video, 1.0, ",".join(map(str, PATCH_BOX)), "widget")
    assert result == {"class": "widget", "created": True, "seeds": 1, "source": "v001"}

    p = Project.load(p.root)
    # Shown by a box, not named in words: no whole-frame sort, no sample review.
    assert status.next_step(p)["args"] == ["find"]
    found = find.find(p, PatchDetector(), PatchEmbedder(), read_at=world["read_at"])
    assert found["proposed"] == {"widget": 10}
    assert found["auto_accepted"] == 0                 # nothing checked yet
    pending = boxes.store(p).pending()
    assert all(world["truth"][b.video] for b in pending)
    assert all(boxes._iou(b.box, PATCH_BOX) > 0.9 for b in pending)
    assert status.next_step(Project.load(p.root))["args"] == ["boxes", "review"]
    assert not find.needed(Project.load(p.root))       # nothing new to look at


def test_answers_teach_the_cutoff_and_the_rest_is_accepted_unasked(world, real_cv2):
    p = Project.create(world["root"], OBJECTS)
    video = world["add"](p, [True] * 12 + [False] * 20, "first")
    seed(p.root, video, 1.0, PATCH_BOX, "widget")
    p = Project.load(p.root)
    find.find(p, PatchDetector(), PatchEmbedder(), read_at=world["read_at"])
    _answer_all(p, world["read_at"], right=lambda b: True)

    p = Project.load(p.root)
    assert len(p.accepted("widget")) == 12
    assert all(s.decided_by == "boxes" and s.is_human for s in p.accepted("widget"))
    limit = cutoff.cutoffs(p, boxes.store(p))["widget"]
    assert limit is not None

    # More footage: what scores like the checked boxes is accepted unasked.
    world["add"](p, [True] * 16 + [False] * 10, "second")
    p = Project.load(p.root)
    assert status.next_step(p)["args"] == ["find"]
    found = find.find(p, PatchDetector(), PatchEmbedder(), read_at=world["read_at"])
    assert found["proposed"] == {"widget": 16}
    assert found["auto_accepted"] >= 1
    p = Project.load(p.root)
    by_box = [s for s in p.samples if s.decided_by == AUTO_BOX]
    assert len(by_box) == found["auto_accepted"]
    # Never a prototype or a held-out sample: nobody looked at them.
    assert not any(s.is_human for s in by_box)
    # The rest were asked: below the cutoff, or a spot check.
    assert found["questions"] == 16 - found["auto_accepted"]
    # Auto-accepted crops do not count as looked at, so they do not re-trigger find.
    assert not find.needed(p)


def test_a_wrong_answer_above_the_cutoff_takes_back_what_it_accepted(world):
    from modules.vision.label_store import LabelledBox, REJECTED

    p = Project.create(world["root"], OBJECTS)
    p.add_class("widget")
    world["add"](p, [True] * 3, "first")
    labels = boxes.store(p)

    def box(n, confidence, verdict):
        return labels.add(LabelledBox(video=f"/checked/{n}.mp4", time=2.5,
                                      class_name="widget", box=PATCH_BOX,
                                      source=find.FOUND, confidence=confidence,
                                      verdict=verdict))

    for n in range(12):                                   # checked, all right
        box(n, 0.80 + 0.01 * n, ACCEPTED)
    candidates = [LabelledBox(video=s.path, time=2.5, class_name="widget", box=PATCH_BOX,
                              source=find.FOUND, confidence=0.95, verdict=PENDING)
                  for s in p.samples]
    target = next(b for b in candidates if not cutoff.is_spot_check(b))
    labels.add(target)
    labels.save()

    assert cutoff.apply(p, labels)["auto_accepted"] == 1
    find.settle(p, labels)
    sample = next(s for s in p.samples if s.path == target.video)
    assert sample.decided_by == AUTO_BOX and not sample.is_human

    # Answers now say boxes this likely are often wrong: no longer safe.
    labels = boxes.store(p)
    for n in range(3):
        box(100 + n, 0.99, REJECTED)
    labels.save()
    result = cutoff.apply(p, labels)
    assert result["taken_back"] == 1 and result["cutoffs"]["widget"] is None
    find.settle(p, labels)
    p = Project.load(p.root)
    sample = next(s for s in p.samples if s.path == target.video)
    assert sample.verdict == PENDING and sample.decided_by == ""
    assert [b.verdict for b in boxes.store(p).boxes if b.video == target.video] == [PENDING]


def test_seed_and_find_from_the_command_line(world, monkeypatch):
    p = Project.create(world["root"], OBJECTS)
    video = world["add"](p, [True] * 3 + [False] * 5, "first")
    code, result = cli.run(["--project", p.root, "seed", "--video", video, "--time", "1",
                            "--box", "0.6,0.2,0.3,0.3", "--class", "widget"])
    assert code == 0 and result["seeds"] == 1
    assert result["next"]["args"] == ["find"]
    monkeypatch.setattr(cli, "make_detector", PatchDetector)
    monkeypatch.setattr(cli, "make_embedder", lambda *_: PatchEmbedder())
    monkeypatch.setattr(boxes, "_read_at", world["read_at"])
    monkeypatch.setattr(find, "_read_at", world["read_at"])
    code, result = cli.run(["--project", p.root, "find"])
    assert code == 0 and result["proposed"] == {"widget": 3}
    assert result["next"]["args"] == ["boxes", "review"]

    code, result = cli.run(["--project", p.root, "seed", "--video", video, "--time", "1",
                            "--box", "0.9,0.2,0.3,0.3", "--class", "widget"])
    assert code == 2 and "inside the frame" in result["error"]
    assert Project.load(p.root).samples[0].verdict == PENDING


def test_an_unreadable_sample_is_searched_once_not_forever(world):
    p = Project.create(world["root"], OBJECTS)
    video = world["add"](p, [True] * 4 + [False] * 4, "first")
    seed(p.root, video, 1.0, PATCH_BOX, "widget")
    p = Project.load(p.root)
    broken = p.samples[0].path

    def read_at(path, moment):
        return None if path == broken else world["read_at"](path, moment)

    found = find.find(p, PatchDetector(), PatchEmbedder(), read_at=read_at)
    assert found["proposed"] == {"widget": 3}
    assert not find.needed(Project.load(p.root))
