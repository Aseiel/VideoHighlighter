"""modules/teach end to end on real video: ffmpeg cuts, OpenCV decodes.

Skipped where OpenCV or the bundled ffmpeg is missing (conftest.py replaces
cv2 with a mock there). The footage is generated — plain colour fields with a
moving shape, never real material — and CLIP is stood in for by a colour
histogram, which is enough to tell those scenes apart. What is under test is
the plumbing: that samples are cut where the plan says, decoded, sorted,
drawn on a sheet, judged, split and laid out for the trainer.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import real_opencv  # noqa: E402

cv2 = real_opencv()
if cv2 is None:
    pytest.skip("real OpenCV is not installed", allow_module_level=True)
imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")

from modules.teach import cli  # noqa: E402
from modules.teach.project import Project  # noqa: E402

RED, BLUE, GREEN = (40, 40, 200), (200, 60, 40), (60, 160, 60)


def _video(path, scenes, fps=15):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 90))
    for seconds, colour, shape in scenes:
        for f in range(int(seconds * fps)):
            img = np.full((90, 160, 3), colour, np.uint8)
            x = 10 + (f * 4) % 130
            if shape == "circle":
                cv2.circle(img, (x, 45), 12, (255, 255, 255), -1)
            elif shape == "box":
                cv2.rectangle(img, (x, 30), (x + 25, 60), (0, 0, 0), -1)
            out.write(img)
    out.release()


class Histogram:
    model_id = "histogram"

    def images(self, frames):
        out = []
        for f in frames:
            h = cv2.calcHist([f], [0, 1, 2], None, [4, 4, 4], [0, 256] * 3).flatten()
            out.append(h / (np.linalg.norm(h) + 1e-8))
        return np.array(out, np.float32)

    def texts(self, texts):
        # What CLIP does with words, reduced to one: "bright" reads as what
        # bright pixels look like. Anything else is noise.
        out = []
        for i, text in enumerate(texts):
            if "bright" in text:
                out.append(self.images([np.full((8, 8, 3), 255, np.uint8)])[0])
            else:
                v = np.random.default_rng(i + len(text)).normal(size=64)
                out.append(v / np.linalg.norm(v))
        return np.array(out, np.float32)


def test_a_project_from_footage_to_a_dataset(tmp_path, monkeypatch):
    # The rest of the suite runs against conftest's cv2 mock; this test lends
    # the real module to the code under test and hands the mock back after.
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    _video(tmp_path / "a.mp4", [(5, RED, "circle"), (5, GREEN, ""), (5, BLUE, "box"),
                                (5, RED, "circle"), (5, GREEN, ""), (5, BLUE, "box")])
    _video(tmp_path / "b.mp4", [(5, BLUE, "box"), (5, RED, "circle"), (5, GREEN, ""),
                                (5, RED, "circle"), (5, BLUE, "box"), (5, GREEN, "")])
    _video(tmp_path / "ex_red.mp4", [(5, RED, "circle")])
    _video(tmp_path / "ex_blue.mp4", [(5, BLUE, "box")])
    monkeypatch.setattr(cli, "make_embedder", Histogram)
    root = str(tmp_path / "project")

    def run(*args):
        code, result = cli.run(["--project", root, *args])
        assert code == 0, result
        return result

    run("init", "--task", "actions")
    run("add-class", "round thing", "--target", "4")
    run("add-class", "square thing", "--target", "4")
    run("add-video", str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4"))
    assert run("cut")["samples_made"] == 12
    run("add-example", "--class", "round thing", "--clip", str(tmp_path / "ex_red.mp4"))
    run("add-example", "--class", "square thing", "--clip", str(tmp_path / "ex_blue.mp4"))

    sorted_ = run("sort")
    assert sorted_["proposed"] == {"round thing": 4, "square thing": 4, "_none": 4}

    sheet = run("review", "--size", "12")
    assert os.path.getsize(sheet["image"]) > 10_000
    # Every guess is right on this footage, so the whole sheet is accepted.
    run("verdict", "--sheet", str(sheet["sheet"]), "--accept-rest")

    project = Project.load(root)
    for sample in project.samples:
        cap = cv2.VideoCapture(sample.path)
        ok, frame = cap.read()
        cap.release()
        assert ok, sample.id
        colour = tuple(int(c) for c in frame[5, 5])
        expected = ("round thing" if abs(colour[2] - RED[2]) < 40 and colour[0] < 100
                    else "square thing" if colour[0] > 150 else "")
        assert sample.label == expected, (sample.id, colour, sample.label)

    built = run("build")
    for split in ("train", "val"):
        for name in ("round thing", "square thing"):
            assert os.listdir(os.path.join(built["dataset"], split, name))
    # Five accepted per class is below what training is worth running on, and
    # status says so rather than offering to train.
    step = run("status")["next"]
    assert "at least" in step["why"] and "add-video" in step["command"]


def test_an_object_project_from_footage_to_a_coco_dataset(tmp_path, monkeypatch):
    """Cut, sort, review, propose boxes, review boxes, build — real decode."""
    import json as _json

    monkeypatch.setitem(sys.modules, "cv2", cv2)

    from modules.teach import boxes as boxes_mod
    from modules.vision.label_store import ACCEPTED

    # The "thing" is a white square on a blue field, in half the scenes.
    scenes = [(5, BLUE, "square"), (5, GREEN, ""), (5, BLUE, "square"), (5, GREEN, ""),
              (5, BLUE, "square"), (5, GREEN, "")]

    def square_video(path, scenes, fps=15):
        out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 90))
        for seconds, colour, shape in scenes:
            for _ in range(int(seconds * fps)):
                img = np.full((90, 160, 3), colour, np.uint8)
                if shape == "square":
                    cv2.rectangle(img, (100, 20), (140, 60), (255, 255, 255), -1)
                out.write(img)
        out.release()

    square_video(tmp_path / "a.mp4", scenes)
    square_video(tmp_path / "ex.mp4", [(5, BLUE, "square")])

    class Det:
        def __init__(self, box):
            self.class_name, self.confidence, self.class_id = "blob", 0.5, 0
            self.x1, self.y1, self.x2, self.y2 = box

    class Detector:
        """Finds anything bright, the way a detector finds anything it knows."""
        def detect(self, frame):
            mask = (frame.mean(axis=2) > 200).astype(np.uint8)
            ys, xs = np.nonzero(mask)
            if not len(xs):
                return [Det((0, 0, 20, 20))]
            return [Det((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)),
                    Det((0, 0, 20, 20))]

    monkeypatch.setattr(cli, "make_embedder", Histogram)
    monkeypatch.setattr(cli, "make_detector", Detector)
    root = str(tmp_path / "objects")

    def run(*args):
        code, result = cli.run(["--project", root, *args])
        assert code == 0, result
        return result

    run("init", "--task", "objects")
    run("add-class", "bright square", "--target", "3")
    run("add-video", str(tmp_path / "a.mp4"))
    run("cut")
    run("add-example", "--class", "bright square", "--clip", str(tmp_path / "ex.mp4"))
    run("sort")
    sheet = run("review", "--size", "12")
    run("verdict", "--sheet", str(sheet["sheet"]), "--accept-rest")

    project = Project.load(root)
    squares = [s for s in project.samples if s.label == "bright square"]
    assert len(squares) == 4                   # three cut samples + the example

    proposed = run("boxes", "propose")
    assert proposed["proposed"] >= 4
    box_sheet = run("boxes", "review")
    assert os.path.getsize(box_sheet["image"]) > 5_000
    # Judged honestly, by where the box is: around the square, or not.
    good = [str(i["n"]) for i in box_sheet["items"] if 0.55 < i["box"][0] < 0.7]
    bad = [str(i["n"]) for i in box_sheet["items"] if str(i["n"]) not in good]
    assert good
    args = ["boxes", "verdict", "--sheet", str(box_sheet["sheet"]), "--accept", ",".join(good)]
    if bad:
        args += ["--reject", ",".join(bad)]
    run(*args)

    labels = boxes_mod.store(Project.load(root))
    accepted = [b for b in labels.boxes if b.verdict == ACCEPTED]
    assert accepted
    for box in accepted:                       # around the square: x 100-140 of 160
        x, y, w, h = box.box
        assert 0.55 < x < 0.7 and 0.2 < w < 0.35, box.box

    built = run("build")
    with open(os.path.join(built["dataset"], "annotations", "train.json")) as fh:
        coco = _json.load(fh)
    assert coco["annotations"] and coco["categories"][0]["name"] == "bright square"
    assert any(n.endswith(".jpg") for n in os.listdir(os.path.join(built["dataset"], "train")))
