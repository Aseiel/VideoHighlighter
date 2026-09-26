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
        v = np.random.default_rng(len(texts)).normal(size=(len(texts), 64))
        return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


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
