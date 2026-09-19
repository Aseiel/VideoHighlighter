"""The round-by-round view of a model that is still learning."""
import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pytest

# conftest is loaded by pytest rather than imported, so its helpers are not on
# the path by default.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import real_opencv          # noqa: E402

from modules.vision import training_preview as tp  # noqa: E402


@dataclass
class Det:
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int = 0


def test_a_good_guess_is_found_and_a_stray_one_is_a_false_alarm():
    truth = [("thing", 10, 10, 50, 50)]
    dets = [Det("thing", 0.9, 12, 11, 52, 49), Det("thing", 0.8, 200, 200, 240, 240)]
    found, expected, alarms, hits = tp.match(dets, truth)
    assert (found, expected, alarms) == (1, 1, 1)
    assert hits == [True, False]


def test_a_box_is_claimed_once_by_the_most_confident_guess():
    truth = [("thing", 10, 10, 50, 50)]
    dets = [Det("thing", 0.4, 10, 10, 50, 50), Det("thing", 0.9, 11, 11, 51, 51)]
    found, _, alarms, hits = tp.match(dets, truth)
    assert found == 1 and alarms == 1
    assert hits == [False, True]


def test_the_right_place_with_the_wrong_class_is_not_found():
    found, expected, alarms, _ = tp.match([Det("other", 0.9, 10, 10, 50, 50)],
                                          [("thing", 10, 10, 50, 50)])
    assert (found, expected, alarms) == (0, 1, 1)


def test_an_empty_frame_with_no_guesses_is_clean():
    assert tp.match([], [])[:3] == (0, 0, 0)


def _snap(epoch, found, expected, history=(), alarms=0, held_out=True):
    return tp.RoundSnapshot(epoch=epoch, total_epochs=30, found=found, expected=expected,
                            false_alarms=alarms, held_out=held_out, history=list(history))


def test_the_sentence_says_what_it_found_and_compares_with_the_first_round():
    s = _snap(12, 5, 8, history=[(1, 0, 8), (6, 3, 8), (12, 5, 8)], alarms=2).sentence()
    assert "Round 12 of 30: found 5 of 8 on frames it has not learned from" in s
    assert "2 wrong guesses" in s
    assert "(round 1: 0 of 8)" in s


def test_the_sentence_admits_when_frames_were_seen_in_training():
    assert "learned from (no held-out" in _snap(3, 2, 4, held_out=False).sentence()


class _FixedDetector:
    def __init__(self, answers):
        self.answers = list(answers)

    def detect(self, _image):
        return self.answers.pop(0)


def test_evaluate_and_snapshot_accumulate_history():
    frames = [tp.PreviewFrame(np.zeros((90, 160, 3), np.uint8), [("thing", 10, 10, 40, 40)]),
              tp.PreviewFrame(np.zeros((90, 160, 3), np.uint8), [])]
    history = []

    @dataclass
    class Report:
        epoch: int
        total_epochs: int = 5
        train_loss: float = 1.0
        val_loss: float = 1.0
        detector: object = None

    r1 = Report(1, detector=_FixedDetector([[], [Det("thing", 0.9, 1, 1, 5, 5)]]))
    s1 = tp.snapshot(r1, frames, history, draw=False)
    assert (s1.found, s1.expected, s1.false_alarms) == (0, 1, 1)
    assert s1.mosaic_rgb is None

    r2 = Report(2, detector=_FixedDetector([[Det("thing", 0.8, 11, 10, 41, 40)], []]))
    s2 = tp.snapshot(r2, frames, history, draw=False)
    assert (s2.found, s2.false_alarms) == (1, 0)
    assert s2.history == [(1, 0, 1), (2, 1, 1)]


def test_low_confidence_guesses_are_not_counted():
    frames = [tp.PreviewFrame(np.zeros((10, 10, 3), np.uint8), [("thing", 0, 0, 5, 5)])]
    found, _, alarms, _ = tp.evaluate(_FixedDetector([[Det("thing", 0.1, 0, 0, 5, 5)]]), frames)
    assert found == 0 and alarms == 0


# --------------------------------------------------------------------------- #
# needs real pixels
# --------------------------------------------------------------------------- #
@pytest.fixture
def cv2_real(monkeypatch):
    cv2 = real_opencv()
    if cv2 is None:
        pytest.skip("OpenCV not installed")
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    return cv2


def test_render_draws_a_mosaic_of_every_frame(cv2_real):
    frames = [tp.PreviewFrame(np.full((360, 640, 3), 90, np.uint8), [("thing", 100, 100, 300, 250)])
              for _ in range(5)]
    per_frame = [([Det("thing", 0.7, 110, 105, 290, 240)], [True])] * 5
    rgb = tp.render(frames, per_frame, caption="Round 1 of 3: found 5 of 5 ✓")
    cols, rows = 3, 2
    assert rgb.shape == (26 + rows * tp.CELL_SIZE[1] + (rows - 1) * tp.GAP,
                         cols * tp.CELL_SIZE[0] + (cols - 1) * tp.GAP, 3)
    assert rgb.dtype == np.uint8
    # a green hit box is present somewhere in the picture
    g = np.array(tp.HIT_COLOUR[::-1])
    assert np.any(np.all(np.abs(rgb.astype(int) - g) < 30, axis=2))


def test_pick_frames_prefers_held_out_frames_with_boxes(cv2_real, tmp_path):
    for split in ("train", "val"):
        (tmp_path / split).mkdir()
    (tmp_path / "annotations").mkdir()
    images, anns = [], []
    for i in range(10):
        name = f"f{i:02d}.jpg"
        cv2_real.imwrite(str(tmp_path / "val" / name), np.full((60, 80, 3), i * 20, np.uint8))
        images.append({"id": i + 1, "file_name": name, "width": 80, "height": 60})
        if i < 8:
            anns.append({"id": i + 1, "image_id": i + 1, "category_id": 1,
                         "bbox": [5, 5, 20, 20], "area": 400, "iscrowd": 0})
    doc = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "thing"}]}
    (tmp_path / "annotations" / "val.json").write_text(json.dumps(doc), encoding="utf-8")

    frames = tp.pick_frames(str(tmp_path), limit=6)
    assert len(frames) == 6
    assert all(f.held_out for f in frames)
    assert sum(1 for f in frames if f.truth) == 5
    assert sum(1 for f in frames if not f.truth) == 1
    assert frames[0].truth[0] == ("thing", 5, 5, 25, 25)


def test_pick_frames_with_no_dataset_is_empty(tmp_path):
    assert tp.pick_frames(str(tmp_path)) == []
