"""Rules about body parts (``person.hand``) decided on pose keypoints.

A person's box is the worst stand-in for their hand. These run the real
engine over cache entries carrying keypoints, and the pose pass with a
stand-in estimator.
"""
from __future__ import annotations

import textwrap
import types

import numpy as np

from modules.rules import body_parts
from modules.vision.keypoints import add_keypoints
from modules.vision.pose_backend import KEYPOINT_NAMES
from video_ai_editor.composition_engine import CompositionEngine


def _engine(tmp_path, rule: str, persist=0.0):
    path = tmp_path / "rules.yaml"
    path.write_text(textwrap.dedent(f"""
        events:
          - name: ev
            window_secs: 0
            persist_secs: {persist}
            rules:
              - {rule}
        """), encoding="utf-8")
    return CompositionEngine(str(path))


def _pose(**parts):
    """COCO-17 keypoints, all invisible except the named ones."""
    kps = [[0.0, 0.0, 0.0] for _ in KEYPOINT_NAMES]
    for name, (x, y) in parts.items():
        kps[KEYPOINT_NAMES.index(name)] = [x, y, 0.9]
    return kps


def _frames(pose, n=4, step=1.0):
    return [{"timestamp": i * step, "objects": ["person", "cup"],
             "bboxes": [[0.2, 0.1, 0.5, 0.8], [0.55, 0.4, 0.1, 0.1]],
             "confidences": [0.9, 0.9], "keypoints": [pose, None]} for i in range(n)]


def _fires(engine, frames):
    events, _ = engine.run(frames)
    return any("ev" in v for v in events.values())


def test_the_hand_decides_not_the_persons_box(tmp_path):
    reaching = _pose(right_wrist=(0.6, 0.45))            # inside the cup's box
    elsewhere = _pose(right_wrist=(0.25, 0.5), left_wrist=(0.3, 0.5))
    by_hand = _engine(tmp_path, "{source: person.hand, region: cup, relation: touches}")
    assert _fires(by_hand, _frames(reaching))
    assert not _fires(by_hand, _frames(elsewhere))
    # The box-level rule cannot tell the two apart: the cup is inside the
    # person's box either way.
    by_box = _engine(tmp_path, "{source: cup, region: person}")
    assert _fires(by_box, _frames(reaching)) and _fires(by_box, _frames(elsewhere))


def test_an_invisible_keypoint_is_not_a_hand(tmp_path):
    faint = _pose()
    faint[KEYPOINT_NAMES.index("right_wrist")] = [0.6, 0.45, 0.1]
    engine = _engine(tmp_path, "{source: person.right_wrist, region: cup}")
    assert not _fires(engine, _frames(faint))


def test_a_remembered_hand_is_not_counted_twice(tmp_path):
    # One hand, seen every 0.25 s, remembered for 1 s: with box IoU matching a
    # point never matched itself and a max_count of 1 failed within a second.
    engine = _engine(tmp_path, "{source: person.hand, region: cup, max_count: 1}",
                     persist=1.0)
    frames = _frames(_pose(right_wrist=(0.6, 0.45)), n=12, step=0.25)
    events, _ = engine.run(frames)
    assert sorted(events) == [0, 1, 2]


def test_rules_ask_for_the_detector_class_and_a_pose_pass():
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "r.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("events:\n  - name: e\n    rules:\n"
                     "      - {source: person.hand, region: cup, max_gap: 0.02}\n")
        engine = CompositionEngine(path)
    assert engine.object_classes == ["cup", "person"]
    assert engine.keypoint_pairs == [("person", "cup", 0.02)]


def test_part_names_and_groups():
    assert body_parts.split("person.hand") == ("person", ("left_wrist", "right_wrist"))
    assert body_parts.split("person.nose") == ("person", ("nose",))
    assert body_parts.split("traffic light") == ("traffic light", ())
    assert body_parts.split("a.b") == ("a.b", ())           # not a part: a class name


class FakeEstimator:
    def __init__(self):
        self.calls = 0

    def estimate(self, frame, boxes):
        self.calls += 1
        pts = np.zeros((17, 3), np.float32)
        pts[9] = (60, 45, 0.9)                     # left wrist, in pixels
        return [types.SimpleNamespace(keypoints=pts) for _ in boxes]


def _reader(video, stamps):
    for ts in stamps:
        yield ts, np.zeros((90, 160, 3), np.uint8)


def test_the_pose_pass_estimates_people_near_the_other_class_once():
    cache = [{"timestamp": 0.0, "objects": ["person", "cup", "person"],
              "bboxes": [[0.2, 0.1, 0.5, 0.8], [0.55, 0.4, 0.1, 0.1], [0.0, 0.0, 0.05, 0.05]],
              "confidences": [0.9, 0.9, 0.9]}]
    est = FakeEstimator()
    stats = add_keypoints("v.mp4", cache, [("person", "cup", 0.0)], est, frame_reader=_reader)
    assert stats == {"frames": 1, "people": 1, "no_pose": 0}
    kps = cache[0]["keypoints"]
    assert kps[1] is None and kps[2] is None          # the cup, and the far person
    assert kps[0][9] == [0.375, 0.5, 0.9]              # normalised by the frame
    again = add_keypoints("v.mp4", cache, [("person", "cup", 0.0)], est, frame_reader=_reader)
    assert again["frames"] == 0 and est.calls == 1
