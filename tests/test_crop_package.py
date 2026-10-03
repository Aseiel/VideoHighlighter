"""Structural + pure-geometry tests for modules/crop.

The package was split out of a single 3,899-line crop_actions.py that had no
test at all, which is exactly what made the split risky. These are cheap and
deliberately narrow: they pin the things a bad split silently breaks — module
boundaries, import purity, and the box math every crop window depends on.

No Qt, no model, no video: CI has neither PySide6 nor an OpenVINO IR, and a
module-level import of either would fail collection rather than skip.
"""
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import real_opencv  # noqa: E402

MODULES = ["core", "config", "pose", "debug", "people", "zones", "track",
           "actions", "avoid"]


@pytest.mark.parametrize("name", MODULES)
def test_every_module_imports(name):
    """No cycles, and no import-time dependency on a model or a display."""
    importlib.import_module(f"modules.crop.{name}")


def test_importing_the_package_touches_no_disk(tmp_path, monkeypatch):
    """crop_actions.py used to run os.makedirs() at module level, so merely
    importing it created output_videos/ and debug_visualizations/ wherever the
    process happened to be. Those calls belong to main(), not to import."""
    monkeypatch.chdir(tmp_path)
    for name in MODULES:
        importlib.reload(importlib.import_module(f"modules.crop.{name}"))
    assert list(tmp_path.iterdir()) == []


def test_layering_is_one_directional():
    """core knows about neither objective; pose is a leaf below the consumers
    that guard on it. A new import that inverts either is a design change, and
    should fail here rather than at runtime as a circular import."""
    import ast
    import pathlib

    def imports_of(name):
        src = pathlib.Path(f"modules/crop/{name}.py").read_text(encoding="utf-8")
        return {n.module for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("modules.crop")}

    assert imports_of("core") == set()
    assert imports_of("pose") <= {"modules.crop.core", "modules.crop.config"}
    assert "modules.crop.actions" not in imports_of("track")
    assert "modules.crop.actions" not in imports_of("zones")


def test_a_called_cropper_never_asks_to_delete(monkeypatch):
    """modules/teach calls main() from the app, where there is no console: the
    windowed exe has no stdin, so a question raised after every crop was made
    and before focus could record them. Called, it must not ask -- only the
    command line does."""
    from modules.crop import actions

    seen = []
    monkeypatch.setattr(actions, "_run_batch", lambda **kw: seen.append(kw))
    actions.main(input_folder="in", output_folder="out", debug=False)
    assert seen == [{"ask_delete": False}]


def test_keeping_the_originals_never_reads_the_console(tmp_path, monkeypatch):
    from modules.crop import actions

    original = tmp_path / "clip.mp4"
    original.write_bytes(b"x")

    def no_console(*_):
        raise AssertionError("input() called")

    monkeypatch.setattr("builtins.input", no_console)
    actions._offer_to_delete([str(original)], ask=False)
    assert original.exists()


def test_the_command_line_deletes_only_on_yes(tmp_path, monkeypatch):
    from modules.crop import actions

    kept, gone = tmp_path / "kept.mp4", tmp_path / "gone.mp4"
    kept.write_bytes(b"x")
    gone.write_bytes(b"x")

    monkeypatch.setattr("builtins.input", lambda *_: "n")
    actions._offer_to_delete([str(kept)], ask=True)
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    actions._offer_to_delete([str(gone)], ask=True)

    assert kept.exists()
    assert not gone.exists()


def test_detector_score_floor_is_below_every_tuned_threshold():
    """The conf= arguments the cropper passes are applied after inference; the
    detector's own score_thr decides what exists at all. If the floor ever rises
    above a tuned threshold, that threshold silently stops meaning anything."""
    from modules.crop import config as cfg

    tuned = [cfg.PERSON_DETECTION_CONF, cfg.PERSON_DETECTION_CONF_ZONES,
             cfg.PERSON_DETECTION_CONF_TRACKING, cfg.ROI_CONFIDENCE_THRESHOLD]
    assert cfg.DETECTOR_SCORE_FLOOR < min(tuned)


def test_iou_of_disjoint_and_identical_boxes():
    from modules.crop.core import calculate_iou

    assert calculate_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert calculate_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert calculate_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_pad_to_size_letterboxes_without_distorting(monkeypatch):
    """Real pixels: the suite shims cv2 with a MagicMock, under which every
    resize is vacuous. Borrow the real module the way conftest documents."""
    import numpy as np

    cv2 = real_opencv()
    if cv2 is None:
        pytest.skip("OpenCV not installed")

    from modules.crop import core

    monkeypatch.setattr(core, "cv2", cv2)
    # 2:1 source into a square target -> scaled to fit width, bars top and bottom.
    out = core.pad_to_size(np.zeros((50, 100, 3), dtype=np.uint8), (200, 200))
    assert out.shape == (200, 200, 3)
    assert out[0, 0].tolist() == [0, 0, 0]


def test_pose_is_inert_without_a_model():
    """The contract the rest of the package is written against: no pose model
    means empty keypoints and validation that abstains, never an exception and
    never a silently dropped detection. The app shipped without a pose model for
    a long time, and every call site still has to tolerate that."""
    from modules.crop.pose import bbox_has_pose_support, get_pose_keypoints_for_frame

    assert get_pose_keypoints_for_frame(None, None) == []
    assert bbox_has_pose_support((0, 0, 10, 10), []) is True


def test_pose_without_boxes_is_empty_not_whole_frame():
    """RTMPose is top-down. Asking for keypoints with no boxes is not 'find
    everyone', it is a question with no subject — and answering [] is what keeps
    a caller that forgot to pass boxes from silently counting zero people while
    believing pose ran."""
    from modules.crop.pose import get_pose_keypoints_for_frame

    sentinel = object()  # never called: the guard returns before touching it
    assert get_pose_keypoints_for_frame(None, sentinel) == []
    assert get_pose_keypoints_for_frame(None, sentinel, person_boxes=[]) == []


def test_keypoint_clustering_takes_a_list_of_per_person_arrays():
    """The top-down backend hands over a list of [K, 3] arrays, one per box,
    where the old whole-frame model produced a single [N, K, 3] tensor. The
    clusterer is shared between them, so pin the shape it now receives."""
    import numpy as np

    from modules.crop.pose import cluster_keypoints_by_person

    def person(x, y):
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[:, 0], kp[:, 1], kp[:, 2] = x, y, 0.9
        return kp

    far = cluster_keypoints_by_person([person(10, 10), person(500, 400)], radius=100)
    near = cluster_keypoints_by_person([person(10, 10), person(30, 30)], radius=100)

    assert len(far) == 2
    assert len(near) == 1


# --- strategy and slot planning, on fake frames and a fake detector ----------

class _Box:
    def __init__(self, xyxy, conf=0.9):
        import numpy as np
        self.xyxy = [np.array(xyxy, dtype=float)]
        self.conf = conf


class _Result:
    def __init__(self, boxes):
        self.boxes = [_Box(b) for b in boxes]


class _Detector:
    """predict() answers with the same boxes on every frame, or per frame."""
    def __init__(self, boxes_for):
        self.boxes_for, self.calls = boxes_for, 0

    def predict(self, frame, **_):
        self.calls += 1
        return [_Result(self.boxes_for(self.calls))]


def _fake_cv2(frames=30, width=600, height=300):
    import types

    import numpy as np

    class Cap:
        def __init__(self, _path):
            self.pos = 0

        def get(self, prop):
            return {7: frames, 3: width, 4: height}.get(prop, 0)

        def set(self, _prop, value):
            self.pos = int(value)

        def read(self):
            if self.pos >= frames:
                return False, None
            self.pos += 1
            return True, np.zeros((height, width, 3), dtype=np.uint8)

        def release(self):
            pass

    return types.SimpleNamespace(
        VideoCapture=Cap, cvtColor=lambda f, _c: f, COLOR_BGR2RGB=4, COLOR_RGB2BGR=4,
        CAP_PROP_FRAME_COUNT=7, CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4,
        CAP_PROP_POS_FRAMES=1)


def test_zone_analysis_sees_the_detected_people(monkeypatch):
    """The box-collecting loop in analyze_region_activity was once replaced by a
    placeholder comment, so every zone saw 0 people for months and every 2-3
    person clip fell back to left + centre. Pin that people are seen, and that
    each counts in the one third holding their centre — counting every third a
    box overlaps put most people in the centre too, so the centre always won."""
    from modules.crop import zones

    monkeypatch.setattr(zones, "cv2", _fake_cv2())
    left_person = (20, 20, 160, 290)      # centre x=90, inside the left third
    right_person = (440, 20, 580, 290)    # centre x=510, inside the right third
    det = _Detector(lambda _n: [left_person, right_person])

    _, people, _, _ = zones.analyze_region_activity("clip.mp4", det, None, sample_frames=5)

    assert people["left"] == [1] * 5
    assert people["center"] == [0] * 5
    assert people["right"] == [1] * 5


def test_only_boxes_with_upper_body_are_people(monkeypatch):
    """A leg boxed on its own (its owner out of frame) holds no action and must
    not get a crop. Top-down pose returns no head/shoulder/elbow joints for it."""
    import numpy as np

    from modules.crop import track

    def pose_for(box, upper):
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[[5, 6, 7] if upper else [13, 14, 15], 2] = 0.9
        return {"bbox": box, "keypoints": kp}

    body, leg = (100, 0, 300, 300), (0, 100, 60, 300)
    monkeypatch.setattr(track, "get_pose_keypoints_for_frame",
                        lambda *a, **k: [pose_for(body, True), pose_for(leg, False)])

    assert track.person_like_boxes(None, [body, leg], pose_model=object()) == [body]
    assert track.person_like_boxes(None, [body, leg], pose_model=None) == [body, leg]


def test_slot_plan_drops_fragments_and_empty_slots(monkeypatch):
    """Slots are fixed from the whole clip: a small box (a hand) is not a
    person's crop, and a slot nobody stands in is dropped rather than locked
    on whatever the first frames showed."""
    from modules.crop import track

    monkeypatch.setattr(track, "cv2", _fake_cv2(frames=30))
    big_left, big_right = (20, 10, 180, 290), (420, 10, 580, 290)
    hand_middle = (280, 120, 320, 160)
    det = _Detector(lambda _n: [big_left, big_right, hand_middle])

    plan = track.plan_slots("clip.mp4", det, None, ["left", "middle", "right"])

    assert plan["middle"] is None
    assert plan["left"] == big_left and plan["right"] == big_right
