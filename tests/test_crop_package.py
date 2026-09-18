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
