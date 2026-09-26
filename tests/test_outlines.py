"""Outlines on demand for composition rules (modules/vision/outlines.py).

The pass must trace only what a rule could need, never twice, and keep the
cache's lists aligned; GrabCut must find the shape inside a box. The last is
checked on real pixels where OpenCV is installed (conftest mocks it in CI).
"""
from __future__ import annotations

import os
import sys
import textwrap

import numpy as np
import pytest

from modules.report.analysis_ondemand import _without_classes
from modules.vision import outlines
from video_ai_editor.composition_engine import CompositionEngine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conftest import real_opencv  # noqa: E402


def _entry(ts, dets):
    return {"timestamp": ts, "objects": [d[0] for d in dets],
            "bboxes": [d[1] for d in dets], "confidences": [0.9] * len(dets)}


PAIRS = [("thing", "holder", 0.0)]


def test_only_detections_whose_boxes_meet_their_partner_are_traced():
    entry = _entry(0.0, [("holder", [0.1, 0.1, 0.3, 0.3]),
                         ("thing", [0.2, 0.2, 0.05, 0.05]),     # meets the holder
                         ("thing", [0.8, 0.8, 0.05, 0.05]),     # nowhere near it
                         ("other", [0.2, 0.2, 0.1, 0.1])])      # not in any rule
    assert outlines.wanted(entry, PAIRS) == {0, 1}
    assert outlines.wanted(entry, [("thing", "holder", 0.5)]) == {0, 1, 2}


class FakeOutliner:
    name = "fake"

    def __init__(self):
        self.calls = []

    def outline(self, frame, boxes_px):
        self.calls.append(len(boxes_px))
        return [[[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]] if i == 0 else None
                for i, _ in enumerate(boxes_px)]


def _frames(video, stamps):
    for ts in stamps:
        yield ts, np.zeros((100, 200, 3), np.uint8)


def test_the_pass_traces_once_and_keeps_lists_aligned():
    cache = [_entry(0.0, [("other", [0, 0, 1, 1]), ("holder", [0.1, 0.1, 0.3, 0.3]),
                          ("thing", [0.2, 0.2, 0.05, 0.05])]),
             _entry(1.0, [("thing", [0.8, 0.8, 0.05, 0.05])])]
    fake = FakeOutliner()
    stats = outlines.add_outlines("v.mp4", cache, PAIRS, outliner=fake, frame_reader=_frames)
    assert stats == {"frames": 1, "traced": 1, "no_outline": 1, "outliner": "fake"}
    contours = cache[0]["contours"]
    assert len(contours) == 3 and contours[0] is None
    assert contours[1] and contours[2] == []          # [] = tried, found nothing
    assert "contours" not in cache[1]

    again = outlines.add_outlines("v.mp4", cache, PAIRS, outliner=fake, frame_reader=_frames)
    assert again["frames"] == 0 and fake.calls == [2]


def test_removing_a_class_keeps_outlines_on_their_own_boxes():
    entry = _entry(0.0, [("a", [0, 0, 0.1, 0.1]), ("b", [0.5, 0.5, 0.1, 0.1])])
    entry["contours"] = [[[0, 0], [0.1, 0], [0.1, 0.1]], [[0.5, 0.5], [0.6, 0.5], [0.6, 0.6]]]
    trimmed = _without_classes([entry], {"a"})[0]
    assert trimmed["objects"] == ["b"]
    assert trimmed["contours"] == [[[0.5, 0.5], [0.6, 0.5], [0.6, 0.6]]]


def test_the_rules_file_says_what_to_trace_and_with_what(tmp_path):
    path = tmp_path / "rules.yaml"
    path.write_text(textwrap.dedent("""
        outliner: sam
        events:
          - name: e
            rules:
              - {source: thing, region: holder, relation: touches, max_gap: 0.02, outline: true}
              - {source: x, region: y}
          - name: off
            enabled: false
            rules:
              - {source: p, region: q, outline: true}
        """), encoding="utf-8")
    engine = CompositionEngine(str(path))
    assert engine.outline_pairs == [("thing", "holder", 0.02)]
    assert engine.outliner == "sam"
    path.write_text("outliner: magic\nevents: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outliner"):
        CompositionEngine(str(path))


# --- on real pixels ----------------------------------------------------------------------

cv2_real = real_opencv()
needs_cv2 = pytest.mark.skipif(cv2_real is None, reason="real OpenCV is not installed")


@pytest.fixture
def cv2(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", cv2_real)
    return cv2_real


@needs_cv2
def test_grabcut_finds_the_shape_inside_the_box(cv2):
    from modules.rules import shapes

    rng = np.random.default_rng(0)
    frame = cv2.GaussianBlur(rng.integers(60, 140, (360, 640, 3)).astype(np.uint8), (0, 0), 3)
    truth = np.zeros((360, 640), np.uint8)
    # An L: most of its bounding box is empty.
    cv2.fillPoly(truth, [np.array([[200, 60], [260, 60], [260, 220], [420, 220],
                                   [420, 280], [200, 280]])], 1)
    frame[truth == 1] = (170, 70, 120)
    (contour,) = outlines.GrabCutOutliner().outline(frame, [(200, 60, 420, 280)])
    assert contour and len(contour) < 40

    drawn = np.zeros_like(truth)
    cv2.fillPoly(drawn, [np.array([[x * 640, y * 360] for x, y in contour], np.int32)], 1)
    iou = (drawn & truth).sum() / (drawn | truth).sum()
    assert iou > 0.9
    # The empty corner of the box is outside the outline.
    assert not shapes.contains(contour, (400 / 640, 80 / 360))


@needs_cv2
def test_nothing_to_find_leaves_the_box(cv2):
    flat = np.full((200, 200, 3), 128, np.uint8)
    assert outlines.GrabCutOutliner().outline(flat, [(50, 50, 150, 150)]) == [None]


@needs_cv2
def test_masks_become_normalised_outlines(cv2):
    mask = np.zeros((100, 200), np.uint8)
    mask[20:60, 50:150] = 1
    contour = outlines.mask_to_contour(mask)
    xs, ys = [p[0] for p in contour], [p[1] for p in contour]
    assert min(xs) == pytest.approx(0.25) and max(ys) == pytest.approx(0.59)


@needs_cv2
def test_sam_glue_turns_box_prompts_into_outlines(cv2, monkeypatch):
    """The model itself cannot be fetched in CI; this pins what is ours: boxes
    in the shape the transformers SAM processor takes, masks out as outlines,
    a near-empty mask treated as no outline."""
    import contextlib
    import types

    class T:                                   # a tensor, as far as the glue cares
        def __init__(self, a):
            self.a = np.asarray(a)

        def cpu(self):
            return self

        def numpy(self):
            return self.a

        def to(self, device):
            return self

        def __getitem__(self, i):
            return T(self.a[i])

        def __iter__(self):
            return (T(row) for row in self.a)

    seen = {}

    class Processor:
        def __call__(self, images, input_boxes, return_tensors):
            seen["boxes"] = input_boxes
            return {"pixel_values": T(0), "original_sizes": T([[100, 200]]),
                    "reshaped_input_sizes": T([[100, 200]])}

        class image_processor:
            @staticmethod
            def post_process_masks(pred, original, reshaped):
                full = np.zeros((2, 1, 100, 200), bool)
                full[0, 0, 10:50, 20:80] = True        # box 1: a real mask
                full[1, 0, 0, 0] = True                # box 2: next to nothing
                return [T(full)]                        # one image: [boxes, 1, H, W]

    class Model:
        def __call__(self, **kw):
            seen["multimask"] = kw.get("multimask_output")
            return types.SimpleNamespace(pred_masks=T(0))

    fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    sam = outlines.SamOutliner()
    sam._model, sam._processor, sam.device = Model(), Processor(), "cpu"
    result = sam.outline(np.zeros((100, 200, 3), np.uint8),
                         [(20, 10, 80, 50), (100, 60, 150, 90)])
    assert seen["boxes"] == [[[20.0, 10.0, 80.0, 50.0], [100.0, 60.0, 150.0, 90.0]]]
    assert seen["multimask"] is False
    assert result[1] is None
    xs = [p[0] for p in result[0]]
    assert min(xs) == pytest.approx(0.1) and max(xs) == pytest.approx(0.395)


def test_the_pipeline_path_traces_on_copies_and_reports_it(tmp_path, monkeypatch):
    from modules.rules import compose_events

    path = tmp_path / "rules.yaml"
    path.write_text(textwrap.dedent("""
        events:
          - name: e
            window_secs: 0
            persist_secs: 0
            rules:
              - {source: thing, region: holder, relation: overlaps, min_overlap: 0.9,
                 outline: true}
        """), encoding="utf-8")
    cache = [_entry(float(t), [("holder", [0.0, 0.0, 0.6, 0.6]),
                               ("thing", [0.45, 0.05, 0.1, 0.1])]) for t in range(3)]

    def fake_add(video, boxes, pairs, outliner=None, **kw):
        for frame in boxes:     # the holder is an L; the thing is in its empty corner
            frame["contours"] = [[[0, 0], [0.2, 0], [0.2, 0.4], [0.6, 0.4],
                                  [0.6, 0.6], [0, 0.6]], []]
        return {"frames": len(boxes), "traced": len(boxes), "no_outline": len(boxes),
                "outliner": "fake"}

    monkeypatch.setattr(outlines, "add_outlines", fake_add)
    stats = {}
    _, boxes, _, hits = compose_events.apply_rules(
        {}, cache, rules_path=str(path), video_path="v.mp4", outline_stats=stats,
        log_fn=lambda *a: None)
    assert hits == 0                          # boxes alone would have said yes
    assert stats["frames"] == 3
    assert "contours" not in cache[0]         # the caller's cache is untouched
    assert boxes[0]["contours"]

    _, _, _, hits = compose_events.apply_rules({}, cache, rules_path=str(path),
                                               log_fn=lambda *a: None)
    assert hits == 3                          # no video: decided on boxes, as before
