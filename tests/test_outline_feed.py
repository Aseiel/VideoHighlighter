"""The overlay's shape switch: outlines traced in the background, nearest the
playhead first, written into the cache, and drawn in place of boxes.

Skipped where Qt cannot load (as test_overlay_outlines.py).
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtWidgets import QApplication
    from video_ai_editor.outline_feed import OutlineFeed
    from video_ai_editor.realtime_overlay import (
        LazyBBoxLoader, OverlayScene, missing_outlines, store_outlines,
    )
except Exception as exc:              # noqa: BLE001 - any missing Qt piece skips
    pytest.skip(f"Qt overlay unavailable: {exc}", allow_module_level=True)

SQUARE = [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class Frames:
    """Frames by timestamp, recording which were asked for."""

    def __init__(self, missing=()):
        self.asked, self.missing = [], set(missing)

    def at(self, timestamp):
        self.asked.append(timestamp)
        return None if timestamp in self.missing else np.zeros((40, 80, 3), np.uint8)


class Outliner:
    name = "fake"

    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def outline(self, frame, boxes_px):
        if self.fail:
            raise RuntimeError("device refused the model")
        self.calls.append(list(boxes_px))
        return [SQUARE if i % 2 == 0 else None for i in range(len(boxes_px))]


def _feed(frames=None, outliner=None, fallback=None):
    out = []
    feed = OutlineFeed(frames=frames or Frames(), start=False,
                       outliner_factory=lambda: outliner or Outliner(),
                       fallback_factory=lambda: fallback or Outliner())
    feed.traced.connect(lambda key, outlines: out.append((key, outlines)))
    feed.status.connect(lambda text: out.append(("status", text)))
    return feed, out


def test_the_job_nearest_the_playhead_goes_first(app):
    feed, out = _feed()
    for key, t in (("far", 9.0), ("near", 5.2), ("mid", 6.5)):
        assert feed.submit(key, [[0.1, 0.1, 0.2, 0.2]], t)
    feed.set_playhead(5.0)
    while feed.run_one():
        pass
    feed.drain()
    assert [k for k, _ in out if k != "status"] == ["near", "mid", "far"]


def test_jobs_the_playhead_left_behind_are_dropped_and_can_come_back(app):
    feed, out = _feed()
    feed.submit("old", [[0, 0, 1, 1]], 1.0)
    feed.submit("new", [[0, 0, 1, 1]], 9.0)
    feed.set_playhead(9.0)
    assert feed.pending() == 1
    assert feed.submit("old", [[0, 0, 1, 1]], 1.0)       # seeking back re-queues it
    assert not feed.submit("old", [[0, 0, 1, 1]], 1.0)   # but only once


def test_outlines_come_back_aligned_with_the_boxes_in_pixels(app):
    frames = Frames()
    outliner = Outliner()
    feed, out = _feed(frames, outliner)
    feed.submit("k", [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 0.25, 0.25]], 3.0)
    feed.run_one()
    feed.drain()
    assert frames.asked == [3.0]                       # the detector's own frame
    assert outliner.calls == [[(0.0, 0.0, 40.0, 20.0), (40.0, 20.0, 60.0, 30.0)]]
    assert ("k", [SQUARE, []]) in out                  # None -> [] (tried, nothing)
    assert feed.traced_count == 1


def test_a_frame_that_cannot_be_read_gives_empty_outlines(app):
    feed, out = _feed(Frames(missing={2.0}))
    feed.submit("k", [[0, 0, 1, 1]], 2.0)
    feed.run_one()
    feed.drain()
    assert ("k", [[]]) in out


def test_sam_that_cannot_load_falls_back_to_grabcut(app):
    def broken():
        raise OSError("no weights and no network")

    feed = OutlineFeed(frames=Frames(), start=False, outliner_factory=broken,
                       fallback_factory=lambda: Outliner())
    out = []
    feed.status.connect(out.append)
    feed.submit("k", [[0, 0, 1, 1]], 1.0)
    feed.run_one()
    feed.drain()
    assert any("SAM unavailable" in s for s in out)


def test_sam_that_fails_on_a_frame_falls_back_for_that_frame_and_after(app):
    fallback = Outliner()
    feed, out = _feed(outliner=Outliner(fail=True), fallback=fallback)
    feed.submit("a", [[0, 0, 1, 1]], 1.0)
    feed.submit("b", [[0, 0, 1, 1]], 1.1)
    feed.run_one()
    feed.run_one()
    feed.drain()
    assert len(fallback.calls) == 2
    assert [o for k, o in out if k in ("a", "b")] == [[SQUARE], [SQUARE]]


def test_warm_up_loads_the_model_before_any_job(app):
    loaded = []

    class Slow(Outliner):
        def _load(self):
            loaded.append(True)

    feed, out = _feed(outliner=Slow())
    assert not feed.run_one()                  # nothing asked for: nothing done
    feed.warm_up()
    assert feed.run_one() and loaded == [True]
    feed.drain()
    assert [t for k, t in out if k == "status"] == ["outlines: loading model…", "outlines: fake"]
    assert not feed.run_one()                  # loaded once


def test_the_worker_thread_runs_jobs(app):
    import time

    feed, out = _feed()
    feed.start()
    try:
        feed.submit("k", [[0, 0, 1, 1]], 1.0)
        deadline = time.time() + 5
        while not any(k == "k" for k, _ in out) and time.time() < deadline:
            feed.drain()
            time.sleep(0.02)
    finally:
        feed.stop()
    assert ("k", [SQUARE]) in out


def test_an_outline_relative_to_its_box_follows_the_box():
    from video_ai_editor.outline_feed import relative_to_box

    rel = relative_to_box(SQUARE, [0.1, 0.1, 0.2, 0.1])
    assert rel == [pytest.approx(p) for p in [(0, 0), (0.5, 0), (0.5, 1), (0, 1)]]
    assert relative_to_box([], [0, 0, 1, 1]) == []
    assert relative_to_box(SQUARE, [0, 0, 0, 1]) == []


# --------------------------------------------------------------- the cache side

def _entry():
    return {"timestamp": 2.0, "objects": ["a", "b", "c", "d"],
            "bboxes": [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0]],
            "confidences": [0.9] * 4, "contours": [SQUARE, [], None]}


def test_only_detections_without_an_outline_are_asked_for():
    entry = _entry()
    # a has one, b was tried, c has none, d has a broken box
    assert missing_outlines(entry) == [2]
    assert missing_outlines(entry, hidden={"c"}) == []


def test_traced_outlines_land_in_the_entry_without_replacing_any():
    entry = _entry()
    entry["objects"].append("e")
    entry["bboxes"].append([0, 0, 1, 1])
    store_outlines(entry, [0, 2, 4], [[[9, 9]] * 3, SQUARE, None])
    assert entry["contours"] == [SQUARE, [], SQUARE, None, []]


def test_the_loader_finds_entries_near_the_playhead_nearest_first():
    cache = {"object_bboxes": [dict(_entry(), timestamp=t) for t in (0.0, 4.0, 5.5, 6.0, 9.0)]}
    loader = LazyBBoxLoader(cache)
    near = loader.object_entries_near(5.0, 0.5, 2.0)
    assert [e["timestamp"] for e in near] == [5.5, 6.0]
    assert near[0] is cache["object_bboxes"][2]           # the entry itself


def test_the_switch_draws_boxes_or_outlines(app):
    entry = dict(_entry(), contours=[SQUARE, None, None])
    scene = OverlayScene()
    scene.load_detections_lazy(LazyBBoxLoader({"object_bboxes": [entry]}))
    scene.show_outlines = False
    scene.update_time(2.0, 0.5)
    assert not any(it._outline_items for it in scene._all_bbox_items)
    scene.set_show_outlines(True)
    assert scene._all_bbox_items == []                    # rebuilt on the next update
    scene.update_time(2.0, 0.5)
    assert sum(bool(it._outline_items) for it in scene._all_bbox_items) == 1


def test_new_outlines_redraw_that_moment(app):
    entry = dict(_entry(), contours=[None, None, None])
    scene = OverlayScene()
    scene.load_detections_lazy(LazyBBoxLoader({"object_bboxes": [entry]}))
    scene.update_time(2.0, 0.5)
    assert not any(it._outline_items for it in scene._all_bbox_items)
    store_outlines(entry, [1], [SQUARE])
    scene.drop_timestamp(2.0)
    scene.update_time(2.0, 0.5)
    assert sum(bool(it._outline_items) for it in scene._all_bbox_items) == 1
