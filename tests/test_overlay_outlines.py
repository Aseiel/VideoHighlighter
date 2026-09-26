"""The player overlay draws a traced outline, not just the box around it.

Skipped where Qt Multimedia cannot load (CI installs only the test
requirements; the overlay module imports the media player at the top).
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtWidgets import QApplication
    from video_ai_editor.realtime_overlay import BBoxOverlayItem, LazyBBoxLoader
except Exception as exc:              # noqa: BLE001 - any missing Qt piece skips
    pytest.skip(f"Qt overlay unavailable: {exc}", allow_module_level=True)

L = [[0.1, 0.1], [0.25, 0.1], [0.25, 0.45], [0.6, 0.45], [0.6, 0.6], [0.1, 0.6]]


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_outlines_travel_from_the_cache_to_the_item(app):
    cache = {"object_bboxes": [{
        "timestamp": 1.0, "objects": ["holder", "thing", "event"],
        "bboxes": [[0.1, 0.1, 0.5, 0.5], [0.45, 0.15, 0.1, 0.1], [0.1, 0.1, 0.5, 0.5]],
        "confidences": [0.9, 0.8, 0.7],
        "contours": [L, [], None],
        "event_contours": [None, None, [L, L]]}]}
    dets = LazyBBoxLoader(cache).get_bboxes_for_time(1.0, 0.5)
    shapes = {d["class_name"]: d["contours"] for d in dets}
    assert shapes == {"holder": [L], "thing": [], "event": [L, L]}


def test_an_item_with_an_outline_draws_the_shape(app):
    item = BBoxOverlayItem((0.1, 0.1, 0.5, 0.5), "holder", 0.9, 1.0, contours=[L])
    item.update_geometry(640, 360)
    (shape,) = item._outline_items
    polygon = shape.polygon()
    assert polygon.count() == len(L)
    assert (polygon.at(3).x(), polygon.at(3).y()) == pytest.approx((0.6 * 640, 0.45 * 360))
    plain = BBoxOverlayItem((0.1, 0.1, 0.5, 0.5), "thing", 0.9, 1.0)
    assert plain._outline_items == []
