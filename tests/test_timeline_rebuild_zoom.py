"""A rebuild of the timeline keeps the zoom the view had.

It used to square it: each rebuild multiplied the view's horizontal zoom by
itself, so after a few filter toggles a zoomed-out video was a few pixels wide
and its bars were gone from the timeline, and a zoomed-in one had run past
everything on it.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from video_ai_editor.signal_timeline import (  # noqa: E402
    SignalTimelineScene, SignalTimelineView)


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture()
def view(app):
    scene = SignalTimelineScene(cache_data={}, video_duration=600.0)
    view = SignalTimelineView(scene)
    view.resize(1000, 300)
    view.show()
    app.processEvents()
    yield view
    view.close()


def _zoom(view, m11):
    t = view.transform()
    view.setTransform(t.__class__(m11, 0, 0, t.m22(), 0, 0))


def _on_screen(view):
    """Pixels per second of video, as drawn."""
    return view.scene().pixels_per_second * view.transform().m11()


@pytest.mark.parametrize("m11", [0.5, 1.0, 2.0])
def test_rebuilding_keeps_the_zoom(app, view, m11):
    _zoom(view, m11)
    for _ in range(4):
        view.scene().build_timeline()
        app.processEvents()
    assert view.transform().m11() == pytest.approx(m11)


def test_rebuilding_keeps_the_scroll_position(app, view):
    _zoom(view, 2.0)
    bar = view.horizontalScrollBar()
    bar.setValue(bar.maximum() // 3)
    where = bar.value()
    view.scene().build_timeline()
    app.processEvents()
    assert bar.value() == where


def test_a_new_scene_scale_keeps_what_the_view_shows(app, view):
    # set_zoom redraws the scene at another pixels-per-second; the view's
    # transform makes up the difference so the screen shows the same.
    _zoom(view, 0.5)
    before = _on_screen(view)
    view.scene().set_zoom(view.scene().pixels_per_second * 2)
    app.processEvents()
    assert _on_screen(view) == pytest.approx(before)
    view.scene().build_timeline()
    app.processEvents()
    assert _on_screen(view) == pytest.approx(before)
