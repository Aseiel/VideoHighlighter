"""The timeline opens on its playhead, not on the middle of the video.

A QGraphicsView shows a scene wider than itself centred, so a long film opened
on its middle, with the clips at its start out of sight and the timeline
reading as empty until something scrolled it.
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
def opened(app):
    """Opens a view on a 1200 s video; detaches it before either is freed.

    A view that is dropped while still showing its scene can take the
    process down as Python frees the two in whatever order it likes; the
    window owns both in the app, so only a test meets this.
    """
    made = []

    def _open(playhead=None, follow=True):
        scene = SignalTimelineScene(cache_data={}, video_duration=1200.0)
        if playhead is not None:
            scene.set_current_time(playhead)
        view = SignalTimelineView(scene)
        view.follow_playhead = follow
        view.resize(800, 300)
        view.show()
        for _ in range(5):
            app.processEvents()
        made.append((view, scene))
        return view

    yield _open
    for view, _scene in made:
        view.close()
        view.setScene(None)
    app.processEvents()


def _seconds_shown(view):
    pps = view.scene().pixels_per_second
    r = view.mapToScene(view.viewport().rect()).boundingRect()
    return r.left() / pps, r.right() / pps


def test_a_long_video_opens_at_its_start(opened):
    first, _last = _seconds_shown(opened())
    assert first == pytest.approx(0.0, abs=0.5)


def test_it_opens_where_the_playhead_is(opened):
    first, last = _seconds_shown(opened(playhead=700.0))
    assert first <= 700.0 <= last


def test_it_opens_there_with_follow_playhead_off(opened):
    # Following is about playback; where the view opens is not.
    first, _last = _seconds_shown(opened(follow=False))
    assert first == pytest.approx(0.0, abs=0.5)
