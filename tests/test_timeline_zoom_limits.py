"""The wheel zooms the timeline between the whole video and 1000 px a second."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from video_ai_editor.signal_timeline import (  # noqa: E402
    MAX_PX_PER_SECOND, SignalTimelineScene, SignalTimelineView, clamp_zoom)


def test_zooming_out_stops_when_the_whole_video_fits():
    # 1000 px view, 500 s video: the whole video is 2 px a second.
    assert clamp_zoom(2.3, 1 / 1.15, 1000, 500) == pytest.approx(2.0 / 2.3)
    assert clamp_zoom(2.0, 1 / 1.15, 1000, 500) == 1.0


def test_zooming_in_stops_at_the_frame_scale():
    assert clamp_zoom(950, 1.15, 1000, 500) == pytest.approx(MAX_PX_PER_SECOND / 950)
    assert clamp_zoom(MAX_PX_PER_SECOND, 1.15, 1000, 500) == 1.0


def test_inside_the_limits_the_wheel_is_untouched():
    assert clamp_zoom(50, 1.15, 1000, 500) == 1.15
    assert clamp_zoom(50, 1 / 1.15, 1000, 500) == pytest.approx(1 / 1.15)


def test_from_outside_a_limit_the_way_back_is_open():
    # Zoomed out past the whole video (an older session, a window made wider):
    # one notch in lands on the whole video, zooming further out does nothing.
    assert clamp_zoom(1.0, 1.15, 1000, 500) == pytest.approx(2.0)
    assert clamp_zoom(1.0, 1 / 1.15, 1000, 500) == 1.0


def test_a_short_clip_never_has_its_limits_crossed():
    # 1000 px for a 0.5 s clip is 2000 px a second for the whole of it: that
    # is the most and the least at once, and the wheel does nothing.
    assert clamp_zoom(2000, 1.15, 1000, 0.5) == 1.0
    assert clamp_zoom(2000, 1 / 1.15, 1000, 0.5) == 1.0


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


def _wheel(view, up: bool, times: int):
    for _ in range(times):
        pos = QPointF(view.viewport().width() / 2, 20)
        ev = QWheelEvent(pos, QPointF(view.mapToGlobal(QPoint(int(pos.x()), 20))),
                         QPoint(0, 0), QPoint(0, 120 if up else -120),
                         Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
                         Qt.ScrollPhase.NoScrollPhase, False)
        view.wheelEvent(ev)


def test_the_wheel_obeys_both_limits(app):
    scene = SignalTimelineScene(cache_data={}, video_duration=600.0)
    view = SignalTimelineView(scene)
    view.resize(1000, 300)
    view.show()
    app.processEvents()
    eff = lambda: scene.pixels_per_second * view.transform().m11()  # noqa: E731
    _wheel(view, up=False, times=80)
    assert eff() == pytest.approx(view.viewport().width() / 600.0, rel=1e-3)
    _wheel(view, up=True, times=200)
    assert eff() == pytest.approx(MAX_PX_PER_SECOND, rel=1e-3)
    view.close()
