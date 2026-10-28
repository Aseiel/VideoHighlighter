"""Right-click and draw-a-box on the plain video surface.

Teaching a model from a box used to need the Live overlay switched on, because
only its graphics view took a right-click. The plain surface (Off and Precomp
modes, vr_video_view + VRVideoOutput.qml) now takes one too, and hands back the
box as fractions of the whole frame — the same thing the overlay hands over.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6.QtQuickWidgets")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from video_ai_editor.vr_video_view import VRVideoView, shown_to_frame_roi


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return QApplication.instance() or QApplication([])


@pytest.fixture
def view(app):
    v = VRVideoView()
    if not v.can_draw_region():
        pytest.skip("QML surface unavailable here")
    v.set_native_resolution(200, 100)          # 2:1 picture
    v.resize(400, 400)                         # -> 400x200, bars above and below
    v.show()
    QApplication.processEvents()
    yield v
    v.close()


def _drag(widget, a, b):
    QTest.mousePress(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, a)
    QTest.mouseMove(widget, b)
    QTest.mouseRelease(widget, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, b)
    QApplication.processEvents()


def test_vr_boxes_land_on_the_left_eye():
    assert shown_to_frame_roi((0.5, 0.2, 0.5, 0.4), False) == (0.5, 0.2, 0.5, 0.4)
    assert shown_to_frame_roi((0.5, 0.2, 0.5, 0.4), True) == (0.25, 0.2, 0.25, 0.4)


def test_right_click_asks_for_the_menu(view):
    asked = []
    view.context_requested.connect(asked.append)
    QTest.mouseClick(view._quick, Qt.MouseButton.RightButton,
                     Qt.KeyboardModifier.NoModifier, QPoint(50, 60))
    QApplication.processEvents()
    assert len(asked) == 1


def test_a_left_drag_draws_nothing_until_armed(view):
    drawn = []
    view.region_drawn.connect(drawn.append)
    _drag(view._quick, QPoint(100, 150), QPoint(200, 250))
    assert drawn == []


def test_an_armed_drag_becomes_a_box_on_the_picture(view):
    drawn = []
    view.region_drawn.connect(drawn.append)
    view.begin_region_draw()
    # Picture spans y 100..300; the drag starts in the bar above it.
    _drag(view._quick, QPoint(100, 50), QPoint(200, 200))
    assert len(drawn) == 1
    assert drawn[0] == pytest.approx((0.25, 0.0, 0.25, 0.5), abs=0.01)
    # One box per arming.
    _drag(view._quick, QPoint(100, 150), QPoint(200, 250))
    assert len(drawn) == 1


def test_a_right_click_cancels_an_armed_box(view):
    drawn, asked = [], []
    view.region_drawn.connect(drawn.append)
    view.context_requested.connect(asked.append)
    view.begin_region_draw()
    QTest.mouseClick(view._quick, Qt.MouseButton.RightButton,
                     Qt.KeyboardModifier.NoModifier, QPoint(50, 150))
    _drag(view._quick, QPoint(100, 150), QPoint(200, 250))
    assert drawn == [] and asked == []
