"""
The wheel scrolls the settings panel instead of changing what is under it.

The Scoring Points panel is a column of spin boxes inside a scroll area, so
Qt's default — wheel over a spin box edits it — meant scrolling past the panel
silently changed how the next run scores. These tests use real widgets and real
wheel events: the value must not move, and the scroll area must.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (QApplication, QComboBox, QDoubleSpinBox, QLabel,
                               QScrollArea, QSlider, QSpinBox, QTabWidget,
                               QVBoxLayout, QWidget)

from modules.ui import wheel_guard


@pytest.fixture(scope="module")
def app():
    existing = QApplication.instance()
    yield existing or QApplication([])


@pytest.fixture
def guard(app):
    g = wheel_guard.install(app)
    yield g
    app.removeEventFilter(g)


def wheel(widget, delta=-120):
    """A wheel notch over `widget`, delivered the way Qt delivers a real one."""
    pos = QPointF(widget.rect().center())
    event = QWheelEvent(
        pos, widget.mapToGlobal(QPoint(int(pos.x()), int(pos.y()))),
        QPoint(0, delta), QPoint(0, delta),
        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase, False)
    QApplication.sendEvent(widget, event)
    return event


@pytest.fixture
def panel(app):
    """A scroll area with a tall column of value widgets — the settings tab."""
    area = QScrollArea()
    inner = QWidget()
    layout = QVBoxLayout(inner)

    widgets = {}
    widgets["spin"] = QSpinBox(); widgets["spin"].setRange(0, 100); widgets["spin"].setValue(50)
    widgets["dspin"] = QDoubleSpinBox(); widgets["dspin"].setRange(0, 10); widgets["dspin"].setValue(5.0)
    widgets["combo"] = QComboBox(); widgets["combo"].addItems(["a", "b", "c"]); widgets["combo"].setCurrentIndex(1)
    widgets["slider"] = QSlider(Qt.Orientation.Horizontal); widgets["slider"].setRange(0, 100); widgets["slider"].setValue(50)
    for w in widgets.values():
        layout.addWidget(w)
    for i in range(40):                       # make it actually scrollable
        layout.addWidget(QLabel(f"row {i}"))

    area.setWidget(inner)
    area.setWidgetResizable(True)
    area.resize(300, 200)
    area.show()
    yield area, widgets
    area.hide()
    area.deleteLater()


class TestValuesStayPut:
    def test_spin_box(self, panel, guard):
        area, w = panel
        wheel(w["spin"])
        assert w["spin"].value() == 50

    def test_double_spin_box(self, panel, guard):
        area, w = panel
        wheel(w["dspin"])
        assert w["dspin"].value() == pytest.approx(5.0)

    def test_combo_box(self, panel, guard):
        area, w = panel
        wheel(w["combo"])
        assert w["combo"].currentIndex() == 1

    def test_slider(self, panel, guard):
        area, w = panel
        wheel(w["slider"])
        assert w["slider"].value() == 50

    def test_a_focused_spin_box_is_guarded_too(self, panel, guard):
        # Clicking a box to read it should not arm it for the next scroll.
        area, w = panel
        w["spin"].setFocus()
        wheel(w["spin"])
        assert w["spin"].value() == 50

    def test_tabs_do_not_change(self, app, guard):
        tabs = QTabWidget()
        for name in ("Basic", "Transcript", "Advanced"):
            tabs.addTab(QWidget(), name)
        tabs.setCurrentIndex(0)
        tabs.resize(300, 200)
        tabs.show()
        wheel(tabs.tabBar())
        assert tabs.currentIndex() == 0
        tabs.hide()
        tabs.deleteLater()


class TestWithoutTheGuard:
    """What the default does — the behaviour being removed."""

    def test_spin_box_would_change(self, panel):
        area, w = panel
        wheel(w["spin"])
        assert w["spin"].value() != 50


class TestThePanelStillScrolls:
    @pytest.mark.parametrize("name", ["spin", "dspin", "combo", "slider"])
    def test_wheel_over_a_value_widget_scrolls_the_area(self, panel, guard, name):
        area, w = panel
        bar = area.verticalScrollBar()
        bar.setValue(0)
        wheel(w[name])
        assert bar.value() > 0, "the wheel must reach the panel"

    def test_scrolls_the_way_the_wheel_turned(self, panel, guard):
        area, w = panel
        bar = area.verticalScrollBar()
        bar.setValue(200)
        wheel(w["spin"], delta=+120)     # wheel up
        assert bar.value() < 200

    def test_the_nearest_scroll_area_is_the_one_that_moves(self, app, guard):
        outer = QScrollArea()
        outer_inner = QWidget(); outer_lay = QVBoxLayout(outer_inner)
        inner_area = QScrollArea()
        inner_inner = QWidget(); inner_lay = QVBoxLayout(inner_inner)
        spin = QSpinBox(); spin.setRange(0, 100); spin.setValue(50)
        inner_lay.addWidget(spin)
        for i in range(40):
            inner_lay.addWidget(QLabel(f"inner {i}"))
        inner_area.setWidget(inner_inner); inner_area.setWidgetResizable(True)
        inner_area.setFixedHeight(150)
        outer_lay.addWidget(inner_area)
        for i in range(40):
            outer_lay.addWidget(QLabel(f"outer {i}"))
        outer.setWidget(outer_inner); outer.setWidgetResizable(True)
        outer.resize(400, 300); outer.show()

        outer.verticalScrollBar().setValue(0)
        inner_area.verticalScrollBar().setValue(0)
        wheel(spin)
        assert inner_area.verticalScrollBar().value() > 0
        assert outer.verticalScrollBar().value() == 0
        outer.hide(); outer.deleteLater()

    def test_outside_a_scroll_area_nothing_happens_at_all(self, app, guard):
        # A combo in a plain dialog: swallowing beats editing.
        dialog = QWidget()
        lay = QVBoxLayout(dialog)
        combo = QComboBox(); combo.addItems(["a", "b", "c"]); combo.setCurrentIndex(1)
        lay.addWidget(combo)
        dialog.resize(200, 100); dialog.show()
        wheel(combo)
        assert combo.currentIndex() == 1
        dialog.hide(); dialog.deleteLater()


class TestOptOut:
    def test_a_widget_can_ask_for_its_wheel_back(self, panel, guard):
        area, w = panel
        w["spin"].setProperty("wheelGuard", False)
        wheel(w["spin"])
        assert w["spin"].value() != 50


class TestWhatIsNotGuarded:
    def test_a_scroll_bar_still_scrolls(self, panel, guard):
        area, _w = panel
        bar = area.verticalScrollBar()
        bar.setValue(0)
        wheel(bar)
        assert bar.value() > 0

    def test_a_custom_wheel_handler_still_runs(self, app, guard):
        """The timeline zooms on wheel from its own QGraphicsView handler."""
        seen = []

        class Zoomer(QWidget):
            def wheelEvent(self, event):
                seen.append(event.angleDelta().y())
                event.accept()

        z = Zoomer()
        z.resize(100, 100)
        z.show()
        wheel(z)
        assert seen == [-120]
        z.hide()
        z.deleteLater()
