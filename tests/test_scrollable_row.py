"""A toolbar row that runs out of width must stay reachable.

Qt's box layout does not shrink its children past their minimum when the window
is too narrow — it stops drawing what does not fit. On a 4K screen the edit
toolbar's last control was reported cut in half at the right edge, with nothing
to say the rest was there.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="Qt not available")

from PySide6.QtCore import Qt                                   # noqa: E402
from PySide6.QtWidgets import (                                 # noqa: E402
    QApplication, QHBoxLayout, QPushButton, QWidget)

from modules.ui.fit import scrollable_row                       # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    return QApplication.instance() or QApplication([])


def _row(buttons=9, label="Render Highlight Video"):
    row = QWidget()
    layout = QHBoxLayout(row)
    for i in range(buttons):
        layout.addWidget(QPushButton(f"{label} {i}"))
    return row


class TestScrollableRow:
    def test_the_row_keeps_its_full_width(self, qt_app):
        """The wrapper must not solve clipping by squashing the buttons — the
        row still asks for everything it needs, and the viewport scrolls."""
        row = _row()
        wanted = row.sizeHint().width()

        scroller = scrollable_row(row)
        scroller.resize(300, scroller.height())
        qt_app.processEvents()

        assert scroller.widget() is row
        assert row.minimumSizeHint().width() >= wanted * 0.5
        assert row.width() >= scroller.viewport().width()

    def test_a_narrow_window_scrolls_instead_of_clipping(self, qt_app):
        scroller = scrollable_row(_row())
        scroller.resize(300, scroller.height())
        scroller.show()
        qt_app.processEvents()

        bar = scroller.horizontalScrollBar()

        assert bar.maximum() > 0, "nothing to scroll — the row was clipped"

    def test_a_wide_window_needs_no_scrolling(self, qt_app):
        scroller = scrollable_row(_row(buttons=2, label="OK"))
        scroller.resize(2000, scroller.height())
        scroller.show()
        qt_app.processEvents()

        assert scroller.horizontalScrollBar().maximum() == 0

    def test_the_height_leaves_room_for_the_scrollbar(self, qt_app):
        """Otherwise the buttons lose a few pixels off the bottom at the moment
        the bar appears, which is worse than the clipping it fixes."""
        row = _row()
        row_height = row.sizeHint().height()

        scroller = scrollable_row(row)

        assert scroller.height() > row_height

    def test_it_never_scrolls_vertically(self, qt_app):
        scroller = scrollable_row(_row())

        assert (scroller.verticalScrollBarPolicy()
                == Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
