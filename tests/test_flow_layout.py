"""The wrapping row: one line where there is room, more where there is not."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import QRect  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout,
    QWidget,
)

from modules.ui.flow_layout import FlowLayout, FlowRow  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def box(width, height=20):
    """A widget of a known, fixed size."""
    w = QWidget()
    w.setFixedSize(width, height)
    return w


def row_of(widths, **kwargs):
    row = FlowRow(**kwargs)
    items = [box(w) for w in widths]
    for item in items:
        row.flow.addWidget(item)
    return row, items


def lay_out(row, width):
    row.flow.setGeometry(QRect(0, 0, width, 1000))


def test_everything_is_on_one_line_when_there_is_room(app):
    row, items = row_of([100, 100, 100], spacing=10)
    lay_out(row, 400)
    assert [i.y() for i in items] == [0, 0, 0]
    assert [i.x() for i in items] == [0, 110, 220]
    assert row.flow.heightForWidth(400) == 20


def test_an_item_that_does_not_fit_starts_a_new_line(app):
    row, items = row_of([100, 100, 100], spacing=10, line_spacing=4)
    lay_out(row, 250)
    assert [i.y() for i in items] == [0, 0, 24]
    assert items[2].x() == 0
    assert row.flow.heightForWidth(250) == 44


def test_height_grows_as_the_width_shrinks(app):
    row, _items = row_of([100, 100, 100, 100], spacing=0, line_spacing=0)
    heights = [row.flow.heightForWidth(w) for w in (400, 300, 200, 100)]
    assert heights == [20, 40, 40, 80]


def test_an_item_wider_than_the_row_still_gets_a_line_of_its_own(app):
    row, items = row_of([300, 50], spacing=0, line_spacing=0)
    lay_out(row, 120)
    assert items[0].y() == 0 and items[1].y() == 20


def test_the_one_line_hint_and_the_floor(app):
    row, _items = row_of([100, 60, 40], spacing=10, margins=(4, 2, 4, 2))
    assert row.flow.sizeHint().width() == 100 + 60 + 40 + 2 * 10 + 8
    assert row.flow.minimumSize().width() == 100 + 8
    assert row.flow.minimumSize().height() == 20 + 4


def test_a_stretch_pushes_what_follows_to_the_right_edge(app):
    row = FlowRow(spacing=10)
    left, right = box(100), box(60)
    row.flow.addWidget(left)
    row.flow.addStretch()
    row.flow.addWidget(right)
    lay_out(row, 500)
    assert left.x() == 0
    assert right.x() + right.width() == 500


def test_a_stretch_takes_nothing_on_a_line_with_no_room(app):
    row = FlowRow(spacing=10, line_spacing=0)
    left, right = box(100), box(60)
    row.flow.addWidget(left)
    row.flow.addStretch()
    row.flow.addWidget(right)
    lay_out(row, 120)
    assert (left.x(), left.y()) == (0, 0)
    assert (right.x(), right.y()) == (0, 20)


def test_an_expanding_field_absorbs_the_spare_width(app):
    row = FlowRow(spacing=10)
    label, field, button = box(50), QLineEdit(), box(80)
    field.setMinimumWidth(100)
    for w in (label, field, button):
        row.flow.addWidget(w)
    lay_out(row, 600)
    assert field.x() == 60
    assert field.width() == 600 - 50 - 80 - 2 * 10
    assert button.x() + button.width() == 600


def test_an_expanding_field_wraps_at_its_minimum_not_its_hint(app):
    row = FlowRow(spacing=10, line_spacing=0)
    label, field = box(50), QLineEdit()
    field.setMinimumWidth(100)
    row.flow.addWidget(label)
    row.flow.addWidget(field)
    lay_out(row, 160)                 # exactly label + gap + minimum
    assert field.x() == 60            # beside the label
    lay_out(row, 159)
    assert field.x() == 0             # on a line of its own
    assert field.y() >= label.y() + label.height()


def test_a_capped_field_passes_spare_width_on(app):
    row = FlowRow(spacing=0)
    capped, free = QLineEdit(), QLineEdit()
    for f in (capped, free):
        f.setMinimumWidth(50)
    capped.setMaximumWidth(80)
    row.flow.addWidget(capped)
    row.flow.addWidget(free)
    lay_out(row, 400)
    assert capped.width() == 80
    assert free.width() == 320


def test_a_hidden_widget_takes_no_place(app):
    row, items = row_of([100, 100, 100], spacing=10)
    row.show()
    items[1].hide()
    lay_out(row, 400)
    assert items[2].x() == 110
    assert row.flow.sizeHint().width() == 210
    row.close()


def test_a_group_stays_together(app):
    row = FlowRow(spacing=10, line_spacing=0)
    first = box(100)
    pair = QHBoxLayout()
    pair.setSpacing(0)
    a, b = box(60), box(30)
    pair.addWidget(a)
    pair.addWidget(b)
    row.flow.addWidget(first)
    group = row.flow.addLayout(pair)
    lay_out(row, 150)                 # not enough for 100 + 10 + 90
    assert group.y() == 20
    assert group.width() == 90


def test_the_row_tells_its_parent_how_tall_it_is(app):
    host = QWidget()
    outer = QVBoxLayout(host)
    outer.setContentsMargins(0, 0, 0, 0)
    row, _items = row_of([100, 100, 100], spacing=0, line_spacing=0)
    outer.addWidget(row)
    outer.addWidget(QLabel("below"))
    host.resize(400, 300)
    host.show()
    app.processEvents()
    assert row.height() == 20

    host.resize(150, 300)
    app.processEvents()
    assert row.height() == 60
    assert row.minimumHeight() == 60        # what a splitter or dock reads

    host.resize(400, 300)
    app.processEvents()
    assert row.height() == 20
    host.close()


def test_showing_a_child_rewraps_the_row(app):
    host = QWidget()
    outer = QVBoxLayout(host)
    outer.setContentsMargins(0, 0, 0, 0)
    row, items = row_of([100, 100, 100], spacing=0, line_spacing=0)
    outer.addWidget(row)
    outer.addStretch(1)
    items[2].hide()
    host.resize(250, 300)
    host.show()
    app.processEvents()
    assert row.height() == 20

    items[2].show()
    app.processEvents()
    assert row.height() == 40
    host.close()


def test_a_row_not_yet_on_screen_claims_only_one_line(app):
    # Its width is a placeholder until it is shown; a height derived from that
    # would inflate the minimum size of the window being built around it.
    row, _items = row_of([100] * 12, spacing=0, line_spacing=0)
    row.resize(100, 20)
    app.processEvents()
    assert row.minimumSizeHint().height() == 20


def test_a_plain_layout_can_host_a_flow_layout(app):
    host = QWidget()
    flow = FlowLayout(host, spacing=5)
    flow.addWidget(QPushButton("one"))
    flow.addWidget(QPushButton("two"))
    assert flow.count() == 2
    assert flow.hasHeightForWidth()
    taken = flow.takeAt(0)
    assert taken is not None and flow.count() == 1
    assert flow.itemAt(5) is None
