"""A row of controls that wraps instead of forcing its window wider.

A ``QHBoxLayout`` full of buttons has a minimum width equal to all of them side
by side, and the window that holds it inherits that minimum. Three such rows
made a window that needed 2717 pixels of width — invisible on a wide monitor,
and on an ordinary one the buttons were squeezed until they overlapped.

``FlowLayout`` is the same row with one difference: when the next item does not
fit, it starts a new line. Where there *is* room it lays out exactly like the
box layout it replaces — items that can expand (a line edit, a slider, a
stretch) share whatever width is left on their line — so nothing changes on a
screen that was already wide enough.

``FlowRow`` is the widget to put it in. A wrapped row is taller than an
unwrapped one, and splitters, docks and scroll areas do not ask a layout how
tall it is at a given width; the row therefore tells them, by keeping its own
minimum height equal to the height it currently needs.

No application imports: this file is identical in both editions.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from PySide6.QtCore import QEvent, QRect, QSize, Qt
from PySide6.QtWidgets import (
    QLayout, QLayoutItem, QSizePolicy, QSpacerItem, QWidget,
)

_EXPAND = QSizePolicy.PolicyFlag.ExpandFlag.value
_GROW = QSizePolicy.PolicyFlag.GrowFlag.value


def _is_stretch(item: QLayoutItem) -> bool:
    return item.spacerItem() is not None


def _expands(item: QLayoutItem) -> bool:
    """Whether this item should take a share of its line's spare width."""
    if _is_stretch(item):
        return True
    widget = item.widget()
    if widget is None:
        return False
    return bool(widget.sizePolicy().horizontalPolicy().value & _EXPAND)


class FlowLayout(QLayout):
    """Left to right, wrapping to a new line when the next item does not fit."""

    def __init__(self, parent: Optional[QWidget] = None, *,
                 spacing: int = 6, line_spacing: int = 4):
        super().__init__(parent)
        self._items: List[QLayoutItem] = []
        self._gap = int(spacing)
        self._line_gap = int(line_spacing)
        self.setContentsMargins(0, 0, 0, 0)

    # ── building ─────────────────────────────────────────────────────────
    def addItem(self, item: QLayoutItem) -> None:  # noqa: N802 (Qt name)
        self._items.append(item)
        self.invalidate()

    def addStretch(self, _stretch: int = 0) -> None:  # noqa: N802
        """Push what follows to the right edge of its line, as a box layout's
        stretch does. On a line with no room to spare it takes none."""
        self.addItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding,
                                 QSizePolicy.Policy.Minimum))

    def addLayout(self, layout: QLayout) -> QWidget:  # noqa: N802
        """Add a group that must stay together (a button and its options, a
        label and its field). Returns the widget that now holds it."""
        box = QWidget()
        layout.setContentsMargins(0, 0, 0, 0)
        box.setLayout(layout)
        self.addWidget(box)
        return box

    # ── the QLayout contract ─────────────────────────────────────────────
    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int) -> Optional[QLayoutItem]:  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int) -> Optional[QLayoutItem]:  # noqa: N802
        if 0 <= index < len(self._items):
            item = self._items.pop(index)
            self.invalidate()
            return item
        return None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._arrange(QRect(0, 0, int(width), 0), apply=False)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._arrange(rect, apply=True)

    def sizeHint(self) -> QSize:  # noqa: N802
        """Everything on one line: what the row wants when there is room."""
        width, height, shown = 0, 0, 0
        for item in self._items:
            if _is_stretch(item) or item.isEmpty():
                continue
            hint = item.sizeHint()
            width += hint.width()
            height = max(height, hint.height())
            shown += 1
        width += self._gap * max(0, shown - 1)
        m = self.contentsMargins()
        return QSize(width + m.left() + m.right(),
                     height + m.top() + m.bottom())

    def minimumSize(self) -> QSize:  # noqa: N802
        """The widest single item: narrower than that and nothing fits."""
        width, height = 0, 0
        for item in self._items:
            if _is_stretch(item) or item.isEmpty():
                continue
            width = max(width, self._base_width(item))
            height = max(height, item.sizeHint().height())
        m = self.contentsMargins()
        return QSize(width + m.left() + m.right(),
                     height + m.top() + m.bottom())

    # ── the arrangement ──────────────────────────────────────────────────
    @staticmethod
    def _base_width(item: QLayoutItem) -> int:
        """The width an item is given before spare room is shared out.

        An expanding item starts from its minimum — it will be given more — and
        everything else from the width it asks for, since nothing will add to it.
        """
        if _is_stretch(item):
            return 0
        if _expands(item):
            return item.minimumSize().width()
        return item.sizeHint().width()

    def _lines(self, width: int) -> List[List[Tuple[QLayoutItem, int]]]:
        lines: List[List[Tuple[QLayoutItem, int]]] = []
        line: List[Tuple[QLayoutItem, int]] = []
        used = 0                      # width taken on this line, gaps included
        solid = 0                     # how many real (non-stretch) items it holds
        for item in self._items:
            stretch = _is_stretch(item)
            if not stretch and item.isEmpty():
                continue              # a hidden widget takes no place
            base = self._base_width(item)
            if stretch:
                line.append((item, 0))
                continue
            need = used + (self._gap if solid else 0) + base
            if solid and need > width:
                lines.append(line)
                line, used, solid = [], 0, 0
                need = base
            line.append((item, base))
            used = need
            solid += 1
        if line:
            lines.append(line)
        # A line holding nothing but a stretch is not a line.
        return [ln for ln in lines if any(not _is_stretch(i) for i, _w in ln)]

    def _arrange(self, rect: QRect, *, apply: bool) -> int:
        m = self.contentsMargins()
        left = rect.x() + m.left()
        top = rect.y() + m.top()
        width = max(0, rect.width() - m.left() - m.right())

        y = top
        lines = self._lines(width)
        for index, line in enumerate(lines):
            solid = [(i, w) for i, w in line if not _is_stretch(i)]
            height = max(i.sizeHint().height() for i, _w in solid)
            if apply:
                used = sum(w for _i, w in solid) + self._gap * (len(solid) - 1)
                self._place(line, left, y, height, max(0, width - used))
            y += height
            if index < len(lines) - 1:
                y += self._line_gap
        return (y - top) + m.top() + m.bottom()

    def _place(self, line, left: int, y: int, height: int, spare: int) -> None:
        takers = [i for i, _w in line if _expands(i)]
        # Shared equally; an item at its maximum passes the rest on.
        extra = {id(i): 0 for i in takers}
        remaining = spare
        while remaining > 0 and takers:
            share = max(1, remaining // len(takers))
            before = remaining
            unfilled = []
            for item in takers:
                if remaining <= 0:
                    break
                if _is_stretch(item):
                    room = remaining
                else:
                    room = max(0, item.maximumSize().width()
                               - self._base_width(item) - extra[id(item)])
                give = min(share, room, remaining)
                extra[id(item)] += give
                remaining -= give
                if room > give:
                    unfilled.append(item)
            if remaining == before:
                break                   # nobody could take any more
            takers = unfilled

        x = left
        placed_solid = False
        for item, base in line:
            if _is_stretch(item):
                x += extra[id(item)]
                continue
            if placed_solid:
                x += self._gap
            w = base + extra.get(id(item), 0)
            # As a box layout does: an item that may grow (a label, a rule, a
            # group) fills the line, one that may not (a button, a field)
            # keeps its own height and sits centred.
            h = min(height, max(item.sizeHint().height(),
                                item.minimumSize().height()))
            if _is_stretch(item) or (item.widget() is not None and (
                    item.widget().sizePolicy().verticalPolicy().value & _GROW)):
                h = min(height, item.maximumSize().height())
            item.setGeometry(QRect(x, y + (height - h) // 2, w, h))
            x += w
            placed_solid = True


class FlowRow(QWidget):
    """A widget laid out by a :class:`FlowLayout`, as tall as it currently needs.

    Use ``row.flow`` as you would the ``QHBoxLayout`` it replaces.
    """

    def __init__(self, parent: Optional[QWidget] = None, *,
                 spacing: int = 6, line_spacing: int = 4,
                 margins: Tuple[int, int, int, int] = (0, 0, 0, 0)):
        super().__init__(parent)
        self.flow = FlowLayout(self, spacing=spacing, line_spacing=line_spacing)
        self.flow.setContentsMargins(*margins)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def sync_height(self) -> None:
        """Make the row's height the one its current width requires.

        Only once it is on screen. Before that its width is a placeholder, and
        a height worked out from a placeholder would be a dozen lines tall —
        enough to push the window's own minimum past what it was sized to,
        which Qt answers by growing the window and never shrinking it back.
        """
        if not self.isVisible():
            return
        needed = self.flow.heightForWidth(max(self.width(), 1))
        if needed != self.minimumHeight() or needed != self.maximumHeight():
            self.setFixedHeight(needed)

    def sizeHint(self) -> QSize:  # noqa: N802
        hint = self.flow.sizeHint()
        return QSize(hint.width(), max(hint.height(), self.minimumHeight()))

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        floor = self.flow.minimumSize()
        return QSize(floor.width(), max(floor.height(), self.minimumHeight()))

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.sync_height()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.sync_height()

    def event(self, event) -> bool:
        handled = super().event(event)
        # A child shown or hidden changes where the row wraps.
        if event.type() == QEvent.Type.LayoutRequest:
            self.sync_height()
        return handled
