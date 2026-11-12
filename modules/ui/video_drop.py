"""Dropping video files onto the detailed settings view.

The Simple view has a drop zone; the detailed view had only an "Add Videos"
button and a file list that, while empty, was an unlabelled dark strip. Dragging
a file out of Explorer is how most people hand a video to an app, and here it
either did nothing or — over a text field — typed the file's URL into it.

Two pieces, both small:

- `VideoList`: the file list, which takes drops and says so while it is empty.
- `accept_video_drops`: makes any other widget (the whole page) take them too,
  so a drop that misses the 46px list still lands.
"""
from __future__ import annotations

from typing import Callable, List

from PySide6.QtCore import QEvent, QObject, Qt, QTimer
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QListWidget

from modules.segments.simple_run import is_video_path
from modules.ui.theme import DARK


def video_paths(mime) -> List[str]:
    """The local video files in a drag's payload, in the order given."""
    if mime is None or not mime.hasUrls():
        return []
    out = []
    for url in mime.urls():
        path = url.toLocalFile()
        if path and is_video_path(path):
            out.append(path)
    return out


def _deliver(on_paths: Callable[[List[str]], None], paths: List[str]) -> None:
    """Hand the paths over after the drop has returned.

    Adding a video can open a dialog (the low-res-copy offer). Opening it from
    inside the drop handler keeps the drag's source — Explorer — waiting on the
    answer, with its window frozen.
    """
    QTimer.singleShot(0, lambda: on_paths(list(paths)))


class VideoList(QListWidget):
    """The input list: accepts dropped videos, and shows a hint while empty."""

    def __init__(self, on_paths: Callable[[List[str]], None],
                 hint: str = "Drop videos here, or press Add Videos",
                 parent=None):
        super().__init__(parent)
        self._on_paths = on_paths
        self._hint = hint
        self.setAcceptDrops(True)

    # QListWidget implements these for moving its own rows and refuses
    # everything else, so all three are needed for an outside drop to land.
    def dragEnterEvent(self, event):  # noqa: N802
        if video_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):  # noqa: N802
        if video_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):  # noqa: N802
        paths = video_paths(event.mimeData())
        if paths:
            event.acceptProposedAction()
            _deliver(self._on_paths, paths)
        else:
            event.ignore()

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        if self.count() or not self._hint:
            return
        painter = QPainter(self.viewport())
        try:
            painter.setPen(QColor(DARK.text_mute))
            painter.drawText(self.viewport().rect(), Qt.AlignCenter, self._hint)
        finally:
            painter.end()


class _DropFilter(QObject):
    def __init__(self, widget, on_paths):
        super().__init__(widget)
        self._on_paths = on_paths

    def eventFilter(self, _obj, event):  # noqa: N802
        kind = event.type()
        if kind in (QEvent.DragEnter, QEvent.DragMove):
            if video_paths(event.mimeData()):
                event.acceptProposedAction()
                return True
        elif kind == QEvent.Drop:
            paths = video_paths(event.mimeData())
            if paths:
                event.acceptProposedAction()
                _deliver(self._on_paths, paths)
                return True
        return False


def accept_video_drops(widget, on_paths: Callable[[List[str]], None]) -> QObject:
    """Let ``widget`` take dropped video files, wherever on it they land.

    A drag over a child that does not want it is offered to the parent, so one
    filter on a page covers its labels, group boxes and empty space. Children
    with drop behaviour of their own (text fields) keep it.
    """
    widget.setAcceptDrops(True)
    drop_filter = _DropFilter(widget, on_paths)
    widget.installEventFilter(drop_filter)
    return drop_filter
