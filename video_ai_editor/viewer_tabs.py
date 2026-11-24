"""One timeline window, one tab per video.

The viewer (`signal_timeline_viewer.SignalTimelineWindow`) is built for one
video, and opening another used to build a second top-level window while the
first stayed open, untracked: a report with times in two videos opened a new
window on every click, each holding its players and caches (a viewer cannot be
torn down cheaply -- see its closeEvent). Here every viewer is a tab of one
window, and a video that already has a viewer gets its tab back.

* Leaving a tab pauses it, so a hidden video does not keep playing.
* Closing a tab closes that viewer the way closing its window did (paused,
  muted, kept for reuse); closing the window closes every tab.
* ``current_changed(viewer)`` says which viewer is in front, so the main
  window's "the open viewer" follows the tab.
"""
from __future__ import annotations

import os
from typing import List, Optional

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget

GEOMETRY_KEY = "viewer_tabs/geometry"


def pause(viewer) -> None:
    """Stop a viewer playing, the source video and the edit alike."""
    try:
        if getattr(viewer, "_edit_playback_active", False):
            viewer._pause_edit_playback()
    except Exception:  # noqa: BLE001 - best effort; the players below still pause
        pass
    players = [getattr(viewer, "video_player", None)]
    live = getattr(viewer, "realtime_preview", None)
    players.append(getattr(live, "player", None) if live is not None else None)
    for player in players:
        try:
            if player is not None:
                player.pause()
        except Exception:  # noqa: BLE001
            pass


class ViewerTabs(QMainWindow):
    """The timeline viewers, as tabs of one window."""

    current_changed = Signal(object)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Signal Timeline")
        self.tabs = QTabWidget(self)
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)
        self.tabs.currentChanged.connect(self._on_current)
        self.tabs.tabCloseRequested.connect(self.close_tab)
        self.setCentralWidget(self.tabs)
        self._front = None
        geometry = QSettings().value(GEOMETRY_KEY)
        if geometry is None or not self.restoreGeometry(geometry):
            screen = QApplication.primaryScreen().availableGeometry()
            w, h = min(1600, screen.width() - 20), min(1000, screen.height() - 20)
            self.resize(w, h)
            self.move(screen.x() + (screen.width() - w) // 2, screen.y())

    # -- the tabs ------------------------------------------------------------
    def viewers(self) -> List[object]:
        return [self.tabs.widget(i) for i in range(self.tabs.count())]

    def current(self):
        return self.tabs.currentWidget()

    def show_viewer(self, viewer) -> None:
        """Put ``viewer`` in front, as a tab of its own when it has none yet."""
        i = self.tabs.indexOf(viewer)
        if i < 0:
            viewer.setWindowFlags(Qt.WindowType.Widget)
            path = str(getattr(viewer, "video_path", "") or "")
            i = self.tabs.addTab(viewer, _short(os.path.basename(path) or "Video"))
            self.tabs.setTabToolTip(i, path)
        self.tabs.setCurrentIndex(i)
        viewer.show()
        if self.isMinimized():
            self.showNormal()
        self.show()
        self.raise_()
        self.activateWindow()

    def close_tab(self, index: int) -> None:
        viewer = self.tabs.widget(index)
        self.tabs.removeTab(index)
        if viewer is not None:
            viewer.close()                 # paused and muted, kept for reuse
            viewer.hide()
        if self.tabs.count() == 0:
            self._remember()
            self.hide()

    def _on_current(self, index: int) -> None:
        viewer = self.tabs.widget(index) if index >= 0 else None
        if self._front is not None and self._front is not viewer:
            pause(self._front)
        self._front = viewer
        if viewer is not None:
            self.setWindowTitle(viewer.windowTitle() or "Signal Timeline")
        self.current_changed.emit(viewer)

    # -- the window ------------------------------------------------------------
    def _remember(self) -> None:
        QSettings().setValue(GEOMETRY_KEY, self.saveGeometry())

    def closeEvent(self, event):           # noqa: N802 - Qt override
        self._remember()
        while self.tabs.count():
            self.close_tab(self.tabs.count() - 1)
        super().closeEvent(event)


def _short(name: str, limit: int = 40) -> str:
    stem = os.path.splitext(name)[0]
    return stem if len(stem) <= limit else stem[: limit - 1] + "…"
