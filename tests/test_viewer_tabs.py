"""One timeline window with a tab per video (video_ai_editor/viewer_tabs.py),
and the main window finding a video's viewer again instead of building another."""
import os
import types

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication, QMainWindow  # noqa: E402

from video_ai_editor import viewer_tabs as vt  # noqa: E402


@pytest.fixture
def app(tmp_path):
    a = QApplication.instance() or QApplication([])
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    yield a


class _Player:
    def __init__(self):
        self.paused = 0

    def pause(self):
        self.paused += 1


class _Viewer(QMainWindow):
    """What the tabs touch of a SignalTimelineWindow."""

    def __init__(self, path):
        super().__init__()
        self.video_path = path
        self.video_player = _Player()
        self.closed = 0
        self.setWindowTitle(f"Signal Timeline - {os.path.basename(path)}")

    def closeEvent(self, event):
        self.closed += 1
        super().closeEvent(event)


@pytest.fixture
def tabs(app):
    w = vt.ViewerTabs()
    yield w
    w.close()
    w.deleteLater()
    app.processEvents()


def test_a_video_gets_one_tab_however_often_it_is_shown(app, tabs):
    a, b = _Viewer(r"D:\m\a.mp4"), _Viewer(r"D:\m\b.mp4")
    tabs.show_viewer(a)
    tabs.show_viewer(b)
    tabs.show_viewer(a)
    app.processEvents()
    assert tabs.viewers() == [a, b] and tabs.current() is a
    assert tabs.tabs.tabText(1) == "b" and tabs.tabs.tabToolTip(1) == r"D:\m\b.mp4"
    assert a.window() is tabs and b.window() is tabs     # inside, not windows of their own


def test_leaving_a_tab_pauses_its_video(app, tabs):
    a, b = _Viewer("a.mp4"), _Viewer("b.mp4")
    tabs.show_viewer(a)
    tabs.show_viewer(b)
    app.processEvents()
    assert a.video_player.paused == 1 and b.video_player.paused == 0
    assert tabs.windowTitle() == "Signal Timeline - b.mp4"


def test_the_front_tab_is_announced(app, tabs):
    seen = []
    tabs.current_changed.connect(seen.append)
    a, b = _Viewer("a.mp4"), _Viewer("b.mp4")
    tabs.show_viewer(a)
    tabs.show_viewer(b)
    tabs.show_viewer(a)
    assert seen == [a, b, a]


def test_closing_a_tab_closes_that_viewer_and_the_last_hides_the_window(app, tabs):
    a, b = _Viewer("a.mp4"), _Viewer("b.mp4")
    tabs.show_viewer(a)
    tabs.show_viewer(b)
    tabs.close_tab(tabs.tabs.indexOf(a))
    assert a.closed == 1 and tabs.viewers() == [b] and tabs.isVisible()
    tabs.close_tab(0)
    assert b.closed == 1 and not tabs.isVisible()


def test_closing_the_window_closes_every_tab(app, tabs):
    a, b = _Viewer("a.mp4"), _Viewer("b.mp4")
    tabs.show_viewer(a)
    tabs.show_viewer(b)
    tabs.close()
    assert a.closed == 1 and b.closed == 1 and tabs.viewers() == []


def test_a_closed_viewer_can_come_back_as_a_tab(app, tabs):
    a = _Viewer("a.mp4")
    tabs.show_viewer(a)
    tabs.close_tab(0)
    tabs.show_viewer(a)
    app.processEvents()
    assert tabs.viewers() == [a] and tabs.isVisible() and a.isVisible()


class TestMainWindowFindsTheViewer:
    """`VideoHighlighterGUI._viewer_for`: by path, whatever its spelling."""

    @pytest.fixture
    def gui(self):
        main = pytest.importorskip("main")
        return main.VideoHighlighterGUI

    def test_same_file_different_spelling(self, gui, app):
        me = types.SimpleNamespace(_viewers={}, _viewer_key=gui._viewer_key)
        v = _Viewer(r"D:\movies\a.mp4")
        me._viewers[gui._viewer_key(r"D:\movies\a.mp4")] = v
        assert gui._viewer_for(me, "D:/movies/a.mp4") is v
        assert gui._viewer_for(me, r"D:\movies\b.mp4") is None

    def test_a_deleted_viewer_is_forgotten(self, gui, app):
        import shiboken6
        me = types.SimpleNamespace(_viewers={}, _viewer_key=gui._viewer_key)
        v = _Viewer("a.mp4")
        me._viewers[gui._viewer_key("a.mp4")] = v
        shiboken6.delete(v)
        assert gui._viewer_for(me, "a.mp4") is None and me._viewers == {}
