"""The edit timeline opens on its first clip.

A QGraphicsView shows a scene wider than itself centred, and the edit scene
is at least 1000 px however few clips it holds, so a short edit opened on its
empty middle and its clips were out of sight.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


@pytest.fixture()
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             10, (160, 90))
    if not writer.isOpened():
        pytest.skip("no mp4 encoder available to make a test clip")
    for i in range(1200):           # two minutes at 10 fps
        writer.write(np.full((90, 160, 3), (i * 3) % 255, dtype=np.uint8))
    writer.release()
    return str(path)


@pytest.fixture()
def private_settings(tmp_path):
    """Keep the window's saved preferences out of the real user's."""
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                      str(tmp_path / "settings"))
    yield
    QSettings.setDefaultFormat(QSettings.Format.NativeFormat)


def test_a_short_edit_opens_on_its_clips(app, clip, private_settings):
    try:
        from signal_timeline_viewer import SignalTimelineWindow
        win = SignalTimelineWindow(clip, {"final_segments": [[8, 14], [40, 47]]})
    except Exception as e:  # a missing optional runtime in this interpreter
        pytest.skip(f"the viewer cannot be built here: {e}")
    try:
        win.setFixedSize(1366, 728)
        win.show()
        for _ in range(60):
            app.processEvents()
        view, scene = win.edit_view, win.edit_scene
        assert scene.clip_items, "the run's segments were not loaded"
        shown = view.mapToScene(view.viewport().rect()).boundingRect()
        first = scene.clip_items[0].sceneBoundingRect()
        assert shown.left() <= first.left() < first.right() <= shown.right()
    finally:
        win.close()


def test_neither_timeline_can_be_dragged_out_of_sight(app, clip, private_settings):
    # QSplitter collapses a pane whose handle is dragged past its minimum;
    # the edit timeline went to 0 px and took every clip with it.
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QSplitter
    try:
        from signal_timeline_viewer import SignalTimelineWindow
        win = SignalTimelineWindow(clip, {"final_segments": [[8, 14]]})
    except Exception as e:  # a missing optional runtime in this interpreter
        pytest.skip(f"the viewer cannot be built here: {e}")
    try:
        win.setFixedSize(1366, 728)
        win.show()
        for _ in range(60):
            app.processEvents()
        splitter = next(
            s for s in win.findChildren(QSplitter)
            if s.orientation() == Qt.Orientation.Vertical
            and s.isAncestorOf(win.edit_view) and s.isAncestorOf(win.signal_view))
        total = sum(splitter.sizes())
        for far in (total + 100, -100):
            splitter.moveSplitter(far, 1)
            for _ in range(10):
                app.processEvents()
            assert min(splitter.sizes()) > 0, splitter.sizes()
    finally:
        win.close()


def test_closed_docks_can_be_reopened_from_timeline_status_bar(app, clip, private_settings):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDockWidget, QToolButton
    from signal_timeline_viewer import SignalTimelineWindow

    win = SignalTimelineWindow(clip, {"final_segments": [[8, 14]]})
    try:
        win.show()
        app.processEvents()
        docks = win.findChildren(QDockWidget, options=Qt.FindDirectChildrenOnly)
        controls = next(dock for dock in docks if dock.windowTitle() == "Controls")
        for dock in docks:
            dock.close()
        app.processEvents()
        assert all(dock.isHidden() for dock in docks)

        panels = next(
            (button for button in win.statusBar().findChildren(QToolButton)
             if button.text() == "Panels"),
            None,
        )
        assert panels is not None
        assert panels.isVisible()
        action = controls.toggleViewAction()
        assert action in panels.menu().actions()
        assert not action.isChecked()
        action.trigger()
        app.processEvents()
        assert controls.isVisible()
        assert action.isChecked()
    finally:
        win.close()
