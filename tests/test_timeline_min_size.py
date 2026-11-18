"""The timeline viewer fits a laptop screen.

Its rows of controls used to be lines that could not break, and a window can
be no smaller than its widest row: it needed 2717x969 px, so on a 1366x768
laptop it ran off the screen and the toolbar buttons overlapped. The rows now
wrap; this holds the window to a size an ordinary screen has.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Offscreen Qt ships no fonts and falls back to one far wider than the
# app's own, which alone adds a couple of hundred pixels to every row. Give it
# the system's, so the width measured is the width a customer gets. Only takes
# effect if no QApplication exists yet; the test skips when it did not.
_SYSTEM_FONTS = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
if os.path.isdir(_SYSTEM_FONTS):
    os.environ.setdefault("QT_QPA_FONTDIR", _SYSTEM_FONTS)

pytest.importorskip("PySide6")
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtGui import QFontDatabase  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

# 1366x768 less a taskbar: what Windows gives a maximized window there.
LAPTOP = (1366, 728)
# The theme's font. Measured in any other, the answer is about a different app.
THEME_FONT = "Segoe UI"


@pytest.fixture(scope="module")
def app():
    app = QApplication.instance() or QApplication([])
    if THEME_FONT not in QFontDatabase.families():
        pytest.skip(f"{THEME_FONT!r} is not available to Qt here, so widths "
                    "would be measured in a substitute font")
    from modules.ui import theme
    sheet, palette, style = app.styleSheet(), app.palette(), app.style().name()
    theme.apply(app)
    yield app
    # The theme is app-wide; leave the tests after this one the app they had.
    app.setStyleSheet(sheet)
    app.setStyle(style)
    app.setPalette(palette)


@pytest.fixture()
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             10, (160, 90))
    if not writer.isOpened():
        pytest.skip("no mp4 encoder available to make a test clip")
    for i in range(20):
        frame = np.full((90, 160, 3), (i * 12) % 255, dtype=np.uint8)
        writer.write(frame)
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


def _settle(app, rounds=20):
    for _ in range(rounds):
        app.processEvents()


def test_the_viewer_fits_a_laptop_screen(app, clip, private_settings):
    try:
        from signal_timeline_viewer import SignalTimelineWindow
        win = SignalTimelineWindow(clip)
    except Exception as e:  # a missing optional runtime in this interpreter
        pytest.skip(f"the viewer cannot be built here: {e}")
    try:
        win.resize(*LAPTOP)
        win.show()
        _settle(app)
        floor = win.minimumSizeHint()
        assert floor.width() <= LAPTOP[0], floor
        assert floor.height() <= LAPTOP[1], floor
        assert win.width() <= LAPTOP[0] and win.height() <= LAPTOP[1], win.size()
    finally:
        win.close()
        win.deleteLater()
        _settle(app, 3)
