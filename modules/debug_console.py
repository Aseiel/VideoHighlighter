"""Always-on debug logging + on-demand live log window.

The packaged exe is built --windowed, so sys.stdout/stderr go nowhere and
every diagnostic print (CLIP search, workers, tracebacks) is lost. This
module fixes that in two layers:

1. install() — called before anything else in main.py — tees stdout/stderr
   into ``debug.log`` next to the exe (project root in dev), with per-line
   timestamps, a crash handler (faulthandler + sys.excepthook), and a ring
   buffer of recent output.

2. set_console_visible(True) — opens a Qt log window that preloads the ring
   buffer (so enabling it AFTER an error still shows what happened) and then
   mirrors all further output live. A native AllocConsole window was tried
   first but is useless in dev (the process already has a console — often a
   hidden one under VS Code — so nothing appears) and its close button kills
   the whole app; the Qt window behaves identically from source and frozen,
   on Windows and macOS. The GUIs expose it as a "Debug log" checkbox; the
   preference persists via QSettings and is re-applied on next launch.

Keep module-level imports light: install() runs before the heavy imports so
it can capture warnings they print. Qt is only imported inside the
window-related functions, which run after QApplication exists.
"""
from __future__ import annotations

import io
import os
import sys
import time
import threading
import weakref
from collections import deque
from typing import Optional

_SETTINGS_ORG = "VideoHighlighter"
_SETTINGS_APP = "Debug"
_SETTINGS_KEY = "console_visible"

_lock = threading.RLock()
_installed = False
_log_fh: Optional[io.TextIOWrapper] = None
_orig_stdout = None
_orig_stderr = None
_backlog: deque = deque(maxlen=5000)  # recent raw chunks, preloaded into the window
_gui_sink = None      # bridge.chunk.emit while the log window exists
_window = None        # singleton _LogWindow (created on first show, then reused)
_checkboxes: list = []  # weakrefs to registered GUI checkboxes, kept in sync


def log_file_path() -> str:
    from modules.app_paths import user_data_dir
    return os.path.join(user_data_dir(), "debug.log")


class _Tee(io.TextIOBase):
    """Fan out writes to the log file, the live log window (if open), the ring
    backlog, and the original stream (visible when running from a terminal).
    Prefixes each new line with a timestamp in the file/window copies."""

    def __init__(self, original):
        self._original = original
        self._at_line_start = True

    def _stamped(self, s: str) -> str:
        out = []
        for ch in s:
            if self._at_line_start and ch not in "\r\n":
                out.append(time.strftime("[%H:%M:%S] "))
            out.append(ch)
            self._at_line_start = ch == "\n"
        return "".join(out)

    def write(self, s: str) -> int:
        if not s:
            return 0
        with _lock:
            stamped = self._stamped(s)
            _backlog.append(stamped)
            if _log_fh is not None:
                try:
                    _log_fh.write(stamped)
                    _log_fh.flush()
                except Exception:
                    pass
            if _gui_sink is not None:
                try:
                    _gui_sink(stamped)  # thread-safe: queued Qt signal emit
                except Exception:
                    pass
            if self._original is not None:
                try:
                    self._original.write(s)
                    self._original.flush()
                except Exception:
                    pass
        return len(s)

    def flush(self):
        pass

    def isatty(self):
        return False

    @property
    def encoding(self):  # some libs probe stream.encoding
        return "utf-8"


def install() -> None:
    """Redirect stdout/stderr through the tee and arm the crash handlers.
    Idempotent; safe in multiprocessing children (they append, never rotate,
    so they can't clobber the parent's live log)."""
    global _installed, _log_fh, _orig_stdout, _orig_stderr
    if _installed:
        return
    _installed = True

    path = log_file_path()
    try:
        import multiprocessing
        is_child = multiprocessing.parent_process() is not None
    except Exception:
        is_child = False
    try:
        if not is_child:
            # keep exactly one previous run for comparison
            prev = path[:-4] + ".prev.log"
            if os.path.exists(path):
                if os.path.exists(prev):
                    os.remove(prev)
                os.replace(path, prev)
        _log_fh = open(path, "a", encoding="utf-8", errors="replace", buffering=1)
    except Exception:
        _log_fh = None  # read-only install dir etc. — tee still feeds the backlog

    # Real streams exist in dev / console builds; None or NullWriter when frozen
    # --windowed. Only mirror to them if they can actually display something.
    _orig_stdout = sys.stdout if _is_real_stream(sys.stdout) else None
    _orig_stderr = sys.stderr if _is_real_stream(sys.stderr) else None
    sys.stdout = _Tee(_orig_stdout)
    sys.stderr = _Tee(_orig_stderr)

    if _log_fh is not None:
        banner = (
            f"\n===== VideoHighlighter session {time.strftime('%Y-%m-%d %H:%M:%S')} "
            f"(frozen={bool(getattr(sys, 'frozen', False))}, child={is_child}) =====\n"
        )
        _log_fh.write(banner)
        try:
            import faulthandler
            faulthandler.enable(file=_log_fh)  # hard crashes (segfault, abort)
        except Exception:
            pass

    _orig_hook = sys.excepthook

    def _hook(exc_type, exc, tb):
        import traceback
        try:
            sys.stderr.write("UNCAUGHT EXCEPTION:\n"
                             + "".join(traceback.format_exception(exc_type, exc, tb)))
        except Exception:
            pass
        _orig_hook(exc_type, exc, tb)

    sys.excepthook = _hook


def _is_real_stream(stream) -> bool:
    if stream is None:
        return False
    # PyInstaller's NullWriter swallows everything; writing to it is pointless.
    return stream.__class__.__name__ != "NullWriter"


def is_supported() -> bool:
    """The Qt log window works wherever the GUI does."""
    return True


def is_console_visible() -> bool:
    return _window is not None and _window.isVisible()


def set_console_visible(visible: bool, persist: bool = True) -> bool:
    """Show/hide the live log window. Returns the resulting visibility.
    Must be called from the GUI thread (it's wired to checkbox signals)."""
    global _window
    try:
        from PySide6.QtWidgets import QApplication
        if QApplication.instance() is None:
            return False
        if visible:
            if _window is None:
                _window = _create_window()
            _window.show()
            _window.raise_()
            _window.activateWindow()
        elif _window is not None:
            _window.hide()
    except Exception:
        return False
    _sync_checkboxes(visible)
    if persist:
        _persist_preference(visible)
    return is_console_visible()


def register_checkbox(chk) -> None:
    """GUIs call this so their checkbox reflects state changes made elsewhere
    (the other GUI's checkbox, the window's close button, startup restore)."""
    _checkboxes.append(weakref.ref(chk))


def _sync_checkboxes(state: bool) -> None:
    for ref in list(_checkboxes):
        chk = ref()
        if chk is None:
            _checkboxes.remove(ref)
        elif chk.isChecked() != state:
            chk.blockSignals(True)
            chk.setChecked(state)
            chk.blockSignals(False)


def _create_window():
    global _gui_sink
    from PySide6.QtCore import QObject, Signal
    from PySide6.QtGui import QTextCursor
    from PySide6.QtWidgets import QPlainTextEdit

    class _Bridge(QObject):
        chunk = Signal(str)

    class _LogWindow(QPlainTextEdit):
        def __init__(self):
            super().__init__()
            self.setReadOnly(True)
            self.setWindowTitle("VideoHighlighter — debug log")
            self.setMaximumBlockCount(10000)  # drop oldest lines, bound memory
            self.setLineWrapMode(QPlainTextEdit.NoWrap)
            self.setStyleSheet(
                "QPlainTextEdit{background:#111;color:#ddd;"
                "font-family:'Consolas','Courier New',monospace;font-size:9pt;}"
            )
            self.resize(950, 500)

        def append_chunk(self, text: str):
            sb = self.verticalScrollBar()
            follow = sb.value() >= sb.maximum() - 4  # keep following the tail
            self.moveCursor(QTextCursor.End)
            self.insertPlainText(text)
            if follow:
                sb.setValue(sb.maximum())

        def closeEvent(self, event):
            # Closing the window is the same as unchecking the box.
            set_console_visible(False)
            event.ignore()  # hide (via the call above), don't destroy

    win = _LogWindow()
    win.append_chunk(f"(full log: {log_file_path()})\n"
                     f"--- replaying last {len(_backlog)} output chunks ---\n")
    with _lock:
        win.append_chunk("".join(_backlog))
        bridge = _Bridge(win)
        # Queued cross-thread delivery: worker threads emit, GUI thread appends.
        bridge.chunk.connect(win.append_chunk)
        _gui_sink = bridge.chunk.emit
    win.append_chunk("--- live output from here on ---\n")
    return win


def _persist_preference(visible: bool) -> None:
    try:
        from PySide6.QtCore import QSettings
        QSettings(_SETTINGS_ORG, _SETTINGS_APP).setValue(_SETTINGS_KEY, visible)
    except Exception:
        pass


def restore_console_preference() -> None:
    """Reopen the log window at startup if it was on last session. Call once
    the QApplication exists."""
    try:
        from PySide6.QtCore import QSettings
        want = QSettings(_SETTINGS_ORG, _SETTINGS_APP).value(_SETTINGS_KEY, False, type=bool)
    except Exception:
        want = False
    if want:
        set_console_visible(True, persist=False)
