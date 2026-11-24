"""A report beside the timeline, with its times clickable.

Reports are HTML pages written for a browser: the highlight report
(`modules.report.highlight_report`), and any page another tool writes about a
video. Read in a browser they are a dead end at the one point that matters,
"what happens at 23:56?" -- the reader has to find that second by hand. Shown
here, a click on a time moves the viewer's playhead there, and a time in
*another* video opens that video first, so a page about several videos works
the same way.

How a time becomes a link, without the page knowing about the app:

* A page marks a moment of a source video as ``<span data-vh-t="83.5">1:23</span>``,
  optionally with ``data-vh-video="<path>"`` when it is not the video the
  viewer shows. In a browser that is plain text, so one page serves both.
* The highlight report's own player buttons (``button.seek``/``button.qseek``
  with ``data-t``) seek the report's inline player in a browser; here they
  move the timeline instead, since their seconds are source seconds too.
* A clip's own player (``<video src="...#t=4,9">``) gets a "Show in timeline"
  button for the start of its fragment. The panel cannot play most source
  codecs itself, so the timeline is where the clip is watched.
* Clicks are turned into a navigation to ``https://seek.videohighlighter.invalid/``
  (``.invalid`` is reserved and never resolves) and refused in
  :meth:`_Page.acceptNavigationRequest`, so nothing leaves the machine. Plain
  ``m:ss`` text is NOT linked: the highlight report also prints times inside
  the rendered output, and guessing which clock a bare time belongs to would
  jump to the wrong place.

Qt WebEngine is optional: without it the panel says so and opens the page in
the browser instead.
"""
from __future__ import annotations

import html as _html
import os
import re
from typing import List, Optional, Tuple
from urllib.parse import parse_qs, urlencode

from PySide6.QtCore import QEvent, QObject, QSettings, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QPushButton, QVBoxLayout, QWidget)

POPPED_KEY = "report/popped_out"
GEOMETRY_KEY = "report/window_geometry"

SEEK_HOST = "seek.videohighlighter.invalid"
VIDEOS_META = "videohighlighter-videos"      # <meta name=... content="a|b">

# Injected once the page is ready. Kept dependency-free (no QWebChannel): a
# click becomes a navigation the page object refuses.
#
# The time clicked last glows -- the app's highlighter yellow, the same glow
# as the website's active tab -- so the reader can tell which moment the
# playhead was sent to. It is remembered per page (localStorage, by the
# element's place among the page's times), so it survives the reload when a
# click in another video reopens the viewer, and the page scrolls back to it.
LINKIFY_JS = r"""
(function () {
  if (window.__vhTimes) return; window.__vhTimes = true;
  var SEL = '[data-vh-t], button.seek[data-t], button.qseek[data-t]';
  var KEY = 'vh-last-time:' + location.pathname;
  var css = document.createElement('style');
  css.textContent = '[data-vh-t]{cursor:pointer;text-decoration:underline dotted;' +
    'text-underline-offset:3px}[data-vh-t]:hover{text-decoration:underline}' +
    '.vh-show{margin:4px 0 0;font:12px system-ui,sans-serif;cursor:pointer}' +
    '.vh-current{outline:1px solid #FFF03A !important;outline-offset:1px;' +
    'border-radius:4px;background-color:rgba(255,240,58,.14) !important;' +
    'box-shadow:0 0 8px rgba(255,240,58,.55),0 0 22px rgba(255,240,58,.22) !important;' +
    'text-shadow:0 0 8px rgba(255,240,58,.55)}';
  (document.head || document.documentElement).appendChild(css);
  document.querySelectorAll('video[src*="#t="]').forEach(function (v) {
    var m = /#t=([\d.]+)/.exec(v.getAttribute('src') || '');
    if (!m) return;
    var b = document.createElement('button');
    b.className = 'vh-show'; b.textContent = 'Show in timeline'; b.dataset.vhT = m[1];
    v.insertAdjacentElement('afterend', b);
  });
  function mark(el) {
    document.querySelectorAll('.vh-current').forEach(function (o) {
      o.classList.remove('vh-current');
    });
    if (el) el.classList.add('vh-current');
  }
  try {
    var kept = localStorage.getItem(KEY);
    if (kept !== null) {
      var el = document.querySelectorAll(SEL)[parseInt(kept, 10)];
      if (el) { mark(el); el.scrollIntoView({block: 'center'}); }
    }
  } catch (err) {}
  document.addEventListener('click', function (e) {
    var el = e.target.closest && e.target.closest(SEL);
    if (!el) return;
    e.preventDefault(); e.stopPropagation();
    mark(el);
    try {
      localStorage.setItem(KEY, Array.prototype.indexOf.call(document.querySelectorAll(SEL), el));
    } catch (err) {}
    var t = el.dataset.vhT !== undefined ? el.dataset.vhT : el.dataset.t;
    var q = 't=' + encodeURIComponent(t);
    if (el.dataset.vhVideo) q += '&video=' + encodeURIComponent(el.dataset.vhVideo);
    window.location.href = 'https://__HOST__/?' + q;
  }, true);
})();
""".replace("__HOST__", SEEK_HOST)


def seek_url(seconds: float, video: Optional[str] = None) -> str:
    """The address a click on a time navigates to."""
    query = {"t": f"{float(seconds):.2f}"}
    if video:
        query["video"] = str(video)
    return f"https://{SEEK_HOST}/?{urlencode(query)}"


def parse_seek(url: str) -> Optional[Tuple[float, Optional[str]]]:
    """``(seconds, video or None)`` for a seek address, ``None`` for any other."""
    m = re.match(r"https?://([^/?#]+)/?\?(.*)$", str(url or ""))
    if not m or m.group(1).lower() != SEEK_HOST:
        return None
    query = parse_qs(m.group(2))
    try:
        seconds = float((query.get("t") or [""])[0])
    except ValueError:
        return None
    if seconds != seconds or seconds < 0:          # NaN or negative
        return None
    video = (query.get("video") or [""])[0] or None
    return seconds, video


def same_file(a: Optional[str], b: Optional[str]) -> bool:
    return bool(a and b) and (os.path.normcase(os.path.abspath(a))
                              == os.path.normcase(os.path.abspath(b)))


def videos_named_in(html_path: str) -> List[str]:
    """The videos a page says it is about (its ``videohighlighter-videos`` meta)."""
    try:
        with open(html_path, encoding="utf-8", errors="replace") as fh:
            head = fh.read(16384)
    except OSError:
        return []
    m = re.search(r'<meta\s+name="%s"\s+content="([^"]*)"' % re.escape(VIDEOS_META),
                  head, re.I)
    return [_html.unescape(p) for p in m.group(1).split("|") if p] if m else []


def reports_for(video_path: str) -> List[str]:
    """Pages about ``video_path`` beside it, newest first.

    Its own highlight reports by name (``<stem>*_why.html``), and any page in
    the folder whose meta lists it -- a comparison of several videos.
    """
    if not video_path:
        return []
    folder = os.path.dirname(os.path.abspath(video_path))
    stem = os.path.splitext(os.path.basename(video_path))[0]
    found = []
    try:
        names = os.listdir(folder)
    except OSError:
        return []
    for name in names:
        if not name.lower().endswith(".html"):
            continue
        path = os.path.join(folder, name)
        if name.startswith(stem) and name.lower().endswith("_why.html"):
            found.append(path)
        elif any(same_file(v, video_path) for v in videos_named_in(path)):
            found.append(path)
    return sorted(found, key=lambda p: -os.path.getmtime(p))


def web_engine_available() -> bool:
    try:
        from PySide6.QtWebEngineWidgets import QWebEngineView  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - missing module or its process
        return False


class ReportPanel(QWidget):
    """A report page with clickable times. ``seek_requested(seconds, video)``:
    ``video`` is empty when the time belongs to the viewer's own video."""

    seek_requested = Signal(float, str)

    def __init__(self, video_path: str = "", parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.path: Optional[str] = None
        self.view = None

        top = QHBoxLayout()
        self.picker = QComboBox()
        self.picker.setToolTip("Reports about this video found beside it")
        # Report names are long; sized to them, the combo alone pushed the
        # viewer's minimum past a laptop screen (tests/test_timeline_min_size.py).
        self.picker.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.picker.setMinimumContentsLength(6)
        self.picker.activated.connect(lambda i: self.load(self.picker.itemData(i)))
        open_btn = QPushButton("Open…")
        open_btn.setToolTip("Show any HTML report here")
        open_btn.clicked.connect(self._choose)
        browser_btn = QPushButton("In browser")
        browser_btn.setToolTip("Open this report in the web browser")
        browser_btn.clicked.connect(self._in_browser)
        self.pop_btn = QPushButton("Pop out")
        self.pop_btn.setCheckable(True)
        self.pop_btn.setToolTip("Show the report in a window of its own, which Windows "
                                "can snap to a side of the screen")
        # The picker on a row of its own: side by side with the buttons, the
        # row alone was wider than the viewer's column may be on a laptop.
        top.addWidget(open_btn)
        top.addWidget(browser_btn)
        top.addWidget(self.pop_btn)
        top.addStretch(1)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.picker)
        lay.addLayout(top)
        if web_engine_available():
            from PySide6.QtWebEngineCore import QWebEngineScript
            from PySide6.QtWebEngineWidgets import QWebEngineView
            self.view = QWebEngineView(self)
            page = _make_page(self.view, self._on_seek)
            script = QWebEngineScript()
            script.setName("videohighlighter-times")
            script.setSourceCode(LINKIFY_JS)
            script.setInjectionPoint(QWebEngineScript.InjectionPoint.DocumentReady)
            script.setWorldId(QWebEngineScript.ScriptWorldId.MainWorld)
            page.scripts().insert(script)
            self.view.setPage(page)
            lay.addWidget(self.view, 1)
            self.note = None
            # Qt releases the browser profile at exit before Python gets round
            # to the widgets, and warns "WebEnginePage still not deleted.
            # Expect troubles" -- a crash on quit, at worst. Drop the view while
            # the event loop still runs.
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                app.aboutToQuit.connect(self._drop_view)
        else:
            self.note = QLabel("This build has no web view, so reports open in the "
                               "browser (“In browser”). Times are clickable only here.")
            self.note.setWordWrap(True)
            lay.addWidget(self.note, 1)
        self.refresh()

    # -- which page ----------------------------------------------------------
    def refresh(self) -> None:
        """Re-list the reports beside the video, keeping the one shown."""
        found = reports_for(self.video_path)
        if self.path and not any(same_file(self.path, p) for p in found):
            found.insert(0, self.path)
        self.picker.clear()
        if not found:
            self.picker.addItem("No report beside this video yet", None)
        for p in found:
            self.picker.addItem(os.path.basename(p), p)
            self.picker.setItemData(self.picker.count() - 1, p, 3)   # tooltip role
        if self.path is None and found:
            self.load(found[0])
        elif self.path:
            self.picker.setCurrentIndex(max(0, self.picker.findData(self.path)))

    def load(self, path: Optional[str]) -> None:
        if not path or not os.path.isfile(path):
            return
        self.path = path
        i = self.picker.findData(path)
        if i < 0:
            self.picker.insertItem(0, os.path.basename(path), path)
            i = 0
        self.picker.setCurrentIndex(i)
        if self.view is not None:
            self.view.load(QUrl.fromLocalFile(path))

    def _drop_view(self) -> None:
        view, self.view = self.view, None
        if view is not None:
            try:
                import shiboken6
                if shiboken6.isValid(view):
                    shiboken6.delete(view)        # its page goes with it
            except Exception as exc:  # noqa: BLE001 - quitting anyway
                print(f"⚠️ Report view not released: {exc}")

    def _choose(self) -> None:
        start = os.path.dirname(self.path or self.video_path or "")
        path, _ = QFileDialog.getOpenFileName(self, "Show a report", start,
                                              "Reports (*.html *.htm)")
        if path:
            self.load(path)

    def _in_browser(self) -> None:
        if self.path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.path))

    def _on_seek(self, seconds: float, video: Optional[str]) -> None:
        if video and same_file(video, self.video_path):
            video = None
        # Out of the navigation callback first: opening another video builds a
        # whole viewer and pumps events, which is no place to be inside
        # Chromium's navigation hook.
        QTimer.singleShot(0, lambda: self.seek_requested.emit(float(seconds), video or ""))


def _make_page(view, on_seek):
    """The page that turns seek navigations into calls and sends other links
    to the browser. Built here so the import stays optional."""
    from PySide6.QtWebEngineCore import QWebEnginePage

    class _Page(QWebEnginePage):
        def acceptNavigationRequest(self, url, nav_type, is_main_frame):
            text = url.toString()
            seek = parse_seek(text)
            if seek is not None:
                on_seek(*seek)
                return False
            if url.scheme() in ("http", "https", "mailto"):
                QDesktopServices.openUrl(url)      # never browse inside the panel
                return False
            return True                            # the report file itself, its anchors

    return _Page(view)


class ReportWindow(QWidget):
    """The report in a window of its own.

    A real top-level window rather than a floating dock: Qt floats a dock as a
    tool window, and Windows neither snaps a tool window to a screen edge nor
    gives it Win+arrow, which is the whole point of having the report beside
    the viewer. ``closed`` fires on the window's own close button."""

    closed = Signal()

    def __init__(self, title: str):
        super().__init__(None, Qt.WindowType.Window)
        self.setWindowTitle(title)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._closing_from_code = False

    def close_quietly(self) -> None:
        self._closing_from_code = True
        self.close()
        self._closing_from_code = False

    def closeEvent(self, event):              # noqa: N802 - Qt override
        QSettings().setValue(GEOMETRY_KEY, self.saveGeometry())
        super().closeEvent(event)
        if not self._closing_from_code:
            # After the close has finished: docking back closes this window
            # too, which must not happen inside its own closeEvent.
            QTimer.singleShot(0, self.closed.emit)


class ReportHost(QObject):
    """Where a viewer's panel lives: its Report tab, or the app's report window.

    There is **one** report window for the whole app (:func:`shared_window`),
    not one per viewer. A click on a time in another video switches the
    viewer's tab; the report must stay exactly as it was -- same page, same
    scroll, the glowing time -- and stay on top for the next click. With a
    window per viewer, the switch hid one window and showed another: a fresh
    copy of the page, from the top, which read as the report closing.

    The window shows one panel at a time: popping another viewer's report out
    puts the previous one back in its own tab. A new viewer pops out by itself
    (the remembered choice) only when the window is free, so opening a video
    from a report never takes the report's place. Moving between the tab and
    the window keeps the panel itself, so nothing reloads.
    """

    def __init__(self, viewer, dock, panel: ReportPanel, title: str = "Report"):
        super().__init__(viewer)
        self.viewer, self.dock, self.panel = viewer, dock, panel
        self.title = title
        # The dock holds a holder rather than the panel itself, so the panel
        # can leave and come back without the dock losing its widget.
        self.holder = QWidget()
        lay = QVBoxLayout(self.holder)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(panel)
        dock.setWidget(self.holder)
        panel.pop_btn.toggled.connect(self.set_popped)
        viewer.installEventFilter(self)
        viewer.destroyed.connect(self._viewer_gone)
        if _setting_bool(POPPED_KEY):
            QTimer.singleShot(0, self._pop_if_free)

    @property
    def popped(self) -> bool:
        return self.panel.pop_btn.isChecked()

    @property
    def window(self) -> Optional["ReportWindow"]:
        win = _SHARED.get("window")
        return win if win is not None and _SHARED.get("host") is self else None

    def _pop_if_free(self) -> None:
        if _SHARED.get("host") is None:
            self.panel.pop_btn.setChecked(True)

    def set_popped(self, on: bool) -> None:
        if on:
            QSettings().setValue(POPPED_KEY, True)
            win = shared_window()
            other = _SHARED.get("host")
            if other is not None and other is not self:
                other.take_back()                 # one report in the window at a time
            _SHARED["host"] = self
            win.setWindowTitle(self.title)
            win.layout().addWidget(self.panel)
            self.panel.pop_btn.setText("Dock")
            self.dock.hide()
            win.show()
            win.raise_()
        else:
            if _SHARED.get("host") is self:
                QSettings().setValue(POPPED_KEY, False)
                _SHARED["host"] = None
                win = _SHARED.get("window")
                if win is not None and win.isVisible():
                    win.close_quietly()
            self._to_dock()
        self.panel.show()

    def take_back(self) -> None:
        """Back into this viewer's tab without touching the window or the
        remembered choice (another viewer's report is taking the window)."""
        if _SHARED.get("host") is self:
            _SHARED["host"] = None
        self.panel.pop_btn.blockSignals(True)
        self.panel.pop_btn.setChecked(False)
        self.panel.pop_btn.blockSignals(False)
        self._to_dock()
        self.panel.show()

    def _to_dock(self) -> None:
        self.holder.layout().addWidget(self.panel)
        self.panel.pop_btn.setText("Pop out")
        self.dock.show()

    def reveal(self) -> None:
        """Bring the report forward wherever it lives."""
        win = self.window
        if self.popped and win is not None:
            win.show()
            win.raise_()
            win.activateWindow()
        else:
            self.dock.show()
            self.dock.raise_()

    def keep_on_top(self) -> None:
        """After the viewer window has come forward (a click switched videos),
        put the report back above it, so the next click is where it was."""
        win = self.window
        if win is not None and win.isVisible():
            QTimer.singleShot(60, lambda: (win.raise_(), win.activateWindow()))

    def eventFilter(self, obj, event):         # noqa: N802 - Qt override
        # A tab switch hides the viewer; the report window stays. Only closing
        # the viewer (its tab, or the whole window) takes its report home.
        # The remembered choice stays: closing a film is not "dock the report".
        if obj is self.viewer and event.type() == QEvent.Type.Close and self.window is not None:
            win = self.window
            self.take_back()
            win.close_quietly()
        return False

    def _viewer_gone(self, *_):
        if _SHARED.get("host") is self:
            _SHARED["host"] = None
            win = _SHARED.get("window")
            if win is not None:
                win.close_quietly()


# The one report window, and which viewer's panel is in it.
_SHARED: dict = {"window": None, "host": None}


def shared_window() -> "ReportWindow":
    win = _SHARED.get("window")
    if win is None:
        win = ReportWindow("Report")
        geometry = QSettings().value(GEOMETRY_KEY)
        if geometry is None or not win.restoreGeometry(geometry):
            win.resize(760, 900)
        win.closed.connect(_window_closed_by_user)
        _SHARED["window"] = win
    return win


def _window_closed_by_user() -> None:
    host = _SHARED.get("host")
    if host is not None:
        host.panel.pop_btn.setChecked(False)      # docks it back and forgets the choice


def _setting_bool(key: str) -> bool:
    value = QSettings().value(key, False)
    return value is True or str(value).lower() in ("true", "1")
