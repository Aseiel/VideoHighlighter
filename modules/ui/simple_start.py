"""Simple view — a lasting one-button workspace, not a splash into the full UI.

People who do not want scoring knobs stay here: drop a video, pick a
highlight length, press Analyze. That run uses a built-in default (motion
peaks + loudness bursts) and does not rewrite the Detailed settings widgets.
Professionals open Detailed settings and use Run Highlighter as before.

Asking *why* a moment was picked belongs here too, so the assistant panel is
folded into this page rather than being a reason to leave it. There is one
LLMChatWidget in the app; `attach_chat` borrows it and the LLM Chat tab takes
it back when Detailed settings opens, so no second model is ever loaded.
"""
from __future__ import annotations

import os

from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFrame, QHBoxLayout, QLabel, QListWidget, QProgressBar,
    QPushButton, QScrollArea, QSizePolicy, QTextEdit, QVBoxLayout, QWidget,
)

from modules.system.app_paths import resource_path
from modules.segments.simple_run import SIMPLE_LENGTHS, idle_status_text, is_video_path
from modules.ui.collapsible import CollapsibleSection
from modules.ui.theme import DARK
from version import __build_date__, __edition__, __version__

SETTINGS_ORG = "VideoHighlighter"
SETTINGS_APP = "ui-sections"
SETTINGS_KEY = "ui/simple_start"

LOGO_ASSET = os.path.join("assets", "icon.png")


def build_brand_header(parent: QWidget | None = None) -> QWidget:
    """Logo + product line — same identity as the splash, compact for the home screen."""
    p = DARK
    bar = QWidget(parent)
    row = QHBoxLayout(bar)
    row.setContentsMargins(0, 0, 0, 4)
    row.setSpacing(14)

    logo = QLabel()
    logo.setStyleSheet("background: transparent;")
    pix = QPixmap(resource_path(LOGO_ASSET))
    if not pix.isNull():
        logo.setPixmap(pix.scaled(
            52, 52,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        ))
    row.addWidget(logo, 0, Qt.AlignmentFlag.AlignTop)

    text_col = QVBoxLayout()
    text_col.setSpacing(2)
    title = QLabel(f"Video Highlighter {__edition__}")
    title.setStyleSheet(
        f"color: {p.text}; font-size: 15pt; font-weight: 600;"
        "background: transparent;")
    text_col.addWidget(title)
    meta = QLabel(f"Version {__version__}  ·  {__build_date__}")
    meta.setStyleSheet(
        f"color: {p.text_mute}; font-size: 9.5pt; background: transparent;")
    text_col.addWidget(meta)
    row.addLayout(text_col)
    row.addStretch()
    return bar


def simple_start_enabled(default: bool = True) -> bool:
    stored = QSettings(SETTINGS_ORG, SETTINGS_APP).value(SETTINGS_KEY, None)
    if stored is None:
        return default
    return str(stored).lower() in ("true", "1")


def persist_simple_start(on: bool) -> None:
    QSettings(SETTINGS_ORG, SETTINGS_APP).setValue(SETTINGS_KEY, on)


class DropZone(QFrame):
    """Click or drop videos here. Forwards paths to the host GUI file list."""

    def __init__(self, on_paths, on_browse, parent=None):
        super().__init__(parent)
        self._on_paths = on_paths
        self._on_browse = on_browse
        self.setAcceptDrops(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("simpleDrop")
        self.setMinimumHeight(180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignCenter)
        p = DARK
        self._title = QLabel("Drop a video here")
        self._title.setStyleSheet(f"color: {p.text}; font-size: 16pt; font-weight: 600;"
                                  "background: transparent; border: none;")
        self._title.setAlignment(Qt.AlignCenter)
        self._title.setWordWrap(True)
        self._hint = QLabel("or click to choose a file  ·  mp4, mov, mkv, avi")
        self._hint.setStyleSheet(f"color: {p.text_dim}; background: transparent; border: none;")
        self._hint.setAlignment(Qt.AlignCenter)
        self._hint.setWordWrap(True)
        lay.addWidget(self._title)
        lay.addWidget(self._hint)
        self.set_loaded([])

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._on_browse()
        super().mousePressEvent(event)

    def dragEnterEvent(self, event):  # noqa: N802
        if _video_urls(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):  # noqa: N802
        paths = _video_urls(event.mimeData())
        if paths:
            self._on_paths(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    def set_loaded(self, paths: list[str]) -> None:
        """Show the accepted file(s) on the drop zone itself.

        The file list under the zone is easy to miss; the zone is what the
        user just interacted with, so that is where "yes, it landed" belongs.
        """
        names = [os.path.basename(p) for p in paths if p]
        if not names:
            self._title.setText("Drop a video here")
            self._hint.setText("or click to choose a file  ·  mp4, mov, mkv, avi")
            self._set_chrome(loaded=False)
            return
        if len(names) == 1:
            self._title.setText(f"Loaded: {names[0]}")
            self._hint.setText("click to add another  ·  or drop more here")
        else:
            self._title.setText(f"Loaded: {len(names)} videos")
            extra = f"  ·  +{len(names) - 2} more" if len(names) > 2 else ""
            self._hint.setText(", ".join(names[:2]) + extra)
        self._set_chrome(loaded=True)

    def _set_chrome(self, loaded: bool) -> None:
        p = DARK
        border = (f"2px solid {p.accent}" if loaded
                  else f"2px dashed {p.border_strong}")
        bg = p.surface_hi if loaded else p.surface
        self.setStyleSheet(f"""
            QFrame#simpleDrop {{
                background: {bg};
                border: {border};
                border-radius: {p.radius_card}px;
            }}
            QFrame#simpleDrop:hover {{
                border-color: {p.accent};
                background: {p.surface_hi};
            }}
        """)


def _video_urls(mime) -> list[str]:
    if not mime.hasUrls():
        return []
    out = []
    for url in mime.urls():
        path = url.toLocalFile()
        if path and is_video_path(path):
            out.append(path)
    return out


class SimpleStartPage(QWidget):
    """Workspace for people who want a highlight without learning the knobs."""

    def __init__(self, gui: QWidget, parent: QWidget | None = None):
        super().__init__(parent if parent is not None else gui)
        self._gui = gui
        p = DARK

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 20, 28, 16)
        root.setSpacing(10)

        root.addWidget(build_brand_header(self))

        headline = QLabel("Find and explain the moments that matter")
        headline.setWordWrap(True)
        headline.setStyleSheet(
            f"color: {p.text}; font-size: 18pt; font-weight: 600;")
        root.addWidget(headline)

        blurb = QLabel(
            "Footage stays on your disk. Analyze finds strong moments with "
            "built-in defaults (motion peaks and loudness), writes a highlight "
            "reel plus separate clips, and shows why each moment scored — "
            "timeline, report, and chat. Open Detailed settings only when you "
            "need full control or to teach it what to look for.")
        blurb.setWordWrap(True)
        blurb.setStyleSheet(f"color: {p.text_dim}; font-size: 11pt;")
        root.addWidget(blurb)

        self.drop = DropZone(self._add_paths, self._browse, self)
        root.addWidget(self.drop, 1)

        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(64)
        self.file_list.setVisible(False)
        root.addWidget(self.file_list)

        length_row = QHBoxLayout()
        length_lab = QLabel("Highlight length")
        length_lab.setStyleSheet(f"color: {p.text_dim};")
        self.length = QComboBox()
        self.length.addItem("Short  — about 1–2 minutes", "short")
        self.length.addItem("Medium  — about 4 minutes", "medium")
        self.length.addItem("Longer  — about 7 minutes", "long")
        self.length.setCurrentIndex(1)
        self.length.setMinimumWidth(200)
        length_row.addWidget(length_lab)
        length_row.addWidget(self.length)
        length_row.addStretch()
        self.clear_btn = QPushButton("Remove")
        self.clear_btn.clicked.connect(self._remove_selected)
        self.clear_btn.setVisible(False)
        length_row.addWidget(self.clear_btn)
        root.addLayout(length_row)

        self.status = QLabel("Ready")
        self.status.setStyleSheet(f"color: {p.text_dim}; font-weight: 600;")
        root.addWidget(self.status)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        actions = QHBoxLayout()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(gui.cancel_pipeline)
        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.setMinimumHeight(48)
        self.analyze_btn.setMinimumWidth(180)
        self.analyze_btn.setStyleSheet(
            f"QPushButton {{ background-color: {p.success}; color: {p.on_accent};"
            f" font-weight: 700; font-size: 14pt; padding: 10px 32px;"
            f" border-radius: {p.radius}px; }}"
            f"QPushButton:hover {{ background-color: #46c160; }}"
            f"QPushButton:disabled {{ background-color: {p.surface};"
            f" color: {p.text_mute}; }}"
        )
        self.analyze_btn.clicked.connect(lambda: gui.toggle_run(simple=True))
        actions.addWidget(self.cancel_btn)
        actions.addStretch()
        actions.addWidget(self.analyze_btn)
        root.addLayout(actions)

        result_row = QHBoxLayout()
        self.timeline_btn = QPushButton("Open timeline")
        self.timeline_btn.setVisible(False)
        self.timeline_btn.clicked.connect(gui.open_timeline_viewer)
        self.report_btn = QPushButton("Open report")
        self.report_btn.setVisible(False)
        self.report_btn.clicked.connect(gui.open_why_report)
        result_row.addWidget(self.timeline_btn)
        result_row.addWidget(self.report_btn)
        result_row.addStretch()
        root.addLayout(result_row)

        self.chat_section = CollapsibleSection(
            "Ask about this video", self,
            expanded=False, settings_key="ui/simple_chat")
        self._chat_layout = QVBoxLayout()
        self._chat_layout.setSpacing(4)
        # The chat panel's control rows don't wrap, so its ~1000px minimum width
        # became this page's minimum width — which pushed Analyze off the right
        # edge behind a horizontal scrollbar. Given its own scroller the panel
        # keeps its rows intact and stops dictating the width of the page.
        self._chat_host = QScrollArea()
        self._chat_host.setWidgetResizable(True)
        self._chat_host.setFrameShape(QScrollArea.NoFrame)
        self._chat_host.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._chat_host.setMinimumHeight(560)
        self._chat_layout.addWidget(self._chat_host)
        self.chat_section.setContentLayout(self._chat_layout)
        self.chat_section.set_hint("why a moment was picked")
        root.addWidget(self.chat_section)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(96)
        self.log.setPlaceholderText("Progress shows up here while Analyze runs.")
        self.log.setStyleSheet(
            "QTextEdit { font-family: 'Courier New', monospace; font-size: 9pt; }")
        root.addWidget(self.log)

        detailed = QPushButton("Detailed settings…")
        detailed.setFlat(True)
        detailed.setCursor(Qt.PointingHandCursor)
        detailed.setStyleSheet(f"QPushButton {{ color: {p.text_dim}; text-align: left; }}"
                               f"QPushButton:hover {{ color: {p.accent}; }}")
        detailed.setToolTip(
            "Scoring, objects, transcript, training, and every other control. "
            "Optional — you are not sent here automatically.")
        detailed.clicked.connect(lambda: gui.set_simple_start(False))
        root.addWidget(detailed)

        self.refresh_files()

    def length_key(self) -> str:
        data = self.length.currentData()
        return data if data in SIMPLE_LENGTHS else "medium"

    def refresh_files(self, *, update_idle_status: bool = True) -> None:
        self.file_list.clear()
        paths = self._gui.get_file_list()
        for path in paths:
            self.file_list.addItem(path)
        has = bool(paths)
        self.file_list.setVisible(has)
        self.clear_btn.setVisible(has)
        self.drop.set_loaded(paths)
        if update_idle_status:
            self.status.setText(idle_status_text(len(paths)))

    def show_results(self, on: bool) -> None:
        self.timeline_btn.setVisible(on)
        self.report_btn.setVisible(on)

    def attach_chat(self, chat: QWidget | None) -> None:
        """Borrow the app's one chat panel into the folded section.

        addWidget reparents, so the LLM Chat tab is emptied for as long as this
        view is the one on screen and refilled the moment Detailed opens. State,
        loaded model and analysis cache all survive the move.
        """
        if chat is None or self.owns_chat(chat):
            return
        self._chat_host.setWidget(chat)
        chat.setVisible(True)

    def release_chat(self, chat: QWidget | None) -> None:
        """Hand the panel back so the LLM Chat tab can re-adopt it."""
        if self.owns_chat(chat):
            self._chat_host.takeWidget()

    def owns_chat(self, chat: QWidget | None) -> bool:
        return chat is not None and self._chat_host.widget() is chat

    def reveal_chat(self) -> None:
        """Unfold the chat and scroll it into view — used when the app hands the
        assistant something to talk about (a finished report)."""
        self.chat_section.expand(True)
        area = self._scroll_area()
        if area is not None:
            QTimer.singleShot(
                0, lambda: area.ensureWidgetVisible(self.chat_section))

    def _scroll_area(self):
        node = self.parentWidget()
        while node is not None:
            if isinstance(node, QScrollArea):
                return node
            node = node.parentWidget()
        return None

    def _browse(self) -> None:
        self._gui.browse_files()
        self.refresh_files()

    def _add_paths(self, paths: list[str]) -> None:
        existing = set(self._gui.get_file_list())
        for path in paths:
            if path not in existing:
                self._gui.file_list.addItem(path)
                existing.add(path)
        if paths:
            from os.path import basename, exists, splitext
            first = paths[0]
            if exists(first):
                self._gui.update_video_duration(first)
            out = self._gui.output_input.text().strip()
            if not out or out == "highlight.mp4":
                self._gui.output_input.setText(
                    f"{splitext(basename(first))[0]}_highlight.mp4")
        self.refresh_files()

    def _remove_selected(self) -> None:
        row = self.file_list.currentRow()
        if row < 0:
            row = self.file_list.count() - 1
        if row >= 0:
            self._gui.file_list.takeItem(row)
        self.refresh_files()

    def sync_run_chrome(self) -> None:
        gui = self._gui
        self.analyze_btn.setEnabled(gui.run_btn.isEnabled())
        text = gui.run_btn.text()
        if text == "Run Highlighter":
            self.analyze_btn.setText("Analyze")
        else:
            self.analyze_btn.setText(text)
        self.cancel_btn.setEnabled(gui.cancel_btn.isEnabled())
        busy = not gui.file_list.isEnabled()
        self.drop.setEnabled(not busy)
        self.clear_btn.setEnabled(not busy)
        self.length.setEnabled(not busy)
        self.file_list.setEnabled(gui.file_list.isEnabled())
        task = gui.task_label.text() or ""
        done = "Complete" in task
        self.show_results(done)
        self.chat_section.set_hint(
            "ask why these moments were picked" if done
            else "why a moment was picked")
        self.progress.setVisible(gui.process_progress_bar.isVisible()
                                 or gui.progress_group.isVisible())
        self.progress.setRange(gui.process_progress_bar.minimum(),
                               gui.process_progress_bar.maximum())
        self.progress.setValue(gui.process_progress_bar.value())
        self.refresh_files(update_idle_status=not busy and not done)
        if busy or done:
            self.status.setText(task)

    def append_log(self, text: str) -> None:
        from PySide6.QtGui import QTextCursor
        bar = self.log.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 4
        cursor = QTextCursor(self.log.document())
        cursor.movePosition(QTextCursor.End)
        if not self.log.document().isEmpty():
            cursor.insertBlock()
        cursor.insertText(text)
        if at_bottom:
            bar.setValue(bar.maximum())
