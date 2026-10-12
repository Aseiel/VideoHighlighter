"""Datasets, inside the app: make one from your own footage, step by step.

The list of datasets you have, and **New dataset…**; opening one walks it
through four steps, each a page:

    1 Videos   add footage; cut it into 5 s samples (and group them)
    2 Groups   samples that look alike, together: say what each group shows
    3 Check    check the machine's guesses for everything not yet decided
    4 Health   is it ready to train, and how good was the automatic sorting

Like the "From videos" tab, every button runs exactly one ``modules.teach``
command (``cli.run``) on a worker thread, so the app and the command line
cannot disagree. Pages only *read* the project to show it.
"""
from __future__ import annotations

import html
import io
import os
import re
import sys
import threading
from typing import Callable, Optional

from PySide6.QtCore import QObject, QSize, QThread, QUrl, Qt, Signal
from PySide6.QtGui import QColor, QDesktopServices, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMenu, QMessageBox, QProgressBar, QPushButton,
    QScrollArea,
    QSpinBox, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from modules.teach.teach_panel import _Job, describe

GREEN, AMBER, RED, GREY = "#2e9d57", "#d4a017", "#c0392b", "#7f8c8d"
BLUE = "#3a8ee6"
# Tiles a folder in Browse shows at once; "Show more" adds as many again.
PAGE = 240
LEVEL_COLOURS = {"ok": GREEN, "warn": AMBER, "bad": RED}
HEALTH_TEXT = {"ready": ("Healthy", GREEN), "needs_videos": ("Needs more videos", AMBER),
               "not_ready": ("Not ready", RED)}
STATE_TEXT = {"open": ("not named yet", AMBER), "named": ("named", GREEN),
              "nothing": ("nothing interesting", GREY), "mixed": ("mixed", AMBER)}
SOURCE_TEXT = {"group": "Groups (tiles you were shown)", "sort": "Sorting (review guesses)",
               "rule": "Rules (objects found)",
               "auto": "Auto-accept (spot checks)"}
THUMB_WIDTH = 168
TILES_PER_ROW = 6

STEPS = ("Videos", "Groups", "Browse", "Check guesses", "Health")
VIDEOS, GROUPS, BROWSE, CHECK, HEALTH = range(5)
# Which page a `status` next step belongs on.
STEP_OF = {"add-video": VIDEOS, "cut": VIDEOS, "focus": VIDEOS, "group": GROUPS,
           "name-group": GROUPS, "add-class": GROUPS, "sort": CHECK, "review": CHECK,
           "boxes": CHECK, "find": CHECK, "add-example": CHECK, "detect": CHECK, "build": HEALTH,
           "train": HEALTH}
# What a command (or an ``auto: <step>`` line) is doing, for the progress line.
DOING = {"add-video": "Adding videos", "cut": "Cutting into samples",
         "focus": "Cropping samples to the people in them",
         "group": "Grouping samples that look alike", "sort": "Sorting samples",
         "build": "Writing the dataset", "train": "Training", "auto": "Working",
         "delete": "Deleting"}
# Progress lines with a count in them: cut, embedding, the cropper's per-video
# "[i/n]" (and not its per-frame "Frame i/n", which would make the bar jump).
COUNTED = re.compile(r"(?:^cut \S+:|embedded|vectors|\[)\s*(\d+)/(\d+)")


def _item(text, colour: str = "", align_right: bool = False) -> QTableWidgetItem:
    item = QTableWidgetItem("" if text is None else str(text))
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    if colour:
        item.setForeground(QColor(colour))
    if align_right:
        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return item


def _table(headers) -> QTableWidget:
    table = QTableWidget(0, len(headers))
    table.setHorizontalHeaderLabels(list(headers))
    table.verticalHeader().setVisible(False)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setSelectionMode(QAbstractItemView.SingleSelection)
    table.setWordWrap(True)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    table.horizontalHeader().setStretchLastSection(True)
    return table


def _fill(table: QTableWidget, rows) -> None:
    table.setRowCount(0)
    for row in rows:
        r = table.rowCount()
        table.insertRow(r)
        for c, cell in enumerate(row):
            table.setItem(r, c, cell if isinstance(cell, QTableWidgetItem) else _item(cell))
    table.resizeRowsToContents()


def _percent(value) -> str:
    return "–" if value is None else f"{value:.0%}"


def _count(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _clock(seconds: float) -> str:
    seconds = int(seconds or 0)
    return f"{seconds // 60}:{seconds % 60:02d}"


# --- running commands -----------------------------------------------------------

class _Lines(io.TextIOBase):
    """``sys.stderr`` while a command runs. Everything still goes where it went
    (the debug log); each line the command's own thread writes is also handed
    to ``say``, which is how the panel shows progress the CLI prints."""

    def __init__(self, original, say: Callable):
        self.original, self.say = original, say
        self.thread = threading.get_ident()
        self._partial = ""

    def write(self, s: str) -> int:
        if self.original is not None:
            self.original.write(s)
        if self.say is not None and threading.get_ident() == self.thread:
            *lines, self._partial = re.split(r"[\r\n]", self._partial + s)
            for line in lines:
                if line.strip():
                    self.say(line.strip())
        return len(s)

    def flush(self):
        if self.original is not None:
            self.original.flush()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return getattr(self.original, "encoding", None) or "utf-8"


def _with_progress(fn: Callable, say: Callable) -> Callable:
    def run():
        lines = _Lines(sys.stderr, say)
        sys.stderr = lines
        try:
            return fn()
        finally:
            lines.say = None
            if sys.stderr is lines:     # unless someone swapped it since
                sys.stderr = lines.original
    return run


class Runner(QObject):
    """One ``modules.teach`` command at a time, off the UI thread. ``progress``
    is each line of what the command prints while it works."""

    started = Signal(list)
    finished = Signal(list, int, dict)
    progress = Signal(str)

    def __init__(self, parent=None, run_cli: Optional[Callable] = None):
        super().__init__(parent)
        from modules.teach import cli
        self._run_cli = run_cli or cli.run
        self._thread = None
        self._job = None
        self._argv: list = []

    @property
    def busy(self) -> bool:
        return self._thread is not None

    def run(self, argv: list) -> bool:
        if self.busy:
            return False
        self._argv = list(argv)
        self._thread = QThread(self)
        self._job = _Job(_with_progress(lambda: self._run_cli(self._argv),
                                        self.progress.emit))
        self._job.moveToThread(self._thread)
        self._thread.started.connect(self._job.run)
        self._job.done.connect(self._done)
        self._thread.finished.connect(self._job.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self.started.emit(self._argv)
        self._thread.start()
        return True

    def _done(self, outcome):
        self._thread.quit()
        self._thread.wait()
        self._thread = self._job = None
        code, result = outcome if isinstance(outcome, tuple) else (0, outcome)
        self.finished.emit(self._argv, int(code), dict(result or {}))

    def wait(self):
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(60000)


class _ThumbJob(QObject):
    """Middle frames of samples as small images, one signal per sample."""

    image = Signal(str, QImage)
    done = Signal()

    def __init__(self, items: list, width: int):
        super().__init__()
        self.items, self.width, self.stop = items, width, False

    def run(self):
        import cv2

        from modules.teach.embed import read_frames
        for sample_id, path in self.items:
            if self.stop:
                break
            try:
                frames = read_frames(path, 1)
                if not frames:
                    continue
                rgb = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
            except Exception as exc:            # a broken clip shows as its number
                print(f"[datasets] no thumbnail for {path}: {exc}")
                continue
            self.image.emit(sample_id, image.scaledToWidth(self.width,
                                                           Qt.SmoothTransformation))
        self.done.emit()


class Thumbnails(QObject):
    """Loads and keeps sample thumbnails; tells whoever shows one when it is in."""

    ready = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cache: dict = {}
        self._asked: set = set()
        self._running: list = []

    def get(self, sample_id: str) -> Optional[QPixmap]:
        return self.cache.get(sample_id)

    def want(self, items: list) -> None:
        items = [(i, p) for i, p in items if i not in self.cache and i not in self._asked]
        if not items:
            return
        self._asked.update(i for i, _ in items)
        thread = QThread(self)
        job = _ThumbJob(items, THUMB_WIDTH)
        job.moveToThread(thread)
        thread.started.connect(job.run)
        job.image.connect(self._got)
        job.done.connect(thread.quit)
        thread.finished.connect(lambda: self._forget(thread, job))
        self._running.append((thread, job))
        thread.start()

    def _got(self, sample_id: str, image: QImage):
        self.cache[sample_id] = QPixmap.fromImage(image)
        self.ready.emit(sample_id)

    def _forget(self, thread, job):
        self._running = [(t, j) for t, j in self._running if t is not thread]
        job.deleteLater()
        thread.deleteLater()

    def stop(self):
        for thread, job in list(self._running):
            job.stop = True
            thread.quit()
            thread.wait(10000)


# --- the list of datasets -------------------------------------------------------

class NewDatasetDialog(QDialog):
    def __init__(self, parent=None, taken=()):
        super().__init__(parent)
        self.setWindowTitle("New dataset")
        self.taken = {t.lower() for t in taken}
        self.name = QLineEdit()
        self.name.setPlaceholderText("a short name, e.g. my-first")
        self.task = QComboBox()
        self.task.addItem("Actions: something that happens over time", "actions")
        self.task.addItem("Objects: a thing visible in one frame", "objects")
        self.focus = QCheckBox("Crop samples to the people in them")
        self.focus.setToolTip("Actions only. Each sample becomes one clip per person, "
                              "labelled on its own: two people doing two things in one "
                              "shot become two samples.")
        self.focus.setChecked(True)
        self.task.currentIndexChanged.connect(
            lambda: self.focus.setEnabled(self.task.currentData() == "actions"))
        form = QFormLayout()
        form.addRow("Name", self.name)
        form.addRow("What it finds", self.task)
        form.addRow("", self.focus)
        note = QLabel("Your footage stays where it is and never leaves this computer. "
                      "The dataset is kept in the app's user data folder.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#999;")
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(note)
        layout.addWidget(buttons)

    def _accept(self):
        from modules.teach.project import slugify
        name = self.name.text().strip()
        if not name:
            QMessageBox.information(self, "New dataset", "Give it a name.")
            return
        if slugify(name) in self.taken:
            QMessageBox.information(self, "New dataset",
                                    f"There is already a dataset called {name!r}.")
            return
        self.accept()

    def args(self) -> list:
        from modules.teach.project import slugify
        name = self.name.text().strip()
        out = ["--project", slugify(name), "init", "--task", self.task.currentData(),
               "--name", name]
        if self.focus.isEnabled() and self.focus.isChecked():
            out.append("--focus")
        return out


class ListPage(QWidget):
    open_project = Signal(str)

    def __init__(self, runner: Runner, parent=None):
        super().__init__(parent)
        self.runner = runner
        self.projects: list = []
        title = QLabel("<h3>Your datasets</h3>Each one is footage you cut into samples "
                       "and labelled. Open one to carry on where you left it, or make a "
                       "new one.")
        title.setWordWrap(True)
        self.table = _table(["Name", "Finds", "Videos", "Samples", "Decided", "Classes",
                             "Health", "Next"])
        self.table.doubleClicked.connect(lambda *_: self.open_selected())
        self.empty = QLabel("No datasets yet. Press <b>New dataset…</b> to make the "
                            "first one from a video.")
        self.empty.setWordWrap(True)
        self.new_btn = QPushButton("New dataset…")
        self.new_btn.clicked.connect(self.new)
        self.open_btn = QPushButton("Open")
        self.open_btn.clicked.connect(self.open_selected)
        self.folder_btn = QPushButton("Show folder")
        self.folder_btn.clicked.connect(self.show_folder)
        self.delete_btn = QPushButton("Delete…")
        self.delete_btn.setToolTip("Move the selected dataset to the Recycle Bin (Del)")
        self.delete_btn.clicked.connect(self.delete_selected)
        delete_key = QShortcut(QKeySequence.Delete, self.table)
        delete_key.setContext(Qt.WidgetShortcut)
        delete_key.activated.connect(self.delete_selected)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        buttons = QHBoxLayout()
        for b in (self.new_btn, self.open_btn, self.folder_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(self.delete_btn)
        buttons.addWidget(self.refresh_btn)
        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addLayout(buttons)
        layout.addWidget(self.empty)
        layout.addWidget(self.table, 1)

    def refresh(self):
        self.runner.run(["projects"])

    def show_projects(self, result: dict):
        self.projects = [p for p in result.get("projects") or [] if not p.get("error")]
        rows = []
        for p in self.projects:
            health, colour = HEALTH_TEXT.get(p.get("health"), ("", ""))
            decided = (f"{p['decided']} of {p['samples']}" if p.get("samples") else "–")
            rows.append([p["name"], p["task"], p["videos"], p["samples"], decided,
                         ", ".join(p.get("classes") or []) or "–",
                         _item(health, colour), (p.get("next") or {}).get("why", "")])
        _fill(self.table, rows)
        self.empty.setVisible(not self.projects)
        self.table.setVisible(bool(self.projects))
        if self.projects:
            self.table.selectRow(0)

    def selected(self) -> str:
        row = self.table.currentRow()
        return self.projects[row]["project"] if 0 <= row < len(self.projects) else ""

    def open_selected(self):
        root = self.selected()
        if root:
            self.open_project.emit(root)

    def show_folder(self):
        from modules.teach.project import projects_root
        root = self.selected() or projects_root()
        os.makedirs(root, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(root))

    def delete_selected(self):
        row = self.table.currentRow()
        if not 0 <= row < len(self.projects) or self.runner.busy:
            return
        p = self.projects[row]
        sure = QMessageBox(QMessageBox.Warning, "Delete dataset",
                           f"Delete the dataset <b>{html.escape(p['name'])}</b>?", parent=self)
        sure.setInformativeText(
            f"Its {_count(p['samples'], 'sample')}, groups, labels and trained rounds "
            "go to the Recycle Bin (a dataset too big for the Bin is deleted for "
            "good). Videos you added from your disk stay where they are; videos "
            "downloaded from a link are kept inside the dataset and go with it. A "
            "model you installed from it stays installed.")
        sure.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        sure.button(QMessageBox.Yes).setText("Delete")
        sure.setDefaultButton(QMessageBox.Cancel)
        if sure.exec() == QMessageBox.Yes:
            self.runner.run(["--project", p["project"], "delete"])

    def new(self):
        dialog = NewDatasetDialog(self, taken=[os.path.basename(p["project"])
                                               for p in self.projects])
        if dialog.exec() == QDialog.Accepted:
            self.runner.run(dialog.args())


# --- the steps ------------------------------------------------------------------

class VideosPage(QWidget):
    def __init__(self, wizard, parent=None):
        super().__init__(parent)
        self.wizard = wizard
        intro = QLabel("<h3>1 · Videos</h3>Add the footage to learn from. It is cut into "
                       "5-second samples, and samples that look alike are put together "
                       "in groups for the next step. One video is a fine start; for a "
                       "model that works on new footage, each thing should appear in "
                       "at least 3 videos.")
        intro.setWordWrap(True)
        self.list = QListWidget()
        self.add_files = QPushButton("Add video files…")
        self.add_files.clicked.connect(self.pick_files)
        self.add_folder = QPushButton("Add a folder…")
        self.add_folder.clicked.connect(self.pick_folder)
        self.add_url = QPushButton("Add from a link…")
        self.add_url.clicked.connect(self.ask_url)
        self.prepare = QPushButton("Cut into samples and group them")
        self.prepare.setToolTip("Cuts every new video into 5 s samples (one per person "
                                "when cropping is on), then groups them.")
        self.prepare.clicked.connect(lambda: self.wizard.command(["auto"]))
        self.crop = QCheckBox("Crop samples to the people in them: one sample per person")
        self.crop.setToolTip(
            "A shot often holds two people doing two different things, and a sample "
            "has one label. Cropped, each person is a sample of their own, grouped and "
            "labelled by itself. Applies to samples not decided yet; the cropper takes "
            "a few seconds per sample.")
        self.crop.clicked.connect(
            lambda on: self.wizard.command(["set", f"focus={'true' if on else 'false'}"]))
        buttons = QHBoxLayout()
        for b in (self.add_files, self.add_folder, self.add_url):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(self.prepare)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self.list, 1)
        layout.addWidget(self.crop)
        layout.addLayout(buttons)
        self.buttons = [self.add_files, self.add_folder, self.add_url, self.prepare,
                        self.crop]

    def show_project(self, project, next_step: dict):
        self.list.clear()
        per_source, crops, whole = {}, {}, {}
        for s in project.samples:
            per_source[s.source] = per_source.get(s.source, 0) + 1
            if s.crop_of:
                crops[s.source] = crops.get(s.source, 0) + 1
        for s in project.replaced:
            whole[s.source] = whole.get(s.source, 0) + 1
        for source in project.sources:
            name = os.path.basename(source.path) or source.url
            state = (f"{per_source.get(source.id, 0)} samples" if source.cut
                     else "not cut yet")
            if crops.get(source.id):
                state += (f" ({crops[source.id]} people crops from "
                          f"{whole.get(source.id, 0)} shots)")
            length = f" · {_clock(source.duration)}" if source.duration else ""
            self.list.addItem(f"{source.id}   {name}{length}   ·   {state}")
        if not project.sources:
            self.list.addItem("No videos yet: add one below.")
        step = (next_step.get("args") or [""])[0]
        self.prepare.setText("Crop to people and group them" if step == "focus"
                             else "Group them" if step == "group"
                             else "Cut into samples and group them")
        self.prepare.setEnabled(step in ("cut", "focus", "group"))
        self.crop.setVisible(project.task == "actions")
        self.crop.setChecked(project.settings.focus)
        self.crop.setEnabled(True)
        for b in (self.add_files, self.add_folder, self.add_url):
            b.setEnabled(True)

    def pick_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Choose videos", "", "Videos (*.mp4 *.mov *.mkv *.avi *.webm *.m4v);;"
                                       "All files (*)")
        if files:
            self.wizard.command(["add-video", *files])

    def pick_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose a folder of videos")
        if folder:
            self.wizard.command(["add-video", folder])

    def ask_url(self):
        url, ok = QInputDialog.getText(self, "Add from a link", "Video link:")
        if ok and url.strip():
            self.wizard.command(["add-video", url.strip()])


class Thumb(QLabel):
    """One sample as a picture. Click selects it, Shift-click a range,
    double-click plays it. ``flag`` is a colour for what it already is."""

    clicked = Signal(object, object)          # the thumb, the keyboard modifiers

    def __init__(self, n: int, sample, mark: str = RED, flag: str = "",
                 tip: str = "", parent=None):
        super().__init__(parent)
        self.n, self.sample, self.mark, self.flag = n, sample, mark, flag
        self.selected = False
        self.setFixedSize(THUMB_WIDTH + 8, THUMB_WIDTH * 9 // 16 + 8)
        self.setAlignment(Qt.AlignCenter)
        self.setToolTip(f"{n}: {sample.source} at {_clock(sample.start)}"
                        + (f"\n{tip}" if tip else "")
                        + f"\n{os.path.abspath(sample.path)}"
                        + "\nClick: select.  Shift-click: select a range.  "
                          "Double-click: play.")
        self.picture: Optional[QPixmap] = None
        self.refresh()

    # Group cards call a selected tile "wrong": it does not belong there.
    @property
    def wrong(self) -> bool:
        return self.selected

    def set_picture(self, pixmap: QPixmap):
        self.picture = pixmap
        self.refresh()

    def set_selected(self, on: bool):
        if on != self.selected:
            self.selected = on
            self.refresh()

    def refresh(self):
        colour = self.mark if self.selected else (self.flag or "#444")
        self.setStyleSheet(f"QLabel {{ border: 4px solid {colour}; background:#111; "
                           "color:#ddd; }")
        if self.picture is not None:
            self.setPixmap(self.picture)
        else:
            self.setText(str(self.n))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self, event.modifiers())

    def mouseDoubleClickEvent(self, event):
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self.sample.path)))


class TileGrid(QWidget):
    """Thumbs in rows as wide as there is room for; keeps which are selected
    (Shift-click for a range)."""

    changed = Signal()
    CELL = THUMB_WIDTH + 12

    def __init__(self, thumbnails: "Thumbnails", columns: int = TILES_PER_ROW,
                 parent=None):
        super().__init__(parent)
        self.thumbnails, self.columns = thumbnails, columns
        self.thumbs: list = []
        self._last = None
        self.grid = QGridLayout(self)
        self.grid.setSpacing(4)
        self.grid.setContentsMargins(0, 0, 0, 0)
        thumbnails.ready.connect(self._thumb)

    def add(self, thumb: Thumb):
        i = len(self.thumbs)
        pixmap = self.thumbnails.get(thumb.sample.id)
        if pixmap is not None:
            thumb.set_picture(pixmap)
        thumb.clicked.connect(self._clicked)
        self.grid.addWidget(thumb, i // self.columns, i % self.columns)
        self.thumbs.append(thumb)

    def load(self):
        self.thumbnails.want([(t.sample.id, t.sample.path) for t in self.thumbs
                              if t.picture is None])

    # Reflow to the width there is: a fixed number of columns either leaves a
    # wide window half empty or hides tiles behind a sideways scroll bar.

    def minimumSizeHint(self) -> QSize:          # noqa: N802 (Qt override)
        return QSize(self.CELL, super().minimumSizeHint().height())

    def resizeEvent(self, event):                # noqa: N802 (Qt override)
        super().resizeEvent(event)
        columns = max(1, (event.size().width() + 4) // self.CELL)
        if columns != self.columns:
            self.columns = columns
            for thumb in self.thumbs:
                self.grid.removeWidget(thumb)
            for i, thumb in enumerate(self.thumbs):
                self.grid.addWidget(thumb, i // columns, i % columns)
            self.updateGeometry()

    def _thumb(self, sample_id: str):
        for thumb in self.thumbs:
            if thumb.sample.id == sample_id:
                thumb.set_picture(self.thumbnails.get(sample_id))

    def _clicked(self, thumb: Thumb, modifiers):
        if modifiers & Qt.ShiftModifier and self._last in self.thumbs:
            a, b = sorted((self.thumbs.index(self._last), self.thumbs.index(thumb)))
            for t in self.thumbs[a:b + 1]:
                t.set_selected(True)
        else:
            thumb.set_selected(not thumb.selected)
        self._last = thumb
        self.changed.emit()

    def select_all(self, on: bool = True):
        for t in self.thumbs:
            t.set_selected(on)
        self.changed.emit()

    def selected(self) -> list:
        return [t for t in self.thumbs if t.selected]


def describe_sample(sample) -> tuple:
    """``(text, colour)``: what a sample is now, for a tooltip and its border."""
    from modules.teach.project import ACCEPTED, NEGATIVE, REJECTED
    who = ("auto" if not sample.is_human else
           "group" if sample.decided_by.startswith("group:") else "you")
    if sample.verdict == ACCEPTED:
        return f"{sample.label} (decided by {who})", GREEN
    if sample.verdict == NEGATIVE:
        return f"nothing interesting (decided by {who})", GREY
    if sample.verdict == REJECTED:
        return f"unusable (decided by {who})", RED
    guess = sample.proposed if sample.proposed and not sample.proposed.startswith("_") \
        else ""
    return (f"not decided; guess: {guess}" if guess else "not decided"), ""


def move_menu(wizard, ids: list, parent=None, here_group: Optional[int] = None) -> QMenu:
    """Every place selected samples can be moved to, each one ``move`` command."""
    project = wizard.project
    menu = QMenu(parent)
    joined = ",".join(ids)

    def go(*where):
        wizard.command(["move", "--samples", joined, *where])

    def new_class():
        from modules.teach.naming import check_name, normalize_name
        name, ok = QInputDialog.getText(parent, "New class", "What do these show?")
        if not ok or not name.strip():
            return
        blocking = [x for x in check_name(normalize_name(name), project.class_names(),
                                          project.task, vocabulary=[]) if x.blocking]
        if blocking:
            QMessageBox.information(parent, "New class",
                                    "\n".join(x.message for x in blocking))
            return
        go("--to-class", name.strip())

    for name in project.class_names():
        menu.addAction(name, lambda n=name: go("--to-class", n))
    menu.addAction("New class…", new_class)
    menu.addSeparator()
    menu.addAction("Nothing interesting", lambda: go("--nothing"))
    menu.addAction("Unusable (bad cut, unclear)", lambda: go("--reject"))
    menu.addAction("Back to not decided", lambda: go("--undecided"))
    menu.addSeparator()
    groups = menu.addMenu("Group")
    for g in project.groups:
        if g.state == "split" or g.id == here_group:
            continue
        label = f"Group {g.id}" + (f" ({g.name})" if g.name else
                                   f" ({g.state})" if g.state != "open" else "")
        groups.addAction(label, lambda i=g.id: go("--to-group", str(i)))
    groups.addSeparator()
    groups.addAction("New group", lambda: go("--new-group"))
    for action in menu.actions():
        action.setEnabled(bool(ids) and action.isEnabled())
    return menu


class GroupCard(QFrame):
    def __init__(self, wizard, project, info: dict, parent=None):
        super().__init__(parent)
        from modules.teach import group as group_mod

        self.wizard, self.id, self.state = wizard, info["id"], info["state"]
        spec = project.get_group(self.id)
        self.setFrameShape(QFrame.StyledPanel)
        editable = self.state == "open"
        wrong = set(spec.wrong)
        self.tiles = TileGrid(wizard.thumbnails)
        for n, sample in enumerate(group_mod.tiles(project, self.id), 1):
            text, colour = describe_sample(sample)
            flag = AMBER if sample.id in wrong else ("" if editable else colour)
            self.tiles.add(Thumb(n, sample, mark=RED if editable else BLUE, flag=flag,
                                 tip=text))
        self.tiles.load()
        self.thumbs = self.tiles.thumbs
        self.tiles.changed.connect(self._count)

        text, colour = STATE_TEXT.get(self.state, (self.state, GREY))
        if self.state == "named":
            text = f"named: {info['name']}"
        more = info["size"] - len(self.thumbs)
        title = QLabel(f"<b>Group {self.id}</b> · {_count(info['size'], 'sample')} from "
                       f"{_count(info['videos'], 'video')}"
                       + (f" · {more} more not shown" if more > 0 else "")
                       + f" · <span style='color:{colour}'>{text}</span>")
        if info["videos"] == 1 and len(project.sources) > 1 and info["size"] >= 4:
            title.setToolTip("All of this group comes from one video: check it is one "
                             "thing, not one scene.")
        self.hint = QLabel()
        self.hint.setStyleSheet("color:#999;")
        self.editable = editable

        row = QHBoxLayout()
        if editable:
            self.name = QComboBox()
            self.name.setEditable(True)
            self.name.addItems(project.class_names())
            self.name.setCurrentText("")
            self.name.lineEdit().setPlaceholderText("what do these show?")
            self.name.lineEdit().returnPressed.connect(self.name_it)
            self.name.setMinimumWidth(220)
            name_btn = QPushButton("Name it")
            name_btn.setToolTip("The tiles shown, except the selected ones, are this.")
            name_btn.clicked.connect(self.name_it)
            nothing = QPushButton("Nothing interesting")
            nothing.setToolTip("These show none of the things to find. The model "
                               "learns them as 'none of these'.")
            nothing.clicked.connect(lambda: self.say(["--nothing"]))
            mixed = QPushButton("Mixed")
            mixed.setToolTip("More than one thing: split it into smaller groups, or "
                             "check its samples one by one.")
            mixed.clicked.connect(lambda: self.say(["--mixed"]))
            for w in (self.name, name_btn, nothing, mixed):
                row.addWidget(w)
        else:
            if self.state == "mixed":
                split = QPushButton("Split into smaller groups")
                split.clicked.connect(
                    lambda: self.wizard.command(["group", "--split", str(self.id)]))
                row.addWidget(split)
            change = QPushButton("Change")
            change.clicked.connect(lambda: self.say(["--reopen"]))
            row.addWidget(change)
        self.move_btn = QPushButton("Move selected to…")
        self.move_btn.clicked.connect(self._move)
        show_all = QPushButton(f"Show all {info['size']}")
        show_all.setToolTip("Every sample of this group, in Browse")
        show_all.clicked.connect(lambda: self.wizard.browse(("group", self.id)))
        row.addWidget(self.move_btn)
        row.addWidget(show_all)
        row.addWidget(self.hint, 1)
        self._wrong_before = len(wrong)
        self._count()

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(self.tiles)
        layout.addLayout(row)

    def _count(self):
        n = len(self.tiles.selected())
        self.move_btn.setEnabled(n > 0)
        if n:
            self.hint.setText(f"{n} selected" + (": naming leaves them out"
                                                 if self.editable else ""))
        elif self.editable:
            self.hint.setText("Select tiles that do not belong, then name the group.")
        elif self.state == "mixed":
            self.hint.setText("Checked one by one in step 4 unless you split it.")
        elif self._wrong_before:
            self.hint.setText(f"Named with {self._wrong_before} left out as not "
                              "belonging" + (" (amber)." if any(
                                  t.flag == AMBER for t in self.thumbs) else "."))
        else:
            self.hint.setText("")

    def wrong(self) -> str:
        return ",".join(str(t.n) for t in self.tiles.selected())

    def _move(self):
        ids = [t.sample.id for t in self.tiles.selected()]
        move_menu(self.wizard, ids, self, here_group=self.id).exec(
            self.move_btn.mapToGlobal(self.move_btn.rect().bottomLeft()))

    def name_it(self):
        name = self.name.currentText().strip()
        if not name:
            self.name.setFocus()
            return
        from modules.teach.naming import check_name, normalize_name
        clean = normalize_name(name)
        others = [n for n in self.wizard.project.class_names() if n != clean]
        problems = check_name(clean, others, self.wizard.project.task, vocabulary=[])
        blocking = [p for p in problems if p.blocking]
        if blocking:
            QMessageBox.information(self, "Name", "\n".join(p.message for p in blocking))
            return
        self.say(["--name", name])

    def say(self, what: list):
        args = ["name-group", str(self.id), *what]
        if "--name" in what or "--nothing" in what:
            if self.wrong():
                args += ["--wrong", self.wrong()]
        self.wizard.command(args)


class GroupsPage(QWidget):
    def __init__(self, wizard, parent=None):
        super().__init__(parent)
        self.wizard = wizard
        self.cards: list = []
        intro = QLabel("<h3>2 · Groups</h3>Samples that look alike are shown together. "
                       "For each group, click any tile that does not belong, then say "
                       "what the rest show. Only the tiles you saw are labelled; the rest "
                       "of each group is checked in step 4. Select tiles and use Move "
                       "to put them anywhere; Show all lists a whole group.")
        intro.setWordWrap(True)
        self.progress = QLabel()
        self.k = QSpinBox()
        self.k.setRange(0, 24)
        self.k.setSpecialValueText("auto")
        self.k.setToolTip("How many groups to make (auto: from the number of samples)")
        self.regroup = QPushButton("Group again")
        self.regroup.setToolTip("Regroups every sample not in a group you already "
                                "named; named groups are kept.")
        self.regroup.clicked.connect(self._regroup)
        top = QHBoxLayout()
        top.addWidget(self.progress, 1)
        top.addWidget(QLabel("Groups:"))
        top.addWidget(self.k)
        top.addWidget(self.regroup)
        self.body = QVBoxLayout()
        self.body.addStretch(1)
        holder = QWidget()
        holder.setLayout(self.body)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(holder)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(top)
        layout.addWidget(self.scroll, 1)
        self.buttons = [self.regroup]

    def _regroup(self):
        args = ["group"]
        if self.k.value():
            args += ["--k", str(self.k.value())]
        self.wizard.command(args)

    def show_project(self, project, next_step: dict):
        from modules.teach import group as group_mod
        position = self.scroll.verticalScrollBar().value()
        for card in self.cards:
            self.body.removeWidget(card)
            card.hide()
            card.deleteLater()
        self.cards = []
        found = group_mod.summary(project)
        if not found["total"]:
            self.progress.setText("No groups yet: cut the videos in step 1 (that groups "
                                  "them too), or press Group again.")
        else:
            decided = found["total"] - found["open"]
            colour = GREEN if not found["open"] else AMBER
            self.progress.setText(
                f"<b style='color:{colour}'>{decided} of {found['total']} groups named</b>"
                f" · {found['named']} as a class, {found['nothing']} nothing interesting, "
                f"{found['mixed']} mixed")
        # Open groups first: they are what this page is for.
        order = sorted(found["groups"], key=lambda g: (g["state"] != "open", g["id"]))
        for info in order:
            card = GroupCard(self.wizard, project, info)
            self.body.insertWidget(self.body.count() - 1, card)
            self.cards.append(card)
        self.regroup.setEnabled(bool(project.samples))
        self.scroll.verticalScrollBar().setValue(position)


class BrowsePage(QWidget):
    """Every sample, folder by folder: a group, a class, nothing, unusable or not
    decided. Select some and move them anywhere."""

    def __init__(self, wizard, parent=None):
        super().__init__(parent)
        self.wizard = wizard
        self.key = ("pending",)
        self.limit = PAGE
        intro = QLabel("<h3>3 · Browse</h3>Everything, like folders. Pick one on the "
                       "left, select samples (Shift-click for a range), and move them "
                       "where they belong. Double-click plays one.")
        intro.setWordWrap(True)
        self.folders = QListWidget()
        self.folders.setFixedWidth(230)
        self.folders.currentItemChanged.connect(self._picked)
        self.header = QLabel()
        self.count = QLabel()
        self.count.setStyleSheet("color:#999;")
        self.all_btn = QPushButton("Select all")
        self.all_btn.clicked.connect(lambda: self.tiles.select_all(True))
        self.none_btn = QPushButton("Clear")
        self.none_btn.clicked.connect(lambda: self.tiles.select_all(False))
        self.move_btn = QPushButton("Move selected to…")
        self.move_btn.clicked.connect(self._move)
        self.copy_btn = QPushButton("Copy paths")
        self.copy_btn.setToolTip("The selected samples' files, one per line")
        self.copy_btn.clicked.connect(self._copy_paths)
        self.reveal_btn = QPushButton("Show in folder")
        self.reveal_btn.setToolTip("Open the folder holding the first selected sample")
        self.reveal_btn.clicked.connect(self._reveal)
        self.more_btn = QPushButton("Show more")
        self.more_btn.clicked.connect(self._more)
        bar = QHBoxLayout()
        bar.addWidget(self.header, 1)
        bar.addWidget(self.count)
        for b in (self.all_btn, self.none_btn, self.copy_btn, self.reveal_btn,
                  self.move_btn):
            bar.addWidget(b)
        self.holder = QVBoxLayout()
        body = QWidget()
        body.setLayout(self.holder)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(body)
        right = QVBoxLayout()
        right.addLayout(bar)
        right.addWidget(self.scroll, 1)
        right.addWidget(self.more_btn)
        middle = QHBoxLayout()
        middle.addWidget(self.folders)
        middle.addLayout(right, 1)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(middle, 1)
        self.tiles = TileGrid(wizard.thumbnails)
        self.holder.addWidget(self.tiles)
        self.holder.addStretch(1)
        self.buttons = [self.move_btn, self.all_btn, self.none_btn, self.more_btn,
                        self.copy_btn, self.reveal_btn]
        self._showing = False

    # The folders and what is in each.

    @staticmethod
    def folder_list(project) -> list:
        from modules.teach.project import ACCEPTED, NEGATIVE, PENDING, REJECTED
        readable = [s for s in project.samples if not s.unreadable]
        out = [(("pending",), "Not decided",
                [s for s in readable if s.verdict == PENDING])]
        for name in project.class_names():
            out.append((("class", name), name,
                        [s for s in readable if s.verdict == ACCEPTED and s.label == name]))
        out.append((("none",), "Nothing interesting",
                    [s for s in readable if s.verdict == NEGATIVE]))
        out.append((("rejected",), "Unusable",
                    [s for s in readable if s.verdict == REJECTED]))
        by_id = {s.id: s for s in readable}
        for g in project.groups:
            if g.state == "split":
                continue
            label = f"Group {g.id}" + (f" · {g.name}" if g.name else
                                       f" · {g.state}" if g.state != "open" else "")
            out.append((("group", g.id), label,
                        [by_id[i] for i in g.members if i in by_id]))
        return out

    def show_project(self, project, next_step: dict):
        self._showing = True
        folders = self.folder_list(project)
        self.folders.clear()
        current = 0
        for i, (key, label, samples) in enumerate(folders):
            item = QListWidgetItem(f"{label}   ({len(samples)})")
            item.setData(Qt.UserRole, key)
            if key[0] == "class":
                item.setForeground(QColor(GREEN))
            elif key[0] == "group":
                item.setForeground(QColor("#999"))
            self.folders.addItem(item)
            if key == self.key:
                current = i
        self._folders = {key: (label, samples) for key, label, samples in folders}
        self.folders.setCurrentRow(current)
        self._showing = False
        self.key = folders[current][0]
        self._fill(keep_scroll=True)

    def show_folder(self, key):
        self.key, self.limit = key, PAGE
        for i in range(self.folders.count()):
            if self.folders.item(i).data(Qt.UserRole) == key:
                self.folders.setCurrentRow(i)
                return

    def _picked(self, item, _previous=None):
        if self._showing or item is None:
            return
        key = item.data(Qt.UserRole)
        if key != self.key:
            self.key, self.limit = key, PAGE
        self._fill()

    def _more(self):
        self.limit += PAGE
        self._fill(keep_scroll=True)

    def _fill(self, keep_scroll: bool = False):
        position = self.scroll.verticalScrollBar().value()
        self.holder.removeWidget(self.tiles)
        self.tiles.hide()               # gone now, not when Qt gets round to it
        self.tiles.deleteLater()
        self.tiles = TileGrid(self.wizard.thumbnails, columns=TILES_PER_ROW)
        self.tiles.changed.connect(self._count)
        self.holder.insertWidget(0, self.tiles)
        label, samples = self._folders.get(self.key, ("", []))
        for n, sample in enumerate(samples[:self.limit], 1):
            text, colour = describe_sample(sample)
            self.tiles.add(Thumb(n, sample, mark=BLUE, flag=colour, tip=text))
        self.tiles.load()
        self.header.setText(f"<b>{label}</b>")
        self.more_btn.setVisible(len(samples) > self.limit)
        self.more_btn.setText(f"Show more ({len(samples) - self.limit} not shown)")
        self._count()
        if keep_scroll:
            self.scroll.verticalScrollBar().setValue(position)
        else:
            self.scroll.verticalScrollBar().setValue(0)

    def _count(self):
        label, samples = self._folders.get(self.key, ("", []))
        n = len(self.tiles.selected())
        shown = min(len(samples), self.limit)
        self.count.setText(f"{n} of {shown} selected" if n else
                           f"{len(samples)} samples")
        self.move_btn.setEnabled(n > 0 and not self.wizard.runner.busy)
        self.copy_btn.setEnabled(n > 0)
        self.reveal_btn.setEnabled(n > 0)
        for b in (self.all_btn, self.none_btn, self.more_btn):
            b.setEnabled(not self.wizard.runner.busy)

    def selected_paths(self) -> list:
        return [os.path.abspath(t.sample.path) for t in self.tiles.selected()]

    def _copy_paths(self):
        from PySide6.QtWidgets import QApplication
        paths = self.selected_paths()
        QApplication.clipboard().setText("\n".join(paths))
        self.wizard.message.setText(f"Copied {_count(len(paths), 'path')}.")

    def _reveal(self):
        paths = self.selected_paths()
        if paths:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(paths[0])))

    def _move(self):
        ids = [t.sample.id for t in self.tiles.selected()]
        here = self.key[1] if self.key[0] == "group" else None
        move_menu(self.wizard, ids, self, here_group=here).exec(
            self.move_btn.mapToGlobal(self.move_btn.rect().bottomLeft()))


class CheckPage(QWidget):
    def __init__(self, wizard, parent=None):
        super().__init__(parent)
        self.wizard = wizard
        intro = QLabel("<h3>4 · Check guesses</h3>What you named becomes the examples, and "
                       "every other sample gets a guess. Check them a batch at a time: "
                       "each tile already shows its guess, so click only the wrong ones "
                       "and press Enter. After a class has 5 checked samples, its "
                       "confident guesses are accepted for you, and a few of those come "
                       "back as spot checks.")
        intro.setWordWrap(True)
        self.table = _table(["Class", "Checked by you", "Accepted automatically",
                             "Waiting with this guess", "Spot checks owed"])
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.sort_btn = QPushButton("Sort with what is named")
        self.sort_btn.setToolTip("Guess a class for every sample not decided yet")
        self.sort_btn.clicked.connect(lambda: self.wizard.command(["sort"]))
        self.review_btn = QPushButton("Check guesses…")
        self.review_btn.clicked.connect(self.wizard.open_review)
        self.auto_btn = QPushButton("Run the automatic steps")
        self.auto_btn.setToolTip("Everything that needs nobody, up to the next point "
                                 "where someone has to look")
        self.auto_btn.clicked.connect(lambda: self.wizard.command(["auto"]))
        buttons = QHBoxLayout()
        buttons.addWidget(self.sort_btn)
        buttons.addWidget(self.auto_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.review_btn)

        # Rules: a detector of your own counts objects in each sample, and
        # rules say which combination is which class (modules/teach/evidence).
        rules_box = QFrame()
        rules_box.setFrameShape(QFrame.StyledPanel)
        rules_title = QLabel("<b>Rules from objects found</b> (optional) — for classes "
                             "that look alike but differ in how many of something "
                             "there are, or what touches what.")
        rules_title.setWordWrap(True)
        self.detector = QLineEdit()
        self.detector.setPlaceholderText("your detector model (.onnx or .xml)")
        self.detector.editingFinished.connect(
            lambda: self._set("detector_model", self.detector.text().strip()))
        self.det_classes = QLineEdit()
        self.det_classes.setPlaceholderText("its class names: a labels file, or a, b, c")
        self.det_classes.editingFinished.connect(
            lambda: self._set("detector_classes", self.det_classes.text().strip()))
        pick_model = QPushButton("Choose…")
        pick_model.clicked.connect(lambda: self._pick(self.detector, "detector_model",
                                                      "Models (*.onnx *.xml)"))
        pick_classes = QPushButton("Choose…")
        pick_classes.clicked.connect(lambda: self._pick(
            self.det_classes, "detector_classes", "Labels (*.json *.txt);;All files (*)"))
        self.detect_btn = QPushButton("Find objects")
        self.detect_btn.setToolTip("Run the detector over every sample not seen yet "
                                   "(a few minutes for several hundred)")
        self.detect_btn.clicked.connect(lambda: self.wizard.command(["detect"]))
        self.rules_btn = QPushButton("Edit rules…")
        self.rules_btn.setToolTip("Opens this dataset's rules.yaml (made from a template "
                                  "the first time). Sort again after saving it.")
        self.rules_btn.clicked.connect(self._edit_rules)
        self.rules_note = QLabel()
        self.rules_note.setWordWrap(True)
        self.rules_note.setStyleSheet("color:#999;")
        form = QGridLayout()
        form.addWidget(QLabel("Detector"), 0, 0)
        form.addWidget(self.detector, 0, 1)
        form.addWidget(pick_model, 0, 2)
        form.addWidget(QLabel("Classes"), 1, 0)
        form.addWidget(self.det_classes, 1, 1)
        form.addWidget(pick_classes, 1, 2)
        rules_row = QHBoxLayout()
        rules_row.addWidget(self.detect_btn)
        rules_row.addWidget(self.rules_btn)
        rules_row.addWidget(self.rules_note, 1)
        box = QVBoxLayout(rules_box)
        box.addWidget(rules_title)
        box.addLayout(form)
        box.addLayout(rules_row)

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.note)
        layout.addLayout(buttons)
        layout.addWidget(rules_box)
        self.buttons = [self.sort_btn, self.review_btn, self.auto_btn, self.detect_btn,
                        self.rules_btn]

    def _set(self, key: str, value: str):
        project = self.wizard.project
        if project is None or getattr(project.settings, key) == value:
            return
        self.wizard.command(["set", f"{key}={value}"])

    def _pick(self, edit: QLineEdit, key: str, pattern: str):
        path, _ = QFileDialog.getOpenFileName(self, "Choose", edit.text(), pattern)
        if path:
            edit.setText(path)
            self._set(key, path)

    def _edit_rules(self):
        from modules.teach import evidence
        path = evidence.init_rules(self.wizard.project)
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        self.wizard.message.setText(f"Rules: {path} — save it, then Sort again.")

    def show_rules(self, project):
        from modules.teach import evidence
        self.detector.setText(project.settings.detector_model)
        self.det_classes.setText(project.settings.detector_classes)
        has_detector = bool(project.settings.detector_model)
        seen = len(evidence.load(project)["samples"]) if has_detector else 0
        waiting = len(evidence.todo(project))
        bits = []
        if has_detector:
            bits.append(f"objects found in {seen} samples"
                        + (f", {waiting} not looked at yet" if waiting else ""))
        rule_guesses = sum(1 for s in project.samples if s.rule_guess)
        if os.path.exists(evidence.rules_path(project)):
            bits.append(f"rules guessed {rule_guesses}")
            record = evidence.rule_record(project)
            for cls, row in sorted(record.items()):
                trust = " — decides alone" if evidence.trusted(project, cls, record) else ""
                bits.append(f"{cls}: right {row['right']} of {row['checked']} checked{trust}")
        self.rules_note.setText(" · ".join(bits) or "No detector set.")
        self.detect_btn.setEnabled(has_detector and waiting > 0)
        self.rules_btn.setEnabled(True)

    def show_project(self, project, next_step: dict, report: dict):
        from modules.teach.project import NONE, PENDING
        rows = []
        for name, r in report["classes"].items():
            label = "none of these" if name == NONE else name
            owed = r.get("spot_checks_owed") or 0
            rows.append([label, r["by_person"], r["by_auto"],
                         r.get("waiting", "–") if name != NONE else "–",
                         _item(owed or "–", RED if owed else "")])
        _fill(self.table, rows)
        pending = sum(1 for s in project.samples if s.verdict == PENDING
                      and not s.unreadable)
        unsorted = sum(1 for s in project.samples if not s.scores and not s.unreadable)
        bits = [f"{pending} samples not decided yet."]
        if unsorted and project.classes:
            bits.append(f"{unsorted} have no guess yet: press Sort.")
        bits.append("Next: " + next_step.get("why", ""))
        self.note.setText(" ".join(bits))
        has_classes = bool(project.classes)
        self.show_rules(project)
        self.auto_btn.setEnabled(True)
        self.sort_btn.setEnabled(has_classes)
        self.review_btn.setEnabled(has_classes and pending > 0)


class HealthPage(QWidget):
    def __init__(self, wizard, parent=None):
        super().__init__(parent)
        self.wizard = wizard
        title = QLabel("<h3>5 · Health</h3>")
        self.banner = QLabel()
        self.banner.setWordWrap(True)
        self.checks = _table(["", "Check", "Class", "What it found", "What to do"])
        sorting_title = QLabel("<b>How good was the automatic sorting?</b> Of the samples "
                               "you looked at, how often the guess you were shown was "
                               "right.")
        sorting_title.setWordWrap(True)
        self.overall = QLabel()
        self.overall.setWordWrap(True)
        self.sources = _table(["Guess came from", "You looked at", "Right", "Wrong",
                               "Unusable", "No guess", "Accuracy"])
        self.per_class = _table(["Guessed as", "Times", "Right", "Accuracy", "Was really"])
        self.refresh_btn = QPushButton("Check again")
        self.refresh_btn.clicked.connect(self.wizard.reload)
        self.train_btn = QPushButton("Train a model")
        self.train_btn.setToolTip("Builds the dataset and trains (minutes to hours). The "
                                  "model is installed only if it beats the one before.")
        self.train_btn.clicked.connect(self.train)
        self.folder_btn = QPushButton("Show dataset folder")
        self.folder_btn.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(self.wizard.root)))
        buttons = QHBoxLayout()
        buttons.addWidget(self.refresh_btn)
        buttons.addWidget(self.folder_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.train_btn)

        body = QVBoxLayout()
        body.addWidget(title)
        body.addWidget(self.banner)
        body.addWidget(self.checks)
        body.addWidget(sorting_title)
        body.addWidget(self.overall)
        body.addWidget(self.sources)
        body.addWidget(self.per_class)
        holder = QWidget()
        holder.setLayout(body)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(holder)
        layout = QVBoxLayout(self)
        layout.addWidget(scroll, 1)
        layout.addLayout(buttons)
        self.buttons = [self.train_btn]
        self.verdict = ""

    def show_project(self, project, next_step: dict, report: dict):
        from modules.teach.project import NONE
        self.verdict = report["verdict"]
        text, colour = HEALTH_TEXT[self.verdict]
        self.banner.setText(f"<span style='font-size:15px; color:{colour}'><b>{text}.</b>"
                            f"</span> {report['says']}")
        rows = []
        order = {"bad": 0, "warn": 1, "ok": 2}
        for c in sorted(report["checks"], key=lambda c: order[c["level"]]):
            rows.append([_item("●", LEVEL_COLOURS[c["level"]]), c["name"],
                         c.get("class", ""), c["detail"], c.get("fix", "")])
        _fill(self.checks, rows)
        self.checks.setMinimumHeight(min(60 + 34 * len(rows), 420))

        s = report["sorting"]
        if s["accuracy"] is None and not s["groups"]["shown"]:
            self.overall.setText("Nothing to measure yet: name groups and check guesses "
                                 "first.")
        else:
            src = s["by_source"]
            right = sum(v["right"] for v in src.values())
            wrong = sum(v["wrong"] for v in src.values())
            total = right + wrong
            self.overall.setText(
                f"The machine's guess was right <b>{right} of {total}</b> times "
                f"({_percent(right / total if total else None)}). Each wrong guess you "
                "fixed was recorded; the fixes are what make the next sort better.")
        _fill(self.sources, [
            [SOURCE_TEXT.get(k, k), v["shown"], v["right"], v["wrong"], v["rejected"],
             v["unsure"], _item(_percent(v["accuracy"]), self._rate_colour(v["accuracy"]))]
            for k, v in sorted(s["by_source"].items())])
        self.sources.setVisible(bool(s["by_source"]))
        self.sources.setMinimumHeight(min(60 + 34 * len(s["by_source"]), 200))
        rows = []
        for name, v in sorted(s["by_class"].items()):
            really = ", ".join(f"{('none of these' if k == NONE else k)} ×{n}"
                               for k, n in v["really"].items())
            rows.append([("none of these" if name == NONE else name), v["guessed"],
                         v["right"], _item(_percent(v["accuracy"]),
                                           self._rate_colour(v["accuracy"])),
                         really or "–"])
        _fill(self.per_class, rows)
        self.per_class.setVisible(bool(rows))
        self.per_class.setMinimumHeight(min(60 + 34 * len(rows), 300))
        self.train_btn.setEnabled(self.verdict != "not_ready")

    @staticmethod
    def _rate_colour(value) -> str:
        if value is None:
            return ""
        return GREEN if value >= 0.9 else AMBER if value >= 0.7 else RED

    def train(self):
        if self.verdict == "needs_videos":
            answer = QMessageBox.question(
                self, "Train a model",
                "Some classes come from fewer than 3 videos. It will train, but the "
                "model will likely mistake those videos' scenes for the thing, and its "
                "score cannot be trusted. Train anyway?")
            if answer != QMessageBox.Yes:
                return
        self.wizard.command(["auto", "--train"])


class Wizard(QWidget):
    back = Signal()

    def __init__(self, runner: Runner, thumbnails: Thumbnails, parent=None):
        super().__init__(parent)
        self.runner, self.thumbnails = runner, thumbnails
        self.root = ""
        self.project = None
        self._review = None
        back = QPushButton("← All datasets")
        back.clicked.connect(self.back)
        self.title = QLabel()
        self.next = QLabel()
        self.next.setWordWrap(True)
        self.next.setStyleSheet("color:#999;")
        self.where = QLabel()
        self.where.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.where.setStyleSheet("color:#999;")
        self.where.setToolTip("Where this dataset is kept (select to copy)")
        folder = QPushButton("Open folder")
        folder.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(self.root)) if self.root else None)
        heading = QVBoxLayout()
        heading.addWidget(self.title)
        heading.addWidget(self.where)
        top = QHBoxLayout()
        top.addWidget(back)
        top.addLayout(heading, 1)
        top.addWidget(folder)

        self.steps = QListWidget()
        self.steps.setFixedWidth(170)
        for i, name in enumerate(STEPS, 1):
            self.steps.addItem(QListWidgetItem(f"{i}  {name}"))
        self.pages = QStackedWidget()
        self.videos = VideosPage(self)
        self.groups = GroupsPage(self)
        self.browser = BrowsePage(self)
        self.check = CheckPage(self)
        self.health = HealthPage(self)
        for page in self.all_pages():
            self.pages.addWidget(page)
        self.steps.currentRowChanged.connect(self.pages.setCurrentIndex)
        middle = QHBoxLayout()
        middle.addWidget(self.steps)
        middle.addWidget(self.pages, 1)

        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # While a command runs: what it is doing, and how far (when it says).
        self.doing = ""
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        runner.started.connect(self._started)
        runner.progress.connect(self.show_progress)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.next)
        layout.addLayout(middle, 1)
        layout.addWidget(self.bar)
        layout.addWidget(self.message)

    def all_pages(self) -> tuple:
        return (self.videos, self.groups, self.browser, self.check, self.health)

    # --- showing ---------------------------------------------------------------

    def browse(self, key):
        """Show one folder (a group, a class...) in Browse."""
        self.browser.show_folder(key)
        self.steps.setCurrentRow(BROWSE)

    def open(self, root: str):
        self.root = root
        self.message.clear()
        self.reload(jump=True)

    def reload(self, jump: bool = False):
        from modules.teach import health
        from modules.teach.project import Project
        from modules.teach.status import next_step
        if not self.root:
            return
        try:
            self.project = Project.load(self.root)
        except (OSError, ValueError) as exc:
            self.message.setText(f"Could not open it: {exc}")
            return
        project = self.project
        step = next_step(project)
        report = health.report(project)
        self.title.setText(f"<h2 style='margin:0'>{project.name}</h2>"
                           f"<span style='color:#999'>{project.task} · "
                           f"{_count(len(project.sources), 'video')} · "
                           f"{_count(len(project.samples), 'sample')}</span>")
        self.where.setText(project.root)
        self.next.setText("Next: " + step.get("why", ""))
        self.videos.show_project(project, step)
        self.groups.show_project(project, step)
        self.browser.show_project(project, step)
        self.check.show_project(project, step, report)
        self.health.show_project(project, step, report)
        self._mark_steps(project, report)
        if jump:
            # Here footage comes first, whatever `status` would start with.
            self.steps.setCurrentRow(STEP_OF.get((step.get("args") or [""])[0], VIDEOS)
                                     if project.sources else VIDEOS)
        self.set_busy(self.runner.busy)

    def _mark_steps(self, project, report):
        groups = report["groups"]
        done = [
            bool(project.sources) and all(s.cut for s in project.sources),
            bool(groups["total"]) and not groups["open"],
            bool(project.samples) and all(s.verdict != "pending" or s.unreadable
                                          for s in project.samples),
            bool(project.classes) and not any(
                c.get("spot_checks_owed") for c in report["classes"].values())
            and any(c["by_person"] for c in report["classes"].values()),
            report["verdict"] == "ready",
        ]
        for i, ok in enumerate(done):
            item = self.steps.item(i)
            item.setText(f"{'✓' if ok else ' '}  {i + 1}  {STEPS[i]}")
            item.setForeground(QColor(GREEN) if ok else QColor())

    # --- doing -----------------------------------------------------------------

    def command(self, args: list):
        if not self.root:
            return
        if not self.runner.run(["--project", self.root, *args]):
            self.message.setText("Still working on the last step; try again in a moment.")

    def finished(self, argv: list, code: int, result: dict):
        command = argv[2] if len(argv) > 2 else ""
        if result.get("error") or result.get("errors"):
            errors = result.get("errors") or [result.get("error")]
            self.message.setText("<span style='color:%s'>%s</span>"
                                 % (RED, "<br>".join(str(e) for e in errors)))
        elif command == "name-group":
            self.message.setText(self._named(result))
        elif command == "move":
            self.message.setText(self._moved(result))
        else:
            self.message.setText(describe(result).replace("\n", "<br>"))
        jump = command in ("auto", "add-video", "group")
        self.reload(jump=jump)

    @staticmethod
    def _moved(result: dict) -> str:
        n = _count(result.get("moved", 0), "sample")
        if "group" in result:
            return f"Moved {n} to group {result['group']}."
        where = {"accepted": repr(result.get("label", "")), "negative": "nothing interesting",
                 "rejected": "unusable", "pending": "not decided"}.get(
            result.get("verdict"), result.get("verdict", ""))
        return f"Moved {n} to {where}."

    @staticmethod
    def _named(result: dict) -> str:
        if result.get("state") == "named":
            text = f"Group {result['group']} is {result['name']!r}: {result['decided']} labelled"
        elif result.get("state") == "nothing":
            text = f"Group {result['group']}: {result['decided']} marked as nothing"
        elif result.get("state") == "mixed":
            text = f"Group {result['group']} is mixed"
        else:
            text = f"Group {result['group']} reopened"
        if result.get("left_for_review"):
            text += f", {result['left_for_review']} left to check in step 4"
        if result.get("background_class"):
            text += (f". 'Nothing interesting' is trained as "
                     f"{result['background_class']!r}")
        return text + "."

    def set_busy(self, busy: bool):
        """Busy: nothing can be pressed. Not busy: ``reload`` sets each button."""
        if busy and self.bar.isHidden():
            for page in self.all_pages():
                for b in page.buttons:
                    b.setEnabled(False)
            self.message.setText(f"{self.doing or 'Working'}…")
        self.bar.setVisible(busy)
        for card in self.groups.cards:
            card.setEnabled(not busy)
        self.browser.tiles.setEnabled(not busy)

    def _started(self, argv: list):
        command = argv[2] if argv[:1] == ["--project"] and len(argv) > 2 else argv[0]
        self.doing = DOING.get(command, "Working")
        self.bar.setRange(0, 0)                 # busy, no count yet

    def show_progress(self, line: str):
        """One line the running command printed: its step, and a count if any."""
        if line.startswith("auto: "):
            self.doing = DOING.get(line.split()[1] if len(line.split()) > 1 else "",
                                   self.doing)
            self.bar.setRange(0, 0)
            line = ""
        counted = COUNTED.search(line)
        if counted:
            done, total = int(counted.group(1)), max(int(counted.group(2)), 1)
            self.bar.setRange(0, total)
            self.bar.setValue(min(done, total))
            self.bar.setFormat(f"{done} of {total}")
        text = f"{self.doing}…"
        if line:
            text += ("<br><span style='color:#999'>%s</span>"
                     % html.escape(line if len(line) <= 160 else line[:159] + "…"))
        self.message.setText(text)

    def open_review(self):
        from modules.teach.review_window import BoxReviewWindow, ReviewWindow
        from modules.teach.teach_panel import TeachPanel
        if self._review is not None:
            self._review.raise_()
            return
        boxes = TeachPanel.wants_box_review(self.root)
        window = BoxReviewWindow(self.root) if boxes else ReviewWindow(self.root)
        window.setAttribute(Qt.WA_DeleteOnClose)
        window.destroyed.connect(self._review_closed)
        self._review = window
        window.show()

    def _review_closed(self, *_):
        self._review = None
        self.reload()


class DatasetsPanel(QWidget):
    """The list of datasets, and the steps of the one that is open."""

    def __init__(self, parent=None, run_cli: Optional[Callable] = None):
        super().__init__(parent)
        self.runner = Runner(self, run_cli=run_cli)
        self.thumbnails = Thumbnails(self)
        self.list = ListPage(self.runner)
        self.wizard = Wizard(self.runner, self.thumbnails)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.list)
        self.stack.addWidget(self.wizard)
        self.list.open_project.connect(self.open)
        self.wizard.back.connect(self.show_list)
        self.runner.started.connect(lambda *_: self._busy(True))
        self.runner.finished.connect(self._finished)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.stack)
        self.list.refresh()

    def open(self, root: str):
        self.stack.setCurrentWidget(self.wizard)
        self.wizard.open(root)

    def show_list(self):
        self.stack.setCurrentWidget(self.list)
        self.list.refresh()

    def _busy(self, busy: bool):
        for b in (self.list.new_btn, self.list.open_btn, self.list.delete_btn,
                  self.list.refresh_btn):
            b.setEnabled(not busy)
        if self.stack.currentWidget() is self.wizard:
            self.wizard.set_busy(busy)

    def _finished(self, argv: list, code: int, result: dict):
        self._busy(False)
        command = argv[2] if argv[:1] == ["--project"] and len(argv) > 2 else argv[0]
        if command == "projects":
            self.list.show_projects(result)
        elif command == "init":
            if result.get("error"):
                QMessageBox.warning(self, "New dataset", result["error"])
                self.list.refresh()
            else:
                self.open(result["created"])
        elif command == "delete":
            if result.get("error"):
                QMessageBox.warning(self, "Delete dataset", result["error"])
            self.list.refresh()
        elif self.stack.currentWidget() is self.wizard:
            self.wizard.finished(argv, code, result)

    def closeEvent(self, event):        # noqa: N802 (Qt override)
        self.thumbnails.stop()
        self.runner.wait()
        super().closeEvent(event)
