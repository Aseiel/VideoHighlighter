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

import os
from typing import Callable, Optional

from PySide6.QtCore import QObject, QThread, QUrl, Qt, Signal
from PySide6.QtGui import QColor, QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
    QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QScrollArea,
    QSpinBox, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from modules.teach.teach_panel import _Job, describe

GREEN, AMBER, RED, GREY = "#2e9d57", "#d4a017", "#c0392b", "#7f8c8d"
LEVEL_COLOURS = {"ok": GREEN, "warn": AMBER, "bad": RED}
HEALTH_TEXT = {"ready": ("Healthy", GREEN), "needs_videos": ("Needs more videos", AMBER),
               "not_ready": ("Not ready", RED)}
STATE_TEXT = {"open": ("not named yet", AMBER), "named": ("named", GREEN),
              "nothing": ("nothing interesting", GREY), "mixed": ("mixed", AMBER)}
SOURCE_TEXT = {"group": "Groups (tiles you were shown)", "sort": "Sorting (review guesses)",
               "auto": "Auto-accept (spot checks)"}
THUMB_WIDTH = 168
TILES_PER_ROW = 6

STEPS = ("Videos", "Groups", "Check guesses", "Health")
VIDEOS, GROUPS, CHECK, HEALTH = range(4)
# Which page a `status` next step belongs on.
STEP_OF = {"add-video": VIDEOS, "cut": VIDEOS, "focus": VIDEOS, "group": GROUPS,
           "name-group": GROUPS, "add-class": GROUPS, "sort": CHECK, "review": CHECK,
           "boxes": CHECK, "find": CHECK, "add-example": CHECK, "build": HEALTH,
           "train": HEALTH}


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

class Runner(QObject):
    """One ``modules.teach`` command at a time, off the UI thread."""

    started = Signal(list)
    finished = Signal(list, int, dict)

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
        self._job = _Job(lambda: self._run_cli(self._argv))
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
                              "which is what action models are trained and used on.")
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
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh)
        buttons = QHBoxLayout()
        for b in (self.new_btn, self.open_btn, self.folder_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)
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
        self.prepare.setToolTip("Cuts every new video into 5 s samples (crops them to "
                                "people if the dataset is set to), then groups them.")
        self.prepare.clicked.connect(lambda: self.wizard.command(["auto"]))
        buttons = QHBoxLayout()
        for b in (self.add_files, self.add_folder, self.add_url):
            buttons.addWidget(b)
        buttons.addStretch(1)
        buttons.addWidget(self.prepare)
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self.list, 1)
        layout.addLayout(buttons)
        self.buttons = [self.add_files, self.add_folder, self.add_url, self.prepare]

    def show_project(self, project, next_step: dict):
        self.list.clear()
        per_source = {}
        for s in project.samples:
            per_source[s.source] = per_source.get(s.source, 0) + 1
        for source in project.sources:
            name = os.path.basename(source.path) or source.url
            state = (f"{per_source.get(source.id, 0)} samples" if source.cut
                     else "not cut yet")
            length = f" · {_clock(source.duration)}" if source.duration else ""
            self.list.addItem(f"{source.id}   {name}{length}   ·   {state}")
        if not project.sources:
            self.list.addItem("No videos yet: add one below.")
        waiting = (next_step.get("args") or [""])[0] in ("cut", "focus", "group")
        self.prepare.setEnabled(waiting)
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
    """One sample in a group: click marks it as not belonging, double-click plays."""

    toggled = Signal()

    def __init__(self, n: int, sample, editable: bool, wrong: bool, parent=None):
        super().__init__(parent)
        self.n, self.sample, self.editable, self.wrong = n, sample, editable, wrong
        self.setFixedSize(THUMB_WIDTH + 8, THUMB_WIDTH * 9 // 16 + 8)
        self.setAlignment(Qt.AlignCenter)
        self.setToolTip(f"{n}: {sample.source} at {_clock(sample.start)}.  "
                        + ("Click: does not belong.  " if editable else "")
                        + "Double-click: play.")
        self.picture: Optional[QPixmap] = None
        self.refresh()

    def set_picture(self, pixmap: QPixmap):
        self.picture = pixmap
        self.refresh()

    def refresh(self):
        colour = RED if self.wrong else ("#444" if self.editable else GREEN)
        self.setStyleSheet(f"QLabel {{ border: 4px solid {colour}; background:#111; "
                           "color:#ddd; }}")
        if self.picture is not None:
            self.setPixmap(self.picture)
        else:
            self.setText(str(self.n))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.editable:
            self.wrong = not self.wrong
            self.refresh()
            self.toggled.emit()

    def mouseDoubleClickEvent(self, event):
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self.sample.path)))


class GroupCard(QFrame):
    def __init__(self, wizard, project, info: dict, parent=None):
        super().__init__(parent)
        from modules.teach import group as group_mod

        self.wizard, self.id, self.state = wizard, info["id"], info["state"]
        spec = project.get_group(self.id)
        self.setFrameShape(QFrame.StyledPanel)
        editable = self.state == "open"
        wrong = set(spec.wrong)
        self.thumbs = []
        grid = QGridLayout()
        grid.setSpacing(4)
        for n, sample in enumerate(group_mod.tiles(project, self.id), 1):
            thumb = Thumb(n, sample, editable, sample.id in wrong)
            pixmap = wizard.thumbnails.get(sample.id)
            if pixmap is not None:
                thumb.set_picture(pixmap)
            thumb.toggled.connect(self._count)
            grid.addWidget(thumb, (n - 1) // TILES_PER_ROW, (n - 1) % TILES_PER_ROW)
            self.thumbs.append(thumb)
        wizard.thumbnails.want([(t.sample.id, t.sample.path) for t in self.thumbs])

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
            self._count()
        else:
            if self.state == "mixed":
                split = QPushButton("Split into smaller groups")
                split.clicked.connect(
                    lambda: self.wizard.command(["group", "--split", str(self.id)]))
                row.addWidget(split)
                self.hint.setText("Its samples are checked one by one in step 3 unless "
                                  "you split it.")
            elif wrong:
                self.hint.setText(f"{len(wrong)} marked as not belonging: they are "
                                  "checked in step 3.")
            change = QPushButton("Change")
            change.clicked.connect(lambda: self.say(["--reopen"]))
            row.addWidget(change)
        row.addWidget(self.hint, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addLayout(grid)
        layout.addLayout(row)

    def _count(self):
        n = sum(t.wrong for t in self.thumbs)
        self.hint.setText(f"{n} marked as not belonging" if n
                          else "Click any tile that does not belong.")

    def wrong(self) -> str:
        return ",".join(str(t.n) for t in self.thumbs if t.wrong)

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
                       "of each group is checked in step 3.")
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
        wizard.thumbnails.ready.connect(self._thumb)

    def _regroup(self):
        args = ["group"]
        if self.k.value():
            args += ["--k", str(self.k.value())]
        self.wizard.command(args)

    def _thumb(self, sample_id: str):
        pixmap = self.wizard.thumbnails.get(sample_id)
        for card in self.cards:
            for thumb in card.thumbs:
                if thumb.sample.id == sample_id:
                    thumb.set_picture(pixmap)

    def show_project(self, project, next_step: dict):
        from modules.teach import group as group_mod
        position = self.scroll.verticalScrollBar().value()
        for card in self.cards:
            self.body.removeWidget(card)
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


class CheckPage(QWidget):
    def __init__(self, wizard, parent=None):
        super().__init__(parent)
        self.wizard = wizard
        intro = QLabel("<h3>3 · Check guesses</h3>What you named becomes the examples, and "
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
        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addWidget(self.table, 1)
        layout.addWidget(self.note)
        layout.addLayout(buttons)
        self.buttons = [self.sort_btn, self.review_btn, self.auto_btn]

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
        self.auto_btn.setEnabled(True)
        self.sort_btn.setEnabled(has_classes)
        self.review_btn.setEnabled(has_classes and pending > 0)


class HealthPage(QWidget):
    def __init__(self, wizard, parent=None):
        super().__init__(parent)
        self.wizard = wizard
        title = QLabel("<h3>4 · Health</h3>")
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
        top = QHBoxLayout()
        top.addWidget(back)
        top.addWidget(self.title, 1)

        self.steps = QListWidget()
        self.steps.setFixedWidth(170)
        for i, name in enumerate(STEPS, 1):
            self.steps.addItem(QListWidgetItem(f"{i}  {name}"))
        self.pages = QStackedWidget()
        self.videos = VideosPage(self)
        self.groups = GroupsPage(self)
        self.check = CheckPage(self)
        self.health = HealthPage(self)
        for page in (self.videos, self.groups, self.check, self.health):
            self.pages.addWidget(page)
        self.steps.currentRowChanged.connect(self.pages.setCurrentIndex)
        middle = QHBoxLayout()
        middle.addWidget(self.steps)
        middle.addWidget(self.pages, 1)

        self.message = QLabel()
        self.message.setWordWrap(True)
        self.message.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self.next)
        layout.addLayout(middle, 1)
        layout.addWidget(self.message)

    # --- showing ---------------------------------------------------------------

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
        self.next.setText("Next: " + step.get("why", ""))
        self.videos.show_project(project, step)
        self.groups.show_project(project, step)
        self.check.show_project(project, step, report)
        self.health.show_project(project, step, report)
        self._mark_steps(project, report)
        if jump:
            # Here footage comes first, whatever `status` would start with.
            self.steps.setCurrentRow(STEP_OF.get((step.get("args") or [""])[0], VIDEOS)
                                     if project.sources else VIDEOS)
        if self.runner.busy:
            self.set_busy(True)

    def _mark_steps(self, project, report):
        groups = report["groups"]
        done = [
            bool(project.sources) and all(s.cut for s in project.sources),
            bool(groups["total"]) and not groups["open"],
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
        else:
            self.message.setText(describe(result).replace("\n", "<br>"))
        jump = command in ("auto", "add-video", "group")
        self.reload(jump=jump)

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
            text += f", {result['left_for_review']} left to check in step 3"
        if result.get("background_class"):
            text += (f". 'Nothing interesting' is trained as "
                     f"{result['background_class']!r}")
        return text + "."

    def set_busy(self, busy: bool):
        """Busy: nothing can be pressed. Not busy: ``reload`` sets each button."""
        if busy:
            for page in (self.videos, self.groups, self.check, self.health):
                for b in page.buttons:
                    b.setEnabled(False)
            self.message.setText("Working…")
        for card in self.groups.cards:
            card.setEnabled(not busy)

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
        for b in (self.list.new_btn, self.list.open_btn, self.list.refresh_btn):
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
        elif self.stack.currentWidget() is self.wizard:
            self.wizard.finished(argv, code, result)

    def closeEvent(self, event):        # noqa: N802 (Qt override)
        self.thumbnails.stop()
        self.runner.wait()
        super().closeEvent(event)
