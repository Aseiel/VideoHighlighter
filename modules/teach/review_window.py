"""Review by clicking: the contact sheet as a window.

The same batch and verdicts as ``review`` (``next_sheet`` / ``apply_verdicts``),
without typing tile numbers. Each tile starts at the answer the guess implies,
so for a batch of good guesses the whole job is one key:

* a tile guessed as a class starts **accepted** (green);
* a tile guessed "none of these" starts **none** (grey);
* an unsure tile starts **undecided** (yellow) and stays in the queue unless
  it is given an answer.

Click a tile to cycle accept -> reject -> none -> undecided. Right-click to
say which class it really is. Double-click plays the clip. Enter saves and
loads the next batch, and by default re-sorts first, so what was just
accepted sharpens the guesses on the next one.

    python -m modules.teach --project <name> review --window
    python -m modules.teach --project <name> boxes review --window

The box window is the same with one frame per tile and the proposed box drawn
on it: accept, reject or leave undecided.

Qt lives only here; everything it does goes through the modules the command
line uses, so the two can never disagree about what a verdict means.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

from PySide6.QtCore import QObject, QThread, QUrl, Qt, Signal
from PySide6.QtGui import QDesktopServices, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QGridLayout, QHBoxLayout, QLabel, QMenu, QMessageBox,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from modules.teach import review
from modules.teach.project import NONE, Project

ACCEPT, REJECT, NEGATIVE, UNDECIDED = "accept", "reject", "none", "undecided"
CYCLE = (ACCEPT, REJECT, NEGATIVE, UNDECIDED)
COLOURS = {ACCEPT: "#2e9d57", REJECT: "#c0392b", NEGATIVE: "#7f8c8d",
           UNDECIDED: "#d4a017"}
MARKS = {ACCEPT: "✓", REJECT: "✗", NEGATIVE: "∅", UNDECIDED: "?"}


def _pixmap(image, max_width: int) -> QPixmap:
    rgb = image.convert("RGB")
    data = rgb.tobytes("raw", "RGB")
    qimage = QImage(data, rgb.width, rgb.height, rgb.width * 3, QImage.Format_RGB888)
    pixmap = QPixmap.fromImage(qimage.copy())
    if pixmap.width() > max_width:
        pixmap = pixmap.scaledToWidth(max_width, Qt.SmoothTransformation)
    return pixmap


def initial_state(item: dict, class_names) -> tuple:
    """``(state, label)`` a tile starts in, from what it was guessed as."""
    guess = item["proposed"]
    if guess in class_names:
        return ACCEPT, guess
    if guess == NONE:
        return NEGATIVE, ""
    return UNDECIDED, ""


def verdict_args(states: dict, guesses: dict) -> dict:
    """Tile states -> ``review.apply_verdicts`` keyword arguments.

    ``states`` maps tile number to ``(state, label)``; ``guesses`` to what the
    tile was guessed as. A tile accepted as its guess is an accept; accepted
    as anything else, a relabel; undecided tiles are left out.
    """
    accept, reject, negative, relabel = [], [], [], []
    for n, (state, label) in sorted(states.items()):
        if state == ACCEPT and label == guesses.get(n):
            accept.append(str(n))
        elif state == ACCEPT and label:
            relabel.append(f"{n}={label}")
        elif state == REJECT:
            reject.append(str(n))
        elif state == NEGATIVE:
            negative.append(str(n))
    return {"accept": ",".join(accept), "reject": ",".join(reject),
            "negative": ",".join(negative), "relabel": relabel}


class Tile(QLabel):
    changed = Signal()

    def __init__(self, n: int, item: dict, image, sample_path: str, class_names,
                 max_width: int = 560, choices=CYCLE):
        super().__init__()
        self.n, self.item, self.path = n, item, sample_path
        self.class_names = list(class_names)
        self.choices = tuple(choices)
        self.state, self.label = initial_state(item, self.class_names)
        self.picture = _pixmap(image, max_width)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAlignment(Qt.AlignCenter)
        if NEGATIVE in self.choices:
            self.setToolTip("Click: accept / reject / none / undecided.  Right-click: "
                            "it is another class.  Double-click: play.")
        else:
            self.setToolTip("Click: accept / reject / undecided.  Double-click: play.")
        self.refresh()

    def refresh(self):
        colour = COLOURS[self.state]
        answer = {ACCEPT: self.label, REJECT: "reject", NEGATIVE: "none of these",
                  UNDECIDED: "undecided"}[self.state]
        self.setStyleSheet(f"QLabel {{ border: 5px solid {colour}; background: #111; }}"
                           f"QLabel:focus {{ border: 5px solid #ffffff; }}")
        self.setPixmap(self.picture)
        self.caption = f"{self.n}  {MARKS[self.state]} {answer}   ({self.item['caption']})"
        self.changed.emit()

    def set_state(self, state: str, label: str = ""):
        self.state = state
        if state == ACCEPT:
            guess = self.item["proposed"]
            self.label = label or (guess if guess in self.class_names else self.label)
            if not self.label:                      # nothing to accept it as
                self.state = UNDECIDED
        else:
            self.label = ""
        self.refresh()

    def cycle(self):
        order = list(self.choices)
        at = order.index(self.state) if self.state in order else -1
        self.set_state(order[(at + 1) % len(order)])

    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() == Qt.LeftButton:
            self.cycle()
        elif event.button() == Qt.RightButton and NEGATIVE in self.choices:
            menu = QMenu(self)
            for name in self.class_names:
                menu.addAction(f"it is: {name}", lambda n=name: self.set_state(ACCEPT, n))
            menu.addSeparator()
            menu.addAction("none of these", lambda: self.set_state(NEGATIVE))
            menu.addAction("reject (unclear / bad cut)", lambda: self.set_state(REJECT))
            menu.exec(event.globalPosition().toPoint())

    def mouseDoubleClickEvent(self, event):
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(self.path)))

    def keyPressEvent(self, event):
        keys = {Qt.Key_A: ACCEPT, Qt.Key_R: REJECT, Qt.Key_N: NEGATIVE,
                Qt.Key_U: UNDECIDED}
        if event.key() in keys and keys[event.key()] in self.choices:
            self.set_state(keys[event.key()])
        elif event.key() == Qt.Key_Space:
            self.cycle()
        else:
            super().keyPressEvent(event)


class _SortWorker(QObject):
    done = Signal(object)

    def __init__(self, root: str, make_embedder: Callable):
        super().__init__()
        self.root, self.make_embedder = root, make_embedder

    def run(self):
        from modules.teach.sort import sort_project
        try:
            project = Project.load(self.root)
            self.done.emit(sort_project(project, self.make_embedder()))
        except Exception as exc:                 # shown, never raised into Qt
            self.done.emit({"error": f"{type(exc).__name__}: {exc}"})


class ReviewWindow(QWidget):
    """One batch at a time; Enter saves it and moves on."""

    def __init__(self, root: str, size: int = 24, class_name: Optional[str] = None,
                 frame_reader: Optional[Callable] = None,
                 make_embedder: Optional[Callable] = None):
        super().__init__()
        self.root, self.size, self.class_name = root, size, class_name
        self.frame_reader = frame_reader
        self.make_embedder = make_embedder
        self.tiles: list = []
        self.holders: list = []
        self.record: dict = {}
        self._thread = None
        self.setWindowTitle("Check the guesses")
        self.resize(1300, 900)

        self.header = QLabel()
        self.header.setWordWrap(True)
        self.progress = QLabel()
        self.grid = QGridLayout()
        body = QWidget()
        body.setLayout(self.grid)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)

        self.resort = QCheckBox("Re-sort with what I just checked before the next batch")
        self.resort.setChecked(make_embedder is not None or frame_reader is None)
        self.save_next = QPushButton("Save and next batch  (Enter)")
        self.save_next.clicked.connect(lambda: self.save(and_next=True))
        self.save_close = QPushButton("Save and close")
        self.save_close.clicked.connect(lambda: self.save(and_next=False))
        all_ok = QPushButton("All shown are right")
        all_ok.clicked.connect(self.accept_all)

        buttons = QHBoxLayout()
        buttons.addWidget(all_ok)
        buttons.addWidget(self.resort)
        buttons.addStretch(1)
        buttons.addWidget(self.save_close)
        buttons.addWidget(self.save_next)

        layout = QVBoxLayout(self)
        layout.addWidget(self.header)
        layout.addWidget(scroll, 1)
        layout.addWidget(self.progress)
        layout.addLayout(buttons)
        for key in (Qt.Key_Return, Qt.Key_Enter):
            QShortcut(QKeySequence(key), self, activated=lambda: self.save(and_next=True))
        self.load_batch()

    # --- batches --------------------------------------------------------------

    def load_batch(self):
        for holder in self.holders:
            self.grid.removeWidget(holder)
            holder.deleteLater()
        self.tiles, self.holders = [], []
        project = Project.load(self.root)
        self.record = self.fetch(project)
        if not self.record:
            self.header.setText("<b>Nothing left to check.</b> Close this window and run "
                                "<code>status</code> for the next step.")
            self.save_next.setEnabled(False)
            self.update_progress(project)
            return
        self.header.setText(self.intro())
        columns = self.columns(project)
        for i, (item, image) in enumerate(zip(self.record["items"], self.record["tiles"])):
            tile = self.make_tile(project, item, image,
                                  max_width=560 if columns == 2 else 280)
            caption = QLabel()
            caption.setWordWrap(True)
            tile.caption_label = caption
            tile.changed.connect(lambda t=tile: t.caption_label.setText(t.caption))
            tile.changed.emit()
            cell = QVBoxLayout()
            cell.addWidget(tile)
            cell.addWidget(caption)
            holder = QWidget()
            holder.setLayout(cell)
            self.grid.addWidget(holder, i // columns, i % columns)
            self.tiles.append(tile)
            self.holders.append(holder)
        if self.tiles:
            self.tiles[0].setFocus()
        self.update_progress(project)

    # What a batch is, and how its tiles behave: the box window overrides these.

    def fetch(self, project: Project) -> dict:
        return review.next_sheet(project, size=self.size, class_name=self.class_name,
                                 frame_reader=self.frame_reader, keep_tiles=True)

    def intro(self) -> str:
        return (f"<b>Batch {self.record['sheet']}</b> — each tile already shows its "
                "guess. Click the wrong ones: click cycles accept / reject / none of "
                "these / undecided; right-click to say which class it really is; "
                "double-click to play. Then press Enter.")

    def columns(self, project: Project) -> int:
        return 2 if project.task == "actions" else 4

    def make_tile(self, project: Project, item: dict, image, max_width: int) -> Tile:
        paths = {s.id: s.path for s in project.samples}
        return Tile(item["n"], item, image, paths.get(item["sample"], ""),
                    project.class_names(), max_width=max_width)

    def apply(self, project: Project) -> dict:
        guesses = {t.n: t.item["proposed"] for t in self.tiles}
        return review.apply_verdicts(project, self.record["sheet"],
                                     **verdict_args(self.states(), guesses), by="window")

    def update_progress(self, project: Project):
        counts = project.counts()
        self.progress.setText("   ".join(
            f"<b>{name}</b>: {c['accepted']} of {c['target']}" for name, c in counts.items()))

    def accept_all(self):
        for tile in self.tiles:
            if tile.state == UNDECIDED and tile.item["proposed"] not in tile.class_names:
                continue
            tile.set_state(ACCEPT if tile.item["proposed"] in tile.class_names else NEGATIVE)

    def states(self) -> dict:
        return {t.n: (t.state, t.label) for t in self.tiles}

    def save(self, and_next: bool = True) -> dict:
        if not self.record:
            if not and_next:
                self.close()
            return {}
        result = self.apply(Project.load(self.root))
        if result.get("errors"):
            QMessageBox.warning(self, "Not saved", "\n".join(result["errors"]))
            return result
        if not and_next:
            self.close()
            return result
        if self.resort.isChecked():
            self.resort_then_load()
        else:
            self.load_batch()
        return result

    def resort_then_load(self):
        from modules.teach.cli import make_embedder
        self.save_next.setEnabled(False)
        self.header.setText("<b>Re-sorting with what you just checked…</b>")
        self._thread = QThread(self)
        self._worker = _SortWorker(self.root, self.make_embedder or make_embedder)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._sorted)
        self._thread.start()

    def _sorted(self, result):
        self._thread.quit()
        self._thread.wait()
        self.save_next.setEnabled(True)
        if isinstance(result, dict) and result.get("error"):
            QMessageBox.warning(self, "Re-sort failed", result["error"])
        self.load_batch()


class BoxReviewWindow(ReviewWindow):
    """The proposed boxes, one frame per tile: is the box around the thing?

    Same window, same keys; a tile is accepted, rejected or left undecided
    (there is no "none of these" or relabel for a box). Verdicts go through
    ``boxes.apply_verdicts``, as ``boxes verdict`` does.
    """

    def __init__(self, root: str, size: int = 24, read_at: Optional[Callable] = None):
        self.read_at = read_at
        super().__init__(root, size, make_embedder=None, frame_reader=read_at)
        self.setWindowTitle("Check the boxes")
        self.resort.setChecked(False)
        self.resort.hide()

    def fetch(self, project: Project) -> dict:
        from modules.teach import boxes
        record = boxes.next_sheet(project, size=self.size, read_at=self.read_at,
                                  renderer=lambda *a, **k: None, keep_tiles=True)
        for item in record.get("items") or []:
            item["proposed"] = item["class_name"]
        return record

    def intro(self) -> str:
        return (f"<b>Boxes {self.record['sheet']}</b> — is the yellow box around the "
                "named thing, and tight? Click the wrong ones to reject them (click "
                "cycles accept / reject / undecided). Then press Enter.")

    def columns(self, project: Project) -> int:
        return 4

    def make_tile(self, project: Project, item: dict, image, max_width: int) -> Tile:
        return Tile(item["n"], item, image, item["video"], [item["class_name"]],
                    max_width=max_width, choices=(ACCEPT, REJECT, UNDECIDED))

    def apply(self, project: Project) -> dict:
        from modules.teach import boxes
        states = self.states()
        pick = lambda wanted: ",".join(str(n) for n, (state, _) in sorted(states.items())  # noqa: E731
                                       if state == wanted)
        return boxes.apply_verdicts(project, self.record["sheet"],
                                    accept=pick(ACCEPT), reject=pick(REJECT))

    def update_progress(self, project: Project):
        from modules.teach import boxes
        labels = boxes.store(project)
        self.progress.setText(f"<b>{len(labels.accepted())}</b> boxes accepted, "
                              f"<b>{len(labels.pending())}</b> waiting")


def open_window(root: str, size: int = 24, class_name: Optional[str] = None,
                boxes: bool = False) -> int:
    app = QApplication.instance() or QApplication([])
    window = BoxReviewWindow(root, size) if boxes else ReviewWindow(root, size, class_name)
    window.show()
    return app.exec()
