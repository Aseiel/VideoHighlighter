"""Add a reviewed video to a hand-sorted dataset, and find out if it helped.

One window for the loop ``intake`` and ``compare`` make possible:

    Preview   what would be added, class by class, and why the rest is not
    Add       link those crops into the dataset (recorded, so it can be undone)
    Check     train with and without them and score both on the dataset's
              other videos: better, worse, or no clear change
    Take out  the parts that measured worse, or everything

Every button runs one ``modules.teach`` command on a worker thread, as the
rest of the Datasets tab does, so the window and the command line cannot
disagree. Nothing here decides anything: the tables show what the commands
returned.
"""
from __future__ import annotations

import os

import re
from types import SimpleNamespace

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QScrollArea, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from modules.teach.dataset_panel import (
    AMBER, GREEN, GREY, RED, Runner, Thumb, Thumbnails, TileGrid, _fill, _item, _table,
)

# "<video>__<milliseconds>..." is how the sorter names a crop.
_WHEN = re.compile(r"__(\d+)")

LEVEL = {"ok": GREEN, "warn": AMBER, "bad": RED}
VERDICT = {"better": GREEN, "worse": RED, "no clear change": GREY}
SEEDS = 5


def _points(value: float) -> str:
    return f"{value * 100:+.1f}"


def _sample(path: str, video: str):
    """What a ``Thumb`` needs to show a crop: where it is and when it starts."""
    match = _WHEN.search(os.path.basename(path))
    return SimpleNamespace(id=path, path=path, source=video,
                           start=int(match.group(1)) / 1000 if match else 0.0)


class _FolderRow(QWidget):
    def __init__(self, title: str, parent=None, files: str = ""):
        super().__init__(parent)
        self.title, self.files = title, files
        self.edit = QLineEdit()
        button = QPushButton("Browse…")
        button.clicked.connect(self.browse)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.edit, 1)
        row.addWidget(button)

    def browse(self):
        if self.files:
            path, _ = QFileDialog.getOpenFileName(self, self.title, self.edit.text(),
                                                  self.files)
        else:
            path = QFileDialog.getExistingDirectory(self, self.title, self.edit.text())
        if path:
            self.edit.setText(os.path.normpath(path))

    def text(self) -> str:
        return self.edit.text().strip()


class ReviewedVideoDialog(QDialog):
    """See the module's text. ``run_cli`` is injectable for tests."""

    def __init__(self, parent=None, run_cli=None):
        super().__init__(parent)
        self.setWindowTitle("Add a reviewed video to a dataset")
        self.resize(900, 680)
        self.runner = Runner(self, run_cli)
        self.runner.started.connect(self._started)
        self.runner.progress.connect(self._progress)
        self.runner.finished.connect(self._finished)
        self.drop_folders: list = []

        intro = QLabel(
            "A video you sorted and checked is the best new material a dataset can get. "
            "All of it at once makes the model worse on every other video, so only "
            "classes the dataset is thin on are taken, and a limited number of each. "
            "<b>Check it helped</b> then measures the change on the dataset's other "
            "videos; anything that measured worse can be taken out again.")
        intro.setWordWrap(True)

        self.dataset = _FolderRow("The dataset (the folder holding train, val and test)")
        self.review = _FolderRow("The reviewed video's crops (one folder per class)")
        self.aliases = _FolderRow("A file renaming folders (optional)", files="JSON (*.json)")
        self.aliases.edit.setPlaceholderText(
            'optional: a JSON file of folder name -> class name ("" leaves it out)')
        self.video = QLineEdit()
        self.video.setPlaceholderText("a short name for this video, e.g. holiday-2")
        self.thin = QSpinBox()
        self.thin.setRange(0, 100000)
        self.thin.setValue(200)
        self.thin.setSuffix(" clips")
        self.thin.setToolTip("A class is taken only when the dataset has fewer clips of it "
                             "than this. 0 takes every class.")
        self.cap = QSpinBox()
        self.cap.setRange(0, 100000)
        self.cap.setValue(50)
        self.cap.setSuffix(" crops")
        self.cap.setToolTip("The most crops taken from one class folder, spread evenly "
                            "over the video. 0 takes them all.")
        form = QFormLayout()
        form.addRow("Dataset", self.dataset)
        form.addRow("Reviewed crops", self.review)
        form.addRow("Folder names", self.aliases)
        form.addRow("Name in the dataset", self.video)
        form.addRow("Take a class when the dataset has fewer than", self.thin)
        form.addRow("At most, per class", self.cap)

        self.health_btn = QPushButton("Dataset health")
        self.preview_btn = QPushButton("Preview")
        self.add_btn = QPushButton("Add to dataset")
        self.check_btn = QPushButton("Check it helped")
        self.check_btn.setToolTip("Trains small models with and without the added crops and "
                                  "scores both on the dataset's other videos. Minutes.")
        self.drop_btn = QPushButton("Take out what measured worse")
        self.undo_btn = QPushButton("Take it all out")
        self.health_btn.clicked.connect(self.health)
        self.preview_btn.clicked.connect(lambda: self.plan(False))
        self.add_btn.clicked.connect(lambda: self.plan(True))
        self.check_btn.clicked.connect(self.check)
        self.drop_btn.clicked.connect(self.drop)
        self.undo_btn.clicked.connect(self.undo)
        self.buttons = (self.health_btn, self.preview_btn, self.add_btn, self.check_btn,
                        self.drop_btn, self.undo_btn)
        row = QHBoxLayout()
        for button in self.buttons:
            row.addWidget(button)
        row.insertStretch(4, 1)
        self.add_btn.setEnabled(False)
        self.drop_btn.setEnabled(False)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.working = QLabel("")
        self.working.setStyleSheet(f"color: {GREY};")

        self.plan_table = _table(["Class folder", "Found", "Taken", "In the dataset", "Why"])
        self.result_table = _table(["Class", "Without", "With", "Change", "Held-out clips",
                                    "Added", ""])
        self.parts_table = _table(["Part of what was added", "Crops", "Change (points)", ""])
        self.health_table = _table(["", "Finding", "What to do", "Which"])
        result = QWidget()
        layout = QVBoxLayout(result)
        layout.setContentsMargins(0, 0, 0, 0)
        self.parts_table.setMinimumHeight(110)
        self.parts_table.setMaximumHeight(150)
        layout.addWidget(self.parts_table)
        layout.addWidget(self.result_table, 1)
        self.tabs = QTabWidget()
        self.thumbnails = Thumbnails(self)
        self.samples_title = QLabel("Click a class above to see the crops it takes.")
        self.samples_title.setWordWrap(True)
        self.samples_box = QScrollArea()
        self.samples_box.setWidgetResizable(True)
        self.tiles = None
        self.paths: dict = {}
        self.plan_table.itemSelectionChanged.connect(self.show_samples)
        samples = QWidget()
        box = QVBoxLayout(samples)
        box.setContentsMargins(0, 0, 0, 0)
        box.addWidget(self.samples_title)
        box.addWidget(self.samples_box, 1)
        split = QSplitter(Qt.Vertical)
        split.addWidget(self.plan_table)
        split.addWidget(samples)
        split.setSizes([260, 300])
        self.tabs.addTab(split, "What would be added")
        self.tabs.addTab(result, "Did it help")
        self.tabs.addTab(self.health_table, "Dataset health")

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addLayout(row)
        layout.addWidget(self.status)
        layout.addWidget(self.working)
        layout.addWidget(self.tabs, 1)

    # -- commands -----------------------------------------------------------

    def _base(self, command: str) -> list:
        argv = ["--project", "datasets", command, self.dataset.text()]
        if self.aliases.text():
            argv += ["--aliases", self.aliases.text()]
        return argv

    def _need(self, review: bool = False, video: bool = True) -> bool:
        missing = [name for name, ok in (
            ("the dataset folder", bool(self.dataset.text())),
            ("the folder of reviewed crops", not review or bool(self.review.text())),
            ("a name for the video", not video or bool(self.video.text().strip()))) if not ok]
        if missing:
            self._say("Choose " + " and ".join(missing) + " first.", AMBER)
        return not missing

    def health(self):
        if self._need(video=False):
            self.runner.run(self._base("dataset-health"))

    def plan(self, apply: bool):
        if not self._need(review=True):
            return
        argv = self._base("add-reviewed") + [
            self.review.text(), "--video", self.video.text().strip(),
            "--thin-under", str(self.thin.value()), "--cap", str(self.cap.value()),
            "--paths"]
        if not self.thin.value() and not self.cap.value():
            argv.append("--everything")
        self.runner.run(argv + (["--apply"] if apply else []))

    def check(self):
        if self._need():
            self.runner.run(self._base("compare-added") + [
                "--added", self.video.text().strip(), "--seeds", str(SEEDS),
                "--parts", "kinds"])

    def drop(self):
        if self.drop_folders and self._need():
            self.runner.run(self._base("add-reviewed") + [
                "--video", self.video.text().strip(), "--drop", *self.drop_folders])

    def undo(self):
        if self._need():
            self.runner.run(self._base("add-reviewed") + [
                "--video", self.video.text().strip(), "--undo"])

    # -- results ------------------------------------------------------------

    def _say(self, text: str, colour: str = ""):
        style = f' style="color:{colour};"' if colour else ""
        self.status.setText(f"<span{style}>{text}</span>")

    def _started(self, argv):
        for button in self.buttons:
            button.setEnabled(False)
        self.working.setText("Working…")

    def _progress(self, line: str):
        # Only what the command says about its own work: libraries it calls
        # print warnings (with their source lines) on the same stream.
        line = line.strip()
        if line and not ("warn" in line.lower() or line.startswith(("C:\\", "/", "File "))):
            self.working.setText(line[-160:])

    def _finished(self, argv, code, result):
        for button in self.buttons:
            button.setEnabled(True)
        self.add_btn.setEnabled(False)
        self.drop_btn.setEnabled(bool(self.drop_folders))
        self.working.setText("")
        if result.get("error"):
            self._say(result["error"], RED)
            return
        command = argv[2]
        if command == "dataset-health":
            self.show_health(result)
        elif command == "compare-added":
            self.show_comparison(result)
        elif "--undo" in argv:
            self.drop_folders = []
            self.drop_btn.setEnabled(False)
            self._say(f"Took out {result['removed']} crops. The dataset is as it was "
                      f"before this video.", GREEN)
        elif "--drop" in argv:
            self.drop_folders = []
            self.drop_btn.setEnabled(False)
            self._say(f"Took out {result['removed']} crops; {result['in_dataset']} of this "
                      f"video's stay. Check again to see where that leaves it.", GREEN)
        else:
            self.show_plan(result)

    def show_plan(self, result: dict):
        rows = []
        for row in result["folders"]:
            have = ", ".join(f"{n} clips from {result_v} videos" for n, result_v in zip(
                row["dataset_clips"].values(), row["dataset_videos"].values())) or "–"
            taken = _item(row["take"], GREEN if row["take"] else GREY, True)
            rows.append([row["folder"], _item(row["found"], align_right=True), taken, have,
                         row["why"] or ("new to the dataset" if not any(
                             row["dataset_clips"].values()) else "the dataset is thin on it")])
        self.paths = {row["folder"]: (row.get("paths") or [], row["found"], row["why"])
                      for row in result["folders"]}
        self.plan_rows = [row["folder"] for row in result["folders"]]
        _fill(self.plan_table, rows)
        self.tabs.setCurrentIndex(0)
        taken = [i for i, row in enumerate(result["folders"]) if row["take"]]
        if taken:
            self.plan_table.selectRow(taken[0])
        applied = result.get("applied")
        text = (f"{result['take']} of {result['found']} crops, "
                f"{result['share_of_train']:.0%} of the training clips afterwards.")
        if applied:
            text = (f"Added {applied['added']} crops ({applied['in_dataset']} of this video "
                    f"are in the dataset now). Next: <b>Check it helped</b>. " + text)
        else:
            text = "Would add " + text + " Nothing has been changed yet."
        for warning in result.get("warnings") or []:
            text += f'<br><span style="color:{AMBER};">{warning}</span>'
        self._say(text)
        self.add_btn.setEnabled(not applied and result["take"] > 0)

    def show_samples(self):
        """The crops the selected class folder gives, in the order they occur
        in the video. Double-click one to play it."""
        row = self.plan_table.currentRow()
        if not 0 <= row < len(getattr(self, "plan_rows", [])):
            return
        folder = self.plan_rows[row]
        paths, found, why = self.paths[folder]
        if self.tiles is not None:
            self.tiles.deleteLater()
        self.tiles = TileGrid(self.thumbnails)
        video = self.video.text().strip()
        for n, path in enumerate(paths, 1):
            self.tiles.add(Thumb(n, _sample(path, video), mark=GREEN,
                                 tip=f"{folder}: crop {n} of the {len(paths)} taken"))
        self.samples_box.setWidget(self.tiles)
        self.tiles.load()
        if paths:
            self.samples_title.setText(
                f"<b>{folder}</b>: the {len(paths)} of {found} crops it takes, spread over "
                f"the video. Double-click one to play it.")
        else:
            self.samples_title.setText(f"<b>{folder}</b>: none of its {found} crops are "
                                       f"taken ({why or 'nothing to take'}).")

    def done(self, code):                       # noqa: D102 - Qt override
        self.thumbnails.stop()
        self.runner.wait()
        super().done(code)

    def show_comparison(self, result: dict):
        rows = []
        for row in result["per_class"]:
            rows.append([row["class"], f"{row['recall_without']:.2f}",
                         f"{row['recall_with']:.2f}",
                         _item(_points(row["change"]), VERDICT[row["verdict"]], True),
                         _item(row["held_out_clips"], align_right=True),
                         _item(row["added_clips"], align_right=True),
                         _item("" if row["verdict"] == "no clear change" else row["verdict"],
                               VERDICT[row["verdict"]])])
        _fill(self.result_table, rows)
        _fill(self.parts_table, [
            [row["part"], _item(row["clips"], align_right=True),
             _item(_points(row["top1_change"]), VERDICT[row["verdict"]], True),
             _item("keep" if row["keep"] else "measured worse: take out",
                   GREEN if row["keep"] else RED)]
            for row in result.get("parts") or []])
        self.parts_table.setVisible(bool(result.get("parts")))
        self.tabs.setCurrentIndex(1)
        self.drop_folders = list(result.get("drop_folders") or [])
        self.drop_btn.setEnabled(bool(self.drop_folders))
        without, with_ = result["top1_without"], result["top1_with"]
        text = (f"<b>{result['summary']}</b><br>Right first guess on the dataset's other "
                f"videos: {without['mean']:.1%} without, {with_['mean']:.1%} with "
                f"({result['how']['seeds']} runs).")
        if self.drop_folders:
            text += ("<br>Measured worse on their own: " + ", ".join(self.drop_folders)
                     + ". <b>Take out what measured worse</b> removes them and keeps the rest.")
        elif not result["keep"]:
            text += "<br><b>Take it all out</b> puts the dataset back as it was."
        self._say(text, "" if result["keep"] else RED)

    def show_health(self, result: dict):
        _fill(self.health_table, [
            [_item({"warn": "check", "bad": "fix"}.get(f["level"], ""), LEVEL[f["level"]]),
             f["text"], f["fix"], ", ".join(str(i) for i in f["items"])
             + (f" … and {f['count'] - len(f['items'])} more" if f["count"] > len(f["items"])
                else "")]
            for f in result["findings"]])
        self.tabs.setCurrentIndex(2)
        wanted = result.get("wanted") or []
        text = (f"{result['clips']} clips, {result['classes']} classes, "
                f"{result['videos']} source videos.")
        if wanted:
            text += (" More videos would help most for: "
                     + ", ".join(w["class"] for w in wanted[:8])
                     + (f" and {len(wanted) - 8} more." if len(wanted) > 8 else "."))
        if not result["findings"]:
            text += " Nothing found that would skew training."
        self._say(text, LEVEL[result["level"]] if result["findings"] else GREEN)
