"""Teach from videos, inside the app: the ``modules.teach`` loop behind buttons.

The command line (``python -m modules.teach``) is the whole feature. This is
the same thing for someone who never opens a terminal. Every button runs
exactly the CLI command it names (``cli.run``), so the two can't disagree,
and the panel always shows what ``status`` says comes next.

    [task] [project] [examples folder] [videos]
    Check this computer | Start | Check guesses | Continue | Train
    <what happened, and what comes next>
"""
from __future__ import annotations

import os
from typing import Callable, Optional

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)


class _Job(QObject):
    done = Signal(object)

    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn())
        except BaseException as exc:          # shown, never raised into Qt, and
            # never lost: a SystemExit here would end the thread with the
            # buttons still disabled.
            self.done.emit((2, {"error": f"{type(exc).__name__}: {exc}"}))


def describe(result: dict) -> str:
    """A command's JSON result as a few lines for a person."""
    if not isinstance(result, dict):
        return str(result)
    if result.get("error"):
        return f"Stopped: {result['error']}"
    lines = []
    if "checks" in result:
        for c in result["checks"]:
            mark = "OK " if c["ok"] else ("MISSING " if c["level"] == "required" else "note ")
            line = f"{mark}{c['name']}: {c['detail']}"
            if not c["ok"] and c.get("fix"):
                line += f"  -> {c['fix']}"
            lines.append(line)
        lines.append("Ready." if result.get("ready") else "Not ready yet: fix the MISSING items.")
        return "\n".join(lines)
    for step in result.get("ran") or []:
        lines.append("done: " + " ".join(step.get("args") or []))
    if "classes" in result and isinstance(result["classes"], dict):
        for name, c in result["classes"].items():
            lines.append(f"{name}: {c['accepted']} of {c['target']} accepted")
    nxt = result.get("next") or result.get("stopped_at") or {}
    if nxt.get("why"):
        lines.append("Next: " + nxt["why"])
    if result.get("message"):
        lines.append(result["message"])
    return "\n".join(lines) or "Done."


class TeachPanel(QWidget):
    """The loop for people who never open a terminal."""

    def __init__(self, parent=None, run_cli: Optional[Callable] = None,
                 open_review: Optional[Callable] = None,
                 host_busy: Optional[Callable[[], bool]] = None):
        super().__init__(parent)
        from modules.teach import cli
        self._run_cli = run_cli or cli.run
        self._open_review = open_review
        self._host_busy = host_busy or (lambda: False)
        self._thread = None
        self._job = None
        self._review = None

        intro = QLabel(
            "Teach it something new from your own videos. Put a few short example "
            "clips in one folder per thing, named after what it shows, and choose "
            "the videos to learn from. It cuts, sorts and labels by itself, and asks "
            "you only about the samples it is unsure of.")
        intro.setWordWrap(True)

        self.task = QComboBox()
        self.task.addItems(["actions", "objects"])
        self.task.setToolTip("actions: something that happens over time (a movement)\n"
                             "objects: a thing visible in one frame")
        self.project = QLineEdit("my-first")
        self.project.setToolTip("A name, or a folder. Projects live in your user data.")
        self.examples = QLineEdit()
        self.examples.setPlaceholderText("folder with one subfolder of clips per thing")
        self.videos = QLineEdit()
        self.videos.setPlaceholderText("a folder of videos (or type the path of one video)")
        self.focus = QCheckBox("Crop samples to the people in them (actions)")

        form = QFormLayout()
        form.addRow("What kind", self.task)
        form.addRow("Project", self.project)
        form.addRow("Examples", self._with_browse(self.examples, folder=True))
        form.addRow("Videos", self._with_browse(self.videos, folder=True))
        form.addRow("", self.focus)

        self.doctor_btn = QPushButton("Check this computer")
        self.doctor_btn.clicked.connect(lambda: self._run(["doctor"]))
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start)
        self.review_btn = QPushButton("Check guesses…")
        self.review_btn.clicked.connect(self.review)
        self.continue_btn = QPushButton("Continue")
        self.continue_btn.setToolTip("Run every step that needs nobody")
        self.continue_btn.clicked.connect(lambda: self._run(["auto"]))
        self.train_btn = QPushButton("Train")
        self.train_btn.setToolTip("Continue, including training (can take a while)")
        self.train_btn.clicked.connect(lambda: self._run(["auto", "--train"]))
        self.share_btn = QPushButton("Share…")
        self.share_btn.setToolTip("Share the trained detector on the model hub "
                                  "(the model only, never your footage)")
        self.share_btn.clicked.connect(self.share)
        buttons = QHBoxLayout()
        for b in (self.doctor_btn, self.start_btn, self.review_btn, self.continue_btn,
                  self.train_btn, self.share_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)

        from PySide6.QtWidgets import QApplication

        from modules.teach import background
        self.background_box = QCheckBox(
            "Keep improving in the background while the app is idle")
        self.background_box.setToolTip(
            "Runs every step that needs nobody (finding, sorting, training) when you "
            "have not touched the app for a couple of minutes, and keeps a few "
            "questions for you. Your footage never leaves this computer.")
        self.background_status = QLabel()
        self.background_status.setWordWrap(True)
        app = QApplication.instance()
        self.idle = background.IdleWatch(app, self) if app is not None else None
        self.background = background.BackgroundTeacher(self._background_root,
                                                       self._free, parent=self)
        self.background.started.connect(lambda: self._set_busy(True))
        self.background.report.connect(self._background_report)
        self.background_box.toggled.connect(self._set_background)
        self.project.editingFinished.connect(self._show_background_setting)
        self._last_background = ""
        self._waiting: dict = {}
        self.background.set_enabled(True)       # acts only on opted-in projects
        self._show_background_setting()

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("What happened, and what comes next, shows here.")

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self.background_box)
        layout.addWidget(self.background_status)
        layout.addWidget(self.output, 1)

    def _with_browse(self, edit: QLineEdit, folder: bool) -> QWidget:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(edit, 1)
        btn = QPushButton("Choose…")

        def pick():
            path = (QFileDialog.getExistingDirectory(self, "Choose a folder") if folder
                    else QFileDialog.getOpenFileName(self, "Choose a video")[0])
            if path:
                edit.setText(path)
        btn.clicked.connect(pick)
        row.addWidget(btn)
        holder = QWidget()
        holder.setLayout(row)
        return holder

    # --- actions ----------------------------------------------------------------

    def project_arg(self) -> str:
        return self.project.text().strip() or "my-first"

    def start(self):
        args = ["quick", "--task", self.task.currentText()]
        if self.examples.text().strip():
            args += ["--examples", self.examples.text().strip()]
        if self.videos.text().strip():
            args += ["--videos", self.videos.text().strip()]
        if self.focus.isChecked():
            args.append("--focus")
        self._run(args)

    def review(self):
        from modules.teach.cli import resolve_root
        root = resolve_root(self.project_arg())
        if self._waiting and root not in self._waiting:
            # Questions from something taught in the player: those first.
            root = next(iter(self._waiting))
            self.project.setText(root)
            self._show_background_setting()
        if not os.path.exists(os.path.join(root, "project.json")):
            self.output.setPlainText("Start a project first.")
            return
        if self._open_review is not None:
            self._open_review(root)
            return
        from PySide6.QtCore import Qt

        from modules.teach.review_window import BoxReviewWindow, ReviewWindow
        window = BoxReviewWindow(root) if self.wants_box_review(root) else ReviewWindow(root)
        # Deleted on close, so ``destroyed`` fires and the panel says what is next.
        window.setAttribute(Qt.WA_DeleteOnClose)
        window.destroyed.connect(self._review_closed)
        self._review = window
        window.show()

    def _review_closed(self, *_):
        self._review = None
        self._run(["status"])

    @staticmethod
    def wants_box_review(root: str) -> bool:
        """Boxes when that is what the project is waiting on; samples otherwise."""
        from modules.teach.project import Project
        from modules.teach.status import next_step
        return next_step(Project.load(root)).get("args", [])[:2] == ["boxes", "review"]

    def share(self):
        from modules.teach.cli import resolve_root
        from modules.teach.project import Project
        from modules.teach.share import NotShareable, share_draft

        root = resolve_root(self.project_arg())
        try:
            onnx, draft = share_draft(Project.load(root))
        except FileNotFoundError:
            self.output.setPlainText("Start a project first.")
            return
        except NotShareable as exc:
            self.output.setPlainText(str(exc))
            return
        from model_hub.gui import PublishWizard
        PublishWizard(self, model_path=onnx, draft=draft).exec()

    # --- the background --------------------------------------------------------

    def _panel_root(self) -> str:
        from modules.teach.cli import resolve_root
        root = resolve_root(self.project_arg())
        return root if os.path.exists(os.path.join(root, "project.json")) else ""

    def _roots(self) -> list:
        """Projects set to improve in the background: those under the user
        data (teaching from the player puts them there), and the panel's own."""
        from modules.teach.background import opted_in
        from modules.teach.project import Project
        roots = opted_in()
        mine = self._panel_root()
        if mine and mine not in roots and Project.load(mine).settings.background:
            roots.append(mine)
        return roots

    def _background_root(self) -> str:
        from modules.teach.background import pick
        root = pick(self._roots(), self._last_background)
        self._last_background = root or self._last_background
        return root

    def _show_background_setting(self):
        from modules.teach.project import Project
        root = self._panel_root()
        on = bool(root) and Project.load(root).settings.background
        self.background_box.blockSignals(True)
        self.background_box.setChecked(on)
        self.background_box.blockSignals(False)

    def _set_background(self, on: bool):
        from modules.teach.project import Project
        root = self._panel_root()
        if not root:
            self.output.setPlainText("Start a project first.")
            self._show_background_setting()
            return
        project = Project.load(root)
        project.settings.background = bool(on)
        project.save()

    def _free(self) -> bool:
        """Nobody is using the app, and nothing else is writing the project."""
        from modules.teach.background import IDLE_SECONDS
        idle = self.idle.idle_seconds() if self.idle is not None else 0.0
        return (idle >= IDLE_SECONDS and self._thread is None and self._review is None
                and not self._host_busy())

    def _background_report(self, report: dict):
        from modules.teach.background import summary
        self._set_busy(False)
        self.background_status.setText(summary(report))
        from modules.teach.background import waiting
        self._waiting = waiting(self._roots())
        n = sum(self._waiting.values())
        self.review_btn.setText(f"Check guesses… ({n})" if n else "Check guesses…")

    def _run(self, args: list):
        if self.background.running:
            self.output.setPlainText("Improving in the background; this finishes its "
                                     "current step first. Try again in a moment.")
            return
        self._set_busy(True)
        self.output.setPlainText("Working… (" + " ".join(args) + ")")
        argv = ["--project", self.project_arg(), *args]
        self._thread = QThread(self)
        self._job = _Job(lambda: self._run_cli(argv))
        self._job.moveToThread(self._thread)
        self._thread.started.connect(self._job.run)
        self._job.done.connect(self._finished)
        self._thread.finished.connect(self._job.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    def _finished(self, outcome):
        self._thread.quit()
        self._thread.wait()
        self._thread = self._job = None
        self._set_busy(False)
        code, result = outcome if isinstance(outcome, tuple) else (0, outcome)
        self.last_result = result
        self.output.setPlainText(describe(result))

    def _set_busy(self, busy: bool):
        for b in (self.doctor_btn, self.start_btn, self.review_btn, self.continue_btn,
                  self.train_btn, self.share_btn):
            b.setEnabled(not busy)
