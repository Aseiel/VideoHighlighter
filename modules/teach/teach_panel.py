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
        except Exception as exc:              # shown, never raised into Qt
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
                 open_review: Optional[Callable] = None):
        super().__init__(parent)
        from modules.teach import cli
        self._run_cli = run_cli or cli.run
        self._open_review = open_review
        self._thread = None
        self._job = None

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
        self.videos.setPlaceholderText("a video, or a folder of videos")
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
        buttons = QHBoxLayout()
        for b in (self.doctor_btn, self.start_btn, self.review_btn, self.continue_btn,
                  self.train_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("What happened, and what comes next, shows here.")

        layout = QVBoxLayout(self)
        layout.addWidget(intro)
        layout.addLayout(form)
        layout.addLayout(buttons)
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
        if not os.path.exists(os.path.join(root, "project.json")):
            self.output.setPlainText("Start a project first.")
            return
        if self._open_review is not None:
            self._open_review(root)
            return
        from modules.teach.review_window import ReviewWindow
        self._review = ReviewWindow(root)
        self._review.destroyed.connect(lambda *_: self._run(["status"]))
        self._review.show()

    def _run(self, args: list):
        self._set_busy(True)
        self.output.setPlainText("Working… (" + " ".join(args) + ")")
        argv = ["--project", self.project_arg(), *args]
        self._thread = QThread(self)
        self._job = _Job(lambda: self._run_cli(argv))
        self._job.moveToThread(self._thread)
        self._thread.started.connect(self._job.run)
        self._job.done.connect(self._finished)
        self._thread.start()

    def _finished(self, outcome):
        self._thread.quit()
        self._thread.wait()
        self._set_busy(False)
        code, result = outcome if isinstance(outcome, tuple) else (0, outcome)
        self.last_result = result
        self.output.setPlainText(describe(result))

    def _set_busy(self, busy: bool):
        for b in (self.doctor_btn, self.start_btn, self.review_btn, self.continue_btn,
                  self.train_btn):
            b.setEnabled(not busy)
