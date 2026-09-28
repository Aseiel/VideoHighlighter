"""Keep teaching while nobody is using the app.

Everything a project does unattended is ``auto``: cut, find, sort, the
learnt cutoff, and (with ``train``) training a round and installing it only
if it beats the last. This runs that same command when the app has been idle
for a while, one pass at a time, for every project whose ``background``
setting is on (teaching from the player turns it on), taking turns between
them. It says how many questions are waiting:
the boxes the cutoff was not sure enough about. A person answers those when
they like (a handful at a time is enough; every answer can move the cutoff
and accept the rest), or an agent does, through ``boxes review``.

It never runs while someone is working: any mouse or key input in the app
postpones it, so does a review window being open, and so does anything the
host says is busy (an analysis running). A pass that is already running
finishes its current step; training in particular is not interrupted.

The engine (``run_once``, ``questions``) is plain Python; ``BackgroundTeacher``
and ``IdleWatch`` are the Qt glue.
"""
from __future__ import annotations

import time
from typing import Callable, Optional

from modules.teach.project import OBJECTS, PENDING, Project

IDLE_SECONDS = 120          # no input for this long counts as idle
CHECK_EVERY_MS = 60_000


def questions(project: Project) -> int:
    """What is waiting for a person: proposed boxes, or unsure samples."""
    if project.task == OBJECTS:
        from modules.teach.boxes import store
        return len(store(project).pending())
    names = set(project.class_names())
    return sum(1 for s in project.samples if s.verdict == PENDING and s.proposed in names)


def opted_in(projects_root: Optional[str] = None) -> list:
    """Every project set to improve in the background, oldest first."""
    import os

    from modules.teach.project import PROJECT_FILE, projects_root as default_root

    base = projects_root or default_root()
    roots = []
    for name in sorted(os.listdir(base)) if os.path.isdir(base) else []:
        root = os.path.join(base, name)
        if not os.path.exists(os.path.join(root, PROJECT_FILE)):
            continue
        try:
            if Project.load(root).settings.background:
                roots.append(root)
        except (OSError, ValueError, KeyError, TypeError):
            continue                 # a damaged project is not ours to touch
    return roots


def pick(roots, last: str = "") -> str:
    """The next project with unattended work, taking turns; "" if none has any."""
    from modules.teach.status import next_step

    roots = list(dict.fromkeys(roots))
    if last in roots:                # start after the one that ran last
        at = roots.index(last) + 1
        roots = roots[at:] + roots[:at]
    for root in roots:
        try:
            step = next_step(Project.load(root))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if step.get("who") == "auto" and step.get("args"):
            return root
    return ""


def waiting(roots) -> dict:
    """``{root: questions}`` for the projects that have any."""
    out = {}
    for root in dict.fromkeys(roots):
        try:
            n = questions(Project.load(root))
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if n:
            out[root] = n
    return out


def run_once(root: str, *, train: bool = True, run_cli: Optional[Callable] = None) -> dict:
    """One ``auto`` pass; what it did, what waits for a person, what is next."""
    from modules.teach import cli
    from modules.teach.status import next_step

    run_cli = run_cli or cli.run
    code, result = run_cli(["--project", root, "auto", *(["--train"] if train else [])])
    project = Project.load(root)
    step = next_step(project)
    return {"root": root, "code": code,
            "ran": [" ".join(r["args"]) for r in (result or {}).get("ran") or []],
            "error": (result or {}).get("error", ""), "questions": questions(project),
            "next": step}


def summary(report: dict) -> str:
    """One line for the panel."""
    if report.get("error"):
        return f"Background: stopped ({report['error']})"
    import os

    ran = ", ".join(report.get("ran") or []) or "nothing new"
    if report.get("root"):
        ran = f"{os.path.basename(report['root'])}: {ran}"
    n = report.get("questions", 0)
    ask = f"; {n} question{'s' if n != 1 else ''} waiting" if n else ""
    return f"Background: {ran}{ask}. Next: {report.get('next', {}).get('why', '')}"


try:
    from PySide6.QtCore import QEvent, QObject, QThread, QTimer, Signal
except ImportError:              # the engine above works without Qt
    QObject = None

if QObject is not None:

    class IdleWatch(QObject):
        """Seconds since the last mouse or key input anywhere in the app."""

        INPUT = {QEvent.MouseButtonPress, QEvent.MouseMove, QEvent.KeyPress,
                 QEvent.Wheel, QEvent.TouchBegin}

        def __init__(self, app, parent=None):
            super().__init__(parent)
            self.last = time.monotonic()
            app.installEventFilter(self)

        def eventFilter(self, obj, event):
            if event.type() in self.INPUT:
                self.last = time.monotonic()
            return False

        def idle_seconds(self) -> float:
            return time.monotonic() - self.last

    class _Pass(QObject):
        done = Signal(object)

        def __init__(self, fn: Callable):
            super().__init__()
            self.fn = fn

        def run(self):
            try:
                self.done.emit(self.fn())
            except BaseException as exc:        # reported, never raised into Qt
                self.done.emit({"error": f"{type(exc).__name__}: {exc}", "questions": 0})

    class BackgroundTeacher(QObject):
        """Runs ``run_once`` whenever ``may_run()`` says the app is free."""

        started = Signal()
        report = Signal(object)

        def __init__(self, root: Callable[[], str], may_run: Callable[[], bool],
                     parent=None, train: bool = True,
                     run: Optional[Callable] = None, interval_ms: int = CHECK_EVERY_MS):
            super().__init__(parent)
            self.root, self.may_run, self.train = root, may_run, train
            self._run = run or run_once
            self._thread = None
            self._job = None
            self.timer = QTimer(self)
            self.timer.setInterval(interval_ms)
            self.timer.timeout.connect(self.tick)

        @property
        def running(self) -> bool:
            return self._thread is not None

        def set_enabled(self, on: bool) -> None:
            if on:
                self.timer.start()
            else:
                self.timer.stop()

        def tick(self) -> bool:
            """Start a pass if the app is free; True if one started."""
            if self.running or not self.may_run():
                return False
            root = self.root()
            if not root:
                return False
            self.started.emit()
            self._thread = QThread(self)
            self._job = _Pass(lambda: self._run(root, train=self.train))
            self._job.moveToThread(self._thread)
            self._thread.started.connect(self._job.run)
            self._job.done.connect(self._finished)
            self._thread.finished.connect(self._job.deleteLater)
            self._thread.finished.connect(self._thread.deleteLater)
            self._thread.start()
            return True

        def _finished(self, report):
            self._thread.quit()
            self._thread.wait()
            self._thread = self._job = None
            self.report.emit(report)
