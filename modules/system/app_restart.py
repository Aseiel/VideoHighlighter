"""Restart the running app on request, and come back where it was.

Code changed on disk reaches a running app only through a restart. Asked over
the live link (``restart``), the app does it itself instead of having its
process killed from outside: it refuses while work is in progress, closes
through its own close path (settings and rules are saved there), and the new
process reopens the timeline viewer on the same video at the same moment.

The order matters. The new process is started first and the old one closes
after: the old one ends with a hard kill of itself (``_hard_exit``), which
takes nothing else with it, and the new one takes seconds to import, by which
time the old one and its live-link pipe are gone.

What to reopen travels in two environment variables the new process reads once
(``take_reopen``) and removes, so a later restart by hand opens nothing.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Callable, Iterable, Optional, Sequence

ENV_VIDEO = "VH_REOPEN_VIDEO"
ENV_SECONDS = "VH_REOPEN_AT"

# Attributes of the main window and the timeline viewer that hold work in
# progress, with the words for the refusal. Anything with ``isRunning()``
# (QThread) or ``is_alive()`` (threading.Thread) counts.
MAIN_WORK = (("worker", "a highlight run"),
             ("_signal_worker", "an analysis run"),
             ("download_worker", "a download"),
             ("_scan_worker", "a face scan"))
VIEWER_WORK = (("_analysis_thread", "an analysis in the timeline viewer"),)


def _running(obj) -> bool:
    if obj is None:
        return False
    for probe in ("isRunning", "is_alive"):
        fn = getattr(obj, probe, None)
        if callable(fn):
            try:
                return bool(fn())
            except RuntimeError:          # a deleted Qt object
                return False
    return False


def busy(holders: Iterable[tuple]) -> list:
    """What is in progress: ``holders`` is ``(object, ((attr, words), ...))``
    pairs; returns the words for each attribute whose work is running."""
    found = []
    for owner, attrs in holders:
        if owner is None:
            continue
        for attr, words in attrs:
            if _running(getattr(owner, attr, None)):
                found.append(words)
    return found


def relaunch_command(argv: Optional[Sequence[str]] = None,
                     executable: Optional[str] = None,
                     frozen: Optional[bool] = None) -> list:
    """The command that started this process. A packaged build's argv[0] is its
    own exe, which ``executable`` already is; from source it is main.py, which
    the interpreter needs as its first argument."""
    argv = list(sys.argv if argv is None else argv)
    executable = executable or sys.executable
    frozen = getattr(sys, "frozen", False) if frozen is None else frozen
    return [executable] + (argv[1:] if frozen else argv)


def spawn(video: Optional[str] = None, seconds: Optional[float] = None, *,
          command: Optional[Sequence[str]] = None,
          popen: Callable = subprocess.Popen):
    """Start the new process, detached from this one, in the same folder (the
    app reads ./cache and config.yaml relative to it)."""
    env = dict(os.environ)
    env.pop(ENV_VIDEO, None)
    env.pop(ENV_SECONDS, None)
    if video:
        env[ENV_VIDEO] = str(video)
        if seconds is not None:
            env[ENV_SECONDS] = f"{float(seconds):.3f}"
    kwargs = {"cwd": os.getcwd(), "env": env, "close_fds": True,
              "stdin": subprocess.DEVNULL}
    if sys.platform == "win32":
        kwargs["creationflags"] = (getattr(subprocess, "DETACHED_PROCESS", 0)
                                   | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
    else:
        kwargs["start_new_session"] = True
    return popen(list(command or relaunch_command()), **kwargs)


def take_reopen(environ=None) -> Optional[tuple]:
    """``(video, seconds)`` a restart asked this process to reopen, or None.
    Removed as it is read, so it happens once."""
    environ = os.environ if environ is None else environ
    video = environ.pop(ENV_VIDEO, None)
    raw = environ.pop(ENV_SECONDS, None)
    if not video or not os.path.isfile(video):
        return None
    try:
        seconds = max(0.0, float(raw)) if raw is not None else None
    except ValueError:
        seconds = None
    return video, seconds
