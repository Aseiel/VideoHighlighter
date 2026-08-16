"""A flight recorder for the timeline's rebuilds.

The crash this exists for leaves nothing behind. Adjusting a confidence filter
or a merge gap rebuilds the whole scene, and the process dies somewhere inside
Qt's C++ side, where a Python traceback cannot form and `sys.excepthook` never
runs. The only evidence is whatever reached the disk *before* the process
stopped — so this writes eagerly and flushes every line, and never holds a
breadcrumb back to fold into a tidier summary later. A summary is exactly the
thing that never gets written.

Two things it records, because the two candidate failures look identical from
outside and need opposite fixes:

* **What a rebuild was for** — which control moved, what it moved to, and
  whether a rebuild was already running when this one started. Re-entering
  `build_timeline` while a previous `clear()` is still unwinding is a live
  suspect, and it is invisible unless something counts.

* **Whether the Python-side references to scene items are still backed by a
  live C++ object.** `QGraphicsScene.clear()` deletes those objects but leaves
  the Python attributes pointing at the corpses. A reference that outlives a
  rebuild does not fail at the point of the mistake — it fails later, on the
  next playhead tick, which is why the traceback (when there is one at all)
  never names the code that actually went wrong.

Nothing here changes what the app does, and every entry point swallows its own
errors: a recorder that can raise is a second bug stacked on the one being
chased. `probe` in particular must be safe to call on a dangling wrapper, since
that is its entire job — it asks shiboken whether the object is valid rather
than poking the object and hoping.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager

LOG_NAME = "repaint_trace.log"


def default_path() -> str:
    """Where the trace goes: beside `debug.log`, not beside the exe.

    A frozen build's own directory is often read-only (Program Files), and a
    recorder that fails to open its file is worse than useless — it is a second
    silent failure in the middle of diagnosing the first.
    """
    try:
        from modules.app_paths import user_data_dir
        return os.path.join(user_data_dir(), LOG_NAME)
    except Exception:
        return LOG_NAME

_lock = threading.Lock()
_fh = None
_armed = False
_depth = 0          # rebuilds currently on the stack, for re-entrancy
_generation = 0     # how many rebuilds this process has started


def arm(path: str | None = None) -> bool:
    """Open the trace file and point `faulthandler` at it. Idempotent.

    The handle is deliberately kept open for the life of the process:
    `faulthandler` writes the C-level traceback of a hard crash straight to a
    file descriptor, which is the one kind of output that survives the app
    being killed rather than raising. Reopening per line, the way the timeline's
    own `debug_log` does, would leave nothing for it to write into at the moment
    it matters.
    """
    global _fh, _armed
    with _lock:
        if _armed:
            return True
        try:
            _fh = open(path or default_path(), "a", encoding="utf-8", buffering=1)
            _fh.write(
                f"\n===== repaint trace armed {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"(pid {os.getpid()}) =====\n")
            _fh.flush()
            try:
                import faulthandler
                # A hard crash in Qt lands here as a C traceback. Without this
                # the repaint crash is silent, which is how it stayed unexplained.
                faulthandler.enable(file=_fh)
            except Exception:
                pass
            _armed = True
            return True
        except Exception:
            _fh = None
            return False


def note(event: str, **fields) -> None:
    """Record one breadcrumb. Best-effort and immediately flushed."""
    if _fh is None:
        return
    try:
        stamp = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
        detail = " ".join(f"{k}={v!r}" for k, v in fields.items())
        line = f"[{stamp}] {event}"
        if detail:
            line += f" {detail}"
        with _lock:
            _fh.write(line + "\n")
            _fh.flush()
    except Exception:
        pass


def _valid(obj) -> str:
    """`live`, `dangling`, or `unset` for one possibly-deleted Qt wrapper."""
    if obj is None:
        return "unset"
    try:
        from shiboken6 import isValid
    except Exception:
        # Without shiboken there is no way to ask that is guaranteed not to
        # crash on a dead wrapper, so decline rather than risk taking the
        # process down inside the tool meant to explain why it went down.
        return "unknown"
    try:
        return "live" if isValid(obj) else "dangling"
    except Exception:
        return "dangling"


def probe(owner, *names: str) -> dict:
    """Liveness of each named scene-item attribute of `owner`.

    Call it *before* touching those attributes, on any path that can run after a
    rebuild. A `dangling` here is the crash, caught one moment early and
    attributed to the reference that caused it.
    """
    out = {}
    for name in names:
        out[name] = _valid(getattr(owner, name, None))
    return out


def on_gui_thread() -> str:
    """`yes`, `no`, or `unknown` — Qt objects may only be touched from the GUI
    thread, and doing it from another is a textbook silent abort."""
    try:
        from PySide6.QtCore import QThread
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            return "unknown"
        return "yes" if QThread.currentThread() is app.thread() else "no"
    except Exception:
        return "unknown"


@contextmanager
def rebuild(reason: str, **fields):
    """Bracket one scene rebuild.

    Logs before and after, so a trace ending on a `begin` with no `end` says the
    process died *inside* the rebuild — which is the single most useful fact the
    current logs cannot express. `depth` above 1 means a rebuild started while
    another was still running.
    """
    global _depth, _generation
    with _lock:
        _generation += 1
        _depth += 1
        gen, depth = _generation, _depth
    note("rebuild.begin", reason=reason, gen=gen, depth=depth,
         gui_thread=on_gui_thread(), **fields)
    started = time.monotonic()
    try:
        yield gen
    except BaseException as exc:
        note("rebuild.raised", reason=reason, gen=gen,
             error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        with _lock:
            _depth -= 1
        note("rebuild.end", reason=reason, gen=gen,
             ms=round((time.monotonic() - started) * 1000, 1))


def reset_for_tests() -> None:
    """Drop the handle and counters. Tests only."""
    global _fh, _armed, _depth, _generation
    with _lock:
        try:
            if _fh is not None:
                _fh.close()
        except Exception:
            pass
        _fh, _armed, _depth, _generation = None, False, 0, 0
