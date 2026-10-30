"""A cancel stops a run at its next check; it never kills the thread.

What happened: Cancel during a transcript waited 5 s, then QThread.terminate()
killed the worker inside PyTorch (Whisper on the processor checks the flag
once per 30 s window). The model and its threads stayed in memory, the next
run (action recognition) loaded its own on top, Windows ran out of virtual
memory and the process died in ntdll — with nothing in the debug log.
"""

from __future__ import annotations

import ast
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _source(name: str) -> str:
    with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
        return fh.read()


def _method(source: str, cls: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return item
    raise AssertionError(f"{cls}.{name} not found")


def _calls(node: ast.AST) -> set:
    return {n.func.attr for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}


def test_no_window_ever_terminates_a_thread():
    for name in ("main.py", "signal_timeline_viewer.py"):
        assert ".terminate()" not in _source(name), name


def test_cancelling_a_run_only_trips_its_flag():
    """No wait() either: that froze the window for the grace period."""
    called = _calls(_method(_source("main.py"), "Worker", "cancel"))
    assert "set" in called
    assert "wait" not in called and "terminate" not in called


def test_a_new_run_waits_for_the_cancelled_one_to_finish():
    source = _source("main.py")
    run = _method(source, "VideoHighlighterGUI", "run_pipeline")
    text = ast.get_source_segment(source, run)
    guard = text.index("self.worker.isRunning()")
    assert text.index("return", guard) < text.index("Validate scoring points")
    assert "is_cancelled()" in text[guard:text.index("Validate scoring points")]
