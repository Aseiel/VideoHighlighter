"""
A frozen build's multiprocessing children must stop being the app early.

Such a child re-runs the exe, so it starts at the top of main.py. Until it
reaches freeze_support() it believes it is a fresh launch: it rotated the live
debug.log away, and it imported everything main.py imports. Object detection
starts six of them, and on a Mac that took the whole machine's memory.
"""

from __future__ import annotations

import ast
import os

from modules.system import debug_console

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_spawn_child_is_known_by_its_command_line():
    argv = ["VideoHighlighter", "--multiprocessing-fork",
            "tracker_fd=5", "pipe_handle=7"]
    assert debug_console.is_multiprocessing_child(argv)


def test_resource_tracker_is_known_by_its_command_line():
    argv = ["VideoHighlighter", "-B", "-s", "-c",
            "from multiprocessing.resource_tracker import main;main(5)"]
    assert debug_console.is_multiprocessing_child(argv)


def test_an_ordinary_launch_is_not_a_child():
    assert not debug_console.is_multiprocessing_child(["VideoHighlighter"])
    assert not debug_console.is_multiprocessing_child(
        ["VideoHighlighter", "--timeline", "/videos/clip.mov"])


def test_freeze_support_runs_before_the_heavy_imports():
    with open(os.path.join(ROOT, "main.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())

    freeze_line = heavy_line = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "freeze_support"):
            freeze_line = min(freeze_line or node.lineno, node.lineno)
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node.col_offset == 0:
            names = ([node.module or ""] if isinstance(node, ast.ImportFrom)
                     else [a.name for a in node.names])
            if any(n.split(".")[0] in ("cv2", "PySide6", "torch", "openvino")
                   for n in names):
                heavy_line = min(heavy_line or node.lineno, node.lineno)

    assert freeze_line is not None, "main.py no longer calls freeze_support()"
    assert heavy_line is not None
    assert freeze_line < heavy_line, (
        f"freeze_support() at line {freeze_line} runs after the first heavy "
        f"import at line {heavy_line}: every frozen child would load the app")
