"""The run timer must own its name for the whole of `run_highlighter`.

`run_highlighter` is one 2,000-line function, and it timed itself with a
variable called `start_time`. Two loops inside it unpack action sequences --
`start_time, end_time, duration, confidence, action_name = sequence` -- into the
same function scope, so by the time the timer was read it held the offset of the
last selected sequence. The run then subtracted a few seconds from the wall
clock and reported the age of the Unix epoch:

    Processing time: 29831566m 49s

Nothing caught it because it is not a type error, not an exception, and the
number it prints is only obviously wrong to a person reading the log. Source
level, because the alternative is a full pipeline run over a real video.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PIPELINE = Path(__file__).resolve().parent.parent / "pipeline.py"
TIMER = "run_started_at"


def _run_highlighter() -> ast.FunctionDef:
    tree = ast.parse(PIPELINE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_highlighter":
            return node
    pytest.fail("run_highlighter is gone from pipeline.py")


def _bindings(fn: ast.FunctionDef, name: str) -> list[int]:
    """Every line in `fn` that binds `name` -- assignment, unpacking, for-target."""
    lines = []
    for node in ast.walk(fn):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            targets = [node.target]
        for target in targets:
            for sub in ast.walk(target):
                if isinstance(sub, ast.Name) and sub.id == name:
                    lines.append(node.lineno)
    return lines


def test_the_timer_is_bound_exactly_once():
    bindings = _bindings(_run_highlighter(), TIMER)
    assert bindings, f"{TIMER} is not assigned in run_highlighter"
    assert len(bindings) == 1, (
        f"{TIMER} is rebound at lines {bindings}; the elapsed time it reports "
        "will be measured from whatever was assigned last"
    )


def test_the_elapsed_time_is_measured_from_the_timer():
    src = PIPELINE.read_text(encoding="utf-8")
    assert f"elapsed = time.time() - {TIMER}" in src


def test_sequence_unpacking_still_uses_the_shadowing_names():
    """Not a style check -- it is what makes the test above worth running.

    If those loops were renamed instead, this file would pass while guarding
    nothing.
    """
    fn = _run_highlighter()
    assert _bindings(fn, "start_time"), (
        "no loop unpacks into start_time any more; check whether the timer "
        "still needs a name of its own"
    )
