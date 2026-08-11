"""Every run that can show detection frames must be handed a way to show them.

The live preview is opt-in from one checkbox, so the user reads it as a property
of the app: tick it, and runs show their frames. It is actually a property of
each *call site* — `run_highlighter(..., preview_fn=...)` — and a call site that
omits the argument does not fail, warn, or look any different. The window opens,
holds its "Waiting for the detection stage" placeholder for the whole run, and
the only conclusion available to the user is that the preview is broken.

That is what happened: the Run button passed `preview_fn`, and the
download-and-process path (`start_download` → `process_video_callback`) did not,
so a downloaded video processed on the spot never fed the window while the same
video started from Run did.

Source-level rather than behavioural, deliberately. Reaching these call sites
for real means a QApplication, a download worker and a full pipeline over a real
video with the OpenVINO models loaded — none of which CI has (see conftest: the
suite runs without the heavy deps and without Qt). The defect is an omitted
keyword argument at a call site, so a check that reads the call sites catches
exactly it, in milliseconds, on a machine with nothing installed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _calls_named(path: Path, func_name: str):
    """Every call to `func_name` in `path`, as (line, set-of-keyword-names)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = (fn.id if isinstance(fn, ast.Name) else
                fn.attr if isinstance(fn, ast.Attribute) else None)
        if name == func_name:
            found.append((node.lineno, {kw.arg for kw in node.keywords}))
    return found


def test_every_run_highlighter_call_passes_preview_fn():
    """A pipeline run started from anywhere can feed the preview window."""
    offenders = []
    for rel in ("main.py", "pipeline.py"):
        path = REPO / rel
        for lineno, kwargs in _calls_named(path, "run_highlighter"):
            if "preview_fn" not in kwargs:
                offenders.append(f"{rel}:{lineno}")

    assert not offenders, (
        "run_highlighter called without preview_fn at "
        + ", ".join(offenders)
        + " — a run started there leaves the live preview window on its "
          "placeholder, which is indistinguishable from a broken preview."
    )


def test_run_highlighter_call_sites_are_actually_found():
    """Guard the guard: an AST walk that matches nothing would pass silently."""
    sites = (_calls_named(REPO / "main.py", "run_highlighter")
             + _calls_named(REPO / "pipeline.py", "run_highlighter"))
    assert len(sites) >= 3, f"expected the known call sites, found {len(sites)}"


@pytest.mark.parametrize("module, func", [
    ("action_recognition.py", "run_action_detection"),
    ("object_recognition.py", "run_object_detection_single"),
])
def test_detector_accepts_a_preview_hook(module, func):
    """Both detection stages still take `preview_fn` — the pipeline passes it
    positionally to neither, so a rename here fails loudly rather than landing
    in the `**kwargs` of something that ignores it."""
    tree = ast.parse((REPO / module).read_text(encoding="utf-8"))
    defs = [n for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == func]
    assert defs, f"{func} not found in {module}"
    args = {a.arg for a in defs[0].args.args + defs[0].args.kwonlyargs}
    assert "preview_fn" in args, f"{module}:{func} no longer takes preview_fn"


ONDEMAND_CALLERS = ("main.py", "signal_timeline_viewer.py")
ONDEMAND_RUNNERS = ("run_actions", "run_objects")


@pytest.mark.parametrize("runner", ONDEMAND_RUNNERS)
def test_ondemand_runner_forwards_preview_fn(runner):
    """`run_actions`/`run_objects` take a preview hook and hand it to the
    detector — an on-demand run detects over the whole video exactly as the
    pipeline's stage does, so it has the same frames to show."""
    tree = ast.parse((REPO / "modules" / "analysis_ondemand.py")
                     .read_text(encoding="utf-8"))
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == runner), None)
    assert fn is not None, f"{runner} not found in analysis_ondemand.py"

    accepted = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    assert "preview_fn" in accepted, f"{runner} does not accept preview_fn"

    detector_calls = [n for n in ast.walk(fn)
                      if isinstance(n, ast.Call)
                      and getattr(n.func, "id", None) in
                      ("run_action_detection", "run_object_detection_single")]
    assert detector_calls, f"{runner} calls no detector"
    for call in detector_calls:
        assert "preview_fn" in {kw.arg for kw in call.keywords}, (
            f"{runner} accepts preview_fn but does not pass it on — the hook is "
            f"dropped and the preview window stays on its placeholder.")


@pytest.mark.parametrize("caller", ONDEMAND_CALLERS)
def test_ondemand_call_sites_pass_preview_fn(caller):
    """Both Analyze panels (main window and timeline viewer) feed the preview.

    They are the same feature in two windows, so a preview that works in one
    and not the other is the same bug reported twice.
    """
    path = REPO / caller
    offenders = []
    for runner in ONDEMAND_RUNNERS:
        for lineno, kwargs in _calls_named(path, runner):
            if "preview_fn" not in kwargs:
                offenders.append(f"{caller}:{lineno} ({runner})")
    assert not offenders, (
        "on-demand run started without preview_fn at " + ", ".join(offenders))


def test_ondemand_call_sites_are_actually_found():
    """Guard the guard, as above."""
    total = sum(len(_calls_named(REPO / c, r))
                for c in ONDEMAND_CALLERS for r in ONDEMAND_RUNNERS)
    assert total >= 4, f"expected both panels' calls, found {total}"


def test_preview_failures_are_not_swallowed_silently():
    """The preview block reports a failure instead of `except Exception: pass`.

    A bare pass there is why an empty window could not be told apart from a
    window nobody fed: both look identical from the outside and neither leaves
    a line in debug.log.
    """
    for module in ("action_recognition.py", "object_recognition.py"):
        src = (REPO / module).read_text(encoding="utf-8")
        head = src.index("preview_fn is not None")
        tail = src.index("\n", src.index("_preview_failed = True", head))
        block = src[head:tail]
        assert "except Exception:\n" not in block, (
            f"{module}: the live-preview block swallows exceptions silently")
