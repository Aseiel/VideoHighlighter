"""Nothing the app runs may depend on an AGPL detector.

A model trained on top of an AGPL detector inherits that licence, and the
models people make in this app are meant to be shared and used by anyone, in
any build. So the app's own code — everything except dev-only ``tools/`` —
must not import ultralytics or reach for its model files. ``tools/labeler.py``
may use it for optional pose pre-fill behind a guarded import.

This reads source text rather than importing anything, so it runs without the
app's dependencies installed.
"""
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# dev-only tooling and this test itself are allowed to name it
EXCLUDED_DIRS = {"tools", "tests", "node_modules", ".git", "__pycache__",
                 "build", "dist", ".venv", ".venv-test", "frontend"}

FORBIDDEN = [
    re.compile(r"^\s*(from|import)\s+ultralytics\b", re.MULTILINE),
    re.compile(r"""["']yolo(11|v8)[a-z]*(-pose|-seg|-worldv2)?\.pt["']"""),
]


def _python_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel = os.path.relpath(dirpath, ROOT)
        top = rel.split(os.sep)[0]
        if top in EXCLUDED_DIRS:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for name in filenames:
            if name.endswith(".py"):
                yield os.path.join(dirpath, name)


def test_app_code_never_imports_an_agpl_detector():
    offenders = []
    for path in _python_files():
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        for pattern in FORBIDDEN:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{os.path.relpath(path, ROOT)}:{line}: {match.group(0).strip()}")
    assert not offenders, "AGPL detector reference in app code:\n" + "\n".join(offenders)


def test_requirements_do_not_pull_it_in():
    for name in ("requirements.txt", os.path.join("sidecar", "requirements.txt")):
        path = os.path.join(ROOT, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            lines = [ln.split("#", 1)[0].strip().lower() for ln in fh]
        assert not any(ln.startswith("ultralytics") for ln in lines), name
