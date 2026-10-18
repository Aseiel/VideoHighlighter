"""List the UI strings a language's catalog is missing.

Two passes, because neither alone finds everything:

* **scan** -- every string literal passed straight to a Qt call that
  modules/i18n/qt_hooks.py translates (QLabel(...), setToolTip(...),
  addRow(...), QMenu.addAction(...), ...). Finds strings that only appear in a
  context menu or a message box, which no window shows at startup.
* **runtime** -- builds the main window and the timeline window offscreen in
  that language and records every string that reached t() and fell back to
  English. Finds strings that reach Qt through a helper or a table (the points
  panel's groups, for one), which no scan of call arguments sees.

    python tools/i18n_misses.py zh-CN            # both passes, prints a summary
    python tools/i18n_misses.py zh-CN --json out.json

A string with no letters ("0%", "00:00") is never listed; f-strings are not
listed either, since there is no fixed text to translate.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Calls whose first text argument the hooks translate, and that argument's
# position.
_CTORS = {
    "QLabel": 0, "QPushButton": 0, "QCheckBox": 0, "QRadioButton": 0,
    "QToolButton": 0, "QGroupBox": 0, "QDockWidget": 0, "QMenu": 0,
    "QAction": 0, "QListWidgetItem": 0, "QGraphicsTextItem": 0,
    "CollapsibleSection": 0,
}
_METHODS = {
    "setText": 0, "setToolTip": 0, "setStatusTip": 0, "setPlaceholderText": 0,
    "setTitle": 0, "setWindowTitle": 0, "addAction": 0, "addMenu": 0,
    "addSection": 0, "addItem": 0, "addTab": 1, "setTabText": 1,
    "setTabToolTip": 1, "addRow": 0, "showMessage": 0, "addText": 0,
    "setPlainText": 0, "setInformativeText": 0, "addButton": 0,
}
_MESSAGE_BOX = {"information", "warning", "critical", "question"}

_SCAN_ROOTS = ("main.py", "signal_timeline_viewer.py", "video_ai_editor",
               "modules", "llm", "model_hub", "training")
_SKIP_PARTS = {"tests", "tools", "teach", ".claude", "__pycache__"}


def _is_copy(value: str) -> bool:
    if not value.strip() or not any(ch.isalpha() for ch in value):
        return False
    v = value.lstrip()
    if v.startswith(("{", "<html", "http://", "https://", "/", "\\")):
        return False
    return not ("px;" in value or "background-color:" in value
                or "font-family:" in value)


def _literal(node) -> str | None:
    """A plain string, including implicit concatenation; not an f-string."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def scan(roots=_SCAN_ROOTS) -> dict[str, str]:
    """{msgid: 'file:line'} for every literal reaching a hooked call."""
    found: dict[str, str] = {}
    files = []
    for root in roots:
        p = REPO / root
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(f for f in p.rglob("*.py")
                         if not _SKIP_PARTS & set(f.relative_to(REPO).parts))
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = (fn.id if isinstance(fn, ast.Name)
                    else fn.attr if isinstance(fn, ast.Attribute) else None)
            if name is None:
                continue
            positions = []
            if name in _CTORS:
                positions = [_CTORS[name]]
            elif name in _METHODS:
                positions = [_METHODS[name]]
                if name in ("addAction", "addMenu", "addItem"):
                    positions.append(1)        # after an icon
            elif name in _MESSAGE_BOX:
                positions = [1, 2]             # (parent, title, text)
            for pos in positions:
                if pos < len(node.args):
                    value = _literal(node.args[pos])
                    if value is not None and _is_copy(value):
                        rel = path.relative_to(REPO).as_posix()
                        found.setdefault(value, f"{rel}:{node.lineno}")
            if name in ("setHeaderLabels", "setHorizontalHeaderLabels") and node.args:
                arg = node.args[0]
                if isinstance(arg, (ast.List, ast.Tuple)):
                    for elt in arg.elts:
                        value = _literal(elt)
                        if value is not None and _is_copy(value):
                            found.setdefault(value, f"{path.name}:{node.lineno}")
    return found


def runtime(lang: str, video: str | None = None) -> dict[str, str]:
    """{msgid: 'main window' | 'timeline window'} for strings that fell back."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)
    from modules.i18n import core

    core.set_language(lang)
    misses: dict[str, str] = {}
    stage = {"name": "startup"}

    class Recording(dict):
        def get(self, key, default=None):
            # t() tries a padded string as-is before trimming it, so only a
            # miss on the trimmed text is a real gap.
            if key not in self and key.strip() not in self and _is_copy(key):
                misses.setdefault(key, stage["name"])
            return super().get(key, default)

    core._catalog[core.current_language()] = Recording(
        core._catalog.get(core.current_language()) or {})

    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])  # noqa: F841
    from modules import i18n
    i18n.install_qt_hooks()

    stage["name"] = "main window"
    import main as app_main
    app_main.VideoHighlighterGUI()

    stage["name"] = "timeline window"
    import signal_timeline_viewer as stv
    cache = {"video_metadata": {"duration": 90.0, "fps": 30.0},
             "motion_peaks": [10.0, 30.0], "motion_events": [10.0, 10.2],
             "scenes": [{"start": 0, "end": 45}, {"start": 45, "end": 90}]}
    stv.SignalTimelineWindow(video or "missing.mp4", cache)
    return misses


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("lang")
    ap.add_argument("--json", help="write {msgid: where} here")
    ap.add_argument("--no-runtime", action="store_true",
                    help="scan only (no Qt needed)")
    ap.add_argument("--video", help="a real video for the timeline window")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(REPO))
    from modules.i18n import catalog
    from modules.i18n.languages import normalize_code
    have = catalog.TRANSLATIONS.get(normalize_code(args.lang), {})

    where = {k: v for k, v in scan().items() if k not in have}
    if not args.no_runtime:
        for k, v in runtime(args.lang, args.video).items():
            if k not in have:
                where.setdefault(k, v)

    if args.json:
        Path(args.json).write_text(
            json.dumps(dict(sorted(where.items())), ensure_ascii=False, indent=1),
            encoding="utf-8")
    print(f"{len(where)} strings missing from {args.lang}")
    sys.stdout.flush()
    os._exit(0)                     # skip Qt teardown of half-built windows


if __name__ == "__main__":
    raise SystemExit(main())
