"""Remove parts of the frozen bundle that the app never loads.

    python tools/strip_bundle.py dist/VideoHighlighter

PyInstaller's PySide6 hooks collect every QML plugin, and the QtWebEngine one
drags in Chromium: Qt6WebEngineCore.dll alone is 194 MB, about a fifth of the
whole compressed core. Nothing here uses it — no Python module imports
QtWebEngine, QtWebChannel or QtPdf, and the one QML file
(video_ai_editor/VRVideoOutput.qml) imports only QtQuick and QtMultimedia.

Every removal is by name, listed below, and printed with its size, so a new
Qt release that renames something shows up as "nothing matched" rather than
as a quietly larger build. The smoke test runs after this step.
"""
from __future__ import annotations

import fnmatch
import os
import shutil
import sys

# Patterns matched against paths relative to the bundle, "/"-separated.
UNUSED = (
    "_internal/PySide6/Qt6WebEngine*",
    "_internal/PySide6/QtWebEngine*",
    "_internal/PySide6/Qt6WebChannel*",
    "_internal/PySide6/QtWebChannel*",
    "_internal/PySide6/Qt6Pdf*",
    "_internal/PySide6/QtPdf*",
    "_internal/PySide6/resources/qtwebengine*",
    "_internal/PySide6/translations/qtwebengine_locales",
    "_internal/PySide6/qml/QtWebEngine",
    "_internal/PySide6/qml/QtWebChannel",
    "_internal/PySide6/qml/QtQuick/Pdf",
)


def _size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(d, f))
               for d, _, files in os.walk(path) for f in files)


def strip(root: str) -> int:
    removed = 0
    matched = set()
    for d, dirs, files in os.walk(root, topdown=True):
        for name in list(dirs) + files:
            full = os.path.join(d, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            hit = next((p for p in UNUSED if fnmatch.fnmatch(rel, p)), None)
            if hit is None:
                continue
            matched.add(hit)
            size = _size(full)
            if os.path.isdir(full):
                shutil.rmtree(full)
                dirs.remove(name)
            else:
                os.remove(full)
            removed += size
            print(f"  - {size / 2**20:7.1f} MB  {rel}")
    for pattern in UNUSED:
        if pattern not in matched:
            print(f"  (nothing matched {pattern})")
    print(f"strip_bundle: removed {removed / 2**20:.0f} MB")
    return removed


if __name__ == "__main__":
    if len(sys.argv) != 2 or not os.path.isdir(sys.argv[1]):
        sys.exit("usage: strip_bundle.py <dist/VideoHighlighter>")
    strip(sys.argv[1])
