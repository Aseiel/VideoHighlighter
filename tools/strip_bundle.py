"""Remove parts of the frozen bundle that the app never loads.

    python tools/strip_bundle.py dist/VideoHighlighter
    python tools/strip_bundle.py dist/VideoHighlighter.app --report 3

The second form is the macOS .app, whose layout differs (Contents/Frameworks
for binaries, Contents/Resources for data, symlinks between the two); its
patterns are listed separately. ``--report N`` then prints where the rest of
the bundle's size is, grouped by the first N path components, and the largest
files.

PyInstaller's PySide6 hooks collect every QML plugin, and the QtWebEngine one
drags in Chromium: Qt6WebEngineCore.dll alone is 194 MB, about a fifth of the
whole compressed core. Nothing here uses it — no Python module imports
QtWebEngine, QtWebChannel or QtPdf, and the one QML file
(video_ai_editor/VRVideoOutput.qml) imports only QtQuick and QtMultimedia.

Every removal is by name, listed below, and printed with its size, so a new
Qt release that renames something shows up as "nothing matched" rather than
as a quietly larger build. The smoke test runs after this step.

On a Mac, a removed folder can leave symlinks pointing at it from the other
half of the bundle; those are removed too, since codesign rejects a bundle with
a dangling link.
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

# The same in a macOS .app (PySide6's wheel keeps Qt under PySide6/Qt), plus
# OpenVINO's readers for TensorFlow, TFLite and Paddle models: the app reads IR
# and ONNX, and converts PyTorch checkpoints, nothing else. "*" matches across
# "/" here, so Contents/*/ covers both Frameworks and Resources.
UNUSED_MAC = (
    "Contents/*/PySide6/Qt/lib/QtWebEngine*",
    "Contents/*/PySide6/Qt/lib/QtWebChannel*",
    "Contents/*/PySide6/Qt/lib/QtPdf*",
    "Contents/*/PySide6/QtWebEngine*",
    "Contents/*/PySide6/QtWebChannel*",
    "Contents/*/PySide6/QtPdf*",
    "Contents/*/PySide6/Qt/qml/QtWebEngine",
    "Contents/*/PySide6/Qt/qml/QtWebChannel",
    "Contents/*/PySide6/Qt/qml/QtQuick/Pdf",
    "Contents/*/PySide6/Qt/translations/qtwebengine_locales",
    "Contents/*/openvino/libs/libopenvino_tensorflow_frontend*",
    "Contents/*/openvino/libs/libopenvino_tensorflow_lite_frontend*",
    "Contents/*/openvino/libs/libopenvino_paddle_frontend*",
)


def _size(path: str) -> int:
    """Bytes under ``path``, not following symlinks (a link costs nothing, and
    following one would count its target twice)."""
    if os.path.islink(path):
        return 0
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(os.path.getsize(os.path.join(d, f))
               for d, _, files in os.walk(path) for f in files
               if not os.path.islink(os.path.join(d, f)))


def _is_app(root: str) -> bool:
    return os.path.isdir(os.path.join(root, "Contents"))


def strip(root: str) -> int:
    patterns = UNUSED_MAC if _is_app(root) else UNUSED
    removed = 0
    matched = set()
    for d, dirs, files in os.walk(root, topdown=True):
        for name in list(dirs) + files:
            full = os.path.join(d, name)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            hit = next((p for p in patterns if fnmatch.fnmatch(rel, p)), None)
            if hit is None:
                continue
            matched.add(hit)
            size = _size(full)
            if os.path.islink(full):
                os.unlink(full)
            elif os.path.isdir(full):
                shutil.rmtree(full)
            else:
                os.remove(full)
            if name in dirs:
                dirs.remove(name)
            removed += size
            print(f"  - {size / 2**20:7.1f} MB  {rel}")
    for pattern in patterns:
        if pattern not in matched:
            print(f"  (nothing matched {pattern})")
    dangling = _remove_dangling_links(root)
    if dangling:
        print(f"  removed {dangling} symlink(s) left pointing at removed files")
    print(f"strip_bundle: removed {removed / 2**20:.0f} MB")
    return removed


def _remove_dangling_links(root: str) -> int:
    gone = 0
    for d, dirs, files in os.walk(root):
        for name in list(dirs) + files:
            full = os.path.join(d, name)
            if os.path.islink(full) and not os.path.exists(full):
                os.unlink(full)
                gone += 1
    return gone


def report(root: str, depth: int = 3, top: int = 30) -> None:
    """Where the bundle's bytes are: by the first ``depth`` path components,
    then the largest single files. Symlinks are not counted."""
    groups, files = {}, []
    for d, _, names in os.walk(root):
        for name in names:
            full = os.path.join(d, name)
            if os.path.islink(full):
                continue
            size = os.path.getsize(full)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            key = "/".join(rel.split("/")[:depth])
            groups[key] = groups.get(key, 0) + size
            files.append((size, rel))
    total = sum(groups.values())
    print(f"bundle size: {total / 2**20:.0f} MB in {len(files)} files")
    for key, size in sorted(groups.items(), key=lambda kv: -kv[1])[:top]:
        print(f"  {size / 2**20:8.1f} MB  {key}")
    print("largest files:")
    for size, rel in sorted(files, reverse=True)[:top]:
        print(f"  {size / 2**20:8.1f} MB  {rel}")


if __name__ == "__main__":
    args = sys.argv[1:]
    depth = None
    if "--report" in args:
        i = args.index("--report")
        depth = int(args[i + 1]) if i + 1 < len(args) else 3
        del args[i:i + 2]
    if len(args) != 1 or not os.path.isdir(args[0]):
        sys.exit("usage: strip_bundle.py <dist/VideoHighlighter | dist/VideoHighlighter.app>"
                 " [--report DEPTH]")
    strip(args[0])
    if depth:
        report(args[0], depth)
