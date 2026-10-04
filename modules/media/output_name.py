"""Where the highlight mp4 is written.

The GUI has one field for this (``highlights.output``). Leaving it blank keeps
the name the pipeline already used: ``<video>_highlight.mp4`` beside the
source. Anything else is the file's base name, and is actually used — a batch
still puts each source's own stem in front so two files in one folder do not
overwrite each other.
"""
from __future__ import annotations

import os
import re

_UNSAFE = re.compile(r"['\"@#$%^&*()]")


def sanitize_output_base(name: str) -> str:
    """Drop quotes and a few characters ffmpeg's concat list treats specially.

    The source stem is left alone: existing ``<video>_highlight.mp4`` files
    must keep resolving to the same path.
    """
    return _UNSAFE.sub("", name).strip()


def highlight_output_path(video_path: str, output_base: str | None, *,
                          multiple: bool) -> str:
    """Absolute-or-relative path of the highlight for one source video.

    ``output_base`` is a file name, not a directory. A blank value means
    ``<stem>_highlight.mp4`` next to ``video_path`` (what a GUI run writes
    today when the field is left empty). A name is used as-is for a single
    video, and as ``<stem>_<name>.mp4`` when ``multiple`` is set. ``.mp4`` is
    added when the name has no extension. The file stays in the source's
    folder; a directory typed into the field is ignored.
    """
    source_dir = os.path.dirname(video_path) or "."
    stem = os.path.splitext(os.path.basename(video_path))[0] or "video"
    raw = os.path.basename(str(output_base or "").replace("\\", "/").strip())
    raw = sanitize_output_base(raw)

    if not raw:
        return os.path.join(source_dir, f"{stem}_highlight.mp4")

    root, ext = os.path.splitext(raw)
    if ext.lower() not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        root = raw
        ext = ".mp4"
    root = root.strip() or "highlight"

    if multiple:
        filename = f"{stem}_{root}{ext}"
    else:
        filename = f"{root}{ext}"
    return os.path.join(source_dir, filename)
