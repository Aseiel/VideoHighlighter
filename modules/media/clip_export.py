"""Separate clip paths for highlight export.

The concatenated reel stays the default output. When
``export_separate_clips`` is on, each scored segment is also written under
``{video}_clips/`` with stable, human-readable names so the user can pick
individual moments without re-cutting from the timeline.
"""
from __future__ import annotations

import os
import re


def sanitize_base_name(name: str) -> str:
    name = re.sub(r"['\"]", "", name)
    return re.sub(r"[@#$%^&*()]", "_", name)


def format_clip_stamp(seconds: float) -> str:
    """Compact timestamp for filenames (no colons — Windows-safe)."""
    sec = max(0, int(round(float(seconds))))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}h{m:02d}m{s:02d}s"
    return f"{m:02d}m{s:02d}s"


def clip_filename(index: int, start: float, end: float) -> str:
    """1-based index + source range, e.g. ``clip_03_01m20s-01m30s.mp4``."""
    i = max(1, int(index))
    a = format_clip_stamp(start)
    b = format_clip_stamp(end)
    return f"clip_{i:02d}_{a}-{b}.mp4"


def clips_directory(output_file: str, video_base_name: str) -> str:
    """Folder next to the reel: ``<output_dir>/<base>_clips``."""
    output_dir = os.path.dirname(os.path.abspath(output_file)) or "."
    base = sanitize_base_name(video_base_name) or "highlight"
    return os.path.join(output_dir, f"{base}_clips")


def segment_clip_path(
    output_file: str,
    video_base_name: str,
    index: int,
    start: float,
    end: float,
) -> str:
    return os.path.join(
        clips_directory(output_file, video_base_name),
        clip_filename(index, start, end),
    )
