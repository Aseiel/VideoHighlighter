"""A shot list as files other tools read: a CMX 3600 EDL and a plain list.

The EDL holds one cut-only video event per shot, with the source and record
times the same - the video *is* the edited programme, so a shot sits in the
timeline where it sits in the file. That is the form subtitle editors and
NLEs read as "the cuts of this video" (checked against Tero Subtitler's
importer, which takes the record in and out of every event as edit points).
The record clock starts at 00:00:00:00 rather than the editing convention of
01:00:00:00: these times are positions in a file, and an importer that
subtracts an hour base then has nothing to guess.

The plain list is one cut per line as ``HH:MM:SS.mmm``, which needs no frame
rate to read back and so survives a variable-frame-rate file exactly.

Timecode arithmetic is ``video_ai_editor.timeline_export``'s, so a shot list
and an exported edit of the same video count frames the same way.
"""
from __future__ import annotations

import os

from modules.segments.shot_boundaries import ShotList
from video_ai_editor.timeline_export import (
    Timebase, edl_frame_index, frames_to_timecode, reel_name, _event_number,
    _write_text,
)


def _stem(shots: ShotList) -> str:
    return os.path.splitext(os.path.basename(shots.video_path))[0]


def edl_text(shots: ShotList) -> str:
    """CMX 3600 text: one event per shot."""
    timebase = Timebase.from_fraction(shots.fps_num, shots.fps_den)
    stem = _stem(shots)
    reel = reel_name(stem)
    lines = [f"TITLE: {stem} shots", "FCM: NON-DROP FRAME", ""]
    number = 0
    for start, end in shots.shots:
        a = edl_frame_index(start, timebase)
        b = edl_frame_index(end, timebase)
        if b <= a:              # shorter than one frame on the EDL clock
            continue
        tc_in = frames_to_timecode(a, timebase.edl_fps)
        tc_out = frames_to_timecode(b, timebase.edl_fps)
        lines.append(f"{_event_number(number):03d}  {reel} V     C        "
                     f"{tc_in} {tc_out} {tc_in} {tc_out}")
        if number == 0:
            lines.append(f"* FROM CLIP NAME: {os.path.basename(shots.video_path)}")
        number += 1
    return "\n".join(lines) + "\n"


def _clock(seconds: float) -> str:
    ms = max(0, int(round(float(seconds) * 1000.0)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def plain_text(shots: ShotList) -> str:
    """One cut per line, ``HH:MM:SS.mmm`` from the start of the video."""
    return "".join(_clock(t) + "\n" for t in shots.cut_times)


def default_path(shots: ShotList, extension: str) -> str:
    base = os.path.splitext(shots.video_path)[0]
    return f"{base}_shots.{extension}"


def write_edl(shots: ShotList, output_path: str | None = None) -> str:
    path = output_path or default_path(shots, "edl")
    _write_text(path, edl_text(shots))
    return path


def write_text(shots: ShotList, output_path: str | None = None) -> str:
    path = output_path or default_path(shots, "txt")
    _write_text(path, plain_text(shots))
    return path
