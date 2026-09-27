"""Cut each source video into fixed-length samples.

This is the step done by hand before: eight-minute videos cut into
five-second pieces, each one a candidate example. Five seconds holds one
action from start to end, and it is the unit the sorter, the review sheet and
the action trainer all work in.

Cuts are re-encoded rather than stream-copied. A stream copy can only start on
a keyframe, so its "five seconds from 0:40" really starts wherever the last
keyframe was — up to several seconds early — and the sample's label then
describes a clip that is not the one in the file. Re-encoding a five-second
H.264 clip at ``veryfast`` costs well under a second.
"""
from __future__ import annotations

import os
import subprocess
from typing import Callable, Optional

from modules.teach.project import Project, Sample

# The last piece of a video is kept if it is at least this fraction of a full
# sample; shorter ones are mostly the fade to black.
MIN_TAIL_FRACTION = 0.6

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".wmv", ".flv")


def plan_segments(duration: float, length: float, stride: Optional[float] = None) -> list:
    """``[(start, length), ...]`` covering ``duration`` seconds.

    ``stride`` is the step between starts (default: ``length``, back to back).
    A stride shorter than the length overlaps samples; that makes more of them
    from little footage, at the cost of neighbours that are nearly the same.
    """
    if duration <= 0 or length <= 0:
        return []
    stride = stride or length
    out = []
    start = 0.0
    while start < duration:
        piece = min(length, duration - start)
        if piece >= length * MIN_TAIL_FRACTION:
            out.append((round(start, 3), round(piece, 3)))
        start += stride
    return out


def sample_id(source_id: str, start: float) -> str:
    return f"{source_id}__{int(round(start * 1000)):08d}"


def ffmpeg_exe() -> str:
    from modules.media.ffmpeg_tools import _bundled_ffmpeg
    return _bundled_ffmpeg() or "ffmpeg"


def probe_duration(path: str) -> float:
    try:
        from modules.media.ffmpeg_tools import probe
        seconds = float((probe(path).get("format") or {}).get("duration") or 0.0)
        if seconds > 0:
            return seconds
    except Exception:
        pass
    import cv2
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        return float(frames / fps) if fps else 0.0
    finally:
        cap.release()


def cut_command(src: str, dst: str, start: float, length: float,
                ffmpeg: str = "ffmpeg") -> list:
    return [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{start:.3f}", "-i", src, "-t", f"{length:.3f}",
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", dst]


def cut_project(project: Project, *, run: Optional[Callable] = None,
                duration_of: Optional[Callable] = None,
                progress: Optional[Callable] = None) -> dict:
    """Cut every source not cut yet. Re-running cuts nothing twice."""
    run = run or subprocess.run
    duration_of = duration_of or probe_duration
    settings = project.settings
    out_dir = project.path("samples")
    os.makedirs(out_dir, exist_ok=True)
    known = {s.id for s in project.samples}
    ffmpeg = ffmpeg_exe()
    made, failed = 0, []

    for source in project.sources:
        if source.cut:
            continue
        if not os.path.exists(source.path):
            failed.append({"source": source.id, "error": f"missing: {source.path}"})
            continue
        source.duration = source.duration or duration_of(source.path)
        plan = plan_segments(source.duration, settings.clip_seconds,
                             settings.stride_seconds)
        ok = True
        for i, (start, length) in enumerate(plan):
            sid = sample_id(source.id, start)
            dst = os.path.join(out_dir, sid + ".mp4")
            if sid not in known:
                if not os.path.exists(dst):
                    result = run(cut_command(source.path, dst, start, length, ffmpeg),
                                 capture_output=True, text=True)
                    if getattr(result, "returncode", 0) != 0 or not os.path.exists(dst):
                        ok = False
                        failed.append({"source": source.id, "sample": sid,
                                       "error": (getattr(result, "stderr", "") or "").strip()[-300:]})
                        continue
                project.samples.append(Sample(id=sid, source=source.id, path=dst,
                                              start=start, duration=length))
                known.add(sid)
                made += 1
            if progress:
                progress(source.id, i + 1, len(plan))
        source.cut = ok
        project.save()

    return {"samples_made": made, "samples_total": len(project.samples),
            "failed": failed}
