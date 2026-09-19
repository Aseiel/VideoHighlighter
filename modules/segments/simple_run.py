"""Built-in scoring for the one-button Simple view.

No Qt. Simple Analyze overlays this onto the pipeline config for that run
only — Detailed settings widgets are left alone. Path and status helpers live
here for the same reason: CI does not install PySide6.
"""
from __future__ import annotations

from pathlib import Path

SIMPLE_SCORING = {
    "scene_points": 2,
    "motion_event_points": 0,
    "motion_peak_points": 5,
    "audio_peak_points": 0,
    "loudness_burst_points": 5,
    "keyword_points": 0,
    "transcript_points": 0,
    "object_points": 0,
    "action_points": 0,
    "face_expression_points": 0,
    "face_expression_labels": [],
    "beginning_points": 0,
    "ending_points": 0,
    "use_transcript": False,
    "create_subtitles": False,
    "highlight_objects": None,
    "interesting_actions": None,
    "write_highlight_report": True,
    # Reel plus a folder of named clips — Simple's default packaging.
    "export_separate_clips": True,
}

# max_duration seconds, clip_time seconds
SIMPLE_LENGTHS = {
    "short": (90, 8),
    "medium": (240, 10),
    "long": (420, 12),
}


def apply_simple_run(config: dict, length: str = "medium") -> dict:
    """Overlay the one-button preset onto a pipeline config (in place)."""
    config.update(SIMPLE_SCORING)
    max_dur, clip = SIMPLE_LENGTHS.get(length, SIMPLE_LENGTHS["medium"])
    config["max_duration"] = max_dur
    config["clip_time"] = clip
    config["exact_duration"] = None
    return config


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def is_video_path(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


def idle_status_text(n: int) -> str:
    """What the status line says when a run is not in progress."""
    if n <= 0:
        return "Ready"
    if n == 1:
        return "1 video ready — press Analyze"
    return f"{n} videos ready — press Analyze"


def simple_scoring_total() -> int:
    keys = (
        "scene_points", "motion_event_points", "motion_peak_points",
        "audio_peak_points", "loudness_burst_points", "keyword_points",
        "transcript_points", "object_points", "action_points",
        "face_expression_points", "beginning_points", "ending_points",
    )
    return sum(int(SIMPLE_SCORING[k]) for k in keys)
