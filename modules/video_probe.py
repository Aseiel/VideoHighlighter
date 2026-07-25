"""
video_probe.py — one-shot ffprobe metadata for a video file.

The engine already asks ffprobe questions in several places, each with its own
subprocess and parsing quirks (`pipeline.get_video_duration`,
`modules.encoder_select.probe_video_size`). The web UI needs more than any of
them expose — display rotation in particular, so a portrait phone clip can be
previewed the right way up — and running one probe per question multiplies
process spawns per file. `probe_video` makes a single JSON call and returns
everything at once.

Rotation convention
===================
``rotation`` is the clockwise degrees a *player* must rotate the stored frames
for correct display, one of {0, 90, 180, 270}.

Two metadata sources exist in the wild:

  - The modern displaymatrix side data. ffprobe reports its angle
    counter-clockwise, so the typical portrait phone clip shows ``-90`` —
    meaning the player rotates 90 CW. The sign flips on the way through.
  - The legacy QuickTime ``rotate`` tag, which is already clockwise and passes
    through unchanged.

Negative and over-360 values from either source normalize mod 360, so -90,
270 and 630 in the same source mean the same orientation. When both sources
are present the displaymatrix wins — it is what modern muxers actually
maintain; stale rotate tags are the leftover.
"""

from __future__ import annotations

import json
import os
import subprocess

from modules.app_paths import ffmpeg_exe


def ffprobe_exe() -> str:
    """Resolve ffprobe the same way `app_paths.ffmpeg_exe` resolves ffmpeg.

    Prefer the binary sitting next to that ffmpeg so the pair stays
    version-matched, falling back to bare "ffprobe" on PATH. imageio-ffmpeg
    ships no ffprobe, so a frozen app without a system install can still miss;
    the caller then sees FileNotFoundError from subprocess.
    """
    ff = ffmpeg_exe()
    base = os.path.basename(ff)
    if "ffmpeg" in base.lower():
        sibling = os.path.join(os.path.dirname(ff),
                               base.lower().replace("ffmpeg", "ffprobe"))
        if os.path.exists(sibling):
            return sibling
    return "ffprobe"


def _float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fps(stream: dict) -> float:
    """Frame rate from the ffprobe fraction strings (e.g. "30000/1001").

    r_frame_rate first; avg_frame_rate as fallback because mis-tagged streams
    sometimes report r_frame_rate as "0/0".
    """
    for key in ("r_frame_rate", "avg_frame_rate"):
        num, _, den = str(stream.get(key) or "").partition("/")
        try:
            value = float(num) / (float(den) if den else 1.0)
        except (ValueError, ZeroDivisionError):
            continue
        if value > 0:
            return value
    return 0.0


def _normalize_cw(degrees: float) -> int:
    """Snap arbitrary clockwise degrees into {0, 90, 180, 270}."""
    return int(round(degrees / 90.0)) * 90 % 360


def _rotation_from_stream(stream: dict) -> int:
    """Player-clockwise rotation from an ffprobe video-stream dict (see module
    docstring for the sign convention). Missing or garbage metadata is 0 —
    an unreadable orientation must never break a probe."""
    for side in stream.get("side_data_list") or []:
        if "rotation" in side:
            try:
                # ffprobe reports counter-clockwise; flip to player-clockwise.
                return _normalize_cw(-float(side["rotation"]))
            except (TypeError, ValueError):
                pass
    try:
        return _normalize_cw(float((stream.get("tags") or {}).get("rotate")))
    except (TypeError, ValueError):
        return 0


def probe_video(path: str) -> dict:
    """All display-relevant metadata for ``path`` from a single ffprobe call.

    Returns ``{"duration": float, "width": int, "height": int, "fps": float,
    "rotation": int}``. Duration prefers the container (format) value —
    per-stream durations lie more often on VFR files — falling back to the
    first video stream. Width/height are the *stored* dimensions; when
    rotation is 90/270 the displayed aspect is their swap, and that stays the
    caller's job so the raw numbers remain trustworthy.

    ffprobe failures propagate (CalledProcessError / FileNotFoundError):
    a file we cannot probe is the caller's decision, not a silent zero.
    """
    out = subprocess.run(
        [ffprobe_exe(), "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", str(path)],
        capture_output=True, encoding="utf-8", errors="replace",
        check=True, timeout=30,
    ).stdout
    data = json.loads(out or "{}")
    stream = next((s for s in data.get("streams") or []
                   if s.get("codec_type") == "video"), {})
    fmt = data.get("format") or {}
    return {
        "duration": _float(fmt.get("duration")) or _float(stream.get("duration")),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": _fps(stream),
        "rotation": _rotation_from_stream(stream),
    }


def get_rotation(path: str) -> int:
    """Convenience for callers that only care about orientation."""
    return probe_video(path)["rotation"]
