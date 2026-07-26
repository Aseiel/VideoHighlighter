"""
music_track.py — mux a music bed onto a finished highlight video.

The video stream is NEVER re-encoded (``-c:v copy`` always); only the audio
side of the container changes, so applying music to a long reel takes seconds
and cannot degrade the picture.

Public API
==========

    apply_music(video, music, out, mode='replace', music_volume=0.8,
                log_fn=print) -> str

Modes
-----
    replace  Drop the original audio entirely. The music loops when shorter
             than the video (``-stream_loop -1``), ends with it (``-shortest``)
             and fades out over the last 2 seconds so the reel never cuts off
             mid-bar.
    mix      Original audio and music play together; the music is scaled by
             ``music_volume`` and the output keeps the video's duration
             (``amix duration=first``).
    duck     Like mix, but the music is side-chain compressed against the
             original audio: it dips when people talk or the source gets loud,
             and swells back during quiet stretches.

A video with no audio stream cannot mix or duck; both silently degrade to
replace (with a log note) rather than fail the whole run. Output always goes
to ``out`` — never in-place.
"""

from __future__ import annotations

import os
import subprocess

from modules.app_paths import ffmpeg_exe

_MODES = ("replace", "mix", "duck")


def _ffprobe_exe() -> str:
    """ffprobe sitting next to whatever ffmpeg_exe() resolved (system installs
    ship them side by side); bare "ffprobe" when there is no sibling."""
    fp = ffmpeg_exe()
    base = os.path.basename(fp)
    if "ffmpeg" in base.lower():
        cand = os.path.join(os.path.dirname(fp),
                            base.lower().replace("ffmpeg", "ffprobe"))
        if os.path.exists(cand):
            return cand
    return "ffprobe"


def _has_audio_stream(path: str) -> bool:
    """True when ffprobe sees at least one audio stream. Optimistic on probe
    failure: a wrong True makes the mix/duck ffmpeg call fail loudly instead
    of silently dropping the original audio."""
    try:
        out = subprocess.run(
            [_ffprobe_exe(), "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=30)
        return bool((out.stdout or "").strip())
    except Exception:
        return True


def _video_duration(path: str) -> float:
    # Deferred import: video_probe is a sibling module that may not exist in
    # minimal environments; music_track itself must stay importable there.
    from modules.video_probe import probe_video
    return float(probe_video(path)["duration"])


def apply_music(video, music, out, mode="replace", music_volume=0.8,
                log_fn=print) -> str:
    """Write a copy of ``video`` to ``out`` with ``music`` applied per ``mode``.

    Returns ``out``. Raises ValueError for an unknown mode or missing input
    files, RuntimeError when ffmpeg itself fails.
    """
    if mode not in _MODES:
        raise ValueError(f"unknown music mode: {mode!r} (expected one of {_MODES})")
    if not video or not os.path.exists(video):
        raise ValueError(f"video not found: {video!r}")
    if not music or not os.path.exists(music):
        raise ValueError(f"music file not found: {music!r}")
    if os.path.abspath(out) == os.path.abspath(video):
        raise ValueError("output must not overwrite the input video")

    vol = max(0.0, min(1.0, float(music_volume)))

    effective = mode
    if mode in ("mix", "duck") and not _has_audio_stream(video):
        log_fn(f"🎵 Video has no audio stream; '{mode}' degrades to 'replace'")
        effective = "replace"

    if effective == "replace":
        duration = _video_duration(video)
        if duration > 0.25:
            fade = min(2.0, duration)
            fade_start = max(0.0, duration - fade)
            graph = (f"[1:a]volume={vol:g},"
                     f"afade=t=out:st={fade_start:.3f}:d={fade:.3f}[a]")
        else:
            # Unprobeable/near-zero duration: a fade would be invalid; the
            # -shortest cut still bounds the looped music.
            graph = f"[1:a]volume={vol:g}[a]"
        extra = ["-shortest"]
    elif effective == "mix":
        # normalize=0 (ffmpeg >= 4.4) keeps the original audio at full level
        # instead of amix halving every input.
        graph = (f"[1:a]volume={vol:g}[m];"
                 f"[0:a][m]amix=inputs=2:duration=first:normalize=0[a]")
        extra = []
    else:  # duck
        graph = (
            f"[0:a]asplit=2[key][orig];"
            f"[1:a]volume={vol:g}[m];"
            f"[m][key]sidechaincompress="
            f"threshold=0.05:ratio=10:attack=20:release=400[duck];"
            f"[orig][duck]amix=inputs=2:duration=first:normalize=0[a]")
        extra = []

    cmd = [ffmpeg_exe(), "-y", "-i", video,
           "-stream_loop", "-1", "-i", music,
           "-filter_complex", graph,
           "-map", "0:v:0", "-map", "[a]",
           "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
           *extra, out]
    log_fn(f"🎵 Applying music ({effective}, volume {vol:.2f}): "
           f"{os.path.basename(str(music))}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not os.path.exists(out):
        tail = (proc.stderr or "").strip().splitlines()[-8:]
        raise RuntimeError("ffmpeg music mux failed:\n" + "\n".join(tail))
    return out
