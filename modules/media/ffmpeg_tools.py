"""
ffmpeg_tools.py — run on the ffmpeg pip already installed, and probe without ffprobe.

Nobody should have to install ffmpeg to use this app. ``imageio-ffmpeg`` is a
requirement already and carries an ffmpeg binary for Windows, Linux and macOS,
and the frozen builds bundle that binary. Two things kept it from counting:

1. Its file is named for its version (``ffmpeg-win-x86_64-v7.1.exe``), so every
   place that runs a bare ``"ffmpeg"`` — ours, and Whisper's audio loader, which
   is not ours to change — never found it. ``ensure_ffmpeg_on_path`` gives it
   the plain name in a private directory and puts that directory on this
   process's PATH, which child processes inherit.
2. It ships no ffprobe. ``probe`` answers in ffprobe's own JSON shape: from
   ffprobe when there is one, from PyAV (also a requirement already) when there
   is not. Callers parse one format either way.

A system ffmpeg or ffprobe on PATH always wins; nothing here overrides it.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

_EXE = ".exe" if sys.platform == "win32" else ""

# What a bare "ffmpeg" resolved to after the last successful ensure.
_ensured: str | None = None


def _bin_dir() -> str:
    """Per-user directory the bundled ffmpeg is staged into under its plain name.

    Not beside the app: an install folder can refuse writes, and from source
    that folder is the repo.
    """
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "VideoHighlighter", "ffmpeg-bin")


def _bundled_ffmpeg() -> str | None:
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    return exe if exe and os.path.isfile(exe) else None


def _same_file(a: str, b: str) -> bool:
    """Size and whole-second mtime: equal for a hard link, and for a copy2."""
    try:
        sa, sb = os.stat(a), os.stat(b)
    except OSError:
        return False
    return sa.st_size == sb.st_size and int(sa.st_mtime) == int(sb.st_mtime)


def _stage(src: str, dst: str) -> None:
    """Put ``src`` at ``dst``: a hard link when the volume allows, else a copy.

    Written under a temporary name and renamed into place, so a second process
    starting at the same moment never runs a half-copied binary.
    """
    if _same_file(src, dst):
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = f"{dst}.{os.getpid()}.tmp"
    try:
        try:
            os.link(src, tmp)
        except OSError:
            shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def ensure_ffmpeg_on_path(log_fn=print) -> str | None:
    """Make a bare ``"ffmpeg"`` resolve in this process and its children.

    Returns what it resolves to, or None when there is no ffmpeg at all — the
    only case reported through ``log_fn``, since it is the only one the user
    has to act on. Cheap to call repeatedly.
    """
    global _ensured
    found = shutil.which("ffmpeg")
    if found:
        if found != _ensured:
            _ensured = found
        return found

    src = _bundled_ffmpeg()
    if not src:
        log_fn("❌ No ffmpeg found. It normally comes with the app's requirements "
               "(imageio-ffmpeg) — reinstall them, or put ffmpeg on PATH.")
        return None

    bin_dir = _bin_dir()
    dst = os.path.join(bin_dir, "ffmpeg" + _EXE)
    try:
        _stage(src, dst)
    except OSError as e:
        # An older staged copy still runs (e.g. held open by another instance
        # while imageio-ffmpeg was upgraded); only a missing one is a failure.
        if not os.path.isfile(dst):
            print(f"⚠️ [ffmpeg_tools] could not stage {src} as {dst}: {e}")
            return None
        print(f"⚠️ [ffmpeg_tools] kept the previously staged ffmpeg ({e})")

    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    print(f"[ffmpeg_tools] no ffmpeg on PATH; using the bundled one as {dst}")
    _ensured = dst
    return dst


def ffprobe_exe() -> str | None:
    """A real ffprobe on PATH, or None (imageio-ffmpeg does not ship one)."""
    return shutil.which("ffprobe")


def probe(path, timeout: float = 30) -> dict:
    """``{"streams": [...], "format": {...}}`` for ``path``, shaped like
    ``ffprobe -print_format json -show_streams -show_format``.

    From ffprobe when it is installed, otherwise from PyAV, which fills the
    keys this app reads: per stream ``index``, ``codec_type``, ``codec_name``,
    ``duration``, ``bit_rate``, ``tags``; for video ``width``, ``height``,
    ``pix_fmt``, ``r_frame_rate``, ``avg_frame_rate`` and a Display Matrix
    ``side_data_list`` entry when the clip is rotated; for audio
    ``sample_rate`` and ``channels``; and ``duration``, ``size``, ``bit_rate``,
    ``format_name`` and ``tags`` on the format. Numbers that ffprobe prints as
    strings are strings here too.

    A file that cannot be read raises (CalledProcessError from ffprobe, PyAV's
    own error otherwise) — whether that is fatal is the caller's decision.
    """
    exe = ffprobe_exe()
    if exe:
        out = subprocess.run(
            [exe, "-v", "error", "-print_format", "json",
             "-show_streams", "-show_format", str(path)],
            capture_output=True, encoding="utf-8", errors="replace",
            check=True, timeout=timeout,
        ).stdout
        return json.loads(out or "{}")
    return _probe_with_pyav(str(path))


def _rate(value) -> str:
    return f"{value.numerator}/{value.denominator}" if value else "0/0"


def _probe_with_pyav(path: str) -> dict:
    import av

    with av.open(path) as container:
        streams = []
        video_entry = None
        for s in container.streams:
            cc = getattr(s, "codec_context", None)
            entry = {"index": s.index, "codec_type": s.type,
                     "codec_name": getattr(cc, "name", None),
                     "tags": dict(s.metadata)}
            if s.duration is not None and s.time_base:
                entry["duration"] = f"{float(s.duration * s.time_base):.6f}"
            if getattr(cc, "bit_rate", None):
                entry["bit_rate"] = str(cc.bit_rate)
            if s.type == "video":
                entry.update(
                    width=cc.width, height=cc.height, pix_fmt=cc.pix_fmt or "",
                    r_frame_rate=_rate(s.base_rate or s.guessed_rate),
                    avg_frame_rate=_rate(s.average_rate),
                )
                if video_entry is None:
                    video_entry = entry
            elif s.type == "audio":
                layout = getattr(cc, "layout", None)
                entry.update(sample_rate=str(cc.sample_rate),
                             channels=getattr(layout, "nb_channels", None)
                             or getattr(cc, "channels", 0))
            streams.append(entry)

        fmt = {"filename": path, "format_name": container.format.name,
               "nb_streams": len(streams), "size": str(os.path.getsize(path)),
               "tags": dict(container.metadata)}
        if container.duration is not None:
            fmt["duration"] = f"{container.duration / av.time_base:.6f}"
        if container.bit_rate:
            fmt["bit_rate"] = str(container.bit_rate)

        # Rotation lives in the display matrix, which PyAV exposes per decoded
        # frame rather than per stream. Same counter-clockwise sign ffprobe
        # prints. An orientation we cannot read must never fail the probe.
        if video_entry is not None:
            try:
                video = container.streams.video[0]
                rotation = getattr(next(container.decode(video)), "rotation", 0)
                if rotation:
                    video_entry["side_data_list"] = [
                        {"side_data_type": "Display Matrix", "rotation": rotation}]
            except Exception:
                pass

    return {"streams": streams, "format": fmt}
