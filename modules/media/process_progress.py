"""Run a download tool and show its progress; stop it when it stalls.

yt-dlp and ffmpeg each report progress in their own way. Both are run the same
way here: every output line goes to a parser that logs progress about once a
second, and a watcher kills the whole process tree (the tool and any ffmpeg it
started) on a user's cancel or when no output has arrived for STALL_SECONDS.

No size or time limit: an hour-long stream downloads as long as it moves.
Kept free of the downloader's heavy imports so it can be tested on its own.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Callable, Dict, Optional


def _flag_is_cancelled(flag) -> bool:
    """True if a cancel flag has been tripped. Accepts either a threading.Event
    (`is_set()`) or a worker object exposing `is_cancelled()`. None => never."""
    if flag is None:
        return False
    try:
        if hasattr(flag, "is_cancelled") and flag.is_cancelled():
            return True
        if hasattr(flag, "is_set") and flag.is_set():
            return True
    except Exception:
        pass
    return False


def _terminate_tree(proc) -> None:
    """Kill a subprocess *and its children*. yt-dlp spawns ffmpeg (for
    --download-sections, HLS, and muxing); ffmpeg inherits our stdout pipe, so
    killing only yt-dlp leaves the pipe open and hangs the progress reader. On
    Windows we use `taskkill /T` to take down the whole tree."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True, check=False,
            )
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


# A download is stopped when it stops moving, never for being long: a fixed
# cap of a few minutes killed every hour-long stream part-way. yt-dlp and
# ffmpeg both report progress at least once a second while data flows, so
# this much silence means the source has stalled.
STALL_SECONDS = 180

# Appended to every yt-dlp download command: one machine-readable line per
# progress update, which _ytdlp_progress turns into the speed / ETA log line.
YTDLP_PROGRESS_ARGS = [
    "--newline",
    "--progress-template",
    "DLP|%(progress._percent_str)s|%(progress._speed_str)s|"
    "%(progress._eta_str)s|%(progress._total_bytes_str)s",
]


def _stream_process(cmd, on_line: Callable, cancel_flag=None,
                    idle_timeout: Optional[float] = None,
                    timeout: Optional[float] = None):
    """Run ``cmd``, hand each output line to ``on_line``, keep the lines it
    does not consume (it returns True for a line it used).

    The process and its children are killed when ``cancel_flag`` trips, when
    no line has arrived for ``idle_timeout`` seconds, or after ``timeout``
    seconds in all. Returns a CompletedProcess with ``.cancelled`` and
    ``.stalled`` set, so a caller can tell a user's cancel from a dead source.
    """
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
    )
    started = time.time()
    last_line = [started]
    cancelled = threading.Event()
    stalled = threading.Event()

    def _watch():
        while proc.poll() is None:
            now = time.time()
            if _flag_is_cancelled(cancel_flag):
                cancelled.set()
            elif idle_timeout and now - last_line[0] > idle_timeout:
                stalled.set()
            elif timeout and now - started > timeout:
                stalled.set()
            if cancelled.is_set() or stalled.is_set():
                _terminate_tree(proc)
                return
            time.sleep(0.3)

    watcher = threading.Thread(target=_watch, daemon=True)
    watcher.start()
    lines = []
    try:
        for line in proc.stdout:
            last_line[0] = time.time()
            if not on_line(line.strip()):
                lines.append(line)
    finally:
        proc.wait()
        watcher.join(timeout=1)
    result = subprocess.CompletedProcess(cmd, proc.returncode, stdout="".join(lines), stderr="")
    result.cancelled = cancelled.is_set()
    result.stalled = stalled.is_set()
    return result


def _throttled(log_fn: Callable, every: float = 1.0) -> Callable:
    """``log_fn`` that drops calls made less than ``every`` seconds apart."""
    last = [0.0]

    def emit(text: str) -> None:
        now = time.time()
        if now - last[0] >= every:
            last[0] = now
            log_fn(text)
    return emit


def _ytdlp_progress(log_fn: Callable) -> Callable:
    """Line handler for a yt-dlp run with YTDLP_PROGRESS_ARGS."""
    emit = _throttled(log_fn)

    def on_line(s: str) -> bool:
        if not s.startswith("DLP|"):
            return False
        parts = [p.strip() for p in s.split("|")]
        if len(parts) >= 4:
            pct, speed, eta = parts[1], parts[2], parts[3]
            size = parts[4] if len(parts) >= 5 else ""
            tail = f"  •  {size}" if size and size not in ("NA", "") else ""
            emit(f"⬇️ {pct}  •  {speed}  •  ETA {eta}{tail}")
        return True
    return on_line


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _ffmpeg_progress(log_fn: Callable, total_seconds: Optional[float] = None) -> Callable:
    """Line handler for ffmpeg run with ``-progress pipe:1 -nostats``: its
    key=value block ends in ``progress=continue`` once a second."""
    emit = _throttled(log_fn)
    state: Dict[str, str] = {}

    def on_line(s: str) -> bool:
        key, sep, value = s.partition("=")
        if not sep or not key or " " in key:
            return False
        state[key] = value
        if key == "progress":
            done = int(state.get("out_time_us", "0") or 0) / 1_000_000
            size = int(state.get("total_size", "0") or 0) / (1024 * 1024)
            where = (f"{_clock(done)} of {_clock(total_seconds)} "
                     f"({done * 100 / total_seconds:.0f}%)"
                     if total_seconds else f"{_clock(done)} so far")
            emit(f"⬇️ {where}  •  {size:.1f} MB  •  {state.get('speed', '?')}")
        return True
    return on_line


def _run_ytdlp_with_progress(cmd, timeout=None, log_fn: Callable = print, cancel_flag=None,
                             idle_timeout: Optional[float] = STALL_SECONDS):
    """Run yt-dlp (with YTDLP_PROGRESS_ARGS in ``cmd``) and log speed / ETA
    live, about once a second. stdout holds everything but the progress
    lines, so the callers' parsing of "Destination:" and errors still works."""
    return _stream_process(cmd, _ytdlp_progress(log_fn), cancel_flag,
                           idle_timeout=idle_timeout, timeout=timeout)


def _run_ffmpeg_with_progress(cmd, log_fn: Callable = print, cancel_flag=None,
                              total_seconds: Optional[float] = None,
                              idle_timeout: Optional[float] = STALL_SECONDS):
    """Run ffmpeg with machine-readable progress on stdout, logged live."""
    cmd = list(cmd)
    cmd[1:1] = ["-hide_banner", "-nostats", "-loglevel", "error", "-progress", "pipe:1"]
    return _stream_process(cmd, _ffmpeg_progress(log_fn, total_seconds), cancel_flag,
                           idle_timeout=idle_timeout)


def _report_stop(result, log_fn: Callable, what: str) -> None:
    """Say why a run ended early, if it did."""
    if getattr(result, "cancelled", False):
        log_fn("⏹️ Download cancelled by user")
    elif getattr(result, "stalled", False):
        log_fn(f"⚠ {what}: no progress for {STALL_SECONDS // 60} minutes — stopped")
