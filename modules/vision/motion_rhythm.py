"""How much the picture moves each second, and how *regularly* it moves.

The analysis cache already holds a motion curve, but it is binned at half a
second: it can say that something moved, not how. That is the wrong question
for telling a sustained activity from the stretches between activities.
Repositioning, walking into frame or a camera move is plenty of motion with no
beat; a sustained activity repeats at roughly one to a few cycles a second.
The difference is rhythm, and rhythm needs a sample rate well above the beat.

So this decodes the video once, small and grey, at its own frame rate, and
reduces each frame pair to the mean optical-flow magnitude. Per second:

``level``
    Mean flow, in pixels per frame at ``WIDTH`` x ``HEIGHT``. Camera moves
    count: nothing separates the subject from the camera, so a handheld pan
    scores like a movement in frame.
``rhythm``
    0-1: the highest *peak* of the normalised autocorrelation of the
    high-passed flow over a ``WINDOW`` second window centred on the second, at
    lags of one cycle of ``MIN_HZ``-``MAX_HZ``. Near 1 the motion repeats on a
    steady beat; near 0 it does not, or there is too little of it
    (``MIN_LEVEL``). The film this was measured on had its median at 0.35.
``beat_hz``
    How often the flow repeats, or 0 where it does not. Flow speed peaks on the
    way out and on the way back, so this can read twice the rate of the
    movement itself (4.2 for a 2.1 Hz movement); ``rhythm`` is the number to
    threshold.

Measured 2026-11-19 on one 48-minute film, against moments checked by eye:
sustained activity 0.6-0.96 (beat 2.1 Hz, which matched a separate per-face
track), walking to a new place 0.0-0.27, being set down between two positions
0.30 falling to 0.0.

Three ways the naive version fails, each of which produced a confident wrong
beat before it was fixed:

* **Resampling.** Decoding 25 fps at 15 drops frames in a fixed pattern, and
  the pattern is a beat (5 Hz). Decode at the native rate.
* **Codec structure.** Frame *differences* pulse with the encoder's frame
  types (a P frame every 4 here, 6.25 Hz) and alias to a false 1 Hz. Optical
  flow is far less sensitive to that than a pixel difference, and ``MAX_HZ``
  stays below the usual P-frame rates.
* **Smoothness.** Any slowly varying signal correlates with itself at short
  lags, so "the largest autocorrelation" always picks the shortest lag. Only a
  local *peak* after high-passing is a beat.

None of this knows what is moving or why. A rule supplies the meaning.

Cost: one low-resolution decode plus flow, about 2.5 minutes for a 48-minute
film on CPU, cached beside the analysis cache and keyed on the video's size
and mtime, so applying rules again is free.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Callable, Iterable, Optional

import numpy as np

WIDTH, HEIGHT = 160, 90
WINDOW = 4.0
MIN_HZ, MAX_HZ = 0.8, 5.0
# Below this mean flow the frame is effectively still, and any autocorrelation
# is the codec's noise.
MIN_LEVEL = 0.05
CACHE_VERSION = 2


def _video_key(video_path: str) -> str:
    try:
        st = os.stat(video_path)
        return f"{st.st_size}:{int(st.st_mtime)}:{CACHE_VERSION}"
    except OSError:
        return f"missing:{CACHE_VERSION}"


def cache_path_for(video_path: str, cache_dir: str = "./cache") -> str:
    digest = hashlib.sha256(
        os.path.abspath(video_path).encode("utf-8", "replace")).hexdigest()[:16]
    return os.path.join(cache_dir, f"{digest}.motion_rhythm.json")


def flow_magnitudes(frames: Iterable[np.ndarray]) -> np.ndarray:
    """Mean optical-flow magnitude into each grey frame (0 for the first)."""
    import cv2

    out = []
    prev = None
    for g in frames:
        if prev is None:
            out.append(0.0)
        else:
            flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 2, 9, 2, 5, 1.1, 0)
            out.append(float(np.linalg.norm(flow, axis=2).mean()))
        prev = g
    return np.asarray(out, dtype=np.float64)


def _high_pass(x: np.ndarray, fps: float) -> np.ndarray:
    n = max(1, int(round(fps)))
    return x - np.convolve(x, np.ones(n) / n, mode="same")


def _rhythm_at(window: np.ndarray, fps: float) -> tuple:
    """``(peak autocorrelation, beat in Hz)``; ``(0, 0)`` where there is no peak."""
    x = window - window.mean()
    denom = float((x * x).sum())
    lo = max(1, int(fps / MAX_HZ))
    hi = min(len(x) - 2, int(fps / MIN_HZ) + 1)
    if denom <= 1e-12 or hi - lo < 2:
        return 0.0, 0.0
    lags = np.arange(lo, hi + 1)
    r = np.array([float((x[:-lag] * x[lag:]).sum()) / denom * len(x) / (len(x) - lag)
                  for lag in lags])
    peaks = [(float(r[i]), int(lags[i])) for i in range(1, len(r) - 1)
             if r[i] > r[i - 1] and r[i] >= r[i + 1] and r[i] > 0]
    if not peaks:
        return 0.0, 0.0
    best = max(p[0] for p in peaks)
    # A beat also peaks at two and three cycles, sometimes a shade higher when
    # the period falls between frames; the rate is the first of those, not the
    # tallest.
    lag = min(p[1] for p in peaks if p[0] >= 0.85 * best)
    return min(1.0, best), fps / float(lag)


def per_second(flow: np.ndarray, fps: float, window: float = WINDOW) -> dict:
    """``{"level": [...], "rhythm": [...], "beat_hz": [...]}``, one per second."""
    flow = np.asarray(flow, dtype=np.float64)
    hp = _high_pass(flow, fps)
    n_sec = int(len(flow) // fps)
    half = int(round(window * fps / 2))
    level, rhythm, beat = [], [], []
    for s in range(n_sec):
        a, b = int(round(s * fps)), int(round((s + 1) * fps))
        lv = float(flow[a:b].mean()) if b > a else 0.0
        level.append(round(lv, 3))
        c = (a + b) // 2
        w = hp[max(0, c - half): min(len(hp), c + half)]
        if lv < MIN_LEVEL or len(w) < fps:
            rhythm.append(0.0)
            beat.append(0.0)
            continue
        r, hz = _rhythm_at(w, fps)
        rhythm.append(round(r, 3))
        beat.append(round(hz, 2))
    return {"level": level, "rhythm": rhythm, "beat_hz": beat}


def _frame_rate(video_path: str) -> float:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=avg_frame_rate,r_frame_rate", "-of", "json",
         video_path], capture_output=True, text=True, creationflags=flags).stdout
    stream = (json.loads(out or "{}").get("streams") or [{}])[0]
    for key in ("avg_frame_rate", "r_frame_rate"):
        num, _, den = str(stream.get(key) or "0/0").partition("/")
        try:
            fps = float(num) / float(den or 1)
        except (ValueError, ZeroDivisionError):
            continue
        if 1.0 <= fps <= 240.0:
            return fps
    return 0.0


def _decode(video_path: str, cancel=None, progress: Optional[Callable] = None,
            fps: float = 25.0):
    cmd = ["ffmpeg", "-v", "error", "-i", video_path,
           "-vf", f"scale={WIDTH}:{HEIGHT}",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    size = WIDTH * HEIGHT
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, creationflags=flags)
    try:
        i = 0
        step = max(1, int(fps * 60))
        while True:
            if cancel is not None and cancel():
                return
            buf = proc.stdout.read(size)
            if len(buf) < size:
                return
            yield np.frombuffer(buf, np.uint8).reshape(HEIGHT, WIDTH)
            i += 1
            if progress is not None and i % step == 0:
                progress(i / fps)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def analyse(video_path: str, *, cache_dir: str = "./cache",
            progress: Optional[Callable] = None, cancel=None,
            log_fn=print) -> dict:
    """Per-second curves for ``video_path``, from the cache when it is current.

    ``{}`` when the video cannot be measured: a rule set that never mentions
    motion should not lose its run to this measurement.
    """
    path = cache_path_for(video_path, cache_dir)
    key = _video_key(video_path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cached = json.load(fh)
        if cached.get("key") == key:
            return cached.get("curves") or {}
    except (OSError, ValueError):
        pass
    try:
        fps = _frame_rate(video_path)
        if not fps:
            log_fn("⚠️ Motion rhythm: could not read the frame rate.")
            return {}
        flow = flow_magnitudes(_decode(video_path, cancel, progress, fps))
    except Exception as exc:
        log_fn(f"⚠️ Motion rhythm unavailable: {exc}")
        return {}
    if (cancel is not None and cancel()) or len(flow) < fps * 2:
        return {}
    curves = per_second(flow, fps)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"key": key, "curves": curves}, fh)
    except OSError:
        pass
    return curves
