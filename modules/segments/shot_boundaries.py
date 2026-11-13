"""Where the shot changes are, to the frame.

``motion_scene_detect_optimized`` also finds cuts, but for a different job: it
looks at every few frames, and it is tuned to miss a cut rather than invent
one, because a false chapter boundary is worse than a coarse one. A shot list
for subtitling needs the opposite on both counts - the exact frame, and every
cut - so this is a separate pass rather than a setting on that one.

Every frame of the video is decoded at 48x27 (a size at which a feature film
decodes in a couple of minutes, and the size learned shot-boundary models take
as input) and each frame is compared with the one before it. Two numbers per
frame come out of that:

* ``step`` - how much the picture changed from the previous frame;
* ``across`` - how much it differs between two frames *before* and one frame
  *after*. A photographic flash or a single corrupt frame changes the picture
  twice and leaves it where it was, so ``step`` is large and ``across`` small.

A cut is then a frame whose ``step`` stands alone: the largest for a quarter
of a second either way, and several times the next largest in that
neighbourhood. That is a ratio against the footage around the cut and not a
level, which is why there is no sensitivity to set - a fixed level describes
how a video was graded, not where it was edited (``scene_cuts`` has the
measurements). Fast motion raises every frame around it and so does not pass;
a cut inside fast motion can be missed, and a dissolve or fade has no single
frame that stands alone and is not reported by this scorer at all.

That is the fallback. When the learned shot-boundary model is installed
(``modules/segments/shot_model.py``: TransNetV2, MIT, through OpenVINO) the
same 48x27 frames go to it instead, and it answers for each frame whether a
transition is happening there. It was trained on hand-marked cuts, so it
reports dissolves and fades as well, and it does not mistake a lighting change
for an edit. The decoding, the timing and the export are the same either way.

Times come from the file's own frame timestamps, not from ``index / fps``: on
variable-frame-rate footage the second drifts, and a shot list that is a
frame out is not a shot list.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

WIDTH, HEIGHT = 48, 27
FRAME_BYTES = WIDTH * HEIGHT * 3

# How far either side of a frame its step is compared with, in seconds. Two
# cuts closer together than this are a flicker, not two shots.
NEIGHBOURHOOD_SECONDS = 0.25

# A cut's step against the next largest step near it. Checked 2026-11-13 on a
# 20-minute animated episode (305 shots, 14.6 cuts a minute): of 28 cuts
# sampled evenly and looked at frame by frame, 26 were cuts and 2 were a
# lighting change and a camera jolt inside one shot. What it misses was not
# measured - there is no hand-made list for that episode to count against.
STAND_OUT_RATIO = 3.0

# Below this (on 0-1) nothing is a cut whatever the ratio: in a still scene the
# neighbourhood is sensor noise, and noise over noise is a large ratio too.
MIN_STEP = 0.02

# ...and above this it is one whatever the ratio, provided it is still the
# largest nearby: two different pictures, with a busy neighbourhood (a cut in
# the middle of an action sequence).
SURE_STEP = 0.25

# A flash returns to the picture it left: across/step below this is a flash.
FLASH_RATIO = 0.4


class ShotDetectionCancelled(RuntimeError):
    """The run was cancelled while the video was being read."""


@dataclass(frozen=True)
class ShotList:
    """The cuts of one video.

    ``cut_frames[i]`` is the index of the first frame of a new shot and
    ``cut_times[i]`` that frame's timestamp in seconds from the start of the
    video. The first shot starts at frame 0 and is not listed.
    """
    video_path: str
    fps_num: int
    fps_den: int
    frame_count: int
    duration: float
    cut_frames: tuple = ()
    cut_times: tuple = ()
    method: str = "frame-difference"
    exact_times: bool = True
    # ``(first, last)`` frame of every transition that takes more than one
    # frame (a dissolve, a fade, a wipe). Its cut is listed at its middle.
    gradual: tuple = ()
    notes: tuple = field(default_factory=tuple)

    @property
    def shots(self) -> list:
        """``[(start, end), ...]`` in seconds, covering the whole video."""
        edges = [0.0, *self.cut_times, float(self.duration)]
        return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]


def cuts_from_steps(step: Sequence[float], across: Sequence[float],
                    radius: int) -> np.ndarray:
    """Frame indices where a new shot starts.

    ``step[i]`` is frame ``i`` against ``i-1`` and ``across[i]`` is frame
    ``i+1`` against ``i-2``, both on 0-1. ``radius`` is the neighbourhood in
    frames.
    """
    step = np.asarray(step, dtype=np.float64)
    across = np.asarray(across, dtype=np.float64)
    n = step.size
    radius = max(2, int(radius))
    cuts = []
    for i in np.flatnonzero(step >= MIN_STEP):
        lo, hi = max(0, i - radius), min(n, i + radius + 1)
        window = step[lo:hi]
        if step[i] < window.max():
            continue
        # A tie is the same cut seen twice (a duplicated frame straddling
        # it); the first one is where the picture changed.
        if np.any(step[lo:i] == step[i]):
            continue
        others = np.delete(window, i - lo)
        rival = float(others.max()) if others.size else 0.0
        if step[i] < SURE_STEP and step[i] < STAND_OUT_RATIO * rival:
            continue
        if across[i] < FLASH_RATIO * step[i]:
            continue
        cuts.append(int(i))
    return np.array(cuts, dtype=int)


class _Differ:
    """``step`` and ``across`` from frames fed in order, a block at a time,
    holding only the last three frames between blocks."""

    def __init__(self):
        self._tail = np.zeros((0, HEIGHT, WIDTH, 3), dtype=np.int16)
        self.step: List[np.ndarray] = []
        self._across: List[np.ndarray] = []
        self.count = 0

    def feed(self, block: np.ndarray) -> None:
        frames = np.concatenate([self._tail, block.astype(np.int16)])
        k = self._tail.shape[0]
        # step for the new frames: each against the one before it.
        prev = frames[max(k - 1, 0):-1] if k else frames[:-1]
        cur = frames[k:] if k else frames[1:]
        d = np.abs(cur - prev).mean(axis=(1, 2, 3)) / 255.0
        self.step.append(d if k else np.concatenate([[0.0], d]))
        # across[i] needs frames i-2 and i+1; emitted for i as soon as i+1
        # exists, so it trails the frames by one.
        first = self.count - 1 if self.count else 0   # first i not yet emitted
        last = self.count + block.shape[0] - 2         # last i with i+1 known
        base = self.count - k                          # frame index of frames[0]
        values = []
        for i in range(first, last + 1):
            a, b = i - 2, i + 1
            if a < 0:
                values.append(1.0)
            else:
                values.append(float(
                    np.abs(frames[b - base] - frames[a - base]).mean() / 255.0))
        self._across.append(np.array(values))
        self.count += block.shape[0]
        self._tail = frames[-3:]

    def result(self):
        step = np.concatenate(self.step) if self.step else np.zeros(0)
        across = np.concatenate(self._across) if self._across else np.zeros(0)
        # The last frame has no frame after it: it cannot be a flash.
        if across.size < step.size:
            across = np.concatenate([across, np.ones(step.size - across.size)])
        return step, across


def _creation_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def read_frame_times(video_path: str) -> Optional[np.ndarray]:
    """Every video frame's timestamp in seconds, in display order, from the
    container (no decoding), or None when the file does not say."""
    from modules.media.ffmpeg_tools import ffprobe_exe
    exe = ffprobe_exe() or "ffprobe"
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "packet=pts_time", "-of", "csv=p=0", video_path],
            capture_output=True, text=True, timeout=600,
            creationflags=_creation_flags())
    except Exception as e:  # noqa: BLE001 - the caller falls back to the rate
        print(f"shot_boundaries: no frame times for {video_path}: "
              f"{type(e).__name__}: {e}")
        return None
    times = []
    for line in out.stdout.splitlines():
        token = line.strip().strip(",")
        try:
            times.append(float(token))
        except ValueError:
            continue
    if not times:
        return None
    times = np.sort(np.array(times, dtype=np.float64))
    return times - times[0]


def _probe(video_path: str):
    """``(fps_num, fps_den, duration)`` of the first video stream."""
    from modules.media.ffmpeg_tools import probe
    data = probe(video_path)
    stream = next(s for s in data.get("streams", [])
                  if s.get("codec_type") == "video")
    num, den = 0, 0
    for key in ("r_frame_rate", "avg_frame_rate"):
        a, _, b = str(stream.get(key) or "").partition("/")
        try:
            num, den = int(a), int(b or 1)
        except ValueError:
            continue
        if num > 0 and den > 0:
            break
    if num <= 0 or den <= 0:
        raise ValueError(f"{video_path}: no frame rate")
    duration = 0.0
    for holder in (data.get("format") or {}, stream):
        try:
            duration = float(holder.get("duration") or 0.0)
        except (TypeError, ValueError):
            duration = 0.0
        if duration > 0:
            break
    return num, den, duration


def detect_shots(video_path: str,
                 progress_fn: Optional[Callable] = None,
                 should_cancel: Optional[Callable[[], bool]] = None,
                 log_fn=print, block_frames: int = 500,
                 scorer: str = "auto") -> ShotList:
    """Decode ``video_path`` once and return its ``ShotList``.

    ``scorer`` is "auto" (the learned model when it is installed, otherwise
    frame differences), "learned" (raise when it is not) or "difference".
    """
    model = None
    if scorer in ("auto", "learned"):
        from modules.segments import shot_model
        model = shot_model.load(log_fn=log_fn)
        if model is None and scorer == "learned":
            raise RuntimeError("the shot-boundary model is not installed")
    fps_num, fps_den, duration = _probe(video_path)
    fps = fps_num / fps_den
    expected = max(1, int(round(duration * fps)))

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
           "-i", video_path, "-map", "0:v:0", "-an", "-sn",
           # Every frame the file holds, once: no frames dropped or doubled
           # to fit a constant rate.
           "-fps_mode", "passthrough",
           "-vf", f"scale={WIDTH}:{HEIGHT}:flags=area",
           "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            creationflags=_creation_flags())
    differ = _Differ()
    try:
        while True:
            if should_cancel and should_cancel():
                raise ShotDetectionCancelled("Shot detection cancelled")
            raw = proc.stdout.read(FRAME_BYTES * block_frames)
            usable = len(raw) - len(raw) % FRAME_BYTES
            if usable <= 0:
                break
            block = np.frombuffer(raw[:usable], dtype=np.uint8).reshape(
                -1, HEIGHT, WIDTH, 3)
            differ.feed(block)
            if model is not None:
                model.feed(block)
            if progress_fn:
                progress_fn(min(95, int(95 * differ.count / expected)), 100,
                            "Shots", f"frame {differ.count}")
    finally:
        if proc.poll() is None:
            proc.kill()
        err = proc.stderr.read().decode("utf-8", "replace").strip()
        proc.wait()
    if differ.count == 0:
        raise RuntimeError(f"no frames decoded from {video_path}: {err[-300:]}")

    gradual = ()
    method = "frame-difference"
    if model is not None:
        from modules.segments import shot_model
        cuts, gradual = shot_model.cuts_from_scores(model.result())
        method = shot_model.METHOD
    else:
        step, across = differ.result()
        radius = int(round(NEIGHBOURHOOD_SECONDS * fps))
        cuts = cuts_from_steps(step, across, radius)

    notes = []
    times = read_frame_times(video_path)
    exact = times is not None and abs(times.size - differ.count) <= 1
    if exact:
        cut_times = [float(times[min(i, times.size - 1)]) for i in cuts]
    else:
        cut_times = [float(i) / fps for i in cuts]
        notes.append("frame timestamps unavailable; times are frame / rate")
        print(f"shot_boundaries: {video_path}: decoded {differ.count} frames, "
              f"container lists {0 if times is None else times.size}")
    if duration <= 0:
        duration = differ.count / fps

    minutes = max(duration / 60.0, 1e-6)
    log_fn(f"🎬 Shots: {len(cuts) + 1} ({len(cuts) / minutes:.1f} cuts/min) "
           f"from {differ.count} frames, by {method}"
           + (f"; {len(gradual)} of the cuts are dissolves or fades"
              if gradual else ""))
    if progress_fn:
        progress_fn(100, 100, "Shots", f"{len(cuts) + 1} shots")
    return ShotList(video_path=video_path, fps_num=fps_num, fps_den=fps_den,
                    frame_count=differ.count, duration=float(duration),
                    cut_frames=tuple(int(i) for i in cuts),
                    cut_times=tuple(cut_times), exact_times=exact,
                    notes=tuple(notes), method=method,
                    gradual=tuple((int(a), int(b)) for a, b in gradual))


def to_dict(shots: ShotList) -> dict:
    """Cache / JSON shape."""
    return {"fps": [shots.fps_num, shots.fps_den],
            "frame_count": shots.frame_count, "duration": shots.duration,
            "cut_frames": list(shots.cut_frames),
            "cut_times": [round(t, 4) for t in shots.cut_times],
            "method": shots.method, "exact_times": shots.exact_times,
            "gradual": [list(g) for g in shots.gradual]}


def main(argv=None) -> int:
    """``python -m modules.segments.shot_boundaries VIDEO [--edl] [--txt]``"""
    import argparse
    from modules.segments import shot_list_export as export
    parser = argparse.ArgumentParser(description="List a video's shot changes.")
    parser.add_argument("video")
    parser.add_argument("--edl", action="store_true", help="write VIDEO_shots.edl")
    parser.add_argument("--txt", action="store_true", help="write VIDEO_shots.txt")
    parser.add_argument("--json", action="store_true", help="print JSON")
    parser.add_argument("--scorer", default="auto",
                        choices=("auto", "learned", "difference"))
    args = parser.parse_args(argv)
    shots = detect_shots(args.video, scorer=args.scorer)
    if args.edl:
        print(export.write_edl(shots))
    if args.txt:
        print(export.write_text(shots))
    if args.json or not (args.edl or args.txt):
        print(json.dumps(to_dict(shots)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
