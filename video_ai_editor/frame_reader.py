"""Exact frames for thumbnails, read forwards where that is cheaper than seeking.

A seek to a frame decodes from the keyframe before it up to the frame itself.
On a film with a keyframe every eight seconds that is 125 frames on average,
about 130 ms, and it is paid again in full for the next thumbnail even when
that one is a second further on: a zoomed-in filmstrip of thirty slots decoded
3750 frames to show 900 frames' worth of video.

The capture is already standing just before the next frame wanted, so reading
on to it costs only the frames in between - measured at about 1.2 ms each with
`grab`, which decodes without making a picture. A slot one second on is then
some 35 ms instead of 130.

So this keeps one capture per video and, for each frame asked for, takes the
cheaper of the two ways there. Which is cheaper depends on the file (how far
apart its keyframes are, how large its frames), so both costs are measured as
they are paid rather than assumed.

One capture, one frame at a time. Thumbnail workers running several seeks side
by side gained nothing - the decoder underneath already uses every core, so
three at once took three times as long each - and reading forwards only works
if the frames are asked for in order. Waiters are therefore let in by priority
first and by time second, which is the order a filmstrip paints in.
"""
from __future__ import annotations

import heapq
import itertools
import threading
import time
from typing import Optional

import cv2

# Until a file's own costs are known: what a seek and one skipped frame cost.
# Deliberately modest about the seek, so the first frames do not read forward
# across a stretch a seek would have crossed faster.
ASSUMED_SEEK_MS = 60.0
ASSUMED_SKIP_MS = 1.3

# How much of each new measurement to believe.
LEARN = 0.3


class _Turns:
    """One at a time: most urgent first, then earliest in the video."""

    def __init__(self):
        self._condition = threading.Condition()
        self._waiting: list = []
        self._busy = False
        self._arrival = itertools.count()

    def enter(self, priority: int, time_ms: float) -> None:
        with self._condition:
            ticket = (priority, time_ms, next(self._arrival))
            heapq.heappush(self._waiting, ticket)
            while self._busy or self._waiting[0] != ticket:
                self._condition.wait()
            heapq.heappop(self._waiting)
            self._busy = True

    def leave(self) -> None:
        with self._condition:
            self._busy = False
            self._condition.notify_all()


class FrameReader:
    """The frame at a given time of one video, as a BGR array. Thread-safe."""

    def __init__(self, video_path: str, opencv=None):
        self.video_path = str(video_path)
        # The caller's OpenCV where it has one of its own (the thumbnail
        # cache does, and tests swap it), else the one imported here.
        self._cv2 = opencv if opencv is not None else cv2
        self._turns = _Turns()
        self._cap = None
        self._fps = 0.0
        self._at_ms: Optional[float] = None   # time of the frame last read
        self._last = None                     # that frame
        self.seek_ms = ASSUMED_SEEK_MS
        self.skip_ms = ASSUMED_SKIP_MS
        self.seeks = 0                        # how each frame was reached,
        self.read_on = 0                      # for the log and the tests
        self._closed = False

    def frame_at(self, time_ms: float, priority: int = 0):
        """The frame at `time_ms`, or None if it cannot be read."""
        if self._closed:
            return None
        self._turns.enter(priority, float(time_ms))
        try:
            if self._closed or not self._open():
                return None
            ahead = self._frames_ahead(time_ms)
            if ahead is not None and ahead * self.skip_ms <= self.seek_ms:
                frame = self._read_on(ahead)
                if frame is not None:
                    return frame
            return self._seek(time_ms)
        finally:
            self._turns.leave()

    def close(self) -> None:
        self._closed = True
        self._turns.enter(-(10 ** 9), 0.0)      # after whoever is mid-frame
        try:
            if self._cap is not None:
                self._cap.release()
            self._cap, self._last, self._at_ms = None, None, None
        finally:
            self._turns.leave()

    # ── internals ──

    def _open(self) -> bool:
        if self._cap is not None and self._cap.isOpened():
            return True
        self._cap = self._cv2.VideoCapture(self.video_path)
        if not self._cap.isOpened():
            self._cap = None
            return False
        self._fps = float(self._cap.get(self._cv2.CAP_PROP_FPS) or 0.0)
        self._at_ms, self._last = None, None
        return True

    def _frames_ahead(self, time_ms: float) -> Optional[int]:
        """How many frames on from the last one read `time_ms` is; None when
        it is behind, or there is nothing to count from."""
        if self._at_ms is None or self._fps <= 0:
            return None
        ahead = int(round((time_ms - self._at_ms) * self._fps / 1000.0))
        return ahead if ahead >= 0 else None

    def _read_on(self, frames: int):
        if frames == 0:
            return self._last
        started = time.perf_counter()
        for _ in range(frames):
            if not self._cap.grab():
                self._at_ms, self._last = None, None
                return None
        skipped = time.perf_counter() - started
        ok, frame = self._cap.retrieve()
        if not ok or frame is None:
            self._at_ms, self._last = None, None
            return None
        self.skip_ms += LEARN * (skipped * 1000.0 / frames - self.skip_ms)
        self._at_ms += frames * 1000.0 / self._fps
        self._last = frame
        self.read_on += 1
        return frame

    def _seek(self, time_ms: float):
        started = time.perf_counter()
        self._cap.set(self._cv2.CAP_PROP_POS_MSEC, time_ms)
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._at_ms, self._last = None, None
            return None
        took = (time.perf_counter() - started) * 1000.0
        # The first seek also pays for opening the decoder; do not learn it.
        if self.seeks:
            self.seek_ms += LEARN * (took - self.seek_ms)
        self.seeks += 1
        self._at_ms, self._last = float(time_ms), frame
        return frame
