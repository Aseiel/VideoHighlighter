"""Where frames come from: a recording on disk, a capture device, or OBS itself.

Each source hands out ``(t, frame_bgr)`` with ``t`` in seconds. Every source is
built to keep the CPU out of it as far as it can, since that is the point of
putting the detector on the NPU:

- ``VideoFileSource`` decodes with the GPU's video engine (D3D11 / VA-API via
  OpenCV's FFmpeg backend) unless told not to. Scanning a 48-minute 1440p60
  H.264 recording that way took 1.6 cores against 3.5 for software decoding,
  in the same time.
- ``ObsScreenshotSource`` asks OBS for the program output already scaled down
  on the GPU, so only a small JPEG ever reaches this process.
- ``CaptureSource`` reads a capture device such as the OBS Virtual Camera; a
  reader thread keeps only the newest frame, so a slow sampler never sees a
  stale, buffered one.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Iterator


def prefetch(items, depth: int = 4) -> Iterator:
    """Iterate ``items`` on a worker thread, ``depth`` ahead, so decoding the
    next frame overlaps detecting this one. Errors surface in the consumer.

    Close the generator before releasing whatever ``items`` reads from: that is
    what stops the worker, and a capture released under a decoding thread can
    take the process down with it."""
    import queue
    q: "queue.Queue" = queue.Queue(maxsize=depth)
    stop = threading.Event()
    done = object()

    def put(item) -> bool:
        while not stop.is_set():
            try:
                q.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def work() -> None:
        try:
            for item in items:
                if not put(item):
                    return
            put(done)
        except BaseException as e:  # noqa: BLE001 - re-raised in the consumer
            put(e)

    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    try:
        while True:
            item = q.get()
            if item is done:
                return
            if isinstance(item, BaseException):
                raise item
            yield item
    finally:
        stop.set()
        worker.join(timeout=5)


class VideoFileSource:
    """The first frame at or after every ``every`` seconds of a video file,
    decoded in order (no seeking between samples: seeking re-decodes from the
    previous keyframe and was 5x slower here).

    Times are the frames' own timestamps, not ``index / fps``: OBS records at a
    variable frame rate (one 60 fps recording here averaged 29 fps, with frame
    gaps from 17 to 50 ms), and counting frames drifts from the real clock.
    """

    def __init__(self, path: str, every: float = 0.5, hw_decode: bool = True,
                 start: float = 0.0, end: float | None = None):
        import cv2  # lazy
        self.path = path
        params = ([cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY]
                  if hw_decode else [])
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, params)
        if hw_decode and not cap.isOpened():
            cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {path}")
        self._cap = cap
        self.hw_decode = bool(cap.get(cv2.CAP_PROP_HW_ACCELERATION) or 0)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0     # average, for display
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration = self.frame_count / self.fps if self.frame_count else 0.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.every = float(every)
        self.start, self.end = max(0.0, start), end
        stop = min(end, self.duration) if end else self.duration
        self.samples = max(0, math.ceil(round((stop - self.start) / self.every, 6)))  # progress

    def __iter__(self) -> Iterator[tuple[float, object]]:
        import cv2  # lazy
        cap = self._cap
        if self.start > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, self.start * 1000)
        due = self.start
        eps = 1e-3
        while True:
            if not cap.grab():
                return
            t = (cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000
            if t + eps < due:
                continue
            if self.end is not None and t > self.end + eps:
                return
            ok, frame = cap.retrieve()
            if not ok:
                return
            yield t, frame
            while due <= t + eps:    # after a gap, skip the slots it swallowed
                due += self.every

    def close(self) -> None:
        self._cap.release()


class CaptureSource:
    """A live capture device (index, e.g. the OBS Virtual Camera) or stream URL."""

    def __init__(self, device: int | str, width: int = 1280, height: int = 720):
        import cv2  # lazy
        if isinstance(device, int):
            # DirectShow lists the OBS Virtual Camera and opens it quickly;
            # Media Foundation (OpenCV's default on Windows) takes seconds.
            backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
            cap = cv2.VideoCapture(device, backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        else:
            cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open capture device {device!r}")
        self._cap = cap
        self._latest = None
        self._error: str | None = None
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        self._t0 = time.monotonic()

    def _reader(self) -> None:
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            with self._cond:
                if not ok:
                    self._error = "capture device stopped delivering frames"
                    self._cond.notify_all()
                    return
                self._latest = frame
                self._cond.notify_all()

    def read(self, timeout: float = 5.0) -> tuple[float, object]:
        """The newest frame, waiting for the first one. Raises when the
        device stops."""
        with self._cond:
            ok = self._cond.wait_for(lambda: self._latest is not None or self._error,
                                     timeout)
            if self._latest is None:  # a frame still waiting beats the error
                if self._error:
                    raise RuntimeError(self._error)
                if not ok:
                    raise TimeoutError("No frame from the capture device")
            frame, self._latest = self._latest, None
        return time.monotonic() - self._t0, frame

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._cap.release()


class ObsScreenshotSource:
    """Frames from a running OBS via obs-websocket ``GetSourceScreenshot``.

    ``source`` is a scene or source name; empty means "whatever scene is on
    program", looked up again whenever a screenshot fails (the user switched
    or renamed it).
    """

    def __init__(self, client, source: str = "", width: int = 640, quality: int = 80):
        self.client = client
        self.fixed_source = source
        self.source = source or client.program_scene()
        self.width = width
        self.quality = quality
        self._t0 = time.monotonic()

    def read(self) -> tuple[float, object]:
        import cv2  # lazy
        import numpy as np
        from npu_obs.obs_ws import ObsError
        try:
            jpeg = self.client.screenshot_jpeg(self.source, self.width, self.quality)
        except ObsError:
            if self.fixed_source:
                raise
            self.source = self.client.program_scene()
            jpeg = self.client.screenshot_jpeg(self.source, self.width, self.quality)
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("OBS sent an image that could not be decoded")
        return time.monotonic() - self._t0, frame

    def record_ms(self) -> int | None:
        """Position in the recording OBS is writing, in ms; None when not recording.
        This is what lines a live detection up with the file on disk."""
        try:
            st = self.client.record_status()
        except Exception:
            return None
        return int(st.get("outputDuration") or 0) if st.get("outputActive") else None

    def close(self) -> None:
        pass
