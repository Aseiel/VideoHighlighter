"""Outlines for the overlay, traced in the background while the video plays.

A detector gives boxes. With the overlay's shape switch on "Outlines", the
detections around the playhead are outlined (SAM prompted with each box, see
``modules/vision/outlines.py``) so the overlay can draw the shape the model
found instead of the rectangle around it. That is for understanding how
accurate a trained model is: a box says roughly where; an outline shows what
it actually picked out, and a wrong one is easy to see. An outline is a second
model's reading of the box, so the box stays drawn behind it.

One worker thread, one job at a time:

* **Nearest first.** Jobs carry a time; the worker takes the one nearest the
  playhead, and drops jobs the playhead has left behind, so seeking does not
  leave a queue of work nobody will see.
* **The frame the detector saw.** Cached detections are traced on the video
  frame at their own timestamp, read from the file, not on whatever frame is
  on screen when the job runs. A live caller can hand over the frame itself.
* **Results on the GUI thread.** The worker only fills a queue; a timer on the
  GUI thread drains it and emits ``traced``, so nothing touches Qt objects or
  the cache from the worker.
* **A fallback.** If SAM cannot be loaded (no weights, no network on first
  use), the feed says so once and traces with GrabCut instead.
"""
from __future__ import annotations

import queue
import threading
from typing import Callable, Optional, Sequence

from PySide6.QtCore import QObject, QTimer, Signal

# Jobs this far behind the playhead are dropped unrun (seconds).
STALE_SECS = 1.5
# How often the GUI thread collects finished outlines (ms).
DRAIN_MS = 60


def relative_to_box(outline: Sequence, box: Sequence) -> list:
    """A frame-normalised outline as points relative to its box ``[x, y, w, h]``
    ((0, 0) its top-left corner, (1, 1) its bottom-right), so it can be drawn
    inside the box wherever a tracker has moved it since. ``[]`` stays ``[]``."""
    if not outline:
        return []
    x, y, w, h = (float(v) for v in box[:4])
    if w <= 0 or h <= 0:
        return []
    return [((float(px) - x) / w, (float(py) - y) / h) for px, py in outline]


def default_outliner():
    """SAM on the best device there is; the caller falls back if it fails."""
    from modules.vision.outlines import SamOutliner
    return SamOutliner()


def fallback_outliner():
    from modules.vision.outlines import GrabCutOutliner
    return GrabCutOutliner()


class VideoFrames:
    """Frames by timestamp from one video file, for the worker thread only."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self._cap = None

    def at(self, timestamp: float):
        import cv2

        if self._cap is None:
            self._cap = cv2.VideoCapture(self.video_path)
        self._cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(timestamp)) * 1000.0)
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


class OutlineFeed(QObject):
    """Background outliner. ``submit`` jobs; ``traced`` delivers results.

    ``traced(key, outlines)``: ``outlines`` is aligned with the job's boxes,
    each a normalised polygon, or ``[]`` where nothing usable was found (the
    box stands in, and it is not tried again).
    """

    traced = Signal(object, list)
    status = Signal(str)

    def __init__(self, video_path: Optional[str] = None, *,
                 outliner_factory: Callable = default_outliner,
                 fallback_factory: Callable = fallback_outliner,
                 frames=None, start: bool = True, parent=None):
        super().__init__(parent)
        self._frames = frames if frames is not None else (VideoFrames(video_path) if video_path else None)
        self._make_outliner, self._make_fallback = outliner_factory, fallback_factory
        self._outliner = None
        self._jobs: dict = {}                 # key -> (time, boxes, frame)
        self._now = 0.0
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._done: "queue.Queue" = queue.Queue()
        self._busy_key = None
        self._warm_wanted = False
        self.traced_count = 0
        self._thread = None
        self._timer = QTimer(self)
        self._timer.setInterval(DRAIN_MS)
        self._timer.timeout.connect(self.drain)
        if start:
            self.start()

    # ------------------------------------------------------------------ API

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="outline-feed", daemon=True)
            self._thread.start()
            self._timer.start()

    def submit(self, key, boxes: Sequence, timestamp: float, frame=None) -> bool:
        """Ask for outlines of ``boxes`` (normalised ``[x, y, w, h]``) at
        ``timestamp``. ``frame`` (BGR) if the caller has it; otherwise it is
        read from the video. ``False`` if the key is already queued or running
        (nothing changes); a job dropped as stale can be submitted again."""
        if not boxes:
            return False
        with self._lock:
            if key in self._jobs or key == self._busy_key:
                return False
            self._jobs[key] = (float(timestamp), [list(b) for b in boxes], frame)
        self._wake.set()
        return True

    def set_playhead(self, seconds: float):
        """Where the viewer is: jobs nearest it go first, ones left behind go."""
        with self._lock:
            self._now = float(seconds)
            for key in [k for k, (t, _, _) in self._jobs.items() if t < self._now - STALE_SECS]:
                del self._jobs[key]

    def clear(self):
        """Forget every queued job (the one running still finishes)."""
        with self._lock:
            self._jobs.clear()

    def pending(self) -> int:
        with self._lock:
            return len(self._jobs) + (self._busy_key is not None)

    def stop(self):
        self._stop.set()
        self._wake.set()
        self._timer.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._frames is not None and hasattr(self._frames, "close"):
            self._frames.close()

    def drain(self):
        """Deliver finished outlines (GUI thread)."""
        while True:
            try:
                kind, key, payload = self._done.get_nowait()
            except queue.Empty:
                return
            if kind == "status":
                self.status.emit(payload)
            else:
                self.traced_count += sum(1 for o in payload if o)
                self.traced.emit(key, payload)

    # ------------------------------------------------------------ the worker

    def _next_job(self):
        with self._lock:
            if not self._jobs:
                self._busy_key = None
                return None
            key = min(self._jobs, key=lambda k: abs(self._jobs[k][0] - self._now))
            self._busy_key = key
            return key, self._jobs.pop(key)

    def warm_up(self):
        """Load the outliner now, in the background, rather than on the first
        job: loading SAM the first time in a session takes tens of seconds
        (most of it importing ``transformers``)."""
        self._warm_wanted = True
        self._wake.set()

    def _ensure_outliner(self):
        if self._outliner is not None:
            return
        self._done.put(("status", None, "outlines: loading model…"))
        try:
            outliner = self._make_outliner()
            load = getattr(outliner, "_load", None)
            if callable(load):                    # fail here, not mid-frame
                load()
            self._outliner = outliner
            device = getattr(outliner, "device", "")
            label = getattr(outliner, "name", "sam")
            self._done.put(("status", None, f"outlines: {label}" + (f" on {device}" if device else "")))
        except Exception as exc:                  # noqa: BLE001 - any load failure falls back
            self._use_fallback(exc)

    def _use_fallback(self, exc):
        print(f"[outlines] SAM unavailable ({exc}); tracing with GrabCut instead")
        self._outliner = self._make_fallback()
        self._done.put(("status", None, "outlines: GrabCut (SAM unavailable)"))

    def run_one(self) -> bool:
        """Trace the nearest queued job. ``False`` when there was none.
        The worker's loop body; callable directly in tests."""
        job = self._next_job()
        if job is None:
            if self._warm_wanted and self._outliner is None:
                self._warm_wanted = False
                self._ensure_outliner()
                return True
            return False
        key, (timestamp, boxes, frame) = job
        try:
            if frame is None and self._frames is not None:
                frame = self._frames.at(timestamp)
            outlines = [[] for _ in boxes]
            if frame is not None:
                self._ensure_outliner()
                h, w = frame.shape[:2]
                boxes_px = [(x * w, y * h, (x + bw) * w, (y + bh) * h) for x, y, bw, bh in boxes]
                try:
                    found = self._outliner.outline(frame, boxes_px)
                except Exception as exc:          # noqa: BLE001 - e.g. a GPU that refuses the model
                    if getattr(self._outliner, "name", "") == "grabcut":
                        raise
                    self._use_fallback(exc)
                    found = self._outliner.outline(frame, boxes_px)
                outlines = [o or [] for o in found]
            self._done.put(("traced", key, outlines))
        except Exception as exc:                  # noqa: BLE001 - one bad frame must not stop the feed
            print(f"[outlines] tracing at {timestamp:.2f}s failed: {exc}")
            self._done.put(("traced", key, [[] for _ in boxes]))
        finally:
            with self._lock:
                self._busy_key = None
        return True

    def _run(self):
        while not self._stop.is_set():
            if not self.run_one():
                self._wake.wait(0.25)
                self._wake.clear()
