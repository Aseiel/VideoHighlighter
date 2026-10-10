"""
Runs LiveBoxTracker (live_tracking.py) on every displayed frame.

Taps the video item's sink — the frames display_frames.py has already shrunk to
screen size — takes the luma plane, steps the tracker, and emits the boxes in
the same (results, frame_w, frame_h) shape the live overlays draw from. A
detector's results come in through add_detections() with the time of the frame
they were computed on; the tracker re-anchors from them.

Latest frame only, on a daemon thread: if a step ever takes longer than a frame,
frames are skipped, never queued.
"""

from __future__ import annotations

import math
import threading
from typing import Optional

import numpy as np

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtMultimedia import QVideoFrame, QVideoFrameFormat

from video_ai_editor.display_frames import _map_readonly
from video_ai_editor.live_tracking import LiveBoxTracker

_PF = QVideoFrameFormat.PixelFormat
# Formats whose plane 0 is the luma plane, and how far to shift it down to 8 bits.
_LUMA_SHIFT = {
    _PF.Format_NV12: 0, _PF.Format_NV21: 0, _PF.Format_YUV420P: 0, _PF.Format_YV12: 0,
    _PF.Format_YUV422P: 0, _PF.Format_P010: 8, _PF.Format_P016: 8,
    _PF.Format_YUV420P10: 2,
}

_MAX_TRACK_W = 1280     # luma wider than this is decimated; the tracker gains nothing from more


def luma_of(vframe, crop_left_half: bool = False) -> Optional[np.ndarray]:
    """The frame's luma as a contiguous uint8 array, at most _MAX_TRACK_W wide."""
    w, h = vframe.width(), vframe.height()
    if w <= 0 or h <= 0:
        return None
    shift = _LUMA_SHIFT.get(vframe.pixelFormat())
    if shift is None:
        img = vframe.toImage()
        if img.isNull():
            return None
        from video_ai_editor.live_face import qimage_to_bgr
        bgr = qimage_to_bgr(img)
        gray = bgr.mean(axis=2).astype(np.uint8)
        return gray[:, : w // 2] if crop_left_half else gray
    cw = w // 2 if crop_left_half else w
    k = max(1, math.ceil(cw / _MAX_TRACK_W))
    if not _map_readonly(vframe):
        return None
    try:
        dt, bpp = (np.uint16, 2) if shift else (np.uint8, 1)
        y = np.frombuffer(vframe.bits(0), dtype=dt).reshape(h, vframe.bytesPerLine(0) // bpp)
        y = y[::k, :cw:k]
        y = (y >> shift).astype(np.uint8) if shift else np.ascontiguousarray(y)
    finally:
        vframe.unmap()
    return y


class LiveTrackFeed(QObject):
    """Usage::

        feed = LiveTrackFeed(video_item.videoSink())
        detector_controller.detected_at.connect(feed.add_detections)
        feed.results_ready.connect(overlay.update_boxes)
        feed.set_enabled(True)
    """

    results_ready = Signal(list, int, int)
    _boxes = Signal(list, int, int)      # tracker thread -> UI thread

    # Keys of a detector result copied onto the track it starts or corrects.
    PAYLOAD_KEYS = ("identity_id", "name", "sim", "det_score")

    def __init__(self, video_sink, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._tracker = LiveBoxTracker(identity_key="identity_id")
        self._enabled = False
        self._vr_mode = False
        self._cond = threading.Condition()
        self._frame = None
        self._detections: list = []        # [(results, w, h, t)] waiting for the thread
        self._reset = False
        self._stopped = False
        self._boxes.connect(self._deliver)
        self._video_sink = video_sink
        self._video_sink.videoFrameChanged.connect(self._on_frame)
        self._thread = threading.Thread(target=self._run, name="live-track", daemon=True)
        self._thread.start()

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        if not enabled:
            with self._cond:
                self._reset = True
                self._detections.clear()

    def set_vr_mode(self, enabled: bool) -> None:
        self._vr_mode = enabled
        with self._cond:
            self._reset = True

    @property
    def tracker(self) -> LiveBoxTracker:
        return self._tracker

    @Slot(object)
    def _on_frame(self, vframe):
        if not self._enabled:
            return
        with self._cond:
            self._frame = vframe
            self._cond.notify()

    @Slot(list, int, int, float)
    def add_detections(self, results, w, h, t):
        if not self._enabled:
            return
        with self._cond:
            self._detections.append((results, w, h, t))
            self._cond.notify()

    def _run(self):
        while True:
            with self._cond:
                while self._frame is None and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    return
                vframe, self._frame = self._frame, None
                pending, self._detections = self._detections, []
                reset, self._reset = self._reset, False
            try:
                self._process(vframe, pending, reset)
            except Exception as e:
                print(f"⚠️ LiveTrackFeed error: {e}")

    def _process(self, vframe, pending, reset):
        if reset:
            self._tracker.reset()
        gray = luma_of(vframe, crop_left_half=self._vr_mode)
        if gray is None:
            return
        gh, gw = gray.shape
        self._tracker.step(gray, vframe.startTime() / 1e6)
        for results, w, h, t in pending:
            sx = gw / w if w else 1.0
            sy = gh / h if h else 1.0
            dets = []
            for r in results:
                x1, y1, x2, y2 = r["bbox"]
                dets.append(((x1 * sx, y1 * sy, x2 * sx, y2 * sy),
                             {k: r.get(k) for k in self.PAYLOAD_KEYS}))
            self._tracker.correct(dets, t)
        self._boxes.emit(self._tracker.boxes(), gw, gh)

    @Slot(list, int, int)
    def _deliver(self, boxes, w, h):
        # Checked here, on the UI thread, so a batch the tracker queued just
        # before set_enabled(False) can't redraw boxes the overlay just cleared.
        if self._enabled:
            self.results_ready.emit(boxes, w, h)

    def shutdown(self) -> None:
        self._enabled = False
        try:
            self._video_sink.videoFrameChanged.disconnect(self._on_frame)
        except Exception:
            pass
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        self._thread.join(timeout=0.5)
