"""
Live Face Recognition — TRUE real-time overlay (runs the model on the frame
showing right now). This is the "Live (real-time)" mode, as opposed to
"Live (cache)" which only replays pre-computed detections.

Why face recognition is the right first true-live feature:
    Identity here comes from FaceIdentityBank — appearance matching, NOT
    tracking. So it has no temporal dependency: every frame is independent.
    You can scrub, jump, or play a never-analysed video and it still names
    people correctly. Anything tracking-based (object/person track_id) would
    break the instant the playhead jumps; face+bank doesn't.

How it stays smooth:
    Video plays ~30fps; face recognition is slower than that. So we:
      - tap frames from the player's QVideoSink (videoFrameChanged),
      - keep only the LATEST frame (drop stale backlog),
      - run inference on a WORKER THREAD,
      - emit boxes back to the UI thread to draw.
    Boxes lag the picture by a few ms in fast motion — fine for labels.

Wiring (inside RealtimeOverlayPreview, for the new mode):
    from video_ai_editor.face_identity import FaceIdentityBank
    from video_ai_editor.live_face import LiveFaceController, LiveFaceOverlay

    self._live_face = LiveFaceController(
        bank=self._face_bank,
        video_sink=self._scene.video_item.videoSink(),   # tap the frames
    )
    self._live_overlay = LiveFaceOverlay(self._scene)
    self._live_face.results_ready.connect(self._live_overlay.update_boxes)

    # toggle:
    self._live_face.set_enabled(True)   # when "Live (real-time)" is selected

NOTE: the QVideoSink frame tap and QImage->numpy conversion are Qt-version
sensitive. This targets PySide6 / Qt6. Verify the conversion on your machine
(print frame.shape once) before trusting the box coordinates.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from PySide6.QtCore import (
    Qt, QObject, QThread, QTimer, QMutex, QMetaObject, Signal, Slot, QRectF,
)
from PySide6.QtGui import QImage, QColor, QPen, QBrush, QFont
from PySide6.QtWidgets import QGraphicsScene, QGraphicsRectItem, QGraphicsSimpleTextItem


# ──────────────────────────────────────────────────────────────────
# QImage -> numpy BGR  (PySide6 / Qt6)
# ──────────────────────────────────────────────────────────────────

def qimage_to_bgr(qimg: QImage) -> Optional[np.ndarray]:
    """Convert a QImage to a contiguous BGR numpy array (what cv2/InsightFace want)."""
    if qimg is None or qimg.isNull():
        return None
    img = qimg.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    bpl = img.bytesPerLine()                       # may be padded > w*3
    ptr = img.constBits()                          # PySide6: memoryview, already sized
    buf = np.frombuffer(ptr, dtype=np.uint8).reshape(h, bpl)
    rgb = buf[:, : w * 3].reshape(h, w, 3)
    return rgb[:, :, ::-1].copy()                  # RGB -> BGR, copy() detaches from Qt buffer


# ──────────────────────────────────────────────────────────────────
# Worker — runs on a background thread, processes the latest frame only
# ──────────────────────────────────────────────────────────────────

class LiveFaceWorker(QObject):
    """
    Lives on a worker QThread. Holds the latest submitted QVideoFrame; on each
    tick it converts + downsamples + runs face recognition, dropping stale frames.
    """

    results_ready = Signal(list, int, int)

    # Max dimension (width) passed to InsightFace. Larger frames are downsampled
    # first — InsightFace resizes to 640×640 internally anyway, so running on a
    # 4K frame just wastes numpy allocation and BGR copy time.
    MAX_INFERENCE_W = 960

    def __init__(self, bank, auto_enroll: bool = True, interval_ms: int = 500,
                 learn_threshold: float = 0.55):
        super().__init__()
        self._bank = bank
        self._auto_enroll = auto_enroll
        self._interval_ms = interval_ms  # default 500ms — face recog on CPU is slow
        self._latest_vframe = None       # raw QVideoFrame, converted on worker thread
        self._mutex = QMutex()
        self._timer: Optional[QTimer] = None
        self._busy = False
        self._learn_threshold = learn_threshold
        self._vr_mode = False
        self._stopped = False   # hard stop — checked at top of _tick

    @Slot()
    def start_processing(self):
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(self._interval_ms)

    @Slot()
    def stop_processing(self):
        self._stopped = True
        if self._timer:
            self._timer.stop()

    @Slot()
    def _tick(self):
        if self._stopped or self._busy:
            return
        return self._do_tick()

    def submit_vframe(self, vframe) -> None:
        """Thread-safe: stash a raw QVideoFrame, overwriting any unprocessed one."""
        self._mutex.lock()
        self._latest_vframe = vframe
        self._mutex.unlock()

    def _do_tick(self):
        self._mutex.lock()
        vframe = self._latest_vframe
        self._latest_vframe = None
        self._mutex.unlock()
        if vframe is None:
            return

        self._busy = True
        try:
            # GPU→CPU conversion happens here on the worker thread (not main thread)
            img = vframe.toImage()
            if img.isNull():
                return
            frame = qimage_to_bgr(img)
            if frame is None:
                return

            # For SBS VR video crop to left half before inference — right half is
            # a duplicate eye view and doubles the work for no extra information.
            if self._vr_mode:
                frame = frame[:, : frame.shape[1] // 2]

            # Downsample to MAX_INFERENCE_W so InsightFace doesn't chew through a
            # 4K BGR array; it resizes to 640px internally anyway.
            h, w = frame.shape[:2]
            if w > self.MAX_INFERENCE_W:
                import cv2
                scale = self.MAX_INFERENCE_W / w
                frame = cv2.resize(frame,
                                   (self.MAX_INFERENCE_W, int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
                h, w = frame.shape[:2]

            faces = self._bank.detect_faces(frame)
            results = []
            for f in faces:
                iid, sim = self._bank.match(f["embedding"])
                if iid is None and self._auto_enroll:
                    thumb = _crop(frame, f["bbox"])
                    iid = self._bank.assign(f["embedding"], thumbnail=thumb,
                                            det_score=f["det_score"])
                elif iid is not None and sim >= self._learn_threshold and f["det_score"] >= 0.6:
                    self._bank.reinforce(iid, f["embedding"])
                results.append({
                    "bbox": f["bbox"],
                    "identity_id": iid,
                    "name": self._bank.name_for(iid) if iid else None,
                    "sim": float(sim),
                    "det_score": f["det_score"],
                })
            self.results_ready.emit(results, w, h)
        except Exception as e:
            print(f"⚠️ LiveFaceWorker error: {e}")
        finally:
            self._busy = False


# ──────────────────────────────────────────────────────────────────
# Controller — owns the worker thread + the frame tap
# ──────────────────────────────────────────────────────────────────

class LiveFaceController(QObject):
    """
    Connects a player's QVideoSink to the worker thread and re-emits results
    on the UI thread for drawing.
    """

    results_ready = Signal(list, int, int)   # forwarded from the worker

    # Minimum seconds between frame submissions to the worker.
    # Face recognition on CPU is slow — no point sending 60 frames/s.
    _SUBMIT_INTERVAL = 0.5   # 2 fps cap for inference

    def __init__(self, bank, video_sink, auto_enroll: bool = True, parent=None):
        super().__init__(parent)
        self._enabled = False
        self._last_submit = 0.0

        self._worker = LiveFaceWorker(bank, auto_enroll=auto_enroll)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.start_processing)
        self._worker.results_ready.connect(self._on_results)
        self._thread.start()

        self._video_sink = video_sink
        self._video_sink.videoFrameChanged.connect(self._on_frame)

    def set_enabled(self, enabled: bool):
        self._enabled = enabled

    def set_vr_mode(self, enabled: bool):
        """Crop frames to left half before inference (side-by-side VR videos)."""
        self._worker._vr_mode = enabled

    @Slot(object)
    def _on_frame(self, video_frame):
        """Called at source framerate (up to 60fps). We only forward once per
        _SUBMIT_INTERVAL and do NOT convert to BGR here — that happens on the
        worker thread so the main thread is never blocked by GPU→CPU transfers."""
        if not self._enabled:
            return
        now = time.monotonic()
        if now - self._last_submit < self._SUBMIT_INTERVAL:
            return
        self._last_submit = now
        # Store raw QVideoFrame; conversion to BGR happens on the worker thread
        self._worker.submit_vframe(video_frame)

    @Slot(list, int, int)
    def _on_results(self, results, w, h):
        self.results_ready.emit(results, w, h)

    def shutdown(self):
        """Stop the worker thread (call from closeEvent).

        We do NOT block waiting for an in-flight InsightFace inference to finish
        (that's what made close hang). Instead:
          1. Stop feeding the worker and disconnect its output signal so it can
             never emit into Qt objects that are being torn down (no crash).
          2. Set the hard-stop flag + ask its timer to stop on the worker thread.
          3. quit() and wait only briefly. If an inference is still running, the
             thread finishes it in the background and exits on its own; the final
             app exit (os._exit) reaps anything still alive.
        """
        self._enabled = False
        try:
            self._video_sink.videoFrameChanged.disconnect(self._on_frame)
        except Exception:
            pass
        try:
            self._worker.results_ready.disconnect(self._on_results)
        except Exception:
            pass

        self._worker._stopped = True
        # Stop the worker's QTimer ON the worker thread — stopping it from the GUI
        # thread throws "Timers cannot be stopped from another thread".
        QMetaObject.invokeMethod(
            self._worker, "stop_processing", Qt.ConnectionType.QueuedConnection
        )
        self._thread.quit()
        # Short, non-blocking-ish wait. Don't sit here for seconds on a 4K frame.
        self._thread.wait(300)


# ──────────────────────────────────────────────────────────────────
# Overlay — draws the recognised faces onto an existing scene
# ──────────────────────────────────────────────────────────────────

def _color_for_id(identity_id: Optional[str]) -> QColor:
    if not identity_id:
        return QColor(180, 180, 180)
    h = (int(identity_id[:8], 16) % 360) / 360.0 if _is_hex(identity_id[:8]) else 0.5
    return QColor.fromHsvF(h, 0.8, 0.95)


def _is_hex(s: str) -> bool:
    try:
        int(s, 16)
        return True
    except ValueError:
        return False


class LiveFaceOverlay:
    """
    Manages a small pool of rect+label items on a QGraphicsScene and updates
    them each time the worker reports results. Boxes are scaled from frame
    pixel coords to the scene's coordinate system.
    """

    def __init__(self, scene: QGraphicsScene, z: float = 60):
        self._scene = scene
        self._z = z
        self._rects: list[QGraphicsRectItem] = []
        self._labels: list[QGraphicsSimpleTextItem] = []
        self._vr_mode = False
        # identity_ids the user has hidden via the overlay filter
        self._hidden_ids: set[str] = set()

    @property
    def hidden_ids(self) -> set[str]:
        return self._hidden_ids

    def set_identity_hidden(self, identity_id: str, hidden: bool):
        """Show/hide one recognised identity on the overlay. Takes effect on the
        next results frame (and immediately for boxes already on screen)."""
        if hidden:
            self._hidden_ids.add(identity_id)
        else:
            self._hidden_ids.discard(identity_id)

    def set_vr_mode(self, enabled: bool):
        """Map boxes into the left half of the scene (SBS VR videos).

        In VR mode the worker runs detection on the left-half frame, so box
        coords span only the left eye. The scene, however, is full SBS width —
        scale x against half the scene so boxes land on the visible left half.
        """
        self._vr_mode = enabled

    def _ensure_pool(self, n: int):
        while len(self._rects) < n:
            rect = QGraphicsRectItem()
            pen = QPen(QColor(0, 255, 0), 2.5)
            pen.setCosmetic(True)
            rect.setPen(pen)
            rect.setBrush(QBrush(Qt.NoBrush))
            rect.setZValue(self._z)
            rect.setVisible(False)
            self._scene.addItem(rect)

            label = QGraphicsSimpleTextItem()
            label.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
            label.setBrush(QBrush(QColor(255, 255, 255)))
            label.setZValue(self._z + 1)
            label.setVisible(False)
            self._scene.addItem(label)

            self._rects.append(rect)
            self._labels.append(label)

    @Slot(list, int, int)
    def update_boxes(self, results, frame_w, frame_h):
        """Redraw the overlay from the latest worker results."""
        # Drop identities the user hid via the 'Facial recognition' filter.
        if self._hidden_ids:
            results = [r for r in results if r.get("identity_id") not in self._hidden_ids]
        scene_rect = self._scene.sceneRect()
        # In VR mode boxes come from the left-half frame, so map them into the
        # left half of the (full-width SBS) scene.
        target_w = scene_rect.width() / 2 if self._vr_mode else scene_rect.width()
        sx = target_w / frame_w if frame_w else 1.0
        sy = scene_rect.height() / frame_h if frame_h else 1.0

        self._ensure_pool(len(results))

        for i, rect in enumerate(self._rects):
            if i >= len(results):
                rect.setVisible(False)
                self._labels[i].setVisible(False)
                continue

            r = results[i]
            x1, y1, x2, y2 = r["bbox"]
            rx, ry = x1 * sx, y1 * sy
            rw, rh = (x2 - x1) * sx, (y2 - y1) * sy

            color = _color_for_id(r["identity_id"])
            pen = rect.pen()
            pen.setColor(color)
            rect.setPen(pen)
            rect.setRect(rx, ry, rw, rh)
            rect.setData(0, r["identity_id"])
            rect.setVisible(True)

            name = r["name"] or f"? {r['sim']:.2f}"
            label = self._labels[i]
            label.setText(name)
            label.setBrush(QBrush(color.lighter(150)))
            label.setPos(rx, max(0, ry - 18))
            label.setVisible(True)

    def clear(self):
        for rect in self._rects:
            rect.setVisible(False)
        for label in self._labels:
            label.setVisible(False)


# ──────────────────────────────────────────────────────────────────
# helper
# ──────────────────────────────────────────────────────────────────

def _crop(frame_bgr: np.ndarray, bbox) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2].copy()