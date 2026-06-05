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

from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, QObject, QThread, QTimer, QMutex, Signal, Slot, QRectF
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
    Lives on a worker QThread. Holds the latest submitted frame; on each tick
    it runs face recognition on that frame (and drops anything older).
    """

    # results: list of dicts, plus the frame size they were computed at
    results_ready = Signal(list, int, int)

    def __init__(self, bank, auto_enroll: bool = True, interval_ms: int = 33):
        super().__init__()
        self._bank = bank
        self._auto_enroll = auto_enroll
        self._interval_ms = interval_ms
        self._latest: Optional[np.ndarray] = None
        self._mutex = QMutex()
        self._timer: Optional[QTimer] = None
        self._busy = False

    @Slot()
    def start_processing(self):
        """Called once the worker is on its thread (connect to QThread.started)."""
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(self._interval_ms)

    @Slot()
    def stop_processing(self):
        if self._timer:
            self._timer.stop()

    def submit(self, frame_bgr: np.ndarray):
        """Thread-safe: stash the latest frame, overwriting any unprocessed one."""
        self._mutex.lock()
        self._latest = frame_bgr
        self._mutex.unlock()

    @Slot()
    def _tick(self):
        if self._busy:
            return  # still chewing the previous frame; skip
        self._mutex.lock()
        frame = self._latest
        self._latest = None
        self._mutex.unlock()
        if frame is None:
            return

        self._busy = True
        try:
            faces = self._bank.detect_faces(frame)
            results = []
            for f in faces:
                iid, sim = self._bank.match(f["embedding"])
                if iid is None and self._auto_enroll:
                    thumb = _crop(frame, f["bbox"])
                    iid = self._bank.assign(f["embedding"], thumbnail=thumb,
                                            det_score=f["det_score"])
                results.append({
                    "bbox": f["bbox"],                      # pixel coords in this frame
                    "identity_id": iid,
                    "name": self._bank.name_for(iid) if iid else None,
                    "sim": float(sim),
                    "det_score": f["det_score"],
                })
            h, w = frame.shape[:2]
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

    def __init__(self, bank, video_sink, auto_enroll: bool = True, parent=None):
        super().__init__(parent)
        self._enabled = False

        self._worker = LiveFaceWorker(bank, auto_enroll=auto_enroll)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.start_processing)
        self._worker.results_ready.connect(self._on_results)   # queued -> UI thread
        self._thread.start()

        # Tap frames
        self._video_sink = video_sink
        self._video_sink.videoFrameChanged.connect(self._on_frame)

    def set_enabled(self, enabled: bool):
        """Turn the live face pass on/off (e.g. when the mode dropdown changes)."""
        self._enabled = enabled

    @Slot(object)
    def _on_frame(self, video_frame):
        if not self._enabled:
            return
        try:
            img = video_frame.toImage()
        except Exception:
            return
        bgr = qimage_to_bgr(img)
        if bgr is not None:
            self._worker.submit(bgr)

    @Slot(list, int, int)
    def _on_results(self, results, w, h):
        self.results_ready.emit(results, w, h)

    def shutdown(self):
        """Stop the worker thread cleanly (call from closeEvent)."""
        try:
            self._video_sink.videoFrameChanged.disconnect(self._on_frame)
        except Exception:
            pass
        QTimer.singleShot(0, self._worker.stop_processing)
        self._thread.quit()
        self._thread.wait(1000)


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
        scene_rect = self._scene.sceneRect()
        sx = scene_rect.width() / frame_w if frame_w else 1.0
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