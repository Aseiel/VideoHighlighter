"""
Async thumbnail extraction with two-tier caching (memory + disk).

Public API:
    cache = ThumbnailCache(video_path)
    cache.thumbnail_ready.connect(my_repaint_slot)
    pixmap = cache.request(time_seconds=12.3, height_px=60)
    # → returns QPixmap immediately if cached, else None
    # → emits thumbnail_ready(time, height, pixmap) when extraction finishes

Times are quantized to 100ms so hover scrubbing doesn't fire a thousand grabs.
Disk cache lives under ./cache/thumbnails/<video_hash>/ and survives restarts.
"""

from __future__ import annotations

import hashlib
import os
import threading
from collections import OrderedDict
from pathlib import Path
from queue import Empty, Queue

import cv2
from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap


# Quantize hover times to 100ms buckets so we don't extract near-duplicate frames
TIME_QUANT_MS = 100


class ThumbnailCache(QObject):
    """
    Per-video thumbnail cache. Create one instance per video.

    Signals:
        thumbnail_ready(time_seconds: float, height_px: int, pixmap: QPixmap)
            Emitted on the main thread after an async extraction finishes.
            Connect this to your clip item's update() or scene.invalidate().
    """

    # Internal: worker → main thread, carries a QImage (thread-safe)
    _image_ready = Signal(int, int, QImage)  # time_ms, height_px, image
    # Public: main thread, carries the final QPixmap
    thumbnail_ready = Signal(float, int, QPixmap)

    def __init__(
        self,
        video_path: str,
        cache_dir: str = "./cache/thumbnails",
        mem_limit: int = 300,
    ):
        super().__init__()
        self.video_path = str(video_path)
        self.mem_limit = mem_limit

        # Hash includes mtime + size so editing the source file invalidates cache
        self.video_hash = self._compute_video_hash()
        self.disk_dir = Path(cache_dir) / self.video_hash
        self.disk_dir.mkdir(parents=True, exist_ok=True)

        # In-memory LRU cache: (time_ms, height_px) → QPixmap
        self._mem: "OrderedDict[tuple, QPixmap]" = OrderedDict()

        # Request bookkeeping
        self._queue: "Queue[tuple]" = Queue()
        self._in_flight: set = set()
        self._lock = threading.Lock()
        self._stopped = False

        # Worker → main thread marshaling. QueuedConnection is implicit
        # for cross-thread signal emissions, so the slot runs on the main thread.
        self._image_ready.connect(self._on_image_ready, Qt.QueuedConnection)

        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="ThumbnailWorker"
        )
        self._worker.start()

    # ── Public API ────────────────────────────────────────────────────────

    def request(self, time_seconds: float, height_px: int) -> QPixmap | None:
        """
        Return a cached pixmap or None.

        If None is returned, an async extraction has been queued and
        `thumbnail_ready` will fire when it's ready.
        """
        key = self._make_key(time_seconds, height_px)

        # 1. Memory hit
        if key in self._mem:
            self._mem.move_to_end(key)
            return self._mem[key]

        # 2. Disk hit → load to memory, return
        disk_path = self._disk_path(key)
        if disk_path.exists():
            pix = QPixmap(str(disk_path))
            if not pix.isNull():
                self._add_to_mem(key, pix)
                return pix

        # 3. Miss → queue extraction
        with self._lock:
            if key not in self._in_flight:
                self._in_flight.add(key)
                self._queue.put(key)
        return None

    def stop(self):
        """Stop the worker thread. Call on app shutdown."""
        self._stopped = True

    # ── Internals ─────────────────────────────────────────────────────────

    def _make_key(self, time_seconds: float, height_px: int) -> tuple:
        time_ms = int(time_seconds * 1000)
        time_ms = (time_ms // TIME_QUANT_MS) * TIME_QUANT_MS
        return (time_ms, int(height_px))

    def _disk_path(self, key: tuple) -> Path:
        time_ms, height = key
        return self.disk_dir / f"{time_ms}_{height}.jpg"

    def _add_to_mem(self, key: tuple, pixmap: QPixmap):
        self._mem[key] = pixmap
        self._mem.move_to_end(key)
        while len(self._mem) > self.mem_limit:
            self._mem.popitem(last=False)

    def _compute_video_hash(self) -> str:
        try:
            st = os.stat(self.video_path)
            sig = f"{self.video_path}|{st.st_size}|{int(st.st_mtime)}"
        except OSError:
            sig = self.video_path
        return hashlib.md5(sig.encode("utf-8")).hexdigest()[:16]

    def _worker_loop(self):
        """Background thread: pull keys, sort by time, extract frames."""
        cap = None
        try:
            while not self._stopped:
                # Block for the first key
                try:
                    first = self._queue.get(timeout=0.5)
                except Empty:
                    continue

                # Drain everything else that's already waiting, then sort by time.
                # Forward seeks are much cheaper than backward seeks.
                batch = [first]
                try:
                    while True:
                        batch.append(self._queue.get_nowait())
                except Empty:
                    pass
                batch.sort(key=lambda k: k[0])  # by time_ms

                # Lazy-open capture once
                if cap is None:
                    cap = cv2.VideoCapture(self.video_path)
                    if not cap.isOpened():
                        print(f"⚠️ ThumbnailCache: cannot open {self.video_path}")
                        return

                for key in batch:
                    if self._stopped:
                        break
                    self._extract_one(cap, key)
        finally:
            if cap is not None:
                cap.release()

    def _extract_one(self, cap, key):
        """Extract, persist, and emit a single thumbnail."""
        time_ms, height = key

        cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
        ok, frame = cap.read()

        with self._lock:
            self._in_flight.discard(key)

        if not ok or frame is None:
            return

        h, w = frame.shape[:2]
        if h != height:
            new_w = max(1, int(round(w * height / h)))
            frame = cv2.resize(frame, (new_w, height), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]

        try:
            cv2.imwrite(
                str(self._disk_path(key)),
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 60],  # ← lowered from 80
            )
        except Exception as e:
            print(f"⚠️ ThumbnailCache disk write failed: {e}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        self._image_ready.emit(time_ms, height, qimg)

    def prefetch_range(self, start_time: float, end_time: float,
                    height_px: int, n_slots: int):
        """
        Queue up all thumbs for a clip's filmstrip in one go.
        Match the slot calculation in filmstrip_painter so we request
        exactly the frames paint() will ask for.
        """
        duration = max(1e-6, end_time - start_time)
        for i in range(n_slots):
            t = start_time + (i + 0.5) / n_slots * duration
            self.request(t, height_px)  # return value ignored — just queue it

    @Slot(int, int, QImage)
    def _on_image_ready(self, time_ms: int, height: int, qimg: QImage):
        """Runs on the main thread."""
        pix = QPixmap.fromImage(qimg)
        if pix.isNull():
            return
        key = (time_ms, height)
        self._add_to_mem(key, pix)
        self.thumbnail_ready.emit(time_ms / 1000.0, height, pix)