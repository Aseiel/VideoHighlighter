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
import subprocess
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from queue import Empty, Queue

import cv2
from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap

try:
    from modules.app_paths import ffmpeg_exe
except Exception:  # pragma: no cover - fall back to OpenCV-only extraction
    ffmpeg_exe = None


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
        vr_mode: bool = False,
        n_workers: int = 3,
    ):
        super().__init__()
        self.video_path = str(video_path)
        self.mem_limit = mem_limit
        self._vr_mode = vr_mode

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

        # ffmpeg is the fast path (scaled/cropped decode); OpenCV is the fallback.
        self._ffmpeg = ffmpeg_exe() if ffmpeg_exe is not None else None
        # Self-disables after the first hwaccel failure so we don't keep paying
        # for a doomed GPU attempt on systems where it isn't usable.
        self._use_hwaccel = True

        # Multiple workers: each ffmpeg extraction is independent (its own fast
        # seek), so high-res VR filmstrips fill in parallel instead of crawling
        # through one thread.
        self._workers = []
        for i in range(max(1, n_workers)):
            t = threading.Thread(
                target=self._worker_loop, daemon=True,
                name=f"ThumbnailWorker-{i}",
            )
            t.start()
            self._workers.append(t)

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

    def set_vr_mode(self, enabled: bool):
        """Enable/disable VR half-frame crop. Clears memory cache so new frames are extracted."""
        if self._vr_mode == enabled:
            return
        self._vr_mode = enabled
        with self._lock:
            self._mem.clear()
            self._in_flight.clear()
        # Wipe disk cache so old full-frame (or half-frame) thumbnails are not reused
        for f in self.disk_dir.glob("*.jpg"):
            try:
                f.unlink()
            except OSError:
                pass

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
        """Background thread: pull one key at a time and extract it.

        Each extraction is an independent ffmpeg fast-seek, so several workers
        can run at once without sharing a VideoCapture — this is what keeps high
        resolution VR filmstrips from serializing every full-res decode through a
        single thread.
        """
        while not self._stopped:
            try:
                key = self._queue.get(timeout=0.5)
            except Empty:
                continue
            if self._stopped:
                break
            try:
                self._extract_one(key)
            except Exception as e:
                with self._lock:
                    self._in_flight.discard(key)
                print(f"⚠️ ThumbnailCache extract failed: {e}")

    def _extract_one(self, key):
        """Extract, persist, and emit a single thumbnail.

        Fast path: ffmpeg decodes a single keyframe already scaled (and VR
        cropped) to the tiny target size, writing straight to the disk cache.
        Fallback: OpenCV full-frame decode + resize, used only when ffmpeg is
        unavailable or fails for this frame.
        """
        time_ms, height = key
        out_path = self._disk_path(key)

        frame = None
        if self._extract_via_ffmpeg(time_ms, height, out_path):
            frame = cv2.imread(str(out_path))

        if frame is None:
            frame = self._extract_via_opencv(time_ms, height)
            if frame is not None:
                try:
                    cv2.imwrite(
                        str(out_path), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 60],
                    )
                except Exception as e:
                    print(f"⚠️ ThumbnailCache disk write failed: {e}")

        with self._lock:
            self._in_flight.discard(key)

        if frame is None:
            return

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        self._image_ready.emit(time_ms, height, qimg)

    def _extract_via_ffmpeg(self, time_ms: int, height: int, out_path: Path) -> bool:
        """Decode one frame with ffmpeg, scaled (and VR-cropped) to `height`.

        Input seeking (-ss before -i) makes ffmpeg jump to the nearest keyframe
        and decode ~one frame instead of a whole GOP — the key win on very high
        resolution VR footage. The crop/scale run inside the decode filter, so a
        full 8K frame is never handed back to Python.
        """
        if not self._ffmpeg:
            return False

        seconds = max(0.0, time_ms / 1000.0)
        if self._vr_mode:
            # Crop the left eye first, then scale — half the pixels to scale.
            vf = f"crop=iw/2:ih:0:0,scale=-2:{height}"
        else:
            vf = f"scale=-2:{height}"

        # Try hardware-accelerated decode first; if it ever fails, remember that
        # and stick to software decode for the rest of the session.
        attempts = [True, False] if self._use_hwaccel else [False]
        for use_hw in attempts:
            cmd = [self._ffmpeg, "-nostdin", "-v", "error"]
            if use_hw:
                cmd += ["-hwaccel", "auto"]
            cmd += [
                "-ss", f"{seconds:.3f}",
                "-i", self.video_path,
                "-frames:v", "1",
                "-vf", vf,
                "-q:v", "5",
                "-y", str(out_path),
            ]
            if self._run_ffmpeg(cmd) and out_path.exists():
                return True
            if use_hw:
                # hwaccel path failed — don't pay for it again this session.
                self._use_hwaccel = False

        return False

    def _run_ffmpeg(self, cmd: list) -> bool:
        creationflags = 0
        if sys.platform.startswith("win"):
            # Keep console windows from flashing for every extraction.
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            r = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=30,
                creationflags=creationflags,
            )
            return r.returncode == 0
        except Exception:
            return False

    def _extract_via_opencv(self, time_ms: int, height: int):
        """Fallback: full-frame decode + resize (slow on VR, but always works)."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
            ok, frame = cap.read()
        finally:
            cap.release()

        if not ok or frame is None:
            return None

        h, w = frame.shape[:2]
        if self._vr_mode:
            frame = frame[:, : w // 2]
            w = w // 2
        if h != height:
            new_w = max(1, int(round(w * height / h)))
            frame = cv2.resize(frame, (new_w, height), interpolation=cv2.INTER_AREA)
        return frame

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