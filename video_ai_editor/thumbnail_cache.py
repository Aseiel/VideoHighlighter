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
import itertools
import os
import subprocess
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from queue import Empty, PriorityQueue

import cv2
from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QImage, QPixmap

try:
    from modules.app_paths import ffmpeg_exe
except Exception:  # pragma: no cover - fall back to OpenCV-only extraction
    ffmpeg_exe = None


# Quantize hover times to 100ms buckets so we don't extract near-duplicate frames
TIME_QUANT_MS = 100

# Lower number = extracted first. On-screen requests must jump ahead of the bulk
# prefetch backlog so the frames you're actually looking at fill in immediately
# instead of waiting behind every off-screen clip. Hover is the sharpest "I want
# this exact frame now" signal, so it even outranks the visible filmstrip.
PRIORITY_HOVER = -10
PRIORITY_VISIBLE = 0
PRIORITY_PREFETCH = 10

# Extraction strategy is chosen by source resolution. A persistent OpenCV
# VideoCapture seek costs ~5-10ms/frame on normal footage; spawning a fresh
# ffmpeg per thumbnail costs ~90-250ms. ffmpeg only wins when the frame is so
# large that a full-res software decode dominates (VR / 4K+), where its
# keyframe-seek + in-decode crop/scale avoids ever handing Python a huge frame.
# Above this longest-edge pixel count we prefer ffmpeg; at or below it, OpenCV.
FFMPEG_MIN_LONG_EDGE = 2600


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

        # Request bookkeeping.
        #   _queue    : PriorityQueue of (priority, seq, key). Lowest priority
        #               number pops first; seq is a monotonic tiebreaker so
        #               same-priority requests stay FIFO and keys never get
        #               compared.
        #   _pending  : key → best (lowest) priority currently sitting in the
        #               queue. Lets an on-screen request promote a key that was
        #               already queued at prefetch priority.
        #   _active   : keys a worker is extracting right now, so a repeat
        #               request doesn't queue duplicate work mid-flight.
        self._queue: "PriorityQueue[tuple]" = PriorityQueue()
        self._pending: dict = {}
        self._active: set = set()
        self._seq = itertools.count()
        self._lock = threading.Lock()
        self._stopped = False

        # Worker → main thread marshaling. QueuedConnection is implicit
        # for cross-thread signal emissions, so the slot runs on the main thread.
        self._image_ready.connect(self._on_image_ready, Qt.QueuedConnection)

        self._ffmpeg = ffmpeg_exe() if ffmpeg_exe is not None else None
        # Ordered hardware-decode candidates for the ffmpeg (VR/high-res) path.
        # These are GPU-agnostic OS decode APIs, so on an Intel box d3d11va/dxva2
        # drive the Intel iGPU (no QSV/CUDA needed — "auto"/cuda proved unreliable
        # here: cuda can't load nvcuda.dll, auto silently ran software). The first
        # one that actually produces a frame gets pinned; an empty list means
        # software decode (still fast thanks to -skip_frame nokey).
        self._hwaccels = self._default_hwaccels()

        # Probe the source size once and pick the extraction strategy. Persistent
        # OpenCV is the fast path for normal footage; ffmpeg keyframe-seek is kept
        # for VR / very-high-res where a full-frame decode would be the bottleneck.
        self._src_w, self._src_h = self._probe_src_size()
        self._prefer_ffmpeg = self._compute_prefer_ffmpeg()

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

    def request(self, time_seconds: float, height_px: int,
                priority: int = PRIORITY_VISIBLE) -> QPixmap | None:
        """
        Return a cached pixmap or None.

        If None is returned, an async extraction has been queued and
        `thumbnail_ready` will fire when it's ready.

        `priority` controls queue ordering: PRIORITY_HOVER (the frame under the
        mouse) beats PRIORITY_VISIBLE (the default, on-screen filmstrip), which
        beats PRIORITY_PREFETCH (bulk look-ahead). A higher-priority request for
        a key already sitting in the queue promotes it to the front.
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

        # 3. Miss → queue extraction (or promote an already-queued one)
        with self._lock:
            if key in self._active:
                return None  # a worker is already extracting this exact frame
            existing = self._pending.get(key)
            if existing is None or priority < existing:
                self._pending[key] = priority
                self._queue.put((priority, next(self._seq), key))
        return None

    def set_vr_mode(self, enabled: bool):
        """Enable/disable VR half-frame crop. Clears memory cache so new frames are extracted."""
        if self._vr_mode == enabled:
            return
        self._vr_mode = enabled
        # VR footage is high-res side-by-side, so it flips us onto the ffmpeg
        # keyframe path (which also does the left-eye crop inside the decode).
        self._prefer_ffmpeg = self._compute_prefer_ffmpeg()
        with self._lock:
            self._mem.clear()
            self._pending.clear()
            self._active.clear()
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

    def _probe_src_size(self) -> tuple:
        """Read source width/height cheaply (metadata only, no frame decode)."""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                return (0, 0)
            try:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            finally:
                cap.release()
            return (w, h)
        except Exception:
            return (0, 0)

    def _default_hwaccels(self) -> list:
        """Ordered hardware-decode APIs to try for this platform (best first).

        All of these are GPU-vendor-agnostic OS decode paths, so they work on
        Intel/AMD/NVIDIA alike — the point is to use *a* GPU, not a vendor SDK.
        """
        if not self._ffmpeg:
            return []
        if sys.platform.startswith("win"):
            return ["d3d11va", "dxva2"]
        if sys.platform.startswith("linux"):
            return ["vaapi"]
        if sys.platform == "darwin":
            return ["videotoolbox"]
        return []

    def _compute_prefer_ffmpeg(self) -> bool:
        """True when ffmpeg keyframe extraction should win over OpenCV.

        Only worthwhile for VR (needs the in-decode crop) or footage so large a
        full-frame software decode dominates. Everything else — and the case
        where we couldn't probe the size — goes to the much faster persistent
        OpenCV path. If ffmpeg isn't available we always use OpenCV.
        """
        if self._ffmpeg is None:
            return False
        if self._vr_mode:
            return True
        long_edge = max(self._src_w, self._src_h)
        return long_edge >= FFMPEG_MIN_LONG_EDGE

    def _worker_loop(self):
        """Background thread: pull one key at a time and extract it.

        Each worker keeps its own persistent OpenCV VideoCapture (created lazily,
        never shared between threads) so the fast path is a cheap seek+read
        instead of reopening the file — or, on VR/high-res, an independent ffmpeg
        keyframe seek. Either way workers run fully in parallel.
        """
        cap = None
        try:
            while not self._stopped:
                try:
                    priority, _seq, key = self._queue.get(timeout=0.5)
                except Empty:
                    continue
                if self._stopped:
                    break

                # Claim the key, skipping stale entries. A key is stale if it's
                # already been extracted (gone from _pending) or a higher-priority
                # duplicate was queued after this one (pending priority now beats
                # ours) — in that case the better entry will do the work.
                with self._lock:
                    cur = self._pending.get(key)
                    if cur is None or cur < priority:
                        continue
                    del self._pending[key]
                    self._active.add(key)

                try:
                    cap = self._extract_one(key, cap)
                except Exception as e:
                    print(f"⚠️ ThumbnailCache extract failed: {e}")
                finally:
                    with self._lock:
                        self._active.discard(key)
        finally:
            if cap is not None:
                cap.release()

    def _extract_one(self, key, cap=None):
        """Extract, persist, and emit a single thumbnail.

        Strategy is chosen by source resolution (see _compute_prefer_ffmpeg):
          - normal footage → persistent OpenCV seek+read (~5-10ms), the fast
            path; `cap` is the worker's reusable VideoCapture and is returned so
            it can be reused for the next frame.
          - VR / very-high-res → ffmpeg keyframe seek with in-decode crop/scale,
            so a huge frame never reaches Python.
        The other path is used as a fallback if the preferred one produces
        nothing. Returns the (possibly newly-opened) OpenCV capture, or `cap`
        unchanged.
        """
        time_ms, height = key
        out_path = self._disk_path(key)

        frame = None
        if self._prefer_ffmpeg:
            if self._extract_via_ffmpeg(time_ms, height, out_path):
                frame = cv2.imread(str(out_path))
            if frame is None:
                frame, cap = self._extract_via_opencv(time_ms, height, cap)
                self._write_disk(out_path, frame)
        else:
            frame, cap = self._extract_via_opencv(time_ms, height, cap)
            if frame is not None:
                self._write_disk(out_path, frame)
            elif self._extract_via_ffmpeg(time_ms, height, out_path):
                frame = cv2.imread(str(out_path))

        if frame is None:
            return cap

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        self._image_ready.emit(time_ms, height, qimg)
        return cap

    def _write_disk(self, out_path: Path, frame):
        if frame is None:
            return
        try:
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        except Exception as e:
            print(f"⚠️ ThumbnailCache disk write failed: {e}")

    def _extract_via_ffmpeg(self, time_ms: int, height: int, out_path: Path) -> bool:
        """Decode one keyframe with ffmpeg, scaled (and VR-cropped) to `height`.

        Three things keep this fast enough for 8K VR (~0.4s vs ~4-5s naively):
          - `-ss` before `-i` seeks the demuxer to the nearest keyframe;
          - `-noaccurate_seek` stops ffmpeg decoding the rest of the GOP just to
            land on the exact timestamp — a filmstrip thumb doesn't need it;
          - `-skip_frame nokey` makes the decoder decode *only* keyframes, so we
            pay for one 8K intra frame instead of a whole run of them.
        The crop/scale run inside the decode filter, so a full 8K frame is never
        handed back to Python. The thumb snaps to the nearest keyframe, which is
        imperceptible at 60px.
        """
        if not self._ffmpeg:
            return False

        seconds = max(0.0, time_ms / 1000.0)
        if self._vr_mode:
            # Crop the left eye first, then scale — half the pixels to scale.
            vf = f"crop=iw/2:ih:0:0,scale=-2:{height}"
        else:
            vf = f"scale=-2:{height}"

        # Try each hardware-decode candidate, then software (None). The first
        # that yields a frame gets pinned so later frames don't re-probe the
        # ones that don't work here (software + skip_frame is still ~5x faster
        # than the old path, so the software fallback is never a disaster).
        attempts = list(self._hwaccels) + [None]
        for hw in attempts:
            cmd = [self._ffmpeg, "-nostdin", "-v", "error"]
            if hw:
                cmd += ["-hwaccel", hw]
            cmd += [
                "-ss", f"{seconds:.3f}",
                "-noaccurate_seek",
                "-skip_frame", "nokey",
                "-i", self.video_path,
                "-frames:v", "1",
                "-vf", vf,
                "-q:v", "5",
                "-y", str(out_path),
            ]
            if self._run_ffmpeg(cmd) and out_path.exists():
                # Pin the winner so later frames skip the candidates that failed:
                # [hw] for hardware, [] when only software worked (i.e. every
                # hardware candidate failed — no point re-spawning doomed GPU
                # attempts on every future frame).
                self._hwaccels = [hw] if hw else []
                return True

        # Not even software produced a frame (bad seek / unreadable). Leave the
        # candidate list untouched and let the caller's OpenCV fallback try.
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

    def _extract_via_opencv(self, time_ms: int, height: int, cap=None):
        """Full-frame decode + resize using a reusable VideoCapture.

        `cap` is the worker's persistent capture (or None to open one). Reusing
        it across frames turns each thumbnail into a cheap seek+read instead of
        paying VideoCapture open cost every time. Returns (frame_or_None, cap)
        so the caller can keep the (possibly newly-opened) capture alive.
        """
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                return None, None

        cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
        ok, frame = cap.read()

        if not ok or frame is None:
            return None, cap

        h, w = frame.shape[:2]
        if self._vr_mode:
            frame = frame[:, : w // 2]
            w = w // 2
        if h != height:
            new_w = max(1, int(round(w * height / h)))
            frame = cv2.resize(frame, (new_w, height), interpolation=cv2.INTER_AREA)
        return frame, cap

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
            # Bulk look-ahead: queue behind anything on screen. When one of
            # these clips scrolls into view its paint() re-requests at
            # PRIORITY_VISIBLE and promotes the frame to the front.
            self.request(t, height_px, priority=PRIORITY_PREFETCH)

    @Slot(int, int, QImage)
    def _on_image_ready(self, time_ms: int, height: int, qimg: QImage):
        """Runs on the main thread."""
        pix = QPixmap.fromImage(qimg)
        if pix.isNull():
            return
        key = (time_ms, height)
        self._add_to_mem(key, pix)
        self.thumbnail_ready.emit(time_ms / 1000.0, height, pix)