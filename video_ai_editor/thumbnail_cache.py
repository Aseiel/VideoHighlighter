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
import time
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

try:
    from modules import repaint_trace
except Exception:  # pragma: no cover - the cache works without the recorder
    repaint_trace = None


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

# How many extractions may run at once across *every* cache in the process.
# There are two caches (the signal timeline's filmstrip and the edit
# timeline's clips), each with its own workers, so the app was running six
# concurrent 8K decodes. Measured on 5760×2880: throughput is 2.7 thumbs/s at
# one worker, 6.4 at three, 6.9 at four and 6.9 at six — it saturates at three
# to four, and the extra two buy nothing while doubling the latency of every
# frame (0.52s → 0.90s) and the memory in flight. A shared ceiling is the only
# way to bound it, because neither cache knows about the other.
MAX_CONCURRENT_EXTRACTIONS = 4
_extraction_slots = threading.Semaphore(MAX_CONCURRENT_EXTRACTIONS)

# Try the CPU before the GPU for thumbnails. `-skip_frame nokey` means exactly
# one intra frame is ever decoded, which a CPU does about as fast as a GPU can
# be set up for it, so the hardware path's per-process device creation is paid
# for nothing — and paid six times over, against the player's own decoder on
# the same GPU. Named rather than inlined so the claim stays measurable.
PREFER_SOFTWARE_DECODE = True


def _memory_mb() -> float:
    """Resident size of this process, or 0.0 when it cannot be read.

    Recorded with each traced extraction so a death by exhaustion leaves a
    rising number behind it rather than nothing at all.
    """
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return 0.0


class ThumbnailCache(QObject):
    """
    Per-video thumbnail cache. Create one instance per video.

    Signals:
        thumbnail_ready(time_seconds: float, height_px: int, pixmap: QPixmap)
            Emitted on the main thread after an async extraction finishes.
            Connect this to your clip item's update() or scene.invalidate().
    """

    # Internal: worker → main thread, carries a QImage (thread-safe). The crop
    # travels with the frame rather than being read from the cache on arrival,
    # because the flag can be toggled while an extraction is in flight and the
    # frame belongs to the mode it was decoded for.
    _image_ready = Signal(int, int, bool, QImage)  # time_ms, height_px, vr, image
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

        # In-memory LRU cache: (time_ms, height_px, vr_mode) → QPixmap.
        # The crop is part of the key so switching it is a lookup somewhere
        # else rather than a reason to throw everything away — see set_vr_mode.
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
        # (hwaccel, message) pairs already reported, so a decoder that fails on
        # every frame is explained once rather than a thousand times.
        self._reported_failures: set = set()

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
        self._src_w, self._src_h, self._src_duration = self._probe_src_size()
        self._prefer_ffmpeg = self._compute_prefer_ffmpeg()
        self._gaps = 0

        # A decoder that stays open, shared with any other cache on the same
        # video. Acquired on first need rather than here: whether it is worth a
        # child process depends on _prefer_ffmpeg, and that can turn on later —
        # a modest side-by-side video sits below the resolution threshold until
        # the VR crop is switched on, and would otherwise never get one.
        self._decoder = None

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

    def frame_aspect(self) -> float | None:
        """Aspect ratio of the pixmaps this cache hands out, or None if unknown.

        The VR crop changes it — a side-by-side frame is 2:1, one eye of it is
        square — and everything that lays thumbnails out in slots needs the
        shape of what will actually arrive, or it sizes the slot for a picture
        that never comes and letterboxes or crops the one that does.
        """
        if self._src_w <= 0 or self._src_h <= 0:
            return None
        width = self._src_w / 2 if self._vr_mode else self._src_w
        return width / self._src_h

    def set_vr_mode(self, enabled: bool):
        """Enable/disable the VR half-frame crop.

        Nothing is thrown away. Both crops are cached side by side — the key
        and the file name carry the mode — so flipping the checkbox is a lookup
        against a different set of keys, and flipping it back finds the frames
        that were already there. It used to clear memory and delete every
        thumbnail on disk, which made each toggle re-extract the whole visible
        strip, and on VR footage that is exactly where the cost is.

        In-flight extractions are left alone for the same reason: each carries
        the crop it was queued for, so the work still lands somewhere useful.
        """
        if self._vr_mode == enabled:
            return
        self._vr_mode = enabled
        # VR footage is high-res side-by-side, so it flips us onto the ffmpeg
        # keyframe path (which also does the left-eye crop inside the decode).
        self._prefer_ffmpeg = self._compute_prefer_ffmpeg()

    def stop(self):
        """Stop the worker threads. Call on app shutdown."""
        self._stopped = True
        if self._decoder is not None:
            self._decoder = None
            try:
                from . import thumbnail_decoder
                # Shared per video: the child only stops once the other
                # timeline's cache has let go of it too.
                thumbnail_decoder.release(self.video_path)
            except Exception:
                pass

    # ── Internals ─────────────────────────────────────────────────────────

    def _make_key(self, time_seconds: float, height_px: int) -> tuple:
        time_ms = int(time_seconds * 1000)
        time_ms = (time_ms // TIME_QUANT_MS) * TIME_QUANT_MS
        return (time_ms, int(height_px), bool(self._vr_mode))

    def _disk_path(self, key: tuple) -> Path:
        # The suffix keeps the two crops apart on disk. Full-frame thumbs keep
        # the name they always had, so an existing cache stays usable.
        time_ms, height, vr = key
        suffix = "_left" if vr else ""
        return self.disk_dir / f"{time_ms}_{height}{suffix}.jpg"

    def peek_nearest(self, time_seconds: float, max_delta: float = 1.5):
        """The closest frame already in memory, at any height, or None.

        Never decodes and never queues: this is only ever a stand-in to put on
        screen *now*, while the frame actually wanted is extracted.

        It exists because the cache key is (time, height) and the same moment
        is routinely already decoded at a different size — the filmstrips ask
        for ~54px thumbs all along a clip, the hover popup asks for 180px, and
        those never share a key. So the frame a hover wants has usually been
        read off disk seconds ago at another size, and showing it upscaled beats
        showing the word "loading".

        Ranked by distance in time first and height second, so an exact-moment
        frame at the wrong size always beats a nearby moment at the right one:
        soft is a much smaller lie than showing a different part of the video.
        """
        target_ms = int(time_seconds * 1000)
        window_ms = int(max_delta * 1000)
        best = None
        best_rank = None
        # A snapshot, matching how request() reads _mem: the dict is only
        # mutated from the GUI thread, and a stale entry here would at worst
        # cost one stand-in.
        for (t_ms, height, vr), pix in list(self._mem.items()):
            if vr != self._vr_mode:
                continue        # the other crop is a different picture, not a size
            delta = abs(t_ms - target_ms)
            if delta > window_ms:
                continue
            rank = (delta, -int(height))
            if best_rank is None or rank < best_rank:
                best_rank, best = rank, pix
        return best

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
        """Read source width/height/duration cheaply (metadata, no decode).

        The duration matters as much as the size: a timeline is as long as the
        analysis cache says, which is not always to the frame what the file
        says, and a request past the last keyframe produces no packets and no
        frame — the expensive way to learn nothing.
        """
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                return (0, 0, 0.0)
            try:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
            finally:
                cap.release()
            duration = frames / fps if fps > 0 and frames > 0 else 0.0
            return (w, h, duration)
        except Exception:
            return (0, 0, 0.0)

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
                    cap = self._extract_one(key, cap, priority)
                except Exception as e:
                    print(f"⚠️ ThumbnailCache extract failed: {e}")
                finally:
                    with self._lock:
                        self._active.discard(key)
        finally:
            if cap is not None:
                cap.release()

    def _extract_one(self, key, cap=None, priority: int = PRIORITY_VISIBLE):
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
        time_ms, height, vr = key
        out_path = self._disk_path(key)

        # Only the expensive path is recorded, which is the same test that
        # selects it: VR and very-high-res sources. Ordinary footage would bury
        # the trace in thousands of uninteresting lines, and it is not where
        # the crash with no traceback happens — a filmstrip loading 8K
        # side-by-side is. A `thumb.begin` with no `thumb.end` after it says
        # the process died inside this extraction, and names the frame and the
        # decoder it was using when it did.
        traced = self._prefer_ffmpeg and repaint_trace is not None
        if traced:
            repaint_trace.note("thumb.begin", t_ms=time_ms, h=height, vr=vr,
                               src=f"{self._src_w}x{self._src_h}",
                               hw=(self._hwaccels[0] if self._hwaccels else "sw"),
                               worker=threading.current_thread().name,
                               rss_mb=_memory_mb())
        started = time.monotonic()
        # The process-wide ceiling. Taken around the decode only, so a worker
        # waiting for a slot is not holding one.
        with _extraction_slots:
            try:
                cap = self._extract_one_frame(key, out_path, cap, priority)
            finally:
                if traced:
                    repaint_trace.note(
                        "thumb.end", t_ms=time_ms, h=height,
                        ms=round((time.monotonic() - started) * 1000, 1))
        return cap

    def _extract_one_frame(self, key, out_path: Path, cap=None,
                           priority: int = PRIORITY_VISIBLE):
        """The decode itself; see _extract_one for the strategy it implements."""
        time_ms, height, vr = key

        frame = None
        if self._prefer_ffmpeg:
            # The persistent decoder first: it keeps the file open and the GPU
            # device created between frames, which is where the cost of a
            # thumbnail actually lives. Falls through to spawning ffmpeg when
            # it is unavailable or its child has just died.
            if self._extract_via_decoder(time_ms, height, out_path, vr, priority):
                frame = cv2.imread(str(out_path))
            if frame is None and self._extract_via_ffmpeg(time_ms, height, out_path, vr):
                frame = cv2.imread(str(out_path))
            if frame is None:
                # Deliberately no OpenCV fallback on this path. It decodes the
                # *whole* frame in-process — 47 MB at 5760×2880, more at 8K —
                # and every worker doing it at once is both the slowest and the
                # heaviest thing this module can do: measured at 12–17 seconds
                # per thumbnail when the ffmpeg grab was failing. A slot that
                # stays a placeholder costs the viewer one missing thumbnail;
                # this costs them the strip, and possibly the process.
                self._report_extraction_gap(time_ms, height)
        else:
            frame, cap = self._extract_via_opencv(time_ms, height, cap, vr)
            if frame is not None:
                self._write_disk(out_path, frame)
            elif self._extract_via_ffmpeg(time_ms, height, out_path, vr):
                frame = cv2.imread(str(out_path))

        if frame is None:
            return cap

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
        self._image_ready.emit(time_ms, height, vr, qimg)
        return cap

    def _write_disk(self, out_path: Path, frame):
        if frame is None:
            return
        try:
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
        except Exception as e:
            print(f"⚠️ ThumbnailCache disk write failed: {e}")

    def _ensure_decoder(self):
        """The persistent decoder for this video, started on first need."""
        if self._decoder is not None or not self._prefer_ffmpeg:
            return self._decoder
        with self._lock:
            if self._decoder is None:
                try:
                    from . import thumbnail_decoder
                    self._decoder = thumbnail_decoder.acquire(self.video_path)
                except Exception as e:      # pragma: no cover - defensive
                    print(f"⚠️ No persistent thumbnail decoder: {e}")
        return self._decoder

    def _extract_via_decoder(self, time_ms: int, height: int, out_path: Path,
                             vr: bool, priority: int = PRIORITY_VISIBLE) -> bool:
        """Ask the persistent decoder for one frame. False → use ffmpeg."""
        decoder = self._ensure_decoder()
        if decoder is None:
            return False
        # The priority the request was queued with decides its turn at the
        # decoder too, or a hover waits behind every filmstrip slot ahead of it.
        return decoder.extract(self._seekable_seconds(time_ms) * 1000.0,
                               height, vr, out_path, priority=priority)

    def _extract_via_ffmpeg(self, time_ms: int, height: int, out_path: Path,
                            vr: bool) -> bool:
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

        seconds = self._seekable_seconds(time_ms)
        if vr:
            # Crop the left eye first, then scale — half the pixels to scale.
            vf = f"crop=iw/2:ih:0:0,scale=-2:{height}"
        else:
            vf = f"scale=-2:{height}"

        # Software first, hardware as the fallback — the reverse of what this
        # used to do. Measured on 5760×2880 across six workers, software ran
        # 8.5 thumbs/s against 6.9 for d3d11va: the hardware attempt is not
        # merely not faster, it costs a D3D11 device *per process*, six of them
        # competing with the player's own decoder for the same GPU. That is
        # where "Failed setup for format d3d11: hwaccel initialisation returned
        # error" comes from on every seek. Hardware stays in the list for a
        # source software cannot handle.
        hardware = list(self._hwaccels)
        attempts = ([None] + hardware if PREFER_SOFTWARE_DECODE
                    else hardware + [None])
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
            ok, error = self._run_ffmpeg(cmd)
            if ok and out_path.exists():
                # Pin the winner so later frames skip what failed here: [] when
                # software worked (the normal case now), [hw] when it took
                # hardware to get a frame at all.
                self._hwaccels = [hw] if hw else []
                return True
            self._report_ffmpeg_failure(hw, error)

        # Nothing produced a frame (bad seek / unreadable). Leave the candidate
        # list untouched and let the caller's OpenCV fallback try.
        return False

    def _seekable_seconds(self, time_ms: int) -> float:
        """The requested moment, pulled back inside the file if it sits past it.

        The last slot of a filmstrip is at the end of the *timeline*, whose
        length comes from the analysis cache; when that runs even slightly long
        the seek lands past the final keyframe and ffmpeg returns nothing. The
        nearest real frame is a far better answer than an empty slot, and it
        costs nothing to ask for.
        """
        seconds = max(0.0, time_ms / 1000.0)
        if self._src_duration > 0:
            seconds = min(seconds, max(0.0, self._src_duration - 0.5))
        return seconds

    def _report_extraction_gap(self, time_ms: int, height: int):
        """A frame this source could not produce. Said once, then counted."""
        self._gaps += 1
        if self._gaps == 1:
            print(f"⚠️ No thumbnail at {time_ms / 1000:.1f}s "
                  f"({self._src_w}x{self._src_h}) — leaving the slot empty "
                  f"rather than decoding the whole frame")
        if repaint_trace is not None:
            repaint_trace.note("thumb.gap", t_ms=time_ms, h=height,
                               total=self._gaps)

    def _report_ffmpeg_failure(self, hw, error: str):
        """Say why an extraction failed — once per distinct reason.

        These used to go to DEVNULL, which is how a filmstrip could spend its
        time failing over from a decoder that never works on this machine
        without anyone being able to tell. Once per message, because the same
        failure repeats for every frame and a line each would bury everything
        else in the log.
        """
        message = (error or "").strip().splitlines()
        message = message[-1] if message else "no output"
        key = (hw, message)
        with self._lock:
            if key in self._reported_failures:
                return
            self._reported_failures.add(key)
        via = hw or "software"
        print(f"⚠️ Thumbnail decode via {via} failed: {message}")
        if repaint_trace is not None:
            repaint_trace.note("thumb.decoder_failed", via=via, error=message)

    def _run_ffmpeg(self, cmd: list) -> tuple:
        """Run one extraction. → (succeeded, stderr text)."""
        creationflags = 0
        if sys.platform.startswith("win"):
            # Keep console windows from flashing for every extraction.
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            r = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=30,
                creationflags=creationflags,
            )
            error = (r.stderr or b"").decode("utf-8", "replace")
            return (r.returncode == 0, error)
        except Exception as e:
            return (False, f"{type(e).__name__}: {e}")

    def _extract_via_opencv(self, time_ms: int, height: int, cap=None,
                            vr: bool = False):
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
        if vr:
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

    @Slot(int, int, bool, QImage)
    def _on_image_ready(self, time_ms: int, height: int, vr: bool, qimg: QImage):
        """Runs on the main thread."""
        pix = QPixmap.fromImage(qimg)
        if pix.isNull():
            return
        key = (time_ms, height, vr)
        self._add_to_mem(key, pix)
        self.thumbnail_ready.emit(time_ms / 1000.0, height, pix)