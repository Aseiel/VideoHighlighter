"""A decoder that outlives the frame it was asked for.

Extracting a thumbnail used to mean starting ffmpeg, opening a 5 GB file,
decoding one keyframe and throwing all of it away — 356ms, of which the decode
was the smallest part. Measured on 5760×2880:

    one ffmpeg process per thumbnail        356ms
    kept open, software decode              156ms
    kept open, d3d11va                       54ms

The hardware decoder is worth about 3× *once the device is created once* rather
than per frame. That is the whole reason the cache prefers software for its
per-process path: there, setting the GPU up costs more than it saves. Here it
does not, because nothing is set up twice.

Why a child process rather than decoding in the app
---------------------------------------------------
The subprocess design being replaced had one property worth keeping: an ffmpeg
that fell over took nothing with it. Handing an 8K stream and a GPU device to
an in-process decoder trades that away, and it would be traded away exactly
while a crash that leaves no traceback is still unexplained — the filmstrip
loading is when it happens.

So the decoder keeps its own process. It holds the open file and the GPU device
across every request, which is where the speed comes from, and when it dies it
dies alone: the parent notices, says so, starts another, and the app carries on
with one missing thumbnail. `restarts` counts how often that happened, because
a decoder that keeps dying is worth knowing about even when nothing crashes.

One decoder is shared per video path. Both timelines cache thumbnails from the
same file, and two children would mean two open files and two GPU devices for
one video.
"""

from __future__ import annotations

import heapq
import itertools
import multiprocessing
import threading

try:
    from modules import repaint_trace
except Exception:  # pragma: no cover - the decoder works without the recorder
    repaint_trace = None


# A request must not be able to hang a worker thread forever. Generous next to
# a 54ms decode: this is the "the child is wedged" threshold, not a deadline.
REQUEST_TIMEOUT = 30.0

# Hardware decoders to try, best first, then software. Same list as the cache's
# per-process path uses, for the same reason: these are OS-level decode APIs
# rather than vendor SDKs, so they work whoever made the GPU.
HWACCEL_CANDIDATES = ("d3d11va", "dxva2", "videotoolbox", "vaapi")


def thumb_width(src_w: int, src_h: int, height: int, vr: bool) -> int:
    """Width of the thumbnail a source of this shape yields at `height`.

    Even, because the scalers want it, and matching what the cache's ffmpeg
    path produces with `scale=-2:h` so the two paths are interchangeable.
    """
    if src_w <= 0 or src_h <= 0 or height <= 0:
        return 2
    eye_w = src_w / 2 if vr else src_w
    width = int(round(eye_w * height / src_h / 2)) * 2
    return max(2, width)


# ── the child ────────────────────────────────────────────────────────────

def _serve(conn, video_path: str) -> None:
    """Answer thumbnail requests until told to stop. Runs in the child."""
    try:
        container, stream, decoder = _open(video_path)
    except Exception as e:
        try:
            conn.send(("fatal", f"{type(e).__name__}: {e}"))
        except Exception:
            pass
        return

    try:
        while True:
            request = conn.recv()
            if request is None:
                break
            time_ms, height, vr, out_path = request
            try:
                _write_thumbnail(container, stream, time_ms, height, vr, out_path)
                conn.send(("ok", decoder))
            except Exception as e:
                conn.send(("error", f"{type(e).__name__}: {e}"))
    except (EOFError, OSError):
        pass                      # parent went away; nothing to report to
    finally:
        try:
            container.close()
        except Exception:
            pass


def _open(video_path: str):
    """Open the file with the best decoder that actually works. → (container, stream, name)"""
    import av

    try:
        from av.codec.hwaccel import HWAccel
    except Exception:
        HWAccel = None

    if HWAccel is not None:
        for device in HWACCEL_CANDIDATES:
            try:
                accel = HWAccel(device_type=device, allow_software_fallback=False)
                container = av.open(video_path, hwaccel=accel)
                stream = container.streams.video[0]
                stream.codec_context.skip_frame = "NONKEY"
                return container, stream, device
            except Exception:
                continue

    container = av.open(video_path)
    stream = container.streams.video[0]
    # Only the keyframes are ever wanted, and decoding the frames between them
    # to reach one is the cost this whole module exists to avoid.
    stream.codec_context.skip_frame = "NONKEY"
    stream.thread_type = "AUTO"
    return container, stream, "software"


def _write_thumbnail(container, stream, time_ms, height, vr, out_path) -> None:
    import cv2

    container.seek(int((time_ms / 1000.0) / stream.time_base),
                   stream=stream, any_frame=False, backward=True)
    frame = next(container.decode(video=0))

    width = thumb_width(frame.width, frame.height, height, vr)
    # Scale the whole frame down first and take the left half of the result:
    # half of a uniformly scaled picture is the same as scaling the half, and
    # this way the full-size frame is never converted to BGR — that conversion
    # alone measured 30ms of the 180 before it was removed.
    total = width * 2 if vr else width
    image = frame.reformat(width=total, height=height, format="bgr24").to_ndarray()
    if vr:
        image = image[:, :width]
    if not cv2.imwrite(str(out_path), image, [cv2.IMWRITE_JPEG_QUALITY, 60]):
        raise OSError(f"could not write {out_path}")


# ── the parent ───────────────────────────────────────────────────────────

class _Turnstile:
    """One request at a time, most urgent first.

    A plain lock would be enough to keep the child to one frame at a time, but
    it hands out its turn in whatever order threads happen to arrive. That is
    the difference between a hover taking 70ms and taking 300: the frame under
    the cursor waits behind however many filmstrip slots were queued first,
    each 54ms, and the queue is deepest exactly when someone is scrubbing
    across a strip that is still loading.

    So waiters are ordered by the same priority the cache queued them with,
    with a sequence number to keep equals in arrival order.
    """

    def __init__(self):
        self._condition = threading.Condition()
        self._waiting: list = []
        self._busy = False
        self._sequence = itertools.count()

    def acquire(self, priority: int) -> None:
        with self._condition:
            ticket = (priority, next(self._sequence))
            heapq.heappush(self._waiting, ticket)
            while self._busy or self._waiting[0] != ticket:
                self._condition.wait()
            heapq.heappop(self._waiting)
            self._busy = True

    def release(self) -> None:
        with self._condition:
            self._busy = False
            self._condition.notify_all()


class PersistentDecoder:
    """One child process holding one video open. Thread-safe."""

    def __init__(self, video_path: str):
        self.video_path = str(video_path)
        self.decoder = None          # which decoder the child reported using
        self.restarts = 0
        self.available = True        # cleared when the child cannot be started
        self._process = None
        self._conn = None
        self._closed = False
        self._turnstile = _Turnstile()
        self._lock = threading.Lock()

    def extract(self, time_ms: int, height: int, vr: bool, out_path,
                priority: int = 0) -> bool:
        """Write one thumbnail. False means the caller should fall back.

        The child decodes one frame at a time — at 54ms that already outruns
        the four-process pipeline it replaces, and a second child would mean a
        second GPU device for one video. `priority` decides who gets the next
        turn, so the frame under the cursor is not stuck behind a screenful of
        filmstrip slots.
        """
        if not self.available or self._closed:
            return False
        self._turnstile.acquire(priority)
        try:
            return self._request(time_ms, height, vr, out_path)
        finally:
            self._turnstile.release()

    def _request(self, time_ms: int, height: int, vr: bool, out_path) -> bool:
        with self._lock:
            if not self._start():
                return False
            try:
                self._conn.send((int(time_ms), int(height), bool(vr), str(out_path)))
                if not self._conn.poll(REQUEST_TIMEOUT):
                    self._fail(f"no answer in {REQUEST_TIMEOUT:.0f}s")
                    return False
                status, detail = self._conn.recv()
            except (EOFError, OSError, BrokenPipeError) as e:
                # The isolation earning its keep: the decoder died, the app did
                # not, and the next request gets a fresh one.
                self._fail(f"{type(e).__name__}: {e}")
                return False
            if status == "ok":
                if self.decoder != detail:
                    self.decoder = detail
                    print(f"🎞 Thumbnail decoder: {detail}")
                    _note("thumb.decoder", using=detail, video=self.video_path)
                return True
            if status == "fatal":
                self._fail(detail, permanent=True)
            return False

    def stop(self) -> None:
        """Shut the child down for good.

        Closing has to be one-way. Stopping the decoder while a worker thread
        is mid-request breaks its pipe, which reads exactly like a decoder that
        died -- and answering that by starting a replacement, during shutdown,
        spawns a child the interpreter is in no state to create.
        """
        with self._lock:
            self._closed = True
            self._shutdown()

    # ── internals ──

    def _start(self) -> bool:
        if self._closed:
            return False
        process = self._process
        if process is not None and process.is_alive():
            return True
        if process is not None:
            # Started once, gone now: it died between requests. That is the
            # same event as dying during one and has to be counted the same,
            # or `restarts` reads zero for a decoder that dies every time —
            # which is precisely the log line somebody would be reading.
            self._died(f"exited with code {process.exitcode}")
        self._shutdown()
        try:
            context = multiprocessing.get_context("spawn")
            parent_conn, child_conn = context.Pipe(duplex=True)
            process = context.Process(
                target=_serve, args=(child_conn, self.video_path),
                name="ThumbnailDecoder", daemon=True)
            process.start()
            child_conn.close()      # the child holds the only other end now
            self._process, self._conn = process, parent_conn
            return True
        except Exception as e:
            print(f"⚠️ Thumbnail decoder would not start: {e}")
            _note("thumb.decoder_unavailable", error=f"{type(e).__name__}: {e}")
            self.available = False
            return False

    def _died(self, reason: str, permanent: bool = False) -> None:
        """Account for one death, however it was noticed."""
        self.restarts += 1
        print(f"⚠️ Thumbnail decoder stopped ({reason}) — "
              f"{'giving up on it' if permanent else 'starting another'}")
        _note("thumb.decoder_died", reason=reason, restarts=self.restarts,
              video=self.video_path)

    def _fail(self, reason: str, permanent: bool = False) -> None:
        self._died(reason, permanent)
        self._shutdown()
        if permanent:
            self.available = False

    def _shutdown(self) -> None:
        if self._conn is not None:
            try:
                self._conn.send(None)
            except Exception:
                pass
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None
        if self._process is not None:
            try:
                self._process.join(timeout=2.0)
                if self._process.is_alive():
                    self._process.terminate()
            except Exception:
                pass
            self._process = None


def _note(event: str, **fields) -> None:
    if repaint_trace is not None:
        repaint_trace.note(event, **fields)


# ── shared, per video ────────────────────────────────────────────────────

_decoders: dict = {}
_users: dict = {}
_registry_lock = threading.Lock()


def acquire(video_path: str):
    """The decoder for this video, starting one if nobody holds it yet."""
    key = str(video_path)
    with _registry_lock:
        decoder = _decoders.get(key)
        if decoder is None:
            decoder = _decoders[key] = PersistentDecoder(key)
            _users[key] = 0
        _users[key] += 1
        return decoder


def release(video_path: str) -> None:
    """Let go. The child is stopped once nothing is using it."""
    key = str(video_path)
    with _registry_lock:
        if key not in _users:
            return
        _users[key] -= 1
        if _users[key] > 0:
            return
        decoder = _decoders.pop(key, None)
        _users.pop(key, None)
    if decoder is not None:
        decoder.stop()
