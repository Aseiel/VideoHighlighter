"""
Display frames sized for the screen, not the source.

QGraphicsVideoItem paints by calling QVideoFrame.toImage() on the UI thread,
at the source's native resolution, for every frame, and then scales the
result down to the view. Measured on this pipeline (Arc A750, FFmpeg backend,
which hands frames over as CPU-side P010 with no texture handle):

    7260x3630 P010   toImage ~85 ms   ->  10 fps painted on a 30 fps source
    3840x2160 P010   toImage ~30 ms   ->  27 fps, UI thread busy ~100%

That's with no live detection running at all — the preview alone can't keep
up, and every live worker's UI-thread work queues behind it.

DisplayFrameFeed sits between the player and the video item. Each frame goes
to one background thread, which shrinks it to the width the view can actually
show and presents that as an 8-bit NV12 frame carrying the source's colour
metadata — so Qt still does the colour conversion, just on ~1/30th of the
pixels. The shrink works on the raw Y/UV planes before any colour math
(downscale_nv12_planes): ~14 ms for a 7K frame, ~4 ms for 4K, with the GIL
released inside OpenCV. Only the latest frame is kept; if the thread falls
behind, frames are dropped rather than queued, so the UI never waits.

The live workers tap the video item's sink, so they now receive the shrunk
frame too and convert ~1280 px instead of ~7000 px each.

The first map() of a hardware-decoded frame is where the decoder copies it
from the GPU (~25 ms for 7K), and PySide holds the GIL for the whole call —
which starved every other Python thread, the live workers and the UI's slots
included. _map_readonly makes the same call through ctypes, which releases
the GIL; that is also what lets two shrink threads overlap usefully.

Frames that are already small, aren't NV12/P010, or are HDR (PQ/HLG — the
8-bit copy would lose what tone mapping needs) pass through untouched.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from typing import Optional

import numpy as np

from PySide6.QtCore import QObject, QSize, Signal, Slot
from PySide6.QtMultimedia import QVideoFrame, QVideoFrameFormat, QVideoSink


def _native_map_fn():
    """QVideoFrame::map(MapMode) from the Qt library PySide6 loaded, callable
    through ctypes (which drops the GIL for the call), or None."""
    try:
        import PySide6
        import shiboken6
        if sys.platform == "win32":
            lib = ctypes.CDLL(os.path.join(os.path.dirname(PySide6.__file__), "Qt6Multimedia.dll"))
            fn = lib["?map@QVideoFrame@@QEAA_NW4MapMode@1@@Z"]
        else:
            fn = ctypes.CDLL(None)["_ZN11QVideoFrame3mapENS_7MapModeE"]
        fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
        fn.restype = ctypes.c_bool
        return lambda vframe, mode: fn(shiboken6.getCppPointer(vframe)[0], int(mode.value))
    except Exception as e:
        print(f"ℹ️ display_frames: no GIL-free QVideoFrame.map ({e}); using PySide's")
        return None


_native_map = _native_map_fn()


def _map_readonly(vframe) -> bool:
    if _native_map is not None:
        return _native_map(vframe, QVideoFrame.MapMode.ReadOnly)
    return vframe.map(QVideoFrame.MapMode.ReadOnly)


def downscale_nv12_planes(vframe, target_w: int, crop_left_half: bool = False):
    """Shrink a P010/NV12 QVideoFrame to ``target_w`` wide (optionally cropping
    to the left half first), returning 8-bit ``(y, uv)`` planes — ``y`` is
    (h, w), ``uv`` is (h/2, w/2, 2) interleaved — or None for any other pixel
    format or a frame that won't map.

    Reads the mapped planes in place (no copy) and resizes them at their native
    bit depth, so the 10->8 bit shift and the caller's colour conversion run on
    the small frame only. Copying the full planes out with bytes() and shifting
    them at native size was ~85% of the old cost on a 7K source.
    """
    import cv2

    fmt = vframe.pixelFormat()
    is_p010 = fmt == QVideoFrameFormat.PixelFormat.Format_P010
    if not is_p010 and fmt != QVideoFrameFormat.PixelFormat.Format_NV12:
        return None
    w, h = vframe.width(), vframe.height()
    crop_w = (w // 2) if crop_left_half else w
    crop_w -= crop_w % 2
    if w <= 0 or h <= 1 or crop_w <= 0:
        return None
    tw = max(2, min(target_w, crop_w))
    tw -= tw % 2
    th = max(2, int(round(h * tw / crop_w)))
    th -= th % 2
    if not _map_readonly(vframe):
        return None
    try:
        dt, bpp = (np.uint16, 2) if is_p010 else (np.uint8, 1)
        y = np.frombuffer(vframe.bits(0), dtype=dt).reshape(h, vframe.bytesPerLine(0) // bpp)
        uv = np.frombuffer(vframe.bits(1), dtype=dt).reshape(h // 2, vframe.bytesPerLine(1) // bpp)
        y = y[:, :crop_w]
        uv = uv[:, :crop_w].reshape(h // 2, crop_w // 2, 2)
        y_small = cv2.resize(y, (tw, th), interpolation=cv2.INTER_AREA)
        uv_small = cv2.resize(uv, (tw // 2, th // 2), interpolation=cv2.INTER_AREA)
    finally:
        vframe.unmap()
    if is_p010:
        y_small = (y_small >> 8).astype(np.uint8)
        uv_small = (uv_small >> 8).astype(np.uint8)
    return y_small, uv_small


_HDR_TRANSFERS = {
    QVideoFrameFormat.ColorTransfer.ColorTransfer_ST2084,
    QVideoFrameFormat.ColorTransfer.ColorTransfer_STD_B67,
}


def shrink_for_display(vframe, target_w: int):
    """A small NV12 copy of ``vframe`` for painting, or ``vframe`` itself when
    shrinking wouldn't help (already near ``target_w``, other pixel format, HDR)."""
    w, h = vframe.width(), vframe.height()
    if w <= 0 or w <= target_w * 1.15:
        return vframe
    src_fmt = vframe.surfaceFormat()
    if src_fmt.colorTransfer() in _HDR_TRANSFERS:
        return vframe
    planes = downscale_nv12_planes(vframe, target_w)
    if planes is None:
        return vframe
    y, uv = planes
    th, tw = y.shape

    fmt = QVideoFrameFormat(QSize(tw, th), QVideoFrameFormat.PixelFormat.Format_NV12)
    fmt.setColorSpace(src_fmt.colorSpace())
    fmt.setColorTransfer(src_fmt.colorTransfer())
    fmt.setColorRange(src_fmt.colorRange())
    fmt.setStreamFrameRate(src_fmt.streamFrameRate())
    out = QVideoFrame(fmt)
    if not out.map(QVideoFrame.MapMode.WriteOnly):
        return vframe
    try:
        dst_y = np.frombuffer(out.bits(0), dtype=np.uint8).reshape(th, out.bytesPerLine(0))
        dst_uv = np.frombuffer(out.bits(1), dtype=np.uint8).reshape(th // 2, out.bytesPerLine(1))
        dst_y[:, :tw] = y
        dst_uv[:, :tw] = uv.reshape(th // 2, tw)
    finally:
        out.unmap()
    out.setStartTime(vframe.startTime())
    out.setEndTime(vframe.endTime())
    # Phone footage carries its orientation on the frame, not in the pixels.
    try:
        out.setRotation(vframe.rotation())
        out.setMirrored(vframe.mirrored())
    except AttributeError:
        pass
    return out


class DisplayFrameFeed(QObject):
    """Player -> (shrink on a background thread) -> video item.

    Usage::

        feed = DisplayFrameFeed(video_item.videoSink(), parent=self)
        player.setVideoSink(feed.input_sink)      # instead of setVideoOutput(video_item)
        feed.set_target_width(px)                 # whenever the view resizes
    """

    _frame_ready = Signal(object)

    def __init__(self, output_sink: QVideoSink, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._output_sink = output_sink
        self.input_sink = QVideoSink(self)
        self.input_sink.videoFrameChanged.connect(self._on_frame)
        self._frame_ready.connect(self._present)

        self._target_w = 1280
        self._cond = threading.Condition()
        self._pending = None
        self._seq = 0              # numbers frames in arrival order
        self._shown_seq = -1       # last one presented (UI thread only)
        self._stopped = False
        # Daemon threads rather than QThreads: they have no Qt timers or slots
        # to run, and can't hold up interpreter exit. Two, because one spends
        # most of a 7K frame's time waiting on the GPU copy in map(): while it
        # waits, the other resizes, which takes 7K from ~25 to 30 fps.
        self._threads = [
            threading.Thread(target=self._run, name=f"display-frames-{i}", daemon=True)
            for i in range(self._THREADS)
        ]
        for t in self._threads:
            t.start()

    _THREADS = 2

    def set_target_width(self, px: int) -> None:
        self._target_w = max(2, int(px))

    @Slot(object)
    def _on_frame(self, vframe):
        with self._cond:
            self._seq += 1
            self._pending = (self._seq, vframe)   # newer frame replaces any unprocessed one
            self._cond.notify()

    def _run(self):
        while True:
            with self._cond:
                while self._pending is None and not self._stopped:
                    self._cond.wait()
                if self._stopped:
                    return
                (seq, vframe), self._pending = self._pending, None
            try:
                out = shrink_for_display(vframe, self._target_w)
            except Exception as e:
                print(f"⚠️ DisplayFrameFeed: shrink failed ({e}), showing the frame as-is")
                out = vframe
            self._frame_ready.emit((seq, out))     # queued to the UI thread

    @Slot(object)
    def _present(self, item):
        seq, vframe = item
        # The other thread may have finished a newer frame first. Order by
        # arrival, not by timestamp — a backwards seek legitimately goes back.
        if seq < self._shown_seq:
            return
        self._shown_seq = seq
        self._output_sink.setVideoFrame(vframe)

    def shutdown(self) -> None:
        with self._cond:
            self._stopped = True
            self._cond.notify_all()
        for t in self._threads:
            t.join(timeout=0.5)
