"""Where the picture is, inside black bars.

A widescreen film is often stored letterboxed in a 16:9 frame, and a 4:3
programme pillarboxed in one. The bars never change, and anything that
measures change over the whole frame is diluted by them: a frame difference
averaged over every pixel is a quarter smaller when a quarter of the pixels
are bars, so a cut that cleared the bar on the picture alone falls under it.

Measured on a 20-minute film with the same picture given bars and not
(2026-11-18): the frame-difference shot scorer lost about a fifth of the real
cuts to letterbox and pillarbox bars alike and found nothing new; the learned
scorer lost about a tenth to pillarbox bars. So the shot passes read the
picture, not the frame.

How the area is found:

* Up to ``SAMPLES`` frames between 10% and 90% of the video, where the
  opening and closing titles and fades are not. A frame with nothing lit in
  it (a fade, a black frame) says nothing about where the picture ends and is
  skipped.
* A row is a bar row when its median brightness is black in **every** usable
  frame. The median and not the maximum, because subtitles and logos are
  often burned into the bars: text covers far less than half a row, so it
  does not lift the median, and the bars around it still count as bars.
  Every frame and not most, because a dark scene can make the top of the
  picture black in one frame; across the whole film it will not stay black.
* The same for columns.

Bars narrower than ``MIN_BAR_FRACTION`` of the frame are left alone (encoders
leave a dark pixel or two at the edges, and cropping those changes nothing),
and so is a "picture" smaller than ``MIN_PICTURE_FRACTION``: that is a dark
video, not a small picture in a big frame.

Never raises; an unreadable video is the whole frame (``None``).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import NamedTuple, Optional, Sequence

import numpy as np

SAMPLES = 20
SPAN = (0.10, 0.90)

# A row or column whose median brightness (0-255) is at or below this is
# black. Bars decode to 0-10 on lossy files; picture rarely sits this low
# across a whole row in every frame of a film.
BAR_LEVEL = 24

# A frame is usable when its brightest tenth is at least this bright: it has
# something lit in it, wherever the picture is.
LIT_LEVEL = 60

# Fewer usable frames than this and nothing is cropped.
MIN_USABLE = 3

MIN_BAR_FRACTION = 0.01
MIN_PICTURE_FRACTION = 0.4

# Frames are measured at this width at most; a 4K frame says no more about
# where its bars are than a 480-pixel one does.
MEASURE_WIDTH = 480


class PictureArea(NamedTuple):
    """The picture, in the video's own pixels: ``x, y, w, h`` of ``width x height``."""

    x: int
    y: int
    w: int
    h: int
    width: int
    height: int

    @property
    def fraction(self) -> float:
        """How much of the frame is picture."""
        return (self.w * self.h) / float(self.width * self.height)

    def crop(self, frame):
        """``frame`` cut to the picture. A frame of another size (already
        scaled or cropped elsewhere) is scaled to it first, by proportion."""
        fh, fw = frame.shape[:2]
        if (fw, fh) == (self.width, self.height):
            return frame[self.y:self.y + self.h, self.x:self.x + self.w]
        sx, sy = fw / self.width, fh / self.height
        x0, y0 = int(round(self.x * sx)), int(round(self.y * sy))
        x1 = max(x0 + 1, int(round((self.x + self.w) * sx)))
        y1 = max(y0 + 1, int(round((self.y + self.h) * sy)))
        return frame[y0:y1, x0:x1]

    def ffmpeg_filter(self) -> str:
        """The ffmpeg ``crop`` filter, as proportions of the input, so it holds
        whatever size ffmpeg decodes the video at."""
        return ("crop="
                f"iw*{self.w / self.width:.6f}:ih*{self.h / self.height:.6f}:"
                f"iw*{self.x / self.width:.6f}:ih*{self.y / self.height:.6f}")

    def describe(self) -> str:
        return (f"{self.w}x{self.h} at ({self.x},{self.y}) of "
                f"{self.width}x{self.height}")


def _edges(bar: np.ndarray) -> Optional[tuple[int, int]]:
    """``(first, last)`` index that is not bar, or None when all of it is."""
    on = np.flatnonzero(~bar)
    if on.size == 0:
        return None
    return int(on[0]), int(on[-1])


def area_in_frames(frames: Sequence[np.ndarray], rows_only: bool = False
                   ) -> Optional[tuple[int, int, int, int]]:
    """``(x, y, w, h)`` of the picture in these frames, or None for the whole.

    ``frames`` are BGR or grey, all the same size. ``rows_only`` crops only
    top and bottom bars, for footage whose columns mean something else (one
    eye of side-by-side VR is a half of the frame, not a picture with bars).
    """
    grey = []
    for f in frames:
        g = f if f.ndim == 2 else f.max(axis=2)
        if np.percentile(g, 90) >= LIT_LEVEL:
            grey.append(g)
    if len(grey) < MIN_USABLE:
        return None
    h, w = grey[0].shape
    row_bar = np.ones(h, dtype=bool)
    col_bar = np.ones(w, dtype=bool)
    for g in grey:
        row_bar &= np.median(g, axis=1) <= BAR_LEVEL
        col_bar &= np.median(g, axis=0) <= BAR_LEVEL
    rows, cols = _edges(row_bar), _edges(col_bar)
    if rows is None or cols is None:
        return None
    y0, y1 = rows
    x0, x1 = (0, w - 1) if rows_only else cols

    def trimmed(lo, hi, size):
        # Ignore a sliver of a bar on either side.
        if lo < size * MIN_BAR_FRACTION:
            lo = 0
        if size - 1 - hi < size * MIN_BAR_FRACTION:
            hi = size - 1
        return lo, hi

    y0, y1 = trimmed(y0, y1, h)
    x0, x1 = trimmed(x0, x1, w)
    ph, pw = y1 - y0 + 1, x1 - x0 + 1
    if ph < h * MIN_PICTURE_FRACTION or pw < w * MIN_PICTURE_FRACTION:
        return None
    if (ph, pw) == (h, w):
        return None
    return x0, y0, pw, ph


def _sample(video_path: str):
    """Up to ``SAMPLES`` frames from the middle of the video, measured small,
    with the video's own size. ``(frames, width, height)`` or None."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return None
        # A rotated phone video is turned by OpenCV and by ffmpeg each in its
        # own way; a crop measured in one may not be the other's. Leave it.
        if int(cap.get(getattr(cv2, "CAP_PROP_ORIENTATION_META", -1)) or 0) % 360:
            return None
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if width <= 0 or height <= 0 or count <= 0:
            return None
        scale = min(1.0, MEASURE_WIDTH / width)
        size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        frames = []
        for at in np.linspace(SPAN[0], SPAN[1], SAMPLES):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(count * at))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                return None              # not the size the header said
            if scale < 1.0:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            frames.append(frame)
        return frames, width, height
    finally:
        cap.release()


@lru_cache(maxsize=32)
def _find(video_path: str, size: int, mtime: float, rows_only: bool
          ) -> Optional[PictureArea]:
    sampled = _sample(video_path)
    if not sampled:
        return None
    frames, width, height = sampled
    found = area_in_frames(frames, rows_only=rows_only)
    if found is None:
        return None
    x, y, w, h = found
    mh, mw = frames[0].shape[:2]
    sx, sy = width / mw, height / mh
    # Back to the video's own pixels, one measured pixel inward on each
    # cropped side (the edge of a bar is soft once scaled), on even numbers.
    x0 = x if x == 0 else int(np.ceil((x + 1) * sx))
    y0 = y if y == 0 else int(np.ceil((y + 1) * sy))
    x1 = width if x + w == mw else int(np.floor((x + w - 1) * sx))
    y1 = height if y + h == mh else int(np.floor((y + h - 1) * sy))
    x0, y0 = x0 + (x0 % 2), y0 + (y0 % 2)
    pw, ph = (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2
    if pw <= 0 or ph <= 0:
        return None
    return PictureArea(x0, y0, pw, ph, width, height)


def find(video_path: str, rows_only: bool = False) -> Optional[PictureArea]:
    """The picture inside ``video_path``'s black bars, or None when there are
    none worth cropping (or the video cannot be read). Remembered per file
    for the session, so every pass over one video asks once."""
    try:
        st = os.stat(video_path)
        return _find(os.path.abspath(video_path), st.st_size, st.st_mtime,
                     bool(rows_only))
    except Exception as e:  # noqa: BLE001 - never stop an analysis over this
        print(f"picture_area: {video_path}: {type(e).__name__}: {e}")
        return None
