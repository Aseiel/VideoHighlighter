"""The AUDIO WAVEFORM lane of the signal timeline, as one painted item.

The lane used to be one line item per waveform bin, drawn at the bin's
centre. That capped the bins at a few thousand for the whole file - more
items made the scene slow to build - and at that count a bin on a long video
was about a second: a sound showed up to half a second away from the playhead
standing on it. Bins are now 50 ms (``waveform.BINS_PER_SECOND``), which is
tens of thousands for a film, so they are painted here instead: only the ones
in the exposed rect, each across its own time span rather than as a line at
its centre. Zoomed out, where several bins share one screen pixel, they are
merged first (lowest minimum, highest maximum, loudest energy), so what is
painted never exceeds the viewport's width in rectangles.

It takes no mouse buttons, so a click on the lane still seeks as a click
anywhere else on the timeline does.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QGraphicsItem

# Below this many screen pixels per bin, bins are merged per pixel.
MIN_PX_PER_BIN = 1.5
# The waveform fills this share of the lane's half-height at full scale.
FILL = 0.8


def bin_levels(energy: Sequence[float], lo_pct: float = 0.10, hi_pct: float = 0.97) -> np.ndarray:
    """Energy mapped onto 0..1 against the file's own ``lo_pct``..``hi_pct``
    percentiles, clipped: one loud transient does not flatten the rest."""
    e = np.asarray(energy, dtype=float)
    if e.size == 0:
        return e
    s = np.sort(e)
    lo = s[min(s.size - 1, int(s.size * lo_pct))]
    hi = s[min(s.size - 1, int(s.size * hi_pct))]
    return np.clip((e - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def merge_bins(mins: np.ndarray, maxs: np.ndarray, levels: np.ndarray, group: int):
    """Every ``group`` consecutive bins as one: (starts, mins, maxs, levels)."""
    starts = np.arange(0, mins.size, max(1, int(group)))
    if starts.size == 0:
        return starts, mins, maxs, levels
    return (starts, np.minimum.reduceat(mins, starts), np.maximum.reduceat(maxs, starts),
            np.maximum.reduceat(levels, starts))


class WaveformLane(QGraphicsItem):
    """``waveform`` ((min, max[, rms]) per bin) tiling ``[0, duration]``."""

    def __init__(self, waveform, duration: float, pixels_per_second: float,
                 width: float, height: float, colors: List[QColor], parent=None):
        super().__init__(parent)
        pts = list(waveform or [])
        n = len(pts)
        self._mins = np.array([float(p[0]) for p in pts], dtype=float)
        self._maxs = np.array([float(p[1]) for p in pts], dtype=float)
        energy = [float(p[2]) if len(p) > 2 else (abs(float(p[0])) + abs(float(p[1]))) / 2
                  for p in pts]
        self._levels = bin_levels(energy)
        self._bin_px = (float(duration) / n * float(pixels_per_second)) if n and duration else 0.0
        self._width = float(width)
        self._height = float(height)
        self._colors = []
        for c in colors or [QColor(47, 129, 247, 200)]:
            c = QColor(c)
            c.setAlpha(min(255, c.alpha() + 50))
            self._colors.append(c)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        # Without this Qt passes no exposedRect worth the name, and every
        # repaint would walk the whole film.
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption, True)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self._width, self._height)

    def bin_span(self, i: int):
        """Scene x of bin ``i``'s start and end."""
        return i * self._bin_px, (i + 1) * self._bin_px

    def paint(self, painter, option, widget=None):
        n = self._mins.size
        if not n or self._bin_px <= 0:
            return
        exposed = option.exposedRect if option is not None else self.boundingRect()
        first = max(0, int(exposed.left() // self._bin_px) - 1)
        last = min(n, int(exposed.right() // self._bin_px) + 2)
        if last <= first:
            return
        t = painter.worldTransform()
        sx = abs(t.m11()) or 1.0
        sy = abs(t.m22()) or 1.0
        group = 1
        if self._bin_px * sx < MIN_PX_PER_BIN:
            group = int(np.ceil(MIN_PX_PER_BIN / (self._bin_px * sx)))
            first -= first % group          # the same groups on every repaint
        starts, mins, maxs, levels = merge_bins(self._mins[first:last], self._maxs[first:last],
                                                self._levels[first:last], group)
        mid = self._height / 2.0
        half = self._height / 2.0 * FILL
        one_px = 1.0 / sy
        painter.setPen(Qt.PenStyle.NoPen)
        top_index = len(self._colors) - 1
        for k, s in enumerate(starts):
            i0 = first + int(s)
            i1 = min(n, i0 + group)
            x0, _ = self.bin_span(i0)
            x1 = i1 * self._bin_px
            y0 = mid - float(maxs[k]) * half
            y1 = mid - float(mins[k]) * half
            if y1 - y0 < one_px:
                y0, y1 = mid - one_px / 2, mid + one_px / 2
            painter.setBrush(self._colors[min(top_index, int(float(levels[k]) * top_index))])
            painter.drawRect(QRectF(x0, y0, x1 - x0, y1 - y0))
