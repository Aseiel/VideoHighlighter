"""A filmstrip of the whole video, as one lane of the signal timeline.

The signal lanes above it are abstract on purpose: a bar says *where* something
is, and colour and position are the right way to answer that because you can
scan a whole video at a glance. What they cannot answer is *what* is there, and
until now the only way to find out was to seek and look. One strip along the
bottom answers it for every lane at once, because they all share the x-axis —
which is also why the other lanes keep their colours rather than growing thumbs
of their own. Per-lane thumbnails would decode the same frames at the same
timestamps several times over and make each row taller for no new information.

Two things make this affordable on an hour-long video:

* **Only the exposed slots are drawn.** `paint()` is handed the region Qt
  actually needs, so scrolling a 63-minute timeline requests the dozen frames
  on screen rather than the thousands that are not.
* **The cache is asynchronous.** A missing frame paints as a placeholder now
  and arrives later on `thumbnail_ready`; nothing blocks the GUI thread waiting
  on a decode.

Slots are aligned to a grid anchored at x=0 rather than to the exposed
rectangle, so the thumbnails stay put while you scroll instead of sliding
around under the cursor as the exposed region changes.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QRectF, Qt
from PySide6.QtWidgets import QGraphicsItem

from .filmstrip_painter import DEFAULT_ASPECT, MIN_THUMB_W, paint_filmstrip

# Tall enough that a 16:9 frame is worth looking at (54px high is 96px wide)
# without taking the room the signal lanes need to stay readable.
LANE_HEIGHT = 54

# Hard ceiling on how many slots one paint may ask the cache for. A 4K-wide
# viewport holds about 40, so this never binds in normal use -- it exists
# because the failure it guards against is not a slow frame but an unusable
# window, and it should not depend on a Qt flag being set correctly forever.
# Slots past the ceiling still draw, as placeholders.
MAX_SLOTS_PER_PAINT = 96


class FilmstripLane(QGraphicsItem):
    """Thumbnails spanning 0 → `duration` across the full scene width."""

    def __init__(self, width: float, duration: float, thumb_cache,
                 height: float = LANE_HEIGHT, aspect: float = DEFAULT_ASPECT,
                 parent=None):
        super().__init__(parent)
        self._width = max(1.0, float(width))
        self._height = float(height)
        self._duration = max(1e-6, float(duration))
        self._cache = thumb_cache
        self._aspect = aspect
        # Cheap to redraw and never interactive: let clicks through to the
        # scene beneath so the strip does not eat seeks aimed at the timeline.
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        # Without this flag Qt does not compute `exposedRect` and passes the
        # whole boundingRect instead -- which for an hour of video is ~2400
        # slots, re-requested on *every* repaint, and the window stops
        # responding. Everything else here assumes the exposed region is
        # roughly a viewport; this is what makes that true.
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemUsesExtendedStyleOption,
                     True)

    def boundingRect(self) -> QRectF:
        return QRectF(0.0, 0.0, self._width, self._height)

    def slot_width(self) -> float:
        """Width of one thumbnail slot, matching filmstrip_painter's maths.

        Kept in step with the painter deliberately: if the two disagreed, the
        rectangle handed over would not be a whole number of slots and the grid
        would drift, which is exactly the sliding this alignment prevents.
        """
        return float(max(MIN_THUMB_W, int(self._height * self._aspect)))

    def paint(self, painter, option, widget=None):
        if self._cache is None:
            return

        full = self.boundingRect()
        exposed = QRectF(option.exposedRect).intersected(full)
        if exposed.isEmpty():
            return

        slot_w = self.slot_width()
        # Snap outwards to the global slot grid so a partly-visible thumbnail is
        # drawn whole rather than resampled at a different offset each frame.
        left = max(0.0, math.floor(exposed.left() / slot_w) * slot_w)
        right = min(self._width, math.ceil(exposed.right() / slot_w) * slot_w)
        if right - left < slot_w:
            right = min(self._width, left + slot_w)
        if right <= left:
            return

        # The guard described at MAX_SLOTS_PER_PAINT. Reached only if the
        # exposed region is not viewport-shaped, in which case drawing the
        # leftmost stretch beats stalling the GUI thread on thousands of
        # decodes.
        span = right - left
        capped = MAX_SLOTS_PER_PAINT * slot_w
        if span > capped:
            right = left + capped
            span = capped

        rect = QRectF(left, 0.0, span, self._height)
        start = left / self._width * self._duration
        end = right / self._width * self._duration
        paint_filmstrip(painter, rect, start, end, self._cache,
                        aspect=self._aspect)
