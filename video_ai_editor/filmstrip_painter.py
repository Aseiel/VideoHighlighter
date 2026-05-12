"""
Filmstrip painter — draws a row of evenly-spaced thumbnails into a QPainter.

Stateless. Call from any QGraphicsItem.paint() override:

    paint_filmstrip(painter, rect, start_time, end_time, thumb_cache)

The number of thumb slots is auto-computed from rect width. For each slot, the
frame at the slot's midpoint time is requested from `thumb_cache`. Missing
thumbs render as dark placeholders; they'll fill in as the cache emits
`thumbnail_ready` — your clip item just needs to call update() in response.
"""

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen


# Slots are sized assuming this aspect ratio. Real thumbs can be any aspect;
# they get center-cropped (if too wide) or letterboxed (if too narrow).
DEFAULT_ASPECT = 16 / 9
MIN_THUMB_W = 40
PLACEHOLDER_COLOR = QColor(20, 20, 30)
SEPARATOR_COLOR = QColor(10, 10, 18)


def paint_filmstrip(painter: QPainter, rect: QRectF, start_time: float,
                    end_time: float, thumb_cache, aspect: float = DEFAULT_ASPECT):
    """
    Paint a filmstrip into `rect` covering source time start_time → end_time.

    Parameters
    ----------
    painter : QPainter
        Active painter; state is saved/restored internally.
    rect : QRectF
        Target area in painter coordinates (the clip's local rect).
    start_time, end_time : float
        Source video times in seconds for the left and right edges of `rect`.
    thumb_cache : ThumbnailCache
        Provides .request(time_seconds, height_px) → QPixmap | None.
    aspect : float
        Assumed video aspect ratio for slot sizing. 16/9 is fine for most
        footage; vertical/square videos still render correctly, just with
        letterbox bars.
    """
    if rect.width() < MIN_THUMB_W or rect.height() < 8:
        return

    H = int(rect.height())
    target_slot_w = max(MIN_THUMB_W, int(H * aspect))
    n_slots = max(1, int(rect.width() // target_slot_w))
    slot_w = rect.width() / n_slots
    duration = max(1e-6, end_time - start_time)

    painter.save()
    painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
    # Clip to the target rect so half-cropped thumbs at the edges
    # don't bleed outside the clip bounds.
    painter.setClipRect(rect)

    for i in range(n_slots):
        slot_x = rect.x() + i * slot_w
        slot_rect = QRectF(slot_x, rect.y(), slot_w, H)

        # Sample at slot midpoint (better than left edge — feels more
        # representative of "what's in this section of the clip").
        t = start_time + (i + 0.5) / n_slots * duration
        pix = thumb_cache.request(t, H)

        if pix is None or pix.isNull():
            painter.fillRect(slot_rect, PLACEHOLDER_COLOR)
        else:
            _draw_aspect_fit(painter, pix, slot_rect)

        # Thin separator between slots (except last)
        if i < n_slots - 1:
            x = slot_x + slot_w
            painter.setPen(QPen(SEPARATOR_COLOR, 1))
            painter.drawLine(int(x), int(rect.y()),
                             int(x), int(rect.y() + H))

    painter.restore()


def _draw_aspect_fit(painter: QPainter, pix, slot_rect: QRectF):
    """
    Draw `pix` into `slot_rect` preserving aspect ratio.

    - If pix is wider than the slot at slot height → center-crop horizontally.
    - If pix is narrower than the slot at slot height → letterbox (paint
      placeholder bars on left/right).
    """
    pw, ph = pix.width(), pix.height()
    if pw == 0 or ph == 0:
        painter.fillRect(slot_rect, PLACEHOLDER_COLOR)
        return

    H = slot_rect.height()
    scale = H / ph
    scaled_w = pw * scale

    if scaled_w >= slot_rect.width():
        # Pixmap is wider than the slot at this height → crop horizontally.
        # Source rect: take a centered strip slot_w/scale wide.
        src_w = slot_rect.width() / scale
        src_x = (pw - src_w) / 2
        src = QRectF(src_x, 0, src_w, ph)
        painter.drawPixmap(slot_rect, pix, src)
    else:
        # Pixmap is narrower → letterbox horizontally.
        painter.fillRect(slot_rect, PLACEHOLDER_COLOR)
        offset_x = (slot_rect.width() - scaled_w) / 2
        target = QRectF(
            slot_rect.x() + offset_x, slot_rect.y(),
            scaled_w, H
        )
        painter.drawPixmap(target, pix, QRectF(0, 0, pw, ph))