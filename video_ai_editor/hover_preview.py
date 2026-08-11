"""
Floating thumbnail popup that follows the cursor on hover.

Single instance per scene. Driven by clip item hover events:

    popup.show_at(global_pos, pixmap, caption="12.34s")  # mouse moved
    popup.hide()                                          # mouse left clip

Auto-positions to stay on screen, prefers above-and-right of cursor.
"""

from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtGui import (
    QBrush, QColor, QGuiApplication, QPainter, QPen, QPixmap
)
from PySide6.QtWidgets import QWidget


PADDING = 6
CAPTION_H = 18
PREVIEW_HEIGHT = 180   # height of the thumbnail inside the popup
CURSOR_OFFSET = 18     # gap between cursor and popup


class HoverPreview(QWidget):
    """Frameless tooltip-style popup showing a single thumbnail + caption."""

    def __init__(self):
        super().__init__(
            None,
            Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        # Don't steal focus or eat mouse events from the underlying scene.
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._pixmap: QPixmap | None = None
        self._caption = ""
        self.hide()

    # ── Public API ────────────────────────────────────────────────────

    def show_at(self, global_pos, pixmap: QPixmap | None, caption: str = ""):
        """Show (or move) the popup near `global_pos` with given content."""
        self._pixmap = pixmap if pixmap is not None and not pixmap.isNull() else None
        self._caption = caption or ""

        # Compute popup dimensions from the pixmap's aspect (or 16:9 fallback)
        if self._pixmap is not None:
            pw, ph = self._pixmap.width(), self._pixmap.height()
            inner_w = int(pw * PREVIEW_HEIGHT / ph) if ph > 0 else int(PREVIEW_HEIGHT * 16 / 9)
        else:
            inner_w = int(PREVIEW_HEIGHT * 16 / 9)

        total_w = inner_w + 2 * PADDING
        total_h = PREVIEW_HEIGHT + 2 * PADDING + CAPTION_H
        self.resize(total_w, total_h)

        self.move(self._compute_position(global_pos, total_w, total_h))

        if not self.isVisible():
            self.show()
        else:
            self.update()  # force repaint when only content changed

    def hide_preview(self):
        self.hide()

    # ── Positioning ───────────────────────────────────────────────────

    def _compute_position(self, global_pos, total_w, total_h) -> QPoint:
        """
        Prefer above-and-right of cursor. Flip / clamp to keep on screen.
        """
        gx, gy = int(global_pos.x()), int(global_pos.y())

        x = gx + CURSOR_OFFSET
        y = gy - total_h - 8  # above cursor

        screen = QGuiApplication.screenAt(QPoint(gx, gy))
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is None:
            return QPoint(x, y)
        geo = screen.availableGeometry()

        # Flip to left side if it'd overflow right
        if x + total_w > geo.right():
            x = gx - total_w - CURSOR_OFFSET
        # Flip below cursor if it'd overflow top
        if y < geo.top():
            y = gy + 24

        # Final clamp
        x = max(geo.left(), min(x, geo.right() - total_w))
        y = max(geo.top(),  min(y, geo.bottom() - total_h))
        return QPoint(x, y)

    # ── Painting ──────────────────────────────────────────────────────

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)

        # Card background
        bg = QColor(15, 15, 15, 240)
        border = QColor(47, 129, 247, 200)
        outer = self.rect().adjusted(0, 0, -1, -1)
        p.setBrush(QBrush(bg))
        p.setPen(QPen(border, 1))
        p.drawRoundedRect(outer, 6, 6)

        # Thumbnail area
        thumb_rect = QRect(
            PADDING, PADDING,
            self.width() - 2 * PADDING, PREVIEW_HEIGHT
        )

        if self._pixmap is not None:
            scaled = self._pixmap.scaled(
                thumb_rect.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            dx = (thumb_rect.width() - scaled.width()) // 2
            dy = (thumb_rect.height() - scaled.height()) // 2
            p.drawPixmap(thumb_rect.x() + dx, thumb_rect.y() + dy, scaled)
        else:
            p.fillRect(thumb_rect, QColor(30, 30, 30))
            p.setPen(QPen(QColor(145, 145, 145)))
            p.drawText(thumb_rect, Qt.AlignCenter, "loading…")

        # Caption
        if self._caption:
            p.setPen(QPen(QColor(225, 225, 225)))
            cap_rect = QRect(
                PADDING, PADDING + PREVIEW_HEIGHT,
                self.width() - 2 * PADDING, CAPTION_H
            )
            p.drawText(cap_rect, Qt.AlignCenter, self._caption)