"""Painted button icons — the counterpart to `theme.py` for glyphs.

Emoji were doing this job. They come from whatever emoji font the OS ships, so
they change between machines, ignore the palette entirely, and render at a size
and weight nobody chose. These are drawn with QPainter instead: monochrome,
tinted from the theme, and generated at several pixel sizes so Qt can pick a
sharp one on any DPI. No new dependency — QtGui already ships with PySide6.

Every path is drawn on a 16x16 logical grid (`_GRID`) and scaled to the
requested size, so a glyph is defined once and stays proportional everywhere.
To add one, write a `_paint_*` that draws inside 16x16 and expose it via
`_icon`.
"""
from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap, QPolygonF

from modules.ui.theme import DARK, Palette

# The design grid every _paint_* below assumes.
_GRID = 16.0

# Rendered pixel sizes. Qt picks the closest to what the button asks for, so a
# 2x display gets the 32px art instead of upscaling the 16px one.
_SIZES = (16, 24, 32, 48)


def _icon(paint, color: str, width: float = 1.5) -> QIcon:
    """Run `paint` on the 16x16 grid at each size in `_SIZES`, in `color`."""
    icon = QIcon()
    for size in _SIZES:
        pm = QPixmap(size, size)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        try:
            p.setRenderHint(QPainter.Antialiasing, True)
            p.scale(size / _GRID, size / _GRID)
            pen = QPen(QColor(color))
            # Pen width is in grid units, so it scales with the art instead of
            # going hairline-thin on the 48px variant.
            pen.setWidthF(width)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            paint(p)
        finally:
            p.end()
        icon.addPixmap(pm)
    return icon


def _paint_grid(p: QPainter) -> None:
    """Four tiles — a thumbnail grid, i.e. "you get to look and choose"."""
    for x, y in ((2.0, 2.0), (9.0, 2.0), (2.0, 9.0), (9.0, 9.0)):
        p.drawRoundedRect(QRectF(x, y, 5.0, 5.0), 1.2, 1.2)


def _paint_download(p: QPainter) -> None:
    """Arrow dropping into a tray — the usual download idiom."""
    p.drawLine(QPointF(8.0, 2.0), QPointF(8.0, 9.6))
    p.drawPolyline(QPolygonF([QPointF(4.6, 6.4), QPointF(8.0, 9.9), QPointF(11.4, 6.4)]))
    p.drawPolyline(QPolygonF([
        QPointF(3.0, 11.4), QPointF(3.0, 13.6),
        QPointF(13.0, 13.6), QPointF(13.0, 11.4),
    ]))


def picker(palette: Palette = DARK, color: str | None = None) -> QIcon:
    """Grid of tiles, for the "pick from a page" action. Defaults to body text
    colour, since it sits on a quiet secondary button."""
    return _icon(_paint_grid, color or palette.text)


def download(palette: Palette = DARK, color: str | None = None) -> QIcon:
    """Download arrow. Defaults to `on_accent` — it rides the accent-filled
    primary button, where body-text grey would disappear."""
    return _icon(_paint_download, color or palette.on_accent)
