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

import math

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


def _paint_play(p: QPainter) -> None:
    """Filled right-pointing triangle."""
    p.setBrush(p.pen().color())
    p.drawPolygon(QPolygonF([
        QPointF(5.0, 3.2), QPointF(13.2, 8.0), QPointF(5.0, 12.8),
    ]))


def _paint_stop(p: QPainter) -> None:
    """Filled square — transport stop."""
    p.setBrush(p.pen().color())
    p.drawRoundedRect(QRectF(4.2, 4.2, 7.6, 7.6), 1.2, 1.2)


def _paint_plus(p: QPainter) -> None:
    p.drawLine(QPointF(8.0, 3.0), QPointF(8.0, 13.0))
    p.drawLine(QPointF(3.0, 8.0), QPointF(13.0, 8.0))


def _paint_trash(p: QPainter) -> None:
    """Lidded bin."""
    p.drawLine(QPointF(3.0, 4.6), QPointF(13.0, 4.6))
    p.drawLine(QPointF(6.2, 2.6), QPointF(9.8, 2.6))
    p.drawPolyline(QPolygonF([
        QPointF(4.4, 4.6), QPointF(5.2, 13.4),
        QPointF(10.8, 13.4), QPointF(11.6, 4.6),
    ]))
    p.drawLine(QPointF(6.7, 7.0), QPointF(6.9, 11.2))
    p.drawLine(QPointF(9.3, 7.0), QPointF(9.1, 11.2))


def _paint_scissors(p: QPainter) -> None:
    """Two crossed blades over finger-ring circles."""
    p.drawLine(QPointF(5.4, 9.8), QPointF(12.4, 2.8))
    p.drawLine(QPointF(10.6, 9.8), QPointF(3.6, 2.8))
    p.drawEllipse(QPointF(4.6, 11.6), 2.0, 2.0)
    p.drawEllipse(QPointF(11.4, 11.6), 2.0, 2.0)


def _paint_save(p: QPainter) -> None:
    """Floppy: outer shell, label window, slider notch."""
    p.drawRoundedRect(QRectF(2.8, 2.8, 10.4, 10.4), 1.6, 1.6)
    p.drawPolyline(QPolygonF([
        QPointF(5.6, 2.8), QPointF(5.6, 6.2),
        QPointF(10.4, 6.2), QPointF(10.4, 2.8),
    ]))
    p.drawPolyline(QPolygonF([
        QPointF(4.8, 13.2), QPointF(4.8, 9.4),
        QPointF(11.2, 9.4), QPointF(11.2, 13.2),
    ]))


def _paint_export(p: QPainter) -> None:
    """Arrow rising out of a tray — download's mirror."""
    p.drawLine(QPointF(8.0, 10.2), QPointF(8.0, 2.6))
    p.drawPolyline(QPolygonF([QPointF(4.6, 6.1), QPointF(8.0, 2.6), QPointF(11.4, 6.1)]))
    p.drawPolyline(QPolygonF([
        QPointF(3.0, 11.4), QPointF(3.0, 13.6),
        QPointF(13.0, 13.6), QPointF(13.0, 11.4),
    ]))


def _paint_render(p: QPainter) -> None:
    """A frame with a play wedge — "produce the video"."""
    p.drawRoundedRect(QRectF(2.4, 3.4, 11.2, 9.2), 1.6, 1.6)
    p.setBrush(p.pen().color())
    p.drawPolygon(QPolygonF([
        QPointF(6.6, 5.9), QPointF(10.4, 8.0), QPointF(6.6, 10.1),
    ]))


def _paint_gear(p: QPainter) -> None:
    """Hub circle with radial teeth."""
    p.drawEllipse(QPointF(8.0, 8.0), 2.4, 2.4)
    for i in range(8):
        a = math.tau * i / 8.0
        p.drawLine(
            QPointF(8.0 + 4.2 * math.cos(a), 8.0 + 4.2 * math.sin(a)),
            QPointF(8.0 + 6.2 * math.cos(a), 8.0 + 6.2 * math.sin(a)),
        )


def _paint_tally(p: QPainter) -> None:
    """Four tally strokes and the crossing fifth — counting."""
    for x in (3.6, 6.0, 8.4, 10.8):
        p.drawLine(QPointF(x, 3.6), QPointF(x, 12.4))
    p.drawLine(QPointF(2.2, 11.4), QPointF(12.6, 4.6))


def _paint_ban(p: QPainter) -> None:
    """No-entry: circle with the diagonal strike."""
    p.drawEllipse(QPointF(8.0, 8.0), 5.4, 5.4)
    p.drawLine(QPointF(4.2, 4.2), QPointF(11.8, 11.8))


def _paint_search(p: QPainter) -> None:
    """Magnifier."""
    p.drawEllipse(QPointF(6.8, 6.8), 4.0, 4.0)
    p.drawLine(QPointF(9.8, 9.8), QPointF(13.4, 13.4))


def picker(palette: Palette = DARK, color: str | None = None) -> QIcon:
    """Grid of tiles, for the "pick from a page" action. Defaults to body text
    colour, since it sits on a quiet secondary button."""
    return _icon(_paint_grid, color or palette.text)


def download(palette: Palette = DARK, color: str | None = None) -> QIcon:
    """Download arrow. Defaults to `on_accent` — it rides the accent-filled
    primary button, where body-text grey would disappear."""
    return _icon(_paint_download, color or palette.on_accent)


def play(palette: Palette = DARK, color: str | None = None) -> QIcon:
    """Transport play. Defaults to `on_accent` for accent-filled buttons."""
    return _icon(_paint_play, color or palette.on_accent)


def stop(palette: Palette = DARK, color: str | None = None) -> QIcon:
    """Transport stop. Defaults to `on_accent` — usually rides a red button."""
    return _icon(_paint_stop, color or palette.on_accent)


def plus(palette: Palette = DARK, color: str | None = None) -> QIcon:
    """Add. Body-text tint for quiet secondary buttons."""
    return _icon(_paint_plus, color or palette.text, width=1.8)


def trash(palette: Palette = DARK, color: str | None = None) -> QIcon:
    """Delete."""
    return _icon(_paint_trash, color or palette.text, width=1.3)


def scissors(palette: Palette = DARK, color: str | None = None) -> QIcon:
    """Cut mode."""
    return _icon(_paint_scissors, color or palette.text, width=1.3)


def save(palette: Palette = DARK, color: str | None = None) -> QIcon:
    """Save / persist."""
    return _icon(_paint_save, color or palette.text, width=1.3)


def export(palette: Palette = DARK, color: str | None = None) -> QIcon:
    """Send out of the app."""
    return _icon(_paint_export, color or palette.text)


def render(palette: Palette = DARK, color: str | None = None) -> QIcon:
    """Render the video. Defaults to `on_accent` — rides the green CTA."""
    return _icon(_paint_render, color or palette.on_accent, width=1.3)


def gear(palette: Palette = DARK, color: str | None = None) -> QIcon:
    """Settings. Defaults to `on_accent` (sits on the render CTA's ⚙ half)."""
    return _icon(_paint_gear, color or palette.on_accent, width=1.3)


def tally(palette: Palette = DARK, color: str | None = None) -> QIcon:
    """Counting / verification against a counter."""
    return _icon(_paint_tally, color or palette.text, width=1.3)


def ban(palette: Palette = DARK, color: str | None = None) -> QIcon:
    """Exclude / avoid."""
    return _icon(_paint_ban, color or palette.text, width=1.3)


def search(palette: Palette = DARK, color: str | None = None) -> QIcon:
    """Magnifier. Defaults to `on_accent` — rides the filled search button."""
    return _icon(_paint_search, color or palette.on_accent, width=1.5)
