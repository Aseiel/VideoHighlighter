"""Size a compact control to the text it actually has to show.

Small buttons beside a field -- "Open", "Refresh", "Labels…" -- were given
hard-coded pixel widths. A pixel count is a guess about a font: it holds for
the one the author had and fails for anyone whose font, DPI scaling or Qt style
differs by a few percent, and the failure is silent and ugly. "Refresh" in a
60px button renders as "efres": the label is centred, so it loses a bit from
each end and reads as a different word rather than an obviously cut one, which
is why these survive so long unnoticed.

Measuring the string in the widget's own font instead costs nothing and cannot
be wrong on someone else's machine.
"""

from __future__ import annotations

from modules.ui.theme import BUTTON_CHROME_H

# What the button spends on padding and border before any text is drawn, taken
# from the stylesheet that sets it rather than guessed here -- the two drifting
# apart is the whole bug. A few px of slack on top absorbs the rounding some
# styles add.
CHROME_PX = BUTTON_CHROME_H + 4


def fit_width(widget, text: str | None = None, *, minimum: int = 0,
              chrome: int = CHROME_PX) -> int:
    """Set `widget`'s minimum width so `text` fits, and return that width.

    Deliberately a *minimum* rather than a fixed size: a layout is then free to
    give the control more room, which is what should happen when the panel is
    wide. Fixing the width is what made these clip in the first place.
    """
    label = widget.text() if text is None else text
    advance = widget.fontMetrics().horizontalAdvance(label or "")
    width = max(int(advance) + chrome, int(minimum))
    widget.setMinimumWidth(width)
    return width


def fit_icon_button(widget, *, side: int = 30) -> int:
    """Square a button that shows only an icon.

    Separate from `fit_width` because there is no text to measure: the size is
    the icon's, and the caller wants it square rather than merely wide enough.
    """
    widget.setMinimumWidth(side)
    widget.setMinimumHeight(side)
    return side
