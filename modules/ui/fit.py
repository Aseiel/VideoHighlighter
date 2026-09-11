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


def scrollable_row(row):
    """Wrap a toolbar row so a narrow window scrolls it instead of clipping it.

    A QHBoxLayout given less width than its contents need does not shrink them
    past their minimum — it simply stops drawing what does not fit, and the
    controls at the right end become unreachable with no indication that they
    exist. Reported from a 4K screen where "Edit duration" was cut in half at
    the edge of the edit toolbar.

    Scrolling rather than wrapping, for the same reason the dock tab bars use
    scroll buttons: a control you can reach beats one that has been folded onto
    a second line the splitter above it then has to give up height for.

    The height reserves room for the scrollbar whether or not it is showing, so
    the row does not lose a few pixels off its buttons at the moment the bar
    appears.
    """
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QFrame, QScrollArea

    scroller = QScrollArea()
    scroller.setWidget(row)
    scroller.setWidgetResizable(True)
    scroller.setFrameShape(QFrame.NoFrame)
    scroller.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroller.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroller.setStyleSheet("QScrollArea { background: transparent; }")

    bar = scroller.horizontalScrollBar().sizeHint().height()
    scroller.setFixedHeight(row.sizeHint().height() + bar)
    return scroller
