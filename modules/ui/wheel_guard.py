"""The wheel scrolls the panel. It does not change values.

Qt's default is that a spin box, combo box, slider or tab bar under the cursor
takes the wheel and changes its value. In a settings screen that is a long
scrolling column of exactly those widgets, that default is a trap: scrolling
past the Scoring Points panel silently re-scores a run, and nothing announces
it. The user's intent when they turn the wheel over a form is to move the form.

So a guarded widget never sees a wheel event: the guard swallows it and hands
it to the nearest enclosing scroll area instead. The panel scrolls, the value
does not move.

Handing it over is deliberate rather than leaving it to Qt's "ignore it and the
parent gets it" propagation. That propagation is real but only runs for events
the platform delivers, which makes it untestable here — and the scrolling half
is the whole point of the change, so it should be the half that is proven.
The event goes to the scroll area's *viewport*; sending it to the scroll area
itself does nothing (measured, not assumed).

Values still change by every deliberate means — typing, clicking the arrows,
dragging a slider, arrow keys once focused. A widget that genuinely wants the
wheel can opt out with `setProperty("wheelGuard", False)`.

Installed on the QApplication rather than on each widget, so it also covers
widgets built later (rule rows, dialogs) which an install-time sweep would
miss.
"""

from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import (QAbstractScrollArea, QAbstractSpinBox, QApplication,
                               QComboBox, QSlider, QTabBar)

#: Widgets whose wheel behaviour is a value change rather than a scroll.
#: QAbstractSpinBox covers QSpinBox, QDoubleSpinBox and QDateTimeEdit.
#: Deliberately absent: QScrollBar, QAbstractItemView, QTextEdit and the
#: timeline's QGraphicsView — for those the wheel really does mean "scroll me"
#: (or, in the timeline's case, "zoom"), which is what the user is asking for.
GUARDED = (QAbstractSpinBox, QComboBox, QSlider, QTabBar)


def scrolling_ancestor(widget):
    """The nearest enclosing scroll area, or None if the widget is not in one."""
    parent = widget.parentWidget()
    while parent is not None:
        if isinstance(parent, QAbstractScrollArea):
            return parent
        parent = parent.parentWidget()
    return None


class WheelGuard(QObject):
    """Application event filter: swallows wheel events on value widgets."""

    def eventFilter(self, obj, event):
        if event.type() != QEvent.Type.Wheel or not isinstance(obj, GUARDED):
            return False
        if obj.property("wheelGuard") is False:
            return False

        area = scrolling_ancestor(obj)
        if area is not None:
            # Re-aimed at the panel. The viewport is what scrolls; the guard
            # does not match it, so this does not come back round.
            event.setAccepted(False)
            QApplication.sendEvent(area.viewport(), event)
        # Swallowed either way: outside a scroll area there is nothing the
        # wheel could mean here, and doing nothing beats editing a value.
        return True


def install(app) -> WheelGuard:
    """Guard every value widget in `app`, now and in future. Returns the guard.

    The guard is parented to `app` so it lives as long as the application and
    is not collected out from under the filter chain.
    """
    guard = WheelGuard(app)
    app.installEventFilter(guard)
    return guard
