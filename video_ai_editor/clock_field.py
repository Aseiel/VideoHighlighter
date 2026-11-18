"""The player's clock, which is also where you type a time to jump to.

It reads like the label it replaced (``21:31 / 48:31``) until it is clicked;
then it holds just the current time, selected, so typing replaces it. Enter
jumps there and Escape or clicking away puts the clock back. Any form
:func:`modules.media.timecode.parse_clock` reads is accepted (``21:31``,
``1:09:05``, ``4145``, ``1h9m5s``); anything else flashes the field red and
leaves it open to fix, instead of jumping somewhere unintended.

Ticks from the player do not overwrite a time being typed: ``show`` keeps the
latest text and puts it up when editing ends.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QLineEdit

from modules.media.timecode import parse_clock

HINT = ("Click to jump to a time: type 21:31, 1:09:05, 4145 (seconds) or 1h9m5s, "
        "then Enter. Escape cancels. Ctrl+G does the same from anywhere.")

_STYLE = """
    QLineEdit {
        color: #a0ffa0; font-family: 'Consolas', monospace;
        font-weight: bold; padding: 8px; background-color: #141414;
        border: 1px solid #141414; border-radius: 4px; min-width: 120px;
    }
    QLineEdit:hover { border: 1px solid #3a5a3a; }
    QLineEdit[editing="true"] { border: 1px solid #a0ffa0; background-color: #1d261d; }
    QLineEdit[bad="true"] { border: 1px solid #ff6060; color: #ff9090; }
"""


class ClockField(QLineEdit):
    jumpRequested = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._shown = "00:00 / 00:00"
        self._editing = False
        self.setReadOnly(True)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(HINT)
        self.setStyleSheet(_STYLE)
        self.setText(self._shown)

    # -- what the player shows ------------------------------------------------
    def show_time(self, text: str) -> None:
        self._shown = text
        if not self._editing:
            self.setText(text)

    def editing(self) -> bool:
        return self._editing

    # -- typing a time --------------------------------------------------------
    def start_editing(self) -> None:
        if self._editing:
            return
        self._editing = True
        self._set_flag("editing", True)
        self.setReadOnly(False)
        self.setCursor(Qt.IBeamCursor)
        self.setText(self._shown.split(" / ")[0].strip())
        self.setFocus(Qt.ShortcutFocusReason)
        self.selectAll()

    def stop_editing(self) -> None:
        if not self._editing:
            return
        self._editing = False
        self._set_flag("editing", False)
        self._set_flag("bad", False)
        self.setReadOnly(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setText(self._shown)
        self.deselect()
        self.clearFocus()

    def mousePressEvent(self, event):
        if not self._editing and event.button() == Qt.LeftButton:
            self.start_editing()
            return
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if self._editing and event.key() in (Qt.Key_Return, Qt.Key_Enter):
            seconds = parse_clock(self.text())
            if seconds is None:
                self._set_flag("bad", True)
                QTimer.singleShot(900, lambda: self._set_flag("bad", False))
                return
            self.stop_editing()
            self.jumpRequested.emit(float(seconds))
            return
        if self._editing and event.key() == Qt.Key_Escape:
            self.stop_editing()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self.stop_editing()

    def _set_flag(self, name: str, on: bool) -> None:
        self.setProperty(name, "true" if on else "false")
        self.style().unpolish(self)
        self.style().polish(self)
