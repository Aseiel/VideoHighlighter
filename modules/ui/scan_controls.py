"""The main window's "scan this one signal now" buttons, and what they say.

Each button runs one detector over the videos in the list and saves the result
— no highlight is cut. They used to be bare buttons named after the signal
("Audio", "Objects") sitting next to a points field, which read as a label or a
toggle for the points rather than as something that starts minutes of work, and
they said nothing about whether that work had already been done.

So every button here is the same three things:

- a play icon and a verb ("Scan audio"), so it reads as an action;
- a status beside it — what is already saved for the listed videos, the
  progress while it runs, why it cannot run;
- its own Stop: the button that started a scan is the one that ends it, which
  is how the timeline viewer's Analyze panel already behaves.

Kept in its own module so the windows that use it stay an import and a call
site: this file is identical in both editions.
"""
from __future__ import annotations

import threading
from typing import Callable, Dict, List, Optional

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSizePolicy, QWidget,
)

from modules.report import scan_state
from modules.ui import icons as ui_icons
from modules.ui.fit import fit_width
from modules.ui.theme import DARK

# Icon + the gap Qt puts between it and the text.
_ICON_ROOM = 22


class _ElidedLabel(QLabel):
    """A label that shortens with "…" instead of forcing its row wider.

    Status text is as long as a file name. In a settings column that must not
    push the layout around, so the label claims no width of its own and shows
    as much as fits; the whole text stays in the tooltip.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(0)

    def set_full_text(self, text: str) -> None:
        self._full = text or ""
        self.setToolTip(self._full)
        self._fit()

    def full_text(self) -> str:
        return self._full

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._fit()

    def _fit(self) -> None:
        shown = self.fontMetrics().elidedText(
            self._full, Qt.ElideRight, max(0, self.width()))
        if shown != self.text():
            super().setText(shown)


class ScanControl(QWidget):
    """One scan button and the status beside it."""

    def __init__(self, kind: str, label: str, tooltip: str, *,
                 with_status: bool = True, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.label = label
        self.base_tooltip = tooltip

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.button = QPushButton(label)
        self.button.setIcon(ui_icons.play(color=DARK.text))
        # Wide enough for the longer of its two faces, so the row does not
        # shift when the button turns into Stop.
        fit_width(self.button, max((label, "Stop"), key=len),
                  minimum=0)
        self.button.setMinimumWidth(self.button.minimumWidth() + _ICON_ROOM)
        row.addWidget(self.button)

        self._tone: Optional[str] = None
        self.status: Optional[_ElidedLabel] = None
        if with_status:
            self.status = _ElidedLabel()
            row.addWidget(self.status, 1)

    def show_idle(self, enabled: bool, tooltip: str) -> None:
        self.button.setText(self.label)
        self.button.setIcon(ui_icons.play(color=DARK.text))
        self.button.setEnabled(enabled)
        self.button.setToolTip(tooltip)

    def show_running(self) -> None:
        self.button.setText("Stop")
        self.button.setIcon(ui_icons.stop(color=DARK.danger))
        self.button.setEnabled(True)
        self.button.setToolTip("Stop this scan. What finished so far stays saved.")

    def set_status(self, text: str, tone: str = "mute") -> None:
        if self.status is None:
            return
        # Restyle only on a change of tone: a stylesheet is re-parsed on every
        # set, and progress arrives many times a second.
        if tone != self._tone:
            self._tone = tone
            color = {"ok": DARK.success, "warn": DARK.warning,
                     "busy": DARK.text_dim}.get(tone, DARK.text_mute)
            self.status.setStyleSheet(
                f"color: {color}; font-size: 9pt; background: transparent;")
        if text != self.status.full_text():
            self.status.set_full_text(text)


class ScanControls(QObject):
    """Every scan button in a window, kept in step.

    One scan runs at a time and a full run excludes them all, so the buttons
    cannot be managed one by one: starting any of them changes what the others
    may do. This owns that, and the background look at the cache that fills in
    the statuses.

    ``list_paths``  -> the videos a scan would run over, right now.
    ``on_start(kind)`` / ``on_stop()`` -> the window's own run and cancel.
    """

    _probed = Signal(int, object)

    def __init__(self, parent, *, list_paths: Callable[[], List[str]],
                 on_start: Callable[[str], None],
                 on_stop: Callable[[], None]):
        super().__init__(parent)
        self._list_paths = list_paths
        self._on_start = on_start
        self._on_stop = on_stop
        self._controls: Dict[str, List[ScanControl]] = {}
        self._running: Optional[str] = None
        self._stopping = False
        self._blocked = False               # a full run owns the machine
        self._paths: List[str] = []
        self._counts: Optional[Dict[str, int]] = None   # None = not looked yet
        self._notes: Dict[str, tuple] = {}  # kind -> (text, tone), until cleared
        self._generation = 0

        self._probed.connect(self._on_probed)
        # Adding ten files fires ten list changes; one look afterwards is enough.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(150)
        self._debounce.timeout.connect(self._refresh_now)
        # Progress is drawn at most ten times a second, whatever rate it
        # arrives at. A detector reports per batch of frames — hundreds of
        # times a second on a fast pass — and the window's own progress handler
        # pumps the event loop, so work done per report is work done nested
        # inside the previous report. Kept here it is one assignment.
        self._pending_progress: Optional[tuple] = None
        self._progress_tick = QTimer(self)
        self._progress_tick.setSingleShot(True)
        self._progress_tick.setInterval(100)
        self._progress_tick.timeout.connect(self._draw_progress)

    # ── building ─────────────────────────────────────────────────────────
    def control(self, kind: str, label: str, tooltip: str, *,
                with_status: bool = True) -> ScanControl:
        """A new button for ``kind``. A kind may have several (composition has
        one beside its editor and one in the signals list)."""
        ctl = ScanControl(kind, label, tooltip, with_status=with_status)
        ctl.button.clicked.connect(lambda _=False, k=kind: self._clicked(k))
        self._controls.setdefault(kind, []).append(ctl)
        self._apply()
        return ctl

    def kinds(self) -> List[str]:
        return list(self._controls)

    def controls(self, kind: str) -> List[ScanControl]:
        return list(self._controls.get(kind, []))

    # ── state the window reports ─────────────────────────────────────────
    def set_blocked(self, blocked: bool) -> None:
        """A full run started (True) or ended (False)."""
        self._blocked = bool(blocked)
        self._apply()
        if not blocked:
            self.refresh()          # the run just wrote every signal

    def set_running(self, kind: Optional[str]) -> None:
        """A scan of ``kind`` started, or (None) the running one ended."""
        self._running = kind
        self._pending_progress = None
        self._stopping = False
        if kind is not None:
            self._notes.pop(kind, None)
            self._apply()
            for ctl in self._controls.get(kind, []):
                ctl.set_status("starting…", "busy")
        else:
            self._apply()
            self.refresh()

    def set_progress(self, kind: str, percent: int, text: str = "") -> None:
        if kind != self._running:
            return
        self._pending_progress = (kind, int(percent), text or "")
        if not self._progress_tick.isActive():
            self._progress_tick.start()

    def _draw_progress(self) -> None:
        pending, self._pending_progress = self._pending_progress, None
        if not pending or pending[0] != self._running:
            return                  # it ended before this got drawn
        if self._stopping:
            return
        kind, percent, text = pending
        line = f"{percent}%" + (f" · {text}" if text else "")
        for ctl in self._controls.get(kind, []):
            ctl.set_status(line, "busy")

    def note(self, kind: str, text: str, tone: str = "warn") -> None:
        """Say something about ``kind`` that outranks the cache status — why it
        could not start, that it failed — until its next run or a list change."""
        self._notes[kind] = (text, tone)
        self._apply()

    def refresh(self) -> None:
        """The list or the cache may have changed: look again, shortly."""
        self._debounce.start()

    # ── internals ────────────────────────────────────────────────────────
    def _clicked(self, kind: str) -> None:
        if self._running == kind:
            self._stopping = True   # keep saying so; progress still arrives
            for ctl in self._controls.get(kind, []):
                ctl.set_status("stopping after the current step…", "busy")
            self._on_stop()
            return
        self._on_start(kind)

    def _refresh_now(self) -> None:
        try:
            paths = list(self._list_paths() or [])
        except Exception:
            paths = []
        if paths != self._paths:
            self._notes.clear()     # they were about a different list
            self._counts = None
        self._paths = paths
        self._apply()
        if not paths or self._running is not None:
            return                  # nothing to read, or it is being written

        self._generation += 1
        generation = self._generation

        def work():
            try:
                counts = scan_state.count_saved(paths)
            except Exception as e:  # a status line must never break the window
                print(f"⚠️ Could not read saved scans: {e}")
                return
            try:
                self._probed.emit(generation, counts)
            except RuntimeError:
                pass                # the window closed while we were reading

        threading.Thread(target=work, name="scan-status", daemon=True).start()

    def _on_probed(self, generation: int, counts) -> None:
        if generation != self._generation:
            return                  # an older look, overtaken by a newer list
        self._counts = dict(counts or {})
        self._apply()

    def _tooltip(self, ctl: ScanControl, total: int) -> str:
        if self._blocked:
            lead = "Waiting for the current run to finish."
        elif self._running is not None:
            lead = "Another scan is running — one at a time."
        elif total == 0:
            lead = "Add a video to the list first."
        elif total == 1:
            lead = "Runs now on the video in the list and saves the result."
        else:
            lead = (f"Runs now on all {total} videos in the list and saves "
                    "the results.")
        return f"{lead}\n\n{ctl.base_tooltip}"

    def _apply(self) -> None:
        total = len(self._paths)
        for kind, group in self._controls.items():
            for ctl in group:
                if self._running == kind:
                    ctl.show_running()
                    continue        # its status is the progress line
                enabled = (not self._blocked and self._running is None
                           and total > 0)
                ctl.show_idle(enabled, self._tooltip(ctl, total))

                if kind in self._notes:
                    ctl.set_status(*self._notes[kind])
                elif total == 0:
                    ctl.set_status("add a video first")
                elif self._counts is None or kind not in scan_state.KINDS:
                    # Not looked yet, or a kind whose output is a file beside
                    # the video rather than something the cache records.
                    ctl.set_status("")
                else:
                    saved = int(self._counts.get(kind, 0))
                    text = scan_state.describe(saved, total)
                    ctl.set_status(("✓ " + text) if saved >= total else text,
                                   "ok" if saved >= total else "mute")
