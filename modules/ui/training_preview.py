"""The "watch it learn" window: what the model finds, round by round.

The counterpart of the live detection preview for a model that is still
training. Each round arrives as a ``modules.vision.training_preview.RoundSnapshot``;
the window keeps every one, so a person can drag back to round 3 and forward
to round 20 and see the guesses tighten. It follows the newest round unless
they have scrubbed away from it.

Closing the window only stops the drawing — the run carries on, and the
panel keeps its one-line summary either way.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget,
)


def _pixmap(rgb) -> QPixmap:
    h, w = rgb.shape[:2]
    image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(image.copy())      # copy: rgb's buffer is not ours to keep


def history_line(history) -> str:
    """"found per round: 0/8 · 2/8 · 5/8" — the trend at a glance, newest last."""
    if not history:
        return ""
    shown = history if len(history) <= 12 else [history[0], None] + list(history[-10:])
    parts = ["…" if h is None else f"{h[1]}/{h[2]}" for h in shown]
    return "Found per round: " + " · ".join(parts)


class TrainingPreviewWindow(QWidget):
    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("👁 Watch it learn")
        self.setMinimumSize(620, 460)
        self.resize(980, 640)
        self._rounds: list = []          # [(QPixmap | None, caption, snapshot)]
        self._follow = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self.image_label = QLabel(
            "Waiting for the first round to finish…\n\n"
            "After every round the model looks at the same few frames it was not "
            "trained on. Grey boxes are yours; green is a guess that matches one, "
            "orange a guess that matches nothing.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setWordWrap(True)
        self.image_label.setStyleSheet(
            "QLabel { background:#101010; color:#8c8c8c; border:1px solid #333; }")
        layout.addWidget(self.image_label, 1)

        self.caption = QLabel("")
        self.caption.setAlignment(Qt.AlignCenter)
        self.caption.setWordWrap(True)
        self.caption.setStyleSheet("font-size:10pt; font-weight:bold;")
        layout.addWidget(self.caption)

        self.trend = QLabel("")
        self.trend.setAlignment(Qt.AlignCenter)
        self.trend.setStyleSheet("color:#888; font-size:9pt;")
        layout.addWidget(self.trend)

        controls = QHBoxLayout()
        self.prev_btn = QPushButton("◀")
        self.prev_btn.setFixedWidth(36)
        self.prev_btn.clicked.connect(lambda: self._show(self._index() - 1, follow=False))
        controls.addWidget(self.prev_btn)

        self.scrub = QSlider(Qt.Horizontal)
        self.scrub.setRange(0, 0)
        self.scrub.valueChanged.connect(lambda v: self._show(v, follow=v == len(self._rounds) - 1))
        controls.addWidget(self.scrub, 1)

        self.next_btn = QPushButton("▶")
        self.next_btn.setFixedWidth(36)
        self.next_btn.clicked.connect(lambda: self._show(self._index() + 1, follow=False))
        controls.addWidget(self.next_btn)

        self.live_btn = QPushButton("⏭ Latest")
        self.live_btn.setToolTip("Jump to the newest round and keep following")
        self.live_btn.clicked.connect(lambda: self._show(len(self._rounds) - 1, follow=True))
        controls.addWidget(self.live_btn)
        layout.addLayout(controls)
        self._update_controls()

    # ── fed by the panel ────────────────────────────────────────────────
    def add_round(self, snap) -> None:
        pix = _pixmap(snap.mosaic_rgb) if getattr(snap, "mosaic_rgb", None) is not None else None
        self._rounds.append((pix, snap.sentence(), snap))
        self.scrub.blockSignals(True)
        self.scrub.setRange(0, len(self._rounds) - 1)
        self.scrub.blockSignals(False)
        if self._follow:
            self._show(len(self._rounds) - 1, follow=True)
        else:
            self._update_controls()

    def reset(self) -> None:
        self._rounds.clear()
        self._follow = True
        self.scrub.setRange(0, 0)
        self.caption.setText("")
        self.trend.setText("")
        self._update_controls()

    # ── internals ───────────────────────────────────────────────────────
    def _index(self) -> int:
        return self.scrub.value()

    def _show(self, index: int, follow: bool) -> None:
        if not self._rounds:
            return
        index = max(0, min(index, len(self._rounds) - 1))
        self._follow = follow and index == len(self._rounds) - 1
        self.scrub.blockSignals(True)
        self.scrub.setValue(index)
        self.scrub.blockSignals(False)
        pix, caption, snap = self._rounds[index]
        if pix is not None:
            self.image_label.setPixmap(pix.scaled(
                self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            # Never leave another round's picture up under this round's caption.
            self.image_label.clear()
            self.image_label.setText(
                "No picture for this round — the live view was not open while it ran. "
                "The numbers below are still real.")
        tag = "  • LIVE" if self._follow else f"  • round {index + 1} of {len(self._rounds)} so far"
        self.caption.setText(caption + tag)
        self.trend.setText(history_line(snap.history))
        self._update_controls()

    def _update_controls(self) -> None:
        n = len(self._rounds)
        i = self._index()
        self.prev_btn.setEnabled(n > 0 and i > 0)
        self.next_btn.setEnabled(n > 0 and i < n - 1)
        self.live_btn.setEnabled(n > 0 and not self._follow)
        self.scrub.setEnabled(n > 1)

    def resizeEvent(self, event):        # noqa: N802 (Qt override)
        super().resizeEvent(event)
        if self._rounds:
            self._show(self._index(), follow=self._follow)

    def closeEvent(self, event):         # noqa: N802 (Qt override)
        self.closed.emit()
        super().closeEvent(event)
