"""Foldable section — the collapsible counterpart to a QGroupBox.

The Controls dock and the LLM panel stack half a dozen always-open group boxes,
so the columns run long and the parts a user actually touches share space with
ones they set once. This widget replaces the static frame with a clickable
header bar (caret + title) that folds its body away, and remembers the fold
state per section across sessions via QSettings.

Kept in `modules/ui` next to `theme.py`/`icons.py` so every screen folds the
same way; colours come from the theme palette so a retheme carries over.

Usage mirrors QGroupBox closely so call sites stay small:

    section = CollapsibleSection("Filters", settings_key="controls/filters")
    section.setContentLayout(some_layout)      # instead of group.setLayout(...)
    parent_layout.addWidget(section)
"""
from __future__ import annotations

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QLabel, QSizePolicy, QToolButton, QVBoxLayout, QWidget,
)

from modules.ui.theme import DARK

# One org/app pair for every fold state so the keys live together in the
# registry instead of scattering per-screen.
_SETTINGS_ORG = "VideoHighlighter"
_SETTINGS_APP = "ui-sections"


class CollapsibleSection(QWidget):
    """A titled section whose body folds away when the header is clicked.

    `settings_key` (e.g. "controls/filters") persists the expanded state;
    without it the section just starts at `expanded` every run. An optional
    `hint` renders dimmed at the header's right edge — useful for showing a
    live value ("Off", "75%") while the body is folded.
    """

    toggled = Signal(bool)  # True when expanded

    def __init__(self, title: str, parent: QWidget | None = None, *,
                 expanded: bool = True, settings_key: str | None = None):
        super().__init__(parent)
        self._settings_key = settings_key
        p = DARK

        self._header = QToolButton()
        self._header.setText(title)
        self._header.setCheckable(True)
        self._header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._header.setArrowType(Qt.DownArrow)
        self._header.setCursor(Qt.PointingHandCursor)
        # Sized/styled as a flat full-width bar, not a button: the arrow is the
        # native QToolButton one so it tracks the palette, the bar picks up
        # hover like a list row.
        self._header.setStyleSheet(f"""
            QToolButton {{
                border: none;
                border-radius: {p.radius}px;
                background: {p.surface};
                color: {p.text};
                font-weight: 600;
                padding: 6px 8px;
                text-align: left;
            }}
            QToolButton:hover {{ background: {p.surface_hi}; }}
        """)
        self._header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        # Right-edge hint label, overlaid on the header bar.
        self._hint = QLabel("", self._header)
        self._hint.setStyleSheet(
            f"color: {p.text_dim}; font-weight: 400; font-size: 9pt;"
            "background: transparent;")
        self._hint.setAttribute(Qt.WA_TransparentForMouseEvents)

        self._body = QWidget()
        self._body.setObjectName("sectionBody")
        # The fold line under the header: a thin inset left rule marks the body
        # as belonging to the header above it.
        self._body.setStyleSheet(f"""
            QWidget#sectionBody {{
                background: transparent;
                border-left: 2px solid {p.border};
                margin-left: 4px;
            }}
        """)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        lay.addWidget(self._header)
        lay.addWidget(self._body)

        if settings_key is not None:
            settings = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
            stored = settings.value(settings_key, None)
            if stored is not None:
                expanded = str(stored).lower() in ("true", "1")
        self._header.setChecked(expanded)
        self._apply(expanded)
        self._header.toggled.connect(self._on_toggled)
        # Persist only on real clicks: a programmatic setChecked (or a stray
        # state change during construction/teardown) must never overwrite the
        # user's saved preference.
        self._header.clicked.connect(self._persist)

    # ------------------------------------------------------------- QGroupBox-ish API

    def setContentLayout(self, layout) -> None:
        layout.setContentsMargins(10, 6, 4, 8)
        self._body.setLayout(layout)

    # Alias so call sites converting from QGroupBox read naturally either way.
    setLayout = setContentLayout  # type: ignore[assignment]

    def set_hint(self, text: str) -> None:
        """Dim right-aligned text on the header (live value while folded)."""
        self._hint.setText(text)
        self._reposition_hint()

    def expand(self, on: bool = True) -> None:
        self._header.setChecked(on)

    def is_expanded(self) -> bool:
        return self._header.isChecked()

    # ------------------------------------------------------------------ internals

    def _on_toggled(self, on: bool) -> None:
        self._apply(on)
        self.toggled.emit(on)

    def _persist(self, on: bool) -> None:
        if self._settings_key is not None:
            QSettings(_SETTINGS_ORG, _SETTINGS_APP).setValue(self._settings_key, on)

    def _apply(self, on: bool) -> None:
        self._header.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self._body.setVisible(on)

    def _reposition_hint(self) -> None:
        self._hint.adjustSize()
        m = 10
        self._hint.move(self._header.width() - self._hint.width() - m,
                        (self._header.height() - self._hint.height()) // 2)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        super().resizeEvent(event)
        self._reposition_hint()
