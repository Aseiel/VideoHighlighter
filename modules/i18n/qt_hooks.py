"""Translate UI strings at the Qt boundary.

Wrapping every literal in ``main.py`` by hand is endless and brittle.
Once a language is chosen, constructors and setters that take a visible
string go through ``t()`` so anything in the catalog shows in that
language. Missing keys still fall back to English (the msgid).

Idempotent with explicit ``i18n.t(...)`` call sites: translating twice
is a no-op when the first result is not itself an English msgid.
"""
from __future__ import annotations

from modules.i18n.core import t

_installed = False


def _wrap_arg(args: tuple, index: int) -> tuple:
    if index >= len(args):
        return args
    value = args[index]
    if not isinstance(value, str) or not value.strip():
        return args
    # Skip stylesheets / paths / URLs — not user-facing copy.
    if value.lstrip().startswith(("{", "<html", "http://", "https://", "/", "\\")):
        return args
    if "px;" in value or "background-color:" in value or "font-family:" in value:
        return args
    out = list(args)
    out[index] = t(value)
    return tuple(out)


def _patch_method(cls, name: str, index: int = 0) -> None:
    original = getattr(cls, name)

    def wrapped(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return original(self, *_wrap_arg(args, index), **kwargs)

    setattr(cls, name, wrapped)


def _patch_add_item(cls) -> None:
    """QComboBox / QListWidget addItem — text may be arg 0 or 1 (with icon)."""
    from PySide6.QtGui import QIcon

    original = cls.addItem

    def wrapped(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if not args:
            return original(self, *args, **kwargs)
        if isinstance(args[0], QIcon) and len(args) >= 2 and isinstance(args[1], str):
            args = (args[0], t(args[1]), *args[2:])
        elif isinstance(args[0], str):
            args = (t(args[0]), *args[1:])
        return original(self, *args, **kwargs)

    cls.addItem = wrapped  # type: ignore[method-assign]


def _patch_add_tab(cls) -> None:
    original = cls.addTab

    def wrapped(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # addTab(widget, label) or addTab(widget, icon, label)
        if len(args) >= 2 and isinstance(args[1], str):
            args = (args[0], t(args[1]), *args[2:])
        elif len(args) >= 3 and isinstance(args[2], str):
            args = (args[0], args[1], t(args[2]), *args[3:])
        return original(self, *args, **kwargs)

    cls.addTab = wrapped  # type: ignore[method-assign]


def install() -> None:
    """Patch common PySide6 widgets. Safe to call once after Qt is imported."""
    global _installed
    if _installed:
        return

    from PySide6.QtWidgets import (
        QCheckBox,
        QComboBox,
        QDialogButtonBox,
        QGroupBox,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QRadioButton,
        QTabWidget,
        QTextEdit,
        QToolButton,
    )

    for cls in (QLabel, QPushButton, QCheckBox, QRadioButton, QToolButton):
        _patch_method(cls, "__init__", 0)
        if hasattr(cls, "setText"):
            _patch_method(cls, "setText", 0)
        if hasattr(cls, "setToolTip"):
            _patch_method(cls, "setToolTip", 0)
        if hasattr(cls, "setStatusTip"):
            _patch_method(cls, "setStatusTip", 0)
        if hasattr(cls, "setPlaceholderText"):
            _patch_method(cls, "setPlaceholderText", 0)

    _patch_method(QLineEdit, "setPlaceholderText", 0)
    _patch_method(QLineEdit, "setToolTip", 0)
    _patch_method(QTextEdit, "setPlaceholderText", 0)
    _patch_method(QTextEdit, "setToolTip", 0)

    # Common static message boxes used for notices.
    for static_name in ("information", "warning", "critical", "question"):
        original = getattr(QMessageBox, static_name)

        def _make(orig):
            def wrapped(parent, title, text, *args, **kwargs):  # type: ignore[no-untyped-def]
                if isinstance(title, str):
                    title = t(title)
                if isinstance(text, str):
                    text = t(text)
                return orig(parent, title, text, *args, **kwargs)

            return wrapped

        setattr(QMessageBox, static_name, staticmethod(_make(original)))

    _patch_method(QGroupBox, "__init__", 0)
    _patch_method(QGroupBox, "setTitle", 0)
    _patch_method(QGroupBox, "setToolTip", 0)

    _patch_add_item(QComboBox)
    _patch_add_tab(QTabWidget)

    # Form rows: addRow(label, field) when label is a str
    from PySide6.QtWidgets import QFormLayout

    _orig_add_row = QFormLayout.addRow

    def add_row(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if args and isinstance(args[0], str):
            args = (t(args[0]), *args[1:])
        return _orig_add_row(self, *args, **kwargs)

    QFormLayout.addRow = add_row  # type: ignore[method-assign]

    # Message-box buttons created with a bare string
    _orig_box_add = QDialogButtonBox.addButton

    def box_add(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        if args and isinstance(args[0], str):
            args = (t(args[0]), *args[1:])
        return _orig_box_add(self, *args, **kwargs)

    QDialogButtonBox.addButton = box_add  # type: ignore[method-assign]

    _installed = True
