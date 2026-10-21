"""The first start's one question: which language.

Shown when no language has been chosen and no installer asked (the mac app,
the portable build, a run from source). Every language is listed by its own
name, the system's language is preselected, and the dialog's own words follow
the selection, so somebody who reads only Chinese can still answer it.
"""
from __future__ import annotations

from PySide6.QtCore import QLocale, Qt
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QLabel, QListWidget,
                               QListWidgetItem, QVBoxLayout)

from modules.i18n import content_language
from modules.i18n.core import current_language, set_language, t
from modules.i18n.languages import display_label, selectable_languages

TITLE = "Choose your language"
EXPLAIN = ("VideoHighlighter's menus, and the language it listens for in "
           "transcripts, writes subtitles and narration in, and understands in "
           "search, all follow this. You can change it later under About.")


def system_guess() -> str:
    try:
        return content_language.guess_from_locale(QLocale.system().uiLanguages())
    except Exception:  # noqa: BLE001
        return "en"


class LanguageDialog(QDialog):
    def __init__(self, preselect: str, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.WindowContextHelpButtonHint, False)
        self._heading = QLabel()
        self._heading.setStyleSheet("font-size: 15pt; font-weight: 600;")
        self._explain = QLabel()
        self._explain.setWordWrap(True)
        self._list = QListWidget()
        for lang in selectable_languages():
            item = QListWidgetItem(display_label(lang))
            item.setData(Qt.UserRole, lang.code)
            self._list.addItem(item)
            if lang.code == preselect:
                self._list.setCurrentItem(item)
        if self._list.currentItem() is None:
            self._list.setCurrentRow(0)
        self._list.currentItemChanged.connect(lambda *_: self._retranslate())
        self._list.itemDoubleClicked.connect(lambda *_: self.accept())
        self._buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        self._buttons.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(self._heading)
        layout.addWidget(self._explain)
        layout.addWidget(self._list, 1)
        layout.addWidget(self._buttons)
        self.resize(460, 520)
        self._retranslate()

    def code(self) -> str:
        item = self._list.currentItem()
        return item.data(Qt.UserRole) if item else "en"

    def _retranslate(self) -> None:
        before = current_language()
        set_language(self.code())
        try:
            self.setWindowTitle(t(TITLE))
            self._heading.setText(t(TITLE))
            self._explain.setText(t(EXPLAIN))
        finally:
            set_language(before)


def ask_language(parent=None) -> str:
    """The language the user picks; the system's guess if they close it, and
    without asking where nobody can answer (tests, a headless run)."""
    from PySide6.QtWidgets import QApplication
    guess = system_guess()
    app = QApplication.instance()
    if app is None or app.platformName() in ("offscreen", "minimal"):
        return guess
    dialog = LanguageDialog(guess, parent)
    dialog.exec()
    return dialog.code() if dialog.result() == QDialog.Accepted else guess
