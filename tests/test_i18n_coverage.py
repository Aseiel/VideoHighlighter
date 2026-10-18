"""UI text in other languages: what reaches the catalog, and what it holds.

The scoring panel and the timeline came up half in English in every language.
Two reasons: strings that were never in the catalog, and text the Qt hooks
never saw (dock titles, menus, the status bar, text drawn into a graphics
scene, padded labels). These pin the second, and the catalog's integrity.
"""
from __future__ import annotations

import pytest

from modules import i18n
from modules.i18n import catalog
from modules.i18n.checks import problems


@pytest.fixture()
def lang():
    def use(code):
        i18n.set_language(code)
    yield use
    i18n.set_language("en")


class TestLookup:
    def test_padding_is_kept_around_the_translation(self, lang):
        lang("zh-CN")
        assert i18n.t("  Objects") == "  " + i18n.t("Objects")
        assert i18n.t("  Objects") != "  Objects"

    def test_an_empty_entry_falls_back_to_english(self, lang, monkeypatch):
        lang("ko")
        from modules.i18n import core
        monkeypatch.setitem(core._catalog, "ko", {"⏸ Pause": ""})
        assert i18n.t("⏸ Pause") == "⏸ Pause"


class TestChecks:
    def test_markup_and_fields_must_survive(self):
        assert problems("<b>Hi</b> {n}", "<b>Salut</b> {n}") == []
        assert "HTML tags differ" in problems("<b>Hi</b>", "Salut")
        assert "format fields differ" in problems("{n} clips", "clips")
        assert "line breaks differ" in problems("a\nb", "a b")
        assert "empty" in problems("Pause", " ")

    def test_a_lone_ampersand_is_a_shortcut_marker(self):
        assert problems("Movement && scenes", "Bewegung && Szenen") == []
        assert problems("Movement && scenes", "运动和场景") == []
        assert problems("Movement && scenes", "Bewegung & Szenen")


class TestCatalog:
    def test_every_language_has_the_same_strings(self):
        keys = {lang: set(t) for lang, t in catalog.TRANSLATIONS.items()}
        first = next(iter(keys.values()))
        assert {lang for lang, k in keys.items() if k != first} == set()

    def test_no_entry_is_structurally_broken(self):
        bad = {(lang, k): problems(k, v)
               for lang, t in catalog.TRANSLATIONS.items()
               for k, v in t.items() if problems(k, v)}
        assert bad == {}


class TestHooks:
    """Text outside labels and buttons now goes through t() too."""

    @pytest.fixture()
    def qt(self, lang):
        pytest.importorskip("PySide6")
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        lang("zh-CN")
        i18n.install_qt_hooks()
        return i18n.t("Objects")

    def test_menus_and_actions(self, qt):
        from PySide6.QtWidgets import QMenu
        menu = QMenu()
        assert menu.addAction("Objects").text() == qt

    def test_dock_and_window_titles(self, qt):
        from PySide6.QtWidgets import QDockWidget, QWidget
        assert QDockWidget("Objects").windowTitle() == qt
        w = QWidget()
        w.setWindowTitle("Objects")
        assert w.windowTitle() == qt

    def test_status_bar(self, qt):
        from PySide6.QtWidgets import QStatusBar
        bar = QStatusBar()
        bar.showMessage("Objects")
        assert bar.currentMessage() == qt

    def test_scene_text(self, qt):
        from PySide6.QtWidgets import QGraphicsScene
        assert QGraphicsScene().addText("Objects").toPlainText() == qt

    def test_numbers_are_left_alone(self, qt):
        from PySide6.QtWidgets import QLabel
        assert QLabel("00:00 / 23:01").text() == "00:00 / 23:01"
