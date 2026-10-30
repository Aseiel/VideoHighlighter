"""The main window's menu bar, and the How to use page it opens.

A stand-in window with the attributes the menu reads: the menu calls the
window's own handlers and buttons, so those are what is checked.
"""

from __future__ import annotations

import os
import re

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QComboBox, QMenu, QTabWidget, QWidget

from modules import i18n
from modules.i18n import catalog
from modules.ui import main_menu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELP = os.path.join(ROOT, "assets", "help")


def _app():
    return QApplication.instance() or QApplication([])


class _Window(QWidget):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.tabs = QTabWidget()
        for title in ("Download", "Basic Settings", "Train", "About"):
            self.tabs.addTab(QWidget(), i18n.t(title))
        self.ui_language_combo = QComboBox()
        for label in ("English", "Polski", "简体中文"):
            self.ui_language_combo.addItem(label)

    def set_simple_start(self, on, persist=True):
        self.calls.append(("simple", on))

    def _start_update_check(self, force=False):
        self.calls.append(("update", force))


@pytest.fixture()
def window():
    _app()
    i18n.set_language("en")
    win = _Window()
    yield win, main_menu.build_menu_bar(win)
    i18n.set_language("en")


def _menu(bar, title):
    # By title among the bar's menus: QAction.menu() hands back a wrapper
    # PySide may already have let go of.
    for menu in bar.findChildren(QMenu):
        if menu.title() == title:
            return menu
    raise AssertionError(f"no {title!r} menu")


def _item(menu, text):
    for action in menu.actions():
        if action.text() == text:
            return action
    raise AssertionError(f"no {text!r} item")


def _all_items(bar):
    return [a.text() for menu in bar.findChildren(QMenu) for a in menu.actions()
            if a.text()]


def test_view_and_help_only(window):
    _win, bar = window
    assert [a.text() for a in bar.actions()] == ["View", "Help"]


def test_nothing_the_window_already_shows_is_repeated(window):
    """The bottom bar, the checkboxes and the settings tabs have these."""
    _win, bar = window
    items = _all_items(bar)
    for shown in ("Add videos…", "Clear video list", "Simple view", "Timeline Viewer",
                  "Highlight Report", "Live detection preview", "Import model…",
                  "Community models…", "Teach a model…", "Download videos from a page…",
                  "Local AI (Ollama) setup…"):
        assert shown not in items, shown


def test_help_items_call_the_windows_own_handlers(window):
    win, bar = window
    _item(_menu(bar, "Help"), "Check for updates").trigger()
    _item(_menu(bar, "Help"), "About VideoHighlighter").trigger()

    assert win.calls == [("update", True), ("simple", False)]
    assert win.tabs.tabText(win.tabs.currentIndex()) == "About"


def test_language_items_follow_the_about_tab_picker(window):
    win, bar = window
    languages = _menu(bar, "Language")
    items = languages.actions()

    items[2].trigger()
    assert win.ui_language_combo.currentIndex() == 2
    items[2].trigger()              # choosing it again keeps it chosen
    assert [a.isChecked() for a in items] == [False, False, True]
    win.ui_language_combo.setCurrentIndex(1)
    assert [a.isChecked() for a in items] == [False, True, False]


def test_help_has_how_to_use_on_f1(window):
    _win, bar = window
    how = _item(_menu(bar, "Help"), "How to use")
    assert how.shortcut().toString() == "F1"


def test_menu_titles_are_translated_in_every_language():
    for lang, table in catalog.TRANSLATIONS.items():
        for msgid in ("View", "Help", "How to use", "Language", "Debug log"):
            assert table.get(msgid), (lang, msgid)


# ---------------------------------------------------------------------------
# The How to use page
# ---------------------------------------------------------------------------

def test_the_page_follows_the_interface_language():
    assert main_menu.help_page_path("zh-CN").endswith("how-to-use.zh-CN.html")
    assert main_menu.help_page_path("en").endswith("how-to-use.en.html")
    # No page of its own yet: English rather than nothing.
    assert main_menu.help_page_path("pl").endswith("how-to-use.en.html")


@pytest.mark.parametrize("name", sorted(os.listdir(HELP)))
def test_every_page_works_offline(name):
    with open(os.path.join(HELP, name), encoding="utf-8") as fh:
        page = fh.read()
    # Nothing fetched from the network: the page has to open in China, and
    # with no connection at all.
    assert not re.search(r'(src|href)\s*=\s*"https?:', page)
    assert "<link" not in page


def test_every_page_has_the_same_sections():
    ids = {}
    for name in os.listdir(HELP):
        with open(os.path.join(HELP, name), encoding="utf-8") as fh:
            ids[name] = re.findall(r'<h2 id="([a-z]+)"', fh.read())
    assert len(set(map(tuple, ids.values()))) == 1, ids


def test_the_chinese_page_names_buttons_as_the_chinese_interface_does():
    zh = catalog.TRANSLATIONS["zh-CN"]
    with open(os.path.join(HELP, "how-to-use.zh-CN.html"), encoding="utf-8") as fh:
        page = fh.read()
    for msgid in ("Simple view", "Analyze", "Add Videos", "Basic Settings",
                  "Run Highlighter", "Score only (no video)", "Export clips",
                  "Timeline Viewer",
                  "Highlight Report", "AI Summary", "Render Highlight Video"):
        assert zh[msgid] in page, msgid


def test_the_main_window_has_the_menu_and_no_debug_checkbox_in_the_bar():
    with open(os.path.join(ROOT, "main.py"), encoding="utf-8") as fh:
        source = fh.read()
    assert "root.setMenuBar(build_menu_bar(self))" in source
    assert 'QCheckBox("Debug log")' not in source


def test_the_bar_has_one_clip_setting_and_score_only_rides_run():
    with open(os.path.join(ROOT, "main.py"), encoding="utf-8") as fh:
        source = fh.read()
    # Keep temp clips has no control; its config value is kept and written back.
    assert "keep_temp_chk" not in source
    assert source.count('"keep_temp": self.keep_temp,') == 3
    assert 'QCheckBox(i18n.t("Export clips"))' in source
    # Scoring without a video is a run: an option of Run, not a report button.
    assert "report_only_btn" not in source
    assert "_joined_buttons(self.run_btn, self.run_opts_btn)" in source
    assert "self.run_pipeline(report_only=True)" in source


def test_joined_buttons_keep_their_shared_edge_through_a_restyle():
    """Run turns orange (Pause) and back by replacing its own stylesheet; the
    join is a theme rule on a property, so it outlives that."""
    from modules.ui import theme
    css = theme.build_qss(theme.DARK)
    assert 'QPushButton[joined="main"]' in css
    assert 'QPushButton[joined="side"]' in css
