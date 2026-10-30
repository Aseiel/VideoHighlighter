"""The main window's menu bar: View and Help.

Only what the window does not already show: the debug log switch (moved here
from the bottom bar), the interface language, and help. Everything else has
its own control in the window and is not repeated. Nothing here keeps state
of its own; items call the window's handlers.
"""
from __future__ import annotations

import os

from PySide6.QtCore import QUrl
from PySide6.QtGui import QActionGroup, QDesktopServices, QKeySequence
from PySide6.QtWidgets import QMenuBar

from modules import i18n
from modules.system import debug_console
from modules.system.app_paths import resource_path

HELP_DIR = os.path.join("assets", "help")


def help_page_path(language: str | None = None) -> str:
    """The bundled How to use page in ``language``, else in English."""
    code = language or i18n.current_language()
    for candidate in (code, code.split("-")[0], "en"):
        path = resource_path(os.path.join(HELP_DIR, f"how-to-use.{candidate}.html"))
        if os.path.isfile(path):
            return path
    return resource_path(os.path.join(HELP_DIR, "how-to-use.en.html"))


def open_help_page(language: str | None = None) -> bool:
    return QDesktopServices.openUrl(QUrl.fromLocalFile(help_page_path(language)))


def _show_tab(gui, title: str) -> None:
    """Leave Simple view and bring the settings tab called ``title`` forward."""
    if hasattr(gui, "set_simple_start"):
        gui.set_simple_start(False)
    tabs = gui.tabs
    for index in range(tabs.count()):
        if tabs.tabText(index) == i18n.t(title):
            tabs.setCurrentIndex(index)
            return


def _language_menu(menu, gui) -> None:
    """Every interface language; choosing one goes through the About tab's
    picker, which saves it and says a restart applies it."""
    combo = gui.ui_language_combo
    sub = menu.addMenu(i18n.t("Language"))
    group = QActionGroup(sub)
    group.setExclusive(True)
    actions = []
    for index in range(combo.count()):
        action = sub.addAction(combo.itemText(index))
        action.setCheckable(True)
        action.setChecked(index == combo.currentIndex())
        group.addAction(action)
        action.triggered.connect(
            lambda _checked=False, i=index: combo.setCurrentIndex(i))
        actions.append(action)

    def _sync(current):
        for i, action in enumerate(actions):
            action.setChecked(i == current)

    combo.currentIndexChanged.connect(_sync)


def build_menu_bar(gui) -> QMenuBar:
    """The menu bar for ``gui``, the main window. Call once its widgets exist."""
    bar = QMenuBar(gui)

    # Nothing the window already shows is repeated here: the bottom bar,
    # the checkboxes and the settings tabs are where those things live.
    view_menu = bar.addMenu(i18n.t("View"))
    debug_log = view_menu.addAction(i18n.t("Debug log"))
    debug_log.setCheckable(True)
    debug_log.setChecked(debug_console.is_console_visible())
    debug_log.toggled.connect(debug_console.set_console_visible)
    debug_console.register_checkbox(debug_log)
    gui.debug_log_action = debug_log
    view_menu.addSeparator()
    _language_menu(view_menu, gui)

    help_menu = bar.addMenu(i18n.t("Help"))
    how = help_menu.addAction(i18n.t("How to use"), lambda: open_help_page())
    how.setShortcut(QKeySequence.HelpContents)
    help_menu.addSeparator()
    help_menu.addAction(
        i18n.t("Open debug log file"),
        lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(debug_console.log_file_path())))
    help_menu.addAction(i18n.t("Check for updates"),
                        lambda: gui._start_update_check(force=True))
    help_menu.addSeparator()
    from modules.ui import pro_offer
    if pro_offer.is_free_build() and pro_offer.is_enabled():
        help_menu.addAction(
            "VideoHighlighter Pro",           # a name: never translated
            lambda: QDesktopServices.openUrl(QUrl(pro_offer.PRO_URL)))
    help_menu.addAction(i18n.t("About VideoHighlighter"), lambda: _show_tab(gui, "About"))
    return bar
