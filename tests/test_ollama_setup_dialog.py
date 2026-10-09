"""The "use Ollama on another PC" dialog: copy works, the test tells cases apart.

The probe is always injected — a test that dials a real address passes or fails
on what the network is doing, and a dead one costs its timeout every run.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

HOST = "http://192.168.1.102:11434"


@pytest.fixture(scope="module")
def app():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def make(app, probe):
    from modules.ui.ollama_setup_dialog import OllamaSetupDialog
    return OllamaSetupDialog(host=HOST, probe=probe)


def test_one_tab_per_system_with_that_systems_commands(app):
    from llm.ollama_host import SETUP_SYSTEMS, server_setup
    dialog = make(app, lambda host: [])
    assert [dialog.tabs.tabText(i) for i in range(dialog.tabs.count())] == list(SETUP_SYSTEMS)
    for system in SETUP_SYSTEMS:
        assert dialog.texts[system].toPlainText() == server_setup(system, HOST)


def test_copy_puts_the_commands_on_the_clipboard(app):
    from PySide6.QtGui import QGuiApplication
    dialog = make(app, lambda host: [])
    copied = dialog.copy("Windows")
    assert QGuiApplication.clipboard().text() == copied
    assert "New-NetFirewallRule" in copied


def test_unreachable_is_not_reported_as_connected(app):
    def dead(host):
        raise ConnectionError("timed out")
    dialog = make(app, dead)
    assert dialog.test() is False
    assert not dialog.connected
    assert "No answer" in dialog.result.text()


def test_reachable_with_models(app):
    asked = []
    dialog = make(app, lambda host: asked.append(host) or ["llama3.2:latest"])
    assert dialog.test() is True and dialog.connected
    assert asked == [HOST]
    assert "1 model" in dialog.result.text()


def test_reachable_but_empty_is_still_connected(app):
    dialog = make(app, lambda host: [])
    assert dialog.test() is True
    assert "no models yet" in dialog.result.text()
