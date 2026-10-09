"""What to run on another PC so this one can use its Ollama.

Pointing the app at a server on the network is one field, but the server will
not answer until it is told to: Ollama listens on 127.0.0.1 only, and Windows'
firewall drops port 11434 without a reply. From this side that is
indistinguishable from a machine that is off, and the fix is on a machine none
of our code runs on. So the most the app can do is show the exact lines, let
them be copied in one click, and check the result from here afterwards.

The commands themselves are :func:`llm.ollama_host.server_setup`; the check is
injected so a test opens this with no network.
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase, QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
    QPlainTextEdit, QPushButton, QTabWidget, QVBoxLayout, QWidget,
)

from llm.ollama_host import SETUP_SYSTEMS, is_remote, resolve, server_setup


class OllamaSetupDialog(QDialog):
    """Per-OS setup commands with a Copy button each, and a connection test."""

    def __init__(self, parent=None, host: Optional[str] = None,
                 probe: Optional[Callable[[str], list]] = None):
        super().__init__(parent)
        self.setWindowTitle("Use Ollama on another PC")
        self.setMinimumWidth(640)
        self.host = resolve(host)
        self._probe = probe or self._default_probe
        # Whether a test here reached the server, so the caller knows its
        # model list is worth asking for again.
        self.connected = False

        layout = QVBoxLayout(self)

        if is_remote(self.host):
            intro = (f"This app is set to use Ollama at <b>{self.host}</b>.<br>"
                     "Ollama only answers its own machine until told otherwise. "
                     "Run these <b>on that PC</b>, not this one:")
        else:
            intro = ("Ollama host is set to this machine. Put the other PC's "
                     "address in the <b>Ollama host</b> field first, then run "
                     "these <b>on that PC</b>:")
        head = QLabel(intro)
        head.setWordWrap(True)
        head.setTextFormat(Qt.RichText)
        layout.addWidget(head)

        self.tabs = QTabWidget()
        self.texts = {}
        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        for system in SETUP_SYSTEMS:
            page = QWidget()
            col = QVBoxLayout(page)
            text = QPlainTextEdit(server_setup(system, self.host))
            text.setReadOnly(True)
            text.setFont(mono)
            text.setLineWrapMode(QPlainTextEdit.NoWrap)
            col.addWidget(text)
            row = QHBoxLayout()
            row.addStretch()
            copy = QPushButton("Copy")
            copy.clicked.connect(lambda _=False, s=system: self.copy(s))
            row.addWidget(copy)
            col.addLayout(row)
            self.texts[system] = text
            self.tabs.addTab(page, system)
        layout.addWidget(self.tabs)

        note = QLabel(
            "Newer Ollama apps can do the first step from their own Settings "
            "(\"Expose Ollama to the network\"); Windows still needs the "
            "firewall rule. If it still fails, check that PC's network is set "
            "to Private, not Public.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#999;")
        layout.addWidget(note)

        test_row = QHBoxLayout()
        self.test_btn = QPushButton("Test connection")
        self.test_btn.clicked.connect(self.test)
        test_row.addWidget(self.test_btn)
        self.result = QLabel("")
        self.result.setWordWrap(True)
        test_row.addWidget(self.result, 1)
        layout.addLayout(test_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _default_probe(host: str) -> list:
        """The server's model list; raises if nothing answered.

        Not ``get_ollama_models``, which returns ``[]`` for both "unreachable"
        and "reachable with nothing pulled" — exactly the two this button is
        here to tell apart.
        """
        import requests
        resp = requests.get(f"{host}/api/tags", timeout=4)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]

    def copy(self, system: str) -> str:
        text = self.texts[system].toPlainText()
        QGuiApplication.clipboard().setText(text)
        self.result.setText(f"{system} commands copied.")
        self.result.setStyleSheet("")
        return text

    def test(self) -> bool:
        self.result.setText(f"Asking {self.host}…")
        self.result.setStyleSheet("")
        self.test_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            models = list(self._probe(self.host) or [])
        except Exception as exc:
            print(f"⚠️ Ollama connection test to {self.host} failed: {exc}")
            self.result.setText(f"No answer from {self.host} yet.")
            self.result.setStyleSheet("color:#f44336;")
            return False
        finally:
            QApplication.restoreOverrideCursor()
            self.test_btn.setEnabled(True)
        if models:
            self.result.setText(f"Connected — {len(models)} model(s) on {self.host}.")
        else:
            self.result.setText(f"Connected, but {self.host} has no models yet "
                                "— run `ollama pull <name>` there.")
        self.result.setStyleSheet("color:#4CAF50;")
        self.connected = True
        return True
