"""Choosing which local model writes the report, on one screen.

The report used to ask for a model with a chain of prompts — backend, then name,
then projector, then label — which is four decisions with no view of what was
already configured and no way back from the second one. The chat panel had
solved the same problem years earlier with a form: a backend, a path with a
Browse button beside it, and the fields that only matter for one backend hidden
until it is picked. This is that form, over the report's list of models.

The list is the part the chat panel does not need and the report does. Reading a
scene and advising on weights suit different models, so the question is not
"which model" once, it is "which of mine, this time" — and a list you can see is
the difference between switching and remembering what you typed last week.

The same argument applies one level down, and for a while did not hold here. The
Ollama field was free text while the panel next door filled a dropdown from the
server, so the report asked the user to remember a tag that the machine it was
about to talk to could simply be asked for. A mistyped tag is indistinguishable
from one that has not been pulled, and the failure surfaces at the end of a
generation somebody waited a minute for. Both fields now offer what is already
there — see :mod:`modules.llm_discovery` — and both stay typable, because a
model being pulled right now is not a reason to refuse the name.

Where the choices come from is injected rather than imported, so the dialog
still opens in a test with no server, no settings and no network.
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton, QVBoxLayout,
    QWidget,
)

from modules.llm_models import BACKENDS, label_for

GGUF_FILTER = "GGUF models (*.gguf);;All files (*)"


class ModelDialog(QDialog):
    """The report's models: what is configured, and how to add another.

    Returns nothing on its own — the caller reads :attr:`models` and
    :attr:`chosen` after ``exec()``, so the report's own model list is never
    written from in here.

    The three callables are everything that reaches outside this window: what
    the Ollama server holds, which GGUF files have been used before, and putting
    one at the top of that shortlist. They default to
    :mod:`modules.llm_discovery`; a test passes its own and touches neither the
    network nor the settings.
    """

    def __init__(self, parent=None, models: Optional[Sequence] = None,
                 chosen: Optional[str] = None,
                 list_models: Optional[Callable] = None,
                 list_recent: Optional[Callable] = None,
                 remember: Optional[Callable] = None):
        super().__init__(parent)
        self._list_models = list_models or self._default_models
        self._list_recent = list_recent or self._default_recent
        self._remember_fn = remember or self._default_remember
        self.setWindowTitle("Models for the report")
        self.models = [dict(m) for m in (models or ())]
        self.chosen = chosen

        root = QVBoxLayout(self)
        root.addWidget(QLabel(
            "The report can be written with any of these. The one selected is "
            "used until you pick another."))

        self.list = QListWidget()
        self.list.setMinimumHeight(110)
        root.addWidget(self.list)

        row = QHBoxLayout()
        self.remove_btn = QPushButton("Remove selected")
        self.remove_btn.clicked.connect(self._remove_selected)
        row.addWidget(self.remove_btn)
        row.addStretch(1)
        root.addLayout(row)

        root.addWidget(QLabel("<b>Add a model</b>"))
        form = QFormLayout()

        self.backend = QComboBox()
        for name in BACKENDS:
            self.backend.addItem(
                "Ollama (local server)" if name == "ollama"
                else "llama-cpp (GGUF file)", name)
        self.backend.currentIndexChanged.connect(self._backend_changed)
        form.addRow("Backend:", self.backend)

        # Ollama takes a tag, llama-cpp takes a file. Two fields rather than one
        # that means different things: the Browse button belongs to only one of
        # them, and a field that sometimes has one is worse than two that are
        # each always themselves.
        #
        # Editable, both of them. The list is what the machine has *now*, and a
        # model being pulled in another window is a perfectly good thing to name
        # — a dropdown that refuses it would be a downgrade on the free text it
        # replaced.
        self.tag = QComboBox()
        self.tag.setEditable(True)
        self.tag.lineEdit().setPlaceholderText("llama3.2")
        self.tag_row = self._labelled("Model name:", self.tag,
                                      button=("Refresh", self._refresh_tags))
        form.addRow(self.tag_row)

        # What the ask turned up, said plainly. "Nothing found" is an ordinary
        # state — Ollama not started, or a user who only runs GGUF files — and
        # showing it as an error would send somebody looking for a fault.
        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status_row = self._labelled("", self.status)
        form.addRow(self.status_row)

        # The files the chat panel has already been used with, above the path
        # they fill in. "Pick one of these, or browse for a new one" reads down
        # the form in the order it is done; the shortlist stays a shortcut to
        # Browse rather than a second, competing way of saying the same thing.
        self.recent = QComboBox()
        self.recent.activated.connect(self._recent_chosen)
        self.recent_row = self._labelled("Recently used:", self.recent)
        form.addRow(self.recent_row)

        self.gguf = QLineEdit()
        self.gguf.setPlaceholderText("D:/models/some-model.Q4_K_M.gguf")
        self.gguf_row = self._labelled("GGUF path:", self.gguf,
                                       browse="Select a GGUF model")
        form.addRow(self.gguf_row)

        self.mmproj = QLineEdit()
        self.mmproj.setPlaceholderText(
            "optional — the mmproj file, for a vision model")
        self.mmproj_row = self._labelled("Vision projector:", self.mmproj,
                                         browse="Select the mmproj file")
        form.addRow(self.mmproj_row)

        self.label = QLineEdit()
        self.label.setPlaceholderText("optional — what to call it in the menu")
        form.addRow("Call it:", self.label)

        add = QPushButton("Add to the list")
        add.clicked.connect(self._add)
        form.addRow("", add)
        root.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.accept)
        buttons.accepted.connect(self.accept)
        root.addWidget(buttons)

        self._fill_tags()
        self._fill_recent()
        self._backend_changed()
        self._refill()

    # ------------------------------------------------------------------ #
    # What this machine already has
    # ------------------------------------------------------------------ #
    @staticmethod
    def _default_models(refresh: bool = False) -> list:
        from modules.llm_discovery import ollama_models
        return ollama_models(refresh=refresh)

    @staticmethod
    def _default_recent() -> list:
        from modules.llm_discovery import recent_gguf
        return recent_gguf()

    @staticmethod
    def _default_remember(path: str) -> None:
        from modules.llm_discovery import remember_gguf
        remember_gguf(path)

    def _fill_tags(self, refresh: bool = False):
        """Offer what the server holds, keeping whatever is half-typed."""
        typed = self.tag.currentText().strip()
        try:
            found = list(self._list_models(refresh=refresh) or [])
        except Exception as exc:                   # pragma: no cover - defensive
            print(f"⚠️ Could not list models: {exc}")
            found = []
        self.tag.clear()
        self.tag.addItems(found)
        self.tag.setCurrentText(typed)
        self.status.setText(
            f"{len(found)} model(s) on the Ollama server." if found else
            "No Ollama server answered — type a name, or start it and press "
            "Refresh.")

    def _refresh_tags(self):
        self._fill_tags(refresh=True)

    def _fill_recent(self):
        try:
            found = list(self._list_recent() or [])
        except Exception as exc:                   # pragma: no cover - defensive
            print(f"⚠️ Could not list recent models: {exc}")
            found = []
        self.recent.clear()
        for path in found:
            self.recent.addItem(os.path.basename(path), path)
        if not found:
            self.recent.addItem("(none yet — use Browse)", "")
        self.recent.setEnabled(bool(found))

    def _recent_chosen(self, _index):
        path = self.recent.currentData()
        if path:
            self.gguf.setText(str(path))

    def _labelled(self, text: str, field: QWidget,
                  browse: Optional[str] = None,
                  button: Optional[tuple] = None) -> QWidget:
        """A form row carrying its own label, so it can be hidden as one unit."""
        wrap = QWidget()
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        caption = QLabel(text)
        caption.setMinimumWidth(110)
        row.addWidget(caption)
        row.addWidget(field, 1)
        if browse:
            find = QPushButton("Browse…")
            find.clicked.connect(lambda: self._browse(field, browse))
            row.addWidget(find)
        if button:
            label, handler = button
            extra = QPushButton(label)
            extra.clicked.connect(handler)
            row.addWidget(extra)
        return wrap

    def _browse(self, field: QLineEdit, title: str):
        path, _ = QFileDialog.getOpenFileName(self, title, field.text().strip(),
                                              GGUF_FILTER)
        if path:
            field.setText(path)

    def _backend_changed(self, *_args):
        """Only the fields that mean something for this backend."""
        is_gguf = self.backend.currentData() == "llama-cpp"
        self.tag_row.setVisible(not is_gguf)
        self.status_row.setVisible(not is_gguf)
        self.gguf_row.setVisible(is_gguf)
        self.recent_row.setVisible(is_gguf)
        self.mmproj_row.setVisible(is_gguf)

    def _refill(self):
        self.list.clear()
        for entry in self.models:
            item = QListWidgetItem(label_for(entry))
            item.setToolTip(f"{entry['backend']} · {entry['model']}")
            item.setData(Qt.UserRole, entry)
            self.list.addItem(item)
            if label_for(entry) == self.chosen:
                item.setSelected(True)
                self.list.setCurrentItem(item)
        if self.list.currentItem() is None and self.list.count():
            self.list.setCurrentRow(0)

    def _add(self):
        backend = self.backend.currentData()
        name = (self.gguf.text() if backend == "llama-cpp"
                else self.tag.currentText()).strip()
        if not name:
            return
        entry = {"backend": backend, "model": name}
        if backend == "llama-cpp" and self.mmproj.text().strip():
            entry["mmproj"] = self.mmproj.text().strip()
        if self.label.text().strip():
            entry["label"] = self.label.text().strip()
        if entry not in self.models:
            self.models.append(entry)
        self.chosen = label_for(entry)
        # A file picked here is one the chat panel should offer next time too --
        # it is the same shortlist, and the user has now used this model twice.
        if backend == "llama-cpp":
            self._remember(name)
        for field in (self.gguf, self.mmproj, self.label):
            field.clear()
        # Not `clear()` on the combo: that empties the offered list along with
        # the text, and the next model would have to be typed after all.
        self.tag.setCurrentText("")
        self._refill()

    def _remember(self, path: str):
        try:
            self._remember_fn(path)
        except Exception as exc:                   # pragma: no cover - defensive
            print(f"⚠️ Could not remember the GGUF path: {exc}")
        self._fill_recent()

    def _remove_selected(self):
        item = self.list.currentItem()
        if item is None:
            return
        entry = item.data(Qt.UserRole)
        self.models = [m for m in self.models if m != entry]
        if self.chosen == label_for(entry):
            self.chosen = label_for(self.models[0]) if self.models else None
        self._refill()

    def accept(self):
        """Whatever is selected on the way out is the one that will be used."""
        item = self.list.currentItem()
        if item is not None:
            self.chosen = label_for(item.data(Qt.UserRole))
        super().accept()
