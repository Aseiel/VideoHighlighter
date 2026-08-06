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
"""
from __future__ import annotations

from typing import Optional, Sequence

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
    :attr:`chosen` after ``exec()``, so the dialog never touches settings and
    can be opened in a test without one.
    """

    def __init__(self, parent=None, models: Optional[Sequence] = None,
                 chosen: Optional[str] = None):
        super().__init__(parent)
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
        self.tag = QLineEdit()
        self.tag.setPlaceholderText("llama3.2")
        self.tag_row = self._labelled("Model name:", self.tag)
        form.addRow(self.tag_row)

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

        self._backend_changed()
        self._refill()

    def _labelled(self, text: str, field: QLineEdit,
                  browse: Optional[str] = None) -> QWidget:
        """A form row carrying its own label, so it can be hidden as one unit."""
        wrap = QWidget()
        row = QHBoxLayout(wrap)
        row.setContentsMargins(0, 0, 0, 0)
        caption = QLabel(text)
        caption.setMinimumWidth(110)
        row.addWidget(caption)
        row.addWidget(field, 1)
        if browse:
            button = QPushButton("Browse…")
            button.clicked.connect(lambda: self._browse(field, browse))
            row.addWidget(button)
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
        self.gguf_row.setVisible(is_gguf)
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
        name = (self.gguf if backend == "llama-cpp" else self.tag).text().strip()
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
        for field in (self.tag, self.gguf, self.mmproj, self.label):
            field.clear()
        self._refill()

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
