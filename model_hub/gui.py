"""PySide6 UI: a publish wizard and a community model browser.

    from model_hub.gui import PublishWizard, ModelBrowserDialog
    PublishWizard(self, model_path=onnx, draft=draft).exec()   # after training
    ModelBrowserDialog(self).exec()

When the Train tab opens the wizard for a model it just trained, ``draft``
already holds every technical field — task, labels, input format, what the
model measured — and those rows stay hidden. The person says what the model
is, picks where it belongs, and confirms the checklist. That is the whole job.
"""
from __future__ import annotations

import re
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QObject, Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QCompleter, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QWizard, QWizardPage,
)

from . import hub
from .manifest import (
    CATEGORY_RE, COMPLIANCE_ITEMS, COLOR_ORDERS, LAYOUTS, LICENSES, NORMALIZATIONS, SLUG_RE,
    TASKS, USABLE_TASKS, InputSpec, Manifest,
)
from .package import build_package

LICENSE_NAMES = {
    "apache-2.0": "Apache 2.0 (recommended)",
    "mit": "MIT",
    "cc-by-4.0": "CC BY 4.0",
    "cc0-1.0": "CC0 (no conditions)",
}


# ------------------------------------------------------------------ worker
class _Worker(QObject):
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable[..., Any], *args, **kwargs):
        super().__init__()
        self._fn, self._args, self._kwargs = fn, args, kwargs

    def run(self):
        try:
            result = self._fn(*self._args, progress=self.progress.emit, **self._kwargs)
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.failed.emit(str(exc) or traceback.format_exc())
        else:
            self.finished.emit(result)


def run_in_thread(owner: QObject, fn, *args, on_progress=None, on_done=None, on_error=None,
                  **kwargs) -> QThread:
    thread = QThread(owner)
    worker = _Worker(fn, *args, **kwargs)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    if on_progress:
        worker.progress.connect(on_progress)
    if on_done:
        worker.finished.connect(on_done)
    if on_error:
        worker.failed.connect(on_error)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread._worker = worker  # keep a reference
    thread.start()
    return thread


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return slug[:64].strip("-")


def _known_categories(progress=None) -> list[str]:
    return hub.categories_in(hub.search_catalog(limit=500))


# ------------------------------------------------------------ wizard pages
class DetailsPage(QWizardPage):
    def __init__(self, model_path: str = "", draft: Manifest | None = None):
        super().__init__()
        self._draft = draft
        self.setTitle("Describe your model")
        self.setSubTitle("Say what it finds and where it belongs. People browse the "
                         "community models by category.")

        self.model_path = QLineEdit(model_path)
        browse = QPushButton("Choose…")
        browse.clicked.connect(self._browse)
        self._model_row = self._hbox(self.model_path, browse)

        self.display_name = QLineEdit(placeholderText="What it finds, e.g. Helicopter detector")
        self.name = QLineEdit(placeholderText="helicopter-detector")
        self._name_edited = False
        self.name.textEdited.connect(lambda _t: setattr(self, "_name_edited", True))
        self.display_name.textChanged.connect(self._sync_name)
        self.description = QPlainTextEdit()
        self.description.setPlaceholderText("What it finds, and on what kind of footage it works well.")
        self.description.setFixedHeight(70)

        self.category = QLineEdit(placeholderText="animals/horses")
        self.category.setToolTip("Where the model is listed: lowercase words, up to three "
                                 "levels, e.g. sports/tennis or games/some-title.")
        self.category_hint = QLabel("")
        self.category_hint.setStyleSheet("color:#999;")
        self.category.textChanged.connect(self._category_changed)

        self.author = QLineEdit(placeholderText="Your name or handle, shown on the model page")
        self.game = QLineEdit(placeholderText="Optional, descriptive only (no logos)")
        self.content_type = QLineEdit(placeholderText="Optional: gameplay, sports, music video…")
        self.license = QComboBox()
        for lic in LICENSES:
            self.license.addItem(LICENSE_NAMES.get(lic, lic), lic)

        # -- technical: filled in from training, shown only for a hand-picked file
        self.task = QComboBox()
        for key in sorted(USABLE_TASKS):
            self.task.addItem(TASKS[key][0], key)
        self.output_format = QComboBox()
        self.task.currentIndexChanged.connect(self._task_changed)
        self.labels = QLineEdit(placeholderText="helicopter, airplane")
        self.width_ = QSpinBox(minimum=16, maximum=2048, value=416)
        self.height_ = QSpinBox(minimum=16, maximum=2048, value=416)
        self.frames = QSpinBox(minimum=1, maximum=128, value=1)
        self.layout_ = QComboBox()
        self.layout_.addItems(sorted(LAYOUTS))
        self.color = QComboBox()
        self.color.addItems(sorted(COLOR_ORDERS))
        self.normalize = QComboBox()
        self.normalize.addItems(sorted(NORMALIZATIONS))
        self.threshold = QDoubleSpinBox(minimum=0.01, maximum=0.99, singleStep=0.05, value=0.3)
        size = QHBoxLayout()
        for w, label in ((self.width_, "W"), (self.height_, "H"), (self.frames, "Frames")):
            size.addWidget(QLabel(label))
            size.addWidget(w)
        size_box = QWidget()
        size_box.setLayout(size)

        outer = QVBoxLayout(self)
        form = QFormLayout()
        outer.addLayout(form)
        outer.addStretch(1)          # extra height below the form, not between rows
        self._form = form
        form.addRow("Model name", self.display_name)
        form.addRow("Description", self.description)
        form.addRow("Category", self.category)
        form.addRow("", self.category_hint)
        form.addRow("Author", self.author)
        form.addRow("Licence", self.license)
        form.addRow("Game", self.game)
        form.addRow("Content type", self.content_type)

        self.measured = QLabel("")
        self.measured.setWordWrap(True)
        self.show_technical = QCheckBox("Show technical details")
        form.addRow("", self.measured)
        form.addRow("", self.show_technical)
        self._technical_rows = []
        for label, widget in (("ONNX model", self._model_row), ("Short name", self.name),
                              ("Task", self.task), ("Output format", self.output_format),
                              ("Labels (comma separated)", self.labels), ("Input size", size_box),
                              ("Layout / color / scaling",
                               self._hbox(self.layout_, self.color, self.normalize)),
                              ("Default confidence", self.threshold)):
            form.addRow(label, widget)
            self._technical_rows.append(widget)
        self.show_technical.toggled.connect(self._set_technical_visible)

        self._task_changed()
        if draft is not None:
            self._apply_draft(draft)
        self._set_technical_visible(draft is None)
        self.show_technical.setVisible(draft is not None)

        for w in (self.model_path, self.name, self.display_name, self.author, self.labels,
                  self.category):
            w.textChanged.connect(self.completeChanged)
        self.description.textChanged.connect(self.completeChanged)

        # Suggest categories people already use; typing a new one is fine too.
        run_in_thread(self, _known_categories, on_done=self._set_categories,
                      on_error=lambda _m: None)

    @staticmethod
    def _hbox(*widgets):
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        for w in widgets:
            lay.addWidget(w)
        return box

    def _apply_draft(self, d: Manifest) -> None:
        idx = self.task.findData(d.task)
        if idx >= 0:
            self.task.setCurrentIndex(idx)
        self._task_changed()
        self.output_format.setCurrentText(d.output_format)
        self.labels.setText(", ".join(d.labels))
        self.width_.setValue(d.input.width)
        self.height_.setValue(d.input.height)
        self.layout_.setCurrentText(d.input.layout)
        self.color.setCurrentText(d.input.color)
        self.normalize.setCurrentText(d.input.normalize)
        self.threshold.setValue(d.confidence_threshold)
        if d.author:
            self.author.setText(d.author)
        found = d.found_sentence()
        self.measured.setText(
            f"Finds: {', '.join(d.labels)}."
            + (f" {found} — this goes on the model page." if found else ""))

    def _set_technical_visible(self, visible: bool) -> None:
        for w in self._technical_rows:
            if hasattr(self._form, "setRowVisible"):      # Qt 6.4+: no gap left behind
                self._form.setRowVisible(w, visible)
            else:
                w.setVisible(visible)
                label = self._form.labelForField(w)
                if label is not None:
                    label.setVisible(visible)

    def _sync_name(self, text: str) -> None:
        if not self._name_edited:
            self.name.setText(slugify(text))

    def _category_changed(self, text: str) -> None:
        text = text.strip()
        if not text:
            self.category_hint.setText("Lowercase words separated by /, e.g. animals/horses")
        elif CATEGORY_RE.match(text):
            self.category_hint.setText(f"Listed under: {' › '.join(text.split('/'))}")
        else:
            self.category_hint.setText("Use lowercase letters, digits and hyphens, "
                                       "up to three levels separated by /")

    def _set_categories(self, categories: list[str]) -> None:
        completer = QCompleter(categories, self)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        completer.setFilterMode(Qt.MatchContains)
        self.category.setCompleter(completer)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose ONNX model", "", "ONNX model (*.onnx)")
        if path:
            self.model_path.setText(path)
            if not self.display_name.text():
                self.display_name.setText(Path(path).stem.replace("_", " "))

    def _task_changed(self):
        task = self.task.currentData()
        self.output_format.clear()
        self.output_format.addItems(sorted(TASKS[task][1]))
        is_action = task == "action_recognition"
        self.frames.setEnabled(is_action)
        self.frames.setValue(16 if is_action else 1)

    def isComplete(self) -> bool:
        return (Path(self.model_path.text()).is_file()
                and bool(SLUG_RE.match(self.name.text()))
                and bool(CATEGORY_RE.match(self.category.text().strip()))
                and all(w.text().strip() for w in (self.display_name, self.author, self.labels))
                and bool(self.description.toPlainText().strip()))

    def manifest(self) -> Manifest:
        labels = [x.strip() for x in self.labels.text().split(",") if x.strip()]
        return Manifest(
            name=self.name.text().strip(),
            display_name=self.display_name.text().strip(),
            description=self.description.toPlainText().strip(),
            task=self.task.currentData(),
            labels=labels,
            author=self.author.text().strip(),
            license=self.license.currentData(),
            category=self.category.text().strip(),
            input=InputSpec(width=self.width_.value(), height=self.height_.value(),
                            layout=self.layout_.currentText(), color=self.color.currentText(),
                            normalize=self.normalize.currentText(), frames=self.frames.value()),
            output_format=self.output_format.currentText(),
            confidence_threshold=round(self.threshold.value(), 3),
            game=self.game.text().strip(),
            content_type=self.content_type.text().strip(),
            metrics=dict(self._draft.metrics) if self._draft is not None else {},
        )


class ChecklistPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Before you share")
        self.setSubTitle("Shared models are public. Confirm each item; sharing stays disabled until you do.")
        lay = QVBoxLayout(self)
        self.boxes: dict[str, QCheckBox] = {}
        for key, text in COMPLIANCE_ITEMS.items():
            box = QCheckBox(text)
            box.setStyleSheet("QCheckBox { padding: 4px 0; }")
            box.toggled.connect(self.completeChanged)
            lay.addWidget(box)
            self.boxes[key] = box
        note = QLabel(
            "Only the model is shared — never your videos, frames or audio. Some "
            "publishers forbid AI training on their games in their terms of service. "
            "If you are not sure, don't share. This is general guidance, not legal advice.")
        note.setWordWrap(True)
        lay.addSpacing(8)
        lay.addWidget(note)
        lay.addStretch()

    def isComplete(self) -> bool:
        return all(b.isChecked() for b in self.boxes.values())

    def compliance(self) -> dict[str, bool]:
        return {k: b.isChecked() for k, b in self.boxes.items()}


class CheckPage(QWizardPage):
    def __init__(self, wizard: "PublishWizard"):
        super().__init__()
        self._wiz = wizard
        self.setTitle("Check the model")
        self.setSubTitle("VideoHighlighter builds the package and runs one test inference on your CPU.")
        self.output = QPlainTextEdit(readOnly=True)
        lay = QVBoxLayout(self)
        lay.addWidget(self.output)
        self.package_dir: Path | None = None
        self._ok = False

    def initializePage(self):
        self._ok = False
        self.output.setPlainText("Checking…")
        manifest = self._wiz.details.manifest()
        manifest.compliance = self._wiz.checklist.compliance()
        self._tmp = tempfile.TemporaryDirectory(prefix="vh-package-")
        out = Path(self._tmp.name) / manifest.name
        try:
            self.package_dir, report = build_package(self._wiz.details.model_path.text(), manifest, out)
        except Exception as exc:  # noqa: BLE001
            self.output.setPlainText(f"✖ {exc}")
            self.completeChanged.emit()
            return
        self._ok = report.ok
        verdict = ("Ready to share." if report.ok
                   else "Fix the problems above, then go back and try again.")
        self.output.setPlainText(report.text() + "\n\n" + verdict)
        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self._ok


class PublishPage(QWizardPage):
    def __init__(self, wizard: "PublishWizard"):
        super().__init__()
        self._wiz = wizard
        self.setTitle("Share on Hugging Face")
        self.setSubTitle("The model is stored in your own free Hugging Face account and "
                         "listed in VideoHighlighter's community models.")
        self.token = QLineEdit(echoMode=QLineEdit.Password)
        self.token.setPlaceholderText("hf_… (token with write permission)")
        saved = hub.get_token()
        if saved:
            self.token.setPlaceholderText("Saved token will be used")
        self.remember = QCheckBox("Remember token in the system credential store")
        self.remember.setChecked(True)
        get_token = QLabel('No account yet? <a href="https://huggingface.co/join">Create one</a>, '
                           'then <a href="https://huggingface.co/settings/tokens">create a token</a> '
                           'with write permission.')
        get_token.setOpenExternalLinks(True)
        get_token.setWordWrap(True)
        self.repo = QLineEdit()
        self.publish_btn = QPushButton("Share")
        self.publish_btn.clicked.connect(self._publish)
        self.log = QPlainTextEdit(readOnly=True)
        self.url: str | None = None

        form = QFormLayout()
        form.addRow("Access token", self.token)
        form.addRow("", get_token)
        form.addRow("", self.remember)
        form.addRow("Repository name", self.repo)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(self.publish_btn, alignment=Qt.AlignLeft)
        lay.addWidget(self.log)

    def initializePage(self):
        self.repo.setText(self._wiz.details.name.text())
        self.url = None
        self.log.clear()

    def _publish(self):
        token = self.token.text().strip() or hub.get_token()
        if not token:
            QMessageBox.warning(self, "Token needed", "Paste a Hugging Face access token with write permission.")
            return
        if self.token.text().strip() and self.remember.isChecked():
            if not hub.save_token(token):
                self.log.appendPlainText("⚠ Couldn't save the token; it will be used only this time.")
        self.publish_btn.setEnabled(False)
        run_in_thread(self, hub.publish, self._wiz.checkpage.package_dir,
                      repo_name=self.repo.text().strip() or None, token=token,
                      on_progress=self.log.appendPlainText,
                      on_done=self._done, on_error=self._error)

    def _done(self, url: str):
        self.url = url
        self.log.appendPlainText(f"\nShared: {url}")
        self.log.appendPlainText("It appears in the community models within a few minutes. Thank you!")
        self.completeChanged.emit()

    def _error(self, message: str):
        self.log.appendPlainText(f"\n✖ {message}")
        self.publish_btn.setEnabled(True)

    def isComplete(self) -> bool:
        return self.url is not None


class PublishWizard(QWizard):
    def __init__(self, parent=None, model_path: str = "", draft: Manifest | None = None):
        super().__init__(parent)
        self.setWindowTitle("Share a model")
        self.setWizardStyle(QWizard.ModernStyle)
        self.resize(720, 640)
        self.details = DetailsPage(model_path=model_path, draft=draft)
        self.checklist = ChecklistPage()
        self.checkpage = CheckPage(self)
        self.publishpage = PublishPage(self)
        for page in (self.details, self.checklist, self.checkpage, self.publishpage):
            self.addPage(page)
        self.setButtonText(QWizard.FinishButton, "Open model page")
        self.finished.connect(self._open_page)

    def _open_page(self, result: int):
        if result == QDialog.Accepted and self.publishpage.url:
            QDesktopServices.openUrl(QUrl(self.publishpage.url))


# ------------------------------------------------------------------ browser
class ModelBrowserDialog(QDialog):
    """Search, install and remove community models."""

    COLUMNS = ["Model", "Category", "Task", "Status", "Downloads", "Updated"]
    installed = Signal(object)          # InstalledModel, so the host can refresh its model list

    def __init__(self, parent=None, models_dir: str | Path | None = None):
        super().__init__(parent)
        self.setWindowTitle("Community models")
        self.resize(900, 540)
        self.models_dir = models_dir
        self.entries: list[hub.CatalogEntry] = []
        self._all: list[hub.CatalogEntry] = []

        self.query = QLineEdit(placeholderText="Search models, e.g. helicopter")
        self.query.returnPressed.connect(self.refresh)
        self.category = QComboBox()
        self.category.addItem("All categories", "")
        self.category.currentIndexChanged.connect(self._filter)
        search = QPushButton("Search")
        search.clicked.connect(self.refresh)
        top = QHBoxLayout()
        top.addWidget(self.query, 1)
        top.addWidget(self.category)
        top.addWidget(search)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.doubleClicked.connect(self._open_page)

        self.status = QLabel("Community models are made by other users. Only models "
                             "marked Verified were reviewed by the VideoHighlighter team.")
        self.status.setWordWrap(True)

        self.install_btn = QPushButton("Install")
        self.install_btn.clicked.connect(self._install)
        self.page_btn = QPushButton("Open model page")
        self.page_btn.clicked.connect(self._open_page)
        self.report_btn = QPushButton("Report a problem")
        self.report_btn.clicked.connect(self._report)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        actions = QHBoxLayout()
        for b in (self.install_btn, self.page_btn, self.report_btn):
            actions.addWidget(b)
        actions.addStretch()
        actions.addWidget(buttons)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.status)
        lay.addLayout(actions)
        self._selection_changed()
        self.refresh()

    # ---------------------------------------------------------------- data
    def _installed_ids(self) -> set[str]:
        return {m.repo_id for m in hub.list_installed(self.models_dir)}

    def refresh(self):
        self.status.setText("Loading community models…")
        run_in_thread(self, _search_with_progress, self.query.text().strip(),
                      on_done=self._loaded, on_error=self._load_failed)

    def _load_failed(self, message: str):
        self.status.setText(f"Couldn't load the community models. Check your internet connection. ({message})")

    def _loaded(self, entries: list[hub.CatalogEntry]):
        self._all = entries
        current = self.category.currentData() or ""
        self.category.blockSignals(True)
        self.category.clear()
        self.category.addItem("All categories", "")
        for cat in hub.categories_in(entries):
            depth = cat.count("/")
            self.category.addItem("    " * depth + cat.split("/")[-1], cat)
        idx = self.category.findData(current)
        self.category.setCurrentIndex(max(0, idx))
        self.category.blockSignals(False)
        self._filter()

    def _filter(self, *_):
        cat = self.category.currentData() or ""
        shown = [e for e in self._all
                 if not cat or e.category == cat or e.category.startswith(cat + "/")]
        self._show(shown)

    def _show(self, entries: list[hub.CatalogEntry]):
        self.entries = entries
        installed = self._installed_ids()
        self.table.setRowCount(len(entries))
        for row, e in enumerate(entries):
            parts = []
            if e.verified:
                parts.append("Verified")
            if e.repo_id in installed:
                parts.append("installed")
            if not e.usable:
                parts.append("needs a newer app")
            status = ", ".join(parts) or "Community"
            values = [e.repo_id, e.category or "—", TASKS.get(e.task, ("—",))[0], status,
                      str(e.downloads), e.last_modified[:10]]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 4:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row, col, item)
        if entries:
            self.status.setText(f"{len(entries)} model(s).")
        elif not self._all:
            self.status.setText("No community models yet. Train one in the Train tab "
                                "and be the first to share it.")
        else:
            self.status.setText("No models match. Try another search or category.")
        self._selection_changed()

    def _current(self) -> hub.CatalogEntry | None:
        rows = self.table.selectionModel().selectedRows()
        return self.entries[rows[0].row()] if rows else None

    def _selection_changed(self):
        e = self._current()
        for b in (self.page_btn, self.report_btn):
            b.setEnabled(e is not None)
        self.install_btn.setEnabled(e is not None and e.usable)

    # ------------------------------------------------------------- actions
    def _install(self):
        e = self._current()
        if not e:
            return
        if not e.verified:
            answer = QMessageBox.question(
                self, "Install community model",
                f"{e.repo_id} was made by another user and hasn't been reviewed.\n\n"
                "VideoHighlighter checks the file before use and runs it only through "
                "ONNX Runtime or OpenVINO. Install it?")
            if answer != QMessageBox.Yes:
                return
        self.install_btn.setEnabled(False)
        run_in_thread(self, hub.install, e.repo_id, e.sha or None, self.models_dir,
                      on_progress=self.status.setText,
                      on_done=self._installed, on_error=self._install_failed)

    def _installed(self, result):
        model, report = result
        if model is None:
            QMessageBox.warning(self, "Model not installed",
                                "The model failed the safety and compatibility checks:\n\n" + report.text())
            self.status.setText("Install cancelled: the model failed the checks.")
        else:
            self.status.setText(f"Installed {model.manifest.display_name}. "
                                "Pick it under Advanced → object model.")
            self.installed.emit(model)
        self._filter()

    def _install_failed(self, message: str):
        self.status.setText(f"Install failed: {message}")
        self._selection_changed()

    def _open_page(self, *_):
        e = self._current()
        if e:
            QDesktopServices.openUrl(QUrl(e.url))

    def _report(self):
        e = self._current()
        if e:
            QDesktopServices.openUrl(QUrl(f"{e.url}/discussions/new"))


def _search_with_progress(query: str, progress=None):
    return hub.search_catalog(query=query)
