"""The panel that turns labelled examples into a model, without a command line.

Everything this drives already exists and is tested without Qt:
``modules.vision.label_store`` assembles a COCO dataset, ``training.train_yolox_run``
fine-tunes on whatever device is present, and ``training.export_yolox`` converts
the result into the IR the app's detector loads. Until now the only way to reach
any of it was a Python prompt, which meant in practice it was run by whoever
wrote it.

So this widget holds no logic of its own. It picks paths, starts a worker, shows
what the worker says, and stops it when asked. Every number it displays comes
from a callback the training loop already emitted.

**Progress is reported in the user's terms.** Before the run: how long it will
take, on which hardware, and whether that number was measured on this computer
(``training.train_estimate``). During it: which stage it is in, time elapsed and
left, and after every round a sentence about what the model now finds on frames
it was not trained on (``modules.vision.training_preview``) — with a "Watch it learn"
window for anyone who wants to see it. Loss values go to the debug log — they
are the right diagnostic and the wrong progress indicator, because nobody
outside this file can say whether 2.48 is good.

Placement: currently a tab, which is the interim home. The design in
``docs/CUSTOM-MODEL-TRAINING.md`` puts this in the dock beside the video, as a
list of things being taught, each row offering exactly one next action. Nothing
in this widget depends on where it lives, so that move is a change of parent.
"""
from __future__ import annotations

import os
import re
from typing import Optional

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QMessageBox,
    QProgressBar, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from modules.ui.collapsible import CollapsibleSection
from modules.ui.theme import DARK as THEME


class ObjectTrainingWorker(QObject):
    """Assemble, train, export — off the GUI thread.

    One worker for the whole chain rather than three, because the user asked
    for a model and the intermediate artifacts are not decisions they made.
    A failure anywhere surfaces as one error with the stage named.
    """

    progress = Signal(int, str)        # percent, human-readable status
    stage = Signal(int)                # index into STAGES
    round_done = Signal(object)        # modules.vision.training_preview.RoundSnapshot
    finished = Signal(object)          # ExportResult
    error = Signal(str)

    STAGES = ("Collecting frames", "Preparing the model", "Learning", "Saving")

    def __init__(self, store_path: str, work_dir: str, dest_dir: str,
                 epochs: int, batch_size: int, size: str):
        super().__init__()
        self._store_path = store_path
        self._work_dir = work_dir
        self._dest_dir = dest_dir
        self._epochs = epochs
        self._batch_size = batch_size
        self._size = size
        self._stop = False
        # Set from the GUI thread when the live view opens or closes; a plain
        # bool read between rounds, so no lock is needed.
        self.draw_rounds = False

    def cancel(self) -> None:
        """Thread-safe: the loop polls this between steps."""
        self._stop = True

    def _should_stop(self) -> bool:
        return self._stop

    @Slot()
    def run(self) -> None:
        stage = "starting"
        try:
            from modules.vision.label_store import LabelStore, build_dataset
            from training.train_yolox_run import Cancelled, train
            from training.export_yolox import install

            stage = "reading the labels"
            store = LabelStore(self._store_path).load()
            counts = store.counts()
            if not counts:
                raise ValueError(
                    "No accepted labels in this store. Mark some examples and "
                    "accept them before training.")

            import time
            run_started = time.perf_counter()
            stage = "collecting the frames"
            self.stage.emit(0)
            self.progress.emit(0, "Collecting frames from your videos...")
            dataset_dir = os.path.join(self._work_dir, "dataset")
            summary = build_dataset(
                store, dataset_dir,
                progress=lambda done, total: self.progress.emit(
                    int(5 * done / max(1, total)),
                    f"Collecting frames... {done} of {total}"),
            )
            if self._should_stop():
                raise Cancelled("stopped before training")

            trained = summary["splits"]["train"]["images"]
            checked = summary["splits"].get("val", {}).get("images", 0)
            extract_per_frame = ((time.perf_counter() - run_started)
                                 / max(1, trained + checked))
            if trained == 0:
                raise ValueError("No frames could be read from your videos.")

            stage = "preparing the model"
            self.stage.emit(1)
            from training.train_yolox_run import pretrained_path
            first_time = not os.path.exists(pretrained_path(self._size))
            self.progress.emit(5, "Preparing the model..." + (
                " The first run downloads its starting weights (20-70 MB)."
                if first_time else ""))

            from modules.vision.training_preview import pick_frames, snapshot
            preview_frames = pick_frames(dataset_dir)
            history: list = []
            learning_started = [False]

            def on_epoch(report):
                snap = snapshot(report, preview_frames, history, draw=self.draw_rounds)
                print(f"[train] round {report.epoch}: found {snap.found}/{snap.expected}, "
                      f"{snap.false_alarms} false alarm(s), train loss "
                      f"{report.train_loss:.4f}, val loss {report.val_loss:.4f}")
                self.round_done.emit(snap)

            stage = "training"

            def on_progress(update):
                if not learning_started[0]:
                    learning_started[0] = True
                    self.stage.emit(2)
                # 5% was the frame collection; the rest is the training run.
                percent = 5 + int(93 * update.fraction)
                left = _friendly_time(update.eta)
                self.progress.emit(percent, (
                    f"Learning... round {update.epoch} of {update.total_epochs}"
                    + (f", about {left} left" if left else "")))

            result = train(
                dataset_dir=dataset_dir,
                output_dir=os.path.join(self._work_dir, "checkpoint"),
                epochs=self._epochs,
                batch_size=self._batch_size,
                size=self._size,
                progress=on_progress,
                should_stop=self._should_stop,
                on_epoch=on_epoch,
            )
            export_started = time.perf_counter()

            stage = "saving the model"
            self.stage.emit(3)
            self.progress.emit(98, "Saving the model...")
            exported = install(result.weights_path, dest_dir=self._dest_dir)
            _record_speed(
                result, self._size, extract_per_frame,
                # setup + export; skipped on a first run, whose download would skew it
                None if first_time else
                (result.seconds - result.loop_seconds) + (time.perf_counter() - export_started))
            exported.trained_on = trained          # for the finished message
            exported.checked_on = checked
            exported.best_val_loss = result.best_val_loss
            exported.last_round = history[-1] if history else None
            exported.device = result.device
            self.progress.emit(100, "Done.")
            self.finished.emit(exported)

        except Exception as exc:                   # noqa: BLE001 - never crash the GUI
            name = type(exc).__name__
            if name == "Cancelled":
                self.error.emit("Stopped.")
                return
            import traceback
            traceback.print_exc()
            self.error.emit(f"Failed while {stage}: {exc}")


def _record_speed(result, size: str, extract_per_frame=None, fixed_seconds=None) -> None:
    """Remember how fast this computer actually trained, so the next estimate
    is measured rather than guessed. Never fails a run."""
    try:
        from training.train_estimate import ThroughputStore, default_store_path, device_kind
        from training.train_yolox_run import DEFAULT_IMAGE_SIZE
        store = ThroughputStore(default_store_path()).load()
        store.record(device_kind(result.device), size, DEFAULT_IMAGE_SIZE,
                     result.train_images_per_second, result.val_images_per_second)
        store.record_overheads(extract_per_frame, fixed_seconds)
        store.save()
        print(f"[train] measured {result.train_images_per_second:.1f} img/s training, "
              f"{result.val_images_per_second:.1f} img/s validating on {result.device}")
    except Exception as exc:                    # noqa: BLE001
        print(f"[train] could not record training speed: {exc}")


def _probe_training_device() -> tuple:
    """(torch device string, human name) the run will use. Imports torch, so
    it is called off the GUI thread."""
    from training.train_yolox_run import resolve_device
    device = resolve_device("AUTO")
    name = ""
    try:
        import torch
        if device.startswith("xpu"):
            name = torch.xpu.get_device_name(0)
        elif device.startswith("cuda"):
            name = torch.cuda.get_device_name(0)
    except Exception:                           # noqa: BLE001
        pass
    return device, name


def can_finetune(device: str) -> bool:
    """True for a torch device the image model fine-tunes on: Intel (XPU) or
    NVIDIA (CUDA). DirectML and the processor are not offered in the GUI."""
    return str(device).startswith(("xpu", "cuda"))


def _elapsed(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _parse_friendly(text: str) -> float:
    """Inverse of ``_friendly_time``, for the clock between progress events."""
    m = re.match(r"([\d.]+) (second|minute|hour)", text or "")
    if not m:
        return 0.0
    return float(m.group(1)) * {"second": 1, "minute": 60, "hour": 3600}[m.group(2)]


def _friendly_time(seconds: float) -> str:
    """"about 3 minutes left" beats "eta 184.2s" for someone deciding whether
    to go and make tea. Empty string when there is no estimate yet, so the
    caller can leave the phrase out rather than print "about 0 seconds"."""
    seconds = int(seconds or 0)
    if seconds <= 0:
        return ""
    if seconds < 90:
        return f"{seconds} seconds"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours = seconds / 3600
    return f"{hours:.1f} hours"


class ObjectTrainingSection(QWidget):
    """Pick a set of labels, train a detector, install it.

    Objects are taught from **boxes in frames**: where a thing is, in a still.
    That is a different kind of example from an action, which is why this and
    :class:`ActionTrainingSection` are separate rather than one form with a
    mode switch — they take different data and produce different models.
    """

    model_installed = Signal(object)     # ExportResult, for the host to react to
    _device_found = Signal(str, str)     # from the probe thread: device, name

    # Deliberately modest defaults. A first run should finish while somebody is
    # still interested in it; the advanced section is there for the second one.
    DEFAULT_EPOCHS = 30
    DEFAULT_BATCH = 8

    def __init__(self, parent=None, store_path: str = ""):
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        self._worker: Optional[TrainingWorker] = None
        self._store_path = store_path
        self._frames = (0, 0)                # (train, val) the store will produce
        self._device: Optional[tuple] = None  # (device, name) once probed
        self._probing = False
        self._started_at = 0.0
        self._time_left = 0.0
        self._estimate_seconds = 0.0
        self._preview = None                 # TrainingPreviewWindow, when open
        self._rounds: list = []              # every RoundSnapshot of this run
        self._tick = QTimer(self)
        self._tick.setInterval(1000)
        self._tick.timeout.connect(self._update_clock)
        self._device_found.connect(self._on_device_found)
        self._build_ui()
        if store_path:
            self._load_store(store_path)

    # ── layout ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout()

        explain = QLabel(
            "Teach the app to find things of your own. Mark examples, and this "
            "trains a small detector that looks for them in every video.\n"
            "The first model finds some of them, not all — it improves each "
            "time you add more examples."
        )
        explain.setWordWrap(True)
        explain.setStyleSheet("color:#999;")
        root.addWidget(explain)

        # -- where the labels come from --
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Examples:"))
        self.store_label = QLabel("none chosen")
        self.store_label.setStyleSheet("font-style:italic;color:#999;")
        source_row.addWidget(self.store_label, 1)
        browse = QPushButton("Choose...")
        browse.clicked.connect(self._browse_store)
        source_row.addWidget(browse)
        self.import_btn = QPushButton("Import from labeller...")
        self.import_btn.setToolTip(
            "Read a tools/labeler.py export. Its points become boxes of a fixed "
            "size, so they arrive needing review rather than accepted.")
        self.import_btn.clicked.connect(self._import_labeler)
        source_row.addWidget(self.import_btn)
        root.addLayout(source_row)

        self.counts_label = QLabel("")
        self.counts_label.setWordWrap(True)
        root.addWidget(self.counts_label)

        # -- advanced, folded: the point is that nobody has to open it --
        advanced = CollapsibleSection("Advanced", settings_key="training/advanced")
        form = QFormLayout()
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(self.DEFAULT_EPOCHS)
        self.epochs_spin.valueChanged.connect(self._refresh_estimate)
        form.addRow("Rounds of learning:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 64)
        self.batch_spin.setValue(self.DEFAULT_BATCH)
        form.addRow("Frames at a time:", self.batch_spin)

        self.size_combo = QComboBox()
        for size, hint in (("nano", "smallest and fastest"),
                           ("tiny", "recommended"),
                           ("s", "slower, a little more accurate")):
            self.size_combo.addItem(f"{size} - {hint}", size)
        self.size_combo.setCurrentIndex(1)
        self.size_combo.currentIndexChanged.connect(self._refresh_estimate)
        form.addRow("Model size:", self.size_combo)
        advanced.setContentLayout(form)
        root.addWidget(advanced)

        # -- how long, said before the button is pressed --
        self.estimate_label = QLabel("")
        self.estimate_label.setWordWrap(True)
        root.addWidget(self.estimate_label)

        # -- the one button --
        self.train_btn = QPushButton("Train a model")
        self.train_btn.setStyleSheet(
            f"QPushButton{{background:{THEME.success};color:white;"
            f"font-weight:bold;padding:10px 18px;}}")
        self.train_btn.setEnabled(False)
        self.train_btn.clicked.connect(self._start)
        root.addWidget(self.train_btn)

        self.cancel_btn = QPushButton("Stop")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setVisible(False)
        root.addWidget(self.cancel_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        self.stage_label = QLabel("")
        self.stage_label.setTextFormat(Qt.RichText)
        self.stage_label.setVisible(False)
        root.addWidget(self.stage_label)

        self.clock_label = QLabel("")
        self.clock_label.setStyleSheet("color:#999;")
        self.clock_label.setVisible(False)
        root.addWidget(self.clock_label)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        # "How is it going" — one sentence per finished round.
        self.round_label = QLabel("")
        self.round_label.setWordWrap(True)
        self.round_label.setVisible(False)
        root.addWidget(self.round_label)

        self.watch_btn = QPushButton("👁 Watch it learn")
        self.watch_btn.setToolTip(
            "Open a live view: after every round the model looks at the same few "
            "frames it was not trained on, and you can see what it finds.")
        self.watch_btn.clicked.connect(self._open_preview)
        self.watch_btn.setVisible(False)
        root.addWidget(self.watch_btn)

        # Asked once the model has worked for the person, never before, and
        # never automatically: sharing is their decision, made in the wizard.
        self.share_box = QWidget()
        share_layout = QVBoxLayout(self.share_box)
        share_layout.setContentsMargins(0, 8, 0, 0)
        share_note = QLabel(
            "It works for you — would you like to share it, so other people can find "
            "the same things in their videos? Only the model is shared, never your "
            "videos, frames or audio.")
        share_note.setWordWrap(True)
        share_layout.addWidget(share_note)
        self.share_btn = QPushButton("Share this model with the community…")
        self.share_btn.clicked.connect(self._share)
        share_layout.addWidget(self.share_btn)
        self.share_box.setVisible(False)
        root.addWidget(self.share_box)
        self._last_export = None

        root.addStretch()
        self.setLayout(root)

    # ── choosing labels ──────────────────────────────────────────────────

    def _browse_store(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose a set of examples", "", "Labels (*.json);;All files (*)")
        if path:
            self._load_store(path)

    def _load_store(self, path: str) -> None:
        try:
            from modules.vision.label_store import LabelStore
            store = LabelStore(path).load()
        except Exception as exc:
            self._say(f"Could not read that file: {exc}", THEME.danger)
            return

        self._store_path = path
        self.store_label.setText(os.path.basename(path))
        self.store_label.setStyleSheet("")
        counts = store.counts()
        pending = len(store.pending())

        if counts:
            described = ", ".join(f"{name} ({n})" for name, n in sorted(counts.items()))
            note = f"Ready to learn: {described}."
            if pending:
                note += f"  {pending} more still need checking."
            self.counts_label.setText(note)
            self.counts_label.setStyleSheet("")
            self.train_btn.setEnabled(True)
            try:
                from training.train_estimate import frames_in_store
                self._frames = frames_in_store(store)
            except Exception as exc:            # noqa: BLE001
                print(f"[training] could not count frames: {exc}")
                self._frames = (0, 0)
            self._refresh_estimate()
        else:
            self.counts_label.setText(
                f"Nothing accepted yet"
                + (f" — {pending} example(s) are waiting to be checked."
                   if pending else " in this file."))
            self.counts_label.setStyleSheet(f"color:{THEME.warning};")
            self.train_btn.setEnabled(False)
            self._frames = (0, 0)
            self.estimate_label.setText("")

    def _import_labeler(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import a labeller export", "", "Labels (*.json);;All files (*)")
        if not path:
            return
        try:
            from modules.vision.label_store import LabelStore, from_labeler_export
            imported = from_labeler_export(path)
        except Exception as exc:
            self._say(f"Could not import that export: {exc}", THEME.danger)
            return
        if not imported:
            self._say("That export contains no labelled points.", THEME.warning)
            return

        target = self._store_path or os.path.splitext(path)[0] + ".examples.json"
        store = LabelStore(target).load()
        store.extend(imported)
        store.save()
        self._load_store(target)
        self._say(
            f"Imported {len(imported)} example(s). They need checking before "
            f"training, because the labeller records a point rather than the "
            f"size of the thing.", THEME.warning)

    # ── the estimate ─────────────────────────────────────────────────────

    def _refresh_estimate(self, *_args) -> None:
        if sum(self._frames) == 0:
            return
        if self._device is None:
            self.estimate_label.setText("Working out how long training will take...")
            self.estimate_label.setStyleSheet("color:#999;")
            if not self._probing:
                self._probing = True
                import threading

                def probe():
                    try:
                        device, name = _probe_training_device()
                    except Exception as exc:    # noqa: BLE001
                        print(f"[training] device probe failed: {exc}")
                        device, name = "cpu", ""
                    self._device_found.emit(device, name)

                threading.Thread(target=probe, daemon=True).start()
            return
        try:
            from training.train_estimate import (
                ThroughputStore, default_store_path, estimate, friendly_device,
                device_kind)
            from training.train_yolox_run import DEFAULT_IMAGE_SIZE, pretrained_path
            device, name = self._device
            size = self.size_combo.currentData()
            est = estimate(
                self._frames[0], self._frames[1], self.epochs_spin.value(), size,
                device, DEFAULT_IMAGE_SIZE,
                store=ThroughputStore(default_store_path()).load(),
                pretrained_cached=os.path.exists(pretrained_path(size)))
        except Exception as exc:                # noqa: BLE001
            print(f"[training] estimate failed: {exc}")
            self.estimate_label.setText("")
            return
        self._estimate_seconds = est.seconds
        text = est.sentence(friendly_device(device, name))
        if device_kind(device) == "cpu":
            text += (" A graphics card usually makes this several times faster; "
                     "a smaller model size is quicker too.")
        self.estimate_label.setText(text)
        self.estimate_label.setStyleSheet("" if est.seconds < 1800 else f"color:{THEME.warning};")

    @Slot(str, str)
    def _on_device_found(self, device: str, name: str) -> None:
        self._device = (device, name)
        self._probing = False
        self._refresh_estimate()

    # ── the live parts of a run ──────────────────────────────────────────

    def _show_stage(self, index: int) -> None:
        stages = ObjectTrainingWorker.STAGES
        parts = []
        for i, label in enumerate(stages):
            if i < index:
                parts.append(f"<span style='color:{THEME.success};'>✓ {label}</span>")
            elif i == index:
                parts.append(f"<b>▶ {label}</b>")
            else:
                parts.append(f"<span style='color:#777;'>{label}</span>")
        self.stage_label.setText("&nbsp;&nbsp;→&nbsp;&nbsp;".join(parts))

    def _update_clock(self) -> None:
        import time
        spent = time.monotonic() - self._started_at
        text = f"{_elapsed(spent)} elapsed"
        if self._time_left > 0:
            text += f" · about {_friendly_time(self._time_left)} left"
        elif self._estimate_seconds > 0:
            text += f" · expected about {_friendly_time(max(0.0, self._estimate_seconds - spent)) or 'a moment'} more"
        self.clock_label.setText(text)

    @Slot(object)
    def _on_round(self, snap) -> None:
        self._rounds.append(snap)
        self.round_label.setText(snap.sentence())
        self.round_label.setVisible(True)
        if self._preview is not None:
            self._preview.add_round(snap)

    def _share(self) -> None:
        exported = self._last_export
        onnx_path = getattr(exported, "onnx_path", "") if exported is not None else ""
        if not onnx_path or not os.path.exists(onnx_path):
            self._say("The trained model's ONNX file is no longer there, so it cannot "
                      "be shared. Train it again to share it.", THEME.warning)
            return
        try:
            from model_hub.gui import PublishWizard
            from model_hub.package import draft_for_trained_detector
        except Exception as exc:                # noqa: BLE001
            self._say(f"Sharing is unavailable: {exc}", THEME.danger)
            return
        last = getattr(exported, "last_round", None)
        metrics = {
            "heldout_found": last[1] if last else None,
            "heldout_expected": last[2] if last else None,
            "rounds": len(self._rounds) or None,
            "train_frames": getattr(exported, "trained_on", None),
        }
        if self._rounds:
            metrics["false_alarms"] = self._rounds[-1].false_alarms
        draft = draft_for_trained_detector(onnx_path, metrics=metrics)
        PublishWizard(self, model_path=onnx_path, draft=draft).exec()

    def _open_preview(self) -> None:
        from modules.ui.training_preview import TrainingPreviewWindow
        if self._preview is None:
            self._preview = TrainingPreviewWindow(self)
            self._preview.closed.connect(self._on_preview_closed)
            for snap in self._rounds:           # rounds before it opened, as numbers
                self._preview.add_round(snap)
        if self._worker is not None:
            self._worker.draw_rounds = True
        self._preview.show()
        self._preview.raise_()
        self._preview.activateWindow()

    def _on_preview_closed(self) -> None:
        if self._worker is not None:
            self._worker.draw_rounds = False
        self._preview = None

    # ── the run ──────────────────────────────────────────────────────────

    def _start(self) -> None:
        if self._thread is not None:
            return

        # The dataset and checkpoints sit beside the user's own labels, not in
        # the install directory: they are working files about their footage,
        # they are large, and they belong wherever that footage is organised.
        work_dir = os.path.join(os.path.dirname(self._store_path), "training_run")
        # The model, by contrast, goes where the detector looks for it.
        repo_root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        dest_dir = os.path.join(repo_root, "models", "custom")

        self._worker = ObjectTrainingWorker(
            store_path=self._store_path,
            work_dir=work_dir,
            dest_dir=dest_dir,
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            size=self.size_combo.currentData(),
        )
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.stage.connect(self._show_stage)
        self._worker.round_done.connect(self._on_round)
        self._worker.draw_rounds = self._preview is not None
        self._rounds = []
        if self._preview is not None:
            self._preview.reset()
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._set_running(True)
        self._thread.start()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._say("Stopping after this step...", THEME.warning)

    def _set_running(self, running: bool) -> None:
        import time
        self.train_btn.setVisible(not running)
        self.cancel_btn.setVisible(running)
        self.progress_bar.setVisible(running)
        self.import_btn.setEnabled(not running)
        self.stage_label.setVisible(running)
        self.clock_label.setVisible(running)
        self.estimate_label.setVisible(not running)
        self.watch_btn.setVisible(running or bool(self._rounds))
        if running:
            self.share_box.setVisible(False)
        if running:
            self.progress_bar.setValue(0)
            self.round_label.setText("")
            self.round_label.setVisible(False)
            self._started_at = time.monotonic()
            self._time_left = 0.0
            self._show_stage(0)
            self._update_clock()
            self._tick.start()
        else:
            self._tick.stop()

    @Slot(int, str)
    def _on_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(max(0, min(100, percent)))
        # The loop's own estimate, once it has one, replaces the up-front guess.
        match = re.search(r"about (.+) left", message)
        self._time_left = _parse_friendly(match.group(1)) if match else self._time_left
        self._say(re.sub(r", about .+ left", "", message), "")

    @Slot(object)
    def _on_finished(self, exported) -> None:
        import time
        took = time.monotonic() - self._started_at
        self._teardown()
        trained = getattr(exported, "trained_on", 0)
        checked = getattr(exported, "checked_on", 0)
        names = ", ".join(exported.class_names)
        message = (f"Your model is ready. It learned {names} from {trained} "
                   f"frame(s)")
        if checked:
            message += f", checked against {checked} it had not seen"
        last = getattr(exported, "last_round", None)
        if last and last[2]:
            message += (f". On frames it never trained on it found {last[1]} "
                        f"of {last[2]}")
        message += f". Took {_elapsed(took)}."
        message += "\nIt is installed and will be used when you run a scan."
        self._say(message, THEME.success)
        self._last_export = exported
        self.share_box.setVisible(bool(getattr(exported, "onnx_path", "")))
        self._device = None                     # re-probe: speeds were just measured
        self._refresh_estimate()
        self.model_installed.emit(exported)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._teardown()
        self._say(message, THEME.danger)
        if not message.startswith("Stopped"):
            QMessageBox.warning(self, "Training", message)

    def _teardown(self) -> None:
        self._set_running(False)
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread = None
        self._worker = None

    def _say(self, text: str, colour: str) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color:{colour};" if colour else "")

    def closeEvent(self, event):        # noqa: N802 (Qt override)
        """A training run must not outlive its window."""
        self._cancel()
        self._teardown()
        super().closeEvent(event)


class ActionTrainingWorker(QObject):
    """Run the action-head trainer on this worker's thread, reporting what it says.

    ``model_training.action_head.train`` encodes every clip once with the
    SigLIP2 frame encoder and trains a small head on the vectors, scored on
    source videos it never saw. Every line it logs goes to the debug log and
    is read for progress here.

    **In this process, not a child one.** It used to run as
    ``sys.executable -m model_training.action_head.train``, which is right from
    source and wrong in the packaged app: there ``sys.executable`` is the app
    itself, so pressing Train opened a second copy of the app, sat at 0 %, and
    the copy rotated the first one's debug log away. Object training already
    runs in-process for the same reason. Stopping is cooperative: the trainer
    asks ``should_stop`` between clips and between held-out folds, a few
    seconds apart at most, and saves nothing when stopped.

    With ``finetune_blocks`` it also fine-tunes the image model's top blocks
    (``--finetune-blocks``, about an hour on a graphics card). Its progress
    then has three more phases: decoding every clip into the frame cache
    first, and after the frozen folds the fine-tune's folds and its final run.
    """

    progress = Signal(int, str)
    finished = Signal(str)
    error = Signal(str)

    # What the head trainer prints on its way: clips encoded ("  120/400
    # clips"), then held-out folds ("fold 2/5"), then the saved head.
    _ENCODED = re.compile(r"^\s*(\d+)/(\d+) clips,")
    _FOLD = re.compile(r"fold\s+(\d+)\s*/\s*(\d+)")
    # Each training length it compares ("  750 steps") runs every fold again.
    _ROUND = re.compile(r"^\s*\d+ steps$")
    _HELDOUT = re.compile(r"Held out .*accuracy\s+([\d.]+)")
    # Fine-tuning only: clips decoded into the frame cache, then the
    # fine-tune's folds ("fine-tune fold 2/5") and epochs ("epoch 3/10"), then
    # the final run ("final epoch 3/9").
    _DECODED = re.compile(r"^\s*(\d+)/(\d+) clips decoded,")
    _FT_START = "Fine-tuning the top"
    _FT_FOLD = re.compile(r"fine-tune fold\s+(\d+)\s*/\s*(\d+)")
    _EPOCH = re.compile(r"^\s*(final )?epoch\s+(\d+)\s*/\s*(\d+)")
    _FINAL = "Final model:"

    def __init__(self, data_path: str, name: str, finetune_blocks: int = 0):
        super().__init__()
        self._data_path = data_path
        self._name = name
        self._finetune_blocks = int(finetune_blocks)
        self._stop = False
        self._note = ""
        self._saved = ""
        self._last = ""
        self._round = 0
        self._phase = ""
        self._ft_fold = (0, 1)

    @property
    def out_dir(self) -> str:
        """Where the trained model lands: the app's action models folder,
        where the action pass finds it (newest first)."""
        from model_training.action_head.train import default_out
        return default_out(self._name)

    def cancel(self) -> None:
        self._stop = True

    def _log(self, text: str) -> None:
        for line in str(text).splitlines():
            line = line.rstrip()
            if line:
                print(f"[actions] {line}")   # the debug log keeps everything
                self._last = line
                self._note = self._read(line) or self._note

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit(0, "Encoding the clips...")
            try:
                from model_training.action_head import train
            except ImportError as exc:
                import traceback
                traceback.print_exc()
                self.error.emit(
                    f"Action training is not available in this installation "
                    f"({exc}).")
                return
            code = train.main(self._args(), log=self._log,
                              should_stop=lambda: self._stop)
            if self._stop or code == train.STOPPED:
                self.error.emit("Stopped.")
                return
            if code != 0:
                reason = self._last.lstrip("❌⚠️ ").strip()
                self.error.emit(
                    f"Training did not finish: {reason or 'see the debug log'}.")
                return
            self.progress.emit(100, "Done.")
            self.finished.emit("; ".join(n for n in (self._note, self._saved) if n))
        except Exception as exc:                    # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.error.emit(f"Could not run the action trainer: {exc}")

    def _args(self) -> list:
        """The head trainer on the chosen folder; the head is written where the
        app looks for action models (models/actions/<name>)."""
        args = ["--data-path", self._data_path, "--name", self._name]
        if self._finetune_blocks > 0:
            args += ["--finetune-blocks", str(self._finetune_blocks)]
        return args

    def _span(self, phase: str) -> tuple:
        """The share of the bar a phase fills, (start, width) in percent."""
        if self._finetune_blocks > 0:
            return {"decode": (0, 15), "encode": (15, 10), "folds": (25, 10),
                    "finetune": (35, 55), "final": (90, 9)}[phase]
        return {"encode": (0, 60), "folds": (60, 35)}[phase]

    def _emit(self, phase: str, done: float, total: float, message: str) -> None:
        """``done`` of ``total`` through ``phase``, on the bar."""
        start, width = self._span(phase)
        self.progress.emit(start + int(width * max(0.0, min(done, total)) / total), message)

    def _read(self, line: str):
        """Turn one line of the trainer's output into a progress update."""
        from model_training.action_head import train as trainer
        if line.startswith(trainer.SAVED_FINETUNED):
            self._saved = "the image model was fine-tuned too"
            return None
        if line.startswith(trainer.SAVED_FROZEN) and self._finetune_blocks > 0:
            self._saved = ("fine-tuning the image model did not do better on videos it "
                           "had not seen, so the small model was kept")
            return None
        if self._finetune_blocks > 0:
            decoded = self._DECODED.search(line)
            if decoded:
                done, total = int(decoded.group(1)), max(1, int(decoded.group(2)))
                self._emit("decode", done, total, f"Reading the clips... {done} of {total}")
                return None
            if line.startswith(self._FT_START):
                self._phase = "finetune"
                self._emit("finetune", 0, 1, "Training the image model...")
                return None
            if line.startswith(self._FINAL):
                self._phase = "final"
                self._emit("final", 0, 1, "Training the final model...")
                return None
            ft_fold = self._FT_FOLD.search(line)
            if ft_fold:
                self._ft_fold = (int(ft_fold.group(1)), max(1, int(ft_fold.group(2))))
                return None
            epoch = self._EPOCH.search(line)
            if epoch:
                done, total = int(epoch.group(2)), max(1, int(epoch.group(3)))
                if self._phase == "final":
                    self._emit("final", done, total,
                               f"Training the final model... epoch {done} of {total}")
                else:
                    fold, folds = self._ft_fold
                    self._emit("finetune", fold - 1 + done / total, folds,
                               f"Training the image model on videos it has not seen... "
                               f"round {fold} of {folds}, epoch {done} of {total}")
                return None
        encoded = self._ENCODED.search(line)
        if encoded:
            done, total = int(encoded.group(1)), max(1, int(encoded.group(2)))
            self._emit("encode", done, total, f"Encoding clips... {done} of {total}")
            return None
        if self._ROUND.search(line):
            self._round += 1
            return None
        fold = self._FOLD.search(line)
        if fold and not self._phase:
            from model_training.action_head.train import DEFAULT_STEPS
            done, total = int(fold.group(1)), max(1, int(fold.group(2)))
            rounds = len(DEFAULT_STEPS.split(","))
            step = max(0, min(self._round, rounds) - 1)
            self._emit("folds", step + done / total, rounds,
                       f"Testing on videos it has not seen... "
                       f"{step * total + done} of {rounds * total}")
            return None
        held = self._HELDOUT.search(line)
        if held:
            # Accuracy is the one number here worth showing: unlike a loss, a
            # person can read it without knowing the model.
            return (f"recognised {float(held.group(1)) * 100:.0f}% of the clips "
                    f"from videos it had not seen")
        return None


class ActionTrainingSection(QWidget):
    """Train the app to recognise an action of the user's own.

    Actions are taught from **whole clips**, not boxes: one folder per action,
    videos inside it. That is why this is its own section rather than a mode of
    the object one - the example a user has to supply is a different kind of
    thing, and a shared form would ask for the wrong input.
    """

    # The folder of a model it saved, for the host to say so and pick it up.
    model_installed = Signal(str)

    # From the probe thread: the torch device and the card's name.
    _device_found = Signal(str, str)

    # The trainer's own minimums: below these it skips the class.
    MIN_TRAIN_CLIPS = 5
    MIN_VAL_CLIPS = 2
    # Blocks the checkbox fine-tunes: the best single-action accuracy measured
    # (docs/plans/2026-10-08-overfitting-and-siglip2-fine-tune.md).
    FINETUNE_BLOCKS = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        self._worker: Optional[ActionTrainingWorker] = None
        self._data_path = ""
        self._probing = False
        self._device_found.connect(self._on_device_found)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout()

        explain = QLabel(
            "Teach the app to recognise something that happens over time, "
            "rather than something visible in a single frame.\n"
            "Give it one folder per action, with a few short clips inside each."
        )
        explain.setWordWrap(True)
        explain.setStyleSheet("color:#999;")
        root.addWidget(explain)

        row = QHBoxLayout()
        row.addWidget(QLabel("Clips folder:"))
        self.folder_label = QLabel("none chosen")
        self.folder_label.setStyleSheet("font-style:italic;color:#999;")
        row.addWidget(self.folder_label, 1)
        browse = QPushButton("Choose...")
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        root.addLayout(row)

        self.classes_label = QLabel("")
        self.classes_label.setWordWrap(True)
        root.addWidget(self.classes_label)

        self.device_label = QLabel(
            "Each clip is read once by the SigLIP2 action model, then a small "
            "model is trained on top: minutes, on a processor too. It is "
            "scored on source videos it never saw, so name clips "
            "<video>_temp_<n> to let it tell videos apart.")
        self.device_label.setWordWrap(True)
        self.device_label.setStyleSheet("color:#999;")
        root.addWidget(self.device_label)

        # Off by default: an hour on a graphics card and a 186 MB model, for a
        # few points more on videos it has not seen. Enabled once the probe
        # finds a card PyTorch can train on.
        self.finetune_box = QCheckBox(
            "Also train the image model (graphics card, about an hour; the model "
            "is 186 MB instead of 2 MB)")
        self.finetune_box.setEnabled(False)
        self.finetune_box.setToolTip("Choose a clips folder first.")
        root.addWidget(self.finetune_box)

        self.train_btn = QPushButton("Train an action model")
        self.train_btn.setStyleSheet(
            f"QPushButton{{background:{THEME.success};color:white;"
            f"font-weight:bold;padding:10px 18px;}}")
        self.train_btn.setEnabled(False)
        self.train_btn.clicked.connect(self._start)
        root.addWidget(self.train_btn)

        self.cancel_btn = QPushButton("Stop")
        self.cancel_btn.clicked.connect(self._cancel)
        self.cancel_btn.setVisible(False)
        root.addWidget(self.cancel_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        root.addStretch()
        self.setLayout(root)

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose the clips folder")
        if path:
            self._load_folder(path)

    def _probe_device(self) -> None:
        """Find out, off the GUI thread (it imports torch), whether this
        computer can fine-tune the image model."""
        if self._probing:
            return
        self._probing = True
        self.finetune_box.setToolTip("Looking for a graphics card...")
        import threading

        def probe():
            try:
                device, name = _probe_training_device()
            except Exception as exc:    # noqa: BLE001
                print(f"[training] device probe failed: {exc}")
                device, name = "cpu", ""
            self._device_found.emit(device, name)

        threading.Thread(target=probe, daemon=True).start()

    @Slot(str, str)
    def _on_device_found(self, device: str, name: str) -> None:
        if can_finetune(device):
            self.finetune_box.setEnabled(True)
            self.finetune_box.setToolTip(
                f"Trains the top {self.FINETUNE_BLOCKS} blocks of the image model with the "
                f"action model on {name or device}. Saved only if it does better on videos "
                f"it has not seen than the small model.")
        else:
            self.finetune_box.setChecked(False)
            self.finetune_box.setEnabled(False)
            self.finetune_box.setToolTip(
                "Needs an Intel Arc or NVIDIA graphics card that PyTorch can train on; "
                "none was found here. The small model trains on any computer.")

    def _load_folder(self, path: str) -> None:
        """Check the folder is laid out the way the trainer reads it.

        The layout is ``<folder>/train/<action>/*.mp4`` and ``<folder>/val/...``
        — *not* a folder per action at the top level, which is the arrangement
        that looks natural and silently yields "No training samples found".
        The minimums are the trainer's own: below them it skips a class, so
        they are worth stating here rather than after a wasted run.
        """
        self._probe_device()
        train_root = os.path.join(path, "train")
        val_root = os.path.join(path, "val")
        if not os.path.isdir(train_root):
            self._data_path = ""
            self.folder_label.setText(os.path.basename(path.rstrip(os.sep)) or path)
            self.folder_label.setStyleSheet("")
            self.classes_label.setText(
                "This folder needs a 'train' folder inside it, with one folder "
                "per action in there (and a 'val' folder the same way).")
            self.classes_label.setStyleSheet(f"color:{THEME.warning};")
            self.train_btn.setEnabled(False)
            return

        def count(root: str) -> dict:
            out = {}
            if not os.path.isdir(root):
                return out
            for name in sorted(os.listdir(root)):
                folder = os.path.join(root, name)
                if not os.path.isdir(folder):
                    continue
                clips = [f for f in os.listdir(folder)
                         if f.lower().endswith((".mp4", ".avi", ".mov"))]
                if clips:
                    out[name] = len(clips)
            return out

        train_counts, val_counts = count(train_root), count(val_root)
        self._data_path = path
        self.folder_label.setText(os.path.basename(path.rstrip(os.sep)) or path)
        self.folder_label.setStyleSheet("")

        if len(train_counts) < 2:
            # One class cannot be learned: a classifier needs something to tell
            # its class apart from, and a single folder trains a model that
            # answers "yes" to everything it is ever shown.
            self.classes_label.setText(
                "Needs at least two actions, one folder of clips each. A model "
                "with only one answer gives that answer to everything.")
            self.classes_label.setStyleSheet(f"color:{THEME.warning};")
            self.train_btn.setEnabled(False)
            return

        short = [name for name, n in train_counts.items()
                 if n < self.MIN_TRAIN_CLIPS or val_counts.get(name, 0) < self.MIN_VAL_CLIPS]
        described = ", ".join(
            f"{name} ({n} + {val_counts.get(name, 0)})" for name, n in train_counts.items())
        if short:
            self.classes_label.setText(
                f"{described}. These would be skipped for having too few clips: "
                f"{', '.join(short)} — each action needs at least "
                f"{self.MIN_TRAIN_CLIPS} to learn from and {self.MIN_VAL_CLIPS} "
                f"to check against.")
            self.classes_label.setStyleSheet(f"color:{THEME.warning};")
        else:
            self.classes_label.setText(f"Ready to learn: {described}.")
            self.classes_label.setStyleSheet("")
        self.train_btn.setEnabled(len(train_counts) - len(short) >= 2)

    def _start(self) -> None:
        if self._thread is not None:
            return
        from modules.vision import frame_encoder
        if not frame_encoder.is_installed():
            self._say("Training needs the SigLIP2 action model: download it from "
                      "Advanced > Action Recognition first.", THEME.warning)
            return
        name = os.path.basename(self._data_path.rstrip(os.sep)) or "my-actions"
        blocks = (self.FINETUNE_BLOCKS
                  if self.finetune_box.isEnabled() and self.finetune_box.isChecked() else 0)
        self._worker = ActionTrainingWorker(data_path=self._data_path, name=name,
                                            finetune_blocks=blocks)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._set_running(True)
        self._thread.start()

    def _cancel(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._say("Stopping...", THEME.warning)

    def _set_running(self, running: bool) -> None:
        self.train_btn.setVisible(not running)
        self.cancel_btn.setVisible(running)
        self.progress_bar.setVisible(running)
        self.finetune_box.setVisible(not running)
        if running:
            self.progress_bar.setValue(0)

    @Slot(int, str)
    def _on_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(max(0, min(100, percent)))
        self._say(message, "")

    @Slot(str)
    def _on_finished(self, note: str) -> None:
        folder = self._worker.out_dir if self._worker is not None else ""
        self._teardown()
        message = "Your action model is ready."
        if note:
            message += f" It {note}."
        if folder:
            message += (f" Saved in {folder}; the Actions pass uses it from the next run "
                        f"(Advanced > Action Recognition names it).")
        self._say(message, THEME.success)
        if folder:
            self.model_installed.emit(folder)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._teardown()
        self._say(message, THEME.danger)

    def _teardown(self) -> None:
        self._set_running(False)
        if self._thread is not None:
            self._thread.quit()
            # The trainer stops between steps (a held-out fold at most, seconds);
            # a QThread dropped while still running takes the app down with it.
            self._thread.wait(60000)
            self._thread = None
        self._worker = None

    def _say(self, text: str, colour: str) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"color:{colour};" if colour else "")

    def closeEvent(self, event):        # noqa: N802 (Qt override)
        """A training run must not outlive its window."""
        self._cancel()
        self._teardown()
        super().closeEvent(event)


class TrainingPanel(QWidget):
    """The two kinds of training, side by side.

    Separate tabs rather than one form, because the *example* differs: an
    object is a box in a frame, an action is a clip that runs over time. They
    take different data from disk, train different models with different
    scripts, and share nothing but the word "training" - so a combined form
    would only hide which inputs each one needs.
    """

    model_installed = Signal(object)
    action_model_installed = Signal(str)     # the trained action model's folder

    def __init__(self, parent=None, store_path: str = ""):
        super().__init__(parent)
        from PySide6.QtWidgets import QTabWidget

        self.objects = ObjectTrainingSection(store_path=store_path)
        self.objects.model_installed.connect(self.model_installed)
        self.actions = ActionTrainingSection()
        self.actions.model_installed.connect(self.action_model_installed)

        tabs = QTabWidget()
        # First: the automated loop (modules/teach), cutting, sorting and
        # labelling by itself from example clips and videos. The two tabs after
        # it train from data that is already labelled.
        try:
            from modules.teach.teach_panel import TeachPanel
            self.teach = TeachPanel()
            tabs.addTab(self.teach, "From videos")
        except Exception as exc:                # pragma: no cover - never cost the rest
            self.teach = None
            print(f"[training] teach panel unavailable: {exc}")
        tabs.addTab(self.objects, "Objects")
        tabs.addTab(self.actions, "Actions")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._hardware_label())
        layout.addWidget(tabs)
        self.setLayout(layout)

    @staticmethod
    def _hardware_label() -> QLabel:
        """Name the hardware, above both kinds of training.

        Shown so somebody can confirm a run is about to use the card they think
        it is, before committing hours to it. Above the tabs rather than inside
        them because it is the same machine either way, and a fact repeated in
        two places is a fact that can disagree with itself.
        """
        label = QLabel()
        label.setWordWrap(True)
        try:
            from modules.system.device_utils import describe_devices
            devices = describe_devices()
        except Exception as exc:                # pragma: no cover - defensive
            devices = []
            print(f"[training] could not list devices: {exc}")

        if devices:
            label.setText("Training hardware: " + "; ".join(devices))
            label.setStyleSheet("color:#999;")
        else:
            label.setText(
                "Training hardware: no GPU found - training will use the "
                "processor and be much slower.")
            label.setStyleSheet(f"color:{THEME.warning};")
        return label
