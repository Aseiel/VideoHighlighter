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
    QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QMessageBox,
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
    """Drive the R3D trainer in a child process, reporting what it prints.

    A subprocess rather than an import, for one reason: ``model_training.r3d``
    is an existing, working training script with its own configuration, cache
    handling and ONNX export. Refactoring it to take progress callbacks would
    be the larger and riskier change, and it would be a change to code that is
    not broken. Reading its output costs a parser and leaves it alone.

    It also means cancelling is a terminated process rather than a cooperative
    flag, which for a run holding a large clip cache is the more reliable stop.
    """

    progress = Signal(int, str)
    finished = Signal(str)
    error = Signal(str)

    # "Epoch 3/30" from the trainer's own progress bar, and the per-epoch
    # summary it prints afterwards. Everything else it says goes to the debug
    # log unchanged.
    _EPOCH = re.compile(r"Epoch\s+(\d+)\s*/\s*(\d+)")
    _VAL = re.compile(r"Val\s+Loss:\s*([\d.]+)\s*\|\s*Acc:\s*([\d.]+)")

    def __init__(self, data_path: str, epochs: int, batch_size: int,
                 pipeline: str, variant: str, device: str):
        super().__init__()
        self._data_path = data_path
        self._epochs = epochs
        self._batch_size = batch_size
        self._pipeline = pipeline
        self._variant = variant
        self._device = device
        self._process = None
        self._stop = False

    def cancel(self) -> None:
        self._stop = True
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:                       # pragma: no cover - defensive
                pass

    @Slot()
    def run(self) -> None:
        import subprocess
        import sys

        repo_root = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        command = self._command(sys.executable)
        try:
            self.progress.emit(
                0, "Preparing clips - the first pass is slow...")
            creation = 0
            if sys.platform.startswith("win"):
                creation = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            # The trainers print emoji, and a Windows console here is cp1250:
            # without this the child dies on UnicodeEncodeError at its first
            # line of output, long before it touches the user's data. Inside
            # the app the same prints survive because debug_console tees them
            # through a UTF-8 stream; a subprocess gets the raw console.
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
            self._process = subprocess.Popen(
                command, cwd=repo_root, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                errors="replace", bufsize=1, creationflags=creation, env=env,
            )
            note = ""
            for line in self._process.stdout:
                line = line.rstrip()
                if line:
                    print(f"[actions] {line}")   # the debug log keeps everything
                    note = self._read(line) or note
            code = self._process.wait()

            if self._stop:
                self.error.emit("Stopped.")
                return
            if code != 0:
                self.error.emit(
                    f"Training stopped with exit code {code}. The debug log "
                    f"has the trainer's own output.")
                return
            self.progress.emit(100, "Done.")
            self.finished.emit(note)
        except Exception as exc:                    # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.error.emit(f"Could not run the action trainer: {exc}")

    def _command(self, python: str) -> list:
        """The trainer to run, and its flags.

        Two pipelines, because the hardware genuinely differs:

        ``intel`` runs Intel's action-recognition encoder under OpenVINO and
        trains only a decoder on top. The encoder is frozen, so its output is
        cached once and every later epoch is fast. This is the path that
        produced the classifier the app already ships.

        ``r3d`` fine-tunes a 3D CNN end to end. More capable and far heavier,
        and it wants a CUDA card.

        Their flags are not the same: the Intel trainer picks its own device
        and takes a decoder type, while the R3D one takes a device and a model
        variant. So this builds each command rather than sharing one.
        """
        common = [
            python, "-u", "-m", f"model_training.{self._pipeline}.train",
            "--data-path", self._data_path,
            "--epochs", str(self._epochs),
            "--batch-size", str(self._batch_size),
            "--no-viz",
        ]
        if self._pipeline == "intel":
            # No --device: intel/config.py already resolves XPU itself, and
            # the encoder half runs under OpenVINO regardless.
            return common
        return common + ["--model", self._variant, "--device", self._device]

    def _read(self, line: str):
        """Turn one line of the trainer's output into a progress update."""
        epoch = self._EPOCH.search(line)
        if epoch:
            done, total = int(epoch.group(1)), max(1, int(epoch.group(2)))
            self.progress.emit(int(100 * done / total),
                               f"Learning... round {done} of {total}")
            return None
        val = self._VAL.search(line)
        if val:
            # Accuracy is the one number here worth showing: unlike a loss, a
            # person can read it without knowing the model.
            share = float(val.group(2)) * 100
            return f"recognised {share:.0f}% of the clips it had not seen"
        return None


class ActionTrainingSection(QWidget):
    """Train the app to recognise an action of the user's own.

    Actions are taught from **whole clips**, not boxes: one folder per action,
    videos inside it. That is why this is its own section rather than a mode of
    the object one - the example a user has to supply is a different kind of
    thing, and a shared form would ask for the wrong input.
    """

    DEFAULT_EPOCHS = 30
    DEFAULT_BATCH = 4          # 3D clips are far heavier than stills
    # The trainer's own minimums: below these it skips the class.
    MIN_TRAIN_CLIPS = 5
    MIN_VAL_CLIPS = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self._thread: Optional[QThread] = None
        self._worker: Optional[ActionTrainingWorker] = None
        self._data_path = ""
        self._device = "cpu"
        self._pipeline = "intel"
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

        advanced = CollapsibleSection(
            "Advanced", settings_key="training/actions_advanced")
        form = QFormLayout()
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(self.DEFAULT_EPOCHS)
        form.addRow("Rounds of learning:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(self.DEFAULT_BATCH)
        form.addRow("Clips at a time:", self.batch_spin)

        self.pipeline_combo = QComboBox()
        self.pipeline_combo.addItem("Automatic (match my hardware)", "auto")
        self.pipeline_combo.addItem("Intel - OpenVINO encoder", "intel")
        self.pipeline_combo.addItem("NVIDIA - 3D CNN", "r3d")
        self.pipeline_combo.currentIndexChanged.connect(self._choose_pipeline)
        form.addRow("Method:", self.pipeline_combo)

        self.variant_combo = QComboBox()
        for variant, hint in (("r3d_18", "recommended"),
                              ("mc3_18", "lighter"),
                              ("r2plus1d_18", "slower, often better")):
            self.variant_combo.addItem(f"{variant} - {hint}", variant)
        form.addRow("3D CNN model:", self.variant_combo)
        # Held so the row can be hidden: it belongs to the 3D CNN only, and a
        # visible-but-irrelevant control reads as a setting that was ignored.
        self._advanced_form = form
        advanced.setContentLayout(form)
        root.addWidget(advanced)

        self.device_label = QLabel("")
        self.device_label.setWordWrap(True)
        root.addWidget(self.device_label)
        self._choose_pipeline()

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

    def _choose_pipeline(self) -> None:
        """Decide which trainer to use, and say so before anything is started.

        **Hardware detection goes through `modules.system.device_utils`, not a torch
        probe.** That module is the app's single source of truth and it knows
        something a torch probe cannot: the released build ships a *CUDA* torch
        wheel, on which `torch.xpu` exists but reports `is_available()` False —
        so on a packaged app an Intel Arc looks like no GPU at all. Its own
        comment says so. `device_utils` falls through to asking OpenVINO, which
        still sees the card. Probing torch here reproduced exactly that bug:
        "No GPU found" on a machine with an A750 in it.

        **Intel is not a fallback.** ``model_training/intel`` is a purpose-built
        pipeline — Intel's action-recognition encoder run under OpenVINO with a
        decoder trained on top — and it produced the classifier this app ships.
        On an Intel machine it is the right answer, not a consolation for
        lacking CUDA. The end-to-end 3D CNN is the better tool only where there
        is an NVIDIA card to run it on.
        """
        backend, gpu_present = "CPU", False
        try:
            from modules.system.device_utils import detect_best_device
            info = detect_best_device(log_fn=lambda *a, **k: None)
            backend = str(getattr(info, "backend_name", "CPU"))
            gpu_present = bool(getattr(info, "gpu_available", False))
        except Exception as exc:                    # pragma: no cover - defensive
            print(f"[training] device detection failed: {exc}")

        has_cuda = backend.upper().startswith("CUDA")

        # What torch itself can train on. Separate from the question above,
        # because device_utils answers for the *inference* pipeline — where
        # Intel deliberately goes through OpenVINO and its `pytorch_device`
        # stays "cpu" — while training is the other case.
        self._device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                self._device = "cuda"
            elif getattr(torch, "xpu", None) and torch.xpu.is_available():
                self._device = "xpu"
        except Exception:
            pass

        chosen = (self.pipeline_combo.currentData()
                  if hasattr(self, "pipeline_combo") else "auto")
        self._pipeline = ("r3d" if has_cuda else "intel") if chosen == "auto" else chosen

        colour = "#999"
        if self._pipeline == "intel":
            where = {"cuda": "your NVIDIA GPU", "xpu": "your Intel GPU"}.get(
                self._device, "the processor")
            note = (f"Intel method, using {backend}. The encoder runs under "
                    f"OpenVINO and only the decoder is trained ({where}), so "
                    f"the first pass is slow and the rest are quick.")
            if not gpu_present:
                note += " No GPU found, so expect the first pass to be long."
                colour = THEME.warning
        else:
            where = {"cuda": "your NVIDIA GPU", "xpu": "your Intel GPU"}.get(
                self._device, "the processor")
            note = f"3D CNN, training every layer on {where}."
            if self._device == "cpu":
                note += (" That is hours rather than minutes - the Intel "
                         "method is usually the better choice here.")
                colour = THEME.warning
        self.device_label.setText(note)
        self.device_label.setStyleSheet(f"color:{colour};")

        # The 3D CNN variants mean nothing to the Intel method, which trains a
        # decoder on a fixed encoder. Leaving the row on screen invites someone
        # to pick r2plus1d_18 and then wonder why nothing about the run changed.
        form = getattr(self, "_advanced_form", None)
        if form is not None:
            form.setRowVisible(self.variant_combo, self._pipeline == "r3d")

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose the clips folder")
        if path:
            self._load_folder(path)

    def _load_folder(self, path: str) -> None:
        """Check the folder is laid out the way the trainer reads it.

        The layout is ``<folder>/train/<action>/*.mp4`` and ``<folder>/val/...``
        — *not* a folder per action at the top level, which is the arrangement
        that looks natural and silently yields "No training samples found".
        The minimums are the trainer's own: below them it skips a class, so
        they are worth stating here rather than after a wasted run.
        """
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
        self._worker = ActionTrainingWorker(
            data_path=self._data_path,
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            pipeline=self._pipeline,
            variant=self.variant_combo.currentData(),
            device=self._device,
        )
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
        if running:
            self.progress_bar.setValue(0)

    @Slot(int, str)
    def _on_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(max(0, min(100, percent)))
        self._say(message, "")

    @Slot(str)
    def _on_finished(self, note: str) -> None:
        self._teardown()
        message = "Your action model is ready."
        if note:
            message += f" It {note}."
        self._say(message, THEME.success)

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self._teardown()
        self._say(message, THEME.danger)

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


class TrainingPanel(QWidget):
    """The two kinds of training, side by side.

    Separate tabs rather than one form, because the *example* differs: an
    object is a box in a frame, an action is a clip that runs over time. They
    take different data from disk, train different models with different
    scripts, and share nothing but the word "training" - so a combined form
    would only hide which inputs each one needs.
    """

    model_installed = Signal(object)

    def __init__(self, parent=None, store_path: str = ""):
        super().__init__(parent)
        from PySide6.QtWidgets import QTabWidget

        self.objects = ObjectTrainingSection(store_path=store_path)
        self.objects.model_installed.connect(self.model_installed)
        self.actions = ActionTrainingSection()

        tabs = QTabWidget()
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
