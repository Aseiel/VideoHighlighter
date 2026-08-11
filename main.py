import os
import sys

# Capture every print/warning/traceback from the very first import: the
# packaged exe is --windowed (no stdout), so modules/debug_console tees all
# output into debug.log next to the exe and can mirror it to a live console
# window. Must run before the heavy imports below — some of them print
# warnings worth keeping.
from modules import debug_console
debug_console.install()

# Progress reporting for the launch itself. Imported here, before the heavy
# imports below, because in a frozen build *they* are the slow part — several
# seconds of decompressing and initialising cv2/OpenVINO/transformers before
# any window can exist. The bootloader's splash covers that stretch with the
# logo; these stage() calls are what make a slow launch readable afterwards in
# debug.log, and they would drive the splash text too if the build ever moves
# to a .spec (see modules/startup_splash.py on why the CLI flag cannot).
# This module deliberately pulls in no Qt, so it cannot disturb the import
# order below, which matters on Windows.
from modules import startup_splash
startup_splash.stage("Loading the video engine…")

import cv2
import json
import subprocess
import threading
import time
import yaml
import multiprocessing

from PySide6.QtWidgets import (
    QApplication, QCompleter, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QLineEdit, QSpinBox, QDoubleSpinBox,
    QGroupBox, QTextEdit, QFormLayout, QProgressBar, QCheckBox,
    QComboBox, QTabWidget, QListWidget, QSplitter,
    QDialog, QDialogButtonBox, QAbstractItemView,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
    QGridLayout, QSlider, QSizePolicy, QToolButton, QMenu,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QMetaObject, Q_ARG, Slot, QStringListModel
from downloader import download_videos_with_immediate_processing, extract_video_links, DownloadError, reset_duration_method_cache
startup_splash.stage("Loading the assistant…")
from llm.llm_chat_widget import LLMChatWidget
from modules.video_cache import VideoAnalysisCache, CachedAnalysisData, build_analysis_cache_params
from modules import analysis_stats
from modules.ui import icons as _ui_icons, theme as _ui_theme
# The five classes the expression scan can report. Imported for the Basic
# tab's picker; the module itself loads no model until something asks it to scan.
from modules.face_emotions import EMOTION_LABELS

startup_splash.stage("Loading the detection runtime…")
try:
    import openvino  # registers OpenVINO's DLL dir on Windows
except Exception:
    pass

from modules.app_paths import resource_path as _resource_path, data_file as _data_file, config_path
from version import __version__, __edition__

# --- Contact / support details shown in the About tab ---
SUPPORT_EMAIL = "przkreft@gmail.com"
WEBSITE_URL = "https://aseiel.github.io/VideoHighlighter-site/"
DISCORD_URL = "https://discord.gg/cUPJqPAMmm"
REPO_URL = "https://github.com/Aseiel/VideoHighlighter"

# User-editable config: lives next to the exe when frozen (so saves persist),
# seeded from the bundled default; just the project-root file when run from source.
CONFIG_FILE = config_path("config.yaml")

YOLO_OBJECTS_LABELS_FILE = _resource_path("yolo_objects_labels.json")
KINETICS_400_LABELS_FILE = _resource_path("kinetics_400_labels.json")
INTEL_CUSTOM_LABELS_FILE = _data_file("intel_finetuned_classifier_3d_mapping.json")
R3D_CUSTOM_LABELS_FILE = _data_file("r3d_finetuned_mapping.json")

class LabelSelectorDialog(QDialog):
    """Dialog with search/filter and multi-select for labels."""

    def __init__(self, title, labels, current_selection=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(480, 520)
        self.all_labels = sorted(labels)
        self.current_selection = set(current_selection or [])

        layout = QVBoxLayout()

        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Filter:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Type to filter labels...")
        self.search_input.textChanged.connect(self._filter_labels)
        search_layout.addWidget(self.search_input)
        layout.addLayout(search_layout)

        self.info_label = QLabel(f"{len(self.all_labels)} labels available")
        self.info_label.setStyleSheet("color: #666; font-size: 9pt;")
        layout.addWidget(self.info_label)

        self.label_list = QListWidget()
        self.label_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._populate_list(self.all_labels)
        layout.addWidget(self.label_list)

        quick_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All Visible")
        select_all_btn.clicked.connect(self._select_all_visible)
        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(self._deselect_all)
        quick_layout.addWidget(select_all_btn)
        quick_layout.addWidget(deselect_all_btn)
        quick_layout.addStretch()
        layout.addLayout(quick_layout)

        self.selection_label = QLabel("0 selected")
        self.selection_label.setStyleSheet("font-weight: bold; color: #2f81f7;")
        layout.addWidget(self.selection_label)
        self.label_list.itemSelectionChanged.connect(self._update_selection_count)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self._preselect_current()

    def _populate_list(self, labels):
        self.label_list.clear()
        for label in labels:
            self.label_list.addItem(label)

    def _preselect_current(self):
        for i in range(self.label_list.count()):
            item = self.label_list.item(i)
            if item.text() in self.current_selection:
                item.setSelected(True)
        self._update_selection_count()

    def _filter_labels(self, text):
        text = text.strip().lower()
        filtered = [l for l in self.all_labels if text in l.lower()] if text else self.all_labels
        self._populate_list(filtered)
        self.info_label.setText(f"{len(filtered)} of {len(self.all_labels)} labels shown")
        self._preselect_current()

    def _select_all_visible(self):
        for i in range(self.label_list.count()):
            self.label_list.item(i).setSelected(True)
        self._update_selection_count()

    def _deselect_all(self):
        self.label_list.clearSelection()
        self._update_selection_count()

    def _update_selection_count(self):
        self.selection_label.setText(f"{len(self.label_list.selectedItems())} selected")

    def get_selected_labels(self):
        return [item.text() for item in self.label_list.selectedItems()]

class NoAnalysisWarningDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("No Analysis Data")
        self.setFixedWidth(420)
        
        layout = QVBoxLayout()
        
        icon_label = QLabel("⚠️")
        icon_label.setStyleSheet("font-size: 32px;")
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)
        
        msg = QLabel(
            "No analysis cache found for this video.\n\n"
            "You can still use the timeline viewer to seek through\n"
            "the video and chat with the LLM — but motion, audio,\n"
            "object and action signals won't be available.\n\n"
            "Run the pipeline first to get full signal data."
        )
        msg.setWordWrap(True)
        msg.setAlignment(Qt.AlignCenter)
        layout.addWidget(msg)
        
        self.dont_show_chk = QCheckBox("Don't show this warning again")
        self.dont_show_chk.setStyleSheet("color: #666;")
        layout.addWidget(self.dont_show_chk)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Open Anyway")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        self.setLayout(layout)

class MultiCompleter(QCompleter):
    """QCompleter that works on comma-separated fields, completing only the current token.
    Matches labels where any word starts with the typed text."""

    def __init__(self, labels=None, parent=None):
        super().__init__(parent)
        self._all_labels = labels or []
        self._source_model = QStringListModel(self._all_labels)
        self.setModel(self._source_model)
        self.setCaseSensitivity(Qt.CaseInsensitive)
        self.setFilterMode(Qt.MatchContains)

    def setLabels(self, labels):
        """Update the full label list."""
        self._all_labels = labels
        self._source_model.setStringList(labels)

    def pathFromIndex(self, index):
        completion = super().pathFromIndex(index)
        widget = self.widget()
        if not widget:
            return completion
        text = widget.text()
        cursor = widget.cursorPosition()
        before = text[:cursor]
        last_comma = before.rfind(",")
        prefix = text[:last_comma + 1] + " " if last_comma >= 0 else ""
        after_cursor = text[cursor:]
        next_comma = after_cursor.find(",")
        suffix = after_cursor[next_comma:] if next_comma >= 0 else ""
        return prefix + completion + suffix

    def splitPath(self, path):
        widget = self.widget()
        if not widget:
            return [path.strip()]
        cursor = widget.cursorPosition()
        before = path[:cursor]
        last_comma = before.rfind(",")
        current_token = before[last_comma + 1:].strip().lower()

        # Filter: any word in label starts with typed text
        if current_token:
            filtered = [l for l in self._all_labels
                        if any(w.startswith(current_token) for w in l.lower().split())]
        else:
            filtered = self._all_labels
        self._source_model.setStringList(filtered)
        return [current_token]

class DownloadWorker(QThread):
    """
    Worker thread for downloading videos (with optional immediate processing after each file).
    
    Emits signals for:
    - progress updates
    - logging
    - finished list of downloaded paths
    - cancellation
    - individual video processed (when immediate processing is active)
    """
    finished = Signal(list)              # List of downloaded file paths
    progress = Signal(int, int, str, str)  # current, total, status, message
    log = Signal(str)                    # log messages
    cancelled = Signal()                 # emitted when cancelled
    video_processed = Signal(str, dict)  # filepath, processing result dict
    add_to_file_list = Signal(str)       # emits filepath to be added

    def __init__(self, url, save_dir, pattern, time_range=None, download_full=True,
                 use_percentages=False, immediate_processing=False, max_concurrent=1,
                 process_callback=None, video_urls=None):
        super().__init__()
        self.url = url
        self.save_dir = save_dir
        self.pattern = pattern
        self.time_range = time_range                  # (start, end) seconds or percentages
        self.download_full = download_full
        self.use_percentages = use_percentages
        self.immediate_processing = immediate_processing
        self.max_concurrent = max_concurrent
        self.process_callback = process_callback      # called after each download if immediate_processing
        self.video_urls = video_urls                  # explicit selection from the picker (skips scrape)
        self._cancelled = False
        self._is_running = False
        self._download_results = []                   # store all download metadata

    def run(self):
        try:
            self._is_running = True
            self.log.emit(f"🚀 Starting download from: {self.url}")

            def log_fn(message):
                self.log.emit(message)

            def progress_fn(current, total, status, message):
                self.progress.emit(current, total, status, message)

            # Wraps the GUI-supplied callback. Emits video_processed so the GUI
            # can react per file. Only used when immediate_processing is on
            # AND a real callback was provided; otherwise the downloader runs
            # without per-video processing.
            def wrapped_process_callback(filepath, metadata):
                if self._cancelled:
                    return {'cancelled': True}
                self.log.emit(f"🔧 Processing: {os.path.basename(filepath)}")
                try:
                    result = self.process_callback(filepath, metadata)
                    self.log.emit(f"✅ Processed: {os.path.basename(filepath)}")
                    self.video_processed.emit(filepath, result)
                    return result
                except Exception as e:
                    self.log.emit(f"❌ Processing failed: {e}")
                    return {'error': str(e)}

            callback = (wrapped_process_callback
                        if (self.immediate_processing and self.process_callback)
                        else None)

            results = download_videos_with_immediate_processing(
                search_url=self.url,
                save_dir=self.save_dir,
                pattern=self.pattern,
                log_fn=log_fn,
                progress_fn=progress_fn,
                process_callback=callback,
                cancel_flag=self,
                time_range=self.time_range,
                download_full=self.download_full,
                use_percentages=self.use_percentages,
                max_workers=self.max_concurrent,
                video_urls=self.video_urls,
            )

            # Collect downloaded files
            downloaded_files = []
            for result in results:
                if result.get('success') and result.get('filepath'):
                    downloaded_files.append(result['filepath'])
                    self._download_results.append(result)

            if self._cancelled:
                self.log.emit("⏹️ Download was cancelled")
                self.cancelled.emit()
                self.finished.emit([])
            else:
                self.finished.emit(downloaded_files)

        except Exception as e:
            self.log.emit(f"❌ Download thread error: {e}")
            import traceback
            self.log.emit(traceback.format_exc())
            self.finished.emit([])
        finally:
            self._is_running = False

    def cancel(self):
        """Request cancellation – called from GUI.

        Non-blocking: just trip the flag and return. run() unwinds and emits
        cancelled/finished, which drive the UI cleanup. (Previously this called
        self.wait()/terminate() on the GUI thread, which froze the UI and — on
        timeout — killed the thread before it could emit its signals, leaving the
        Download button stuck disabled. force_download_cleanup is the safety net
        for a worker genuinely stuck in a non-cancellable subprocess.)"""
        if self._is_running:
            self.log.emit("⏹️ Cancellation requested - stopping download...")
            self._cancelled = True

    def is_cancelled(self):
        """Public method used by downloader module to check cancellation"""
        return self._cancelled

    def is_set(self):
        """Compatibility alias – matches threading.Event.is_set()"""
        return self._cancelled
    
class DetectionPreviewWindow(QWidget):
    """Standalone window showing live detection frames during processing.

    Supports pause (freezes the pipeline) and rewind (scrub back through a ring
    buffer of recently shown frames).
    """

    closed = Signal()

    BUFFER_SIZE = 250  # rewind history (~30s at 8 fps); ~125MB of pixmaps

    def __init__(self, parent=None):
        super().__init__(parent)
        from collections import deque
        self.setWindowTitle("🔍 Live Detection Preview")
        self.setMinimumSize(560, 400)
        self.resize(720, 540)

        self._frames = deque(maxlen=self.BUFFER_SIZE)  # (pixmap, caption)
        self._paused = False
        self._view_index = -1  # -1 = follow live (latest)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self.image_label = QLabel("Waiting for the detection stage (objects / actions)…")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "QLabel { background:#101010; color:#8c8c8c; border:1px solid #333; }"
        )
        layout.addWidget(self.image_label, 1)

        self.caption = QLabel("")
        self.caption.setStyleSheet("color:#9aa; font-size:10pt;")
        self.caption.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.caption)

        # ── Controls: pause/resume, step, scrub slider ──
        controls = QHBoxLayout()
        self.pause_btn = QPushButton("⏸ Freeze")
        self.pause_btn.setFixedWidth(90)
        self.pause_btn.setToolTip("Freeze the preview to inspect a frame.\n"
                                  "Processing keeps running in the background.")
        self.pause_btn.clicked.connect(self._toggle_pause)
        controls.addWidget(self.pause_btn)

        self.prev_btn = QPushButton("◀")
        self.prev_btn.setFixedWidth(36)
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        controls.addWidget(self.prev_btn)

        self.scrub = QSlider(Qt.Horizontal)
        self.scrub.setMinimum(0)
        self.scrub.setMaximum(0)
        self.scrub.sliderPressed.connect(self._on_scrub_pressed)
        self.scrub.valueChanged.connect(self._on_scrub_moved)
        controls.addWidget(self.scrub, 1)

        self.next_btn = QPushButton("▶")
        self.next_btn.setFixedWidth(36)
        self.next_btn.clicked.connect(lambda: self._step(1))
        controls.addWidget(self.next_btn)

        self.live_btn = QPushButton("⏭ Live")
        self.live_btn.setFixedWidth(70)
        self.live_btn.setToolTip("Jump back to the live frame and resume following")
        self.live_btn.clicked.connect(self._go_live)
        controls.addWidget(self.live_btn)

        layout.addLayout(controls)
        self._update_controls_enabled()

    # ── public: called from the GUI when a new frame arrives ──
    def set_frame(self, pixmap, caption=""):
        # Background processing keeps feeding frames even while frozen.
        was_full = len(self._frames) == self._frames.maxlen
        self._frames.append((pixmap, caption))
        # If frozen and the buffer just dropped its oldest frame, shift the view
        # index down by one so we keep looking at the SAME content.
        if self._paused and was_full and self._view_index > 0:
            self._view_index -= 1

        self.scrub.blockSignals(True)
        self.scrub.setMaximum(len(self._frames) - 1)
        self.scrub.setValue(self._view_index if self._paused else len(self._frames) - 1)
        self.scrub.blockSignals(False)

        if not self._paused:
            self._view_index = len(self._frames) - 1
            self._render_current()
        self._update_controls_enabled()

    # ── internals ──
    def _render_current(self):
        if not self._frames:
            return
        idx = self._view_index if self._view_index >= 0 else len(self._frames) - 1
        idx = max(0, min(idx, len(self._frames) - 1))
        pix, cap = self._frames[idx]
        self.image_label.setPixmap(
            pix.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        live_tag = "  • LIVE" if (not self._paused and idx == len(self._frames) - 1) else \
                   f"  • frozen {idx + 1}/{len(self._frames)}"
        self.caption.setText((cap or "") + live_tag)

    def _toggle_pause(self):
        self._paused = not self._paused
        self.pause_btn.setText("▶ Resume" if self._paused else "⏸ Freeze")
        if not self._paused:
            # Un-freeze → follow live again
            self._view_index = len(self._frames) - 1
            self._render_current()
        self._update_controls_enabled()

    def _on_scrub_pressed(self):
        # Touching the slider implies you want to review → auto-pause
        if not self._paused:
            self._toggle_pause()

    def _on_scrub_moved(self, value):
        if self._paused:
            self._view_index = value
            self._render_current()

    def _step(self, delta):
        if not self._paused:
            self._toggle_pause()
        self._view_index = max(0, min(self._view_index + delta, len(self._frames) - 1))
        self.scrub.blockSignals(True)
        self.scrub.setValue(self._view_index)
        self.scrub.blockSignals(False)
        self._render_current()

    def _go_live(self):
        if self._paused:
            self._toggle_pause()  # resume → follows live
        else:
            self._view_index = len(self._frames) - 1
            self._render_current()

    def _update_controls_enabled(self):
        has = len(self._frames) > 0
        self.prev_btn.setEnabled(has)
        self.next_btn.setEnabled(has)
        self.scrub.setEnabled(has)

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


class Worker(QThread):
    finished = Signal(object)
    progress = Signal(int, int, str, str)
    log = Signal(str)
    cancelled = Signal()
    preview = Signal(object, object, int)   # frame_bgr (ndarray), boxes (list), sec
    timeline_requested = Signal(str, object)  # video_path, analysis_data

    def __init__(self, video_path, gui_config=None):
        super().__init__()
        self.video_path = video_path
        self.gui_config = gui_config
        self._cancel_flag = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # starts unpaused
        self._is_running = False
        self.preview_enabled = False

    def pause(self):
        self._pause_event.clear()

    def resume(self):
        self._pause_event.set()

    def is_paused(self):
        return not self._pause_event.is_set()

    def run(self):
        from pipeline import run_highlighter
        try:
            self._is_running = True

            def pausing_progress(cur, tot, task, det):
                self._pause_event.wait()  # blocks while paused
                if not self._cancel_flag.is_set():
                    self.progress.emit(cur, tot, task, det)

            # Check if single or multiple files
            if isinstance(self.video_path, list):
                self.log.emit(f"🚀 Starting batch processing of {len(self.video_path)} videos...")
            else:
                self.log.emit("🚀 Starting video highlighter pipeline...")

            # Gate the preview emit on the live flag, checked per call so the
            # checkbox works mid-run. (The detector only builds/resizes a frame
            # ~8x/sec, negligible next to inference.)
            def preview_gate(frame, boxes, sec):
                if self.preview_enabled and not self._cancel_flag.is_set():
                    self.preview.emit(frame, boxes, sec)

            output = run_highlighter(
                self.video_path,
                gui_config=self.gui_config,
                log_fn=self.log.emit,
                progress_fn=pausing_progress,
                cancel_flag=self._cancel_flag,
                preview_fn=preview_gate,
                # Qt widgets may only be built on the main thread; emitting
                # hands the request off to the GUI's reuse-aware handler.
                timeline_fn=lambda path, data: self.timeline_requested.emit(str(path), data),
            )

            if self._cancel_flag.is_set():
                self.log.emit("⏹️ Pipeline was cancelled")
                self.cancelled.emit()
                self.finished.emit("")
            else:
                self.finished.emit(output or "")

        except Exception as e:
            self.log.emit(f"❌ Worker error: {e}")
            import traceback
            self.log.emit(f"Full traceback: {traceback.format_exc()}")
            self.finished.emit("")
        finally:
            self._is_running = False

    def cancel(self):
        if self._is_running:
            self.log.emit("⏹️ Cancellation requested - stopping pipeline...")
            self._cancel_flag.set()
            if not self.wait(5000):
                self.log.emit("⚠️ Force terminating thread...")
                self.terminate()
                self.wait()

    def is_cancelled(self):
        return self._cancel_flag.is_set()


class SignalRunWorker(QThread):
    """Run ONE analysis signal on demand over a list of videos, folding each
    result into that video's cache (leaving the other signals intact).

    This is the main-window twin of the timeline viewer's "Analyze" panel: same
    engine (`modules.analysis_ondemand`), same fold-into-cache behaviour, just
    looped over the whole file list instead of one loaded video. It never cuts
    highlights — it only produces the standalone `.srt`/`.txt` (subtitles /
    transcript) and/or warms the cache for a later highlight run or the viewer.
    """
    finished = Signal(str)                  # short summary, or "" on hard error
    progress = Signal(int, int, str, str)   # current, total, task, message
    log = Signal(str)
    cancelled = Signal()
    preview = Signal(object, object, int)   # frame_bgr (ndarray), boxes, sec

    def __init__(self, kind, video_paths, params=None):
        super().__init__()
        self.kind = kind
        self.video_paths = list(video_paths)
        self.params = params or {}
        self._cancel_flag = threading.Event()
        self._is_running = False
        self.preview_enabled = False

    def run(self):
        from modules import analysis_ondemand as aod
        self._is_running = True
        n = len(self.video_paths)
        done = 0
        try:
            self.log.emit(f"🚀 {self.kind.title()} on demand over {n} video(s)...")
            for i, vp in enumerate(self.video_paths):
                if self._cancel_flag.is_set():
                    break
                name = os.path.basename(vp)
                self.log.emit(f"▶️ {self.kind} [{i+1}/{n}]: {name}")

                def progress(cur, tot, task, det, _i=i, _name=name):
                    frac = (cur / tot) if tot else 0.0
                    overall = int(((_i + frac) / n) * 100)
                    self.progress.emit(overall, 100, self.kind.title(), f"{_name}: {det}")

                try:
                    patch = self._run_one(aod, vp, progress)
                except aod._Cancelled:
                    break
                except Exception as e:
                    self.log.emit(f"❌ {name}: {e}")
                    continue

                if patch:
                    aod.merge_into_cache(vp, patch, log=self.log.emit)
                    done += 1

            if self._cancel_flag.is_set():
                self.log.emit("⏹️ On-demand run cancelled")
                self.cancelled.emit()
                self.finished.emit("")
            else:
                self.finished.emit(f"{self.kind.title()}: {done}/{n} done")
        except Exception as e:
            import traceback
            self.log.emit(f"❌ {self.kind} run error: {e}")
            self.log.emit(traceback.format_exc())
            self.finished.emit("")
        finally:
            self._is_running = False

    def _run_one(self, aod, video_path, progress):
        """Dispatch to the matching on-demand runner and shape the cache patch.
        Mirrors the timeline viewer's per-kind cache keys."""
        c = self._cancel_flag
        p = self.params

        # An on-demand object/action run detects over the whole video exactly as
        # the pipeline's stage does — same detector, same length — so it feeds
        # the preview window from the same checkbox. Without this the window
        # opened and stayed on its placeholder for the entire run, which is the
        # one place the wait is longest and the reassurance worth most.
        # Checked per call so the checkbox still works mid-run.
        def preview_fn(frame, boxes, sec):
            if self.preview_enabled and not self._cancel_flag.is_set():
                self.preview.emit(frame, boxes, sec)

        if self.kind == "motion":
            return aod.run_motion(video_path, progress=progress, cancel=c, log=self.log.emit)
        if self.kind == "audio":
            return aod.run_audio(video_path, progress=progress, cancel=c, log=self.log.emit)
        if self.kind == "objects":
            result = aod.run_objects(video_path, p.get("objects") or [],
                                     progress=progress, cancel=c, log=self.log.emit,
                                     preview_fn=preview_fn)
            return {"objects": result}
        if self.kind == "actions":
            result = aod.run_actions(video_path, interesting_actions=p.get("actions") or [],
                                     progress=progress, cancel=c, log=self.log.emit,
                                     preview_fn=preview_fn)
            return {"actions": result, "actions_all": result}
        if self.kind == "transcript":
            result = aod.run_transcript(video_path, language=p.get("language"),
                                        progress=progress, cancel=c, log=self.log.emit)
            return {"transcript": result}
        if self.kind == "subtitles":
            result = aod.run_subtitles(video_path, language=p.get("language"),
                                       source_lang=p.get("source_lang"),
                                       target_lang=p.get("target_lang"),
                                       progress=progress, cancel=c, log=self.log.emit)
            return {"transcript": result}
        raise ValueError(f"unknown signal kind: {self.kind}")

    def cancel(self):
        if self._is_running:
            self.log.emit("⏹️ Cancellation requested — finishing current video...")
            self._cancel_flag.set()

    def is_cancelled(self):
        return self._cancel_flag.is_set()


class FaceScanWorker(QThread):
    """Offline identity pass over a video to populate the face bank with everyone
    who appears, so they show up in the Avoid list (the 'dry run')."""
    log = Signal(str)
    done = Signal(int)   # identity count after scan, or -1 on error

    def __init__(self, video_path, db_path):
        super().__init__()
        self.video_path = video_path
        self.db_path = db_path

    def run(self):
            try:
                from video_ai_editor.face_identity import FaceIdentityBank
                from modules.compute_forbidden import build_tracking_model, tag_entries

                bank = FaceIdentityBank(db_path=self.db_path)
                model = build_tracking_model("n", log_fn=self.log.emit)
                self.log.emit(f"🔍 Scanning {os.path.basename(self.video_path)} for faces…")
                # tag_entries caches the per-frame tagging so the pipeline's avoid step
                # reuses this same pass instead of re-running face recognition.
                tag_entries(
                    self.video_path, bank,
                    yolo_model=model,
                    model_size="n",
                    face_every=15,
                    vid_stride=3,
                    save_bank=True,
                    log_fn=self.log.emit,
                )
                self.done.emit(len(bank))
            except Exception as e:
                self.log.emit(f"❌ Face scan failed: {e}")
                self.done.emit(-1)

class UpdateCheckWorker(QThread):
    """Ask the update manifest whether a newer build exists.

    Off the GUI thread because a network round trip on it would freeze the
    window for the whole timeout on a bad connection — at the exact moment the
    user is trying to start work. Nothing here touches a widget; the answer
    travels back as a signal, and silence means "nothing to say".
    """

    found = Signal(object)   # modules.update_check.UpdateInfo
    nothing = Signal(str)    # only for an explicit "check now": why it found nothing

    def __init__(self, force=False, parent=None):
        super().__init__(parent)
        self.force = force

    def run(self):
        try:
            from modules import update_check
            info = update_check.check_for_update(force=self.force)
        except Exception as e:
            # An update check must never be the reason anything goes wrong.
            print(f"update_check: check failed ({type(e).__name__}: {e})")
            if self.force:
                self.nothing.emit("Could not check right now.")
            return
        if info:
            self.found.emit(info)
        elif self.force:
            # The automatic check says nothing when there is nothing; a user who
            # pressed a button is owed an answer either way.
            self.nothing.emit(f"You're on the latest version ({__version__}).")


class UpdateInstallWorker(QThread):
    """Download and install a release, off the GUI thread.

    All the logic lives in modules/update_install; this only marshals progress
    and the result back to the window.
    """

    progress = Signal(str, int, int, str)   # phase, done, total, detail
    finished_with = Signal(object)          # update_install.InstallResult

    def __init__(self, manifest_url, root, parent=None):
        super().__init__(parent)
        self.manifest_url = manifest_url
        self.root = root
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        from modules import update_install
        try:
            result = update_install.install_update(
                self.manifest_url, self.root,
                progress=lambda *a: self.progress.emit(*a),
                should_cancel=lambda: self._cancel,
            )
        except Exception as e:
            print(f"update_install: unexpected failure ({type(e).__name__}: {e})")
            result = update_install.InstallResult(
                ok=False, message=f"The update failed: {e}")
        self.finished_with.emit(result)


class RangeSlider(QWidget):
    """Single slider with two handles for selecting a range"""
    startChanged = Signal(int)
    endChanged = Signal(int)

    def __init__(self, minimum=0, maximum=100, parent=None):
        super().__init__(parent)
        self._min = minimum
        self._max = maximum
        self._start = minimum
        self._end = maximum
        self._dragging = None  # 'start', 'end', or None
        self.setFixedHeight(32)
        self.setMinimumWidth(200)
        self.setCursor(Qt.PointingHandCursor)

    def start(self):
        return self._start

    def end(self):
        return self._end

    def setStart(self, val):
        val = max(self._min, min(val, self._end - 1))
        if val != self._start:
            self._start = val
            self.startChanged.emit(val)
            self.update()

    def setEnd(self, val):
        val = min(self._max, max(val, self._start + 1))
        if val != self._end:
            self._end = val
            self.endChanged.emit(val)
            self.update()

    def setRangeValues(self, start, end):
        """Set both handles at once.

        setStart()/setEnd() clamp against the *current* opposite handle, so
        calling them in sequence fails when the whole window moves past the old
        range (e.g. switching from 'first 5min' to 'last 5min' clamps the new
        start to the old end). Setting both together avoids that cross-clamp.
        """
        start = max(self._min, min(int(start), self._max))
        end = max(self._min, min(int(end), self._max))
        if start > end:
            start, end = end, start
        if end <= start:
            end = min(self._max, start + 1)
        changed_start = (start != self._start)
        changed_end = (end != self._end)
        self._start = start
        self._end = end
        if changed_start:
            self.startChanged.emit(start)
        if changed_end:
            self.endChanged.emit(end)
        if changed_start or changed_end:
            self.update()

    def setRange(self, minimum, maximum):
        self._min = minimum
        self._max = maximum
        self._start = max(self._start, minimum)
        self._end = min(self._end, maximum)
        self.update()

    def _val_to_x(self, val):
        inset = 8
        w = self.width() - 2 * inset
        if self._max == self._min:
            return inset
        return inset + int((val - self._min) / (self._max - self._min) * w)

    def _x_to_val(self, x):
        inset = 8
        w = self.width() - 2 * inset
        if w <= 0:
            return self._min
        ratio = max(0.0, min(1.0, (x - inset) / w))
        return int(self._min + ratio * (self._max - self._min))

    def paintEvent(self, event):
        from PySide6.QtGui import QPainter, QColor
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        x0 = self._val_to_x(self._start)
        x1 = self._val_to_x(self._end)
        track_y = self.height() // 2 - 3
        track_h = 6

        # Full track background
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(60, 60, 60))
        inset = 8
        p.drawRoundedRect(inset, track_y, self.width() - 2 * inset, track_h, 3, 3)

        # Selected range
        p.setBrush(QColor(47, 129, 247))
        p.drawRoundedRect(x0, track_y, max(2, x1 - x0), track_h, 3, 3)

        # Start handle
        p.setBrush(QColor(222, 222, 222))
        p.setPen(QColor(47, 129, 247))
        p.drawEllipse(x0 - 7, self.height() // 2 - 7, 14, 14)

        # End handle
        p.drawEllipse(x1 - 7, self.height() // 2 - 7, 14, 14)

        p.end()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        x = event.position().toPoint().x()
        x0 = self._val_to_x(self._start)
        x1 = self._val_to_x(self._end)

        dist_start = abs(x - x0)
        dist_end = abs(x - x1)

        if dist_start <= dist_end and dist_start < 20:
            self._dragging = 'start'
        elif dist_end < 20:
            self._dragging = 'end'
        elif x0 < x < x1:
            # Click between handles — move nearest
            self._dragging = 'start' if dist_start < dist_end else 'end'

    def mouseMoveEvent(self, event):
        if self._dragging is None:
            return
        val = self._x_to_val(event.position().toPoint().x())
        if self._dragging == 'start':
            self.setStart(val)
        else:
            self.setEnd(val)

    def mouseReleaseEvent(self, event):
        self._dragging = None

class VideoHighlighterGUI(QWidget):
    #: A detection frame on its way to the preview window. The Run button has
    #: `Worker.preview` for this, but the download-and-process path calls
    #: `run_highlighter` straight from the download worker's thread and has no
    #: Worker to borrow a signal from — so the window owns one, and the hop to
    #: the GUI thread happens here rather than in each caller.
    preview_frame = Signal(object, object, int)   # frame_bgr, boxes, sec

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Video Highlighter v{__version__} {__edition__}")
        screen = QApplication.primaryScreen().availableGeometry()
        w = min(1000, screen.width() - 20)
        # Open as tall as the screen comfortably allows. The fixed sections
        # above the tabs (input list, output name, time range) cost ~410px
        # before a single tab row is drawn, so an 800px window spent half its
        # height before the settings even started. The cap only binds on very
        # tall screens; everywhere else the available height decides, and the
        # window can still shrink to its ~794px minimum.
        h = min(1200, screen.height() - 20)
        self.resize(w, h)
        self.move(screen.x() + (screen.width() - w) // 2, screen.y())

        
        self.worker = None

        self.config_data = self.load_config()

        layout = QVBoxLayout()
        # A little breathing room, but tight enough that the ~8 stacked sections
        # don't add up to a screenful of gaps (that empty space pushed the tabs
        # and Run row down). Trimmed from the original 20/16/14.
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(6)

        # --- Update notice (hidden unless there is actually a newer build) ---
        # Costs no vertical space while hidden, which is the whole reason it is
        # a banner and not a startup dialog: nothing interrupts a launch, and
        # nothing is permanently occupying a row on a small screen.
        self.update_banner = self._build_update_banner()
        layout.addWidget(self.update_banner)

        # Store video duration
        self.current_video_duration = 0

        # --- File picker ---
        file_group = QGroupBox("Input Videos")
        file_layout = QVBoxLayout()
        file_layout.setContentsMargins(12, 6, 12, 6)
        file_layout.setSpacing(6)

        # Buttons row
        btn_layout = QHBoxLayout()
        self.browse_btn = QPushButton("Add Videos")
        self.browse_btn.clicked.connect(self.browse_files)
        self.remove_btn = QPushButton("Remove Selected")
        self.remove_btn.clicked.connect(self.remove_selected_file)
        self.clear_btn = QPushButton("Clear All")
        self.clear_btn.clicked.connect(self.clear_files)
        
        btn_layout.addWidget(self.browse_btn)
        btn_layout.addWidget(self.remove_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addStretch()  # Push buttons to the left

        file_layout.addLayout(btn_layout)

        # File list — compact; it scrolls when there are more files. The tall
        # white box was the biggest chunk of empty space up top.
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(46)
        file_layout.addWidget(self.file_list)

        saved_paths = self.config_data.get("video", {}).get("paths", [])
        if saved_paths:
            for path in saved_paths:
                if os.path.exists(path):
                    self.file_list.addItem(path)
        
        file_group.setLayout(file_layout)
        # Vertical Maximum: the group takes only its natural height and never gets
        # stretched by slack. Without this, extra window height inflated this box
        # into a tall mostly-empty white panel up top.
        file_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout.addWidget(file_group)

        # --- Output filename ---
        out_layout = QHBoxLayout()
        self.output_input = QLineEdit(self.config_data.get("highlights", {}).get("output", "highlight.mp4"))
        out_layout.addWidget(QLabel("Output base name:"))
        out_layout.addWidget(self.output_input)
        layout.addLayout(out_layout)

        highlights_cfg = self.config_data.get("highlights", {})
        scoring_cfg = self.config_data.get("scoring", {})

        # --- Time Range Selection with Slider ---
        time_range_group = QGroupBox("Processing Time Range")
        time_range_layout = QVBoxLayout()

        # Enable/disable checkbox
        self.use_time_range_chk = QCheckBox("Process only specific time range")
        self.use_time_range_chk.setChecked(highlights_cfg.get("use_time_range", False))
        self.use_time_range_chk.toggled.connect(self.on_time_range_toggle)
        time_range_layout.addWidget(self.use_time_range_chk)

        # Everything below the checkbox lives in a body that is only shown while
        # "Process only specific time range" is ticked. Off (the default), this
        # whole group collapses to one line — the slider, %-labels, selection
        # text and preset buttons no longer take a chunk of the window.
        self.time_range_body = QWidget()
        time_range_body_layout = QVBoxLayout(self.time_range_body)
        time_range_body_layout.setContentsMargins(0, 0, 0, 0)
        time_range_body_layout.setSpacing(4)

        # Video duration label
        self.video_duration_label = QLabel("Set time range in percentages (0-100%) - loads actual times when video is selected")
        self.video_duration_label.setStyleSheet("color: #666; font-style: italic;")
        time_range_body_layout.addWidget(self.video_duration_label)

        # Range slider container
        slider_container = QWidget()
        slider_layout = QVBoxLayout()
        slider_layout.setContentsMargins(0, 0, 0, 0)

        # Range slider (single bar with two handles)
        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Start:"))
        self.range_slider = RangeSlider(0, 100)
        self.range_slider.setStart(highlights_cfg.get("range_start_pct", 0))
        self.range_slider.setEnd(highlights_cfg.get("range_end_pct", 100))
        self.range_slider.setEnabled(False)
        self.range_slider.startChanged.connect(self.on_slider_changed)
        self.range_slider.endChanged.connect(self.on_slider_changed)
        range_row.addWidget(self.range_slider, stretch=1)
        range_row.addWidget(QLabel("End"))

        self.start_time_label = QLabel("0%")
        self.start_time_label.setMinimumWidth(80)
        self.start_time_label.setStyleSheet("font-weight: bold;")

        self.end_time_label = QLabel("100%")
        self.end_time_label.setMinimumWidth(80)
        self.end_time_label.setStyleSheet("font-weight: bold;")

        labels_row = QHBoxLayout()
        labels_row.addWidget(self.start_time_label)
        labels_row.addStretch()
        labels_row.addWidget(self.end_time_label)

        slider_layout.addLayout(range_row)
        slider_layout.addLayout(labels_row)

        slider_container.setLayout(slider_layout)
        time_range_body_layout.addWidget(slider_container)

        # Selection info
        self.selection_info_label = QLabel("Selection: Full video")
        self.selection_info_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 10pt;")
        time_range_body_layout.addWidget(self.selection_info_label)

        # Quick presets
        presets_layout = QHBoxLayout()
        presets_layout.addWidget(QLabel("Quick presets:"))
        self.first_5min_btn = QPushButton("First 5min")
        self.first_5min_btn.clicked.connect(lambda: self.set_slider_preset("first_5"))
        self.first_5min_btn.setEnabled(False)
        self.last_5min_btn = QPushButton("Last 5min")
        self.last_5min_btn.clicked.connect(lambda: self.set_slider_preset("last_5"))
        self.last_5min_btn.setEnabled(False)
        self.last_10min_btn = QPushButton("Last 10min")
        self.last_10min_btn.clicked.connect(lambda: self.set_slider_preset("last_10"))
        self.last_10min_btn.setEnabled(False)
        self.middle_btn = QPushButton("Middle")
        self.middle_btn.clicked.connect(lambda: self.set_slider_preset("middle"))
        self.middle_btn.setEnabled(False)
        self.full_video_btn = QPushButton("Full video")
        self.full_video_btn.clicked.connect(lambda: self.set_slider_preset("full"))
        self.full_video_btn.setEnabled(False)
        presets_layout.addWidget(self.first_5min_btn)
        presets_layout.addWidget(self.last_5min_btn)
        presets_layout.addWidget(self.last_10min_btn)
        presets_layout.addWidget(self.middle_btn)
        presets_layout.addWidget(self.full_video_btn)
        presets_layout.addStretch()
        time_range_body_layout.addLayout(presets_layout)

        time_range_layout.addWidget(self.time_range_body)
        # Collapsed unless the box is already ticked from saved config.
        self.time_range_body.setVisible(self.use_time_range_chk.isChecked())

        time_range_group.setLayout(time_range_layout)
        time_range_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout.addWidget(time_range_group)

        # Enable slider if checkbox was already checked from config
        if self.use_time_range_chk.isChecked():
            self.range_slider.setEnabled(True)

        # Initialize the selection info display with saved values
        self.update_selection_info()

        # Load duration from first saved video
        if self.file_list.count() > 0:
            first_path = self.file_list.item(0).text()
            if os.path.exists(first_path):
                self.update_video_duration(first_path)

        # --- Live detection preview (opens a separate window) ---
        # Added before the progress group so that expanding the progress bars
        # (when the pipeline starts) does not push these controls off-screen.
        self.live_preview_checkbox = QCheckBox("Live detection preview (separate window)")
        self.live_preview_checkbox.setToolTip(
            "Open a window showing frames + detected object boxes live while the\n"
            "pipeline runs. Throttled and downscaled — does not slow processing."
        )
        self.live_preview_checkbox.toggled.connect(self._on_live_preview_toggled)
        layout.addWidget(self.live_preview_checkbox)
        self.preview_window = None  # DetectionPreviewWindow, created on demand
        # Read from detection threads, so it mirrors the checkbox rather than
        # being queried across threads (same reason Worker keeps its own copy).
        self._preview_enabled = False
        self.preview_frame.connect(self.on_preview_frame)

        # Force reprocess — the live preview only shows frames while detection
        # actually runs. If results are cached, detection is skipped and the
        # preview stays blank. Tick this to ignore the cache and re-run.
        self.force_reprocess_checkbox = QCheckBox("Force reprocess (ignore cache)")
        self.force_reprocess_checkbox.setToolTip(
            "Re-run analysis even if cached results exist.\n"
            "Required for the live detection preview to show anything on an\n"
            "already-processed video."
        )
        layout.addWidget(self.force_reprocess_checkbox)

        # --- Progress Section (hidden when idle) ---
        self.progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()
        progress_layout.setContentsMargins(4, 4, 4, 4)
        progress_layout.setSpacing(2)

        self.download_progress_bar = QProgressBar()
        self.download_progress_bar.setVisible(False)
        self.download_progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.download_progress_bar)

        # Batch counter as text, not a bar. The whole group is hidden when idle
        # and shown while running, so every row in it is height the window gains
        # at the moment a run starts — and on a screen where the window is
        # already at its limit, that pushes the buttons under the taskbar. The
        # label below said "Video 1/1" anyway, so the bar was a second copy of
        # the same fact costing a row.
        self.batch_label = QLabel()
        self.batch_label.setVisible(False)
        self.batch_label.setStyleSheet("color: #666; font-weight: bold;")
        progress_layout.addWidget(self.batch_label)

        self.process_progress_bar = QProgressBar()
        self.process_progress_bar.setVisible(False)
        self.process_progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.process_progress_bar)

        self.task_label = QLabel("Ready")
        self.task_label.setStyleSheet("color: #666; font-weight: bold;")
        progress_layout.addWidget(self.task_label)

        self.progress_group.setLayout(progress_layout)
        # Hidden when idle; sits here near the top (above the tabs) — its original
        # spot. It only appears while a download or pipeline/analysis runs.
        self.progress_group.setVisible(False)
        layout.addWidget(self.progress_group)

        # --- Tabs ---
        # Kept on self so features elsewhere can bring a tab forward — the
        # advisor hands a run to the LLM Chat tab rather than to a window.
        self.tabs = tabs = QTabWidget()

        # --- Tab 0: Download ---
        download_tab = QWidget()
        download_layout = QVBoxLayout()

        download_group = QGroupBox("Download Videos from Website")
        download_form = QVBoxLayout()

        # URL input
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("Page URL:"))
        self.download_url_input = QLineEdit()
        self.download_url_input.setText(self.config_data.get("download", {}).get("last_url", ""))
        self.download_url_input.setPlaceholderText("https://example.com/videos")
        url_layout.addWidget(self.download_url_input)
        download_form.addLayout(url_layout)

        # Link pattern is auto-detected from the listing page (see
        # downloader.detect_link_pattern), so there's no manual field.

        # Save directory
        save_dir_layout = QHBoxLayout()
        save_dir_layout.addWidget(QLabel("Save directory:"))
        self.download_save_dir_input = QLineEdit()
        self.download_save_dir_input.setText(self.config_data.get("download", {}).get("save_dir", "D:\\movies"))
        save_dir_layout.addWidget(self.download_save_dir_input)
        self.browse_save_dir_btn = QPushButton("Browse...")
        self.browse_save_dir_btn.clicked.connect(self.browse_save_directory)
        save_dir_layout.addWidget(self.browse_save_dir_btn)
        download_form.addLayout(save_dir_layout)

        # Time range selection for downloads. One mode picker instead of two
        # overlapping checkboxes — the modes are mutually exclusive.
        time_range_group = QGroupBox("Download Time Range")
        time_range_layout = QVBoxLayout()

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Download:"))
        self.download_mode_combo = QComboBox()
        self.download_mode_combo.addItem("Full video", "full")
        self.download_mode_combo.addItem("Same range as processing", "same")
        self.download_mode_combo.addItem("Specific range (seconds)", "specific")
        self.download_mode_combo.setToolTip(
            "Full video — download the whole thing.\n"
            "Same range as processing — reuse the Processing Time Range above.\n"
            "Specific range — download only the seconds you set below."
        )
        mode_row.addWidget(self.download_mode_combo)
        mode_row.addStretch()
        time_range_layout.addLayout(mode_row)

        # Manual seconds range — shown only in "Specific range" mode.
        self.download_range_widget = QWidget()
        range_col = QVBoxLayout(self.download_range_widget)
        range_col.setContentsMargins(0, 0, 0, 0)
        time_input_layout = QHBoxLayout()
        time_input_layout.addWidget(QLabel("Start time (seconds):"))
        self.download_start_input = QSpinBox()
        self.download_start_input.setRange(0, 86400)  # 0 to 24 hours
        self.download_start_input.setValue(0)
        time_input_layout.addWidget(self.download_start_input)

        time_input_layout.addWidget(QLabel("End time (seconds):"))
        self.download_end_input = QSpinBox()
        self.download_end_input.setRange(1, 86400)  # 1 second to 24 hours
        self.download_end_input.setValue(300)  # Default: 5 minutes
        time_input_layout.addWidget(self.download_end_input)
        time_input_layout.addStretch()
        range_col.addLayout(time_input_layout)

        self.download_duration_label = QLabel("Duration: 300s (5:00)")
        range_col.addWidget(self.download_duration_label)
        time_range_layout.addWidget(self.download_range_widget)

        # Connect signals
        self.download_start_input.valueChanged.connect(self.update_download_duration)
        self.download_end_input.valueChanged.connect(self.update_download_duration)
        self.download_mode_combo.currentIndexChanged.connect(self.on_download_mode_changed)

        time_range_group.setLayout(time_range_layout)
        download_form.addWidget(time_range_group)

        # Options
        self.auto_add_downloaded_chk = QCheckBox("Automatically add downloaded videos to file list")
        self.auto_add_downloaded_chk.setChecked(self.config_data.get("download", {}).get("auto_add", True))
        download_form.addWidget(self.auto_add_downloaded_chk)

        # After-download processing: one mode picker instead of two overlapping
        # (and half-dead) checkboxes.
        process_row = QHBoxLayout()
        process_row.addWidget(QLabel("After download:"))
        self.process_mode_combo = QComboBox()
        self.process_mode_combo.addItem("Don't process", "none")
        self.process_mode_combo.addItem("Process each video as it downloads", "immediate")
        self.process_mode_combo.addItem("Process all after downloads finish", "batch")
        self.process_mode_combo.setToolTip(
            "Don't process — just download.\n"
            "Process each as it downloads — run the pipeline per video, overlapping with remaining downloads.\n"
            "Process all after downloads finish — download everything first, then run the pipeline over the list."
        )
        process_row.addWidget(self.process_mode_combo)
        process_row.addStretch()
        download_form.addLayout(process_row)

        # Concurrent downloads — only meaningful while processing overlaps
        # downloads (the "immediate" mode).
        concurrent_layout = QHBoxLayout()
        concurrent_layout.addWidget(QLabel("Concurrent downloads:"))
        self.concurrent_spinbox = QSpinBox()
        self.concurrent_spinbox.setRange(1, 10)
        self.concurrent_spinbox.setValue(self.config_data.get("download", {}).get("concurrent_downloads", 1))
        self.concurrent_spinbox.setToolTip("Number of videos to download simultaneously (higher = faster but more resource intensive)")
        concurrent_layout.addWidget(self.concurrent_spinbox)
        concurrent_layout.addStretch()
        download_form.addLayout(concurrent_layout)

        self.process_mode_combo.currentIndexChanged.connect(self.on_process_mode_changed)
        self.on_process_mode_changed()  # sync spinner enabled state

        # Download buttons. The pair is "choose some" vs "take everything", so
        # the labels say which is which, and only the second gets accent fill.
        download_btn_layout = QHBoxLayout()
        self.browse_select_btn = QPushButton("Pick Videos from Page…")
        self.browse_select_btn.setIcon(_ui_icons.picker())
        # No inline style: this is a plain secondary button, so it inherits the
        # theme's default QPushButton and stays in step if the palette changes.
        self.browse_select_btn.setToolTip("Open a grid of the site's videos (thumbnails) and pick which ones to download")
        self.browse_select_btn.clicked.connect(self.browse_and_select_videos)

        self.download_btn = QPushButton("Download All")
        self.download_btn.setIcon(_ui_icons.download())
        self.download_btn.setToolTip("Download every video found on the page, without picking")
        # Accent fill marks the primary action. The disabled rule matters: this
        # button is switched off for the whole download, and without it the fill
        # stays bright blue and keeps inviting clicks that do nothing.
        _p = _ui_theme.DARK
        self.download_btn.setStyleSheet(
            f"QPushButton {{ background-color: {_p.accent}; color: {_p.on_accent};"
            f" font-weight: bold; padding: 8px; border: none; border-radius: {_p.radius}px; }}"
            f"QPushButton:hover {{ background-color: {_p.accent_hover}; }}"
            f"QPushButton:pressed {{ background-color: {_p.accent_press}; }}"
            f"QPushButton:disabled {{ background-color: {_p.surface}; color: {_p.text_mute}; }}"
        )
        # lambda so the clicked(bool) arg isn't passed as start_download's video_urls
        self.download_btn.clicked.connect(lambda: self.start_download())
        download_btn_layout.addStretch()
        download_btn_layout.addWidget(self.browse_select_btn)
        download_btn_layout.addWidget(self.download_btn)
        download_form.addLayout(download_btn_layout)

        download_group.setLayout(download_form)
        download_layout.addWidget(download_group)
        download_layout.addStretch()
        download_tab.setLayout(download_layout)
        tabs.addTab(self._scrollable(download_tab), "Download")

        # --- Tab 1: Basic Settings ---
        basic_tab = QWidget()
        # Grid so the two tall groups (Scoring Points / Duration) sit side by side
        # and use horizontal space instead of stacking into one tall column that
        # overflows the window. Mirrors the Advanced tab layout.
        basic_layout = QGridLayout()

        # On-demand run buttons sit right next to the scoring row they run. Each
        # runs that one signal over every video in the list and folds the result
        # into each cache — no highlights are cut. Registered here so a full
        # pipeline run (or another on-demand run) can grey them out.
        self._analyze_buttons = {}
        self._signal_worker = None

        # ── Group 1: Scoring Points ──
        points_box = QGroupBox("Scoring Points")
        points_layout = QVBoxLayout()
        points_layout.setSpacing(6)

        self.spin_scene_points = QSpinBox(); self.spin_scene_points.setRange(0,100); self.spin_scene_points.setValue(scoring_cfg.get("scene_points", 0))
        self.spin_scene_points.setToolTip("Points awarded when a new scene cut is detected (abrupt visual change)")

        self.spin_motion_event_points = QSpinBox(); self.spin_motion_event_points.setRange(0,100); self.spin_motion_event_points.setValue(scoring_cfg.get("motion_event_points", 0))
        self.spin_motion_event_points.setToolTip("Points for any frame with detected movement above the threshold")

        self.spin_motion_peak = QSpinBox(); self.spin_motion_peak.setRange(0,100); self.spin_motion_peak.setValue(scoring_cfg.get("motion_peak_points", 3))
        self.spin_motion_peak.setToolTip("Points for a sudden burst of motion followed by stillness (e.g. a goal followed by replay, an explosion then calm)")

        self.spin_audio_peak = QSpinBox(); self.spin_audio_peak.setRange(0,100); self.spin_audio_peak.setValue(scoring_cfg.get("audio_peak_points", 0))
        self.spin_audio_peak.setToolTip("Points when audio intensity spikes (e.g. crowd roar, explosions, bells, loud impacts)")

        self.spin_loudness_burst = QSpinBox(); self.spin_loudness_burst.setRange(0,100)
        self.spin_loudness_burst.setValue(scoring_cfg.get("loudness_burst_points", 0))
        self.spin_loudness_burst.setToolTip(
            "Points where the audio rises above its OWN local level, not a fixed "
            "threshold.\n\nUse this instead of audio peak points when the interesting "
            "moments are loud for this part of this video rather than loud in "
            "absolute terms \u2014 it self-calibrates, so the same setting works on "
            "quietly and loudly mastered files.\n\nFinds brief moments that stand out "
            "from their surroundings; a passage that is loud throughout will not "
            "stand out from itself and is not reported.")

        self.spin_keyword_points = QSpinBox(); self.spin_keyword_points.setRange(0,100); self.spin_keyword_points.setValue(scoring_cfg.get("keyword_points", 2))
        self.spin_keyword_points.setToolTip("Points when a search keyword is found in speech (needs transcript enabled)")
        # Keyword scoring only works with a transcript — grey it out until then.
        self.spin_keyword_points.setEnabled(self.config_data.get("transcript", {}).get("enabled", False))

        self.spin_transcript_points = QSpinBox(); self.spin_transcript_points.setRange(0,100); self.spin_transcript_points.setValue(scoring_cfg.get("transcript_points", 2))
        self.spin_transcript_points.setToolTip("Points for any moment where speech is detected, regardless of content")

        self.spin_object = QSpinBox(); self.spin_object.setRange(0,100); self.spin_object.setValue(scoring_cfg.get("object_points", 1))
        self.spin_object.setToolTip("Points when a configured object class is detected in the frame")

        self.spin_action = QSpinBox(); self.spin_action.setRange(0,1000); self.spin_action.setValue(scoring_cfg.get("action_points", 10))
        self.spin_action.setToolTip("Points when a configured action is recognized (e.g. punching, jumping, dancing)")

        self.spin_face_expression = QSpinBox(); self.spin_face_expression.setRange(0,100)
        self.spin_face_expression.setValue(scoring_cfg.get("face_expression_points", 0))
        self.spin_face_expression.setToolTip(
            "Points for a second whose strongest face reads as one of the "
            "expressions you pick beside this.\n\nThe scan runs only when this "
            "is above 0 AND at least one expression is chosen — with either "
            "missing there is no outcome it could change, so it is skipped.\n\n"
            "It reports what a five-class classifier saw on a face, not what "
            "anyone felt: it has no notion of intensity, degrades on profile "
            "and occlusion, and cannot tell a performed expression from a felt "
            "one.")
        self._face_label_actions = {}
        self.btn_face_labels = QToolButton()
        self.btn_face_labels.setPopupMode(QToolButton.InstantPopup)
        self.btn_face_labels.setToolTip(
            "Which expressions earn the points above. Picking all of them "
            "scores every second a face is visible, which distinguishes "
            "nothing — so choose the ones that mark the moments you want.")
        face_menu = QMenu(self.btn_face_labels)
        chosen = {str(x).lower()
                  for x in (scoring_cfg.get("face_expression_labels") or [])}
        for _label in EMOTION_LABELS:
            act = face_menu.addAction(_label)
            act.setCheckable(True)
            act.setChecked(_label in chosen)
            act.toggled.connect(self._update_face_labels_button)
            self._face_label_actions[_label] = act
        self.btn_face_labels.setMenu(face_menu)
        self._update_face_labels_button()

        self.spin_beginning_seconds = QSpinBox(); self.spin_beginning_seconds.setRange(0,3600); self.spin_beginning_seconds.setSuffix(" s"); self.spin_beginning_seconds.setValue(scoring_cfg.get("beginning_seconds", 60))
        self.spin_beginning_seconds.setToolTip("How many seconds from the start of the video count as the intro window")

        self.spin_beginning_points = QSpinBox(); self.spin_beginning_points.setRange(0,100); self.spin_beginning_points.setSuffix(" pts"); self.spin_beginning_points.setValue(scoring_cfg.get("beginning_points", 0))
        self.spin_beginning_points.setToolTip("Points added to every second in the intro window — raise to make the intro more likely to be picked as a highlight, 0 to score it like anything else")

        self.spin_ending_seconds = QSpinBox(); self.spin_ending_seconds.setRange(0,3600); self.spin_ending_seconds.setSuffix(" s"); self.spin_ending_seconds.setValue(scoring_cfg.get("ending_seconds", 120))
        self.spin_ending_seconds.setToolTip("How many seconds before the end of the video count as the outro window")

        self.spin_ending_points = QSpinBox(); self.spin_ending_points.setRange(0,100); self.spin_ending_points.setSuffix(" pts"); self.spin_ending_points.setValue(scoring_cfg.get("ending_points", 0))
        self.spin_ending_points.setToolTip("Points added to every second in the outro window — raise to make the outro more likely to be picked as a highlight, 0 to score it like anything else")

        intro_row = QHBoxLayout()
        intro_row.addWidget(self.spin_beginning_seconds)
        intro_row.addWidget(self.spin_beginning_points)
        intro_row.addStretch(1)
        intro_widget = QWidget(); intro_widget.setLayout(intro_row)

        outro_row = QHBoxLayout()
        outro_row.addWidget(self.spin_ending_seconds)
        outro_row.addWidget(self.spin_ending_points)
        outro_row.addStretch(1)
        outro_widget = QWidget(); outro_widget.setLayout(outro_row)

        # One box per signal rather than one list of eleven rows. The labels
        # only mean anything once the reader knows which detector a row belongs
        # to, and in a flat column that has to be inferred from the wording of
        # each label — which is why the shortest ones ("Scene points") were the
        # hardest to place. A box answers it before the row is read.
        face_row = QWidget()
        face_h = QHBoxLayout(face_row)
        face_h.setContentsMargins(0, 0, 0, 0)
        face_h.setSpacing(6)
        face_h.addWidget(self.spin_face_expression)
        face_h.addWidget(self.btn_face_labels)
        face_h.addStretch(1)

        # Scene / motion-event / motion-peak all come from ONE detector pass, so
        # the button lives on the first of the three and runs all three.
        # Keyword + transcript points likewise share one transcription pass.
        groups = (
            ("Movement && scenes", (
                ("Scene points:", self._points_row_with_button(
                    self.spin_scene_points, "motion", "Motion & scenes",
                    "Detect scene cuts and motion across every video in the list "
                    "and cache them (covers scene, motion event and motion peak "
                    "— one pass). No highlights are cut.")),
                ("Motion event points:", self.spin_motion_event_points),
                ("Motion peak points:", self.spin_motion_peak),
            )),
            ("Audio", (
                ("Audio peak points:", self._points_row_with_button(
                    self.spin_audio_peak, "audio", "Audio",
                    "Detect audio peaks across every video in the list and cache "
                    "them. No highlights are cut.")),
                ("Loudness burst points (vs local level):",
                 self.spin_loudness_burst),
            )),
            # Composition earns a row here rather than living only beside its
            # editor in Advanced. Rules built from signal conditions are an
            # analysis pass like any other on this page — they measure the file
            # and cache a result — and they are the one kind that needs no
            # previous run, so requiring a trip to another tab to start them put
            # the cheapest signal behind the most navigation.
            ("Composition rules", (
                ("Apply saved rules:", self._rules_run_row()),
            )),
            ("Speech", (
                ("Keyword points (keywords in transcript):",
                 self.spin_keyword_points),
                ("Transcript points (all words):", self._points_row_with_button(
                    self.spin_transcript_points, "transcript", "Transcribe",
                    "Transcribe every video in the list to a _transcript.txt "
                    "sidecar and cache it (uses the model/language in the "
                    "Transcript tab). No highlights are cut.")),
            )),
            ("Objects && actions", (
                ("Object points:", self._points_row_with_button(
                    self.spin_object, "objects", "Objects",
                    "Detect the classes from the 'Object detection' field below, "
                    "across every video in the list, and cache them. No "
                    "highlights are cut.")),
                ("Action points:", self._points_row_with_button(
                    self.spin_action, "actions", "Actions",
                    "Detect the actions from the 'Action keywords' field below "
                    "(blank = all actions), across every video in the list, and "
                    "cache them. No highlights are cut.")),
            )),
            ("Face expression", (
                ("Points, and which expressions:", face_row),
            )),
            ("Where in the video", (
                ("Intro (window, points):", intro_widget),
                ("Outro (window, points):", outro_widget),
            )),
            ("Speech", (
                ("Keyword points (keywords in transcript):",
                 self.spin_keyword_points),
                ("Transcript points (all words):", self._points_row_with_button(
                    self.spin_transcript_points, "transcript", "Transcribe",
                    "Transcribe every video in the list to a _transcript.txt "
                    "sidecar and cache it (uses the model/language in the "
                    "Transcript tab). No highlights are cut.")),
            )),
            ("Objects && actions", (
                ("Object points:", self._points_row_with_button(
                    self.spin_object, "objects", "Objects",
                    "Detect the classes from the 'Object detection' field below, "
                    "across every video in the list, and cache them. No "
                    "highlights are cut.")),
                ("Action points:", self._points_row_with_button(
                    self.spin_action, "actions", "Actions",
                    "Detect the actions from the 'Action keywords' field below "
                    "(blank = all actions), across every video in the list, and "
                    "cache them. No highlights are cut.")),
            )),
            ("Face expression", (
                ("Points, and which expressions:", face_row),
            )),
            ("Where in the video", (
                ("Intro (window, points):", intro_widget),
                ("Outro (window, points):", outro_widget),
            )),
        )
        for title, rows in groups:
            points_layout.addWidget(self._points_group(title, rows))
        points_layout.addStretch(1)

        points_box.setLayout(points_layout)
        basic_layout.addWidget(points_box, 0, 0, Qt.AlignTop)

        # ── Group 2: Duration & Cutting ──
        duration_box = QGroupBox("Duration && Cutting")
        duration_layout = QVBoxLayout()

        # Main duration controls (always visible)
        duration_form = QFormLayout()

        self.spin_max_duration = QSpinBox(); self.spin_max_duration.setRange(1,3600); self.spin_max_duration.setValue(highlights_cfg.get("max_duration", 420))
        self.spin_exact_duration = QSpinBox(); self.spin_exact_duration.setRange(0,3600); self.spin_exact_duration.setValue(highlights_cfg.get("exact_duration", 0))
        self.spin_clip_time = QSpinBox(); self.spin_clip_time.setRange(0,300); self.spin_clip_time.setValue(highlights_cfg.get("clip_time", 10))

        duration_form.addRow("Max highlight duration (s):", self.spin_max_duration)
        duration_form.addRow("Exact duration (s, 0 = off):", self.spin_exact_duration)
        duration_form.addRow("Clip time (s, 0 = auto):", self.spin_clip_time)

        # ── Best moments <-> Full story ──
        # How far the cut is allowed to follow the score. Left, it takes the
        # highest-scoring moments wherever they are, which on a video whose
        # action is concentrated can mean the whole cut comes from one stretch.
        # Right, every part of the video contributes.
        self.slider_coverage = QSlider(Qt.Horizontal)
        self.slider_coverage.setRange(0, 100)
        self.slider_coverage.setValue(int(round(float(highlights_cfg.get("coverage", 0.0)) * 100)))
        self.slider_coverage.setTickPosition(QSlider.TicksBelow)
        self.slider_coverage.setTickInterval(25)

        coverage_row = QVBoxLayout()
        coverage_row.addWidget(self.slider_coverage)
        self.coverage_hint_label = QLabel("")
        self.coverage_hint_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        self.coverage_hint_label.setWordWrap(True)
        coverage_row.addWidget(self.coverage_hint_label)

        def on_coverage_changed(value):
            if value <= 5:
                hint = "Best moments — highest-scoring parts only, wherever they fall."
            elif value >= 95:
                hint = "Full story — every part of the video is represented."
            else:
                hint = f"{value}% toward full story — best moments, spread across the video."
            self.coverage_hint_label.setText(hint)

        self.slider_coverage.valueChanged.connect(on_coverage_changed)
        on_coverage_changed(self.slider_coverage.value())

        duration_form.addRow("Best moments ↔ Full story:", coverage_row)

        duration_layout.addLayout(duration_form)

        # Auto-segmentation info label (always visible, updates dynamically)
        self.auto_seg_info_label = QLabel("")
        self.auto_seg_info_label.setStyleSheet("color: #2f81f7; font-style: italic; padding: 4px;")
        self.auto_seg_info_label.setWordWrap(True)
        duration_layout.addWidget(self.auto_seg_info_label)

        # ── Auto-segmentation controls (shown only when clip_time = 0) ──
        self.auto_seg_group = QGroupBox("Auto-Segmentation Settings")
        auto_seg_layout = QFormLayout()

        self.spin_auto_min_clip = QSpinBox()
        self.spin_auto_min_clip.setRange(1, 30)
        self.spin_auto_min_clip.setValue(highlights_cfg.get("auto_min_clip", 2))
        self.spin_auto_min_clip.setToolTip("Shortest clip the auto-cutter will produce")

        self.spin_auto_max_clip = QSpinBox()
        self.spin_auto_max_clip.setRange(3, 120)
        self.spin_auto_max_clip.setValue(highlights_cfg.get("auto_max_clip", 30))
        self.spin_auto_max_clip.setToolTip("Longest single clip before it gets trimmed to the best sub-window")

        self.spin_auto_merge_gap = QSpinBox()
        self.spin_auto_merge_gap.setRange(0, 10)
        self.spin_auto_merge_gap.setValue(highlights_cfg.get("auto_merge_gap", 2))
        self.spin_auto_merge_gap.setToolTip("Merge interest regions that are within this gap into one clip")

        auto_seg_layout.addRow("Min clip length (s):", self.spin_auto_min_clip)
        auto_seg_layout.addRow("Max clip length (s):", self.spin_auto_max_clip)
        auto_seg_layout.addRow("Merge gap (s):", self.spin_auto_merge_gap)

        self.auto_seg_group.setLayout(auto_seg_layout)
        duration_layout.addWidget(self.auto_seg_group)

        duration_box.setLayout(duration_layout)
        basic_layout.addWidget(duration_box, 0, 1, Qt.AlignTop)

        # ── Connect clip_time spinner to show/hide auto-seg controls ──
        def on_clip_time_changed(value):
            is_auto = (value == 0)
            self.auto_seg_group.setVisible(is_auto)
            if is_auto:
                self.auto_seg_info_label.setText(
                    "🔧 Auto mode: the app will determine clip boundaries from signal structure "
                    "(action durations, scene cuts, keyword timing, object clusters, audio/motion peaks)."
                )
            else:
                self.auto_seg_info_label.setText(
                    f"✂️ Fixed mode: each highlight clip will be {value}s long."
                )

        self.spin_clip_time.valueChanged.connect(on_clip_time_changed)
        # Trigger once to set initial state
        on_clip_time_changed(self.spin_clip_time.value())

        # Highlight object classes
        obj_layout = QHBoxLayout()
        self.objects_input = QLineEdit(",".join(self.config_data.get("objects", {}).get("interesting", [])))
        self.objects_input.setPlaceholderText("person,glass,wine glass,sports ball")
        obj_layout.addWidget(QLabel("Object detection:"))
        obj_layout.addWidget(self.objects_input)
        self.load_objects_btn = QPushButton("Load Labels")
        self.load_objects_btn.setToolTip("Load labels from yolo_objects_labels.json")
        self.load_objects_btn.clicked.connect(self.open_object_label_selector)
        obj_layout.addWidget(self.load_objects_btn)
        basic_layout.addLayout(obj_layout, 1, 0, 1, 2)

        # Action keywords
        action_kw_layout = QHBoxLayout()
        self.actions_input = QLineEdit(",".join(self.config_data.get("actions", {}).get("interesting", [])))
        self.actions_input.setPlaceholderText("high jump, high kick, archery")
        action_kw_layout.addWidget(QLabel("Action keywords:"))
        action_kw_layout.addWidget(self.actions_input)
        self.load_actions_btn = QPushButton("Load Labels")
        self.load_actions_btn.setToolTip("Load labels from kinetics_400_labels.json (or custom Intel model)")
        self.load_actions_btn.clicked.connect(self.open_action_label_selector)
        action_kw_layout.addWidget(self.load_actions_btn)
        basic_layout.addLayout(action_kw_layout, 2, 0, 1, 2)

        # Transcript search keywords — moved here from the Transcript tab: it's a
        # common highlight signal (score moments where these words are spoken).
        # Greyed out unless transcript processing is enabled (nothing to search
        # otherwise); tracks the transcript toggle via on_transcript_toggle.
        _kw_enabled = self.config_data.get("transcript", {}).get("enabled", False)
        kw_layout = QHBoxLayout()
        self.search_keywords_input = QLineEdit(",".join(self.config_data.get("transcript", {}).get("search_keywords", [])))
        self.search_keywords_input.setPlaceholderText("goal, score, win")
        self.search_keywords_input.setToolTip("Score moments where these spoken words appear (needs transcript enabled)")
        self.search_keywords_input.setEnabled(_kw_enabled)
        self.search_keywords_label = QLabel("Transcript keywords:")
        self.search_keywords_label.setEnabled(_kw_enabled)
        kw_layout.addWidget(self.search_keywords_label)
        kw_layout.addWidget(self.search_keywords_input)
        basic_layout.addLayout(kw_layout, 3, 0, 1, 2)

        # Conditional action scoring checkbox
        self.actions_require_objects_chk = QCheckBox("Only score actions when objects detected")
        self.actions_require_objects_chk.setChecked(self.config_data.get("actions", {}).get("require_objects", False))
        self.actions_require_objects_chk.setToolTip("Actions will only add points if objects are also detected in that timeframe")
        basic_layout.addWidget(self.actions_require_objects_chk, 4, 0, 1, 2)

        # (The old "Skip highlights" checkbox is gone. Producing a transcript or
        # subtitles without cutting highlights is now a per-signal run button in
        # the Scoring Points panel above — Transcribe/Objects/Actions/etc. — so
        # scores never have to be zeroed by hand.)

        # Combine every processed video's highlights into one master video.
        # Config key stays under "download" (auto_combine) so saved configs load.
        self.auto_combine_chk = QCheckBox("Combine highlights from all processed videos into one video")
        self.auto_combine_chk.setChecked(self.config_data.get("download", {}).get("auto_combine", True))
        self.auto_combine_chk.setToolTip("When enabled, the highlights from every processed video are merged into a single master video")
        basic_layout.addWidget(self.auto_combine_chk, 5, 0, 1, 2)

        # Equal-width columns; trailing stretch row keeps groups packed at the top.
        basic_layout.setColumnStretch(0, 1)
        basic_layout.setColumnStretch(1, 1)
        basic_layout.setRowStretch(6, 1)

        basic_tab.setLayout(basic_layout)
        tabs.addTab(self._scrollable(basic_tab), "Basic Settings")

        # --- Tab 2: Transcript & Subtitles ---
        transcript_cfg = self.config_data.get("transcript", {})
        subtitles_cfg = self.config_data.get("subtitles", {})

        transcript_tab = QWidget()
        transcript_layout = QVBoxLayout()

        transcript_group = QGroupBox("Transcript Settings")
        transcript_form = QFormLayout()
        self.transcript_checkbox = QCheckBox("Enable transcript processing")
        self.transcript_checkbox.setChecked(transcript_cfg.get("enabled", False))
        self.transcript_checkbox.toggled.connect(self.on_transcript_toggle)
        transcript_form.addRow("Use transcript:", self.transcript_checkbox)

        # Source language for transcription
        self.transcript_source_lang = QComboBox()
        self.transcript_source_lang.addItems(["auto","en","pl","es","fr","de","it","pt","ru","ja","ko","zh"])
        self.transcript_source_lang.setCurrentText(transcript_cfg.get("source_lang", "en"))
        self.transcript_source_lang.setEnabled(transcript_cfg.get("enabled", False))
        transcript_form.addRow("Source language:", self.transcript_source_lang)

        self.transcript_model_combo = QComboBox()
        self.transcript_model_combo.addItems(["tiny","base","small","medium","large"])
        self.transcript_model_combo.setCurrentText(transcript_cfg.get("model", "base"))
        self.transcript_model_combo.setEnabled(transcript_cfg.get("enabled", False))
        transcript_form.addRow("Whisper model:", self.transcript_model_combo)

        # (Search keywords moved to the Basic Settings tab — a common highlight
        # signal, editable without opening this tab.)
        transcript_group.setLayout(transcript_form)
        transcript_layout.addWidget(transcript_group)

        subtitle_group = QGroupBox("Subtitle Settings")
        subtitle_form = QFormLayout()
        self.subtitles_checkbox = QCheckBox("Generate subtitles (.srt)")
        self.subtitles_checkbox.setChecked(subtitles_cfg.get("enabled", False))
        self.subtitles_checkbox.toggled.connect(self.on_subtitles_toggle)
        # Disable subtitle checkbox if transcript is not enabled
        self.subtitles_checkbox.setEnabled(transcript_cfg.get("enabled", False))
        subtitle_form.addRow("Create subtitles:", self.subtitles_checkbox)

        # No "source language" here. What is spoken is a property of the video,
        # it is already declared in Transcript Settings above (which is what
        # Whisper is actually told), and asking twice only let the two answers
        # disagree — a subtitle box saying "en" never made Russian audio English,
        # it just mislabelled the translation and named the file wrong. When a
        # cached transcript is reused, its own recorded language is used.
        self.subtitle_target_lang = QComboBox()
        self.subtitle_target_lang.addItems(["en","pl","es","fr","de","it","pt","ru","ja","ko","zh"])
        self.subtitle_target_lang.setCurrentText(subtitles_cfg.get("target_lang", "pl"))
        self.subtitle_target_lang.setEnabled(subtitles_cfg.get("enabled", False) and transcript_cfg.get("enabled", False))
        subtitle_form.addRow("Target language:", self.subtitle_target_lang)
        _sub_run = self._make_analyze_button(
            "subtitles", "Make subtitles",
            "Transcribe every video in the list and write a .srt next to each "
            "(translated when the target language differs). No highlights are cut.")
        subtitle_form.addRow("", _sub_run)
        subtitle_group.setLayout(subtitle_form)
        transcript_layout.addWidget(subtitle_group)

        transcript_tab.setLayout(transcript_layout)
        tabs.addTab(self._scrollable(transcript_tab), "Transcript && Subtitles")

        # --- Tab 3: Advanced Tab ---
        advanced_cfg = self.config_data.get("advanced", {})
        visualization_cfg = self.config_data.get("visualization", {})

        advanced_tab = QWidget()
        # Grid so the small groups sit side by side and use horizontal space
        # (especially when maximized) instead of one tall scrolling column.
        advanced_layout = QGridLayout()

        # ── Group 1: Motion Recognition ──
        motion_box = QGroupBox("Motion Recognition")
        motion_layout = QFormLayout()

        self.frame_skip_spin = QSpinBox()
        self.frame_skip_spin.setRange(1, 30)
        self.frame_skip_spin.setValue(advanced_cfg.get("frame_skip", 5))
        self.frame_skip_spin.setToolTip("Analyze every Nth frame for motion detection (higher = faster, less precise)")

        motion_layout.addRow("Frame skip:", self.frame_skip_spin)
        self.vr_mode_chk = QCheckBox("VR side-by-side optimization")
        self.vr_mode_chk.setChecked(bool(advanced_cfg.get("vr_mode", False)))
        self.vr_mode_chk.setToolTip(
            "Run visual analysis on the left half only for side-by-side VR/3D videos."
        )
        motion_layout.addRow("", self.vr_mode_chk)
        motion_box.setLayout(motion_layout)
        advanced_layout.addWidget(motion_box, 1, 0)

        # ── Group 2: Object Recognition ──
        object_box = QGroupBox("Object Recognition")
        object_layout = QFormLayout()

        self.obj_frame_skip_spin = QSpinBox()
        self.obj_frame_skip_spin.setRange(1, 60)
        self.obj_frame_skip_spin.setValue(advanced_cfg.get("object_frame_skip", 10))
        self.obj_frame_skip_spin.setToolTip("Analyze every Nth frame for object detection (higher = faster, less precise)")

        self.yolo_type_combo = QComboBox()
        self.yolo_type_combo.addItem("Standard YOLO (80 objects)", "standard")

        # Pro v1 keeps pose/keypoints disabled until a permissive backend lands.
        self._custom_pose_model = None

        self.yolo_model_combo = QComboBox()

        # Object model selector: standard COCO / Custom / Mixed, auto-discovered
        # from models/custom/. Loaded natively by ultralytics; class names come
        # from each model's own metadata.
        self.object_model_combo = QComboBox()
        self.object_model_combo.setToolTip(
            "Standard — the 80 COCO objects\n"
            "Custom — a model you trained (auto-detected from models/custom/)\n"
            "Mixed — standard YOLO + your custom model together")

        import_obj_btn = QPushButton("Import model…")
        import_obj_btn.setToolTip("Copy a trained model (.pt / .onnx) into models/custom/")
        obj_model_row = QHBoxLayout()
        obj_model_row.setContentsMargins(0, 0, 0, 0)
        obj_model_row.addWidget(self.object_model_combo, 1)
        obj_model_row.addWidget(import_obj_btn)
        self.object_model_widget = QWidget()
        self.object_model_widget.setLayout(obj_model_row)

        def _populate_object_models(select_type=None, select_path=""):
            """Rebuild the combo from discovery. Each entry's data is
            (yolo_type, path), matching what the pipeline consumes."""
            from modules.app_paths import discover_object_models
            self.object_model_combo.blockSignals(True)
            self.object_model_combo.clear()
            self.object_model_combo.addItem("Standard YOLO (80 objects)", ("standard", ""))
            models = []
            try:
                models = discover_object_models()
            except Exception as e:
                print(f"⚠️ object model discovery failed: {e}")
            for m in models:
                n = len(m["classes"])
                self.object_model_combo.addItem(
                    f"Custom — {m['name']} ({n} classes)", ("custom", m["path"]))
            for m in models:
                n = len(m["classes"])
                self.object_model_combo.addItem(
                    f"Mixed — standard + {m['name']} (80 + {n})", ("custom_mixed", m["path"]))

            # Restore selection by (type, path); fall back to standard.
            target = (select_type or "standard", select_path or "")
            idx = next((i for i in range(self.object_model_combo.count())
                        if self.object_model_combo.itemData(i) == target), 0)
            self.object_model_combo.setCurrentIndex(idx)
            self.object_model_combo.blockSignals(False)

        def _import_object_model():
            from modules.app_paths import object_models_dir
            src, _ = QFileDialog.getOpenFileName(
                self, "Import object detector model", "",
                "YOLO models (*.pt *.onnx);;All files (*)")
            if not src:
                return
            dst_dir = object_models_dir()
            try:
                os.makedirs(dst_dir, exist_ok=True)
                import shutil
                dst = os.path.join(dst_dir, os.path.basename(src))
                shutil.copy2(src, dst)
                _populate_object_models(select_type="custom", select_path=dst)
                self.append_log(f"✅ Imported object model: {os.path.basename(dst)}")
            except Exception as e:
                self.append_log(f"⚠️ Object model import failed: {e}")

        import_obj_btn.clicked.connect(_import_object_model)

        def on_object_model_changed(index=0):
            yolo_type = self.object_detector_choice()[0]
            prev_size = self.yolo_model_combo.currentData()
            self.yolo_model_combo.blockSignals(True)
            self.yolo_model_combo.clear()

            # Mixed still runs the standard detector, so the size stays live;
            # only custom-only makes it moot.
            custom_only = (yolo_type == "custom")

            if custom_only:
                # Size applies to the standard detector, which isn't used here
                self.yolo_model_combo.addItem("(custom model — size N/A)", "n")
                self.yolo_model_combo.setEnabled(False)
            else:
                self.yolo_model_combo.addItem("Nano (fastest, lowest accuracy)", "n")
                self.yolo_model_combo.addItem("Small (fast, good balance)", "s")
                self.yolo_model_combo.addItem("Medium (balanced)", "m")
                self.yolo_model_combo.addItem("Large (accurate, slower)", "l")
                self.yolo_model_combo.addItem("Extra-Large (most accurate, slowest)", "x")
                self.yolo_model_combo.setEnabled(True)

            restore_idx = self.yolo_model_combo.findData(prev_size)
            if restore_idx >= 0:
                self.yolo_model_combo.setCurrentIndex(restore_idx)
            self.yolo_model_combo.blockSignals(False)

        _populate_object_models(
            select_type=advanced_cfg.get("yolo_type", "standard"),
            select_path=advanced_cfg.get("yolo_custom_model_path", "") or "",
        )
        self.object_model_combo.currentIndexChanged.connect(on_object_model_changed)

        current_model = advanced_cfg.get("yolo_model_size", "n")
        on_object_model_changed()
        idx = self.yolo_model_combo.findData(current_model)
        self.yolo_model_combo.setCurrentIndex(idx if idx >= 0 else 0)

        self.obj_confidence_spin = QSpinBox()
        self.obj_confidence_spin.setRange(5, 95)
        self.obj_confidence_spin.setSuffix("%")
        self.obj_confidence_spin.setValue(int(self.config_data.get("objects", {}).get("confidence", 30)))
        self.obj_confidence_spin.setToolTip("Minimum confidence threshold for object detection (lower = more detections, more false positives)")

        object_layout.addRow("Frame skip:", self.obj_frame_skip_spin)
        object_layout.addRow("Detector type:", self.yolo_type_combo)
        object_layout.addRow("Detector model size:", self.yolo_model_combo)
        object_layout.addRow("Object model:", self.object_model_widget)
        object_layout.addRow("Confidence threshold:", self.obj_confidence_spin)

        object_box.setLayout(object_layout)
        advanced_layout.addWidget(object_box, 2, 0)

        # ── Group 3: Action Recognition ──
        action_box = QGroupBox("Action Recognition")
        action_layout = QFormLayout()

        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(1, 30)
        self.sample_rate_spin.setValue(advanced_cfg.get("sample_rate", 5))
        self.sample_rate_spin.setToolTip("Sample every Nth frame for action recognition clips")

        self.action_backend_combo = QComboBox()
        self.action_backend_combo.addItem("Auto (CUDA / OpenVINO / CPU)", "auto")
        self.action_backend_combo.addItem("OpenVINO (Intel GPU / CPU)", "openvino")
        self.action_backend_combo.addItem("R3D + CUDA (NVIDIA GPU)", "r3d_cuda")
        self.action_backend_combo.addItem("R3D + CPU (PyTorch, slow)", "r3d_cpu")
        current_backend = advanced_cfg.get("action_backend", "auto")
        idx_ab = self.action_backend_combo.findData(current_backend)
        self.action_backend_combo.setCurrentIndex(idx_ab if idx_ab >= 0 else 0)

        self._intel_count = len(self.load_labels_from_json(KINETICS_400_LABELS_FILE)) if os.path.exists(KINETICS_400_LABELS_FILE) else 0
        self._custom_ov_count = len(self.load_labels_from_json(INTEL_CUSTOM_LABELS_FILE)) if os.path.exists(INTEL_CUSTOM_LABELS_FILE) else 0
        self._r3d_custom_count = len(self.load_labels_from_json(R3D_CUSTOM_LABELS_FILE)) if os.path.exists(R3D_CUSTOM_LABELS_FILE) else 0

        self.action_models_combo = QComboBox()

        import_action_btn = QPushButton("Import model…")
        import_action_btn.setToolTip(
            "Copy a trained custom action model into the app's custom-model slot:\n"
            "  • OpenVINO decoder (.xml + .bin)\n"
            "  • R3D fine-tuned weights (.pth)\n"
            "A same-named .json (labels / mapping) next to it is picked up "
            "automatically, or you'll be asked to pick one.")
        action_model_row = QHBoxLayout()
        action_model_row.addWidget(self.action_models_combo, 1)
        action_model_row.addWidget(import_action_btn)
        action_model_widget = QWidget()
        action_model_widget.setLayout(action_model_row)

        def _import_action_model():
            src, _ = QFileDialog.getOpenFileName(
                self, "Import custom action model", "",
                "Action models (*.xml *.pth);;OpenVINO IR (*.xml);;"
                "R3D weights (*.pth);;All files (*)")
            if not src:
                return
            is_r3d = src.lower().endswith(".pth")
            labels_src = ""
            if not os.path.exists(os.path.splitext(src)[0] + ".json"):
                prompt = ("R3D mapping file (idx_to_label + metadata JSON)" if is_r3d
                          else "Labels file for this decoder (idx_to_label JSON)")
                labels_src, _ = QFileDialog.getOpenFileName(
                    self, prompt, "", "JSON (*.json);;All files (*)")
            try:
                # Fresh re-resolution (not the frozen *_LABELS_FILE constants) so
                # the newly imported model's class count shows up immediately,
                # without requiring an app restart.
                if is_r3d:
                    from modules.app_paths import (
                        import_r3d_action_model, r3d_custom_action_paths)
                    n_classes, variant = import_r3d_action_model(src, labels_src)
                    if n_classes == 0:
                        print("⚠️ R3D model imported without a mapping file — it won't "
                              "be usable until one is provided")
                    elif not variant:
                        print("⚠️ R3D mapping has no model_variant — the loader will use "
                              "the 'R3D model variant' dropdown selection")
                    fresh = r3d_custom_action_paths()[1]
                    self._r3d_custom_count = (
                        len(self.load_labels_from_json(fresh)) if os.path.exists(fresh) else 0)
                    select_mode = "r3d_custom_only"
                else:
                    from modules.app_paths import (
                        import_custom_action_model, custom_action_decoder_paths)
                    n_classes = import_custom_action_model(src, labels_src)
                    if n_classes == 0:
                        print("⚠️ Custom action decoder imported without a labels file "
                              "— it won't be usable until one is provided")
                    fresh = custom_action_decoder_paths()[2]
                    self._custom_ov_count = (
                        len(self.load_labels_from_json(fresh)) if os.path.exists(fresh) else 0)
                    select_mode = "custom_only"

                on_action_backend_changed(0)
                idx = self.action_models_combo.findData(select_mode)
                if idx < 0 and n_classes:
                    # The mode isn't offered under the current Backend, so switch to
                    # one that enables the just-imported model — preferring GPU —
                    # instead of leaving the user to hunt through the dropdown:
                    #   • R3D custom needs an R3D backend. "Auto" would *disable* R3D
                    #     on a non-CUDA machine (see pipeline.py), which would make
                    #     r3d_custom_only fail to load — so pick r3d_cuda when an
                    #     NVIDIA GPU is present, else r3d_cpu (slow, but it runs).
                    #   • OpenVINO custom → "Auto" (lists it, and uses the Intel
                    #     GPU / CPU at runtime).
                    if select_mode == "r3d_custom_only":
                        try:
                            from modules.device_utils import detect_best_device
                            has_cuda = detect_best_device(
                                log_fn=lambda *a, **k: None).pytorch_device == "cuda"
                        except Exception:
                            has_cuda = False
                        target_backend = "r3d_cuda" if has_cuda else "r3d_cpu"
                    else:
                        target_backend = "auto"
                    # Setting the combo fires on_action_backend_changed, which
                    # rebuilds the models list, so re-query the index afterward.
                    ab_idx = self.action_backend_combo.findData(target_backend)
                    if ab_idx >= 0:
                        self.action_backend_combo.setCurrentIndex(ab_idx)
                        idx = self.action_models_combo.findData(select_mode)
                if idx >= 0:
                    self.action_models_combo.setCurrentIndex(idx)
            except Exception as e:
                print(f"⚠️ action model import failed: {e}")

        import_action_btn.clicked.connect(_import_action_model)

        self.r3d_model_combo = QComboBox()
        self.r3d_model_combo.addItem("R3D-18 (fastest)", "r3d_18")
        self.r3d_model_combo.addItem("MC3-18 (mixed convolution)", "mc3_18")
        self.r3d_model_combo.addItem("R(2+1)D-18 (most accurate)", "r2plus1d_18")
        current_r3d = advanced_cfg.get("r3d_model", "r3d_18")
        idx_r3d = self.r3d_model_combo.findData(current_r3d)
        self.r3d_model_combo.setCurrentIndex(idx_r3d if idx_r3d >= 0 else 0)

        def on_action_backend_changed(index):
            backend = self.action_backend_combo.currentData()
            self.r3d_model_combo.setEnabled(backend in ("auto", "r3d_cuda", "r3d_cpu"))

            prev_data = self.action_models_combo.currentData()
            self.action_models_combo.blockSignals(True)
            self.action_models_combo.clear()

            if backend in ("openvino",):
                if self._intel_count:
                    self.action_models_combo.addItem(f"Intel Kinetics-400 ({self._intel_count} classes)", "intel_only")
                if self._custom_ov_count:
                    self.action_models_combo.addItem(f"Custom OpenVINO ({self._custom_ov_count} classes)", "custom_only")
                if self._intel_count and self._custom_ov_count:
                    total = self._intel_count + self._custom_ov_count
                    self.action_models_combo.addItem(f"Mixed — both decoders ({total} classes)", "mixed")
            elif backend in ("r3d_cuda", "r3d_cpu"):
                if self._intel_count:
                    self.action_models_combo.addItem(f"R3D Kinetics-400 pretrained ({self._intel_count} classes)", "intel_only")
                if self._r3d_custom_count:
                    self.action_models_combo.addItem(f"R3D fine-tuned ({self._r3d_custom_count} classes)", "r3d_custom_only")
                if self._intel_count and self._r3d_custom_count:
                    total = self._intel_count + self._r3d_custom_count
                    self.action_models_combo.addItem(f"Mixed — both R3D ({total} classes)", "mixed")
            else:
                if self._intel_count:
                    self.action_models_combo.addItem(f"Intel Kinetics-400 ({self._intel_count} classes)", "intel_only")
                if self._custom_ov_count:
                    self.action_models_combo.addItem(f"Custom OpenVINO ({self._custom_ov_count} classes)", "custom_only")
                if self._r3d_custom_count:
                    self.action_models_combo.addItem(f"R3D fine-tuned ({self._r3d_custom_count} classes)", "r3d_custom_only")
                available = sum(1 for c in [self._intel_count, self._custom_ov_count, self._r3d_custom_count] if c > 0)
                if available >= 2:
                    total = self._intel_count + self._custom_ov_count + self._r3d_custom_count
                    self.action_models_combo.addItem(f"Mixed — all models ({total} classes)", "mixed")

            restore_idx = self.action_models_combo.findData(prev_data)
            if restore_idx >= 0:
                self.action_models_combo.setCurrentIndex(restore_idx)
            self.action_models_combo.blockSignals(False)
            self.update_actions_completer()

        self.action_backend_combo.currentIndexChanged.connect(on_action_backend_changed)
        self.action_models_combo.currentIndexChanged.connect(lambda: self.update_actions_completer())
        on_action_backend_changed(0)
        current_action_models = advanced_cfg.get("action_models", "mixed")
        restore_idx = self.action_models_combo.findData(current_action_models)
        if restore_idx >= 0:
            self.action_models_combo.setCurrentIndex(restore_idx)

        action_layout.addRow("Frame skip:", self.sample_rate_spin)
        action_layout.addRow("Backend:", self.action_backend_combo)
        action_layout.addRow("Models:", action_model_widget)
        action_layout.addRow("R3D model variant:", self.r3d_model_combo)

        action_box.setLayout(action_layout)
        advanced_layout.addWidget(action_box, 2, 1)

        # ── Group 4: Bounding Box Visualization ──
        # ── Group 4: Composition Rules ──
        comp_box = QGroupBox("Composition Rules")
        comp_outer = QVBoxLayout()

        comp_info = QLabel(
            "Compose higher-level actions from the spatial relationships between detected objects. "
            "Example: if object A appears inside region B a certain number of times, fire action X. "
            "Each row is one spatial condition; multiple rows with the same Event Name must ALL be "
            "satisfied together (AND logic). "
            "Window = how many seconds of frames to smooth over (reduces flicker). "
            "Persist = how long to keep an object 'alive' after YOLO loses sight of it (handles occlusion). "
            "Saved to composition_rules.yaml next to the application."
        )
        comp_info.setWordWrap(True)
        comp_info.setStyleSheet("color: #888; font-size: 9pt;")
        comp_outer.addWidget(comp_info)

        # Table: Event Name | Label | Source | Region | Min | Max | Window | Persist | [Del]
        # Two kinds of condition share this table. A spatial one is geometry
        # (this class inside that class); a signal one is a threshold on a
        # per-second measurement. They need different fields, and the earlier
        # version had no row shape for the second kind — so signal rules were
        # invisible here and could only be edited as YAML.
        self.COMP_MIN_UNSET = -9999.0
        self.COMP_MAX_UNSET = 9999.0
        self.comp_table = QTableWidget(0, 13)
        self.comp_table.setHorizontalHeaderLabels([
            "On", "Kind", "Event Name", "Display Label",
            "Object / Signal", "Region / Equals",
            "Min", "Max", "Sustain (s)", "Within (s)",
            "Window (s)", "Persist (s)", "",
        ])
        # chr(10) rather than an escape: this block is generated, and a
        # literal backslash-n did not survive the round trip intact.
        self.comp_table.horizontalHeader().setToolTip(chr(10).join([
            "On: untick to keep a rule but stop it running",
            "Kind - Spatial: the object in Object must be inside Region.",
            "       Signal: the measurement in Signal must sit between Min and Max.",
            "Min/Max: counts for Spatial, thresholds for Signal. Left at the",
            "       extreme they mean no bound and are not written out.",
            "Region / Equals: the containing class (Spatial), or a label the",
            "       signal must equal, such as an expression name (Signal).",
            "Sustain: the condition must hold this many seconds in a row (Signal).",
            "Within: accept it if it held this many seconds either side, for",
            "       signals not sampled at the same moments (Signal).",
            "Window: seconds of frames to smooth over (reduces flicker)",
            "Persist: seconds to keep a source alive after it disappears",
        ]))
        self.comp_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.comp_table.horizontalHeader().setStretchLastSection(False)
        for _c, _w in ((0, 34), (1, 78), (2, 140), (3, 140), (4, 150), (5, 120),
                       (6, 70), (7, 70), (8, 80), (9, 80), (10, 80), (11, 80),
                       (12, 32)):
            self.comp_table.setColumnWidth(_c, _w)
        self.comp_table.setMinimumHeight(160)
        self.comp_table.setMaximumHeight(280)
        self.comp_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        comp_outer.addWidget(self.comp_table)

        comp_btn_row = QHBoxLayout()
        comp_add_btn = QPushButton("+ Add Spatial")
        comp_add_btn.setToolTip("Add a condition on object geometry: "
                                "this class inside that class")
        comp_add_signal_btn = QPushButton("+ Add Signal")
        comp_add_signal_btn.setToolTip(
            "Add a condition on a per-second measurement, e.g. "
            "vocal_density_pct or waveform_peak_density. Needs no detections.")
        comp_save_btn = QPushButton("Save Rules")
        comp_save_btn.setToolTip("Save composition rules to composition_rules.yaml")
        comp_btn_row.addWidget(comp_add_btn)
        comp_btn_row.addWidget(comp_add_signal_btn)
        comp_btn_row.addStretch()
        # Run lives here, beside the editor, because this is where rules are
        # changed. Applying them is seconds against what is already cached; the
        # alternative was a full pipeline run just to see a rule edit.
        comp_btn_row.addWidget(self._make_analyze_button(
            "composition", "Run Rules",
            "Run the ticked rules over every video in the list.\n\n"
            "Fetches whatever they read that is missing: a rule naming an\n"
            "object class the cache does not have starts a detection pass for\n"
            "the classes the rules name, and a signal rule measures the audio\n"
            "(cached, so only the first run pays for it). Anything already\n"
            "there is reused, so a re-run after a threshold edit is seconds.\n\n"
            "Save your rules first. Unticked rules are skipped and cost\n"
            "nothing. Safe to run repeatedly: previous results for these rules\n"
            "are replaced, not stacked."))
        comp_btn_row.addWidget(comp_save_btn)
        comp_outer.addLayout(comp_btn_row)

        comp_box.setLayout(comp_outer)
        advanced_layout.addWidget(comp_box, 3, 0, 1, 2)

        # ---- load existing rules into table ----
        def _comp_load_rules():
            from modules.app_paths import composition_rules_path, user_data_dir
            path = composition_rules_path()
            events = []
            if path:
                try:
                    with open(path, encoding='utf-8') as _f:
                        events = (yaml.safe_load(_f) or {}).get('events', [])
                except Exception:
                    pass
            self.comp_table.setRowCount(0)
            # Only an event with neither kind of condition is unrepresentable
            # now. Kept aside and written back untouched on save, because the
            # table rebuilds this file from its own rows and would otherwise
            # delete it silently.
            self._comp_passthrough = [ev for ev in events
                                      if not ev.get('rules') and not ev.get('signals')]
            # The whole original entry, by name. Saving rebuilds this file from
            # the table's rows, so any field without a column is lost — which is
            # how `min_duration_secs` and `ignore_edges_secs` disappeared the
            # first time somebody pressed Save after they were added. Keeping
            # the original and overwriting only what the table owns means a
            # field added later survives without needing a column first.
            self._comp_original = {str(ev.get('name', '')): dict(ev)
                                   for ev in events if ev.get('name')}
            for ev in events:
                common = dict(
                    ev_name=ev.get('name', ''),
                    ev_label=ev.get('label', ev.get('name', '')),
                    window=ev.get('window_secs', 0.75),
                    persist=ev.get('persist_secs', 0.5),
                    enabled=bool(ev.get('enabled', True)),
                )
                for rule in ev.get('rules', []) or []:
                    _comp_add_table_row(
                        kind='Spatial',
                        source=rule.get('source', ''),
                        region=rule.get('region', ''),
                        min_c=rule.get('min_count', 1),
                        max_c=rule.get('max_count', 999),
                        **common)
                for cond in ev.get('signals', []) or []:
                    equals = cond.get('equals')
                    if equals is None and cond.get('any_of'):
                        # any_of has no column of its own; showing the first
                        # value would quietly drop the rest, so the row is
                        # rendered read-only-ish by leaving Equals blank and the
                        # condition is preserved through passthrough instead.
                        equals = None
                    _comp_add_table_row(
                        kind='Signal',
                        source=str(cond.get('signal', '')),
                        region='' if equals is None else str(equals),
                        min_c=cond.get('min'),
                        max_c=cond.get('max'),
                        sustain=cond.get('sustained_secs', 0),
                        within=cond.get('within_secs', 0),
                        **common)
            if self._comp_passthrough:
                names = ', '.join(str(ev.get('name', '?'))
                                  for ev in self._comp_passthrough)
                # print(), not append_log(): this runs from __init__, before the
                # log pane exists, and calling it there took the whole
                # application down before its first window.
                print(f"Composition rules: {len(self._comp_passthrough)} "
                      f"rule(s) with no conditions ({names}); preserved on save.")


        def _comp_kind_of(row):
            combo = self.comp_table.cellWidget(row, 1)
            return (combo.currentText() if combo else "Spatial").strip()

        def _comp_apply_kind(row):
            """Grey the cells the chosen kind does not use.

            Left editable they would look like fields that simply had not been
            filled in, and a spatial rule carrying a Sustain value that is
            silently dropped on save is worse than one that never offered it.
            """
            signal = _comp_kind_of(row) == "Signal"
            for col in (8, 9):                       # Sustain / Within
                w = self.comp_table.cellWidget(row, col)
                if w:
                    w.setEnabled(signal)
            for col in (10, 11):                     # Window / Persist
                w = self.comp_table.cellWidget(row, col)
                if w:
                    w.setEnabled(True)
            head = self.comp_table.horizontalHeaderItem(4)
            if head:
                head.setText("Object / Signal")

        def _comp_add_table_row(ev_name='', ev_label='', source='', region='',
                                min_c=None, max_c=None, window=0.75, persist=0.5,
                                enabled=True, kind='Spatial',
                                sustain=0, within=0):
            r = self.comp_table.rowCount()
            self.comp_table.insertRow(r)

            # Enabled is a property of the *event*, and one event can occupy
            # several rows. Toggling any of them moves the rest, so the table
            # cannot be left saying a rule is both on and off.
            on_chk = QCheckBox()
            on_chk.setChecked(bool(enabled))
            on_chk.setToolTip("Run this rule. Unticked keeps it in the file "
                              "but stops it matching.")
            def _sync(state, box=on_chk):
                row = next((i for i in range(self.comp_table.rowCount())
                            if self.comp_table.cellWidget(i, 0) is box), None)
                if row is None:
                    return
                item = self.comp_table.item(row, 2)
                name = item.text().strip() if item else ''
                if not name:
                    return
                for i in range(self.comp_table.rowCount()):
                    other = self.comp_table.cellWidget(i, 0)
                    twin = self.comp_table.item(i, 2)
                    if other is None or other is box or twin is None:
                        continue
                    if twin.text().strip() == name and other.isChecked() != box.isChecked():
                        other.blockSignals(True)
                        other.setChecked(box.isChecked())
                        other.blockSignals(False)
            on_chk.stateChanged.connect(_sync)
            # In the cell directly, not inside a centring wrapper: every lookup
            # finds it with cellWidget(row, 0), and a wrapper would return the
            # wrapper instead.
            self.comp_table.setCellWidget(r, 0, on_chk)

            kind_combo = QComboBox()
            kind_combo.addItems(["Spatial", "Signal"])
            kind_combo.setCurrentText("Signal" if str(kind) == "Signal" else "Spatial")
            def _on_kind(_i, box=kind_combo):
                row = next((i for i in range(self.comp_table.rowCount())
                            if self.comp_table.cellWidget(i, 1) is box), None)
                if row is not None:
                    _comp_apply_kind(row)
            kind_combo.currentIndexChanged.connect(_on_kind)
            self.comp_table.setCellWidget(r, 1, kind_combo)

            self.comp_table.setItem(r, 2, QTableWidgetItem(ev_name))
            self.comp_table.setItem(r, 3, QTableWidgetItem(ev_label))
            self.comp_table.setItem(r, 4, QTableWidgetItem(source))
            self.comp_table.setItem(r, 5, QTableWidgetItem(region))

            # One widget type for both kinds. Counts are whole numbers and
            # thresholds are not, and a spin box per kind would have to be
            # rebuilt every time the Kind cell changed; the save path rounds
            # counts back to integers instead.
            min_spin = QDoubleSpinBox()
            min_spin.setDecimals(2)
            min_spin.setRange(self.COMP_MIN_UNSET, self.COMP_MAX_UNSET)
            min_spin.setValue(self.COMP_MIN_UNSET if min_c is None else float(min_c))
            min_spin.setToolTip("At the minimum this means 'no lower bound' "
                                "and is not written to the file.")
            self.comp_table.setCellWidget(r, 6, min_spin)

            max_spin = QDoubleSpinBox()
            max_spin.setDecimals(2)
            max_spin.setRange(self.COMP_MIN_UNSET, self.COMP_MAX_UNSET)
            max_spin.setValue(self.COMP_MAX_UNSET if max_c is None else float(max_c))
            max_spin.setToolTip("At the maximum this means 'no upper bound' "
                                "and is not written to the file.")
            self.comp_table.setCellWidget(r, 7, max_spin)

            sus_spin = QSpinBox()
            sus_spin.setRange(0, 600)
            sus_spin.setValue(int(sustain or 0))
            sus_spin.setToolTip("0 = not required")
            self.comp_table.setCellWidget(r, 8, sus_spin)

            win_secs_spin = QSpinBox()
            win_secs_spin.setRange(0, 600)
            win_secs_spin.setValue(int(within or 0))
            win_secs_spin.setToolTip("0 = must coincide exactly")
            self.comp_table.setCellWidget(r, 9, win_secs_spin)

            win_spin = QDoubleSpinBox()
            win_spin.setRange(0.0, 10.0)
            win_spin.setSingleStep(0.25)
            win_spin.setValue(float(window))
            self.comp_table.setCellWidget(r, 10, win_spin)

            per_spin = QDoubleSpinBox()
            per_spin.setRange(0.0, 10.0)
            per_spin.setSingleStep(0.25)
            per_spin.setValue(float(persist))
            self.comp_table.setCellWidget(r, 11, per_spin)

            del_btn = QPushButton()
            del_btn.setIcon(_ui_icons.cross())
            del_btn.setToolTip("Remove this condition")
            del_btn.setFixedWidth(28)
            del_btn.setFlat(True)
            del_btn.setStyleSheet("border: none;")
            def _make_del(btn):
                def _del():
                    for i in range(self.comp_table.rowCount()):
                        if self.comp_table.cellWidget(i, 12) is btn:
                            self.comp_table.removeRow(i)
                            return
                return _del
            del_btn.clicked.connect(_make_del(del_btn))
            self.comp_table.setCellWidget(r, 12, del_btn)

            _comp_apply_kind(r)


        def _comp_collect_events():
            """The table as the rules file's ``events`` list."""
            # Group rows by event name (preserving order of first appearance).
            # A row is one *condition*; several rows can belong to one event,
            # and they may now be of either kind, so an event can carry spatial
            # and signal conditions together.
            events_ordered = []
            events_map = {}
            for r in range(self.comp_table.rowCount()):
                def _txt(c):
                    it = self.comp_table.item(r, c)
                    return it.text().strip() if it else ''
                kind     = _comp_kind_of(r)
                ev_name  = _txt(2)
                ev_label = _txt(3)
                first    = _txt(4)
                second   = _txt(5)
                min_v    = self.comp_table.cellWidget(r, 6).value()
                max_v    = self.comp_table.cellWidget(r, 7).value()
                sustain  = self.comp_table.cellWidget(r, 8).value()
                within   = self.comp_table.cellWidget(r, 9).value()
                window   = self.comp_table.cellWidget(r, 10).value()
                persist  = self.comp_table.cellWidget(r, 11).value()
                on_box   = self.comp_table.cellWidget(r, 0)
                enabled  = True if on_box is None else bool(on_box.isChecked())

                if not ev_name or not first:
                    continue
                if kind == "Spatial" and not second:
                    continue

                if ev_name not in events_map:
                    # Start from what was read, so fields this table has no
                    # column for are carried across untouched, then overwrite
                    # the ones it does own. `rules` and `signals` are rebuilt
                    # from the rows below and must not survive from the
                    # original, or a deleted condition would come back.
                    entry = dict(getattr(self, '_comp_original', {}).get(ev_name, {}))
                    entry.pop('rules', None)
                    entry.pop('signals', None)
                    entry.update({
                        'name': ev_name,
                        'label': ev_label or ev_name,
                        'enabled': enabled,
                        'window_secs': window,
                        'persist_secs': persist,
                    })
                    events_map[ev_name] = entry
                    events_ordered.append(entry)
                entry = events_map[ev_name]

                if kind == "Signal":
                    cond = {'signal': first}
                    # The extremes mean "no bound". Writing them out would turn
                    # an open-ended condition into one clamped at an arbitrary
                    # number that happens to be this widget's range.
                    if min_v > self.COMP_MIN_UNSET:
                        cond['min'] = round(float(min_v), 4)
                    if max_v < self.COMP_MAX_UNSET:
                        cond['max'] = round(float(max_v), 4)
                    if second:
                        cond['equals'] = second
                    if int(sustain) > 0:
                        cond['sustained_secs'] = int(sustain)
                    if int(within) > 0:
                        cond['within_secs'] = int(within)
                    entry.setdefault('signals', []).append(cond)
                else:
                    entry.setdefault('rules', []).append({
                        'source': first,
                        'region': second,
                        # Counts are whole numbers; the shared spin box carries
                        # decimals for the signal case.
                        'min_count': int(round(min_v)) if min_v > self.COMP_MIN_UNSET else 0,
                        'max_count': int(round(max_v)) if max_v < self.COMP_MAX_UNSET else 999,
                    })

            # Anything the table still cannot represent — an event with neither
            # kind of condition — is written back as it was read. The table
            # rebuilds this file from its own rows, so without this such an
            # entry would be deleted with nothing on screen to show it going.
            events_ordered.extend(getattr(self, '_comp_passthrough', []) or [])
            return events_ordered

        def _comp_save_rules(quiet=False):
            """Write the table to the rules file. Returns True if it wrote.

            ``quiet`` skips the write when nothing changed and reports only
            failures — for the automatic saves (on Run, on close), where a log
            line per close is noise and a silent loss of a ticked box is not.
            """
            from modules.app_paths import user_data_dir
            import os as _os
            out = {'events': _comp_collect_events()}
            if quiet and out == getattr(self, '_comp_saved_state', None):
                return False
            save_path = _os.path.join(user_data_dir(), 'composition_rules.yaml')
            try:
                with open(save_path, 'w', encoding='utf-8') as _f:
                    yaml.dump(out, _f, allow_unicode=True, sort_keys=False, default_flow_style=False)
                self._comp_saved_state = out
                if not quiet:
                    self.append_log(f"✅ Composition rules saved → {save_path}")
                return True
            except Exception as _e:
                # Never quiet: this is the one outcome the user has to know
                # about, and on close it is their last chance to.
                self.append_log(f"❌ Could not save composition rules: {_e}")
                return False


        comp_add_btn.clicked.connect(
            lambda: _comp_add_table_row(kind='Spatial'))
        comp_add_signal_btn.clicked.connect(
            lambda: _comp_add_table_row(kind='Signal'))
        comp_save_btn.clicked.connect(_comp_save_rules)
        _comp_load_rules()

        # ── Group 5: Bounding Box Visualization ──
        bbox_box = QGroupBox("Bounding Box Visualization")
        bbox_layout = QVBoxLayout()

        info_label = QLabel("ℹ️ Enable bounding boxes, creates new file with extension _annotated.mp4 for debugging")
        info_label.setStyleSheet("color: #666; font-size: 9pt; font-style: italic;")
        bbox_layout.addWidget(info_label)

        self.bbox_objects_chk = QCheckBox("Draw bounding boxes for object detection")
        self.bbox_objects_chk.setChecked(visualization_cfg.get("draw_object_boxes", False))
        self.bbox_objects_chk.setToolTip("Visualize detected objects with labeled bounding boxes")
        bbox_layout.addWidget(self.bbox_objects_chk)

        self.bbox_actions_chk = QCheckBox("Draw labels for action recognition")
        self.bbox_actions_chk.setChecked(visualization_cfg.get("draw_action_labels", False))
        self.bbox_actions_chk.setToolTip("Display detected action names on frames")
        bbox_layout.addWidget(self.bbox_actions_chk)

        # On by default: it costs one frame grab per kept segment and answers the
        # question every user asks first — why these moments and not others.
        self.why_report_chk = QCheckBox("Write a highlight report")
        self.why_report_chk.setChecked(
            visualization_cfg.get("write_highlight_report", True))
        self.why_report_chk.setToolTip(
            "Writes <output>_why.html next to the highlight: every kept segment with\n"
            "its score breakdown, the objects and actions that triggered it, and the\n"
            "moments that scored well but were left out.\n\n"
            "One self-contained file with thumbnails embedded — openable in any\n"
            "browser and sendable to a client. A matching .json holds the same data.")
        bbox_layout.addWidget(self.why_report_chk)

        bbox_box.setLayout(bbox_layout)
        advanced_layout.addWidget(bbox_box, 1, 1)

        # ── Group: Video Output ──
        # How the final highlight is re-encoded. CPU (libx265) is VR-safe but slow;
        # GPU is fast but its HEVC may not play in some VR players. Placed at the top
        # of Advanced so it's easy to find when a VR player rejects a render.
        output_box = QGroupBox("Video Output")
        output_layout = QFormLayout()
        self.render_mode_combo = QComboBox()
        self.render_mode_combo.addItem("CPU x265 (VR-safe, slow)", "cpu")
        self.render_mode_combo.addItem("GPU (fast, may break VR)", "gpu")
        self.render_mode_combo.setToolTip(
            "How the highlight video is encoded:\n"
            "CPU x265 — re-encode on the CPU with libx265 (HEVC), matching how VR\n"
            "   sources are authored. VR-safe, but slow at 6K.\n"
            "GPU — re-encode with the hardware encoder. Fast, but the HEVC output\n"
            "   may not play in VR players like HereSphere."
        )
        _saved_render_mode = highlights_cfg.get("render_mode", "cpu")
        _rm_idx = self.render_mode_combo.findData(_saved_render_mode)
        if _rm_idx >= 0:
            self.render_mode_combo.setCurrentIndex(_rm_idx)
        output_layout.addRow("Cut / encode:", self.render_mode_combo)
        output_box.setLayout(output_layout)
        advanced_layout.addWidget(output_box, 0, 0, 1, 2)

        # Equal column widths; let the row below the composition table absorb slack
        advanced_layout.setColumnStretch(0, 1)
        advanced_layout.setColumnStretch(1, 1)
        advanced_layout.setRowStretch(4, 1)

        advanced_scroll = QScrollArea()
        advanced_scroll.setWidgetResizable(True)
        _adv_container = QWidget()
        _adv_container.setLayout(advanced_layout)
        advanced_scroll.setWidget(_adv_container)
        advanced_tab.setLayout(QVBoxLayout())
        advanced_tab.layout().setContentsMargins(0, 0, 0, 0)
        advanced_tab.layout().addWidget(advanced_scroll)
        tabs.addTab(self._scrollable(advanced_tab), "Advanced")

        content_splitter = QSplitter(Qt.Vertical)
        # A floor, not the old behaviour. Wrapping the pages in scroll areas
        # dropped the tab widget's minimum height to almost nothing, which is
        # what let the window finally shrink — but it also left nothing
        # resisting the splitter, so the tabs collapsed to a couple of rows
        # while the empty log pane kept its share. Small enough that the window
        # still fits a 1080p screen, big enough to show a form without folding.
        tabs.setMinimumHeight(280)
        content_splitter.addWidget(tabs)
        self.content_splitter = content_splitter
        layout.addWidget(content_splitter)

        # --- Tab 4: LLM Chat ---
        llm_tab = QWidget()
        llm_layout = QVBoxLayout()
        self.llm_chat = LLMChatWidget(parent=self)
        llm_layout.addWidget(self.llm_chat)
        llm_tab.setLayout(llm_layout)
        tabs.addTab(self._scrollable(llm_tab), "LLM Chat")

        # --- Tab 5: Avoid ---
        avoid_tab = QWidget()
        avoid_layout = QVBoxLayout()

        avoid_group = QGroupBox("Avoid People")
        avoid_group_layout = QVBoxLayout()

        self.avoid_face_recognition_chk = QCheckBox("Enable face recognition")
        self.avoid_face_recognition_chk.setChecked(self.config_data.get("avoid", {}).get("face_recognition_enabled", False))
        self.avoid_face_recognition_chk.setToolTip(
            "When enabled, the pipeline runs face recognition to locate avoided people and skip or crop them out.\n"
            "Disable to skip the face-recognition step entirely (faster, no avoid enforcement)."
        )
        avoid_group_layout.addWidget(self.avoid_face_recognition_chk)

        avoid_info = QLabel(
            "People you name in the Timeline Viewer (right-click a face → Name) "
            "show up here. Tick someone to exclude them from generated highlights."
        )
        avoid_info.setWordWrap(True)
        avoid_info.setStyleSheet("color: #666; font-size: 9pt;")
        avoid_group_layout.addWidget(avoid_info)
        avoid_method_row = QHBoxLayout()
        avoid_method_row.addWidget(QLabel("When found:"))
        self.avoid_method_combo = QComboBox()
        self.avoid_method_combo.addItem("Skip those moments", "skip")
        self.avoid_method_combo.addItem("Crop them out (experimental)", "crop")
        self.avoid_method_combo.currentIndexChanged.connect(
            lambda: setattr(self, "_avoid_method", self.avoid_method_combo.currentData()))
        avoid_method_row.addWidget(self.avoid_method_combo)
        avoid_method_row.addStretch()
        avoid_group_layout.addLayout(avoid_method_row)

        avoid_row = QHBoxLayout()
        self.avoid_refresh_btn = QPushButton("🔄 Refresh from face database")
        self.avoid_refresh_btn.clicked.connect(self.refresh_avoid_list)
        avoid_row.addWidget(self.avoid_refresh_btn)
        self.avoid_scan_btn = QPushButton("🔍 Scan video for faces")
        self.avoid_scan_btn.setToolTip("Run face recognition over the first video in the list "
                                       "to collect everyone who appears, then tick who to avoid.")
        self.avoid_scan_btn.clicked.connect(self._on_scan_faces)
        avoid_row.addWidget(self.avoid_scan_btn)
        self.avoid_count_label = QLabel("")
        self.avoid_count_label.setStyleSheet("color: #2f81f7; font-weight: bold;")
        avoid_row.addWidget(self.avoid_count_label)
        avoid_row.addStretch()
        avoid_group_layout.addLayout(avoid_row)
        self.avoid_clear_btn = QPushButton("🗑 Clear faces")
        self.avoid_clear_btn.setToolTip("Remove scanned faces from the bank (keeps named/avoided people).")
        self.avoid_clear_btn.clicked.connect(self._on_clear_faces)
        avoid_row.addWidget(self.avoid_clear_btn)

        self.avoid_scroll = QScrollArea()
        self.avoid_scroll.setWidgetResizable(True)
        self.avoid_list_container = QWidget()
        self.avoid_list_layout = QVBoxLayout(self.avoid_list_container)
        self.avoid_list_layout.addStretch()
        self.avoid_scroll.setWidget(self.avoid_list_container)
        avoid_group_layout.addWidget(self.avoid_scroll)

        avoid_group.setLayout(avoid_group_layout)
        avoid_layout.addWidget(avoid_group, 1)
        avoid_tab.setLayout(avoid_layout)
        tabs.addTab(self._scrollable(avoid_tab), "Avoid")

        # --- Tab: About & Contact ---
        tabs.addTab(self._scrollable(self._build_about_tab()), "About")

        # --- Tab: About & Contact ---
        tabs.addTab(self._build_about_tab(), "ℹ️ About")

        # Defer first populate until after __init__ finishes (so log_output exists)
        QTimer.singleShot(0, self.refresh_avoid_list)
        # Let the window finish painting first — the check is never urgent, and
        # its own throttle means most launches do no network at all.
        QTimer.singleShot(3000, self._start_update_check)
        # Clear whatever the last update displaced. It can only be deleted now,
        # on a launch after the process that had those files open has exited.
        QTimer.singleShot(1500, self._sweep_updated_files)

        # --- Run / Cancel Controls ---
        ctrl_layout = QHBoxLayout()
        self.keep_temp_chk = QPushButton("Keep temp clips: ON" if highlights_cfg.get("keep_temp", False) else "Keep temp clips: OFF")
        self.keep_temp_chk.setCheckable(True)
        self.keep_temp_chk.setChecked(highlights_cfg.get("keep_temp", False))
        self.keep_temp_chk.clicked.connect(lambda: self.keep_temp_chk.setText(
            "Keep temp clips: ON" if self.keep_temp_chk.isChecked() else "Keep temp clips: OFF"))

        self.timeline_btn = QPushButton("Timeline Viewer")
        self.timeline_btn.setStyleSheet("QPushButton { background-color: #2f81f7; color: white; font-weight: bold; padding: 8px; }")
        self.timeline_btn.clicked.connect(self.open_timeline_viewer)

        self.why_report_btn = QPushButton("Highlight Report")
        self.why_report_btn.setToolTip(
            "Open the report explaining why each highlight was chosen.\n"
            "Written next to the highlight on every run (Advanced tab toggles it).")
        self.why_report_btn.clicked.connect(self.open_why_report)
        # "AI Summary" writes a few plain-language sentences into that same
        # report. Separate button because it costs a model run and tens of
        # seconds — the report itself must stay instant.
        self.ai_summary_btn = QPushButton("AI Summary")
        self.ai_summary_btn.setToolTip(
            "Add a short plain-language summary to the Highlight Report:\n"
            "what shaped this cut, and the one change most likely to improve it.\n"
            "Runs your local model, so it takes a moment. The report's findings\n"
            "are always there without it.")
        self.ai_summary_btn.clicked.connect(self.write_ai_summary)

        # The wheel keeps the choices that most users never touch out of sight.
        # A drawn icon, not a "⚙" glyph: the packaged build has no guarantee of
        # a font carrying it, and a blank button is what you get when it does
        # not.
        self.ai_summary_opts_btn = QPushButton()
        self.ai_summary_opts_btn.setIcon(_ui_icons.gear())
        self.ai_summary_opts_btn.setFixedWidth(28)
        self.ai_summary_opts_btn.setToolTip("Summary options — ask a question, "
                                            "discuss in chat, choose the model")
        self.ai_summary_opts_btn.clicked.connect(self.show_ai_summary_menu)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet("QPushButton:enabled { background-color: #ff4444; color: white; font-weight: bold; }")
        self.cancel_btn.clicked.connect(self.cancel_pipeline)

        # Scores and reports without encoding. Detection is cached, so this is
        # seconds once a video has been analysed — cheap enough to use as the
        # normal way of trying a setting.
        self.report_only_btn = QPushButton("Report Only")
        self.report_only_btn.setToolTip(
            "Re-score with the current settings and write the Highlight Report,\n"
            "without rendering a video. Fast on a video already analysed.")
        self.report_only_btn.clicked.connect(lambda: self.run_pipeline(report_only=True))

        self.run_btn = QPushButton("Run Highlighter")
        self.run_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        self.run_btn.clicked.connect(self.toggle_run)

        ctrl_layout.addWidget(self.cancel_btn)
        ctrl_layout.addWidget(self.keep_temp_chk)
        ctrl_layout.addWidget(self.timeline_btn)
        ctrl_layout.addWidget(self.why_report_btn)
        ctrl_layout.addWidget(self.ai_summary_btn)
        ctrl_layout.addWidget(self.ai_summary_opts_btn)
        self.debug_console_chk = QCheckBox("Debug log")
        self.debug_console_chk.setChecked(debug_console.is_console_visible())
        self.debug_console_chk.setToolTip(
            "Open a live window mirroring all app output\n"
            "(recent output is replayed, so it works after an error too).\n"
            f"Everything is always saved to:\n{debug_console.log_file_path()}"
        )
        self.debug_console_chk.toggled.connect(debug_console.set_console_visible)
        debug_console.register_checkbox(self.debug_console_chk)
        ctrl_layout.addWidget(self.debug_console_chk)
        self.session_analyzed_count = 0
        self.analyzed_counter_label = QLabel()
        self.analyzed_counter_label.setStyleSheet("color: #2196F3; font-weight: bold;")
        self.analyzed_counter_label.setToolTip(
            "Videos successfully analyzed by the pipeline.\n"
            f"Lifetime total persists in:\n{analysis_stats.stats_path()}"
        )
        self.update_analyzed_counter()
        ctrl_layout.addWidget(self.analyzed_counter_label)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.report_only_btn)
        ctrl_layout.addWidget(self.run_btn)
        layout.addLayout(ctrl_layout)

        # --- Log view (inside splitter) ---
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(80)
        self.log_output.setStyleSheet("QTextEdit { font-family: 'Courier New', monospace; font-size: 9pt; }")
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.addWidget(QLabel("Log Output:"))
        log_layout.addWidget(self.log_output)
        content_splitter.addWidget(log_widget)
        content_splitter.setStretchFactor(0, 3)
        content_splitter.setStretchFactor(1, 1)
        # Sized against the splitter's own height, not the window's. Asking for
        # window-height-derived sizes here requests more than the splitter
        # actually receives (the input list and time range are above it), and
        # QSplitter then scales BOTH panes down proportionally — which is what
        # squeezed the tabs while the log kept its 80px minimum.
        QTimer.singleShot(0, self._balance_content_splitter)

        self.setLayout(layout)

        self.setup_label_completers()
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_worker_status)

        # Load download time range settings (AFTER all widgets are created)
        download_cfg = self.config_data.get("download", {})
        self.download_start_input.setValue(download_cfg.get("time_range_start", 0))
        self.download_end_input.setValue(download_cfg.get("time_range_end", 300))

        # Restore the download mode. Fall back to the old two-checkbox keys so
        # existing configs keep working: use_same_time_range -> "same",
        # download_full -> "full", else "specific".
        mode = download_cfg.get("download_mode")
        if mode is None:
            if download_cfg.get("use_same_time_range", False):
                mode = "same"
            elif download_cfg.get("download_full", False):
                mode = "full"
            else:
                mode = "full"
        idx = self.download_mode_combo.findData(mode)
        self.download_mode_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.on_download_mode_changed()  # sync visibility

        # Restore the after-download processing mode, with fall-back from the old
        # keys: immediate_processing -> "immediate", auto_process -> "batch",
        # else "none".
        pmode = download_cfg.get("process_mode")
        if pmode is None:
            if download_cfg.get("immediate_processing", False):
                pmode = "immediate"
            elif download_cfg.get("auto_process", False):
                pmode = "batch"
            else:
                pmode = "none"
        pidx = self.process_mode_combo.findData(pmode)
        self.process_mode_combo.setCurrentIndex(pidx if pidx >= 0 else 0)
        self.on_process_mode_changed()  # sync spinner enabled

    # --- About / Contact tab ---
    @staticmethod
    def _scrollable(page):
        """Wrap a tab page so it can shrink, and scroll instead of clipping.

        QTabWidget takes its minimum height from its tallest page, so one big
        tab set the floor for the entire window — 1049px, more than a 1080p
        screen has once the taskbar is accounted for. The window then could not
        shrink to fit, and the row of buttons at the bottom ended up under the
        taskbar, which is where this started.
        """
        from PySide6.QtWidgets import QScrollArea

        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.NoFrame)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        area.setWidget(page)
        return area

    def _balance_content_splitter(self):
        """Give the log a fixed slice and the tabs everything else.

        Runs after the first layout pass, when the splitter knows how tall it
        actually is. The log is a status pane — it wants a readable few lines,
        not a proportional share of the window.
        """
        splitter = getattr(self, "content_splitter", None)
        if splitter is None:
            return
        available = splitter.height()
        log_height = 150 if available >= 460 else 110
        splitter.setSizes([max(280, available - log_height), log_height])

    # --- Update notice ---
    def _build_update_banner(self):
        """The hidden-by-default "a newer version exists" strip.

        Built once at startup and only ever shown/hidden, so the check that
        fills it in never has to construct widgets from its own thread.
        """
        banner = QWidget()
        banner.setVisible(False)
        banner.setStyleSheet(
            "QWidget { background: #2d4a63; border-radius: 4px; }"
            "QLabel { color: #e8f1f8; background: transparent; }"
        )
        row = QHBoxLayout()
        row.setContentsMargins(10, 6, 6, 6)
        row.setSpacing(8)

        self.update_label = QLabel()
        self.update_label.setWordWrap(True)
        row.addWidget(self.update_label, 1)

        # Shown only while an install is running.
        self.update_progress = QProgressBar()
        self.update_progress.setVisible(False)
        self.update_progress.setMaximumWidth(220)
        row.addWidget(self.update_progress)

        self.update_install_btn = QPushButton("Download and install")
        self.update_install_btn.clicked.connect(self._install_update)
        self.update_install_btn.setVisible(False)
        row.addWidget(self.update_install_btn)

        self.update_get_btn = QPushButton("Get it")
        self.update_get_btn.clicked.connect(self._open_update_download)
        row.addWidget(self.update_get_btn)

        self.update_skip_btn = QPushButton("Skip this version")
        self.update_skip_btn.clicked.connect(self._skip_update)
        row.addWidget(self.update_skip_btn)

        self.update_close_btn = QPushButton("✕")
        self.update_close_btn.setFixedWidth(28)
        self.update_close_btn.setToolTip("Hide until the next check")
        self.update_close_btn.clicked.connect(
            lambda: self.update_banner.setVisible(False))
        row.addWidget(self.update_close_btn)

        banner.setLayout(row)
        return banner

    def _start_update_check(self, force=False):
        """Kick off the manifest check in the background.

        Deliberately after the window is up: startup must not wait on the
        network, and a user who never sees a newer build should never know
        this ran.
        """
        self.update_worker = UpdateCheckWorker(force=force, parent=self)
        self.update_worker.found.connect(self._on_update_available)
        self.update_worker.nothing.connect(self._on_update_check_quiet)
        self.update_worker.start()

    def _on_update_available(self, info):
        """Show the banner. Runs on the GUI thread (queued signal)."""
        self._pending_update = info
        text = f"<b>{info.headline}</b>"
        if info.notes:
            text += f"<br>{info.notes}"
        self.update_label.setText(text)

        # Installing in place is only offered to a packaged build. From source
        # the "install root" is the git checkout, and an update would overwrite
        # working files with release ones — so a dev build gets the download
        # link like any release published before the updater existed.
        can_install = bool(info.can_self_install) and getattr(sys, "frozen", False)
        self.update_install_btn.setVisible(can_install)
        self.update_get_btn.setVisible(not can_install)

        self.update_banner.setVisible(True)
        print(f"update_check: {info.version} available (running {__version__})"
              f"{' [self-install]' if can_install else ''}")

    def _sweep_updated_files(self):
        from modules import update_apply

        try:
            freed = update_apply.sweep_old(update_apply.install_root())
        except Exception as e:
            print(f"update_apply: sweep failed ({e})")
            return
        if freed:
            print(f"update_apply: reclaimed {freed / (1024 ** 2):.1f} MB "
                  "from the previous update")

    def _install_update(self):
        """Download and apply the pending release."""
        info = getattr(self, "_pending_update", None)
        if not info or not info.manifest_url:
            return
        from modules import update_apply

        self.update_install_btn.setEnabled(False)
        self.update_skip_btn.setVisible(False)
        self.update_close_btn.setVisible(False)
        self.update_progress.setVisible(True)
        self.update_progress.setRange(0, 0)     # indeterminate until sizes known
        self.update_label.setText("<b>Preparing update…</b>")

        self.update_installer = UpdateInstallWorker(
            info.manifest_url, update_apply.install_root(), parent=self)
        self.update_installer.progress.connect(self._on_install_progress)
        self.update_installer.finished_with.connect(self._on_install_finished)
        self.update_installer.start()

    def _on_install_progress(self, phase, done, total, detail):
        from modules import update_install

        if phase == update_install.DOWNLOADING and total:
            self.update_progress.setRange(0, total)
            self.update_progress.setValue(done)
            mb_done, mb_total = done / (1024 ** 2), total / (1024 ** 2)
            self.update_label.setText(
                f"<b>Downloading {mb_done:.1f} / {mb_total:.1f} MB</b><br>{detail}")
        else:
            self.update_progress.setRange(0, 0)
            self.update_label.setText(f"<b>{detail or phase}</b>")

    def _on_install_finished(self, result):
        from PySide6.QtWidgets import QMessageBox

        self.update_progress.setVisible(False)
        self.update_install_btn.setEnabled(True)
        self.update_close_btn.setVisible(True)

        if not result.ok:
            self.update_skip_btn.setVisible(True)
            self.update_label.setText(f"<b>{result.message}</b>")
            self.append_log(f"⚠️ Update: {result.message}")
            return

        self.update_label.setText(f"<b>{result.message}</b>")
        self.append_log(f"✅ Update: {result.message}")
        if not result.restart_required:
            return

        answer = QMessageBox.question(
            self, "Restart now?",
            f"{result.message}\n\nRestart Video Highlighter now?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if answer == QMessageBox.Yes:
            self._restart_for_update()

    def _restart_for_update(self):
        """Relaunch the (now updated) app and quit this process.

        The new files are already in place; this process is still running the
        copies that were moved aside, which is why a restart is what actually
        switches versions. The displaced files are swept on the next launch,
        once nothing holds them open.
        """
        import subprocess
        from modules import update_apply

        try:
            subprocess.Popen(update_apply.relaunch_command(),
                             cwd=update_apply.install_root(), close_fds=True)
        except Exception as e:
            print(f"update_install: could not relaunch ({e})")
            return
        QApplication.quit()

    def _on_update_check_quiet(self, message):
        """Answer an explicit "check now" that turned up nothing."""
        if hasattr(self, "update_status_label"):
            self.update_status_label.setText(message)

    def _open_update_download(self):
        info = getattr(self, "_pending_update", None)
        if not info or not info.download_url:
            return
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl(info.download_url))

    def _skip_update(self):
        info = getattr(self, "_pending_update", None)
        if info:
            from modules import update_check
            update_check.skip_version(info.version)
        self.update_banner.setVisible(False)

    def _build_about_tab(self):
        """A read-only About & Contact panel: version, support links, licensing."""
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer_layout.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        scroll.setWidget(content)

        # Header
        title = QLabel(f"🎬 Video Highlighter ({__edition__})")
        title.setStyleSheet("font-size: 16pt; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QLabel(f"Version {__version__} — free & open source (AGPLv3)")
        subtitle.setStyleSheet("color: #888;")
        subtitle.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(subtitle)

        # --- Updates ---
        from modules import update_check as _update_check

        upd_group = QGroupBox("Updates")
        upd_layout = QVBoxLayout(upd_group)

        upd_auto = QCheckBox("Check for a newer version automatically")
        upd_auto.setChecked(_update_check.is_enabled())
        upd_auto.setToolTip(
            "Once a day, downloads a small text file listing the latest "
            "version. Nothing about you or this computer is sent."
        )
        upd_auto.toggled.connect(_update_check.set_enabled)
        upd_layout.addWidget(upd_auto)

        upd_row = QHBoxLayout()
        upd_now_btn = QPushButton("Check now")
        upd_now_btn.clicked.connect(lambda: self._start_update_check(force=True))
        upd_row.addWidget(upd_now_btn)
        self.update_status_label = QLabel("")
        self.update_status_label.setStyleSheet("color: #888;")
        upd_row.addWidget(self.update_status_label)
        upd_row.addStretch()
        upd_layout.addLayout(upd_row)
        layout.addWidget(upd_group)

        # --- Upgrade to Pro ---
        pro_group = QGroupBox("VideoHighlighter Pro")
        pro_layout = QVBoxLayout(pro_group)
        pro_line = QLabel(
            "You're running the free, open-source edition. "
            "<b>Pro</b> adds faster detection backends and extra features, "
            "and supports continued development.<br>"
            f'👉 <a href="{WEBSITE_URL}">Learn more / Get Pro</a>'
        )
        pro_line.setOpenExternalLinks(True)
        pro_line.setTextInteractionFlags(Qt.TextBrowserInteraction)
        pro_line.setWordWrap(True)
        pro_layout.addWidget(pro_line)
        layout.addWidget(pro_group)

        # --- Contact & support ---
        support_group = QGroupBox("Contact & Support")
        support_layout = QVBoxLayout(support_group)
        intro = QLabel("Need help, found a bug, or have a feature request? Reach us here:")
        intro.setWordWrap(True)
        support_layout.addWidget(intro)

        links = QLabel(
            f'📧 Email: <a href="mailto:{SUPPORT_EMAIL}?subject=VideoHighlighter%20support">{SUPPORT_EMAIL}</a><br>'
            f'💬 Discord: <a href="{DISCORD_URL}">{DISCORD_URL}</a><br>'
            f'🌐 Website: <a href="{WEBSITE_URL}">{WEBSITE_URL}</a><br>'
            f'⭐ Source code: <a href="{REPO_URL}">{REPO_URL}</a>'
        )
        links.setOpenExternalLinks(True)
        links.setTextInteractionFlags(Qt.TextBrowserInteraction)
        links.setWordWrap(True)
        support_layout.addWidget(links)

        tip = QLabel(
            "💡 When reporting a bug, please include your OS and the debug log "
            "(toggle “Debug log” next to Run) — it speeds up diagnosis."
        )
        tip.setStyleSheet("color: #888; font-size: 9pt;")
        tip.setWordWrap(True)
        support_layout.addWidget(tip)
        layout.addWidget(support_group)

        # --- Legal ---
        legal_group = QGroupBox("Legal")
        legal_layout = QVBoxLayout(legal_group)
        legal = QLabel(
            "© 2026 Przemysław Kreft and contributors.<br>"
            "VideoHighlighter is free software licensed under the "
            f'<a href="{REPO_URL}/blob/main/LICENSE">GNU AGPLv3</a>. '
            f'Contributions are accepted under a <a href="{REPO_URL}/blob/main/CLA.md">CLA</a>.<br>'
            "Includes third-party components (e.g. PySide6, FFmpeg) under their "
            "respective licenses."
        )
        legal.setOpenExternalLinks(True)
        legal.setTextInteractionFlags(Qt.TextBrowserInteraction)
        legal.setWordWrap(True)
        legal_layout.addWidget(legal)
        layout.addWidget(legal_group)

        layout.addStretch()
        return outer

    # --- About / Contact tab ---
    def _build_about_tab(self):
        """A read-only About & Contact panel: version, support links, licensing."""
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer_layout.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        scroll.setWidget(content)

        # Header
        title = QLabel(f"🎬 Video Highlighter ({__edition__})")
        title.setStyleSheet("font-size: 16pt; font-weight: bold;")
        layout.addWidget(title)

        subtitle = QLabel(f"Version {__version__} — free & open source (AGPLv3)")
        subtitle.setStyleSheet("color: #888;")
        subtitle.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(subtitle)

        # --- Upgrade to Pro ---
        pro_group = QGroupBox("VideoHighlighter Pro")
        pro_layout = QVBoxLayout(pro_group)
        pro_line = QLabel(
            "You're running the free, open-source edition. "
            "<b>Pro</b> adds faster detection backends and extra features, "
            "and supports continued development.<br>"
            f'👉 <a href="{WEBSITE_URL}">Learn more / Get Pro</a>'
        )
        pro_line.setOpenExternalLinks(True)
        pro_line.setTextInteractionFlags(Qt.TextBrowserInteraction)
        pro_line.setWordWrap(True)
        pro_layout.addWidget(pro_line)
        layout.addWidget(pro_group)

        # --- Contact & support ---
        support_group = QGroupBox("Contact & Support")
        support_layout = QVBoxLayout(support_group)
        intro = QLabel("Need help, found a bug, or have a feature request? Reach us here:")
        intro.setWordWrap(True)
        support_layout.addWidget(intro)

        links = QLabel(
            f'📧 Email: <a href="mailto:{SUPPORT_EMAIL}?subject=VideoHighlighter%20support">{SUPPORT_EMAIL}</a><br>'
            f'💬 Discord: <a href="{DISCORD_URL}">{DISCORD_URL}</a><br>'
            f'🌐 Website: <a href="{WEBSITE_URL}">{WEBSITE_URL}</a><br>'
            f'⭐ Source code: <a href="{REPO_URL}">{REPO_URL}</a>'
        )
        links.setOpenExternalLinks(True)
        links.setTextInteractionFlags(Qt.TextBrowserInteraction)
        links.setWordWrap(True)
        support_layout.addWidget(links)

        tip = QLabel(
            "💡 When reporting a bug, please include your OS and the debug log "
            "(toggle “Debug log” next to Run) — it speeds up diagnosis."
        )
        tip.setStyleSheet("color: #888; font-size: 9pt;")
        tip.setWordWrap(True)
        support_layout.addWidget(tip)
        layout.addWidget(support_group)

        # --- Legal ---
        legal_group = QGroupBox("Legal")
        legal_layout = QVBoxLayout(legal_group)
        legal = QLabel(
            "© 2026 Przemysław Kreft and contributors.<br>"
            "VideoHighlighter is free software licensed under the "
            f'<a href="{REPO_URL}/blob/main/LICENSE">GNU AGPLv3</a>. '
            f'Contributions are accepted under a <a href="{REPO_URL}/blob/main/CLA.md">CLA</a>.<br>'
            "Includes third-party components (e.g. PySide6, FFmpeg) under their "
            "respective licenses."
        )
        legal.setOpenExternalLinks(True)
        legal.setTextInteractionFlags(Qt.TextBrowserInteraction)
        legal.setWordWrap(True)
        legal_layout.addWidget(legal)
        layout.addWidget(legal_group)

        layout.addStretch()
        return outer

    # --- Avoid methods ---
    def _get_face_bank(self):
        """Lazily create / reload the shared face identity bank."""
        try:
            from video_ai_editor.face_identity import FaceIdentityBank
        except ImportError as e:
            if hasattr(self, "log_output"):
                self.append_log(f"⚠️ Face bank unavailable: {e}")
            return None
        if getattr(self, "_face_bank", None) is None:
            self._face_bank = FaceIdentityBank(db_path="./cache/face_db.json")
        else:
            self._face_bank.load()   # pick up names/avoids set in the timeline viewer
        return self._face_bank

    def refresh_avoid_list(self):
        """Rebuild the people rows from the face database."""
        import base64
        from PySide6.QtGui import QPixmap

        # clear existing rows (keep the trailing stretch)
        while self.avoid_list_layout.count() > 1:
            item = self.avoid_list_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        bank = self._get_face_bank()
        if bank is None:
            self.avoid_count_label.setText("face database not available")
            return

        identities = bank.all_identities()
        identities.sort(key=lambda i: (i["name"] is None, -(i.get("count") or 0)))

        named = 0
        for ident in identities:
            r = QWidget()
            rl = QHBoxLayout(r)
            rl.setContentsMargins(4, 2, 4, 2)

            thumb = QLabel()
            thumb.setFixedSize(48, 48)
            if ident.get("thumb"):
                pix = QPixmap()
                pix.loadFromData(base64.b64decode(ident["thumb"]), "JPEG")
                if not pix.isNull():
                    thumb.setPixmap(pix.scaled(48, 48, Qt.KeepAspectRatio,
                                               Qt.SmoothTransformation))
            rl.addWidget(thumb)

            display = ident["name"] or f"Person {ident['id'][:8]}"
            if ident["name"]:
                named += 1
            name_label = QLabel(
                f"<b>{display}</b><br>"
                f"<span style='color:#888;font-size:8pt;'>seen {ident.get('count', 0)}×</span>"
            )
            rl.addWidget(name_label, 1)

            chk = QCheckBox("Avoid")
            chk.setChecked(bool(ident.get("avoid", False)))
            chk.toggled.connect(lambda checked, iid=ident["id"]: self._on_avoid_toggled(iid, checked))
            rl.addWidget(chk)

            rm = QPushButton("✕")
            rm.setFixedWidth(28)
            rm.setToolTip("Remove this person from the face bank")
            rm.clicked.connect(lambda _=False, iid=ident["id"]: self._on_remove_identity(iid))
            rl.addWidget(rm)

            self.avoid_list_layout.insertWidget(self.avoid_list_layout.count() - 1, r)

        self.avoid_count_label.setText(
            f"{len(identities)} people · {named} named · {len(bank.avoided_ids())} avoided"
        )

    def _on_avoid_toggled(self, identity_id, checked):
        """Persist an avoid toggle to the face database."""
        bank = getattr(self, "_face_bank", None)
        if bank is None:
            return
        bank.set_avoid(identity_id, checked)
        bank.save()
        name = bank.name_for(identity_id)
        self.append_log(f"{'🚫 Avoiding' if checked else '✅ Allowing'} {name} "
                        f"({len(bank.avoided_ids())} avoided)")
        self.avoid_count_label.setText(
            f"{len(bank.all_identities())} people · "
            f"{sum(1 for i in bank.all_identities() if i['name'])} named · "
            f"{len(bank.avoided_ids())} avoided"
        )

    def _on_scan_faces(self):
        videos = self.get_file_list()
        if not videos:
            self.append_log("⚠️ Add a video first, then scan it for faces.")
            return
        video = videos[0]
        if not os.path.exists(video):
            self.append_log(f"⚠️ Video not found: {video}")
            return
        self.avoid_scan_btn.setEnabled(False)
        self.avoid_scan_btn.setText("🔍 Scanning…")
        self._scan_worker = FaceScanWorker(video, "./cache/face_db.json")
        self._scan_worker.log.connect(self.append_log)
        self._scan_worker.done.connect(self._on_scan_done)
        self._scan_worker.start()

    def _on_remove_identity(self, identity_id):
            bank = self._get_face_bank()
            if bank is None:
                return
            if bank.remove(identity_id):
                bank.save()
                self.append_log("🗑 Removed 1 person from the face bank")
            self.refresh_avoid_list()

    def _on_clear_faces(self):
            from PySide6.QtWidgets import QMessageBox
            bank = self._get_face_bank()
            if not bank or len(bank) == 0:
                self.append_log("ℹ️ Face bank is already empty.")
                return
            box = QMessageBox(self)
            box.setWindowTitle("Clear faces")
            box.setText(f"Clear the face bank ({len(bank)} identities)?")
            box.setInformativeText("Choose what to remove.")
            btn_all   = box.addButton("Clear everything", QMessageBox.ButtonRole.DestructiveRole)
            btn_keep  = box.addButton("Keep named / avoided", QMessageBox.ButtonRole.AcceptRole)
            btn_cancel = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            box.exec()
            clicked = box.clickedButton()
            if clicked is btn_cancel:
                return
            kept = bank.clear(keep_named=(clicked is btn_keep))
            bank.save()
            self.append_log(f"🗑 Face bank cleared — {kept} identities kept")
            self.refresh_avoid_list()

    def _on_scan_done(self, n):
        self.avoid_scan_btn.setEnabled(True)
        self.avoid_scan_btn.setText("🔍 Scan video for faces")
        if n >= 0:
            self.append_log(f"✅ Face scan complete — {n} identities in the bank")
        self.refresh_avoid_list()

    # --- Downloader methods ---
    def browse_save_directory(self):
        """Browse for save directory"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Save Directory", self.download_save_dir_input.text()
        )
        if directory:
            self.download_save_dir_input.setText(directory)

    def browse_and_select_videos(self):
        """Open the thumbnail picker for the listing URL, then download the chosen videos."""
        url = self.download_url_input.text().strip()
        if not url.startswith(("http://", "https://")):
            self.append_log("⚠️ Enter a listing URL (http:// or https://) first")
            return
        try:
            from video_picker_dialog import VideoPickerDialog
        except Exception as e:
            self.append_log(f"❌ Video picker unavailable: {e}")
            return
        dlg = VideoPickerDialog(url, pattern="auto", use_browser="auto", parent=self)
        if dlg.exec():
            urls = [e["url"] for e in dlg.selected_entries()]
            if not urls:
                self.append_log("No videos selected.")
                return
            self.append_log(f"🗂 Selected {len(urls)} video(s) from picker")
            self.start_download(video_urls=urls)

    def start_download(self, video_urls=None):
        """Start the download process. If video_urls is given (from the picker),
        those exact URLs are downloaded instead of scraping the listing."""
        url = self.download_url_input.text().strip()
        save_dir = self.download_save_dir_input.text().strip()
        pattern = "auto"  # link pattern is auto-detected from the listing page

        # After-download processing mode: none / immediate / batch.
        process_mode = self.process_mode_combo.currentData()
        immediate_processing = (process_mode == "immediate")
        max_concurrent = self.concurrent_spinbox.value() if immediate_processing else 1
        
        # Get time range settings from the download-mode picker.
        mode = self.download_mode_combo.currentData()
        time_range = None
        use_percentages = False
        download_full = False

        if mode == "same":
            # Reuse the processing range. Selecting this mode auto-enables the
            # processing checkbox (see on_download_mode_changed); guard anyway.
            if not self.use_time_range_chk.isChecked():
                self.use_time_range_chk.setChecked(True)
            start_pct = self.range_slider.start()
            end_pct = self.range_slider.end()
            if end_pct <= start_pct:
                self.append_log("⚠️ Invalid time range - end must be greater than start")
                return
            time_range = (float(start_pct), float(end_pct))
            use_percentages = True
            self.append_log(f"⏱️ Downloading percentage range: {start_pct}% - {end_pct}%")
        elif mode == "specific":
            start_s = self.download_start_input.value()
            end_s = self.download_end_input.value()
            if end_s <= start_s:
                self.append_log("⚠️ Invalid range - end must be greater than start")
                return
            time_range = (float(start_s), float(end_s))
            use_percentages = False
            self.append_log(f"⏱️ Downloading seconds range: {start_s}s - {end_s}s")
        else:  # "full"
            download_full = True
            self.append_log("📥 Downloading full videos")
        
        # Validation
        if not url:
            self.append_log("⚠️ Please enter a URL")
            return
        
        if not save_dir:
            self.append_log("⚠️ Please enter a save directory")
            return
        
        # Check if URL is valid
        if not url.startswith(("http://", "https://")):
            self.append_log("⚠️ URL must start with http:// or https://")
            return
        
        # Check if already running
        if hasattr(self, 'download_worker') and self.download_worker and self.download_worker.isRunning():
            self.append_log("⚠️ Download already in progress!")
            return
        
        # Clear log and start
        self.log_output.clear()
        self._show_progress(True)
        self.append_log("=== Starting Video Download ===")
        self.append_log(f"🌐 URL: {url}")
        self.append_log(f"📁 Save directory: {save_dir}")
        self.append_log("🔍 Link pattern: auto-detect")
        
        if immediate_processing:
            self.append_log(f"⚡ Mode: Immediate processing after each download")
            self.append_log(f"   Concurrent downloads: {max_concurrent}")
        else:
            self.append_log("📦 Mode: Batch download (process all videos at once)")
        
        # (Range already logged per-mode above.)

        self.append_log("")
        
        # UI state changes
        self.download_progress_bar.setVisible(True)
        self.download_progress_bar.setRange(0, 100)
        self.download_progress_bar.setValue(0)
        self.process_progress_bar.setVisible(False)
        self.process_progress_bar.setRange(0, 100)
        self.process_progress_bar.setValue(0)
        self.task_label.setText("🌐 Extracting video links...")
        self.download_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        
        # Define processing callback for immediate processing
        def process_video_callback(filepath, metadata):
            """Process video immediately after download using the pipeline.
            Skips processing if *_highlight.mp4 already exists next to the file.
            """
            try:
                filename = os.path.basename(filepath)
                base_name = os.path.splitext(filename)[0]
                source_dir = os.path.dirname(filepath)

                # Expected highlight output path
                output_file = os.path.join(source_dir, f"{base_name}_highlight.mp4")

                # Decide whether to skip existing highlights
                # If you later add a checkbox like self.skip_existing_highlights_chk, this will pick it up.
                skip_existing = True
                if hasattr(self, "skip_existing_highlights_chk"):
                    skip_existing = self.skip_existing_highlights_chk.isChecked()

                # Header in log
                self.append_log(f"\n{'='*60}")
                self.append_log(f"🎬 IMMEDIATE PROCESSING: {filename}")
                self.append_log(f"{'='*60}")

                # Auto-add downloaded video to file list (GUI-thread safe)
                if self.auto_add_downloaded_chk.isChecked():
                    existing = self.get_file_list()
                    if filepath not in existing:
                        QMetaObject.invokeMethod(
                            self.file_list, "addItem",
                            Qt.QueuedConnection,
                            Q_ARG(str, filepath)
                        )
                        self.append_log(f"📋 Added to file list: {filename}")

                # --- SKIP if highlight already exists ---
                if skip_existing and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    self.append_log(f"⏭️ Skipping processing (highlight exists): {os.path.basename(output_file)}")
                    self.append_log(f"{'='*60}\n")

                    return {
                        'processed_at': time.time(),
                        'filename': filename,
                        'highlight_file': output_file,
                        'success': True,
                        'skipped': True
                    }

                # Build config for this single video
                config = self.build_pipeline_config()
                config['output_file'] = output_file

                self.append_log(f"📁 Output will be: {os.path.basename(output_file)}")
                self.append_log("")

                # Run pipeline synchronously (this blocks the download worker thread by design)
                try:
                    from pipeline import run_highlighter
                    cancel_flag = threading.Event()

                    # Show indeterminate processing state in GUI
                    QMetaObject.invokeMethod(
                        self, "set_process_busy",
                        Qt.QueuedConnection,
                        Q_ARG(str, f"🔧 Processing: {filename} | Initializing…")
                    )

                    # Thread-safe logging back to GUI
                    def log_fn(msg):
                        QMetaObject.invokeMethod(
                            self, "append_log",
                            Qt.QueuedConnection,
                            Q_ARG(str, f"  [{filename}] {msg}")
                        )

                    # Thread-safe progress updates back to GUI
                    def progress_fn(current, total, task, details):
                        QMetaObject.invokeMethod(
                            self, "update_process_progress",
                            Qt.QueuedConnection,
                            Q_ARG(int, int(current)),
                            Q_ARG(int, int(total)),
                            Q_ARG(str, f"{filename} | {task}"),
                            Q_ARG(str, str(details))
                        )

                    # Feed the live preview window, exactly as the Run button
                    # does. Without this the checkbox looks like it applies to
                    # every run, but a downloaded video processed here handed
                    # the pipeline no preview_fn at all — so the window opened,
                    # said "Waiting for the detection stage", and stayed on that
                    # for the whole run. The signal makes the thread hop; the
                    # flag is read instead of the checkbox so this stays off the
                    # GUI thread's widgets.
                    def preview_fn(frame, boxes, sec):
                        if self._preview_enabled and not cancel_flag.is_set():
                            self.preview_frame.emit(frame, boxes, sec)

                    result = run_highlighter(
                        filepath,
                        gui_config=config,
                        log_fn=log_fn,
                        progress_fn=progress_fn,
                        cancel_flag=cancel_flag,
                        preview_fn=preview_fn,
                    )

                    # If pipeline returns a path, use it; otherwise fall back to our expected output_file
                    highlight_path = result or output_file

                    if highlight_path and os.path.exists(highlight_path) and os.path.getsize(highlight_path) > 0:
                        self.append_log(f"✅ Highlight created: {os.path.basename(highlight_path)}")
                        self.append_log(f"{'='*60}\n")

                        return {
                            'processed_at': time.time(),
                            'filename': filename,
                            'highlight_file': highlight_path,
                            'success': True,
                            'skipped': False
                        }

                    self.append_log("⚠️ Processing completed but no highlight generated (or file missing/empty)")
                    self.append_log(f"{'='*60}\n")
                    return {'success': False, 'error': 'No highlight generated'}

                except Exception as e:
                    self.append_log(f"❌ Processing error: {e}")
                    import traceback
                    self.append_log(f"Traceback:\n{traceback.format_exc()}")
                    self.append_log(f"{'='*60}\n")
                    return {'success': False, 'error': str(e)}

            except Exception as e:
                self.append_log(f"❌ Callback setup error: {e}")
                import traceback
                self.append_log(f"Traceback:\n{traceback.format_exc()}")
                return {'success': False, 'error': str(e)}
            
        # Videos processed straight off the download feed the preview window
        # too, so the flag those runs read has to match the checkbox before the
        # first one starts.
        self._preview_enabled = self.live_preview_checkbox.isChecked()

        # Create download worker with processing callback
        self.download_worker = DownloadWorker(
            url, save_dir, pattern,
            time_range=time_range,
            download_full=download_full,
            use_percentages=use_percentages,
            immediate_processing=immediate_processing,
            max_concurrent=max_concurrent,
            process_callback=process_video_callback if immediate_processing else None,
            video_urls=video_urls
        )
        
        # Connect signals
        self.download_worker.log.connect(self.append_log)
        self.download_worker.progress.connect(self.update_download_progress)
        self.download_worker.finished.connect(self.download_done)
        self.download_worker.cancelled.connect(self.download_cancelled)
        if immediate_processing:
            self.download_worker.video_processed.connect(self.on_video_processed)
        
        self.status_timer.start(100)
        self.download_worker.start()

    def build_pipeline_config(self):
        """Build pipeline configuration from GUI settings"""
        
        def get_list_from_input(input_field):
            text = input_field.text().strip()
            if not text:
                return None
            items = [s.strip() for s in text.split(",") if s.strip()]
            return items if items else None
        
        highlight_objects = get_list_from_input(self.objects_input)
        interesting_actions = get_list_from_input(self.actions_input)
        use_transcript = self.transcript_checkbox.isChecked()
        search_keywords = get_list_from_input(self.search_keywords_input) if use_transcript else []
        
        exact_duration_val = int(self.spin_exact_duration.value())
        exact_duration = exact_duration_val if exact_duration_val > 0 else None
        
        config = {
            "scene_points": int(self.spin_scene_points.value()),
            "motion_event_points": int(self.spin_motion_event_points.value()),
            "motion_peak_points": int(self.spin_motion_peak.value()),
            "audio_peak_points": int(self.spin_audio_peak.value()),
            "loudness_burst_points": int(self.spin_loudness_burst.value()),
            "keyword_points": int(self.spin_keyword_points.value()),
            "transcript_points": int(self.spin_transcript_points.value()),
            "beginning_points": int(self.spin_beginning_points.value()),
            "ending_points": int(self.spin_ending_points.value()),
            "beginning_seconds": int(self.spin_beginning_seconds.value()),
            "ending_seconds": int(self.spin_ending_seconds.value()),
            "object_points": int(self.spin_object.value()),
            "action_points": int(self.spin_action.value()),
            "face_expression_points": int(self.spin_face_expression.value()),
            "face_expression_labels": self.selected_face_labels(),
            "clip_time": int(self.spin_clip_time.value()),
            "coverage": self.slider_coverage.value() / 100.0,
            "report_only": bool(getattr(self, "_report_only", False)),
            "max_duration": int(self.spin_max_duration.value()),
            "exact_duration": exact_duration,
            "multi_signal_boost": 1.2,
            "min_signals_for_boost": 2,
            "keep_temp": self.keep_temp_chk.isChecked(),
            "render_mode": self.render_mode_combo.currentData(),
            "highlight_objects": highlight_objects,
            "interesting_actions": interesting_actions,
            "actions_require_objects": self.actions_require_objects_chk.isChecked(),
            "use_transcript": use_transcript,
            "transcript_model": self.transcript_model_combo.currentText(),
            "transcript_source_lang": self.transcript_source_lang.currentText(),
            "search_keywords": search_keywords,
            "create_subtitles": self.subtitles_checkbox.isChecked() and use_transcript,
            # The spoken language has one home: Transcript Settings.
            "source_lang": self.transcript_source_lang.currentText(),
            "target_lang": self.subtitle_target_lang.currentText(),
            "frame_skip": int(self.frame_skip_spin.value()),
            "vr_mode": self.vr_mode_chk.isChecked(),
            "object_frame_skip": int(self.obj_frame_skip_spin.value()),
            "yolo_type": self.object_detector_choice()[0],
            "yolo_model_size": self.yolo_model_combo.currentData(),
            "yolo_custom_model_path": self.object_detector_choice()[1] or getattr(self, "_custom_pose_model", None),
            "sample_rate": int(self.sample_rate_spin.value()),
            "auto_min_clip": float(self.spin_auto_min_clip.value()),
            "auto_max_clip": float(self.spin_auto_max_clip.value()),
            "auto_merge_gap": float(self.spin_auto_merge_gap.value()),
            "draw_object_boxes": self.bbox_objects_chk.isChecked(),
            "write_highlight_report": self.why_report_chk.isChecked(),
            "draw_action_labels": self.bbox_actions_chk.isChecked(),
            "action_backend": self.action_backend_combo.currentData(),
            "r3d_model": self.r3d_model_combo.currentData(),
            "action_models": self.action_models_combo.currentData(),
            "object_confidence": self.obj_confidence_spin.value() / 100.0,
            "force_reprocess": self.force_reprocess_checkbox.isChecked(),
        }
      
        # Add time range if enabled
        if self.use_time_range_chk.isChecked() and self.current_video_duration > 0:
            start_pct = self.range_slider.start() / 100
            end_pct = self.range_slider.end() / 100
            config["use_time_range"] = True
            config["range_start"] = int(start_pct * self.current_video_duration)
            config["range_end"] = int(end_pct * self.current_video_duration)
        else:
            config["use_time_range"] = False
        
        # Remove None values
        return {k: v for k, v in config.items() if v is not None}


    def on_video_processed(self, filepath, result):
        """Handle when a video is processed immediately after download"""
        filename = os.path.basename(filepath)
        if result.get('success'):
            self.append_log(f"✅ {filename} downloaded and processed successfully")
        else:
            self.append_log(f"⚠️ {filename} downloaded but processing failed")


    def on_process_mode_changed(self):
        """Concurrent downloads only matter while processing overlaps downloads
        (the 'immediate' mode); grey the spinner otherwise."""
        self.concurrent_spinbox.setEnabled(
            self.process_mode_combo.currentData() == "immediate"
        )

    def on_download_mode_changed(self):
        """Show the seconds inputs only for 'specific', and make 'same' pull in
        a processing range to reuse (auto-enable 'Process only specific time
        range' so there's an actual range instead of the whole video)."""
        mode = self.download_mode_combo.currentData()
        if hasattr(self, "download_range_widget"):
            self.download_range_widget.setVisible(mode == "specific")
        if mode == "same" and not self.use_time_range_chk.isChecked():
            self.use_time_range_chk.setChecked(True)
        if mode == "specific":
            self.update_download_duration()

    def update_download_duration(self):
        """Update the duration label for the specific-range mode."""
        if self.download_mode_combo.currentData() != "specific":
            return

        start = self.download_start_input.value()
        end = self.download_end_input.value()
        
        # Ensure end is after start
        if end <= start:
            end = start + 1
            self.download_end_input.setValue(end)
        
        duration = end - start
        minutes = duration // 60
        seconds = duration % 60
        
        self.download_duration_label.setText(
            f"Duration: {duration}s ({minutes}:{seconds:02d})"
        )

    def download_done(self, downloaded_files):
        """Handle download completion with immediate processing support"""
        self.status_timer.stop()
        
        if hasattr(self, 'download_worker') and self.download_worker and self.download_worker.is_cancelled():
            self.append_log("\n⏹️ === DOWNLOAD CANCELLED ===")
            self.task_label.setText("⏹️ Cancelled")
            self.task_label.setStyleSheet("color: #ff9800; font-weight: bold;")
            self.download_cleanup()
            return
        
        if downloaded_files:
            self.append_log(f"\n✅ === DOWNLOAD COMPLETED ===")
            self.append_log(f"📊 Successfully downloaded {len(downloaded_files)} videos")
            
            # Check if immediate processing was enabled
            if self.process_mode_combo.currentData() == "immediate":
                # Count successful processing
                if hasattr(self.download_worker, '_download_results'):
                    processed_count = sum(1 for r in self.download_worker._download_results 
                                        if r.get('processed', False))
                    self.append_log(f"🎬 Successfully processed {processed_count}/{len(downloaded_files)} videos")
                    
                    # List all results
                    for result in self.download_worker._download_results:
                        if result.get('success') and result.get('processed'):
                            highlight = result.get('process_result', {}).get('highlight_file')
                            if highlight:
                                self.append_log(f"  ✅ {os.path.basename(highlight)}")
                
                # Combine highlights if enabled and we have multiple
                if self.auto_combine_chk.isChecked() and len(downloaded_files) > 1:
                    self.append_log("\n🎬 Combining all highlights...")
                    highlight_files = []
                    
                    if hasattr(self.download_worker, '_download_results'):
                        for result in self.download_worker._download_results:
                            highlight = result.get('process_result', {}).get('highlight_file')
                            if highlight and os.path.exists(highlight):
                                highlight_files.append(highlight)
                    
                    if len(highlight_files) > 1:
                        first_video_dir = os.path.dirname(highlight_files[0])
                        combined_output = os.path.join(first_video_dir, "all_highlights_combined.mp4")
                        combined_file = self.combine_highlights(highlight_files, combined_output)
                        
                        if combined_file:
                            self.append_log(f"🎉 Combined highlight: {combined_file}")
            
            self.task_label.setText("✅ Complete!")
            self.task_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.append_log("\n⚠️ === DOWNLOAD COMPLETED WITH NO FILES ===")
            self.task_label.setText("❌ Download Failed")
            self.task_label.setStyleSheet("color: #f44336; font-weight: bold;")

        # Batch mode: downloads are done, now run the pipeline over them. Add the
        # files to the list first (batch needs them there regardless of the
        # auto-add toggle), then hand off to the pipeline.
        if self.process_mode_combo.currentData() == "batch" and downloaded_files:
            existing = self.get_file_list()
            for f in downloaded_files:
                if f not in existing and os.path.exists(f):
                    self.file_list.addItem(f)
            if self.file_list.count() > 0:
                self.append_log("\n▶️ Starting batch processing of downloaded videos...")
                self.auto_start_pipeline()
                return

        self.download_cleanup()
        self._show_progress(False)

    def auto_start_pipeline(self):
        """Automatically start pipeline processing after download"""
        # Clean up download state
        self.download_cleanup()
        
        # Small delay to ensure UI updates
        QApplication.processEvents()
        
        # Now start the pipeline
        self.run_pipeline()

    def download_cancelled(self):
        """Handle download cancellation"""
        self.status_timer.stop()
        self.append_log("\n⏹️ === DOWNLOAD CANCELLED BY USER ===")
        self.task_label.setText("⏹️ Download Cancelled")
        self.task_label.setStyleSheet("color: #ff9800; font-weight: bold;")
        self.download_cleanup()

    def download_cleanup(self):
        """Clean up UI state after download completion/cancellation"""
        # Hide progress bar only if not auto-processing
        if self.process_mode_combo.currentData() != "batch" or self.file_list.count() == 0:
            self.download_progress_bar.setVisible(False)
            # If you're not auto-processing, also hide processing bar
            self.process_progress_bar.setVisible(False)

        
        # Re-enable controls
        self.download_btn.setEnabled(True)
        
        # Only re-enable cancel if not auto-processing
        if self.process_mode_combo.currentData() != "batch" or self.file_list.count() == 0:
            self.cancel_btn.setEnabled(False)
            self.cancel_btn.setText("Cancel")
        
        # Reset task label style after 5 seconds (only if not auto-processing)
        if self.process_mode_combo.currentData() != "batch" or self.file_list.count() == 0:
            QTimer.singleShot(5000, lambda: self.task_label.setStyleSheet("color: #666; font-weight: bold;"))
        
        # Clean up worker
        if hasattr(self, 'download_worker') and self.download_worker:
            if self.download_worker.isRunning():
                self.download_worker.wait(1000)
            self.download_worker = None

    # --- Multi-file support methods ---
    def browse_files(self):
        """Add one or more video files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Video(s)", "", "Videos (*.mp4 *.mov *.avi *.mkv)"
        )
        existing = self.get_file_list()
        for path in file_paths:
            if path not in existing:
                self.file_list.addItem(path)
        
        # Auto-set output filename based on first video if output is empty or default
        if file_paths and (not self.output_input.text().strip() or 
                        self.output_input.text().strip() == "highlight.mp4"):
            first_video = file_paths[0]
            base_name = os.path.splitext(os.path.basename(first_video))[0]
            self.output_input.setText(f"{base_name}_highlight.mp4")
        
        # Update video duration for time range slider (use first video)
        if file_paths:
            self.update_video_duration(file_paths[0])

    def remove_selected_file(self):
        """Remove selected file from the list"""
        current_row = self.file_list.currentRow()
        if current_row >= 0:
            self.file_list.takeItem(current_row)

    def clear_files(self):
        """Clear all files from the list and reset output name"""
        self.file_list.clear()
        self.output_input.setText("highlight.mp4")
        # Reset video duration info
        self.current_video_duration = 0
        self.video_duration_label.setText("Select a video to enable time range controls")
        self.video_duration_label.setStyleSheet("color: #666; font-style: italic;")
        self.update_selection_info()

    def get_file_list(self):
        """Get list of all files in the list widget"""
        return [self.file_list.item(i).text() for i in range(self.file_list.count())]
    
    def combine_highlights(self, highlight_files, output_path):
        """Combine multiple highlight videos into one.

        Thin delegate to modules.combine_videos.combine_videos (the same engine
        the sidecar drives), keeping this method's original contract for the Qt
        callers: None when there is nothing to combine, the lone file passed
        straight through when there is only one, otherwise the combined output
        path. All engine logging is routed through append_log."""
        if not highlight_files:
            self.append_log("⚠️ No highlight files to combine")
            return None

        # Filter out None values and non-existent files
        valid_files = [f for f in highlight_files if f and os.path.exists(f)]

        if not valid_files:
            self.append_log("⚠️ No valid highlight files found")
            return None

        if len(valid_files) == 1:
            self.append_log("ℹ️ Only one highlight file, no combining needed")
            return valid_files[0]

        try:
            from modules.combine_videos import combine_videos

            return combine_videos(
                valid_files, output_path, log_fn=self.append_log,
            )
        except Exception as e:
            self.append_log(f"❌ Failed to combine highlights: {e}")
            import traceback
            self.append_log(f"Traceback:\n{traceback.format_exc()}")
            return None
            
    # --- Config persistence ---
    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def save_config(self):
        # Helper function to get non-empty text or empty list
        def get_text_list(input_field):
            text = input_field.text().strip()
            if not text:
                return []
            return [s.strip() for s in text.split(",") if s.strip()]

        data = {
            "video": {"paths": self.get_file_list()},
            "download": {
                "last_url": self.download_url_input.text().strip(),
                "save_dir": self.download_save_dir_input.text().strip(),
                "auto_add": self.auto_add_downloaded_chk.isChecked(),
                "auto_combine": self.auto_combine_chk.isChecked(),
                "download_mode": self.download_mode_combo.currentData(),
                "process_mode": self.process_mode_combo.currentData(),
                "concurrent_downloads": self.concurrent_spinbox.value(),
                "time_range_start": self.download_start_input.value(),
                "time_range_end": self.download_end_input.value(),
            },
            "highlights": {
                "clip_time": int(self.spin_clip_time.value()),
                "coverage": self.slider_coverage.value() / 100.0,
                "output": self.output_input.text().strip(),
                "max_duration": int(self.spin_max_duration.value()),
                "exact_duration": int(self.spin_exact_duration.value()),
                "keep_temp": self.keep_temp_chk.isChecked(),
                "render_mode": self.render_mode_combo.currentData(),
                "auto_min_clip": int(self.spin_auto_min_clip.value()),
                "auto_max_clip": int(self.spin_auto_max_clip.value()),
                "auto_merge_gap": int(self.spin_auto_merge_gap.value()),
                "use_time_range": self.use_time_range_chk.isChecked(),
                "range_start_pct": self.range_slider.start(),
                "range_end_pct": self.range_slider.end(),
            },
            "scoring": {
                "scene_points": int(self.spin_scene_points.value()),
                "motion_event_points": int(self.spin_motion_event_points.value()),
                "motion_peak_points": int(self.spin_motion_peak.value()),
                "audio_peak_points": int(self.spin_audio_peak.value()),
                "loudness_burst_points": int(self.spin_loudness_burst.value()),
                "keyword_points": int(self.spin_keyword_points.value()),
                "transcript_points": int(self.spin_transcript_points.value()),
                "object_points": int(self.spin_object.value()),
                "action_points": int(self.spin_action.value()),
                "face_expression_points": int(self.spin_face_expression.value()),
                "face_expression_labels": self.selected_face_labels(),
                "beginning_points": int(self.spin_beginning_points.value()),
                "ending_points": int(self.spin_ending_points.value()),
                "beginning_seconds": int(self.spin_beginning_seconds.value()),
                "ending_seconds": int(self.spin_ending_seconds.value()),
                "multi_signal_boost": 1.2,
                "min_signals_for_boost": 2,
            },
            "actions": {
                "interesting": get_text_list(self.actions_input),
                "require_objects": self.actions_require_objects_chk.isChecked()
            },
            "objects": {
                "interesting": get_text_list(self.objects_input),
                "confidence": self.obj_confidence_spin.value(),
            },
            "keywords": {
                "transcript_file": "transcript.txt",
                "interesting": get_text_list(self.search_keywords_input),
            },
            "transcript": {
                "enabled": self.transcript_checkbox.isChecked(),
                "model": self.transcript_model_combo.currentText(),
                "source_lang": self.transcript_source_lang.currentText(),
                "search_keywords": get_text_list(self.search_keywords_input),
            },
            "subtitles": {
                "enabled": self.subtitles_checkbox.isChecked(),
                # Mirrors transcript.source_lang so an older build (and anything
                # still reading subtitles.source_lang) sees one answer, not two.
                "source_lang": self.transcript_source_lang.currentText(),
                "target_lang": self.subtitle_target_lang.currentText(),
            },
            # Detector knobs with no widget of their own. Carried through from
            # whatever is on disk rather than omitted, because this dict is
            # written whole - anything missing here is deleted from config.yaml
            # the first time the user saves settings.
            "loudness_bursts": self.config_data.get("loudness_bursts", {}),
            "advanced": {
                "frame_skip": int(self.frame_skip_spin.value()),
                "vr_mode": self.vr_mode_chk.isChecked(),
                "object_frame_skip": int(self.obj_frame_skip_spin.value()),
                "sample_rate": int(self.sample_rate_spin.value()),
                "yolo_type": self.object_detector_choice()[0],
                "yolo_model_size": self.yolo_model_combo.currentData(),
                "yolo_custom_model_path": self.object_detector_choice()[1],
                "action_backend": self.action_backend_combo.currentData(),
                "r3d_model": self.r3d_model_combo.currentData(),
                "action_models": self.action_models_combo.currentData(),
            },
            "visualization": {
                "draw_object_boxes": self.bbox_objects_chk.isChecked(),
                "write_highlight_report": self.why_report_chk.isChecked(),
                "draw_action_labels": self.bbox_actions_chk.isChecked(),
            },
            "avoid": {
                "face_recognition_enabled": self.avoid_face_recognition_chk.isChecked(),
            },
            "ui": {
                "suppress_no_cache_warning": self.config_data.get("ui", {}).get("suppress_no_cache_warning", False),
            },
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True)
            
    def closeEvent(self, event):
        self.save_config()
        event.accept()
        # Hard-kill on main-window close. We can't rely on app.exec() returning:
        # the timeline viewer window is kept alive (hidden) for reuse, and a
        # lingering hidden window can stop Qt from quitting. Killing here, the
        # moment the user closes the main GUI, guarantees the process dies even
        # if native FFmpeg/onnxruntime threads are stuck (which deadlock the
        # normal os._exit/ExitProcess path on Windows).
        _hard_exit(0)

    def check_worker_status(self):
        """Periodic check of worker status for UI responsiveness"""
        if self.worker and not self.worker.isRunning():
            self.status_timer.stop()

    def on_transcript_toggle(self, checked):
        """Handle transcript checkbox toggle"""
        self.transcript_source_lang.setEnabled(checked)
        self.transcript_model_combo.setEnabled(checked)
        # Keyword scoring controls live in Basic Settings but only work with a
        # transcript, so they grey out with it.
        self.search_keywords_input.setEnabled(checked)
        self.search_keywords_label.setEnabled(checked)
        self.spin_keyword_points.setEnabled(checked)
        self.subtitles_checkbox.setEnabled(checked)
        
        # If transcript is disabled, also disable subtitles
        if not checked:
            self.subtitles_checkbox.setChecked(False)
            self.on_subtitles_toggle(False)

    def on_subtitles_toggle(self, checked):
        """Handle subtitles checkbox toggle"""
        # Subtitles can only be enabled if transcript is enabled
        transcript_enabled = self.transcript_checkbox.isChecked()
        final_state = checked and transcript_enabled
        
        self.subtitle_target_lang.setEnabled(final_state)

    # --- Labels ---
    def load_labels_from_json(self, filepath):
            """Load label list from a JSON file. Handles list, dict, and nested dict formats."""
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return [str(item) for item in data]
                elif isinstance(data, dict):
                    # Intel custom: has "label_to_idx" key
                    if "label_to_idx" in data:
                        return list(data["label_to_idx"].keys())
                    # Intel custom alt: has "idx_to_label" key
                    if "idx_to_label" in data:
                        return list(data["idx_to_label"].values())
                    # YOLO: has "class" key with {index: label}
                    if "class" in data:
                        return list(data["class"].values())
                    # Flat dict: {index: label} or {label: index}
                    values = list(data.values())
                    if values and isinstance(values[0], str):
                        return list(data.values())
                    else:
                        return list(data.keys())
                else:
                    self.append_log(f"⚠️ Unexpected JSON format in {filepath}")
                    return []
            except Exception as e:
                self.append_log(f"❌ Failed to load labels from {filepath}: {e}")
                return []

    def object_detector_choice(self):
        """The Advanced tab's object-model selection as (yolo_type, path) —
        exactly the pair the pipeline consumes. ("standard", "") when nothing
        custom is selected."""
        data = self.object_model_combo.currentData()
        if not data:
            return ("standard", "")
        return (data[0] or "standard", data[1] or "")

    def custom_object_class_names(self):
        """Class names of the selected custom object detector, read from the
        model's own metadata. Empty for keypoint/pose models and when no custom
        model is selected — callers fall back to the pose sidecars."""
        path = self.object_detector_choice()[1]
        if not path or not os.path.exists(path):
            return []
        from modules.app_paths import object_model_names
        return object_model_names(path)

    def open_object_label_selector(self):
        """Open label selector. For the custom model this offers your trained
        class names; for 'mixed' it merges those with the COCO objects;
        otherwise the standard YOLO objects."""
        yolo_type = self.object_detector_choice()[0]

        labels = []
        if "custom" in yolo_type:
            # A custom *detector* carries its class names in the model itself;
            # only fall back to the pose sidecars when it's a keypoint model.
            labels = self.custom_object_class_names()
            if not labels:
                try:
                    from modules.app_paths import custom_keypoint_names
                    labels = custom_keypoint_names()
                except Exception:
                    labels = []
            if not labels:
                self.append_log("⚠️ No custom class names found (choose a model / check labels).")

        if yolo_type != "custom":  # standard or mixed -> include COCO objects
            if os.path.exists(YOLO_OBJECTS_LABELS_FILE):
                labels = labels + self.load_labels_from_json(YOLO_OBJECTS_LABELS_FILE)

        if not labels:
            self.append_log("⚠️ No labels available for the selected model.")
            return

        current = [s.strip() for s in self.objects_input.text().split(",") if s.strip()]
        title = ("Select Object Labels (custom + YOLO)" if yolo_type == "custom_mixed"
                 else "Select Labels (custom model)" if yolo_type == "custom"
                 else "Select Object Labels (YOLO)")
        dlg = LabelSelectorDialog(title, labels, current, self)
        if dlg.exec() == QDialog.Accepted:
            selected = dlg.get_selected_labels()
            self.objects_input.setText(", ".join(selected))
            self.append_log(f"✅ Loaded {len(selected)} object labels")

    def open_action_label_selector(self):
        """Open label selector based on current backend and action models settings."""
        backend = self.action_backend_combo.currentData()
        action_models = self.action_models_combo.currentData()

        # R3D-only always uses Kinetics-400
        if backend in ("r3d_cuda", "r3d_cpu"):
            action_models = "intel_only"

        if action_models == "custom_only":
            label_file = INTEL_CUSTOM_LABELS_FILE
            title = f"Select Action Labels (Custom Fine-tuned — {self._custom_ov_count} classes)"
        elif action_models == "intel_only":
            label_file = KINETICS_400_LABELS_FILE
            title = "Select Action Labels (Intel Kinetics-400 — 400 classes)"
        elif action_models == "r3d_custom_only":
            label_file = R3D_CUSTOM_LABELS_FILE
            title = "Select Action Labels (R3D Fine-tuned)"
        elif action_models == "mixed":
            # Show labels tagged with source model
            custom_labels = []
            intel_labels = []
            if os.path.exists(INTEL_CUSTOM_LABELS_FILE):
                custom_labels = self.load_labels_from_json(INTEL_CUSTOM_LABELS_FILE)
            if os.path.exists(KINETICS_400_LABELS_FILE):
                intel_labels = self.load_labels_from_json(KINETICS_400_LABELS_FILE)

            tagged = []
            custom_set = set(l.lower() for l in custom_labels)
            intel_set = set(l.lower() for l in intel_labels)
            # Labels in both → show tagged versions
            overlap = custom_set & intel_set
            for label in sorted(custom_labels):
                if label.lower() in overlap:
                    tagged.append(f"{label} [custom]")
                else:
                    tagged.append(label)
            for label in sorted(intel_labels):
                if label.lower() in overlap:
                    tagged.append(f"{label} [intel]")
                else:
                    if label.lower() not in custom_set:  # avoid duplicates for non-overlap
                        tagged.append(label)
            tagged.sort()

            if not tagged:
                self.append_log("⚠️ No label files found")
                return
            current = [s.strip() for s in self.actions_input.text().split(",") if s.strip()]
            overlap_count = len(overlap)
            dlg = LabelSelectorDialog(
                f"Select Action Labels (Mixed — {len(tagged)} labels, {overlap_count} shared)",
                tagged, current, self)
            if dlg.exec() == QDialog.Accepted:
                selected = dlg.get_selected_labels()
                self.actions_input.setText(", ".join(selected))
                self.append_log(f"✅ Loaded {len(selected)} action labels (mixed)")
            return
        else:
            label_file = KINETICS_400_LABELS_FILE
            title = "Select Action Labels"

        if not os.path.exists(label_file):
            self.append_log(f"⚠️ Label file not found: {label_file}")
            return

        labels = self.load_labels_from_json(label_file)
        if not labels:
            self.append_log(f"⚠️ No labels found in {label_file}")
            return

        current = [s.strip() for s in self.actions_input.text().split(",") if s.strip()]
        dlg = LabelSelectorDialog(title, labels, current, self)
        if dlg.exec() == QDialog.Accepted:
            selected = dlg.get_selected_labels()
            self.actions_input.setText(", ".join(selected))
            self.append_log(f"✅ Loaded {len(selected)} action labels from {os.path.basename(label_file)}")

    def setup_label_completers(self):
        if os.path.exists(YOLO_OBJECTS_LABELS_FILE):
            obj_labels = self.load_labels_from_json(YOLO_OBJECTS_LABELS_FILE)
            if obj_labels:
                completer = MultiCompleter(obj_labels, self)
                completer.setMaxVisibleItems(10)
                self.objects_input.setCompleter(completer)

    def update_actions_completer(self):
        """Update actions auto-complete labels based on selected backend and action models.

        Called from several places that can fire in one cascade (a backend change
        repopulates the model combo, which re-emits currentIndexChanged), so it
        no-ops when the selection resolves to the labels already installed."""
        backend = self.action_backend_combo.currentData()
        action_models = self.action_models_combo.currentData()

        # R3D-only always uses Kinetics-400
        if backend in ("r3d_cuda", "r3d_cpu"):
            action_models = "intel_only"

        if action_models == getattr(self, "_actions_completer_models", -1):
            return
        self._actions_completer_models = action_models

        action_labels = []
        source = None

        if action_models == "custom_only":
            if os.path.exists(INTEL_CUSTOM_LABELS_FILE):
                action_labels = self.load_labels_from_json(INTEL_CUSTOM_LABELS_FILE)
                source = f"Custom fine-tuned ({self._custom_ov_count} classes)"
        elif action_models == "intel_only":
            if os.path.exists(KINETICS_400_LABELS_FILE):
                action_labels = self.load_labels_from_json(KINETICS_400_LABELS_FILE)
                source = "Intel Kinetics-400 (400 classes)"
        elif action_models == "r3d_custom_only":
            if os.path.exists(R3D_CUSTOM_LABELS_FILE):
                action_labels = self.load_labels_from_json(R3D_CUSTOM_LABELS_FILE)
                source = f"R3D fine-tuned ({len(action_labels)} classes)"
        elif action_models == "mixed":
            custom_labels = []
            intel_labels = []
            if os.path.exists(INTEL_CUSTOM_LABELS_FILE):
                custom_labels = self.load_labels_from_json(INTEL_CUSTOM_LABELS_FILE)
            if os.path.exists(KINETICS_400_LABELS_FILE):
                intel_labels = self.load_labels_from_json(KINETICS_400_LABELS_FILE)
            # Build tagged list for overlapping labels
            custom_set = set(l.lower() for l in custom_labels)
            intel_set = set(l.lower() for l in intel_labels)
            overlap = custom_set & intel_set
            tagged = []
            for label in custom_labels:
                tagged.append(f"{label} [custom]" if label.lower() in overlap else label)
            for label in intel_labels:
                if label.lower() in overlap:
                    tagged.append(f"{label} [intel]")
                elif label.lower() not in custom_set:
                    tagged.append(label)
            action_labels = sorted(set(tagged))
            source = f"Mixed ({len(custom_labels)} custom + {len(intel_labels)} Kinetics-400, {len(overlap)} shared, {len(action_labels)} total)"

        if action_labels:
            completer = MultiCompleter(action_labels, self)
            completer.setMaxVisibleItems(10)
            self.actions_input.setCompleter(completer)
            print(f"🔤 Actions auto-complete: {source}")
        else:
            self.actions_input.setCompleter(None)

    @Slot(str)
    def append_log(self, text: str):
        """Thread-safe log append (always executes on GUI thread)."""
        app = QApplication.instance()
        gui_thread = app.thread() if app else None

        if gui_thread and QThread.currentThread() != gui_thread:
            QMetaObject.invokeMethod(
                self, "append_log",
                Qt.QueuedConnection,
                Q_ARG(str, text)
            )
            return

        # --- GUI thread only below ---
        # Insert through a standalone cursor rather than QTextEdit.append(),
        # which moves the widget's own cursor and clears the user's selection —
        # that made text impossible to select/copy while logs were streaming.
        from PySide6.QtGui import QTextCursor
        scrollbar = self.log_output.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 4

        cursor = QTextCursor(self.log_output.document())
        cursor.movePosition(QTextCursor.End)
        if not self.log_output.document().isEmpty():
            cursor.insertBlock()
        cursor.insertText(text)

        # Only follow the tail if the user was already at the bottom; don't yank
        # them down (and away from a selection) while they scroll back through it.
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    def _show_progress(self, visible=True):
        # Show/hide the whole progress box. Hidden when idle so it doesn't sit
        # there empty; the tabs+log splitter above absorbs the size change.
        # The bars are made visible again by update_download/process_progress().
        self.progress_group.setVisible(visible)
        if not visible:
            self.download_progress_bar.setVisible(False)
            self.process_progress_bar.setVisible(False)
            self.hide_batch_progress()
            self.task_label.setText("Ready")

    @Slot(int, int, str, str)
    def update_pipeline_progress(self, current: int, total: int, task_name: str, details: str = ""):
        """Split the batch counter off to its own row. Everything else the
        pipeline emits is a stage of the video currently being worked on."""
        if task_name.lower().startswith("batch"):
            self.update_batch_progress(current, total, task_name, details)
        else:
            self.update_process_progress(current, total, task_name, details)

    @Slot(int, int, str, str)
    def update_batch_progress(self, current: int, total: int, task_name: str, details: str = ""):
        """Videos finished out of total, kept visible for the whole batch run."""
        # Only worth stating when there is more than one video: "1 / 1" tells
        # nobody anything they cannot see from the file list.
        counter = f"{max(0, min(current, total))}/{total}  " if total > 1 else ""
        self.batch_label.setText(f"📦 {counter}{details}")
        self.batch_label.setVisible(True)
        QApplication.processEvents()

    def hide_batch_progress(self):
        self.batch_label.setVisible(False)

    @Slot(str)
    def set_process_busy(self, text: str):
        self.process_progress_bar.setVisible(True)
        self.process_progress_bar.setRange(0, 0)  # indeterminate
        self.task_label.setText(text)

    @Slot(int, int, str, str)
    def update_download_progress(self, current: int, total: int, task_name: str, details: str = ""):
        if total > 0:
            self.download_progress_bar.setRange(0, 100)
            pct = min(100, max(0, int((current / total) * 100)))
            self.download_progress_bar.setValue(pct)
            self.download_progress_bar.setVisible(True)
            self.task_label.setText(f"⬇️ {task_name}: {pct}% - {details}")
        else:
            self.download_progress_bar.setVisible(True)
            self.download_progress_bar.setRange(0, 0)
            self.task_label.setText(f"⬇️ {task_name} - {details}")

        QApplication.processEvents()

    @Slot(int, int, str, str)
    def update_process_progress(self, current: int, total: int, task_name: str, details: str = ""):
        if total > 0:
            self.process_progress_bar.setRange(0, 100)
            pct = min(100, max(0, int((current / total) * 100)))
            self.process_progress_bar.setValue(pct)
            self.process_progress_bar.setVisible(True)
            self.task_label.setText(f"🔧 {task_name}: {pct}% - {details}")
        else:
            self.process_progress_bar.setVisible(True)
            self.process_progress_bar.setRange(0, 0)
            self.task_label.setText(f"🔧 {task_name} - {details}")

        # Keep UI responsive
        QApplication.processEvents()

    def format_time(self, seconds):
        """Format seconds as MM:SS or HH:MM:SS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    def on_time_range_toggle(self, checked):
        """Show/enable the time-range controls only while the box is ticked;
        collapse the group to a single line otherwise."""
        if hasattr(self, "time_range_body"):
            self.time_range_body.setVisible(checked)
        # Always enable sliders when checkbox is checked, even without video
        self.range_slider.setEnabled(checked)
        
        # Preset buttons only work when video duration is known
        has_duration = self.current_video_duration > 0
        self.first_5min_btn.setEnabled(checked and has_duration)
        self.last_5min_btn.setEnabled(checked and has_duration)
        self.last_10min_btn.setEnabled(checked and has_duration)
        self.middle_btn.setEnabled(checked and has_duration)
        self.full_video_btn.setEnabled(checked and has_duration)
        
        self.update_selection_info()

    def on_slider_changed(self):
        self.update_selection_info()

    def update_selection_info(self):
        """Update the selection information labels"""
        start_pct = self.range_slider.start()
        end_pct = self.range_slider.end()
        
        if self.current_video_duration == 0:
            # No video loaded - show percentages
            self.start_time_label.setText(f"{start_pct}%")
            self.end_time_label.setText(f"{end_pct}%")
            
            if self.use_time_range_chk.isChecked():
                range_pct = end_pct - start_pct
                self.selection_info_label.setText(
                    f"Selection: {start_pct}% to {end_pct}% ({range_pct}% of video)"
                )
                self.selection_info_label.setStyleSheet("color: #2f81f7; font-weight: bold; font-size: 10pt;")
            else:
                self.selection_info_label.setText("Selection: Full video")
                self.selection_info_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 10pt;")
            return
        
        # Calculate actual times when video is loaded
        start_seconds = int((start_pct / 100) * self.current_video_duration)
        end_seconds = int((end_pct / 100) * self.current_video_duration)
        duration = end_seconds - start_seconds
        
        # Update labels with time and percentage
        self.start_time_label.setText(f"{self.format_time(start_seconds)} ({start_pct}%)")
        self.end_time_label.setText(f"{self.format_time(end_seconds)} ({end_pct}%)")
        
        # Update selection info
        percentage = end_pct - start_pct
        
        if self.use_time_range_chk.isChecked():
            self.selection_info_label.setText(
                f"Selection: {self.format_time(duration)} ({percentage}% of video)"
            )
            self.selection_info_label.setStyleSheet("color: #2f81f7; font-weight: bold; font-size: 10pt;")
        else:
            self.selection_info_label.setText("Selection: Full video")
            self.selection_info_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 10pt;")

    def update_video_duration(self, video_path):
        """Update slider ranges based on video duration"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            duration = int(total_frames / fps) if fps else 0
            cap.release()
            
            if duration > 0:
                self.current_video_duration = duration
                
                # Update sliders with 100 steps (0-100 representing 0%-100% of video)
                self.range_slider.setRange(0, 100)
                
                # Keep existing slider values (don't reset user's choice)
                # Only update the display labels
                
                # Update labels
                self.video_duration_label.setText(
                    f"Video duration: {self.format_time(duration)} ({duration}s)"
                )
                self.video_duration_label.setStyleSheet("color: #4CAF50; font-style: italic;")
                
                # Enable controls if checkbox is checked
                if self.use_time_range_chk.isChecked():
                    self.range_slider.setEnabled(True)
                    self.first_5min_btn.setEnabled(True)
                    self.last_5min_btn.setEnabled(True)
                    self.last_10min_btn.setEnabled(True)
                    self.middle_btn.setEnabled(True)
                    self.full_video_btn.setEnabled(True)
                
                self.update_selection_info()
                return True
            else:
                self.current_video_duration = 0
                self.video_duration_label.setText("Could not determine video duration")
                self.video_duration_label.setStyleSheet("color: #f44336; font-style: italic;")
                return False
                
        except Exception as e:
            self.current_video_duration = 0
            self.video_duration_label.setText(f"Error reading video: {e}")
            self.video_duration_label.setStyleSheet("color: #f44336; font-style: italic;")
            return False

    def set_slider_preset(self, preset_type):
        """Set quick preset time ranges using sliders"""
        if self.current_video_duration == 0:
            self.append_log("⚠️ No video loaded")
            return
        
        duration = self.current_video_duration
        
        if preset_type == "first_5":
            # First 5 minutes or entire video if shorter
            end_seconds = min(300, duration)
            start_pct = 0
            end_pct = int((end_seconds / duration) * 100)
        elif preset_type == "last_5":
            # Last 5 minutes
            start_seconds = max(0, duration - 300)
            start_pct = int((start_seconds / duration) * 100)
            end_pct = 100
        elif preset_type == "last_10":
            # Last 10 minutes
            start_seconds = max(0, duration - 600)
            start_pct = int((start_seconds / duration) * 100)
            end_pct = 100
        elif preset_type == "middle":
            # Middle third of video
            third = duration / 3
            start_pct = int((third / duration) * 100)
            end_pct = int((2 * third / duration) * 100)
        elif preset_type == "full":
            start_pct = 0
            end_pct = 100
        else:
            return
        
        self.range_slider.setRangeValues(start_pct, end_pct)

        start_time = int((start_pct / 100) * duration)
        end_time = int((end_pct / 100) * duration)
        self.append_log(f"✅ Preset '{preset_type}': {self.format_time(start_time)} to {self.format_time(end_time)}")


    def _position_preview_window(self):
        """Place the preview window just to the right of the main GUI."""
        if self.preview_window is None:
            return
        try:
            g = self.frameGeometry()
            x = g.x() + g.width() + 8
            y = g.y()
            # Keep it on-screen: if it would overflow the screen, clamp.
            screen = QApplication.primaryScreen()
            if screen is not None:
                avail = screen.availableGeometry()
                pw = self.preview_window.width() or 720
                if x + pw > avail.right():
                    x = max(avail.left(), avail.right() - pw)
            self.preview_window.move(x, y)
        except Exception:
            pass

    def _on_live_preview_toggled(self, checked):
        """Open/close the separate preview window. Applies live to a running job."""
        if checked:
            if self.preview_window is None:
                # Top-level window (no parent) so it's freely movable and not
                # clipped to the main window; we position it ourselves.
                self.preview_window = DetectionPreviewWindow()
                self.preview_window.closed.connect(
                    lambda: self.live_preview_checkbox.setChecked(False)
                )
            self._position_preview_window()
            self.preview_window.show()
            self.preview_window.raise_()
            self.preview_window.activateWindow()
        else:
            if self.preview_window is not None:
                self.preview_window.hide()
        self._preview_enabled = checked
        # Every kind of run that can be in flight: the pipeline, an on-demand
        # Analyze run, and an Analyze run started in an open timeline viewer.
        # Ticking the box mid-run starts showing frames in any of them without
        # waiting for the next one.
        for attr in ("worker", "_signal_worker", "timeline_window"):
            running = getattr(self, attr, None)
            if running is not None:
                try:
                    running.preview_enabled = checked
                except (RuntimeError, AttributeError):
                    pass   # viewer's C++ side already deleted, or no such worker

    @Slot(object, object, int)
    def on_preview_frame(self, frame_bgr, boxes, sec):
        """Draw a live detection frame (BGR ndarray + normalised boxes)."""
        if not self.live_preview_checkbox.isChecked() or self.preview_window is None:
            return
        try:
            from PySide6.QtGui import QImage, QPainter, QPen, QColor, QFont, QPixmap
            import numpy as np

            # Ensure a contiguous uint8 BGR array, then convert to RGB
            frame_bgr = np.ascontiguousarray(frame_bgr)
            h, w = frame_bgr.shape[:2]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            rgb = np.ascontiguousarray(rgb)
            qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()
            pix = QPixmap.fromImage(qimg)

            painter = QPainter(pix)
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(QColor(0, 230, 90), 2)
            painter.setFont(QFont("Arial", 9, QFont.Bold))
            for item in boxes or []:
                name, nx, ny, nw, nh, conf = item
                rx, ry, rw, rh = int(nx * w), int(ny * h), int(nw * w), int(nh * h)
                painter.setPen(pen)
                painter.drawRect(rx, ry, rw, rh)
                label = f"{name} {conf:.2f}"
                painter.fillRect(rx, max(0, ry - 14), 8 + len(label) * 6, 14, QColor(0, 0, 0, 160))
                painter.setPen(QColor(0, 255, 120))
                painter.drawText(rx + 3, max(10, ry - 3), label)
            painter.end()

            n = len(boxes or [])
            cap = f"t={sec//60:d}:{sec%60:02d}"
            if n:
                cap += f"  •  {n} object{'s' if n != 1 else ''}"
            self.preview_window.set_frame(pix, caption=cap)
        except Exception as e:
            print(f"⚠️ preview draw error: {e}")

    def run_pipeline(self, report_only: bool = False):
        from pipeline import run_highlighter
        """Start the pipeline processing (UPDATED for multi-file).

        ``report_only`` scores and reports without encoding anything. Tuning
        weights is cheap — detection is cached — but re-rendering a highlight
        to find out what the new weights did is not, and that cost is what
        makes trying a setting feel expensive.
        """
        self._report_only = bool(report_only)
        video_paths = self.get_file_list()
        
        if not video_paths:
            self.append_log("⚠️ No videos selected!")
            return

        # Check if all files exist
        missing_files = [p for p in video_paths if not os.path.exists(p)]
        if missing_files:
            self.append_log(f"⚠️ Video file(s) not found:")
            for f in missing_files:
                self.append_log(f"  - {f}")
            return

        if self.worker and self.worker.isRunning():
            self.append_log("⚠️ Pipeline already running!")
            return
        
        # --- Validate scoring points ---
        scene_points = int(self.spin_scene_points.value())
        motion_event_points = int(self.spin_motion_event_points.value())
        motion_peak_points = int(self.spin_motion_peak.value())
        audio_peak_points = int(self.spin_audio_peak.value())
        loudness_burst_points = int(self.spin_loudness_burst.value())
        
        # Object points only count if objects are configured
        highlight_objects = [s.strip() for s in self.objects_input.text().split(",") if s.strip()]
        object_points = int(self.spin_object.value()) if highlight_objects else 0
        
        # Action points only count if actions are configured
        interesting_actions = [s.strip() for s in self.actions_input.text().split(",") if s.strip()]
        action_points = int(self.spin_action.value()) if interesting_actions else 0
        
        # Transcript and keyword points only count if transcript is enabled
        use_transcript = self.transcript_checkbox.isChecked()
        keyword_points = int(self.spin_keyword_points.value()) if use_transcript else 0
        transcript_points = int(self.spin_transcript_points.value()) if use_transcript else 0
        
        beginning_points = int(self.spin_beginning_points.value())
        ending_points = int(self.spin_ending_points.value())
        
        # Expressions only count when a class is chosen, for the same reason
        # objects need a class list: the scan is skipped otherwise, so counting
        # the weight would promise points nothing can earn.
        face_points = (int(self.spin_face_expression.value())
                       if self.selected_face_labels() else 0)

        total_points = (scene_points + motion_event_points + motion_peak_points + 
                       audio_peak_points + loudness_burst_points +
                       keyword_points + transcript_points + 
                       beginning_points + ending_points + object_points + action_points
                       + face_points)
        
        if total_points == 0:
            self.append_log("❌ ERROR: All scoring points are set to 0!")
            self.append_log("")
            self.append_log("Please configure at least one scoring point:")
            self.append_log("  • Scene points")
            self.append_log("  • Motion event points")
            self.append_log("  • Motion peak points")
            self.append_log("  • Audio peak points")
            self.append_log("  • Object points")
            self.append_log("  • Action points")
            if use_transcript:
                self.append_log("  • Keyword points (transcript enabled)")
                self.append_log("  • Transcript points (transcript enabled)")
            else:
                self.append_log("")
                self.append_log("Note: Transcript is disabled - keyword and transcript")
                self.append_log("points are not counted. Enable transcript to use them.")
            return

        exact_duration_val = int(self.spin_exact_duration.value())
        exact_duration = exact_duration_val if exact_duration_val > 0 else None
        
        # Get output base name from input
        output_base = self.output_input.text().strip() or "highlight.mp4"
        
        # If multiple files, we'll handle output paths per file in the pipeline
        # For single file, use the same directory as source video
        if len(video_paths) == 1:
            # Single file - use the same directory as source video
            source_dir = os.path.dirname(video_paths[0])
            output_file = os.path.join(source_dir, output_base)
        else:
            # Multiple files - the pipeline will handle appending '_highlight' to each
            # But we still want to use the output_base as a template
            output_file = output_base

        exact_duration_val = int(self.spin_exact_duration.value())
        exact_duration = exact_duration_val if exact_duration_val > 0 else None

        # Helper function to get non-empty lists
        def get_list_from_input(input_field):
            text = input_field.text().strip()
            if not text:
                return None
            items = [s.strip() for s in text.split(",") if s.strip()]
            return items if items else None
        
        highlight_objects = get_list_from_input(self.objects_input)
        interesting_actions = get_list_from_input(self.actions_input)
        use_transcript = self.transcript_checkbox.isChecked()
        search_keywords = get_list_from_input(self.search_keywords_input) if use_transcript else []
        # Avoid: pull flagged identities from the shared face bank
        avoid_bank = self._get_face_bank()
        avoid_ids = avoid_bank.avoided_ids() if avoid_bank else []

        config = {
            "scene_points": int(self.spin_scene_points.value()),
            "motion_event_points": int(self.spin_motion_event_points.value()),
            "motion_peak_points": int(self.spin_motion_peak.value()),
            "audio_peak_points": int(self.spin_audio_peak.value()),
            "loudness_burst_points": int(self.spin_loudness_burst.value()),
            "keyword_points": int(self.spin_keyword_points.value()),
            "transcript_points": int(self.spin_transcript_points.value()),
            "beginning_points": int(self.spin_beginning_points.value()),
            "ending_points": int(self.spin_ending_points.value()),
            "beginning_seconds": int(self.spin_beginning_seconds.value()),
            "ending_seconds": int(self.spin_ending_seconds.value()),
            "object_points": int(self.spin_object.value()),
            "action_points": int(self.spin_action.value()),
            "face_expression_points": int(self.spin_face_expression.value()),
            "face_expression_labels": self.selected_face_labels(),
            "clip_time": int(self.spin_clip_time.value()),
            "coverage": self.slider_coverage.value() / 100.0,
            "report_only": bool(getattr(self, "_report_only", False)),
            "max_duration": int(self.spin_max_duration.value()),
            "exact_duration": exact_duration,
            "multi_signal_boost": 1.2,
            "min_signals_for_boost": 2,
            "keep_temp": self.keep_temp_chk.isChecked(),
            "render_mode": self.render_mode_combo.currentData(),
            "output_file": output_file,
            "highlight_objects": highlight_objects,
            "interesting_actions": interesting_actions,
            "actions_require_objects": self.actions_require_objects_chk.isChecked(),
            "use_transcript": use_transcript,
            "transcript_model": self.transcript_model_combo.currentText(),
            "transcript_source_lang": self.transcript_source_lang.currentText(),
            "search_keywords": search_keywords,
            "create_subtitles": self.subtitles_checkbox.isChecked() and use_transcript,
            # The spoken language has one home: Transcript Settings.
            "source_lang": self.transcript_source_lang.currentText(),
            "target_lang": self.subtitle_target_lang.currentText(),
            "frame_skip": int(self.frame_skip_spin.value()),
            "object_frame_skip": int(self.obj_frame_skip_spin.value()),
            "yolo_type": self.object_detector_choice()[0],
            "yolo_model_size": self.yolo_model_combo.currentData(),
            "yolo_custom_model_path": self.object_detector_choice()[1] or getattr(self, "_custom_pose_model", None),
            "sample_rate": int(self.sample_rate_spin.value()),
            "auto_min_clip": float(self.spin_auto_min_clip.value()),
            "auto_max_clip": float(self.spin_auto_max_clip.value()),
            "auto_merge_gap": float(self.spin_auto_merge_gap.value()),
            "draw_object_boxes": self.bbox_objects_chk.isChecked(),
            "write_highlight_report": self.why_report_chk.isChecked(),
            "draw_action_labels": self.bbox_actions_chk.isChecked(),
            "action_backend": self.action_backend_combo.currentData(),
            "r3d_model": self.r3d_model_combo.currentData(),
            "avoid_enabled": self.avoid_face_recognition_chk.isChecked() and bool(avoid_ids),
            "avoid_method": getattr(self, "_avoid_method", "skip"),
            "avoid_identity_ids": avoid_ids,
            "avoid_manual_ranges": self._get_manual_avoid_ranges(),
            "face_db_path": "./cache/face_db.json",
            "force_reprocess": self.force_reprocess_checkbox.isChecked(),
        }

        # Remove None values
        config = {k: v for k,v in config.items() if v is not None}

        # Clear previous logs
        self.log_output.clear()
        self._show_progress(True)
        self.append_log("=== Starting Video Highlighter Pipeline ===")
        self.append_log(f"📁 Input: {video_paths}")
        self.append_log(f"📁 Output: {config.get('output_file', 'highlight.mp4')}")
        if config.get('draw_object_boxes') or config.get('draw_action_labels'):
            self.append_log("🎨 Bounding box visualization enabled for temp files")
        self.append_log("")

        if self.use_time_range_chk.isChecked() and self.current_video_duration > 0:
            start_pct = self.range_slider.start() / 100
            end_pct = self.range_slider.end() / 100
            config["use_time_range"] = True
            config["range_start"] = int(start_pct * self.current_video_duration)
            config["range_end"] = int(end_pct * self.current_video_duration)
        else:
            config["use_time_range"] = False

        # UI state changes
        self.process_progress_bar.setVisible(True)
        self.process_progress_bar.setRange(0, 100)
        self.process_progress_bar.setValue(0)
        self.download_progress_bar.setVisible(False)
        # The pipeline reveals this again on the first batch update; a single-file
        # run must not inherit the last batch's counter.
        self.hide_batch_progress()
        self.task_label.setText("🚀 Initializing...")
        self.run_btn.setText("⏸ Pause")
        self.run_btn.setStyleSheet("QPushButton { background-color: #ff8c00; color: white; font-weight: bold; padding: 8px; }")
        self._set_analyze_buttons_enabled(False)   # no on-demand run while a pipeline runs
        self.cancel_btn.setEnabled(True)

        # Disable form inputs during processing
        self.file_list.setEnabled(False)
        self.output_input.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.remove_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

        # Create and start worker
        self.worker = Worker(video_paths, config)
        self._preview_enabled = self.live_preview_checkbox.isChecked()
        self.worker.preview_enabled = self._preview_enabled
        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self.update_pipeline_progress)
        self.worker.finished.connect(self.pipeline_done)
        self.worker.cancelled.connect(self.pipeline_cancelled)
        self.worker.preview.connect(self.on_preview_frame)
        self.worker.timeline_requested.connect(self.on_timeline_requested)
        
        # Start status checking timer
        self.status_timer.start(100)  # Check every 100ms
        
        self.worker.start()

    def cancel_pipeline(self):
        """Cancel the running pipeline or download"""
        # Check if download is running
        if hasattr(self, 'download_worker') and self.download_worker and self.download_worker.isRunning():
            self.append_log("\n⏹️ === CANCELLATION REQUESTED ===")
            self.append_log("⏹️ Stopping download...")
            self.task_label.setText("⏹️ Cancelling download...")
            self.cancel_btn.setEnabled(False)
            self.cancel_btn.setText("Cancelling...")
            worker = self.download_worker
            worker.cancel()
            QTimer.singleShot(10000, lambda: self.force_download_cleanup(worker))
            return
        
        # Check if pipeline is running
        if self.worker and self.worker.isRunning():
            self.append_log("\n⏹️ === CANCELLATION REQUESTED ===")
            self.append_log("⏹️ Stopping pipeline...")
            self.task_label.setText("⏹️ Cancelling pipeline...")
            self.cancel_btn.setEnabled(False)
            self.cancel_btn.setText("Cancelling...")
            self.worker.cancel()
            QTimer.singleShot(10000, self.force_worker_cleanup)
            return

        # Check if an on-demand signal run is going
        if self._signal_worker and self._signal_worker.isRunning():
            self.append_log("\n⏹️ === CANCELLATION REQUESTED ===")
            self.append_log("⏹️ Stopping on-demand run...")
            self.task_label.setText("⏹️ Cancelling on-demand run...")
            self.cancel_btn.setText("Cancelling...")
            self._signal_worker.cancel()
            return

        # Nothing is running
        self.append_log("⚠️ Nothing to cancel - no active process")

    def _make_analyze_button(self, kind, label, tooltip):
        """A small 'run this one signal on demand' button, registered so the
        pipeline can grey it out while a full run (or another on-demand run) is
        active."""
        btn = QPushButton(label)
        btn.setToolTip(tooltip)
        btn.clicked.connect(lambda _=False, k=kind: self.start_signal_run(k))
        # A list, not one button per kind. Composition has two Run buttons — one
        # beside the rules editor where rules are changed, one in the signals
        # list beside every other on-demand run — and keying by kind alone let
        # the second registration drop the first, leaving a live button during a
        # run that is meant to disable them all.
        self._analyze_buttons.setdefault(kind, []).append(btn)
        return btn

    def _points_group(self, title, rows):
        """One signal's scoring rows, in their own titled box.

        ``rows`` is ``((label, field), ...)`` — the same pairs a form layout
        takes, so a row moves between groups by moving one tuple.
        """
        box = QGroupBox(title)
        form = QFormLayout()
        form.setContentsMargins(8, 4, 8, 4)
        form.setSpacing(4)
        for label, field in rows:
            form.addRow(label, field)
        box.setLayout(form)
        return box

    def selected_face_labels(self):
        """The expression classes chosen to score, lowercased for the pipeline."""
        return [name for name, act in
                getattr(self, "_face_label_actions", {}).items()
                if act.isChecked()]

    def _update_face_labels_button(self, *_args):
        """Keep the button reading as what it will actually score.

        Named rather than counted while the list is short: "happy, surprise" is
        the setting itself, where "2 selected" makes the user open the menu to
        find out what they chose. The empty case has to say the consequence —
        points with nothing selected score nothing at all, and a button reading
        "none" would look like a valid state.
        """
        chosen = self.selected_face_labels()
        self.btn_face_labels.setText(
            ", ".join(chosen) if chosen else "pick expressions…")

    def _rules_run_row(self):
        """The composition Run button plus a word on what it will actually do.

        No points spinbox: composed events are not scored the way the signals
        above are, so the row carries the button and a note instead of a number
        nobody would set.
        """
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(self._make_analyze_button(
            "composition", "Apply rules", chr(10).join([
                "Run the saved composition rules over every video in the",
                "list and cache the events.",
                "",
                "Fetches what the ticked rules read and the cache lacks:",
                "signal rules measure the file directly, and a spatial rule",
                "starts a detection pass for the classes it names. Both are",
                "cached, so re-running after a threshold edit is seconds.",
                "",
                "Safe to run repeatedly: previous results for these rules",
                "are replaced, not stacked.",
            ])))
        note = QLabel("edit them in Advanced → Composition Rules")
        note.setStyleSheet("color: #888; font-size: 9pt;")
        h.addWidget(note)
        h.addStretch(1)
        return w

    def _points_row_with_button(self, spin, kind, label, tooltip):
        """Wrap a scoring-point spinbox and its on-demand Run button into one
        form-row field: [spinbox] [Run button] [stretch]."""
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(spin)
        h.addWidget(self._make_analyze_button(kind, label, tooltip))
        h.addStretch(1)
        return w

    def _set_analyze_buttons_enabled(self, enabled):
        for btn in [b for group in getattr(self, "_analyze_buttons", {}).values()
                    for b in group]:
            btn.setEnabled(enabled)

    def start_signal_run(self, kind):
        """Run one analysis signal (objects / actions / transcript / subtitles /
        motion / audio) over every video in the list, folding each result into
        that video's cache. No highlights are cut — this is the main-window twin
        of the timeline viewer's Analyze panel."""
        if self.worker and self.worker.isRunning():
            self.append_log("⚠️ A pipeline run is active — let it finish first.")
            return
        if self._signal_worker and self._signal_worker.isRunning():
            self.append_log("⚠️ An on-demand run is already going.")
            return

        video_paths = self.get_file_list()
        if not video_paths:
            self.append_log("⚠️ No videos in the list.")
            return
        missing = [p for p in video_paths if not os.path.exists(p)]
        if missing:
            self.append_log("⚠️ Video file(s) not found:")
            for f in missing:
                self.append_log(f"  - {f}")
            return

        # Per-kind params + validation.
        params = {}
        if kind == "objects":
            objs = [s.strip() for s in self.objects_input.text().split(",") if s.strip()]
            if not objs:
                self.append_log("⚠️ Type at least one object class first (e.g. person, car).")
                return
            params["objects"] = objs
        elif kind == "actions":
            # Blank = detect every action (same as the timeline viewer).
            params["actions"] = [s.strip() for s in self.actions_input.text().split(",") if s.strip()]
        elif kind == "transcript":
            params["language"] = self.transcript_source_lang.currentText()
        elif kind == "subtitles":
            # One spoken language, from Transcript Settings. run_subtitles takes
            # the .srt's source from the transcript it actually used, so a reused
            # cached one is labelled with its own language rather than this.
            params["language"] = self.transcript_source_lang.currentText()
            params["target_lang"] = self.subtitle_target_lang.currentText()

        # UI state — reuse the pipeline's progress row.
        self.log_output.clear()
        self._show_progress(True)
        self.process_progress_bar.setVisible(True)
        self.process_progress_bar.setRange(0, 100)
        self.process_progress_bar.setValue(0)
        self.download_progress_bar.setVisible(False)
        self.hide_batch_progress()
        self.task_label.setText(f"🚀 {kind.title()} (on demand)…")
        self._set_analyze_buttons_enabled(False)
        self.run_btn.setEnabled(False)   # no full run while an on-demand run goes
        self.cancel_btn.setEnabled(True)

        self._signal_run_paths = list(video_paths)   # for live-refreshing an open viewer
        self._signal_worker = SignalRunWorker(kind, video_paths, params)
        self._preview_enabled = self.live_preview_checkbox.isChecked()
        self._signal_worker.preview_enabled = self._preview_enabled
        self._signal_worker.log.connect(self.append_log)
        self._signal_worker.progress.connect(self.update_pipeline_progress)
        self._signal_worker.finished.connect(self._signal_run_finished)
        self._signal_worker.preview.connect(self.on_preview_frame)
        self._signal_worker.start()

    @Slot(str)
    def _signal_run_finished(self, summary):
        if summary:
            self.append_log(f"✅ {summary}")
        self._set_analyze_buttons_enabled(True)
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("Cancel")
        self.task_label.setText("Ready")
        self.process_progress_bar.setValue(100)
        self._signal_worker = None

        # If a timeline viewer is open for one of the videos we just ran, refresh
        # it live so the new signal appears without reopening.
        tw = getattr(self, 'timeline_window', None)
        if tw is not None:
            try:
                if getattr(tw, 'video_path', None) in getattr(self, '_signal_run_paths', []):
                    tw.refresh_from_disk()
                    self.append_log("🔄 Refreshed the open timeline viewer.")
            except RuntimeError:
                self.timeline_window = None   # underlying window was destroyed
            except Exception as e:
                self.append_log(f"⚠️ Could not refresh timeline viewer: {e}")

    def toggle_run(self):
        """Run / Pause / Resume - single button"""
        # Not running → start pipeline
        if not self.worker or not self.worker._is_running:
            self.run_pipeline(report_only=False)
            return

        # Running and not paused → pause
        if not self.worker.is_paused():
            self.worker.pause()
            self.run_btn.setText("▶ Resume")
            self.run_btn.setStyleSheet("QPushButton { background-color: #2f81f7; color: white; font-weight: bold; padding: 8px; }")
            self.task_label.setText("⏸ Paused")
            self.task_label.setStyleSheet("color: #ff8c00; font-weight: bold;")
            self.append_log("⏸ Pipeline paused")
            return

        # Paused → resume
        self.worker.resume()
        self.run_btn.setText("⏸ Pause")
        self.run_btn.setStyleSheet("QPushButton { background-color: #ff8c00; color: white; font-weight: bold; padding: 8px; }")
        self.run_btn.setEnabled(True)  # keep enabled for pause

    def force_download_cleanup(self, worker=None):
        """Safety net (fires ~10s after a cancel request) in case the worker
        never emitted its finished/cancelled signal — e.g. it's stuck in a
        non-cancellable subprocess. Runs download_cleanup() unconditionally so
        the Download button always comes back."""
        worker = worker or getattr(self, 'download_worker', None)
        # A newer download may have replaced this worker in the meantime; don't
        # touch it — the new download owns the UI now.
        if worker is not getattr(self, 'download_worker', None):
            return
        if worker and worker.isRunning():
            self.append_log("⚠️ Forcing download termination...")
            worker.terminate()
            worker.wait(3000)
        self.download_cleanup()

    def force_worker_cleanup(self):
        """Force cleanup if worker doesn't stop gracefully"""
        if self.worker and self.worker.isRunning():
            self.append_log("⚠️ Forcing pipeline termination...")
            self.worker.terminate()
            self.worker.wait(3000)  # Wait up to 3 seconds
            self.pipeline_cleanup()
            self._show_progress(False)

    def update_analyzed_counter(self):
        """Refresh the analyzed-videos counter label (lifetime + this session)."""
        total = analysis_stats.get_analyzed_count()
        self.analyzed_counter_label.setText(
            f"📈 Analyzed videos: {total} (session: {self.session_analyzed_count})"
        )

    def pipeline_done(self, output_file):
        """Handle pipeline completion"""
        self.status_timer.stop()
        was_cancelled = bool(self.worker and self.worker.is_cancelled())
        
        if output_file and not was_cancelled:
            self.append_log(f"\n✅ === PIPELINE COMPLETED SUCCESSFULLY ===")
            
            # Handle both single file (string) and multiple files (list of tuples)
            if isinstance(output_file, list):
                self.append_log(f"🎬 Processed {len(output_file)} videos:")
                
                highlight_files = []  # Track valid highlight files
                
                for item in output_file:
                    # Handle tuple format: (input_path, output_path)
                    if isinstance(item, tuple):
                        input_path, result_path = item
                        file = result_path
                    else:
                        file = item
                    
                    if file:
                        self.append_log(f"   • {file}")
                        highlight_files.append(file)  # Add to list for combining
                        
                        # Check for additional files for each video
                        base_name = os.path.splitext(file)[0]
                        srt_file = f"{base_name}_{self.subtitle_target_lang.currentText()}.srt"
                        transcript_file = f"{base_name}_transcript.txt"
                        
                        if os.path.exists(srt_file): 
                            self.append_log(f"     📝 Subtitle: {srt_file}")
                        if os.path.exists(transcript_file): 
                            self.append_log(f"     📄 Transcript: {transcript_file}")
                    else:
                        self.append_log(f"   ❌ Failed to process")
                
                # Combine highlights if enabled and we have multiple files
                if len(highlight_files) > 1 and self.auto_combine_chk.isChecked():
                    self.append_log("")
                    self.append_log("=" * 60)
                    
                    # Auto-generate combined output name in same directory as first highlight
                    first_video_dir = os.path.dirname(highlight_files[0])
                    combined_output = os.path.join(first_video_dir, "all_highlights_combined.mp4")
                    
                    # Call the combine method
                    combined_file = self.combine_highlights(highlight_files, combined_output)
                    
                    if combined_file:
                        self.append_log(f"🎉 All highlights combined into: {combined_file}")
                        
                        # Calculate and display total duration
                        try:
                            cap = cv2.VideoCapture(combined_file)
                            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                            duration = total_frames / fps if fps else 0
                            cap.release()
                            self.append_log(f"   Total duration: {int(duration//60)}:{int(duration%60):02d} ({duration:.1f}s)")
                        except Exception as e:
                            self.append_log(f"   (Could not determine duration: {e})")
                    
                    self.append_log("=" * 60)
                
            else:
                # Single file
                self.append_log(f"🎬 Output saved to: {output_file}")
                
                # Check for additional files
                base_name = os.path.splitext(output_file)[0]
                srt_file = f"{base_name}_{self.subtitle_target_lang.currentText()}.srt"
                transcript_file = f"{base_name}_transcript.txt"
                
                if os.path.exists(srt_file): 
                    self.append_log(f"📝 Subtitle file: {srt_file}")
                if os.path.exists(transcript_file): 
                    self.append_log(f"📄 Transcript file: {transcript_file}")
                
            newly_analyzed = len(highlight_files) if isinstance(output_file, list) else 1
            if newly_analyzed:
                self.session_analyzed_count += newly_analyzed
                total_analyzed = analysis_stats.increment_analyzed(newly_analyzed)
                self.update_analyzed_counter()
                self.append_log(
                    f"📈 Analyzed videos: +{newly_analyzed} this run — lifetime total: {total_analyzed}"
                )

            self.task_label.setText("✅ Complete!")
            self.task_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        elif not was_cancelled:
            self.append_log("\n⚠️ === PIPELINE COMPLETED WITH ERRORS ===")
            self.append_log("❌ No output file was generated. Check the log for errors.")
            self.task_label.setText("❌ Failed")
            self.task_label.setStyleSheet("color: #f44336; font-weight: bold;")
        
        # Feed analysis data to LLM chat
        if hasattr(self, 'llm_chat'):
            try:
                from modules.video_cache import VideoAnalysisCache
                cache = VideoAnalysisCache()
                video_path = self.get_file_list()[0] if self.get_file_list() else ""
                
                # Try loading from cache
                config = self.build_pipeline_config()
                cache_data = cache.load(video_path, params=None)  # load latest
                
                if cache_data:
                    self.llm_chat.set_analysis_data(cache_data, video_path)
                    self.append_log("🤖 LLM chat context updated with analysis data")
            except Exception as e:
                self.append_log(f"⚠️ Could not update LLM context: {e}")

        # feed cache to bot after finished pipeline
        if hasattr(self, 'llm_chat') and output_file:
            try:
                video_paths = self.get_file_list()
                video_path = video_paths[0] if video_paths else ""
                if video_path and os.path.exists(video_path):
                    config = self.build_pipeline_config()
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    video_duration = total_frames / fps if fps else 0
                    cap.release()
                    cfg_data = {}
                    if os.path.exists(CONFIG_FILE):
                        with open(CONFIG_FILE, "r", encoding="utf-8") as _f:
                            cfg_data = yaml.safe_load(_f) or {}
                    analysis_params = build_analysis_cache_params(
                        gui_config=config, config=cfg_data,
                        sample_rate=int(self.sample_rate_spin.value()),
                        video_duration=video_duration,
                    )
                    cache = VideoAnalysisCache(cache_dir=config.get("cache_dir", "./cache"))
                    cache_data = cache.load(video_path, params=analysis_params)
                    if cache_data:
                        self.llm_chat.set_analysis_data(cache_data, video_path)
                        self.append_log("🤖 LLM chat context updated with analysis data")
            except Exception as e:
                self.append_log(f"⚠️ Could not update LLM context: {e}")

        self.pipeline_cleanup()

    def pipeline_cancelled(self):
        """Handle pipeline cancellation"""
        self.status_timer.stop()
        self.append_log("\n⏹️ === PIPELINE CANCELLED ===")
        self.task_label.setText("⏹️ Cancelled")
        self.task_label.setStyleSheet("color: #ff9800; font-weight: bold;")
        self.pipeline_cleanup()

    def pipeline_cleanup(self):
        """Clean up UI state after pipeline completion/cancellation"""
        # Hide progress bar
        self.process_progress_bar.setVisible(False)
        # (Optional) keep download bar hidden too
        self.download_progress_bar.setVisible(False)

        
        # Re-enable controls
        self.run_btn.setText("Run Highlighter")
        self.run_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        self.run_btn.setEnabled(True)
        self._set_analyze_buttons_enabled(True)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("Cancel")

        # Re-enable file inputs
        self.file_list.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.remove_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)
        self.output_input.setEnabled(True)

        # Reset task label style
        QTimer.singleShot(5000, lambda: self.task_label.setStyleSheet("color: #666; font-weight: bold;"))
        
        # Clean up worker
        if self.worker:
            if self.worker.isRunning():
                self.worker.wait(1000)  # Wait up to 1 second
            self.worker = None

    def _get_manual_avoid_ranges(self):
        """Manual avoid ranges marked on the timeline.

        Prefer the live window (same process, always current). Fall back to the
        shared store so ranges still apply after the viewer is closed, or when
        they were marked in an earlier session. Safe if neither exists."""
        tw = getattr(self, "timeline_window", None)
        if tw is not None and hasattr(tw, "get_avoid_ranges"):
            try:
                return tw.get_avoid_ranges()
            except Exception:
                pass
        try:
            from modules.manual_avoid import load_ranges
            paths = self.get_file_list()
            if paths:
                return [list(r) for r in load_ranges(paths[0])]
        except Exception:
            pass
        return []

    def on_timeline_requested(self, video_path, analysis_data):
        """Open the timeline viewer at the pipeline's request.

        Runs on the main thread (the worker emits timeline_requested), so it is
        safe to build Qt widgets here. Reuses an already-open window for the same
        video rather than constructing a second one — each SignalTimelineWindow
        pins itself in memory and can't be torn down, so a fresh one per run
        leaks ~2.5GB. See open_timeline_viewer() for the same guard.
        """
        try:
            existing = getattr(self, 'timeline_window', None)
            if existing is not None:
                try:
                    if getattr(existing, 'video_path', None) == video_path:
                        existing.show()
                        existing.raise_()
                        existing.activateWindow()
                        self.append_log("📊 Reusing open timeline viewer.")
                        return
                except RuntimeError:
                    # Underlying C++ object was deleted — fall through.
                    self.timeline_window = None

            from signal_timeline_viewer import SignalTimelineWindow
            self.append_log(f"📊 Opening timeline viewer for: {os.path.basename(video_path)}")
            self.timeline_window = SignalTimelineWindow(video_path, analysis_data)
            self.timeline_window.show()
            self.llm_chat.set_timeline_window(self.timeline_window)
            self.llm_chat.set_video_path(video_path)
            self.llm_chat.load_cache_for_video(video_path)
        except Exception as e:
            self.append_log(f"❌ Failed to open timeline viewer: {e}")
    def _why_report_candidates(self) -> list:
        """Where a report for the current selection could be, newest first.

        Mirrors pipeline.py's own naming (`os.path.splitext(OUTPUT_FILE)[0] +
        "_why.html"`, falling back to the source video's stem) rather than
        guessing, so the button and the writer cannot disagree about the path.
        Several candidates because the output name is resolved differently for a
        single file than for a batch.
        """
        out = []
        video_paths = self.get_file_list()
        output_base = self.output_input.text().strip() or "highlight.mp4"

        for vp in video_paths:
            source_dir = os.path.dirname(vp)
            # single-file run: <source dir>/<output name>
            out.append(os.path.join(source_dir,
                                    os.path.splitext(output_base)[0] + "_why.html"))
            # batch run: the pipeline appends _highlight per input
            base = os.path.splitext(os.path.basename(vp))[0]
            out.append(os.path.join(source_dir, f"{base}_highlight_why.html"))
            # OUTPUT_FILE empty → pipeline falls back to the video's own stem
            out.append(os.path.splitext(vp)[0] + "_why.html")

        seen, uniq = set(), []
        for p in out:
            if p not in seen:
                seen.add(p)
                uniq.append(p)
        return [p for p in uniq if os.path.exists(p)]

    def open_why_report(self):
        """Open the newest "why these moments" report for the current selection."""
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        found = self._why_report_candidates()
        if not found:
            if not self.get_file_list():
                self.append_log("⚠️ Add a video first — the report sits next to its highlight.")
                return
            self.append_log(
                "⚠️ No report found yet. Run the highlighter with “Write a highlight "
                "report” enabled (Advanced tab) — it is written before the video is "
                "encoded, so it appears early in the run.")
            return

        newest = max(found, key=lambda p: os.path.getmtime(p))
        if QDesktopServices.openUrl(QUrl.fromLocalFile(newest)):
            self.append_log(f"📄 Opened report: {os.path.basename(newest)}")
        else:
            # No browser association is plausible on a stripped Windows install;
            # the path is more useful than a silent failure.
            self.append_log(f"⚠️ Could not open a browser. The report is at: {newest}")

    # ── AI summary of the highlight report ─────────────────────────────
    def _newest_why_report_json(self):
        """The JSON beside the newest report, or None with a logged reason."""
        found = self._why_report_candidates()
        if not found:
            self.append_log("⚠️ No highlight report yet — run the highlighter "
                            "first, the summary is written into that report.")
            return None
        newest = max(found, key=lambda p: os.path.getmtime(p))
        json_path = os.path.splitext(newest)[0] + ".json"
        if not os.path.exists(json_path):
            self.append_log(f"⚠️ {os.path.basename(newest)} has no .json beside "
                            "it, so there is nothing to summarise from.")
            return None
        return json_path

    def _ai_summary_settings(self):
        """The model a report is written with, as ``(backend, name-or-path)``."""
        entry = self._active_llm_model()
        if not entry:
            return ("ollama", "llama3")
        return (entry["backend"], entry["model"])

    def _llm_models(self):
        """Every configured model, oldest single-model setting folded in."""
        from PySide6.QtCore import QSettings
        from modules.llm_models import migrate, parse

        s = QSettings("VideoHighlighter", "Pro")
        models = parse(s.value("advisor/models"))
        if not models:
            models = migrate(models, s.value("advisor/backend"),
                             s.value("advisor/model"))
        return models

    def _save_llm_models(self, models, chosen=None):
        from PySide6.QtCore import QSettings
        from modules.llm_models import label_for, serialise

        s = QSettings("VideoHighlighter", "Pro")
        s.setValue("advisor/models", serialise(models))
        if chosen is not None:
            s.setValue("advisor/model_chosen", label_for(chosen))

    def _active_llm_model(self):
        from PySide6.QtCore import QSettings
        from modules.llm_models import active

        s = QSettings("VideoHighlighter", "Pro")
        return active(self._llm_models(), s.value("advisor/model_chosen"))

    def write_ai_summary(self, question=None, reading=False, model=None):
        """Generate the summary and put it in the report, then open it.

        ``reading`` swaps the task: the default asks what to change about the
        run, this asks what the footage looks like it is doing. Two different
        questions, kept in two fields on the report so a reader can always tell
        which one they are looking at.
        """
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import QApplication

        json_path = self._newest_why_report_json()
        if not json_path:
            return

        from modules import advisor
        entry = model or self._active_llm_model()
        backend = (entry or {}).get("backend", "ollama")
        model = (entry or {}).get("model", "llama3")
        mmproj = (entry or {}).get("mmproj")
        self.append_log(
            f"🤖 Asking {backend}/{model} to "
            f"{'read what happens in this cut' if reading else 'summarise the report'}… "
            "this takes a moment.")
        # The call blocks; without this the window looks hung rather than busy.
        self.ai_summary_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            llm = advisor.load_llm(backend, model, mmproj=mmproj)
            if llm is None:
                self.append_log(
                    f"⚠️ Could not reach {backend}/{model}. The report's findings "
                    "are there without it — only the summary needs a model.")
                return
            from modules.llm_models import label_for
            text = advisor.summarise_report_file(
                json_path, llm=llm, question=question or None, reading=reading,
                model_name=label_for(entry))
        except Exception as exc:
            self.append_log(f"⚠️ Summary failed: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.ai_summary_btn.setEnabled(True)

        if not text:
            self.append_log("⚠️ The model returned nothing; report unchanged.")
            return
        self.append_log(f"💡 {text}")
        html_path = os.path.splitext(json_path)[0] + ".html"
        if os.path.exists(html_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(html_path))

    def write_chapter_story(self, model=None):
        """Narrate every chapter of the newest report, then open it.

        One model call per chapter, so this is minutes rather than seconds and
        the log has to show progress — a silent wait of that length reads as a
        hang. The projector is asked for here, unlike everywhere else in the
        report: this is the one narration that sends pictures, and a model that
        can see the footage is the difference between describing what was said
        and describing what happened.
        """
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtWidgets import QApplication

        json_path = self._newest_why_report_json()
        if not json_path:
            return

        import json

        from modules import advisor, chapter_story
        from modules.llm_models import label_for

        try:
            with open(json_path, encoding="utf-8") as fh:
                chapters = json.load(fh).get("chapters") or []
        except Exception as exc:
            self.append_log(f"⚠️ Could not read the report: {exc}")
            return
        if not chapters:
            self.append_log("⚠️ This report has no chapters to tell.")
            return

        entry = model or self._active_llm_model()
        backend = (entry or {}).get("backend", "ollama")
        name = (entry or {}).get("model", "llama3")
        mmproj = (entry or {}).get("mmproj")
        self.append_log(
            f"📖 Asking {backend}/{name} to tell {len(chapters)} chapters — "
            "one call each, so this takes minutes, not seconds.")
        self.ai_summary_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            llm = advisor.load_llm(backend, name, mmproj=mmproj, vision=True)
            if llm is None:
                self.append_log(
                    f"⚠️ Could not reach {backend}/{name}. The chapters keep "
                    "their measurements — only the telling needs a model.")
                return

            def progress(line):
                # Straight to the user's pane rather than the debug log: this
                # is the only thing moving for the next several minutes.
                self.append_log(line)
                QApplication.processEvents()

            told = chapter_story.tell_report_file(
                json_path, llm=llm, model_name=label_for(entry),
                log_fn=progress)
        except Exception as exc:
            self.append_log(f"⚠️ Telling the chapters failed: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.ai_summary_btn.setEnabled(True)

        if not told:
            self.append_log("⚠️ The model returned nothing; report unchanged.")
            return
        self.append_log(f"📖 Told {told} of {len(chapters)} chapters.")
        html_path = os.path.splitext(json_path)[0] + ".html"
        if os.path.exists(html_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(html_path))

    def propose_composition_rule(self, model=None):
        """Draft a rule that would let the next run check something that was said.

        The one place in this app where a model's output becomes configuration,
        so the sequence is fixed: it proposes, `rule_proposal` rejects anything
        naming a class this video has no detections for, the user reads the YAML
        and says yes, and only then is the file written. There is no path that
        skips the middle two.
        """
        import json

        from PySide6.QtWidgets import (QApplication, QInputDialog, QMessageBox)

        from modules import advisor, rule_proposal
        from modules.app_paths import composition_rules_path
        from modules.llm_models import label_for

        json_path = self._newest_why_report_json()
        if not json_path:
            return
        try:
            with open(json_path, encoding="utf-8") as fh:
                report = json.load(fh)
        except Exception as exc:
            self.append_log(f"⚠️ Could not read the report: {exc}")
            return

        vocabulary = report.get("vocabulary") or {}
        classes = vocabulary.get("classes") or []
        if not classes:
            self.append_log(
                "⚠️ This report has no detections to build a rule from. Run "
                "object detection with a transcript first.")
            return

        # What the user wants tested. Seeded from the strongest gap so the
        # common case is a keypress, and editable because the gap is a
        # candidate rather than a question.
        # Seeded with the longest line among the gaps rather than the most
        # distinctive one. Keyness ranks "Mm-hmm." top on real footage — it is
        # genuinely concentrated and genuinely not a claim — and a dialog that
        # opens with it reads as the feature being broken. Length is a crude
        # proxy for "contains an assertion" and beats the alternative.
        gaps = vocabulary.get("gaps") or []
        seed = max((str(where.get("quote") or "")
                    for gap in gaps for where in (gap.get("chapters") or [])),
                   key=len, default="")
        claim, ok = QInputDialog.getMultiLineText(
            self, "Check something that was said",
            "Which claim should the next run try to check?\n"
            "A line from the transcript works best — the rule is built to "
            "confirm or contradict it.\n"
            f"Classes available in this video: {', '.join(classes)}",
            seed)
        if not ok or not claim.strip():
            return

        entry = model or self._active_llm_model()
        backend = (entry or {}).get("backend", "ollama")
        name = (entry or {}).get("model", "llama3")
        rules_path = composition_rules_path()
        self.append_log(f"🧩 Asking {backend}/{name} for a rule that would "
                        "check that…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            llm = advisor.load_llm(backend, name)
            if llm is None:
                self.append_log(f"⚠️ Could not reach {backend}/{name}.")
                return
            proposal = rule_proposal.propose(
                claim.strip(), classes, llm=llm,
                existing=rule_proposal.existing_rules(rules_path),
                gaps=gaps, claim_at=self._claim_second(report, claim),
                model_name=label_for(entry))
        except Exception as exc:
            self.append_log(f"⚠️ Rule proposal failed: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        if proposal is None:
            self.append_log(
                "⚠️ No usable rule came back. Either the claim cannot be "
                "expressed with the classes this video has, or the model named "
                "one it does not have — the debug log says which.")
            return

        answer = QMessageBox.question(
            self, "Add this rule?",
            f"<b>{proposal.label}</b><br><br>"
            f"{proposal.why}<br><br>"
            f"<pre>{proposal.as_yaml()}</pre>"
            f"Add it to your composition rules?<br>"
            f"<small>{rules_path}<br>The current file is backed up first. "
            f"Object detection must re-run before this can fire.</small>",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            self.append_log("ℹ️ Rule not added.")
            return

        try:
            rule_proposal.apply(rules_path, proposal,
                                video_path=(report.get("video") or {}).get("path"))
        except Exception as exc:
            self.append_log(f"⚠️ Could not write the rule: {exc}")
            return
        self.append_log(
            f"✅ Added '{proposal.name}' to {os.path.basename(rules_path)}. "
            "Re-run with object detection forced (a cached detection pass "
            "skips the composition engine), then tell the chapters again — the "
            "report will say whether it fired.")

    @staticmethod
    def _claim_second(report, claim):
        """Where in the video a claim was said, if it is a line of transcript.

        Matched on the stored text so the check can be filed under the chapter
        the sentence belongs to. Returns None when the user typed a question of
        their own rather than pasting a line, which is a perfectly good way to
        use this and simply carries no timestamp.
        """
        wanted = " ".join(str(claim or "").split()).lower()
        if not wanted:
            return None
        for chapter in (report.get("chapters") or []):
            for line in ((chapter.get("dialogue") or [])
                         + (chapter.get("quotes") or [])):
                text = " ".join(str(line.get("text") or "").split()).lower()
                if text and (text in wanted or wanted in text):
                    return float(line.get("start") or 0.0)
        return None

    def show_ai_summary_menu(self):
        from PySide6.QtWidgets import QMenu

        from modules.llm_models import label_for

        menu = QMenu(self)
        # First, because it is the one that answers "what is in this video"
        # rather than "what should I change" — and the one people reach for.
        act_read = menu.addAction("Read what happens in this cut…")
        # The chapter walk-through is the slow one — a call per chapter rather
        # than one for the report — so it says so on the menu rather than in a
        # log line the user reads after committing to the wait.
        act_story = menu.addAction("Tell the story, chapter by chapter… (slow)")
        # Closes the loop the other two open: they describe what was said, this
        # is how the next run gets a signal that can check it.
        act_rule = menu.addAction("Check something that was said…")
        act_wrong = menu.addAction("Something's wrong with this cut…")
        act_ask = menu.addAction("Ask a question about this cut…")
        act_chat = menu.addAction("Discuss in LLM chat")

        # Which model writes it. A submenu rather than a setting to go and
        # change, because the choice belongs to the run: reading a scene and
        # advising on weights suit different models, and picking one here is the
        # difference between switching and going to look for where switching
        # lives.
        models = self._llm_models()
        active_model = self._active_llm_model()
        read_with = {}
        if len(models) > 1:
            menu.addSeparator()
            sub = menu.addMenu("Read with…")
            for entry in models:
                item = sub.addAction(label_for(entry))
                item.setCheckable(True)
                item.setChecked(entry == active_model)
                item.setToolTip(f"{entry['backend']} · {entry['model']}")
                read_with[item] = entry

        menu.addSeparator()
        act_model = menu.addAction(
            f"Models: {label_for(active_model)}…" if models else "Add a model…")

        chosen = menu.exec(self.ai_summary_opts_btn.mapToGlobal(
            self.ai_summary_opts_btn.rect().bottomLeft()))
        if chosen in read_with:
            entry = read_with[chosen]
            self._save_llm_models(models, chosen=entry)
            self.write_ai_summary(reading=True, model=entry)
        elif chosen is act_read:
            self.write_ai_summary(reading=True)
        elif chosen is act_story:
            self.write_chapter_story()
        elif chosen is act_rule:
            self.propose_composition_rule()
        elif chosen is act_wrong:
            self._report_what_is_wrong()
        elif chosen is act_ask:
            self._ask_ai_summary_question()
        elif chosen is act_chat:
            self._discuss_report_in_chat()
        elif chosen is act_model:
            self._choose_ai_summary_model()

    def _report_what_is_wrong(self):
        """Ask what disappointed the user, then answer that.

        Without this the advisor can only list everything it noticed. Naming
        the complaint is what turns "give another signal a weight" into which
        one, and why that one.
        """
        import json

        from PySide6.QtWidgets import QInputDialog
        from modules.highlight_advice import CONCERNS, attach_advice

        json_path = self._newest_why_report_json()
        if not json_path:
            return

        labels = list(CONCERNS.values())
        picked, ok = QInputDialog.getItem(
            self, "What is wrong with this highlight?",
            "Pick the closest one — the report is re-read with that in mind:",
            labels, 0, False)
        if not ok:
            return
        concern = next(k for k, v in CONCERNS.items() if v == picked)

        try:
            with open(json_path, encoding="utf-8") as fh:
                report = json.load(fh)
            attach_advice(report, concern=concern)
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(report, fh, indent=1)

            from modules.highlight_report import render_html
            html_path = os.path.splitext(json_path)[0] + ".html"
            with open(html_path, "w", encoding="utf-8") as fh:
                fh.write(render_html(report))
        except Exception as exc:
            self.append_log(f"⚠️ Could not re-read the report: {exc}")
            return

        findings = report.get("advice") or []
        self.append_log(f"💡 Re-read with '{picked}' in mind — "
                        f"{len(findings)} suggestion(s):")
        for finding in findings[:3]:
            self.append_log(f"   • {finding.get('title', '')}")

        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices
        QDesktopServices.openUrl(QUrl.fromLocalFile(html_path))

    def _ask_ai_summary_question(self):
        """A typed question is about the footage, and goes to the reader.

        It went to the advisor, whose system prompt opens "you help someone tune
        a video highlight tool" and whose rules push every answer toward a weight
        to change. So a question about what happens in a video was answered by
        the persona hired to talk about settings, and came back sounding like it
        had refused — when it had simply been asked by the wrong one of the two.

        Tuning questions have their own item on this menu.
        """
        from PySide6.QtWidgets import QInputDialog

        question, ok = QInputDialog.getText(
            self, "Ask about this cut",
            "What would you like to know about this video?\n"
            "The model answers from what the run measured — the marks, their "
            "order, and how often the video repeats them.",
            text="What does the pattern across these clips look like to you?")
        if ok and question.strip():
            self.write_ai_summary(question.strip(), reading=True)

    def _choose_ai_summary_model(self):
        """The report's models, on one screen — the chat panel's form, listed.

        Was a chain of four prompts with no view of what was already configured
        and no way back from the second one.
        """
        from PySide6.QtWidgets import QApplication

        from modules.llm_models import label_for
        from modules.ui.model_dialog import ModelDialog

        models = self._llm_models()
        active = self._active_llm_model()
        # Building it asks the Ollama server what it holds, so the dialog can
        # offer the names instead of asking the user to remember them. That is
        # a request with a timeout, and a server that is not running spends all
        # of it — once per session, since the answer is cached, but the first
        # time it should look like waiting rather than like a hang.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            dialog = ModelDialog(self, models=models,
                                 chosen=label_for(active) if active else None)
        finally:
            QApplication.restoreOverrideCursor()
        dialog.exec()

        self._save_llm_models(dialog.models)
        if dialog.chosen:
            from PySide6.QtCore import QSettings
            QSettings("VideoHighlighter", "Pro").setValue(
                "advisor/model_chosen", dialog.chosen)
            self.append_log(f"🤖 The report will be written with {dialog.chosen}.")
        elif not dialog.models:
            self.append_log("🤖 No model configured for the report.")

    def _discuss_report_in_chat(self):
        """Open the LLM chat with this run's findings already in front of it."""
        json_path = self._newest_why_report_json()
        if not json_path:
            return
        widget = self._open_llm_chat_widget()
        if widget is None:
            self.append_log("⚠️ The LLM chat window is not available in this build.")
            return
        try:
            widget.seed_from_report(json_path)
        except Exception as exc:
            self.append_log(f"⚠️ Could not hand the report to the chat: {exc}")

    def _open_llm_chat_widget(self):
        """The LLM Chat tab, brought to the front."""
        widget = getattr(self, "llm_chat", None)
        if widget is None:
            return None
        tabs = getattr(self, "tabs", None)
        if tabs is not None:
            # The chat is a tab, not a window: handing it a report without
            # showing it would look like nothing happened.
            index = tabs.indexOf(widget)
            if index == -1 and widget.parentWidget() is not None:
                index = tabs.indexOf(widget.parentWidget())
            if index != -1:
                tabs.setCurrentIndex(index)
        return widget

    def open_timeline_viewer(self):
        """Open timeline viewer for the selected video"""
        video_paths = self.get_file_list()
        
        if not video_paths:
            self.append_log("⚠️ No video selected. Please add a video first.")
            return
        
        # Use the first video in the list
        video_path = video_paths[0]
        
        if not os.path.exists(video_path):
            self.append_log(f"⚠️ Video file not found: {video_path}")
            return
        
        try:
            from signal_timeline_viewer import SignalTimelineWindow

            # Reuse an existing timeline window for the same video instead of
            # building a new one. The timeline window pins itself in memory
            # (it installs an app-wide event filter, and is referenced by the
            # LLM chat), and its 4K players can't be torn down without blocking,
            # so creating a fresh one each open leaks ~2.5GB per cycle. Re-show
            # the existing one when the video matches.
            existing = getattr(self, 'timeline_window', None)
            if existing is not None:
                try:
                    same_video = (getattr(existing, 'video_path', None) == video_path)
                    if same_video:
                        # Pick up any signals added on demand since it was opened
                        # (per-signal Run buttons fold into the cache on disk).
                        #
                        # Reusing the window is not the instant path it looks
                        # like: the refresh re-ingests every signal and redraws
                        # the timeline, which on a long video takes as long as
                        # building the window did — and it blocks the GUI
                        # thread, with the old view still on screen. Without the
                        # splash, reopening looked like the app had frozen.
                        startup_splash.begin("Reopening timeline viewer",
                                             os.path.basename(video_path),
                                             steps=4, parent=self)
                        try:
                            existing.refresh_from_disk()
                        except Exception as e:
                            self.append_log(f"⚠️ Could not refresh timeline cache: {e}")
                        finally:
                            startup_splash.finish(existing)
                        # Un-mute (close() muted the audio outputs) and re-show
                        for ao_attr, obj in (('audio_output', existing),
                                             ('_audio', getattr(existing, 'realtime_preview', None))):
                            ao = getattr(obj, ao_attr, None) if obj is not None else None
                            if ao is not None:
                                try:
                                    ao.setMuted(False)
                                except Exception:
                                    pass
                        existing.show()
                        existing.raise_()
                        existing.activateWindow()
                        self.append_log("📊 Reusing open timeline viewer.")
                        return
                except RuntimeError:
                    # Underlying C++ object was deleted — fall through to recreate
                    self.timeline_window = None

            # Check if cache exists - use the same parameters as in pipeline
            from modules.video_cache import VideoAnalysisCache, build_analysis_cache_params
            
            # Build the same parameters that were used when processing
            # We need to recreate the analysis_params that were used
            # Let's get the current config from GUI
            config = self.build_pipeline_config()
            
            # Get video duration for parameter building
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            video_duration = total_frames / fps if fps else 0
            cap.release()
            
            # Build analysis params that match what was used
            sample_rate = int(self.sample_rate_spin.value())
            
            # Load config.yaml defaults
            cfg_data = {}
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    cfg_data = yaml.safe_load(f) or {}
            
            analysis_params = build_analysis_cache_params(
                gui_config=config,
                config=cfg_data,
                sample_rate=sample_rate,
                video_duration=video_duration
            )
            
            # Try to load with these params first
            cache = VideoAnalysisCache()
            cache_data = cache.load(video_path, params=analysis_params)
            
            if not cache_data:
                import json
                from pathlib import Path
                
                video_hash = cache._get_video_hash(video_path)
                cache_dir = Path("./cache")
                matching_files = list(cache_dir.glob(f"{video_hash}*.cache.json"))
                
                if matching_files:
                    latest_file = max(matching_files, key=lambda p: p.stat().st_mtime)
                    with open(latest_file, 'r') as f:
                        cache_data = json.load(f)
                    self.append_log(f"✅ Loaded cache: {latest_file.name}")
                else:
                    # Check if user suppressed this warning
                    suppress = self.config_data.get("ui", {}).get("suppress_no_cache_warning", False)
                    
                    if not suppress:
                        dlg = NoAnalysisWarningDialog(self)
                        if dlg.exec() != QDialog.Accepted:
                            return  # User clicked Cancel
                        
                        if dlg.dont_show_chk.isChecked():
                            # Persist the preference
                            if "ui" not in self.config_data:
                                self.config_data["ui"] = {}
                            self.config_data["ui"]["suppress_no_cache_warning"] = True
                            self.save_config()
                    
                    self.append_log("⚠️ Opening timeline without signal data — run pipeline to populate signals.")
                    cache_data = {}

            
            self.append_log(f"📊 Opening timeline viewer for: {os.path.basename(video_path)}")

            # Building this window takes several seconds on a real analysis —
            # the signal timeline and the assistant panel are most of it — and
            # it blocks the GUI thread, so without the splash the app just
            # appears to hang. The window reports its own stages (see
            # signal_timeline_viewer.init_ui).
            startup_splash.begin("Opening timeline viewer",
                                 os.path.basename(video_path), steps=6,
                                 parent=self)
            startup_splash.stage("Reading the analysis cache…")
            window = None
            try:
                # Create and show the timeline window
                window = SignalTimelineWindow(video_path, cache_data)
                # An Analyze run started over there detects over the whole
                # video just as a pipeline stage does, so it feeds the preview
                # window this side owns. Queued (the frames come off the
                # viewer's analysis thread), and gated on the checkbox by the
                # emitting end.
                window.preview_frame.connect(self.on_preview_frame)
                window.preview_enabled = self._preview_enabled
                self.timeline_window = window
                window.show()
            finally:
                # finally: a viewer that fails half-way must not leave an
                # always-on-top splash stranded over the app with no window
                # behind it to explain itself. `window` stays None in that
                # case, so a *previous* viewer is never raised by mistake.
                startup_splash.finish(window)
            # Connect LLM chat to timeline and video
            self.llm_chat.set_timeline_window(self.timeline_window)
            self.llm_chat.set_video_path(video_path)
            self.llm_chat.load_cache_for_video(video_path)

        except ImportError as e:
            self.append_log(f"❌ Failed to import timeline viewer: {e}")
            self.append_log("   Make sure signal_timeline_viewer.py is in the same directory.")
        except Exception as e:
            self.append_log(f"❌ Failed to open timeline viewer: {e}")
            import traceback
            self.append_log(traceback.format_exc())

def _hard_exit(exit_code: int = 0):
    """Terminate the process immediately, bypassing slow/hanging native teardown.

    os._exit() is NOT safe enough on Windows — it calls ExitProcess, which runs
    DLL detach and tries to terminate threads cleanly. If a native thread is
    stuck (FFmpeg 4K decoder, onnxruntime/InsightFace mid-inference), ExitProcess
    deadlocks and the process never dies. TerminateProcess on our own process is
    the hardest kill available: it terminates every thread immediately with no
    cleanup, so a stuck decoder/inference thread can't block exit.

    Config is saved in the main window's closeEvent, so nothing is left to
    persist by the time this runs.
    """
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    if sys.platform == "win32":
        # On Windows, os.kill() with any signal other than CTRL_C_EVENT /
        # CTRL_BREAK_EVENT unconditionally calls TerminateProcess on our own
        # process — the hardest kill available, and (unlike a raw ctypes
        # TerminateProcess call) Python handles the process handle correctly,
        # so it can't be silently truncated/failed on 64-bit.
        try:
            import signal
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception:
            pass
        # Fallback: raw TerminateProcess with correct 64-bit handle types.
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
            kernel32.TerminateProcess(kernel32.GetCurrentProcess(), exit_code & 0xFFFF)
        except Exception:
            pass
    os._exit(exit_code)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass
    reset_duration_method_cache()
    # Disable D3D11VA hardware acceleration in Qt multimedia's FFmpeg backend.
    # On some Windows systems D3D11VA initialisation fails for H.264, causing
    # noisy warnings even though playback still works via software decoding.
    os.environ.setdefault("QT_FFMPEG_DECODING_HWACCEL", "none")
    os.environ.setdefault("QT_LOGGING_RULES", "qt.multimedia.ffmpeg=false")

    # Give Windows an explicit AppUserModelID so the taskbar groups this app
    # under our own icon instead of the generic python.exe host. Must run
    # before any window is created.
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "VideoHighlighter.App"
            )
        except Exception:
            pass

    app = QApplication(sys.argv)

    # Central theme: one graphite + accent stylesheet for all base widgets.
    # Additive — screens with their own inline styles still override it.
    _ui_theme.apply(app)
    # The settings screens are tall scrolling columns of spin boxes; Qt's
    # default hands the wheel to whichever one is under the cursor, so scrolling
    # past the Scoring Points panel quietly re-scores the next run. Wheel scrolls
    # the panel instead, everywhere in the app.
    from modules.ui.wheel_guard import install as _install_wheel_guard
    _install_wheel_guard(app)
    # One icon for every window the app opens (main window, timeline viewer,
    # dialogs). Set on the QApplication so nothing has to remember to do it.
    # The .ico carries every size from 16 to 256px, all of them the same logo
    # — 16-48px have their brightness and contrast lifted, because at taskbar
    # size the artwork's thin outlines and dark glass otherwise disappear into a
    # dark taskbar. It previously carried a separate simplified mark at those
    # sizes, which was drawn from an older logo and so kept shipping it long
    # after the artwork changed.
    _icon_path = _resource_path(os.path.join("assets", "icon.ico"))
    if os.path.exists(_icon_path):
        from PySide6.QtGui import QIcon
        app.setWindowIcon(QIcon(_icon_path))

    # `--timeline <video>` opens just the Signal Timeline viewer for one video
    # instead of the full GUI. The packaged build is a single exe, so this is how
    # another process (the web UI's sidecar) asks for the viewer — without it,
    # the only way in was to launch the whole application.
    #
    # Placed after the theme, wheel guard and window icon are installed on the
    # QApplication: the viewer is a window like any other, and launching it
    # ahead of that setup shipped it unthemed and carrying the old mark.
    if "--timeline" in sys.argv:
        idx = sys.argv.index("--timeline")
        video = sys.argv[idx + 1] if len(sys.argv) > idx + 1 else ""
        if not video or not os.path.exists(video):
            print(f"--timeline needs an existing video path (got {video!r})")
            _hard_exit(2)
        from signal_timeline_viewer import SignalTimelineWindow
        win = SignalTimelineWindow(video)
        win.show()
        _hard_exit(app.exec())

    # Hand over from the bootloader's splash to the Qt one, which can keep
    # reporting through the window build (the remaining seconds) and follows
    # the app's theme. begin() closes the native splash once this is painted.
    startup_splash.begin(f"VideoHighlighter {__edition__}",
                         f"Version {__version__}", steps=2)

    # Reopen the live debug-log window if it was on last session (needs the
    # QApplication, hence here and not earlier).
    debug_console.restore_console_preference()

    startup_splash.stage("Building the workspace…")
    gui = VideoHighlighterGUI()
    gui.show()
    startup_splash.finish(gui)
    exit_code = app.exec()

    # Backup hard-exit in case app.exec() does return (main closeEvent already
    # hard-exits, so this is belt-and-suspenders).
    _hard_exit(exit_code)