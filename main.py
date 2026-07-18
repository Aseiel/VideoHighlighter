import os
import sys

# Capture every print/warning/traceback from the very first import: the
# packaged exe is --windowed (no stdout), so modules/debug_console tees all
# output into debug.log next to the exe and can mirror it to a live console
# window. Must run before the heavy imports below — some of them print
# warnings worth keeping.
from modules import debug_console
debug_console.install()

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
    QGridLayout, QSlider,
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QMetaObject, Q_ARG, Slot, QStringListModel
from downloader import download_videos_with_immediate_processing, extract_video_links, DownloadError, reset_duration_method_cache
from llm.llm_chat_widget import LLMChatWidget
from modules.video_cache import VideoAnalysisCache, CachedAnalysisData, build_analysis_cache_params
from modules.ui import icons as _ui_icons, theme as _ui_theme

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
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Video Highlighter v{__version__} {__edition__}")
        screen = QApplication.primaryScreen().availableGeometry()
        w = min(1000, screen.width() - 20)
        h = min(800, screen.height() - 20)
        self.resize(w, h)
        self.move(screen.x() + (screen.width() - w) // 2, screen.y())

        
        self.worker = None

        self.config_data = self.load_config()

        layout = QVBoxLayout()
        # Breathing room — the window read cramped with Qt's default tight
        # margins. Generous outer margin + row spacing is the single biggest
        # lever on how modern it feels.
        layout.setContentsMargins(20, 16, 20, 12)
        layout.setSpacing(14)

        # Store video duration
        self.current_video_duration = 0

        # --- File picker ---
        file_group = QGroupBox("Input Videos")
        file_layout = QVBoxLayout()
        file_layout.setContentsMargins(14, 12, 14, 12)
        file_layout.setSpacing(10)

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

        # File list
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(120)
        file_layout.addWidget(self.file_list)

        saved_paths = self.config_data.get("video", {}).get("paths", [])
        if saved_paths:
            for path in saved_paths:
                if os.path.exists(path):
                    self.file_list.addItem(path)
        
        file_group.setLayout(file_layout)
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

        # Video duration label
        self.video_duration_label = QLabel("Set time range in percentages (0-100%) - loads actual times when video is selected")
        self.video_duration_label.setStyleSheet("color: #666; font-style: italic;")
        time_range_layout.addWidget(self.video_duration_label)

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
        time_range_layout.addWidget(slider_container)

        # Selection info
        self.selection_info_label = QLabel("Selection: Full video")
        self.selection_info_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 10pt;")
        time_range_layout.addWidget(self.selection_info_label)

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
        time_range_layout.addLayout(presets_layout)

        time_range_group.setLayout(time_range_layout)
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

        # Batch counter — its own row because the process bar and task_label
        # below are rewritten by every pipeline stage of the current video.
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setVisible(False)
        self.batch_progress_bar.setFormat("%v / %m videos")
        progress_layout.addWidget(self.batch_progress_bar)

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
        # Hidden when idle — the progress box only appears while a download or
        # pipeline runs. The Basic/Advanced tabs use side-by-side layouts, so the
        # tabs+log splitter has room to shrink when this box appears instead of
        # pushing the Run/Cancel row and log box off the bottom of the window.
        self.progress_group.setVisible(False)
        layout.addWidget(self.progress_group)

        # --- Tabs ---
        tabs = QTabWidget()

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
        tabs.addTab(download_tab, "Download")

        # --- Tab 1: Basic Settings ---
        basic_tab = QWidget()
        # Grid so the two tall groups (Scoring Points / Duration) sit side by side
        # and use horizontal space instead of stacking into one tall column that
        # overflows the window. Mirrors the Advanced tab layout.
        basic_layout = QGridLayout()

        # ── Group 1: Scoring Points ──
        points_box = QGroupBox("Scoring Points")
        points_layout = QFormLayout()

        self.spin_scene_points = QSpinBox(); self.spin_scene_points.setRange(0,100); self.spin_scene_points.setValue(scoring_cfg.get("scene_points", 0))
        self.spin_scene_points.setToolTip("Points awarded when a new scene cut is detected (abrupt visual change)")

        self.spin_motion_event_points = QSpinBox(); self.spin_motion_event_points.setRange(0,100); self.spin_motion_event_points.setValue(scoring_cfg.get("motion_event_points", 0))
        self.spin_motion_event_points.setToolTip("Points for any frame with detected movement above the threshold")

        self.spin_motion_peak = QSpinBox(); self.spin_motion_peak.setRange(0,100); self.spin_motion_peak.setValue(scoring_cfg.get("motion_peak_points", 3))
        self.spin_motion_peak.setToolTip("Points for a sudden burst of motion followed by stillness (e.g. a goal followed by replay, an explosion then calm)")

        self.spin_audio_peak = QSpinBox(); self.spin_audio_peak.setRange(0,100); self.spin_audio_peak.setValue(scoring_cfg.get("audio_peak_points", 0))
        self.spin_audio_peak.setToolTip("Points when audio intensity spikes (e.g. crowd roar, explosions, bells, loud impacts)")

        self.spin_keyword_points = QSpinBox(); self.spin_keyword_points.setRange(0,100); self.spin_keyword_points.setValue(scoring_cfg.get("keyword_points", 2))
        self.spin_keyword_points.setToolTip("Points when a search keyword (configured in Transcript & Subtitles tab) is found in speech")

        self.spin_transcript_points = QSpinBox(); self.spin_transcript_points.setRange(0,100); self.spin_transcript_points.setValue(scoring_cfg.get("transcript_points", 2))
        self.spin_transcript_points.setToolTip("Points for any moment where speech is detected, regardless of content")

        self.spin_object = QSpinBox(); self.spin_object.setRange(0,100); self.spin_object.setValue(scoring_cfg.get("object_points", 1))
        self.spin_object.setToolTip("Points when a configured object class is detected in the frame")

        self.spin_action = QSpinBox(); self.spin_action.setRange(0,1000); self.spin_action.setValue(scoring_cfg.get("action_points", 10))
        self.spin_action.setToolTip("Points when a configured action is recognized (e.g. punching, jumping, dancing)")

        points_layout.addRow("Scene points:", self.spin_scene_points)
        points_layout.addRow("Motion event points:", self.spin_motion_event_points)
        points_layout.addRow("Motion peak points:", self.spin_motion_peak)
        points_layout.addRow("Audio peak points:", self.spin_audio_peak)
        points_layout.addRow("Keyword points (keywords in transcript):", self.spin_keyword_points)
        points_layout.addRow("Transcript points (all words):", self.spin_transcript_points)
        points_layout.addRow("Object points:", self.spin_object)
        points_layout.addRow("Action points:", self.spin_action)

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

        # Conditional action scoring checkbox
        self.actions_require_objects_chk = QCheckBox("Only score actions when objects detected")
        self.actions_require_objects_chk.setChecked(self.config_data.get("actions", {}).get("require_objects", False))
        self.actions_require_objects_chk.setToolTip("Actions will only add points if objects are also detected in that timeframe")
        basic_layout.addWidget(self.actions_require_objects_chk, 3, 0, 1, 2)

        self.skip_highlights_chk = QCheckBox("Skip highlights")
        self.skip_highlights_chk.setChecked(highlights_cfg.get("skip_highlights", False))
        basic_layout.addWidget(self.skip_highlights_chk, 4, 0, 1, 2)

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
        tabs.addTab(basic_tab, "Basic Settings")

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

        self.search_keywords_input = QLineEdit(",".join(transcript_cfg.get("search_keywords", [])))
        self.search_keywords_input.setPlaceholderText("goal, score, win")
        self.search_keywords_input.setEnabled(transcript_cfg.get("enabled", False))
        transcript_form.addRow("Search keywords:", self.search_keywords_input)
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

        self.subtitle_source_lang = QComboBox()
        self.subtitle_source_lang.addItems(["en","pl","es","fr","de","it","pt","ru","ja","ko","zh"])
        self.subtitle_source_lang.setCurrentText(subtitles_cfg.get("source_lang", "en"))
        self.subtitle_source_lang.setEnabled(subtitles_cfg.get("enabled", False) and transcript_cfg.get("enabled", False))
        subtitle_form.addRow("Source language:", self.subtitle_source_lang)

        self.subtitle_target_lang = QComboBox()
        self.subtitle_target_lang.addItems(["en","pl","es","fr","de","it","pt","ru","ja","ko","zh"])
        self.subtitle_target_lang.setCurrentText(subtitles_cfg.get("target_lang", "pl"))
        self.subtitle_target_lang.setEnabled(subtitles_cfg.get("enabled", False) and transcript_cfg.get("enabled", False))
        subtitle_form.addRow("Target language:", self.subtitle_target_lang)
        subtitle_group.setLayout(subtitle_form)
        transcript_layout.addWidget(subtitle_group)

        transcript_tab.setLayout(transcript_layout)
        tabs.addTab(transcript_tab, "Transcript && Subtitles")

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
        self.comp_table = QTableWidget(0, 9)
        self.comp_table.setHorizontalHeaderLabels([
            "Event Name", "Display Label", "Source Object", "Region Object",
            "Min Count", "Max Count", "Window (s)", "Persist (s)", "",
        ])
        self.comp_table.horizontalHeader().setToolTip(
            "Source Object: the object that must appear inside the Region Object\n"
            "Min/Max Count: how many Source instances must be inside the region\n"
            "Window: seconds of frames to smooth over (reduces single-frame flicker)\n"
            "Persist: seconds to keep a source 'alive' after it disappears (handles occlusion)"
        )
        self.comp_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.comp_table.horizontalHeader().setSectionResizeMode(8, QHeaderView.Fixed)
        self.comp_table.setColumnWidth(8, 32)
        self.comp_table.setMinimumHeight(160)
        self.comp_table.setMaximumHeight(280)
        self.comp_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        comp_outer.addWidget(self.comp_table)

        comp_btn_row = QHBoxLayout()
        comp_add_btn = QPushButton("+ Add Rule")
        comp_add_btn.setToolTip("Add a new rule row")
        comp_save_btn = QPushButton("Save Rules")
        comp_save_btn.setToolTip("Save composition rules to composition_rules.yaml")
        comp_btn_row.addWidget(comp_add_btn)
        comp_btn_row.addStretch()
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
            for ev in events:
                for rule in ev.get('rules', []):
                    _comp_add_table_row(
                        ev.get('name', ''),
                        ev.get('label', ev.get('name', '')),
                        rule.get('source', ''),
                        rule.get('region', ''),
                        rule.get('min_count', 1),
                        rule.get('max_count', 999),
                        ev.get('window_secs', 0.75),
                        ev.get('persist_secs', 0.5),
                    )

        def _comp_add_table_row(ev_name='', ev_label='', source='', region='',
                                min_c=1, max_c=999, window=0.75, persist=0.5):
            r = self.comp_table.rowCount()
            self.comp_table.insertRow(r)
            self.comp_table.setItem(r, 0, QTableWidgetItem(ev_name))
            self.comp_table.setItem(r, 1, QTableWidgetItem(ev_label))
            self.comp_table.setItem(r, 2, QTableWidgetItem(source))
            self.comp_table.setItem(r, 3, QTableWidgetItem(region))

            min_spin = QSpinBox()
            min_spin.setRange(0, 99)
            min_spin.setValue(int(min_c))
            self.comp_table.setCellWidget(r, 4, min_spin)

            max_spin = QSpinBox()
            max_spin.setRange(0, 999)
            max_spin.setValue(int(max_c))
            self.comp_table.setCellWidget(r, 5, max_spin)

            win_spin = QDoubleSpinBox()
            win_spin.setRange(0.1, 10.0)
            win_spin.setSingleStep(0.25)
            win_spin.setValue(float(window))
            self.comp_table.setCellWidget(r, 6, win_spin)

            per_spin = QDoubleSpinBox()
            per_spin.setRange(0.0, 10.0)
            per_spin.setSingleStep(0.25)
            per_spin.setValue(float(persist))
            self.comp_table.setCellWidget(r, 7, per_spin)

            del_btn = QPushButton("✕")
            del_btn.setFixedWidth(28)
            del_btn.setStyleSheet("color: #c33; border: none; font-weight: bold;")
            # Find the row at click-time by locating this button in column 8
            def _make_del(btn):
                def _del():
                    for i in range(self.comp_table.rowCount()):
                        if self.comp_table.cellWidget(i, 8) is btn:
                            self.comp_table.removeRow(i)
                            return
                return _del
            del_btn.clicked.connect(_make_del(del_btn))
            self.comp_table.setCellWidget(r, 8, del_btn)

        def _comp_save_rules():
            from modules.app_paths import user_data_dir
            import os as _os
            # Group rows by event name (preserving order of first appearance)
            events_ordered = []
            events_map = {}
            for r in range(self.comp_table.rowCount()):
                ev_name  = (self.comp_table.item(r, 0) or QTableWidgetItem()).text().strip()
                ev_label = (self.comp_table.item(r, 1) or QTableWidgetItem()).text().strip()
                source   = (self.comp_table.item(r, 2) or QTableWidgetItem()).text().strip()
                region   = (self.comp_table.item(r, 3) or QTableWidgetItem()).text().strip()
                min_c    = self.comp_table.cellWidget(r, 4).value()
                max_c    = self.comp_table.cellWidget(r, 5).value()
                window   = self.comp_table.cellWidget(r, 6).value()
                persist  = self.comp_table.cellWidget(r, 7).value()
                if not ev_name or not source or not region:
                    continue
                if ev_name not in events_map:
                    ev_entry = {
                        'name': ev_name,
                        'label': ev_label or ev_name,
                        'rules': [],
                        'window_secs': window,
                        'persist_secs': persist,
                    }
                    events_map[ev_name] = ev_entry
                    events_ordered.append(ev_entry)
                events_map[ev_name]['rules'].append({
                    'source': source,
                    'region': region,
                    'min_count': min_c,
                    'max_count': max_c,
                })
            out = {'events': events_ordered}
            save_path = _os.path.join(user_data_dir(), 'composition_rules.yaml')
            try:
                with open(save_path, 'w', encoding='utf-8') as _f:
                    yaml.dump(out, _f, allow_unicode=True, sort_keys=False, default_flow_style=False)
                self.append_log(f"✅ Composition rules saved → {save_path}")
            except Exception as _e:
                self.append_log(f"❌ Could not save composition rules: {_e}")

        comp_add_btn.clicked.connect(lambda: _comp_add_table_row())
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
        tabs.addTab(advanced_tab, "Advanced")

        content_splitter = QSplitter(Qt.Vertical)
        content_splitter.addWidget(tabs)
        layout.addWidget(content_splitter)

        # --- Tab 4: LLM Chat ---
        llm_tab = QWidget()
        llm_layout = QVBoxLayout()
        self.llm_chat = LLMChatWidget(parent=self)
        llm_layout.addWidget(self.llm_chat)
        llm_tab.setLayout(llm_layout)
        tabs.addTab(llm_tab, "LLM Chat")

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
        tabs.addTab(avoid_tab, "Avoid")

        # --- Tab: About & Contact ---
        tabs.addTab(self._build_about_tab(), "About")

        # Defer first populate until after __init__ finishes (so log_output exists)
        QTimer.singleShot(0, self.refresh_avoid_list)

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

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet("QPushButton:enabled { background-color: #ff4444; color: white; font-weight: bold; }")
        self.cancel_btn.clicked.connect(self.cancel_pipeline)

        self.run_btn = QPushButton("Run Highlighter")
        self.run_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        self.run_btn.clicked.connect(self.toggle_run)

        ctrl_layout.addWidget(self.cancel_btn)
        ctrl_layout.addWidget(self.keep_temp_chk)
        ctrl_layout.addWidget(self.timeline_btn)
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
        ctrl_layout.addStretch()
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
        content_splitter.setSizes([h - 200, 150])  # give log ~150px

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

                    result = run_highlighter(
                        filepath,
                        gui_config=config,
                        log_fn=log_fn,
                        progress_fn=progress_fn,
                        cancel_flag=cancel_flag
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
            "keyword_points": int(self.spin_keyword_points.value()),
            "transcript_points": int(self.spin_transcript_points.value()),
            "beginning_points": 0,
            "ending_points": 0,
            "object_points": int(self.spin_object.value()),
            "action_points": int(self.spin_action.value()),
            "clip_time": int(self.spin_clip_time.value()),
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
            "source_lang": self.subtitle_source_lang.currentText(),
            "target_lang": self.subtitle_target_lang.currentText(),
            "skip_highlights": self.skip_highlights_chk.isChecked(),
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
        """Combine multiple highlight videos into one with robust resolution/framerate handling"""
        if not highlight_files:
            self.append_log("⚠️ No highlight files to combine")
            return None
        
        try:
            # Filter out None values and non-existent files
            valid_files = [f for f in highlight_files if f and os.path.exists(f)]
            
            if not valid_files:
                self.append_log("⚠️ No valid highlight files found")
                return None
            
            if len(valid_files) == 1:
                self.append_log("ℹ️ Only one highlight file, no combining needed")
                return valid_files[0]
            
            self.append_log(f"🎬 Combining {len(valid_files)} highlights into one video...")
            
            # Analyze all input videos to determine target specs
            self.append_log("🔍 Analyzing input videos...")
            video_specs = []
            for video_file in valid_files:
                try:
                    cmd = [
                        "ffprobe", "-v", "error",
                        "-select_streams", "v:0",
                        "-show_entries", "stream=width,height,r_frame_rate",
                        "-of", "json",
                        video_file
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                    import json
                    info = json.loads(result.stdout)
                    
                    if 'streams' in info and len(info['streams']) > 0:
                        stream = info['streams'][0]
                        width = stream.get('width', 1920)
                        height = stream.get('height', 1080)
                        fps_str = stream.get('r_frame_rate', '30/1')
                        
                        # Parse fps fraction (e.g., "30000/1001" or "30/1")
                        if '/' in fps_str:
                            num, den = fps_str.split('/')
                            fps = float(num) / float(den)
                        else:
                            fps = float(fps_str)
                        
                        video_specs.append({
                            'file': video_file,
                            'width': width,
                            'height': height,
                            'fps': fps
                        })
                        self.append_log(f"  {os.path.basename(video_file)}: {width}x{height} @ {fps:.2f}fps")
                except Exception as e:
                    self.append_log(f"  ⚠️ Could not analyze {os.path.basename(video_file)}: {e}")
            
            if not video_specs:
                self.append_log("❌ Could not analyze any input videos")
                return None
            
            # Determine target resolution (use most common or largest)
            widths = [s['width'] for s in video_specs]
            heights = [s['height'] for s in video_specs]
            target_width = max(set(widths), key=widths.count)  # Most common width
            target_height = max(set(heights), key=heights.count)  # Most common height
            target_fps = 30  # Standard fps
            
            self.append_log(f"🎯 Target format: {target_width}x{target_height} @ {target_fps}fps")
            
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Create temp directory for normalized files
            temp_dir = os.path.join(output_dir or ".", "temp_combine")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Normalize each video to common format
            self.append_log("⚙️ Normalizing all videos to common format...")
            normalized_files = []
            
            for i, spec in enumerate(video_specs):
                video_file = spec['file']
                temp_file = os.path.join(temp_dir, f"normalized_{i:03d}.mp4")
                normalized_files.append(temp_file)
                
                self.append_log(f"  Processing {i+1}/{len(video_specs)}: {os.path.basename(video_file)}")
                
                # Normalize: scale, pad, set fps, and re-encode
                cmd = [
                    "ffmpeg", "-y", "-i", video_file,
                    # VIDEO: Scale to fit, pad to exact size, set fps, ensure proper timestamps
                    "-vf", f"scale={target_width}:{target_height}:force_original_aspect_ratio=decrease,"
                        f"pad={target_width}:{target_height}:(ow-iw)/2:(oh-ih)/2,"
                        f"setsar=1,fps={target_fps},setpts=N/FRAME_RATE/TB",
                    # AUDIO: Resample and re-timestamp
                    "-af", "aresample=48000,asetpts=N/SR/TB",
                    # VIDEO CODEC: Consistent encoding settings
                    "-c:v", "libx264",
                    "-preset", "medium",
                    "-crf", "23",
                    "-pix_fmt", "yuv420p",
                    "-profile:v", "high",
                    "-level", "4.0",
                    "-g", str(target_fps * 2),  # GOP size = 2 seconds
                    "-keyint_min", str(target_fps),
                    "-sc_threshold", "0",
                    # AUDIO CODEC
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-ar", "48000",
                    # TIMING & SYNC
                    "-vsync", "cfr",  # Constant frame rate
                    "-async", "1",  # Audio sync
                    "-max_muxing_queue_size", "1024",
                    "-fflags", "+genpts",
                    "-avoid_negative_ts", "make_zero",
                    temp_file
                ]
                
                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=300,  # 5 minute timeout per file
                        check=True
                    )
                    
                    # Verify the normalized file
                    if os.path.exists(temp_file) and os.path.getsize(temp_file) > 0:
                        self.append_log(f"    ✅ Normalized successfully")
                    else:
                        raise Exception("Normalized file is empty or missing")
                        
                except subprocess.CalledProcessError as e:
                    self.append_log(f"    ❌ Normalization failed: {e.stderr[:200]}")
                    raise
                except Exception as e:
                    self.append_log(f"    ❌ Error: {e}")
                    raise
            
            # Now concatenate the normalized files
            self.append_log("🔗 Concatenating normalized videos...")
            concat_file = os.path.join(temp_dir, "concat_list.txt")
            with open(concat_file, "w", encoding="utf-8") as f:
                for temp_file in normalized_files:
                    abs_path = os.path.abspath(temp_file).replace('\\', '/')
                    f.write(f"file '{abs_path}'\n")
            
            # Simple concatenation (copy) since all files now have identical format
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file,
                "-c", "copy",  # Direct copy - no re-encoding
                "-movflags", "+faststart",
                output_path
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=True
                )
                
                # Verify output
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    self.append_log(f"✅ Combined video saved: {output_path}")
                    
                    # Get final info
                    try:
                        cmd = [
                            "ffprobe", "-v", "error",
                            "-select_streams", "v:0",
                            "-show_entries", "stream=r_frame_rate,width,height",
                            "-show_entries", "format=duration,size",
                            "-of", "json",
                            output_path
                        ]
                        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                        import json
                        info = json.loads(result.stdout)
                        
                        if 'streams' in info and len(info['streams']) > 0:
                            stream = info['streams'][0]
                            width = stream.get('width', 'N/A')
                            height = stream.get('height', 'N/A')
                            fps = stream.get('r_frame_rate', 'N/A')
                            
                        if 'format' in info:
                            format_info = info['format']
                            duration = float(format_info.get('duration', 0))
                            size = int(format_info.get('size', 0)) / (1024 * 1024)  # MB
                            
                            self.append_log(f"📊 Final: {width}x{height}, {fps} fps, {duration:.1f}s, {size:.1f}MB")
                            
                    except Exception as e:
                        pass  # Info is optional
                    
                    # Clean up temp files
                    try:
                        os.remove(concat_file)
                        for temp_file in normalized_files:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                        os.rmdir(temp_dir)
                    except Exception as e:
                        self.append_log(f"⚠️ Could not clean up temp files: {e}")
                    
                    return output_path
                else:
                    raise Exception("Output file is empty or missing")
                    
            except Exception as e:
                self.append_log(f"❌ Failed to concatenate: {e}")
                
                # Clean up on failure
                try:
                    if os.path.exists(concat_file):
                        os.remove(concat_file)
                    for temp_file in normalized_files:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    if os.path.exists(temp_dir):
                        os.rmdir(temp_dir)
                except:
                    pass
                
                return None
                
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
                "output": self.output_input.text().strip(),
                "max_duration": int(self.spin_max_duration.value()),
                "exact_duration": int(self.spin_exact_duration.value()),
                "keep_temp": self.keep_temp_chk.isChecked(),
                "render_mode": self.render_mode_combo.currentData(),
                "skip_highlights": self.skip_highlights_chk.isChecked(),
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
                "keyword_points": int(self.spin_keyword_points.value()),
                "transcript_points": int(self.spin_transcript_points.value()),
                "object_points": int(self.spin_object.value()),
                "action_points": int(self.spin_action.value()),
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
                "source_lang": self.subtitle_source_lang.currentText(),
                "target_lang": self.subtitle_target_lang.currentText(),
            },
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
        self.search_keywords_input.setEnabled(checked)
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
        
        self.subtitle_source_lang.setEnabled(final_state)
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
        self.batch_progress_bar.setRange(0, max(1, total))
        self.batch_progress_bar.setValue(max(0, min(current, total)))
        self.batch_progress_bar.setVisible(True)
        self.batch_label.setText(f"📦 {details}")
        self.batch_label.setVisible(True)
        QApplication.processEvents()

    def hide_batch_progress(self):
        self.batch_progress_bar.setVisible(False)
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
        """Enable/disable time range controls"""
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
        if hasattr(self, 'worker') and self.worker is not None:
            try:
                self.worker.preview_enabled = checked
            except Exception:
                pass

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

    def run_pipeline(self):
        from pipeline import run_highlighter
        """Start the pipeline processing (UPDATED for multi-file)"""
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
        
        beginning_points = 0  # Not configurable in GUI
        ending_points = 0     # Not configurable in GUI
        
        total_points = (scene_points + motion_event_points + motion_peak_points + 
                       audio_peak_points + keyword_points + transcript_points + 
                       beginning_points + ending_points + object_points + action_points)
        
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
            "keyword_points": int(self.spin_keyword_points.value()),
            "transcript_points": int(self.spin_transcript_points.value()),
            "beginning_points": 0,
            "ending_points": 0,
            "object_points": int(self.spin_object.value()),
            "action_points": int(self.spin_action.value()),
            "clip_time": int(self.spin_clip_time.value()),
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
            "source_lang": self.subtitle_source_lang.currentText(),
            "target_lang": self.subtitle_target_lang.currentText(),
            "skip_highlights": self.skip_highlights_chk.isChecked(),
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

        # --- Skip highlights logic ---
        if config.get("skip_highlights", False):
            config["scene_points"] = 0
            config["motion_event_points"] = 0
            config["motion_peak_points"] = 0
            config["audio_peak_points"] = 0
            config["object_points"] = 0
            config["action_points"] = 0
            config["keyword_points"] = 0
            config["clip_time"] = 0
            config["max_duration"] = 0
            config["exact_duration"] = None

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
        self.cancel_btn.setEnabled(True)

        # Disable form inputs during processing
        self.file_list.setEnabled(False)
        self.output_input.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.remove_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

        # Create and start worker
        self.worker = Worker(video_paths, config)
        self.worker.preview_enabled = self.live_preview_checkbox.isChecked()
        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self.update_pipeline_progress)
        self.worker.finished.connect(self.pipeline_done)
        self.worker.cancelled.connect(self.pipeline_cancelled)
        self.worker.preview.connect(self.on_preview_frame)
        
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
        
        # Nothing is running
        self.append_log("⚠️ Nothing to cancel - no active process")

    def toggle_run(self):
        """Run / Pause / Resume - single button"""
        # Not running → start pipeline
        if not self.worker or not self.worker._is_running:
            self.run_pipeline()
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
                srt_file = f"{base_name}_{self.target_lang_combo.currentText()}.srt"
                transcript_file = f"{base_name}_transcript.txt"
                
                if os.path.exists(srt_file): 
                    self.append_log(f"📝 Subtitle file: {srt_file}")
                if os.path.exists(transcript_file): 
                    self.append_log(f"📄 Transcript file: {transcript_file}")
                
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
        """Manual avoid ranges marked on the timeline (same process, so read them
        straight off the live timeline window). Safe if it's closed/absent."""
        tw = getattr(self, "timeline_window", None)
        if tw is not None and hasattr(tw, "get_avoid_ranges"):
            try:
                return tw.get_avoid_ranges()
            except Exception:
                pass
        return []

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
            
            # Create and show the timeline window
            self.timeline_window = SignalTimelineWindow(video_path, cache_data)
            self.timeline_window.show()
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
    app = QApplication(sys.argv)
    # Central theme: one graphite + accent stylesheet for all base widgets.
    # Additive — screens with their own inline styles still override it.
    _ui_theme.apply(app)
    # Reopen the live debug-log window if it was on last session (needs the
    # QApplication, hence here and not earlier).
    debug_console.restore_console_preference()
    gui = VideoHighlighterGUI()
    gui.show()
    exit_code = app.exec()

    # Backup hard-exit in case app.exec() does return (main closeEvent already
    # hard-exits, so this is belt-and-suspenders).
    _hard_exit(exit_code)