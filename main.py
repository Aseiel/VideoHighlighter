import sys
import os
import subprocess
import threading
import time
import yaml
import cv2
import re
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QLineEdit, QSpinBox,
    QGroupBox, QTextEdit, QFormLayout, QProgressBar, QCheckBox,
    QComboBox, QTabWidget, QListWidget, QSlider, 
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QMetaObject, Q_ARG, Slot
from pipeline import run_highlighter
from downloader import download_videos_with_immediate_processing, extract_video_links, DownloadError, reset_duration_method_cache

# At app startup
reset_duration_method_cache()

CONFIG_FILE = "config.yaml"

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

    # NEW: Signal to safely add items to GUI list widget from worker thread
    add_to_file_list = Signal(str)       # emits filepath to be added

    def __init__(self, url, save_dir, pattern, time_range=None, download_full=True,
                 use_percentages=False, immediate_processing=False, max_concurrent=1,
                 process_callback=None):
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
        self._cancelled = False
        self._is_running = False
        self._download_results = []                   # store all download metadata

    def run(self):
        try:
            self._is_running = True
            self.log.emit(f"🚀 Starting download from: {self.url}")

            # Decide which downloader to use
            if self.immediate_processing:
                try:
                    from downloader import download_videos_with_immediate_processing
                    use_new_downloader = True
                except ImportError:
                    self.log.emit("⚠️ Immediate processing module not found → falling back to standard mode")
                    use_new_downloader = False
            else:
                use_new_downloader = False

            def log_fn(message):
                self.log.emit(message)

            def progress_fn(current, total, status, message):
                self.progress.emit(current, total, status, message)

            # Define processing callback wrapper
            def wrapped_process_callback(filepath, metadata):
                if self._cancelled:
                    return {'cancelled': True}

                self.log.emit(f"🔧 Processing: {os.path.basename(filepath)}")

                if self.process_callback:
                    try:
                        result = self.process_callback(filepath, metadata)
                        self.log.emit(f"✅ Processed: {os.path.basename(filepath)}")
                        # Emit signal so GUI can react (e.g. show highlight created)
                        self.video_processed.emit(filepath, result)
                        return result
                    except Exception as e:
                        self.log.emit(f"❌ Processing failed: {e}")
                        return {'error': str(e)}
                return {'status': 'processed'}

            if use_new_downloader:
                results = download_videos_with_immediate_processing(
                    search_url=self.url,
                    save_dir=self.save_dir,
                    pattern=self.pattern,
                    log_fn=log_fn,
                    progress_fn=progress_fn,
                    process_callback=wrapped_process_callback,
                    cancel_flag=self,                           # pass self → uses .is_set() / .is_cancelled()
                    time_range=self.time_range,
                    download_full=self.download_full,
                    use_percentages=self.use_percentages,
                    max_workers=self.max_concurrent
                )

                # Collect downloaded files
                downloaded_files = []
                for result in results:
                    if result.get('success') and result.get('filepath'):
                        downloaded_files.append(result['filepath'])
                        self._download_results.append(result)

            else:
                # Fallback / legacy path (adjust if you still have old function)
                from downloader import download_videos  # assuming this exists
                downloaded_files = download_videos(
                    search_url=self.url,
                    save_dir=self.save_dir,
                    pattern=self.pattern,
                    log_fn=log_fn,
                    progress_fn=progress_fn,
                    cancel_flag=self,
                    time_range=self.time_range,
                    download_full=self.download_full,
                    use_percentages=self.use_percentages
                )

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
        """Request cancellation – called from GUI"""
        if self._is_running:
            self.log.emit("⏹️ Cancellation requested - stopping download...")
            self._cancelled = True
            # Give some time for graceful stop
            if not self.wait(5000):
                self.log.emit("⚠️ Thread did not stop gracefully → forcing termination")
                self.terminate()
                self.wait()

    def is_cancelled(self):
        """Public method used by downloader module to check cancellation"""
        return self._cancelled

    def is_set(self):
        """Compatibility alias – matches threading.Event.is_set()"""
        return self._cancelled
    
class Worker(QThread):
    finished = Signal(object)
    progress = Signal(int, int, str, str)
    log = Signal(str)
    cancelled = Signal()

    def __init__(self, video_path, gui_config=None):
        super().__init__()
        self.video_path = video_path
        self.gui_config = gui_config
        self._cancel_flag = threading.Event()
        self._is_running = False

    def run(self):
        try:
            self._is_running = True
            # Check if single or multiple files
            if isinstance(self.video_path, list):
                self.log.emit(f"🚀 Starting batch processing of {len(self.video_path)} videos...")
            else:
                self.log.emit("🚀 Starting video highlighter pipeline...")


            output = run_highlighter(
                self.video_path,
                gui_config=self.gui_config,
                log_fn=self.log.emit,
                progress_fn=lambda cur, tot, task, det: self.progress.emit(cur, tot, task, det),
                cancel_flag=self._cancel_flag
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


class VideoHighlighterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Highlighter - Highlights & Subtitles")
        self.setGeometry(200, 200, 1000, 800)
        self.worker = None

        self.config_data = self.load_config()

        layout = QVBoxLayout()

        # Store video duration
        self.current_video_duration = 0

        # --- File picker ---
        file_group = QGroupBox("Input Videos")
        file_layout = QVBoxLayout()

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
        info_label = QLabel("ℹ️ For multiple files, '_highlight' will be appended to each filename")
        info_label.setStyleSheet("color: #666; font-size: 9pt;")
        out_layout.addWidget(info_label)
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

        # Start position slider
        start_slider_layout = QHBoxLayout()
        start_slider_layout.addWidget(QLabel("Start:"))
        self.start_time_slider = QSlider(Qt.Horizontal)
        self.start_time_slider.setMinimum(0)
        self.start_time_slider.setMaximum(100)
        self.start_time_slider.setValue(highlights_cfg.get("range_start_pct", 0))
        self.start_time_slider.setEnabled(False)
        self.start_time_slider.valueChanged.connect(self.on_slider_changed)
        self.start_time_label = QLabel("0%")
        self.start_time_label.setMinimumWidth(80)
        self.start_time_label.setStyleSheet("font-weight: bold;")
        start_slider_layout.addWidget(self.start_time_slider, stretch=1)
        start_slider_layout.addWidget(self.start_time_label)
        slider_layout.addLayout(start_slider_layout)

        # End position slider
        end_slider_layout = QHBoxLayout()
        end_slider_layout.addWidget(QLabel("End:"))
        self.end_time_slider = QSlider(Qt.Horizontal)
        self.end_time_slider.setMinimum(0)
        self.end_time_slider.setMaximum(100)
        self.end_time_slider.setValue(highlights_cfg.get("range_end_pct", 100))
        self.end_time_slider.setEnabled(False)
        self.end_time_slider.valueChanged.connect(self.on_slider_changed)
        self.end_time_label = QLabel("100%")
        self.end_time_label.setMinimumWidth(80)
        self.end_time_label.setStyleSheet("font-weight: bold;")
        end_slider_layout.addWidget(self.end_time_slider, stretch=1)
        end_slider_layout.addWidget(self.end_time_label)
        slider_layout.addLayout(end_slider_layout)

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

        # Initialize the selection info display with saved values
        self.update_selection_info()

        # --- Progress Section ---
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout()

        progress_layout.addWidget(QLabel("Download"))
        self.download_progress_bar = QProgressBar()
        self.download_progress_bar.setVisible(False)
        self.download_progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.download_progress_bar)

        progress_layout.addWidget(QLabel("Processing"))
        self.process_progress_bar = QProgressBar()
        self.process_progress_bar.setVisible(False)
        self.process_progress_bar.setRange(0, 100)
        progress_layout.addWidget(self.process_progress_bar)

        self.task_label = QLabel("Ready")
        self.task_label.setStyleSheet("color: #666; font-weight: bold;")
        progress_layout.addWidget(self.task_label)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

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

        # Pattern input
        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("Link pattern:"))
        self.download_pattern_input = QLineEdit()
        self.download_pattern_input.setText(self.config_data.get("download", {}).get("link_pattern", "/video/"))
        self.download_pattern_input.setPlaceholderText("/video/")
        self.download_pattern_input.setToolTip("Pattern to match in video links (e.g., /video/, /watch/)")
        pattern_layout.addWidget(self.download_pattern_input)
        download_form.addLayout(pattern_layout)

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

        # Time range selection for downloads
        time_range_group = QGroupBox("Download Time Range (Optional)")
        time_range_layout = QVBoxLayout()

        # Full download checkbox (default: unchecked = download only time range)
        self.download_full_chk = QCheckBox("Download full video")
        self.download_full_chk.setChecked(False)  # Default: download only time range
        self.download_full_chk.setToolTip("When unchecked, only downloads the specified time range")
        time_range_layout.addWidget(self.download_full_chk)

        # Time range inputs
        time_input_layout = QHBoxLayout()
        time_input_layout.addWidget(QLabel("Start time (seconds):"))
        self.download_start_input = QSpinBox()
        self.download_start_input.setRange(0, 86400)  # 0 to 24 hours
        self.download_start_input.setValue(0)
        self.download_start_input.setEnabled(True)  # Enabled by default
        time_input_layout.addWidget(self.download_start_input)

        time_input_layout.addWidget(QLabel("End time (seconds):"))
        self.download_end_input = QSpinBox()
        self.download_end_input.setRange(1, 86400)  # 1 second to 24 hours
        self.download_end_input.setValue(300)  # Default: 5 minutes
        self.download_end_input.setEnabled(True)  # Enabled by default
        time_input_layout.addWidget(self.download_end_input)

        time_range_layout.addLayout(time_input_layout)

        # Duration label
        self.download_duration_label = QLabel("Duration: 300s (5:00)")
        time_range_layout.addWidget(self.download_duration_label)

        # Connect signals
        self.download_start_input.valueChanged.connect(self.update_download_duration)
        self.download_end_input.valueChanged.connect(self.update_download_duration)
        self.download_full_chk.toggled.connect(self.on_download_full_toggle)

        time_range_group.setLayout(time_range_layout)
        download_form.addWidget(time_range_group)

        # Download time range options
        download_time_group = QGroupBox("Download Time Range")
        download_time_layout = QVBoxLayout()

        # Checkbox to use the same time range as processing
        self.use_same_time_range_chk = QCheckBox("Use same time range as processing")
        self.use_same_time_range_chk.setChecked(False)  # Default: download full
        self.use_same_time_range_chk.setToolTip("When checked, downloads only the time range specified in 'Processing Time Range' section")
        download_time_layout.addWidget(self.use_same_time_range_chk)

        # Info label
        download_time_info = QLabel("ℹ️ Unchecked: Download full videos\n   Checked: Download only selected time range")
        download_time_info.setStyleSheet("color: #666; font-size: 9pt; font-style: italic;")
        download_time_layout.addWidget(download_time_info)

        download_time_group.setLayout(download_time_layout)
        download_form.addWidget(download_time_group)

        # Options
        self.auto_add_downloaded_chk = QCheckBox("Automatically add downloaded videos to file list")
        self.auto_add_downloaded_chk.setChecked(self.config_data.get("download", {}).get("auto_add", True))
        download_form.addWidget(self.auto_add_downloaded_chk)

        # Auto-process checkbox
        self.auto_process_chk = QCheckBox("Automatically start processing after download completes")
        self.auto_process_chk.setChecked(self.config_data.get("download", {}).get("auto_process", False))
        self.auto_process_chk.setToolTip("When enabled, the highlighter pipeline will start automatically after videos are downloaded")
        download_form.addWidget(self.auto_process_chk)

        # Immediate processing checkbox
        self.immediate_processing_chk = QCheckBox("Process each video immediately after download")
        self.immediate_processing_chk.setChecked(self.config_data.get("download", {}).get("immediate_processing", True))
        self.immediate_processing_chk.setToolTip("Process videos as soon as they're downloaded, instead of waiting for all downloads to complete")
        download_form.addWidget(self.immediate_processing_chk)

        # Concurrent downloads spinner
        concurrent_layout = QHBoxLayout()
        concurrent_layout.addWidget(QLabel("Concurrent downloads:"))
        self.concurrent_spinbox = QSpinBox()
        self.concurrent_spinbox.setRange(1, 10)
        self.concurrent_spinbox.setValue(self.config_data.get("download", {}).get("concurrent_downloads", 1))
        self.concurrent_spinbox.setToolTip("Number of videos to download simultaneously (higher = faster but more resource intensive)")
        self.concurrent_spinbox.setEnabled(self.immediate_processing_chk.isChecked())
        concurrent_layout.addWidget(self.concurrent_spinbox)
        concurrent_layout.addStretch()
        download_form.addLayout(concurrent_layout)

        # Connect checkbox to enable/disable spinner
        self.immediate_processing_chk.toggled.connect(self.concurrent_spinbox.setEnabled)


        # Download button
        download_btn_layout = QHBoxLayout()
        self.download_btn = QPushButton("🌐 Download Videos")
        self.download_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 8px; }")
        self.download_btn.clicked.connect(self.start_download)
        download_btn_layout.addStretch()
        download_btn_layout.addWidget(self.download_btn)
        download_form.addLayout(download_btn_layout)

        # Combine highlights
        self.auto_combine_chk = QCheckBox("Automatically combine all highlights into one video")
        self.auto_combine_chk.setChecked(self.config_data.get("download", {}).get("auto_combine", True))
        self.auto_combine_chk.setToolTip("When enabled, all individual highlights will be combined into one master video")
        download_form.addWidget(self.auto_combine_chk)
        
        # Info label
        info_label = QLabel("ℹ️ Requires yt-dlp: pip install yt-dlp")
        info_label.setStyleSheet("color: #666; font-size: 9pt; font-style: italic;")
        download_form.addWidget(info_label)
        
        download_group.setLayout(download_form)
        download_layout.addWidget(download_group)
        download_layout.addStretch()
        download_tab.setLayout(download_layout)
        tabs.addTab(download_tab, "Download")

        # --- Tab 1: Basic Settings ---
        basic_tab = QWidget()
        basic_layout = QVBoxLayout()
        scores_box = QGroupBox("Points & durations")
        scores_layout = QFormLayout()

        # Scoring points
        self.spin_scene_points = QSpinBox(); self.spin_scene_points.setRange(0,100); self.spin_scene_points.setValue(scoring_cfg.get("scene_points", 0))
        self.spin_motion_event_points = QSpinBox(); self.spin_motion_event_points.setRange(0,100); self.spin_motion_event_points.setValue(scoring_cfg.get("motion_event_points", 0))
        self.spin_motion_peak = QSpinBox(); self.spin_motion_peak.setRange(0,100); self.spin_motion_peak.setValue(scoring_cfg.get("motion_peak_points", 3))
        self.spin_audio_peak = QSpinBox(); self.spin_audio_peak.setRange(0,100); self.spin_audio_peak.setValue(scoring_cfg.get("audio_peak_points", 0))
        self.spin_keyword_points = QSpinBox(); self.spin_keyword_points.setRange(0,100); self.spin_keyword_points.setValue(scoring_cfg.get("keyword_points", 2))
        self.spin_transcript_points = QSpinBox(); self.spin_transcript_points.setRange(0,100); self.spin_transcript_points.setValue(scoring_cfg.get("transcript_points", 2))
        self.spin_object = QSpinBox(); self.spin_object.setRange(0,100); self.spin_object.setValue(scoring_cfg.get("object_points", 1))
        self.spin_action = QSpinBox(); self.spin_action.setRange(0,1000); self.spin_action.setValue(scoring_cfg.get("action_points", 10))
        self.spin_clip_time = QSpinBox(); self.spin_clip_time.setRange(1,300); self.spin_clip_time.setValue(highlights_cfg.get("clip_time", 10))
        self.spin_max_duration = QSpinBox(); self.spin_max_duration.setRange(1,3600); self.spin_max_duration.setValue(highlights_cfg.get("max_duration", 420))
        self.spin_exact_duration = QSpinBox(); self.spin_exact_duration.setRange(0,3600); self.spin_exact_duration.setValue(highlights_cfg.get("exact_duration", 0))

         # Add to form layout
        scores_layout.addRow("Scene points:", self.spin_scene_points)
        scores_layout.addRow("Motion event points:", self.spin_motion_event_points)
        scores_layout.addRow("Motion peak points:", self.spin_motion_peak)
        scores_layout.addRow("Audio peak points:", self.spin_audio_peak)
        scores_layout.addRow("Keyword points (keywords in transcript):", self.spin_keyword_points)
        scores_layout.addRow("Transcript points (all words):", self.spin_transcript_points)
        scores_layout.addRow("Object points:", self.spin_object)
        scores_layout.addRow("Action points:", self.spin_action)
        scores_layout.addRow("Clip time (s):", self.spin_clip_time)
        scores_layout.addRow("Max highlight duration (s):", self.spin_max_duration)
        scores_layout.addRow("Exact duration (s, 0 = off):", self.spin_exact_duration)
        scores_box.setLayout(scores_layout)
        basic_layout.addWidget(scores_box)

        # Highlight object classes
        obj_layout = QHBoxLayout()
        self.objects_input = QLineEdit(",".join(self.config_data.get("objects", {}).get("interesting", [])))
        self.objects_input.setPlaceholderText("person,glass,wine glass,sports ball")
        obj_layout.addWidget(QLabel("Object detection:"))
        obj_layout.addWidget(self.objects_input)
        basic_layout.addLayout(obj_layout)

        # Action keywords
        action_kw_layout = QHBoxLayout()
        self.actions_input = QLineEdit(",".join(self.config_data.get("actions", {}).get("interesting", [])))
        self.actions_input.setPlaceholderText("high jump, high kick, archery")
        action_kw_layout.addWidget(QLabel("Action keywords:"))
        action_kw_layout.addWidget(self.actions_input)
        basic_layout.addLayout(action_kw_layout)

        # Conditional action scoring checkbox
        self.actions_require_objects_chk = QCheckBox("Only score actions when objects detected")
        self.actions_require_objects_chk.setChecked(self.config_data.get("actions", {}).get("require_objects", False))
        self.actions_require_objects_chk.setToolTip("Actions will only add points if objects are also detected in that timeframe")
        basic_layout.addWidget(self.actions_require_objects_chk)

        self.skip_highlights_chk = QCheckBox("Skip highlights")
        self.skip_highlights_chk.setChecked(highlights_cfg.get("skip_highlights", False))
        basic_layout.addWidget(self.skip_highlights_chk)

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
        
        self.source_lang_combo = QComboBox()
        self.source_lang_combo.addItems(["en","pl","es","fr","de","it","pt","ru","ja","ko","zh"])
        self.source_lang_combo.setCurrentText(subtitles_cfg.get("source_lang", "en"))
        self.source_lang_combo.setEnabled(subtitles_cfg.get("enabled", False) and transcript_cfg.get("enabled", False))
        subtitle_form.addRow("Source language:", self.source_lang_combo)
        
        self.target_lang_combo = QComboBox()
        self.target_lang_combo.addItems(["en","pl","es","fr","de","it","pt","ru","ja","ko","zh"])
        self.target_lang_combo.setCurrentText(subtitles_cfg.get("target_lang", "pl"))
        self.target_lang_combo.setEnabled(subtitles_cfg.get("enabled", False) and transcript_cfg.get("enabled", False))
        subtitle_form.addRow("Target language:", self.target_lang_combo)
        subtitle_group.setLayout(subtitle_form)
        transcript_layout.addWidget(subtitle_group)

        transcript_tab.setLayout(transcript_layout)
        tabs.addTab(transcript_tab, "Transcript & Subtitles")

        # --- Tab3: Advanced Tab ---
        advanced_cfg = self.config_data.get("advanced", {})
        visualization_cfg = self.config_data.get("visualization", {})

        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout()
        misc_box = QGroupBox("Advanced / Optional")
        misc_layout = QFormLayout()
        self.frame_skip_spin = QSpinBox(); self.frame_skip_spin.setRange(1,30); self.frame_skip_spin.setValue(advanced_cfg.get("frame_skip", 5))
        self.obj_frame_skip_spin = QSpinBox(); self.obj_frame_skip_spin.setRange(1,60); self.obj_frame_skip_spin.setValue(advanced_cfg.get("object_frame_skip", 10))
        self.yolo_pt_path = QLineEdit(advanced_cfg.get("yolo_pt_path", ""))
        self.openvino_model_folder = QLineEdit(advanced_cfg.get("openvino_model_folder", ""))
        self.sample_rate_spin = QSpinBox()
        self.sample_rate_spin.setRange(1,30)  # 1 = process every frame, 30 = every 30th frame
        self.sample_rate_spin.setValue(advanced_cfg.get("sample_rate", 5))  # default 5
        misc_layout.addRow("Frame skip (motion):", self.frame_skip_spin)
        misc_layout.addRow("Frame skip (objects):", self.obj_frame_skip_spin)
        misc_layout.addRow("Sample rate (actions):", self.sample_rate_spin)
        misc_layout.addRow("YOLO .pt path (optional):", self.yolo_pt_path)
        misc_layout.addRow("OpenVINO model folder (optional):", self.openvino_model_folder)
        misc_box.setLayout(misc_layout)
        advanced_layout.addWidget(misc_box)
        
        # --- Bounding Box Visualization Options ---
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
        advanced_layout.addWidget(bbox_box)
        
        advanced_tab.setLayout(advanced_layout)
        tabs.addTab(advanced_tab, "Advanced")

        layout.addWidget(tabs)

        # --- Run / Cancel Controls ---
        ctrl_layout = QHBoxLayout()
        self.keep_temp_chk = QPushButton("Keep temp clips: ON" if highlights_cfg.get("keep_temp", False) else "Keep temp clips: OFF")
        self.keep_temp_chk.setCheckable(True)
        self.keep_temp_chk.setChecked(highlights_cfg.get("keep_temp", False))
        self.keep_temp_chk.clicked.connect(lambda: self.keep_temp_chk.setText(
            "Keep temp clips: ON" if self.keep_temp_chk.isChecked() else "Keep temp clips: OFF"))

        self.timeline_btn = QPushButton("📊 Show Timeline Viewer")
        self.timeline_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 8px; }")
        self.timeline_btn.clicked.connect(self.open_timeline_viewer)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet("QPushButton:enabled { background-color: #ff4444; color: white; font-weight: bold; }")
        self.cancel_btn.clicked.connect(self.cancel_pipeline)

        self.run_btn = QPushButton("Run Highlighter")
        self.run_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        self.run_btn.clicked.connect(self.run_pipeline)

        ctrl_layout.addWidget(self.cancel_btn)
        ctrl_layout.addWidget(self.keep_temp_chk)
        ctrl_layout.addWidget(self.timeline_btn)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.run_btn)
        layout.addLayout(ctrl_layout)

        # --- Log view ---
        layout.addWidget(QLabel("Log Output:"))
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("QTextEdit { font-family: 'Courier New', monospace; font-size: 9pt; }")
        layout.addWidget(self.log_output)

        self.setLayout(layout)

        # Load download config
        download_cfg = self.config_data.get("download", {})
        self.use_same_time_range_chk.setChecked(download_cfg.get("use_same_time_range", False))


        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_worker_status)

        # Load download time range settings (AFTER all widgets are created)
        download_cfg = self.config_data.get("download", {})
        self.download_full_chk.setChecked(download_cfg.get("download_full", False))
        self.download_start_input.setValue(download_cfg.get("time_range_start", 0))
        self.download_end_input.setValue(download_cfg.get("time_range_end", 300))

        # Initialize the UI state
        self.on_download_full_toggle(self.download_full_chk.isChecked())

    # --- Downloader methods ---
    def browse_save_directory(self):
        """Browse for save directory"""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Save Directory", self.download_save_dir_input.text()
        )
        if directory:
            self.download_save_dir_input.setText(directory)

    def start_download(self):
        """Start the download process"""
        url = self.download_url_input.text().strip()
        save_dir = self.download_save_dir_input.text().strip()
        pattern = self.download_pattern_input.text().strip() or "/video/"
        
        # Get immediate processing settings
        immediate_processing = self.immediate_processing_chk.isChecked()
        max_concurrent = self.concurrent_spinbox.value() if immediate_processing else 1
        
        # Get time range settings
        use_same_time_range = self.use_same_time_range_chk.isChecked()
        time_range = None
        use_percentages = False
        
        if use_same_time_range:
            if not self.use_time_range_chk.isChecked():
                self.append_log("⚠️ 'Process only specific time range' is not enabled")
                return
            
            # Get percentage values directly from sliders
            start_pct = self.start_time_slider.value()
            end_pct = self.end_time_slider.value()
            
            if end_pct <= start_pct:
                self.append_log("⚠️ Invalid time range - end must be greater than start")
                return
            
            time_range = (float(start_pct), float(end_pct))
            use_percentages = True  # Use percentages directly!
            download_full = False
            
            # Log the percentage range
            self.append_log(f"⏱️ Downloading percentage range: {start_pct}% - {end_pct}%")
            self.append_log(f"   (yt-dlp will handle the percentage conversion automatically)")
        else:
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
        self.append_log("=== Starting Video Download ===")
        self.append_log(f"🌐 URL: {url}")
        self.append_log(f"📁 Save directory: {save_dir}")
        self.append_log(f"🔍 Pattern: {pattern}")
        
        if immediate_processing:
            self.append_log(f"⚡ Mode: Immediate processing after each download")
            self.append_log(f"   Concurrent downloads: {max_concurrent}")
        else:
            self.append_log("📦 Mode: Batch download (process all videos at once)")
        
        if download_full:
            self.append_log("📥 Downloading: Full videos")
        else:
            start_pct, end_pct = time_range
            self.append_log(f"⏱️ Downloading: Percentage range {start_pct}% - {end_pct}%")
        
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
            process_callback=process_video_callback if immediate_processing else None
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
            "highlight_objects": highlight_objects,
            "interesting_actions": interesting_actions,
            "actions_require_objects": self.actions_require_objects_chk.isChecked(),
            "use_transcript": use_transcript,
            "transcript_model": self.transcript_model_combo.currentText(),
            "search_keywords": search_keywords,
            "create_subtitles": self.subtitles_checkbox.isChecked() and use_transcript,
            "source_lang": self.source_lang_combo.currentText(),
            "target_lang": self.target_lang_combo.currentText(),
            "skip_highlights": self.skip_highlights_chk.isChecked(),
            "frame_skip": int(self.frame_skip_spin.value()),
            "object_frame_skip": int(self.obj_frame_skip_spin.value()),
            "yolo_pt_path": self.yolo_pt_path.text().strip() or None,
            "openvino_model_folder": self.openvino_model_folder.text().strip() or None,
            "sample_rate": int(self.sample_rate_spin.value()),
            "draw_object_boxes": self.bbox_objects_chk.isChecked(),
            "draw_action_labels": self.bbox_actions_chk.isChecked(),
        }
        
        # Add time range if enabled
        if self.use_time_range_chk.isChecked() and self.current_video_duration > 0:
            start_pct = self.start_time_slider.value() / 100
            end_pct = self.end_time_slider.value() / 100
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


    def on_download_full_toggle(self, checked):
        """Enable/disable time range inputs based on full download checkbox"""
        self.download_start_input.setEnabled(not checked)
        self.download_end_input.setEnabled(not checked)
        if checked:
            self.download_duration_label.setText("Downloading full videos")
        else:
            self.update_download_duration()

    def update_download_duration(self):
        """Update the duration label for download time range"""
        if self.download_full_chk.isChecked():
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
            if self.immediate_processing_chk.isChecked():
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
        
        self.download_cleanup()

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
        if not self.auto_process_chk.isChecked() or self.file_list.count() == 0:
            self.download_progress_bar.setVisible(False)
            # If you're not auto-processing, also hide processing bar
            self.process_progress_bar.setVisible(False)

        
        # Re-enable controls
        self.download_btn.setEnabled(True)
        
        # Only re-enable cancel if not auto-processing
        if not self.auto_process_chk.isChecked() or self.file_list.count() == 0:
            self.cancel_btn.setEnabled(False)
            self.cancel_btn.setText("Cancel")
        
        # Reset task label style after 5 seconds (only if not auto-processing)
        if not self.auto_process_chk.isChecked() or self.file_list.count() == 0:
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
                "link_pattern": self.download_pattern_input.text().strip() or "/video/",
                "save_dir": self.download_save_dir_input.text().strip(),
                "auto_add": self.auto_add_downloaded_chk.isChecked(),
                "auto_process": self.auto_process_chk.isChecked(),
                "auto_combine": self.auto_combine_chk.isChecked(),
                "use_same_time_range": self.use_same_time_range_chk.isChecked(),
                "immediate_processing": self.immediate_processing_chk.isChecked(),  # NEW
                "concurrent_downloads": self.concurrent_spinbox.value(),  # NEW
                "download_full": self.download_full_chk.isChecked(),  # Already exists
                "time_range_start": self.download_start_input.value(),  # Already exists
                "time_range_end": self.download_end_input.value(),  # Already exists
            },
            "highlights": {
                "clip_time": int(self.spin_clip_time.value()),
                "output": self.output_input.text().strip(),
                "max_duration": int(self.spin_max_duration.value()),
                "exact_duration": int(self.spin_exact_duration.value()),
                "keep_temp": self.keep_temp_chk.isChecked(),
                "skip_highlights": self.skip_highlights_chk.isChecked(),
                "use_time_range": self.use_time_range_chk.isChecked(),
                "range_start_pct": self.start_time_slider.value(),
                "range_end_pct": self.end_time_slider.value(),
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
            "objects": {"interesting": get_text_list(self.objects_input)},
            "keywords": {
                "transcript_file": "transcript.txt",
                "interesting": get_text_list(self.search_keywords_input),
            },
            "transcript": {
                "enabled": self.transcript_checkbox.isChecked(),
                "model": self.transcript_model_combo.currentText(),
                "search_keywords": get_text_list(self.search_keywords_input),
            },
            "subtitles": {
                "enabled": self.subtitles_checkbox.isChecked(),
                "source_lang": self.source_lang_combo.currentText(),
                "target_lang": self.target_lang_combo.currentText(),
            },
            "advanced": {
                "frame_skip": int(self.frame_skip_spin.value()),
                "object_frame_skip": int(self.obj_frame_skip_spin.value()),
                "sample_rate": int(self.sample_rate_spin.value()),
                "yolo_pt_path": self.yolo_pt_path.text().strip(),
                "openvino_model_folder": self.openvino_model_folder.text().strip(),
            },
            "visualization": {
                "draw_object_boxes": self.bbox_objects_chk.isChecked(),
                "draw_action_labels": self.bbox_actions_chk.isChecked(),
            },
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False, allow_unicode=True)
            
    def closeEvent(self, event):
        self.save_config()
        event.accept()

    def check_worker_status(self):
        """Periodic check of worker status for UI responsiveness"""
        if self.worker and not self.worker.isRunning():
            self.status_timer.stop()

    def on_transcript_toggle(self, checked):
        """Handle transcript checkbox toggle"""
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
        
        self.source_lang_combo.setEnabled(final_state)
        self.target_lang_combo.setEnabled(final_state)

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
        self.log_output.append(text)
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_progress(self, current, total, task_name, details=""):
        # Decide which bar based on task_name or status
        if "download" in task_name.lower() or "extract" in task_name.lower():
            self.update_download_progress(current, total, task_name, details)
        else:
            self.update_process_progress(current, total, task_name, details)

    @Slot(str)
    def set_download_busy(self, text: str):
        self.download_progress_bar.setVisible(True)
        self.download_progress_bar.setRange(0, 0)  # indeterminate
        self.task_label.setText(text)

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
        self.start_time_slider.setEnabled(checked)
        self.end_time_slider.setEnabled(checked)
        
        # Preset buttons only work when video duration is known
        has_duration = self.current_video_duration > 0
        self.first_5min_btn.setEnabled(checked and has_duration)
        self.last_5min_btn.setEnabled(checked and has_duration)
        self.last_10min_btn.setEnabled(checked and has_duration)
        self.middle_btn.setEnabled(checked and has_duration)
        self.full_video_btn.setEnabled(checked and has_duration)
        
        self.update_selection_info()

    def on_slider_changed(self):
        """Handle slider value changes"""
        # Ensure start is always before end
        if self.start_time_slider.value() >= self.end_time_slider.value():
            if self.sender() == self.start_time_slider:
                # Start moved, adjust to be 1 second before end
                self.start_time_slider.setValue(max(0, self.end_time_slider.value() - 1))
            else:
                # End moved, adjust to be 1 second after start
                self.end_time_slider.setValue(min(self.start_time_slider.maximum(), 
                                                self.start_time_slider.value() + 1))
        
        self.update_selection_info()

    def update_selection_info(self):
        """Update the selection information labels"""
        start_pct = self.start_time_slider.value()
        end_pct = self.end_time_slider.value()
        
        if self.current_video_duration == 0:
            # No video loaded - show percentages
            self.start_time_label.setText(f"{start_pct}%")
            self.end_time_label.setText(f"{end_pct}%")
            
            if self.use_time_range_chk.isChecked():
                range_pct = end_pct - start_pct
                self.selection_info_label.setText(
                    f"Selection: {start_pct}% to {end_pct}% ({range_pct}% of video)"
                )
                self.selection_info_label.setStyleSheet("color: #2196F3; font-weight: bold; font-size: 10pt;")
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
            self.selection_info_label.setStyleSheet("color: #2196F3; font-weight: bold; font-size: 10pt;")
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
                self.start_time_slider.setMaximum(100)
                self.end_time_slider.setMaximum(100)
                
                # Keep existing slider values (don't reset user's choice)
                # Only update the display labels
                
                # Update labels
                self.video_duration_label.setText(
                    f"Video duration: {self.format_time(duration)} ({duration}s)"
                )
                self.video_duration_label.setStyleSheet("color: #4CAF50; font-style: italic;")
                
                # Enable controls if checkbox is checked
                if self.use_time_range_chk.isChecked():
                    self.start_time_slider.setEnabled(True)
                    self.end_time_slider.setEnabled(True)
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
        
        self.start_time_slider.setValue(start_pct)
        self.end_time_slider.setValue(end_pct)
        
        start_time = int((start_pct / 100) * duration)
        end_time = int((end_pct / 100) * duration)
        self.append_log(f"✅ Preset '{preset_type}': {self.format_time(start_time)} to {self.format_time(end_time)}")


    def run_pipeline(self):
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
            "output_file": output_file,
            "highlight_objects": highlight_objects,
            "interesting_actions": interesting_actions,
            "actions_require_objects": self.actions_require_objects_chk.isChecked(),
            "use_transcript": use_transcript,
            "transcript_model": self.transcript_model_combo.currentText(),
            "search_keywords": search_keywords,
            "create_subtitles": self.subtitles_checkbox.isChecked() and use_transcript,
            "source_lang": self.source_lang_combo.currentText(),
            "target_lang": self.target_lang_combo.currentText(),
            "skip_highlights": self.skip_highlights_chk.isChecked(),
            "frame_skip": int(self.frame_skip_spin.value()),
            "object_frame_skip": int(self.obj_frame_skip_spin.value()),
            "yolo_pt_path": self.yolo_pt_path.text().strip() or None,
            "openvino_model_folder": self.openvino_model_folder.text().strip() or None,
            "sample_rate": int(self.sample_rate_spin.value()),
            "draw_object_boxes": self.bbox_objects_chk.isChecked(),
            "draw_action_labels": self.bbox_actions_chk.isChecked(),
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
        self.append_log("=== Starting Video Highlighter Pipeline ===")
        self.append_log(f"📁 Input: {video_paths}")
        self.append_log(f"📁 Output: {config.get('output_file', 'highlight.mp4')}")
        if config.get('draw_object_boxes') or config.get('draw_action_labels'):
            self.append_log("🎨 Bounding box visualization enabled for temp files")
        self.append_log("")

        if self.use_time_range_chk.isChecked() and self.current_video_duration > 0:
            start_pct = self.start_time_slider.value() / 100
            end_pct = self.end_time_slider.value() / 100
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
        self.task_label.setText("🚀 Initializing...")
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        
        # Disable form inputs during processing
        self.file_list.setEnabled(False)
        self.output_input.setEnabled(False)
        self.browse_btn.setEnabled(False)
        self.remove_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

        # Create and start worker
        self.worker = Worker(video_paths, config)
        self.worker.log.connect(self.append_log)
        self.worker.progress.connect(self.update_process_progress)
        self.worker.finished.connect(self.pipeline_done)
        self.worker.cancelled.connect(self.pipeline_cancelled)
        
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
            self.download_worker.cancel()
            QTimer.singleShot(10000, self.force_download_cleanup)
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

    def force_download_cleanup(self):
        """Force cleanup if download worker doesn't stop gracefully"""
        if hasattr(self, 'download_worker') and self.download_worker and self.download_worker.isRunning():
            self.append_log("⚠️ Forcing download termination...")
            self.download_worker.terminate()
            self.download_worker.wait(3000)
            self.download_cleanup()

    def force_worker_cleanup(self):
        """Force cleanup if worker doesn't stop gracefully"""
        if self.worker and self.worker.isRunning():
            self.append_log("⚠️ Forcing pipeline termination...")
            self.worker.terminate()
            self.worker.wait(3000)  # Wait up to 3 seconds
            self.pipeline_cleanup()

    def pipeline_done(self, output_file):
        """Handle pipeline completion"""
        self.status_timer.stop()
        
        if output_file and not self.worker.is_cancelled():
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
                        srt_file = f"{base_name}_{self.target_lang_combo.currentText()}.srt"
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
        elif not self.worker.is_cancelled():
            self.append_log("\n⚠️ === PIPELINE COMPLETED WITH ERRORS ===")
            self.append_log("❌ No output file was generated. Check the log for errors.")
            self.task_label.setText("❌ Failed")
            self.task_label.setStyleSheet("color: #f44336; font-weight: bold;")
        
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
            
            # Check if cache exists
            from modules.video_cache import VideoAnalysisCache
            cache = VideoAnalysisCache()
            cache_data = cache.load(video_path)
            
            if not cache_data:
                self.append_log("⚠️ No analysis cache found for this video.")
                self.append_log("   Please run the highlighter pipeline first to generate analysis data.")
                return
            
            self.append_log(f"📊 Opening timeline viewer for: {os.path.basename(video_path)}")
            
            # Create and show the timeline window
            self.timeline_window = SignalTimelineWindow(video_path, cache_data)
            self.timeline_window.show()
            
        except ImportError as e:
            self.append_log(f"❌ Failed to import timeline viewer: {e}")
            self.append_log("   Make sure signal_timeline_viewer.py is in the same directory.")
        except Exception as e:
            self.append_log(f"❌ Failed to open timeline viewer: {e}")
            import traceback
            self.append_log(traceback.format_exc())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = VideoHighlighterGUI()
    gui.show()
    sys.exit(app.exec())