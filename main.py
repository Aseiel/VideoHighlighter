import sys
import os
import threading
import yaml
import cv2
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QLineEdit, QSpinBox,
    QGroupBox, QTextEdit, QFormLayout, QProgressBar, QCheckBox,
    QComboBox, QTabWidget, QListWidget, QSlider
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from pipeline import run_highlighter


CONFIG_FILE = "config.yaml"


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

        # Load saved paths if any
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
        self.video_duration_label = QLabel("Select a video to see duration")
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
        self.start_time_slider.setMaximum(100)  # Will be updated when video is loaded
        self.start_time_slider.setValue(0)
        self.start_time_slider.setEnabled(False)
        self.start_time_slider.valueChanged.connect(self.on_slider_changed)
        self.start_time_label = QLabel("00:00")
        self.start_time_label.setMinimumWidth(60)
        self.start_time_label.setStyleSheet("font-weight: bold;")
        start_slider_layout.addWidget(self.start_time_slider, stretch=1)
        start_slider_layout.addWidget(self.start_time_label)
        slider_layout.addLayout(start_slider_layout)

        # End position slider
        end_slider_layout = QHBoxLayout()
        end_slider_layout.addWidget(QLabel("End:"))
        self.end_time_slider = QSlider(Qt.Horizontal)
        self.end_time_slider.setMinimum(0)
        self.end_time_slider.setMaximum(100)  # Will be updated when video is loaded
        self.end_time_slider.setValue(100)
        self.end_time_slider.setEnabled(False)
        self.end_time_slider.valueChanged.connect(self.on_slider_changed)
        self.end_time_label = QLabel("00:00")
        self.end_time_label.setMinimumWidth(60)
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

        # Store video duration
        self.current_video_duration = 0

        # --- Progress Section ---
        progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        self.task_label = QLabel("Ready")
        self.task_label.setStyleSheet("color: #666; font-weight: bold;")
        progress_layout.addWidget(self.task_label)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # --- Tabs ---
        tabs = QTabWidget()

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

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setStyleSheet("QPushButton:enabled { background-color: #ff4444; color: white; font-weight: bold; }")
        self.cancel_btn.clicked.connect(self.cancel_pipeline)

        self.run_btn = QPushButton("Run Highlighter")
        self.run_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px; }")
        self.run_btn.clicked.connect(self.run_pipeline)

        ctrl_layout.addWidget(self.cancel_btn)
        ctrl_layout.addWidget(self.keep_temp_chk)
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

        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.check_worker_status)

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

    def get_file_list(self):
        """Get list of all files in the list widget"""
        return [self.file_list.item(i).text() for i in range(self.file_list.count())]


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

    def append_log(self, text):
        """Append text to log and auto-scroll"""
        self.log_output.append(text)
        # Auto-scroll to bottom
        scrollbar = self.log_output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        # Process events to keep UI responsive
        QApplication.processEvents()

    def update_progress(self, current, total, task_name, details=""):
        """Update progress bar and task label"""
        if total > 0:
            percentage = min(100, max(0, int((current/total)*100)))
            self.progress_bar.setValue(percentage)
            self.progress_bar.setVisible(True)
            self.task_label.setText(f"🔄 {task_name}: {percentage}% - {details}")
        else:
            self.task_label.setText(f"🔄 {task_name} - {details}")
        
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
        self.start_time_slider.setEnabled(checked and self.current_video_duration > 0)
        self.end_time_slider.setEnabled(checked and self.current_video_duration > 0)
        self.first_5min_btn.setEnabled(checked and self.current_video_duration > 0)
        self.last_5min_btn.setEnabled(checked and self.current_video_duration > 0)
        self.last_10min_btn.setEnabled(checked and self.current_video_duration > 0)
        self.middle_btn.setEnabled(checked and self.current_video_duration > 0)
        self.full_video_btn.setEnabled(checked and self.current_video_duration > 0)
        
        if checked and self.current_video_duration == 0:
            self.append_log("⚠️ Select a video first to enable time range")
            self.use_time_range_chk.setChecked(False)
        
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
        if self.current_video_duration == 0:
            self.start_time_label.setText("00:00")
            self.end_time_label.setText("00:00")
            self.selection_info_label.setText("Selection: No video loaded")
            return
        
        # Calculate actual times
        start_seconds = int((self.start_time_slider.value() / 100) * self.current_video_duration)
        end_seconds = int((self.end_time_slider.value() / 100) * self.current_video_duration)
        duration = end_seconds - start_seconds
        
        # Update labels
        self.start_time_label.setText(self.format_time(start_seconds))
        self.end_time_label.setText(self.format_time(end_seconds))
        
        # Update selection info
        percentage = (duration / self.current_video_duration) * 100 if self.current_video_duration > 0 else 0
        
        if self.use_time_range_chk.isChecked():
            self.selection_info_label.setText(
                f"Selection: {self.format_time(duration)} ({percentage:.1f}% of video)"
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
                
                # Reset to full range
                self.start_time_slider.setValue(0)
                self.end_time_slider.setValue(100)
                
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
        # For single file, use the source video's directory
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
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
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
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.pipeline_done)
        self.worker.cancelled.connect(self.pipeline_cancelled)
        
        # Start status checking timer
        self.status_timer.start(100)  # Check every 100ms
        
        self.worker.start()

    def cancel_pipeline(self):
        """Cancel the running pipeline"""
        if self.worker and self.worker.isRunning():
            self.append_log("\n⏹️ === CANCELLATION REQUESTED ===")
            self.task_label.setText("⏹️ Cancelling...")
            self.cancel_btn.setEnabled(False)
            self.cancel_btn.setText("Cancelling...")
            
            # Request cancellation
            self.worker.cancel()
            
            # Set a timeout for forced termination
            QTimer.singleShot(10000, self.force_worker_cleanup)  # 10 second timeout

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
                for item in output_file:
                    # Handle tuple format: (input_path, output_path)
                    if isinstance(item, tuple):
                        input_path, result_path = item
                        file = result_path
                    else:
                        file = item
                    
                    if file:
                        self.append_log(f"   • {file}")
                        
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
        self.progress_bar.setVisible(False)
        
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = VideoHighlighterGUI()
    gui.show()
    sys.exit(app.exec())