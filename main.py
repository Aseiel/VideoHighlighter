import sys
import os
import threading
import yaml
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QLineEdit, QSpinBox,
    QGroupBox, QTextEdit, QFormLayout, QProgressBar, QCheckBox,
    QComboBox, QTabWidget
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from pipeline import run_highlighter


CONFIG_FILE = "config.yaml"


class Worker(QThread):
    finished = Signal(str)
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
        file_layout = QHBoxLayout()
        self.video_input = QLineEdit(self.config_data.get("video", {}).get("path", ""))
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_file)
        file_layout.addWidget(QLabel("Input Video:"))
        file_layout.addWidget(self.video_input)
        file_layout.addWidget(browse_btn)
        layout.addLayout(file_layout)

        # --- Output filename ---
        out_layout = QHBoxLayout()
        self.output_input = QLineEdit(self.config_data.get("highlights", {}).get("output", "highlight.mp4"))
        out_layout.addWidget(QLabel("Output file:"))
        out_layout.addWidget(self.output_input)
        layout.addLayout(out_layout)

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

        scoring_cfg = self.config_data.get("scoring", {})
        highlights_cfg = self.config_data.get("highlights", {})

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
        scores_layout.addRow("Keyword points:", self.spin_keyword_points)
        scores_layout.addRow("Transcript points:", self.spin_transcript_points)
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
        self.sample_rate_spin.setValue(advanced_cfg.get("action_sample_rate", 5))  # default 5
        misc_layout.addRow("Frame skip (motion):", self.frame_skip_spin)
        misc_layout.addRow("Frame skip (objects):", self.obj_frame_skip_spin)
        misc_layout.addRow("Sample rate (actions):", self.sample_rate_spin)
        misc_layout.addRow("YOLO .pt path (optional):", self.yolo_pt_path)
        misc_layout.addRow("OpenVINO model folder (optional):", self.openvino_model_folder)
        misc_box.setLayout(misc_layout)
        advanced_layout.addWidget(misc_box)
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
            "video": {"path": self.video_input.text().strip()},
            "highlights": {
                "clip_time": int(self.spin_clip_time.value()),
                "output": self.output_input.text().strip(),
                "max_duration": int(self.spin_max_duration.value()),
                "exact_duration": int(self.spin_exact_duration.value()),
                "keep_temp": self.keep_temp_chk.isChecked(),
                "skip_highlights": self.skip_highlights_chk.isChecked(),
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
                "require_objects": self.actions_require_objects_chk.isChecked()  # NEW
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
                "yolo_pt_path": self.yolo_pt_path.text().strip(),
                "openvino_model_folder": self.openvino_model_folder.text().strip(),
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

    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Videos (*.mp4 *.mov *.avi *.mkv)")
        if file_path:
            self.video_input.setText(file_path)
            base, ext = os.path.splitext(file_path)
            self.output_input.setText(base + "_highlight" + (ext or ".mp4"))

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

    def run_pipeline(self):
        """Start the pipeline processing"""
        video = self.video_input.text().strip()
        if not video:
            self.append_log("⚠️ No video selected!")
            return

        if not os.path.exists(video):
            self.append_log(f"⚠️ Video file not found: {video}")
            return

        # Stop any existing worker
        if self.worker and self.worker.isRunning():
            self.append_log("⚠️ Pipeline already running!")
            return

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
            "output_file": self.output_input.text().strip() or None,
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
        self.append_log(f"📁 Input: {video}")
        self.append_log(f"📁 Output: {config.get('output_file', 'highlight.mp4')}")
        self.append_log("")

        # UI state changes
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.task_label.setText("🚀 Initializing...")
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        
        # Disable form inputs during processing
        self.video_input.setEnabled(False)
        self.output_input.setEnabled(False)

        # Create and start worker
        self.worker = Worker(video, config)
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
        self.video_input.setEnabled(True)
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