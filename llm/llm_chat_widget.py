"""
llm_chat_widget.py - Embeddable Qt chat panel for VideoHighlighter.

Key feature: AUTO-LOADS the most recent cache file from ./cache/ on startup,
so the LLM always has video context even after restarting the app.

Usage:
    chat = LLMChatWidget(parent=self, cache_dir="./cache")
    layout.addWidget(chat)
"""

from __future__ import annotations

import os
import json
import time as _time
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QLineEdit, QComboBox,
    QGroupBox, QFileDialog, QApplication,
    QDialog, QDialogButtonBox,
)
from PySide6.QtCore import Qt, Signal, Slot, QThread
from PySide6.QtGui import QTextCursor

from .llm_module import LLMModule, VideoContextBuilder, get_available_backends, get_ollama_models

# timeline bridge (only available when timeline viewer is present)
try:
    from .llm_timeline_bridge import TimelineBridge
    HAS_TIMELINE_BRIDGE = True
except ImportError:
    HAS_TIMELINE_BRIDGE = False


# ---------------------------------------------------------------------------
# Worker thread for LLM queries
# ---------------------------------------------------------------------------
class _LLMWorker(QThread):
    """Runs LLM.query() off the GUI thread with token-by-token streaming."""

    token_received = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, llm: LLMModule, message: str,
                 analysis_data: dict | None = None,
                 video_path: str = "",
                 timeline_context: str = "",
                 frame_base64: str | None = None):
        super().__init__()
        self.llm = llm
        self.message = message
        self.analysis_data = analysis_data
        self.video_path = video_path
        self.timeline_context = timeline_context
        self.frame_base64 = frame_base64
        self._cancel = False

    def cancel(self):
        """Request cancellation of the current generation."""
        self._cancel = True

    def run(self):
        try:
            def _stream_callback(token: str):
                if self._cancel:
                    raise _GenerationCancelled()
                self.token_received.emit(token)

            full_response = self.llm.query(
                user_message=self.message,
                analysis_data=self.analysis_data,
                video_path=self.video_path,
                timeline_context=self.timeline_context,
                frame_base64=self.frame_base64,
                stream_callback=_stream_callback,
            )
            if self._cancel:
                self.finished.emit(full_response + "\n[stopped]")
            else:
                self.finished.emit(full_response)
        except _GenerationCancelled:
            self.finished.emit("[stopped by user]")
        except Exception as e:
            if self._cancel:
                self.finished.emit("[stopped by user]")
            else:
                self.error.emit(str(e))


class _GenerationCancelled(Exception):
    """Raised inside stream callback to abort generation."""
    pass

# ---------------------------------------------------------------------------
# Chat widget
# ---------------------------------------------------------------------------
class LLMChatWidget(QWidget):
    """
    Self-contained chat panel. Auto-loads latest cache from disk on startup.
    """

    llm_replied = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None, compact: bool = False,
                 cache_dir: str = "./cache"):
        super().__init__(parent)
        self._llm: Optional[LLMModule] = None
        self._analysis_data: Optional[dict] = None
        self._video_path: str = ""
        self._chat_history: list[dict] = []
        self._worker: Optional[_LLMWorker] = None
        self._compact = compact
        self._cache_dir = cache_dir
        self._timeline_bridge: Optional['TimelineBridge'] = None
        if HAS_TIMELINE_BRIDGE:
            self._timeline_bridge = TimelineBridge()

        self._build_ui()
        self._auto_load_latest_cache()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(4, 4, 4, 4)

        # --- Settings group ---
        settings_group = QGroupBox("LLM Settings")
        settings_layout = QVBoxLayout()

        # Row 1: backend + model
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Backend:"))
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("Ollama (local server)", "ollama")
        self.backend_combo.addItem("llama-cpp (GGUF file)", "llama-cpp")
        self.backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        row1.addWidget(self.backend_combo)

        row1.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setMinimumWidth(180)
        row1.addWidget(self.model_combo)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setFixedWidth(60)
        self.refresh_btn.clicked.connect(self._refresh_models)
        row1.addWidget(self.refresh_btn)
        settings_layout.addLayout(row1)

        # Row 2: GGUF path (hidden by default)
        self.gguf_row_widget = QWidget()
        gguf_inner = QHBoxLayout()
        gguf_inner.setContentsMargins(0, 0, 0, 0)
        gguf_inner.addWidget(QLabel("GGUF path:"))
        self.gguf_path_input = QLineEdit()
        self.gguf_path_input.setPlaceholderText("/path/to/model.gguf")
        gguf_inner.addWidget(self.gguf_path_input)
        self.gguf_browse_btn = QPushButton("Browse...")
        self.gguf_browse_btn.clicked.connect(self._browse_gguf)
        gguf_inner.addWidget(self.gguf_browse_btn)
        self.gguf_row_widget.setLayout(gguf_inner)
        self.gguf_row_widget.setVisible(False)
        settings_layout.addWidget(self.gguf_row_widget)

        # Row 3: connect + status
        row3 = QHBoxLayout()
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setStyleSheet(
            "QPushButton{background:#4CAF50;color:white;font-weight:bold;padding:6px 16px;}"
        )
        self.connect_btn.clicked.connect(self._connect_llm)
        row3.addWidget(self.connect_btn)

        self.status_label = QLabel("Not connected")
        self.status_label.setStyleSheet("color:#999;font-style:italic;")
        row3.addWidget(self.status_label)
        row3.addStretch()
        settings_layout.addLayout(row3)

        # Row 4: context indicator + Load Cache + Show Context
        row4 = QHBoxLayout()
        self.context_label = QLabel("No video context")
        self.context_label.setStyleSheet("color:#f44336;font-size:9pt;font-weight:bold;")
        row4.addWidget(self.context_label)
        row4.addStretch()

        self.load_cache_btn = QPushButton("Load Cache")
        self.load_cache_btn.setToolTip("Manually pick a .cache.json file")
        self.load_cache_btn.clicked.connect(self._load_cache_from_file)
        row4.addWidget(self.load_cache_btn)

        self.show_context_btn = QPushButton("Show Context")
        self.show_context_btn.setToolTip("See exactly what text the LLM receives")
        self.show_context_btn.clicked.connect(self._show_context_debug)
        row4.addWidget(self.show_context_btn)

        settings_layout.addLayout(row4)
        settings_group.setLayout(settings_layout)
        if self._compact:
            settings_group.setMaximumHeight(180)
        root.addWidget(settings_group)

        # --- Chat display ---
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet(
            "QTextEdit{background:#1e1e1e;color:#ddd;"
            "font-family:'Segoe UI','SF Pro',sans-serif;font-size:10pt;"
            "border:1px solid #333;border-radius:4px;padding:8px;}"
        )
        self.chat_display.setMinimumHeight(150 if self._compact else 250)
        root.addWidget(self.chat_display, stretch=1)

        # --- Input bar ---
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask about the video analysis...")
        self.input_field.setStyleSheet(
            "QLineEdit{padding:8px;font-size:10pt;border:1px solid #555;border-radius:4px;}"
        )
        self.input_field.returnPressed.connect(self._send_message)
        self.input_field.setEnabled(False)
        input_layout.addWidget(self.input_field, stretch=1)

        self.send_btn = QPushButton("Send")
        self.send_btn.setStyleSheet(
            "QPushButton{background:#2196F3;color:white;font-weight:bold;"
            "padding:8px 20px;border-radius:4px;}"
            "QPushButton:disabled{background:#555;}"
        )
        self.send_btn.clicked.connect(self._send_message)
        self.send_btn.setEnabled(False)
        input_layout.addWidget(self.send_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setStyleSheet(
            "QPushButton{background:#c62828;color:white;font-weight:bold;"
            "padding:8px 16px;border-radius:4px;}"
            "QPushButton:disabled{background:#555;}"
        )
        self.stop_btn.clicked.connect(self._stop_generation)
        self.stop_btn.setEnabled(False)
        input_layout.addWidget(self.stop_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self._clear_chat)
        input_layout.addWidget(self.clear_btn)

        root.addLayout(input_layout)
        self.setLayout(root)
        self._refresh_models()

    # --------------------------------------------------------- Public API

    def set_analysis_data(self, data: dict, video_path: str = ""):
        """Feed video analysis cache so the LLM has context."""
        self._analysis_data = data
        self._video_path = video_path
        self._update_context_label()

    def set_timeline_window(self, window):
        """Connect to a SignalTimelineWindow for timeline control.
        
        Call this from your timeline viewer's __init__:
            self.llm_chat = LLMChatWidget(parent=self, compact=True)
            self.llm_chat.set_timeline_window(self)
        """
        if self._timeline_bridge:
            self._timeline_bridge.set_timeline_window(window)
            self._update_context_label()
            self._append_system(
                "Timeline connected! You can now ask me to edit the timeline.\n"
                "Examples: 'add a clip at 0:10 to 0:15', 'remove clip 2', "
                "'show only person detections', 'play the clip at 0:30'"
            )

    def get_llm_module(self) -> Optional[LLMModule]:
        return self._llm

    def load_cache_for_video(self, video_path: str) -> bool:
        """Try to find and load a cache file for a specific video."""
        if not video_path or not os.path.exists(video_path):
            return False
        cache_dir = Path(self._cache_dir)
        if not cache_dir.exists():
            return False

        # Match by video filename substring in cache filenames
        video_stem = Path(video_path).stem.lower()
        all_caches = sorted(
            cache_dir.glob("*.cache.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for cf in all_caches:
            if video_stem in cf.stem.lower():
                return self._load_cache_file(str(cf), video_path)

        # Fallback: try via VideoAnalysisCache hash lookup
        try:
            from modules.video_cache import VideoAnalysisCache
            cache = VideoAnalysisCache(cache_dir=self._cache_dir)
            vhash = cache._get_video_hash(video_path)
            matching = sorted(
                cache_dir.glob(f"{vhash}*.cache.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if matching:
                return self._load_cache_file(str(matching[0]), video_path)
        except Exception:
            pass

        return False

    # ------------------------------------------------ Cache loading

    def _auto_load_latest_cache(self):
        """On startup, find and load the most recent cache file from disk."""
        cache_dir = Path(self._cache_dir)
        if not cache_dir.exists():
            self._append_system(
                f"No cache directory at '{self._cache_dir}'. "
                "Run pipeline first or use 'Load Cache'."
            )
            return

        all_caches = sorted(
            cache_dir.glob("*.cache.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not all_caches:
            self._append_system(
                "No cache files found. Run the pipeline on a video first."
            )
            return

        latest = all_caches[0]
        age_sec = _time.time() - latest.stat().st_mtime
        if age_sec < 60:
            age_str = f"{age_sec:.0f}s ago"
        elif age_sec < 3600:
            age_str = f"{age_sec/60:.0f}m ago"
        elif age_sec < 86400:
            age_str = f"{age_sec/3600:.1f}h ago"
        else:
            age_str = f"{age_sec/86400:.1f}d ago"

        self._append_system(f"Auto-loading latest cache: {latest.name} ({age_str})")
        self._load_cache_file(str(latest))

    def _load_cache_file(self, filepath: str, video_path: str = "") -> bool:
        """Load a .cache.json file and set it as analysis context."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                self._append_system(f"Invalid cache: expected dict, got {type(data).__name__}")
                return False

            if not video_path:
                video_path = os.path.basename(filepath)

            self.set_analysis_data(data, video_path)

            # Summary
            meta = data.get("video_metadata", {})
            dur = meta.get("duration", 0)
            n_obj = len(data.get("objects", []))
            n_act = len(data.get("actions", []))
            t_data = data.get("transcript", {})
            n_trans = len(t_data.get("segments", [])) if isinstance(t_data, dict) else 0
            n_scenes = len(data.get("scenes", []))
            n_motion = len(data.get("motion_events", []))
            n_peaks = len(data.get("motion_peaks", []))
            audio = data.get("audio", {})
            n_audio = len(audio.get("peaks", [])) if isinstance(audio, dict) else 0

            self._append_system(
                f"Cache loaded: {os.path.basename(filepath)}\n"
                f"  Duration: {int(dur)}s ({int(dur)//60}m{int(dur)%60:02d}s) | "
                f"Objects: {n_obj} | Actions: {n_act} | "
                f"Transcript: {n_trans} segs | Scenes: {n_scenes}\n"
                f"  Motion: {n_motion} events, {n_peaks} peaks | Audio peaks: {n_audio}"
            )
            return True

        except json.JSONDecodeError as e:
            self._append_system(f"Invalid JSON in {os.path.basename(filepath)}: {e}")
            return False
        except Exception as e:
            self._append_system(f"Failed to load cache: {e}")
            return False

    def _load_cache_from_file(self):
        """Manual cache file picker dialog."""
        start_dir = self._cache_dir if os.path.isdir(self._cache_dir) else "."
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Cache File", start_dir,
            "Cache Files (*.cache.json);;JSON Files (*.json);;All Files (*)"
        )
        if path:
            self._load_cache_file(path)

    def _show_context_debug(self):
        """Show exactly what context text would be sent to the LLM."""
        if not self._analysis_data:
            self._append_system(
                "NO ANALYSIS DATA LOADED!\n"
                "This is why the LLM halluccinates - it has nothing to work with.\n"
                "Use 'Load Cache' or run the pipeline first."
            )
            return

        context_text = VideoContextBuilder.build(self._analysis_data, self._video_path)
        ctx_chars = len(context_text)
        ctx_lines = context_text.count("\n") + 1

        dlg = QDialog(self)
        dlg.setWindowTitle("LLM Context Debug - What the LLM sees")
        dlg.setMinimumSize(700, 500)
        layout = QVBoxLayout()

        stats = QLabel(
            f"Context: {ctx_chars:,} chars | {ctx_lines} lines | "
            f"~{ctx_chars // 4:,} tokens (approx)\n"
            f"Video: {self._video_path}\n"
            f"Data keys: {', '.join(sorted(self._analysis_data.keys()))}"
        )
        stats.setStyleSheet("font-weight:bold;padding:4px;")
        layout.addWidget(stats)

        if ctx_chars > 8000:
            warn = QLabel(
                "WARNING: Context is large! Small models (3B) may ignore parts. "
                "Use 8B+ models for best results."
            )
            warn.setStyleSheet("color:#ff9800;font-weight:bold;padding:4px;")
            layout.addWidget(warn)

        text_view = QTextEdit()
        text_view.setReadOnly(True)
        text_view.setPlainText(context_text)
        text_view.setStyleSheet(
            "QTextEdit{font-family:'Consolas','Courier New',monospace;font-size:9pt;}"
        )
        layout.addWidget(text_view, stretch=1)

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(dlg.close)
        layout.addWidget(btn_box)

        dlg.setLayout(layout)
        dlg.exec()

    def _update_context_label(self):
        if not self._analysis_data:
            self.context_label.setText("No video context - LLM will hallucinate!")
            self.context_label.setStyleSheet("color:#f44336;font-size:9pt;font-weight:bold;")
            return

        meta = self._analysis_data.get("video_metadata", {})
        dur = meta.get("duration", 0)
        n_obj = len(self._analysis_data.get("objects", []))
        n_act = len(self._analysis_data.get("actions", []))
        t_data = self._analysis_data.get("transcript", {})
        n_trans = len(t_data.get("segments", [])) if isinstance(t_data, dict) else 0

        vname = os.path.basename(self._video_path) if self._video_path else "loaded"

        # Timeline status
        tl_status = ""
        if self._timeline_bridge and self._timeline_bridge.is_connected:
            tl_status = " | TL: connected"

        self.context_label.setText(
            f"Context: {vname} | {int(dur)}s | "
            f"{n_obj} obj | {n_act} act | {n_trans} transcript{tl_status}"
        )
        self.context_label.setStyleSheet("color:#4CAF50;font-size:9pt;font-weight:bold;")

    # ------------------------------------------------ Handlers

    def _on_backend_changed(self, _index):
        backend = self.backend_combo.currentData()
        is_gguf = backend == "llama-cpp"
        self.gguf_row_widget.setVisible(is_gguf)
        self.refresh_btn.setVisible(not is_gguf)
        if not is_gguf:
            self._refresh_models()

    def _refresh_models(self):
        self.model_combo.clear()
        backend = self.backend_combo.currentData()
        if backend == "ollama":
            models = get_ollama_models()
            if models:
                for m in models:
                    self.model_combo.addItem(m)
                self.status_label.setText(f"Found {len(models)} Ollama models")
                self.status_label.setStyleSheet("color:#4CAF50;font-style:italic;")
            else:
                for m in ["llama3.2", "llama3.1", "mistral", "phi3", "gemma2"]:
                    self.model_combo.addItem(m)
                self.status_label.setText("Ollama not running - showing defaults")
                self.status_label.setStyleSheet("color:#ff9800;font-style:italic;")
        else:
            self.model_combo.addItem("(select GGUF file below)")

    def _browse_gguf(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model", "", "GGUF Models (*.gguf);;All Files (*)"
        )
        if path:
            self.gguf_path_input.setText(path)

    def _connect_llm(self):
        backend = self.backend_combo.currentData()
        model = self.model_combo.currentText().strip()

        self.status_label.setText("Connecting...")
        self.status_label.setStyleSheet("color:#2196F3;font-style:italic;")
        self.connect_btn.setEnabled(False)
        QApplication.processEvents()

        try:
            if backend == "ollama":
                self._llm = LLMModule(backend="ollama", model=model, log_fn=self._log)
            elif backend == "llama-cpp":
                gguf_path = self.gguf_path_input.text().strip()
                if not gguf_path:
                    raise ValueError("Select a GGUF model file first")
                self._llm = LLMModule(backend="llama-cpp", model_path=gguf_path, log_fn=self._log)
            else:
                raise ValueError(f"Unknown backend: {backend}")

            self._llm.load()

            self.status_label.setText(f"Connected: {model}")
            self.status_label.setStyleSheet("color:#4CAF50;font-weight:bold;")
            self.connect_btn.setText("Reconnect")
            self.input_field.setEnabled(True)
            self.send_btn.setEnabled(True)

            if self._analysis_data:
                n_obj = len(self._analysis_data.get("objects", []))
                n_act = len(self._analysis_data.get("actions", []))
                self._append_system(
                    f"Connected to {model}. "
                    f"Context ready: {n_obj} object entries, {n_act} actions."
                )
            else:
                self._append_system(
                    f"Connected to {model}. "
                    f"WARNING: No video context! Use 'Load Cache' first."
                )

        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            self.status_label.setStyleSheet("color:#f44336;font-style:italic;")
            self.input_field.setEnabled(False)
            self.send_btn.setEnabled(False)
        finally:
            self.connect_btn.setEnabled(True)

    def _send_message(self):
        text = self.input_field.text().strip()
        if not text:
            return
        if not self._llm or not self._llm.is_loaded():
            self._append_system("Not connected. Click 'Connect' first.")
            return
        if self._worker and self._worker.isRunning():
            self._append_system("Still generating... please wait.")
            return

        if not self._analysis_data:
            self._append_system(
                "WARNING: No analysis data loaded! LLM will hallucinate.\n"
                "Use 'Load Cache' to load a cache file first."
            )

        self._append_user(text)
        self.input_field.clear()
        self._chat_history.append({"role": "user", "content": text})

        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.send_btn.setText("...")
        self.stop_btn.setEnabled(True)

        self._append_html(
            '<div style="color:#8BC34A;margin-top:8px;"><b>Assistant:</b></div>'
        )

        # Build timeline context if connected
        timeline_ctx = ""
        if self._timeline_bridge and self._timeline_bridge.is_connected:
            timeline_ctx = (
                self._timeline_bridge.get_timeline_state() + "\n" +
                self._timeline_bridge.get_available_commands_text()
            )

        # Capture current frame for vision models
        frame_b64 = None
        if self._timeline_bridge and self._timeline_bridge._window:
            window = self._timeline_bridge._window
            if hasattr(window, 'capture_current_frame_base64'):
                frame_b64 = window.capture_current_frame_base64()
                if frame_b64:
                    self._append_system(f"📷 Frame captured at {window.current_time:.1f}s ({len(frame_b64)//1024}KB)")
                else:
                    self._append_system("⚠️ Frame capture failed")
            else:
                self._append_system("⚠️ capture_current_frame_base64 method not found on timeline window")

        self._worker = _LLMWorker(
            llm=self._llm,
            message=text,
            analysis_data=self._analysis_data,
            video_path=self._video_path,
            timeline_context=timeline_ctx,
            frame_base64=frame_b64,
        )
        self._worker.token_received.connect(self._on_token)
        self._worker.finished.connect(self._on_response_done)
        self._worker.error.connect(self._on_response_error)
        self._worker.start()

    def _stop_generation(self):
        """Stop the current LLM generation."""
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self.stop_btn.setEnabled(False)
            self._append_system("⏹ Stopping generation...")

    @Slot(str)
    def _on_token(self, token: str):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(token)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    @Slot(str)
    def _on_response_done(self, full_text: str):
        self._chat_history.append({"role": "assistant", "content": full_text})

        # Parse and execute any timeline commands from the response
        if self._timeline_bridge and self._timeline_bridge.is_connected:
            try:
                from .llm_timeline_bridge import parse_commands
                commands = parse_commands(full_text)
                if commands:
                    cursor = self.chat_display.textCursor()
                    cursor.movePosition(QTextCursor.End)
                    cursor.insertText("\n")
                    self.chat_display.setTextCursor(cursor)

                    _, results = self._timeline_bridge.process_response(full_text)
                    for result in results:
                        self._append_html(
                            f'<div style="color:#FFD700;margin:2px 0;font-style:italic;">'
                            f'{result}</div>'
                        )
            except Exception as e:
                self._append_html(
                    f'<div style="color:#ff9800;">Command error: {e}</div>'
                )

        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText("\n\n")
        self.chat_display.setTextCursor(cursor)

        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Send")
        self.stop_btn.setEnabled(False)
        self.input_field.setFocus()

    @Slot(str)
    def _on_response_error(self, error_msg: str):
        self._append_html(
            f'<div style="color:#f44336;margin-left:12px;">Error: {error_msg}</div><br>'
        )
        self.input_field.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.send_btn.setText("Send")
        self.stop_btn.setEnabled(False)

    def _clear_chat(self):
        self.chat_display.clear()
        self._chat_history.clear()
        if self._llm and self._llm.is_loaded():
            self._append_system("Chat cleared. Ready for new questions.")

    # ------------------------------------------------ Display helpers

    def _append_html(self, html: str):
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertHtml(html)
        self.chat_display.setTextCursor(cursor)
        self.chat_display.ensureCursorVisible()

    def _append_user(self, text: str):
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        safe = safe.replace("\n", "<br>")
        self._append_html(
            f'<div style="color:#64B5F6;margin-top:8px;"><b>You:</b></div>'
            f'<div style="color:#ccc;margin-left:12px;">{safe}</div><br>'
        )

    def _append_system(self, text: str):
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        safe = safe.replace("\n", "<br>")
        self._append_html(
            f'<div style="color:#aaa;font-style:italic;margin:4px 0;">{safe}</div>'
        )

    def _log(self, msg: str):
        self.status_label.setText(msg)


# ---------------------------------------------------------------------------
# Standalone window for testing
# ---------------------------------------------------------------------------
class LLMChatWindow(QWidget):
    def __init__(self, cache_dir: str = "./cache"):
        super().__init__()
        self.setWindowTitle("VideoHighlighter - LLM Chat")
        self.setMinimumSize(600, 500)
        layout = QVBoxLayout()
        self.chat = LLMChatWidget(parent=self, cache_dir=cache_dir)
        layout.addWidget(self.chat)
        self.setLayout(layout)


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    win = LLMChatWindow()
    win.show()
    sys.exit(app.exec())