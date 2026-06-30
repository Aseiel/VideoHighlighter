"""
llm_chat_widget.py - Embeddable Qt chat panel for VideoHighlighter.

Key feature: AUTO-LOADS the most recent cache file from ./cache/ on startup,
so the LLM always has video context even after restarting the app.

Now with VideoSeekAnalyzer integration for visual search and seeking capabilities.

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
    QGroupBox, QFileDialog, QApplication, QCheckBox,
    QDialog, QDialogButtonBox, QDoubleSpinBox, QSpinBox,
)
from PySide6.QtCore import Qt, Signal, Slot, QThread, QSettings, QObject
from PySide6.QtGui import QTextCursor

from .llm_module import (
    LLMModule, VideoContextBuilder, get_available_backends, get_ollama_models,
    VideoSeekAnalyzer, CancellationToken, GenerationCancelled,
)
from .llm_reasoning import ReasoningLLMIntegration

# timeline bridge (only available when timeline viewer is present)
try:
    from .llm_timeline_bridge import TimelineBridge
    HAS_TIMELINE_BRIDGE = True
except ImportError:
    HAS_TIMELINE_BRIDGE = False


# ---------------------------------------------------------------------------
# Worker thread for LLM queries
# ---------------------------------------------------------------------------
class _LLMWorker(QObject):
    """Runs LLM.query() off the GUI thread with token-by-token streaming.

    Uses the QObject + moveToThread() pattern. CancellationToken supports
    mid-generation cancellation; blocking image-decode calls still have to
    finish, but the worker emits `finished` cleanly via the event loop and
    the `finished → deleteLater` chain handles teardown.
    """

    token_received = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, llm: LLMModule, message: str,
                 analysis_data: dict | None = None,
                 video_path: str = "",
                 timeline_context: str = "",
                 frame_base64: str | None = None,
                 free_chat_mode: bool = False):
        super().__init__()
        self.llm = llm
        self.message = message
        self.analysis_data = analysis_data
        self.video_path = video_path
        self.timeline_context = timeline_context
        self.frame_base64 = frame_base64
        self.free_chat_mode = free_chat_mode
        self._cancel_token = CancellationToken()

    def cancel(self):
        """Request cancellation — works even for blocking GGUF vision calls."""
        self._cancel_token.cancel()

    @Slot()
    def run(self):
        try:
            def _stream_callback(token: str):
                if self._cancel_token.is_cancelled:
                    raise GenerationCancelled()
                self.token_received.emit(token)

            full_response = self.llm.query(
                user_message=self.message,
                analysis_data=self.analysis_data,
                video_path=self.video_path,
                timeline_context=self.timeline_context,
                frame_base64=self.frame_base64,
                stream_callback=_stream_callback,
                cancellation_token=self._cancel_token,
                free_chat_mode=self.free_chat_mode,
            )
            if self._cancel_token.is_cancelled:
                self.finished.emit(full_response + "\n[stopped]")
            else:
                self.finished.emit(full_response)
        except GenerationCancelled:
            self.finished.emit("[stopped by user]")
        except Exception as e:
            if self._cancel_token.is_cancelled:
                self.finished.emit("[stopped by user]")
            else:
                self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Worker thread for visual search
# ---------------------------------------------------------------------------
class _VisualSearchWorker(QObject):
    """Runs VideoSeekAnalyzer visual search in background.

    Uses the QObject + moveToThread() pattern. CancellationToken interrupts
    per-frame LLM calls; the image decode itself is a ~20s blocking C call,
    so cancellation takes effect at the next frame boundary.
    """

    progress = Signal(int, int, float, str)  # current, total, timestamp, preview
    frame_analyzed = Signal(float, str, str, bool)  # timestamp, timestamp_str, response, contains_target
    found = Signal(float, str, str)  # timestamp, timestamp_str, analysis
    finished = Signal(list)  # all results
    error = Signal(str)

    def __init__(self, analyzer: VideoSeekAnalyzer, target: str,
                 interval: float = 1.0, max_seeks: int = 100,
                 stop_on_first_match: bool = True,
                 start_time: float = 0.0,
                 scene_threshold: float = 8.0,
                 mode: str = "llm", top_k: int = 30, clip_device: str = "GPU"):
        super().__init__()
        self.analyzer = analyzer
        self.target = target
        self.interval = interval
        self.max_seeks = max_seeks
        self.stop_on_first_match = stop_on_first_match
        self.start_time = start_time
        # Engine: "llm" (VLM brute force), "clip" (fast GPU ranker only), or
        # "clip_llm" (CLIP ranks all frames on the GPU, VLM confirms top_k).
        self.mode = mode
        self.top_k = top_k
        self.clip_device = clip_device
        self._clip = None  # lazily created ClipFramePrefilter
        # Content-aware sampling: skip a frame when its mean per-pixel change from
        # the last *analyzed* frame is below this (0-255 scale). 0 disables it.
        # Higher = skip more aggressively. This only avoids redundant ~3s vision
        # encodes on visually-static stretches; it never alters what the model is
        # asked, and the downscale used for the metric never reaches the model.
        self.scene_threshold = scene_threshold
        self._prev_small = None  # downscaled grayscale of the last analyzed frame
        self._cancel_token = CancellationToken()

    def cancel(self):
        self._cancel_token.cancel()

    @Slot()
    def run(self):
        """Dispatch to the selected engine. Each path emits finished/error."""
        try:
            if self.mode == "clip":
                self._run_clip()
            elif self.mode == "clip_llm":
                self._run_clip_llm()
            else:
                self._run_llm()
        except GenerationCancelled:
            self.finished.emit([])
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

    # ------------------------------------------------------------------ shared
    def _vlm_analyze(self, timestamp, frame, stage_totals, scene_diff=-1.0):
        """Two-phase VLM presence check on one decoded frame. Emits
        frame_analyzed and returns the result dict (or None if cancelled).
        Frames go to the model at full resolution — downscaling wrecks accuracy.
        """
        import time
        frame_t0 = time.perf_counter()
        try:
            h, w = frame.shape[:2]
            frame_dims = f"{w}x{h}"
        except Exception:
            frame_dims = "?"

        t0 = time.perf_counter()
        frame_b64 = self.analyzer.frame_to_base64(frame)
        t_encode = time.perf_counter() - t0
        stage_totals['encode'].append(t_encode)
        b64_kb = len(frame_b64) / 1024
        if self._cancel_token.is_cancelled:
            return None

        # one-word YES/NO presence check (cheap)
        t0 = time.perf_counter()
        response = self.analyzer.llm.query(
            user_message=f"Does this frame contain a {self.target}? Answer with only one word: YES or NO.",
            frame_base64=frame_b64,
            system_prompt=LLMModule.SYSTEM_PROMPT_VISUAL_SEARCH,
            temperature=0.0, max_tokens=5, cancellation_token=self._cancel_token,
        )
        t_llm = time.perf_counter() - t0
        stage_totals['llm'].append(t_llm)
        starts_with_yes = response.strip().lower().startswith("yes")

        # describe only on a positive hit
        analysis_text = response
        t_confirm = 0.0
        if starts_with_yes:
            t0 = time.perf_counter()
            analysis_text = self.analyzer.llm.query(
                user_message=f"This frame contains a {self.target}. Briefly describe what you see.",
                frame_base64=frame_b64,
                system_prompt=LLMModule.SYSTEM_PROMPT_VISUAL_SEARCH,
                temperature=0.0, max_tokens=150, cancellation_token=self._cancel_token,
            )
            t_confirm = time.perf_counter() - t0
            stage_totals['confirm'].append(t_confirm)

        result = {
            "timestamp": timestamp,
            "timestamp_str": f"{int(timestamp)//60}:{int(timestamp)%60:02d}",
            "analysis": analysis_text,
            "contains_target": starts_with_yes,
        }
        self.frame_analyzed.emit(timestamp, result["timestamp_str"],
                                 analysis_text, starts_with_yes)
        t_total = time.perf_counter() - frame_t0
        print(f"[t={timestamp:6.1f}s] {frame_dims} {b64_kb:5.0f}KB Δ={scene_diff:5.1f} | "
              f"encode={t_encode*1000:5.0f}ms  llm={t_llm*1000:6.0f}ms  "
              f"confirm={t_confirm*1000:6.0f}ms  TOTAL={t_total*1000:6.0f}ms "
              f"({'YES' if starts_with_yes else 'no '})")
        return result

    def _print_summary(self, stage_totals, n_skipped):
        if not stage_totals.get('llm'):
            return
        print("\n" + "=" * 70)
        print("PROFILE SUMMARY")
        print("=" * 70)
        print(f"{'stage':<10} {'count':>5} {'mean':>8} {'median':>8} {'min':>8} {'max':>8} {'total':>8}")
        for stage in ('encode', 'llm', 'confirm'):
            vals = stage_totals.get(stage) or []
            if not vals:
                continue
            vs = sorted(vals)
            print(f"{stage:<10} {len(vals):>5} {sum(vals)/len(vals)*1000:>7.0f}ms "
                  f"{vs[len(vs)//2]*1000:>7.0f}ms {vs[0]*1000:>7.0f}ms {vs[-1]*1000:>7.0f}ms "
                  f"{sum(vals):>7.1f}s")
        print(f"\nFrames analyzed: {len(stage_totals['llm'])}   "
              f"skipped (scene-stable): {n_skipped}")
        print("=" * 70 + "\n")

    # ----------------------------------------------------------------- engines
    def _run_llm(self):
        """LLM-only: walk every interval frame and ask the VLM (brute force)."""
        import time
        import numpy as np
        from collections import defaultdict

        stage_totals = defaultdict(list)
        n_skipped = 0
        results = []
        num = min(int(self.analyzer.duration / self.interval) + 1, self.max_seeks)
        timestamps = [self.start_time + i * self.interval for i in range(num)]

        for i, timestamp in enumerate(timestamps):
            if self._cancel_token.is_cancelled or timestamp > self.analyzer.duration + 0.1:
                break
            self.progress.emit(i + 1, len(timestamps), timestamp, f"Analyzing {timestamp:.1f}s")
            self.analyzer.current_time = timestamp
            frame = self.analyzer.seek_to_time(timestamp)
            if frame is None:
                print(f"[t={timestamp:6.1f}s] seek failed")
                continue

            # content-aware skip (cheap downscaled diff; never sent to the model)
            scene_diff = -1.0
            if self.scene_threshold > 0:
                small = frame[::8, ::8]
                small = small.mean(axis=2).astype('float32') if small.ndim == 3 else small.astype('float32')
                if self._prev_small is not None and small.shape == self._prev_small.shape:
                    scene_diff = float(np.abs(small - self._prev_small).mean())
                    if scene_diff < self.scene_threshold:
                        n_skipped += 1
                        print(f"[t={timestamp:6.1f}s] skipped (scene Δ={scene_diff:4.1f} < {self.scene_threshold})")
                        continue
                self._prev_small = small

            if self._cancel_token.is_cancelled:
                break
            try:
                result = self._vlm_analyze(timestamp, frame, stage_totals, scene_diff)
            except GenerationCancelled:
                break
            except Exception as e:
                print(f"Error at {timestamp:.1f}s: {e}")
                continue
            if result is None:
                break
            results.append(result)
            if result["contains_target"]:
                self.found.emit(timestamp, result["timestamp_str"], result["analysis"])
                if self.stop_on_first_match:
                    break

        self._print_summary(stage_totals, n_skipped)
        self.finished.emit(results)

    def _clip_scan(self):
        """Score every interval frame with CLIP on the GPU. Returns a list of
        (timestamp, score) sorted by score desc, or None if CLIP is unavailable."""
        import time
        try:
            from .clip_prefilter import ClipFramePrefilter
        except Exception as e:
            self.error.emit(f"CLIP prefilter import failed: {e}")
            return None
        if not ClipFramePrefilter.available():
            self.error.emit('CLIP deps missing — pip install "optimum[openvino]" pillow')
            return None

        if self._clip is None:
            self.progress.emit(0, 1, self.start_time,
                               "Loading CLIP on GPU (first run downloads the model)...")
            self._clip = ClipFramePrefilter(device=self.clip_device)
            self._clip.load()
        self._clip.set_query(self.target)

        num = min(int(self.analyzer.duration / self.interval) + 1, self.max_seeks)
        timestamps = [self.start_time + i * self.interval for i in range(num)
                      if self.start_time + i * self.interval <= self.analyzer.duration + 0.1]

        scored, buf_f, buf_t = [], [], []
        BATCH = 16
        t0 = time.perf_counter()
        for i, ts in enumerate(timestamps):
            if self._cancel_token.is_cancelled:
                break
            self.analyzer.current_time = ts
            frame = self.analyzer.seek_to_time(ts)
            if frame is None:
                continue
            buf_f.append(frame)
            buf_t.append(ts)
            if len(buf_f) >= BATCH:
                for t, s in zip(buf_t, self._clip.score_frames_bgr(buf_f)):
                    scored.append((t, s))
                buf_f, buf_t = [], []
            if i % 10 == 0:
                self.progress.emit(i + 1, len(timestamps), ts, f"CLIP scan {ts:.0f}s")
        if buf_f and not self._cancel_token.is_cancelled:
            for t, s in zip(buf_t, self._clip.score_frames_bgr(buf_f)):
                scored.append((t, s))

        elapsed = time.perf_counter() - t0
        print(f"\nCLIP scan: {len(scored)} frames in {elapsed:.1f}s "
              f"({elapsed/max(1,len(scored))*1000:.1f} ms/frame) on {self._clip.device}")
        scored.sort(key=lambda x: -x[1])
        return scored

    def _run_clip(self):
        """CLIP-only: return the top_k frames ranked by similarity (no VLM)."""
        scored = self._clip_scan()
        if scored is None:
            return
        top = scored[:self.top_k]
        print(f"CLIP: top {len(top)} of {len(scored)} frames "
              f"(scores {top[0][1]:.2f}..{top[-1][1]:.2f})" if top else "CLIP: no frames")
        results = []
        for ts, score in sorted(top, key=lambda x: x[0]):  # chronological for playback
            tstr = f"{int(ts)//60}:{int(ts)%60:02d}"
            analysis = f"CLIP match (similarity {score:.2f})"
            results.append({"timestamp": ts, "timestamp_str": tstr,
                            "analysis": analysis, "contains_target": True,
                            "clip_score": score})
            self.frame_analyzed.emit(ts, tstr, analysis, True)
            self.found.emit(ts, tstr, analysis)
        self.finished.emit(results)

    def _run_clip_llm(self):
        """CLIP ranks all frames on the GPU; the VLM confirms only the top_k."""
        from collections import defaultdict
        scored = self._clip_scan()
        if scored is None:
            return
        candidates = scored[:self.top_k]
        print(f"CLIP+LLM: confirming top {len(candidates)} candidates with the VLM")

        stage_totals = defaultdict(list)
        results = []
        for rank, (ts, score) in enumerate(candidates):  # strongest first
            if self._cancel_token.is_cancelled:
                break
            self.progress.emit(rank + 1, len(candidates), ts,
                               f"VLM confirm {ts:.0f}s (CLIP {score:.2f})")
            self.analyzer.current_time = ts
            frame = self.analyzer.seek_to_time(ts)
            if frame is None:
                continue
            try:
                result = self._vlm_analyze(ts, frame, stage_totals)
            except GenerationCancelled:
                break
            except Exception as e:
                print(f"Error at {ts:.1f}s: {e}")
                continue
            if result is None:
                break
            result["clip_score"] = score
            results.append(result)
            if result["contains_target"]:
                self.found.emit(ts, result["timestamp_str"], result["analysis"])
                if self.stop_on_first_match:
                    break

        self._print_summary(stage_totals, 0)
        self.finished.emit(results)

# Keywords that indicate the user wants visual/frame analysis
_VISION_KEYWORDS = (
    "see", "look", "show me", "what is this", "describe frame",
    "describe image", "what's in the frame", "what do you see",
    "visual", "screenshot", "current frame", "this frame",
    "what's happening here", "what is happening here",
)

# ---------------------------------------------------------------------------
# Time spinbox — displays seconds as mm:ss, accepts both formats as input
# ---------------------------------------------------------------------------
class TimeSpinBox(QDoubleSpinBox):
    """
    QDoubleSpinBox that stores a float seconds value but displays it as mm:ss.
    Accepts input in either form:  '90'  or  '1:30'  or  '1:30.5'
    """

    def textFromValue(self, value: float) -> str:
        total = max(0.0, value)
        minutes = int(total // 60)
        secs = total - minutes * 60
        if abs(secs - round(secs)) < 0.01:
            return f"{minutes}:{int(round(secs)):02d}"
        return f"{minutes}:{secs:05.2f}"  # e.g. 1:05.50

    def valueFromText(self, text: str) -> float:
        text = text.strip()
        if not text:
            return 0.0
        if ':' in text:
            parts = text.split(':')
            try:
                minutes = int(parts[0]) if parts[0] else 0
                seconds = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
                return minutes * 60.0 + seconds
            except ValueError:
                return self.value()
        try:
            return float(text)
        except ValueError:
            return self.value()

    def validate(self, text: str, pos: int):
        from PySide6.QtGui import QValidator
        import re
        stripped = text.strip()
        if not stripped:
            return (QValidator.Intermediate, text, pos)
        if ':' in stripped:
            if re.fullmatch(r'\d{0,3}:\d{0,2}(\.\d*)?', stripped):
                return (QValidator.Acceptable, text, pos)
            if re.fullmatch(r'\d{0,3}:', stripped):
                return (QValidator.Intermediate, text, pos)
            return (QValidator.Invalid, text, pos)
        return super().validate(text, pos)

class LLMChatWidget(QWidget):
    """
    Self-contained chat panel. Auto-loads latest cache from disk on startup.
    Now with VideoSeekAnalyzer integration for visual search and seeking.
    """
    MAX_RECENT_GGUF = 5
    SETTINGS_KEY = "VideoHighlighter/LLMChat"

    llm_replied = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None, compact: bool = False,
                 cache_dir: str = "./cache", video_path: str = ""):
        super().__init__(parent)
        self._llm: Optional[LLMModule] = None
        self._analyzer: Optional[VideoSeekAnalyzer] = None
        self._analysis_data: Optional[dict] = None
        self._video_path: str = video_path
        self._chat_history: list[dict] = []
        self._worker: Optional[_LLMWorker] = None
        self._worker_thread: Optional[QThread] = None
        self._search_worker: Optional[_VisualSearchWorker] = None
        self._search_worker_thread: Optional[QThread] = None
        self._compact = compact
        self._cache_dir = cache_dir
        self._timeline_bridge: Optional['TimelineBridge'] = None
        self._preview_window = None
        self.reasoning_engine = None
        self.reasoning_enabled = True
        if HAS_TIMELINE_BRIDGE:
            self._timeline_bridge = TimelineBridge()

        self._build_ui()
        self._auto_load_latest_cache()
        
        # Initialize analyzer if video path provided
        if self._video_path and os.path.exists(self._video_path):
            self._init_analyzer()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        root = QVBoxLayout()
        root.setContentsMargins(4, 4, 4, 4)

        # --- Settings group ---
        settings_group = QGroupBox("LLM Settings")
        settings_layout = QVBoxLayout()
        settings_layout.setSpacing(4)
        settings_layout.setContentsMargins(6, 6, 6, 6)

        # Row 1: backend + model + connect + status (one row to save height)
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

        # Connect + status share this row (was a separate row) to save vertical space
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setStyleSheet(
            "QPushButton{background:#4CAF50;color:white;font-weight:bold;padding:6px 16px;}"
        )
        self.connect_btn.clicked.connect(self._connect_llm)
        row1.addWidget(self.connect_btn)

        self.status_label = QLabel("Not connected")
        self.status_label.setStyleSheet("color:#999;font-style:italic;")
        row1.addWidget(self.status_label)
        row1.addStretch()
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

        # Row 3: mmproj path (for vision models, hidden by default)
        self.mmproj_row_widget = QWidget()
        mmproj_inner = QHBoxLayout()
        mmproj_inner.setContentsMargins(0, 0, 0, 0)
        mmproj_inner.addWidget(QLabel("mmproj path:"))
        self.mmproj_path_input = QLineEdit()
        self.mmproj_path_input.setPlaceholderText("/path/to/mmproj-model.gguf (optional, for vision)")
        mmproj_inner.addWidget(self.mmproj_path_input)
        self.mmproj_browse_btn = QPushButton("Browse...")
        self.mmproj_browse_btn.clicked.connect(self._browse_mmproj)
        mmproj_inner.addWidget(self.mmproj_browse_btn)
        self.mmproj_row_widget.setLayout(mmproj_inner)
        self.mmproj_row_widget.setVisible(False)
        settings_layout.addWidget(self.mmproj_row_widget)
        # Restore last-used GGUF paths
        settings = QSettings(self.SETTINGS_KEY, "LLMChat")
        self.gguf_path_input.setText(settings.value("last_gguf_path", ""))
        self.mmproj_path_input.setText(settings.value("last_mmproj_path", ""))


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
        
        # Row 5: Visual search controls
        search_group = QGroupBox("Visual Search")
        search_layout = QHBoxLayout()
        
        search_layout.addWidget(QLabel("Search for:"))
        self.search_target = QLineEdit()
        self.search_target.setPlaceholderText("explosion, person, car, etc.")
        search_layout.addWidget(self.search_target)
        
        search_layout.addWidget(QLabel("From:"))
        self.search_start_time = TimeSpinBox()
        self.search_start_time.setRange(0, 99999)
        self.search_start_time.setValue(0)
        self.search_start_time.setSingleStep(10)
        self.search_start_time.setDecimals(2)
        self.search_start_time.setToolTip(
            "Start searching from this timestamp. Type seconds (90) or mm:ss (1:30)."
        )
        search_layout.addWidget(self.search_start_time)

        search_layout.addWidget(QLabel("Interval:"))
        self.search_interval = TimeSpinBox()
        self.search_interval.setRange(0.5, 600.0)  # up to 10 minutes per step
        self.search_interval.setValue(1.0)
        self.search_interval.setSingleStep(1.0)
        self.search_interval.setDecimals(2)
        self.search_interval.setToolTip(
            "Time between sampled frames. Type seconds (60) or mm:ss (1:00)."
        )
        search_layout.addWidget(self.search_interval)
        
        # Engine selector: CLIP (fast GPU ranker) / LLM (VLM) / CLIP+LLM (funnel)
        search_layout.addWidget(QLabel("Engine:"))
        self.search_engine_combo = QComboBox()
        self.search_engine_combo.addItem("CLIP + LLM", "clip_llm")
        self.search_engine_combo.addItem("CLIP only", "clip")
        self.search_engine_combo.addItem("LLM only", "llm")
        self.search_engine_combo.setToolTip(
            "CLIP only  : fast GPU similarity ranking; best for broad concepts.\n"
            "LLM only   : the vision model checks every frame (slow, reasons, uncensored).\n"
            "CLIP + LLM : CLIP ranks all frames on the GPU, the LLM confirms the top-K."
        )
        self.search_engine_combo.currentIndexChanged.connect(self._on_search_engine_changed)
        search_layout.addWidget(self.search_engine_combo)

        self.search_topk_label = QLabel("Top-K:")
        search_layout.addWidget(self.search_topk_label)
        self.search_topk = QSpinBox()
        self.search_topk.setRange(1, 1000)
        self.search_topk.setValue(30)
        self.search_topk.setToolTip(
            "How many top CLIP candidates to keep (CLIP only) or send to the LLM "
            "to confirm (CLIP + LLM). Higher = better recall, slower."
        )
        search_layout.addWidget(self.search_topk)
        self.search_topk.valueChanged.connect(self._save_search_prefs)

        # Restore last-used engine + Top-K (first run defaults to CLIP only —
        # frictionless, needs no connected model).
        _ss = QSettings(self.SETTINGS_KEY, "LLMChat")
        _eidx = self.search_engine_combo.findData(_ss.value("search_engine", "clip"))
        if _eidx >= 0:
            self.search_engine_combo.setCurrentIndex(_eidx)
        try:
            self.search_topk.setValue(int(_ss.value("search_topk", 30)))
        except (TypeError, ValueError):
            pass
        self._on_search_engine_changed()  # sync Top-K visibility to restored engine

        # Add "stop on first match" checkbox
        self.stop_on_match_cb = QCheckBox("Stop on find")
        self.stop_on_match_cb.setChecked(True)
        self.stop_on_match_cb.setToolTip(
            "When checked, search stops and seeks to the first match.\n"
            "When unchecked, search scans entire video and reports all matches."
        )
        search_layout.addWidget(self.stop_on_match_cb)

        self.search_btn = QPushButton("🔍 Search")
        self.search_btn.setStyleSheet(
            "QPushButton{background:#FF9800;color:white;font-weight:bold;padding:6px 12px;border-radius:4px;}"
        )
        self.search_btn.clicked.connect(self._start_visual_search)
        search_layout.addWidget(self.search_btn)
        
        self.stop_search_btn = QPushButton("⏹ Stop")
        self.stop_search_btn.setStyleSheet(
            "QPushButton{background:#c62828;color:white;font-weight:bold;padding:6px 12px;border-radius:4px;}"
        )
        self.stop_search_btn.clicked.connect(self._stop_visual_search)
        self.stop_search_btn.setEnabled(False)
        search_layout.addWidget(self.stop_search_btn)
        
        search_group.setLayout(search_layout)
        settings_layout.addWidget(search_group)
        
        # Search progress
        self.search_progress = QLabel("")
        self.search_progress.setStyleSheet("color:#2196F3;font-style:italic;font-size:9pt;")
        settings_layout.addWidget(self.search_progress)
        
        # ===== Reasoning controls (plain row — no group box, to save height) =====
        reasoning_layout = QHBoxLayout()
        self.reasoning_checkbox = QCheckBox("Enable reasoning")
        self.reasoning_checkbox.setChecked(True)
        self.reasoning_checkbox.setToolTip(
            "When enabled, the LLM will infer relationships between detected objects and actions\n"
            "Examples: 'Person is punching person', 'Person drinking from cup', 'Multiple people talking'"
        )
        reasoning_layout.addWidget(self.reasoning_checkbox)

        self.reasoning_stats_btn = QPushButton("📊 Stats")
        self.reasoning_stats_btn.setFixedWidth(60)
        self.reasoning_stats_btn.setToolTip("Show reasoning statistics")
        self.reasoning_stats_btn.clicked.connect(self._show_reasoning_stats)
        reasoning_layout.addWidget(self.reasoning_stats_btn)

        self.reasoning_save_btn = QPushButton("💾 Save")
        self.reasoning_save_btn.setFixedWidth(60)
        self.reasoning_save_btn.setToolTip("Save inferred facts to cache")
        self.reasoning_save_btn.clicked.connect(self._save_reasoning_facts)
        reasoning_layout.addWidget(self.reasoning_save_btn)

        reasoning_layout.addStretch()
        settings_layout.addLayout(reasoning_layout)

        settings_group.setLayout(settings_layout)
        if self._compact:
            settings_group.setMaximumHeight(450)
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
        self.input_field.setPlaceholderText("Ask about the video analysis or search visually...")
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

        self.free_chat_chk = QCheckBox("💬 Free chat")
        self.free_chat_chk.setToolTip(
            "When checked, the LLM answers freely without being restricted to video data"
        )
        input_layout.addWidget(self.free_chat_chk)

        root.addLayout(input_layout)
        self.setLayout(root)
        self._refresh_models()

    # --------------------------------------------------------- Public API

    def set_analysis_data(self, data: dict, video_path: str = ""):
        """Feed video analysis cache so the LLM has context."""
        self._analysis_data = data
        self._video_path = video_path
        self._update_context_label()
        
        # Initialize analyzer if we have video path
        if video_path and os.path.exists(video_path):
            self._init_analyzer()
        
        # Initialize reasoning engine if enabled
        if hasattr(self, 'reasoning_checkbox') and self.reasoning_checkbox.isChecked():
            try:
                from .llm_reasoning import ReasoningLLMIntegration
                llm = self._llm if (self._llm and self._llm.is_loaded()) else None
                self.reasoning_engine = ReasoningLLMIntegration(
                    llm, data, video_path
                )
                stats = self.reasoning_engine.reasoning_engine.get_action_statistics()
                self._append_system(
                    f"🧠 Reasoning engine initialized with {stats['total_actions']} actions analyzed"
                )
                if not llm:
                    self._append_system(
                        "ℹ️ Stats available now. Connect a model for reasoning questions."
                    )
            except Exception as e:
                self._append_system(f"⚠️ Could not initialize reasoning: {e}")
                self.reasoning_engine = None

    def set_preview_window(self, preview):
        """Connect to a VideoPreviewWindow for seek sync."""
        self._preview_window = preview

    def set_timeline_window(self, window):
        """Connect to a SignalTimelineWindow for timeline control."""
        if self._timeline_bridge:
            self._timeline_bridge.set_timeline_window(window)
            self._timeline_bridge.set_scan_callback(self._trigger_visual_scan)
            self._update_context_label()
            self._append_system(
                "Timeline connected! You can now ask me to edit the timeline.\n"
                "Examples: 'add a clip at 0:10 to 0:15', 'remove clip 2', "
                "'show only person detections', 'play the clip at 0:30'"
            )

        # Always grab video path from timeline window
        if hasattr(window, 'video_path') and window.video_path:
            self._video_path = window.video_path
            self._update_context_label()

        # Sync search start time with timeline clicks
        if hasattr(window, 'signal_scene'):
            window.signal_scene.time_clicked.connect(self._update_search_start_time)

    def _update_search_start_time(self, time):
        """Update visual search start time when user clicks timeline."""
        if hasattr(self, 'search_start_time'):
            self.search_start_time.setValue(time)

    def get_llm_module(self) -> Optional[LLMModule]:
        return self._llm

    def get_analyzer(self) -> Optional[VideoSeekAnalyzer]:
        return self._analyzer

    def load_cache_for_video(self, video_path: str) -> bool:
        """Try to find and load a cache file for a specific video."""
        if not video_path or not os.path.exists(video_path):
            return False
        cache_dir = Path(self._cache_dir)
        if not cache_dir.exists():
            return False

        video_stem = Path(video_path).stem.lower()
        all_caches = sorted(
            cache_dir.glob("*.cache.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for cf in all_caches:
            if video_stem in cf.stem.lower():
                success = self._load_cache_file(str(cf), video_path)
                if success:
                    self._init_analyzer(video_path)
                return success

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
                success = self._load_cache_file(str(matching[0]), video_path)
                if success:
                    self._init_analyzer(video_path)
                return success
        except Exception:
            pass

        return False

    def set_video_path(self, video_path: str):
        """Set the video path and initialize analyzer."""
        self._video_path = video_path
        if video_path and os.path.exists(video_path):
            self._init_analyzer(video_path)

    # ------------------------------------------------ Analyzer initialization

    def _init_analyzer(self, video_path: str = None, require_llm: bool = True):
        """Initialize VideoSeekAnalyzer for visual operations.

        require_llm=False allows CLIP-only search, which never touches the VLM.
        """
        path = video_path or self._video_path
        if not path or not os.path.exists(path):
            return False

        if require_llm and (not self._llm or not self._llm.is_loaded()):
            self._append_system("⚠️ LLM not connected. This engine requires a connected vision model.")
            return False
        
        try:
            if self._analyzer:
                self._analyzer.close()
            
            self._analyzer = VideoSeekAnalyzer(path, self._llm, verbose=False)
            self._append_system(f"✅ Video analyzer ready: {os.path.basename(path)}")
            return True
        except Exception as e:
            self._append_system(f"❌ Failed to initialize analyzer: {e}")
            return False

    # ------------------------------------------------ Visual search

    def _on_search_engine_changed(self):
        """Top-K only applies to the CLIP-based engines; hide it for LLM-only."""
        engine = self.search_engine_combo.currentData()
        is_clip = engine in ("clip", "clip_llm")
        self.search_topk_label.setVisible(is_clip)
        self.search_topk.setVisible(is_clip)
        self._save_search_prefs()

    def _save_search_prefs(self):
        """Persist the engine + Top-K so they're remembered next session."""
        try:
            s = QSettings(self.SETTINGS_KEY, "LLMChat")
            s.setValue("search_engine", self.search_engine_combo.currentData())
            s.setValue("search_topk", self.search_topk.value())
        except Exception:
            pass

    def _start_visual_search(self):
        """Start visual search for target in video."""
        target = self.search_target.text().strip()
        if not target:
            self._append_system("❌ Please enter something to search for")
            return

        engine = self.search_engine_combo.currentData()  # "clip_llm" | "clip" | "llm"
        top_k = self.search_topk.value()

        if not self._analyzer:
            if not self._init_analyzer(require_llm=(engine != "clip")):
                self._append_system("❌ Cannot start search: No video loaded or analyzer not ready")
                return

        # Engines that use the VLM need a live model. The analyzer may have been
        # created earlier by a CLIP-only run (with no LLM), so sync the current one.
        if engine != "clip":
            if not self._llm or not self._llm.is_loaded():
                self._append_system("⚠️ This engine needs a connected vision model. Connect one, or use CLIP only.")
                return
            self._analyzer.llm = self._llm
        
        if self._search_worker_thread and self._search_worker_thread.isRunning():
            # Cancel the previous worker. The QObject+moveToThread pattern
            # handles cleanup via the finished→deleteLater chain — old worker
            # finishes its current frame, emits finished, thread quits, both
            # objects schedule themselves for deletion through the event loop.
            # Qt's parent-child ownership keeps the old thread alive until then,
            # even though we're about to reassign self._search_worker_thread.
            if self._search_worker:
                self._search_worker.cancel()
                try:
                    self._search_worker.disconnect()
                except (RuntimeError, TypeError):
                    pass
            self._append_system(
                "⏹ Cancelling previous search (will finish current frame in background)..."
            )

        interval = self.search_interval.value()
        start_from = self.search_start_time.value()
        remaining = max(0, self._analyzer.duration - start_from)
        max_seeks = int(remaining / interval) + 1
        stop_on_match = self.stop_on_match_cb.isChecked()
        
        mode_str = "stop on first match" if stop_on_match else "scan entire video"
        start_str = f"{int(start_from)//60}:{int(start_from)%60:02d}"

        # Generate a unique scan ID and clear any previous findings for this query
        # so the timeline shows fresh results.
        self._current_scan_id = f"scan_{int(_time.time())}"
        if (self._timeline_bridge
                and self._timeline_bridge._window
                and hasattr(self._timeline_bridge._window, 'signal_scene')):
            scene = self._timeline_bridge._window.signal_scene
            if hasattr(scene, 'clear_visual_findings'):
                scene.clear_visual_findings(query=target)

        engine_label = {"clip_llm": "CLIP+LLM", "clip": "CLIP", "llm": "LLM"}.get(engine, engine)
        extra = f", top-{top_k}" if engine in ("clip", "clip_llm") else ""
        self._append_system(
            f"🔍 [{engine_label}{extra}] searching for '{target}' from {start_str} "
            f"every {interval}s ({mode_str})..."
        )
        self.search_progress.setText(f"Searching for '{target}' [{engine_label}]...")
        self.search_btn.setEnabled(False)
        self.stop_search_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)  # Also enable main Stop button
        
        self._search_worker_thread = QThread(self)  # parent=self for Qt ownership
        self._search_worker = _VisualSearchWorker(
            analyzer=self._analyzer,
            target=target,
            interval=interval,
            max_seeks=max_seeks,
            stop_on_first_match=stop_on_match,
            start_time=start_from,
            mode=engine,
            top_k=top_k,
        )
        self._search_worker.moveToThread(self._search_worker_thread)

        # Lifecycle
        self._search_worker_thread.started.connect(self._search_worker.run)
        self._search_worker.finished.connect(self._search_worker_thread.quit)
        self._search_worker.finished.connect(self._search_worker.deleteLater)
        self._search_worker.error.connect(self._search_worker_thread.quit)
        self._search_worker_thread.finished.connect(self._search_worker_thread.deleteLater)

        # Our handlers
        self._search_worker.progress.connect(self._on_search_progress)
        self._search_worker.frame_analyzed.connect(self._on_frame_analyzed)
        self._search_worker.found.connect(self._on_search_found)
        self._search_worker.finished.connect(self._on_search_finished)
        self._search_worker.error.connect(self._on_search_error)

        self._search_worker_thread.start()

    def _stop_visual_search(self):
        """Stop ongoing visual search."""
        if self._search_worker_thread and self._search_worker_thread.isRunning():
            self._search_worker.cancel()
            self._append_system(
                "⏹ Stop requested — will stop after current frame finishes.\n"
                "   (GGUF image decoding cannot be interrupted mid-frame)"
            )
            self.search_progress.setText("Stopping after current frame...")
            self.search_btn.setEnabled(True)
            self.stop_search_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)

    def _trigger_visual_scan(self, target: str, interval: float):
        """Called by TimelineBridge when LLM generates [CMD:visual_scan]."""
        self.search_target.setText(target)
        self.search_interval.setValue(interval)
        self._start_visual_search()

    @Slot(int, int, float, str)
    def _on_search_progress(self, current: int, total: int, timestamp: float, preview: str):
        """Update search progress."""
        percent = (current / total) * 100
        self.search_progress.setText(
            f"Searching: {current}/{total} ({percent:.1f}%) - {timestamp:.1f}s"
        )

    @Slot(float, str, str, bool)
    def _on_frame_analyzed(self, timestamp: float, timestamp_str: str, response: str, contains_target: bool):
        """Show each frame's YES/NO result in chat and update preview for ALL frames."""
        icon = "✅" if contains_target else "❌"
        short = response[:120] + "..." if len(response) > 120 else response
        self._append_system(f"  {icon} [{timestamp_str}] {short}")

        # ── Live push YES hits onto the signal timeline ──
        if (contains_target
                and self._timeline_bridge
                and self._timeline_bridge._window
                and hasattr(self._timeline_bridge._window, 'add_visual_findings')):
            window = self._timeline_bridge._window

            # Resolve model name for metadata
            model_name = ''
            if self._llm:
                model_name = (getattr(self._llm, 'model', None)
                            or getattr(self._llm, 'model_path', None) or '')
                if model_name and ('/' in model_name or '\\' in model_name):
                    model_name = os.path.basename(model_name)

            finding = {
                'timestamp':  timestamp,
                'query':      self.search_target.text().strip(),
                'confidence': 1.0,  # YES/NO classification — see notes
                'model':      model_name,
                'scan_id':    getattr(self, '_current_scan_id', ''),
                'analysis':   (response or '')[:200],
            }
            try:
                # save=False — defer disk write until scan finishes (see D3)
                window.add_visual_findings([finding], save=False)
            except Exception as e:
                print(f"⚠️ Failed to push visual finding to timeline: {e}")
        
        # Update preview window directly (QMediaPlayer has its own decoder,
        # so this is safe). Do NOT call _seek_to_timestamp() here — that
        # would re-seek the analyzer's cv2.VideoCapture from the GUI thread
        # while the worker thread is still using it, causing the FFmpeg
        # assertion: "!dst->progress failed at libavcodec/utils.c:954"
        if hasattr(self, '_preview_window') and self._preview_window:
            self._preview_window.seek_to_time(timestamp)
            self._preview_window.force_frame_update()
            
            if hasattr(self._preview_window, 'show_frame_analysis_status'):
                self._preview_window.show_frame_analysis_status(timestamp, contains_target)
            
            QApplication.processEvents()
        
        # Update timeline playhead (no decoder involved, just UI)
        if self._timeline_bridge and self._timeline_bridge.is_connected:
            self._timeline_bridge._cmd_seek({'time': str(timestamp)})
        
        self.search_progress.setText(f"Analyzing frame at {timestamp_str}...")

    @Slot(float, str, str)
    def _on_search_found(self, timestamp: float, timestamp_str: str, analysis: str):
        """Handle found target — auto-seek preview and timeline to the found timestamp."""
        self._append_system(
            f"🎯 FOUND at {timestamp_str}!\n"
            f"   {analysis[:150]}..."
        )
        
        # Only update preview + timeline, NOT the analyzer (worker may still be running)
        if hasattr(self, '_preview_window') and self._preview_window:
            self._preview_window.seek_to_time(timestamp)
            self._preview_window.force_frame_update()
        
        if self._timeline_bridge and self._timeline_bridge.is_connected:
            self._timeline_bridge._cmd_seek({'time': str(timestamp)})

    @Slot(list)
    def _on_search_finished(self, results: list):
        """Handle search completion — worker is done, safe to seek analyzer."""
        found_results = [r for r in results if r.get("contains_target", False)]
        found_count = len(found_results)
        
        if found_count > 0:
            ts_list = ", ".join(r["timestamp_str"] for r in found_results)
            self._append_system(
                f"✅ Search complete. Found '{self.search_target.text()}' at {found_count} timestamp(s): {ts_list}"
            )
            
            # Worker is finished so _seek_to_timestamp (which touches the
            # analyzer's cv2.VideoCapture) is safe now — no race condition.
            self._seek_to_timestamp(found_results[0]["timestamp"])

        else:
            self._append_system(
                f"❌ Search complete. No '{self.search_target.text()}' found in video."
            )

        # ── Persist findings (added live during scan) to disk once ──
        if (self._timeline_bridge
                and self._timeline_bridge._window
                and hasattr(self._timeline_bridge._window, 'save_visual_findings_to_cache')):
            saved = self._timeline_bridge._window.save_visual_findings_to_cache()
            if saved and found_count > 0:
                self._append_system(
                    f"💾 Saved {found_count} '{self.search_target.text()}' "
                    f"finding(s) to cache (will reload next session)"
                )
        
        self.search_progress.setText("")
        self.search_btn.setEnabled(True)
        self.stop_search_btn.setEnabled(False)
        # Disable main stop button too (unless a chat query is still running)
        if not (self._worker_thread and self._worker_thread.isRunning()):
            self.stop_btn.setEnabled(False)

    @Slot(str)
    def _on_search_error(self, error: str):
        """Handle search error."""
        self._append_system(f"❌ Search error: {error}")
        self.search_progress.setText("Search failed")
        self.search_btn.setEnabled(True)
        self.stop_search_btn.setEnabled(False)
        if not (self._worker_thread and self._worker_thread.isRunning()):
            self.stop_btn.setEnabled(False)

    def _seek_to_timestamp(self, seconds: float):
        """Seek the timeline AND preview to a specific timestamp and ensure frame is displayed."""
        ts_str = f"{int(seconds)//60}:{int(seconds)%60:02d}"
        
        # Update analyzer position
        if self._analyzer:
            self._analyzer.current_time = seconds
            # Actually seek and read a frame to ensure it's loaded
            frame = self._analyzer.seek_to_time(seconds)
            if frame is not None:
                # Optionally cache the frame for later use
                if hasattr(self, '_current_frame'):
                    self._current_frame = frame
        
        # Update the preview window with forced frame update
        if hasattr(self, '_preview_window') and self._preview_window:
            # First seek to the time
            self._preview_window.seek_to_time(seconds)
            
            # Then force a frame update
            self._preview_window.force_frame_update()
            
            # Process events to ensure UI updates
            QApplication.processEvents()
            
            # For debugging: check if frame was captured
            if hasattr(self._preview_window, 'capture_current_frame'):
                frame_image = self._preview_window.capture_current_frame()
                if frame_image:
                    self._append_system(f"📸 Frame captured at {ts_str}")
                else:
                    self._append_system(f"⚠️ Could not capture frame at {ts_str}")
        
        # Also use timeline bridge if available
        if self._timeline_bridge and self._timeline_bridge.is_connected:
            result = self._timeline_bridge._cmd_seek({'time': str(seconds)})
            self._append_system(result)
        
        return True

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

        analyzer_status = " | Analyzer: ready" if self._analyzer else ""

        tl_status = ""
        if self._timeline_bridge and self._timeline_bridge.is_connected:
            tl_status = " | TL: connected"

        self.context_label.setText(
            f"Context: {vname} | {int(dur)}s | "
            f"{n_obj} obj | {n_act} act | {n_trans} transcript{tl_status}{analyzer_status}"
        )
        self.context_label.setStyleSheet("color:#4CAF50;font-size:9pt;font-weight:bold;")

    def _show_reasoning_stats(self):
        """Show reasoning engine statistics.
        
        Properly scoped — no more referencing `lines` from a different 
        branch, and handles missing attributes gracefully.
        """
        if not hasattr(self, 'reasoning_engine') or not self.reasoning_engine:
            if self._analysis_data:
                self._append_system("⚠️ Reasoning engine not initialized. Try toggling 'Enable reasoning' checkbox.")
            else:
                self._append_system("⚠️ No analysis data loaded. Load a cache file first.")
            return
        
        try:
            # Try to get action summary first (newer API)
            summary = self.reasoning_engine.get_action_summary()
            self._append_system(summary)
        except AttributeError:
            # Fallback to older method
            try:
                stats = self.reasoning_engine.reasoning_engine.get_action_statistics()
                lines = [
                    "📊 **Action Statistics**",
                    f"• Total actions: {stats['total_actions']}",
                    f"• Unique action types: {stats['unique_actions']}",
                    f"• Timestamps with actions: {stats['timestamps_with_actions']}",
                    f"• Action clusters: {stats['action_clusters']}",
                    "\nMost common actions:"
                ]
                for action, count in stats['most_common'][:5]:
                    lines.append(f"  • {action}: {count} times")
                
                # This block was previously OUTSIDE the except, referencing
                # `lines` that only existed inside. Now it's properly scoped.
                if hasattr(self.reasoning_engine, 'reasoning_engine') and \
                   hasattr(self.reasoning_engine.reasoning_engine, 'action_sequences') and \
                   self.reasoning_engine.reasoning_engine.action_sequences:
                    lines.append("\n🎬 **Detected Action Sequences:**")
                    for seq in self.reasoning_engine.reasoning_engine.action_sequences[:3]:
                        lines.append(
                            f"  • {seq.description} "
                            f"({int(seq.start_time)//60}:{int(seq.start_time)%60:02d} - "
                            f"{int(seq.end_time)//60}:{int(seq.end_time)%60:02d})"
                        )
                
                self._append_system("\n".join(lines))
            except Exception as e:
                self._append_system(f"⚠️ Could not retrieve statistics: {e}")

    def _save_reasoning_facts(self):
        """Save inferred facts to cache."""
        if not hasattr(self, 'reasoning_engine') or not self.reasoning_engine:
            self._append_system("⚠️ No reasoning engine to save.")
            return
        
        try:
            saved_path = self.reasoning_engine.save_analysis(self._cache_dir)
            self._append_system(f"💾 Reasoning facts saved to: {os.path.basename(saved_path)}")
        except Exception as e:
            self._append_system(f"❌ Failed to save: {e}")

    # ------------------------------------------------ Handlers

    def _on_backend_changed(self, _index):
        backend = self.backend_combo.currentData()
        is_gguf = backend == "llama-cpp"
        self.gguf_row_widget.setVisible(is_gguf)
        self.mmproj_row_widget.setVisible(is_gguf)
        self.refresh_btn.setVisible(not is_gguf)
        self.model_combo.setEnabled(not is_gguf)

        if is_gguf:
            self._populate_recent_gguf()
        else:
            self._refresh_models()

    def _populate_recent_gguf(self):
        """Fill model combo with recently used GGUF files."""
        self.model_combo.clear()
        settings = QSettings(self.SETTINGS_KEY, "LLMChat")
        recent = settings.value("recent_gguf_paths", [])
        # QSettings may return a string instead of list if only 1 item
        if isinstance(recent, str):
            recent = [recent] if recent else []

        if recent:
            for path in recent:
                # Show just filename in dropdown, store full path as data
                self.model_combo.addItem(os.path.basename(path), path)
            # Auto-fill the path input when user picks from dropdown
            self.model_combo.currentIndexChanged.connect(self._on_recent_gguf_selected)
            # Select the most recent one
            self.model_combo.setCurrentIndex(0)
            self._on_recent_gguf_selected(0)
        else:
            self.model_combo.addItem("(no recent models — use Browse)")

    def _on_recent_gguf_selected(self, index):
        """When user picks a recent GGUF from dropdown, fill the path input."""
        path = self.model_combo.currentData()
        if path and os.path.exists(path):
            self.gguf_path_input.setText(path)

    def _refresh_models(self):
        self.model_combo.clear()
        backend = self.backend_combo.currentData()
        
        if backend == "llama-cpp":
            self.model_combo.addItem("(select GGUF file below)")
            return
            
        if backend == "ollama":
            models = get_ollama_models()
            if models:
                for m in models:
                    self.model_combo.addItem(m)
                self.status_label.setText(f"Found {len(models)} Ollama models")
                self.status_label.setStyleSheet("color:#4CAF50;font-style:italic;")
            else:
                for m in ["llama3.2", "llama3.2-vision", "llava", "bakllava", "llava-llama3"]:
                    self.model_combo.addItem(m)
                self.status_label.setText("Ollama not running - showing defaults (vision models recommended)")
                self.status_label.setStyleSheet("color:#ff9800;font-style:italic;")

    def _browse_gguf(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select GGUF Model", "", "GGUF Models (*.gguf);;All Files (*)"
        )
        if path:
            self.gguf_path_input.setText(path)

    def _browse_mmproj(self):
        """Browse for mmproj file (for vision models)."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select mmproj File", "", "GGUF Models (*.gguf);;All Files (*)"
        )
        if path:
            self.mmproj_path_input.setText(path)

    def _connect_llm(self):
        backend = self.backend_combo.currentData()
        model = self.model_combo.currentText().strip()
        gguf_path = ""
        mmproj_path = None

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
                
                mmproj_path = self.mmproj_path_input.text().strip() or None
                
                self._llm = LLMModule(
                    backend="llama-cpp", 
                    model_path=gguf_path,
                    mmproj_path=mmproj_path,
                    log_fn=self._log
                )
                
                model_name = os.path.basename(gguf_path)
                self.status_label.setText(f"Loading {model_name}...")
                QApplication.processEvents()
            else:
                raise ValueError(f"Unknown backend: {backend}")

            self._llm.load()

            self._save_gguf_to_recent(gguf_path)
            settings = QSettings(self.SETTINGS_KEY, "LLMChat")
            settings.setValue("last_gguf_path", gguf_path)
            if mmproj_path:
                settings.setValue("last_mmproj_path", mmproj_path)

            if backend == "llama-cpp":
                model_name = os.path.basename(gguf_path)
                if mmproj_path:
                    model_name += " (with vision)"
                self.status_label.setText(f"Connected: {model_name}")
            else:
                self.status_label.setText(f"Connected: {model}")
                
            self.status_label.setStyleSheet("color:#4CAF50;font-weight:bold;")
            self.connect_btn.setText("Reconnect")
            self.input_field.setEnabled(True)
            self.send_btn.setEnabled(True)

            if self._video_path and os.path.exists(self._video_path):
                self._init_analyzer()

            # Update reasoning engine with now-connected LLM
            if hasattr(self, 'reasoning_engine') and self.reasoning_engine:
                self.reasoning_engine.llm = self._llm
                self._append_system("🧠 Reasoning engine now has LLM — 'why' questions enabled")

            if self._analysis_data:
                n_obj = len(self._analysis_data.get("objects", []))
                n_act = len(self._analysis_data.get("actions", []))
                self._append_system(
                    f"Connected to {model if backend=='ollama' else os.path.basename(gguf_path)}. "
                    f"Context ready: {n_obj} object entries, {n_act} actions."
                )
            else:
                self._append_system(
                    f"Connected to {model if backend=='ollama' else os.path.basename(gguf_path)}. "
                    f"WARNING: No video context! Use 'Load Cache' first."
                )

        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            self.status_label.setStyleSheet("color:#f44336;font-style:italic;")
            self.input_field.setEnabled(False)
            self.send_btn.setEnabled(False)
        finally:
            self.connect_btn.setEnabled(True)

    def _save_gguf_to_recent(self, path: str):
        """Add a GGUF path to the recent list (most recent first, no duplicates)."""
        settings = QSettings(self.SETTINGS_KEY, "LLMChat")
        recent = settings.value("recent_gguf_paths", [])
        if isinstance(recent, str):
            recent = [recent] if recent else []

        # Remove if already present, then prepend
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)

        # Cap the list
        recent = recent[:self.MAX_RECENT_GGUF]
        settings.setValue("recent_gguf_paths", recent)

        # Refresh dropdown if currently showing GGUF mode
        if self.backend_combo.currentData() == "llama-cpp":
            self._populate_recent_gguf()

    def _send_message(self):
        text = self.input_field.text().strip()
        if not text:
            return
        if not self._llm or not self._llm.is_loaded():
            self._append_system("Not connected. Click 'Connect' first.")
            return
        if self._worker_thread and self._worker_thread.isRunning():
            self._append_system("Still generating... please wait.")
            return
        
        # Special handling for reasoning questions
        if (hasattr(self, 'reasoning_engine') and self.reasoning_engine and 
            text.lower().startswith(('why ', 'how do you know', 'explain '))):
            
            current_time = 0
            if self._timeline_bridge and self._timeline_bridge._window:
                current_time = getattr(self._timeline_bridge._window, 'current_time', 0)
            
            answer = self.reasoning_engine.answer_why_question(text, current_time)
            if answer:
                self._append_user(text)
                self._append_html(
                    f'<div style="color:#FFD700;margin:8px 0;padding:8px;'
                    f'background:#2a2a2a;border-left:4px solid #FFD700;'
                    f'font-family:monospace;">'
                    f'🔍 {answer}</div>'
                )
                self.input_field.clear()
                self._chat_history.append({"role": "user", "content": text})
                self._chat_history.append({"role": "assistant", "content": answer})

                MAX_HISTORY_PAIRS = 6
                if len(self._chat_history) > MAX_HISTORY_PAIRS * 2:
                    self._chat_history = self._chat_history[-(MAX_HISTORY_PAIRS * 2):]

                return

        # Parse search requests from chat (e.g. "search for explosion every 60s")
        if self._try_parse_chat_search(text):
            self._append_user(text)
            self.input_field.clear()
            return

        # Check for mode commands
        force_visual = False
        force_text = False
        actual_message = text
        
        if text.startswith('!visual'):
            force_visual = True
            actual_message = text[7:].strip()
            self._append_system("🎯 Forcing VISUAL mode - will capture and analyze current frame")
        elif text.startswith('!text'):
            force_text = True
            actual_message = text[5:].strip()
            self._append_system("📝 Forcing TEXT mode - will ignore vision keywords and use only analysis data")

        # Handle seek commands (still useful)
        if self._handle_seek_command(actual_message):
            self.input_field.clear()
            return

        if not self._analysis_data:
            self._append_system(
                "WARNING: No analysis data loaded! LLM will hallucinate.\n"
                "Use 'Load Cache' to load a cache file first."
            )

        self._append_user(actual_message)
        self.input_field.clear()
        self._chat_history.append({"role": "user", "content": actual_message})

        self.input_field.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.send_btn.setText("...")
        self.stop_btn.setEnabled(True)

        self._append_html(
            '<div style="color:#8BC34A;margin-top:8px;"><b>Assistant:</b></div>'
        )

        # Build timeline context if connected
        timeline_ctx = ""
        has_timeline = self._timeline_bridge and self._timeline_bridge.is_connected
        if has_timeline:
            timeline_ctx = (
                self._timeline_bridge.get_timeline_state() + "\n" +
                self._timeline_bridge.get_available_commands_text()
            )

        if has_timeline:
            timeline_ctx = self._timeline_bridge.get_timeline_state()
            # Truncate clip list for small models
            lines = timeline_ctx.split('\n')
            if len(lines) > 15:
                timeline_ctx = '\n'.join(lines[:15]) + f'\n... ({len(lines)-15} more lines)'
            timeline_ctx += '\n' + self._timeline_bridge.get_available_commands_text()

        # Capture frame based on mode
        frame_b64 = None
        _text_lower = actual_message.lower()
        
        if force_visual:
            if self._timeline_bridge and self._timeline_bridge._window:
                window = self._timeline_bridge._window
                if hasattr(window, 'capture_current_frame_base64'):
                    frame_b64 = window.capture_current_frame_base64()
                    if frame_b64:
                        self._append_system(
                            f"📷 Frame captured at {window.current_time:.1f}s "
                            f"({len(frame_b64)//1024}KB)"
                        )
                        if has_timeline:
                            self._append_system(
                                "ℹ️ Combining frame analysis with timeline context..."
                            )
                    else:
                        self._append_system("⚠️ Frame capture failed")
            else:
                self._append_system("⚠️ Cannot capture frame: No timeline window connected")
        
        elif force_text:
            frame_b64 = None
            self._append_system("ℹ️ Text mode active: using analysis data only, ignoring vision")
        
        else:
            _wants_vision = any(kw in _text_lower for kw in _VISION_KEYWORDS)
            _asks_about_current = any(phrase in _text_lower for phrase in 
                                    ["current frame", "this frame", "what do you see", 
                                    "what's happening now", "describe this frame"])
            
            if (_wants_vision or _asks_about_current) and self._timeline_bridge and self._timeline_bridge._window:
                window = self._timeline_bridge._window
                if hasattr(window, 'capture_current_frame_base64'):
                    frame_b64 = window.capture_current_frame_base64()
                    if frame_b64:
                        self._append_system(
                            f"📷 Frame captured at {window.current_time:.1f}s "
                            f"({len(frame_b64)//1024}KB)"
                        )
                        if has_timeline:
                            self._append_system(
                                "ℹ️ Combining frame analysis with timeline context..."
                            )
                    else:
                        self._append_system("⚠️ Frame capture failed")

        # Create worker and host thread
        _free_chat = self.free_chat_chk.isChecked()
        self._worker_thread = QThread(self)
        self._worker = _LLMWorker(
            llm=self._llm,
            message=actual_message,
            analysis_data=self._analysis_data if not _free_chat else None,
            video_path=self._video_path,
            timeline_context=timeline_ctx,
            frame_base64=frame_b64,
            free_chat_mode=_free_chat,
        )
        self._worker.moveToThread(self._worker_thread)

        # Lifecycle — worker.finished triggers thread.quit + deleteLater on both
        self._worker_thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.error.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)

        # Our handlers
        self._worker.token_received.connect(self._on_token)
        self._worker.finished.connect(self._on_response_done)
        self._worker.error.connect(self._on_response_error)

        self._worker_thread.start()

    def _try_parse_chat_search(self, text: str) -> bool:
        """Parse visual search requests typed in the chat input.
        
        Handles natural language like:
            "search for explosion every 60s"
            "seek through video every 60s until you find a goal"
            "please seek every 60s until you will find X"
            "go through video every 30s looking for X"
            "scan every 10s for fire"
            "find person every 30s"
            "search for car" (uses current interval)
        
        Returns True if the message was handled as a search command.
        """
        import re
        text_lower = text.lower().strip()
        
        # Quick check: does this look like a search/seek request at all?
        # Must contain at least one action keyword AND either "every" or "search/find for"
        action_words = ('seek', 'search', 'scan', 'find', 'look for', 'looking for',
                        'go through', 'scrub through')
        has_action = any(w in text_lower for w in action_words)
        if not has_action:
            return False
        
        # --- Extract interval (if present) ---
        interval = None
        interval_match = re.search(r'every\s+(\d+(?:\.\d+)?)\s*s(?:ec(?:ond)?s?)?', text_lower)
        if interval_match:
            interval = float(interval_match.group(1))
        
        # --- Extract target ---
        target = None
        
        # Pattern group 1: "until you [will] find <target>"  /  "until <target> is found"
        # Covers: "seek every 60s until you find X", "seek until you will find X"
        m = re.search(r'until\s+(?:you\s+)?(?:will\s+)?(?:find|see|spot|locate)\s+(.+)', text_lower)
        if m:
            target = m.group(1).strip()
        
        if not target:
            m = re.search(r'until\s+(.+?)\s+is\s+(?:found|seen|spotted|located|detected)', text_lower)
            if m:
                target = m.group(1).strip()

        # Pattern group 2: "looking for <target>"  /  "for <target>" at end
        if not target:
            m = re.search(r'(?:looking|searching|scanning)\s+for\s+(.+)', text_lower)
            if m:
                target = m.group(1).strip()
        
        # Pattern group 3: "search/scan/find for <target> [every Ns]"
        if not target:
            m = re.search(r'(?:search|scan|find|look)\s+(?:for|the)\s+(.+?)(?:\s+every\s+|$)', text_lower)
            if m:
                target = m.group(1).strip()
        
        # Pattern group 3b: "find <target> every Ns" (no "for")
        if not target:
            m = re.search(r'(?:find|seek)\s+(.+?)\s+every\s+', text_lower)
            if m:
                target = m.group(1).strip()
                # Don't match filler like "find through video every"
                filler = {'through', 'the', 'video', 'in', 'a', 'an', 'this', 'my'}
                if target in filler or all(w in filler for w in target.split()):
                    target = None
        
        # Pattern group 4: "every Ns for <target>"  (interval before target)
        if not target:
            m = re.search(r'every\s+\d+(?:\.\d+)?\s*s(?:ec(?:ond)?s?)?\s+(?:for|to find|looking for)\s+(.+)', text_lower)
            if m:
                target = m.group(1).strip()
        
        if not target:
            return False
        
        # --- Clean up target ---
        # Remove common trailing filler words and punctuation
        target = re.sub(
            r'\s+(?:in\s+(?:the\s+)?(?:video|frame|frames|clip)|every\s+\d+.*|please|thanks?)\.?$',
            '', target
        ).strip()
        target = target.rstrip('.,!?')
        
        # If we still have "every Ns" embedded in the target, remove it
        target = re.sub(r'\s*every\s+\d+(?:\.\d+)?\s*s(?:ec(?:ond)?s?)?\s*', ' ', target).strip()
        
        if not target or len(target) < 2:
            return False
        
        # --- Apply ---
        self.search_target.setText(target)
        if interval is not None:
            self.search_interval.setValue(interval)
            self._append_system(f"🔍 Parsed search: '{target}' every {interval}s")
        else:
            self._append_system(
                f"🔍 Parsed search: '{target}' "
                f"(using current interval: {self.search_interval.value()}s)"
            )
        self._start_visual_search()
        return True

    def _handle_seek_command(self, text: str) -> bool:
        """Handle direct seek commands without going through LLM."""
        text_lower = text.lower()
        
        if text_lower.startswith(('seek ', 'go to ', 'jump to ')):
            parts = text.split()
            if len(parts) >= 2:
                time_str = parts[-1]
                try:
                    if ':' in time_str:
                        minutes, seconds = map(int, time_str.split(':'))
                        seconds = minutes * 60 + seconds
                    else:
                        seconds = float(time_str)
                    
                    return self._seek_to_timestamp(seconds)
                except ValueError:
                    pass
        
        return False
  
    def _stop_generation(self):
        """Stop the current LLM generation or visual search.
        
        Note: For GGUF vision models, the image encoding/decoding step is a
        single blocking C call (~20s) that cannot be interrupted from Python.
        Stop takes effect after the current frame finishes processing.
        """
        stopped_something = False
        if self._worker_thread and self._worker_thread.isRunning():
            if self._worker:
                self._worker.cancel()
            stopped_something = True
        if self._search_worker_thread and self._search_worker_thread.isRunning():
            if self._search_worker:
                self._search_worker.cancel()
            self.search_btn.setEnabled(True)
            self.stop_search_btn.setEnabled(False)
            self.search_progress.setText("Stopping after current frame...")
            stopped_something = True
        if stopped_something:
            self.stop_btn.setEnabled(False)
            self._append_system(
                "⏹ Stop requested — will stop after current frame finishes processing.\n"
                "   (GGUF image decoding is ~20s and cannot be interrupted mid-frame)"
            )

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

    def closeEvent(self, event):
        """Clean up resources on close.

        Wait for running threads to finish — destroying a QThread while
        its run-loop is still active triggers qFatal. Cancellation sets a
        flag; uninterruptible C calls (GGUF image decode) need to finish
        naturally before the worker can emit finished.
        """
        # LLM worker
        if self._worker:
            self._worker.cancel()
        if self._worker_thread and self._worker_thread.isRunning():
            if not self._worker_thread.wait(5000):
                print("⚠️ LLM thread did not stop in time, terminating")
                self._worker_thread.terminate()
                self._worker_thread.wait(2000)

        # Visual search worker
        if self._search_worker:
            self._search_worker.cancel()
        if self._search_worker_thread and self._search_worker_thread.isRunning():
            if not self._search_worker_thread.wait(5000):
                print("⚠️ Search thread did not stop in time, terminating")
                self._search_worker_thread.terminate()
                self._search_worker_thread.wait(2000)

        if self._analyzer:
            self._analyzer.close()
        super().closeEvent(event)

# ---------------------------------------------------------------------------
# Standalone window for testing
# ---------------------------------------------------------------------------
class LLMChatWindow(QWidget):
    def __init__(self, cache_dir: str = "./cache", video_path: str = ""):
        super().__init__()
        self.setWindowTitle("VideoHighlighter - LLM Chat with Visual Search")
        self.setMinimumSize(700, 600)
        layout = QVBoxLayout()
        self.chat = LLMChatWidget(parent=self, cache_dir=cache_dir, video_path=video_path)
        layout.addWidget(self.chat)
        self.setLayout(layout)


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Chat with Visual Search")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--cache-dir", type=str, default="./cache", help="Cache directory")
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    win = LLMChatWindow(cache_dir=args.cache_dir, video_path=args.video or "")
    win.show()
    sys.exit(app.exec())