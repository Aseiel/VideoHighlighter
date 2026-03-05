"""
Complete Signal Timeline Viewer with Filters and Edit Timeline
- Signal visualization with filtering
- Edit timeline with clip management
- Action/object filtering
- Exact time playback
"""

import sys
import os
import threading
from pathlib import Path
import json
import numpy as np
from collections import defaultdict
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, 
    QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel,
    QCheckBox, QGroupBox, QSplitter, QScrollArea,
    QFrame, QLineEdit, QSlider, QGraphicsRectItem, QGraphicsTextItem,
    QMessageBox, QDockWidget, QMenu, QGraphicsLineItem,
    QComboBox, QListWidget, QListWidgetItem, QDialog,
    QDialogButtonBox, QFormLayout, QTabWidget
)
from PySide6.QtCore import Qt, QRectF, Signal, Slot, QPointF, QTimer, QPoint, QMimeData
from PySide6.QtGui import (
    QColor, QPen, QBrush, QPainter, QFont, QPainterPath, 
    QLinearGradient, QRadialGradient, QCursor, QAction,
    QPainterPath, QFontMetrics, QDrag, QPixmap
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
import subprocess
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from datetime import datetime, timedelta


# modules
from video_ai_editor.video_preview import TimelineWithPreview
from video_ai_editor.bbox_overlay import AnnotatedVideoManager
from video_ai_editor.timeline_export import TimelineExporter
from video_ai_editor.waveform import WaveformVisualizer
from video_ai_editor.timeline_bars import TimelineBar
from video_ai_editor.signal_timeline import SignalTimelineScene, SignalTimelineView
from video_ai_editor.edit_timeline import EditTimelineScene
from video_ai_editor.filter_dialogs import FilterDialog, ConfidenceFilterDialog



class SignalTimelineWindow(QMainWindow):
    """Main window for signal timeline viewer with edit timeline and filters"""
    waveform_ready = Signal(object)
    render_finished = Signal(bool, str)
    
    def __init__(self, video_path, cache_data=None):
        # Add this IMMEDIATELY at the start of __init__
        debug_log(f"SignalTimelineWindow.__init__ CALLED with video_path={video_path}")
        debug_log(f"  cache_data provided: {cache_data is not None}")
        debug_log(f"\n{'='*60}")
        debug_log(f"🔍 [TIMELINE] SignalTimelineWindow.__init__ START")
        debug_log(f"{'='*60}")
        debug_log(f"  - video_path: {video_path}")
        debug_log(f"  - cache_data provided: {cache_data is not None}")
        debug_log(f"  - cache_data type: {type(cache_data)}")
        
        if cache_data is not None:
            debug_log(f"  - cache_data keys: {list(cache_data.keys()) if cache_data else 'None'}")
        
        super().__init__()
        self.video_path = video_path
        
        # If cache_data was provided, use it directly
        if cache_data is not None:
            debug_log(f"  ✓ Using provided cache_data")
            self.cache_data = cache_data
        else:
            debug_log(f"  ⚠️ No cache_data provided, attempting to load...")
            self.cache_data = self.load_cache_data()
            
            # If still no cache_data, create minimal structure
            if not self.cache_data:
                debug_log(f"  ⚠️ Creating minimal cache data structure")
                self.cache_data = {
                    "video_metadata": {"duration": 0, "fps": 30},
                    "transcript": {"segments": []},
                    "objects": [],
                    "actions": [],
                    "scenes": [],
                    "motion_events": [],
                    "motion_peaks": [],
                    "audio_peaks": []
                }
        
        debug_log(f"\n  📊 FINAL CACHE DATA STATE:")
        debug_log(f"  - self.cache_data is None? {self.cache_data is None}")
        if self.cache_data:
            debug_log(f"  - self.cache_data keys: {list(self.cache_data.keys())}")
            # Check for motion data specifically
            debug_log(f"    - 'motion_events' present: {'motion_events' in self.cache_data}")
            debug_log(f"    - 'motion_peaks' present: {'motion_peaks' in self.cache_data}")
            debug_log(f"    - 'scenes' present: {'scenes' in self.cache_data}")
            debug_log(f"    - 'video_metadata' present: {'video_metadata' in self.cache_data}")
            
            if 'video_metadata' in self.cache_data:
                debug_log(f"      - duration: {self.cache_data['video_metadata'].get('duration', 'N/A')}")
        
        # Get video duration from cache or fallback
        self.video_duration = self.cache_data.get('video_metadata', {}).get('duration', 0) if self.cache_data else 0
        debug_log(f"  - video_duration from cache: {self.video_duration}")
        
        # If we still don't have duration, try to get it from the video file
        if self.video_duration == 0 and os.path.exists(video_path):
            try:
                import cv2
                debug_log(f"  - Attempting to get duration from video file...")
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.video_duration = total_frames / fps if fps else 0
                cap.release()
                debug_log(f"  - Got video duration from file: {self.video_duration:.1f}s")
            except Exception as e:
                debug_log(f"  ⚠️ Could not get video duration: {e}")
                self.video_duration = 60  # fallback
        
        self.cache = self.get_cache_instance()
        debug_log(f"  - cache instance: {self.cache is not None}")
        
        self.current_time = 0
        
        # Track clip removals for batch updates
        self.pending_clip_removals = []
        self.removal_timer = QTimer()
        self.removal_timer.setSingleShot(True)
        self.removal_timer.timeout.connect(self.process_pending_removals)
        
        # Extract info for display
        self.action_types = self._extract_action_types()
        self.object_classes = self._extract_object_classes()
        
        debug_log(f"\n  📊 EXTRACTED INFO:")
        debug_log(f"  - action_types: {self.action_types}")
        debug_log(f"  - object_classes: {self.object_classes}")
        
        self.setWindowTitle(f"Signal Timeline - {os.path.basename(video_path)}")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Make window semi-transparent
        self.setWindowOpacity(0.98)
        
        # Load waveform from cache - store it in instance variable
        self.waveform = self.load_waveform_from_cache()
        debug_log(f"  - waveform loaded: {self.waveform is not None}, length: {len(self.waveform) if self.waveform else 0}")
        
        # Initialize UI - PASS waveform to constructor
        debug_log(f"\n  🎨 Initializing UI...")
        self.init_ui()
        
        # bbox_manager is created inside create_video_preview_dock()
        # — no need to create it again here

        # Start background extraction if we don't have cached waveform
        if not self.waveform or len(self.waveform) == 0:
            debug_log(f"  ⚠️ No cached waveform or empty waveform, starting extraction...")
            self.init_waveform()
        else:
            debug_log(f"  ✅ Using cached waveform ({len(self.waveform)} points)")
        
        debug_log(f"\n{'='*60}")
        debug_log(f"✅ [TIMELINE] SignalTimelineWindow.__init__ COMPLETE")
        debug_log(f"{'='*60}\n")

    def launch_preview(self):
        """Launch video preview window"""
        chat = getattr(self, 'llm_chat', None)
        self.preview_window = TimelineWithPreview.launch_preview(self, chat_widget=chat)

    def closeEvent(self, event):
        """Close preview when timeline closes"""
        if hasattr(self, 'preview_window') and self.preview_window:
            self.preview_window.close()
        super().closeEvent(event)

    def _on_bbox_toggled(self, label: str):
        """Visual feedback when overlay is toggled."""
        is_original = (label == "🎥 Original")
        state = "Original" if is_original else f"Overlay: {label}"
        self.statusBar().showMessage(f"Video source: {state}", 3000)

        # Hide detection panel when viewing annotated video (avoids double info)
        if hasattr(self, 'detection_panel'):
            self.detection_panel.setVisible(is_original)

    def create_video_preview_dock(self):
        """
        Create video preview dock with dual overlay modes:
          - Off:     Plain video (QVideoWidget)
          - Live:    Real-time bbox overlay from cache (QGraphicsVideoItem + scene)
          - Precomp: Pre-rendered annotated video swap (bbox_overlay.py)
        """
        from PySide6.QtCore import Qt, QUrl
        from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
        from PySide6.QtMultimediaWidgets import QVideoWidget
        from PySide6.QtWidgets import QStackedWidget

        dock = QDockWidget("Video Preview", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        preview_widget = QWidget()
        layout = QVBoxLayout(preview_widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ──────────────────────────────────────────────────────────
        # Mode selector row
        # ──────────────────────────────────────────────────────────
        mode_row = QHBoxLayout()

        mode_label = QLabel("Overlay:")
        mode_label.setStyleSheet("color: #a0c0ff; font-weight: bold;")
        mode_row.addWidget(mode_label)

        self.overlay_mode_combo = QComboBox()
        self.overlay_mode_combo.addItems([
            "Off",
            "Live (real-time)",
            "Precomp (swap video)",
        ])
        self.overlay_mode_combo.setToolTip(
            "Off — plain video, no overlays\n"
            "Live — real-time bboxes from cache (needs bbox data)\n"
            "Precomp — swap to pre-rendered annotated video"
        )
        self.overlay_mode_combo.setStyleSheet("""
            QComboBox {
                background-color: #1a1a2a; color: #ddd;
                border: 1px solid #3a3a5a; border-radius: 4px;
                padding: 4px 8px; min-width: 160px;
            }
            QComboBox:hover { border-color: #5a5a8a; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #1a1a2a; color: #ddd;
                selection-background-color: #3a5fcd;
            }
        """)
        mode_row.addWidget(self.overlay_mode_combo)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        # ──────────────────────────────────────────────────────────
        # Stacked video area: page 0 = QVideoWidget, page 1 = Live overlay
        # ──────────────────────────────────────────────────────────
        video_and_info = QSplitter(Qt.Orientation.Horizontal)

        # -- Left: stacked video widget --
        self.preview_stack = QStackedWidget()

        # Page 0: Plain QVideoWidget (Off + Precomp modes)
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(320, 240)
        self.video_widget.setStyleSheet("background-color: black; border: 2px solid #3a3a5a;")
        self.preview_stack.addWidget(self.video_widget)  # index 0

        # Page 1: Live real-time overlay
        self.realtime_preview = None
        try:
            from video_ai_editor.realtime_overlay import RealtimeOverlayPreview
            self.realtime_preview = RealtimeOverlayPreview(
                video_path=self.video_path,
                cache_data=self.cache_data,
            )
            self.preview_stack.addWidget(self.realtime_preview)  # index 1
            print(f"✅ Live overlay loaded ({self.realtime_preview.get_detection_count()} detections)")
        except ImportError as e:
            print(f"⚠️ realtime_overlay not available: {e}")
        except Exception as e:
            print(f"⚠️ realtime_overlay init failed: {e}")
            import traceback; traceback.print_exc()

        self.preview_stack.setCurrentIndex(0)
        video_and_info.addWidget(self.preview_stack)

        # -- Right: Detection info panel --
        self.detection_panel = QLabel("No detections")
        self.detection_panel.setWordWrap(True)
        self.detection_panel.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.detection_panel.setMinimumWidth(180)
        self.detection_panel.setMaximumWidth(280)
        self.detection_panel.setStyleSheet("""
            QLabel {
                background-color: #0a0a18;
                color: #d0d8ff;
                border: 1px solid #3a3a5a;
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        video_and_info.addWidget(self.detection_panel)
        video_and_info.setSizes([500, 200])

        layout.addWidget(video_and_info, 1)

        # ──────────────────────────────────────────────────────────
        # Media player (shared — used in Off + Precomp modes)
        # ──────────────────────────────────────────────────────────
        self.video_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.video_player.setAudioOutput(self.audio_output)
        self.video_player.setVideoOutput(self.video_widget)
        self.video_player.setSource(QUrl.fromLocalFile(self.video_path))
        self.audio_output.setVolume(0.8)

        # Active player pointer — switches between shared and live
        self._active_player = self.video_player

        # ──────────────────────────────────────────────────────────
        # Transport controls
        # ──────────────────────────────────────────────────────────
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 4, 0, 0)

        # Play button
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.toggle_video_playback)
        self.play_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a5fcd; color: white; font-weight: bold;
                padding: 8px 16px; border-radius: 4px; min-width: 80px;
            }
            QPushButton:hover { background-color: #4a6fdd; }
        """)
        controls_layout.addWidget(self.play_btn)

        # Time slider
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 100)
        self.time_slider.sliderMoved.connect(self.seek_video)
        controls_layout.addWidget(self.time_slider)

        # Time label
        self.preview_time_label = QLabel("00:00 / 00:00")
        self.preview_time_label.setStyleSheet("""
            QLabel {
                color: #a0ffa0; font-family: 'Consolas', monospace;
                font-weight: bold; padding: 8px; background-color: #1a1a2a;
                border-radius: 4px; min-width: 120px;
                qproperty-alignment: AlignCenter;
            }
        """)
        controls_layout.addWidget(self.preview_time_label)

        # Show detections checkbox
        self.show_detections_checkbox = QCheckBox("Show Detections")
        self.show_detections_checkbox.setChecked(True)
        self.show_detections_checkbox.stateChanged.connect(self._toggle_detection_panel)
        controls_layout.addWidget(self.show_detections_checkbox)

        controls_layout.addStretch()

        # Volume
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("🔊"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.valueChanged.connect(self.set_volume)
        self.volume_slider.setFixedWidth(80)
        volume_layout.addWidget(self.volume_slider)
        controls_layout.addLayout(volume_layout)

        layout.addWidget(controls_widget)

        # ──────────────────────────────────────────────────────────
        # Precomp overlay controls (bbox_overlay.py)
        # ──────────────────────────────────────────────────────────
        self.bbox_manager = None
        try:
            from video_ai_editor.bbox_overlay import AnnotatedVideoManager
            self.bbox_manager = AnnotatedVideoManager(
                video_path=self.video_path,
                cache_data=self.cache_data,
                player=self.video_player,
                parent=self,
            )

            # Create widget but start hidden (only shown in Precomp mode)
            self._precomp_widget = self.bbox_manager.create_toggle_widget()
            self._precomp_widget.setVisible(False)
            layout.addWidget(self._precomp_widget)

            # Connect source change signal
            self.bbox_manager.source_changed.connect(self._on_bbox_toggled)

            print(f"✅ Precomp overlay manager ready")
        except ImportError as e:
            print(f"⚠️ bbox_overlay not available: {e}")
            self._precomp_widget = None
        except Exception as e:
            print(f"⚠️ bbox_overlay init failed: {e}")
            import traceback; traceback.print_exc()
            self._precomp_widget = None

        # ──────────────────────────────────────────────────────────
        # Connect shared player signals
        # ──────────────────────────────────────────────────────────
        self.video_player.durationChanged.connect(self.update_video_duration)
        self.video_player.positionChanged.connect(self._on_shared_player_position)
        self.video_player.playbackStateChanged.connect(self.update_play_button)

        # Connect live player signals (if available)
        if self.realtime_preview is not None:
            live_player = self.realtime_preview.player
            live_player.durationChanged.connect(self.update_video_duration)
            live_player.positionChanged.connect(self._on_live_player_position)
            live_player.playbackStateChanged.connect(self.update_play_button)

        # ──────────────────────────────────────────────────────────
        # Connect mode selector (after everything is built)
        # ──────────────────────────────────────────────────────────
        self.overlay_mode_combo.currentTextChanged.connect(self._on_overlay_mode_changed)

        dock.setWidget(preview_widget)
        return dock

    def _on_overlay_mode_changed(self, text):
        """Switch between Off / Live / Precomp overlay modes."""

        # Capture current position from whichever player is active
        current_pos = self._active_player.position()
        was_playing = (
            self._active_player.playbackState() == QMediaPlayer.PlayingState
        )

        # Pause the outgoing player
        self._active_player.pause()

        if "Live" in text:
            # ── Switch to Live real-time overlay ──
            if self.realtime_preview is None:
                self.statusBar().showMessage(
                    "⚠️ Live overlay not available — module not loaded", 3000
                )
                self.overlay_mode_combo.blockSignals(True)
                self.overlay_mode_combo.setCurrentText("Off")
                self.overlay_mode_combo.blockSignals(False)
                return

            self.preview_stack.setCurrentIndex(1)
            self._active_player = self.realtime_preview.player

            # Hide precomp controls
            if self._precomp_widget:
                self._precomp_widget.setVisible(False)

            # Sync position — play briefly to force frame render, then restore state
            self._active_player.setPosition(current_pos)
            if was_playing:
                self._active_player.play()
            else:
                # Must play+pause to force QGraphicsVideoItem to render a frame
                self._active_player.play()
                QTimer.singleShot(100, self._active_player.pause)

            # Refit view after video dimensions are known
            QTimer.singleShot(200, self.realtime_preview._view._fit_video)

            count = self.realtime_preview.get_detection_count()
            self.statusBar().showMessage(
                f"🎯 Live overlay mode — {count} detections from cache", 3000
            )
            
        elif "Precomp" in text:
            # ── Switch to Precomp (annotated video swap) ──
            self.preview_stack.setCurrentIndex(0)
            self._active_player = self.video_player

            # Ensure shared player outputs to QVideoWidget
            self.video_player.setVideoOutput(self.video_widget)

            # Show precomp controls
            if self._precomp_widget:
                self._precomp_widget.setVisible(True)

            # Sync position
            self._active_player.setPosition(current_pos)
            if was_playing:
                self._active_player.play()

            self.statusBar().showMessage(
                "🎬 Precomp mode — select annotated video from dropdown", 3000
            )

        else:
            # ── Off mode ──
            self.preview_stack.setCurrentIndex(0)
            self._active_player = self.video_player

            # Ensure shared player outputs to QVideoWidget
            self.video_player.setVideoOutput(self.video_widget)

            # Reset to original video if bbox_manager swapped it
            if self.bbox_manager and self.bbox_manager._current_source != "🎥 Original":
                self.bbox_manager._switch_to("🎥 Original")

            # Hide precomp controls
            if self._precomp_widget:
                self._precomp_widget.setVisible(False)

            # Sync position
            self._active_player.setPosition(current_pos)
            if was_playing:
                self._active_player.play()

            self.statusBar().showMessage("Video overlay off", 3000)

    def _on_shared_player_position(self, position):
        """Position updates from shared player (Off + Precomp modes)."""
        if self._active_player is not self.video_player:
            return  # Ignore if live mode is active
        self._handle_position_update(position)

    def _on_live_player_position(self, position):
        """Position updates from live overlay player."""
        if self.realtime_preview is None:
            return
        if self._active_player is not self.realtime_preview.player:
            return  # Ignore if not in live mode
        self._handle_position_update(position)

    def _handle_position_update(self, position):
        """Shared logic for any player position update."""
        duration = self._active_player.duration()
        if duration <= 0:
            return

        # Update slider
        percent = (position / duration) * 100
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(int(percent))
        self.time_slider.blockSignals(False)

        # Update time display
        self.update_time_display(position)

        # Update detection panel
        time_seconds = position / 1000.0
        self._update_detection_panel(time_seconds)

        # Update signal timeline playhead during playback
        if self._active_player.playbackState() == QMediaPlayer.PlayingState:
            self.current_time = time_seconds
            self.signal_scene.set_current_time(self.current_time)
            if hasattr(self, 'signal_view'):
                self.signal_view.ensure_time_visible(self.current_time)

    def _toggle_detection_panel(self, state):
        """Show/hide detection panel"""
        if hasattr(self, 'detection_panel'):
            self.detection_panel.setVisible(state == Qt.Checked)

    def _update_detection_panel(self, time_seconds):
        """Update detection info panel with actions/objects at current time"""
        if not hasattr(self, 'detection_panel') or not self.detection_panel.isVisible():
            return
        if not self.cache_data:
            return
        
        time_window = 1.0
        lines = []
        
        # ── Actions ──
        actions = []
        for act in self.cache_data.get('actions', []):
            ts = act.get('timestamp', -999)
            if abs(ts - time_seconds) > time_window:
                continue
            name = act.get('action_name') or act.get('action', '?')
            conf = act.get('confidence', 0)
            model = act.get('model_type', '')
            actions.append((name, conf, model))
        
        actions.sort(key=lambda x: x[1], reverse=True)
        
        if actions:
            lines.append('<b style="color: #80b0ff;">━━ ACTIONS ━━</b>')
            for name, conf, model in actions[:5]:
                # Confidence bar using block chars
                bar_len = int(conf * 12)
                bar = '█' * bar_len + '░' * (12 - bar_len)
                
                if 'custom' in model:
                    color = '#00ff00'
                elif 'cuda' in model or 'r3d' in model:
                    color = '#0080ff'
                else:
                    color = '#00a5ff'
                
                tag = f' <span style="color:#888;">[{model}]</span>' if model else ''
                lines.append(
                    f'<span style="color:{color};">{bar} {conf:.0%}</span><br>'
                    f'  <b>{name}</b>{tag}'
                )
            lines.append('')
        
        # ── Objects ──
        objects = []
        for obj_entry in self.cache_data.get('objects', []):
            ts = obj_entry.get('timestamp', -999)
            if abs(ts - time_seconds) > time_window:
                continue
            for obj_name in obj_entry.get('objects', []):
                if isinstance(obj_name, str) and obj_name not in objects:
                    objects.append(obj_name)
        
        if objects:
            lines.append('<b style="color: #80ff80;">━━ OBJECTS ━━</b>')
            for obj in objects[:8]:
                lines.append(f'  • {obj}')
            lines.append('')
        
        # ── Timestamp ──
        mins, secs = divmod(int(time_seconds), 60)
        ms = int((time_seconds % 1) * 100)
        lines.insert(0, f'<b style="color: #00ffff; font-size: 13px;">{mins:02d}:{secs:02d}.{ms:02d}</b>')
        
        if not actions and not objects:
            lines.append('<span style="color: #666;">No detections</span>')
        
        self.detection_panel.setText('<br>'.join(lines))

    def toggle_video_playback(self):
        if self._active_player.playbackState() == QMediaPlayer.PlayingState:
            self._active_player.pause()
            self.play_btn.setText("▶ Play")
        else:
            self._active_player.play()
            self.play_btn.setText("⏸ Pause")

    def seek_video(self, position):
        """Seek video to specific position"""
        if self.video_player.duration() > 0:
            new_position = (position / 100.0) * self.video_player.duration()
            self.video_player.setPosition(int(new_position))

    def set_volume(self, value):
        """Set video volume"""
        self.audio_output.setVolume(value / 100.0)

    def update_video_duration(self, duration):
        """Update video duration display"""
        if duration > 0:
            self.time_slider.setRange(0, 100)
            total_seconds = duration // 1000
            mins = total_seconds // 60
            secs = total_seconds % 60
            self.total_duration_str = f"{mins:02d}:{secs:02d}"
            self.update_time_display(self.video_player.position())

    def update_video_position(self, position):
        if self.video_player.duration() > 0:
            percent = (position / self.video_player.duration()) * 100
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(int(percent))
            self.time_slider.blockSignals(False)
            
            self.update_time_display(position)
            
            # ── Update detection panel ──
            time_seconds = position / 1000.0
            self._update_detection_panel(time_seconds)
            
            if self.video_player.playbackState() == QMediaPlayer.PlayingState:
                self.current_time = time_seconds
                self.signal_scene.set_current_time(self.current_time)
                if hasattr(self, 'signal_view'):
                    self.signal_view.ensure_time_visible(self.current_time)

    def update_time_display(self, position):
        current_seconds = position // 1000
        mins = current_seconds // 60
        secs = current_seconds % 60
        current_time_str = f"{mins:02d}:{secs:02d}"
        
        if hasattr(self, 'total_duration_str'):
            self.preview_time_label.setText(f"{current_time_str} / {self.total_duration_str}")
        else:
            self.preview_time_label.setText(f"{current_time_str}")

    def update_play_button(self, state):
        """Update play button based on playback state"""
        if state == QMediaPlayer.PlayingState:
            self.play_btn.setText("⏸ Pause")
        else:
            self.play_btn.setText("▶ Play")

    def update_preview_time_label(self):
        """Update the time label in preview"""
        if hasattr(self, 'preview_time_label'):
            current_mins = int(self.current_time // 60)
            current_secs = int(self.current_time % 60)
            total_mins = int(self.video_duration // 60)
            total_secs = int(self.video_duration % 60)
            self.preview_time_label.setText(f"{current_mins:02d}:{current_secs:02d} / {total_mins:02d}:{total_secs:02d}")


    def simulate_playback(self):
        """Simulate video playback progress"""
        if hasattr(self, 'preview_playing') and self.preview_playing:
            # Increment time
            self.current_time += 0.1  # 100ms increments
            
            # Check if we reached the end
            if self.current_time >= self.video_duration:
                self.current_time = 0  # Loop back to start
            
            # Update timeline
            self.signal_scene.set_current_time(self.current_time)
            
            # Update preview display
            if hasattr(self, 'video_display'):
                self.video_display.setText(f"▶ Playing at {self.current_time:.1f}s")
            
            # Update time label
            self.update_preview_time_label()

    def toggle_preview_playback(self):
        """Start external player but show status in preview"""
        if not hasattr(self, 'preview_playing'):
            self.preview_playing = False
        
        if not self.preview_playing:
            # Start external player
            self.play_video_time(self.current_time)  # This opens external player
            
            # Mark as playing in preview
            self.preview_playing = True
            self.preview_play_btn.setText("⏸ Pause")
            
            # Show status in preview window
            if hasattr(self, 'video_display'):
                self.video_display.setText(f"▶ Playing in external player\nTime: {self.current_time:.1f}s")
                self.video_display.setStyleSheet("""
                    QLabel {
                        background-color: #1a3a2a;
                        color: #a0ffa0;
                        border: 2px solid #4a7a5a;
                        border-radius: 6px;
                        font-size: 14px;
                        padding: 20px;
                    }
                """)
            
            self.statusBar().showMessage(f"▶ Playing in external player at {self.current_time:.1f}s", 2000)
        else:
            # Can't actually pause external player, just update UI
            self.preview_playing = False
            self.preview_play_btn.setText("▶ Play")
            
            if hasattr(self, 'video_display'):
                self.video_display.setText(f"⏸ Click to play\nLast time: {self.current_time:.1f}s")
                self.video_display.setStyleSheet("""
                    QLabel {
                        background-color: #0a0a14;
                        color: #c0d0ff;
                        border: 2px solid #3a3a5a;
                        border-radius: 6px;
                        font-size: 14px;
                        padding: 20px;
                    }
                """)
            
            self.statusBar().showMessage("⏸ Preview paused", 2000)

    def update_preview_display_playing(self):
        """Update display to show playing state"""
        if hasattr(self, 'video_display'):
            self.video_display.setText(f"▶ Playing at {self.current_time:.1f}s")
            self.video_display.setStyleSheet("""
                QLabel {
                    background-color: #1a3a2a;
                    color: #a0ffa0;
                    border: 2px solid #4a7a5a;
                    border-radius: 6px;
                    font-size: 14px;
                    padding: 20px;
                }
            """)

    def update_preview_display_paused(self):
        """Update display to show paused state"""
        if hasattr(self, 'video_display'):
            self.video_display.setText(f"⏸ Paused at {self.current_time:.1f}s")
            self.video_display.setStyleSheet("""
                QLabel {
                    background-color: #0a0a14;
                    color: #c0d0ff;
                    border: 2px solid #3a3a5a;
                    border-radius: 6px;
                    font-size: 14px;
                    padding: 20px;
                }
            """)

    def update_preview_display(self):
        """Update preview display during playback"""
        if hasattr(self, 'preview_playing') and self.preview_playing:
            # Simulate time progression (if using external player, this won't be accurate)
            # For now, just show that we're playing
            pass

    def toggle_ai_overlay_preview(self, state):
        """Toggle AI overlay in preview"""
        from PySide6.QtCore import Qt
        
        if state == Qt.Checked:
            self.statusBar().showMessage("AI Overlays: ON (requires annotated video)", 2000)
        else:
            self.statusBar().showMessage("AI Overlays: OFF", 2000)
        
        # Update the display based on overlay state
        self.update_preview_display_based_on_state()

    def update_preview_display_based_on_state(self):
        """Update preview display based on current state"""
        if not hasattr(self, 'video_display'):
            return
        
        if hasattr(self, 'preview_playing') and self.preview_playing:
            state_text = "Playing"
            bg_color = "#1a3a2a"
            text_color = "#a0ffa0"
            border_color = "#4a7a5a"
        else:
            state_text = "Paused"
            bg_color = "#0a0a14"
            text_color = "#c0d0ff"
            border_color = "#3a3a5a"
        
        if hasattr(self, 'preview_overlay_toggle') and self.preview_overlay_toggle.isChecked():
            overlay_text = "\nAI Overlays: ON"
            bg_color = "#2a1a3a"  # Purple for AI mode
            text_color = "#ffa0ff"
            border_color = "#7a4a7a"
        else:
            overlay_text = "\nAI Overlays: OFF"
        
        self.video_display.setText(f"{state_text} at {self.current_time:.1f}s{overlay_text}")
        self.video_display.setStyleSheet(f"""
            QLabel {{
                background-color: {bg_color};
                color: {text_color};
                border: 2px solid {border_color};
                border-radius: 6px;
                font-size: 14px;
                padding: 20px;
            }}
        """)

    def update_waveform_checkbox_state(self):
        """Enable/disable + sync the waveform checkbox with actual waveform availability."""
        if not hasattr(self, "waveform_checkbox"):
            return
        if not hasattr(self, "signal_scene") or self.signal_scene is None:
            return

        has_waveform = bool(self.signal_scene.waveform) and len(self.signal_scene.waveform) > 0

        # Avoid triggering toggle_waveform while we programmatically change the checkbox
        self.waveform_checkbox.blockSignals(True)
        try:
            self.waveform_checkbox.setEnabled(has_waveform)

            # If we have waveform data, reflect the scene's visibility state.
            # If we don't, force unchecked.
            if has_waveform:
                self.waveform_checkbox.setChecked(bool(self.signal_scene.waveform_visible))
            else:
                self.waveform_checkbox.setChecked(False)
        finally:
            self.waveform_checkbox.blockSignals(False)

    def _apply_pending_waveform(self):
        if hasattr(self, '_pending_waveform_data'):
            data = self._pending_waveform_data
            delattr(self, '_pending_waveform_data')

            if not hasattr(self, 'signal_scene') or self.signal_scene is None:
                print("⚠️ No signal_scene yet, cannot apply waveform")
                return

            self.update_waveform_data(data)

    def load_waveform_from_cache(self):
        """Try to load waveform from cache data"""
        try:
            if self.cache_data:
                # Check in various possible locations
                waveform_data = None
                
                # Try direct access
                if 'waveform_data' in self.cache_data:
                    waveform_data = self.cache_data['waveform_data']
                
                # Try under video_metadata
                elif 'video_metadata' in self.cache_data and 'waveform' in self.cache_data['video_metadata']:
                    waveform_data = self.cache_data['video_metadata']['waveform']
                
                if waveform_data and len(waveform_data) > 0:
                    print(f"✅ Loaded waveform from cache ({len(waveform_data)} points)")
                    return waveform_data
                else:
                    print(f"⚠️ Waveform data found but empty")
        except Exception as e:
            print(f"⚠️ Could not load cached waveform: {e}")
        
        return None


    def init_waveform(self):
        """Initialize waveform visualization in background with better debugging"""
        # First check if video even has audio
        try:
            result = subprocess.run([
                "ffprobe", "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=codec_type", "-of", "default=noprint_wrappers=1:nokey=1",
                self.video_path
            ], capture_output=True, text=True, timeout=8)

            if result.returncode != 0 or not result.stdout.strip():
                print("⚠️ Video has NO AUDIO STREAM → no waveform possible")
                self.statusBar().showMessage("Video has no audio track", 5000)
                return
            else:
                print("✓ Video contains audio stream")
        except Exception as e:
            print(f"⚠️ Could not check audio stream: {e}")

        # Start extraction in background
        import threading

        def extract_waveform():
            print("🎵 [thread] Starting waveform extraction...")
            visualizer = WaveformVisualizer(self.video_path)
            data = visualizer.extract_waveform(num_points=2000)

            if data is None:
                print("❌ [thread] extract_waveform() returned None")
                self.waveform_ready.emit(None)
            else:
                print(f"✅ [thread] extract_waveform() returned list len={len(data)} first={data[0] if data else None}")
                self.waveform_ready.emit(data)

            def apply():
                print("🧵 [ui] apply() called")
                if data is None:
                    self.statusBar().showMessage("Failed to extract waveform (None)", 6000)
                else:
                    self.update_waveform_data(data)

            QTimer.singleShot(0, apply)

        thread = threading.Thread(target=extract_waveform, daemon=True)
        thread.start()

    def update_waveform_data(self, waveform_data):
        print(f"🧩 update_waveform_data() called with {len(waveform_data) if waveform_data else 0} points")
        
        if not waveform_data or len(waveform_data) == 0:
            print("❌ No waveform data received → skipping update")
            return
        
        print(f"✅ update_waveform_data received: {len(waveform_data)} points")
        
        self.waveform = waveform_data
        self.save_waveform_to_cache(waveform_data)
        
        if hasattr(self, 'signal_scene') and self.signal_scene is not None:
            # Update scene with new waveform data
            self.signal_scene.set_waveform_data(waveform_data)
            
            # Force checkbox update after scene is built
            QTimer.singleShot(100, lambda: self.update_waveform_checkbox_state())
            
            # Force a view update
            QTimer.singleShot(150, lambda: self.signal_view.viewport().update())
            
            self.statusBar().showMessage(
                f"✅ Waveform loaded ({len(waveform_data)} points)", 5000
            )
        else:
            print(f"Scene not ready yet, storing waveform data")
            self._pending_waveform_data = waveform_data


    def save_waveform_to_cache(self, waveform_data):
        """Save waveform to cache for future use"""
        try:
            if not self.cache_data:
                self.cache_data = {}
            
            if 'video_metadata' not in self.cache_data:
                self.cache_data['video_metadata'] = {}
            
            self.cache_data['video_metadata']['waveform'] = waveform_data
            print(f"💾 Saved waveform to cache ({len(waveform_data)} points)")
        except Exception as e:
            print(f"⚠️ Could not save waveform to cache: {e}")

    def set_waveform_data(self, waveform_data):
        self.waveform = waveform_data or []
        self.waveform_visible = True if self.waveform else False  # FORCE ON when data exists
        self.waveform_colors = self.generate_waveform_colors()
        self.build_timeline()

    def get_cache_instance(self):
        """Get cache instance for highlight loading"""
        print(f"\n🔍 [TIMELINE] get_cache_instance")
        try:
            from modules.video_cache import VideoAnalysisCache
            cache = VideoAnalysisCache()
            
            # List all cache files
            cache_dir = Path("./cache")
            if cache_dir.exists():
                cache_files = list(cache_dir.glob("*.cache.json"))
                print(f"  - Cache directory contains {len(cache_files)} cache files:")
                for f in cache_files:
                    size_kb = f.stat().st_size / 1024
                    print(f"    - {f.name} ({size_kb:.1f} KB)")
                    
                    # Try to peek inside
                    try:
                        with open(f, 'r') as fh:
                            data = json.load(fh)
                            print(f"      Keys: {data.keys()}")
                            if 'video_path' in data:
                                print(f"      Video: {data['video_path']}")
                    except:
                        print(f"      Could not read file")
            
            return cache
        except Exception as e:
            print(f"  ❌ Could not initialize cache: {e}")
            return None

    def load_cache_data(self):
        """Load cache data for the video with extensive debugging"""
        print(f"\n{'='*60}")
        print(f"🔍 [TIMELINE] load_cache_data START")
        print(f"{'='*60}")
        print(f"  - video_path: {self.video_path}")
        
        try:
            from modules.video_cache import VideoAnalysisCache
            cache = VideoAnalysisCache()
            print(f"  ✓ Created VideoAnalysisCache instance")
            
            # Get video hash for debugging
            video_hash = cache._get_video_hash(self.video_path)
            print(f"  - Video hash: {video_hash}")
            
            # List all cache files first
            cache_dir = Path("./cache")
            all_cache_files = list(cache_dir.glob("*.cache.json"))
            print(f"\n  📁 All cache files in directory ({len(all_cache_files)}):")
            for f in all_cache_files:
                size_kb = f.stat().st_size / 1024
                print(f"    - {f.name} ({size_kb:.1f} KB)")
            
            # Look for any cache file with this video hash (wildcard match)
            matching_files = list(cache_dir.glob(f"{video_hash}*.cache.json"))
            print(f"\n  🔍 Files matching video hash ({len(matching_files)}):")
            
            for cache_file in matching_files:
                print(f"    - {cache_file.name}")
                try:
                    # Try to load it directly
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)
                    
                    # Verify it's for this video
                    if cache_data.get("video_hash") == video_hash:
                        print(f"      ✓ Successfully loaded cache file")
                        print(f"      ✓ Contains keys: {list(cache_data.keys())}")
                        
                        # Check for motion data specifically
                        print(f"      - motion_events present: {'motion_events' in cache_data}")
                        print(f"      - motion_peaks present: {'motion_peaks' in cache_data}")
                        print(f"      - scenes present: {'scenes' in cache_data}")
                        
                        if 'motion_events' in cache_data:
                            print(f"      - motion_events count: {len(cache_data['motion_events'])}")
                        if 'motion_peaks' in cache_data:
                            print(f"      - motion_peaks count: {len(cache_data['motion_peaks'])}")
                        if 'scenes' in cache_data:
                            print(f"      - scenes count: {len(cache_data['scenes'])}")
                        
                        print(f"\n  ✅ Successfully loaded cache data from direct file read")
                        print(f"{'='*60}\n")
                        return cache_data
                except Exception as e:
                    print(f"      ✗ Failed to load: {e}")
                    continue
            
            # If we get here, try with default params as fallback
            print(f"\n  🔄 Attempting to load with default params...")
            default_params = {
                "analysis_cache_schema": "analysis_v2",
                "use_transcript": False,
                "transcript_model": "base",
                "search_keywords": [],
                "highlight_objects": [],
                "interesting_actions": [],
                "object_frame_skip": 10,
                "sample_rate": 5,
                "action_use_person_detection": True,
                "action_max_people": 2,
                "yolo_model_size": "n",
                "yolo_pt_path": "yolo11n.pt",
                "openvino_model_folder": "yolo11n_openvino_model/",
                "use_time_range": False,
                "range_start": 0,
                "range_end": None,
                "scene_threshold": 70.0,
                "motion_threshold": 100.0,
                "spike_factor": 1.2,
                "freeze_seconds": 4,
                "freeze_factor": 0.8,
            }
            
            cache_data = cache.load(self.video_path, params=default_params)
            if cache_data:
                print(f"  ✓ Found param-based cache")
                print(f"  ✓ Contains keys: {list(cache_data.keys())}")
                print(f"\n{'-'*40}")
                return cache_data
            
            # Try legacy load (no params)
            print(f"\n  🔄 Attempting legacy load (no params)...")
            cache_data = cache.load(self.video_path)
            if cache_data:
                print(f"  ✓ Found legacy cache")
                print(f"  ✓ Contains keys: {list(cache_data.keys())}")
                return cache_data
            
            print(f"\n  ⚠️ No cache found in any format - creating empty dict")
            
        except Exception as e:
            print(f"  ❌ Error in load_cache_data: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n{'='*60}\n")
        return {}
    
    def _extract_action_types(self):
        """Extract unique action names for info display"""
        actions = set()
        for item in self.cache_data.get('actions', []):
            name = item.get('action_name') or item.get('action') or 'Unknown'
            if isinstance(name, str):
                actions.add(name.strip().title())
        return sorted(list(actions))
    
    def _extract_object_classes(self):
        """Extract unique object classes for info display"""
        objs = set()
        for item in self.cache_data.get('objects', []):
            for obj in item.get('objects', []):
                if isinstance(obj, str):
                    objs.add(obj.strip().title())
        return sorted(list(objs))
    
    def init_ui(self):
        """Initialize the user interface with edit timeline"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)
        
        # Create info bar
        info_bar = self.create_info_bar()
        main_layout.addWidget(info_bar)
        
        # Create splitter for main content
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Create signal timeline view (top)
        signal_widget = QWidget()
        signal_layout = QVBoxLayout(signal_widget)
        
        # Always pass current waveform data (might be None initially)
        print(f"🎵 init_ui: Creating scene with waveform data ({len(self.waveform) if self.waveform else 0} points)")
        
        # Create scene with current waveform data (may be empty initially)
        self.signal_scene = SignalTimelineScene(self.cache_data, self.video_duration, waveform=self.waveform)
        self.signal_view = SignalTimelineView(self.signal_scene)
        
        # Enable drag and drop on the viewport
        self.signal_view.viewport().setAcceptDrops(True)
        
        # Connect signals
        self.signal_scene.time_clicked.connect(self.on_time_clicked)
        self.signal_scene.add_to_edit_requested.connect(self.on_add_to_edit_requested)
        self.signal_scene.filter_changed.connect(self.on_filter_changed)
        
        # Check if waveform clicked signal exists
        if hasattr(self.signal_scene, 'waveform_clicked'):
            self.signal_scene.waveform_clicked.connect(self.on_waveform_clicked)
        
        signal_layout.addWidget(QLabel("Signal Timeline (Drag items to edit timeline below)"))
        signal_layout.addWidget(self.signal_view)
        
        splitter.addWidget(signal_widget)
       
        # Create edit timeline view (bottom)
        edit_widget = QWidget()
        edit_layout = QVBoxLayout(edit_widget)
        
        # Edit timeline
        self.edit_scene = EditTimelineScene(self.video_path, self.video_duration, cache=self.cache)
        self.edit_view = QGraphicsView(self.edit_scene)
        self.edit_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.edit_view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.edit_view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.edit_view.setFixedHeight(120)
        self.edit_view.setAcceptDrops(True)
        self.edit_view.viewport().setAcceptDrops(True)
        self.edit_view.setStyleSheet("""
            QGraphicsView {
                background-color: rgba(30, 30, 40, 200);
                border: 2px solid rgba(100, 100, 150, 150);
                border-radius: 5px;
            }
        """)
        
        # --- LLM Chat Panel (in timeline) ---
        try:
            from llm.llm_chat_widget import LLMChatWidget
            self.llm_chat = LLMChatWidget(parent=self, compact=True, cache_dir="./cache")
            self.llm_chat.set_timeline_window(self)  # <-- THIS connects it!
            
            # If we have cache_data, feed it
            if self.cache_data:
                self.llm_chat.set_analysis_data(self.cache_data, self.video_path)
            
            # Add as a dock widget on the bottom
            llm_dock = QDockWidget("LLM Assistant", self)
            llm_dock.setWidget(self.llm_chat)
            self.addDockWidget(Qt.BottomDockWidgetArea, llm_dock)
        except ImportError:
            pass  # LLM modules not installed

        # Set focus policy to receive key events
        self.edit_view.setFocusPolicy(Qt.StrongFocus)
        
        # Connect edit timeline signals
        self.edit_scene.clip_double_clicked.connect(self.on_clip_double_clicked)
        self.edit_scene.clip_added.connect(self.on_clip_added)
        self.edit_scene.clip_removed.connect(self.on_clip_removed)
        self.edit_scene.time_clicked.connect(self.on_edit_time_clicked)
        self.edit_scene.clip_cut.connect(self.on_clip_cut)
        self.edit_scene.clip_trimmed.connect(self.on_clip_trimmed)

        
        edit_layout.addWidget(QLabel("Edit Timeline (Select clips and press Delete)"))
        edit_layout.addWidget(self.edit_view)
        
        # Add edit controls
        edit_controls = self.create_edit_controls()
        edit_layout.addWidget(edit_controls)
        
        splitter.addWidget(edit_widget)
        
        # Set splitter sizes (signal timeline gets more space)
        splitter.setSizes([700, 300])
        
        main_layout.addWidget(splitter)
        
        # Add controls panel
        controls_dock = self.create_controls_dock()
        self.addDockWidget(Qt.RightDockWidgetArea, controls_dock)

        # 🎬 ADD VIDEO PREVIEW DOCK
        try:
            preview_dock = self.create_video_preview_dock()
            self.addDockWidget(Qt.LeftDockWidgetArea, preview_dock)
        except Exception as e:
            print(f"⚠️ Could not create preview dock: {e}")
            # Continue without preview

        # Connect render signal
        self.render_finished.connect(self.on_render_finished)

        # Apply dark theme
        self.apply_dark_theme()
        
        # Status bar
        self.statusBar().showMessage(f"Video duration: {self.video_duration:.1f}s | Total edit duration: {self.edit_scene.get_total_duration():.1f}s")
        
        # Install event filter to handle global key events
        self.installEventFilter(self)

    def capture_current_frame_base64(self) -> str | None:
        """
        Capture current frame for LLM.
        
        Live mode:  scene.render() → video + bboxes = one image
        Other modes: cv2 grab from current source file + optional annotation
        """
        # ── Live mode: composited scene capture ──
        if (self.realtime_preview is not None
                and self.preview_stack.currentIndex() == 1):
            b64 = self.realtime_preview.capture_frame_base64()
            if b64:
                tag = " [live overlay]" if self.realtime_preview._overlay_enabled else ""
                print(f"📷 Captured frame at {self.current_time:.1f}s "
                      f"({len(b64) // 1024}KB){tag}")
                return b64

        # ── Precomp / Off mode: cv2 capture ──
        import cv2
        import base64

        try:
            # Determine which video file to read from
            source_path = self.video_path
            if (self.bbox_manager is not None
                    and hasattr(self.bbox_manager, '_sources')
                    and hasattr(self.bbox_manager, '_current_source')):
                source_path = self.bbox_manager._sources.get(
                    self.bbox_manager._current_source, self.video_path
                )

            cap = cv2.VideoCapture(source_path)
            cap.set(cv2.CAP_PROP_POS_MSEC, self.current_time * 1000)
            ret, frame = cap.read()
            cap.release()

            if not ret:
                print(f"❌ Could not read frame at {self.current_time:.1f}s")
                return None

            # Optionally annotate with cached action/object labels
            if (hasattr(self, 'annotate_llm_frames')
                    and self.annotate_llm_frames.isChecked()):
                frame = self._annotate_frame_for_llm(frame, self.current_time)

            # Resize
            h, w = frame.shape[:2]
            max_dim = 1024
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

            _, buffer = cv2.imencode(
                '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            b64 = base64.b64encode(buffer).decode('utf-8')

            tag = ""
            if source_path != self.video_path:
                tag = " [precomp annotated]"
            elif (hasattr(self, 'annotate_llm_frames')
                  and self.annotate_llm_frames.isChecked()):
                tag = " [cv2 annotated]"

            print(f"📷 Captured frame at {self.current_time:.1f}s "
                  f"({len(b64) // 1024}KB){tag}")
            return b64

        except Exception as e:
            print(f"❌ Frame capture failed: {e}")
            return None

    def _annotate_frame_for_llm(self, frame, timestamp, time_window=1.0):
        """Draw action/object labels onto frame before sending to LLM."""
        import cv2

        if not self.cache_data:
            return frame

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # ── Actions (top-left) ──
        actions = []
        for act in self.cache_data.get('actions', []):
            ts = act.get('timestamp', -999)
            if abs(ts - timestamp) > time_window:
                continue
            name = act.get('action_name') or act.get('action', '?')
            conf = act.get('confidence', 0)
            model = act.get('model_type', '')
            actions.append((name, conf, model))

        actions.sort(key=lambda x: x[1], reverse=True)

        for i, (name, conf, model) in enumerate(actions[:5]):
            y = 30 + i * 35
            tag = f" [{model}]" if model else ""
            text = f"{name}{tag} {conf:.0%}"

            if 'custom' in model:
                color = (0, 255, 0)
            elif 'cuda' in model or 'r3d' in model:
                color = (0, 128, 255)
            else:
                color = (0, 165, 255)

            # Semi-transparent confidence bar
            bar_w = int(min(w - 20, 350) * conf)
            overlay = annotated.copy()
            cv2.rectangle(overlay, (10, y - 18), (10 + bar_w, y + 10), color, -1)
            cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

            # Text
            cv2.putText(annotated, text, (14, y + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)  # outline
            cv2.putText(annotated, text, (14, y + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ── Objects (bottom-left) ──
        objects = []
        for obj_entry in self.cache_data.get('objects', []):
            ts = obj_entry.get('timestamp', -999)
            if abs(ts - timestamp) > time_window:
                continue
            for obj_name in obj_entry.get('objects', []):
                if isinstance(obj_name, str) and obj_name not in objects:
                    objects.append(obj_name)

        if objects:
            text = "Objects: " + ", ".join(objects[:6])
            overlay = annotated.copy()
            cv2.rectangle(overlay, (8, h - 40), (len(text) * 11 + 20, h - 8), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
            cv2.putText(annotated, text, (12, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        # ── Timestamp (top-right) ──
        mins, secs = divmod(int(timestamp), 60)
        cv2.putText(annotated, f"{mins:02d}:{secs:02d}", (w - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(annotated, f"{mins:02d}:{secs:02d}", (w - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return annotated

    def _annotate_frame_from_cache(self, frame, timestamp, time_window=1.0):
        """Draw action/object labels from cached analysis data onto frame."""
        import cv2

        if not self.cache_data:
            return frame

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # ── 1. Action labels (top-left) ──
        actions_at_time = []
        for act in self.cache_data.get('actions', []):
            act_ts = act.get('timestamp', -999)
            if abs(act_ts - timestamp) > time_window:
                continue
            name = act.get('action_name') or act.get('action', '?')
            conf = act.get('confidence', 0)
            model = act.get('model_type', '')
            actions_at_time.append((name, conf, model))

        actions_at_time.sort(key=lambda x: x[1], reverse=True)
        for i, (name, conf, model) in enumerate(actions_at_time[:5]):
            y_pos = 30 + i * 35
            tag = f" [{model}]" if model else ""
            text = f"{name}{tag} {conf:.0%}"

            # Color by model
            if 'custom' in model:
                color = (0, 255, 0)
            elif 'cuda' in model or 'r3d' in model:
                color = (0, 128, 255)
            else:
                color = (0, 165, 255)

            # Confidence bar background
            bar_width = int(min(w - 20, 350) * conf)
            overlay = annotated.copy()
            cv2.rectangle(overlay, (10, y_pos - 18), (10 + bar_width, y_pos + 8), color, -1)
            cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)

            cv2.putText(annotated, text, (14, y_pos + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ── 2. Object names (bottom-left) ──
        objects_at_time = []
        for obj_entry in self.cache_data.get('objects', []):
            obj_ts = obj_entry.get('timestamp', -999)
            if abs(obj_ts - timestamp) > time_window:
                continue
            for obj_name in obj_entry.get('objects', []):
                if isinstance(obj_name, str) and obj_name not in objects_at_time:
                    objects_at_time.append(obj_name)

        if objects_at_time:
            text = "Objects: " + ", ".join(objects_at_time[:6])
            overlay = annotated.copy()
            cv2.rectangle(overlay, (8, h - 38), (len(text) * 11 + 16, h - 8), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
            cv2.putText(annotated, text, (12, h - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        # ── 3. Timestamp ──
        mins, secs = divmod(int(timestamp), 60)
        cv2.putText(annotated, f"{mins:02d}:{secs:02d}", (w - 90, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return annotated

    def eventFilter(self, obj, event):
        """Global event filter for handling delete, spacebar, and cut mode exit"""
        if event.type() == event.Type.KeyPress:

            # Escape: exit cut mode
            if event.key() == Qt.Key_Escape:
                if hasattr(self, 'cut_mode_btn') and self.cut_mode_btn.isChecked():
                    self.cut_mode_btn.setChecked(False)  # triggers toggle_cut_mode(False)
                    return True

            if event.key() == Qt.Key_Space:
                if hasattr(self, '_edit_playlist') and self._edit_playlist:
                    self.toggle_edit_playback()
                else:
                    self.toggle_video_playback()
                return True

            if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                if (obj == self or
                    (hasattr(self, 'edit_view') and self.edit_view.hasFocus()) or
                    (hasattr(self, 'edit_scene') and len(self.edit_scene.selectedItems()) > 0)):
                    if hasattr(self, 'edit_scene'):
                        self.edit_scene.remove_selected_clips()
                        return True

        return super().eventFilter(obj, event)
    
    def create_info_bar(self):
        """Create information bar with video stats"""
        bar = QFrame()
        bar.setStyleSheet("""
            QFrame {
                background: #1e1e2c;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        
        layout = QHBoxLayout(bar)
        
        # Video info
        duration_mins = int(self.video_duration // 60)
        duration_secs = int(self.video_duration % 60)
        action_count = len(self.action_types)
        object_count = len(self.object_classes)
        
        info_text = f"Duration: {duration_mins:02d}:{duration_secs:02d} • Actions: {action_count} • Objects: {object_count}"
        info_label = QLabel(info_text)
        info_label.setStyleSheet("color: #c0d0ff; font-weight: bold;")
        
        layout.addWidget(info_label)
        layout.addStretch()
        
        # Drag and delete instructions
        instructions = QLabel(
            "🖱️ Drag signal bars → edit timeline  "
            "•  Left-drag background → highlight range, then drag range → edit timeline  "
            "•  Select clip + Delete to remove"
        )
        instructions.setStyleSheet("color: #a0ffa0; font-style: italic; font-size: 11px; padding: 4px; background: rgba(0, 100, 0, 40); border-radius: 4px;")
        layout.addWidget(instructions)
        layout.addStretch()
        
        # Current time display
        self.time_label = QLabel("No time selected")
        self.time_label.setStyleSheet("color: #ff8080; font-family: Consolas; font-weight: bold;")
        
        layout.addWidget(self.time_label)
        
        return bar
    
    def create_filter_controls(self):
        """Create filter controls for the dock widget"""
        filter_group = QGroupBox("Filters")
        filter_layout = QVBoxLayout()
        
        # Filter summary
        self.filter_summary = QLabel("All actions/objects visible")
        self.filter_summary.setStyleSheet("color: #a0ffa0; font-size: 11px;")
        filter_layout.addWidget(self.filter_summary)
        
        # Confidence filter display
        self.confidence_label = QLabel(f"Confidence: {self.signal_scene.min_confidence:.1f}-{self.signal_scene.max_confidence:.1f}")
        self.confidence_label.setStyleSheet("color: #ffa0a0; font-size: 11px;")
        filter_layout.addWidget(self.confidence_label)
        
        # Quick filter buttons
        quick_filter_layout = QHBoxLayout()
        
        show_all_btn = QPushButton("Show All")
        show_all_btn.clicked.connect(self.show_all_filters)
        show_all_btn.setToolTip("Show all actions and objects")
        
        hide_all_btn = QPushButton("Hide All")
        hide_all_btn.clicked.connect(self.hide_all_filters)
        hide_all_btn.setToolTip("Hide all actions and objects")
        
        quick_filter_layout.addWidget(show_all_btn)
        quick_filter_layout.addWidget(hide_all_btn)
        filter_layout.addLayout(quick_filter_layout)
        
        # Confidence filter button
        self.confidence_filter_btn = QPushButton("🎚️ Confidence Filter...")
        self.confidence_filter_btn.clicked.connect(self.open_confidence_filter)
        self.confidence_filter_btn.setStyleSheet("""
            QPushButton {
                background-color: #5a3fcd;
                font-weight: bold;
                padding: 8px;
            }
        """)
        filter_layout.addWidget(self.confidence_filter_btn)
        
        # Advanced filters button
        self.filter_dialog_btn = QPushButton("🎛️ Advanced Filters...")
        self.filter_dialog_btn.clicked.connect(self.open_filter_dialog)
        self.filter_dialog_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a5fcd;
                font-weight: bold;
                padding: 8px;
            }
        """)
        filter_layout.addWidget(self.filter_dialog_btn)
        
        # Current filters display
        self.current_filters_label = QLabel("")
        self.current_filters_label.setStyleSheet("color: #cccccc; font-size: 10px;")
        self.current_filters_label.setWordWrap(True)
        filter_layout.addWidget(self.current_filters_label)
        
        filter_group.setLayout(filter_layout)
        return filter_group
    
    def create_edit_controls(self):
        """Create controls for edit timeline"""
        controls = QWidget()
        layout = QHBoxLayout(controls)
        
        # Play Edited clip
        self.play_edit_btn = QPushButton("▶ Play Edit")
        self.play_edit_btn.clicked.connect(self.toggle_edit_playback)
        self.play_edit_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a6fcd;
                font-weight: bold;
                padding: 8px;
                min-width: 100px;
            }
        """)
        self.play_edit_btn.setToolTip("Play all clips in the edit timeline sequentially")
        layout.addWidget(self.play_edit_btn)
        
        self.stop_edit_btn = QPushButton("⏹ Stop")
        self.stop_edit_btn.clicked.connect(self.stop_edit_playback)
        self.stop_edit_btn.setStyleSheet("""
            QPushButton {
                background-color: #8a2a2a;
                font-weight: bold;
                padding: 8px;
            }
        """)
        layout.addWidget(self.stop_edit_btn)

        # Add clip button
        self.add_clip_btn = QPushButton("➕ Add Clip at Current Time")
        self.add_clip_btn.clicked.connect(self.on_add_clip_clicked)
        
        # Remove selected clips button
        self.remove_clips_btn = QPushButton("🗑️ Delete Selected Clips")
        self.remove_clips_btn.clicked.connect(self.on_remove_clips_clicked)
        
        # Add clip button
        self.add_clip_btn = QPushButton("➕ Add Clip at Current Time")
        self.add_clip_btn.clicked.connect(self.on_add_clip_clicked)

        # Remove selected clips button
        self.remove_clips_btn = QPushButton("🗑️ Delete Selected Clips")
        self.remove_clips_btn.clicked.connect(self.on_remove_clips_clicked)

        # Cut Mode toggle
        self.cut_mode_btn = QPushButton("✂️  Cut Mode")
        self.cut_mode_btn.setCheckable(True)
        self.cut_mode_btn.setToolTip(
            "Cut Mode ON:\n"
            "  • Left-click on a clip to cut it at that point\n"
            "  • Right-click for trim / cut menu\n"
            "  • Press C while hovering to cut at cursor\n\n"
            "Cut Mode OFF: normal drag/select behaviour"
        )
        self.cut_mode_btn.toggled.connect(self.toggle_cut_mode)
        self.cut_mode_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a44;
                color: #d0d8ff;
                font-weight: bold;
                padding: 8px 12px;
                border: 1px solid #4a4a6a;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #3a3a5c;
            }
            QPushButton:checked {
                background-color: #7a2a1a;
                border: 2px solid #ff6040;
                color: #ffccaa;
            }
            QPushButton:checked:hover {
                background-color: #8a3a2a;
            }
        """)

        # Save to cache button - ADD THIS
        self.save_cache_btn = QPushButton("💾 Save to Cache")
        self.save_cache_btn.clicked.connect(self.on_save_cache_clicked)
        self.save_cache_btn.setToolTip("Save current edit timeline to cache for future use")
        
        # Export button
        self.export_btn = QPushButton("📤 Export Edit")
        self.export_btn.clicked.connect(self.on_export_clicked)
        
        # Duration label
        self.edit_duration_label = QLabel("Edit duration: 0.0s")
        self.edit_duration_label.setStyleSheet("color: #a0ffa0; font-weight: bold;")
        
        layout.addWidget(self.add_clip_btn)
        layout.addWidget(self.remove_clips_btn)
        layout.addWidget(self.cut_mode_btn)
        layout.addWidget(self.save_cache_btn)
        layout.addWidget(self.export_btn)
        
        self.render_highlight_btn = QPushButton("🎬 Render Highlight Video")
        self.render_highlight_btn.clicked.connect(self.on_render_highlight_clicked)
        self.render_highlight_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a7a2a;
                font-weight: bold;
                padding: 8px;
            }
        """)
        self.render_highlight_btn.setToolTip("Render edit timeline clips into a single highlight video file")
        layout.addWidget(self.render_highlight_btn)
        
        layout.addStretch()

        layout.addWidget(self.edit_duration_label)
        
        return controls

    def create_controls_dock(self):
        """Create dock widget with controls including filters"""
        dock = QDockWidget("Controls", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        controls_widget = QWidget()
        layout = QVBoxLayout(controls_widget)
        
        # Waveform controls
        waveform_group = QGroupBox("Waveform")
        waveform_layout = QVBoxLayout()

        # Waveform visibility toggle
        self.waveform_checkbox = QCheckBox("Show Waveform")

        # Initialize checkbox state
        self.waveform_checkbox.setChecked(False)  # Start unchecked
        self.waveform_checkbox.setEnabled(False)  # Disabled until data loads

        # Connect signal
        self.waveform_checkbox.stateChanged.connect(self.toggle_waveform)
        waveform_layout.addWidget(self.waveform_checkbox)

        # Waveform opacity slider
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Opacity:"))
        self.waveform_opacity_slider = QSlider(Qt.Horizontal)
        self.waveform_opacity_slider.setRange(30, 100)
        self.waveform_opacity_slider.setValue(70)
        self.waveform_opacity_slider.valueChanged.connect(self.change_waveform_opacity)
        opacity_layout.addWidget(self.waveform_opacity_slider)
        waveform_layout.addLayout(opacity_layout)
        
        waveform_group.setLayout(waveform_layout)
        layout.addWidget(waveform_group)
        
        # ADD FILTER CONTROLS
        filter_controls = self.create_filter_controls()
        layout.addWidget(filter_controls)
        
        # Layer visibility controls
        layer_group = QGroupBox("Visible Layers")
        layer_layout = QVBoxLayout()
        
        self.layer_checkboxes = {}
        for layer_name in self.signal_scene.visible_layers.keys():
            display_name = layer_name.replace('_', ' ').title()
            checkbox = QCheckBox(display_name)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(
                lambda state, name=layer_name: self.toggle_layer(name, state)
            )
            layer_layout.addWidget(checkbox)
            self.layer_checkboxes[layer_name] = checkbox
        
        layer_group.setLayout(layer_layout)
        layout.addWidget(layer_group)
        
        # Zoom controls
        zoom_group = QGroupBox("Zoom")
        zoom_layout = QHBoxLayout()
        
        zoom_slider = QSlider(Qt.Orientation.Horizontal)
        zoom_slider.setMinimum(10)
        zoom_slider.setMaximum(200)
        zoom_slider.setValue(int(self.signal_scene.pixels_per_second))
        zoom_slider.valueChanged.connect(self.on_zoom_changed)
        
        zoom_layout.addWidget(QLabel("Zoom:"))
        zoom_layout.addWidget(zoom_slider)
        
        zoom_group.setLayout(zoom_layout)
        layout.addWidget(zoom_group)
        
        # Playback controls
        playback_group = QGroupBox("Playback")
        playback_layout = QVBoxLayout()
        
        play_btn = QPushButton("▶ Play at Selected Time")
        play_btn.clicked.connect(self.play_video_at_current_time)
        play_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a5fcd;
                font-weight: bold;
            }
        """)
        playback_layout.addWidget(play_btn)

        self.annotate_llm_frames = QCheckBox("Annotate LLM frames (labels + confidence)")
        self.annotate_llm_frames.setChecked(False)
        self.annotate_llm_frames.setToolTip("Draw action/object labels on frames before sending to LLM vision")
        playback_layout.addWidget(self.annotate_llm_frames)

        self.follow_playhead_checkbox = QCheckBox("Follow Playhead")
        self.follow_playhead_checkbox.setChecked(True)
        self.follow_playhead_checkbox.setToolTip(
            "Auto-scroll the timeline to keep the playhead visible during playback"
        )
        self.follow_playhead_checkbox.stateChanged.connect(self.toggle_follow_playhead)
        playback_layout.addWidget(self.follow_playhead_checkbox)
        
        playback_group.setLayout(playback_layout)

        layout.addWidget(playback_group)     
        layout.addStretch()
        
        dock.setWidget(controls_widget)
        return dock
    
    def open_confidence_filter(self):
        """Open the confidence filter dialog"""
        if not hasattr(self, 'confidence_dialog'):
            self.confidence_dialog = ConfidenceFilterDialog(self.signal_scene, self)
            self.confidence_dialog.finished.connect(self.on_confidence_filter_closed)
        
        self.confidence_dialog.show()
        self.confidence_dialog.raise_()
        self.confidence_dialog.activateWindow()
    
    def on_confidence_filter_closed(self):
        """Update filter summary when confidence dialog closes"""
        self.update_filter_summary()
    
    def update_filter_summary(self):
        """Update the filter summary display"""
        if hasattr(self, 'signal_scene'):
            visible_actions = self.signal_scene.get_filtered_actions()
            visible_objects = self.signal_scene.get_filtered_objects()
            
            total_actions = len(self.signal_scene.action_types)
            total_objects = len(self.signal_scene.object_classes)
            
            action_text = f"{len(visible_actions)}/{total_actions} actions"
            object_text = f"{len(visible_objects)}/{total_objects} objects"
            
            self.filter_summary.setText(f"Showing: {action_text}, {object_text}")
            self.confidence_label.setText(f"Confidence: {self.signal_scene.min_confidence:.1f}-{self.signal_scene.max_confidence:.1f}")
            
            # Show which specific filters are active
            filter_details = []
            
            if self.signal_scene.min_confidence > 0 or self.signal_scene.max_confidence < 1:
                filter_details.append(f"Confidence: {self.signal_scene.min_confidence:.1f}-{self.signal_scene.max_confidence:.1f}")
            
            if len(visible_actions) < total_actions:
                if len(visible_actions) <= 3:
                    filter_details.append(f"Actions: {', '.join(visible_actions)}")
                else:
                    filter_details.append(f"Actions: {len(visible_actions)} shown")
            
            if len(visible_objects) < total_objects:
                if len(visible_objects) <= 3:
                    filter_details.append(f"Objects: {', '.join(visible_objects)}")
                else:
                    filter_details.append(f"Objects: {len(visible_objects)} shown")
            
            if filter_details:
                self.current_filters_label.setText(" | ".join(filter_details))
            else:
                self.current_filters_label.setText("No filters applied")

    def get_highlights_from_signal_data(self):
        """Extract highlights from signal timeline cache data"""
        # This would require access to the main window's cache data
        # For now, we'll check if parent has cache_data
        highlights = []
        
        try:
            # Try to get parent window
            parent = self.parent()
            while parent and not hasattr(parent, 'cache_data'):
                parent = parent.parent()
            
            if parent and hasattr(parent, 'cache_data'):
                cache_data = parent.cache_data
                
                # Look for highlight segments in cache data
                if 'final_segments' in cache_data:
                    for segment in cache_data['final_segments']:
                        if isinstance(segment, (list, tuple)) and len(segment) >= 2:
                            start, end = segment[0], segment[1]
                            if end > start:  # Valid duration
                                highlights.append((start, end))
                
                # Also check for segments under analysis data
                elif 'analysis' in cache_data and 'final_segments' in cache_data['analysis']:
                    for segment in cache_data['analysis']['final_segments']:
                        if isinstance(segment, (list, tuple)) and len(segment) >= 2:
                            start, end = segment[0], segment[1]
                            if end > start:
                                highlights.append((start, end))
        except Exception as e:
            print(f"⚠️ Error extracting highlights from signal data: {e}")
        
        return highlights

    @Slot(int)
    def toggle_follow_playhead(self, state):
        """Toggle whether the timeline auto-scrolls to follow the playhead"""
        follow = (state == Qt.Checked)
        if hasattr(self, 'signal_view'):
            self.signal_view.follow_playhead = follow
        self.statusBar().showMessage(f"Follow playhead: {'ON' if follow else 'OFF'}", 2000)

    @Slot(int)
    def toggle_waveform(self, state):
        """Toggle waveform visibility"""
        # Prevent multiple rapid toggles
        if not hasattr(self, 'signal_scene'):
            return
        
        visible = bool(state == Qt.Checked)
        
        # Check if we actually have waveform data
        has_waveform = self.signal_scene.waveform is not None and len(self.signal_scene.waveform) > 0
        
        if visible and not has_waveform:
            # No data available, revert checkbox
            self.waveform_checkbox.setChecked(False)
            self.statusBar().showMessage("⚠️ No waveform data available", 3000)
            return
        
        # Store current view transform for restoration
        view_transform = self.signal_view.transform()
        
        # Store current scroll position
        h_scroll = self.signal_view.horizontalScrollBar().value()
        v_scroll = self.signal_view.verticalScrollBar().value()
        
        # Store current scene position under cursor
        cursor_pos = self.signal_view.mapFromGlobal(QCursor.pos())
        cursor_scene_pos = self.signal_view.mapToScene(cursor_pos)
        
        # Set visibility
        self.signal_scene.waveform_visible = visible
        
        # FORCE complete rebuild with proper dimensions
        self.signal_scene.build_timeline()
        
        # Calculate the scaling factor needed to maintain the same horizontal zoom
        old_width = self.signal_scene.sceneRect().width() / view_transform.m11() if view_transform.m11() != 0 else 1
        new_width = self.signal_scene.sceneRect().width()
        
        # Apply scaling to maintain horizontal zoom
        if old_width > 0 and new_width > 0:
            # Calculate how much we need to scale to maintain the same visible width
            scale_factor = new_width / old_width
            
            # Create a new transform with adjusted scaling
            new_transform = view_transform.scale(scale_factor, 1.0)
            self.signal_view.setTransform(new_transform)
        
        # Restore scroll positions
        self.signal_view.horizontalScrollBar().setValue(h_scroll)
        self.signal_view.verticalScrollBar().setValue(v_scroll)
        
        # Adjust view to maintain cursor position if possible
        if cursor_scene_pos.x() >= 0 and cursor_scene_pos.x() <= self.signal_scene.sceneRect().width():
            # Calculate how much the scene shifted
            scene_shift = cursor_scene_pos.y() - self.signal_view.mapToScene(cursor_pos).y()
            if abs(scene_shift) > 10:  # Significant shift
                self.signal_view.verticalScrollBar().setValue(
                    v_scroll + int(scene_shift * self.signal_view.transform().m22())
                )
        
        if visible:
            self.statusBar().showMessage(f"✅ Waveform visible ({len(self.signal_scene.waveform)} points)", 2000)
        else:
            self.statusBar().showMessage("Waveform hidden", 2000)
        
    @Slot(int)
    def change_waveform_opacity(self, value):
        """Change waveform opacity"""
        if hasattr(self, 'signal_scene'):
            self.signal_scene.waveform_opacity = value / 100.0
            self.signal_scene.waveform_colors = self.signal_scene.generate_waveform_colors()
            self.signal_scene.build_timeline()

    @Slot()
    def on_save_cache_clicked(self):
        """Save current edit timeline to cache"""
        if hasattr(self, 'edit_scene'):
            # Try to save using the cache system
            try:
                if hasattr(self.edit_scene, 'save_clips_to_cache'):
                    success = self.edit_scene.save_clips_to_cache()
                    if success:
                        self.statusBar().showMessage("✅ Edit timeline saved to cache", 3000)
                    else:
                        self.statusBar().showMessage("⚠️ Failed to save to cache", 3000)
                else:
                    self.statusBar().showMessage("⚠️ Cache saving not available in this scene", 3000)
            except Exception as e:
                self.statusBar().showMessage(f"⚠️ Error saving to cache: {str(e)[:50]}...", 3000)
        else:
            self.statusBar().showMessage("⚠️ No edit timeline available", 3000)

    @Slot(str, int)
    def toggle_layer(self, layer_name, state):
        """Toggle visibility of a layer"""
        self.signal_scene.visible_layers[layer_name] = (state == Qt.CheckState.Checked.value)
        self.signal_scene.build_timeline()
    
    @Slot(int)
    def on_zoom_changed(self, value):
        """Handle zoom slider changes"""
        self.signal_scene.set_zoom(value)
    
    @Slot(float)
    def on_time_clicked(self, time):
        """Handle timeline click"""
        # Stop edit playback if running
        if hasattr(self, '_edit_playlist') and self._edit_playlist:
            self.stop_edit_playback()
        
        self.current_time = max(0, min(self.video_duration, time))
        
        # Update signal scene
        self.signal_scene.current_time_seconds = self.current_time
        self.signal_scene.set_current_time(self.current_time)
        if hasattr(self, 'signal_view'):
            self.signal_view.ensure_time_visible(self.current_time)
           
        # Seek video player to this time
        if hasattr(self, 'video_player'):
            milliseconds = int(self.current_time * 1000)
            self._active_player.setPosition(milliseconds)
            if self.video_player.playbackState() != QMediaPlayer.PlayingState:
                self._active_player.play()
                QTimer.singleShot(50, self._active_player.pause)
        
        # Update label
        minutes = int(self.current_time // 60)
        seconds = int(self.current_time % 60)
        milliseconds = int((self.current_time % 1) * 1000)
        self.time_label.setText(f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}")

    @Slot(float, float, float)
    def on_waveform_clicked(self, start_time, end_time, amplitude):
        """Handle waveform clicks - auto-create a clip"""
        print(f"🎵 Waveform clicked at {start_time:.2f}s, amplitude: {amplitude:.2f}")
        # Option A: Increase threshold so only very loud sections add clips
        if amplitude > 0.8:  # Much higher threshold
            # Add to edit timeline
            if hasattr(self, 'edit_scene'):
                self.edit_scene.add_clip(start_time, end_time)
                self.update_edit_duration()
                self.statusBar().showMessage(f"Added audio clip: {start_time:.1f}s to {end_time:.1f}s", 2000)
        
        # Option B: Remove auto-add entirely, just seek
        # Just seek to the clicked time without adding clip
        self.current_time = start_time
        self.signal_scene.set_current_time(start_time)
    
    @Slot(float)
    def on_add_to_edit_requested(self, time):
        """Handle request to add region to edit timeline"""
        # Find a signal region around this time
        start, end = self.find_signal_region_around(time)
        self.edit_scene.add_clip_from_selection(start, end)
        self.update_edit_duration()
        
        self.statusBar().showMessage(f"Added clip: {start:.1f}s to {end:.1f}s", 2000)
    
    @Slot(float)
    def on_edit_time_clicked(self, time):
        """Handle click on edit timeline — seek to source time"""
        self.current_time = max(0, min(self.video_duration, time))
        
        # Update signal timeline playhead
        self.signal_scene.current_time_seconds = self.current_time
        self.signal_scene.set_current_time(self.current_time)
        if hasattr(self, 'signal_view'):
            self.signal_view.ensure_time_visible(self.current_time)
        
        # Seek video player
        if hasattr(self, 'video_player'):
            self.video_player.setPosition(int(self.current_time * 1000))
        
        # If playing — update which clip we're in and let timer handle progress
        is_playing = (hasattr(self, '_edit_paused') and not self._edit_paused 
                      and hasattr(self, '_edit_playlist') and self._edit_playlist)
        
        for i, (start, end) in enumerate(self.edit_scene.clips):
            if start <= self.current_time <= end:
                if is_playing:
                    # During playback: update playlist index, let timer do progress
                    self._edit_playlist_index = i + 1
                    self.edit_scene.set_active_clip(i)
                else:
                    # When paused: set both clip and progress from click
                    self.edit_scene.set_active_clip(i)
                    progress = (self.current_time - start) / (end - start) if end > start else 0
                    self.edit_scene.set_active_progress(progress)
                break
        else:
            self.edit_scene.clear_active_clip()
        
        # Update time label
        minutes = int(self.current_time // 60)
        seconds = int(self.current_time % 60)
        milliseconds = int((self.current_time % 1) * 1000)
        self.time_label.setText(f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}")

    @Slot(float, float)
    def on_clip_double_clicked(self, start_time, end_time):
        """Handle double-click on edit clip - play it"""
        # Play the clip (play from start_time for duration)
        self.current_time = start_time
        self.signal_scene.set_current_time(start_time)
        
        # Update time label
        minutes = int(start_time // 60)
        seconds = int(start_time % 60)
        self.time_label.setText(f"Clip: {minutes:02d}:{seconds:02d}")
        
        # Play the clip
        self.play_video_clip(start_time, end_time)

    @Slot(float, float)
    def on_clip_added(self, start_time: float, end_time: float):
        """Handle when a clip is added to edit timeline"""
        self.update_edit_duration()
        self.statusBar().showMessage(
            f"✅  Added clip  {start_time:.2f}s → {end_time:.2f}s  "
            f"({end_time - start_time:.2f}s)",
            3000
        )
        # Flash the newly added clip
        items = self.edit_scene.clip_items
        if items:
            last = items[-1]
            original_pen = last.pen()
            last.setPen(QPen(QColor(100, 230, 255), 3))
            QTimer.singleShot(400, lambda: self._safe_restore_pen(last, original_pen))

    def _safe_restore_pen(self, item, pen):
        """Restore a clip item's pen safely (item may have been deleted)."""
        try:
            item.setPen(pen)
        except RuntimeError:
            pass
    
    @Slot(int)
    def on_clip_removed(self, index):
        """Handle when a clip is removed from edit timeline"""
        # Add to pending removals
        self.pending_clip_removals.append(index)
        
        # Start or restart the timer
        self.removal_timer.start(100)  # 100ms delay

    def toggle_cut_mode(self, active: bool):
        """
        Enable or disable cut mode on the edit timeline.

        While cut mode is active:
          - The edit view shows a CrossCursor
          - Left-clicking a clip cuts it at the click position
          - A red dashed line follows the mouse on clips
          - The C key cuts at the current hover position
        """
        if not hasattr(self, 'edit_scene'):
            return

        self.edit_scene.cut_mode = active

        if active:
            self.edit_view.setCursor(QCursor(Qt.CrossCursor))
            self.statusBar().showMessage(
                "✂️  Cut Mode ON — left-click a clip to cut it  |  C key = cut at cursor  |  right-click for trim menu",
                0  # 0 = stays until next message
            )
        else:
            self.edit_view.setCursor(QCursor(Qt.ArrowCursor))
            # Make sure no stale indicator line remains
            self.edit_scene._hide_cut_indicator()
            self.statusBar().showMessage("Cut Mode OFF", 3000)

    @Slot(float)
    def on_clip_cut(self, cut_time: float):
        """
        Called after a successful cut.  Updates duration display and
        shows a status bar message with the cut timestamp.
        """
        self.update_edit_duration()

        minutes = int(cut_time // 60)
        seconds = cut_time % 60
        self.statusBar().showMessage(
            f"✂️  Cut at {minutes:02d}:{seconds:05.2f}  —  "
            f"{len(self.edit_scene.clips)} clips in timeline",
            4000
        )

    @Slot(int)
    def on_clip_trimmed(self, clip_index: int):
        """
        Called after a trim operation.  Updates duration display and
        shows a brief status bar message.
        """
        self.update_edit_duration()

        if 0 <= clip_index < len(self.edit_scene.clips):
            start, end = self.edit_scene.clips[clip_index]
            duration = end - start
            self.statusBar().showMessage(
                f"Trimmed clip {clip_index + 1}  →  {start:.2f}s – {end:.2f}s  ({duration:.1f}s)",
                3000
            )
        else:
            self.statusBar().showMessage("Clip trimmed", 2000)

    def process_pending_removals(self):
        """Process multiple clip removals at once"""
        if not self.pending_clip_removals:
            return
        
        # Update duration
        self.update_edit_duration()
        
        # Show status message
        count = len(self.pending_clip_removals)
        if count == 1:
            self.statusBar().showMessage(f"Removed clip {self.pending_clip_removals[0] + 1}", 2000)
        else:
            self.statusBar().showMessage(f"Removed {count} clips", 2000)
        
        # Clear pending removals
        self.pending_clip_removals.clear()

    @Slot()
    def on_add_clip_clicked(self):
        """Add a clip at current time"""
        if hasattr(self, 'current_time') and self.current_time >= 0:
            self.edit_scene.add_clip_from_selection(self.current_time)
            self.update_edit_duration()
            self.statusBar().showMessage(f"Added clip at {self.current_time:.1f}s", 2000)
        else:
            self.statusBar().showMessage("⚠️ Select a time first", 2000)
    
    @Slot()
    def on_remove_clips_clicked(self):
        """Remove selected clips (button click handler)"""
        if hasattr(self, 'edit_scene'):
            self.edit_scene.remove_selected_clips()
            self.update_edit_duration()
            self.statusBar().showMessage("Removed selected clips", 2000)
    
    @Slot()
    def on_export_clicked(self):
        """Export the edit timeline to EDL/XML for DaVinci Resolve"""
        if len(self.edit_scene.clips) == 0:
            QMessageBox.warning(self, "No Clips", "Add some clips to the edit timeline first!")
            return
              
        # Ask user for format
        formats = TimelineExporter.get_export_formats()
        
        # Create simple format selector
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Timeline")
        dialog.resize(400, 200)
        
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select export format:"))
        
        format_combo = QComboBox()
        for name, _ in formats:
            format_combo.addItem(name)
        layout.addWidget(format_combo)
        
        # Info label
        info = QLabel(f"Exporting {len(self.edit_scene.clips)} clips, "
                    f"total duration: {self.edit_scene.get_total_duration():.1f}s")
        info.setStyleSheet("color: #a0ffa0; padding: 8px; background: #1a2a1a; border-radius: 4px;")
        layout.addWidget(info)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec() == QDialog.Accepted:
            format_idx = format_combo.currentIndex()
            format_name, format_pattern = formats[format_idx]
            
            # Ask for save location
            from PySide6.QtWidgets import QFileDialog
            
            default_name = os.path.splitext(os.path.basename(self.video_path))[0] + "_edit"
            if format_name.startswith("EDL"):
                default_path = os.path.join(os.path.dirname(self.video_path), f"{default_name}.edl")
                filter_str = "EDL files (*.edl)"
            elif format_name.startswith("FCPXML"):
                default_path = os.path.join(os.path.dirname(self.video_path), f"{default_name}.xml")
                filter_str = "XML files (*.xml)"
            else:
                default_path = os.path.join(os.path.dirname(self.video_path), f"{default_name}.txt")
                filter_str = "All files (*.*)"
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Timeline", default_path, filter_str
            )
            
            if not file_path:
                return
            
            # Export
            try:
                if format_name.startswith("EDL"):
                    result = TimelineExporter.to_edl(self.edit_scene.clips, self.video_path, file_path)
                    msg = f"EDL exported to: {os.path.basename(result)}"
                elif format_name.startswith("FCPXML"):
                    result = TimelineExporter.to_fcp_xml(self.edit_scene.clips, self.video_path, file_path)
                    msg = f"FCPXML exported to: {os.path.basename(result)}"
                else:
                    # CSV fallback
                    import csv
                    with open(file_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['Clip', 'Start (s)', 'End (s)', 'Duration (s)'])
                        for i, (start, end) in enumerate(self.edit_scene.clips, 1):
                            writer.writerow([i, f"{start:.2f}", f"{end:.2f}", f"{end-start:.2f}"])
                    msg = f"CSV exported to: {os.path.basename(file_path)}"
                
                QMessageBox.information(self, "Export Successful", 
                                    f"✅ Timeline exported successfully!\n\n{msg}")
                
                # Optional: Open containing folder
                reply = QMessageBox.question(self, "Open Folder", 
                                            "Open containing folder?",
                                            QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    import subprocess
                    folder = os.path.dirname(file_path)
                    if sys.platform == 'win32':
                        os.startfile(folder)
                    elif sys.platform == 'darwin':
                        subprocess.run(['open', folder])
                    else:
                        subprocess.run(['xdg-open', folder])
                        
            except Exception as e:
                QMessageBox.critical(self, "Export Failed", 
                                    f"Failed to export timeline:\n{str(e)}")
    
    def update_edit_duration(self):
        """Update edit duration display"""
        total_duration = self.edit_scene.get_total_duration()
        self.edit_duration_label.setText(f"Edit duration: {total_duration:.1f}s")
        self.statusBar().showMessage(f"Video duration: {self.video_duration:.1f}s | Total edit duration: {total_duration:.1f}s")
    
    def find_signal_region_around(self, time):
        """Find meaningful region around clicked time, respecting filters"""
        start = max(0, time - 3)
        end = min(self.video_duration, time + 3)
        
        # Try to find a region with visible actions/objects
        visible_actions = self.signal_scene.get_filtered_actions()
        visible_objects = self.signal_scene.get_filtered_objects()
        
        # If we have filters active, try to find a region that contains them
        if visible_actions or visible_objects:
            # Look for action/object occurrences near this time
            best_start, best_end = start, end
            
            # Check for actions
            for action in self.cache_data.get('actions', []):
                action_name = action.get('action_name') or action.get('action') or 'Unknown'
                action_name = action_name.strip().title()
                timestamp = action.get('timestamp', 0)
                
                if action_name in visible_actions and abs(timestamp - time) < 5:
                    # Expand region to include this action
                    best_start = min(best_start, max(0, timestamp - 2))
                    best_end = max(best_end, min(self.video_duration, timestamp + 2))
            
            # Check for objects
            for obj_data in self.cache_data.get('objects', []):
                timestamp = obj_data.get('timestamp', 0)
                for obj_name in obj_data.get('objects', []):
                    if isinstance(obj_name, str):
                        obj_name = obj_name.strip().title()
                        if obj_name in visible_objects and abs(timestamp - time) < 5:
                            # Expand region to include this object
                            best_start = min(best_start, max(0, timestamp - 2))
                            best_end = max(best_end, min(self.video_duration, timestamp + 2))
            
            return best_start, best_end
        
        return start, end
    
    def open_filter_dialog(self):
        """Open the filter dialog"""
        if not hasattr(self, 'filter_dialog'):
            self.filter_dialog = FilterDialog(self.signal_scene, self)
            self.filter_dialog.finished.connect(self.on_filter_dialog_closed)
        
        self.filter_dialog.show()
        self.filter_dialog.raise_()
        self.filter_dialog.activateWindow()
    
    def on_filter_dialog_closed(self):
        """Update filter summary when dialog closes"""
        self.update_filter_summary()
    
    def show_all_filters(self):
        """Show all actions and objects with full confidence range"""
        if hasattr(self, 'signal_scene'):
            self.signal_scene.set_all_actions_visible(True)
            self.signal_scene.set_all_objects_visible(True)
            self.signal_scene.set_confidence_filter(0.0, 1.0)  # Reset confidence filter
            self.update_filter_summary()
    
    def hide_all_filters(self):
        """Hide all actions and objects"""
        if hasattr(self, 'signal_scene'):
            self.signal_scene.set_all_actions_visible(False)
            self.signal_scene.set_all_objects_visible(False)
            self.update_filter_summary()
    
    def update_filter_summary(self):
        """Update the filter summary display"""
        if hasattr(self, 'signal_scene'):
            visible_actions = self.signal_scene.get_filtered_actions()
            visible_objects = self.signal_scene.get_filtered_objects()
            
            total_actions = len(self.signal_scene.action_types)
            total_objects = len(self.signal_scene.object_classes)
            
            action_text = f"{len(visible_actions)}/{total_actions} actions"
            object_text = f"{len(visible_objects)}/{total_objects} objects"
            
            self.filter_summary.setText(f"Showing: {action_text}, {object_text}")
            
            # Show which specific filters are active
            if len(visible_actions) < total_actions or len(visible_objects) < total_objects:
                filter_details = []
                if len(visible_actions) < total_actions:
                    if len(visible_actions) <= 3:
                        filter_details.append(f"Actions: {', '.join(visible_actions)}")
                    else:
                        filter_details.append(f"Actions: {len(visible_actions)} shown")
                
                if len(visible_objects) < total_objects:
                    if len(visible_objects) <= 3:
                        filter_details.append(f"Objects: {', '.join(visible_objects)}")
                    else:
                        filter_details.append(f"Objects: {len(visible_objects)} shown")
                
                self.current_filters_label.setText(" | ".join(filter_details))
            else:
                self.current_filters_label.setText("No filters applied")
    
    @Slot(dict)
    def on_filter_changed(self, filters):
        """Handle filter changes from the scene"""
        self.update_filter_summary()
    
    def play_video_at_current_time(self):
        """Play video at current time position"""
        if not hasattr(self, 'current_time') or self.current_time < 0:
            self.statusBar().showMessage("⚠️ Click timeline to select a timestamp first", 2000)
            return
        
        milliseconds = int(self.current_time * 1000)
        self.video_player.setPosition(milliseconds)
        self.video_player.play()
        self.play_btn.setText("⏸ Pause")
        
        self.statusBar().showMessage(f"▶ Playing at {self.current_time:.1f}s", 2000)

    def play_edit_timeline(self):
        """Play all clips in the edit timeline sequentially"""
        clips = self.edit_scene.get_clip_times()
        if not clips:
            self.statusBar().showMessage("⚠️ No clips in edit timeline", 2000)
            return

        self._edit_paused = False
        self._edit_playlist = list(clips)
        self._edit_playlist_index = 0
        self.play_edit_btn.setText("⏸ Pause")
        self.statusBar().showMessage(f"▶ Playing edit timeline: {len(clips)} clips", 3000)
        self._play_next_edit_clip()

    def toggle_edit_playback(self):
        """Toggle play/pause for edit timeline"""
        if not hasattr(self, '_edit_playlist') or not self._edit_playlist:
            # Nothing playing — start fresh
            self.play_edit_timeline()
            return

        if hasattr(self, '_edit_paused') and self._edit_paused:
            # Resume
            self._edit_paused = False
            self.play_edit_btn.setText("⏸ Pause")
            self.video_player.play()
            
            # Restart progress timer
            if hasattr(self, '_edit_progress_timer'):
                self._edit_progress_timer.start(33)
            
            # Restart clip end timer with remaining time
            if hasattr(self, '_edit_remaining_ms') and self._edit_remaining_ms > 0:
                if hasattr(self, '_edit_clip_timer'):
                    self._edit_clip_timer.start(self._edit_remaining_ms)
            
            self.statusBar().showMessage("▶ Resumed", 2000)
        else:
            # Pause
            self._edit_paused = True
            self.play_edit_btn.setText("▶ Play Edit")
            self.video_player.pause()
            
            # Stop timers but remember remaining time
            if hasattr(self, '_edit_clip_timer') and self._edit_clip_timer.isActive():
                self._edit_remaining_ms = self._edit_clip_timer.remainingTime()
                self._edit_clip_timer.stop()
            
            if hasattr(self, '_edit_progress_timer'):
                self._edit_progress_timer.stop()
            
            self.statusBar().showMessage("⏸ Paused", 2000)

    def _play_next_edit_clip(self):
        """Play the next clip in the edit playlist"""
        if not hasattr(self, '_edit_playlist') or self._edit_playlist_index >= len(self._edit_playlist):
            self.statusBar().showMessage("✅ Edit timeline playback complete", 3000)
            self.video_player.pause()
            self.edit_scene.clear_active_clip()
            self._edit_playlist = []
            self._edit_paused = False
            self.play_edit_btn.setText("▶ Play Edit")
            if hasattr(self, '_edit_progress_timer'):
                self._edit_progress_timer.stop()
            return

        start, end = self._edit_playlist[self._edit_playlist_index]
        duration = end - start
        self._edit_playlist_index += 1

        clip_num = self._edit_playlist_index
        total = len(self._edit_playlist)
        self.statusBar().showMessage(f"▶ Clip {clip_num}/{total}: {start:.1f}s - {end:.1f}s", int(duration * 1000))

        # Seek and play
        self.current_time = start
        self.signal_scene.set_current_time(start)
        if hasattr(self, 'signal_view'):
            self.signal_view.ensure_time_visible(start)

        self.video_player.setPosition(int(start * 1000))
        self.video_player.play()

        # Highlight active clip in edit timeline
        self.edit_scene.set_active_clip(self._edit_playlist_index - 1)

        # Progress update timer (~30fps)
        if hasattr(self, '_edit_progress_timer'):
            self._edit_progress_timer.stop()
            self._edit_progress_timer.deleteLater()
        self._edit_progress_timer = QTimer()
        self._edit_progress_timer.timeout.connect(self._update_edit_progress)
        self._edit_progress_timer.start(33)

        # Timer to stop at clip end and play next
        if hasattr(self, '_edit_clip_timer'):
            self._edit_clip_timer.stop()
        self._edit_clip_timer = QTimer()
        self._edit_clip_timer.setSingleShot(True)
        self._edit_clip_timer.timeout.connect(self._play_next_edit_clip)
        self._edit_clip_timer.start(int(duration * 1000))

    def _update_edit_progress(self):
        """Update progress line in active edit clip"""
        if not hasattr(self, '_edit_playlist') or self._edit_playlist_index <= 0:
            return
        idx = self._edit_playlist_index - 1
        if idx >= len(self._edit_playlist):
            return
        start, end = self._edit_playlist[idx]
        duration = end - start
        if duration <= 0:
            return
        current = self.video_player.position() / 1000.0
        
        # Ignore updates until player has actually seeked to the clip
        if current < start - 0.5 or current > end + 0.5:
            return
        
        progress = max(0.0, min(1.0, (current - start) / duration))
        self.edit_scene.set_active_progress(progress)

    def stop_edit_playback(self):
        """Stop edit timeline playback"""
        if hasattr(self, '_edit_clip_timer'):
            self._edit_clip_timer.stop()
        if hasattr(self, '_edit_progress_timer'):
            self._edit_progress_timer.stop()
        self.edit_scene.clear_active_clip()
        self._edit_playlist = []
        self._edit_playlist_index = 0
        self._edit_paused = False
        self.play_edit_btn.setText("▶ Play Edit")
        self._active_player.pause()
        
        # Reset to beginning
        self.current_time = 0
        self._active_player.setPosition(0)
        self.signal_scene.set_current_time(0)
        if hasattr(self, 'signal_view'):
            self.signal_view.ensure_time_visible(0)
        self.time_label.setText("00:00.000")
        
        self.statusBar().showMessage("⏹ Edit playback stopped", 2000)

    def play_video_clip(self, start_time, end_time):
        """Play a specific clip in the preview"""
        duration = end_time - start_time
        self.statusBar().showMessage(f"Playing clip: {start_time:.1f}s for {duration:.1f}s", 3000)
        
        # Seek to start time
        milliseconds = int(start_time * 1000)
        self.video_player.setPosition(milliseconds)
        
        # Play the video
        self.video_player.play()
        self.play_btn.setText("⏸ Pause")
        
        # Set up timer to stop at end time
        if hasattr(self, 'clip_timer'):
            self.clip_timer.stop()
        
        self.clip_timer = QTimer()
        self.clip_timer.setSingleShot(True)
        self.clip_timer.timeout.connect(lambda: self.video_player.pause())
        self.clip_timer.start(int(duration * 1000))

    def play_video_time(self, time):
        """Play video at specific time in preview"""
        # This replaces the external player call
        self.current_time = time
        self.play_video_at_current_time()

    @Slot()
    def on_render_highlight_clicked(self):
        """Render edit timeline clips into a single highlight video"""
        clips = self.edit_scene.get_clip_times()
        if not clips:
            QMessageBox.warning(self, "No Clips", "Add some clips to the edit timeline first!")
            return

        from PySide6.QtWidgets import QFileDialog

        default_name = os.path.splitext(os.path.basename(self.video_path))[0] + "_highlight.mp4"
        default_path = os.path.join(os.path.dirname(self.video_path), default_name)

        output_path, _ = QFileDialog.getSaveFileName(
            self, "Save Highlight Video", default_path, "MP4 files (*.mp4);;All files (*.*)"
        )
        if not output_path:
            return

        self.statusBar().showMessage("🎬 Rendering highlight video...")
        self.render_highlight_btn.setEnabled(False)
        self.render_highlight_btn.setText("⏳ Rendering...")

        # Store for the callback
        self._render_output_path = output_path
        self._render_clips = clips

        import threading

        def render():
            try:
                filter_parts = []
                inputs = []

                for i, (start, end) in enumerate(clips):
                    duration = end - start
                    inputs.extend(["-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", self.video_path])
                    filter_parts.append(f"[{i}:v][{i}:a]")

                n = len(clips)
                filter_str = "".join(filter_parts) + f"concat=n={n}:v=1:a=1[outv][outa]"

                cmd = ["ffmpeg", "-y"] + inputs + [
                    "-filter_complex", filter_str,
                    "-map", "[outv]", "-map", "[outa]",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k",
                    output_path
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

                if result.returncode == 0 and os.path.exists(output_path):
                    size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    total_dur = sum(e - s for s, e in clips)
                    msg = (f"✅ Highlight video rendered!\n\n"
                           f"File: {os.path.basename(output_path)}\n"
                           f"Clips: {len(clips)}\n"
                           f"Duration: {total_dur:.1f}s\n"
                           f"Size: {size_mb:.1f} MB")
                    self.render_finished.emit(True, msg)
                else:
                    err = result.stderr[-500:] if result.stderr else "Unknown error"
                    self.render_finished.emit(False, f"FFmpeg error:\n{err}")

            except Exception as e:
                self.render_finished.emit(False, str(e))

        threading.Thread(target=render, daemon=True).start()

    @Slot(bool, str)
    def on_render_finished(self, success, message):
        """Handle render completion on the main thread"""
        self.render_highlight_btn.setEnabled(True)
        self.render_highlight_btn.setText("🎬 Render Highlight Video")

        if success:
            self.statusBar().showMessage("✅ Highlight rendered!", 5000)
            QMessageBox.information(self, "Render Complete", message)
        else:
            self.statusBar().showMessage("❌ Render failed", 5000)
            QMessageBox.critical(self, "Render Failed", message)

    def apply_dark_theme(self):
        """Apply modern dark theme"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0f0f18;
            }
            QGroupBox {
                color: #d0e0ff;
                border: 1px solid #3a3a50;
                border-radius: 6px;
                margin-top: 14px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QCheckBox {
                color: #e0e8ff;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 3px;
                border: 2px solid #4a4a6a;
            }
            QCheckBox::indicator:checked {
                background-color: #3a5fcd;
                border: 2px solid #5a7fdd;
            }
            QPushButton {
                background-color: #2a2a44;
                color: white;
                border: 1px solid #4a4a6a;
                padding: 8px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a3a5c;
            }
            QPushButton:pressed {
                background-color: #1a1a34;
            }
            QLabel {
                color: #d0d8ff;
            }
            QSlider::groove:horizontal {
                height: 8px;
                background: #3a3a5a;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3a5fcd;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QDockWidget {
                color: #d0e0ff;
                border: 1px solid #3a3a50;
                border-radius: 6px;
            }
            QDockWidget::title {
                background: #2a2a3a;
                padding: 6px;
                border-radius: 4px;
            }
            QStatusBar {
                color: #ffffff;
                background-color: rgba(40, 40, 50, 180);
            }
        """)


# Also write to a debug file
DEBUG_FILE = "timeline_debug.log"


def debug_log(msg):
    """Write debug message to both console and file"""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    full_msg = f"[{timestamp}] {msg}"
    
    # Use the original print function directly
    import builtins
    builtins.print(full_msg, flush=True)
    
    # Write to file
    with open(DEBUG_FILE, "a", encoding="utf-8") as f:
        f.write(full_msg + "\n")
        f.flush()

# Keep original print safe
original_print = print

# Now replace debug_log at the module level
print = debug_log

debug_log("="*60)
debug_log("🚀 TIMELINE VIEWER STARTING")
debug_log("="*60)
debug_log(f"Python version: {sys.version}")
debug_log(f"Current working directory: {os.getcwd()}")
debug_log(f"Script location: {__file__}")



def show_timeline_viewer(video_path, cache_data=None):
    """
    Launch the signal timeline viewer with edit timeline

    Args:
        video_path: Path to video file
        cache_data: Optional cache data dict (will load from cache if not provided)
    
    Returns:
        int: Application exit code
    """
    debug_log("="*60)
    debug_log(f"🎬 show_timeline_viewer called")
    debug_log(f"  - video_path: {video_path}")
    debug_log(f"  - cache_data provided: {cache_data is not None}")
    debug_log(f"  - video_path exists: {os.path.exists(video_path)}")
    
    app = QApplication.instance()
    if app is None:
        debug_log("  - Creating new QApplication")
        app = QApplication(sys.argv)
    else:
        debug_log("  - Using existing QApplication")
    
    debug_log("  🔵 ABOUT TO CREATE SignalTimelineWindow...")
    try:
        window = SignalTimelineWindow(video_path, cache_data)
        debug_log("  🟢 SignalTimelineWindow CREATED successfully")
    except Exception as e:
        debug_log(f"  ❌ ERROR creating SignalTimelineWindow: {e}")
        import traceback
        traceback.print_exc()
        return -1
    
    debug_log("  - Showing window...")
    window.show()
    
    debug_log("  - Entering event loop...")
    result = app.exec()
    debug_log(f"  - Event loop exited with code: {result}")
    
    return result

if __name__ == "__main__":
    # Test with a video file
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        show_timeline_viewer(video_path)
    else:
        print("Usage: python signal_timeline_viewer.py <video_path>")