"""
Video Preview Player with AI Overlay Toggle
- Standalone video player
- Syncs with timeline via signals
- Toggle between original and AI-annotated video
- Timeline sync controls
"""

import sys
import os
from typing import Optional
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QComboBox, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QCheckBox, QGroupBox,
    QFileDialog, QMessageBox, QSplitter
)
from PySide6.QtCore import Qt, QUrl, Signal, Slot, QTimer, QRect, QEvent
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtGui import QPainter, QColor, QFont, QPen, QBrush, QPixmap

class AnalysisOverlayWidget(QWidget):
    """Floating transparent overlay that draws labels on top of video.
    
    QVideoWidget uses native rendering that covers child widgets,
    so this is a top-level frameless window that tracks the video widget position.
    """
    
    def __init__(self, video_widget, parent=None):
        # Top-level frameless transparent window
        super().__init__(None, 
                         Qt.FramelessWindowHint | 
                         Qt.WindowStaysOnTopHint |
                         Qt.Tool)  # Tool = no taskbar entry
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        
        self.video_widget = video_widget
        self.cache_data = None
        self.current_time = 0.0
        self.time_window = 1.0
        self._visible = False
        
        # Track video widget movement/resize
        self.video_widget.installEventFilter(self)
        
        # Poll position as backup (some window managers miss move events)
        self._pos_timer = QTimer()
        self._pos_timer.timeout.connect(self._sync_geometry)
        self._pos_timer.start(100)  # 10fps position sync
        
    def set_cache_data(self, cache_data):
        self.cache_data = cache_data
        
    def set_time(self, seconds):
        self.current_time = seconds
        if self._visible:
            self.update()
    
    def set_overlay_visible(self, visible):
        self._visible = visible
        if visible:
            self._sync_geometry()
            self.show()
            self.raise_()
        else:
            self.hide()
        self.update()
    
    def eventFilter(self, obj, event):
        """Track video widget moves and resizes"""
        if obj == self.video_widget:
            if event.type() in (QEvent.Resize, QEvent.Move, QEvent.Show):
                self._sync_geometry()
        return False
    
    def _sync_geometry(self):
        """Match overlay geometry to video widget's screen position"""
        if not self.video_widget.isVisible():
            return
        try:
            global_pos = self.video_widget.mapToGlobal(self.video_widget.rect().topLeft())
            self.setGeometry(
                global_pos.x(), global_pos.y(),
                self.video_widget.width(), self.video_widget.height()
            )
        except RuntimeError:
            pass  # widget deleted
    
    def paintEvent(self, event):
        if not self._visible or not self.cache_data:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        if w < 10 or h < 10:
            painter.end()
            return
        
        # ── Actions (top-left) ──
        actions = []
        for act in self.cache_data.get('actions', []):
            ts = act.get('timestamp', -999)
            if abs(ts - self.current_time) > self.time_window:
                continue
            name = act.get('action_name') or act.get('action', '?')
            conf = act.get('confidence', 0)
            model = act.get('model_type', '')
            actions.append((name, conf, model))
        
        actions.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, conf, model) in enumerate(actions[:5]):
            y = 28 + i * 34
            tag = f" [{model}]" if model else ""
            text = f"{name}{tag} {conf:.0%}"
            
            if 'custom' in model:
                color = QColor(0, 255, 0, 220)
            elif 'cuda' in model or 'r3d' in model:
                color = QColor(0, 128, 255, 220)
            else:
                color = QColor(0, 165, 255, 220)
            
            # Confidence bar
            bar_w = int(min(w - 20, 320) * conf)
            painter.fillRect(10, y - 18, bar_w, 28, 
                           QColor(color.red(), color.green(), color.blue(), 70))
            
            # Text with outline for readability
            font = QFont("Arial", 11, QFont.Bold)
            painter.setFont(font)
            
            # Black outline
            painter.setPen(QPen(QColor(0, 0, 0, 200), 3))
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx or dy:
                        painter.drawText(14 + dx, y + 2 + dy, text)
            
            # Colored text
            painter.setPen(color)
            painter.drawText(14, y + 2, text)
        
        # ── Objects (bottom-left) ──
        objects = []
        for obj_entry in self.cache_data.get('objects', []):
            ts = obj_entry.get('timestamp', -999)
            if abs(ts - self.current_time) > self.time_window:
                continue
            for obj_name in obj_entry.get('objects', []):
                if isinstance(obj_name, str) and obj_name not in objects:
                    objects.append(obj_name)
        
        if objects:
            text = "Objects: " + ", ".join(objects[:6])
            painter.fillRect(8, h - 38, len(text) * 9 + 16, 30, QColor(0, 0, 0, 150))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.setPen(QColor(0, 255, 0, 220))
            painter.drawText(14, h - 16, text)
        
        # ── No data indicator ──
        if not actions and not objects:
            painter.setFont(QFont("Arial", 9))
            painter.setPen(QColor(255, 255, 100, 150))
            painter.drawText(10, h - 10, f"No detections at {self.current_time:.1f}s")
        
        # ── Timestamp (top-right) ──
        mins, secs = divmod(int(self.current_time), 60)
        ts_text = f"{mins:02d}:{secs:02d}"
        painter.setFont(QFont("Consolas", 13, QFont.Bold))
        painter.setPen(QColor(0, 0, 0, 200))
        painter.drawText(w - 78, 27, ts_text)
        painter.setPen(QColor(0, 255, 255, 230))
        painter.drawText(w - 80, 25, ts_text)
        
        painter.end()
    
    def cleanup(self):
        """Stop timers and hide"""
        self._pos_timer.stop()
        self.hide()

class VideoPreviewWindow(QMainWindow):
    """Standalone video preview window with AI overlay toggle"""
    
    # Signals to communicate with timeline
    time_changed = Signal(float)  # When user seeks in preview
    play_state_changed = Signal(bool)  # Play/pause state
    overlay_toggled = Signal(bool)  # AI overlay on/off
    
    def __init__(self, video_path, annotated_video_path=None, parent=None, cache_data=None):
        super().__init__(parent)
        self.video_path = video_path
        self.annotated_video_path = annotated_video_path
        self.cache_data = cache_data
        self.current_source = 'original'
        
        self.setWindowTitle(f"Video Preview - {os.path.basename(video_path)}")
        self.setGeometry(200, 200, 800, 600)
        
        self.init_ui()
        self.init_video_player()
        
    def init_ui(self):
        """Initialize the UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(640, 360)
        layout.addWidget(self.video_widget, 1)
        
        # ── ADD: Floating overlay for analysis labels ──
        self.analysis_overlay = AnalysisOverlayWidget(self.video_widget)
        if self.cache_data:
            self.analysis_overlay.set_cache_data(self.cache_data)
        
        controls = self.create_controls()
        layout.addWidget(controls)
        
        self.status_bar = self.statusBar()
        self.apply_dark_theme()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'analysis_overlay'):
            self.analysis_overlay.setGeometry(self.video_widget.geometry())

    def create_controls(self):
        """Create video controls"""
        controls = QWidget()
        layout = QVBoxLayout(controls)
        
        # Top row: Playback controls
        playback_layout = QHBoxLayout()
        
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedWidth(40)
        self.play_btn.clicked.connect(self.toggle_playback)
        
        self.stop_btn = QPushButton("⏹")
        self.stop_btn.setFixedWidth(40)
        self.stop_btn.clicked.connect(self.stop)
        
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(1000)
        self.time_slider.sliderMoved.connect(self.seek_to_position)
        
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setFixedWidth(120)
        
        playback_layout.addWidget(self.play_btn)
        playback_layout.addWidget(self.stop_btn)
        playback_layout.addWidget(self.time_slider)
        playback_layout.addWidget(self.time_label)
        
        # Middle row: AI overlay and sync controls
        feature_layout = QHBoxLayout()
        
        # AI overlay toggle
        self.overlay_checkbox = QCheckBox("Show AI Labels Overlay")
        self.overlay_checkbox.setEnabled(True)  # Always enabled — reads from cache
        self.overlay_checkbox.stateChanged.connect(self.toggle_overlay)
        
        # Sync with timeline checkbox
        self.sync_checkbox = QCheckBox("Sync with Timeline")
        self.sync_checkbox.setChecked(True)
        
        feature_layout.addWidget(self.overlay_checkbox)
        feature_layout.addStretch()
        feature_layout.addWidget(self.sync_checkbox)
        
        # Bottom row: Volume and other controls
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("Volume:"))
        
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.valueChanged.connect(self.set_volume)
        
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.5x", "0.75x", "Normal", "1.25x", "1.5x", "2.0x"])
        self.speed_combo.setCurrentText("Normal")
        self.speed_combo.currentTextChanged.connect(self.set_playback_speed)
        
        volume_layout.addWidget(self.volume_slider)
        volume_layout.addStretch()
        volume_layout.addWidget(QLabel("Speed:"))
        volume_layout.addWidget(self.speed_combo)
        
        # Assemble all controls
        layout.addLayout(playback_layout)
        layout.addLayout(feature_layout)
        layout.addLayout(volume_layout)
        
        return controls
        
    def init_video_player(self):
        """Initialize the media player"""
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.video_widget)
        
        # Set initial volume
        self.audio_output.setVolume(self.volume_slider.value() / 100.0)
        
        # Connect signals
        self.player.durationChanged.connect(self.update_duration)
        self.player.positionChanged.connect(self.update_position)
        self.player.playbackStateChanged.connect(self.update_play_button)
        
        # Load video
        self.load_video(self.video_path)
        
    def load_video(self, path):
        """Load a video file"""
        if os.path.exists(path):
            self.player.setSource(QUrl.fromLocalFile(path))
            self.status_bar.showMessage(f"Loaded: {os.path.basename(path)}", 3000)
        else:
            self.status_bar.showMessage(f"File not found: {path}", 5000)

    def show_frame_analysis_status(self, timestamp: float, contains_target: bool):
        """Show a temporary overlay indicating frame analysis result"""
        # Update status bar
        status = f"Frame at {int(timestamp)//60}:{int(timestamp)%60:02d} - {'✅ TARGET FOUND' if contains_target else '❌ No target'}"
        self.status_bar.showMessage(status, 1000)  # Show for 1 second
        
        # Optional: Change border color briefly to indicate analysis
        original_style = self.video_widget.styleSheet()
        color = "#4CAF50" if contains_target else "#f44336"
        self.video_widget.setStyleSheet(f"border: 3px solid {color};")
        
        # Reset border after a short delay
        QTimer.singleShot(500, lambda: self.video_widget.setStyleSheet(original_style))


    @Slot()
    def toggle_playback(self):
        """Toggle play/pause"""
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()
            
    @Slot()
    def stop(self):
        """Stop playback"""
        self.player.stop()
        
    @Slot(int)
    def seek_to_position(self, position):
        """Seek to a position in the video"""
        if self.player.duration() > 0:
            new_position = (position / 1000.0) * self.player.duration()
            self.player.setPosition(int(new_position))
            
            # Emit time change if syncing with timeline
            if self.sync_checkbox.isChecked():
                self.time_changed.emit(new_position / 1000.0)  # Convert to seconds
                
    @Slot(int)
    def set_volume(self, value):
        """Set volume level"""
        self.audio_output.setVolume(value / 100.0)
        
    @Slot(str)
    def set_playback_speed(self, speed_text):
        """Set playback speed"""
        speed_map = {
            "0.5x": 0.5,
            "0.75x": 0.75,
            "Normal": 1.0,
            "1.25x": 1.25,
            "1.5x": 1.5,
            "2.0x": 2.0
        }
        self.player.setPlaybackRate(speed_map.get(speed_text, 1.0))
        
    @Slot(int)
    def toggle_overlay(self, state):
        """Toggle analysis labels overlay"""
        if hasattr(self, 'analysis_overlay'):
            self.analysis_overlay.set_overlay_visible(state == Qt.Checked)
        self.overlay_toggled.emit(state == Qt.Checked)
        
    @Slot(int)
    def update_duration(self, duration):
        """Update duration display"""
        if duration > 0:
            self.time_slider.setMaximum(duration)
            total_secs = duration // 1000
            self.total_time = f"{total_secs // 60:02d}:{total_secs % 60:02d}"
            self.update_time_label(self.player.position())
            
    @Slot(int)
    def update_position(self, position):
        if self.player.duration() > 0:
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(position)
            self.time_slider.blockSignals(False)
            self.update_time_label(position)
            
            # ── Update overlay time ──
            if hasattr(self, 'analysis_overlay'):
                self.analysis_overlay.set_time(position / 1000.0)
            
            if (self.sync_checkbox.isChecked() and 
                self.player.playbackState() == QMediaPlayer.PlayingState):
                self.time_changed.emit(position / 1000.0)
                
    def update_time_label(self, position):
        """Update time label"""
        current_secs = position // 1000
        current_time = f"{current_secs // 60:02d}:{current_secs % 60:02d}"
        self.time_label.setText(f"{current_time} / {self.total_time}")
        
    @Slot(QMediaPlayer.PlaybackState)
    def update_play_button(self, state):
        """Update play button based on playback state"""
        if state == QMediaPlayer.PlayingState:
            self.play_btn.setText("⏸")
        else:
            self.play_btn.setText("▶")
            
        self.play_state_changed.emit(state == QMediaPlayer.PlayingState)
        
    # Public API methods for timeline to call
    def seek_to_time(self, seconds):
        if self.player.duration() > 0:
            milliseconds = int(seconds * 1000)
            was_playing = self.player.playbackState() == QMediaPlayer.PlayingState
            self.player.setPosition(milliseconds)
            
            # ── Update overlay ──
            if hasattr(self, 'analysis_overlay'):
                self.analysis_overlay.set_time(seconds)
            
            if not was_playing:
                self.force_frame_update()
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(milliseconds)
            self.time_slider.blockSignals(False)
            self.update_time_label(milliseconds)


    def _pause_if_not_playing(self, was_playing):
        """Helper to pause if we were originally paused"""
        if not was_playing and self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()

    def play_from_time(self, seconds):
        """Play from specific time"""
        self.seek_to_time(seconds)
        self.player.play()
        
    def set_sync_enabled(self, enabled):
        """Enable/disable sync with timeline"""
        self.sync_checkbox.setChecked(enabled)

    def capture_current_frame(self):
        """Capture the current frame as QImage (for AI analysis or display)"""
        if hasattr(self.video_widget, 'grab'):
            # This grabs the current video widget content
            pixmap = self.video_widget.grab()
            return pixmap.toImage()
        return None

    def force_frame_update(self):
        """Force the video widget to update its displayed frame"""
        if self.player.playbackState() != QMediaPlayer.PlayingState:
            # Store current position
            current_pos = self.player.position()
            
            # Method 1: Seek to current position + 1ms then back (forces frame update)
            if current_pos + 1 < self.player.duration():
                self.player.setPosition(current_pos + 1)
                
                # Use a single-shot timer to seek back after a very short delay
                QTimer.singleShot(10, lambda: self._restore_position(current_pos))
            else:
                # Near the end, seek slightly backward then forward
                self.player.setPosition(current_pos - 100)  # 100ms back
                QTimer.singleShot(10, lambda: self._restore_position(current_pos))

    def capture_current_frame_base64(self) -> Optional[str]:
        """Capture current frame and return as base64 string"""
        import base64
        from io import BytesIO
        
        # Ensure we're at the right position and frame is rendered
        self.force_frame_update()
        
        # Give a tiny moment for the frame to render
        from PySide6.QtCore import QCoreApplication
        QCoreApplication.processEvents()
        
        # Try multiple methods to capture the frame
        pixmap = None
        
        # Method 1: Try grabbing from video widget
        if hasattr(self.video_widget, 'grab'):
            pixmap = self.video_widget.grab()
        
        # Method 2: If that failed, try grabbing from the player's video output
        if (pixmap is None or pixmap.isNull()) and hasattr(self.player, 'videoSink'):
            video_sink = self.player.videoSink()
            if video_sink and hasattr(video_sink, 'videoFrame'):
                frame = video_sink.videoFrame()
                if frame and not frame.isNull():
                    # Convert QVideoFrame to QImage to QPixmap
                    image = frame.toImage()
                    if not image.isNull():
                        from PySide6.QtGui import QPixmap
                        pixmap = QPixmap.fromImage(image)
        
        if pixmap and not pixmap.isNull():
            # Convert to base64
            buffer = BytesIO()
            pixmap.save(buffer, 'JPEG', quality=85)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return None

    def _restore_position(self, target_position):
        """Restore to target position after forcing frame update"""
        self.player.setPosition(target_position)

    def _safe_pause(self, timer):
        """Safely pause the player"""
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        timer.deleteLater()

    def apply_dark_theme(self):
        """Apply dark theme styling"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2a;
            }
            QVideoWidget {
                background-color: black;
                border: 2px solid #3a3a5a;
            }
            QPushButton {
                background-color: #2a2a44;
                color: white;
                border: 1px solid #4a4a6a;
                padding: 6px;
                border-radius: 4px;
                min-width: 40px;
            }
            QPushButton:hover {
                background-color: #3a3a5c;
            }
            QCheckBox {
                color: #e0e8ff;
                padding: 4px;
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
            QLabel {
                color: #d0d8ff;
            }
            QComboBox {
                background-color: #2a2a44;
                color: white;
                border: 1px solid #4a4a6a;
                padding: 4px;
                border-radius: 4px;
            }
        """)


# Integration with your timeline viewer
class TimelineWithPreview:
    """Helper to connect timeline with preview window"""
    
    @staticmethod
    def launch_preview(timeline_window, chat_widget=None):
        video_path = timeline_window.video_path
        annotated_path = TimelineWithPreview.find_annotated_video(video_path, timeline_window.cache_data)
        
        # Pass cache_data to preview
        preview = VideoPreviewWindow(video_path, annotated_path, 
                                    cache_data=timeline_window.cache_data)
        
        # Connect signals
        timeline_window.signal_scene.time_clicked.connect(preview.seek_to_time)
        timeline_window.edit_scene.clip_double_clicked.connect(
            lambda start, end: preview.play_from_time(start)
        )
        preview.time_changed.connect(timeline_window.signal_scene.set_current_time)
        
        preview.show()
        if chat_widget:
            chat_widget.set_preview_window(preview)
        
        return preview

    @staticmethod
    def find_annotated_video(video_path, cache_data):
        """Find annotated video path from cache"""
        # This depends on your cache structure
        # Example: look for annotated video in cache
        video_dir = os.path.dirname(video_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Try common locations
        possible_paths = [
            os.path.join(video_dir, f"{video_name}_annotated.mp4"),
            os.path.join(video_dir, f"{video_name}_with_boxes.mp4"),
            os.path.join(video_dir, "annotated", f"{video_name}.mp4"),
        ]
        
        # Also check cache data
        if cache_data and 'annotated_video_path' in cache_data:
            possible_paths.insert(0, cache_data['annotated_video_path'])
            
        for path in possible_paths:
            if os.path.exists(path):
                return path

    def closeEvent(self, event):
        if hasattr(self, 'analysis_overlay'):
            self.analysis_overlay.cleanup()
        super().closeEvent(event)

        return None