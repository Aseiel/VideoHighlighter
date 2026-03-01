"""
Video Preview Player with AI Overlay Toggle
- Standalone video player
- Syncs with timeline via signals
- Toggle between original and AI-annotated video
- Timeline sync controls
"""

import sys
import os
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QComboBox, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QCheckBox, QGroupBox,
    QFileDialog, QMessageBox, QSplitter
)
from PySide6.QtCore import Qt, QUrl, Signal, Slot, QTimer
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtGui import QAction


class VideoPreviewWindow(QMainWindow):
    """Standalone video preview window with AI overlay toggle"""
    
    # Signals to communicate with timeline
    time_changed = Signal(float)  # When user seeks in preview
    play_state_changed = Signal(bool)  # Play/pause state
    overlay_toggled = Signal(bool)  # AI overlay on/off
    
    def __init__(self, video_path, annotated_video_path=None, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.annotated_video_path = annotated_video_path
        self.current_source = 'original'  # 'original' or 'annotated'
        
        self.setWindowTitle(f"Video Preview - {os.path.basename(video_path)}")
        self.setGeometry(200, 200, 800, 600)
        
        self.init_ui()
        self.init_video_player()
        
    def init_ui(self):
        """Initialize the UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Video display area
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(640, 360)
        layout.addWidget(self.video_widget, 1)
        
        # Controls panel
        controls = self.create_controls()
        layout.addWidget(controls)
        
        # Status bar
        self.status_bar = self.statusBar()
        
        # Apply dark theme
        self.apply_dark_theme()
        
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
        self.overlay_checkbox = QCheckBox("Show AI Bounding Boxes")
        self.overlay_checkbox.setEnabled(self.annotated_video_path is not None)
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
        """Toggle between original and AI-annotated video"""
        if state == Qt.Checked and self.annotated_video_path:
            self.load_video(self.annotated_video_path)
            self.current_source = 'annotated'
        else:
            self.load_video(self.video_path)
            self.current_source = 'original'
            
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
        """Update position display"""
        if self.player.duration() > 0:
            # Update slider
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(position)
            self.time_slider.blockSignals(False)
            
            # Update label
            self.update_time_label(position)
            
            # Emit time change if syncing
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
        """Seek to specific time (called from timeline or LLM) and ensure frame is displayed"""
        if self.player.duration() > 0:
            milliseconds = int(seconds * 1000)
            
            # Store current playback state
            was_playing = self.player.playbackState() == QMediaPlayer.PlayingState
            
            # Set position
            self.player.setPosition(milliseconds)
            
            # Force frame to be rendered
            if not was_playing:
                # If was paused, we need to briefly play to force frame update
                self.player.play()
                
                # Create a single-shot timer to pause after frame is rendered
                # Use a longer delay to ensure frame is actually rendered
                QTimer.singleShot(100, lambda: self._pause_if_not_playing(was_playing))
            
            # Update UI
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
            # Briefly play to force frame update
            self.player.play()
            
            # Create a timer to pause after a very short time
            # This ensures the frame is rendered
            timer = QTimer()
            timer.setSingleShot(True)
            timer.timeout.connect(lambda: self._safe_pause(timer))
            timer.start(50)  # 50ms should be enough for one frame

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
        """Launch preview window from timeline"""
        # Get paths from timeline
        video_path = timeline_window.video_path
        
        # Find annotated video (you need to implement this based on your cache)
        annotated_path = TimelineWithPreview.find_annotated_video(video_path, timeline_window.cache_data)
        
        # Create preview window
        preview = VideoPreviewWindow(video_path, annotated_path)
        
        # Connect signals
        timeline_window.signal_scene.time_clicked.connect(preview.seek_to_time)
        timeline_window.edit_scene.clip_double_clicked.connect(
            lambda start, end: preview.play_from_time(start)
        )
        preview.time_changed.connect(timeline_window.signal_scene.set_current_time)
        
        # Show both windows
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
                
        return None