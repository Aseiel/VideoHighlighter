
"""
Complete Signal Timeline Viewer with Filters and Edit Timeline
- Signal visualization with filtering
- Edit timeline with clip management
- Action/object filtering
- Exact time playback
"""

import sys
import os
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

class TimelineExporter:
    """Export edit timeline to various formats"""
    
    @staticmethod
    def to_edl(clips, video_path, output_path=None, fps=30):
        """Export to CMX3600 EDL - DaVinci Resolve compatible"""
        
        def seconds_to_timecode(seconds, fps=30, drop_frame=False):
            """Convert seconds to SMPTE timecode"""
            total_frames = int(round(seconds * fps))
            hours = total_frames // (3600 * fps)
            minutes = (total_frames // (60 * fps)) % 60
            secs = (total_frames // fps) % 60
            frames = total_frames % fps
            return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"
        
        lines = []
        
        # Header
        lines.append("TITLE: AI Video Editor Edit")
        lines.append("FCM: NON-DROP FRAME")
        lines.append("")
        
        # Reel name from filename (without extension)
        reel_name = os.path.splitext(os.path.basename(video_path))[0]
        # Limit reel name to 8 chars for compatibility
        reel_name = reel_name[:8].upper()
        
        # Add source file reference
        lines.append(f"* SOURCE FILE: {video_path}")
        lines.append("")
        
        # Each clip
        for i, (start, end) in enumerate(clips, 1):
            duration = end - start
            
            # Calculate cumulative time for record track
            record_start = sum(clips[j][1] - clips[j][0] for j in range(i-1))
            record_end = record_start + duration
            
            # Convert to timecode
            source_in = seconds_to_timecode(start, fps)
            source_out = seconds_to_timecode(end, fps)
            record_in = seconds_to_timecode(record_start, fps)
            record_out = seconds_to_timecode(record_end, fps)
            
            # EDL entry - proper format for DaVinci Resolve
            lines.append(f"{i:03d}  {reel_name:8} V     C        {source_in} {source_out} {record_in} {record_out}")
            lines.append(f"* FROM CLIP NAME: {os.path.basename(video_path)}")
            lines.append(f"* COMMENT: Clip {i} - {duration:.1f}s")
            lines.append("")
        
        # Write file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        return output_path
    
    @staticmethod
    def to_fcp_xml(clips, video_path, output_path=None, fps=30):
        """
        Export to Final Cut Pro XML format (DaVinci Resolve compatible)
        
        Args:
            clips: List of (start_time, end_time) tuples in seconds
            video_path: Path to source video
            output_path: Output file path (None = auto-generate)
            fps: Frames per second
        """
        if not clips:
            return None
            
        if output_path is None:
            base = os.path.splitext(video_path)[0]
            output_path = f"{base}_edit.xml"
        
        video_name = os.path.basename(video_path)
        
        # Create XML structure
        fcpxml = ET.Element("fcpxml", version="1.9")
        resources = ET.SubElement(fcpxml, "resources")
        library = ET.SubElement(fcpxml, "library")
        event = ET.SubElement(library, "event", name="AI Video Edit")
        project = ET.SubElement(event, "project", name="Edited Timeline")
        sequence = ET.SubElement(project, "sequence", format="r1")
        
        # Add format
        format_elem = ET.SubElement(resources, "format", 
                                   id="r1",
                                   name="FFVideoFormat1080p2997",
                                   frameDuration="1001/30000",
                                   width="1920",
                                   height="1080")
        
        # Add asset
        asset_id = f"asset-{hash(video_path) % 10000}"
        asset = ET.SubElement(resources, "asset",
                            id=asset_id,
                            name=video_name,
                            src=f"file://{video_path}")
        
        # Media duration
        duration_sec = clips[-1][1] - clips[0][0] if clips else 60
        duration_frames = int(duration_sec * fps)
        
        # Add sequence
        spine = ET.SubElement(sequence, "spine")
        
        # Total duration in frames
        total_duration = sum(end - start for start, end in clips)
        sequence.set("duration", f"{int(total_duration * fps * 100)}s")
        
        # Add each clip
        for i, (start, end) in enumerate(clips, 1):
            duration = end - start
            duration_frames = int(duration * fps * 100)
            start_frames = int(start * fps * 100)
            
            clip = ET.SubElement(spine, "clip",
                               name=f"Clip {i}",
                               duration=f"{duration_frames}s",
                               start=f"{start_frames}s")
            
            # Add video
            video = ET.SubElement(clip, "video")
            ET.SubElement(video, "offset", relative="start", value=f"{start_frames}s")
            
            # Add audio
            audio = ET.SubElement(clip, "audio")
            ET.SubElement(audio, "offset", relative="start", value=f"{start_frames}s")
            
            ET.SubElement(clip, "asset-ref", id=asset_id)
        
        # Pretty print
        xml_str = ET.tostring(fcpxml, encoding='utf-8')
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="  ")
        
        # Remove XML declaration if minidom adds it weird
        lines = pretty_xml.split('\n')
        if lines[0].startswith('<?xml'):
            lines[0] = '<?xml version="1.0" encoding="utf-8"?>'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return output_path
    
    @staticmethod
    def get_export_formats():
        """Return list of available export formats"""
        return [
            ("EDL (CMX3600)", "*.edl"),
            ("FCPXML (DaVinci Resolve)", "*.xml"),
            ("CSV", "*.csv"),
            ("JSON", "*.json")
        ]
    
    @staticmethod
    def export_auto(clips, video_path, format='edl'):
        """Auto-export based on format name"""
        format = format.lower()
        if format == 'edl':
            return TimelineExporter.to_edl(clips, video_path)
        elif format in ('fcpxml', 'xml', 'fcp'):
            return TimelineExporter.to_fcp_xml(clips, video_path)
        else:
            return None

class WaveformVisualizer:
    """Extracts and stores waveform data for visualization"""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.waveform_data = None  # List of (min_val, max_val) tuples
        self.duration = 0
        self.sample_rate = 44100
    
    def extract_waveform(self, num_points=1000):
        import os, tempfile, subprocess, wave
        import numpy as np

        fd, wav_file = tempfile.mkstemp(suffix=".wav")
        os.close(fd)  # IMPORTANT: don't keep the file handle open

        try:
            print(f"🎵 Extracting audio from: {self.video_path}")

            cmd = [
                "ffmpeg",
                "-y",
                "-i", self.video_path,
                "-map", "0:a:0",          # pick first audio stream explicitly
                "-vn",
                "-ac", "1",
                "-ar", str(self.sample_rate),
                "-c:a", "pcm_s16le",
                wav_file,
                "-hide_banner",
                "-loglevel", "error",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ FFmpeg failed")
                print("stderr:", result.stderr.strip())
                return None

            if not os.path.exists(wav_file) or os.path.getsize(wav_file) < 44:
                print("❌ WAV output missing/too small (likely no audio or ffmpeg write failed)")
                return None

            with wave.open(wav_file, "rb") as wf:
                rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16)

            if audio.size == 0 or rate <= 0:
                print("❌ No audio samples decoded")
                return None

            self.duration = audio.size / rate

            step = max(1, audio.size // num_points)
            waveform = []
            for i in range(0, audio.size, step):
                chunk = audio[i:i + step]
                if chunk.size:
                    waveform.append((float(chunk.min()) / 32768.0, float(chunk.max()) / 32768.0))

            self.waveform_data = waveform
            print(f"✅ Waveform extracted: {len(waveform)} points, duration={self.duration:.2f}s")
            return waveform

        except Exception as e:
            print(f"❌ Waveform extraction error: {e}")
            import traceback; traceback.print_exc()
            return None

        finally:
            try:
                if os.path.exists(wav_file):
                    os.remove(wav_file)
            except:
                pass



class TimelineBar:
    """Represents a single signal bar on the timeline"""
    def __init__(self, start_time, end_time, y_position, height, color, label, 
                 confidence=None, metadata=None):
        self.start_time = start_time
        self.end_time = end_time
        self.y_position = y_position
        self.height = height
        self.color = color
        self.label = label
        self.confidence = confidence  # Store confidence value (0-10 scale or 0-1 scale)
        self.metadata = metadata or {}
        self.scene = None  # Will be set when drawn
    
    def get_alpha(self):
        """Get transparency based on confidence"""
        if self.confidence is not None:
            # Normalize confidence to 0-10 scale if needed
            if self.confidence <= 1.0:  # Assuming 0-1 scale
                normalized_confidence = self.confidence * 10
            else:
                normalized_confidence = min(self.confidence, 10)
            
            # Map confidence 0-10 to alpha 100-255
            return int(100 + (normalized_confidence / 10.0) * 155)
        return 180  # Default semi-transparent
    
    def get_normalized_confidence(self):
        """Get confidence normalized to 0-1 scale"""
        if self.confidence is None:
            return 0.5  # Default medium confidence
        
        if self.confidence <= 1.0:
            return self.confidence
        else:
            return self.confidence / 10.0

class DraggableTimelineBar(QGraphicsRectItem):
    """A timeline bar that can be dragged multiple times"""
    
    def __init__(self, bar, x, width, parent=None):
        super().__init__(parent)
        self.bar = bar
        self.setRect(0, 0, width, bar.height)
        self.setPos(x, bar.y_position)
        
        # Set the rectangle position and size
        self.setRect(0, 0, width, bar.height)
        self.setPos(x, bar.y_position)
        
        # Create gradient fill
        gradient = QLinearGradient(0, 0, 0, bar.height)
        color = bar.color
        color.setAlpha(bar.get_alpha())
        light_color = color.lighter(130)
        light_color.setAlpha(bar.get_alpha())
        gradient.setColorAt(0, light_color)
        gradient.setColorAt(1, color)
        
        self.setBrush(QBrush(gradient))
        self.setPen(QPen(color.darker(120), 1))
        
        # Enable dragging and hover events
        self.setAcceptHoverEvents(True)
        self.setCursor(QCursor(Qt.OpenHandCursor))
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.ItemIsMovable, False)
        
        # Track mouse press position
        self.mouse_press_pos = None
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_press_pos = event.pos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))

            scene = self.scene()
            if scene:
                time = event.scenePos().x() / scene.pixels_per_second

                # Optional: waveform click -> auto clip
                if scene.waveform_visible and scene.waveform and scene.video_duration > 0:
                    sample_index = int(time * len(scene.waveform) / scene.video_duration)
                    if 0 <= sample_index < len(scene.waveform):
                        min_val, max_val = scene.waveform[sample_index]
                        amplitude = (abs(min_val) + abs(max_val)) / 2
                        if amplitude > 0.3:
                            start = max(0, time - 1.5)
                            end = min(scene.video_duration, time + 1.5)
                            scene.waveform_clicked.emit(start, end, amplitude)

                scene.time_clicked.emit(time)

                if event.modifiers() & Qt.ControlModifier:
                    scene.add_to_edit_requested.emit(time)

            event.accept()
            return

        super().mousePressEvent(event)

    
    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.LeftButton) or self.mouse_press_pos is None:
            super().mouseMoveEvent(event)
            return

        if (event.pos() - self.mouse_press_pos).manhattanLength() < QApplication.startDragDistance():
            return

        self.start_drag(event)
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_press_pos = None
            self.setCursor(QCursor(Qt.OpenHandCursor))
        super().mouseReleaseEvent(event)

    
    def start_drag(self, event):
        import json
        mime_data = QMimeData()
        bar_data = {
            'type': 'timeline_bar',
            'start_time': self.bar.start_time,
            'end_time': self.bar.end_time,
            'duration': self.bar.end_time - self.bar.start_time,
            'label': self.bar.label,
            'metadata': self.bar.metadata
        }
        mime_data.setText(json.dumps(bar_data))

        view = self.scene().views()[0] if self.scene() and self.scene().views() else None
        drag = QDrag(view.viewport() if view else event.widget())
        drag.setMimeData(mime_data)

        rect = self.rect()
        pixmap = QPixmap(int(rect.width()), int(rect.height()))
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setBrush(self.brush())
        painter.setPen(self.pen())
        painter.drawRect(0, 0, int(rect.width()) - 1, int(rect.height()) - 1)
        painter.end()

        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(int(rect.width() / 2), int(rect.height() / 2)))
        drag.exec(Qt.CopyAction)

        self.setCursor(QCursor(Qt.OpenHandCursor))
        self.mouse_press_pos = None
    
    def hoverEnterEvent(self, event):
        """Highlight on hover"""
        self.original_pen = self.pen()
        self.highlight_pen = QPen(QColor(255, 255, 0), 2)
        self.setPen(self.highlight_pen)
        
        # Show tooltip
        duration = self.bar.end_time - self.bar.start_time
        self.setToolTip(f"Drag to add to edit timeline\n{self.bar.label}\n{self.bar.start_time:.1f}s - {self.bar.end_time:.1f}s\nDuration: {duration:.1f}s")
        
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        """Remove highlight"""
        if hasattr(self, 'original_pen'):
            self.setPen(self.original_pen)
        
        super().hoverLeaveEvent(event)

class EditClipItem(QGraphicsRectItem):
    """Represents a clip in the edit timeline that can be dragged for reordering"""
    
    def __init__(self, start_time, end_time, y, height, color, index):
        super().__init__()
        self.start_time = start_time
        self.end_time = end_time
        self.color = color
        self.index = index
        self.is_selected = False
        self.original_pos = None
        
        # Set rectangle properties
        self.setRect(0, 0, 0, height)
        self.setPos(0, y)
        self.setBrush(QBrush(color))
        self.setPen(QPen(color.darker(150), 1))
        
        # Make it draggable
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setCursor(QCursor(Qt.OpenHandCursor))
        
        # Add text label
        self.text_item = QGraphicsTextItem(self)
        self.update_label()
    
    def update_label(self):
        """Update clip label text"""
        duration = self.end_time - self.start_time
        label_text = f"Clip {self.index}\n{self.start_time:.1f}s-{self.end_time:.1f}s\n({duration:.1f}s)"
        self.text_item.setPlainText(label_text)
        self.text_item.setFont(QFont("Arial", 8))
        self.text_item.setDefaultTextColor(Qt.white)
        
        # Center text in clip
        text_rect = self.text_item.boundingRect()
        self.text_item.setPos(self.rect().width()/2 - text_rect.width()/2, 
                             self.rect().height()/2 - text_rect.height()/2)
    
    def set_selected(self, selected):
        """Update selection state"""
        self.is_selected = selected
        if selected:
            self.setPen(QPen(Qt.yellow, 2))
        else:
            self.setPen(QPen(self.color.darker(150), 1))
    
    def mousePressEvent(self, event):
        """Handle mouse press for dragging"""
        if event.button() == Qt.LeftButton:
            # Clear other selections if shift/ctrl not pressed
            if not (event.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier)):
                scene = self.scene()
                if scene:
                    for item in scene.selectedItems():
                        if item != self:
                            item.setSelected(False)
            
            # Set this item as selected
            self.setSelected(True)
            self.original_pos = self.pos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
        
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release - reorder if moved"""
        try:
            if event.button() == Qt.LeftButton and hasattr(self, 'original_pos') and self.original_pos:
                self.setCursor(QCursor(Qt.OpenHandCursor))
                
                # Check if item was moved significantly
                new_pos = self.pos()
                if (new_pos - self.original_pos).manhattanLength() > 10:
                    # Item was dragged - trigger reorder
                    scene = self.scene()
                    if scene and hasattr(scene, 'reorder_clip'):
                        scene.reorder_clip(self.index - 1, new_pos.x())  # Convert to 0-based index
                
                self.original_pos = None
            
            # Call parent only if item still exists
            if self.scene():
                super().mouseReleaseEvent(event)
        except RuntimeError:
            # Item was deleted, ignore
            pass

    def itemChange(self, change, value):
        """Handle item changes (like selection)"""
        if change == QGraphicsRectItem.ItemSelectedChange:
            # Update appearance when selected/deselected
            if value:
                self.setPen(QPen(Qt.yellow, 2))
            else:
                self.setPen(QPen(self.color.darker(150), 1))
        
        return super().itemChange(change, value)
    
    def mouseDoubleClickEvent(self, event):
        """Handle double click - play this clip"""
        if event.button() == Qt.LeftButton:
            # Emit signal to play this clip
            scene = self.scene()
            if hasattr(scene, 'clip_double_clicked'):
                scene.clip_double_clicked.emit(self.start_time, self.end_time)
        super().mouseDoubleClickEvent(event)
    
    def hoverEnterEvent(self, event):
        """Show tooltip on hover"""
        duration = self.end_time - self.start_time
        self.setToolTip(f"Clip {self.index}\n{self.start_time:.1f}s - {self.end_time:.1f}s\nDuration: {duration:.1f}s\nDrag to reorder\nDouble-click to play")
        super().hoverEnterEvent(event)

class EditTimelineScene(QGraphicsScene):
    """Simple timeline showing clips as colored rectangles"""
    
    clip_double_clicked = Signal(float, float)  # start, end
    clip_added = Signal(float, float)  # start, end
    clip_removed = Signal(int)  # index
    
    def __init__(self, video_path, video_duration, parent=None, cache=None):  # ADD cache parameter
        super().__init__(parent)
        self.video_path = video_path
        self.video_duration = video_duration
        self.cache = cache  # Store cache reference
        self.clips = []  # List of (start_time, end_time) tuples
        self.clip_items = []  # List of EditClipItem objects
        self.pixels_per_second = 50
        self.clip_height = 60
        self.clip_spacing = 5
        
        # Visual feedback for drop target
        self.drop_indicator = None
        self.drop_indicator_marker = None
        self.drop_position = None
        self.is_dragging_over = False
        
        # Load initial highlights if available
        self.load_initial_clips()
        
        self.setSceneRect(0, 0, 1000, self.clip_height + 40)
        self.build_timeline()

    def contextMenuEvent(self, event):
        """Show context menu for loading different highlight versions"""
        menu = QMenu()
        
        # Load from cache action
        load_action = menu.addAction("📂 Load from Cache...")
        load_action.triggered.connect(self.load_from_cache_menu)
        
        # Save to cache action
        save_action = menu.addAction("💾 Save to Cache")
        save_action.triggered.connect(lambda: self.save_clips_to_cache())
        
        menu.exec(event.screenPos())
    
    def load_from_cache_menu(self):
        """Show dialog to load different highlight versions from cache"""
        if not self.cache or not hasattr(self.cache, 'get_highlight_history'):
            QMessageBox.warning(None, "Cache Error", 
                               "Enhanced cache not available. Cannot load from cache.")
            return
        
        # Get highlight history
        history = self.cache.get_highlight_history(self.video_path)
        if not history:
            QMessageBox.information(None, "No Cache", 
                                   "No cached highlight versions found for this video.")
            return
        
        # Create selection dialog
        dialog = QDialog()
        dialog.setWindowTitle("Load Highlight Version")
        dialog.resize(500, 400)
        layout = QVBoxLayout(dialog)
        
        list_widget = QListWidget()
        for i, entry in enumerate(history):
            created = entry.get('created_at', 'Unknown')
            segments = entry.get('segments_count', 0)
            duration = entry.get('total_duration', 0)
            item_text = f"Version {i+1}: {segments} clips, {duration:.1f}s ({created})"
            list_widget.addItem(item_text)
        
        layout.addWidget(QLabel("Select cached highlight version:"))
        layout.addWidget(list_widget)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(lambda: self.load_selected_version(dialog, list_widget, history))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        dialog.exec()
    
    def load_selected_version(self, dialog, list_widget, history):
        """Load selected highlight version"""
        selected = list_widget.currentRow()
        if 0 <= selected < len(history):
            entry = history[selected]
            segments = entry.get('segments', [])
            
            if segments:
                self.clips = segments
                self.build_timeline()
                QMessageBox.information(None, "Loaded", 
                                       f"Loaded {len(segments)} clips from cache.")
        
        dialog.accept()


    def save_clips_to_cache(self, parameters=None):
        """Save current clips to cache for future use"""
        if not self.cache or not hasattr(self.cache, 'save_highlight_segments'):
            print("⚠️ Cache not available for saving")
            return False
        
        # Default parameters if none provided
        if parameters is None:
            parameters = {
                'max_duration': 420,
                'clip_time': 10,
                'highlight_objects': [],
                'interesting_actions': [],
                'scene_points': 0,
                'motion_event_points': 0,
                'motion_peak_points': 3,
                'exact_duration': None
            }
        
        # Prepare segments metadata
        segments_metadata = []
        for start, end in self.clips:
            segments_metadata.append({
                'score': 1.0,
                'signals': {'user_edited': 1.0},
                'primary_reason': 'manual_selection'
            })
        
        # Save to cache
        try:
            success = self.cache.save_highlight_segments(
                self.video_path,
                parameters,
                self.clips,
                segments_metadata,
                score_info={'user_edited': True}
            )
            
            if success:
                print(f"✅ Saved {len(self.clips)} clips to cache")
                return True
            return False
        except Exception as e:
            print(f"❌ Failed to save clips to cache: {e}")
            return False

    def keyPressEvent(self, event):
        """Handle key presses for deleting clips"""
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            self.remove_selected_clips()
            event.accept()
        else:
            super().keyPressEvent(event)
    
    def remove_selected_clips(self):
        selected_items = [item for item in self.items() if isinstance(item, EditClipItem) and item.isSelected()]
        
        if not selected_items:
            return
        
        # Collect indices first (from high to low)
        indices_to_remove = sorted(
            (self.clip_items.index(item) for item in selected_items),
            reverse=True
        )
        
        for idx in indices_to_remove:
            # Remove graphics item
            if self.clip_items[idx] in self.items():
                self.removeItem(self.clip_items[idx])
            
            # Remove data
            self.clip_items.pop(idx)
            self.clips.pop(idx)
            
            self.clip_removed.emit(idx)
        
        self.build_timeline()

    def load_initial_clips(self):
        """Load initial clips from highlight cache"""
        self.clips = []

        # Try ANY cached highlight version (not just default params)
        if self.cache and hasattr(self.cache, 'get_highlight_history'):
            # Try with None to get all versions
            history = self.cache.get_highlight_history(self.video_path, analysis_params=None)
            if history:
                # Use the most recent highlight version
                entry = history[0]  # Most recent first
                segments = entry.get('segments', [])
                if segments:
                    self.clips = segments
                    print(f"✅ Loaded {len(self.clips)} highlight segments from cache (most recent)")
                    return
   
        # 2) Fallback: Check if pipeline saved segments in analysis cache
        cache_data = None
        parent = self.parent()
        while parent is not None and not hasattr(parent, "cache_data"):
            parent = parent.parent()

        if parent is not None and hasattr(parent, "cache_data"):
            cache_data = parent.cache_data

        if cache_data and "final_segments" in cache_data:
            loaded = []
            for segment in cache_data.get("final_segments", []):
                if isinstance(segment, (list, tuple)) and len(segment) >= 2:
                    start, end = float(segment[0]), float(segment[1])
                    if end > start and (end - start) >= 0.5:
                        loaded.append((start, end))

            if loaded:
                self.clips = loaded
                print(f"✅ Loaded {len(self.clips)} segments from cache_data['final_segments']")
                return
        
        # 3) Last resort: sample clips
        print("⚠️ No cached highlights found, creating sample clips")
        if self.video_duration > 30:
            sample_points = [(0.1, 0.2), (0.3, 0.4), (0.7, 0.8)]
            for start_ratio, end_ratio in sample_points:
                start = self.video_duration * start_ratio
                end = self.video_duration * end_ratio
                duration = end - start
                if duration > 15:
                    end = start + 15
                elif duration < 3:
                    end = start + 3
                if end <= self.video_duration:
                    self.clips.append((start, end))
        else:
            start = max(0, self.video_duration / 4)
            end = min(self.video_duration, self.video_duration * 3 / 4)
            if end - start >= 3:
                self.clips.append((start, end))
    
    def build_timeline(self):
        """Build the edit timeline visualization"""
        print("build_timeline() called")
        self.clear()
        self.clip_items = []
        
        # Draw background
        self.addRect(self.sceneRect(), QPen(Qt.NoPen), QBrush(QColor(30, 30, 40)))
        
        # Draw title
        title = self.addText("Edit Timeline", QFont("Arial", 12, QFont.Weight.Bold))
        title.setDefaultTextColor(Qt.white)
        title.setPos(10, 5)
        
        # Draw time ruler
        self.draw_time_ruler()
        
        # Draw clips
        current_x = 20
        y_pos = 35
        
        for i, (start, end) in enumerate(self.clips):
            duration = end - start
            width = max(60, duration * self.pixels_per_second)
            
            # Create clip item
            color = self.get_clip_color(i)
            clip_item = EditClipItem(start, end, y_pos, self.clip_height, color, i + 1)
            clip_item.setRect(0, 0, width, self.clip_height)
            clip_item.setPos(current_x, y_pos)
            self.addItem(clip_item)
            self.clip_items.append(clip_item)
            
            # Update clip item with actual width for text positioning
            clip_item.update_label()
            
            current_x += width + self.clip_spacing
        
        # Update scene width based on total clips width
        total_width = max(1000, current_x + 20)
        self.setSceneRect(0, 0, total_width, self.clip_height + 40)
        
        # Draw clip count
        count_text = self.addText(f"{len(self.clips)} clips", QFont("Arial", 10))
        count_text.setDefaultTextColor(QColor(200, 200, 200))
        count_text.setPos(total_width - 100, 5)
    
    def draw_time_ruler(self):
        """Draw a time ruler above the clips"""
        ruler_y = 30
        self.addLine(20, ruler_y, self.sceneRect().width() - 20, ruler_y, 
                    QPen(QColor(150, 150, 150), 1))
        
        # Draw time markers every 10 pixels
        for i in range(0, int(self.sceneRect().width()), 10):
            x = 20 + i
            if i % 50 == 0:  # Major tick every 50 pixels
                self.addLine(x, ruler_y - 8, x, ruler_y, QPen(QColor(200, 200, 200), 1))
                
                # Calculate time
                time_seconds = i / self.pixels_per_second
                if time_seconds <= self.video_duration:
                    time_text = self.addText(f"{time_seconds:.1f}s", QFont("Arial", 8))
                    time_text.setDefaultTextColor(QColor(180, 180, 180))
                    time_text.setPos(x - 10, ruler_y - 25)
            else:
                self.addLine(x, ruler_y - 4, x, ruler_y, QPen(QColor(150, 150, 150), 1))
    
    def get_clip_color(self, index):
        """Get a color for a clip based on its index"""
        colors = [
            QColor(100, 150, 255, 220),  # Blue
            QColor(100, 255, 100, 220),  # Green
            QColor(255, 100, 100, 220),  # Red
            QColor(255, 200, 50, 220),   # Yellow
            QColor(200, 100, 255, 220),  # Purple
            QColor(50, 255, 255, 220),   # Cyan
        ]
        return colors[index % len(colors)]
    
    def add_clip(self, start_time, end_time):
        """Add a new clip to the timeline"""
        # Ensure valid times
        start_time = max(0, min(start_time, self.video_duration - 1))
        end_time = max(start_time + 1, min(end_time, self.video_duration))
        
        self.clips.append((start_time, end_time))
        self.build_timeline()
        self.clip_added.emit(start_time, end_time)
    
    def add_clip_from_selection(self, start_time, end_time=None):
        """Add a clip from a time selection (default 5 second duration)"""
        if end_time is None:
            end_time = start_time + 5  # Default 5 second clip
        
        # Ensure minimum duration
        if end_time - start_time < 0.5:
            end_time = start_time + 3  # Minimum 3 seconds
        
        self.add_clip(start_time, end_time)
    
    def get_total_duration(self):
        """Get total duration of all clips"""
        return sum(end - start for start, end in self.clips)
    
    def get_clip_times(self):
        """Get list of all clip time ranges"""
        return self.clips.copy()
    
    # DRAG AND DROP METHODS
    def clear_drop_indicators(self):
        """Safely remove drop indicators"""
        try:
            if hasattr(self, 'drop_indicator') and self.drop_indicator:
                self.removeItem(self.drop_indicator)
                self.drop_indicator = None
        except:
            self.drop_indicator = None
        
        try:
            if hasattr(self, 'drop_indicator_marker') and self.drop_indicator_marker:
                self.removeItem(self.drop_indicator_marker)
                self.drop_indicator_marker = None
        except:
            self.drop_indicator_marker = None


    def dragEnterEvent(self, event):
        """Accept drag events with timeline bar data"""
        self.clear_drop_indicators()
        
        if event.mimeData().hasText():
            try:
                import json
                data = json.loads(event.mimeData().text())
                if data.get('type') == 'timeline_bar':
                    event.acceptProposedAction()
                    self.is_dragging_over = True
                    
                    # Show drop indicator
                    self.show_drop_indicator(event.scenePos())
                    return
            except:
                pass
        
        event.ignore()

    def dragMoveEvent(self, event):
        """Update drop indicator position"""
        if event.mimeData().hasText():
            try:
                import json
                data = json.loads(event.mimeData().text())
                if data.get('type') == 'timeline_bar':
                    event.acceptProposedAction()
                    self.clear_drop_indicators()
                    self.show_drop_indicator(event.scenePos())
                    return
            except:
                pass
        
        event.ignore()

    def dragLeaveEvent(self, event):
        """Remove drop indicator"""
        self.is_dragging_over = False
        self.clear_drop_indicators()
    
    def dropEvent(self, event):
        """Handle drop to create a new clip"""
        if event.mimeData().hasText():
            try:
                import json
                data = json.loads(event.mimeData().text())
                
                if data.get('type') == 'timeline_bar':
                    # Get drop position
                    pos = event.scenePos()
                    
                    # Calculate which clip position this is (between clips)
                    insert_index = self.get_insert_index(pos.x())
                    
                    # Add the clip
                    start_time = data['start_time']
                    end_time = data['end_time']
                    
                    # Ensure valid duration
                    if end_time - start_time < 0.5:
                        end_time = start_time + 3.0  # Minimum 3 seconds
                    
                    # Insert at the calculated position
                    self.clips.insert(insert_index, (start_time, end_time))
                    self.build_timeline()
                    self.clip_added.emit(start_time, end_time)
                    
                    event.accept()
                    
                    # Show feedback
                    self.show_drop_feedback(insert_index)
                    
                    # Update parent window if available
                    if hasattr(self.parent(), 'update_edit_duration'):
                        self.parent().update_edit_duration()
                    
                    return
            
            except Exception as e:
                print(f"Drop error: {e}")
        
        event.ignore()
        self.is_dragging_over = False
        self.hide_drop_indicator()
    
    def show_drop_indicator(self, pos):
        """Show visual indicator where clip will be inserted"""
        # Clear old indicators first
        self.clear_drop_indicators()
        
        # Calculate insertion point
        insert_index = self.get_insert_index(pos.x())
        
        # Create indicator line
        x_pos = self.calculate_insert_x(insert_index)
        self.drop_indicator = self.addLine(x_pos, 35, x_pos, 35 + self.clip_height, 
                                          QPen(QColor(0, 255, 0), 2, Qt.DashLine))
        
        # Create insertion point marker
        self.drop_indicator_marker = self.addEllipse(x_pos - 5, 30, 10, 10, 
                                                    QPen(Qt.green, 2), QBrush(Qt.green))
        
        self.drop_position = insert_index
    
    def hide_drop_indicator(self):
        """Remove drop indicator - SAFE version"""
        self.clear_drop_indicators()
        
        if hasattr(self, 'drop_indicator_marker'):
            self.removeItem(self.drop_indicator_marker)
    
    def get_insert_index(self, x_pos):
        """Determine where to insert based on x position"""
        # Convert x position to clip index
        current_x = 20
        
        for i, (start, end) in enumerate(self.clips):
            duration = end - start
            width = max(60, duration * self.pixels_per_second)
            
            # Check if position is before this clip
            if x_pos < current_x + width / 2:
                return i
            
            current_x += width + self.clip_spacing
        
        # If beyond all clips, append at end
        return len(self.clips)
    
    def calculate_insert_x(self, index):
        """Calculate x position for insertion at given index"""
        current_x = 20
        
        for i in range(index):
            if i < len(self.clips):
                start, end = self.clips[i]
                duration = end - start
                width = max(60, duration * self.pixels_per_second)
                current_x += width + self.clip_spacing
        
        return current_x
    
    def show_drop_feedback(self, insert_index):
        """Show visual feedback after successful drop"""
        # Highlight the newly inserted clip briefly
        if insert_index < len(self.clip_items):
            item = self.clip_items[insert_index]
            
            # Store original pen
            original_pen = item.pen()
            
            # Flash with highlight color
            def flash():
                item.setPen(QPen(Qt.yellow, 3))
                QTimer.singleShot(300, lambda: item.setPen(original_pen))
            
            QTimer.singleShot(100, flash)
    
    def reorder_clip(self, clip_index, new_x_pos):
        """Reorder a clip based on drag position"""
        if 0 <= clip_index < len(self.clips):
            # Remove clip from current position
            clip = self.clips.pop(clip_index)
            
            # Determine new insertion index
            insert_index = self.get_insert_index(new_x_pos)
            
            # Adjust index if moving forward in list
            if insert_index > clip_index:
                insert_index -= 1
            
            # Insert at new position
            self.clips.insert(insert_index, clip)
            
            # Rebuild timeline
            self.build_timeline()


class SignalTimelineScene(QGraphicsScene):
    """Improved graphics scene with filtering capabilities"""
    
    time_clicked = Signal(float)
    add_to_edit_requested = Signal(float)
    filter_changed = Signal(dict)
    waveform_clicked = Signal(float, float, float)
    
    def __init__(self, cache_data, video_duration, parent=None, waveform=None):
        super().__init__(parent)
        self.cache_data = cache_data
        self.video_duration = max(video_duration, 1.0)
        
        # Waveform visualization
        self.waveform = waveform or []
        self.waveform_visible = bool(self.waveform)
        self.waveform_opacity = 0.7
        
        print(f"🎵 SignalTimelineScene init: waveform={len(self.waveform)} points, visible={self.waveform_visible}")

        if self.waveform_visible:
            # Generate colors for waveform
            self.waveform_colors = self.generate_waveform_colors()
        else:
            self.waveform_colors = []
        
        # Dynamic zoom for short videos
        if video_duration < 30:
            self.pixels_per_second = 120.0
        elif video_duration < 120:
            self.pixels_per_second = 60.0
        else:
            self.pixels_per_second = 50.0
            
        self.layer_height = 40
        self.layer_spacing = 10
        
        # Extract action and object types for better organization
        self.action_types = self._extract_action_types()
        self.object_classes = self._extract_object_classes()
        
        # FILTERS: Track which actions/objects are visible
        self.visible_actions = {action: True for action in self.action_types}
        self.visible_objects = {obj: True for obj in self.object_classes}
        
        # NEW: Confidence filters (0.0 to 1.0 scale)
        self.min_confidence = 0.0  # Minimum confidence threshold
        self.max_confidence = 1.0  # Maximum confidence threshold
        
        # Define logical groups (order matters)
        self.group_order = [
            ('transcript', ['Transcript']),
            ('actions', [f"Action: {a}" for a in self.action_types]),
            ('objects', [f"Object: {o}" for o in self.object_classes]),
            ('scenes', ['Scenes']),
            ('motion', ['Motion Events', 'Motion Peaks']),
            ('audio', ['Audio Peaks']),
            ('highlights', ['Final Highlights'])
        ]
        
        # Layer visibility - initialize all to visible
        self.visible_layers = {}
        for _, tracks in self.group_order:
            for track in tracks:
                key = track.lower().replace(' ', '_')
                if 'action:' in key:
                    key = 'actions'
                elif 'object:' in key:
                    key = 'objects'
                elif 'final highlights' in key.lower():
                    key = 'highlights'
                self.visible_layers[key] = True
        
        # Color scheme
        self.colors = {
            'transcript': QColor(100, 150, 255),
            'actions': QColor(100, 255, 100),
            'objects': QColor(255, 100, 100),
            'scenes': QColor(200, 200, 100),
            'motion_events': QColor(255, 150, 50),
            'motion_peaks': QColor(255, 200, 100),
            'audio_peaks': QColor(150, 100, 255),
            'highlights': QColor(50, 200, 50)
        }
        
        # Create color palettes
        self.action_colors = self._color_palette(len(self.action_types), start_hue=100)
        self.object_colors = self._color_palette(len(self.object_classes), start_hue=340)
        
        self.bars = []
        self.build_timeline()


    
    def generate_waveform_colors(self):
        """Generate color gradient for waveform based on amplitude"""
        colors = []
        for i in range(256):
            # Create gradient from dark blue to bright cyan to yellow to red
            if i < 64:  # Quiet: dark blue to cyan
                r = int(50 + (i / 64) * 100)
                g = int(100 + (i / 64) * 155)
                b = 200
            elif i < 128:  # Medium: cyan to yellow
                r = int(150 + ((i-64) / 64) * 105)
                g = 255
                b = int(200 - ((i-64) / 64) * 200)
            else:  # Loud: yellow to red
                r = 255
                g = int(255 - ((i-128) / 128) * 155)
                b = 0
            colors.append(QColor(r, g, b, int(150 * self.waveform_opacity)))
        return colors

    
    def set_waveform_data(self, waveform_data):
        """Set waveform data for visualization"""
        self.waveform = waveform_data or []
        
        # IMPORTANT: Set visibility based on actual data
        self.waveform_visible = True if waveform_data and len(waveform_data) > 0 else False
        
        # Recompute colors with current opacity
        self.waveform_colors = self.generate_waveform_colors()
        
        print(f"✅ SignalTimelineScene.set_waveform_data: {len(self.waveform) if self.waveform else 0} points, visible={self.waveform_visible}")
        
        # Rebuild timeline to include waveform
        self.build_timeline()

    
    def draw_waveform_layer(self, y_pos, height):
        """Draw the waveform visualization layer"""
        if not self.waveform or len(self.waveform) == 0 or not self.waveform_visible:
            # IMPORTANT: Return the SAME y_pos when not drawing
            return y_pos  # Don't add any height
        
        print(f"🎵 draw_waveform_layer: Drawing at y={y_pos} with height={height}, {len(self.waveform)} points")
        
        # Draw waveform background
        waveform_y = y_pos
        self.addRect(0, waveform_y, self.sceneRect().width(), height, 
                    QPen(Qt.NoPen), QBrush(QColor(10, 10, 20, 50)))
        
        # Draw waveform label
        label = self.addText("AUDIO WAVEFORM", QFont("Arial", 10, QFont.Weight.Bold))
        label.setPos(5, waveform_y - 20)
        label.setDefaultTextColor(QColor(180, 220, 255))
        
        # Draw the actual waveform
        if len(self.waveform) > 0 and self.video_duration > 0:
            # Calculate proper scaling
            total_width = self.sceneRect().width()
            points_per_pixel = len(self.waveform) / total_width
            
            for i, (min_val, max_val) in enumerate(self.waveform):
                # Calculate x position based on time, not index
                time_pos = (i / len(self.waveform)) * self.video_duration
                x = time_pos * self.pixels_per_second
                
                # Skip if beyond visible area
                if x > total_width:
                    break
                
                # Calculate amplitude
                amplitude = (abs(min_val) + abs(max_val)) / 2
                amplitude_index = min(255, int(amplitude * 500))
                
                # Get color
                if self.waveform_colors and amplitude_index < len(self.waveform_colors):
                    color = self.waveform_colors[amplitude_index]
                    color.setAlpha(min(255, color.alpha() + 50))
                else:
                    color = QColor(100, 150, 255, 200)
                
                # Calculate y positions
                y_center = waveform_y + height // 2
                y_min = y_center + int(min_val * height // 2 * 0.8)
                y_max = y_center + int(max_val * height // 2 * 0.8)
                
                # Draw vertical line - ensure minimum width
                line_width = max(2, self.pixels_per_second / (len(self.waveform) / self.video_duration))
                pen = QPen(color, min(5, line_width))  # Cap at 5 pixels thick
                self.addLine(x, y_min, x, y_max, pen)
        
        return y_pos + height + self.layer_spacing



    def _extract_action_types(self):
        """Extract unique action names from cache data"""
        actions = set()
        for item in self.cache_data.get('actions', []):
            name = item.get('action_name') or item.get('action') or item.get('class') or 'unknown'
            if isinstance(name, str):
                actions.add(name.strip().title())
        return sorted(list(actions)) if actions else ['Unknown']
    
    def _extract_object_classes(self):
        """Extract unique object classes from cache data"""
        objs = set()
        for item in self.cache_data.get('objects', []):
            for obj in item.get('objects', []):
                if isinstance(obj, str):
                    objs.add(obj.strip().title())
        return sorted(list(objs)) if objs else ['Unknown']
    
    def _color_palette(self, count, start_hue=0):
        """Generate a color palette"""
        if count == 0:
            return []
        return [QColor.fromHsvF((start_hue + i * 0.618) % 1.0, 0.85, 0.92) 
                for i in range(count)]
    
    # Filter methods
    def set_action_filter(self, action_name, visible):
        """Set visibility for a specific action"""
        if action_name in self.visible_actions:
            self.visible_actions[action_name] = visible
            self.build_timeline()
            self.filter_changed.emit({
                'actions': self.visible_actions.copy(),
                'objects': self.visible_objects.copy()
            })
    
    def set_object_filter(self, object_name, visible):
        """Set visibility for a specific object"""
        if object_name in self.visible_objects:
            self.visible_objects[object_name] = visible
            self.build_timeline()
            self.filter_changed.emit({
                'actions': self.visible_actions.copy(),
                'objects': self.visible_objects.copy()
            })
    
    def set_all_actions_visible(self, visible):
        """Set all actions visible or hidden"""
        for action in self.visible_actions:
            self.visible_actions[action] = visible
        self.build_timeline()
        self.filter_changed.emit({
            'actions': self.visible_actions.copy(),
            'objects': self.visible_objects.copy()
        })
    
    def set_all_objects_visible(self, visible):
        """Set all objects visible or hidden"""
        for obj in self.visible_objects:
            self.visible_objects[obj] = visible
        self.build_timeline()
        self.filter_changed.emit({
            'actions': self.visible_actions.copy(),
            'objects': self.visible_objects.copy()
        })
    
    def get_filtered_actions(self):
        """Get list of currently visible actions"""
        return [action for action, visible in self.visible_actions.items() if visible]
    
    def get_filtered_objects(self):
        """Get list of currently visible objects"""
        return [obj for obj, visible in self.visible_objects.items() if visible]
    
    def build_timeline(self):
        """Rebuild the timeline with waveform"""
        print(f"🔄 SignalTimelineScene.build_timeline() called")
        print(f"   - Waveform data: {self.waveform is not None}, length: {len(self.waveform)}")
        print(f"   - Waveform visible: {self.waveform_visible}")
        
        # If we have a view connected, store its current transform
        views = self.views()
        old_transform = None
        old_h_scroll = None
        if views:
            view = views[0]
            old_transform = view.transform()
            old_h_scroll = view.horizontalScrollBar().value()
        
        # Calculate width based on video duration
        width = self.video_duration * self.pixels_per_second
        
        # Start with base height for time ruler
        height = 50  # Time ruler and labels
        
        # ONLY add waveform height if it's visible AND has data
        if self.waveform_visible and self.waveform and len(self.waveform) > 0:
            height += 80  # Waveform height
            height += self.layer_spacing  # Add spacing after waveform
        
        # Add height for other visible layers
        for _, tracks in self.group_order:
            for track in tracks:
                key = track.lower().replace(' ', '_')
                if 'action:' in key:
                    key = 'actions'
                elif 'object:' in key:
                    key = 'objects'
                elif 'final highlights' in key.lower():
                    key = 'highlights'
                
                if self.visible_layers.get(key, True):
                    height += self.layer_height + self.layer_spacing
        
        self.setSceneRect(0, 0, width, height)
        self.clear()
        self.bars = []
        
        # Draw background and time markers FIRST
        self.draw_background()
        self.draw_time_markers()
        
        # Start drawing below time markers
        current_y = 40
        
        # Draw waveform if visible
        if self.waveform_visible and self.waveform and len(self.waveform) > 0:
            current_y = self.draw_waveform_layer(current_y, 80)
        
        # Draw other layers (rest of your existing code...)
        
        # Draw time markers
        self.draw_time_markers()
        
        # Restore current time indicator if it was set
        if hasattr(self, 'current_time_seconds'):
            self.set_current_time(self.current_time_seconds)
        
        # Restore view transform if we had one
        if views and old_transform:
            view = views[0]
            
            # Calculate the horizontal scaling needed to maintain same visible area
            old_visible_width = self.sceneRect().width() / old_transform.m11()
            new_visible_width = width
            
            # Only adjust horizontal scale if scene width changed significantly
            if abs(old_visible_width - new_visible_width) > 10:
                scale_factor = new_visible_width / old_visible_width
                new_transform = old_transform.scale(scale_factor, 1.0)
                view.setTransform(new_transform)
                view.horizontalScrollBar().setValue(old_h_scroll)
        
        print(f"✅ Timeline rebuilt successfully, final height={height}")

        
        
        # Draw other layers
        # Layer 1: Transcript
        if self.visible_layers.get('transcript', True):
            current_y = self.draw_transcript_layer(current_y)

        
        # Layer 2: Actions (with better naming)
        if self.visible_layers.get('actions', True):
            current_y = self.draw_improved_actions_layer(current_y)
        
        # Layer 3: Objects (organized by class)
        if self.visible_layers.get('objects', True):
            current_y = self.draw_improved_objects_layer(current_y)
        
        # Layer 4: Scenes
        if self.visible_layers.get('scenes', True):
            current_y = self.draw_scenes_layer(current_y)
        
        # Layer 5: Motion Events
        if self.visible_layers.get('motion_events', True):
            current_y = self.draw_motion_events_layer(current_y)
        
        # Layer 6: Motion Peaks
        if self.visible_layers.get('motion_peaks', True):
            current_y = self.draw_motion_peaks_layer(current_y)
        
        # Layer 7: Audio Peaks
        if self.visible_layers.get('audio_peaks', True):
            current_y = self.draw_audio_peaks_layer(current_y)
        
        # Layer 8: Highlight Segments
        if self.visible_layers.get('highlights', True):
            current_y = self.draw_highlights_layer(current_y)
        
        # Draw time markers
        self.draw_time_markers()
        
        # Restore current time indicator if it was set
        if hasattr(self, 'current_time_seconds'):
            self.set_current_time(self.current_time_seconds)
        
        print(f"✅ Timeline rebuilt successfully, final height={height}")

        
    def draw_background(self):
        """Draw gradient background with subtle grid"""
        gradient = QLinearGradient(0, 0, 0, self.sceneRect().height())
        gradient.setColorAt(0, QColor(20, 20, 30))
        gradient.setColorAt(1, QColor(40, 40, 50))
        self.addRect(self.sceneRect(), QPen(Qt.PenStyle.NoPen), QBrush(gradient))
        
        # Add subtle grid lines
        for sec in range(0, int(self.video_duration) + 1, 5):
            x = sec * self.pixels_per_second
            pen = QPen(QColor(45, 45, 55) if sec % 30 else QColor(70, 70, 90), 1)
            self.addLine(x, 0, x, self.sceneRect().height(), pen)
        
        # FIX: Draw time markers AFTER background but BEFORE waveform
        self.draw_time_markers()

    
    def draw_transcript_layer(self, y_pos):
        """Draw transcript segments with improved labeling"""
        label = self.addText("TRANSCRIPT", QFont("Arial", 10, QFont.Weight.Bold))
        label.setPos(5, y_pos - 20)
        label.setDefaultTextColor(QColor(180, 220, 255))
        
        if 'transcript' in self.cache_data and self.cache_data['transcript'].get('segments'):
            for segment in self.cache_data['transcript']['segments']:
                start = segment.get('start', 0)
                end = segment.get('end', start + 1)
                text = segment.get('text', '').strip()
                
                if text:
                    # Calculate visual weight based on text density
                    words = len(text.split())
                    duration = max(0.1, end - start)
                    density = words / duration
                    intensity = min(10, density * 2)
                    
                    bar = TimelineBar(
                        start, end, y_pos, self.layer_height,
                        self.colors['transcript'], text[:30] + "..." if len(text) > 30 else text,
                        confidence=intensity,
                        metadata={'full_text': text, 'words': words}
                    )
                    self.draw_bar(bar)
                    self.bars.append(bar)
        
        return y_pos + self.layer_height + self.layer_spacing
    
    def draw_improved_actions_layer(self, y_pos):
        """Draw action detections with organized classification and filtering"""
        label = self.addText("ACTIONS", QFont("Arial", 10, QFont.Weight.Bold))
        label.setPos(5, y_pos - 20)
        label.setDefaultTextColor(QColor(180, 220, 255))
        
        # Group actions by type
        action_groups = defaultdict(list)
        for action in self.cache_data.get('actions', []):
            # Apply confidence filter
            if not self.should_show_action(action):
                continue
                
            action_name = action.get('action_name') or action.get('action') or 'Unknown'
            action_name = action_name.strip().title()
            if action_name in self.visible_actions and self.visible_actions[action_name]:
                action_groups[action_name].append(action)
        
        # If no visible actions, still show the layer but empty
        if not action_groups:
            # Show filter status message
            if self.min_confidence > 0 or self.max_confidence < 1:
                text = self.addText(f"(filtered: confidence {self.min_confidence:.1f}-{self.max_confidence:.1f})", 
                                   QFont("Arial", 9))
            else:
                text = self.addText("(no actions)", QFont("Arial", 9))
            text.setPos(150, y_pos + 15)
            text.setDefaultTextColor(QColor(150, 150, 150))
            return y_pos + self.layer_height + self.layer_spacing
        
        # Calculate y offset for each action type
        type_height = self.layer_height // max(1, len(action_groups))
        current_type_y = y_pos
        
        for idx, (action_type, actions) in enumerate(sorted(action_groups.items())):
            # Get color from palette
            if action_type in self.action_types:
                try:
                    color_idx = self.action_types.index(action_type)
                    color = self.action_colors[color_idx]
                except (ValueError, IndexError):
                    color = QColor(180, 220, 120)
            else:
                color = QColor(180, 220, 120)
            
            # Draw each action occurrence
            for action in actions:
                timestamp = action.get('timestamp', 0)
                confidence = action.get('confidence', 0.5)  # Default if missing
                
                bar = TimelineBar(
                    timestamp, timestamp + 0.5,
                    current_type_y, type_height,
                    color, action_type,
                    confidence=confidence,
                    metadata={'type': action_type, 'confidence': confidence}
                )
                self.draw_bar(bar)
                self.bars.append(bar)
            
            current_type_y += type_height
        
        return y_pos + self.layer_height + self.layer_spacing
    
    def draw_improved_objects_layer(self, y_pos):
        """Draw object detections organized by class with filtering"""
        label = self.addText("OBJECTS", QFont("Arial", 10, QFont.Weight.Bold))
        label.setPos(5, y_pos - 20)
        label.setDefaultTextColor(QColor(180, 220, 255))
        
        # Group objects by class
        object_groups = defaultdict(list)
        for obj_data in self.cache_data.get('objects', []):
            # Apply confidence filter
            if not self.should_show_object(obj_data):
                continue
                
            timestamp = obj_data.get('timestamp', 0)
            for obj_name in obj_data.get('objects', []):
                if isinstance(obj_name, str):
                    obj_name = obj_name.strip().title()
                    if obj_name in self.visible_objects and self.visible_objects[obj_name]:
                        object_groups[obj_name].append((timestamp, obj_data.get('confidence', 0.5)))
        
        # If no visible objects, still show the layer but empty
        if not object_groups:
            # Show filter status message
            if self.min_confidence > 0 or self.max_confidence < 1:
                text = self.addText(f"(filtered: confidence {self.min_confidence:.1f}-{self.max_confidence:.1f})", 
                                   QFont("Arial", 9))
            else:
                text = self.addText("(no objects)", QFont("Arial", 9))
            text.setPos(150, y_pos + 15)
            text.setDefaultTextColor(QColor(150, 150, 150))
            return y_pos + self.layer_height + self.layer_spacing
        
        # Calculate y offset for each object type
        type_height = self.layer_height // max(1, len(object_groups))
        current_type_y = y_pos
        
        for idx, (obj_type, detections) in enumerate(sorted(object_groups.items())):
            # Get color from palette
            if obj_type in self.object_classes:
                try:
                    color_idx = self.object_classes.index(obj_type)
                    color = self.object_colors[color_idx]
                except (ValueError, IndexError):
                    color = QColor(220, 140, 180)
            else:
                color = QColor(220, 140, 180)
            
            # Draw each object occurrence
            for timestamp, confidence in detections:
                bar = TimelineBar(
                    timestamp, timestamp + 0.3,
                    current_type_y, type_height,
                    color, obj_type,
                    confidence=confidence,
                    metadata={'type': obj_type, 'confidence': confidence}
                )
                self.draw_bar(bar)
                self.bars.append(bar)
            
            current_type_y += type_height
        
        return y_pos + self.layer_height + self.layer_spacing
    
    def draw_scenes_layer(self, y_pos):
        """Draw scene changes with improved labeling"""
        label = self.addText("SCENES", QFont("Arial", 10, QFont.Weight.Bold))
        label.setPos(5, y_pos - 20)
        label.setDefaultTextColor(QColor(180, 220, 255))
        
        if 'scenes' in self.cache_data:
            for i, scene in enumerate(self.cache_data['scenes']):
                start = scene.get('start', 0)
                end = scene.get('end', start + 1)
                
                # Alternate colors for scene differentiation
                scene_color = QColor(200, 200, 100)
                if i % 2 == 0:
                    scene_color = QColor(180, 180, 80)
                
                bar = TimelineBar(
                    start, end, y_pos, self.layer_height,
                    scene_color, f"Scene {i+1}",
                    metadata={'scene_index': i, 'duration': end - start}
                )
                self.draw_bar(bar)
                self.bars.append(bar)
        
        return y_pos + self.layer_height + self.layer_spacing
    
    def draw_motion_events_layer(self, y_pos):
        """Draw motion events as spikes"""
        label = self.addText("MOTION EVENTS", QFont("Arial", 10, QFont.Weight.Bold))
        label.setPos(5, y_pos - 20)
        label.setDefaultTextColor(QColor(180, 220, 255))
        
        if 'motion_events' in self.cache_data:
            print(f"DEBUG: Drawing {len(self.cache_data['motion_events'])} motion events")
            for timestamp in self.cache_data['motion_events']:
                x = timestamp * self.pixels_per_second
                # Draw vertical line with varying height based on intensity
                pen = QPen(self.colors['motion_events'], 2)
                self.addLine(x, y_pos, x, y_pos + self.layer_height, pen)
        else:
            print("DEBUG: No motion_events key in cache_data")
        
        return y_pos + self.layer_height + self.layer_spacing

    
    def draw_motion_peaks_layer(self, y_pos):
        """Draw motion peaks"""
        label = self.addText("MOTION PEAKS", QFont("Arial", 10, QFont.Weight.Bold))
        label.setPos(5, y_pos - 20)
        label.setDefaultTextColor(QColor(180, 220, 255))
        
        if 'motion_peaks' in self.cache_data:
            for timestamp in self.cache_data['motion_peaks']:
                x = timestamp * self.pixels_per_second
                pen = QPen(self.colors['motion_peaks'], 3)
                self.addLine(x, y_pos, x, y_pos + self.layer_height, pen)
        
        return y_pos + self.layer_height + self.layer_spacing
    
    def draw_audio_peaks_layer(self, y_pos):
        """Draw audio peaks"""
        label = self.addText("AUDIO PEAKS", QFont("Arial", 10, QFont.Weight.Bold))
        label.setPos(5, y_pos - 20)
        label.setDefaultTextColor(QColor(180, 220, 255))
        
        if 'audio_peaks' in self.cache_data:
            for timestamp in self.cache_data['audio_peaks']:
                x = timestamp * self.pixels_per_second
                pen = QPen(self.colors['audio_peaks'], 3)
                self.addLine(x, y_pos, x, y_pos + self.layer_height, pen)
        
        return y_pos + self.layer_height + self.layer_spacing
    
    def draw_highlights_layer(self, y_pos):
        """Draw final highlight segments with improved labeling"""
        label = self.addText("HIGHLIGHTS", QFont("Arial", 10, QFont.Weight.Bold))
        label.setPos(5, y_pos - 20)
        label.setDefaultTextColor(QColor(180, 220, 255))
        
        # Check if highlight segments are in cache
        if 'final_segments' in self.cache_data:
            for i, segment in enumerate(self.cache_data['final_segments']):
                if isinstance(segment, (list, tuple)) and len(segment) >= 2:
                    start, end = segment[0], segment[1]
                    duration = end - start
                    bar = TimelineBar(
                        start, end, y_pos, self.layer_height,
                        self.colors['highlights'], f"Highlight {i+1} ({duration:.1f}s)",
                        confidence=10,  # Full opacity
                        metadata={'index': i, 'duration': duration}
                    )
                    self.draw_bar(bar)
                    self.bars.append(bar)
        
        return y_pos + self.layer_height + self.layer_spacing
    
    def draw_bar(self, bar):
        """Draw a single timeline bar with gradient and INTELLIGENTLY SCALED LABELS"""
        x = bar.start_time * self.pixels_per_second
        width = max(2, (bar.end_time - bar.start_time) * self.pixels_per_second)
        
        # Create gradient for 3D effect
        gradient = QLinearGradient(x, bar.y_position, x, bar.y_position + bar.height)
        
        color = bar.color
        color.setAlpha(bar.get_alpha())
        
        # Lighter at top, darker at bottom
        light_color = color.lighter(130)
        light_color.setAlpha(bar.get_alpha())
        
        gradient.setColorAt(0, light_color)
        gradient.setColorAt(1, color)
        
        # FIX: Create draggable item instead of regular rectangle
        bar.scene = self  # Set reference to scene
        draggable_item = DraggableTimelineBar(bar, x, width)
        self.addItem(draggable_item)
        
        # ADVANCED SCALING: Intelligently decide what to show based on available space
        if width > 2:
            # Calculate text metrics based on available width
            # The key is to scale with both width AND pixels_per_second (zoom)
            
            # Determine if this is a "high zoom" scenario (zoomed in)
            is_high_zoom = self.pixels_per_second > 80  # More detailed when zoomed in
            
            # Calculate minimum readable width based on font
            min_readable_width = 10  # Minimum width to show any text
            
            if width >= min_readable_width:
                # Create font based on available space
                font = QFont("Arial")
                
                # Scale font size based on multiple factors
                base_size = 8
                
                # Factor 1: Width scaling
                width_factor = min(1.5, width / 40)  # Normalize to 40px = factor 1
                
                # Factor 2: Zoom level scaling (more detail when zoomed in)
                zoom_factor = min(1.2, self.pixels_per_second / 70)
                
                # Factor 3: Duration scaling (longer bars can have bigger text)
                duration = bar.end_time - bar.start_time
                duration_factor = min(1.3, duration / 3)
                
                # Combine factors
                scale_factor = width_factor * zoom_factor * duration_factor
                font_size = max(6, min(12, int(base_size * scale_factor)))
                font.setPointSize(font_size)
                
                # Also consider making font bold for better visibility
                if width > 50 and zoom_factor > 0.8:
                    font.setBold(True)
                
                # Prepare label text
                label_text = bar.label
                
                # Estimate text width with this font
                fm = QFontMetrics(font)
                estimated_text_width = fm.horizontalAdvance(label_text)
                
                # If text is too wide for the bar, try to fit it
                if estimated_text_width > width * 0.9:  # Leave 10% margin
                    # Strategy 1: Try smaller font
                    smaller_font_size = max(6, font_size - 1)
                    font.setPointSize(smaller_font_size)
                    fm = QFontMetrics(font)
                    estimated_text_width = fm.horizontalAdvance(label_text)
                    
                    # Strategy 2: If still too wide, truncate
                    if estimated_text_width > width * 0.9:
                        # Calculate how many characters we can fit
                        avg_char_width = fm.horizontalAdvance("W")  # Wide character
                        max_chars = max(1, int((width * 0.8) / avg_char_width))
                        
                        if max_chars >= 3:
                            # Truncate with ellipsis
                            label_text = bar.label[:max_chars - 1] + "…"
                        elif max_chars >= 1:
                            # Just show first character
                            label_text = bar.label[0] if bar.label else "•"
                        else:
                            # Not enough space for any text
                            label_text = ""
                elif width < 20 and len(label_text) > 3:
                    # Very narrow bar - show abbreviation
                    label_text = bar.label[:2] + "…" if len(bar.label) > 2 else bar.label
                
                # Create and position text if we have something to show
                if label_text:
                    text = QGraphicsTextItem(label_text, draggable_item)
                    text.setFont(font)
                    
                    # Choose text color based on bar brightness
                    bar_brightness = (color.red() + color.green() + color.blue()) / 3
                    if bar_brightness > 150:  # Light bar
                        text_color = QColor(30, 30, 30, 220)  # Dark text
                    else:
                        text_color = QColor(255, 255, 255, 220)  # Light text
                    
                    text.setDefaultTextColor(text_color)
                    
                    # Center text in bar
                    text_rect = text.boundingRect()
                    text_x = (width - text_rect.width()) / 2
                    text_y = (bar.height - text_rect.height()) / 2
                    
                    # Ensure text stays within bar bounds
                    text_x = max(1, min(text_x, width - text_rect.width() - 1))
                    text_y = max(1, min(text_y, bar.height - text_rect.height() - 1))
                    
                    text.setPos(text_x, text_y)
                    
                    # Always add comprehensive tooltip
                    self.add_bar_tooltip(bar, draggable_item, text)
                else:
                    # Bar too small for text, just add tooltip
                    self.add_bar_tooltip(bar, draggable_item)
            else:
                # Bar too small for any text, just add tooltip
                self.add_bar_tooltip(bar, draggable_item)
        else:
            # Extremely narrow bar (just a line), only tooltip
            self.add_bar_tooltip(bar, draggable_item)
        
        # Store reference to bar for hit testing
        bar.graphics_rect = draggable_item
        
        # Store in bars list
        self.bars.append(bar)

    def add_bar_tooltip(self, bar, draggable_item, text_item=None):
        """Add comprehensive tooltip to bar and text item"""
        # Create detailed tooltip
        duration = bar.end_time - bar.start_time
        tooltip_lines = [
            f"Label: {bar.label}",
            f"Time: {bar.start_time:.2f}s - {bar.end_time:.2f}s",
            f"Duration: {duration:.2f}s"
        ]
        
        # Add confidence if available
        if bar.confidence is not None:
            tooltip_lines.append(f"Confidence: {bar.confidence}/10")
        
        # Add metadata
        if bar.metadata:
            for key, value in bar.metadata.items():
                tooltip_lines.append(f"{key}: {value}")
        
        tooltip = "\n".join(tooltip_lines)
        
        # Set tooltip on draggable item
        draggable_item.setToolTip(tooltip)
        
        # Also set on text item if provided
        if text_item:
            text_item.setToolTip(tooltip)

            
    def draw_time_markers(self):
        """Draw time markers at regular intervals with improved formatting"""
        for second in range(0, int(self.video_duration) + 1, 5):
            x = second * self.pixels_per_second
            
            # Draw vertical line (darker for 30-second intervals)
            if second % 30 == 0:
                pen = QPen(QColor(100, 100, 150, 150), 1, Qt.PenStyle.SolidLine)
            else:
                pen = QPen(QColor(80, 80, 120, 80), 1, Qt.PenStyle.DashLine)
            self.addLine(x, 0, x, self.sceneRect().height(), pen)
            
            # Add time label for 30-second intervals
            if second % 30 == 0:
                minutes = second // 60
                secs = second % 60
                time_label = f"{minutes:02d}:{secs:02d}"
                
                text = self.addText(time_label, QFont("Consolas", 9))
                text.setPos(x + 5, self.sceneRect().height() - 25)
                text.setDefaultTextColor(QColor(200, 200, 200))
    
    def mousePressEvent(self, event):
        """Handle mouse clicks - emit time signal"""
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.scenePos()
            time = pos.x() / self.pixels_per_second
            self.time_clicked.emit(time)
            
            # If Ctrl is pressed, also request to add to edit timeline
            if event.modifiers() & Qt.ControlModifier:
                self.add_to_edit_requested.emit(time)
        
        super().mousePressEvent(event)
    
    def set_zoom(self, zoom_level):
        """Change zoom level (pixels per second)"""
        self.pixels_per_second = zoom_level
        self.build_timeline()
    
    def set_current_time(self, seconds):
        """Set current time indicator"""
        # Store the current time
        self.current_time_seconds = seconds
        
        # Remove old time line if it exists
        if hasattr(self, 'current_time_line') and self.current_time_line in self.items():
            self.removeItem(self.current_time_line)
        
        x = seconds * self.pixels_per_second
        # Make sure x is within scene bounds
        x = max(0, min(x, self.sceneRect().width() - 1))
        
        self.current_time_line = self.addLine(x, 0, x, self.sceneRect().height(),
                                            QPen(QColor(255, 60, 60), 2, Qt.PenStyle.DashLine))
        
    def set_confidence_filter(self, min_confidence, max_confidence=None):
        """Set confidence filter range (0.0 to 1.0)"""
        self.min_confidence = max(0.0, min(min_confidence, 1.0))
        if max_confidence is not None:
            self.max_confidence = min(1.0, max(max_confidence, 0.0))
        else:
            self.max_confidence = 1.0
        
        # Rebuild timeline with new filters
        self.build_timeline()
        
        # Emit filter change signal
        self.filter_changed.emit({
            'actions': self.visible_actions.copy(),
            'objects': self.visible_objects.copy(),
            'min_confidence': self.min_confidence,
            'max_confidence': self.max_confidence
        })
    
    def should_show_action(self, action_data):
        """Check if an action should be shown based on filters"""
        action_name = action_data.get('action_name') or action_data.get('action') or 'Unknown'
        action_name = action_name.strip().title()
        
        # Check if action type is visible
        if not self.visible_actions.get(action_name, True):
            return False
        
        # Check confidence filter
        confidence = action_data.get('confidence')
        if confidence is not None:
            # Normalize confidence to 0-1 scale
            if confidence > 1.0:  # Assuming 0-10 scale
                confidence = confidence / 10.0
            
            if confidence < self.min_confidence or confidence > self.max_confidence:
                return False
        
        return True
    
    def should_show_object(self, obj_data):
        """Check if an object should be shown based on filters"""
        # For objects, we check each object in the detection
        objects = obj_data.get('objects', [])
        timestamp = obj_data.get('timestamp', 0)
        
        # Check if any visible objects are in this detection
        for obj_name in objects:
            if isinstance(obj_name, str):
                obj_name = obj_name.strip().title()
                if obj_name in self.visible_objects and self.visible_objects[obj_name]:
                    # Check confidence if available
                    confidence = obj_data.get('confidence')
                    if confidence is not None:
                        # Normalize confidence to 0-1 scale
                        if confidence > 1.0:
                            confidence = confidence / 10.0
                        
                        if confidence >= self.min_confidence and confidence <= self.max_confidence:
                            return True
                    else:
                        # No confidence data, show it
                        return True
        
        return False


class ConfidenceFilterDialog(QDialog):
    """Dialog for filtering by confidence level"""
    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.setWindowTitle("Confidence Filter")
        self.setModal(False)
        self.resize(400, 300)
        self.init_ui()

    def init_ui(self):
        """Initialize the confidence filter UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Filter by Confidence Level")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #a0c0ff;")
        layout.addWidget(title)
        
        # Description
        desc = QLabel("Adjust the confidence range to filter actions and objects.\nConfidence range: 0.0 (low) to 1.0 (high)")
        desc.setStyleSheet("color: #cccccc; font-size: 11px;")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # Current range display
        self.range_label = QLabel(f"Current: {self.scene.min_confidence:.2f} - {self.scene.max_confidence:.2f}")
        self.range_label.setStyleSheet("font-weight: bold; color: #a0ffa0;")
        layout.addWidget(self.range_label)
        
        # Minimum confidence slider
        min_group = QGroupBox("Minimum Confidence")
        min_layout = QVBoxLayout()
        
        self.min_slider = QSlider(Qt.Horizontal)
        self.min_slider.setRange(0, 100)
        self.min_slider.setValue(int(self.scene.min_confidence * 100))
        self.min_slider.valueChanged.connect(self.on_slider_changed)
        
        self.min_value_label = QLabel(f"{self.scene.min_confidence:.2f}")
        self.min_value_label.setStyleSheet("color: #ffa0a0;")
        
        min_layout.addWidget(self.min_slider)
        min_layout.addWidget(self.min_value_label)
        min_group.setLayout(min_layout)
        layout.addWidget(min_group)
        
        # Maximum confidence slider
        max_group = QGroupBox("Maximum Confidence")
        max_layout = QVBoxLayout()
        
        self.max_slider = QSlider(Qt.Horizontal)
        self.max_slider.setRange(0, 100)
        self.max_slider.setValue(int(self.scene.max_confidence * 100))
        self.max_slider.valueChanged.connect(self.on_slider_changed)
        
        self.max_value_label = QLabel(f"{self.scene.max_confidence:.2f}")
        self.max_value_label.setStyleSheet("color: #ffa0a0;")
        
        max_layout.addWidget(self.max_slider)
        max_layout.addWidget(self.max_value_label)
        max_group.setLayout(max_layout)
        layout.addWidget(max_group)
        
        # Preset buttons
        preset_layout = QHBoxLayout()
        
        high_btn = QPushButton("High (0.7-1.0)")
        high_btn.clicked.connect(lambda: self.set_range(0.7, 1.0))
        
        medium_btn = QPushButton("Medium (0.4-0.7)")
        medium_btn.clicked.connect(lambda: self.set_range(0.4, 0.7))
        
        low_btn = QPushButton("Low (0.0-0.4)")
        low_btn.clicked.connect(lambda: self.set_range(0.0, 0.4))
        
        all_btn = QPushButton("All (0.0-1.0)")
        all_btn.clicked.connect(lambda: self.set_range(0.0, 1.0))
        
        preset_layout.addWidget(high_btn)
        preset_layout.addWidget(medium_btn)
        preset_layout.addWidget(low_btn)
        preset_layout.addWidget(all_btn)
        layout.addLayout(preset_layout)
        
        # Statistics
        self.stats_label = QLabel()
        self.stats_label.setStyleSheet("color: #cccccc; font-size: 10px; margin-top: 10px;")
        self.stats_label.setWordWrap(True)
        layout.addWidget(self.stats_label)
        
        # Update initial statistics
        self.update_statistics()
        
        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Apply | QDialogButtonBox.Close
        )
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply_filters)
        buttons.button(QDialogButtonBox.Close).clicked.connect(self.close)
        layout.addWidget(buttons)
        
        # Apply dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a2a;
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
            QPushButton {
                background-color: #2a2a44;
                color: white;
                border: 1px solid #4a4a6a;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #3a3a5c;
            }
        """)

    def on_slider_changed(self):
        """Update labels when sliders change"""
        min_val = self.min_slider.value() / 100.0
        max_val = self.max_slider.value() / 100.0
        
        # Ensure min <= max
        if min_val > max_val:
            self.max_slider.setValue(int(min_val * 100))
            max_val = min_val
        
        self.min_value_label.setText(f"{min_val:.2f}")
        self.max_value_label.setText(f"{max_val:.2f}")
        self.range_label.setText(f"Current: {min_val:.2f} - {max_val:.2f}")
        
        # Update statistics preview (without applying)
        self.update_statistics()

    def set_range(self, min_val, max_val):
        """Set range from preset"""
        self.min_slider.setValue(int(min_val * 100))
        self.max_slider.setValue(int(max_val * 100))
        self.on_slider_changed()

    def update_statistics(self):
        """Update filter statistics preview"""
        min_val = self.min_slider.value() / 100.0
        max_val = self.max_slider.value() / 100.0
        
        # Count items that would be visible with these settings
        total_actions = len(self.scene.cache_data.get('actions', []))
        total_objects = len(self.scene.cache_data.get('objects', []))
        
        visible_actions = 0
        for action in self.scene.cache_data.get('actions', []):
            confidence = action.get('confidence')
            if confidence is not None:
                if confidence > 1.0:
                    confidence = confidence / 10.0
                if min_val <= confidence <= max_val:
                    visible_actions += 1
            else:
                visible_actions += 1  # Count items without confidence
        
        visible_objects = 0
        for obj_data in self.scene.cache_data.get('objects', []):
            confidence = obj_data.get('confidence')
            if confidence is not None:
                if confidence > 1.0:
                    confidence = confidence / 10.0
                if min_val <= confidence <= max_val:
                    visible_objects += 1
            else:
                visible_objects += 1
        
        filtered_out = (total_actions + total_objects) - (visible_actions + visible_objects)
        
        stats_text = f"""
Statistics (Preview):
- Total actions: {total_actions} → Visible: {visible_actions}
- Total objects: {total_objects} → Visible: {visible_objects}
- Filtered out: {filtered_out} items
"""
        self.stats_label.setText(stats_text.strip())

    def apply_filters(self):
        """Apply the confidence filters"""
        min_val = self.min_slider.value() / 100.0
        max_val = self.max_slider.value() / 100.0
        
        self.scene.set_confidence_filter(min_val, max_val)
        
        # Update statistics with actual results
        self.update_statistics()


class SignalTimelineView(QGraphicsView):
    """Custom view with smooth zooming and panning"""
    
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        
        # Start with no drag mode (we'll handle it manually)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        
        # Semi-transparent background
        self.setStyleSheet("""
            QGraphicsView {
                background-color: rgba(18, 18, 24, 200);
                border: 2px solid rgba(100, 100, 150, 150);
                border-radius: 5px;
            }
        """)
        
        # Enable mouse tracking for better drag experience
        self.setMouseTracking(True)
        
        # Track mouse state for manual panning
        self.panning = False
        self.last_pan_point = QPoint()
    
    def wheelEvent(self, event):
        """Zoom with mouse wheel"""
        zoom_factor = 1.15
        
        # Store scene position before zoom
        old_pos = self.mapToScene(event.position().toPoint())
        
        if event.angleDelta().y() > 0:
            self.scale(zoom_factor, 1.0)
        else:
            self.scale(1.0 / zoom_factor, 1.0)
        
        # Get new position after zoom
        new_pos = self.mapToScene(event.position().toPoint())
        
        # Calculate offset to keep mouse position stable
        delta = new_pos - old_pos
        
        # Adjust view to maintain mouse position
        self.horizontalScrollBar().setValue(
            self.horizontalScrollBar().value() + int(delta.x())
        )
        
        event.accept()
    
    def mousePressEvent(self, event):
        """Handle mouse press in the view"""
        # Check if we're clicking on a draggable item
        item = self.itemAt(event.pos())
        
        if item and isinstance(item, DraggableTimelineBar):
            # Let the item handle the mouse press
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            super().mousePressEvent(event)
            return
        
        # If right-click or middle-click, use scroll hand drag
        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            super().mousePressEvent(event)
            return
        
        # Left-click on background - manual panning
        if event.button() == Qt.LeftButton:
            self.panning = True
            self.last_pan_point = event.pos()
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            event.accept()
            return
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse movement for panning"""
        if self.panning:
            delta = event.pos() - self.last_pan_point
            self.last_pan_point = event.pos()
            
            # Scroll the view
            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            hbar.setValue(hbar.value() - delta.x())
            vbar.setValue(vbar.value() - delta.y())
            
            event.accept()
            return
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release"""
        if event.button() == Qt.LeftButton and self.panning:
            self.panning = False
            self.setCursor(QCursor(Qt.ArrowCursor))
            event.accept()
            return
        
        # Reset drag mode if it was set
        if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        
        super().mouseReleaseEvent(event)


class FilterDialog(QDialog):
    """Dialog for filtering actions and objects"""
    
    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.setWindowTitle("Filter Actions & Objects")
        self.setModal(False)  # Non-modal so users can keep it open
        self.resize(500, 600)
        
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create tab widget
        tabs = QTabWidget()
        
        # Actions tab
        actions_tab = QWidget()
        actions_layout = QVBoxLayout(actions_tab)
        
        # Actions filter controls
        actions_header = QLabel("Filter Actions")
        actions_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #a0c0ff;")
        actions_layout.addWidget(actions_header)
        
        # Search box for actions
        self.action_search = QLineEdit()
        self.action_search.setPlaceholderText("Search actions...")
        self.action_search.textChanged.connect(self.filter_action_list)
        actions_layout.addWidget(self.action_search)
        
        # Actions list
        self.action_list = QListWidget()
        self.action_list.setSelectionMode(QListWidget.MultiSelection)
        self.populate_action_list()
        actions_layout.addWidget(self.action_list)
        
        # Action buttons
        action_buttons = QHBoxLayout()
        self.select_all_actions = QPushButton("Select All")
        self.select_all_actions.clicked.connect(lambda: self.set_all_actions(True))
        self.deselect_all_actions = QPushButton("Deselect All")
        self.deselect_all_actions.clicked.connect(lambda: self.set_all_actions(False))
        
        action_buttons.addWidget(self.select_all_actions)
        action_buttons.addWidget(self.deselect_all_actions)
        actions_layout.addLayout(action_buttons)
        
        tabs.addTab(actions_tab, "Actions")
        
        # Objects tab
        objects_tab = QWidget()
        objects_layout = QVBoxLayout(objects_tab)
        
        # Objects filter controls
        objects_header = QLabel("Filter Objects")
        objects_header.setStyleSheet("font-weight: bold; font-size: 14px; color: #ffa0a0;")
        objects_layout.addWidget(objects_header)
        
        # Search box for objects
        self.object_search = QLineEdit()
        self.object_search.setPlaceholderText("Search objects...")
        self.object_search.textChanged.connect(self.filter_object_list)
        objects_layout.addWidget(self.object_search)
        
        # Objects list
        self.object_list = QListWidget()
        self.object_list.setSelectionMode(QListWidget.MultiSelection)
        self.populate_object_list()
        objects_layout.addWidget(self.object_list)
        
        # Object buttons
        object_buttons = QHBoxLayout()
        self.select_all_objects = QPushButton("Select All")
        self.select_all_objects.clicked.connect(lambda: self.set_all_objects(True))
        self.deselect_all_objects = QPushButton("Deselect All")
        self.deselect_all_objects.clicked.connect(lambda: self.set_all_objects(False))
        
        object_buttons.addWidget(self.select_all_objects)
        object_buttons.addWidget(self.deselect_all_objects)
        objects_layout.addLayout(object_buttons)
        
        tabs.addTab(objects_tab, "Objects")
        
        layout.addWidget(tabs)
        
        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Apply | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.apply_and_close)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply_filters)
        buttons.rejected.connect(self.reject)
        
        layout.addWidget(buttons)
        
        # Apply dark theme
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a2a;
            }
            QListWidget {
                background-color: #2a2a3a;
                color: #ffffff;
                border: 1px solid #444466;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #333344;
            }
            QListWidget::item:selected {
                background-color: #3a5fcd;
            }
            QListWidget::item:hover {
                background-color: #3a3a5a;
            }
            QLineEdit {
                background-color: #2a2a3a;
                color: #ffffff;
                border: 1px solid #444466;
                border-radius: 4px;
                padding: 5px;
            }
            QTabWidget::pane {
                border: 1px solid #444466;
                background-color: #2a2a3a;
            }
            QTabBar::tab {
                background-color: #3a3a5a;
                color: #cccccc;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #4a5fcd;
                color: #ffffff;
            }
        """)
    
    def populate_action_list(self):
        """Populate action list with all action types"""
        self.action_list.clear()
        for action in sorted(self.scene.action_types):
            item = QListWidgetItem(action)
            item.setCheckState(Qt.Checked if self.scene.visible_actions.get(action, True) else Qt.Unchecked)
            self.action_list.addItem(item)
    
    def populate_object_list(self):
        """Populate object list with all object types"""
        self.object_list.clear()
        for obj in sorted(self.scene.object_classes):
            item = QListWidgetItem(obj)
            item.setCheckState(Qt.Checked if self.scene.visible_objects.get(obj, True) else Qt.Unchecked)
            self.object_list.addItem(item)
    
    def filter_action_list(self):
        """Filter action list based on search text"""
        search_text = self.action_search.text().lower()
        for i in range(self.action_list.count()):
            item = self.action_list.item(i)
            item.setHidden(search_text not in item.text().lower())
    
    def filter_object_list(self):
        """Filter object list based on search text"""
        search_text = self.object_search.text().lower()
        for i in range(self.object_list.count()):
            item = self.object_list.item(i)
            item.setHidden(search_text not in item.text().lower())
    
    def set_all_actions(self, selected):
        """Select or deselect all actions"""
        for i in range(self.action_list.count()):
            item = self.action_list.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Checked if selected else Qt.Unchecked)
    
    def set_all_objects(self, selected):
        """Select or deselect all objects"""
        for i in range(self.object_list.count()):
            item = self.object_list.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Checked if selected else Qt.Unchecked)
    
    def apply_filters(self):
        """Apply the selected filters to the scene"""
        # Update action filters
        for i in range(self.action_list.count()):
            item = self.action_list.item(i)
            action_name = item.text()
            self.scene.set_action_filter(action_name, item.checkState() == Qt.Checked)
        
        # Update object filters
        for i in range(self.object_list.count()):
            item = self.object_list.item(i)
            object_name = item.text()
            self.scene.set_object_filter(object_name, item.checkState() == Qt.Checked)
    
    def apply_and_close(self):
        """Apply filters and close dialog"""
        self.apply_filters()
        self.accept()


class SignalTimelineWindow(QMainWindow):
    """Main window for signal timeline viewer with edit timeline and filters"""
    waveform_ready = Signal(object)
    
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
        self.preview_window = TimelineWithPreview.launch_preview(self)

    def closeEvent(self, event):
        """Close preview when timeline closes"""
        if hasattr(self, 'preview_window') and self.preview_window:
            self.preview_window.close()
        super().closeEvent(event)

    def create_video_preview_dock(self):
        """Create dock with video preview player using Qt's built-in player"""
        from PySide6.QtCore import Qt, QUrl
        from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
        from PySide6.QtMultimediaWidgets import QVideoWidget
        
        # Create the dock widget
        dock = QDockWidget("Video Preview", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        # Create the main widget for the dock
        preview_widget = QWidget()
        layout = QVBoxLayout(preview_widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        
        # Preview label
        preview_label = QLabel("🎬 Video Preview")
        preview_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        preview_label.setStyleSheet("color: #a0c0ff; padding: 4px;")
        layout.addWidget(preview_label)
        
        # Create video widget
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumSize(400, 300)
        self.video_widget.setStyleSheet("""
            QVideoWidget {
                background-color: black;
                border: 2px solid #3a3a5a;
                border-radius: 6px;
            }
        """)
        layout.addWidget(self.video_widget)
        
        # Create media player
        self.video_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.video_player.setAudioOutput(self.audio_output)
        self.video_player.setVideoOutput(self.video_widget)
        
        # Load the video
        video_url = QUrl.fromLocalFile(self.video_path)
        self.video_player.setSource(video_url)
        
        # Create controls widget
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 8, 0, 0)
        
        # Play/Pause button
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.toggle_video_playback)
        self.play_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a5fcd;
                color: white;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #4a6fdd;
            }
        """)
        controls_layout.addWidget(self.play_btn)
        
        # Time slider
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 100)
        self.time_slider.sliderMoved.connect(self.seek_video)
        controls_layout.addWidget(self.time_slider)
        
        # Time label
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("""
            QLabel {
                color: #a0ffa0;
                font-family: 'Consolas', monospace;
                font-weight: bold;
                padding: 8px;
                background-color: #1a1a2a;
                border-radius: 4px;
                min-width: 120px;
                qproperty-alignment: AlignCenter;
            }
        """)
        controls_layout.addWidget(self.time_label)
        
        controls_layout.addStretch()
        
        # Volume slider
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("🔊"))
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.valueChanged.connect(self.set_volume)
        self.volume_slider.setFixedWidth(80)
        volume_layout.addWidget(self.volume_slider)
        controls_layout.addLayout(volume_layout)
        
        # Add controls to main layout
        layout.addWidget(controls_widget)
        
        # Connect video player signals
        self.video_player.durationChanged.connect(self.update_video_duration)
        self.video_player.positionChanged.connect(self.update_video_position)
        self.video_player.playbackStateChanged.connect(self.update_play_button)
        
        # Set initial volume
        self.audio_output.setVolume(0.8)
        
        dock.setWidget(preview_widget)
        return dock

    def toggle_video_playback(self):
        """Toggle video playback in the preview"""
        if self.video_player.playbackState() == QMediaPlayer.PlayingState:
            self.video_player.pause()
            self.play_btn.setText("▶ Play")
        else:
            self.video_player.play()
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
        """Update video position display"""
        if self.video_player.duration() > 0:
            # Update slider
            percent = (position / self.video_player.duration()) * 100
            self.time_slider.blockSignals(True)
            self.time_slider.setValue(int(percent))
            self.time_slider.blockSignals(False)
            
            # Update time label
            self.update_time_display(position)
            
            # Sync with timeline if playing
            if self.video_player.playbackState() == QMediaPlayer.PlayingState:
                self.current_time = position / 1000.0
                self.signal_scene.set_current_time(self.current_time)

    def update_time_display(self, position):
        """Update the time display label"""
        current_seconds = position // 1000
        mins = current_seconds // 60
        secs = current_seconds % 60
        current_time_str = f"{mins:02d}:{secs:02d}"
        
        if hasattr(self, 'total_duration_str'):
            self.time_label.setText(f"{current_time_str} / {self.total_duration_str}")
        else:
            self.time_label.setText(f"{current_time_str}")

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
        
        # IMPORTANT: Always pass current waveform data (might be None initially)
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
        self.edit_view.setStyleSheet("""
            QGraphicsView {
                background-color: rgba(30, 30, 40, 200);
                border: 2px solid rgba(100, 100, 150, 150);
                border-radius: 5px;
            }
        """)
        
        # Set focus policy to receive key events
        self.edit_view.setFocusPolicy(Qt.StrongFocus)
        
        # Connect edit timeline signals
        self.edit_scene.clip_double_clicked.connect(self.on_clip_double_clicked)
        self.edit_scene.clip_added.connect(self.on_clip_added)
        self.edit_scene.clip_removed.connect(self.on_clip_removed)
        
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


        # Apply dark theme
        self.apply_dark_theme()
        
        # Status bar
        self.statusBar().showMessage(f"Video duration: {self.video_duration:.1f}s | Total edit duration: {self.edit_scene.get_total_duration():.1f}s")
        
        # Install event filter to handle global key events
        self.installEventFilter(self)

    def eventFilter(self, obj, event):
        """Global event filter for handling delete key"""
        if event.type() == event.Type.KeyPress:
            if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
                # Check if edit view has focus or if delete is pressed globally
                if (obj == self or 
                    (hasattr(self, 'edit_view') and self.edit_view.hasFocus()) or
                    (hasattr(self, 'edit_scene') and len(self.edit_scene.selectedItems()) > 0)):
                    
                    # Remove selected clips
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
        instructions = QLabel("🖱️ Drag bars to edit timeline • Select clips and press Delete to remove")
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
        
        # Add clip button
        self.add_clip_btn = QPushButton("➕ Add Clip at Current Time")
        self.add_clip_btn.clicked.connect(self.on_add_clip_clicked)
        
        # Remove selected clips button
        self.remove_clips_btn = QPushButton("🗑️ Delete Selected Clips")
        self.remove_clips_btn.clicked.connect(self.on_remove_clips_clicked)
        
        # Save to cache button - ADD THIS
        self.save_cache_btn = QPushButton("💾 Save to Cache")
        self.save_cache_btn.clicked.connect(self.on_save_cache_clicked)  # CONNECT IT!
        self.save_cache_btn.setToolTip("Save current edit timeline to cache for future use")
        
        # Export button
        self.export_btn = QPushButton("📤 Export Edit")
        self.export_btn.clicked.connect(self.on_export_clicked)
        
        # Duration label
        self.edit_duration_label = QLabel("Edit duration: 0.0s")
        self.edit_duration_label.setStyleSheet("color: #a0ffa0; font-weight: bold;")
        
        layout.addWidget(self.add_clip_btn)
        layout.addWidget(self.remove_clips_btn)
        layout.addWidget(self.save_cache_btn)  # ADD TO LAYOUT
        layout.addWidget(self.export_btn)
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
        self.current_time = max(0, min(self.video_duration, time))
        
        # Update signal scene
        self.signal_scene.current_time_seconds = self.current_time
        self.signal_scene.set_current_time(self.current_time)
        
        # Seek video player to this time
        if hasattr(self, 'video_player'):
            milliseconds = int(self.current_time * 1000)
            self.video_player.setPosition(milliseconds)
        
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
    def on_clip_added(self, start_time, end_time):
        """Handle when a clip is added to edit timeline"""
        self.update_edit_duration()
        self.statusBar().showMessage(f"Added clip: {start_time:.1f}s to {end_time:.1f}s", 2000)
    
    @Slot(int)
    def on_clip_removed(self, index):
        """Handle when a clip is removed from edit timeline"""
        # Add to pending removals
        self.pending_clip_removals.append(index)
        
        # Start or restart the timer
        self.removal_timer.start(100)  # 100ms delay

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
    
    @Slot()
    def play_video_at_current_time(self):
        """Play video at current time IN PREVIEW WINDOW"""
        if not hasattr(self, 'current_time') or self.current_time < 0:
            self.statusBar().showMessage("⚠️ Click timeline to select a timestamp first", 2000)
            return
        
        # Play in the preview window
        self.play_video_time(self.current_time)

        
        # Seek to current time
        milliseconds = int(self.current_time * 1000)
        self.video_player.setPosition(milliseconds)
        
        # Play the video
        self.video_player.play()
        self.play_btn.setText("⏸ Pause")
        
        self.statusBar().showMessage(f"▶ Playing at {self.current_time:.1f}s", 2000)


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
