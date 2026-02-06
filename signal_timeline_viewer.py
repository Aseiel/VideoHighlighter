"""
Complete Signal Timeline Viewer with Filters and Edit Timeline
- Signal visualization with filtering
- Edit timeline with clip management
- Action/object filtering
- Exact time playback
"""

import sys
import os
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
from PySide6.QtCore import Qt, QRectF, Signal, Slot, QPointF, QTimer, QPoint
from PySide6.QtGui import (
    QColor, QPen, QBrush, QPainter, QFont, QPainterPath, 
    QLinearGradient, QRadialGradient, QCursor, QAction,
    QPainterPath, QFontMetrics
)
import subprocess


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
        self.confidence = confidence
        self.metadata = metadata or {}
    
    def get_alpha(self):
        """Get transparency based on confidence"""
        if self.confidence is not None:
            # Map confidence 0-10 to alpha 100-255
            return int(100 + (self.confidence / 10.0) * 155)
        return 180  # Default semi-transparent


class EditClipItem(QGraphicsRectItem):
    """Represents a clip in the edit timeline"""
    def __init__(self, start_time, end_time, y, height, color, index):
        super().__init__()
        self.start_time = start_time
        self.end_time = end_time
        self.color = color
        self.index = index
        self.is_selected = False
        
        # Set rectangle properties
        self.setRect(0, 0, 0, height)
        self.setPos(0, y)
        self.setBrush(QBrush(color))
        self.setPen(QPen(color.darker(150), 1))
        
        # Make it movable and selectable
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        
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
        """Handle mouse press"""
        self.set_selected(True)
        super().mousePressEvent(event)
    
    def mouseDoubleClickEvent(self, event):
        """Handle double click - play this clip"""
        if event.button() == Qt.LeftButton:
            # Emit signal to play this clip
            scene = self.scene()
            if hasattr(scene, 'clip_double_clicked'):
                scene.clip_double_clicked.emit(self.start_time, self.end_time)
        super().mouseDoubleClickEvent(event)


class EditTimelineScene(QGraphicsScene):
    """Simple timeline showing clips as colored rectangles"""
    
    clip_double_clicked = Signal(float, float)  # start, end
    clip_added = Signal(float, float)  # start, end
    clip_removed = Signal(int)  # index
    
    def __init__(self, video_path, video_duration, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.video_duration = video_duration
        self.clips = []  # List of (start_time, end_time) tuples
        self.clip_items = []  # List of EditClipItem objects
        self.pixels_per_second = 50
        self.clip_height = 60
        self.clip_spacing = 5
        
        # Load initial highlights if available
        self.load_initial_clips()
        
        self.setSceneRect(0, 0, 1000, self.clip_height + 40)
        self.build_timeline()
    
    def load_initial_clips(self):
        """Load initial clips from highlights in cache (simulated for now)"""
        # In a real implementation, this would load from cache
        # For now, create some example clips
        if self.video_duration > 30:
            # Create 3 sample clips at interesting times
            self.clips = [
                (5.0, 12.0),
                (25.0, 35.0),
                (45.0, 55.0)
            ]
    
    def build_timeline(self):
        """Build the edit timeline visualization"""
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
    
    def remove_selected_clips(self):
        """Remove selected clips"""
        selected_items = [item for item in self.clip_items if item.is_selected]
        
        for item in selected_items:
            if item in self.clip_items:
                index = self.clip_items.index(item)
                self.clip_items.remove(item)
                if 0 <= index < len(self.clips):
                    self.clips.pop(index)
        
        self.build_timeline()
    
    def get_total_duration(self):
        """Get total duration of all clips"""
        return sum(end - start for start, end in self.clips)
    
    def get_clip_times(self):
        """Get list of all clip time ranges"""
        return self.clips.copy()


class SignalTimelineScene(QGraphicsScene):
    """Improved graphics scene with filtering capabilities"""
    
    time_clicked = Signal(float)
    add_to_edit_requested = Signal(float)
    filter_changed = Signal(dict)  # New: emit when filters change
    
    def __init__(self, cache_data, video_duration, parent=None):
        super().__init__(parent)
        self.cache_data = cache_data
        self.video_duration = max(video_duration, 1.0)
        
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
        """Build all timeline elements with improved layout"""
        self.clear()
        self.bars = []
        
        # Set scene size
        width = self.video_duration * self.pixels_per_second
        height = 100  # Base height for time ruler
        
        # Calculate height based on visible layers
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
        
        # Draw background with subtle grid
        self.draw_background()
        
        # Draw layers
        current_y = 30
        
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
            action_name = action.get('action_name') or action.get('action') or 'Unknown'
            action_name = action_name.strip().title()
            if action_name in self.visible_actions and self.visible_actions[action_name]:
                action_groups[action_name].append(action)
        
        # If no visible actions, still show the layer but empty
        if not action_groups:
            # Show "No visible actions" message
            text = self.addText("(filtered out)", QFont("Arial", 9))
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
                confidence = action.get('confidence', 0)
                
                bar = TimelineBar(
                    timestamp, timestamp + 0.5,
                    current_type_y, type_height,
                    color, action_type,
                    confidence=confidence * 10,
                    metadata={'type': action_type, 'confidence': confidence}
                )
                self.draw_bar(bar)
                self.bars.append(bar)
        
        return y_pos + self.layer_height + self.layer_spacing
    
    def draw_improved_objects_layer(self, y_pos):
        """Draw object detections organized by class with filtering"""
        label = self.addText("OBJECTS", QFont("Arial", 10, QFont.Weight.Bold))
        label.setPos(5, y_pos - 20)
        label.setDefaultTextColor(QColor(180, 220, 255))
        
        # Group objects by class
        object_groups = defaultdict(list)
        for obj_data in self.cache_data.get('objects', []):
            timestamp = obj_data.get('timestamp', 0)
            for obj_name in obj_data.get('objects', []):
                if isinstance(obj_name, str):
                    obj_name = obj_name.strip().title()
                    if obj_name in self.visible_objects and self.visible_objects[obj_name]:
                        object_groups[obj_name].append(timestamp)
        
        # If no visible objects, still show the layer but empty
        if not object_groups:
            # Show "No visible objects" message
            text = self.addText("(filtered out)", QFont("Arial", 9))
            text.setPos(150, y_pos + 15)
            text.setDefaultTextColor(QColor(150, 150, 150))
            return y_pos + self.layer_height + self.layer_spacing
        
        # Calculate y offset for each object type
        type_height = self.layer_height // max(1, len(object_groups))
        current_type_y = y_pos
        
        for idx, (obj_type, timestamps) in enumerate(sorted(object_groups.items())):
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
            for timestamp in timestamps:
                bar = TimelineBar(
                    timestamp, timestamp + 0.3,
                    current_type_y, type_height,
                    color, obj_type,
                    confidence=5,
                    metadata={'type': obj_type, 'count': len(timestamps)}
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
            for timestamp in self.cache_data['motion_events']:
                x = timestamp * self.pixels_per_second
                # Draw vertical line with varying height based on intensity
                pen = QPen(self.colors['motion_events'], 2)
                self.addLine(x, y_pos, x, y_pos + self.layer_height, pen)
        
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
        """Draw a single timeline bar with gradient and transparency"""
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
        
        # Draw rectangle with rounded corners
        rect = self.addRect(x, bar.y_position, width, bar.height,
                           QPen(color.darker(120), 1), QBrush(gradient))
        
        # Add text label if wide enough
        if width > 40:
            text = self.addText(bar.label, QFont("Arial", 8))
            text.setPos(x + 5, bar.y_position + 5)
            text.setDefaultTextColor(QColor(255, 255, 255, 200))
            
            # Add tooltip with metadata
            if bar.metadata:
                tooltip = "\n".join([f"{k}: {v}" for k, v in bar.metadata.items()])
                text.setToolTip(tooltip)
                rect.setToolTip(tooltip)
    
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
        if hasattr(self, 'current_time_line'):
            self.removeItem(self.current_time_line)
        
        x = seconds * self.pixels_per_second
        self.current_time_line = self.addLine(x, 0, x, self.sceneRect().height(),
                                             QPen(QColor(255, 60, 60), 2, Qt.PenStyle.DashLine))


class SignalTimelineView(QGraphicsView):
    """Custom view with smooth zooming and panning"""
    
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        
        # Semi-transparent background
        self.setStyleSheet("""
            QGraphicsView {
                background-color: rgba(18, 18, 24, 200);
                border: 2px solid rgba(100, 100, 150, 150);
                border-radius: 5px;
            }
        """)
    
    def wheelEvent(self, event):
        """Zoom with mouse wheel"""
        zoom_factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(zoom_factor, 1.0)
        else:
            self.scale(1.0 / zoom_factor, 1.0)


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
    
    def __init__(self, video_path, cache_data=None):
        super().__init__()
        self.video_path = video_path
        self.cache_data = cache_data or self.load_cache_data()
        
        if not self.cache_data:
            print("❌ No cache data available")
            self.close()
            return
        
        self.video_duration = self.cache_data.get('video_metadata', {}).get('duration', 0)
        self.current_time = 0
        
        # Extract info for display
        self.action_types = self._extract_action_types()
        self.object_classes = self._extract_object_classes()
        
        self.setWindowTitle(f"Signal Timeline - {os.path.basename(video_path)}")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Make window semi-transparent
        self.setWindowOpacity(0.98)
        
        self.init_ui()
    
    def load_cache_data(self):
        """Load cache data for the video"""
        try:
            from modules.video_cache import VideoAnalysisCache
            cache = VideoAnalysisCache()
            return cache.load(self.video_path)
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return None
    
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
        
        # Signal timeline
        self.signal_scene = SignalTimelineScene(self.cache_data, self.video_duration)
        self.signal_view = SignalTimelineView(self.signal_scene)
        self.signal_scene.time_clicked.connect(self.on_time_clicked)
        self.signal_scene.add_to_edit_requested.connect(self.on_add_to_edit_requested)
        self.signal_scene.filter_changed.connect(self.on_filter_changed)
        
        signal_layout.addWidget(QLabel("Signal Timeline (Hold Ctrl+Click to add clip)"))
        signal_layout.addWidget(self.signal_view)
        
        splitter.addWidget(signal_widget)
        
        # Create edit timeline view (bottom)
        edit_widget = QWidget()
        edit_layout = QVBoxLayout(edit_widget)
        
        # Edit timeline
        self.edit_scene = EditTimelineScene(self.video_path, self.video_duration)
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
        
        # Connect edit timeline signals
        self.edit_scene.clip_double_clicked.connect(self.on_clip_double_clicked)
        self.edit_scene.clip_added.connect(self.on_clip_added)
        
        edit_layout.addWidget(QLabel("Edit Timeline (Double-click clip to play)"))
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
        
        # Apply dark theme
        self.apply_dark_theme()
        
        # Status bar
        self.statusBar().showMessage(f"Video duration: {self.video_duration:.1f}s | Total edit duration: {self.edit_scene.get_total_duration():.1f}s")
    
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
        
        # Open filter dialog button
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
        self.add_clip_btn.setToolTip("Add a 5-second clip at the currently selected time")
        
        # Remove selected clips button
        self.remove_clips_btn = QPushButton("🗑️ Remove Selected Clips")
        self.remove_clips_btn.clicked.connect(self.on_remove_clips_clicked)
        self.remove_clips_btn.setToolTip("Remove selected clips from edit timeline")
        
        # Export button
        self.export_btn = QPushButton("📤 Export Edit")
        self.export_btn.clicked.connect(self.on_export_clicked)
        self.export_btn.setToolTip("Export the edited timeline")
        
        # Duration label
        self.edit_duration_label = QLabel("Edit duration: 0.0s")
        self.edit_duration_label.setStyleSheet("color: #a0ffa0; font-weight: bold;")
        
        layout.addWidget(self.add_clip_btn)
        layout.addWidget(self.remove_clips_btn)
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
        self.signal_scene.set_current_time(self.current_time)
        
        # Update label
        minutes = int(self.current_time // 60)
        seconds = int(self.current_time % 60)
        milliseconds = int((self.current_time % 1) * 1000)
        self.time_label.setText(f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}")
    
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
        """Remove selected clips"""
        before_count = len(self.edit_scene.clips)
        self.edit_scene.remove_selected_clips()
        after_count = len(self.edit_scene.clips)
        
        removed = before_count - after_count
        if removed > 0:
            self.update_edit_duration()
            self.statusBar().showMessage(f"Removed {removed} clip(s)", 2000)
        else:
            self.statusBar().showMessage("No clips selected", 2000)
    
    @Slot()
    def on_export_clicked(self):
        """Export the edit timeline"""
        if len(self.edit_scene.clips) == 0:
            QMessageBox.warning(self, "No Clips", "Add some clips to the edit timeline first!")
            return
        
        # In a real implementation, this would export the video
        # For now, just show a message with the edit plan
        clip_info = "\n".join([f"Clip {i+1}: {start:.1f}s to {end:.1f}s (duration: {end-start:.1f}s)" 
                              for i, (start, end) in enumerate(self.edit_scene.clips)])
        
        QMessageBox.information(self, "Export Edit", 
                               f"Edit Timeline:\n{clip_info}\n\nTotal duration: {self.edit_scene.get_total_duration():.1f}s\n\n"
                               f"Export functionality will be implemented in a future version.")
    
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
        """Show all actions and objects"""
        if hasattr(self, 'signal_scene'):
            self.signal_scene.set_all_actions_visible(True)
            self.signal_scene.set_all_objects_visible(True)
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
        """Open video at the selected timestamp with exact time"""
        if not hasattr(self, 'current_time') or self.current_time < 0:
            self.statusBar().showMessage("⚠️ Click timeline to select a timestamp first", 2000)
            return
        
        self.play_video_time(self.current_time)
    
    def play_video_clip(self, start_time, end_time):
        """Play a specific clip"""
        duration = end_time - start_time
        self.statusBar().showMessage(f"Playing clip: {start_time:.1f}s for {duration:.1f}s", 3000)
        
        # Play from start_time
        self.play_video_time(start_time)
    
    def play_video_time(self, time):
        """Play video at specific time"""
        exact_time = time
        
        # Try different players
        players_to_try = [
            ('mpv', ['mpv', f'--start={exact_time}', self.video_path]),
            ('vlc', ['vlc', '--start-time', str(exact_time), self.video_path]),
            ('ffplay', ['ffplay', '-ss', str(exact_time), '-autoexit', self.video_path]),
        ]
        
        for player_name, command in players_to_try:
            try:
                # Check if player exists
                if sys.platform == 'win32':
                    check_cmd = ['where', player_name]
                else:
                    check_cmd = ['which', player_name]
                
                result = subprocess.run(check_cmd, capture_output=True)
                if result.returncode == 0:
                    # Launch player
                    subprocess.Popen(
                        command,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                    )
                    
                    self.statusBar().showMessage(f"▶ Playing with {player_name} at {exact_time:.2f}s", 3000)
                    return
                    
            except Exception as e:
                continue
        
        self.statusBar().showMessage("⚠️ Install mpv, VLC, or ffplay for video playback", 4000)
    
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


def show_timeline_viewer(video_path, cache_data=None):
    """
    Launch the signal timeline viewer with edit timeline
    
    Args:
        video_path: Path to video file
        cache_data: Optional cache data dict (will load from cache if not provided)
    
    Returns:
        int: Application exit code
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    window = SignalTimelineWindow(video_path, cache_data)
    window.show()
    
    return app.exec()


if __name__ == "__main__":
    # Test with a video file
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        show_timeline_viewer(video_path)
    else:
        print("Usage: python signal_timeline_viewer.py <video_path>")