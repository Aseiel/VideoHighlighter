from .timeline_bars import TimelineBar, DraggableTimelineBar
from collections import defaultdict
from PySide6.QtWidgets import (
    QGraphicsScene, QGraphicsView, QGraphicsTextItem,
    QGraphicsLineItem, QApplication, QMenu
)
from PySide6.QtCore import Qt, QRectF, Signal, Slot, QPointF, QTimer, QPoint, QMimeData
from PySide6.QtGui import (
    QColor, QPen, QBrush, QFont, QLinearGradient,
    QFontMetrics, QCursor, QPainter, QDrag, QPixmap
)

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

        # Range selection state
        self._selection_rect_item = None   # QGraphicsRectItem — the blue highlight
        self._selection_label_item = None  # QGraphicsTextItem — time label
        self._selection_start_time = None  # float seconds
        self._selection_end_time = None    # float seconds
        self._selection_active = False     # True = selection exists and can be dragged

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
        
        # Clear selection state — items are about to be wiped by self.clear()
        self._selection_rect_item  = None
        self._selection_label_item = None
        self._selection_active     = False

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
            height += 80 + self.layer_spacing  # Waveform height with spacing after waveform
        
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
        
        # Draw background
        self.draw_background()
        
        # Start drawing below time markers
        current_y = 40
        
        # Draw waveform if visible
        if self.waveform_visible and self.waveform:
            current_y = self.draw_waveform_layer(current_y, 80)
               
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

        # Restore playhead
        if hasattr(self, 'current_time_seconds'):
            self.set_current_time(self.current_time_seconds)
        
        # Restore view zoom/scroll
        if views and old_transform:
            view = views[0]
            old_visible_width = self.sceneRect().width() / old_transform.m11()
            if abs(old_visible_width - width) > 10:
                scale_factor = width / old_visible_width
                view.setTransform(old_transform.scale(scale_factor, 1.0))
                view.horizontalScrollBar().setValue(old_h_scroll)

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

        for timestamp in self.cache_data.get('motion_events', []):
            bar = TimelineBar(
                timestamp, timestamp + 0.5,
                y_pos, self.layer_height,
                self.colors['motion_events'], "Motion",
                confidence=7,
                metadata={'timestamp': timestamp}
            )
            self.draw_bar(bar)
            self.bars.append(bar)

        return y_pos + self.layer_height + self.layer_spacing
   
    def draw_motion_peaks_layer(self, y_pos):
        """Draw motion peaks"""
        label = self.addText("MOTION PEAKS", QFont("Arial", 10, QFont.Weight.Bold))
        label.setPos(5, y_pos - 20)
        label.setDefaultTextColor(QColor(180, 220, 255))

        for timestamp in self.cache_data.get('motion_peaks', []):
            bar = TimelineBar(
                timestamp, timestamp + 0.5,
                y_pos, self.layer_height,
                self.colors['motion_peaks'], "Peak",
                confidence=9,
                metadata={'timestamp': timestamp}
            )
            self.draw_bar(bar)
            self.bars.append(bar)

        return y_pos + self.layer_height + self.layer_spacing
    
    def draw_audio_peaks_layer(self, y_pos):
        """Draw audio peaks"""
        label = self.addText("AUDIO PEAKS", QFont("Arial", 10, QFont.Weight.Bold))
        label.setPos(5, y_pos - 20)
        label.setDefaultTextColor(QColor(180, 220, 255))

        for timestamp in self.cache_data.get('audio_peaks', []):
            bar = TimelineBar(
                timestamp, timestamp + 0.5,
                y_pos, self.layer_height,
                self.colors['audio_peaks'], "Audio",
                confidence=8,
                metadata={'timestamp': timestamp}
            )
            self.draw_bar(bar)
            self.bars.append(bar)

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
      
    def set_zoom(self, zoom_level):
        """Change zoom level (pixels per second)"""
        self.pixels_per_second = zoom_level
        self.build_timeline()
    
    def set_current_time(self, seconds):
        """Set current time indicator — moves existing line instead of recreating"""
        self.current_time_seconds = seconds
        
        x = seconds * self.pixels_per_second
        x = max(0, min(x, self.sceneRect().width() - 1))
        h = self.sceneRect().height()
        
        # Move existing line or create if missing
        if hasattr(self, 'current_time_line') and self.current_time_line in self.items():
            self.current_time_line.setLine(x, 0, x, h)
        else:
            self.current_time_line = self.addLine(
                x, 0, x, h,
                QPen(QColor(255, 60, 60), 2, Qt.PenStyle.DashLine)
            )
            self.current_time_line.setZValue(100)

    def update_selection_rect(self, start_time: float, end_time: float):
        """
        Draw or update the blue selection highlight while the user is dragging.
        Called on every mouse-move during a range drag.
        """
        t0 = min(start_time, end_time)
        t1 = max(start_time, end_time)

        self._selection_start_time = t0
        self._selection_end_time   = t1

        x0     = t0 * self.pixels_per_second
        width  = max(2.0, (t1 - t0) * self.pixels_per_second)
        height = self.sceneRect().height()

        if (self._selection_rect_item is not None
                and self._selection_rect_item in self.items()):
            self._selection_rect_item.setRect(x0, 0, width, height)
        else:
            pen   = QPen(QColor(100, 200, 255, 200), 1.5)
            brush = QBrush(QColor(80, 160, 255, 40))
            self._selection_rect_item = self.addRect(x0, 0, width, height, pen, brush)
            self._selection_rect_item.setZValue(90)

        self._update_selection_label(t0, t1, x0)

    def _update_selection_label(self, t0: float, t1: float, x0: float):
        """Update the time-range label that floats above the selection rect."""
        def fmt(t):
            m, s = divmod(t, 60)
            return f"{int(m):02d}:{s:05.2f}"

        duration = t1 - t0
        text = f"{fmt(t0)} → {fmt(t1)}  ({duration:.2f}s)  — drag to add"

        font = QFont("Consolas", 9, QFont.Weight.Bold)

        if (self._selection_label_item is not None
                and self._selection_label_item in self.items()):
            self._selection_label_item.setPlainText(text)
            self._selection_label_item.setPos(x0 + 4, 2)
        else:
            self._selection_label_item = self.addText(text, font)
            self._selection_label_item.setDefaultTextColor(QColor(120, 220, 255))
            self._selection_label_item.setPos(x0 + 4, 2)
            self._selection_label_item.setZValue(91)

    def finalise_selection(self):
        """
        Called on mouse-release after a range drag.
        Keeps the selection rect visible and marks it as ready to drag.
        Returns (start, end) if the selection is valid, else None.
        """
        t0 = self._selection_start_time
        t1 = self._selection_end_time

        if t0 is None or t1 is None or abs(t1 - t0) < 0.3:
            # Too short — discard silently
            self.clear_selection()
            return None

        # Make the rect visually distinct ("ready to drag")
        if (self._selection_rect_item is not None
                and self._selection_rect_item in self.items()):
            # Brighter border, slightly more opaque fill
            self._selection_rect_item.setPen(QPen(QColor(100, 220, 255, 255), 2))
            self._selection_rect_item.setBrush(QBrush(QColor(80, 180, 255, 60)))

        # Update label to show drag hint
        x0 = min(t0, t1) * self.pixels_per_second
        self._update_selection_label(min(t0, t1), max(t0, t1), x0)

        self._selection_active = True
        return (min(t0, t1), max(t0, t1))

    def clear_selection(self):
        """Remove the selection overlay entirely."""
        for attr in ('_selection_rect_item', '_selection_label_item'):
            item = getattr(self, attr, None)
            if item is not None and item in self.items():
                try:
                    self.removeItem(item)
                except RuntimeError:
                    pass
            setattr(self, attr, None)

        self._selection_start_time = None
        self._selection_end_time   = None
        self._selection_active     = False

    def selection_rect_contains(self, scene_pos) -> bool:
        """
        Returns True if scene_pos is inside the current selection rect.
        Used by SignalTimelineView to decide whether a press starts a DnD
        or clears the selection.
        """
        if not self._selection_active:
            return False
        if (self._selection_rect_item is None
                or self._selection_rect_item not in self.items()):
            return False
        return self._selection_rect_item.contains(
            self._selection_rect_item.mapFromScene(scene_pos)
        )
       
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
        self._range_selecting = False   # True while left-drag on background
        self._range_dragging  = False   # True while dragging the selection rect
        self._range_start_time = None   # seconds — where the drag started
        self._drag_press_pos   = None
        self._dnd_started      = False

        # Auto-follow playhead during playback
        self.follow_playhead = True
        self.follow_anchor = 0.35       # keep playhead ~35% from left
        self.follow_margin_left = 0.10  # scroll when playhead < 10% from left
        self.follow_margin_right = 0.85 # scroll when playhead > 85% from left

    def ensure_time_visible(self, time_seconds):
        """Auto-scroll so the playhead stays visible during playback."""
        if not self.follow_playhead:
            return
        scene = self.scene()
        if not scene:
            return

        pps = getattr(scene, 'pixels_per_second', 50)
        playhead_x = time_seconds * pps

        vp = self.viewport().rect()
        left = self.mapToScene(vp.topLeft()).x()
        right = self.mapToScene(vp.topRight()).x()
        width = right - left
        if width <= 0:
            return

        rel = (playhead_x - left) / width

        # Inside comfort zone → do nothing
        if self.follow_margin_left <= rel <= self.follow_margin_right:
            return

        # Use Qt's centerOn — keep vertical position, shift horizontal
        center_y = self.mapToScene(vp.center()).y()
        # Offset so playhead lands at 35% from left (center = 50%, so shift by +15%)
        self.centerOn(playhead_x + width * 0.15, center_y)

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
    
    def _item_is_bar(self, pos) -> bool:
        """True if a DraggableTimelineBar is under pos (walks parent chain)."""
        item = self.itemAt(pos)
        while item is not None:
            if isinstance(item, DraggableTimelineBar):
                return True
            item = item.parentItem()
        return False

    def _pos_in_selection(self, pos) -> bool:
        """True if view pos is inside the active selection rect."""
        scene = self.scene()
        if scene is None:
            return False
        scene_pos = self.mapToScene(pos)
        return scene.selection_rect_contains(scene_pos)

    # ── mouse events ───────────────────────────────────────────────────

    def mousePressEvent(self, event):
        """
        Priority:
          1. Left on active selection rect  → start DnD
          2. Left on a signal bar           → pass to item (existing bar DnD)
          3. Right / middle                 → pan
          4. Left on background             → start range selection
                                              (clears any existing selection)
        """
        scene = self.scene()

        # ── 1. Press inside the active selection rect ─────────────────
        # We record the press position; the actual QDrag is started
        # in mouseMoveEvent once the drag threshold is exceeded.
        if event.button() == Qt.LeftButton and self._pos_in_selection(event.pos()):
            self._range_dragging   = True   # "intent to drag" flag
            self._range_selecting  = False
            self._drag_press_pos   = event.pos()   # ← record where press happened
            self._dnd_started      = False          # ← DnD not yet fired
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            event.accept()
            return

        # ── 2. Signal bar ─────────────────────────────────────────────
        if event.button() == Qt.LeftButton and self._item_is_bar(event.pos()):
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self._range_selecting = False
            self._range_dragging  = False
            if scene:
                scene.clear_selection()
            super().mousePressEvent(event)
            return

        # ── 3. Pan ────────────────────────────────────────────────────
        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self._follow_was_on   = self.follow_playhead
            self.follow_playhead  = False
            self._range_selecting = False
            self._range_dragging  = False
            super().mousePressEvent(event)
            return

        # ── 4. Background — start range selection ─────────────────────
        if event.button() == Qt.LeftButton:
            # Clear any existing selection first
            if scene:
                scene.clear_selection()

            self._range_selecting = True
            self._range_dragging  = False
            self._left_press_pos  = event.pos()
            self.last_pan_point   = event.pos()

            if scene and hasattr(scene, 'pixels_per_second'):
                scene_pos = self.mapToScene(event.pos())
                t = scene_pos.x() / scene.pixels_per_second
                self._range_start_time = t
                # Also seek the video to the click point
                scene.time_clicked.emit(t)

            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene = self.scene()

        # ── Intent to drag the selection rect ────────────────────────
        if self._range_dragging and not getattr(self, '_dnd_started', False):
            # Wait until the cursor has moved past the drag threshold
            press_pos = getattr(self, '_drag_press_pos', event.pos())
            dist = (event.pos() - press_pos).manhattanLength()

            if dist < QApplication.startDragDistance():
                # Not moved enough yet — just update cursor
                event.accept()
                return

            # ── Threshold crossed: fire the real QDrag ────────────────
            # Mark as started so we don't fire twice
            self._dnd_started = True

            t0 = scene._selection_start_time if scene else None
            t1 = scene._selection_end_time   if scene else None

            if t0 is not None and t1 is not None:
                self._start_range_dnd(min(t0, t1), max(t0, t1), event)
                # drag.exec() is blocking — returns here when drop completes
                # or is cancelled.  Clean up regardless of outcome.

            self._range_dragging = False
            self._dnd_started    = False
            self.setCursor(QCursor(Qt.ArrowCursor))
            if scene:
                scene.clear_selection()

            event.accept()
            return

        # ── Drawing the range selection rect ──────────────────────────
        if self._range_selecting:
            if scene and hasattr(self, '_range_start_time'):
                scene_pos = self.mapToScene(event.pos())
                current_t = scene_pos.x() / scene.pixels_per_second
                scene.update_selection_rect(self._range_start_time, current_t)
            event.accept()
            return

        # ── Panning ───────────────────────────────────────────────────
        if self.panning:
            delta = event.pos() - self.last_pan_point
            self.last_pan_point = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        # Threshold check: switch to pan if left held and moved far enough
        if event.buttons() & Qt.LeftButton and hasattr(self, '_left_press_pos'):
            if (event.pos() - self._left_press_pos).manhattanLength() > QApplication.startDragDistance():
                if not self._range_selecting:
                    self.panning = True
                    self.setCursor(QCursor(Qt.ClosedHandCursor))
                    event.accept()
                    return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        scene = self.scene()

        if event.button() == Qt.LeftButton:

            # ── Release while in drag-intent state ────────────────────
            # If the user pressed inside the selection but released
            # without moving far enough to trigger DnD, just clear flags.
            if self._range_dragging:
                self._range_dragging = False
                self._dnd_started    = False
                self.setCursor(QCursor(Qt.ArrowCursor))
                # Leave the selection rect visible — user can try again
                event.accept()
                return

            # ── Release after drawing the range ───────────────────────
            if self._range_selecting:
                self._range_selecting = False
                self.setCursor(QCursor(Qt.ArrowCursor))

                if scene:
                    result = scene.finalise_selection()
                    if result:
                        # Selection is now active — show hint in status bar
                        # via the parent window
                        t0, t1 = result
                        duration = t1 - t0
                        # Find parent SignalTimelineWindow and update status
                        parent = self.parent()
                        while parent and not hasattr(parent, 'statusBar'):
                            parent = parent.parent()
                        if parent and hasattr(parent, 'statusBar'):
                            parent.statusBar().showMessage(
                                f"Selected {duration:.2f}s  "
                                f"({t0:.2f}s → {t1:.2f}s)  "
                                "— drag selection into edit timeline to add",
                                0
                            )

                if hasattr(self, '_left_press_pos'):
                    del self._left_press_pos
                event.accept()
                return

            # ── Normal left release ────────────────────────────────────
            if self.panning:
                self.panning = False
                self.setCursor(QCursor(Qt.ArrowCursor))

            if hasattr(self, '_left_press_pos'):
                del self._left_press_pos

            event.accept()
            return

        # Right / middle — reset scroll-hand drag
        if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            if hasattr(self, '_follow_was_on'):
                self.follow_playhead = self._follow_was_on
                del self._follow_was_on

        super().mouseReleaseEvent(event)

    def _start_range_dnd(self, start_time: float, end_time: float, event):
        """
        Fire a QDrag with the same MIME format that DraggableTimelineBar uses,
        so the existing EditTimelineScene.dropEvent handles it transparently.
        """
        import json

        mime_data = QMimeData()
        bar_data = {
            'type':       'timeline_bar',
            'start_time': start_time,
            'end_time':   end_time,
            'duration':   end_time - start_time,
            'label':      f"Range {start_time:.2f}s–{end_time:.2f}s",
            'metadata':   {'source': 'range_selection'}
        }
        mime_data.setText(json.dumps(bar_data))

        drag = QDrag(self.viewport())
        drag.setMimeData(mime_data)

        # Build a small pixmap that looks like a clip chip
        pps = self.scene().pixels_per_second if self.scene() else 50
        chip_w = min(200, max(60, int((end_time - start_time) * pps)))
        chip_h = 30

        pixmap = QPixmap(chip_w, chip_h)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Gradient fill — same blue as the selection rect
        grad = QLinearGradient(0, 0, 0, chip_h)
        grad.setColorAt(0, QColor(120, 200, 255, 220))
        grad.setColorAt(1, QColor(60,  140, 220, 220))
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(QColor(100, 220, 255), 1.5))
        painter.drawRoundedRect(1, 1, chip_w - 2, chip_h - 2, 4, 4)

        # Duration label
        painter.setPen(QPen(Qt.white))
        painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        painter.drawText(
            pixmap.rect(),
            Qt.AlignCenter,
            f"{end_time - start_time:.2f}s"
        )
        painter.end()

        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(chip_w // 2, chip_h // 2))

        # exec_ is blocking — returns when drop completes or is cancelled
        drag.exec(Qt.CopyAction)