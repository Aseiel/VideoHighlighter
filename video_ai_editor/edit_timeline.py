import os
import json
from pathlib import Path
from PySide6.QtWidgets import (
    QGraphicsRectItem, QGraphicsTextItem, QGraphicsScene,
    QGraphicsView, QDialog, QVBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QDialogButtonBox, QMessageBox, QMenu
)
from PySide6.QtCore import Qt, QRectF, Signal, QTimer
from PySide6.QtGui import (
    QColor, QPen, QBrush, QFont, QLinearGradient, QCursor
)
from .timeline_bars import TimelineBar

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
        """Handle mouse press for dragging + seeking"""
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
            
            # Calculate source time from click position within clip
            local_x = event.pos().x()
            width = self.rect().width()
            if width > 0:
                progress = max(0.0, min(1.0, local_x / width))
                source_time = self.start_time + progress * (self.end_time - self.start_time)
                scene = self.scene()
                if scene and hasattr(scene, 'time_clicked'):
                    scene.time_clicked.emit(source_time)
        
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
    time_clicked = Signal(float)  # source video time
    
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
        
        self.active_clip_index = -1  # Currently playing clip (-1 = none)
        self.active_progress = 0.0   # 0.0 to 1.0 within active clip
        self._active_overlay = None
        self._progress_line = None
        self.setSceneRect(0, 0, 1000, self.clip_height + 40)
        self.build_timeline()

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

        # Restore active clip overlay if playing
        if self.active_clip_index >= 0 and self.active_clip_index < len(self.clip_items):
            self._active_overlay = None
            self._progress_line = None
            self._create_active_overlay()
    
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

    def set_active_clip(self, index):
        """Highlight the currently playing clip"""
        self.active_clip_index = index
        self.active_progress = 0.0
        self._create_active_overlay()

    def set_active_progress(self, progress):
        """Update progress within the active clip (0.0 to 1.0)"""
        self.active_progress = max(0.0, min(1.0, progress))
        self._move_progress_line()

    def clear_active_clip(self):
        """Remove active clip highlight"""
        self.active_clip_index = -1
        self.active_progress = 0.0
        self._remove_active_overlay()

    def set_active_clip(self, index):
        """Highlight the currently playing clip"""
        old_index = self.active_clip_index
        self.active_clip_index = index
        self.active_progress = 0.0

        if old_index != index:
            # Only recreate overlay when clip changes
            self._remove_active_overlay()
            self._create_active_overlay()
        
        self._move_progress_line()

    def _create_active_overlay(self):
        """Create glow overlay and progress line (once per clip)"""
        if self.active_clip_index < 0 or self.active_clip_index >= len(self.clip_items):
            return

        item = self.clip_items[self.active_clip_index]
        rect = item.rect()
        pos = item.pos()

        # Glow overlay
        self._active_overlay = self.addRect(
            pos.x() - 3, pos.y() - 3,
            rect.width() + 6, rect.height() + 6,
            QPen(QColor(80, 180, 255, 200), 3),
            QBrush(QColor(80, 180, 255, 40))
        )
        self._active_overlay.setZValue(10)

        # Progress line — create once, then just move it
        x = pos.x()
        self._progress_line = self.addLine(
            x, pos.y(), x, pos.y() + rect.height(),
            QPen(QColor(255, 255, 255, 220), 2)
        )
        self._progress_line.setZValue(11)

    def _move_progress_line(self):
        """Move the progress line without removing/recreating it"""
        if self.active_clip_index < 0 or self.active_clip_index >= len(self.clip_items):
            return

        # Recreate if missing (e.g. after build_timeline cleared everything)
        if not self._progress_line or self._progress_line not in self.items():
            self._progress_line = None
            self._active_overlay = None
            self._create_active_overlay()
            return

        item = self.clip_items[self.active_clip_index]
        rect = item.rect()
        pos = item.pos()
        x = pos.x() + rect.width() * self.active_progress

        # Just update the line coordinates — no remove/add
        self._progress_line.setLine(x, pos.y(), x, pos.y() + rect.height())

    def _remove_active_overlay(self):
        """Remove overlay and progress line"""
        try:
            if self._active_overlay and self._active_overlay in self.items():
                self.removeItem(self._active_overlay)
        except RuntimeError:
            pass
        self._active_overlay = None

        try:
            if self._progress_line and self._progress_line in self.items():
                self.removeItem(self._progress_line)
        except RuntimeError:
            pass
        self._progress_line = None

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