from PySide6.QtWidgets import (
    QGraphicsRectItem, QGraphicsTextItem, QApplication
)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import (
    QColor, QPen, QBrush, QLinearGradient, QCursor, QPainter, QPixmap
)
from PySide6.QtCore import QMimeData
from PySide6.QtGui import QDrag

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
                if scene.visible_layers.get('waveform', False) and scene.waveform and scene.video_duration > 0:
                    sample_index = int(time * len(scene.waveform) / scene.video_duration)
                    if 0 <= sample_index < len(scene.waveform):
                        pt = scene.waveform[sample_index]   # (min, max) or (min, max, rms)
                        min_val, max_val = pt[0], pt[1]
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

        # Coerce any non-JSON-native values (e.g. numpy scalars from analysis
        # data) so json.dumps can't raise inside this Qt event callback — an
        # unhandled exception here hard-crashes the app on drag.
        def _json_safe(o):
            try:
                return o.item()          # numpy scalar -> python scalar
            except Exception:
                return str(o)

        try:
            start_t = float(self.bar.start_time)
            end_t = float(self.bar.end_time)
            bar_data = {
                'type': 'timeline_bar',
                'start_time': start_t,
                'end_time': end_t,
                'duration': end_t - start_t,
                'label': str(self.bar.label),
                'metadata': self.bar.metadata,
            }
            payload = json.dumps(bar_data, default=_json_safe)
        except Exception as e:
            print(f"⚠️ Drag serialize failed, skipping drag: {e}")
            self.mouse_press_pos = None
            return

        mime_data = QMimeData()
        mime_data.setText(payload)

        view = self.scene().views()[0] if self.scene() and self.scene().views() else None
        drag = QDrag(view.viewport() if view else event.widget())
        drag.setMimeData(mime_data)

        # Cap the drag pixmap size. A merged bar can be tens of thousands of px
        # wide; creating an oversized QPixmap (beyond the backend max, ~32767) can
        # crash natively with no Python traceback. It's only a visual cue, so a
        # small fixed-cap pixmap is fine.
        rect = self.rect()
        pw = max(1, min(int(rect.width()), 300))
        ph = max(1, min(int(rect.height()), 80))
        pixmap = QPixmap(pw, ph)
        pixmap.fill(Qt.transparent)

        painter = QPainter(pixmap)
        painter.setBrush(self.brush())
        painter.setPen(self.pen())
        painter.drawRect(0, 0, pw - 1, ph - 1)
        painter.end()

        drag.setPixmap(pixmap)
        drag.setHotSpot(QPoint(pw // 2, ph // 2))

        # Reset state BEFORE exec: exec() runs a nested event loop and the drop
        # may rebuild the scene and delete this item, so touching self afterwards
        # would be a use-after-free.
        self.mouse_press_pos = None
        drag.exec(Qt.CopyAction)
    
    def hoverEnterEvent(self, event):
        """Highlight on hover"""
        self.original_pen = self.pen()
        self.highlight_pen = QPen(QColor(255, 255, 0), 2)
        self.setPen(self.highlight_pen)
        
        # Show tooltip
        duration = self.bar.end_time - self.bar.start_time
        self.setToolTip(f"Drag to place • Right-click to add (this / all in row)\n{self.bar.label}\n{self.bar.start_time:.1f}s - {self.bar.end_time:.1f}s\nDuration: {duration:.1f}s")
        
        super().hoverEnterEvent(event)
    
    def hoverLeaveEvent(self, event):
        """Remove highlight"""
        if hasattr(self, 'original_pen'):
            self.setPen(self.original_pen)
        
        super().hoverLeaveEvent(event)