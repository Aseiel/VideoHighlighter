import os
import hashlib
import json
from pathlib import Path
from PySide6.QtWidgets import (
    QGraphicsRectItem, QGraphicsTextItem, QGraphicsScene,
    QGraphicsView, QDialog, QVBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QDialogButtonBox, QMessageBox, QMenu,
    QStyle,
)
from PySide6.QtCore import Qt, QRectF, Signal, QTimer, QPointF
from PySide6.QtGui import (
    QColor, QPen, QBrush, QFont, QLinearGradient, QCursor, QPainter
)
from .timeline_bars import TimelineBar

# ── thumbnail / hover modules ─────────────────────────────────────
from .thumbnail_cache import ThumbnailCache
from .hover_preview import HoverPreview
from .filmstrip_painter import paint_filmstrip


# Hover preview pulls a larger thumb than the filmstrip slots.
HOVER_PREVIEW_HEIGHT = 180


def _fmt_time(seconds: float) -> str:
    """Format timestamp as M:SS.t  →  1268.4 becomes '21:08.4'"""
    mins = int(seconds // 60)
    secs = seconds - mins * 60
    return f"{mins}:{secs:04.1f}"


def _fmt_duration(seconds: float) -> str:
    """Durations under a minute keep 'X.Xs', longer ones use M:SS.t"""
    return f"{seconds:.1f}s" if seconds < 60 else _fmt_time(seconds)


class EditClipItem(QGraphicsRectItem):
    """Represents a clip in the edit timeline that can be dragged for reordering"""

    def __init__(self, start_time, end_time, y, height, color, index):
        super().__init__()
        self.start_time = start_time
        self.end_time = end_time
        self.color = color
        self.is_selected = False
        self.original_pos = None
        self.drag_start_pos = None

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
        self.text_item.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.text_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.text_item.setAcceptHoverEvents(False)
        # Ensure text renders above our filmstrip paint() output.
        self.text_item.setZValue(2)
        self.update_label()

    def update_label(self):
        """Update clip label text"""
        duration = self.end_time - self.start_time
        
        # Get current index from scene dynamically
        idx = -1
        scene = self.scene()
        if scene and hasattr(scene, 'clip_items') and self in scene.clip_items:
            idx = scene.clip_items.index(self)
        
        # Build label text with or without clip number
        if idx >= 0:
            label_text = f"Clip {idx + 1}\n{_fmt_time(self.start_time)} - {_fmt_time(self.end_time)}\n({_fmt_duration(duration)})"
        else:
            label_text = f"{_fmt_time(self.start_time)} - {_fmt_time(self.end_time)}\n({_fmt_duration(duration)})"
        
        self.text_item.setPlainText(label_text)
        self.text_item.setFont(QFont("Arial", 8, QFont.Weight.Bold))
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

    # ── custom paint that draws filmstrip + dark label backdrop ──
    def paint(self, painter, option, widget=None):
        """
        Custom paint:
          1. Colored background (uses self.brush() — same color as before)
          2. Filmstrip thumbnails (from scene.thumb_cache)
          3. Semi-transparent dark backdrop behind the text label
          4. Border (selection-aware)

        The QGraphicsTextItem child renders on top automatically (its z=2),
        so the text appears clearly readable on the dark backdrop.
        """
        rect = self.rect()
        painter.setRenderHint(QPainter.Antialiasing, True)

        # 1. Background — keep the existing color, but rounded for polish
        painter.setBrush(self.brush())
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(rect, 4, 4)

        # 2. Filmstrip — only if scene exposes a thumb_cache
        scene = self.scene()
        if scene is not None and getattr(scene, "thumb_cache", None) is not None:
            # Inset 1px so thumbs don't overdraw the rounded corners
            inset = rect.adjusted(1, 1, -1, -1)
            paint_filmstrip(
                painter,
                inset,
                self.start_time,
                self.end_time,
                scene.thumb_cache,
            )

        # 3. Label backdrop — Premiere-style dark pill behind the text
        if self.text_item is not None:
            tr = self.text_item.boundingRect()
            tp = self.text_item.pos()
            backdrop = QRectF(
                tp.x() - 8, tp.y() - 4,
                tr.width() + 16, tr.height() + 8
            )
            # Don't draw if the backdrop wouldn't fit (very narrow clips)
            if backdrop.width() < rect.width() - 2:
                painter.setBrush(QBrush(QColor(0, 0, 0, 160)))
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(backdrop, 5, 5)

        # 4. Border — yellow when selected (matches old behavior), pen otherwise
        if option is not None and (option.state & QStyle.State_Selected):
            painter.setPen(QPen(Qt.yellow, 2))
        else:
            painter.setPen(self.pen())
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(rect, 4, 4)

    def mousePressEvent(self, event):
        """Handle mouse press — cut mode takes priority over drag/seek."""
        scene = self.scene()

        # 1. Cut mode: left-click cuts immediately
        if (event.button() == Qt.LeftButton
                and scene is not None
                and getattr(scene, 'cut_mode', False)):
            local_x = event.pos().x()
            width = self.rect().width()
            progress = max(0.0, min(1.0, local_x / width)) if width > 0 else 0.5
            cut_time = self.start_time + progress * (self.end_time - self.start_time)
            scene.cut_clip_at(cut_time)
            event.accept()
            return

        # 2. Normal left-click: select + seek
        if event.button() == Qt.LeftButton:
            # Clear other selections if shift/ctrl not pressed
            if not (event.modifiers() & (Qt.ShiftModifier | Qt.ControlModifier)):
                if scene:
                    for item in scene.selectedItems():
                        if item != self:
                            item.setSelected(False)

            self.setSelected(True)
            
            # Store the initial drag position in scene coordinates
            self.drag_start_pos = event.scenePos()
            self.original_pos = self.pos()  # Store original item position
            self.setCursor(QCursor(Qt.ClosedHandCursor))
            
            # Seek to the source time at the click position
            local_x = event.pos().x()
            width = self.rect().width()
            if width > 0:
                progress = max(0.0, min(1.0, local_x / width))
                source_time = self.start_time + progress * (self.end_time - self.start_time)
                if scene and hasattr(scene, 'time_clicked'):
                    scene.time_clicked.emit(source_time)
            
            event.accept()  # Accept the event to ensure we get move events
            return
        
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move during drag - manual positioning for better control"""
        if self.original_pos is not None and self.drag_start_pos is not None:
            # Calculate the movement delta in scene coordinates
            current_scene_pos = event.scenePos()
            delta = current_scene_pos - self.drag_start_pos
            
            # Calculate new position (only horizontal movement matters for reordering)
            new_x = self.original_pos.x() + delta.x()
            new_y = self.original_pos.y()  # Keep Y position fixed
            
            # Set the new position
            self.setPos(new_x, new_y)
            
            # Visual feedback
            self.setOpacity(0.55)
            self.setZValue(20)
            
            # Show drop indicator
            scene = self.scene()
            if scene is not None:
                # Use the LEFT edge of the clip for consistent positioning
                left_edge = self.pos().x()
                center_y = self.pos().y() + self.rect().height() / 2
                scene.show_drop_indicator(QPointF(left_edge, center_y))
            
            event.accept()
            return
        
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release - finish drag and drop"""
        try:
            if event.button() == Qt.LeftButton and self.original_pos is not None:
                self.setCursor(QCursor(Qt.OpenHandCursor))
                self.setOpacity(1.0)
                self.setZValue(0)

                scene = self.scene()
                if scene is not None:
                    scene.hide_drop_indicator()

                new_pos = self.pos()
                moved = (new_pos - self.original_pos).manhattanLength() > 10

                if moved and scene and hasattr(scene, 'reorder_clip'):
                    # Get current index dynamically
                    current_index = self._get_current_index()
                    if current_index >= 0:
                        left_edge = new_pos.x()
                        scene.reorder_clip(current_index, left_edge)
                else:
                    # Snap back to original position if not moved enough
                    self._animate_to(self.original_pos)

                # Reset drag state
                self.original_pos = None
                self.drag_start_pos = None
                
                event.accept()
                return

            if self.scene():
                super().mouseReleaseEvent(event)
        except RuntimeError:
            pass

        # Ensure drag state is cleared even if there's an error
        self.original_pos = None
        self.drag_start_pos = None

    def _animate_to(self, target_pos, duration_ms=180, steps=14):
        """Manual ease-out tween — QGraphicsRectItem isn't a QObject for QPropertyAnimation."""
        start = self.pos()
        if (start - target_pos).manhattanLength() < 1:
            return
        interval = max(1, duration_ms // steps)

        def tick(i=[0]):
            try:
                if i[0] > steps or self.scene() is None:
                    self.setPos(target_pos)
                    return
                t = i[0] / steps
                ease = 1 - (1 - t) ** 3
                self.setPos(
                    start.x() + (target_pos.x() - start.x()) * ease,
                    start.y() + (target_pos.y() - start.y()) * ease,
                )
                i[0] += 1
                QTimer.singleShot(interval, tick)
            except RuntimeError:
                pass
        tick()

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

    def _get_current_index(self):
        """Get current index from scene's clip_items list"""
        scene = self.scene()
        if scene and hasattr(scene, 'clip_items') and self in scene.clip_items:
            return scene.clip_items.index(self)
        return -1

    def hoverEnterEvent(self, event):
        """Show tooltip on hover"""
        duration = self.end_time - self.start_time
        idx = self._get_current_index()
        clip_num = idx + 1 if idx >= 0 else "?"
        
        self.setToolTip(
            f"Clip {clip_num}\n"
            f"{self.start_time:.1f}s - {self.end_time:.1f}s\n"
            f"Duration: {duration:.1f}s\n"
            f"Drag to reorder · Double-click to play · Right-click for menu"
        )
        super().hoverEnterEvent(event)

    def contextMenuEvent(self, event):
        """
        Right-click context menu for a clip.
        Provides:
          - Cut here     (at the exact pixel the user right-clicked)
          - Trim start   (move in-point to click position)
          - Trim end     (move out-point to click position)
          - Delete clip
        """
        scene = self.scene()
        if not scene:
            return
        
        current_index = self._get_current_index()
        if current_index < 0:
            return

        # Hide hover preview while the context menu is open
        if hasattr(scene, '_hover_preview') and scene._hover_preview is not None:
            scene._hover_preview.hide_preview()

        # ── Calculate source time at click position ─────────────────────
        local_x = event.pos().x()
        width = self.rect().width()
        progress = max(0.0, min(1.0, local_x / width)) if width > 0 else 0.5
        click_time = self.start_time + progress * (self.end_time - self.start_time)
        duration = self.end_time - self.start_time

        # ── Build menu ───────────────────────────────────────────────────
        menu = QMenu()
        menu.setStyleSheet("""
            QMenu {
                background-color: #1a1a2a;
                color: #d0d8ff;
                border: 1px solid #3a3a5a;
                border-radius: 4px;
                padding: 4px 0;
            }
            QMenu::item { padding: 6px 20px; border-radius: 3px; }
            QMenu::item:selected { background-color: #3a5fcd; }
            QMenu::item:disabled { color: #555577; }
            QMenu::separator { height: 1px; background: #3a3a5a; margin: 4px 8px; }
        """)

        header = menu.addAction(
            f"Clip {current_index}   {_fmt_time(self.start_time)} → {_fmt_time(self.end_time)}  ({_fmt_duration(duration)})"
        )
        header.setEnabled(False)
        menu.addSeparator()

        cut_action = menu.addAction(f"✂️   Cut here  ({_fmt_time(click_time)})")
        too_close = (
            click_time - self.start_time < 0.2
            or self.end_time - click_time < 0.2
        )
        cut_action.setEnabled(not too_close)
        if too_close:
            cut_action.setToolTip("Click closer to the centre of the clip")

        trim_start_action = menu.addAction(f"⬅️   Trim start  →  {_fmt_time(click_time)}")
        trim_end_action   = menu.addAction(f"➡️   Trim end  ←  {_fmt_time(click_time)}")
        trim_start_action.setEnabled(click_time > self.start_time + 0.2)
        trim_end_action.setEnabled(click_time < self.end_time - 0.2)

        menu.addSeparator()
        delete_action = menu.addAction("🗑️   Delete clip")

        chosen = menu.exec(event.screenPos())

        if chosen == cut_action:
            scene.cut_clip_at(click_time)

        elif chosen == trim_start_action:
            if self in scene.clip_items:
                idx = scene.clip_items.index(self)
                scene.trim_clip_start(idx, click_time)

        elif chosen == trim_end_action:
            if self in scene.clip_items:
                idx = scene.clip_items.index(self)
                scene.trim_clip_end(idx, click_time)

        elif chosen == delete_action:
            if self in scene.clip_items:
                idx = scene.clip_items.index(self)
                scene.clips.pop(idx)
                scene.clip_items.pop(idx)
                scene.removeItem(self)
                scene.build_timeline()
                scene.clip_removed.emit(idx)

    def hoverMoveEvent(self, event):
        """
        Track the source-time under the cursor for the C key shortcut,
        update the cut indicator line when in cut mode, and drive the
        hover thumbnail preview when NOT in cut mode.
        """
        scene = self.scene()
        if scene is not None:
            local_x = event.pos().x()
            width = self.rect().width()
            progress = max(0.0, min(1.0, local_x / width)) if width > 0 else 0.5
            source_time = self.start_time + progress * (self.end_time - self.start_time)

            scene._hover_source_time = source_time

            if getattr(scene, 'cut_mode', False):
                # Cut mode: show cut indicator, hide hover preview
                item_pos = self.pos()
                scene._update_cut_indicator(
                    item_pos.x() + local_x,
                    item_pos.y(),
                    self.rect().height()
                )
                self.setCursor(QCursor(Qt.CrossCursor))
                if hasattr(scene, '_hover_preview') and scene._hover_preview is not None:
                    scene._hover_preview.hide_preview()
            else:
                # Normal mode: drive hover preview
                self.setCursor(QCursor(Qt.OpenHandCursor))
                if (hasattr(scene, '_hover_preview')
                        and scene._hover_preview is not None
                        and getattr(scene, 'thumb_cache', None) is not None):
                    pix = scene.thumb_cache.request(source_time, HOVER_PREVIEW_HEIGHT)
                    scene._hover_preview.show_at(
                        event.screenPos(),
                        pix,
                        caption=_fmt_time(source_time),
                    )

        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event):
        """
        Hide the cut indicator and hover preview when the mouse leaves this clip.
        """
        scene = self.scene()
        if scene is not None:
            scene._hide_cut_indicator()
            scene._hover_source_time = None
            if hasattr(scene, '_hover_preview') and scene._hover_preview is not None:
                scene._hover_preview.hide_preview()

        self.setCursor(QCursor(Qt.OpenHandCursor))
        super().hoverLeaveEvent(event)


class EditTimelineScene(QGraphicsScene):
    """Simple timeline showing clips as colored rectangles"""

    clip_double_clicked = Signal(float, float)  # start, end
    clip_added = Signal(float, float)           # start, end
    clip_removed = Signal(int)                  # index
    time_clicked = Signal(float)                # source video time
    clip_cut = Signal(float)                    # cut time after a successful cut
    clip_trimmed = Signal(int)                  # clip index after a trim
    clip_reordered = Signal(int, int)           # from_index, to_index

    def __init__(self, video_path, video_duration, parent=None, cache=None, cache_data=None):
        super().__init__(parent)
        self.video_path = video_path
        self.video_duration = video_duration
        self.cache = cache
        self.cache_data = cache_data or {}
        self.clips = []
        self.clip_items = []
        self.pixels_per_second = 50
        self.clip_height = 60
        self.clip_spacing = 5

        # ── thumbnail cache (per-video, async, persistent) ──
        try:
            self.thumb_cache = ThumbnailCache(video_path)
            self.thumb_cache.thumbnail_ready.connect(self._on_thumb_ready)
        except Exception as e:
            print(f"⚠️ ThumbnailCache init failed: {e} — filmstrip disabled")
            self.thumb_cache = None

        # ── hover preview popup (single instance, shared by clips) ──
        try:
            self._hover_preview = HoverPreview()
        except Exception as e:
            print(f"⚠️ HoverPreview init failed: {e}")
            self._hover_preview = None

        # Visual feedback for drop target
        self.drop_indicator = None
        self.drop_indicator_marker = None
        self.drop_position = None
        self.is_dragging_over = False

        # Load initial highlights if available
        self.load_initial_clips()

        self.active_clip_index = -1
        self.active_progress = 0.0
        self._active_overlay = None
        self._progress_line = None

        # Cut mode state
        self.cut_mode = False
        self._cut_line = None
        self._hover_source_time = None

        self.setSceneRect(0, 0, 1000, self.clip_height + 40)
        self.build_timeline()

    def set_vr_mode(self, enabled: bool):
        """Crop thumbnails to left half for side-by-side VR videos."""
        if self.thumb_cache is not None:
            self.thumb_cache.set_vr_mode(enabled)
        self.update()

    # ── cache → repaint relevant clips ──────────────────────────
    def _on_thumb_ready(self, time_seconds, height, pixmap):
        """When a new thumbnail arrives, repaint every clip whose range covers it."""
        for item in self.clip_items:
            if not hasattr(item, 'start_time'):
                continue
            if item.start_time <= time_seconds <= item.end_time:
                item.update()

    # ── clean shutdown hook (call from SignalTimelineWindow.closeEvent) ──
    def cleanup(self):
        """Tear down the thumbnail worker and hide the hover popup."""
        try:
            if self.thumb_cache is not None:
                self.thumb_cache.stop()
        except Exception:
            pass
        try:
            if self._hover_preview is not None:
                self._hover_preview.hide()
                self._hover_preview.deleteLater()
                self._hover_preview = None
        except Exception:
            pass

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
        # If we're over a clip, EditClipItem.contextMenuEvent handles it.
        # Only fall back here for empty timeline area.
        item = self.itemAt(event.scenePos(), self.views()[0].transform()) if self.views() else None
        if isinstance(item, EditClipItem):
            super().contextMenuEvent(event)
            return

        menu = QMenu()
        load_action = menu.addAction("📂 Load from Cache...")
        load_action.triggered.connect(self.load_from_cache_menu)
        save_action = menu.addAction("💾 Save to Cache")
        save_action.triggered.connect(lambda: self.save_clips_to_cache())
        menu.exec(event.screenPos())

    def load_from_cache_menu(self):
        """Show dialog to load different highlight versions from cache"""
        if not self.cache or not hasattr(self.cache, 'get_highlight_history'):
            QMessageBox.warning(None, "Cache Error",
                               "Enhanced cache not available. Cannot load from cache.")
            return

        history = self.cache.get_highlight_history(self.video_path)
        if not history:
            QMessageBox.information(None, "No Cache",
                                   "No cached highlight versions found for this video.")
            return

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

        segments_metadata = []
        for start, end in self.clips:
            segments_metadata.append({
                'score': 1.0,
                'signals': {'user_edited': 1.0},
                'primary_reason': 'manual_selection'
            })

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
        """Handle key presses for deleting and cutting clips"""
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            self.remove_selected_clips()
            event.accept()
        elif event.key() == Qt.Key_C:
            if self._hover_source_time is not None:
                self.cut_clip_at(self._hover_source_time)
            event.accept()
        else:
            super().keyPressEvent(event)

    def remove_selected_clips(self):
        selected_items = [item for item in self.items()
                          if isinstance(item, EditClipItem) and item.isSelected()]
        if not selected_items:
            return

        indices_to_remove = sorted(
            (self.clip_items.index(item) for item in selected_items),
            reverse=True
        )

        for idx in indices_to_remove:
            if self.clip_items[idx] in self.items():
                self.removeItem(self.clip_items[idx])
            self.clip_items.pop(idx)
            self.clips.pop(idx)
            self.clip_removed.emit(idx)

        self.build_timeline()

    def load_initial_clips(self):
            """Load initial clips. Prefer THIS run's final (post-subtract) segments
            so the edit timeline matches exactly what the pipeline cut — including
            the splits that remove an avoided person. Fall back to cache history
            only when opened standalone (no run data)."""
            self.clips = []

            # 1. PREFERRED — this run's final segments, stamped by the pipeline.
            final = (self.cache_data or {}).get("final_segments")
            if final:
                loaded = []
                for segment in final:
                    if isinstance(segment, (list, tuple)) and len(segment) >= 2:
                        start, end = float(segment[0]), float(segment[1])
                        if end > start and (end - start) >= 0.5:
                            loaded.append((start, end))
                if loaded:
                    self.clips = loaded
                    print(f"✅ Loaded {len(self.clips)} segments from this run's final_segments")
                    return

            # 2. Fallback — most recent highlight version from cache history.
            if self.cache and hasattr(self.cache, 'get_highlight_history'):
                history = self.cache.get_highlight_history(self.video_path, analysis_params=None)
                if history:
                    segments = history[0].get('segments', [])
                    if segments:
                        self.clips = [tuple(s) for s in segments]
                        print(f"✅ Loaded {len(self.clips)} highlight segments from cache history")
                        return

            # 3. Last resort — sample clips.
            print("⚠️ No cached highlights found, creating sample clips")
            if self.video_duration > 30:
                for start_ratio, end_ratio in [(0.1, 0.2), (0.3, 0.4), (0.7, 0.8)]:
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

        # ── Calculate total width FIRST ──
        current_x = 20
        for start, end in self.clips:
            duration = end - start
            width = max(60, duration * self.pixels_per_second)
            current_x += width + self.clip_spacing
        total_width = max(1000, current_x + 20)
        self.setSceneRect(0, 0, total_width, self.clip_height + 40)

        # Background
        self.addRect(self.sceneRect(), QPen(Qt.NoPen), QBrush(QColor(30, 30, 40)))

        # Time ruler — now uses correct sceneRect
        self.draw_time_ruler()

        # Clips
        current_x = 20
        y_pos = 35

        for i, (start, end) in enumerate(self.clips):
            duration = end - start
            width = max(60, duration * self.pixels_per_second)

            color = self.get_clip_color(i)
            # Added missing 'i' parameter for index
            clip_item = EditClipItem(start, end, y_pos, self.clip_height, color, i)
            clip_item.setRect(0, 0, width, self.clip_height)
            clip_item.setPos(current_x, y_pos)
            self.addItem(clip_item)
            self.clip_items.append(clip_item)

            # ── prefetch this clip's filmstrip ─────────────────────
            if self.thumb_cache is not None:
                aspect = 16 / 9
                target_slot_w = max(40, int(self.clip_height * aspect))
                n_slots = max(1, int(width // target_slot_w))
                self.thumb_cache.prefetch_range(
                    start, end, self.clip_height, n_slots
                )

            clip_item.update_label()
            current_x += width + self.clip_spacing

        # Update scene width
        total_width = max(1000, current_x + 20)
        self.setSceneRect(0, 0, total_width, self.clip_height + 40)

        # Restore active clip overlay if playing
        if self.active_clip_index >= 0 and self.active_clip_index < len(self.clip_items):
            self._active_overlay = None
            self._progress_line = None
            self._create_active_overlay()

    def draw_time_ruler(self):
        """Draw time ruler showing accumulated edit duration (gap-aware)."""
        ruler_y = 30
        total_width = self.sceneRect().width()

        # Ruler baseline
        self.addLine(20, ruler_y, total_width - 20, ruler_y,
                    QPen(QColor(150, 150, 150), 1))

        if not self.clips:
            return

        # Build accumulated-time → pixel mapping at clip boundaries
        waypoints = []  # (accumulated_seconds, pixel_x)
        acc = 0.0
        cx = 20.0
        waypoints.append((0.0, cx))

        for start, end in self.clips:
            duration = end - start
            width = max(60, duration * self.pixels_per_second)
            acc += duration
            cx += width
            waypoints.append((acc, cx))
            cx += self.clip_spacing

        total_edit_time = acc
        if total_edit_time <= 0:
            return

        # Interpolate: edit-seconds → pixel x
        def time_to_x(t):
            for i in range(len(waypoints) - 1):
                t0, x0 = waypoints[i]
                t1, x1 = waypoints[i + 1]
                if t0 <= t <= t1 and t1 > t0:
                    return x0 + (t - t0) / (t1 - t0) * (x1 - x0)
            return waypoints[-1][1]

        # Choose tick spacing based on total duration
        if total_edit_time > 120:
            minor, major = 5.0, 30.0
        elif total_edit_time > 30:
            minor, major = 1.0, 5.0
        else:
            minor, major = 0.5, 2.0

        t = 0.0
        while t <= total_edit_time + 0.01:
            x = time_to_x(t)
            is_major = (t % major) < 0.01 or t == 0

            if is_major:
                self.addLine(x, ruler_y - 8, x, ruler_y,
                            QPen(QColor(200, 200, 200), 1))
                label = self.addText(_fmt_time(t), QFont("Arial", 8))
                label.setDefaultTextColor(QColor(180, 180, 180))
                label.setPos(x - 10, ruler_y - 25)
            else:
                self.addLine(x, ruler_y - 4, x, ruler_y,
                            QPen(QColor(150, 150, 150), 1))

            t += minor

    def get_clip_color(self, index):
        """Stable color per clip based on (start, end) — survives reorders
        and app restarts."""
        colors = [
            QColor(100, 150, 255, 220),  # Blue
            QColor(100, 255, 100, 220),  # Green
            QColor(255, 100, 100, 220),  # Red
            QColor(255, 200, 50, 220),   # Yellow
            QColor(200, 100, 255, 220),  # Purple
            QColor(50, 255, 255, 220),   # Cyan
        ]
        if 0 <= index < len(self.clips):
            start, end = self.clips[index]
            key = f"{round(start, 2)}_{round(end, 2)}".encode()
            color_idx = int(hashlib.md5(key).hexdigest(), 16) % len(colors)
            return colors[color_idx]
        return colors[index % len(colors)]
    
    def add_clip(self, start_time, end_time):
        """Add a new clip to the timeline"""
        start_time = max(0, min(start_time, self.video_duration - 1))
        end_time = max(start_time + 1, min(end_time, self.video_duration))

        self.clips.append((start_time, end_time))
        self.build_timeline()
        self.clip_added.emit(start_time, end_time)

    def add_clip_from_selection(self, start_time, end_time=None):
        """Add a clip from a time selection (default 5 second duration)"""
        if end_time is None:
            end_time = start_time + 5

        if end_time - start_time < 0.5:
            end_time = start_time + 3

        self.add_clip(start_time, end_time)

    def get_total_duration(self):
        """Get total duration of all clips"""
        return sum(end - start for start, end in self.clips)

    def get_clip_times(self):
        """Get list of all clip time ranges"""
        return self.clips.copy()

    # ── DRAG AND DROP ──────────────────────────────────────────────────
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
                data = json.loads(event.mimeData().text())
                if data.get('type') == 'timeline_bar':
                    event.acceptProposedAction()
                    self.is_dragging_over = True
                    self.show_drop_indicator(event.scenePos())
                    return
            except:
                pass

        event.ignore()

    def dragMoveEvent(self, event):
        """Update drop indicator position"""
        if event.mimeData().hasText():
            try:
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
                data = json.loads(event.mimeData().text())

                if data.get('type') == 'timeline_bar':
                    pos = event.scenePos()
                    insert_index = self.get_insert_index(pos.x())

                    start_time = data['start_time']
                    end_time = data['end_time']

                    if end_time - start_time < 0.5:
                        end_time = start_time + 3.0

                    self.clips.insert(insert_index, (start_time, end_time))
                    self.build_timeline()
                    self.clip_added.emit(start_time, end_time)

                    event.accept()
                    self.show_drop_feedback(insert_index)

                    if hasattr(self.parent(), 'update_edit_duration'):
                        self.parent().update_edit_duration()

                    return

            except Exception as e:
                print(f"Drop error: {e}")

        event.ignore()
        self.is_dragging_over = False
        self.hide_drop_indicator()

    def show_drop_indicator(self, pos_or_x, y=None):
        """Show visual indicator where clip will be inserted"""
        self.clear_drop_indicators()
        
        # Handle both QPointF and (x, y) parameter patterns
        if isinstance(pos_or_x, QPointF):
            x_pos = pos_or_x.x()
            y_pos = pos_or_x.y()
        else:
            x_pos = pos_or_x
            y_pos = y if y is not None else 35
        
        insert_index = self.get_insert_index(x_pos)
        x_pos = self.calculate_insert_x(insert_index)
        self.drop_indicator = self.addLine(x_pos, 35, x_pos, 35 + self.clip_height,
                                        QPen(QColor(0, 255, 0), 2, Qt.DashLine))
        self.drop_indicator_marker = self.addEllipse(x_pos - 5, 30, 10, 10,
                                                    QPen(Qt.green, 2), QBrush(Qt.green))
        self.drop_position = insert_index

    def hide_drop_indicator(self):
        """Remove drop indicator - SAFE version"""
        self.clear_drop_indicators()

        if hasattr(self, 'drop_indicator_marker'):
            self.removeItem(self.drop_indicator_marker)

    def get_insert_index(self, x_pos):
        """
        Determine where to insert based on x position.
        Compare against clip boundaries (left edges + half width for equal spacing)
        """
        current_x = 20
        
        for i, (start, end) in enumerate(self.clips):
            duration = end - start
            width = max(60, duration * self.pixels_per_second)
            
            # Compare against clip's right edge (more predictable than center)
            right_edge = current_x + width
            if x_pos < current_x + (width / 2):  # Insert before if left of center
                return i
            
            current_x += width + self.clip_spacing
        
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
        if insert_index < len(self.clip_items):
            item = self.clip_items[insert_index]
            original_pen = item.pen()

            def flash():
                item.setPen(QPen(Qt.yellow, 3))
                QTimer.singleShot(300, lambda: item.setPen(original_pen))

            QTimer.singleShot(100, flash)

    # ── ACTIVE CLIP OVERLAY ────────────────────────────────────────────
    def set_active_clip(self, index):
        """Highlight the currently playing clip"""
        old_index = self.active_clip_index
        self.active_clip_index = index
        self.active_progress = 0.0

        if old_index != index:
            self._remove_active_overlay()
            self._create_active_overlay()

        self._move_progress_line()

    def set_active_progress(self, progress):
        """Update progress within the active clip (0.0 to 1.0)"""
        self.active_progress = max(0.0, min(1.0, progress))
        self._move_progress_line()

    def clear_active_clip(self):
        """Remove active clip highlight"""
        self.active_clip_index = -1
        self.active_progress = 0.0
        self._remove_active_overlay()

    def _create_active_overlay(self):
        """Create glow overlay and progress line (once per clip)"""
        if self.active_clip_index < 0 or self.active_clip_index >= len(self.clip_items):
            return

        item = self.clip_items[self.active_clip_index]
        rect = item.rect()
        pos = item.pos()

        self._active_overlay = self.addRect(
            pos.x() - 3, pos.y() - 3,
            rect.width() + 6, rect.height() + 6,
            QPen(QColor(80, 180, 255, 200), 3),
            QBrush(QColor(80, 180, 255, 40))
        )
        self._active_overlay.setZValue(10)

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

        if not self._progress_line or self._progress_line not in self.items():
            self._progress_line = None
            self._active_overlay = None
            self._create_active_overlay()
            return

        item = self.clip_items[self.active_clip_index]
        rect = item.rect()
        pos = item.pos()
        x = pos.x() + rect.width() * self.active_progress
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

    def reorder_clip(self, clip_index, new_left_x):
        """Reorder clip based on its left edge position, not center"""
        if 0 <= clip_index < len(self.clips):
            clip = self.clips.pop(clip_index)
            insert_index = self.get_insert_index(new_left_x)
            
            # Adjust insert_index if we're moving forward
            if insert_index > clip_index:
                insert_index -= 1
            
            self.clips.insert(insert_index, clip)
            self.build_timeline()  # This recreates all clip items with new indices
            
            if insert_index != clip_index:
                self.show_drop_feedback(insert_index)
                self.clip_reordered.emit(clip_index, insert_index)

    # ── CUT / TRIM OPERATIONS ──────────────────────────────────────────
    def cut_clip_at(self, source_time):
        """
        Split whichever clip contains source_time into two clips.
        source_time must be at least 0.2 s from both edges.
        Returns True if a cut was made.
        """
        MIN_SIDE = 0.2

        for i, (start, end) in enumerate(self.clips):
            if start + MIN_SIDE < source_time < end - MIN_SIDE:
                self.clips[i] = (start, source_time)
                self.clips.insert(i + 1, (source_time, end))
                self.build_timeline()
                self.clip_cut.emit(source_time)
                return True

        return False

    def trim_clip_start(self, clip_index, new_start):
        """Move the in-point of a clip to new_start (clamped to ≥0.5s duration)."""
        MIN_DURATION = 0.5
        if not (0 <= clip_index < len(self.clips)):
            return
        start, end = self.clips[clip_index]
        new_start = max(start, min(new_start, end - MIN_DURATION))
        self.clips[clip_index] = (new_start, end)
        self.build_timeline()
        self.clip_trimmed.emit(clip_index)

    def trim_clip_end(self, clip_index, new_end):
        """Move the out-point of a clip to new_end (clamped to ≥0.5s duration)."""
        MIN_DURATION = 0.5
        if not (0 <= clip_index < len(self.clips)):
            return
        start, end = self.clips[clip_index]
        new_end = min(end, max(new_end, start + MIN_DURATION))
        self.clips[clip_index] = (start, new_end)
        self.build_timeline()
        self.clip_trimmed.emit(clip_index)

    # ── CUT INDICATOR LINE ─────────────────────────────────────────────
    def _update_cut_indicator(self, x, y, height):
        """Draw or reposition the dashed red vertical line previewing the cut."""
        if self._cut_line is not None and self._cut_line in self.items():
            self._cut_line.setLine(x, y, x, y + height)
        else:
            self._cut_line = self.addLine(
                x, y, x, y + height,
                QPen(QColor(255, 80, 80, 220), 2, Qt.DashLine)
            )
            self._cut_line.setZValue(50)

    def _hide_cut_indicator(self):
        """Remove the cut indicator line."""
        if self._cut_line is not None and self._cut_line in self.items():
            try:
                self.removeItem(self._cut_line)
            except RuntimeError:
                pass
        self._cut_line = None