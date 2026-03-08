"""
Real-Time BBox Overlay for Video Preview — v1

Uses QGraphicsScene + QGraphicsVideoItem to composite bounding boxes
directly on top of the video frame. No pre-rendering needed.

Two modes available:
  - LIVE:    Real-time overlay from cached detection data (this module)
  - PRECOMP: Pre-rendered annotated video swap (bbox_overlay.py)

Usage in signal_timeline_viewer.py:

    from video_ai_editor.realtime_overlay import RealtimeOverlayPreview

    # Replace QVideoWidget with this:
    self.overlay_preview = RealtimeOverlayPreview(
        video_path=self.video_path,
        cache_data=self.cache_data,
        parent=self,
    )
    layout.addWidget(self.overlay_preview)

    # Player is created internally:
    self.video_player = self.overlay_preview.player

    # Capture composited frame for LLM (video + bboxes = one image):
    base64_str = self.overlay_preview.capture_frame_base64()
"""

from __future__ import annotations

import os
import base64
from typing import Optional
from collections import defaultdict

from PySide6.QtCore import Qt, QRectF, QTimer, QUrl, QPointF, Signal, QSizeF
from PySide6.QtGui import (
    QColor, QPen, QBrush, QFont, QPainter, QImage, QTransform,
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QGraphicsTextItem, QGraphicsEllipseItem,
    QCheckBox, QLabel, QGroupBox, QComboBox, QSlider, QGraphicsItem,
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QGraphicsVideoItem


# ──────────────────────────────────────────────────────────────────
# Color palette for object classes
# ──────────────────────────────────────────────────────────────────

_CLASS_COLORS: dict[str, QColor] = {}
_HUE_STEP = 0.618033988749895  # golden ratio
_next_hue = 0.0


def color_for_class(class_name: str) -> QColor:
    """Deterministic, visually distinct color per object class."""
    global _next_hue
    key = class_name.lower().strip()
    if key not in _CLASS_COLORS:
        _CLASS_COLORS[key] = QColor.fromHsvF(_next_hue % 1.0, 0.85, 0.95)
        _next_hue += _HUE_STEP
    return _CLASS_COLORS[key]


# ──────────────────────────────────────────────────────────────────
# BBoxOverlayItem — a single bounding box with label
# ──────────────────────────────────────────────────────────────────

class BBoxOverlayItem(QGraphicsRectItem):
    """
    One bounding box rendered on the scene.
    Automatically hides/shows based on timestamp proximity.
    """

    def __init__(
        self,
        bbox: tuple[float, float, float, float],  # (x, y, w, h) normalised 0-1
        class_name: str,
        confidence: float,
        timestamp: float,
        parent: QGraphicsItem | None = None,
    ):
        super().__init__(parent)
        self.class_name = class_name
        self.confidence = confidence
        self.timestamp = timestamp
        self.bbox_norm = bbox  # stored normalised, mapped to scene in update_geometry

        color = color_for_class(class_name)

        # Box style
        pen = QPen(color, 2.5)
        pen.setCosmetic(True)  # constant thickness regardless of zoom
        self.setPen(pen)
        self.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 30)))

        # Label background + text
        self._label_bg = QGraphicsRectItem(self)
        self._label_bg.setBrush(QBrush(QColor(0, 0, 0, 160)))
        self._label_bg.setPen(QPen(Qt.NoPen))

        self._label = QGraphicsTextItem(self)
        self._label.setDefaultTextColor(color.lighter(140))
        font = QFont("Consolas", 9, QFont.Weight.Bold)
        self._label.setFont(font)
        self._label.setPlainText(f"{class_name} {confidence:.0%}")

        # Tooltip
        self.setToolTip(
            f"{class_name}\n"
            f"Confidence: {confidence:.1%}\n"
            f"Time: {timestamp:.2f}s"
        )

        self.setZValue(50)
        self.setVisible(False)

    def update_geometry(self, video_width: float, video_height: float):
        """Map normalised bbox to actual video pixel coords."""
        x, y, w, h = self.bbox_norm
        px = x * video_width
        py = y * video_height
        pw = w * video_width
        ph = h * video_height
        self.setRect(px, py, pw, ph)

        # Position label at top of box
        label_rect = self._label.boundingRect()
        self._label.setPos(px + 2, py - label_rect.height() - 1)
        self._label_bg.setRect(
            px, py - label_rect.height() - 2,
            label_rect.width() + 6, label_rect.height() + 2,
        )


# ──────────────────────────────────────────────────────────────────
# OverlayScene — scene with video item + bbox items
# ──────────────────────────────────────────────────────────────────

class OverlayScene(QGraphicsScene):
    """
    QGraphicsScene holding:
      - QGraphicsVideoItem  (bottom layer, z=0)
      - BBoxOverlayItems    (mid layer, z=50)
      - Crosshair / info    (top layer, z=100)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setBackgroundBrush(QBrush(QColor(0, 0, 0)))

        # Video item
        self.video_item = QGraphicsVideoItem()
        self.video_item.setZValue(0)
        self.video_item.setSize(QSizeF(1920, 1080))  # default until nativeSizeChanged
        self.addItem(self.video_item)
        self.setSceneRect(0, 0, 1920, 1080)

        # Bbox items indexed by timestamp bucket (100ms buckets)
        self._bbox_items: dict[int, list[BBoxOverlayItem]] = defaultdict(list)
        self._all_bbox_items: list[BBoxOverlayItem] = []
        self._visible_bucket: int = -1

        # Video dimensions (updated when native size changes)
        self._video_w: float = 1920
        self._video_h: float = 1080

        self.video_item.nativeSizeChanged.connect(self._on_native_size_changed)

    def _on_native_size_changed(self, size):
        """Update scene rect when video dimensions are known."""
        if size.width() > 0 and size.height() > 0:
            self._video_w = size.width()
            self._video_h = size.height()
            self.video_item.setSize(size)
            self.setSceneRect(0, 0, size.width(), size.height())

            # Recompute all bbox geometries
            for item in self._all_bbox_items:
                item.update_geometry(self._video_w, self._video_h)

    def load_detections(self, cache_data: dict, time_window: float = 0.5):
        """
        Load object/action detections from cache and create overlay items.

        Checks multiple cache key formats for maximum compatibility:

        Dedicated bbox keys (preferred):
            cache_data['object_bboxes'] = [
                {'timestamp': 2.5, 'objects': ['person'], 'bboxes': [[x,y,w,h]], 'confidences': [0.9]},
            ]
            cache_data['action_bboxes'] = [
                {'timestamp': 3.0, 'action_name': 'running', 'confidence': 0.85, 'bbox': [x,y,w,h]},
            ]

        Standard keys (fallback — entries with bbox data are used):
            cache_data['objects'] — entries that have a 'bboxes' field
            cache_data['actions'] — entries that have a 'bbox' field
        """
        # Clear existing items
        for item in self._all_bbox_items:
            self.removeItem(item)
        self._all_bbox_items.clear()
        self._bbox_items.clear()
        self._visible_bucket = -1

        count = 0

        # ── Objects with bboxes ──
        # Check dedicated key first, fall back to standard 'objects' key
        object_entries = cache_data.get('object_bboxes') or []
        if not object_entries:
            object_entries = [
                e for e in cache_data.get('objects', [])
                if e.get('bboxes') or e.get('bbox')
            ]

        for entry in object_entries:
            ts = entry.get('timestamp', 0)
            names = entry.get('objects', [])

            # Support both 'bboxes' (list) and 'bbox' (single)
            bboxes = entry.get('bboxes', [])
            if not bboxes and entry.get('bbox'):
                bboxes = [entry['bbox']] * max(1, len(names))

            confidences = entry.get('confidences', [])

            for i, name in enumerate(names):
                if i >= len(bboxes):
                    break

                bbox = bboxes[i]
                if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                    continue

                conf = confidences[i] if i < len(confidences) else 0.5

                item = BBoxOverlayItem(
                    bbox=tuple(bbox[:4]),
                    class_name=str(name),
                    confidence=float(conf),
                    timestamp=float(ts),
                )
                item.update_geometry(self._video_w, self._video_h)
                self.addItem(item)
                self._all_bbox_items.append(item)

                bucket = int(ts * 10)
                self._bbox_items[bucket].append(item)
                count += 1

        # ── Actions with bboxes ──
        # Check dedicated key first, fall back to standard 'actions' key
        action_entries = cache_data.get('action_bboxes') or []
        if not action_entries:
            action_entries = [
                e for e in cache_data.get('actions', [])
                if e.get('bbox')
            ]

        for entry in action_entries:
            ts = entry.get('timestamp', 0)
            bbox = entry.get('bbox')
            if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue

            name = entry.get('action_name') or entry.get('action', 'action')
            conf = entry.get('confidence', 0.5)

            item = BBoxOverlayItem(
                bbox=tuple(bbox[:4]),
                class_name=f"[A] {name}",
                confidence=float(conf),
                timestamp=float(ts),
            )
            item.update_geometry(self._video_w, self._video_h)
            self.addItem(item)
            self._all_bbox_items.append(item)

            bucket = int(ts * 10)
            self._bbox_items[bucket].append(item)
            count += 1

        # ── Debug output ──
        print(f"🎯 Loaded {count} bbox overlays ({len(self._bbox_items)} time buckets)")
        print(f"   Cache keys: object_bboxes={len(cache_data.get('object_bboxes') or [])}, "
            f"action_bboxes={len(cache_data.get('action_bboxes') or [])}, "
            f"objects(w/bbox)={len([e for e in cache_data.get('objects',[]) if e.get('bboxes')])}, "
            f"actions(w/bbox)={len([e for e in cache_data.get('actions',[]) if e.get('bbox')])}")
        if count == 0:
            relevant = [k for k in cache_data if any(s in k.lower() for s in ('bbox','detect','object','action'))]
            print(f"   ⚠️  No bbox data found. Relevant keys in cache: {relevant or 'none'}")
        return count

    def update_time(self, time_seconds: float, window: float = 0.3):
        """
        Show only bboxes near the current timestamp.
        Called every frame (~30fps) from position update.
        """
        center_bucket = int(time_seconds * 10)

        # Skip if same bucket (avoid redundant work)
        if center_bucket == self._visible_bucket:
            return
        self._visible_bucket = center_bucket

        # Calculate which buckets are in range
        half_window_buckets = max(1, int(window * 10))
        active_buckets = set(
            range(center_bucket - half_window_buckets,
                    center_bucket + half_window_buckets + 1)
        )

        # Show/hide items
        for bucket, items in self._bbox_items.items():
            visible = bucket in active_buckets
            for item in items:
                if item.isVisible() != visible:
                    item.setVisible(visible)

    def set_overlays_visible(self, visible: bool):
        """Toggle all overlays on/off."""
        for item in self._all_bbox_items:
            item.setVisible(False)  # always reset
        if not visible:
            self._visible_bucket = -1  # force re-eval when turned back on

    def get_visible_classes(self) -> set[str]:
        """Get set of class names currently having visible bboxes."""
        return {
            item.class_name
            for item in self._all_bbox_items
            if item.isVisible()
        }


# ──────────────────────────────────────────────────────────────────
# OverlayView — QGraphicsView with aspect-ratio-correct fitting
# ──────────────────────────────────────────────────────────────────

class OverlayView(QGraphicsView):
    """View that keeps video aspect ratio and supports smooth resize."""

    def __init__(self, scene: OverlayScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("QGraphicsView { background-color: black; border: none; }")
        self.setMinimumSize(320, 240)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_video()

    def showEvent(self, event):
        super().showEvent(event)
        # Refit when widget becomes visible (e.g. QStackedWidget page switch)
        QTimer.singleShot(50, self._fit_video)

    def _fit_video(self):
        """Fit scene into view maintaining aspect ratio."""
        scene = self.scene()
        if scene and scene.sceneRect().width() > 0:
            self.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


# ──────────────────────────────────────────────────────────────────
# RealtimeOverlayPreview — drop-in replacement for QVideoWidget
# ──────────────────────────────────────────────────────────────────

class RealtimeOverlayPreview(QWidget):
    """
    Complete video preview widget with real-time bbox overlay.

    Drop-in replacement for QVideoWidget + controls.
    Creates its own QMediaPlayer internally.
    """

    # Emitted when overlay mode changes
    overlay_toggled = Signal(bool)

    def __init__(
        self,
        video_path: str,
        cache_data: dict | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.cache_data = cache_data or {}

        self._overlay_enabled = False
        self._detection_count = 0

        self._init_ui()
        self._init_player()
        self._load_detections()

    @property
    def player(self) -> QMediaPlayer:
        """Access the internal media player."""
        return self._player

    @property
    def audio_output(self) -> QAudioOutput:
        return self._audio

    # ── Setup ──────────────────────────────────────────────────────

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Scene + View
        self._scene = OverlayScene()
        self._view = OverlayView(self._scene)
        layout.addWidget(self._view, 1)

        # Controls row
        controls = QHBoxLayout()
        controls.setContentsMargins(4, 0, 4, 4)

        # Overlay toggle
        self._overlay_cb = QCheckBox("🎯 Live BBox Overlay")
        self._overlay_cb.setChecked(False)
        self._overlay_cb.setToolTip(
            "Show bounding boxes from cached detections in real-time.\n"
            "Uses detection data already in cache — no GPU cost."
        )
        self._overlay_cb.stateChanged.connect(self._on_overlay_toggled)
        controls.addWidget(self._overlay_cb)

        # Detection count label
        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color: #888; font-size: 11px;")
        controls.addWidget(self._count_label)

        controls.addStretch()

        # Time window slider
        controls.addWidget(QLabel("Window:"))
        self._window_slider = QSlider(Qt.Horizontal)
        self._window_slider.setRange(1, 20)  # 0.1s to 2.0s
        self._window_slider.setValue(5)       # 0.5s default
        self._window_slider.setFixedWidth(80)
        self._window_slider.setToolTip("Time window for showing nearby detections (0.1s - 2.0s)")
        self._window_slider.valueChanged.connect(self._on_window_changed)
        controls.addWidget(self._window_slider)

        self._window_label = QLabel("0.5s")
        self._window_label.setStyleSheet("color: #aaa; font-size: 11px; min-width: 30px;")
        controls.addWidget(self._window_label)

        layout.addLayout(controls)

        # Style
        self.setStyleSheet("""
            QCheckBox { color: #d0d8ff; spacing: 6px; }
            QCheckBox::indicator { width: 16px; height: 16px; }
            QCheckBox::indicator:checked { background-color: #3a5fcd; border: 2px solid #5a7fdd; border-radius: 3px; }
            QCheckBox::indicator:unchecked { border: 2px solid #4a4a6a; border-radius: 3px; }
            QLabel { color: #d0d8ff; }
        """)

    def _init_player(self):
        """Create media player and wire it to the graphics video item."""
        self._player = QMediaPlayer()
        self._audio = QAudioOutput()
        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(self._scene.video_item)
        self._player.setSource(QUrl.fromLocalFile(self.video_path))
        self._audio.setVolume(0.8)

        # Update overlays on position change
        self._player.positionChanged.connect(self._on_position_changed)

        # Fit view once video dimensions are known
        self._scene.video_item.nativeSizeChanged.connect(
            lambda _: self._view._fit_video()
        )

    def _load_detections(self):
        """Load bbox data from cache."""
        self._detection_count = self._scene.load_detections(self.cache_data)

        if self._detection_count > 0:
            self._count_label.setText(f"({self._detection_count} detections loaded)")
            self._overlay_cb.setEnabled(True)
        else:
            self._count_label.setText("(no bbox data in cache)")
            self._overlay_cb.setEnabled(False)
            self._overlay_cb.setToolTip(
                "No bounding box data found in cache.\n"
                "Run detection with draw_bboxes=True and bbox saving enabled,\n"
                "or use the pre-rendered video swap instead."
            )

    # ── Slots ──────────────────────────────────────────────────────

    def _on_overlay_toggled(self, state):
        self._overlay_enabled = (state == Qt.Checked.value)
        self._scene.set_overlays_visible(self._overlay_enabled)

        if self._overlay_enabled:
            # Force immediate update at current position
            pos_sec = self._player.position() / 1000.0
            window = self._window_slider.value() / 10.0
            self._scene.update_time(pos_sec, window)

        self.overlay_toggled.emit(self._overlay_enabled)

    def _on_position_changed(self, position_ms: int):
        """Called ~30x per second during playback."""
        if not self._overlay_enabled:
            return
        time_sec = position_ms / 1000.0
        window = self._window_slider.value() / 10.0
        self._scene.update_time(time_sec, window)

    def _on_window_changed(self, value):
        window = value / 10.0
        self._window_label.setText(f"{window:.1f}s")

    # ── Public API ─────────────────────────────────────────────────

    def set_cache_data(self, cache_data: dict):
        """Update detection data (e.g. after new analysis)."""
        self.cache_data = cache_data
        self._load_detections()

    def set_overlay_visible(self, visible: bool):
        """Programmatically toggle overlay."""
        self._overlay_cb.setChecked(visible)

    def capture_frame_base64(self, max_dim: int = 1024) -> str | None:
        """
        Capture current video frame WITH bboxes composited into one image.

        Returns base64-encoded JPEG string ready for LLM vision API.
        This is the key advantage over pre-rendered approach:
        one scene.render() call = video + overlays = single image.
        """
        scene = self._scene
        scene_rect = scene.sceneRect()
        if scene_rect.width() <= 0 or scene_rect.height() <= 0:
            return None

        # Determine output size (respect max_dim)
        w = scene_rect.width()
        h = scene_rect.height()
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            w = int(w * scale)
            h = int(h * scale)
        else:
            w, h = int(w), int(h)

        # Render scene → QImage (video + all visible overlays)
        image = QImage(w, h, QImage.Format.Format_RGB888)
        image.fill(QColor(0, 0, 0))

        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        scene.render(painter, QRectF(0, 0, w, h), scene_rect)
        painter.end()

        # Convert to base64 JPEG
        from PySide6.QtCore import QBuffer, QIODevice

        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        image.save(buffer, "JPEG", 90)
        buffer.close()

        b64 = base64.b64encode(buffer.data().data()).decode('utf-8')
        return b64

    def capture_frame_qimage(self) -> QImage | None:
        """Capture composited frame as QImage (for local use)."""
        scene = self._scene
        scene_rect = scene.sceneRect()
        if scene_rect.width() <= 0:
            return None

        w, h = int(scene_rect.width()), int(scene_rect.height())
        image = QImage(w, h, QImage.Format.Format_RGB888)
        image.fill(QColor(0, 0, 0))

        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        scene.render(painter, QRectF(0, 0, w, h), scene_rect)
        painter.end()
        return image

    def get_detection_count(self) -> int:
        """Number of bbox detections loaded from cache."""
        return self._detection_count

    def get_visible_classes(self) -> set[str]:
        """Get currently visible object/action classes."""
        return self._scene.get_visible_classes()


# ──────────────────────────────────────────────────────────────────
# Standalone test
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton

    app = QApplication(sys.argv)

    if len(sys.argv) < 2:
        print("Usage: python realtime_overlay.py <video_path> [cache.json]")
        sys.exit(1)

    video_path = sys.argv[1]

    # Load cache if provided
    cache_data = {}
    if len(sys.argv) > 2:
        import json
        with open(sys.argv[2]) as f:
            cache_data = json.load(f)

    # Create test window
    win = QMainWindow()
    win.setWindowTitle("Real-Time BBox Overlay Test")
    win.resize(1280, 800)

    preview = RealtimeOverlayPreview(
        video_path=video_path,
        cache_data=cache_data,
    )

    # Add play button
    central = QWidget()
    layout = QVBoxLayout(central)
    layout.addWidget(preview, 1)

    btn_row = QHBoxLayout()
    play_btn = QPushButton("▶ Play / Pause")
    play_btn.clicked.connect(
        lambda: preview.player.pause()
        if preview.player.playbackState() == QMediaPlayer.PlayingState
        else preview.player.play()
    )
    btn_row.addWidget(play_btn)

    capture_btn = QPushButton("📷 Capture Frame")
    def _capture():
        b64 = preview.capture_frame_base64()
        if b64:
            print(f"Captured frame: {len(b64) // 1024} KB base64")
            # Save to file for inspection
            import base64 as b64mod
            with open("captured_frame.jpg", "wb") as f:
                f.write(b64mod.b64decode(b64))
            print("Saved to captured_frame.jpg")
        else:
            print("No frame captured")
    capture_btn.clicked.connect(_capture)
    btn_row.addWidget(capture_btn)

    layout.addLayout(btn_row)
    win.setCentralWidget(central)

    win.show()
    sys.exit(app.exec())