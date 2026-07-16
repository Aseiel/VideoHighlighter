"""
Real-Time BBox Overlay for Video Preview — v2 (Memory Optimized)

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

OPTIMIZATIONS:
- Lazy loading: bboxes loaded on demand, not all at once
- Time-based filtering: only loads bboxes for current time window
- Memory cleanup: releases bbox data when not in use
- Bucket caching: caches only active time buckets
"""

from __future__ import annotations

import os
import base64
from typing import Optional
import gc
from typing import Optional, Dict, List, Set, Tuple
from collections import defaultdict
from video_ai_editor.face_identity import FaceIdentityBank
from video_ai_editor.live_face import LiveFaceController, LiveFaceOverlay

from PySide6.QtCore import Qt, QRectF, QTimer, QUrl, QPointF, Signal, QSizeF
from PySide6.QtGui import (
    QColor, QPen, QBrush, QFont, QPainter, QImage, QTransform, QAction,
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QGraphicsTextItem, QGraphicsEllipseItem,
    QCheckBox, QLabel, QGroupBox, QComboBox, QSlider, QGraphicsItem, QMenu, QInputDialog,
    QToolButton,
)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QGraphicsVideoItem

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


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


def _strip_badge(name: str) -> str:
    """Drop a leading '[XX] ' abbreviation badge from an overlay label."""
    s = str(name).strip()
    if s.startswith('[') and ']' in s:
        s = s[s.index(']') + 1:].strip()
    return s


def _palette_color(i: int) -> QColor:
    """Same formula as SignalTimelineScene._color_palette, so overlay colours
    match the timeline for the same sorted class index."""
    return QColor.fromHsvF((i * 0.618) % 1.0, 0.85, 0.92)


def action_abbrev(name: str) -> str:
    """Short code from a class name: initials of its words, e.g.
    'jump' -> 'J', 'high five' -> 'HF', 'sit down' -> 'SD'."""
    words = [w for w in str(name).replace('_', ' ').split() if w]
    if not words:
        return "?"
    return "".join(w[0] for w in words).upper()[:3]


# ──────────────────────────────────────────────────────────────────
# LazyBBoxLoader — loads bboxes on demand
# ──────────────────────────────────────────────────────────────────

class LazyBBoxLoader:
    """
    Loads bbox data lazily from cache.
    Only loads data when needed for a specific time range.
    Caches recently accessed buckets with LRU-like behavior.
    """
    
    def __init__(self, cache_data: dict, max_cached_buckets: int = 20):
        """
        Args:
            cache_data: The full cache dictionary
            max_cached_buckets: Maximum number of time buckets to keep in memory
        """
        self.cache_data = cache_data
        self.max_cached_buckets = max_cached_buckets
        
        # Store raw bbox data (loaded lazily)
        self._object_bboxes: List[dict] = []
        self._action_bboxes: List[dict] = []
        self._loaded = False
        self._load_count = 0
        
        # Cache for time buckets (dict of bucket -> list of bbox items)
        self._bucket_cache: Dict[int, List[dict]] = {}
        self._bucket_access_order: List[int] = []
        
        # Build index of timestamps without loading all data
        self._timestamp_index: Dict[int, List[dict]] = defaultdict(list)
        self._index_loaded = False
        
        # Memory tracking
        self._peak_memory_mb = 0
        
    def _ensure_loaded(self):
        """Load all bbox data from cache (called once)."""
        if self._loaded:
            return
        
        print(f"📦 Loading bbox data from cache...")
        
        # ── Object bboxes ──
        self._object_bboxes = self.cache_data.get('object_bboxes', []) or []
        # Fallback: check objects entries with bboxes
        if not self._object_bboxes:
            self._object_bboxes = [
                e for e in self.cache_data.get('objects', [])
                if e.get('bboxes') or e.get('bbox')
            ]
        
        # ── Action bboxes ──
        self._action_bboxes = self.cache_data.get('action_bboxes', []) or []
        # Fallback: check actions entries with bboxes
        if not self._action_bboxes:
            self._action_bboxes = [
                e for e in self.cache_data.get('actions', [])
                if e.get('bbox')
            ]
        
        self._loaded = True
        self._load_count = len(self._object_bboxes) + len(self._action_bboxes)
        
        print(f"   Loaded {self._load_count} bbox overlays")
        print(f"   Object bboxes: {len(self._object_bboxes)}, Action bboxes: {len(self._action_bboxes)}")
        
        # Build timestamp index
        self._build_timestamp_index()
        
        # Force garbage collection
        gc.collect()
        
        if HAS_PSUTIL:
            mem = psutil.Process().memory_info().rss / (1024 * 1024)
            print(f"   Memory after load: {mem:.1f} MB")
    
    def _build_timestamp_index(self):
        """Build index of timestamps for fast lookup."""
        if self._index_loaded:
            return
        
        # Index object bboxes
        for entry in self._object_bboxes:
            ts = entry.get('timestamp', 0)
            bucket = int(ts * 10)
            self._timestamp_index[bucket].append(('object', entry))
        
        # Index action bboxes
        for entry in self._action_bboxes:
            ts = entry.get('timestamp', 0)
            bucket = int(ts * 10)
            self._timestamp_index[bucket].append(('action', entry))
        
        self._index_loaded = True
        print(f"   Built index: {len(self._timestamp_index)} time buckets")
    
    def get_bboxes_for_time(self, time_seconds: float, window: float = 0.5) -> List[dict]:
        """
        Get bboxes for a specific time range.
        Only loads data from cache when needed.
        
        Args:
            time_seconds: Current time in seconds
            window: Time window in seconds (bboxes within this range are returned)
        
        Returns:
            List of bbox entries with their data
        """
        self._ensure_loaded()
        
        # Calculate bucket range
        center_bucket = int(time_seconds * 10)
        half_window = max(1, int(window * 10))
        start_bucket = center_bucket - half_window
        end_bucket = center_bucket + half_window
        
        # Get bboxes from cache
        result = []
        buckets_to_load = []
        
        for bucket in range(start_bucket, end_bucket + 1):
            if bucket in self._bucket_cache:
                # Already cached
                result.extend(self._bucket_cache[bucket])
            else:
                # Need to load this bucket
                buckets_to_load.append(bucket)
        
        if buckets_to_load:
            # Load missing buckets from index
            for bucket in buckets_to_load:
                if bucket in self._timestamp_index:
                    entries = self._timestamp_index[bucket]
                    # Convert to dict format expected by BBoxOverlayItem
                    bboxes = []
                    for entry_type, entry in entries:
                        if entry_type == 'object':
                            bboxes.extend(self._object_entry_to_bboxes(entry))
                        else:  # action
                            bboxes.extend(self._action_entry_to_bboxes(entry))
                    
                    self._bucket_cache[bucket] = bboxes
                    result.extend(bboxes)
                    
                    # Update access order
                    if bucket in self._bucket_access_order:
                        self._bucket_access_order.remove(bucket)
                    self._bucket_access_order.append(bucket)
                    
                    # LRU cleanup
                    while len(self._bucket_cache) > self.max_cached_buckets:
                        oldest = self._bucket_access_order.pop(0)
                        if oldest in self._bucket_cache:
                            del self._bucket_cache[oldest]
        
        return result
    
    def _object_entry_to_bboxes(self, entry: dict) -> List[dict]:
        """Convert an object entry to bbox dicts."""
        ts = entry.get('timestamp', 0)
        names = entry.get('objects', [])
        bboxes = entry.get('bboxes', [])
        if not bboxes and entry.get('bbox'):
            bboxes = [entry['bbox']] * max(1, len(names))
        confidences = entry.get('confidences', [])
        
        result = []
        for i, name in enumerate(names):
            if i >= len(bboxes):
                break
            bbox = bboxes[i]
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            conf = confidences[i] if i < len(confidences) else 0.5
            result.append({
                'timestamp': ts,
                'class_name': str(name),
                'confidence': float(conf),
                'bbox': tuple(bbox[:4]),
                'type': 'object'
            })
        return result
    
    def _action_entry_to_bboxes(self, entry: dict) -> List[dict]:
        """Convert an action entry to bbox dicts."""
        ts = entry.get('timestamp', 0)
        bbox = entry.get('bbox')
        if not bbox or not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return []
        
        name = entry.get('action_name') or entry.get('action', 'action')
        conf = entry.get('confidence', 0.5)
        
        return [{
            'timestamp': ts,
            'class_name': f"[{action_abbrev(name)}] {name}",
            'confidence': float(conf),
            'bbox': tuple(bbox[:4]),
            'type': 'action'
        }]
    
    def get_all_class_names(self) -> set[str]:
        """All formatted class names present, matching BBoxOverlayItem.class_name
        (objects raw, actions prefixed with their abbreviation)."""
        self._ensure_loaded()
        names: set[str] = set()
        for entry in self._object_bboxes:
            for n in entry.get('objects', []) or []:
                if n:
                    names.add(str(n))
        for entry in self._action_bboxes:
            name = entry.get('action_name') or entry.get('action', 'action')
            names.add(f"[{action_abbrev(name)}] {name}")
        return names

    def clear_cache(self):
        """Clear the bucket cache to free memory."""
        self._bucket_cache.clear()
        self._bucket_access_order.clear()
        gc.collect()
    
    def get_total_count(self) -> int:
        """Get total number of bboxes without loading all data."""
        if self._loaded:
            return self._load_count
        
        # Count from cache without loading
        count = len(self.cache_data.get('object_bboxes', []))
        count += len(self.cache_data.get('action_bboxes', []))
        
        # Check fallback entries
        if count == 0:
            count += len([e for e in self.cache_data.get('objects', []) if e.get('bboxes') or e.get('bbox')])
            count += len([e for e in self.cache_data.get('actions', []) if e.get('bbox')])
        
        return count
    
    def get_memory_usage_mb(self) -> float:
        """Get current memory usage of the loader."""
        if not HAS_PSUTIL:
            return 0.0
        
        try:
            mem = psutil.Process().memory_info().rss / (1024 * 1024)
            if mem > self._peak_memory_mb:
                self._peak_memory_mb = mem
            return mem
        except:
            return 0.0


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
        color: QColor | None = None,
    ):
        super().__init__(parent)
        self.class_name = class_name
        self.confidence = confidence
        self.timestamp = timestamp
        self.bbox_norm = bbox  # stored normalised, mapped to scene in update_geometry

        if color is None:
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

        # Scale the label font with the video resolution. The scene is video-sized
        # (e.g. 1920x1080) and gets downscaled to fit the small preview, so a fixed
        # point-size font shrinks to nothing — and worse on higher-res videos.
        # Sizing it relative to video height keeps labels readable and consistent
        # across resolutions.
        font = self._label.font()
        font.setPixelSize(max(16, int(video_height * 0.035)))
        self._label.setFont(font)

        # Cache geometry so the scene can de-overlap labels without recomputing.
        label_rect = self._label.boundingRect()
        self._px = px
        self._py = py
        self._ph = ph
        self._label_w = label_rect.width()
        self._label_h = label_rect.height()
        self._set_label_top(self.default_label_top())

    def default_label_top(self) -> float:
        """Default label top: just above the box."""
        return self._py - self._label_h - 1

    def below_label_top(self) -> float:
        """Label top when placed below the box (used near the top edge)."""
        return self._py + self._ph + 1

    def _set_label_top(self, top: float):
        """Position the label/background at scene y = top."""
        self._label.setPos(self._px + 2, top)
        self._label_bg.setRect(self._px, top - 1, self._label_w + 6, self._label_h + 2)

    def label_rect_at(self, top: float) -> QRectF:
        """Background rect of the label in scene coords at the given top y."""
        return QRectF(self._px, top - 1, self._label_w + 6, self._label_h + 2)


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
        
        # Track which items are currently visible
        self._visible_count: int = 0

        # Per-class filter + last computed active time window (for re-applying)
        self._hidden_classes: set[str] = set()
        self._active_buckets: set[int] = set()

        # class_name -> QColor, mirroring the timeline's palette (built on load)
        self._class_colors: dict[str, QColor] = {}

        # Video dimensions (updated when native size changes)
        self._video_w: float = 1920
        self._video_h: float = 1080

        self.video_item.nativeSizeChanged.connect(self._on_native_size_changed)

    # Max scene resolution for the overlay compositor.
    # The video item is scaled DOWN to this before Qt renders bboxes on top.
    # Bboxes use normalised [0,1] coords so they map correctly regardless of scale.
    _MAX_SCENE_W = 1280
    _MAX_SCENE_H = 720

    def _on_native_size_changed(self, size):
        """Update scene rect when video dimensions are known."""
        if size.width() <= 0 or size.height() <= 0:
            return

        # Keep track of native resolution (used for bbox coord conversion)
        self._video_w = size.width()
        self._video_h = size.height()

        # Scale DOWN to _MAX_SCENE dimensions so Qt doesn't composite at 4K.
        # Bboxes use normalised coords so they scale with the scene automatically.
        scale = min(1.0,
                    self._MAX_SCENE_W / size.width(),
                    self._MAX_SCENE_H / size.height())
        scene_w = size.width()  * scale
        scene_h = size.height() * scale

        self.video_item.setSize(QSizeF(scene_w, scene_h))
        self.setSceneRect(0, 0, scene_w, scene_h)

        # Recompute bbox geometries against the (possibly downscaled) scene size
        for item in self._all_bbox_items:
            item.update_geometry(scene_w, scene_h)
        self._resolve_label_overlaps()

    def load_detections_lazy(self, bbox_loader: LazyBBoxLoader):
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

        Load detections from the lazy loader.
        Only creates items that will be visible soon.
        """
        # Clear existing items
        for item in self._all_bbox_items:
            self.removeItem(item)
        self._all_bbox_items.clear()
        self._bbox_items.clear()
        self._visible_bucket = -1
        
        # We don't load all bboxes here - they'll be loaded on demand
        # Just store the loader reference
        self._bbox_loader = bbox_loader
        self._detection_count = bbox_loader.get_total_count()
        self._build_class_colors()
        print(f"🎯 Lazy loader ready: {self._detection_count} bboxes available")
        return self._detection_count

    def _build_class_colors(self):
        """Assign each class the SAME colour the timeline uses: the timeline
        palette indexed by the class's sorted position within its group
        (actions / objects). Keyed by the formatted class_name items carry."""
        self._class_colors = {}
        try:
            names = self._bbox_loader.get_all_class_names()
        except Exception:
            return
        action_disp = sorted({_strip_badge(n).title() for n in names if n.startswith('[')})
        object_disp = sorted({n.title() for n in names if not n.startswith('[')})
        action_idx = {d: i for i, d in enumerate(action_disp)}
        object_idx = {d: i for i, d in enumerate(object_disp)}
        for n in names:
            if n.startswith('['):
                self._class_colors[n] = _palette_color(action_idx.get(_strip_badge(n).title(), 0))
            else:
                self._class_colors[n] = _palette_color(object_idx.get(n.title(), 0))

    def update_time(self, time_seconds: float, window: float = 0.3):
        """
        Show only bboxes near the current timestamp.
        Lazily loads bboxes from cache as needed.
        """
        center_bucket = int(time_seconds * 10)

        # Skip if same bucket (avoid redundant work)
        if center_bucket == self._visible_bucket:
            return
        self._visible_bucket = center_bucket
        
        # Get bboxes from lazy loader for this time range
        if hasattr(self, '_bbox_loader'):
            bbox_data = self._bbox_loader.get_bboxes_for_time(time_seconds, window)
            
            # Check if we need to add new items
            existing_timestamps = {item.timestamp for item in self._all_bbox_items}
            new_items = []
            
            for bbox in bbox_data:
                ts = bbox.get('timestamp', 0)
                # Only add if not already in scene
                if ts not in existing_timestamps:
                    item = BBoxOverlayItem(
                        bbox=bbox['bbox'],
                        class_name=bbox['class_name'],
                        confidence=bbox['confidence'],
                        timestamp=ts,
                        color=self._class_colors.get(bbox['class_name']),
                    )
                    scene_r = self.sceneRect()
                    item.update_geometry(scene_r.width(), scene_r.height())
                    self.addItem(item)
                    self._all_bbox_items.append(item)
                    new_items.append(item)
                    bucket = int(ts * 10)
                    self._bbox_items[bucket].append(item)
            
            if new_items:
                print(f"📦 Added {len(new_items)} new bbox items at {time_seconds:.1f}s")
        
        # Calculate which buckets are in range
        half_window_buckets = max(1, int(window * 10))
        active_buckets = set(
            range(center_bucket - half_window_buckets,
                    center_bucket + half_window_buckets + 1)
        )

        # Show/hide items (respecting the per-class filter), then de-overlap labels
        self._active_buckets = active_buckets
        self._apply_visibility()

    def _apply_visibility(self):
        """Show items whose bucket is in the active window and whose class isn't
        hidden by the filter; then de-overlap the visible labels."""
        visible_count = 0
        for bucket, items in self._bbox_items.items():
            in_window = bucket in self._active_buckets
            for item in items:
                vis = in_window and item.class_name not in self._hidden_classes
                if item.isVisible() != vis:
                    item.setVisible(vis)
                if vis:
                    visible_count += 1
        self._visible_count = visible_count
        self._resolve_label_overlaps()

    def set_class_hidden(self, class_name: str, hidden: bool):
        """Toggle whether a detection class is shown in the overlay."""
        if hidden:
            self._hidden_classes.add(class_name)
        else:
            self._hidden_classes.discard(class_name)
        self._apply_visibility()

    def _resolve_label_overlaps(self):
        """Nudge overlapping labels so close boxes don't bury each other's labels.
        Greedy: place left-to-right/top-down, stack a colliding label upward by its
        own height; if that runs off the top of the frame, stack it downward (below
        the box) instead."""
        visible = [it for it in self._all_bbox_items
                   if it.isVisible() and hasattr(it, "_label_h")]
        placed: list[QRectF] = []
        step = lambda it: it._label_h + 2

        def _stack(it, start_top, direction):
            top = start_top
            rect = it.label_rect_at(top)
            guard = 0
            while guard < 40 and any(rect.intersects(r) for r in placed):
                top += direction * step(it)
                rect = it.label_rect_at(top)
                guard += 1
            return top, rect

        for it in sorted(visible, key=lambda i: (i._px, i._py)):
            top, rect = _stack(it, it.default_label_top(), -1)   # upward
            if top < 0:                                          # off the top edge
                top, rect = _stack(it, it.below_label_top(), +1) # downward instead
            it._set_label_top(top)
            placed.append(rect)

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
    
    def get_memory_usage(self) -> int:
        """Get number of items currently in scene."""
        return len(self._all_bbox_items)
    
    def clear_items(self):
        """Remove all bbox items to free memory."""
        for item in self._all_bbox_items:
            self.removeItem(item)
        self._all_bbox_items.clear()
        self._bbox_items.clear()
        self._visible_bucket = -1
        if hasattr(self, '_bbox_loader'):
            self._bbox_loader.clear_cache()
        gc.collect()


# ──────────────────────────────────────────────────────────────────
# OverlayView — QGraphicsView with aspect-ratio-correct fitting
# ──────────────────────────────────────────────────────────────────

class OverlayView(QGraphicsView):
    """View that keeps video aspect ratio and supports smooth resize."""
    identity_context_requested = Signal(object, object)   # (identity_id, global_pos)

    def __init__(self, scene: OverlayScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("QGraphicsView { background-color: black; border: none; }")
        self.setMinimumSize(320, 240)
        self._vr_mode = False

    def set_vr_mode(self, enabled: bool):
        """Show only the left half of the scene (SBS VR videos)."""
        self._vr_mode = enabled
        self._fit_video()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_video()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(50, self._fit_video)

    def _fit_video(self):
        """Fit scene into view maintaining aspect ratio."""
        scene = self.scene()
        if not scene or scene.sceneRect().width() <= 0:
            return
        rect = scene.sceneRect()
        if self._vr_mode:
            # Show left half only — right eye view is identical for SBS content
            rect = QRectF(0, 0, rect.width() / 2, rect.height())
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        
    def contextMenuEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        hit = None
        for item in self.scene().items():
            iid = item.data(0)
            if iid and item.isVisible() and item.sceneBoundingRect().contains(scene_pos):
                if hit is None or item.zValue() >= hit.zValue():
                    hit = item
        if hit is not None:
            self.identity_context_requested.emit(hit.data(0), event.globalPos())
            event.accept()
            return
        super().contextMenuEvent(event)


# ──────────────────────────────────────────────────────────────────
# RealtimeOverlayPreview — drop-in replacement for QVideoWidget
# ──────────────────────────────────────────────────────────────────

class RealtimeOverlayPreview(QWidget):
    """
    Complete video preview widget with real-time bbox overlay.

    Drop-in replacement for QVideoWidget + controls.
    Creates its own QMediaPlayer internally.
    
    OPTIMIZATIONS:
    - Lazy loading: bboxes loaded on demand
    - LRU cache: only keeps recent time buckets in memory
    - Memory cleanup: clears items when overlay is off
    """

    # Emitted when overlay mode changes
    overlay_toggled = Signal(bool)
    avoid_person_requested = Signal(str)

    def __init__(
        self,
        video_path: str,
        cache_data: dict | None = None,
        parent: QWidget | None = None,
        max_cached_buckets: int = 20,
    ):
        super().__init__(parent)
        self.video_path = video_path
        self.cache_data = cache_data or {}

        self._overlay_enabled = False
        self._detection_count = 0
        self._max_cached_buckets = max_cached_buckets

        self._init_ui()
        self._view.identity_context_requested.connect(self._on_identity_context)
        self._init_player()
        
        self._face_bank = None
        self._live_face = None
        self._live_overlay = None
        # True while the mode combo is on "Live (real-time)". Real-time inference
        # only runs when this AND the overlay checkbox are both on, so unchecking
        # the checkbox pauses processing without leaving Live mode.
        self._live_face_mode = False
        # Submenu handle for the live 'Facial recognition' filter group.
        self._face_filter_menu: QMenu | None = None

        # Create lazy loader (doesn't load data yet)
        self._bbox_loader = LazyBBoxLoader(self.cache_data, max_cached_buckets)
        self._load_detections_lazy()

        # Memory tracking
        self._memory_timer = QTimer()
        self._memory_timer.timeout.connect(self._log_memory)
        self._memory_timer.start(5000)  # Log every 5 seconds

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
            "Uses detection data already in cache — no GPU cost.\n"
            "Only loads bboxes near current playhead position."
        )
        self._overlay_cb.stateChanged.connect(self._on_overlay_toggled)
        controls.addWidget(self._overlay_cb)

        # Detection count label
        self._count_label = QLabel("")
        self._count_label.setStyleSheet("color: #888; font-size: 11px;")
        controls.addWidget(self._count_label)

        # Per-class overlay filter (show/hide each detection class)
        self._filter_btn = QToolButton()
        self._filter_btn.setText("🔍 Filter")
        self._filter_btn.setPopupMode(QToolButton.InstantPopup)
        self._filter_btn.setToolTip("Show/hide individual detection classes on the overlay")
        self._filter_menu = QMenu(self._filter_btn)
        self._filter_btn.setMenu(self._filter_menu)
        self._filter_btn.setEnabled(False)
        self._filter_actions: dict[str, QAction] = {}
        controls.addWidget(self._filter_btn)

        controls.addStretch()

        # Memory usage label
        self._mem_label = QLabel("")
        self._mem_label.setStyleSheet("color: #666; font-size: 10px;")
        controls.addWidget(self._mem_label)

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
            QCheckBox { color: #d4d4d4; spacing: 6px; }
            QCheckBox::indicator { width: 16px; height: 16px; }
            QCheckBox::indicator:checked { background-color: #3a5fcd; border: 2px solid #5a7fdd; border-radius: 3px; }
            QCheckBox::indicator:unchecked { border: 2px solid #4a4a4a; border-radius: 3px; }
            QLabel { color: #d4d4d4; }
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

    def _load_detections_lazy(self):
        """Load bbox data lazily from cache."""
        self._detection_count = self._scene.load_detections_lazy(self._bbox_loader)

        if self._detection_count > 0:
            self._count_label.setText(f"({self._detection_count} detections available)")
            self._overlay_cb.setEnabled(True)
        else:
            self._count_label.setText("(no bbox data in cache)")
            # Don't force-disable here — real-time mode re-enables the checkbox
            # even with an empty cache (see set_live_face_enabled).
            if not self._live_face_mode:
                self._overlay_cb.setEnabled(False)
            self._overlay_cb.setToolTip(
                "No bounding box data found in cache.\n"
                "Run detection with draw_bboxes=True and bbox saving enabled,\n"
                "or use the pre-rendered video swap instead."
            )

        # Always build the filter — it now also hosts the live 'Facial recognition'
        # group, which is available in real-time mode regardless of cached data.
        self._build_filter_menu()

    def _build_filter_menu(self):
        """Populate the overlay filter, grouped by detection source so each
        detected thing can be shown/hidden individually:

            🧊 Object recognition   → cached object classes
            🎬 Action recognition   → cached action classes
            🙂 Facial recognition   → recognised faces (live, from the face bank)

        Object/action groups come from the cache and are fixed for the clip.
        The facial group is rebuilt each time it opens, because faces are
        recognised live and identities appear / get named as playback runs.
        """
        self._filter_menu.clear()
        self._filter_actions.clear()
        self._face_filter_menu = None

        names = sorted(self._bbox_loader.get_all_class_names())
        # Actions carry a '[AB] ' badge prefix (see _action_entry_to_bboxes);
        # everything else is an object class.
        object_names = [n for n in names if not n.startswith('[')]
        action_names = [n for n in names if n.startswith('[')]

        if object_names:
            self._add_class_group("🧊 Object recognition", object_names)
        if action_names:
            self._add_class_group("🎬 Action recognition", action_names)

        # Facial recognition is always offered — it's driven by the live face
        # worker, not the cache. Rebuilt on open so newly seen faces show up.
        self._face_filter_menu = self._filter_menu.addMenu("🙂 Facial recognition")
        self._face_filter_menu.aboutToShow.connect(self._rebuild_face_filter)

        if self._filter_actions:
            self._filter_menu.addSeparator()
            show_all = self._filter_menu.addAction("Show all classes")
            show_all.triggered.connect(lambda: self._set_all_classes(True))
            hide_all = self._filter_menu.addAction("Hide all classes")
            hide_all.triggered.connect(lambda: self._set_all_classes(False))

        # Reachable whenever there are cached classes or real-time faces are live.
        self._filter_btn.setEnabled(bool(self._filter_actions) or self._live_face_mode)

    def _add_class_group(self, title: str, names: list[str]):
        """Add one category submenu of per-class show/hide toggles.

        `names` are the raw class_name keys the scene filters on; the badge
        prefix is stripped for display only.
        """
        sub = self._filter_menu.addMenu(title)
        group_actions: list[QAction] = []

        show_all = sub.addAction("Show all")
        hide_all = sub.addAction("Hide all")
        sub.addSeparator()

        for name in names:
            act = QAction(_strip_badge(name), sub)
            act.setCheckable(True)
            act.setChecked(name not in self._scene._hidden_classes)
            act.toggled.connect(
                lambda checked, n=name: self._scene.set_class_hidden(n, not checked)
            )
            sub.addAction(act)
            self._filter_actions[name] = act
            group_actions.append(act)

        show_all.triggered.connect(
            lambda: [a.setChecked(True) for a in group_actions if not a.isChecked()]
        )
        hide_all.triggered.connect(
            lambda: [a.setChecked(False) for a in group_actions if a.isChecked()]
        )

    def _rebuild_face_filter(self):
        """(Re)populate the facial-recognition submenu from the face bank.

        Faces are recognised live, so this runs each time the submenu opens.
        Each identity (named or auto-enrolled) gets its own show/hide toggle.
        """
        menu = self._face_filter_menu
        if menu is None:
            return
        menu.clear()

        if self._face_bank is None:
            act = menu.addAction("Select “Live (real-time)” to recognise faces")
            act.setEnabled(False)
            return

        identities = self._face_bank.all_identities()
        if not identities:
            act = menu.addAction("(no faces recognised yet)")
            act.setEnabled(False)
            return

        face_actions: list[QAction] = []
        show_all = menu.addAction("Show all")
        hide_all = menu.addAction("Hide all")
        menu.addSeparator()

        for ident in identities:
            iid = ident["id"]
            disp = ident.get("name") or f"Person {iid[:8]}"
            act = QAction(disp, menu)
            act.setCheckable(True)
            act.setChecked(not self._is_identity_hidden(iid))
            act.toggled.connect(
                lambda checked, _id=iid: self._set_identity_hidden(_id, not checked)
            )
            menu.addAction(act)
            face_actions.append(act)

        show_all.triggered.connect(
            lambda: [a.setChecked(True) for a in face_actions if not a.isChecked()]
        )
        hide_all.triggered.connect(
            lambda: [a.setChecked(False) for a in face_actions if a.isChecked()]
        )

    def _is_identity_hidden(self, identity_id: str) -> bool:
        return (
            self._live_overlay is not None
            and identity_id in self._live_overlay.hidden_ids
        )

    def _set_identity_hidden(self, identity_id: str, hidden: bool):
        """Show/hide one recognised face on the live overlay."""
        if self._live_overlay is not None:
            self._live_overlay.set_identity_hidden(identity_id, hidden)

    def _set_all_classes(self, show: bool):
        """Check/uncheck every object/action class (each toggle updates the filter)."""
        for act in self._filter_actions.values():
            if act.isChecked() != show:
                act.setChecked(show)

    def _log_memory(self):
        """Log current memory usage."""
        if not HAS_PSUTIL:
            return
        try:
            mem = psutil.Process().memory_info().rss / (1024 * 1024)
            scene_items = self._scene.get_memory_usage()
            cached_buckets = len(self._bbox_loader._bucket_cache) if hasattr(self, '_bbox_loader') else 0
            self._mem_label.setText(f"💾 {mem:.0f}MB | Items: {scene_items} | Cache: {cached_buckets}buckets")
        except:
            pass

    # ── Slots ──────────────────────────────────────────────────────

    def _on_overlay_toggled(self, state):
        self._overlay_enabled = (state == Qt.Checked.value)
        self._scene.set_overlays_visible(self._overlay_enabled)
        
        if self._overlay_enabled:
            # Force immediate update at current position
            pos_sec = self._player.position() / 1000.0
            window = self._window_slider.value() / 10.0
            self._scene.update_time(pos_sec, window)
        else:
            # Clear items when overlay is off to free memory
            self._scene.clear_items()
            self._bbox_loader.clear_cache()

        # In real-time mode the same checkbox gates the face-recognition worker,
        # so unchecking it actually stops processing (not just hides cached boxes).
        self._apply_live_face_state()

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
        
        # Update immediately if overlay is enabled
        if self._overlay_enabled:
            pos_sec = self._player.position() / 1000.0
            self._scene.update_time(pos_sec, window)

    # ── Public API ─────────────────────────────────────────────────

    def set_cache_data(self, cache_data: dict):
        """Update detection data (e.g. after new analysis)."""
        self.cache_data = cache_data
        self._bbox_loader = LazyBBoxLoader(self.cache_data, self._max_cached_buckets)
        self._load_detections_lazy()

    def set_overlay_visible(self, visible: bool):
        """Programmatically toggle overlay."""
        self._overlay_cb.setChecked(visible)

    def set_live_face_enabled(self, enabled: bool):
        """Enter/leave TRUE real-time face-recognition mode.

        This only records the *mode* — the '🎯 Live BBox Overlay' checkbox is the
        actual on/off switch for inference, so the user can pause processing
        without leaving Live mode. Entering real-time mode makes the checkbox
        usable even with no cached bbox data and switches it on so recognition
        starts immediately.
        """
        self._live_face_mode = enabled

        if enabled:
            # Real-time needs no cached detections — make the checkbox usable even
            # with an empty cache, then switch it on so inference starts. The
            # filter button is also enabled so the live 'Facial recognition' group
            # is reachable even when there's no cached object/action data.
            self._overlay_cb.setEnabled(True)
            self._filter_btn.setEnabled(True)
            if self._overlay_cb.isChecked():
                self._apply_live_face_state()
            else:
                self._overlay_cb.setChecked(True)  # fires _on_overlay_toggled → _apply_live_face_state
        else:
            # Leaving real-time: stop inference and restore the checkbox's and
            # filter button's enabled-state (which depend on cached data).
            self._apply_live_face_state()
            self._overlay_cb.setEnabled(self._detection_count > 0)
            self._filter_btn.setEnabled(bool(self._filter_actions))

    def _apply_live_face_state(self):
        """Start or stop the real-time face worker from the current mode + checkbox.

        Inference runs only when Live (real-time) mode is selected AND the overlay
        checkbox is on. The worker (and InsightFace) is created lazily the first
        time both are true.
        """
        want = self._live_face_mode and self._overlay_enabled

        if want and self._live_face is None:
            # lazy init — only loads InsightFace the first time it's switched on
            if self._face_bank is None:
                self._face_bank = FaceIdentityBank(db_path="./cache/face_db.json")
            self._live_overlay = LiveFaceOverlay(self._scene)
            self._live_face = LiveFaceController(
                bank=self._face_bank,
                video_sink=self._scene.video_item.videoSink(),   # the frame tap
            )
            self._live_face.results_ready.connect(self._live_overlay.update_boxes)
            # Inherit current VR state (user may have enabled VR before live face)
            vr = getattr(self._view, "_vr_mode", False)
            self._live_face.set_vr_mode(vr)
            self._live_overlay.set_vr_mode(vr)

        if self._live_face is not None:
            self._live_face.set_enabled(want)
            if not want and self._live_overlay is not None:
                self._live_overlay.clear()

    def _on_identity_context(self, identity_id, global_pos):
        menu = QMenu(self)
        a_name  = menu.addAction("✏️  Name this person…")
        a_avoid = menu.addAction("🚫  Avoid this person")
        chosen = menu.exec(global_pos)

        if chosen == a_name and self._face_bank is not None:
            ident = self._face_bank.get_identity(identity_id)
            prefill = ident["name"] if (ident and ident["name"]) else ""
            name, ok = QInputDialog.getText(self, "Name person", "Name:", text=prefill)
            if ok and name.strip():
                name = name.strip()
                # if this name already belongs to someone, MERGE into them
                existing = next((i for i in self._face_bank.all_identities()
                                 if i["name"] and i["name"].lower() == name.lower()
                                 and i["id"] != identity_id), None)
                if existing:
                    self._face_bank.merge_identities(existing["id"], identity_id)
                else:
                    self._face_bank.name_identity(identity_id, name)
                self._face_bank.save()
        elif chosen == a_avoid:
            self.avoid_person_requested.emit(identity_id)

    def shutdown_live_face(self):
        """Stop the worker thread cleanly. Call from the window's closeEvent."""
        if self._live_face is not None:
            self._live_face.shutdown()

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
        """Number of bbox detections available in cache."""
        return self._detection_count

    def get_visible_classes(self) -> set[str]:
        """Get currently visible object/action classes."""
        return self._scene.get_visible_classes()
    
    def clear_cache(self):
        """Clear bbox cache to free memory."""
        self._bbox_loader.clear_cache()
        self._scene.clear_items()
        gc.collect()
    
    def closeEvent(self, event):
        """Clean up on close."""
        self.clear_cache()
        self._memory_timer.stop()
        self.shutdown_live_face()
        super().closeEvent(event)


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
    win.setWindowTitle("Real-Time BBox Overlay Test (Memory Optimized)")
    win.resize(1280, 800)

    preview = RealtimeOverlayPreview(
        video_path=video_path,
        cache_data=cache_data,
        max_cached_buckets=15,
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

    clear_btn = QPushButton("🧹 Clear Cache")
    clear_btn.clicked.connect(preview.clear_cache)
    btn_row.addWidget(clear_btn)

    layout.addLayout(btn_row)
    win.setCentralWidget(central)

    win.show()
    sys.exit(app.exec())