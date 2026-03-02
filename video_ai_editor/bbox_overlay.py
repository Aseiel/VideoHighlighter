"""
Annotated Video Switcher for Video Timeline — SELF-CONTAINED

Everything is handled internally:
  - Generate buttons call your pipelines directly (threaded)
  - Dropdown swaps the player source reliably
  - No signals to connect, no slots to paste

Usage (inside create_video_preview_dock, after self.video_player exists):

    from video_ai_editor.bbox_overlay import AnnotatedVideoManager
    self.bbox_manager = AnnotatedVideoManager(
        video_path=self.video_path,
        cache_data=self.cache_data,
        player=self.video_player,
    )
    layout.addWidget(self.bbox_manager.create_toggle_widget())

That's it. Nothing else needed.
"""

from __future__ import annotations

import os
import glob
import threading
import traceback
from typing import Optional

from PySide6.QtCore import Qt, Signal, QObject, QTimer, QUrl
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel,
    QPushButton, QGroupBox, QComboBox, QMessageBox,
)


# ---------------------------------------------------------------------------
# Find annotated videos next to the original
# ---------------------------------------------------------------------------

def find_annotated_videos(video_path: str) -> dict[str, str]:
    if not video_path or not os.path.isfile(video_path):
        return {}

    base_dir = os.path.dirname(os.path.abspath(video_path))
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    original_abs = os.path.abspath(video_path)

    found = {}
    patterns = [
        (f"{base_name}*action*annotated*", "🎬 Actions"),
        (f"{base_name}*object*annotated*", "📦 Objects"),
        (f"{base_name}_annotated*", "🎯 Annotated"),
        (f"{base_name}_bbox*", "🎯 BBox"),
        (f"{base_name}_overlay*", "🎯 Overlay"),
    ]

    for pattern, label in patterns:
        matches = glob.glob(os.path.join(base_dir, pattern))
        for match in matches:
            match_abs = os.path.abspath(match)
            if os.path.getsize(match_abs) < 10240:
                continue
            if match_abs == original_abs:
                continue
            if not match.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.webm')):
                continue

            fname = os.path.basename(match).lower()
            if 'action' in fname:
                display = "🎬 Actions"
            elif 'object' in fname:
                display = "📦 Objects"
            else:
                display = label

            if display in found:
                if os.path.getsize(match_abs) > os.path.getsize(found[display]):
                    found[display] = match_abs
            else:
                found[display] = match_abs

    return found


# ---------------------------------------------------------------------------
# AnnotatedVideoManager — SELF-CONTAINED
# ---------------------------------------------------------------------------

class AnnotatedVideoManager(QObject):
    """
    Manages switching between original and annotated videos.

    - Generate buttons call run_action_detection / run_object_detection directly
    - Dropdown swaps QMediaPlayer source
    - No external slots needed
    """

    source_changed = Signal(str)

    def __init__(self, video_path: str, cache_data: dict = None,
                 player=None, parent: QObject | None = None):
        super().__init__(parent)
        self.video_path = os.path.abspath(video_path)
        self.cache_data = cache_data or {}
        self._player = player
        self._current_source = "🎥 Original"
        self._generating_actions = False
        self._generating_objects = False

        # Build source list
        self._sources: dict[str, str] = {"🎥 Original": self.video_path}
        self._sources.update(find_annotated_videos(self.video_path))

        if len(self._sources) > 1:
            names = [k for k in self._sources if k != "🎥 Original"]
            print(f"✅ Found annotated videos: {', '.join(names)}")
        else:
            print("ℹ️ No annotated videos found yet — use Generate buttons")

        # UI (created lazily)
        self._widget: Optional[QWidget] = None
        self._combo: Optional[QComboBox] = None
        self._status: Optional[QLabel] = None
        self._gen_actions_btn: Optional[QPushButton] = None
        self._gen_objects_btn: Optional[QPushButton] = None

    # ---- public API -------------------------------------------------------

    def set_player(self, player):
        self._player = player

    def refresh(self):
        """Re-scan for annotated videos."""
        old_keys = set(self._sources.keys())
        self._sources = {"🎥 Original": self.video_path}
        self._sources.update(find_annotated_videos(self.video_path))

        if set(self._sources.keys()) != old_keys and self._combo:
            current_text = self._combo.currentText()
            self._combo.blockSignals(True)
            self._combo.clear()
            self._combo.addItems(list(self._sources.keys()))
            idx = self._combo.findText(current_text)
            self._combo.setCurrentIndex(idx if idx >= 0 else 0)
            self._combo.blockSignals(False)

        self._update_status()
        self._update_button_labels()

        new_keys = set(self._sources.keys()) - old_keys
        if new_keys:
            print(f"🔄 Refresh found new: {', '.join(new_keys)}")

    def set_generating(self, generating: bool, which: str = "actions"):
        if which == "actions":
            self._generating_actions = generating
            if self._gen_actions_btn:
                self._gen_actions_btn.setEnabled(not generating)
                self._gen_actions_btn.setText(
                    "⏳ Generating Actions…" if generating else "🎬 Generate Actions")
        elif which == "objects":
            self._generating_objects = generating
            if self._gen_objects_btn:
                self._gen_objects_btn.setEnabled(not generating)
                self._gen_objects_btn.setText(
                    "⏳ Generating Objects…" if generating else "📦 Generate Objects")

    # ---- widget factory ---------------------------------------------------

    def create_toggle_widget(self) -> QWidget:
        if self._widget is not None:
            return self._widget

        self._widget = grp = QGroupBox("🎯 Video Source")
        grp.setStyleSheet("""
            QGroupBox {
                font-weight: bold; color: #ccc;
                border: 1px solid #3a3a5a; border-radius: 4px;
                margin-top: 6px; padding-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px; padding: 0 4px;
            }
        """)
        layout = QVBoxLayout(grp)
        layout.setContentsMargins(6, 16, 6, 6)
        layout.setSpacing(4)

        # Row 1: dropdown + refresh
        row1 = QHBoxLayout()
        self._combo = QComboBox()
        self._combo.addItems(list(self._sources.keys()))
        self._combo.setCurrentText(self._current_source)
        self._combo.currentTextChanged.connect(self._on_combo_changed)
        self._combo.setStyleSheet("""
            QComboBox {
                background-color: #1a1a2a; color: #ddd;
                border: 1px solid #3a3a5a; border-radius: 4px;
                padding: 6px 10px; min-width: 160px;
            }
            QComboBox:hover { border-color: #5a5a8a; }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background-color: #1a1a2a; color: #ddd;
                selection-background-color: #3a5fcd;
            }
        """)
        row1.addWidget(self._combo, 1)

        refresh_btn = QPushButton("🔄")
        refresh_btn.setToolTip("Re-scan for annotated videos")
        refresh_btn.setFixedWidth(36)
        refresh_btn.clicked.connect(self.refresh)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a2a4a; color: white;
                border: 1px solid #3a3a5a; border-radius: 4px;
                padding: 6px; font-size: 14px;
            }
            QPushButton:hover { background-color: #3a3a5a; }
        """)
        row1.addWidget(refresh_btn)
        layout.addLayout(row1)

        # Row 2: generate buttons
        row2 = QHBoxLayout()
        self._gen_actions_btn = QPushButton("🎬 Generate Actions")
        self._gen_actions_btn.setToolTip(
            "Run action detection with bounding boxes (uses your existing pipeline)")
        self._gen_actions_btn.clicked.connect(self._on_generate_actions)
        self._gen_actions_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a5f2a; color: white;
                font-weight: bold; padding: 6px 10px; border-radius: 4px;
            }
            QPushButton:hover { background-color: #3a7f3a; }
            QPushButton:disabled { background-color: #333; color: #777; }
        """)
        row2.addWidget(self._gen_actions_btn)

        self._gen_objects_btn = QPushButton("📦 Generate Objects")
        self._gen_objects_btn.setToolTip(
            "Run object detection with bounding boxes (uses your existing pipeline)")
        self._gen_objects_btn.clicked.connect(self._on_generate_objects)
        self._gen_objects_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a4f6f; color: white;
                font-weight: bold; padding: 6px 10px; border-radius: 4px;
            }
            QPushButton:hover { background-color: #3a6f9f; }
            QPushButton:disabled { background-color: #333; color: #777; }
        """)
        row2.addWidget(self._gen_objects_btn)
        layout.addLayout(row2)

        # Row 3: status
        self._status = QLabel()
        self._status.setStyleSheet("color: #888; font-size: 11px;")
        self._update_status()
        layout.addWidget(self._status)

        self._update_button_labels()
        return grp

    # ---- SOURCE SWAP (the core fix) ---------------------------------------

    def _on_combo_changed(self, text: str):
        if not text or text == self._current_source:
            return
        print(f"🔀 Dropdown changed to: {text}")
        self._switch_to(text)

    def _switch_to(self, label: str):
        if label not in self._sources:
            print(f"⚠️ Unknown source: {label}")
            return

        path = self._sources[label]
        if not os.path.isfile(path):
            print(f"⚠️ File missing: {path}")
            return

        if not self._player:
            print("⚠️ No player bound!")
            return

        print(f"🔀 Switching to: {label}")
        print(f"   File: {os.path.basename(path)} ({os.path.getsize(path) / 1024 / 1024:.1f} MB)")

        # Capture current state
        pos_ms = self._player.position()
        was_playing = False
        try:
            from PySide6.QtMultimedia import QMediaPlayer as QMP
            was_playing = (self._player.playbackState() == QMP.PlayingState)
        except Exception:
            pass

        print(f"   Position: {pos_ms}ms, was_playing: {was_playing}")

        # CRITICAL: Stop first, then clear source, then set new source
        self._player.stop()
        self._player.setSource(QUrl())  # Clear source first

        new_url = QUrl.fromLocalFile(path)
        print(f"   Setting source: {new_url.toLocalFile()}")

        # Use a short delay after clearing to let Qt release the old source
        def _set_new_source():
            self._player.setSource(new_url)
            self._current_source = label
            self.source_changed.emit(label)
            self._update_status()

            # Restore position after new source loads
            def _restore_position():
                if self._player:
                    duration = self._player.duration()
                    if duration > 0 and pos_ms < duration:
                        self._player.setPosition(pos_ms)
                        print(f"   ✅ Position restored to {pos_ms}ms")
                    if was_playing:
                        self._player.play()
                    print(f"   ✅ Source swap complete: {label}")

            QTimer.singleShot(500, _restore_position)

        QTimer.singleShot(100, _set_new_source)

    # ---- GENERATE: Actions ------------------------------------------------

    def _on_generate_actions(self):
        if self._generating_actions:
            return

        print("🎬 Generate Actions button clicked")
        self.set_generating(True, "actions")
        self._set_status("🎬 Running action detection with bounding boxes…")

        def _run():
            try:
                from action_recognition import run_action_detection

                base, ext = os.path.splitext(self.video_path)
                output = f"{base}_actions_annotated{ext}"

                # Pull interesting_actions from cache if available
                actions_list = self.cache_data.get('interesting_actions', None)

                print(f"🎬 Starting action detection → {os.path.basename(output)}")
                if actions_list:
                    print(f"   Tracking actions: {actions_list}")

                run_action_detection(
                    video_path=self.video_path,
                    device="AUTO",
                    sample_rate=5,
                    log_file=f"{base}_actions_bbox.csv",
                    debug=False,
                    top_k=10,
                    confidence_threshold=0.01,
                    draw_bboxes=True,
                    annotated_output=output,
                    use_person_detection=True,
                    max_people=2,
                    interesting_actions=actions_list,
                    include_model_type=True,
                    enable_r3d=True,
                    progress_callback=self._action_progress_callback,
                )

                print(f"✅ Action bbox video saved: {output}")
                QTimer.singleShot(0, lambda: self._on_generate_done(True, output, "actions"))

            except Exception as e:
                traceback.print_exc()
                QTimer.singleShot(0, lambda: self._on_generate_done(False, str(e), "actions"))

        threading.Thread(target=_run, daemon=True).start()

    def _action_progress_callback(self, current, total, stage, message):
        QTimer.singleShot(0, lambda: self._set_status(f"🎬 {message}"))

    # ---- GENERATE: Objects ------------------------------------------------

    def _on_generate_objects(self):
        if self._generating_objects:
            return

        print("📦 Generate Objects button clicked")
        self.set_generating(True, "objects")
        self._set_status("📦 Running object detection with bounding boxes…")

        def _run():
            try:
                from object_recognition import run_object_detection

                base, ext = os.path.splitext(self.video_path)
                output = f"{base}_objects_annotated{ext}"

                # Pull object list from cache, fallback to common COCO classes
                highlight_objects = self.cache_data.get('highlight_objects', None)
                if not highlight_objects:
                    highlight_objects = [
                        "person", "bicycle", "car", "motorcycle", "bus", "truck",
                        "dog", "cat", "horse", "bird",
                        "backpack", "umbrella", "handbag", "suitcase",
                        "bottle", "cup", "fork", "knife", "spoon",
                        "chair", "couch", "bed", "dining table",
                        "tv", "laptop", "cell phone", "book",
                        "sports ball", "tennis racket", "baseball bat",
                        "skateboard", "surfboard", "frisbee",
                    ]

                print(f"📦 Starting object detection → {os.path.basename(output)}")
                print(f"   Looking for: {highlight_objects[:5]}...")

                run_object_detection(
                    video_path=self.video_path,
                    highlight_objects=highlight_objects,
                    frame_skip=5,
                    csv_file=f"{base}_objects_bbox.csv",
                    draw_boxes=True,
                    annotated_output=output,
                    progress_fn=None,
                )

                print(f"✅ Object bbox video saved: {output}")
                QTimer.singleShot(0, lambda: self._on_generate_done(True, output, "objects"))

            except Exception as e:
                traceback.print_exc()
                QTimer.singleShot(0, lambda: self._on_generate_done(False, str(e), "objects"))

        threading.Thread(target=_run, daemon=True).start()

    def _object_progress_callback(self, progress, message):
        pct = int(progress * 100)
        QTimer.singleShot(0, lambda: self._set_status(f"📦 {pct}% — {message}"))

    # ---- Generate completion (shared) -------------------------------------

    def _on_generate_done(self, success: bool, result: str, which: str):
        """Called on main thread when generation finishes."""
        self.set_generating(False, which)

        if success:
            self.refresh()  # Re-scan → new video appears in dropdown
            size_mb = os.path.getsize(result) / (1024 * 1024) if os.path.isfile(result) else 0
            self._set_status(f"✅ {which.title()} video ready ({size_mb:.1f} MB)")
            print(f"✅ {which.title()} bbox video ready: {result}")

            # Show a message box
            if self._widget:
                QMessageBox.information(
                    self._widget, "Generation Complete",
                    f"{which.title()} annotated video is ready!\n"
                    f"Select it from the dropdown to view.\n\n"
                    f"File: {os.path.basename(result)} ({size_mb:.1f} MB)"
                )
        else:
            self._set_status(f"❌ {which.title()} generation failed: {result[:60]}")
            print(f"❌ {which.title()} generation failed: {result}")

    # ---- helpers ----------------------------------------------------------

    def _set_status(self, text: str):
        if self._status:
            self._status.setText(text)

    def _update_status(self):
        n = len(self._sources) - 1
        if n == 0:
            self._set_status("No annotated videos found — click Generate")
        else:
            path = self._sources.get(self._current_source, "")
            if path and os.path.isfile(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                self._set_status(
                    f"{n} source{'s' if n > 1 else ''} · "
                    f"Current: {os.path.basename(path)} ({size_mb:.1f} MB)")
            else:
                self._set_status(f"{n} source{'s' if n > 1 else ''} available")

    def _update_button_labels(self):
        has_actions = "🎬 Actions" in self._sources
        has_objects = "📦 Objects" in self._sources
        if self._gen_actions_btn and not self._generating_actions:
            self._gen_actions_btn.setText(
                "🎬 Regenerate Actions" if has_actions else "🎬 Generate Actions")
        if self._gen_objects_btn and not self._generating_objects:
            self._gen_objects_btn.setText(
                "📦 Regenerate Objects" if has_objects else "📦 Generate Objects")