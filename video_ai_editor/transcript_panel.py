"""
Transcript Panel for Signal Timeline Viewer
- Scrollable full transcript with timestamps
- Search with keyword highlighting
- Click any segment to seek video
- Auto-scroll follows video playhead
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QScrollArea, QFrame, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QColor, QTextCursor, QFont


class TranscriptSegmentWidget(QFrame):
    """Single transcript segment — timestamp + text, clickable to seek"""
    clicked = Signal(float)  # emits start time

    def __init__(self, segment: dict, parent=None):
        super().__init__(parent)
        self.start_time = float(segment.get("start", 0))
        self.end_time = float(segment.get("end", self.start_time + 1))
        self.text = segment.get("text", "").strip()
        self._is_active = False
        self._is_match = False

        self.setObjectName("transcriptSegment")
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(36)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Timestamp label
        mins = int(self.start_time // 60)
        secs = int(self.start_time % 60)
        self.time_label = QLabel(f"{mins:02d}:{secs:02d}")
        self.time_label.setFixedWidth(38)
        self.time_label.setAlignment(Qt.AlignTop | Qt.AlignRight)
        self.time_label.setStyleSheet("""
            color: #5a7faa;
            font-family: 'Consolas', monospace;
            font-size: 10px;
            font-weight: bold;
            padding-top: 3px;
        """)
        layout.addWidget(self.time_label)

        # Text label
        self.text_label = QLabel(self.text)
        self.text_label.setWordWrap(True)
        self.text_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.text_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.text_label.setStyleSheet("""
            color: #c8d4e8;
            font-size: 12px;
            line-height: 1.5;
            padding: 2px 0;
        """)
        layout.addWidget(self.text_label)

        self._apply_style()

    def _apply_style(self):
        if self._is_active and self._is_match:
            bg = "#2a3f1a"
            border = "#7aff50"
        elif self._is_active:
            bg = "#1a2a3f"
            border = "#3a7fcd"
        elif self._is_match:
            bg = "#2a2a10"
            border = "#aaaa30"
        else:
            bg = "transparent"
            border = "transparent"

        self.setStyleSheet(f"""
            QFrame#transcriptSegment {{
                background-color: {bg};
                border-left: 3px solid {border};
                border-radius: 3px;
            }}
            QFrame#transcriptSegment:hover {{
                background-color: #1e2a3a;
                border-left: 3px solid #5a8fcd;
            }}
        """)

    def set_active(self, active: bool):
        if self._is_active == active:
            return
        self._is_active = active
        self._apply_style()

    def set_match(self, match: bool):
        if self._is_match == match:
            return
        self._is_match = match
        self._apply_style()

    def highlight_text(self, keyword: str):
        """Highlight keyword in text label"""
        if not keyword:
            self.text_label.setText(self.text)
            return
        
        # Simple case-insensitive highlight using HTML
        import re
        escaped_text = self.text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        highlighted = pattern.sub(
            lambda m: f'<span style="background-color:#5a5a10; color:#ffff60; '
                      f'border-radius:2px; padding:0 2px;">{m.group()}</span>',
            escaped_text
        )
        self.text_label.setText(highlighted)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.start_time)
        super().mousePressEvent(event)


class TranscriptPanel(QWidget):
    """
    Full transcript panel with search and playhead sync.
    
    Signals:
        seek_requested(float): emitted when user clicks a segment
    """
    seek_requested = Signal(float)

    def __init__(self, segments: list, parent=None):
        super().__init__(parent)
        self.segments = segments or []
        self.segment_widgets: list[TranscriptSegmentWidget] = []
        self.current_active_idx = -1
        self.match_indices: list[int] = []
        self.current_match_idx = 0
        self._auto_scroll = True
        self._last_keyword = ""

        self._build_ui()
        self._populate_segments()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Search row ──
        search_row = QHBoxLayout()
        search_row.setSpacing(4)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search transcript...")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.setStyleSheet("""
            QLineEdit {
                background-color: #1a1a2e;
                color: #e0e8ff;
                border: 1px solid #3a3a5a;
                border-radius: 4px;
                padding: 6px 10px;
                font-size: 12px;
            }
            QLineEdit:focus {
                border-color: #4a7fcd;
            }
        """)
        self.search_input.returnPressed.connect(self._next_match)
        self.search_input.textChanged.connect(self._on_search_changed)
        search_row.addWidget(self.search_input)

        self.prev_btn = QPushButton("▲")
        self.prev_btn.setFixedSize(28, 28)
        self.prev_btn.setToolTip("Previous match")
        self.prev_btn.clicked.connect(self._prev_match)
        self.prev_btn.setStyleSheet(self._nav_btn_style())
        search_row.addWidget(self.prev_btn)

        self.next_btn = QPushButton("▼")
        self.next_btn.setFixedSize(28, 28)
        self.next_btn.setToolTip("Next match")
        self.next_btn.clicked.connect(self._next_match)
        self.next_btn.setStyleSheet(self._nav_btn_style())
        search_row.addWidget(self.next_btn)

        layout.addLayout(search_row)

        # ── Match info + auto-scroll toggle ──
        meta_row = QHBoxLayout()

        self.match_label = QLabel("")
        self.match_label.setStyleSheet("color: #7a9aaa; font-size: 10px; padding: 0 4px;")
        meta_row.addWidget(self.match_label)

        meta_row.addStretch()

        self.auto_scroll_btn = QPushButton("⟳ Auto-scroll ON")
        self.auto_scroll_btn.setCheckable(True)
        self.auto_scroll_btn.setChecked(True)
        self.auto_scroll_btn.setFixedHeight(22)
        self.auto_scroll_btn.clicked.connect(self._toggle_auto_scroll)
        self.auto_scroll_btn.setStyleSheet("""
            QPushButton {
                background: #1a2a1a;
                color: #60aa60;
                border: 1px solid #3a5a3a;
                border-radius: 3px;
                font-size: 10px;
                padding: 0 6px;
            }
            QPushButton:checked {
                background: #1a2a1a;
                color: #60aa60;
            }
            QPushButton:!checked {
                background: #1a1a1a;
                color: #666;
                border-color: #333;
            }
        """)
        meta_row.addWidget(self.auto_scroll_btn)

        layout.addLayout(meta_row)

        # ── Scrollable segment list ──
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: #0e0e1a;
                border: 1px solid #2a2a3a;
                border-radius: 4px;
            }
            QScrollBar:vertical {
                background: #1a1a2a;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #3a3a5a;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #4a5aaa;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        self.content_widget = QWidget()
        self.content_widget.setStyleSheet("background-color: #0e0e1a;")
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(4, 4, 4, 4)
        self.content_layout.setSpacing(1)
        self.content_layout.addStretch()

        self.scroll_area.setWidget(self.content_widget)
        layout.addWidget(self.scroll_area)

        # ── No transcript message ──
        if not self.segments:
            empty = QLabel("No transcript available.\nRun the pipeline with transcript enabled.")
            empty.setAlignment(Qt.AlignCenter)
            empty.setStyleSheet("color: #4a5a6a; font-style: italic; padding: 20px;")
            self.content_layout.insertWidget(0, empty)

    def _nav_btn_style(self):
        return """
            QPushButton {
                background-color: #1a1a2e;
                color: #7a9acd;
                border: 1px solid #3a3a5a;
                border-radius: 4px;
                font-size: 10px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2a2a4e;
                color: #aac0ff;
            }
            QPushButton:disabled {
                color: #333;
                border-color: #222;
            }
        """

    def _populate_segments(self):
        """Build all segment widgets"""
        # Remove stretch first
        stretch = self.content_layout.takeAt(self.content_layout.count() - 1)

        for seg in self.segments:
            w = TranscriptSegmentWidget(seg)
            w.clicked.connect(self._on_segment_clicked)
            self.segment_widgets.append(w)
            self.content_layout.addWidget(w)

        # Re-add stretch at end
        self.content_layout.addStretch()

        self.match_label.setText(f"{len(self.segments)} segments")

    def _on_segment_clicked(self, time: float):
        """User clicked a segment — seek video and disable auto-scroll temporarily"""
        self.seek_requested.emit(time)

    def _toggle_auto_scroll(self, checked: bool):
        self._auto_scroll = checked
        self.auto_scroll_btn.setText("⟳ Auto-scroll ON" if checked else "⟳ Auto-scroll OFF")

    # ── Search ────────────────────────────────────────────────────

    def _on_search_changed(self, text: str):
        """Re-run search on every keystroke"""
        self._last_keyword = text.strip()
        self._run_search()

    def _run_search(self):
        keyword = self._last_keyword
        self.match_indices = []

        for i, w in enumerate(self.segment_widgets):
            if keyword and keyword.lower() in w.text.lower():
                w.set_match(True)
                w.highlight_text(keyword)
                self.match_indices.append(i)
            else:
                w.set_match(False)
                w.highlight_text("")

        if self.match_indices:
            self.current_match_idx = 0
            count = len(self.match_indices)
            self.match_label.setText(f"{count} match{'es' if count != 1 else ''}")
            self.match_label.setStyleSheet("color: #aaff60; font-size: 10px; padding: 0 4px;")
            self.prev_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            # Scroll to first match
            self._scroll_to_match(0)
        elif keyword:
            self.match_label.setText("No matches")
            self.match_label.setStyleSheet("color: #ff6060; font-size: 10px; padding: 0 4px;")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
        else:
            self.match_label.setText(f"{len(self.segments)} segments")
            self.match_label.setStyleSheet("color: #7a9aaa; font-size: 10px; padding: 0 4px;")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)

    def _next_match(self):
        if not self.match_indices:
            return
        self.current_match_idx = (self.current_match_idx + 1) % len(self.match_indices)
        self._scroll_to_match(self.current_match_idx)
        self._update_match_counter()

    def _prev_match(self):
        if not self.match_indices:
            return
        self.current_match_idx = (self.current_match_idx - 1) % len(self.match_indices)
        self._scroll_to_match(self.current_match_idx)
        self._update_match_counter()

    def _update_match_counter(self):
        if self.match_indices:
            count = len(self.match_indices)
            current = self.current_match_idx + 1
            self.match_label.setText(f"{current}/{count} matches")

    def _scroll_to_match(self, match_idx: int):
        """Scroll to a specific match and seek video to it"""
        if not self.match_indices or match_idx >= len(self.match_indices):
            return
        seg_idx = self.match_indices[match_idx]
        widget = self.segment_widgets[seg_idx]
        self._scroll_to_widget(widget)
        # Also seek the video to this match
        self.seek_requested.emit(widget.start_time)

    def _scroll_to_widget(self, widget: TranscriptSegmentWidget):
        """Smooth scroll to bring widget into view"""
        self.scroll_area.ensureWidgetVisible(widget, 50, 80)

    # ── Playhead sync ─────────────────────────────────────────────

    def update_current_time(self, time_seconds: float):
        """Called by timeline window as video plays — highlights current segment"""
        if not self.segment_widgets:
            return

        # Find active segment
        new_idx = -1
        for i, w in enumerate(self.segment_widgets):
            if w.start_time <= time_seconds < w.end_time:
                new_idx = i
                break
            # Handle gap between segments — snap to nearest preceding
            if w.start_time > time_seconds:
                new_idx = max(0, i - 1)
                break

        if new_idx == self.current_active_idx:
            return

        # Deactivate old
        if 0 <= self.current_active_idx < len(self.segment_widgets):
            self.segment_widgets[self.current_active_idx].set_active(False)

        # Activate new
        self.current_active_idx = new_idx
        if 0 <= new_idx < len(self.segment_widgets):
            self.segment_widgets[new_idx].set_active(True)

            # Auto-scroll only if enabled and no active search
            if self._auto_scroll and not self._last_keyword:
                self._scroll_to_widget(self.segment_widgets[new_idx])

    def reload_segments(self, segments: list):
        """Replace all segments (e.g. after cache reload)"""
        # Clear existing
        for w in self.segment_widgets:
            self.content_layout.removeWidget(w)
            w.deleteLater()
        self.segment_widgets.clear()
        self.match_indices.clear()
        self.current_active_idx = -1
        self.segments = segments or []
        self._populate_segments()