"""
Face / Identity Search Panel

Shows all detected identities (from object_bboxes cache).
Click a face card → find every segment where that person appears.
Results: scrollable list with Jump and Add-to-Edit-Timeline buttons.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Callable

from PySide6.QtCore import Qt, Signal, QByteArray, QThread
from PySide6.QtGui import QPixmap, QColor, QPainter, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QFrame, QSizePolicy, QSplitter, QComboBox, QMenu,
)

from modules.face_emotions import EMOTION_LABELS

THUMB_SIZE = 56    # face card thumbnail px
MERGE_GAP  = 2.0  # seconds — gaps smaller than this are merged
MIN_DUR    = 0.5  # seconds — segments shorter than this are dropped


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _placeholder_pixmap(size: int) -> QPixmap:
    pix = QPixmap(size, size)
    pix.fill(QColor(60, 60, 60))
    p = QPainter(pix)
    p.setPen(QColor(145, 145, 145))
    p.setFont(QFont("Arial", size // 3))
    p.drawText(pix.rect(), Qt.AlignCenter, "?")
    p.end()
    return pix


def _pixmap_from_b64(b64: str, size: int) -> QPixmap:
    try:
        data = base64.b64decode(b64)
        pix = QPixmap()
        pix.loadFromData(QByteArray(data))
        if not pix.isNull():
            return pix.scaled(size, size, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
    except Exception:
        pass
    return _placeholder_pixmap(size)


def load_identity_entries(video_path: str) -> list[dict]:
    """Load the identity-tagged per-frame entries for a video, straight from the
    Avoid pass's cache (cache/avoid/<key>.entries.json). Load-only — never runs
    tracking/recognition. Returns [] if the video was never face-scanned.

    Resolved cwd-INDEPENDENTLY (repo root via this module's location, then a
    cwd-relative ./cache fallback) so it works no matter where the app launched.
    Matched by exact key only — the key is the sole reliable video↔file link.
    """
    if not video_path:
        return []
    try:
        from modules.compute_forbidden import _entries_key
        key = _entries_key(video_path, "n", 15, 3)
        fname = key + ".entries.json"
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for p in (os.path.join(repo_root, "cache", "avoid", fname),
                  os.path.join("cache", "avoid", fname)):
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as fh:
                    entries = json.load(fh)
                print(f"🪪 SearchPanel: loaded {len(entries)} identity-tagged "
                      f"frame(s) from {p}")
                return entries or []
        print(f"🪪 SearchPanel: no identity entries for "
              f"'{os.path.basename(video_path)}' (key={key})")
    except Exception as e:
        print(f"⚠️ SearchPanel.load_identity_entries failed: {e}")
    return []


def build_segments(
    object_bboxes: list[dict],
    identity_id: str,
    video_duration: float,
    merge_gap: float = MERGE_GAP,
    min_dur: float = MIN_DUR,
) -> list[tuple[float, float]]:
    """Return merged (start, end) segments where identity_id appears."""
    entries = sorted(object_bboxes, key=lambda e: e.get("timestamp", 0))

    hits: list[float] = []
    for entry in entries:
        ids = entry.get("identity_ids") or []
        if identity_id in ids:
            hits.append(entry.get("timestamp", 0.0))

    if not hits:
        return []

    merged: list[list[float]] = [[hits[0], hits[0]]]
    for t in hits[1:]:
        if t - merged[-1][1] <= merge_gap:
            merged[-1][1] = t
        else:
            merged.append([t, t])

    result = []
    for s, e in merged:
        seg_start = max(0.0, s - 0.5)
        seg_end   = min(video_duration, e + 1.5)
        if seg_end - seg_start >= min_dur:
            result.append((seg_start, seg_end))
    return result


# ── Face card ─────────────────────────────────────────────────────────────────

class _FaceCard(QFrame):
    clicked = Signal(str)              # identity_id
    avoid_toggled = Signal(str, bool)  # identity_id, should be avoided

    def __init__(self, identity_id: str, name: str, thumb_b64: str | None,
                 avoided: bool = False, parent=None):
        super().__init__(parent)
        self.identity_id = identity_id
        self.avoided = bool(avoided)
        self.setFixedHeight(THUMB_SIZE + 28)
        self.setMinimumWidth(70)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self._apply_style()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        thumb_lbl = QLabel()
        thumb_lbl.setFixedSize(THUMB_SIZE, THUMB_SIZE)
        thumb_lbl.setAlignment(Qt.AlignCenter)
        pix = _pixmap_from_b64(thumb_b64, THUMB_SIZE) if thumb_b64 else _placeholder_pixmap(THUMB_SIZE)
        thumb_lbl.setPixmap(pix)
        layout.addWidget(thumb_lbl, alignment=Qt.AlignHCenter)

        self._name_lbl = QLabel(name or "Unknown")
        self._name_lbl.setAlignment(Qt.AlignCenter)
        self._name_lbl.setFixedWidth(THUMB_SIZE + 8)
        layout.addWidget(self._name_lbl, alignment=Qt.AlignHCenter)
        self._apply_name_style()

    def _apply_style(self):
        """Avoided people are marked on the card itself.

        The setting lives in the face bank and changes what the next run cuts,
        so it has to be visible where the faces are — not only in a tab
        somewhere else."""
        border = "#c04a4a" if self.avoided else "#3a3a3a"
        self.setStyleSheet(
            "QFrame { background: #141414; border: 1px solid %s;"
            " border-radius: 6px; }"
            "QFrame:hover { border-color: #4a90f5; background: #1f1f1f; }"
            % border
        )

    def _apply_name_style(self):
        colour = "#c88080" if self.avoided else "#cccccc"
        self._name_lbl.setStyleSheet(
            "color: %s; font-size: 10px; border: none; background: transparent;"
            % colour
        )

    def set_avoided(self, avoided: bool):
        self.avoided = bool(avoided)
        self._apply_style()
        self._apply_name_style()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.identity_id)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        action = menu.addAction("Stop avoiding in highlights" if self.avoided
                                else "Avoid this person in highlights")
        if menu.exec(event.globalPos()) is action:
            self.avoid_toggled.emit(self.identity_id, not self.avoided)


# ── Segment row ───────────────────────────────────────────────────────────────

class _SegmentRow(QWidget):
    def __init__(self, idx: int, start: float, end: float,
                 on_jump: Callable, on_add: Callable, parent=None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(2, 1, 2, 1)
        row.setSpacing(4)

        lbl = QLabel(f"#{idx+1}  {_fmt(start)} → {_fmt(end)}  ({end-start:.1f}s)")
        lbl.setStyleSheet("color: #b4b4b4; font-size: 11px;")
        row.addWidget(lbl, 1)

        jump_btn = QPushButton("▶ Jump")
        jump_btn.setFixedWidth(62)
        jump_btn.setStyleSheet(
            "QPushButton{background:#323232;color:#cccccc;border:none;border-radius:3px;padding:3px;font-size:10px;}"
            "QPushButton:hover{background:#424242;}"
        )
        jump_btn.clicked.connect(lambda _=False, s=start: on_jump(s))
        row.addWidget(jump_btn)

        add_btn = QPushButton("+ Edit")
        add_btn.setFixedWidth(52)
        add_btn.setStyleSheet(
            "QPushButton{background:#1a4a2a;color:#88ffaa;border:none;border-radius:3px;padding:3px;font-size:10px;}"
            "QPushButton:hover{background:#2a6a3a;}"
        )
        add_btn.clicked.connect(lambda _=False, s=start, e=end: on_add(s, e))
        row.addWidget(add_btn)


# ── Main panel ────────────────────────────────────────────────────────────────

class _ExpressionScanWorker(QThread):
    """Runs the face sweep off the UI thread.

    Minutes of detection on a feature-length file, so it cannot run inline —
    a frozen window during a scan reads as a crash.
    """
    # Signal(object), not Signal(dict): a dict signal marshals through
    # QVariantMap, which is keyed by string, so a scan keyed by second arrives
    # empty on the other side — the scan succeeds and the panel reports nothing.
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(float, float)

    def __init__(self, video_path: str, cache_dir: str = "./cache",
                 rescan: bool = False, parent=None):
        super().__init__(parent)
        self._video_path = video_path
        self._cache_dir = cache_dir
        self._rescan = rescan
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        try:
            from modules.face_scan import scan_video
            seconds = scan_video(
                self._video_path,
                cache_dir=self._cache_dir,
                use_cache=not self._rescan,
                cancel_fn=lambda: self._cancelled,
                progress_fn=lambda at, total: self.progress.emit(at, total),
            )
            self.done.emit(seconds or {})
        except Exception as exc:                      # noqa: BLE001
            self.failed.emit(str(exc))


class SearchPanel(QWidget):
    """
    Face identity search panel for the timeline viewer.

    Parameters
    ----------
    cache_data      : dict with 'object_bboxes' key (identity-tagged frames)
    video_duration  : total video length in seconds
    on_jump         : callback(time_seconds) — seek the player
    on_add_clip     : callback(start, end)   — add clip to edit timeline
    face_db_path    : path to face_db.json for thumbnails (optional)
    """

    def __init__(
        self,
        cache_data: dict,
        video_duration: float,
        on_jump: Callable[[float], None],
        on_add_clip: Callable[[float, float], None],
        face_db_path: str | None = None,
        video_path: str | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self._cache_data     = cache_data
        self._video_duration = video_duration
        self._jump           = on_jump
        self._add_clip       = on_add_clip
        self._face_db_path   = face_db_path
        self._video_path     = video_path

        self._identities: dict[str, dict] = {}
        self._cards: dict[str, _FaceCard] = {}
        self._current_segments: list[tuple[float, float]] = []

        self._build_ui()
        self._load_identities()

    def _entries(self) -> list[dict]:
        """The identity-tagged frames to search over. Prefer whatever the cache
        already holds; otherwise load them directly from disk by video path (so
        the panel works even if the viewer never populated cache_data)."""
        entries = (self._cache_data.get("identity_entries")
                   or self._cache_data.get("object_bboxes"))
        if entries:
            return entries
        entries = load_identity_entries(self._video_path)
        if entries:
            # Cache it so the People timeline layer and re-queries reuse it.
            self._cache_data["identity_entries"] = entries
        return entries

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        hdr = QLabel("🔍 Search by Person")
        hdr.setStyleSheet("color: #909090; font-size: 12px; font-weight: bold;")
        root.addWidget(hdr)

        hint = QLabel("Click a face to find all segments where they appear.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #686868; font-size: 10px;")
        root.addWidget(hint)

        # ── Expressions ───────────────────────────────────────────────────────
        # Its own question, not a highlight setting: "where is this expression"
        # is asked and answered here, and the results feed the same jump / add
        # controls as a person search.
        expr_row = QHBoxLayout()
        expr_row.setSpacing(4)

        self._expr_scan_btn = QPushButton("Scan expressions")
        self._expr_scan_btn.setToolTip(
            "Detect faces across the video and classify their expression.\n"
            "Runs once and is cached, so asking a second question is instant.")
        self._expr_scan_btn.clicked.connect(self._scan_expressions)
        expr_row.addWidget(self._expr_scan_btn)

        self._expr_combo = QComboBox()
        self._expr_combo.addItem("— expression —", "")
        for _label in EMOTION_LABELS:
            self._expr_combo.addItem(_label, _label)
        self._expr_combo.setEnabled(False)
        self._expr_combo.currentIndexChanged.connect(self._on_expression_picked)
        expr_row.addWidget(self._expr_combo, 1)
        root.addLayout(expr_row)

        self._expr_status = QLabel("")
        self._expr_status.setWordWrap(True)
        self._expr_status.setStyleSheet("color: #686868; font-size: 10px;")
        root.addWidget(self._expr_status)
        self._expr_seconds: dict = {}
        self._expr_worker = None
        self._load_cached_expressions()

        splitter = QSplitter(Qt.Vertical)
        splitter.setChildrenCollapsible(False)

        # ── Face grid (top) ───────────────────────────────────────────────────
        face_scroll = QScrollArea()
        face_scroll.setWidgetResizable(True)
        face_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        face_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        face_scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        face_scroll.setFixedHeight(THUMB_SIZE + 60)

        self._face_grid_widget = QWidget()
        self._face_grid_layout = QHBoxLayout(self._face_grid_widget)
        self._face_grid_layout.setContentsMargins(2, 2, 2, 2)
        self._face_grid_layout.setSpacing(6)
        self._face_grid_layout.addStretch()

        self._no_faces_lbl = QLabel("No tagged identities found.\nRun identity tagging first.")
        self._no_faces_lbl.setWordWrap(True)
        self._no_faces_lbl.setAlignment(Qt.AlignCenter)
        self._no_faces_lbl.setStyleSheet("color: #606060; font-size: 11px;")
        self._face_grid_layout.insertWidget(0, self._no_faces_lbl)

        face_scroll.setWidget(self._face_grid_widget)
        splitter.addWidget(face_scroll)

        # ── Results (bottom) ──────────────────────────────────────────────────
        results_widget = QWidget()
        res_layout = QVBoxLayout(results_widget)
        res_layout.setContentsMargins(0, 0, 0, 0)
        res_layout.setSpacing(4)

        top_row = QHBoxLayout()
        self._results_header = QLabel("Select a person above")
        self._results_header.setStyleSheet("color: #888888; font-size: 11px; font-weight: bold;")
        top_row.addWidget(self._results_header, 1)

        self._add_all_btn = QPushButton("+ Add All to Edit Timeline")
        self._add_all_btn.setEnabled(False)
        self._add_all_btn.setStyleSheet(
            "QPushButton{background:#1a3a2a;color:#66ee88;border:none;border-radius:4px;padding:4px 8px;font-size:10px;}"
            "QPushButton:hover{background:#2a5a3a;}"
            "QPushButton:disabled{color:#446644;background:#111a14;}"
        )
        self._add_all_btn.clicked.connect(self._add_all)
        top_row.addWidget(self._add_all_btn)
        res_layout.addLayout(top_row)

        seg_scroll = QScrollArea()
        seg_scroll.setWidgetResizable(True)
        seg_scroll.setStyleSheet("QScrollArea{border:1px solid #2a2a2a;background:#0d0d0d;}")

        self._seg_list_widget = QWidget()
        self._seg_list_layout = QVBoxLayout(self._seg_list_widget)
        self._seg_list_layout.setContentsMargins(4, 4, 4, 4)
        self._seg_list_layout.setSpacing(2)
        self._seg_list_layout.addStretch()

        self._no_results_lbl = QLabel("No segments to show.")
        self._no_results_lbl.setAlignment(Qt.AlignCenter)
        self._no_results_lbl.setStyleSheet("color: #505050; font-size: 11px;")
        self._seg_list_layout.insertWidget(0, self._no_results_lbl)

        seg_scroll.setWidget(self._seg_list_widget)
        res_layout.addWidget(seg_scroll, 1)
        splitter.addWidget(results_widget)

        splitter.setSizes([THUMB_SIZE + 70, 300])
        root.addWidget(splitter, 1)

    # ── Data ──────────────────────────────────────────────────────────────────

    def _load_identities(self):
        object_bboxes = self._entries()

        seen: dict[str, dict] = {}
        for entry in object_bboxes:
            ids   = entry.get("identity_ids")   or []
            names = entry.get("identity_names") or []
            for iid, name in zip(ids, names):
                if not iid:
                    continue
                if iid not in seen:
                    seen[iid] = {"name": name or iid[:8], "thumb_b64": None, "count": 0}
                seen[iid]["count"] += 1
                if name and not seen[iid]["name"]:
                    seen[iid]["name"] = name

        if not seen:
            return  # leave "no faces" label visible

        # Enrich with thumbnails from face_db
        if self._face_db_path and os.path.exists(self._face_db_path):
            try:
                with open(self._face_db_path, "r", encoding="utf-8") as f:
                    db = json.load(f)
                for rec in db.get("identities", []):
                    iid = rec.get("id")
                    if iid in seen:
                        seen[iid]["thumb_b64"] = rec.get("thumb")
                        seen[iid]["avoided"] = bool(rec.get("avoid"))
                        if rec.get("name"):
                            seen[iid]["name"] = rec["name"]
            except Exception as e:
                print(f"⚠️ SearchPanel: face_db load error: {e}")

        self._identities = seen
        self._no_faces_lbl.hide()

        # Populate face cards, sorted by appearance count
        for iid, info in sorted(seen.items(), key=lambda kv: -kv[1]["count"]):
            card = _FaceCard(iid, info["name"], info["thumb_b64"],
                             avoided=info.get("avoided", False))
            card.clicked.connect(self._on_face_selected)
            card.avoid_toggled.connect(self._on_avoid_toggled)
            self._cards[iid] = card
            # Insert before trailing stretch
            self._face_grid_layout.insertWidget(self._face_grid_layout.count() - 1, card)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_avoid_toggled(self, identity_id: str, avoided: bool):
        """Mark a person as one the highlighter should skip.

        The face bank is the same one the pipeline reads, so this changes what
        the next run cuts. It is written immediately: a preference that only
        survives if the app closes cleanly is one nobody can rely on.
        """
        try:
            from video_ai_editor.face_identity import FaceIdentityBank

            bank = FaceIdentityBank(db_path=self._face_db_path
                                    or "./cache/face_db.json")
            bank.load()
            bank.set_avoid(identity_id, avoided)
            if not bank.save():
                raise IOError("face bank could not be written")
        except Exception as exc:                      # noqa: BLE001
            print(f"⚠️ Could not change avoid for {identity_id}: {exc}")
            self._expr_status.setText(f"Could not save the avoid setting: {exc}")
            return

        card = self._cards.get(identity_id)
        if card is not None:
            card.set_avoided(avoided)
        if identity_id in self._identities:
            self._identities[identity_id]["avoided"] = avoided
        name = (self._identities.get(identity_id) or {}).get("name", "this person")
        self._expr_status.setText(
            f"{name} will be {'skipped' if avoided else 'included'} on the next run.")

    def _on_face_selected(self, identity_id: str):
        info = self._identities.get(identity_id, {})
        name = info.get("name", identity_id[:8])

        object_bboxes = self._entries()
        segments = build_segments(object_bboxes, identity_id, self._video_duration)
        self._current_segments = segments

        total = sum(e - s for s, e in segments)
        self._results_header.setText(
            f"{name} — {len(segments)} segment{'s' if len(segments) != 1 else ''}, {total:.1f}s total"
        )
        self._add_all_btn.setEnabled(bool(segments))
        self._refresh_results(segments)

    # ── expressions ───────────────────────────────────────────────────────────
    def _load_cached_expressions(self):
        """Reuse a scan from a previous session or the highlighter's own run."""
        if not self._video_path:
            return
        try:
            from modules.face_scan import cache_path_for, load as load_scan
            seconds = load_scan(cache_path_for(self._video_path))
        except Exception:
            seconds = None
        if seconds:
            self._apply_expression_scan(seconds, cached=True)

    def _scan_expressions(self):
        if self._expr_worker is not None:
            self._expr_worker.cancel()
            self._expr_status.setText("Cancelling…")
            return
        if not self._video_path:
            self._expr_status.setText("No video to scan.")
            return

        self._expr_scan_btn.setText("Cancel")
        self._expr_status.setText("Scanning…")
        worker = _ExpressionScanWorker(self._video_path)
        worker.done.connect(self._on_expression_scan_done)
        worker.failed.connect(self._on_expression_scan_failed)
        worker.progress.connect(
            lambda at, total: self._expr_status.setText(
                f"Scanning… {at / total * 100:.0f}%" if total else "Scanning…"))
        worker.finished.connect(self._on_expression_worker_finished)
        self._expr_worker = worker
        worker.start()

    def _on_expression_worker_finished(self):
        self._expr_worker = None
        self._expr_scan_btn.setText("Scan expressions")

    def _on_expression_scan_failed(self, message: str):
        self._expr_status.setText(f"Scan failed: {message}")

    def _on_expression_scan_done(self, seconds: dict):
        if not seconds:
            # The classifier is optional; saying so beats an empty result that
            # looks like the video simply had no faces in it.
            self._expr_status.setText(
                "No expressions found. If the expression model is not "
                "installed, that is why — the log says which.")
            return
        self._apply_expression_scan(seconds)

    def _apply_expression_scan(self, seconds: dict, cached: bool = False):
        from modules.face_scan import label_counts

        self._expr_seconds = seconds
        self._expr_combo.setEnabled(True)
        counts = label_counts(seconds)
        summary = ", ".join(f"{label} {count}"
                            for label, count in sorted(counts.items(),
                                                       key=lambda kv: -kv[1])
                            if count)
        prefix = "Cached scan" if cached else "Scanned"
        self._expr_status.setText(
            f"{prefix}: {len(seconds)} second(s) with a readable face"
            + (f" — {summary}" if summary else ""))

    def _on_expression_picked(self, _index: int):
        label = self._expr_combo.currentData() or ""
        if not label or not self._expr_seconds:
            return
        from modules.face_scan import segments_for

        segments = segments_for(self._expr_seconds, label,
                                duration=self._video_duration)
        self._current_segments = segments
        self._results_header.setText(f"{len(segments)} clip(s) of '{label}'")
        self._add_all_btn.setEnabled(bool(segments))
        self._refresh_results(segments)

    def _refresh_results(self, segments: list[tuple[float, float]]):
        # Clear rows (keep trailing stretch)
        while self._seg_list_layout.count() > 1:
            item = self._seg_list_layout.takeAt(0)
            widget = item.widget()
            if widget is None:
                continue
            if widget is self._no_results_lbl:
                # Long-lived and reused across refreshes, unlike the rows.
                # deleteLater() only schedules the destruction, so it survives
                # this call and dies on the next turn of the event loop —
                # leaving self._no_results_lbl pointing at a freed C++ object
                # for whichever refresh comes after. Unparent it instead.
                widget.setParent(None)
                continue
            widget.deleteLater()

        if not segments:
            self._no_results_lbl.show()
            self._seg_list_layout.insertWidget(0, self._no_results_lbl)
            return

        self._no_results_lbl.hide()
        for i, (start, end) in enumerate(segments):
            row = _SegmentRow(i, start, end, self._jump, self._add_clip)
            self._seg_list_layout.insertWidget(i, row)

    def _add_all(self):
        for start, end in self._current_segments:
            self._add_clip(start, end)
