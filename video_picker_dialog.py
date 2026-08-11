"""
Video picker dialog — browse a listing page as a grid of [✓] thumbnail + title
cards and choose which videos to download.

Scraping (downloader.extract_video_entries) and thumbnail fetches run off the GUI
thread so the dialog stays responsive. On accept, selected_entries() returns the
chosen [{url, title, thumbnail_url, duration}] dicts for the caller to download.

Standalone preview (no app needed):
    python video_picker_dialog.py --url "https://example.com/videos" --pattern "/video/"

Be considerate of the target site: respect its Terms of Service and robots.txt,
and only download content you're permitted to.
"""
from __future__ import annotations

from typing import List, Dict

import requests
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QScrollArea, QWidget,
    QLabel, QCheckBox, QPushButton, QFrame, QApplication,
)
from PySide6.QtCore import Qt, QThread, QObject, Signal, Slot
from PySide6.QtGui import QPixmap

from downloader import extract_video_entries

_COLUMNS = 3
_THUMB_W, _THUMB_H = 200, 112


class _ScrapeWorker(QObject):
    done = Signal(list)
    error = Signal(str)
    log = Signal(str)

    def __init__(self, url, pattern, use_browser):
        super().__init__()
        self._url, self._pattern, self._use_browser = url, pattern, use_browser

    @Slot()
    def run(self):
        try:
            entries = extract_video_entries(
                self._url, pattern=self._pattern,
                log_fn=lambda m: self.log.emit(str(m)),
                use_browser=self._use_browser,
            )
            self.done.emit(entries)
        except Exception as e:
            self.error.emit(str(e))


class _ThumbWorker(QObject):
    """Fetch thumbnails sequentially; emit (index, image_bytes) per success."""
    thumb = Signal(int, bytes)
    finished = Signal()

    _HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    def __init__(self, urls: List[str]):
        super().__init__()
        self._urls = urls
        self._stop = False

    def stop(self):
        self._stop = True

    @Slot()
    def run(self):
        for i, u in enumerate(self._urls):
            if self._stop:
                break
            if not u:
                continue
            try:
                r = requests.get(u, headers=self._HEADERS, timeout=15)
                if r.ok and r.content:
                    self.thumb.emit(i, r.content)
            except Exception:
                pass
        self.finished.emit()


class _Card(QFrame):
    def __init__(self, entry: Dict, parent=None):
        super().__init__(parent)
        self.entry = entry
        self.setFrameShape(QFrame.StyledPanel)
        self.setFixedWidth(_THUMB_W + 20)

        v = QVBoxLayout(self)
        v.setContentsMargins(8, 6, 8, 6)

        top = QHBoxLayout()
        self.checkbox = QCheckBox()
        top.addWidget(self.checkbox)
        top.addStretch()
        dur = entry.get("duration")
        if dur:
            top.addWidget(QLabel(str(dur)))
        v.addLayout(top)

        self.thumb_label = QLabel("loading…")
        self.thumb_label.setFixedSize(_THUMB_W, _THUMB_H)
        self.thumb_label.setAlignment(Qt.AlignCenter)
        self.thumb_label.setStyleSheet("background:#222;color:#888;border:1px solid #333;")
        self.thumb_label.setCursor(Qt.PointingHandCursor)
        self.thumb_label.mousePressEvent = lambda e: self.checkbox.toggle()
        v.addWidget(self.thumb_label)

        self.title_label = QLabel(entry.get("title", ""))
        self.title_label.setWordWrap(True)
        self.title_label.setFixedHeight(42)
        self.title_label.setToolTip(entry.get("title", ""))
        self.title_label.setStyleSheet("font-size:9pt;")
        v.addWidget(self.title_label)

    def set_thumb(self, data: bytes):
        pix = QPixmap()
        if pix.loadFromData(data):
            self.thumb_label.setPixmap(
                pix.scaled(_THUMB_W, _THUMB_H, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            self.thumb_label.setText("no preview")

    def is_checked(self) -> bool:
        return self.checkbox.isChecked()


class VideoPickerDialog(QDialog):
    def __init__(self, url: str, pattern: str = "/video/", use_browser: str = "auto", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select videos to download")
        self.resize(_COLUMNS * (_THUMB_W + 28) + 60, 640)

        self._entries: List[Dict] = []
        self._cards: List[_Card] = []
        self._selected: List[Dict] = []
        self._scrape_thread = None
        self._scrape_worker = None
        self._thumb_thread = None
        self._thumb_worker = None

        root = QVBoxLayout(self)

        self.status_label = QLabel("Loading listing…")
        self.status_label.setStyleSheet("color:#2f81f7;font-style:italic;")
        root.addWidget(self.status_label)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_host = QWidget()
        self.grid = QGridLayout(self.grid_host)
        self.grid.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.grid_host)
        root.addWidget(self.scroll, stretch=1)

        btns = QHBoxLayout()
        self.select_all_btn = QPushButton("Select all")
        self.select_all_btn.clicked.connect(lambda: self._set_all(True))
        self.select_none_btn = QPushButton("Select none")
        self.select_none_btn.clicked.connect(lambda: self._set_all(False))
        btns.addWidget(self.select_all_btn)
        btns.addWidget(self.select_none_btn)
        btns.addStretch()
        self.download_btn = QPushButton("Download selected")
        self.download_btn.setStyleSheet(
            "QPushButton{background:#4CAF50;color:white;font-weight:bold;padding:6px 16px;}"
        )
        self.download_btn.setEnabled(False)
        self.download_btn.clicked.connect(self._accept_selection)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btns.addWidget(self.cancel_btn)
        btns.addWidget(self.download_btn)
        root.addLayout(btns)

        self._start_scrape(url, pattern, use_browser)

    # -------------------------------------------------------------- scraping
    def _start_scrape(self, url, pattern, use_browser):
        self._scrape_thread = QThread(self)
        self._scrape_worker = _ScrapeWorker(url, pattern, use_browser)
        self._scrape_worker.moveToThread(self._scrape_thread)
        self._scrape_thread.started.connect(self._scrape_worker.run)
        self._scrape_worker.log.connect(lambda m: self.status_label.setText(m))
        self._scrape_worker.done.connect(self._on_entries)
        self._scrape_worker.error.connect(self._on_scrape_error)
        self._scrape_worker.done.connect(self._scrape_thread.quit)
        self._scrape_worker.error.connect(self._scrape_thread.quit)
        self._scrape_thread.start()

    @Slot(list)
    def _on_entries(self, entries: List[Dict]):
        self._entries = entries or []
        if not self._entries:
            self.status_label.setText("No videos found on that page.")
            return
        self.status_label.setText(f"{len(self._entries)} video(s) — pick which to download.")
        for idx, entry in enumerate(self._entries):
            card = _Card(entry)
            card.checkbox.stateChanged.connect(self._update_download_btn)
            self.grid.addWidget(card, idx // _COLUMNS, idx % _COLUMNS)
            self._cards.append(card)
        self.download_btn.setEnabled(False)
        self._start_thumbs()

    @Slot(str)
    def _on_scrape_error(self, msg: str):
        self.status_label.setText(f"Failed to load listing: {msg}")

    # -------------------------------------------------------------- thumbnails
    def _start_thumbs(self):
        urls = [e.get("thumbnail_url") for e in self._entries]
        self._thumb_thread = QThread(self)
        self._thumb_worker = _ThumbWorker(urls)
        self._thumb_worker.moveToThread(self._thumb_thread)
        self._thumb_thread.started.connect(self._thumb_worker.run)
        self._thumb_worker.thumb.connect(self._on_thumb)
        self._thumb_worker.finished.connect(self._thumb_thread.quit)
        self._thumb_thread.start()

    @Slot(int, bytes)
    def _on_thumb(self, index: int, data: bytes):
        if 0 <= index < len(self._cards):
            self._cards[index].set_thumb(data)

    # -------------------------------------------------------------- selection
    def _set_all(self, checked: bool):
        for c in self._cards:
            c.checkbox.setChecked(checked)

    def _update_download_btn(self):
        n = sum(1 for c in self._cards if c.is_checked())
        self.download_btn.setEnabled(n > 0)
        self.download_btn.setText(f"Download selected ({n})" if n else "Download selected")

    def _accept_selection(self):
        self._selected = [c.entry for c in self._cards if c.is_checked()]
        self._stop_threads()
        self.accept()

    def selected_entries(self) -> List[Dict]:
        return self._selected

    # -------------------------------------------------------------- lifecycle
    def _stop_threads(self):
        if self._thumb_worker:
            self._thumb_worker.stop()
        for th in (self._thumb_thread, self._scrape_thread):
            if th and th.isRunning():
                th.quit()
                th.wait(2000)

    def reject(self):
        self._stop_threads()
        super().reject()

    def closeEvent(self, event):
        self._stop_threads()
        super().closeEvent(event)


def _standalone():
    import argparse
    ap = argparse.ArgumentParser(description="Preview the video picker dialog")
    ap.add_argument("--url", required=True)
    ap.add_argument("--pattern", default="/video/")
    ap.add_argument("--browser", default="auto", choices=["auto", "never", "always"])
    args = ap.parse_args()

    app = QApplication([])
    dlg = VideoPickerDialog(args.url, pattern=args.pattern, use_browser=args.browser)
    if dlg.exec():
        sel = dlg.selected_entries()
        print(f"\nSelected {len(sel)} video(s):")
        for e in sel:
            print(f"  {e['title']}  →  {e['url']}")
    else:
        print("Cancelled.")


if __name__ == "__main__":
    _standalone()
