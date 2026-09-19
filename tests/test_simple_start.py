"""Simple view is the default workspace — the full settings UI stays.

The page exists so a first-time user can drop a video and press Analyze
without seeing every tab. Switching to Detailed settings must not destroy
widgets; the QSettings key only remembers which workspace they used last.
The assistant is one widget shared by both views, so the handoff is checked
against the real window rather than a stand-in.
"""
from __future__ import annotations

from pathlib import Path

import pytest


def test_simple_preset_can_score_without_any_knobs():
    from modules.segments.simple_run import apply_simple_run, simple_scoring_total
    assert simple_scoring_total() > 0
    cfg = apply_simple_run({}, "short")
    assert cfg["motion_peak_points"] > 0
    assert cfg["loudness_burst_points"] > 0
    assert cfg["max_duration"] == 90
    assert cfg["use_transcript"] is False
    assert cfg["write_highlight_report"] is True
    medium = apply_simple_run({}, "medium")
    assert medium["max_duration"] == 240
    assert medium["clip_time"] == 10


def test_main_keeps_the_full_ui_and_adds_the_stack():
    text = Path("main.py").read_text(encoding="utf-8")
    assert "SimpleStartPage" in text
    assert "set_simple_start" in text
    assert 'QPushButton("Run Highlighter")' in text
    assert "view_stack" in text


def test_video_path_accepts_common_containers():
    from modules.segments.simple_run import is_video_path
    assert is_video_path(r"C:\clips\talk.mp4")
    assert is_video_path("/tmp/a.MKV")
    assert not is_video_path("/tmp/notes.txt")
    assert not is_video_path("/tmp/noext")


def test_simple_start_defaults_on_when_unset(monkeypatch):
    pytest.importorskip("PySide6")
    from modules.ui import simple_start as ss
    store = {}

    class _FakeSettings:
        def value(self, key, default=None):
            return store.get(key, default)

        def setValue(self, key, val):
            store[key] = val

    monkeypatch.setattr(ss, "QSettings", lambda *a, **k: _FakeSettings())
    assert ss.simple_start_enabled(default=True) is True
    ss.persist_simple_start(False)
    assert ss.simple_start_enabled(default=True) is False
    ss.persist_simple_start(True)
    assert ss.simple_start_enabled() is True


def test_chat_panel_moves_between_the_two_views():
    """One LLMChatWidget, borrowed by whichever workspace is on screen.

    Building a second one would mean two model connections and two analysis
    caches each claiming to answer for the same video. Skipped where the app's
    runtime dependencies are not installed.
    """
    pytest.importorskip("PySide6")
    app_main = pytest.importorskip("main")
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    gui = app_main.VideoHighlighterGUI()
    try:
        chat = gui.llm_chat
        assert gui.simple_page.owns_chat(chat)

        gui.set_simple_start(False, persist=False)
        assert not gui.simple_page.owns_chat(chat)
        assert gui.llm_tab_layout.indexOf(chat) >= 0

        gui.set_simple_start(True, persist=False)
        assert gui.simple_page.owns_chat(chat)
        # Same object throughout: no rebuild, so a conversation survives the
        # trip to Detailed settings and back.
        assert gui.llm_chat is chat
    finally:
        gui.deleteLater()


def test_idle_status_names_the_loaded_file_count():
    from modules.segments.simple_run import idle_status_text
    assert idle_status_text(0) == "Ready"
    assert idle_status_text(1) == "1 video ready — press Analyze"
    assert idle_status_text(3) == "3 videos ready — press Analyze"


def test_simple_page_calls_the_length_a_highlight():
    text = Path("modules/ui/simple_start.py").read_text(encoding="utf-8")
    assert 'QLabel("Highlight length")' in text
    assert "about 1–2 minutes" in text
    assert "about 4 minutes" in text
    assert "about 7 minutes" in text
    assert "Reel length" not in text
    assert "Find and explain the moments that matter" in text
    assert "separate clips" in text
    assert "Footage stays on your disk" in text
    # No upsell on the empty first screen: an ad shown before the app has
    # produced anything is an ad. Pro belongs where there is a result to
    # extend, and in About.
    assert "Get Pro" not in text


def test_drop_zone_shows_the_filename_once_loaded():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    from modules.ui.simple_start import DropZone

    QApplication.instance() or QApplication([])
    zone = DropZone(lambda p: None, lambda: None)
    try:
        assert zone._title.text() == "Drop a video here"
        zone.set_loaded([r"C:\clips\talk.mp4"])
        assert zone._title.text() == "Loaded: talk.mp4"
        zone.set_loaded([r"C:\a.mp4", r"C:\b.mov"])
        assert zone._title.text() == "Loaded: 2 videos"
        zone.set_loaded([])
        assert zone._title.text() == "Drop a video here"
    finally:
        zone.deleteLater()


def test_drop_zone_accepts_only_video_urls():
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QMimeData, QUrl
    from PySide6.QtWidgets import QApplication
    from modules.ui import simple_start as ss

    QApplication.instance() or QApplication([])
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile("/tmp/clip.mp4"),
                  QUrl.fromLocalFile("/tmp/readme.txt")])
    paths = ss._video_urls(mime)
    assert len(paths) == 1
    assert paths[0].lower().replace("\\", "/").endswith("clip.mp4")
