"""The main window's scan buttons: what they say and when they can be pressed.

These replaced bare buttons named after a signal. The behaviour worth pinning
is the part a user reads: a button is only live when pressing it would do
something, the status says what is already on disk, and the button that
started a scan is the one that stops it.
"""
from __future__ import annotations

import pytest

from modules.report import scan_state

pytest.importorskip("PySide6.QtWidgets")


# ── what counts as saved ────────────────────────────────────────────────────

def test_an_empty_cache_holds_nothing():
    assert scan_state.kinds_in({}) == frozenset()
    assert scan_state.kinds_in(None) == frozenset()


def test_empty_lists_are_not_a_result():
    # A full run with a signal switched off writes the key with nothing in it.
    # Calling that "saved" would tell the user not to scan a video that has
    # never been scanned.
    cache = {"scenes": [], "motion_peaks": [], "audio_peaks": [],
             "audio": {}, "objects": [], "actions": [],
             "transcript": {"segments": []}}
    assert scan_state.kinds_in(cache) == frozenset()


def test_each_kind_is_recognised_from_its_own_keys():
    assert scan_state.kinds_in({"motion_peaks": [1.0]}) == {"motion"}
    assert scan_state.kinds_in({"audio": {"waveform": [0.1]}}) == {"audio"}
    assert scan_state.kinds_in(
        {"transcript": {"segments": [{"text": "hi"}]}}) == {"transcript"}
    assert scan_state.kinds_in(
        {"objects": [{"timestamp": 1, "objects": ["car"]}]}) == {"objects"}
    assert scan_state.kinds_in({"actions_all": [{"t": 1}]}) == {"actions"}


def test_rules_that_matched_nothing_still_count_as_applied():
    # The one kind that can tell "ran, found nothing" from "never ran".
    assert scan_state.kinds_in({"composed_event_names": []}) == {"composition"}


def test_a_video_is_the_union_of_its_cache_files(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text("{}"), b.write_text("{}")
    held = {str(a): {"motion_peaks": [1.0]},
            str(b): {"transcript": {"segments": [{"text": "x"}]}}}
    got = scan_state.saved_kinds(
        "v.mp4", find_files=lambda _vp: [a, b], load=lambda p: held[str(p)])
    assert got == {"motion", "transcript"}


def test_an_unreadable_cache_says_nothing_rather_than_failing(tmp_path):
    a = tmp_path / "broken.json"
    a.write_text("{")

    def load(_path):
        raise ValueError("half-written")

    assert scan_state.saved_kinds(
        "v.mp4", find_files=lambda _vp: [a], load=load) == frozenset()


def test_a_missing_video_has_nothing_saved():
    def find(_vp):
        raise FileNotFoundError

    assert scan_state.saved_kinds("gone.mp4", find_files=find) == frozenset()


def test_counts_are_per_kind_over_the_list(tmp_path):
    files = {}
    for name, data in (("one", {"motion_peaks": [1.0], "audio_peaks": [2.0]}),
                       ("two", {"motion_peaks": [3.0]})):
        f = tmp_path / f"{name}.json"
        f.write_text(name)
        files[name] = (f, data)
    counts = scan_state.count_saved(
        ["one", "two"],
        find_files=lambda vp: [files[vp][0]],
        load=lambda p: next(d for f, d in files.values() if f == p))
    assert counts["motion"] == 2
    assert counts["audio"] == 1
    assert counts["objects"] == 0


def test_status_wording():
    assert scan_state.describe(0, 0) == ""
    assert scan_state.describe(0, 3) == "nothing saved yet"
    assert scan_state.describe(1, 1) == "saved"
    assert scan_state.describe(3, 3) == "saved for all 3"
    assert scan_state.describe(1, 3) == "saved for 1 of 3"


# ── the buttons ─────────────────────────────────────────────────────────────

@pytest.fixture
def controls(monkeypatch):
    from PySide6.QtWidgets import QApplication, QWidget
    from modules.ui import scan_controls as sc

    QApplication.instance() or QApplication([])
    # No cache directory in a unit test: answer the background look directly.
    monkeypatch.setattr(sc.scan_state, "count_saved",
                        lambda paths, **_k: {"motion": len(paths)})
    host = QWidget()
    state = {"paths": [], "started": [], "stopped": 0}

    def stop():
        state["stopped"] += 1

    registry = sc.ScanControls(
        host, list_paths=lambda: list(state["paths"]),
        on_start=state["started"].append, on_stop=stop)
    registry._state = state
    registry._host = host
    return registry


def _settle(registry):
    """Run the debounced look and deliver its (stubbed) answer."""
    from PySide6.QtCore import QCoreApplication
    import time

    registry._debounce.stop()
    registry._refresh_now()
    deadline = time.time() + 2
    while time.time() < deadline:
        QCoreApplication.processEvents()
        if registry._counts is not None or not registry._paths:
            break
        time.sleep(0.01)


def test_a_button_is_dead_and_says_why_while_the_list_is_empty(controls):
    ctl = controls.control("motion", "Scan motion", "Detect scene cuts.")
    _settle(controls)
    assert not ctl.button.isEnabled()
    assert ctl.status.full_text() == "add a video first"
    assert "Add a video" in ctl.button.toolTip()


def test_a_listed_video_makes_it_live_and_shows_what_is_saved(controls):
    ctl = controls.control("motion", "Scan motion", "Detect scene cuts.")
    other = controls.control("audio", "Scan audio", "Detect audio peaks.")
    controls._state["paths"] = ["a.mp4"]
    _settle(controls)
    assert ctl.button.isEnabled()
    assert ctl.status.full_text() == "✓ saved"
    assert other.status.full_text() == "nothing saved yet"
    assert "Detect scene cuts." in ctl.button.toolTip()


def test_pressing_starts_that_kind(controls):
    ctl = controls.control("audio", "Scan audio", "tip")
    controls._state["paths"] = ["a.mp4"]
    _settle(controls)
    ctl.button.click()
    assert controls._state["started"] == ["audio"]


def test_the_running_button_becomes_stop_and_the_rest_wait(controls):
    audio = controls.control("audio", "Scan audio", "tip")
    motion = controls.control("motion", "Scan motion", "tip")
    controls._state["paths"] = ["a.mp4"]
    _settle(controls)

    controls.set_running("audio")
    assert audio.button.text() == "Stop" and audio.button.isEnabled()
    assert not motion.button.isEnabled()

    # Reports are coalesced: only the latest one is drawn, on a timer.
    controls.set_progress("audio", 41, "peaks")
    controls.set_progress("audio", 42, "peaks")
    assert audio.status.full_text() == "starting…"
    controls._draw_progress()
    assert audio.status.full_text() == "42% · peaks"

    audio.button.click()                     # Stop, not a second start
    assert controls._state["stopped"] == 1
    assert controls._state["started"] == []
    # Late progress must not paint over "stopping…".
    controls.set_progress("audio", 43, "peaks")
    controls._draw_progress()
    assert audio.status.full_text().startswith("stopping")

    controls.set_running(None)
    assert audio.button.text() == "Scan audio"
    assert motion.button.isEnabled()


def test_a_full_run_blocks_every_button(controls):
    ctl = controls.control("motion", "Scan motion", "tip")
    controls._state["paths"] = ["a.mp4"]
    _settle(controls)
    controls.set_blocked(True)
    assert not ctl.button.isEnabled()
    controls.set_blocked(False)
    assert ctl.button.isEnabled()


def test_a_note_outranks_the_cache_until_the_list_changes(controls):
    ctl = controls.control("motion", "Scan motion", "tip")
    controls._state["paths"] = ["a.mp4"]
    _settle(controls)
    controls.note("motion", "failed — see the log below")
    assert ctl.status.full_text() == "failed — see the log below"

    controls._state["paths"] = ["a.mp4", "b.mp4"]
    _settle(controls)
    assert ctl.status.full_text() == "✓ saved for all 2"


def test_both_buttons_of_one_kind_move_together(controls):
    # Composition has one beside its editor and one in the signals list.
    first = controls.control("composition", "Apply rules", "tip")
    second = controls.control("composition", "Run rules", "tip")
    controls._state["paths"] = ["a.mp4"]
    _settle(controls)
    controls.set_running("composition")
    assert first.button.text() == second.button.text() == "Stop"


# ── dropping files ──────────────────────────────────────────────────────────

def test_only_video_files_are_taken_from_a_drop():
    from PySide6.QtCore import QMimeData, QUrl
    from modules.ui.video_drop import video_paths

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile("C:/clips/a.mp4"),
                  QUrl.fromLocalFile("C:/clips/notes.txt"),
                  QUrl("https://example.com/b.mp4")])
    assert [p.replace("\\", "/") for p in video_paths(mime)] == ["C:/clips/a.mp4"]
    assert video_paths(QMimeData()) == []


# ── the progress handlers' event pump ───────────────────────────────────────

def test_the_progress_pump_does_not_nest(monkeypatch):
    """A report handled while the window is already pumping must not pump again.

    Each nested pump handles the next queued report one stack frame deeper;
    when reports outrun the repaint that never unwinds, and the process dies
    below Python with no traceback.
    """
    import main as app_main

    depth = {"now": 0, "max": 0, "calls": 0}

    class Window:
        _pump_events = app_main.VideoHighlighterGUI._pump_events

    window = Window()

    def fake_process_events(*_a):
        depth["calls"] += 1
        depth["now"] += 1
        depth["max"] = max(depth["max"], depth["now"])
        if depth["calls"] < 50:
            window._pump_events()        # a queued report arriving mid-pump
        depth["now"] -= 1

    monkeypatch.setattr(app_main.QApplication, "processEvents",
                        staticmethod(fake_process_events))
    window._pump_events()
    assert depth["max"] == 1
    window._pump_events()                # and it works again afterwards
    assert depth["calls"] == 2
