"""Clicking the edit timeline seeks; it never starts playback.

Four faults behind one report ("clicking a clip autoplays the edit"):

1. Stop left `_edit_playback_active` set. A later click on a clip read that as
   "edit playback is running" and rerouted it to the clicked clip, which
   starts the player.
2. The edit scene emitted `time_clicked` with the click's x as a time. That
   signal carries *source* time; x is a position in the cut. Every click first
   seeked the source to an unrelated moment, then the clip seeked it back.
3. `hide_drop_indicator` removed a marker `clear_drop_indicators` had already
   removed, handing Qt a null item on every clip release.
4. The clicked clip was found again by source time, and two clips can cover
   the same source moment: a click on the right of one clip put the playhead
   in the first clip containing that time. The click now carries its index.

The window methods are the window's own, run against a stand-in that holds
just what they touch, as in test_seek_slider.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt, qInstallMessageHandler
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QGraphicsView

import signal_timeline_viewer as stv
from video_ai_editor.edit_timeline import EditTimelineScene


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


class _Player:
    def __init__(self):
        self.plays = 0
        self.position = None

    def play(self):
        self.plays += 1

    def pause(self):
        pass

    def setPosition(self, ms):
        self.position = ms


class _Recorder:
    """Accepts any call; every one is harmless here."""

    def __getattr__(self, name):
        return lambda *a, **k: None


class _Window:
    # The window's own handler, which on_edit_clip_clicked delegates to.
    on_edit_time_clicked = stv.SignalTimelineWindow.on_edit_time_clicked

    def __init__(self, clips):
        self._active_player = _Player()
        self.video_player = self._active_player
        self.video_duration = 600.0
        self.current_time = 0.0
        self.signal_scene = _Recorder()
        self.signal_view = _Recorder()
        self.time_label = _Recorder()
        self.play_edit_btn = _Recorder()
        self.edit_scene = _Recorder()
        self.edit_scene.clips = list(clips)
        self.rerouted = []
        self._edit_playback_active = True    # as left by an earlier Play Edit
        self._edit_paused = False
        self._single_clip_playing = False

    def statusBar(self):
        return _Recorder()

    def _reroute_edit_playback_to(self, index, pos):
        self.rerouted.append((index, pos))


class TestStopThenClick:
    def test_a_click_after_stop_seeks_without_playing(self):
        win = _Window([(100.0, 120.0), (300.0, 330.0)])
        stv.SignalTimelineWindow.stop_edit_playback(win)
        assert win._edit_playback_active is False

        stv.SignalTimelineWindow.on_edit_time_clicked(win, 310.0)
        assert win.rerouted == []
        assert win._active_player.plays == 0
        assert win._active_player.position == 310_000

    def test_overlapping_clips_put_the_playhead_in_the_clicked_one(self):
        # Clip 1 covers 978-992, clip 2 963-995: 987 s is in both. A click on
        # clip 2 used to land the playhead in clip 1.
        win = _Window([(978.4, 991.6), (963.2, 994.9)])
        active = []
        win.edit_scene.set_active_clip = active.append
        win._edit_playback_active = False
        stv.SignalTimelineWindow.on_edit_clip_clicked(win, 1, 987.0)
        assert active == [1]

    def test_a_click_while_playing_still_follows_the_click(self):
        win = _Window([(100.0, 120.0), (300.0, 330.0)])
        stv.SignalTimelineWindow.on_edit_time_clicked(win, 310.0)
        assert win.rerouted == [(1, 310.0)]


@pytest.fixture()
def edit_view(app, monkeypatch):
    # No thumbnails: their workers outlive the test (stop() doesn't join) and
    # fault inside whichever test is running when they next wake. The scene
    # already copes with a cache it couldn't build, by drawing no filmstrip.
    import video_ai_editor.edit_timeline as et

    def no_cache(*_a, **_k):
        raise RuntimeError("thumbnails not needed here")

    monkeypatch.setattr(et, "ThumbnailCache", no_cache)
    scene = EditTimelineScene("missing.mp4", 600.0)
    for start, end in [(100.0, 120.0), (300.0, 330.0)]:
        scene.add_clip(start, end)
    view = QGraphicsView(scene)
    view.resize(1400, 200)
    view.show()
    QTest.qWaitForWindowExposed(view)
    times = []
    scene.clip_clicked.connect(lambda index, t: times.append((index, t)))
    yield scene, view, times
    view.close()
    scene.cleanup()


def _click(view, scene_x, scene_y):
    pos = view.mapFromScene(scene_x, scene_y)
    QTest.mouseClick(view.viewport(), Qt.LeftButton, Qt.NoModifier, QPoint(pos))


class TestEditSceneClicks:
    def test_clicking_a_clip_sends_one_source_time(self, edit_view):
        scene, view, times = edit_view
        item = scene.clip_items[1]
        r = item.sceneBoundingRect()
        _click(view, r.left() + r.width() * 0.5, r.center().y())
        assert len(times) == 1
        index, t = times[0]
        assert index == 1
        assert t == pytest.approx(315.0, abs=0.5)

    def test_clicking_empty_timeline_sends_nothing(self, edit_view):
        scene, view, times = edit_view
        r = scene.clip_items[-1].sceneBoundingRect()
        _click(view, r.right() + 40, r.center().y())
        assert times == []

    def test_releasing_a_clip_logs_no_null_remove(self, edit_view):
        scene, view, times = edit_view
        warnings = []
        previous = qInstallMessageHandler(lambda _t, _c, msg: warnings.append(msg))
        try:
            scene.show_drop_indicator(50.0)
            scene.hide_drop_indicator()
            scene.hide_drop_indicator()
        finally:
            qInstallMessageHandler(previous)
        assert not [w for w in warnings if "cannot remove 0-item" in w]


class TestMotionEventsNavigation:
    def test_motion_events_get_arrows_one_stop_per_bar(self, app):
        from video_ai_editor.signal_timeline import SignalTimelineScene

        # Floats, as the pipeline caches them: two runs of sampled frames.
        events = [10.0 + i / 6 for i in range(12)] + [40.0 + i / 6 for i in range(6)]
        scene = SignalTimelineScene(cache_data={"motion_events": events},
                                    video_duration=60.0)
        stops = scene._nav_timestamps_motion_events()
        assert stops == [pytest.approx(10.0), pytest.approx(40.0)]


class TestSpace:
    """Space plays the timeline you were last working in. Only a running edit
    used to count, so a click on an edit clip then Space played the source."""

    def press_space(self, win):
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QKeyEvent
        event = QKeyEvent(QEvent.KeyPress, Qt.Key_Space, Qt.NoModifier)
        return stv.SignalTimelineWindow.eventFilter(win, None, event)

    def window(self):
        win = _Window([(978.4, 991.6), (963.2, 994.9)])
        win.video_duration = 1200.0      # long enough for these clips
        win._edit_playback_active = False
        win.played = []
        win.toggle_edit_playback = lambda: win.played.append("edit")
        win.toggle_video_playback = lambda: win.played.append("video")
        win.isActiveWindow = lambda: True
        return win

    def test_after_clicking_an_edit_clip_space_plays_the_edit(self, app):
        win = self.window()
        stv.SignalTimelineWindow.on_edit_clip_clicked(win, 1, 987.0)
        assert self.press_space(win) is True
        assert win.played == ["edit"]

    def test_the_main_play_button_hands_space_back(self, app):
        win = self.window()
        stv.SignalTimelineWindow.on_edit_clip_clicked(win, 1, 987.0)
        player = win._active_player
        player.playbackState = lambda: None
        win.play_btn = _Recorder()
        stv.SignalTimelineWindow.toggle_video_playback(win)
        self.press_space(win)
        assert win.played == ["video"]

    def test_nothing_clicked_space_plays_the_video(self, app):
        win = self.window()
        self.press_space(win)
        assert win.played == ["video"]

    def test_play_edit_resumes_in_the_clicked_overlapping_clip(self, app):
        win = self.window()
        stv.SignalTimelineWindow.on_edit_clip_clicked(win, 1, 987.0)
        clips = win.edit_scene.clips
        assert stv.SignalTimelineWindow._edit_resume_target(win, clips) == (1, 987.0)
