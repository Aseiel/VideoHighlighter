"""
Seeking with the preview slider.

Two faults met in one symptom: grabbing the playhead and letting go near where
it already was stopped playback dead and did not seek.

1. The slider ran 0-100, so one step was 1% of the file — 36 seconds of an
   hour-long video. A seek to anywhere inside the current step rounded to the
   value the slider already held, so `valueChanged` never fired and no seek
   happened.
2. `_block_position_updates` was raised on sliderPressed and lowered only
   inside `seek_video`, which runs on `valueChanged`. When (1) swallowed the
   change, the block was never lifted: the slider, the clock and the timeline
   playhead froze for the rest of the session while the video kept playing.

The methods under test are the window's own; the slider is wired here the way
`SignalTimelineWindow` wires it, which `TestTheWiring` pins.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QSlider
from PySide6.QtCore import Qt

import signal_timeline_viewer as stv


VIEWER_SOURCE = Path(stv.__file__).read_text(encoding="utf-8")

ONE_HOUR_MS = 60 * 60 * 1000


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


class _Player:
    """Just enough QMediaPlayer for the seek path."""

    def __init__(self, duration=ONE_HOUR_MS, position=0):
        self._duration = duration
        self._position = position
        self.seeks = []

    def duration(self):
        return self._duration

    def position(self):
        return self._position

    def setPosition(self, ms):
        self._position = ms
        self.seeks.append(ms)

    def playbackState(self):
        return None


class _Window:
    """The window's own seek methods, without building a QMainWindow.

    Bound off the class, so these are the functions that ship — a harness
    rather than a reimplementation.
    """

    _on_slider_pressed = stv.SignalTimelineWindow._on_slider_pressed
    _on_slider_released = stv.SignalTimelineWindow._on_slider_released
    seek_video = stv.SignalTimelineWindow.seek_video
    _handle_position_update = stv.SignalTimelineWindow._handle_position_update


@pytest.fixture
def window(app):
    """A bare window carrying only what the seek path touches."""
    win = _Window()
    win._active_player = _Player()
    win._block_position_updates = False
    win.current_time = 0.0
    # Shadow the collaborators the seek path calls into; signal_scene and
    # signal_view are left absent so the hasattr guards skip them.
    win.update_time_display = lambda ms: None
    win._update_detection_panel = lambda seconds: None

    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, win._active_player.duration())
    slider.sliderPressed.connect(win._on_slider_pressed)
    slider.sliderReleased.connect(win._on_slider_released)
    slider.valueChanged.connect(win.seek_video)
    win.time_slider = slider
    return win


class TestLettingGoWithoutMovingTheHandle:
    """The reported bug: press, release near where it was, playback dies."""

    def test_release_seeks_even_though_the_value_did_not_change(self, window):
        window.time_slider.setValue(30_000)
        window._active_player.seeks.clear()

        window.time_slider.sliderPressed.emit()
        window.time_slider.sliderReleased.emit()

        assert window._active_player.seeks == [30_000]

    def test_release_does_not_leave_updates_blocked_forever(self, window):
        window.time_slider.sliderPressed.emit()
        assert window._block_position_updates is True
        window.time_slider.sliderReleased.emit()

        from PySide6.QtTest import QTest
        QTest.qWait(350)          # the 200ms settle timer seek_video arms
        assert window._block_position_updates is False

    def test_position_updates_flow_again_after_the_release(self, window):
        window.time_slider.sliderPressed.emit()
        window.time_slider.sliderReleased.emit()

        from PySide6.QtTest import QTest
        QTest.qWait(350)

        window._active_player._position = 12_345
        window._handle_position_update(12_345)
        assert window.time_slider.value() == 12_345

    def test_a_seek_with_no_media_still_lifts_the_block(self, window):
        window._active_player._duration = 0
        window.time_slider.sliderPressed.emit()
        window._on_slider_released()
        # No duration means no seek is possible, but the block must not stick.
        assert window._block_position_updates is False
        assert window._active_player.seeks == []


class TestTheSliderIsInMilliseconds:
    def test_value_is_the_position_to_seek_to(self, window):
        window.seek_video(90_000)
        assert window._active_player.seeks == [90_000]

    def test_a_one_second_move_in_an_hour_long_video_seeks(self, window):
        """Under the old 0-100 slider this rounded away to nothing."""
        window.time_slider.setValue(600_000)          # 10:00
        window._active_player.seeks.clear()
        window.time_slider.setValue(601_000)          # 10:01
        assert window._active_player.seeks == [601_000]

    def test_seeking_near_the_beginning_lands_where_asked(self, window):
        window.time_slider.setValue(500)              # half a second in
        assert window._active_player.seeks[-1] == 500

    def test_position_updates_write_milliseconds_to_the_slider(self, window):
        window._handle_position_update(754_000)
        assert window.time_slider.value() == 754_000

    def test_the_range_follows_the_duration(self, window):
        window._active_player._duration = 123_456
        window._handle_position_update(1_000)
        assert window.time_slider.maximum() == 123_456

    def test_out_of_range_values_are_clamped(self, window):
        window.seek_video(ONE_HOUR_MS * 5)
        assert window._active_player.seeks[-1] == ONE_HOUR_MS
        window.seek_video(-9_000)
        assert window._active_player.seeks[-1] == 0

    def test_arrow_and_page_steps_are_time_sized(self):
        # A slider counting milliseconds with Qt's default step of 1 would move
        # one millisecond per arrow key.
        assert "setSingleStep(1000)" in VIEWER_SOURCE
        assert "setPageStep(10000)" in VIEWER_SOURCE


class TestTheWiring:
    """The handlers are only reached if the window still connects them."""

    def test_release_is_connected(self):
        assert re.search(r"time_slider\.sliderReleased\.connect\(\s*self\._on_slider_released",
                         VIEWER_SOURCE)

    def test_press_is_connected(self):
        assert re.search(r"time_slider\.sliderPressed\.connect\(\s*self\._on_slider_pressed",
                         VIEWER_SOURCE)

    def test_the_percent_range_is_gone(self):
        # `setRange(0, 100)` on this slider is the 1%-per-step bug.
        assert not re.search(r"time_slider\.setRange\(0,\s*100\)", VIEWER_SOURCE)
