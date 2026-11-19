"""A seek while the player is stopped must still show the frame.

QMediaPlayer in StoppedState takes the position and renders nothing; the live
overlay's player is stopped when the viewer first switches to it, and every
player is stopped again after the end of the video. Measured on a real file:
0 frames after a stopped seek, 1 after pause+seek. The fix is to pause first,
and only then (a playing or paused player must not be touched).
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6.QtMultimedia")
viewer = pytest.importorskip("signal_timeline_viewer")
QMediaPlayer = viewer.QMediaPlayer


class Player:
    def __init__(self, state):
        self.state, self.calls = state, []

    def playbackState(self):
        return self.state

    def pause(self):
        self.calls.append("pause")
        self.state = QMediaPlayer.PausedState

    def setPosition(self, ms):
        self.calls.append(("seek", ms))


def test_a_stopped_player_is_paused_before_the_seek():
    p = Player(QMediaPlayer.StoppedState)
    viewer.SignalTimelineWindow._seek_showing(p, 1428000)
    assert p.calls == ["pause", ("seek", 1428000)]


@pytest.mark.parametrize("state", ["PausedState", "PlayingState"])
def test_a_paused_or_playing_player_is_only_seeked(state):
    p = Player(getattr(QMediaPlayer, state))
    viewer.SignalTimelineWindow._seek_showing(p, 5000)
    assert p.calls == [("seek", 5000)]
