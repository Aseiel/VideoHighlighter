"""The ◀ arrow beside a timeline row, during playback.

One click returns to the start of the group the playhead is in, so a group can
be watched again and again with the arrow alone. A double-click steps to the
group before. Before that, the second click measured from the playhead, which
during playback had already left the start the first click landed on, so it
restarted the same group and stepping back needed a pause.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

GROUPS = [10.0, 30.0, 50.0]


@pytest.fixture(scope="module")
def app():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


class _Panel:
    """Just the ◀ click state; the real methods are bound onto it."""

    def __init__(self, clock):
        from signal_timeline_viewer import SignalLabelPanel
        self._find_target = SignalLabelPanel._find_target
        self._prev_origin = SignalLabelPanel._prev_origin.__get__(self)
        self.clock = clock

    def click_prev(self, playhead, monkeypatch):
        import signal_timeline_viewer as stv
        monkeypatch.setattr(stv.time, "monotonic", lambda: self.clock[0])
        origin = self._prev_origin("ACTIONS", "prev", playhead)
        target = self._find_target(GROUPS, origin, "prev")
        if target is not None:
            self._last_prev = ("ACTIONS", self.clock[0], target)
        return target


@pytest.fixture
def panel(app):
    return _Panel([100.0])


def test_one_click_restarts_the_current_group(panel, monkeypatch):
    assert panel.click_prev(31.0, monkeypatch) == 30.0


def test_slow_repeat_clicks_keep_restarting_it(panel, monkeypatch):
    # Watching a group again and again: the playhead moves on between clicks.
    assert panel.click_prev(31.0, monkeypatch) == 30.0
    panel.clock[0] += 3.0
    assert panel.click_prev(33.0, monkeypatch) == 30.0


def test_a_double_click_steps_to_the_previous_group(panel, monkeypatch):
    assert panel.click_prev(31.0, monkeypatch) == 30.0
    panel.clock[0] += 0.2
    # Playback has moved the playhead past the start; the double-click
    # measures from where the first click landed, not from the playhead.
    assert panel.click_prev(30.2, monkeypatch) == 10.0


def test_a_double_click_on_another_row_does_not_chain(panel, monkeypatch):
    panel._last_prev = ("OBJECTS", panel.clock[0], 30.0)
    panel.clock[0] += 0.2
    assert panel.click_prev(31.0, monkeypatch) == 30.0


def test_paused_on_a_group_start_one_click_goes_back(panel, monkeypatch):
    assert panel.click_prev(30.0, monkeypatch) == 10.0


def test_next_is_unchanged(app):
    from signal_timeline_viewer import SignalLabelPanel
    assert SignalLabelPanel._find_target(GROUPS, 30.5, "next") == 50.0
    assert SignalLabelPanel._find_target(GROUPS, 50.5, "next") is None
