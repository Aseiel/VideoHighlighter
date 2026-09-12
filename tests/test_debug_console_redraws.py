"""
The debug-log tee keeps tqdm's in-place redraws from flooding debug.log.

A progress bar rewrites its line with "\\r" several times a second. A terminal
overwrites; the log file appended every one, and a user's five-hour video left
over a megabyte of bars behind — more than they could paste into a report.
"""

from __future__ import annotations

import collections
import io

import pytest

from modules import debug_console


@pytest.fixture
def tee(monkeypatch):
    log = io.StringIO()
    clock = [1000.0]
    monkeypatch.setattr(debug_console, "_log_fh", log)
    monkeypatch.setattr(debug_console, "_gui_sink", None)
    monkeypatch.setattr(debug_console, "_backlog", collections.deque(maxlen=50))
    monkeypatch.setattr(debug_console, "_clock", lambda: clock[0])
    terminal = io.StringIO()
    return debug_console._Tee(terminal), log, terminal, clock


def test_redraws_inside_the_interval_collapse_to_the_last_one(tee):
    t, log, _terminal, _clock = tee

    for n in range(1, 6):
        t.write(f"\rbar {n}/5")
    t.write("\n")

    text = log.getvalue()
    assert "bar 1/5" in text and "bar 5/5\n" in text
    assert not any(f"bar {n}/5" in text for n in (2, 3, 4))


def test_a_redraw_after_the_interval_is_kept(tee):
    t, log, _terminal, clock = tee

    t.write("\rbar 1")
    clock[0] += debug_console._REDRAW_INTERVAL + 1
    t.write("\rbar 2")

    assert "bar 2" in log.getvalue()


def test_other_output_brings_the_latest_bar_with_it(tee):
    t, log, _terminal, _clock = tee

    t.write("\rbar 1")
    t.write("\rbar 2")
    t.write("Motion detection error: boom\n")

    assert "bar 2Motion detection error: boom\n" in log.getvalue()


def test_ordinary_lines_are_untouched_and_stamped(tee):
    t, log, _terminal, _clock = tee

    t.write("hello\n")

    assert log.getvalue().startswith("[") and log.getvalue().endswith("] hello\n")


def test_the_terminal_still_sees_every_redraw(tee):
    t, _log, terminal, _clock = tee

    for n in range(1, 4):
        t.write(f"\rbar {n}")

    assert terminal.getvalue() == "\rbar 1\rbar 2\rbar 3"
