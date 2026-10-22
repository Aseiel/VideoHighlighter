"""Tests for hiding child processes' console windows (modules/system/hidden_subprocess.py).

Popen itself is replaced by a recorder, so nothing is started; what's pinned
down is the flag each call ends up with, and when the switch stays off. The
real effect (no window) was measured from a windowless Python on Windows: 10
console windows for 5 ffmpeg runs before, none after.
"""
from __future__ import annotations

import subprocess

import pytest

from modules.system import hidden_subprocess as hs


@pytest.fixture
def recorded(monkeypatch):
    """subprocess.Popen.__init__ replaced by a recorder; restored afterwards."""
    calls = []

    def record(self, *args, **kwargs):
        calls.append(kwargs.get("creationflags", 0))

    monkeypatch.setattr(subprocess.Popen, "__init__", record)
    monkeypatch.setattr(hs, "_installed", False)
    monkeypatch.setattr(hs.sys, "platform", "win32")
    return calls


def _start(**kwargs):
    subprocess.Popen.__init__(object.__new__(subprocess.Popen), ["ffmpeg", "-version"], **kwargs)


def test_every_child_gets_no_window(recorded):
    assert hs.install(force=True)
    _start()
    _start(creationflags=0)
    assert recorded == [hs.CREATE_NO_WINDOW, hs.CREATE_NO_WINDOW]


def test_flags_a_caller_set_are_kept(recorded):
    hs.install(force=True)
    _start(creationflags=0x00000200)          # CREATE_NEW_PROCESS_GROUP
    assert recorded == [0x00000200 | hs.CREATE_NO_WINDOW]


@pytest.mark.parametrize("flag", [hs.CREATE_NEW_CONSOLE, hs.DETACHED_PROCESS])
def test_a_caller_asking_for_its_own_console_arrangement_gets_it(recorded, flag):
    hs.install(force=True)
    _start(creationflags=flag)
    assert recorded == [flag]


def test_nothing_changes_with_a_console(recorded, monkeypatch):
    # Run from a terminal, children share it and their output belongs there.
    monkeypatch.setattr(hs, "_has_console", lambda: True)
    assert not hs.install()
    _start()
    assert recorded == [0]


def test_nothing_changes_off_windows(recorded, monkeypatch):
    monkeypatch.setattr(hs.sys, "platform", "darwin")
    assert not hs.install(force=True)
    _start()
    assert recorded == [0]


def test_installing_twice_wraps_once(recorded):
    assert hs.install(force=True)
    assert not hs.install(force=True)
    _start()
    assert recorded == [hs.CREATE_NO_WINDOW]
