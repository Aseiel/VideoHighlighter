"""What the action loop does when something it waits on never answers.

0.12.0 moved decoding onto a background thread, and a 0.12.0 run wedged with the
progress bar frozen and `debug.log` ending on the last line before the main
loop. Nothing else was recorded: the loop waits on other threads and on two GPU
runtimes, and a wait that never returns raises nothing, exits nothing, prints
nothing.

Two things are pinned here. `_FramePrefetcher.stop()` must not hand back a
capture its decode thread is still inside -- the caller's next line releases it,
and releasing a capture under a live read is a use-after-free in the FFmpeg
backend. And `_StallWatchdog` must name the phase and dump the stacks, because
that is the only thing that will identify the next hang.

`action_recognition` imports cv2/torch at module scope; conftest's shims cover
both, and nothing here touches a real capture.
"""

from __future__ import annotations

import threading
import time

import pytest

from action_recognition import _FramePrefetcher, _StallWatchdog


class FakeCap:
    """A capture that yields `frames` and then reports end of stream."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.released = False

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)


class WedgedCap:
    """A capture whose read() never returns -- a stalled decoder, or a file the
    backend will not give up on."""

    def __init__(self):
        self.entered = threading.Event()
        self.let_go = threading.Event()

    def read(self):
        self.entered.set()
        self.let_go.wait()
        return False, None


def test_frames_come_out_in_order_and_end_exactly():
    reader = _FramePrefetcher(FakeCap([1, 2, 3])).start()
    assert [reader.read() for _ in range(3)] == [1, 2, 3]
    assert reader.read() is None
    assert reader.stop() is True


def test_stop_reports_a_decode_thread_it_could_not_join():
    """False is the whole point: the caller skips cap.release() on it."""
    cap = WedgedCap()
    reader = _FramePrefetcher(cap).start()
    assert cap.entered.wait(5.0)
    began = time.monotonic()
    try:
        assert reader.stop(timeout=0.5) is False
        assert time.monotonic() - began < 5.0  # bounded: it must not hang either
    finally:
        cap.let_go.set()


def test_stop_waits_for_a_reader_that_is_merely_slow():
    """The normal case: one in-flight read, then the thread is really gone."""
    slow = threading.Event()

    class SlowCap:
        def read(self):
            slow.wait(0.4)
            return True, 1

    reader = _FramePrefetcher(SlowCap()).start()
    assert reader.stop(timeout=10.0) is True


def test_stop_drains_a_producer_blocked_on_a_full_queue():
    reader = _FramePrefetcher(FakeCap(list(range(500))), queue_size=2).start()
    time.sleep(0.2)  # let it fill and block on put()
    assert reader.stop(timeout=10.0) is True


def test_the_watchdog_names_the_phase_and_dumps_the_stacks(capsys):
    dog = _StallWatchdog(timeout=0.2, repeat=60.0)
    try:
        dog.beat("R3D inference")
        time.sleep(1.6)
    finally:
        dog.close()
    out = capsys.readouterr().out
    assert "R3D inference" in out
    assert "stalled" in out
    assert "MainThread" in out


def test_the_watchdog_stays_quiet_while_the_loop_moves(capsys):
    dog = _StallWatchdog(timeout=0.5, repeat=60.0)
    try:
        for _ in range(8):
            dog.beat("waiting for a decoded frame")
            time.sleep(0.15)
    finally:
        dog.close()
    assert "stalled" not in capsys.readouterr().out


def test_the_watchdog_reports_once_per_stall_not_once_per_second(capsys):
    dog = _StallWatchdog(timeout=0.2, repeat=600.0)
    try:
        dog.beat("action decoders")
        time.sleep(2.5)
    finally:
        dog.close()
    assert capsys.readouterr().out.count("stalled") == 1
