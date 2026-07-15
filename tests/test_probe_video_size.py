"""
Regression tests for `encoder_select.probe_video_size`.

The parser used to do `stdout.strip().split("x")` over the whole ffprobe blob.
That works for a single clean line, but some files (observed on GoPro HEVC)
make ffprobe emit the entry twice with a blank line between, even under
-select_streams v:0:

    5312x2988\r\n
    \r\n
    5312x2988\r\n

Splitting that on "x" yields ['5312', '2988\r\n\r\n5312', '2988'] and
int(parts[1]) raises. probe_video_size then reported (0, 0), which pushed
encoder_select into a bad chain and every cut_video call failed — the video
produced no highlight at all.
"""

from __future__ import annotations

import subprocess

import pytest

from modules import encoder_select


@pytest.fixture(autouse=True)
def _clear_cache():
    encoder_select._size_cache.clear()
    yield
    encoder_select._size_cache.clear()


def _fake_run(stdout: str):
    def run(*_args, **_kwargs):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
    return run


def test_single_line(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run("1920x1080\n"))
    assert encoder_select.probe_video_size("a.mp4") == (1920, 1080)


def test_duplicated_entry_with_blank_line(monkeypatch):
    """The GoPro HEVC shape that used to raise ValueError."""
    monkeypatch.setattr(subprocess, "run", _fake_run("5312x2988\r\n\r\n5312x2988\r\n"))
    assert encoder_select.probe_video_size("gopro.mp4") == (5312, 2988)


def test_leading_blank_lines(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run("\n\n1280x720\n"))
    assert encoder_select.probe_video_size("b.mp4") == (1280, 720)


def test_garbage_falls_back_to_zero(monkeypatch):
    """Unparseable output must not raise; the caller handles (0, 0) by
    assuming a normal H.264 source."""
    monkeypatch.setattr(subprocess, "run", _fake_run("N/AxN/A\n"))

    # cv2 is a MagicMock under the test shim, so its ints aren't real; force the
    # cv2 fallback to contribute nothing.
    import cv2

    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: (_ for _ in ()).throw(RuntimeError()))
    assert encoder_select.probe_video_size("c.mp4") == (0, 0)


def test_result_is_cached(monkeypatch):
    calls = {"n": 0}

    def counting_run(*_args, **_kwargs):
        calls["n"] += 1
        return subprocess.CompletedProcess([], 0, "800x600\n", "")

    monkeypatch.setattr(subprocess, "run", counting_run)
    assert encoder_select.probe_video_size("d.mp4") == (800, 600)
    assert encoder_select.probe_video_size("d.mp4") == (800, 600)
    assert calls["n"] == 1
