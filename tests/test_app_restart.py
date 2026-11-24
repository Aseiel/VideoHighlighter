"""`modules.system.app_restart`: what a restart refuses, runs and carries over."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import types

from modules.system import app_restart as ar


class Job:
    def __init__(self, running):
        self.running = running

    def isRunning(self):
        return self.running


def test_busy_names_only_the_work_that_is_running():
    main = types.SimpleNamespace(worker=Job(True), _signal_worker=Job(False),
                                 download_worker=None)
    viewer = types.SimpleNamespace(_analysis_thread=None)
    assert ar.busy([(main, ar.MAIN_WORK), (viewer, ar.VIEWER_WORK)]) == ["a highlight run"]


def test_busy_sees_a_plain_thread_and_skips_a_missing_viewer():
    gate = threading.Event()
    t = threading.Thread(target=gate.wait)
    t.start()
    try:
        viewer = types.SimpleNamespace(_analysis_thread=t)
        assert ar.busy([(None, ar.MAIN_WORK), (viewer, ar.VIEWER_WORK)]) == [
            "an analysis in the timeline viewer"]
    finally:
        gate.set()
        t.join()


def test_a_deleted_qt_object_is_not_busy():
    class Gone:
        def isRunning(self):
            raise RuntimeError("Internal C++ object already deleted.")
    assert ar.busy([(types.SimpleNamespace(worker=Gone()), ar.MAIN_WORK)]) == []


def test_relaunch_command_from_source_and_frozen():
    assert ar.relaunch_command(["main.py", "-x"], "py.exe", frozen=False) == ["py.exe", "main.py", "-x"]
    assert ar.relaunch_command(["C:/app/VH.exe", "-x"], "C:/app/VH.exe", frozen=True) == [
        "C:/app/VH.exe", "-x"]


def test_spawn_passes_what_to_reopen_and_detaches(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(ar.ENV_VIDEO, "stale.mp4")
    seen = {}
    ar.spawn("v.mp4", 1428.5, command=["app"], popen=lambda cmd, **kw: seen.update(cmd=cmd, **kw))
    assert seen["cmd"] == ["app"] and seen["cwd"] == os.getcwd()
    assert seen["env"][ar.ENV_VIDEO] == "v.mp4" and seen["env"][ar.ENV_SECONDS] == "1428.500"
    if sys.platform == "win32":
        # A hidden console of its own, never none: without one, every console
        # program the app starts opens a window of its own.
        assert seen["creationflags"] & subprocess.CREATE_NO_WINDOW
        assert not seen["creationflags"] & subprocess.DETACHED_PROCESS
    else:
        assert seen["start_new_session"]


def test_spawn_without_a_video_reopens_nothing(monkeypatch):
    monkeypatch.setenv(ar.ENV_VIDEO, "stale.mp4")
    monkeypatch.setenv(ar.ENV_SECONDS, "5")
    seen = {}
    ar.spawn(None, None, command=["app"], popen=lambda cmd, **kw: seen.update(kw))
    assert ar.ENV_VIDEO not in seen["env"] and ar.ENV_SECONDS not in seen["env"]


def test_take_reopen_reads_once(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    env = {ar.ENV_VIDEO: str(video), ar.ENV_SECONDS: "12.5"}
    assert ar.take_reopen(env) == (str(video), 12.5)
    assert ar.take_reopen(env) is None


def test_take_reopen_ignores_a_missing_file_and_a_bad_time(tmp_path):
    assert ar.take_reopen({ar.ENV_VIDEO: str(tmp_path / "gone.mp4")}) is None
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    assert ar.take_reopen({ar.ENV_VIDEO: str(video), ar.ENV_SECONDS: "soon"}) == (str(video), None)


def test_where_the_windows_were_travels_with_the_restart(monkeypatch):
    monkeypatch.setenv(ar.ENV_GEOMETRY, "stale")
    seen = {}
    ar.spawn(None, None, geometry={"main": "AAA=", "viewer": None},
             command=["app"], popen=lambda cmd, **kw: seen.update(kw))
    env = seen["env"]
    assert ar.take_geometry(env) == {"main": "AAA="}
    assert ar.take_geometry(env) == {}                      # read once


def test_no_geometry_and_bad_geometry_are_nothing(monkeypatch):
    seen = {}
    monkeypatch.delenv(ar.ENV_GEOMETRY, raising=False)
    ar.spawn(None, None, command=["app"], popen=lambda cmd, **kw: seen.update(kw))
    assert ar.ENV_GEOMETRY not in seen["env"]
    assert ar.take_geometry({ar.ENV_GEOMETRY: "not json"}) == {}
    assert ar.take_geometry({ar.ENV_GEOMETRY: "[1, 2]"}) == {}
