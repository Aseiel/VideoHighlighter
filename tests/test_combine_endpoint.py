"""
Validation tests for sidecar.server's POST /combine.

The contract that matters: validation runs BEFORE any job starts, so a bad
request (fewer than 2 files, or a file that doesn't exist, or no output path)
returns {ok:false,error} without ever touching the single run slot. If a bad
request slipped through to manager.start_job it would spawn a child process and
block the next legitimate run.

The env's fastapi TestClient is broken (httpx BaseTransport incompatibility), so
these drive the endpoint coroutine directly with asyncio.run and assert on
manager.start_job NOT being called for the rejection paths.
"""

from __future__ import annotations

import asyncio

import sidecar.server as server
from sidecar.server import CombineRequest, start_combine


def _run(req: CombineRequest) -> dict:
    return asyncio.run(start_combine(req))


def _guard_start_job(monkeypatch):
    """Make manager.start_job explode: any call means validation let a bad
    request through to the run slot."""
    def _boom(*_a, **_k):
        raise AssertionError("start_job must not run for an invalid request")

    monkeypatch.setattr(server.manager, "start_job", _boom)


def test_fewer_than_two_files_rejected(monkeypatch, tmp_path):
    _guard_start_job(monkeypatch)
    only = tmp_path / "one.mp4"
    only.write_text("x")
    res = _run(CombineRequest(files=[str(only)], output=str(tmp_path / "out.mp4")))
    assert res["ok"] is False
    assert "2 files" in res["error"]


def test_missing_file_rejected(monkeypatch, tmp_path):
    _guard_start_job(monkeypatch)
    real = tmp_path / "real.mp4"
    real.write_text("x")
    ghost = tmp_path / "ghost.mp4"  # never created
    res = _run(CombineRequest(files=[str(real), str(ghost)],
                              output=str(tmp_path / "out.mp4")))
    assert res["ok"] is False
    assert "not found" in res["error"]
    assert "ghost.mp4" in res["error"]


def test_empty_output_rejected(monkeypatch, tmp_path):
    _guard_start_job(monkeypatch)
    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_text("x")
    b.write_text("x")
    res = _run(CombineRequest(files=[str(a), str(b)], output="   "))
    assert res["ok"] is False
    assert "output" in res["error"].lower()


def test_valid_request_reaches_start_job(monkeypatch, tmp_path):
    """The happy path: two real files + an output must build the job and call
    start_job. We stub start_job to capture the job dict rather than spawn a
    real child process."""
    captured = {}

    def _fake_start_job(job, _loop):
        captured.update(job)
        return "run-xyz"

    monkeypatch.setattr(server.manager, "start_job", _fake_start_job)

    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    a.write_text("x")
    b.write_text("x")
    music = tmp_path / "song.mp3"
    music.write_text("x")

    res = _run(CombineRequest(
        files=[str(a), str(b)], output=str(tmp_path / "reel.mp4"),
        music_path=str(music), music_mode="mix", music_volume=0.5))

    assert res == {"ok": True, "run_id": "run-xyz"}
    assert captured["kind"] == "combine"
    assert captured["files"] == [str(a), str(b)]
    assert captured["music_path"] == str(music)
    assert captured["music_mode"] == "mix"
    assert captured["music_volume"] == 0.5
