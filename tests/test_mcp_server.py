"""The MCP server (mcp_server/): its tool list, a few calls with the engine mocked,
and the stdout guard that keeps the stdio transport clean.

Byte-identical in both editions, like the package. The search-by-example tests
run only where the edition has ``llm/example_search.py``.

Calls go through FastMCP itself (list_tools / call_tool), so argument schemas
and JSON conversion are exercised, not just the functions behind them.
"""
from __future__ import annotations

import importlib

import asyncio
import json
import subprocess
import sys
import textwrap
import threading
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mcp")

from mcp.server.fastmcp.exceptions import ToolError  # noqa: E402

from mcp_server import tools  # noqa: E402
from mcp_server.jobs import JobManager  # noqa: E402
from mcp_server.server import build_server  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

HAS_EXAMPLE_SEARCH = tools.has_example_search()

EXPECTED_TOOLS = {
    "get_info", "probe_video", "scan_folder", "get_config",
    "start_highlight_run", "start_visual_search", "set_timeline_clips",
    "start_expression_scan",
    "get_job", "list_jobs", "cancel_job",
    "dataset_health", "plan_reviewed_video", "add_reviewed_video",
    "start_dataset_comparison", "drop_reviewed_folders", "undo_reviewed_video",
} | ({"start_example_search"} if HAS_EXAMPLE_SEARCH else set())   | ({"start_counter_check"} if tools.has_counter_check() else set())   | ({"start_counter_guided_run"} if tools.has_counter_guided() else set())

needs_example_search = pytest.mark.skipif(
    not HAS_EXAMPLE_SEARCH, reason="this edition has no search by example")


@pytest.fixture(autouse=True)
def no_live_app(monkeypatch):
    """Never reach a real running app from a test: its open timeline would
    take the test's clips."""
    pushed = []
    monkeypatch.setattr(tools, "show_live", lambda video, spans, note: pushed.append(
        (video, [list(s) for s in spans], note)) or {"live": False, "why": "test"})
    return pushed


@pytest.fixture
def jobs():
    return JobManager()


@pytest.fixture
def server(jobs):
    return build_server(jobs)


@pytest.fixture
def video(tmp_path):
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"\0" * 64)
    return str(p)


def call(server, tool, /, **args):
    content = asyncio.run(server.call_tool(tool, args))
    return json.loads(content[0].text)


def finish(jobs, job_id):
    job = jobs.wait(job_id, timeout=10)
    assert job.status in ("done", "failed", "cancelled"), job.status
    return job


# ── Listing ──────────────────────────────────────────────────────────────


def test_lists_every_tool_with_a_description(server):
    listed = asyncio.run(server.list_tools())
    assert {t.name for t in listed} == EXPECTED_TOOLS
    for t in listed:
        assert t.description and t.description.strip(), t.name


def test_tool_schemas_and_hints(server):
    by_name = {t.name: t for t in asyncio.run(server.list_tools())}
    assert by_name["start_highlight_run"].inputSchema["required"] == ["video_paths"]
    if HAS_EXAMPLE_SEARCH:
        assert set(by_name["start_example_search"].inputSchema["required"]) == {
            "video_path", "roi", "name"}
    for name in ("get_info", "probe_video", "scan_folder", "get_config",
                 "get_job", "list_jobs"):
        assert by_name[name].annotations.readOnlyHint is True, name
    for name in EXPECTED_TOOLS & {"start_highlight_run", "start_visual_search",
                                  "start_example_search", "cancel_job"}:
        assert by_name[name].annotations.readOnlyHint is False, name


def test_example_search_is_listed_only_where_the_edition_has_it(jobs, monkeypatch):
    monkeypatch.setattr(tools, "has_example_search", lambda: False)
    names = {t.name for t in asyncio.run(build_server(jobs).list_tools())}
    assert "start_example_search" not in names
    assert "start_visual_search" in names


# ── Quick tools ──────────────────────────────────────────────────────────


def test_probe_video_reports_metadata(server, video, monkeypatch):
    import modules.media.video_probe as vp
    monkeypatch.setattr(vp, "probe_video", lambda p: {
        "duration": 12.5, "width": 1920, "height": 1080, "fps": 30.0,
        "rotation": 0})
    info = call(server, "probe_video", path=video)
    assert info["duration"] == 12.5 and info["width"] == 1920
    assert info["path"] == video and info["size_bytes"] == 64


def test_missing_file_is_a_tool_error(server, tmp_path):
    with pytest.raises(ToolError, match="File not found"):
        call(server, "probe_video", path=str(tmp_path / "nope.mp4"))


def test_scan_folder_lists_videos_only(server, tmp_path):
    (tmp_path / "b.mp4").write_bytes(b"")
    (tmp_path / "a.MOV").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.mkv").write_bytes(b"")
    flat = call(server, "scan_folder", path=str(tmp_path))
    assert [Path(p).name for p in flat["videos"]] == ["a.MOV", "b.mp4"]
    deep = call(server, "scan_folder", path=str(tmp_path), recursive=True)
    assert deep["count"] == 3


# ── Highlight run ────────────────────────────────────────────────────────


def _fake_pipeline(monkeypatch, run_highlighter):
    mod = types.ModuleType("pipeline")
    mod.run_highlighter = run_highlighter
    monkeypatch.setitem(sys.modules, "pipeline", mod)


def test_highlight_run_returns_a_job_then_its_outputs(server, jobs, video, monkeypatch):
    seen = {}

    def run_highlighter(video_path, gui_config=None, log_fn=print,
                        progress_fn=None, cancel_flag=None):
        seen.update(video_path=video_path, gui_config=dict(gui_config))
        log_fn("analysing")
        progress_fn(1, 2, "Scenes", "half way")
        Path(video_path[0][:-4] + "_mcp_reel_why.json").write_text("{}")
        return [(p, p[:-4] + "_mcp_reel.mp4") for p in video_path]

    _fake_pipeline(monkeypatch, run_highlighter)
    started = call(server, "start_highlight_run", video_paths=[video],
                   length="short", output_name="mcp_reel",
                   overrides={"clip_time": 6})
    assert started["status"] in ("queued", "running")

    finish(jobs, started["job_id"])
    snap = call(server, "get_job", job_id=started["job_id"])
    assert snap["status"] == "done"
    assert snap["result"]["outputs"] == [{
        "input": video, "output": video[:-4] + "_mcp_reel.mp4", "exists": False,
        "report_json": video[:-4] + "_mcp_reel_why.json"}]
    assert "analysing" in snap["log_tail"]
    assert snap["progress"]["task"] == "Scenes"

    # The batch form even for one video, the Simple preset under the
    # overrides, and the output name handed to the pipeline's own namer.
    assert seen["video_path"] == [video]
    cfg = seen["gui_config"]
    assert cfg["max_duration"] == 90 and cfg["clip_time"] == 6
    assert cfg["output_base"] == "mcp_reel"
    assert cfg["write_highlight_report"] is True
    # The preset's None entries are dropped, as main.py drops them: handed
    # through, highlight_objects=None crashed the pipeline's list() of it.
    assert "highlight_objects" not in cfg and None not in cfg.values()


def test_saved_preset_passes_only_the_overrides(server, jobs, video, monkeypatch):
    seen = {}

    def run_highlighter(video_path, gui_config=None, **_):
        seen["cfg"] = dict(gui_config)
        return [(video_path[0], video_path[0] + ".out.mp4")]

    _fake_pipeline(monkeypatch, run_highlighter)
    started = call(server, "start_highlight_run", video_paths=[video],
                   preset="saved", overrides={"max_duration": 60})
    finish(jobs, started["job_id"])
    assert seen["cfg"] == {"max_duration": 60}


def test_a_run_that_writes_nothing_fails(server, jobs, video, monkeypatch):
    _fake_pipeline(monkeypatch, lambda video_path, **_: [(video_path[0], None)])
    started = call(server, "start_highlight_run", video_paths=[video])
    finish(jobs, started["job_id"])
    snap = call(server, "get_job", job_id=started["job_id"])
    assert snap["status"] == "failed" and "without writing" in snap["error"]


def test_jobs_run_one_at_a_time_and_a_queued_one_cancels(server, jobs, video,
                                                         monkeypatch):
    release = threading.Event()
    entered = threading.Event()

    def run_highlighter(video_path, cancel_flag=None, **_):
        entered.set()
        release.wait(5)
        return [(video_path[0], video_path[0] + ".out.mp4")]

    _fake_pipeline(monkeypatch, run_highlighter)
    first = call(server, "start_highlight_run", video_paths=[video])
    assert entered.wait(5)
    second = call(server, "start_highlight_run", video_paths=[video])
    assert call(server, "get_job", job_id=second["job_id"])["status"] == "queued"

    cancelled = call(server, "cancel_job", job_id=second["job_id"])
    assert cancelled["status"] == "cancelled"
    release.set()
    assert finish(jobs, first["job_id"]).status == "done"
    statuses = {j["job_id"]: j["status"] for j in call(server, "list_jobs")["jobs"]}
    assert statuses == {first["job_id"]: "done", second["job_id"]: "cancelled"}


def test_cancelling_a_running_job_trips_the_pipeline_flag(server, jobs, video,
                                                          monkeypatch):
    entered = threading.Event()

    def run_highlighter(video_path, cancel_flag=None, **_):
        entered.set()
        assert cancel_flag.wait(5)
        return []

    _fake_pipeline(monkeypatch, run_highlighter)
    started = call(server, "start_highlight_run", video_paths=[video])
    assert entered.wait(5)
    call(server, "cancel_job", job_id=started["job_id"])
    assert finish(jobs, started["job_id"]).status == "cancelled"


# ── Searches ─────────────────────────────────────────────────────────────


class _FakeEmbedder:
    model_id = "fake-clip"
    logit_scale = 100.0

    def set_query(self, target):
        self.query = target

    def embed_query_labels(self):
        # Row 0 the prompt, row 1 a negative.
        return np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)


def test_visual_search_ranks_frames_and_returns_numbers_only(server, jobs, video,
                                                             monkeypatch):
    import llm.clip_index as ci

    emb = _FakeEmbedder()
    monkeypatch.setitem(tools._EMBEDDER, "AUTO", emb)
    built = {}

    def load_or_build(path, cache_path, interval=1.0, **kw):
        built.update(path=path, interval=interval, embedder=kw.get("embedder"))
        vecs = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0.8, 0.6, 0, 0]],
                        dtype=np.float32)
        return ci.ClipFrameIndex(np.array([0.0, 1.0, 2.0]), vecs,
                                 {"logit_scale": 100.0})

    monkeypatch.setattr(ci, "load_or_build", load_or_build)
    monkeypatch.setattr(ci, "cache_path_for", lambda p, *a, **k: "unused.npz")

    started = call(server, "start_visual_search", video_path=video,
                   query="a red ball", top_k=2)
    finish(jobs, started["job_id"])
    snap = call(server, "get_job", job_id=started["job_id"])
    assert snap["status"] == "done", snap.get("error")
    result = snap["result"]
    assert emb.query == "a red ball"
    assert built == {"path": video, "interval": tools.INDEX_INTERVAL,
                     "embedder": emb}
    assert result["frames_indexed"] == 3
    assert [m["timestamp"] for m in result["matches"]] == [1.0, 2.0]
    for m in result["matches"]:
        assert set(m) == {"timestamp", "score", "margin"}
        assert 0.0 <= m["score"] <= 1.0


def _fake_face_scan(monkeypatch, seconds, seen=None):
    from modules.vision import face_emotions, face_scan
    import modules.media.video_probe as vp

    monkeypatch.setattr(face_emotions.EmotionClassifier, "available",
                        lambda self: True)
    monkeypatch.setattr(vp, "probe_video", lambda p: {"duration": 60.0})

    def scan_video(path, **kw):
        if seen is not None:
            seen.update(path=path, use_cache=kw.get("use_cache"))
        return seconds

    monkeypatch.setattr(face_scan, "scan_video", scan_video)


def test_expression_scan_returns_counts_and_ranges(server, jobs, video,
                                                   monkeypatch):
    seconds = {
        3: {"label": "happy", "confidence": 0.9, "faces": 1},
        4: {"label": "happy", "confidence": 0.8, "faces": 1},
        5: {"label": "neutral", "confidence": 0.7, "faces": 1},
        30: {"label": "happy", "confidence": 0.6, "faces": 2},
        40: {"label": "surprise", "confidence": 0.95, "faces": 1},
    }
    seen = {}
    _fake_face_scan(monkeypatch, seconds, seen)

    started = call(server, "start_expression_scan", video_path=video)
    finish(jobs, started["job_id"])
    snap = call(server, "get_job", job_id=started["job_id"])
    assert snap["status"] == "done", snap.get("error")
    result = snap["result"]
    assert seen == {"path": video, "use_cache": True}
    assert result["seconds_with_a_readable_face"] == 5
    assert result["seconds_by_expression"]["happy"] == 3
    # Neutral is counted, and left out of the ranges unless asked for.
    assert set(result["ranges"]) == {"happy", "surprise"}
    assert result["ranges"]["happy"] == [{"start": 2.5, "end": 5.5},
                                         {"start": 29.5, "end": 31.5}]

    # A confidence floor and a chosen label narrow the ranges, not the counts.
    started = call(server, "start_expression_scan", video_path=video,
                   labels=["Happy"], min_confidence=0.7, rescan=True)
    finish(jobs, started["job_id"])
    result = call(server, "get_job", job_id=started["job_id"])["result"]
    assert seen["use_cache"] is False
    assert result["ranges"] == {"happy": [{"start": 2.5, "end": 5.5}]}
    assert result["seconds_by_expression"]["happy"] == 3


def test_expression_scan_refuses_a_class_the_model_does_not_have(server, video):
    with pytest.raises(ToolError, match="Unknown expression"):
        call(server, "start_expression_scan", video_path=video, labels=["bored"])


def test_expression_scan_fails_plainly_without_the_model(server, jobs, video,
                                                         monkeypatch):
    from modules.vision import face_emotions
    monkeypatch.setattr(face_emotions.EmotionClassifier, "available",
                        lambda self: False)
    started = call(server, "start_expression_scan", video_path=video)
    job = finish(jobs, started["job_id"])
    assert job.status == "failed" and "not installed" in job.error


@needs_example_search
def test_example_search_passes_the_region_and_reports_matches(server, jobs, video,
                                                              monkeypatch):
    es = importlib.import_module("llm.example_search")

    monkeypatch.setitem(tools._EMBEDDER, "AUTO", _FakeEmbedder())
    seen = {}

    def search_region_in_video(path, roi, name, **kw):
        seen.update(path=path, roi=roi, name=name, **kw)
        findings = [{"timestamp": 42.0, "query": name, "confidence": 0.91,
                     "model": "fake-clip", "scan_id": kw["scan_id"]}]
        return findings, object(), [0] * 10

    monkeypatch.setattr(es, "search_region_in_video", search_region_in_video)
    monkeypatch.chdir(Path(video).parent)

    started = call(server, "start_example_search", video_path=video,
                   roi=[0.1, 0.2, 0.3, 0.4], name="my thing",
                   example_timestamps=[12.0])
    finish(jobs, started["job_id"])
    snap = call(server, "get_job", job_id=started["job_id"])
    assert snap["status"] == "done", snap.get("error")
    assert snap["result"]["matches"] == [{"timestamp": 42.0, "score": 0.91}]
    assert snap["result"]["frames_indexed"] == 10
    assert seen["roi"] == (0.1, 0.2, 0.3, 0.4)
    assert seen["example_timestamps"] == (12.0,)
    assert seen["threshold"] == es.DEFAULT_THRESHOLD
    assert Path(seen["cache_path"]).name == "cropidx_clip.mp4_my_thing.npz"


@needs_example_search
@pytest.mark.parametrize("roi", [[0.5, 0.5, 0.6, 0.1], [0, 0, 0, 0.5], [0.1, 0.2]])
def test_example_search_rejects_a_region_outside_the_frame(server, video, roi):
    with pytest.raises(ToolError, match="roi"):
        call(server, "start_example_search", video_path=video, roi=roi, name="x")


needs_counter_check = pytest.mark.skipif(
    not tools.has_counter_check(), reason="this edition has no counter check")


@needs_counter_check
def test_counter_check_reports_ticks_values_and_coverage(server, jobs, video,
                                                         monkeypatch, tmp_path):
    cd = importlib.import_module("modules.vision.counter_detect")

    seen = {}

    def evaluate_counter(path, roi, clips, **kw):
        seen.update(path=path, roi=roi, clips=list(clips), **kw)
        kw["progress_cb"](0.5)
        return cd.CounterReport(
            increments=[100.0, 200.0],
            match=cd.MatchReport(total=2, hits=[100.0], misses=[200.0]),
            suggested_spans=[(195.0, 201.5)], sample_count=40, threshold=1.5,
            values=[1, 2, 3], missing_values=[3], values_confident=True,
            state_previews=[object()])

    monkeypatch.setattr(cd, "evaluate_counter", evaluate_counter)
    report = tmp_path / "r_why.json"
    report.write_text(json.dumps({"segments": [{"start": 95, "end": 105}]}))

    started = call(server, "start_counter_check", video_path=video,
                   roi=[0.0, 0.8, 0.1, 0.15], report_json=str(report))
    finish(jobs, started["job_id"])
    snap = call(server, "get_job", job_id=started["job_id"])
    assert snap["status"] == "done", snap.get("error")
    result = snap["result"]
    assert result["ticks"] == [{"time": 100.0, "covered": True, "value": 2},
                               {"time": 200.0, "covered": False, "value": 3}]
    assert result["first_value"] == 1 and result["recall"] == 0.5
    assert result["suggested_spans"] == [[195.0, 201.5]]
    assert "state_previews" not in json.dumps(result)     # no pictures back
    assert seen["clips"] == [(95.0, 105.0)] and seen["sensitivity"] == "medium"


@pytest.mark.skipif(not tools.has_counter_guided(),
                    reason="this edition has no counter-guided runs")
def test_counter_guided_run_returns_one_event_per_value(server, jobs, video,
                                                        monkeypatch):
    cg = importlib.import_module("modules.segments.counter_guided")

    seen = {}

    def run(path, roi, classes, output, **kw):
        seen.update(path=path, roi=roi, classes=classes, output=output, **kw)
        return {"output": output, "clips_dir": output[:-4] + "_clips",
                "events_json": output[:-4] + "_events.json", "values_read": True,
                "timeline": True,
                "events": [{"value": 1, "appeared": 10.0, "search": [0.0, 70.0],
                            "start": 30.0, "end": 42.0, "peak": 36.0, "score": 0.7,
                            "cls": "a", "clip": "c.mp4"}]}

    monkeypatch.setattr(cg, "run", run)
    started = call(server, "start_counter_guided_run", video_path=video,
                   roi=[0.0, 0.8, 0.1, 0.15], classes=["a", "b"], before_peak=20)
    finish(jobs, started["job_id"])
    snap = call(server, "get_job", job_id=started["job_id"])
    assert snap["status"] == "done", snap.get("error")
    assert snap["result"]["events"] == [{
        "value": 1, "appeared": 10.0, "search": [0.0, 70.0], "start": 30.0,
        "end": 42.0, "peak": 36.0, "score": 0.7, "cls": "a"}]
    assert snap["result"]["in_timeline_viewer"] is True
    assert seen["classes"] == ["a", "b"]
    assert (seen["before_peak"], seen["after_peak"]) == (20.0, 5.0)
    assert seen["output"].endswith("clip_counted.mp4")
    with pytest.raises(ToolError, match="classes"):
        call(server, "start_counter_guided_run", video_path=video,
             roi=[0.0, 0.8, 0.1, 0.15], classes=[])


def test_set_timeline_clips_becomes_the_newest_highlight_version(server, video,
                                                                 monkeypatch,
                                                                 tmp_path):
    """What the edit timeline opens on when no run has just finished."""
    import modules.media.video_probe as vp
    from modules.media.video_cache import VideoAnalysisCache

    monkeypatch.setattr(vp, "probe_video", lambda p: {"duration": 100.0})
    monkeypatch.chdir(tmp_path)
    result = call(server, "set_timeline_clips", video_path=video,
                  clips=[[50, 60], [10, 22.5], [95, 140]], note="fixed by hand")
    assert result["clips"] == [[10.0, 22.5], [50.0, 60.0], [95.0, 100.0]]
    assert result["total_seconds"] == 27.5
    assert result["live"] == {"live": False, "why": "test"}
    history = VideoAnalysisCache(cache_dir="./cache").get_highlight_history(video)
    assert [list(s) for s in history[0]["segments"]] == result["clips"]
    with pytest.raises(ToolError, match="empty or outside"):
        call(server, "set_timeline_clips", video_path=video, clips=[[120, 130]])


def test_show_live_reports_whether_the_open_app_took_the_clips(monkeypatch):
    import importlib.util

    # The autouse stub replaced tools.show_live; put the real one back.
    monkeypatch.undo()
    if importlib.util.find_spec("modules.system.live_link") is None:
        assert tools.show_live("v.mp4", [(1, 2)], "n") == {
            "live": False, "why": "this edition has no live link"}
        return
    live_link = importlib.import_module("modules.system.live_link")

    sent = []
    monkeypatch.setattr(live_link, "send", lambda cmd, args, **kw: sent.append(
        (cmd, args)) or {"shown": 2, "opened_viewer": True})
    assert tools.show_live("v.mp4", [(1, 2), (3, 4)], "fixed") == {
        "live": True, "shown": 2, "opened_viewer": True}
    assert sent == [("show_clips", {"video_path": "v.mp4",
                                    "clips": [[1, 2], [3, 4]], "note": "fixed"})]
    monkeypatch.setattr(live_link, "send", lambda *a, **kw: None)
    assert tools.show_live("v.mp4", [(1, 2)], "n")["why"] == "the app is not running"

    def refuse(*a, **kw):
        raise live_link.LiveLinkError("Video not found")
    monkeypatch.setattr(live_link, "send", refuse)
    assert tools.show_live("v.mp4", [(1, 2)], "n") == {
        "live": False, "why": "Video not found"}


def test_unknown_job_is_a_tool_error(server):
    with pytest.raises(ToolError, match="No such job"):
        call(server, "get_job", job_id="nope")


# ── stdout guard ─────────────────────────────────────────────────────────


def test_only_the_protocol_reaches_stdout():
    """print(), a raw fd-1 write and a child process all land on stderr once
    stdout is claimed; only what is written to the transport reaches stdout."""
    script = textwrap.dedent("""
        import os, subprocess, sys
        sys.path.insert(0, %r)
        from mcp_server.stdio_guard import claim_stdout
        out = claim_stdout()
        print("noise-print")
        os.write(1, b"noise-fd\\n")
        subprocess.run([sys.executable, "-c", "print('noise-child')"])
        out.write(b"PROTO\\n")
    """ % str(REPO))
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                          timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.replace(b"\r\n", b"\n") == b"PROTO\n"
    for noise in (b"noise-print", b"noise-fd", b"noise-child"):
        assert noise in proc.stderr
