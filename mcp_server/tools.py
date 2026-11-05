"""What each MCP tool does, as plain functions — no MCP types in here.

``server.py`` registers these with FastMCP; tests call them through it with the
engine mocked. Engine imports stay inside the functions: importing ``pipeline``
pulls in torch, OpenVINO and Whisper, and ``list_tools`` should not wait on that.

Every result is JSON-able and carries paths, timestamps, scores and text only.
"""
from __future__ import annotations

import os
from typing import Any, Optional, Sequence

from mcp_server.jobs import Job, JobCancelled, JobManager

LENGTHS = ("short", "medium", "long")
PRESETS = ("simple", "saved")


def _existing_file(path: str) -> str:
    path = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isfile(path):
        raise ValueError(f"File not found: {path}")
    return path


# ── Quick, read-only ─────────────────────────────────────────────────────


def get_info(jobs: JobManager) -> dict:
    from modules.system.app_paths import config_path, user_data_dir
    try:
        from version import __edition__, __version__
    except Exception:
        __version__, __edition__ = "unknown", "unknown"
    current = jobs.current
    return {
        "app": "VideoHighlighter",
        "version": __version__,
        "edition": __edition__,
        "data_dir": user_data_dir(),
        "config_path": config_path("config.yaml"),
        "working_dir": os.getcwd(),
        "running_job": current.id if current else None,
        "queued_jobs": [j.id for j in jobs.all() if j.status == "queued"],
    }


def probe_video(path: str) -> dict:
    from modules.media.video_probe import probe_video as _probe

    path = _existing_file(path)
    info = dict(_probe(path))
    info["path"] = path
    info["size_bytes"] = os.path.getsize(path)
    return info


def scan_folder(path: str, recursive: bool = False, limit: int = 1000) -> dict:
    from modules.segments.simple_run import is_video_path

    root = os.path.abspath(os.path.expanduser(str(path)))
    if not os.path.isdir(root):
        raise ValueError(f"Folder not found: {root}")
    found: list[str] = []
    truncated = False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            if is_video_path(name):
                if len(found) >= limit:
                    truncated = True
                    break
                found.append(os.path.join(dirpath, name))
        if truncated or not recursive:
            break
    return {"folder": root, "videos": found, "count": len(found),
            "truncated": truncated}


def get_config() -> dict:
    import yaml
    from modules.system.app_paths import config_path

    path = config_path("config.yaml")
    if not os.path.exists(path):
        return {"path": path, "exists": False, "config": {}}
    with open(path, "r", encoding="utf-8") as f:
        return {"path": path, "exists": True, "config": yaml.safe_load(f) or {}}


# ── Jobs ─────────────────────────────────────────────────────────────────


def start_highlight_run(jobs: JobManager, video_paths: Sequence[str],
                        preset: str = "simple", length: str = "medium",
                        output_name: Optional[str] = None,
                        overrides: Optional[dict] = None) -> dict:
    paths = [_existing_file(p) for p in (video_paths or [])]
    if not paths:
        raise ValueError("No videos given.")
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {PRESETS}")
    if length not in LENGTHS:
        raise ValueError(f"length must be one of {LENGTHS}")

    # The pipeline reads config.yaml itself; gui_config holds only what this run
    # changes, exactly as the GUI's Worker hands it over.
    gui_config: dict = {}
    if preset == "simple":
        from modules.segments.simple_run import apply_simple_run
        apply_simple_run(gui_config, length)
    gui_config.update(overrides or {})
    if output_name:
        gui_config["output_base"] = output_name
    # As main.py does before every run: a None means "not set here, use
    # config.yaml", and the pipeline's .get() would hand the None through.
    gui_config = {k: v for k, v in gui_config.items() if v is not None}

    def run(job: Job) -> dict:
        from pipeline import run_highlighter

        # Always the batch form, even for one video: it names each output
        # beside its source (<stem>_highlight.mp4, or output_name), where the
        # single-video form would fall back to a relative highlight.mp4.
        out = run_highlighter(
            paths,
            gui_config=gui_config,
            log_fn=job.log,
            progress_fn=lambda cur, tot, task="", det="":
                job.set_progress(cur, tot, task, det),
            cancel_flag=job.cancel_flag,
        )
        outputs = []
        for src, dst in (out or []):
            entry = {"input": src, "output": dst or None,
                     "exists": bool(dst) and os.path.exists(dst)}
            # The run's own explanation (pipeline writes <output>_why.*): why
            # each moment was kept, and what scored well but was left out.
            if dst:
                for key, suffix in (("report_json", "_why.json"),
                                    ("report_html", "_why.html")):
                    path = os.path.splitext(dst)[0] + suffix
                    if os.path.exists(path):
                        entry[key] = path
            outputs.append(entry)
        if not job.cancel_flag.is_set() and not any(o["output"] for o in outputs):
            raise RuntimeError("The run finished without writing a highlight; "
                               "see log_tail for why.")
        return {"outputs": outputs}

    job = jobs.submit("highlight_run", {
        "video_paths": paths, "preset": preset, "length": length,
        "output_name": output_name, "overrides": overrides or {},
    }, run)
    return {"job_id": job.id, "status": job.status}


# One embedder for the server's life: loading CLIP costs seconds, and the job
# thread is the only user (jobs run one at a time).
_EMBEDDER: dict[str, Any] = {}


def _embedder(device: str = "AUTO"):
    if device not in _EMBEDDER:
        from llm.clip_index import ClipEmbedder
        emb = ClipEmbedder(device=device)
        emb.load()
        _EMBEDDER[device] = emb
    return _EMBEDDER[device]


def _progress(job: Job, task: str):
    """A (done, total) callback that also carries cancellation into index
    builds, which have no cancel flag of their own."""
    def cb(done, total):
        job.set_progress(done, total, task)
        job.check_cancel()
    return cb


# The frame index is shared with chapters and composition signals, and
# load_or_build replaces it whenever the interval differs, so searches keep the
# interval everything else builds it at rather than offering their own.
INDEX_INTERVAL = 1.0


def start_visual_search(jobs: JobManager, video_path: str, query: str,
                        top_k: int = 20) -> dict:
    """Text search over the video's whole-frame index (./cache/clip/)."""
    path = _existing_file(video_path)
    query = (query or "").strip()
    if not query:
        raise ValueError("query is empty")
    top_k = max(1, min(int(top_k), 500))
    interval = INDEX_INTERVAL

    def run(job: Job) -> dict:
        from llm.clip_index import (cache_path_for, load_or_build, log_odds,
                                    score_embeddings)

        emb = _embedder()
        job.log(f"🔎 Indexing {os.path.basename(path)} every {interval:g}s")
        idx = load_or_build(path, cache_path_for(path), interval=interval,
                            embedder=emb, model_id=emb.model_id,
                            progress=_progress(job, "Indexing frames"))
        job.check_cancel()
        if len(idx) == 0:
            return {"query": query, "frames_indexed": 0, "matches": []}
        emb.set_query(query)
        labels = emb.embed_query_labels()
        scale = idx.meta.get("logit_scale") or emb.logit_scale
        probs = score_embeddings(idx.embeddings, labels, scale)
        # Rank by margin: at a high logit scale many good frames round to a
        # probability of 1.0, and the margin still orders them.
        margins = log_odds(idx.embeddings, labels, scale)
        order = sorted(range(len(idx)), key=lambda i: -margins[i])[:top_k]
        return {
            "query": query,
            "frames_indexed": len(idx),
            "matches": [{"timestamp": round(float(idx.timestamps[i]), 3),
                         "score": round(float(probs[i]), 4),
                         "margin": round(float(margins[i]), 3)} for i in order],
        }

    job = jobs.submit("visual_search", {
        "video_path": path, "query": query, "top_k": top_k,
    }, run)
    return {"job_id": job.id, "status": job.status}


def _roi(roi: Sequence[float]) -> tuple:
    vals = tuple(float(v) for v in (roi or ()))
    if len(vals) != 4:
        raise ValueError("roi must be [x, y, width, height], each 0..1")
    x, y, w, h = vals
    if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1
            and x + w <= 1.0001 and y + h <= 1.0001):
        raise ValueError("roi must lie inside the frame: [x, y, width, height], each 0..1")
    return vals


def _edition_module(name: str):
    """Import a module that only some editions ship, by name.

    The package is the same in every edition, and the tools that need these
    modules are listed only where they exist (``has_example_search`` and the
    rest). A static import would be a promise this file cannot keep in an
    edition without the module, and the local-import check is right to fail
    it. Raises ImportError where the module is absent, as the import did.
    """
    import importlib
    return importlib.import_module(name)


def has_example_search() -> bool:
    """True where this edition ships search by example (llm/example_search.py).
    Looked up, not imported, so listing tools never loads CLIP."""
    import importlib.util
    try:
        return importlib.util.find_spec("llm.example_search") is not None
    except (ImportError, ValueError):
        return False


def start_example_search(jobs: JobManager, video_path: str, roi: Sequence[float],
                         name: str, example_timestamps: Sequence[float] = (),
                         interval: float = 1.5, threshold: Optional[float] = None,
                         top_k: int = 100) -> dict:
    """Find more like this: the region `roi` at `example_timestamps` is the
    example; every other sampled frame's same region is scored against it."""
    path = _existing_file(video_path)
    box = _roi(roi)
    name = (name or "").strip()
    if not name:
        raise ValueError("name is empty")
    examples = tuple(float(t) for t in (example_timestamps or ()))
    interval = float(interval)
    if interval <= 0:
        raise ValueError("interval must be positive")
    top_k = max(1, min(int(top_k), 1000))

    def run(job: Job) -> dict:
        example_search = _edition_module("llm.example_search")

        if threshold is None:
            thr = example_search.DEFAULT_THRESHOLD
        else:
            thr = float(threshold)
        # Same cache file the GUI's region search uses for this video and name.
        os.makedirs("./cache", exist_ok=True)
        safe = "".join(c if c.isalnum() else "_" for c in name)
        cache_path = os.path.join(
            "./cache", f"cropidx_{os.path.basename(path)}_{safe}.npz")
        findings, _category, index = example_search.search_region_in_video(
            path, box, name, example_timestamps=examples, interval=interval,
            threshold=thr, top_k=top_k, embedder=_embedder(),
            cache_path=cache_path, scan_id=f"region_{name}",
            progress=_progress(job, "Indexing region"))
        return {
            "name": name,
            "roi": list(box),
            "example_timestamps": list(examples),
            "threshold": thr,
            "frames_indexed": len(index),
            "matches": [{"timestamp": round(f["timestamp"], 3),
                         "score": round(f["confidence"], 4)} for f in findings],
        }

    job = jobs.submit("example_search", {
        "video_path": path, "roi": list(box), "name": name,
        "example_timestamps": list(examples), "interval": interval,
        "threshold": threshold, "top_k": top_k,
    }, run)
    return {"job_id": job.id, "status": job.status}


def has_counter_check() -> bool:
    """True where this edition ships the on-screen counter check
    (modules/vision/counter_detect.py)."""
    import importlib.util
    try:
        return importlib.util.find_spec("modules.vision.counter_detect") is not None
    except (ImportError, ValueError):
        return False


SENSITIVITIES = ("low", "medium", "high")


def _clips_from_report(report_json: str) -> list:
    import json

    with open(_existing_file(report_json), encoding="utf-8") as fh:
        report = json.load(fh)
    return [(float(s["start"]), float(s["end"]))
            for s in report.get("segments") or []]


def start_counter_check(jobs: JobManager, video_path: str, roi: Sequence[float],
                        clips: Optional[Sequence[Sequence[float]]] = None,
                        report_json: Optional[str] = None,
                        sensitivity: str = "medium") -> dict:
    """Read an on-screen tally in `roi`: every time it goes up, an event
    provably happened. Reports each tick's time and which ones the given clips
    (or a run's report) cover."""
    path = _existing_file(video_path)
    box = _roi(roi)
    if sensitivity not in SENSITIVITIES:
        raise ValueError(f"sensitivity must be one of {SENSITIVITIES}")
    if clips is not None:
        spans = [(float(a), float(b)) for a, b in clips]
    elif report_json:
        spans = _clips_from_report(report_json)
    else:
        spans = []

    def run(job: Job) -> dict:
        evaluate_counter = _edition_module("modules.vision.counter_detect").evaluate_counter

        def progress(fraction):
            job.set_progress(round(fraction * 100), 100, "Reading the counter")
            job.check_cancel()

        job.log(f"🔢 Reading the counter in {list(box)} against {len(spans)} clip(s)")
        rep = evaluate_counter(path, box, spans, sensitivity=sensitivity,
                               progress_cb=progress)
        values = rep.values if rep.values_confident else []
        ticks = []
        for i, t in enumerate(rep.increments):
            tick = {"time": round(float(t), 2),
                    "covered": any(abs(float(t) - h) < 1e-6 for h in rep.match.hits)}
            # values[0] is the state before the first tick.
            if values and i + 1 < len(values):
                tick["value"] = values[i + 1]
            ticks.append(tick)
        return {
            "roi": list(box),
            "ticks": ticks,
            "covered": len(rep.match.hits),
            "missed": len(rep.match.misses),
            "recall": round(rep.match.recall, 3),
            "values_read": rep.values_confident,
            "first_value": values[0] if values else None,
            "suggested_spans": [[round(a, 2), round(b, 2)]
                                for a, b in rep.suggested_spans],
            "samples": rep.sample_count,
        }

    job = jobs.submit("counter_check", {
        "video_path": path, "roi": list(box), "clips": len(spans),
        "report_json": report_json, "sensitivity": sensitivity,
    }, run)
    return {"job_id": job.id, "status": job.status}


def has_counter_guided() -> bool:
    """True where this edition ships counter-guided runs
    (modules/segments/counter_guided.py)."""
    import importlib.util
    try:
        return importlib.util.find_spec("modules.segments.counter_guided") is not None
    except (ImportError, ValueError):
        return False


def start_counter_guided_run(jobs: JobManager, video_path: str, roi: Sequence[float],
                             classes: Sequence[str], output_name: Optional[str] = None,
                             before_peak: float = 15.0, after_peak: float = 5.0,
                             search_before: float = 10.0, search_after: float = 60.0,
                             sensitivity: str = "medium") -> dict:
    """One clip per value of an on-screen counter, placed where the action
    model scores `classes` highest near that value."""
    path = _existing_file(video_path)
    box = _roi(roi)
    classes = [str(c).strip() for c in (classes or []) if str(c).strip()]
    if not classes:
        raise ValueError("classes is empty")
    if sensitivity not in SENSITIVITIES:
        raise ValueError(f"sensitivity must be one of {SENSITIVITIES}")
    before_peak, after_peak = float(before_peak), float(after_peak)
    if not (0 <= before_peak <= 120 and 0 <= after_peak <= 120
            and before_peak + after_peak >= 1):
        raise ValueError("before_peak and after_peak must be 0..120 seconds "
                         "and add up to at least 1")
    stem = (output_name or "").strip() or (
        os.path.splitext(os.path.basename(path))[0] + "_counted")
    stem = os.path.splitext(os.path.basename(stem))[0]
    output = os.path.join(os.path.dirname(path), stem + ".mp4")

    def run(job: Job) -> dict:
        counter_guided = _edition_module("modules.segments.counter_guided")

        try:
            record = counter_guided.run(
                path, box, classes, output, sensitivity=sensitivity,
                before_peak=before_peak, after_peak=after_peak,
                before=float(search_before),
                after=float(search_after), log=job.log,
                progress=lambda cur, tot, task: job.set_progress(cur, tot, task),
                cancel_flag=job.cancel_flag)
        except InterruptedError:
            raise JobCancelled()
        spans = [(e["start"], e["end"]) for e in record["events"]
                 if e.get("start") is not None]
        live = (show_live(path, spans, "counter-guided run") if spans
                else {"live": False, "why": "no clips"})
        return {
            "live": live,
            "output": record["output"],
            "clips_dir": record["clips_dir"],
            "events_json": record["events_json"],
            "values_read": record["values_read"],
            # Saved as the video's newest highlight version: the timeline
            # viewer opens on these clips.
            "in_timeline_viewer": bool(record.get("timeline")),
            "events": [{k: e.get(k) for k in
                        ("value", "appeared", "search", "start", "end", "peak",
                         "score", "cls")}
                       for e in record["events"]],
        }

    job = jobs.submit("counter_guided_run", {
        "video_path": path, "roi": list(box), "classes": classes,
        "output": output, "before_peak": before_peak, "after_peak": after_peak,
        "search": [search_before, search_after], "sensitivity": sensitivity,
    }, run)
    return {"job_id": job.id, "status": job.status}


def show_live(video_path: str, spans: Sequence[Sequence[float]], note: str) -> dict:
    """Put clips on the running app's edit timeline, if an app is listening.

    Where the edition has a live link (modules/system/live_link.py) and the
    app is open, the timeline viewer opens on the video (or is already open)
    and its edit timeline takes the clips at once, undoable there. Otherwise
    nothing happens and the reason is returned: the clips are on disk either
    way, so the next open shows them."""
    try:
        live_link = _edition_module("modules.system.live_link")
    except ImportError:
        return {"live": False, "why": "this edition has no live link"}
    try:
        result = live_link.send("show_clips", {
            "video_path": video_path, "clips": [list(s) for s in spans], "note": note})
    except live_link.LiveLinkError as e:
        return {"live": False, "why": str(e)}
    if result is None:
        return {"live": False, "why": "the app is not running"}
    return {"live": True, **result}


def set_timeline_clips(video_path: str, clips: Sequence[Sequence[float]],
                       note: Optional[str] = None) -> dict:
    """Save `clips` as the video's newest highlight version.

    The timeline viewer's edit timeline opens on the newest version when no run
    has just finished, so opening this video there shows exactly these clips,
    to adjust and export in the app. A run started afterwards replaces them."""
    from modules.media.video_cache import VideoAnalysisCache
    from modules.media.video_probe import probe_video

    path = _existing_file(video_path)
    duration = float(probe_video(path).get("duration") or 0)
    spans = []
    for clip in clips or []:
        start, end = (float(v) for v in clip)
        if duration:
            start, end = max(0.0, start), min(duration, end)
        if end - start < 0.5:
            raise ValueError(f"Clip {list(clip)} is empty or outside the video "
                             f"(0-{duration:.1f} s)")
        spans.append((round(start, 2), round(end, 2)))
    if not spans:
        raise ValueError("No clips given.")
    spans.sort()
    reason = (note or "set over MCP").strip()
    saved = VideoAnalysisCache(cache_dir="./cache").save_highlight_segments(
        path, {"source": "mcp", "note": reason}, spans,
        [{"score": 1.0, "signals": {"chosen": 1.0}, "primary_reason": reason}
         for _ in spans],
        score_info={"source": "mcp"})
    if not saved:
        raise RuntimeError("The video's analysis cache refused the clips; the debug "
                           "log says why.")
    live = show_live(path, spans, reason)
    return {"video": path, "clips": [list(s) for s in spans], "count": len(spans),
            "total_seconds": round(sum(e - s for s, e in spans), 2),
            "live": live,
            "next": ("On the app's edit timeline now; Ctrl+Z there takes them back."
                     if live.get("live") else
                     "Open this video in the timeline viewer: its edit timeline "
                     "holds these clips, ready to adjust and export.")}


def get_job(jobs: JobManager, job_id: str, log_lines: int = 20) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise ValueError(f"No such job: {job_id}")
    return job.snapshot(log_lines=max(0, min(int(log_lines), 300)))


def list_jobs(jobs: JobManager) -> dict:
    return {"jobs": [j.snapshot(log_lines=0) for j in jobs.all()]}


def cancel_job(jobs: JobManager, job_id: str) -> dict:
    job = jobs.cancel(job_id)
    if job is None:
        raise ValueError(f"No such job: {job_id}")
    return {"job_id": job.id, "status": job.status,
            "cancel_requested": job.cancel_flag.is_set()}
