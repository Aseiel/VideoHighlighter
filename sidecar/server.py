"""FastAPI sidecar that exposes the VideoHighlighter engine over HTTP + WebSocket.

This is a thin wrapper around ``pipeline.run_highlighter`` — the exact same entry
point the Qt GUI drives. The web frontend (Vite/React/shadcn, hosted inside the
Tauri v2 shell) talks to this server on localhost. The heavy compute (torch, cv2,
YOLO, whisper) runs here, in-process, exactly as it does today; the browser only
renders controls and receives log/progress text. No performance-sensitive work
crosses the webview boundary.

Contract mirrors ``main.py``'s ``Worker``:
    run_highlighter(video_path, gui_config=..., log_fn=..., progress_fn=...,
                    cancel_flag=..., preview_fn=...)
Callbacks are routed to a per-run WebSocket instead of Qt signals.

Run standalone for dev:
    python -m sidecar.server --port 8756
Packaged: PyInstaller builds this into a single-file binary used as the Tauri
sidecar (see packaging/).
"""

from __future__ import annotations

import argparse
import asyncio
import multiprocessing as mp
import os
import sys
import threading
import traceback
import uuid
from typing import Any, Optional

# Make the project root importable whether run as ``python -m sidecar.server``
# from source or as a frozen one-file binary (sys._MEIPASS layout).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sidecar import worker


@asynccontextmanager
async def lifespan(_app: "FastAPI"):
    # Capture the serving event loop so worker threads can dispatch events even
    # before any WebSocket has connected.
    manager.loop = asyncio.get_running_loop()
    yield


app = FastAPI(title="VideoHighlighter Sidecar", version="1.0.0", lifespan=lifespan)

# The frontend is served from a Tauri custom scheme / localhost dev server; allow
# any origin since we only ever bind to loopback.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class RunManager:
    """Tracks a single active pipeline run and broadcasts its events to every
    connected WebSocket. Only one run at a time (matches the Qt GUI, which
    refuses to start a second while one is active).

    Connections register their own asyncio.Queue in ``subscribers`` for the life
    of the socket; the worker thread fans each event out to all of them via
    ``loop.call_soon_threadsafe``. Using a persistent per-connection queue (rather
    than one shared queue swapped per run) avoids a WS blocking forever on a stale
    queue reference."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.run_id: Optional[str] = None
        # Cross-process primitives, recreated per job.
        self.cancel_flag = mp.get_context("spawn").Event()
        # Set = running. Cleared = paused; the child's progress callback blocks
        # on it, the same gate the Qt Worker uses.
        self.pause_event = mp.get_context("spawn").Event()
        self.pause_event.set()
        self.preview_flag = None
        self.proc: Optional[mp.process.BaseProcess] = None
        self.thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.subscribers: set["asyncio.Queue[dict]"] = set()
        # Live detection preview. Toggleable mid-run, like the Qt checkbox.
        self.preview_enabled = False

    @property
    def is_running(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    @property
    def is_paused(self) -> bool:
        return not self.pause_event.is_set()

    def pause(self) -> bool:
        if self.is_running and not self.is_paused:
            self.pause_event.clear()
            return True
        return False

    def resume(self) -> bool:
        if self.is_running and self.is_paused:
            self.pause_event.set()
            return True
        return False

    def subscribe(self) -> "asyncio.Queue[dict]":
        q: "asyncio.Queue[dict]" = asyncio.Queue()
        self.subscribers.add(q)
        return q

    def unsubscribe(self, q: "asyncio.Queue[dict]") -> None:
        self.subscribers.discard(q)

    def _emit(self, event: dict) -> None:
        """Thread-safe fan-out from the worker thread to every subscriber."""
        loop = self.loop
        if loop is None:
            return

        def _dispatch() -> None:
            for q in list(self.subscribers):
                q.put_nowait(event)

        try:
            loop.call_soon_threadsafe(_dispatch)
        except RuntimeError:
            # Loop is gone (server shutting down) — drop the event.
            pass

    def start_job(self, job: dict, loop: asyncio.AbstractEventLoop) -> str:
        """Spawn the job in a child process and pump its events to subscribers."""
        with self._lock:
            if self.is_running:
                raise RuntimeError("A run is already in progress")
            self.run_id = uuid.uuid4().hex
            job = {**job, "run_id": self.run_id}
            self.loop = loop

            ctx = mp.get_context("spawn")
            self.cancel_flag = ctx.Event()
            self.pause_event = ctx.Event()
            self.pause_event.set()
            self.preview_flag = ctx.Value("b", 1 if self.preview_enabled else 0)

            parent_conn, child_conn = ctx.Pipe(duplex=False)
            proc = ctx.Process(
                target=worker.run_job,
                args=(child_conn, job, self.cancel_flag, self.pause_event,
                      self.preview_flag),
                daemon=True,
            )
            self.proc = proc
            proc.start()
            child_conn.close()  # parent keeps only the read end

            # Hand this pump its own proc handle rather than letting it read
            # self.proc: once this job finishes and the client starts the next
            # one, self.proc points at the *new* child, and a stale pump touching
            # it would join/None the wrong process.
            self.thread = threading.Thread(
                target=self._pump, args=(parent_conn, proc), daemon=True
            )
            self.thread.start()
            return self.run_id

    def _pump(self, conn, proc) -> None:
        """Relay events from `proc` until it finishes, and notice if it dies.

        `proc` is captured at start rather than read from self.proc so a pump
        that outlives its job (still in teardown when the next run starts) never
        observes or clears the successor's handle."""
        saw_done = False
        try:
            while True:
                if not conn.poll(0.5):
                    # No data: only bail once the child is truly gone.
                    if not proc.is_alive():
                        break
                    continue
                try:
                    event = conn.recv()
                except EOFError:
                    break
                if event.get("type") == "done":
                    saw_done = True
                self._emit(event)
                if saw_done:
                    break
        except Exception:
            pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
            proc.join(timeout=5)
            # A child that vanished without a `done` was killed or crashed — most
            # likely torch's native runtime faulting on teardown. Report it rather
            # than leaving the UI stuck on "running".
            if not saw_done:
                if self.cancel_flag.is_set():
                    self._emit({"type": "cancelled"})
                else:
                    self._emit({
                        "type": "error",
                        "message": (
                            f"The processing engine stopped unexpectedly "
                            f"(exit code {proc.exitcode}). The run was not completed."
                        ),
                    })
                self._emit({"type": "done"})
            # Only clear the slot if it is still ours: the next job may have
            # already claimed self.proc while we were tearing down.
            with self._lock:
                if self.proc is proc:
                    self.proc = None

    def cancel(self) -> bool:
        if self.is_running:
            self.cancel_flag.set()
            # Release the pause gate too: a paused worker is parked in
            # pause_event.wait() and would never observe the cancel otherwise.
            self.pause_event.set()
            return True
        return False

    def set_preview(self, enabled: bool) -> None:
        self.preview_enabled = enabled
        if self.preview_flag is not None:
            self.preview_flag.value = 1 if enabled else 0


manager = RunManager()


class RunRequest(BaseModel):
    video_paths: list[str]
    config: dict


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "running": manager.is_running,
        "paused": manager.is_paused,
        "run_id": manager.run_id,
    }


class PreviewRequest(BaseModel):
    enabled: bool


@app.post("/preview")
async def set_preview(req: PreviewRequest) -> dict:
    """Toggle the live detection preview. Takes effect mid-run."""
    manager.set_preview(req.enabled)
    return {"ok": True, "enabled": manager.preview_enabled}


@app.post("/pause")
async def pause_run() -> dict:
    return {"ok": manager.pause()}


@app.post("/resume")
async def resume_run() -> dict:
    return {"ok": manager.resume()}


@app.get("/stats")
async def stats() -> dict:
    """Lifetime count of analyzed videos — the Qt GUI's counter, same file."""
    try:
        from modules import analysis_stats

        return {
            "ok": True,
            "analyzed": analysis_stats.get_analyzed_count(),
            "path": analysis_stats.stats_path(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "analyzed": 0}


@app.post("/run")
async def start_run(req: RunRequest) -> dict:
    if not req.video_paths:
        return {"ok": False, "error": "No videos provided"}
    missing = [p for p in req.video_paths if not os.path.exists(p)]
    if missing:
        return {"ok": False, "error": f"Video file(s) not found: {missing}"}
    try:
        loop = asyncio.get_running_loop()
        run_id = manager.start_job(
            {"kind": "run", "video_paths": list(req.video_paths),
             "config": dict(req.config)},
            loop,
        )
        return {"ok": True, "run_id": run_id}
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}


@app.post("/cancel")
async def cancel_run() -> dict:
    return {"ok": manager.cancel()}


# ── Config persistence ────────────────────────────────────────────────────
# Reads/writes the same config.yaml the Qt GUI uses, so settings carry across
# both UIs. Shape mirrors main.py's save_config().


@app.get("/config")
async def get_config() -> dict:
    import yaml
    from modules.app_paths import config_path

    try:
        path = config_path()
        if not os.path.exists(path):
            return {"ok": True, "config": {}}
        with open(path, encoding="utf-8") as fh:
            return {"ok": True, "config": yaml.safe_load(fh) or {}}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "config": {}}


class ConfigRequest(BaseModel):
    config: dict


@app.post("/config")
async def save_config(req: ConfigRequest) -> dict:
    """Merge-write config.yaml. Merging (rather than replacing) preserves keys
    the web UI doesn't own yet, e.g. ui.suppress_no_cache_warning."""
    import yaml
    from modules.app_paths import config_path

    try:
        path = config_path()
        existing: dict = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                existing = yaml.safe_load(fh) or {}
        for section, values in (req.config or {}).items():
            if isinstance(values, dict) and isinstance(existing.get(section), dict):
                existing[section].update(values)
            else:
                existing[section] = values
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump(existing, fh, sort_keys=False, allow_unicode=True)
        return {"ok": True, "path": path}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


# ── Composition rules ─────────────────────────────────────────────────────
# Flat rows in the UI <-> the grouped {events: [{name, label, rules: [...]}]}
# shape composition_rules.yaml uses. Grouping/ungrouping mirrors main.py's
# _comp_save_rules / _comp_load_rules.


@app.get("/composition-rules")
async def get_composition_rules() -> dict:
    import yaml
    from modules.app_paths import composition_rules_path

    try:
        path = composition_rules_path()
        rows: list[dict] = []
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                events = (yaml.safe_load(fh) or {}).get("events", [])
            for ev in events:
                for rule in ev.get("rules", []):
                    rows.append({
                        "name": ev.get("name", ""),
                        "label": ev.get("label", ev.get("name", "")),
                        "source": rule.get("source", ""),
                        "region": rule.get("region", ""),
                        "min_count": rule.get("min_count", 1),
                        "max_count": rule.get("max_count", 999),
                        "window_secs": ev.get("window_secs", 0.75),
                        "persist_secs": ev.get("persist_secs", 0.5),
                    })
        return {"ok": True, "rules": rows}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "rules": []}


class CompRulesRequest(BaseModel):
    rules: list[dict]


@app.post("/composition-rules")
async def save_composition_rules(req: CompRulesRequest) -> dict:
    import yaml
    from modules.app_paths import user_data_dir

    try:
        events_ordered: list[dict] = []
        events_map: dict[str, dict] = {}
        for row in req.rules:
            name = str(row.get("name", "")).strip()
            source = str(row.get("source", "")).strip()
            region = str(row.get("region", "")).strip()
            # Same validation as the Qt table: incomplete rows are dropped.
            if not name or not source or not region:
                continue
            if name not in events_map:
                entry = {
                    "name": name,
                    "label": str(row.get("label") or name).strip(),
                    "rules": [],
                    "window_secs": float(row.get("window_secs", 0.75)),
                    "persist_secs": float(row.get("persist_secs", 0.5)),
                }
                events_map[name] = entry
                events_ordered.append(entry)
            events_map[name]["rules"].append({
                "source": source,
                "region": region,
                "min_count": int(row.get("min_count", 1)),
                "max_count": int(row.get("max_count", 999)),
            })
        path = os.path.join(user_data_dir(), "composition_rules.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            yaml.dump({"events": events_ordered}, fh, allow_unicode=True,
                      sort_keys=False, default_flow_style=False)
        return {"ok": True, "path": path, "events": len(events_ordered)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


# ── Video probing ─────────────────────────────────────────────────────────


@app.get("/video-info")
async def video_info(path: str) -> dict:
    """Full display metadata for a video, so the UI can show a real time-range
    slider AND orient a preview correctly. Backed by modules.video_probe (one
    ffprobe JSON call); keeps the {ok, duration} back-compat shape and adds
    width/height/fps/rotation. probe_video raises on ffprobe failure, so the
    try/except yields the {ok:false,error} shape the other endpoints use."""
    try:
        if not os.path.exists(path):
            return {"ok": False, "error": "file not found"}
        from modules.video_probe import probe_video

        info = await asyncio.to_thread(probe_video, path)
        return {
            "ok": True,
            "duration": float(info.get("duration") or 0),
            "width": int(info.get("width") or 0),
            "height": int(info.get("height") or 0),
            "fps": float(info.get("fps") or 0),
            "rotation": int(info.get("rotation") or 0),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


# ── Folder scan + combine ─────────────────────────────────────────────────

_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm", ".mts", ".m2ts")
_SCAN_CAP = 2000


def _natural_key(name: str):
    """Split a name into text/number runs so "clip2" sorts before "clip10"."""
    import re

    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", name)]


def _scan_video_files(path: str, recursive: bool) -> list[str]:
    """Video files under `path`, natural-sorted, capped at 2000.

    Extension match is case-insensitive against _VIDEO_EXTS. When `recursive`
    the whole tree is walked (each directory's entries natural-sorted so the
    result is deterministic); otherwise only the top level. Pure — no I/O
    beyond listing, so it is unit-testable without spinning up the server."""
    results: list[str] = []
    if recursive:
        for root, dirs, names in os.walk(path):
            dirs.sort(key=_natural_key)
            for name in sorted(names, key=_natural_key):
                if os.path.splitext(name)[1].lower() in _VIDEO_EXTS:
                    results.append(os.path.join(root, name))
                    if len(results) >= _SCAN_CAP:
                        return results
    else:
        for name in sorted(os.listdir(path), key=_natural_key):
            full = os.path.join(path, name)
            if (os.path.splitext(name)[1].lower() in _VIDEO_EXTS
                    and os.path.isfile(full)):
                results.append(full)
                if len(results) >= _SCAN_CAP:
                    break
    return results


@app.get("/scan-folder")
async def scan_folder(path: str, recursive: int = 0) -> dict:
    """List video files in a folder for the batch picker.

    A missing/non-directory path returns {ok:false,error} (HTTP 200), matching
    the other endpoints' error shape rather than raising."""
    try:
        if not path or not os.path.isdir(path):
            return {"ok": False, "error": f"not a folder: {path!r}"}
        files = await asyncio.to_thread(_scan_video_files, path, bool(recursive))
        return {"ok": True, "files": files, "count": len(files)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


class CombineRequest(BaseModel):
    files: list[str]
    output: str
    music_path: str | None = None
    music_mode: str | None = "replace"
    music_volume: float | None = 0.8


@app.post("/combine")
async def start_combine(req: CombineRequest) -> dict:
    """Join finished highlight clips into one reel via the RunManager (job kind
    'combine'). Validation runs BEFORE any job starts, so a bad request never
    occupies the single run slot."""
    files = [f for f in (req.files or []) if f]
    if len(files) < 2:
        return {"ok": False, "error": "Need at least 2 files to combine"}
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        return {"ok": False, "error": f"File(s) not found: {missing}"}
    if not (req.output or "").strip():
        return {"ok": False, "error": "No output path provided"}

    job: dict = {"kind": "combine", "files": files, "output": req.output}
    if req.music_path:
        job["music_path"] = req.music_path
        job["music_mode"] = req.music_mode or "replace"
        job["music_volume"] = float(
            req.music_volume if req.music_volume is not None else 0.8)

    try:
        loop = asyncio.get_running_loop()
        run_id = manager.start_job(job, loop)
        return {"ok": True, "run_id": run_id}
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


# ── Camera cards, script, music, and the auto pipeline ────────────────────
# Card detection and script validation are cheap and synchronous, so they are
# plain request/response. The pipeline itself is a job on the RunManager, like
# every other long operation, so it shares one cancel/pause/event path.

@app.get("/gopro/cards")
async def gopro_cards() -> dict:
    """Mounted camera cards, with what is on each one.

    Detection walks the filesystem, which can block on a spun-down or
    disconnected drive, so it runs off the event loop.
    """
    try:
        from modules.gopro_ingest import find_gopro_cards, scan_card, suggest_folder_name

        def probe() -> list[dict]:
            found = []
            for card in find_gopro_cards():
                takes = scan_card(card)
                found.append({
                    "root": card.root,
                    "label": card.label,
                    "camera_type": card.camera_type,
                    "firmware": card.firmware,
                    "file_count": card.file_count,
                    "total_bytes": card.total_bytes,
                    "take_count": len(takes),
                    "chaptered_takes": sum(1 for t in takes if t.is_chaptered),
                    "suggested_folder": suggest_folder_name(card, takes),
                })
            return found

        return {"ok": True, "cards": await asyncio.to_thread(probe)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@app.get("/script/example")
async def script_example() -> dict:
    """A commented starter script the UI can drop into an empty editor."""
    try:
        from modules.script_plan import example_script
        return {"ok": True, "text": example_script()}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


class ScriptRequest(BaseModel):
    text: str


@app.post("/script/validate")
async def script_validate(req: ScriptRequest) -> dict:
    """Parse a script and report errors/warnings without running anything.

    This is what makes the script editor usable: the alternative is finding out
    a key was misspelled after twenty minutes of detection.
    """
    try:
        from modules.script_plan import ScriptError, parse_script, validate_script
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    try:
        script = parse_script(req.text or "")
    except ScriptError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": True,
        "title": script.title,
        "beats": [b.name for b in script.beats],
        "clip_count": script.clip_count,
        "target_duration": script.target_duration,
        "music": script.music,
        "warnings": validate_script(script),
    }


@app.get("/music/analysis")
async def music_analysis(path: str) -> dict:
    """Beat grid for a music file, for the UI's waveform/beat display."""
    try:
        from modules.music_analysis import analyze_music

        if not path or not os.path.exists(path):
            return {"ok": False, "error": f"not found: {path!r}"}
        analysis = await asyncio.to_thread(analyze_music, path)
        return {
            "ok": True,
            "bpm": analysis.bpm,
            "duration": analysis.duration,
            "beats": analysis.beats,
            "downbeats": analysis.downbeats,
            "meter": analysis.meter,
            "backend": analysis.backend,
            "sections": [
                {"start": s.start, "end": s.end, "energy": s.energy, "label": s.label}
                for s in analysis.sections
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@app.get("/transitions")
async def list_transitions() -> dict:
    """The transition names the engine accepts, so the UI never offers one the
    renderer would refuse."""
    try:
        from modules.transitions import DEFAULT_DURATION, TRANSITIONS
        return {"ok": True, "transitions": sorted(TRANSITIONS),
                "default_duration": DEFAULT_DURATION}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@app.get("/edl")
async def get_edl(path: str) -> dict:
    """Read a cut list for the timeline editor."""
    try:
        from modules.edl import EdlError, load_edl, validate_edl

        if not path or not os.path.exists(path):
            return {"ok": True, "exists": False}
        try:
            edl = await asyncio.to_thread(load_edl, path)
        except EdlError as exc:
            return {"ok": False, "exists": True, "error": str(exc)}
        return {
            "ok": True, "exists": True, "title": edl.title,
            "music": edl.music, "music_mode": edl.music_mode,
            "music_volume": edl.music_volume,
            "width": edl.width, "height": edl.height, "fps": edl.fps,
            "crf": edl.crf, "duration": edl.duration,
            "source_duration": edl.source_duration,
            "warnings": validate_edl(edl),
            "cuts": [
                {"source": c.source, "start": c.start, "end": c.end,
                 "duration": c.duration, "transition": c.transition,
                 "transition_duration": c.transition_duration,
                 "label": c.label}
                for c in edl.cuts
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


class EdlSaveRequest(BaseModel):
    path: str
    title: str | None = "Untitled"
    music: str | None = ""
    music_mode: str | None = "replace"
    music_volume: float | None = 0.8
    width: int | None = 0
    height: int | None = 0
    fps: int | None = 0
    crf: int | None = 20
    cuts: list[dict]


def _edl_from_request(req: "EdlSaveRequest"):
    from modules.edl import Cut, Edl

    return Edl(
        title=req.title or "Untitled",
        music=req.music or "", music_mode=req.music_mode or "replace",
        music_volume=float(req.music_volume if req.music_volume is not None else 0.8),
        width=int(req.width or 0), height=int(req.height or 0),
        fps=int(req.fps or 0), crf=int(req.crf or 20),
        cuts=[
            Cut(source=str(c.get("source", "")),
                start=float(c.get("start", 0.0)),
                end=float(c.get("end", 0.0)),
                transition=str(c.get("transition", "cut")),
                transition_duration=float(c.get("transition_duration", 0.5)),
                label=str(c.get("label", "") or ""))
            for c in (req.cuts or [])
        ],
    )


@app.post("/edl")
async def save_edl_endpoint(req: EdlSaveRequest) -> dict:
    """Write the timeline back to disk. Validated first so a broken edit is
    reported instead of overwriting a good cut list with a bad one."""
    try:
        from modules.edl import EdlError, parse_edl, save_edl, validate_edl

        edl = _edl_from_request(req)
        # Round-trip through the parser: it owns the rules, and writing a file
        # that the loader would then reject is the one outcome worth ruling out.
        try:
            save_edl(edl, req.path)
            parse_edl(open(req.path, encoding="utf-8").read())
        except EdlError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "path": req.path, "duration": edl.duration,
                "warnings": validate_edl(edl)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


class EdlRenderRequest(EdlSaveRequest):
    output: str


@app.post("/edl/render")
async def render_edl_endpoint(req: EdlRenderRequest) -> dict:
    """Render a timeline as a job, so it streams progress like any other run."""
    if not (req.output or "").strip():
        return {"ok": False, "error": "No output path provided"}
    if not req.cuts:
        return {"ok": False, "error": "The timeline is empty"}
    try:
        edl = _edl_from_request(req)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}

    job = {
        "kind": "edl",
        "output": req.output,
        "edl_path": req.path,
        "edl": {
            "title": edl.title, "music": edl.music,
            "music_mode": edl.music_mode, "music_volume": edl.music_volume,
            "width": edl.width, "height": edl.height, "fps": edl.fps,
            "crf": edl.crf,
            "cuts": [
                {"source": c.source, "start": c.start, "end": c.end,
                 "transition": c.transition,
                 "transition_duration": c.transition_duration,
                 "label": c.label}
                for c in edl.cuts
            ],
        },
    }
    try:
        loop = asyncio.get_running_loop()
        return {"ok": True, "run_id": manager.start_job(job, loop)}
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


class AutoRunRequest(BaseModel):
    dest_root: str
    card_root: str | None = None
    source_paths: list[str] | None = None
    folder_name: str | None = ""
    script_path: str | None = ""
    music_path: str | None = ""
    output_name: str | None = "film.mp4"
    music_mode: str | None = "replace"
    music_volume: float | None = 0.8
    transition: str | None = "cut"
    transition_duration: float | None = 0.5
    transition_bars: float | None = 0.0
    quantise: str | None = ""
    width: int | None = 0
    height: int | None = 0
    fps: int | None = 0
    crf: int | None = 20
    resume: bool = True
    verify: str | None = "size"
    config: dict | None = None


@app.post("/auto/run")
async def start_auto(req: AutoRunRequest) -> dict:
    """Run card-to-film as one job. Validated before the run slot is taken."""
    if not (req.dest_root or "").strip():
        return {"ok": False, "error": "No destination folder provided"}
    if not req.card_root and not req.source_paths:
        return {"ok": False, "error": "Pick a camera card or some source files"}
    for label, path in (("Script", req.script_path), ("Music", req.music_path)):
        if path and not os.path.exists(path):
            return {"ok": False, "error": f"{label} file not found: {path}"}
    # Reject an unknown transition here rather than after the detection pass:
    # the render is the last stage, and finding out about a typo then costs the
    # whole run.
    try:
        from modules.transitions import normalise_kind
        normalise_kind(req.transition or "cut")
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception:
        pass
    if (req.quantise or "") not in ("", "bar", "beat"):
        return {"ok": False,
                "error": f"unknown quantise unit {req.quantise!r} "
                         f"(expected 'bar', 'beat' or nothing)"}

    job = {
        "kind": "auto",
        "dest_root": req.dest_root,
        "card_root": req.card_root or "",
        "source_paths": [p for p in (req.source_paths or []) if p],
        "folder_name": req.folder_name or "",
        "script_path": req.script_path or "",
        "music_path": req.music_path or "",
        "output_name": req.output_name or "film.mp4",
        "music_mode": req.music_mode or "replace",
        "music_volume": float(req.music_volume if req.music_volume is not None else 0.8),
        "transition": req.transition or "cut",
        "transition_duration": float(req.transition_duration or 0.5),
        "transition_bars": float(req.transition_bars or 0.0),
        "quantise": req.quantise or "",
        "width": int(req.width or 0),
        "height": int(req.height or 0),
        "fps": int(req.fps or 0),
        "crf": int(req.crf or 20),
        "resume": bool(req.resume),
        "verify": req.verify or "size",
        "config": req.config or {},
    }
    try:
        loop = asyncio.get_running_loop()
        return {"ok": True, "run_id": manager.start_job(job, loop)}
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@app.get("/auto/job")
async def auto_job(root: str) -> dict:
    """Saved job state for a destination folder.

    Lets the UI show what a previous run finished — and therefore what a resume
    would skip — before anything is started.
    """
    try:
        from modules.auto_pipeline import job_path, load_job

        path = job_path(root)
        if not os.path.exists(path):
            return {"ok": True, "exists": False}
        state = await asyncio.to_thread(load_job, path)
        return {
            "ok": True,
            "exists": True,
            "job_id": state.job_id,
            "created": state.created,
            "clips": state.clips,
            "highlights": state.highlights,
            "reel": state.reel,
            "final": state.final,
            "errors": state.errors,
            "stages": [
                {"name": s.name, "status": s.status, "detail": s.detail,
                 "error": s.error, "seconds": s.seconds,
                 "satisfied": s.is_satisfied}
                for s in state.stages.values()
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


# ── Download ──────────────────────────────────────────────────────────────
# Reuses the run manager's event stream so the UI's log/progress panel shows
# download output exactly like a pipeline run.

class DownloadRequest(BaseModel):
    url: str
    save_dir: str
    pattern: str | None = "auto"
    download_full: bool = True
    time_range_start: int = 0
    time_range_end: int = 300
    concurrent: int = 1
    # When given (from the picker), these exact URLs are fetched and no listing
    # scrape happens — same as the Qt picker's start_download(video_urls=...).
    video_urls: list[str] | None = None


class BrowseRequest(BaseModel):
    url: str
    pattern: str | None = "auto"
    use_browser: str = "auto"


@app.post("/browse-listing")
async def browse_listing(req: BrowseRequest) -> dict:
    """Scrape a listing page into pickable entries — the data behind the Qt
    'Browse & Select…' thumbnail grid."""
    if not req.url.strip().startswith(("http://", "https://")):
        return {"ok": False, "error": "URL must start with http:// or https://",
                "entries": []}
    try:
        from downloader import extract_video_entries

        entries = await asyncio.to_thread(
            extract_video_entries, req.url, req.pattern, lambda *_: None,
            req.use_browser,
        )
        return {"ok": True, "entries": entries or []}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "entries": []}


@app.get("/about")
async def about() -> dict:
    """Version/links for the About panel (mirrors _build_about_tab)."""
    try:
        from version import __version__, __edition__

        return {
            "ok": True,
            "version": __version__,
            "edition": __edition__,
            "support_email": "przkreft@gmail.com",
            "website": "https://videohighlighter.com",
            "discord": "https://discord.gg/cUPJqPAMmm",
            "repo": "https://github.com/Aseiel/VideoHighlighter",
            "log_path": _debug_log_path(),
        }
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


def _debug_log_path() -> str:
    try:
        from modules import debug_console

        return debug_console.log_file_path()
    except Exception:
        return ""


def _reveal(path: str) -> dict:
    """Select `path` in the OS file manager."""
    import subprocess

    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", path])
        else:
            # No portable "select the file" on Linux — open the folder.
            subprocess.Popen(["xdg-open", os.path.dirname(path)])
        return {"ok": True, "path": path}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@app.post("/reveal-log")
async def reveal_log() -> dict:
    """Show the debug log in the file manager.

    The Qt 'Debug log' checkbox opens a live Qt window mirroring stdout, which
    has no meaning here — the web UI already streams the run log. What it can't
    give you is the file itself (the thing you attach to a bug report), so this
    reveals it instead of pretending to port the window."""
    path = _debug_log_path()
    if not path or not os.path.exists(path):
        return {"ok": False, "error": "No debug log yet — run something first."}
    return _reveal(path)


class RevealOutputRequest(BaseModel):
    path: str


@app.post("/reveal-output")
async def reveal_output(req: RevealOutputRequest) -> dict:
    """Show a finished highlight video in the file manager.

    The run's `finished` event carries the output path but the web UI had no way
    to act on it — the browser can't open a local folder, so without this the
    video you just made is only reachable by hunting for it by hand."""
    path = (req.path or "").strip()
    if not path:
        return {"ok": False, "error": "No output path given."}
    if not os.path.exists(path):
        return {"ok": False, "error": f"Output no longer exists: {path}"}
    return _reveal(path)


@app.post("/download")
async def start_download(req: DownloadRequest) -> dict:
    if not req.url.strip():
        return {"ok": False, "error": "No URL provided"}
    try:
        loop = asyncio.get_running_loop()
        run_id = manager.start_job(
            {
                "kind": "download",
                "url": req.url,
                "save_dir": req.save_dir,
                "pattern": req.pattern,
                "download_full": req.download_full,
                "time_range": (
                    None if req.download_full
                    else (float(req.time_range_start), float(req.time_range_end))
                ),
                "concurrent": max(1, req.concurrent),
                "video_urls": req.video_urls or None,
            },
            loop,
        )
        return {"ok": True, "run_id": run_id}
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


# ── Labels ────────────────────────────────────────────────────────────────
# The Basic/Advanced tabs offer autocomplete from the same label JSONs the Qt
# GUI reads ("Load Labels" buttons there).

def _load_label_json(path: str) -> list[str]:
    """Parse a label JSON. Mirrors main.py's load_labels_from_json so the web UI
    and the Qt GUI offer the identical vocabulary (list, flat dict, YOLO's
    {"class": {idx: label}}, and the Intel label_to_idx / idx_to_label shapes)."""
    import json
    try:
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        if isinstance(data, list):
            labels = [str(x) for x in data]
        elif isinstance(data, dict):
            if "label_to_idx" in data:
                labels = list(data["label_to_idx"].keys())
            elif "idx_to_label" in data:
                labels = list(data["idx_to_label"].values())
            elif "class" in data:
                labels = list(data["class"].values())
            else:
                values = list(data.values())
                labels = values if values and isinstance(values[0], str) else list(data.keys())
        else:
            return []
        return [str(x) for x in labels if str(x).strip()]
    except Exception:
        return []


@app.get("/labels/objects")
async def get_object_labels(yolo_type: str = "standard") -> dict:
    """Object vocabulary. Mirrors open_object_label_selector: the source depends
    on the detector type (standard COCO vs custom keypoints vs both)."""
    from modules.app_paths import custom_keypoint_names, data_file

    try:
        coco = _load_label_json(data_file("yolo_objects_labels.json"))
        if yolo_type == "custom":
            return {"ok": True, "labels": custom_keypoint_names(), "source": "custom"}
        if yolo_type == "mixed":
            return {"ok": True, "labels": custom_keypoint_names() + coco,
                    "source": "mixed"}
        return {"ok": True, "labels": coco, "source": "standard"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "labels": []}


@app.get("/labels/actions")
async def get_action_labels(backend: str = "auto", models: str = "intel_only") -> dict:
    """Action vocabulary. Mirrors open_action_label_selector, including the
    r3d_* backends forcing intel_only and the mixed mode's [custom]/[intel]
    disambiguation suffixes for labels present in both sets."""
    from modules.app_paths import data_file

    try:
        if backend in ("r3d_cuda", "r3d_cpu"):
            models = "intel_only"

        intel = _load_label_json(data_file("kinetics_400_labels.json"))
        custom_ov = _load_label_json(
            data_file("intel_finetuned_classifier_3d_mapping.json"))
        r3d_custom = _load_label_json(data_file("r3d_finetuned_mapping.json"))

        if models == "custom_only":
            return {"ok": True, "labels": custom_ov}
        if models == "r3d_custom_only":
            return {"ok": True, "labels": r3d_custom}
        if models == "mixed":
            custom = custom_ov or r3d_custom
            shared = set(custom) & set(intel)
            labels = [f"{c} [custom]" if c in shared else c for c in custom]
            labels += [f"{i} [intel]" if i in shared else i
                       for i in intel if i not in set(custom) or i in shared]
            return {"ok": True, "labels": labels, "shared": len(shared)}
        return {"ok": True, "labels": intel}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "labels": []}


@app.get("/labels/{kind}")
async def get_labels(kind: str) -> dict:
    """Back-compat shim for the simple object/action label fetch."""
    if kind == "objects":
        return await get_object_labels()
    if kind == "actions":
        return await get_action_labels()
    return {"ok": False, "error": f"unknown label kind: {kind}", "labels": []}


# ── Face bank (Avoid tab) ─────────────────────────────────────────────────

FACE_DB_PATH = "./cache/face_db.json"


def _face_bank():
    from video_ai_editor.face_identity import FaceIdentityBank

    bank = FaceIdentityBank(db_path=FACE_DB_PATH)
    bank.load()
    return bank


@app.get("/faces")
async def list_faces() -> dict:
    """Identities from the shared face bank, as shown in the Qt Avoid tab.
    Sorted unnamed-last then by descending sighting count, matching
    refresh_avoid_list. `thumb` is base64 JPEG, rendered as-is by the UI."""
    try:
        bank = _face_bank()
        idents = []
        for ident in bank.all_identities():
            ident_id = ident.get("id")
            idents.append({
                "id": ident_id,
                "name": ident.get("name") or "",
                "label": bank.name_for(ident_id),
                "avoid": bool(ident.get("avoid", False)),
                "count": ident.get("count", 0),
                "thumb": ident.get("thumb") or "",
            })
        idents.sort(key=lambda i: (not i["name"], -i["count"]))
        return {
            "ok": True,
            "identities": idents,
            "named": sum(1 for i in idents if i["name"]),
            "avoided": sum(1 for i in idents if i["avoid"]),
        }
    except Exception as exc:  # noqa: BLE001 — face stack is optional
        return {"ok": False, "error": str(exc), "identities": []}


class IdentityRequest(BaseModel):
    id: str


@app.post("/faces/remove")
async def remove_face(req: IdentityRequest) -> dict:
    try:
        bank = _face_bank()
        removed = bank.remove(req.id)
        if removed:
            bank.save()
        return {"ok": removed}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


class NameRequest(BaseModel):
    id: str
    name: str


@app.post("/faces/name")
async def name_face(req: NameRequest) -> dict:
    """Name an identity. If the name already belongs to someone else, merge into
    them rather than creating a duplicate — same rule as the Timeline Viewer."""
    try:
        bank = _face_bank()
        target = req.name.strip()
        if not target:
            return {"ok": False, "error": "empty name"}
        for ident in bank.all_identities():
            if ident.get("id") != req.id and (ident.get("name") or "") == target:
                bank.merge_identities(ident["id"], req.id)
                bank.save()
                return {"ok": True, "merged_into": ident["id"]}
        bank.name_identity(req.id, target)
        bank.save()
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


class ClearFacesRequest(BaseModel):
    keep_named: bool = True


@app.post("/faces/clear")
async def clear_faces(req: ClearFacesRequest) -> dict:
    """keep_named mirrors the Qt 'Keep named / avoided' vs 'Clear everything'."""
    try:
        bank = _face_bank()
        bank.clear(keep_named=req.keep_named)
        bank.save()
        return {"ok": True, "remaining": len(bank.all_identities())}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@app.get("/avoid-ranges")
async def get_avoid_ranges(path: str) -> dict:
    """Manual avoid ranges marked for a video in the Timeline Viewer."""
    try:
        from modules.manual_avoid import load_ranges

        return {"ok": True, "ranges": [[a, b] for a, b in load_ranges(path)]}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc), "ranges": []}


class AvoidRangesRequest(BaseModel):
    video_path: str
    ranges: list[list[float]]


@app.post("/avoid-ranges")
async def set_avoid_ranges(req: AvoidRangesRequest) -> dict:
    try:
        from modules.manual_avoid import save_ranges

        save_ranges(req.video_path, [tuple(r) for r in req.ranges])
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


class ScanRequest(BaseModel):
    video_path: str


@app.post("/faces/scan")
async def scan_faces(req: ScanRequest) -> dict:
    """Offline face pass over a video to populate the bank, mirroring
    FaceScanWorker. tag_entries caches per-frame tagging so the pipeline's avoid
    step reuses this work instead of re-running recognition."""
    if not os.path.exists(req.video_path):
        return {"ok": False, "error": "video not found"}
    try:
        loop = asyncio.get_running_loop()
        run_id = manager.start_job(
            {"kind": "scan_faces", "video_path": req.video_path,
             "face_db_path": FACE_DB_PATH},
            loop,
        )
        return {"ok": True, "run_id": run_id}
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


class AvoidRequest(BaseModel):
    id: str
    avoid: bool


@app.post("/faces/avoid")
async def set_face_avoid(req: AvoidRequest) -> dict:
    """Tick/untick a person for exclusion, persisted to the shared face bank."""
    try:
        bank = _face_bank()
        ident = bank._id_index.get(req.id)
        if ident is None:
            return {"ok": False, "error": "identity not found"}
        ident["avoid"] = bool(req.avoid)
        bank.save()
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


# ── LLM chat ──────────────────────────────────────────────────────────────
# Thin wrapper over llm.llm_module. The full Qt widget also does vision/CLIP
# prefilter over video frames; this exposes the text-chat path, which is what
# the web UI needs. Heavy vision work stays in the native editor.

_llm_cache: dict[str, Any] = {"key": None, "module": None}


@app.get("/llm/backends")
async def llm_backends() -> dict:
    try:
        from llm.llm_module import get_available_backends, get_ollama_models

        backends = get_available_backends()
        return {
            "ok": True,
            "backends": backends,
            "ollama_models": get_ollama_models() if "ollama" in backends else [],
        }
    except Exception as exc:  # noqa: BLE001 — LLM stack is optional
        return {"ok": False, "error": str(exc), "backends": [], "ollama_models": []}


@app.get("/llm/clip-status")
async def clip_status() -> dict:
    """Whether the OpenVINO CLIP stack imports; names the failing module if not."""
    try:
        from llm.clip_prefilter import ClipFramePrefilter

        err = ClipFramePrefilter.import_error()
        return {"ok": True, "available": err is None, "error": err}
    except Exception as exc:  # noqa: BLE001
        return {"ok": True, "available": False, "error": str(exc)}


class VisionSearchRequest(BaseModel):
    video_path: str
    query: str
    # clip = CLIP ranker only; clip_llm = CLIP then VLM confirms; llm = VLM only.
    mode: str = "clip"
    interval: float = 1.0
    top_k: int = 30
    threshold: float = 0.5
    clip_device: str = "GPU"
    backend: str = "ollama"
    model: str = "llava"


@app.post("/llm/vision-search")
async def vision_search(req: VisionSearchRequest) -> dict:
    """Find moments matching a text query. Runs in the worker process and
    streams progress/results over the run socket."""
    if not os.path.exists(req.video_path):
        return {"ok": False, "error": "video not found"}
    if not req.query.strip():
        return {"ok": False, "error": "empty query"}
    try:
        loop = asyncio.get_running_loop()
        run_id = manager.start_job(
            {"kind": "vision_search", **req.model_dump()}, loop
        )
        return {"ok": True, "run_id": run_id}
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


class ChatRequest(BaseModel):
    backend: str
    model: str
    message: str
    video_path: str | None = None


@app.post("/llm/chat")
async def llm_chat(req: ChatRequest) -> dict:
    """Ask the local LLM about a video, using its cached analysis as context."""
    try:
        from llm.llm_module import LLMModule

        key = f"{req.backend}:{req.model}"
        if _llm_cache["key"] != key:
            module = LLMModule(backend=req.backend, model=req.model)
            module.load()
            _llm_cache.update({"key": key, "module": module})
        module = _llm_cache["module"]

        # Feed the same analysis cache the Qt chat uses, when available.
        analysis = None
        if req.video_path and os.path.exists(req.video_path):
            try:
                from modules.video_cache import VideoAnalysisCache

                analysis = VideoAnalysisCache().load(req.video_path)
            except Exception:
                analysis = None

        answer = await asyncio.to_thread(
            module.query, req.message, analysis_data=analysis
        )
        return {"ok": True, "answer": str(answer), "had_context": analysis is not None}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


# ── Native Qt editor bridge ───────────────────────────────────────────────

class EditorRequest(BaseModel):
    video_path: str | None = None


@app.post("/open-editor")
async def open_editor(req: EditorRequest) -> dict:
    """Open the native Timeline Viewer for a video.

    Launches signal_timeline_viewer directly (it has a standalone entry point
    that builds its own QApplication and loads the analysis cache itself), NOT
    main.py — starting main.py would open the whole Qt application, which is a
    confusing second copy of the app rather than the viewer the user asked for.

    The viewer is a separate process, so it can't be driven in-process the way
    main.py does it; ranges marked there reach us through the shared store in
    modules.manual_avoid.
    """
    import subprocess
    import sys as _sys

    if not req.video_path:
        return {"ok": False, "error": "No video selected. Add a video first."}
    if not os.path.exists(req.video_path):
        return {"ok": False, "error": f"Video not found: {req.video_path}"}

    try:
        if getattr(_sys, "frozen", False):
            # The packaged Qt app is a single exe; `--timeline <video>` asks it
            # for just the viewer rather than the whole GUI (see main.py).
            exe = os.path.join(os.path.dirname(_sys.executable),
                               "VideoHighlighter.exe")
            if not os.path.exists(exe):
                return {"ok": False, "error": "Qt app executable not found"}
            cmd = [exe, "--timeline", req.video_path]
        else:
            viewer = os.path.join(_ROOT, "signal_timeline_viewer.py")
            if not os.path.exists(viewer):
                return {"ok": False, "error": "signal_timeline_viewer.py not found"}
            cmd = [_sys.executable, viewer, req.video_path]

        # Detach: the viewer must outlive this request, and on Windows a child
        # sharing our console would die with us.
        creationflags = 0
        if _sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008  # DETACHED_PROCESS
        subprocess.Popen(cmd, cwd=_ROOT, close_fds=True,
                         creationflags=creationflags)
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


@app.websocket("/ws")
async def ws_events(ws: WebSocket) -> None:
    """Streams run events (log / progress / finished / error) to the frontend.

    The frontend connects, then POSTs /run; events for the active run are pushed
    here until a ``done`` event, after which the socket stays open for the next
    run's events too."""
    await ws.accept()
    if manager.loop is None:
        manager.loop = asyncio.get_running_loop()
    q = manager.subscribe()
    try:
        while True:
            event = await q.get()
            await ws.send_json(event)
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        # Never let a socket error crash the server.
        pass
    finally:
        manager.unsubscribe(q)


def main() -> None:
    # The engine prints emoji; a Windows console defaults to cp1252 and the
    # first such print would raise UnicodeEncodeError. Same reason worker.py
    # reconfigures its stdio.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="VideoHighlighter FastAPI sidecar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8756)
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    # Required before any spawn on Windows / frozen builds: without it the child
    # re-executes the server instead of the worker entry point.
    mp.freeze_support()
    main()
