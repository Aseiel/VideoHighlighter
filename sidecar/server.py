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
            self.proc = ctx.Process(
                target=worker.run_job,
                args=(child_conn, job, self.cancel_flag, self.pause_event,
                      self.preview_flag),
                daemon=True,
            )
            self.proc.start()
            child_conn.close()  # parent keeps only the read end

            self.thread = threading.Thread(
                target=self._pump, args=(parent_conn,), daemon=True
            )
            self.thread.start()
            return self.run_id

    def _pump(self, conn) -> None:
        """Relay child events until it finishes, and notice if it dies."""
        saw_done = False
        try:
            while True:
                if not conn.poll(0.5):
                    # No data: only bail once the child is truly gone.
                    if self.proc is not None and not self.proc.is_alive():
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
            if self.proc is not None:
                self.proc.join(timeout=5)
            # A child that vanished without a `done` was killed or crashed — most
            # likely torch's native runtime faulting on teardown. Report it rather
            # than leaving the UI stuck on "running".
            if not saw_done:
                code = self.proc.exitcode if self.proc else None
                if self.cancel_flag.is_set():
                    self._emit({"type": "cancelled"})
                else:
                    self._emit({
                        "type": "error",
                        "message": (
                            f"The processing engine stopped unexpectedly "
                            f"(exit code {code}). The run was not completed."
                        ),
                    })
                self._emit({"type": "done"})
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
    """Duration/fps for a video, so the UI can show a real time-range slider.
    Uses the pipeline's own ffprobe-based helper (cv2's frame count is unreliable
    on VFR footage)."""
    try:
        if not os.path.exists(path):
            return {"ok": False, "error": "file not found"}
        from pipeline import get_video_duration

        duration = await asyncio.to_thread(get_video_duration, path, lambda *_: None)
        return {"ok": True, "duration": float(duration or 0)}
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
    """Launch the existing PySide6 app (Timeline Viewer / video editor).

    The realtime preview, overlays and VR view stay native Qt — this just starts
    main.py as its own process so the web UI can hand off to it.
    """
    import subprocess
    import sys as _sys

    root = _ROOT
    main_py = os.path.join(root, "main.py")
    if not os.path.exists(main_py):
        return {"ok": False, "error": "main.py not found"}
    try:
        if getattr(_sys, "frozen", False):
            # Packaged: the Qt app ships as its own executable next to ours.
            exe = os.path.join(os.path.dirname(_sys.executable),
                               "VideoHighlighter.exe")
            if not os.path.exists(exe):
                return {"ok": False, "error": "Qt app executable not found"}
            cmd = [exe]
        else:
            cmd = [_sys.executable, main_py]
        subprocess.Popen(cmd, cwd=root, close_fds=True)
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
