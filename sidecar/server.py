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
        self.cancel_flag = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.subscribers: set["asyncio.Queue[dict]"] = set()

    @property
    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

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

    def start(self, video_paths: list[str], gui_config: dict,
              loop: asyncio.AbstractEventLoop) -> str:
        with self._lock:
            if self.is_running:
                raise RuntimeError("A pipeline run is already in progress")
            self.run_id = uuid.uuid4().hex
            self.cancel_flag = threading.Event()
            self.loop = loop
            self.thread = threading.Thread(
                target=self._run, args=(video_paths, gui_config), daemon=True
            )
            self.thread.start()
            return self.run_id

    def cancel(self) -> bool:
        if self.is_running:
            self.cancel_flag.set()
            return True
        return False

    def _run(self, video_paths: list[str], gui_config: dict) -> None:
        # Import lazily: keeps server import cheap and defers the heavy ML import
        # cost until an actual run starts.
        from pipeline import run_highlighter

        def log_fn(msg: Any) -> None:
            self._emit({"type": "log", "message": str(msg)})

        def progress_fn(cur: int, tot: int, task: str, det: str = "") -> None:
            self._emit({
                "type": "progress",
                "current": cur, "total": tot, "task": task, "detail": det,
            })

        try:
            self._emit({"type": "started", "run_id": self.run_id})
            output = run_highlighter(
                video_paths if len(video_paths) > 1 else video_paths[0],
                gui_config=gui_config,
                log_fn=log_fn,
                progress_fn=progress_fn,
                cancel_flag=self.cancel_flag,
                preview_fn=None,  # live preview stays in the native Qt editor
            )
            if self.cancel_flag.is_set():
                self._emit({"type": "cancelled"})
            else:
                self._emit({"type": "finished", "output": output or ""})
        except Exception as exc:  # noqa: BLE001 — surface everything to the UI
            self._emit({"type": "error", "message": str(exc),
                        "traceback": traceback.format_exc()})
        finally:
            self._emit({"type": "done"})


manager = RunManager()


class RunRequest(BaseModel):
    video_paths: list[str]
    config: dict


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "running": manager.is_running, "run_id": manager.run_id}


@app.post("/run")
async def start_run(req: RunRequest) -> dict:
    if not req.video_paths:
        return {"ok": False, "error": "No videos provided"}
    missing = [p for p in req.video_paths if not os.path.exists(p)]
    if missing:
        return {"ok": False, "error": f"Video file(s) not found: {missing}"}
    try:
        loop = asyncio.get_running_loop()
        run_id = manager.start(list(req.video_paths), dict(req.config), loop)
        return {"ok": True, "run_id": run_id}
    except RuntimeError as exc:
        return {"ok": False, "error": str(exc)}


@app.post("/cancel")
async def cancel_run() -> dict:
    return {"ok": manager.cancel()}


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


def _run_download(req: DownloadRequest) -> None:
    from downloader import download_videos_with_immediate_processing

    def log_fn(msg: Any) -> None:
        manager._emit({"type": "log", "message": str(msg)})

    def progress_fn(cur: int, tot: int, task: str = "Download", det: str = "") -> None:
        manager._emit({
            "type": "progress",
            "current": cur, "total": tot, "task": task, "detail": det,
        })

    try:
        manager._emit({"type": "started", "run_id": manager.run_id})
        results = download_videos_with_immediate_processing(
            search_url=req.url,
            save_dir=req.save_dir,
            pattern=req.pattern or "auto",
            log_fn=log_fn,
            progress_fn=progress_fn,
            cancel_flag=manager.cancel_flag,
            time_range=(
                None if req.download_full
                else (float(req.time_range_start), float(req.time_range_end))
            ),
            download_full=req.download_full,
            max_workers=max(1, req.concurrent),
        )
        paths = [r.get("filepath") for r in (results or []) if r.get("filepath")]
        if manager.cancel_flag.is_set():
            manager._emit({"type": "cancelled"})
        else:
            manager._emit({"type": "downloaded", "paths": paths})
            manager._emit({"type": "finished", "output": f"{len(paths)} file(s)"})
    except Exception as exc:  # noqa: BLE001
        manager._emit({"type": "error", "message": str(exc),
                       "traceback": traceback.format_exc()})
    finally:
        manager._emit({"type": "done"})


@app.post("/download")
async def start_download(req: DownloadRequest) -> dict:
    if not req.url.strip():
        return {"ok": False, "error": "No URL provided"}
    if manager.is_running:
        return {"ok": False, "error": "A run is already in progress"}
    try:
        manager.loop = asyncio.get_running_loop()
        manager.run_id = uuid.uuid4().hex
        manager.cancel_flag = threading.Event()
        manager.thread = threading.Thread(
            target=_run_download, args=(req,), daemon=True
        )
        manager.thread.start()
        return {"ok": True, "run_id": manager.run_id}
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


@app.get("/labels/{kind}")
async def get_labels(kind: str) -> dict:
    """kind: 'objects' | 'actions'. Returns the label vocabulary for autocomplete."""
    from modules.app_paths import data_file

    files = {
        "objects": "yolo_objects_labels.json",
        "actions": "kinetics_400_labels.json",
    }
    name = files.get(kind)
    if not name:
        return {"ok": False, "error": f"unknown label kind: {kind}", "labels": []}
    return {"ok": True, "labels": _load_label_json(data_file(name))}


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
    Names/avoid flags are set in the native Timeline Viewer."""
    try:
        bank = _face_bank()
        idents = []
        for ident in bank.all_identities():
            idents.append({
                "id": ident.get("id"),
                "name": ident.get("name") or "",
                "label": bank.name_for(ident.get("id")),
                "avoid": bool(ident.get("avoid", False)),
                "count": ident.get("count", 0),
            })
        return {"ok": True, "identities": idents}
    except Exception as exc:  # noqa: BLE001 — face stack is optional
        return {"ok": False, "error": str(exc), "identities": []}


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
    parser = argparse.ArgumentParser(description="VideoHighlighter FastAPI sidecar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8756)
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
