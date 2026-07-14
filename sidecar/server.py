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
