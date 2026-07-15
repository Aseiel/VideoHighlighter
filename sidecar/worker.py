"""Pipeline worker that runs in its own process.

Why a separate process rather than a thread: torch's native runtime (c10.dll)
can hard-crash the interpreter (access violation 0xc0000005) when a run is
cancelled mid-inference — the same class of problem main.py works around by
hard-exiting on close instead of unwinding cleanly. In-process, that crash takes
the whole HTTP server down and the UI just sees the connection drop. Isolated
here, the server survives, reports the failure, and stays ready for the next run.

Protocol: the parent passes a job dict over a multiprocessing pipe; this module
streams event dicts (log/progress/preview/finished/...) back over the same pipe.
Cancel and pause are Events shared through the process boundary.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any


def _install_path() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    if root not in sys.path:
        sys.path.insert(0, root)


def run_job(conn, job: dict, cancel_evt, pause_evt, preview_flag) -> None:
    """Entry point for the child process. Never raises into the parent."""
    _install_path()

    def emit(event: dict) -> None:
        try:
            conn.send(event)
        except Exception:
            # Parent went away — nothing useful left to do.
            pass

    def log_fn(msg: Any) -> None:
        emit({"type": "log", "message": str(msg)})

    def progress_fn(cur: int, tot: int, task: str, det: str = "") -> None:
        pause_evt.wait()
        if cancel_evt.is_set():
            return
        emit({"type": "progress", "current": cur, "total": tot,
              "task": task, "detail": det})

    def preview_fn(frame, boxes, sec) -> None:
        if not preview_flag.value or cancel_evt.is_set():
            return
        try:
            import base64
            import cv2

            ok, buf = cv2.imencode(".jpg", frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if not ok:
                return
            emit({
                "type": "preview",
                "jpeg": base64.b64encode(buf.tobytes()).decode("ascii"),
                "boxes": [
                    {"name": b[0], "x": b[1], "y": b[2],
                     "w": b[3], "h": b[4], "conf": b[5]}
                    for b in (boxes or [])
                ],
                "sec": int(sec),
            })
        except Exception:
            pass

    kind = job.get("kind", "run")
    try:
        emit({"type": "started", "run_id": job.get("run_id")})

        if kind == "run":
            from pipeline import run_highlighter

            paths = job["video_paths"]
            output = run_highlighter(
                paths if len(paths) > 1 else paths[0],
                gui_config=job["config"],
                log_fn=log_fn,
                progress_fn=progress_fn,
                cancel_flag=cancel_evt,
                preview_fn=preview_fn,
            )
            if cancel_evt.is_set():
                emit({"type": "cancelled"})
            else:
                emit({"type": "finished", "output": output or ""})

        elif kind == "download":
            from downloader import download_videos_with_immediate_processing

            results = download_videos_with_immediate_processing(
                search_url=job["url"],
                save_dir=job["save_dir"],
                pattern=job.get("pattern") or "auto",
                log_fn=log_fn,
                progress_fn=progress_fn,
                cancel_flag=cancel_evt,
                time_range=job.get("time_range"),
                download_full=job.get("download_full", True),
                max_workers=job.get("concurrent", 1),
            )
            paths = [r.get("filepath") for r in (results or []) if r.get("filepath")]
            if cancel_evt.is_set():
                emit({"type": "cancelled"})
            else:
                emit({"type": "downloaded", "paths": paths})
                emit({"type": "finished", "output": f"{len(paths)} file(s)"})

        elif kind == "scan_faces":
            from video_ai_editor.face_identity import FaceIdentityBank
            from modules.compute_forbidden import build_tracking_model, tag_entries

            bank = FaceIdentityBank(db_path=job["face_db_path"])
            yolo_model = build_tracking_model("n")
            tag_entries(
                job["video_path"], bank, yolo_model, model_size="n",
                face_every=15, vid_stride=3, save_bank=True, log_fn=log_fn,
            )
            n = len(bank.all_identities())
            emit({"type": "faces_scanned", "count": n})
            emit({"type": "finished", "output": f"{n} identities"})

        else:
            emit({"type": "error", "message": f"unknown job kind: {kind}"})

    except Exception as exc:  # noqa: BLE001 — report everything, crash nothing
        emit({"type": "error", "message": str(exc),
              "traceback": traceback.format_exc()})
    finally:
        emit({"type": "done"})
        try:
            conn.close()
        except Exception:
            pass
