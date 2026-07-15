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


def _force_utf8_stdio() -> None:
    """Stop emoji in library prints from killing the child.

    The engine prints emoji freely (e.g. clip_prefilter.load()'s "🔧"). A spawned
    child on Windows gets a cp1252 stdout, so the first such print raises
    UnicodeEncodeError and takes the whole job down — which surfaced as a CLIP
    search dying the moment the model loaded. main.py avoids this because
    debug_console.install() reconfigures stdout at import; the worker has no
    such hook, so do it here.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _jpeg_b64(frame, max_w: int = 320) -> str:
    """Small JPEG thumbnail of a BGR frame, for the results list."""
    import base64

    import cv2

    h, w = frame.shape[:2]
    if w > max_w:
        scale = max_w / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                           interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def _vision_search(job: dict, emit, log_fn, progress_fn, cancel_evt) -> None:
    """Find moments in a video matching a text query.

    Three engines, mirroring the Qt chat widget's search modes:
      clip      — CLIP ranks every sampled frame on the GPU. Fast, no VLM.
      clip_llm  — CLIP ranks, then the vision LLM confirms the top K.
      llm       — the vision LLM checks every sampled frame. Slow, most capable.

    Implemented here rather than calling clip_prefilter.scan_video because that
    helper prints progress and can't be cancelled — no good for a long job
    driven from a UI.
    """
    import cv2

    query = job["query"]
    video_path = job["video_path"]
    mode = job.get("mode", "clip")
    interval = float(job.get("interval", 1.0))
    top_k = int(job.get("top_k", 30))
    threshold = float(job.get("threshold", 0.5))
    device = job.get("clip_device", "GPU")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        emit({"type": "error", "message": f"Could not open video: {video_path}"})
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps * interval)))
    expected = max(1, total // step)

    scored: list[tuple[float, float, Any]] = []  # (timestamp, score, frame)

    if mode in ("clip", "clip_llm"):
        from llm.clip_prefilter import ClipFramePrefilter

        err = ClipFramePrefilter.import_error()
        if err:
            cap.release()
            emit({"type": "error", "message": f"CLIP unavailable: {err}"})
            return

        log_fn(f"🔍 CLIP search for '{query}' on {device} (every {interval}s)")
        pf = ClipFramePrefilter(device=device)
        pf.load()
        pf.set_query(query)
        log_fn(f"   CLIP ready on {pf.device}")

        batch_frames, batch_ts = [], []
        idx = 0
        done = 0

        def flush():
            nonlocal done
            if not batch_frames:
                return
            for ts, sc, fr in zip(batch_ts, pf.score_frames_bgr(batch_frames),
                                  batch_frames):
                scored.append((ts, float(sc), fr))
            done += len(batch_frames)
            progress_fn(done, expected, "CLIP scan", f"{done}/{expected} frames")
            batch_frames.clear()
            batch_ts.clear()

        # grab() skips decoding; only sampled frames are retrieved.
        while not cancel_evt.is_set():
            if not cap.grab():
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    batch_frames.append(frame)
                    batch_ts.append(idx / fps)
                    if len(batch_frames) >= 16:
                        flush()
            idx += 1
        if not cancel_evt.is_set():
            flush()
        cap.release()

        scored.sort(key=lambda r: -r[1])
        if mode == "clip":
            hits = [r for r in scored if r[1] >= threshold][:top_k]
            log_fn(f"✅ {len(hits)} frame(s) above {threshold:.2f}")
            emit({"type": "vision_results", "results": [
                {"timestamp": ts, "score": sc, "thumb": _jpeg_b64(fr),
                 "analysis": ""}
                for ts, sc, fr in hits
            ]})
            emit({"type": "finished", "output": f"{len(hits)} match(es)"})
            return

        # clip_llm: keep the best K for the VLM to confirm.
        scored = scored[:top_k]
        log_fn(f"🤖 Confirming top {len(scored)} frame(s) with the vision model…")

    else:
        # Pure LLM: sample the video directly, no ranking pass.
        idx = 0
        while not cancel_evt.is_set():
            if not cap.grab():
                break
            if idx % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    scored.append((idx / fps, 0.0, frame))
            idx += 1
        cap.release()
        log_fn(f"🤖 Checking {len(scored)} frame(s) with the vision model…")

    # VLM confirmation pass (clip_llm and llm).
    from llm.llm_module import LLMModule

    llm = LLMModule(backend=job.get("backend", "ollama"),
                    model=job.get("model", "llava"))
    llm.load()

    results = []
    for i, (ts, sc, frame) in enumerate(scored, 1):
        if cancel_evt.is_set():
            break
        progress_fn(i, len(scored), "Vision check", f"{i}/{len(scored)}")
        try:
            # Vision mode is selected by passing frame_base64; the frame goes at
            # full resolution because downscaling hurts VLM accuracy (the
            # thumbnail below is only for the results list).
            answer = llm.query(
                f"Is there {query} in this image? Answer yes or no, then explain briefly.",
                frame_base64=_jpeg_b64(frame, max_w=10_000),
                free_chat_mode=True,
            )
        except Exception as exc:  # noqa: BLE001 — one bad frame shouldn't end it
            log_fn(f"⚠️ frame {ts:.1f}s: {exc}")
            continue
        text = str(answer)
        hit = text.strip().lower().startswith("yes") or " yes" in text.lower()[:40]
        if hit:
            results.append({"timestamp": ts, "score": sc,
                            "thumb": _jpeg_b64(frame), "analysis": text})
            emit({"type": "vision_hit", "timestamp": ts, "analysis": text})

    results.sort(key=lambda r: r["timestamp"])
    emit({"type": "vision_results", "results": results})
    emit({"type": "finished", "output": f"{len(results)} match(es)"})


def run_job(conn, job: dict, cancel_evt, pause_evt, preview_flag) -> None:
    """Entry point for the child process. Never raises into the parent."""
    _install_path()
    _force_utf8_stdio()

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
                video_urls=job.get("video_urls"),
            )
            paths = [r.get("filepath") for r in (results or []) if r.get("filepath")]
            if cancel_evt.is_set():
                emit({"type": "cancelled"})
            else:
                emit({"type": "downloaded", "paths": paths})
                emit({"type": "finished", "output": f"{len(paths)} file(s)"})

        elif kind == "vision_search":
            _vision_search(job, emit, log_fn, progress_fn, cancel_evt)

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
