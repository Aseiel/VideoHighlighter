"""Background jobs for the MCP tools: queue, run one at a time, report status.

A tool call has to answer in seconds, and an analysis run or a whole-video
search index takes minutes to hours. So a start tool enqueues a job and returns
its id; the client polls ``get_job``. One worker thread runs the queue serially —
the GUI refuses a second concurrent run for the same reason (models in memory,
one GPU) — and later jobs wait as ``queued``.

Nothing here knows about the engine; a job is a callable that receives its
:class:`Job` and reports through ``log`` / ``set_progress`` / ``check_cancel``.
"""
from __future__ import annotations

import queue
import threading
import time
import traceback
import uuid
from collections import deque
from typing import Any, Callable, Optional

QUEUED, RUNNING, DONE, FAILED, CANCELLED = (
    "queued", "running", "done", "failed", "cancelled")
FINISHED = {DONE, FAILED, CANCELLED}

LOG_KEEP = 300          # log lines a job keeps for get_job's tail
FINISHED_KEEP = 50      # finished jobs kept for get_job / list_jobs


class JobCancelled(Exception):
    """Raised from a progress callback to stop work that has no cancel flag."""


class Job:
    def __init__(self, kind: str, params: dict, fn: Callable[["Job"], Any]):
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.params = params
        self.fn = fn
        self.status = QUEUED
        self.created = time.time()
        self.started: Optional[float] = None
        self.finished: Optional[float] = None
        self.progress: dict = {}
        self.result: Any = None
        self.error: Optional[str] = None
        self.cancel_flag = threading.Event()
        self._log: deque = deque(maxlen=LOG_KEEP)
        self._lock = threading.Lock()

    # -- called from the job's own thread ---------------------------------

    def log(self, msg) -> None:
        text = str(msg)
        print(f"[mcp job {self.id}] {text}")
        with self._lock:
            self._log.extend(text.splitlines() or [""])

    def set_progress(self, current, total, task: str = "", detail: str = "") -> None:
        with self._lock:
            self.progress = {"current": current, "total": total,
                             "task": task, "detail": detail}

    def check_cancel(self) -> None:
        if self.cancel_flag.is_set():
            raise JobCancelled()

    # -- read side ---------------------------------------------------------

    def snapshot(self, log_lines: int = 20) -> dict:
        with self._lock:
            tail = list(self._log)[-log_lines:] if log_lines > 0 else []
            progress = dict(self.progress)
        out = {
            "job_id": self.id,
            "kind": self.kind,
            "status": self.status,
            "params": self.params,
            "progress": progress,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "log_tail": tail,
        }
        if self.status == DONE:
            out["result"] = self.result
        if self.error:
            out["error"] = self.error
        return out


class JobManager:
    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._queue: "queue.Queue[Job]" = queue.Queue()
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self.current: Optional[Job] = None

    def submit(self, kind: str, params: dict, fn: Callable[[Job], Any]) -> Job:
        job = Job(kind, params, fn)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._prune()
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._run_forever, name="mcp-jobs", daemon=True)
                self._worker.start()
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        with self._lock:
            return [self._jobs[i] for i in self._order]

    def cancel(self, job_id: str) -> Optional[Job]:
        """Trip the flag. A queued job is dropped when its turn comes; a running
        one stops at the engine's next check. Never kills the thread."""
        job = self.get(job_id)
        if job is not None and job.status not in FINISHED:
            job.cancel_flag.set()
            if job.status == QUEUED:
                job.status = CANCELLED
                job.finished = time.time()
        return job

    def wait(self, job_id: str, timeout: float = 10.0) -> Optional[Job]:
        """Block until the job finishes (tests; never called from a tool)."""
        deadline = time.monotonic() + timeout
        job = self.get(job_id)
        while job is not None and job.status not in FINISHED:
            if time.monotonic() > deadline:
                break
            time.sleep(0.01)
        return job

    def _prune(self) -> None:
        finished = [i for i in self._order if self._jobs[i].status in FINISHED]
        for job_id in finished[:max(0, len(finished) - FINISHED_KEEP)]:
            self._order.remove(job_id)
            del self._jobs[job_id]

    def _run_forever(self) -> None:
        while True:
            job = self._queue.get()
            if job.status == CANCELLED:
                continue
            self.current = job
            job.status = RUNNING
            job.started = time.time()
            try:
                job.result = job.fn(job)
                job.status = CANCELLED if job.cancel_flag.is_set() else DONE
            except JobCancelled:
                job.status = CANCELLED
            except Exception as e:  # noqa: BLE001 — report it, keep serving
                traceback.print_exc()
                job.error = f"{type(e).__name__}: {e}"
                job.status = FAILED
            finally:
                job.finished = time.time()
                self.current = None
