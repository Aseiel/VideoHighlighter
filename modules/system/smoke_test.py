"""`--smoke-test <video>`: the packaged app's riskiest paths, with no window.

Nobody on the project has a Mac, so the first person to run a mac build was a
tester — and the build took his machine down. Every multiprocessing child had
loaded the whole app (freeze_support() sat below the heavy imports), object
detection starts six of them, and the memory ran out. None of that needs a
screen to see. CI runs this against the built .app on a macOS runner:

    VideoHighlighter --smoke-test clip.mp4 --smoke-report report.json

and it checks what hurt the tester, in the frozen binary itself:

* the app gets through its imports, and ``pipeline`` imports (the 0.11.0 crash
  was an import: ultralytics → matplotlib → the macOS font list);
* a spawned child does not load the app — it is asked what it imported;
* object detection runs its workers, and the whole process tree stays under a
  memory ceiling — past it the tree is killed, so a runaway fails the job
  instead of the runner;
* the thumbnail decoder child starts and answers;
* a child does not rotate the parent's debug.log away.

Exit code 0 when every check passed, 1 when one failed. The report is JSON so
the workflow can upload it with the logs.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time

# Modules only the app's own window needs. A multiprocessing child that holds
# any of them ran main.py's imports, which is the bug this exists to catch.
APP_ONLY_MODULES = ("PySide6", "transformers", "llm.llm_chat_widget")

DEFAULT_MAX_TREE_MB = 5000       # macos-latest runners have 7 GB
DEFAULT_MAX_CHILDREN = 12        # 4 workers + Manager + resource tracker ≈ 6
DEFAULT_TIMEOUT_S = 900


def _probe_child() -> list:
    """Runs in a spawned child: which app-only modules got loaded here."""
    return sorted(m for m in APP_ONLY_MODULES if m in sys.modules)


def _arg(argv, name, default=None, cast=str):
    if name in argv:
        i = argv.index(name)
        if i + 1 < len(argv):
            return cast(argv[i + 1])
    return default


class _TreeWatch:
    """Samples this process and all its descendants; kills them past a ceiling."""

    def __init__(self, max_tree_mb: float, max_children: int):
        import psutil
        self._psutil = psutil
        self._me = psutil.Process()
        self.max_tree_mb = max_tree_mb
        self.max_children = max_children
        self.peak_tree_mb = 0.0
        self.peak_child_mb = 0.0
        self.peak_children = 0
        self.children_seen = set()
        self.tripped = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="SmokeTreeWatch")

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def _run(self):
        while not self._stop.wait(0.2):
            try:
                children = self._me.children(recursive=True)
            except Exception:
                continue
            total = self._rss_mb(self._me)
            for child in children:
                self.children_seen.add(child.pid)
                rss = self._rss_mb(child)
                total += rss
                self.peak_child_mb = max(self.peak_child_mb, rss)
            self.peak_tree_mb = max(self.peak_tree_mb, total)
            self.peak_children = max(self.peak_children, len(children))
            if total > self.max_tree_mb:
                self._trip(f"process tree reached {total:.0f} MB "
                           f"(ceiling {self.max_tree_mb:.0f} MB)", children)
            elif len(children) > self.max_children:
                self._trip(f"{len(children)} child processes "
                           f"(ceiling {self.max_children})", children)

    def _rss_mb(self, proc) -> float:
        try:
            return proc.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0

    def _trip(self, reason, children):
        if self.tripped:
            return
        self.tripped = reason
        print(f"🛑 smoke: {reason} — killing the children")
        for child in children:
            try:
                child.kill()
            except Exception:
                pass


def main(argv) -> int:
    started = time.monotonic()
    video = _arg(argv, "--smoke-test", "")
    report_path = _arg(argv, "--smoke-report", "")
    max_tree_mb = _arg(argv, "--smoke-max-tree-mb", DEFAULT_MAX_TREE_MB, float)
    max_children = _arg(argv, "--smoke-max-children", DEFAULT_MAX_CHILDREN, int)
    timeout_s = _arg(argv, "--smoke-timeout", DEFAULT_TIMEOUT_S, float)

    report = {"video": video, "frozen": bool(getattr(sys, "frozen", False)),
              "platform": sys.platform, "checks": {}, "ok": False}
    checks = report["checks"]

    def check(name, ok, detail=""):
        checks[name] = {"ok": bool(ok), "detail": detail}
        print(f"{'✅' if ok else '❌'} smoke: {name}" + (f" — {detail}" if detail else ""))

    def finish() -> int:
        report["ok"] = bool(checks) and all(c["ok"] for c in checks.values())
        report["seconds"] = round(time.monotonic() - started, 1)
        if report_path:
            try:
                with open(report_path, "w", encoding="utf-8") as fh:
                    json.dump(report, fh, indent=2)
            except Exception as e:
                print(f"⚠️ smoke: could not write the report: {e}")
        print(f"{'✅ smoke test passed' if report['ok'] else '❌ smoke test FAILED'}"
              f" in {report['seconds']}s")
        return 0 if report["ok"] else 1

    # A hang is a failure too, and one that would otherwise last until the
    # runner's own six-hour limit.
    def _deadline():
        checks["finished in time"] = {"ok": False, "detail": f"over {timeout_s:.0f}s"}
        print(f"❌ smoke: still running after {timeout_s:.0f}s")
        code = finish()
        os._exit(code or 1)
    watchdog = threading.Timer(timeout_s, _deadline)
    watchdog.daemon = True
    watchdog.start()

    if not video or not os.path.exists(video):
        check("video exists", False, repr(video))
        return finish()

    from modules.system import debug_console
    marker = f"smoke-marker-{os.getpid()}-{time.time():.0f}"
    print(marker)

    try:
        import psutil  # noqa: F401
    except Exception as e:
        check("psutil available", False, f"{type(e).__name__}: {e}")
        return finish()

    try:
        import pipeline  # noqa: F401  — run_pipeline's import, where 0.11.0 died
        check("pipeline imports", True)
    except BaseException as e:
        check("pipeline imports", False, f"{type(e).__name__}: {e}")
        return finish()

    import multiprocessing
    context = multiprocessing.get_context("spawn")
    try:
        with context.Pool(1) as pool:
            loaded = pool.apply_async(_probe_child).get(timeout=120)
        if report["frozen"]:
            check("spawned child does not load the app", not loaded,
                  f"child imported {', '.join(loaded)}" if loaded else "")
        else:
            # From source, spawn re-runs main.py as __mp_main__ by design, so
            # only a frozen build can answer this.
            checks["spawned child does not load the app"] = {
                "ok": True, "detail": f"skipped from source (child had {loaded})"}
    except BaseException as e:
        check("spawned child does not load the app", False, f"{type(e).__name__}: {e}")

    watch = _TreeWatch(max_tree_mb, max_children).start()
    try:
        import object_recognition
        work = tempfile.mkdtemp(prefix="vh-smoke-")
        objects, _bboxes = object_recognition.run_object_detection(
            video, ["person"], frame_skip=5,
            csv_file=os.path.join(work, "objects.csv"), progress_fn=None,
            device="cpu", log_fn=print)
        workers = len(watch.children_seen)
        check("object detection ran its workers",
              not watch.tripped and workers >= object_recognition.NUM_WORKERS,
              watch.tripped or f"{workers} child processes seen, "
                               f"{len(objects)} seconds with objects")
    except BaseException as e:
        check("object detection ran its workers", False,
              watch.tripped or f"{type(e).__name__}: {e}")

    try:
        from video_ai_editor import thumbnail_decoder
        decoder = thumbnail_decoder.acquire(video)
        out = os.path.join(tempfile.mkdtemp(prefix="vh-smoke-"), "thumb.jpg")
        try:
            ok = decoder.extract(1000, 90, False, out)
        finally:
            thumbnail_decoder.release(video)
        check("thumbnail decoder answers",
              ok and os.path.exists(out) and not watch.tripped,
              watch.tripped or f"decoder={decoder.decoder}, restarts={decoder.restarts}")
    except BaseException as e:
        check("thumbnail decoder answers", False, f"{type(e).__name__}: {e}")
    watch.stop()

    report["memory"] = {"peak_tree_mb": round(watch.peak_tree_mb),
                        "peak_child_mb": round(watch.peak_child_mb),
                        "peak_children": watch.peak_children,
                        "children_seen": len(watch.children_seen),
                        "ceiling_tree_mb": max_tree_mb}
    check("memory stayed under the ceiling", not watch.tripped,
          watch.tripped or f"peak {watch.peak_tree_mb:.0f} MB across "
                           f"{watch.peak_children} children at once")

    try:
        sys.stdout.flush()
        with open(debug_console.log_file_path(), encoding="utf-8",
                  errors="replace") as fh:
            kept = marker in fh.read()
        check("debug.log survived the children", kept,
              "" if kept else "a child rotated the live log away")
    except Exception as e:
        check("debug.log survived the children", False, f"{type(e).__name__}: {e}")

    watchdog.cancel()
    return finish()
