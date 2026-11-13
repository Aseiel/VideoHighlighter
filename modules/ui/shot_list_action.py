"""'Export shot list…' for one video, off the GUI thread.

The detection itself is ``modules/segments/shot_boundaries.py`` and the files
are ``modules/segments/shot_list_export.py``; this is only the worker and the
two lines the user reads. It writes ``<video>_shots.edl`` and
``<video>_shots.txt`` beside the video, the way subtitles land beside it.
"""
from __future__ import annotations

import os
import threading

from PySide6.QtCore import QThread, Signal


class ShotListWorker(QThread):
    log = Signal(str)
    progress = Signal(int, int, str, str)
    done = Signal(bool)

    def __init__(self, video_path: str, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def run(self) -> None:
        from modules.segments import shot_boundaries, shot_list_export
        name = os.path.basename(self.video_path)
        try:
            shots = shot_boundaries.detect_shots(
                self.video_path, progress_fn=self.progress.emit,
                should_cancel=self._cancel.is_set, log_fn=self.log.emit)
            edl = shot_list_export.write_edl(shots)
            txt = shot_list_export.write_text(shots)
        except shot_boundaries.ShotDetectionCancelled:
            self.log.emit(f"⏹️ Shot list cancelled: {name}")
            self.done.emit(False)
            return
        except Exception as e:  # noqa: BLE001 - reported, not raised into Qt
            print(f"shot_list_action: {self.video_path}: {type(e).__name__}: {e}")
            self.log.emit(f"⚠️ Could not list the shots of {name}: {e}")
            self.done.emit(False)
            return
        self.log.emit(f"✅ Shot list: {os.path.basename(edl)} and "
                      f"{os.path.basename(txt)}, beside the video")
        if not shots.exact_times:
            self.log.emit("ℹ️ This file does not carry per-frame timestamps; "
                          "the times are frame number / frame rate")
        self.done.emit(True)


def export_shot_list(gui, video_path: str):
    """Start the export for ``video_path``; returns the worker (kept on
    ``gui`` so it is not collected mid-run), or None when one is running."""
    running = getattr(gui, "_shot_list_worker", None)
    if running is not None and running.isRunning():
        gui.append_log("⚠️ A shot list is already being made.")
        return None
    worker = ShotListWorker(video_path, gui)
    worker.log.connect(gui.append_log)
    gui._shot_list_worker = worker
    gui.append_log(f"🎬 Listing the shots of {os.path.basename(video_path)}…")
    worker.start()
    return worker
