"""The motion pass ends when its frame reader does, error or not.

The reader is a thread feeding a queue; the pass waits on the queue for the
reader's end-of-video sentinel. Two ways that sentinel never came:

- the reader raised (a damaged frame, a resize error, memory), and the thread
  died before sending it, with nothing in the log;
- the queue was full when the video ended and a GPU batch took longer than the
  reader's single 1 s attempt to send it.

Either way the pass waited forever, with the run's bar parked at 10%, which is
where the pipeline shows this step.

Each case runs in a fresh interpreter: the pass needs the real torch and OpenCV
(conftest replaces both with mocks for the rest of the suite), and a hang must
fail the test on a timeout rather than hold the suite.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

_SCRIPT = textwrap.dedent('''
    import json, sys
    import numpy as np
    import cv2
    from modules.segments import motion_scene_detect_optimized as msd

    FRAMES, FAIL_AT = int(sys.argv[1]), int(sys.argv[2])

    class FakeCapture:
        """A short grey video that can fail partway through a read."""
        def __init__(self): self.pos = 0
        def isOpened(self): return True
        def get(self, prop):
            if prop == cv2.CAP_PROP_FPS: return 30.0
            if prop == cv2.CAP_PROP_FRAME_COUNT: return float(FRAMES)
            return 0.0
        def set(self, *a): return True
        def read(self):
            if self.pos == FAIL_AT: raise MemoryError("simulated")
            if self.pos >= FRAMES: return False, None
            self.pos += 1
            return True, np.full((48, 64, 3), (self.pos * 7) % 255, np.uint8)
        def release(self): pass

    cv2.VideoCapture = lambda *a, **k: FakeCapture()
    try:
        result = msd.detect_scenes_motion_optimized("fake.mp4", device="cpu",
                                                    debug=False)
        out = {"result": [len(r) for r in result]}
    except msd.FrameReadError as exc:
        out = {"error": "FrameReadError", "message": str(exc)}
    except Exception as exc:
        out = {"error": type(exc).__name__, "message": str(exc)}
    print("RESULT " + json.dumps(out))
''')


def _have_real_deps() -> bool:
    probe = subprocess.run([sys.executable, "-c", "import torch, cv2, tqdm"],
                           cwd=REPO, capture_output=True, timeout=120)
    return probe.returncode == 0


pytestmark = pytest.mark.skipif(not _have_real_deps(),
                                reason="needs the real torch, OpenCV and tqdm")


def _run(frames: int, fail_at: int = -1, timeout: int = 90) -> dict:
    try:
        done = subprocess.run(
            [sys.executable, "-c", _SCRIPT, str(frames), str(fail_at)],
            cwd=REPO, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    except subprocess.TimeoutExpired:
        pytest.fail("the motion pass hung")
    lines = [l for l in done.stdout.splitlines() if l.startswith("RESULT ")]
    assert lines, f"no result\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}"
    return json.loads(lines[-1][len("RESULT "):])


def test_a_reader_error_ends_the_pass_and_says_where():
    out = _run(frames=200, fail_at=37)
    assert out.get("error") == "FrameReadError", out
    assert "after frame 36" in out["message"] and "MemoryError" in out["message"]


def test_a_clean_video_still_finishes():
    out = _run(frames=120)
    assert "error" not in out, out
    assert len(out["result"]) == 3


def test_the_error_is_not_read_as_a_cancel():
    # pipeline.py returns without a word on a RuntimeError from this step.
    src = (REPO / "modules/segments/motion_scene_detect_optimized.py").read_text(
        encoding="utf-8")
    assert "class FrameReadError(Exception):" in src
