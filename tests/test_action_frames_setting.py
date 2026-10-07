"""The "Frames per window" setting reaches every action run and the cache key.

4 or 8 frames per window for actions by name
(docs/plans/2026-10-07-action-frames.md). The two find different actions, so a
cache made at one must not be reused for the other; caches made at the
default, 4, keep their key.
"""
from __future__ import annotations

import ast
from pathlib import Path

from modules.media.video_cache import build_analysis_cache_params

REPO = Path(__file__).resolve().parent.parent


def _params(**gui):
    base = {"interesting_actions": ["punching person (boxing)"], "action_points": 5}
    base.update(gui)
    return build_analysis_cache_params(base, {}, sample_rate=5, video_duration=60.0)


def test_eight_frames_has_its_own_cache_key():
    assert _params(action_frames=8)["action_frames"] == 8


def test_four_frames_keeps_the_key_caches_already_have():
    assert "action_frames" not in _params(action_frames=4)
    assert _params(action_frames=4) == _params()


def _passes_frames(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", getattr(node.func, "id", None))
                == "run_action_detection_siglip"):
            return "frames_per_window" in {k.arg for k in node.keywords}
    return False


def test_every_action_run_is_given_the_setting():
    for rel in ("pipeline.py", "modules/report/analysis_ondemand.py",
                "video_ai_editor/bbox_overlay.py"):
        assert _passes_frames(REPO / rel), f"{rel} runs actions without frames_per_window"
