"""
Regression guard: the helpers moved to `modules.pipeline_helpers` must remain
importable under their old names from `pipeline` itself.

This file exists to catch the obvious break: a future contributor cleans up
the import shim at the top of `pipeline.py` without realising other files
(`object_recognition.py`, possibly user scripts) still do
`from pipeline import seconds_to_mmss`.

If this test ever needs to be deleted, it is because every caller of the
legacy import has been updated and a deprecation cycle has fully completed.
Do not delete it for any other reason.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


def _shim_heavy_for_pipeline_import() -> None:
    """Pipeline.py imports torch / ultralytics / cv2 / openvino at the top.
    The conftest already shims most of these, but pipeline also imports
    `action_recognition` and `object_recognition` (siblings, not under
    `modules.`). Shim them too.
    """
    # NB: do NOT shim `object_recognition` — we want the real module so we
    # can verify it correctly imports `seconds_to_mmss` from
    # `modules.pipeline_helpers` after the duplicate removal. The real module
    # is itself shim-friendly (cv2 / ultralytics are already shimmed by
    # conftest.py, and the rest is stdlib + numpy).
    for name in (
        "action_recognition",  # heavy openvino + torch; keep shimmed
        "modules.audio_peaks",
        "modules.motion_scene_detect_optimized",
        "modules.video_cache",
        "modules.video_cutter",
        "modules.video_cutter",
        "modules.transcript",
        "modules.transcript_srt",
    ):
        sys.modules.setdefault(name, MagicMock())


@pytest.fixture(scope="module", autouse=True)
def _prepare_pipeline_imports():
    _shim_heavy_for_pipeline_import()


def test_legacy_seconds_to_mmss_still_importable():
    from pipeline import seconds_to_mmss
    assert seconds_to_mmss(75) == "01:15"


def test_legacy_collapse_runs_private_alias_still_importable():
    # `_collapse_runs` was the original private name. Keep the alias.
    from pipeline import _collapse_runs
    assert _collapse_runs(["a", "a", "b"]) == "a ×2, b ×1"


def test_legacy_subtract_forbidden_still_importable():
    from pipeline import subtract_forbidden
    assert subtract_forbidden([(0.0, 10.0)], [(3.0, 7.0)]) == [
        (0.0, 3.0),
        (7.0, 10.0),
    ]


def test_legacy_check_cancellation_still_importable():
    from pipeline import check_cancellation
    # None flag is a no-op; just verifying the symbol resolves.
    check_cancellation(None, log_fn=lambda msg: None)


def test_legacy_progress_tracker_still_importable():
    from pipeline import ProgressTracker
    tracker = ProgressTracker()
    tracker.update_progress(1, 10, "step")  # must not raise


def test_object_recognition_uses_helpers_for_seconds_to_mmss():
    # The duplicate in object_recognition.py was removed. The symbol must
    # still resolve via the helpers import.
    from object_recognition import seconds_to_mmss
    assert seconds_to_mmss(125) == "02:05"
