"""Analysis-region helpers.

Utilities for restricting visual analysis to part of a frame. This build always
analyses the full frame, so these are full-frame stubs that keep the shared
motion-detection code importing and running cleanly.
"""

from __future__ import annotations

FULL_FRAME = "full"


def normalize_analysis_region(region: str | None) -> str:
    """Return a known analysis-region identifier.

    This build always analyses the full frame, so the request is ignored and
    ``FULL_FRAME`` is returned.
    """
    return FULL_FRAME


def crop_frame_for_analysis(frame, region: str | None = None):
    """Return the frame unchanged — the full frame is analysed."""
    return frame
