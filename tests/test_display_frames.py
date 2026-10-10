"""Shrinking video frames off the UI thread (video_ai_editor/display_frames.py).

The preview used to paint every frame at the source's full resolution on the UI
thread; it now paints a copy shrunk on the raw planes. These pin the two things
that copy must get right: the columns it keeps (a crop that started one column
off would put every live box beside its subject) and the frame metadata the
video item reads (a shrunk frame without its start time breaks the trackers
that time-match detections against it).
"""

from __future__ import annotations

import numpy as np
import pytest

# PySide6 is not in requirements-dev, so like the other Qt tests this skips in CI.
pytest.importorskip("PySide6", reason="Qt not available in this environment")

from PySide6.QtCore import QSize                                       # noqa: E402
from PySide6.QtMultimedia import QVideoFrame, QVideoFrameFormat       # noqa: E402

from video_ai_editor.display_frames import (                          # noqa: E402
    downscale_nv12_planes, shrink_for_display,
)


@pytest.fixture(autouse=True)
def _real_cv2(monkeypatch):
    """The suite shims cv2 with a MagicMock; these tests are about real pixels."""
    import sys
    from tests.conftest import real_opencv
    cv2 = real_opencv()
    if cv2 is None:
        pytest.skip("OpenCV not installed")
    monkeypatch.setitem(sys.modules, "cv2", cv2)


def _nv12(width: int, height: int, luma_of_column) -> QVideoFrame:
    """An NV12 frame whose luma is a function of the column; chroma neutral."""
    fmt = QVideoFrameFormat(QSize(width, height), QVideoFrameFormat.PixelFormat.Format_NV12)
    frame = QVideoFrame(fmt)
    assert frame.map(QVideoFrame.MapMode.WriteOnly)
    y = np.frombuffer(frame.bits(0), np.uint8).reshape(height, frame.bytesPerLine(0))
    uv = np.frombuffer(frame.bits(1), np.uint8).reshape(height // 2, frame.bytesPerLine(1))
    y[:, :width] = np.array([luma_of_column(x) for x in range(width)], np.uint8)[None, :]
    uv[:] = 128
    frame.unmap()
    frame.setStartTime(1_234_000)
    return frame


def test_whole_frame_keeps_its_left_to_right_order():
    frame = _nv12(256, 64, lambda x: x)            # luma rises left to right
    y, uv = downscale_nv12_planes(frame, 64)
    assert y.shape == (16, 64) and uv.shape == (8, 32, 2)
    assert y[0, 0] < 8 and y[0, -1] > 247


@pytest.mark.parametrize("span,lo,hi", [
    ((0, 128), 0, 128),        # left half
    ((128, 256), 128, 256),    # right half
    ((64, 192), 64, 192),      # centre
])
def test_x_span_keeps_exactly_those_columns(span, lo, hi):
    frame = _nv12(256, 64, lambda x: x)
    y, _ = downscale_nv12_planes(frame, 32, x_span=span)
    assert y.shape[1] == 32
    # INTER_AREA averages 4 source columns per output column.
    assert abs(float(y[0, 0]) - (lo + 1.5)) <= 1
    assert abs(float(y[0, -1]) - (hi - 2.5)) <= 1


def test_left_half_shorthand_matches_the_span():
    frame = _nv12(256, 64, lambda x: x)
    a, _ = downscale_nv12_planes(frame, 32, crop_left_half=True)
    b, _ = downscale_nv12_planes(frame, 32, x_span=(0, 128))
    assert np.array_equal(a, b)


def test_shrunk_frame_carries_the_source_time():
    frame = _nv12(2048, 64, lambda x: x % 256)
    out = shrink_for_display(frame, 512)
    assert out is not frame
    assert out.width() == 512
    assert out.startTime() == 1_234_000


def test_a_frame_already_small_enough_passes_through():
    frame = _nv12(512, 64, lambda x: x % 256)
    assert shrink_for_display(frame, 512) is frame
