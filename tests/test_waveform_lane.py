"""The AUDIO WAVEFORM lane lines a sound up with the playhead.

It used to be one bin per second or so (a fixed 1000-2000 points for a whole
file), each drawn as a line at the bin's centre: a slap at 3:21.9 showed at
3:22.4. Bins are now 50 ms and painted across their own time span.
"""
from __future__ import annotations

import numpy as np
import pytest

from video_ai_editor import waveform as wf

RATE = 16000


def _click_at(seconds, total=10.0):
    audio = np.zeros(int(total * RATE), np.int16)
    i = int(seconds * RATE)
    audio[i:i + 160] = 20000          # 10 ms of loud
    return audio


def test_bins_are_a_twentieth_of_a_second_whatever_the_length():
    for total in (10.0, 600.0):
        bins = wf.bin_audio(np.zeros(int(total * RATE), np.int16), RATE)
        assert len(bins) == pytest.approx(total * wf.BINS_PER_SECOND, abs=1)


def test_a_click_lands_in_the_bin_of_its_own_time():
    bins = wf.bin_audio(_click_at(3.21), RATE)
    loud = [i for i, b in enumerate(bins) if b[1] > 0.5]
    assert loud == [int(3.21 * wf.BINS_PER_SECOND)]
    lo, hi, rms = bins[loud[0]]
    assert -1.0 <= lo <= hi <= 1.0 and rms > 0


def test_a_short_last_bin_is_kept():
    bins = wf.bin_audio(np.ones(RATE + 100, np.int16), RATE)
    assert len(bins) == wf.BINS_PER_SECOND + 1


def test_a_fixed_count_still_works():
    assert len(wf.bin_audio(np.ones(RATE * 10, np.int16), RATE, num_points=1000)) == 1000


def test_an_old_cache_is_coarse_and_a_new_one_is_not():
    assert wf.is_coarse([(0, 0, 0)] * 2000, 2160.0)          # 36 min at 2000 points
    assert not wf.is_coarse([(0, 0, 0)] * 2160 * 20, 2160.0)
    assert not wf.is_coarse([], 2160.0)
    assert not wf.is_coarse([(0, 0, 0)] * 10, 0)


# ── the lane ────────────────────────────────────────────────────────────────

pytest.importorskip("PySide6")
from PySide6.QtCore import QRectF  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication, QStyleOptionGraphicsItem  # noqa: E402

from video_ai_editor.waveform_lane import WaveformLane, merge_bins  # noqa: E402


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


def _paint(lane, width, height, sx=1.0):
    img = QImage(int(width * sx), int(height), QImage.Format.Format_ARGB32)
    img.fill(QColor(0, 0, 0, 0))
    p = QPainter(img)
    p.scale(sx, 1.0)
    opt = QStyleOptionGraphicsItem()
    opt.exposedRect = QRectF(0, 0, width, height)
    lane.paint(p, opt)
    p.end()
    return img


def test_a_bin_is_painted_across_its_own_time(app):
    bins = wf.bin_audio(_click_at(3.21), RATE)
    pps = 100.0                                         # 5 px per 50 ms bin
    lane = WaveformLane(bins, 10.0, pps, 1000, 40, [QColor(0, 0, 255, 255)] * 256)
    img = _paint(lane, 1000, 40)
    # The quiet bins draw a one-pixel line through the middle (row 20); the
    # click (0.61 of full scale) reaches up to row 10.
    top = [x for x in range(img.width()) if QColor.fromRgba(img.pixel(x, 14)).alpha() > 0]
    assert top and min(top) == pytest.approx(3.20 * pps, abs=1)
    assert max(top) == pytest.approx(3.25 * pps, abs=1)


def test_zoomed_out_bins_merge_per_screen_pixel(app):
    mins = np.zeros(1000)
    maxs = np.zeros(1000)
    maxs[537] = 0.9
    starts, m0, m1, lv = merge_bins(mins, maxs, np.zeros(1000), 10)
    assert len(starts) == 100 and m1[53] == pytest.approx(0.9) and m1.sum() == pytest.approx(0.9)
    # 2000 bins in 100 screen pixels still paints, and the click survives.
    bins = [(0.0, 0.0, 0.0)] * 2000
    bins[1234] = (-0.9, 0.9, 0.5)
    lane = WaveformLane(bins, 100.0, 10.0, 1000, 40, [QColor(0, 0, 255, 255)] * 256)
    img = _paint(lane, 1000, 40, sx=0.1)
    top = [x for x in range(img.width()) if QColor.fromRgba(img.pixel(x, 6)).alpha() > 0]
    assert top and abs(top[0] - 61.7) <= 2


def test_the_lane_takes_no_clicks(app):
    from PySide6.QtCore import Qt
    lane = WaveformLane([(0, 0, 0)] * 10, 1.0, 50, 50, 40, [])
    assert lane.acceptedMouseButtons() == Qt.MouseButton.NoButton
