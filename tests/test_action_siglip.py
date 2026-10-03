"""The SigLIP2 action runner's pure parts: windows, seconds, crops, trust.

No model, no video: the encoder, the head's ONNX session and YOLOX are what
the real run adds, and each is tested where it lives.
"""
import json

import numpy as np
import pytest

from modules.vision import action_siglip as A


def test_windows_sample_like_the_trainer():
    """4 frames at the centres of 4 equal parts of a 5 s window, a window every
    2.5 s, and the tail of the video gets a window of its own."""
    w = A.window_frames(total_frames=300, fps=25, k=4)   # 12 s
    assert w[0] == [15, 46, 78, 109]                     # 125 frames / 4
    starts = [f[0] - 15 for f in w]
    # 2.5 s = 62.5 -> 62; a window at 186 would run past the end, so the
    # tail window ends on the last frame instead
    assert starts == [0, 62, 124, 300 - 125]


def test_a_short_video_is_one_window():
    assert A.window_frames(total_frames=40, fps=25, k=4) == [[5, 15, 25, 35]]


def test_every_second_is_reported_once():
    """Overlapping windows each speak for their own stretch, so a second is
    never claimed twice and none is left out — first to last."""
    windows = A.window_frames(total_frames=25 * 60, fps=25, k=4)
    seen = []
    for w in range(len(windows)):
        seen += list(A.window_seconds(windows, w, 25, last_second=60))
    assert seen == list(range(60))


def test_one_region_per_person_with_margin():
    """The two largest people of the busiest frame, each followed over the
    window by overlap and widened by the margin; smaller people are left out."""
    big = [(100, 50, 300, 450), (100, 60, 320, 450)]
    second = [(400, 100, 500, 400), (410, 100, 510, 400)]
    small = (560, 10, 580, 40)
    frames = [[big[0], second[0], small], [big[1], second[1]]]
    regions = A.person_regions(frames, width=640, height=480)
    assert len(regions) == 2
    x1, y1, x2, y2 = regions[0]
    assert (x1, x2) == (int(100 - 0.2 * 220), int(320 + 0.2 * 220))
    assert y2 == 480                                    # clipped to the frame


def test_nobody_found_means_the_whole_frame():
    assert A.person_regions([[], []], width=640, height=480) == [(0, 0, 640, 480)]


class _Head(A.ActionHead):
    """detected() without an ONNX session."""

    def __init__(self, thresholds, pairs=()):
        self.thresholds, self.pairs = thresholds, list(pairs)


def test_only_trusted_actions_are_detected():
    head = _Head([0.5, None, 0.9])
    got = head.detected(np.array([[0.6, 0.99, 0.8]]))[0]
    assert got.tolist() == [True, False, False]         # None = never trusted


def test_a_trusted_pair_adds_both_actions():
    head = _Head([0.9, 0.9], pairs=[(0, 1, 0.4)])
    assert head.detected(np.array([[0.5, 0.45]]))[0].tolist() == [True, True]
    assert head.detected(np.array([[0.5, 0.30]]))[0].tolist() == [False, False]


def test_heads_are_found_by_kind_and_encoder(tmp_path, monkeypatch):
    def head(name, **meta):
        d = tmp_path / name
        d.mkdir()
        (d / A.HEAD_MODEL).write_bytes(b"")
        (d / A.HEAD_META).write_text(json.dumps({"kind": A.HEAD_KIND, **meta}))
        return str(d)

    ours = head("ours", encoder="enc-a")
    head("other-encoder", encoder="enc-b")
    (tmp_path / "not-a-head").mkdir()
    monkeypatch.setattr(A, "_head_dirs", lambda: sorted(str(p) for p in tmp_path.iterdir()))
    assert A.find_heads("enc-a") == [ours]


def test_a_head_json_of_another_kind_is_refused(tmp_path):
    (tmp_path / A.HEAD_MODEL).write_bytes(b"")
    (tmp_path / A.HEAD_META).write_text(json.dumps({"kind": "detector"}))
    with pytest.raises(ValueError):
        A.read_head_meta(str(tmp_path))
