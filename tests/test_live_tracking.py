"""Per-frame tracking between live detections (video_ai_editor/live_tracking.py).

The live face detector runs about twice a second; without tracking its box sat
still between passes and trailed a moving subject by up to ~0.9 s. These pin
the three things that fix it: the tracker follows a moving subject frame to
frame, a late detection corrects the track without dragging it back to where
the subject *was*, and a subject that disappears stops being drawn.
"""

from __future__ import annotations

import numpy as np

from video_ai_editor.live_tracking import LOST, TRACKING, LiveBoxTracker

W, H, S = 640, 360, 48          # frame size, subject size
FPS = 30.0


def _scene(seed=0):
    rng = np.random.default_rng(seed)
    background = rng.integers(0, 256, (H, W)).astype(np.uint8)
    background = ((background.astype(np.float32)
                   + np.roll(background, 1, 0) + np.roll(background, 1, 1)) / 3).astype(np.uint8)
    subject = rng.integers(0, 256, (S, S)).astype(np.uint8)
    return background, subject


def _frame(background, subject, x, y, visible=True):
    f = background.copy()
    if visible:
        f[int(y):int(y) + S, int(x):int(x) + S] = subject
    return f


def _path(i):
    """Subject's top-left at frame i: ~120 px/s to the right, a little down."""
    return 100 + 4.0 * i, 120 + 1.0 * i


def _box(x, y):
    return (x, y, x + S, y + S)


def _centre_err(tb, x, y):
    return np.hypot((tb[0] + tb[2]) / 2 - (x + S / 2), (tb[1] + tb[3]) / 2 - (y + S / 2))


def test_follows_a_moving_subject_between_detections():
    bg, subj = _scene()
    tr = LiveBoxTracker()
    x0, y0 = _path(0)
    tr.step(_frame(bg, subj, x0, y0), 0.0)
    tr.correct([(_box(x0, y0), {"name": "a"})], 0.0)

    for i in range(1, 30):                       # one second, no further detections
        x, y = _path(i)
        tr.step(_frame(bg, subj, x, y), i / FPS)

    boxes = tr.boxes()
    assert len(boxes) == 1
    assert boxes[0]["name"] == "a"
    assert boxes[0]["state"] == TRACKING
    # The frozen detection box would now be 116 px behind; the track is on it.
    assert _centre_err(boxes[0]["bbox"], *_path(29)) < 6


def test_late_detection_corrects_without_snapping_back():
    bg, subj = _scene(1)
    tr = LiveBoxTracker()
    x0, y0 = _path(0)
    tr.step(_frame(bg, subj, x0, y0), 0.0)
    tr.correct([(_box(x0, y0), {})], 0.0)
    for i in range(1, 20):
        tr.step(_frame(bg, subj, *_path(i)), i / FPS)

    # A detection of frame 10 arrives now, at frame 19. Snapping to it would put
    # the box 36 px behind the subject.
    tr.correct([(_box(*_path(10)), {})], 10 / FPS)
    (b,) = tr.boxes()
    assert _centre_err(b["bbox"], *_path(19)) < 6
    assert tr.errors[-1] < 0.15                  # track was within 15% of a box at frame 10


def test_occluded_subject_goes_lost_then_stops_drawing():
    bg, subj = _scene(2)
    tr = LiveBoxTracker()
    x0, y0 = _path(0)
    tr.step(_frame(bg, subj, x0, y0), 0.0)
    tr.correct([(_box(x0, y0), {})], 0.0)
    for i in range(1, 10):
        tr.step(_frame(bg, subj, *_path(i)), i / FPS)
    assert tr.boxes()[0]["state"] == TRACKING

    for i in range(10, 30):                      # subject gone
        tr.step(_frame(bg, subj, *_path(i), visible=False), i / FPS)
    assert tr.boxes() == []                      # coasted briefly, then hidden
    assert tr._tracks[0].state == LOST

    # It comes back and the detector sees it: re-anchored, drawn again.
    x, y = _path(30)
    tr.step(_frame(bg, subj, x, y), 30 / FPS)
    tr.correct([(_box(x, y), {})], 30 / FPS)
    tr.step(_frame(bg, subj, *_path(31)), 31 / FPS)
    (b,) = tr.boxes()
    assert b["state"] == TRACKING


def test_detector_ending_a_track():
    bg, subj = _scene(3)
    tr = LiveBoxTracker()
    tr.step(_frame(bg, subj, 100, 100), 0.0)
    tr.correct([(_box(100, 100), {})], 0.0)
    for k in range(1, 4):                        # three detector passes that don't see it
        tr.step(_frame(bg, subj, 100, 100), k * 0.5)
        tr.correct([], k * 0.5)
    assert tr.boxes() == []


def test_seek_drops_tracks():
    bg, subj = _scene(4)
    tr = LiveBoxTracker()
    tr.step(_frame(bg, subj, 100, 100), 5.0)
    tr.correct([(_box(100, 100), {})], 5.0)
    tr.step(_frame(bg, subj, 100, 100), 2.0)     # backwards seek
    assert tr.boxes() == []


def test_lost_track_is_reanchored_by_identity_not_duplicated():
    bg, subj = _scene(5)
    tr = LiveBoxTracker(identity_key="identity_id")
    tr.step(_frame(bg, subj, 100, 100), 0.0)
    tr.correct([(_box(100, 100), {"identity_id": "p1"})], 0.0)
    for i in range(1, 20):                       # gone long enough to be hidden
        tr.step(_frame(bg, subj, 100, 100, visible=False), i / FPS)
    assert tr.boxes() == []

    # Reappears far from where it was lost — no overlap, same identity.
    tr.step(_frame(bg, subj, 450, 220), 20 / FPS)
    tr.correct([(_box(450, 220), {"identity_id": "p1"})], 20 / FPS)
    (b,) = tr.boxes()
    assert b["identity_id"] == "p1"
    assert b["track_id"] == 0                    # the same track, not a second one
    assert len(tr._tracks) == 1
