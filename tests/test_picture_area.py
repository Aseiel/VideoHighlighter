"""The picture inside black bars, found so the shot passes read only it."""
import shutil
import sys

import numpy as np
import pytest

sys.path.insert(0, __import__("os").path.dirname(__file__))
from conftest import real_opencv  # noqa: E402
from modules.media import picture_area as pa

H, W = 240, 320


def _picture(rng, h, w):
    """Lit, textured picture: every row and column well above black."""
    return rng.integers(60, 255, (h, w, 3), dtype=np.uint8)


def _framed(top=0, bottom=0, left=0, right=0, n=8, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        f = np.zeros((H, W, 3), dtype=np.uint8)
        f[top:H - bottom, left:W - right] = _picture(rng, H - top - bottom,
                                                      W - left - right)
        out.append(f)
    return out


def test_letterbox_is_found():
    assert pa.area_in_frames(_framed(top=40, bottom=40)) == (0, 40, W, H - 80)


def test_pillarbox_is_found():
    assert pa.area_in_frames(_framed(left=40, right=40)) == (40, 0, W - 80, H)


def test_no_bars_is_the_whole_frame():
    assert pa.area_in_frames(_framed()) is None


def test_a_sliver_at_the_edge_is_not_a_bar():
    assert pa.area_in_frames(_framed(top=1, bottom=1)) is None


def test_subtitles_in_the_bar_do_not_hold_it_open():
    frames = _framed(top=40, bottom=40)
    for f in frames[::2]:                  # text in every other frame
        f[H - 30:H - 20, 100:160] = 235    # a fifth of the row's width
    assert pa.area_in_frames(frames) == (0, 40, W, H - 80)


def test_a_dark_scene_does_not_crop_into_the_picture():
    frames = _framed(top=40, bottom=40)
    frames[3][40:90] = 5                   # night sky at the top of one shot
    assert pa.area_in_frames(frames) == (0, 40, W, H - 80)


def test_black_frames_say_nothing():
    frames = _framed(top=40, bottom=40, n=2) + [np.zeros((H, W, 3), np.uint8)] * 6
    assert pa.area_in_frames(frames) is None   # two usable frames are too few


def test_a_dark_video_is_not_a_tiny_picture():
    assert pa.area_in_frames(_framed(top=100, bottom=100)) is None


def test_rows_only_keeps_the_columns():
    frames = _framed(top=40, bottom=40, left=40, right=40)
    assert pa.area_in_frames(frames, rows_only=True) == (0, 40, W, H - 80)
    assert pa.area_in_frames(_framed(left=40, right=40), rows_only=True) is None


def test_crop_scales_to_a_smaller_frame():
    area = pa.PictureArea(0, 48, 640, 264, 640, 360)
    assert area.crop(np.zeros((360, 640))).shape == (264, 640)
    assert area.crop(np.zeros((180, 320))).shape == (132, 320)
    assert area.ffmpeg_filter() == ("crop=iw*1.000000:ih*0.733333:"
                                    "iw*0.000000:ih*0.133333")


@pytest.fixture()
def cv2(monkeypatch):
    """The real OpenCV, for the tests that write and read a video: the suite's
    shim cannot encode a frame (tests/conftest.py)."""
    real = real_opencv()
    if real is None:
        pytest.skip("OpenCV is not installed")
    monkeypatch.setitem(sys.modules, "cv2", real)
    return real


def _video(cv2, path, top=0, bottom=0, left=0, right=0, frames=150, cuts_every=25):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             25, (W, H))
    if not writer.isOpened():
        pytest.skip("no mp4 encoder available to make a test clip")
    rng = np.random.default_rng(1)
    pic = None
    for i in range(frames):
        if i % cuts_every == 0:
            pic = _picture(rng, H - top - bottom, W - left - right)
        f = np.zeros((H, W, 3), dtype=np.uint8)
        f[top:H - bottom, left:W - right] = pic
        writer.write(f)
    writer.release()
    return str(path)


def test_find_reads_it_from_a_video(cv2, tmp_path):
    area = pa.find(_video(cv2, tmp_path / "lb.mp4", top=40, bottom=40))
    assert area is not None and (area.width, area.height) == (W, H)
    # Inside the picture, and nearly all of it.
    assert area.y >= 40 and area.y + area.h <= H - 40
    assert area.h >= (H - 80) - 8 and (area.x, area.w) == (0, W)


def test_find_leaves_an_ordinary_video_alone(cv2, tmp_path):
    assert pa.find(_video(cv2, tmp_path / "plain.mp4")) is None


def test_find_never_raises(cv2, tmp_path):
    assert pa.find(str(tmp_path / "missing.mp4")) is None
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    assert pa.find(str(bad)) is None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_the_shot_list_reads_the_picture(cv2, tmp_path):
    from modules.segments import shot_boundaries as sb
    video = _video(cv2, tmp_path / "cuts.mp4", top=40, bottom=40)
    shots = sb.detect_shots(video, scorer="difference", log_fn=lambda *a: None)
    assert shots.picture_area and shots.picture_area[1] >= 40
    assert sb.to_dict(shots)["picture_area"] == list(shots.picture_area)
    assert len(shots.cut_frames) == 5      # a new picture every 25 frames
