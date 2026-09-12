"""
Tests for modules.ffmpeg_tools — running on the ffmpeg pip installed.

Staging is tested against a fake bundled binary and a controlled PATH, so it
needs nothing installed. The probe tests write a tiny clip with PyAV and skip
where PyAV is absent (the CI test requirements do not carry it); where a real
ffprobe exists they also pin that PyAV's answer matches it.
"""

from __future__ import annotations

import os
import shutil
import sys
import types

import pytest

from modules import ffmpeg_tools

EXE = ".exe" if sys.platform == "win32" else ""


def _fake_binary(path, content=b"fake ffmpeg"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    path.chmod(0o755)
    return path


def _same_path(a, b) -> bool:
    return os.path.normcase(str(a)) == os.path.normcase(str(b))


@pytest.fixture
def isolated(monkeypatch, tmp_path):
    """No ffmpeg on PATH, a fake imageio-ffmpeg, and a private staging dir."""
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    monkeypatch.chdir(empty)  # `which` on Windows looks in the cwd too
    src = _fake_binary(tmp_path / "imageio" / f"ffmpeg-fake-v1{EXE}")
    fake = types.SimpleNamespace(get_ffmpeg_exe=lambda: str(src))
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake)
    bin_dir = tmp_path / "staged"
    monkeypatch.setattr(ffmpeg_tools, "_bin_dir", lambda: str(bin_dir))
    monkeypatch.setattr(ffmpeg_tools, "_ensured", None)
    return types.SimpleNamespace(src=src, bin_dir=bin_dir, fake=fake,
                                 empty=empty)


# ---------------------------------------------------------------------------
# ensure_ffmpeg_on_path
# ---------------------------------------------------------------------------

def test_bundled_ffmpeg_answers_to_its_plain_name(isolated):
    got = ffmpeg_tools.ensure_ffmpeg_on_path()

    staged = isolated.bin_dir / f"ffmpeg{EXE}"
    assert _same_path(got, staged)
    assert staged.read_bytes() == isolated.src.read_bytes()
    assert _same_path(os.environ["PATH"].split(os.pathsep)[0], isolated.bin_dir)
    assert _same_path(shutil.which("ffmpeg"), staged)


def test_system_ffmpeg_wins(isolated, monkeypatch):
    system = _fake_binary(isolated.empty.parent / "system" / f"ffmpeg{EXE}")
    monkeypatch.setenv("PATH", str(system.parent))

    assert _same_path(ffmpeg_tools.ensure_ffmpeg_on_path(), system)
    assert not isolated.bin_dir.exists()


def test_repeat_calls_do_not_grow_path(isolated):
    ffmpeg_tools.ensure_ffmpeg_on_path()
    ffmpeg_tools.ensure_ffmpeg_on_path()

    entries = [os.path.normcase(p) for p in os.environ["PATH"].split(os.pathsep)]
    assert entries.count(os.path.normcase(str(isolated.bin_dir))) == 1


def test_upgraded_bundle_replaces_the_staged_copy(isolated, monkeypatch):
    ffmpeg_tools.ensure_ffmpeg_on_path()
    # A new file rather than an edit, which would write through a hard link.
    isolated.src.unlink()
    _fake_binary(isolated.src, b"fake ffmpeg, a newer and longer build")
    monkeypatch.setenv("PATH", str(isolated.empty))  # the next launch

    ffmpeg_tools.ensure_ffmpeg_on_path()

    staged = isolated.bin_dir / f"ffmpeg{EXE}"
    assert staged.read_bytes() == b"fake ffmpeg, a newer and longer build"


def test_no_ffmpeg_anywhere_is_reported(isolated, monkeypatch):
    def missing():
        raise RuntimeError("no binary for this platform")
    monkeypatch.setattr(isolated.fake, "get_ffmpeg_exe", missing)

    said = []
    assert ffmpeg_tools.ensure_ffmpeg_on_path(said.append) is None
    assert said and "ffmpeg" in said[0]


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------

def _write_clip(path, rotation=0):
    """1 s of 64x48 video at 10 fps plus a mono AAC track."""
    av = pytest.importorskip("av")
    np = pytest.importorskip("numpy")
    with av.open(str(path), "w") as out:
        video = out.add_stream("mpeg4", rate=10)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        if rotation:
            if not hasattr(video, "set_display_rotation"):
                pytest.skip("this PyAV cannot write a display matrix")
            video.set_display_rotation(rotation)
        audio = out.add_stream("aac", rate=16000)
        for i in range(10):
            frame = av.VideoFrame.from_ndarray(
                np.full((48, 64, 3), i * 20, np.uint8), format="rgb24")
            for packet in video.encode(frame):
                out.mux(packet)
        for packet in video.encode():
            out.mux(packet)
        for i in range(16):
            frame = av.AudioFrame.from_ndarray(
                np.zeros((1, 1024), np.float32), format="fltp", layout="mono")
            frame.sample_rate, frame.pts = 16000, i * 1024
            for packet in audio.encode(frame):
                out.mux(packet)
        for packet in audio.encode():
            out.mux(packet)
    return path


def _video(info) -> dict:
    return next(s for s in info["streams"] if s["codec_type"] == "video")


def _summary(info) -> dict:
    """The parts of a probe callers here actually read."""
    video = _video(info)
    return {
        "types": sorted(s["codec_type"] for s in info["streams"]),
        "size": (video["width"], video["height"]),
        "rates": (video["r_frame_rate"], video["avg_frame_rate"]),
        "rotation": next((sd["rotation"] for sd in video.get("side_data_list") or []
                          if "rotation" in sd), 0),
        "duration": round(float(info["format"]["duration"]), 3),
        "bytes": info["format"]["size"],
    }


@pytest.fixture
def without_ffprobe(monkeypatch):
    monkeypatch.setattr(ffmpeg_tools, "ffprobe_exe", lambda: None)


def test_probe_without_ffprobe_reads_the_clip(tmp_path, without_ffprobe):
    clip = _write_clip(tmp_path / "clip.mp4")

    info = ffmpeg_tools.probe(clip)

    assert _video(info)["width"] == 64 and _video(info)["height"] == 48
    assert _video(info)["avg_frame_rate"] == "10/1"
    assert "audio" in [s["codec_type"] for s in info["streams"]]
    assert float(info["format"]["duration"]) == pytest.approx(1.0, abs=0.1)
    assert info["format"]["size"] == str(clip.stat().st_size)
    assert "side_data_list" not in _video(info)


def test_probe_without_ffprobe_reads_rotation(tmp_path, without_ffprobe):
    clip = _write_clip(tmp_path / "rotated.mp4", rotation=90)

    side = _video(ffmpeg_tools.probe(clip))["side_data_list"]

    assert side[0]["rotation"] == 90


@pytest.mark.parametrize("rotation", [0, 90])
def test_pyav_answer_matches_ffprobe(tmp_path, monkeypatch, rotation):
    if not ffmpeg_tools.ffprobe_exe():
        pytest.skip("no ffprobe to compare against")
    clip = _write_clip(tmp_path / "clip.mp4", rotation)
    from_ffprobe = _summary(ffmpeg_tools.probe(clip))

    monkeypatch.setattr(ffmpeg_tools, "ffprobe_exe", lambda: None)

    assert _summary(ffmpeg_tools.probe(clip)) == from_ffprobe


def test_unreadable_file_raises(tmp_path, without_ffprobe):
    pytest.importorskip("av")
    bad = tmp_path / "not_a_video.mp4"
    bad.write_bytes(b"definitely not a video")

    with pytest.raises(Exception):
        ffmpeg_tools.probe(bad)
