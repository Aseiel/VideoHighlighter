"""
End-to-end tests for modules.combine_videos against real ffmpeg.

Three tiny generated clips cover the traps the combiner exists to handle:
a large landscape clip with audio, a sideways clip (rotation metadata, the
phone-footage case), and a clip with no audio stream at all. The combined
reel must come out upright (rotation baked, not copied), on one uniform
canvas with the portrait clip pillarboxed rather than stretched, with a
continuous audio track.

Rotation metadata cannot be written by every ffmpeg build; when neither
-display_rotation nor the rotate tag sticks, the rotation-specific
assertions skip instead of lying.

probe checks go through modules.video_probe when present (the contract
probe), with a local ffprobe fallback so these tests do not depend on that
module landing first.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import types

import pytest

from modules.app_paths import ffmpeg_exe
from modules.combine_videos import CombineCancelled, _ffprobe_exe, combine_videos

FFMPEG = ffmpeg_exe()
FFPROBE = _ffprobe_exe()


def _tool_ok(cmd: str) -> bool:
    try:
        return subprocess.run([cmd, "-version"], capture_output=True).returncode == 0
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    not (_tool_ok(FFMPEG) and _tool_ok(FFPROBE)),
    reason="ffmpeg/ffprobe not available",
)


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{result.stderr[-500:]}")
    return result


def _ffprobe_json(path: str) -> dict:
    out = _run([FFPROBE, "-v", "error", "-show_streams", "-show_format",
                "-of", "json", path])
    return json.loads(out.stdout)


def _local_probe(path: str) -> dict:
    """Same shape as the video_probe contract, built from raw ffprobe."""
    info = _ffprobe_json(path)
    v = next(s for s in info["streams"] if s.get("codec_type") == "video")
    rotation = 0
    for sd in v.get("side_data_list") or []:
        if "rotation" in sd:
            # displaymatrix rotation is CCW; the contract wants CW-to-display
            rotation = (-int(round(float(sd["rotation"])))) % 360
    if not rotation:
        try:
            rotation = int(round(float((v.get("tags") or {}).get("rotate", 0)))) % 360
        except (TypeError, ValueError):
            rotation = 0
    fps_str = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1"
    num, _, den = fps_str.partition("/")
    fps = float(num) / float(den) if den and float(den) else float(num or 0)
    return {
        "duration": float(info.get("format", {}).get("duration", 0)),
        "width": int(v["width"]),
        "height": int(v["height"]),
        "fps": fps,
        "rotation": rotation,
    }


def _probe(path: str) -> dict:
    try:
        from modules.video_probe import probe_video
        return probe_video(path)
    except ImportError:
        return _local_probe(path)


def _expected_canvas() -> tuple[int, int]:
    """Canvas = dims of the largest input when video_probe can measure them,
    else the module's documented 1920x1080 fallback."""
    try:
        import modules.video_probe  # noqa: F401
        return 1280, 720
    except ImportError:
        return 1920, 1080


def _has_audio_stream(path: str) -> bool:
    info = _ffprobe_json(path)
    return any(s.get("codec_type") == "audio" for s in info["streams"])


def _region_luma(path: str, t: float, crop: str):
    """Average luma (YAVG) of one frame's cropped region, or None when this
    ffmpeg build can't report signalstats."""
    result = subprocess.run(
        [FFMPEG, "-v", "error", "-ss", str(t), "-i", path, "-frames:v", "1",
         "-vf", f"crop={crop},signalstats,metadata=print:file=-",
         "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    for line in (result.stdout or "").splitlines() + (result.stderr or "").splitlines():
        if "signalstats.YAVG" in line:
            try:
                return float(line.rsplit("=", 1)[1])
            except ValueError:
                return None
    return None


def _make_clip(path: str, size: str, audio: bool) -> None:
    cmd = [FFMPEG, "-y", "-v", "error",
           "-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30:duration=2"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
                "-c:a", "aac", "-shortest"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", path]
    _run(cmd)


def _make_rotated(base: str, path: str) -> bool:
    """Remux `base` with 90-degree rotation metadata. True when the metadata
    actually stuck (verified by probing back), False otherwise."""
    for cmd in (
        [FFMPEG, "-y", "-v", "error", "-display_rotation", "90", "-i", base,
         "-c", "copy", path],
        [FFMPEG, "-y", "-v", "error", "-i", base, "-c", "copy",
         "-metadata:s:v:0", "rotate=90", path],
    ):
        try:
            _run(cmd)
        except RuntimeError:
            continue
        if _local_probe(path)["rotation"] % 180 == 90:
            return True
    return False


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    d = tmp_path_factory.mktemp("clips")
    a = str(d / "a_landscape.mp4")
    b_base = str(d / "b_base.mp4")
    b = str(d / "b_rotated.mp4")
    c = str(d / "c_noaudio.mp4")
    _make_clip(a, "1280x720", audio=True)
    _make_clip(b_base, "640x360", audio=True)
    _make_clip(c, "640x360", audio=False)
    rotation_supported = _make_rotated(b_base, b)
    if not rotation_supported:
        shutil.copyfile(b_base, b)
    return {"files": [a, b, c], "rotation_supported": rotation_supported}


@pytest.fixture(scope="module")
def combined(fixtures, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("reel") / "combined.mp4")
    logs, progress = [], []
    result = combine_videos(
        fixtures["files"], out,
        log_fn=logs.append,
        progress_fn=lambda cur, tot, task, det: progress.append((cur, tot, task, det)),
    )
    return {"output": result, "logs": logs, "progress": progress}


def test_output_exists_with_expected_duration(combined):
    out = combined["output"]
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0
    assert _probe(out)["duration"] == pytest.approx(6.0, abs=1.5)


def test_output_has_no_rotation_metadata(combined):
    # Normalization must bake rotation into pixels; leftover metadata would
    # make players rotate the already-upright reel.
    assert _probe(combined["output"])["rotation"] == 0


def test_output_dims_are_uniform_canvas(combined):
    info = _probe(combined["output"])
    assert (info["width"], info["height"]) == _expected_canvas()
    # r_frame_rate/avg_frame_rate on a concat -c copy output are guesses that
    # the tiny AAC-priming gap at each segment boundary skews (observed 120/1),
    # so measure the effective rate as frames over duration instead.
    v = next(s for s in _ffprobe_json(combined["output"])["streams"]
             if s.get("codec_type") == "video")
    if v.get("nb_frames") and v.get("duration"):
        assert float(v["nb_frames"]) / float(v["duration"]) == pytest.approx(30, abs=1)


def test_output_has_audio_track(combined):
    # The no-audio input got silent AAC, so the reel carries audio end to end.
    assert _has_audio_stream(combined["output"])


def test_progress_reported_per_file(combined):
    calls = combined["progress"]
    assert all(task == "Combining" for _, _, task, _ in calls)
    details = [det for _, _, _, det in calls]
    for n in (1, 2, 3):
        assert f"file {n}/3" in details
    assert all(tot == 3 for _, tot, _, _ in calls)


def test_rotated_clip_is_pillarboxed_upright(combined, fixtures):
    """The empirical autorotation check: the sideways clip occupies the middle
    2s of the reel. Displayed upright it is portrait on a landscape canvas, so
    its left fifth must be black padding — while the full-canvas landscape
    clip's left fifth is testsrc2 content. A squished or unrotated clip would
    fill the frame and light the pillar up."""
    if not fixtures["rotation_supported"]:
        pytest.skip("this ffmpeg build cannot write rotation metadata")
    out = combined["output"]
    pillar = _region_luma(out, 3.0, "iw/5:ih:0:0")
    center = _region_luma(out, 3.0, "iw/5:ih:(iw-iw/5)/2:0")
    landscape_edge = _region_luma(out, 1.0, "iw/5:ih:0:0")
    if pillar is None or center is None or landscape_edge is None:
        pytest.skip("signalstats not available in this ffmpeg build")
    assert pillar < 36, f"expected black pillarbox, got YAVG={pillar}"
    assert center > 48, f"expected clip content in the center, got YAVG={center}"
    assert landscape_edge > 48, (
        f"landscape clip should fill the canvas, got YAVG={landscape_edge}"
    )


def test_cancel_before_work(fixtures, tmp_path):
    out = str(tmp_path / "cancelled.mp4")
    with pytest.raises(CombineCancelled):
        combine_videos(fixtures["files"], out, log_fn=lambda _m: None,
                       cancel_check=lambda: True)
    assert not os.path.exists(out)


def test_all_inputs_missing_raises(tmp_path):
    with pytest.raises(ValueError):
        combine_videos([str(tmp_path / "nope.mp4")], str(tmp_path / "out.mp4"),
                       log_fn=lambda _m: None)


def test_cancel_during_music_leaves_no_output(fixtures, tmp_path):
    """Cancelling after concat but before music is applied must leave nothing at
    the destination — not the finished-but-music-less reel."""
    out = str(tmp_path / "cancel_music.mp4")
    calls = {"n": 0}

    # Two inputs -> checks fire: file0, file1, pre-concat, pre-music, pre-copy.
    # Trip on the pre-music check (the 4th), so the concat has already written
    # the staged reel by the time cancellation lands.
    def cancel_check() -> bool:
        calls["n"] += 1
        return calls["n"] >= 4

    c = fixtures["files"][2]
    with pytest.raises(CombineCancelled):
        combine_videos([c, c], out, log_fn=lambda _m: None,
                       cancel_check=cancel_check,
                       music={"path": "song.mp3", "mode": "replace"})
    assert not os.path.exists(out)


def test_music_failure_after_concat_leaves_no_output(fixtures, tmp_path, monkeypatch):
    """If music application blows up after the concat, the destination stays
    clean rather than holding a silent-but-complete reel."""
    def apply_music(*_a, **_k):
        raise RuntimeError("boom")

    fake = types.ModuleType("modules.music_track")
    fake.apply_music = apply_music
    fake._MODES = ("replace", "mix", "duck")
    monkeypatch.setitem(sys.modules, "modules.music_track", fake)
    monkeypatch.setattr(sys.modules["modules"], "music_track", fake, raising=False)

    out = str(tmp_path / "music_fail.mp4")
    c = fixtures["files"][2]
    with pytest.raises(RuntimeError):
        combine_videos([c, c], out, log_fn=lambda _m: None,
                       music={"path": "song.mp3", "mode": "replace"})
    assert not os.path.exists(out)


def test_invalid_music_mode_raises_before_work(fixtures, tmp_path):
    """A bad mode is rejected up front, before any normalize/concat runs."""
    out = str(tmp_path / "bad_mode.mp4")
    with pytest.raises(ValueError):
        combine_videos(fixtures["files"], out, log_fn=lambda _m: None,
                       music={"path": "song.mp3", "mode": "nonsense"})
    assert not os.path.exists(out)


def test_music_is_applied_via_music_track(fixtures, tmp_path, monkeypatch):
    calls = {}

    def apply_music(video, music, out, mode="replace", music_volume=0.8, log_fn=print):
        calls.update(video=video, music=music, out=out,
                     mode=mode, volume=music_volume)
        shutil.copyfile(video, out)
        return out

    fake = types.ModuleType("modules.music_track")
    fake.apply_music = apply_music
    monkeypatch.setitem(sys.modules, "modules.music_track", fake)
    # `from modules import music_track` prefers the package attribute over
    # sys.modules once the real module has been imported elsewhere in the run.
    # `modules` is a namespace package already loaded via the import above; grab
    # it from sys.modules rather than a bare `import modules` (which the
    # local-import completeness check flags, there being no modules/__init__.py).
    modules_pkg = sys.modules["modules"]
    monkeypatch.setattr(modules_pkg, "music_track", fake, raising=False)

    out = str(tmp_path / "with_music.mp4")
    c = fixtures["files"][2]
    result = combine_videos(
        [c, c], out, log_fn=lambda _m: None,
        music={"path": "song.mp3", "mode": "duck", "volume": 0.25},
    )
    assert result == out
    assert os.path.exists(out)
    # The reel is staged in a temp dir; music is applied there and only the
    # finished result is copied to `out`. So apply_music sees temp paths for
    # both its input (the concatenated reel) and output, never the destination.
    assert calls["video"] != out
    assert calls["music"] == "song.mp3"
    assert calls["mode"] == "duck"
    assert calls["volume"] == 0.25
    assert calls["out"] != out
    # Both temp paths are cleaned up with the staging dir.
    assert not os.path.exists(calls["video"])
    assert not os.path.exists(calls["out"])
