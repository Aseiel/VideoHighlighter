"""
combine_videos.py — join finished highlight clips into one reel, GUI-free.

Port of the Qt combine_highlights flow (main.py) so the sidecar can run it as
a job. The two-phase shape is deliberate and must stay:

1. Normalize: every input is re-encoded to one uniform format (H.264 + AAC,
   same canvas / fps / SAR / pix_fmt, 48 kHz stereo) into a temp dir.
2. Concat: the uniform files are joined with the concat demuxer and `-c copy`.
   Stream copy is only safe *because* of phase 1 — mixed codecs, resolutions
   or timebases through the concat demuxer produce desync or hard failures.

Design constraints
==================
- Aspect ratio is preserved: scale with force_original_aspect_ratio=decrease
  then center-pad to the canvas, so portrait clips are pillarboxed on a
  landscape canvas rather than stretched.
- Rotation metadata must be *baked*, not copied: ffmpeg's autorotation (on by
  default during a re-encode) turns sideways-shot clips upright before the
  concat step throws the metadata away. Never pass -noautorotate here.
- Inputs without an audio stream get a silent AAC track from anullsrc; the
  concat demuxer requires every segment to carry the same stream layout.
- The canvas is the displayed size of the largest input (rotation swaps w/h),
  falling back to 1920x1080 when probing is unavailable.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from modules.app_paths import ffmpeg_exe


class CombineCancelled(RuntimeError):
    """Raised between steps when cancel_check() says stop. ffmpeg is never
    killed mid-file, so cancellation leaves no partial file outside the temp
    dir (which is removed on the way out)."""


def _ffprobe_exe() -> str:
    """ffprobe resolved like ffmpeg_exe(): PATH first, then next to whatever
    ffmpeg we found. imageio-ffmpeg ships only ffmpeg, so callers must cope
    with ffprobe being genuinely absent (see _has_audio)."""
    found = shutil.which("ffprobe")
    if found:
        return found
    ff = ffmpeg_exe()
    name = "ffprobe.exe" if ff.lower().endswith(".exe") else "ffprobe"
    candidate = os.path.join(os.path.dirname(os.path.abspath(ff)), name)
    if os.path.exists(candidate):
        return candidate
    return "ffprobe"


def _has_audio(path: str, log_fn=print) -> bool:
    try:
        result = subprocess.run(
            [_ffprobe_exe(), "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30,
        )
        if result.returncode == 0:
            return "audio" in (result.stdout or "")
    except Exception:
        pass
    # No usable ffprobe (frozen app with only imageio-ffmpeg's ffmpeg).
    # `ffmpeg -i` with no output exits non-zero but still lists the streams.
    try:
        result = subprocess.run(
            [ffmpeg_exe(), "-hide_banner", "-i", path],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30,
        )
        return "Audio:" in (result.stderr or "")
    except Exception as e:
        log_fn(f"⚠️ Could not detect audio in {os.path.basename(path)}: {e}")
        return False


def _target_canvas(files, log_fn) -> tuple[int, int, int]:
    """(width, height, fps) of the largest input's *displayed* size — rotation
    of 90/270 swaps stored w/h. Defaults to 1920x1080 @ 30 when probing fails
    so a broken probe degrades quality, never the whole combine."""
    try:
        from modules.video_probe import probe_video
    except Exception as e:
        log_fn(f"⚠️ Video probe unavailable ({e}); using default 1920x1080")
        return 1920, 1080, 30
    best = None
    for f in files:
        try:
            info = probe_video(f)
            w, h = int(info["width"]), int(info["height"])
            if int(info.get("rotation", 0) or 0) in (90, 270):
                w, h = h, w
            fps = float(info.get("fps") or 0)
            log_fn(f"  {os.path.basename(f)}: {w}x{h} @ {fps:.2f}fps")
            if best is None or w * h > best[0] * best[1]:
                best = (w, h, fps)
        except Exception as e:
            log_fn(f"⚠️ Could not analyze {os.path.basename(f)}: {e}")
    if best is None:
        return 1920, 1080, 30
    w, h, fps = best
    # x264 + yuv420p needs even dimensions.
    w, h = max(2, w - w % 2), max(2, h - h % 2)
    fps = int(round(fps)) if fps > 0 else 30
    return w, h, min(max(fps, 10), 60)


def _normalize(src: str, dst: str, width: int, height: int, fps: int,
               log_fn=print) -> None:
    with_silence = not _has_audio(src, log_fn)
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
          f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
          f"setsar=1,fps={fps},setpts=N/FRAME_RATE/TB")
    cmd = [ffmpeg_exe(), "-y", "-v", "error", "-i", src]
    if with_silence:
        cmd += ["-f", "lavfi", "-i",
                "anullsrc=channel_layout=stereo:sample_rate=48000"]
    cmd += [
        "-map", "0:v:0",
        "-map", "1:a:0" if with_silence else "0:a:0",
        "-vf", vf,
        "-af", "aresample=48000,asetpts=N/SR/TB",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-level", "4.0",
        "-g", str(fps * 2),
        "-keyint_min", str(fps),
        "-sc_threshold", "0",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        "-fps_mode", "cfr",
        "-max_muxing_queue_size", "1024",
        "-fflags", "+genpts",
        "-avoid_negative_ts", "make_zero",
    ]
    if with_silence:
        cmd += ["-shortest"]  # anullsrc is infinite
    cmd += [dst]
    result = subprocess.run(cmd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=600)
    if result.returncode != 0 or not os.path.exists(dst) or os.path.getsize(dst) == 0:
        err = (result.stderr or "").strip()[-500:] or "unknown error"
        raise RuntimeError(f"Normalization failed for {os.path.basename(src)}: {err}")


def _check_cancel(cancel_check) -> None:
    if cancel_check is not None and cancel_check():
        raise CombineCancelled("Combine cancelled")


def combine_videos(files, output, log_fn=print, progress_fn=None,
                   cancel_check=None, music=None) -> str:
    """Combine `files` (in order) into `output`; returns the output path.

    music: optional {'path': str, 'mode': 'replace'|'mix'|'duck',
    'volume': float 0..1} applied to the finished reel via
    modules.music_track (imported lazily so combining works without it).

    Raises ValueError when no input file exists, CombineCancelled when
    cancel_check fires, RuntimeError when ffmpeg fails.
    """
    valid = []
    for f in files or []:
        if f and os.path.exists(f):
            valid.append(f)
        elif f:
            log_fn(f"⚠️ Skipping missing input: {f}")
    if not valid:
        raise ValueError("No valid input files to combine")

    # Validate the music request up front: rejecting a bad mode after the
    # expensive normalize/concat has already run would waste minutes and leave
    # a reel behind. music_track owns the canonical mode list; fall back to the
    # known set if it can't be read (minimal env / test double).
    if music and music.get("path"):
        try:
            from modules import music_track
            modes = music_track._MODES
        except Exception:
            modes = ("replace", "mix", "duck")
        mode = music.get("mode", "replace")
        if mode not in modes:
            raise ValueError(
                f"unknown music mode: {mode!r} (expected one of {modes})")

    output = os.path.abspath(output)
    out_dir = os.path.dirname(output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    log_fn(f"🎬 Combining {len(valid)} videos into one...")
    log_fn("🔍 Analyzing input videos...")
    width, height, fps = _target_canvas(valid, log_fn)
    log_fn(f"🎯 Target format: {width}x{height} @ {fps}fps")

    total = len(valid)
    temp_dir = tempfile.mkdtemp(prefix="vh_combine_")
    # The reel is built entirely inside temp_dir and only moved to `output` once
    # every step (concat AND music) has succeeded. If cancel fires or music
    # application fails after concat, `output` is never touched — the file the
    # user sees is either the finished reel or nothing, never a music-less
    # stand-in (see CombineCancelled).
    _, out_ext = os.path.splitext(output)
    staged = os.path.join(temp_dir, f"reel{out_ext or '.mp4'}")
    try:
        normalized = []
        for i, src in enumerate(valid):
            _check_cancel(cancel_check)
            if progress_fn:
                try:
                    progress_fn(i, total, "Combining", f"file {i + 1}/{total}")
                except Exception:
                    pass
            log_fn(f"⚙️ Normalizing {i + 1}/{total}: {os.path.basename(src)}")
            dst = os.path.join(temp_dir, f"normalized_{i:03d}.mp4")
            _normalize(src, dst, width, height, fps, log_fn)
            normalized.append(dst)

        _check_cancel(cancel_check)
        if progress_fn:
            try:
                progress_fn(total, total, "Combining", "concatenating")
            except Exception:
                pass
        log_fn("🔗 Concatenating normalized videos...")
        concat_list = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for dst in normalized:
                # concat-demuxer quoting: single-quoted path, embedded quotes
                # closed/escaped/reopened.
                p = os.path.abspath(dst).replace("\\", "/").replace("'", "'\\''")
                f.write(f"file '{p}'\n")
        result = subprocess.run(
            [ffmpeg_exe(), "-y", "-v", "error", "-f", "concat", "-safe", "0",
             "-i", concat_list, "-c", "copy", "-movflags", "+faststart", staged],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=600,
        )
        if result.returncode != 0 or not os.path.exists(staged) or os.path.getsize(staged) == 0:
            err = (result.stderr or "").strip()[-500:] or "unknown error"
            raise RuntimeError(f"Concatenation failed: {err}")

        if music and music.get("path"):
            _check_cancel(cancel_check)
            from modules import music_track
            music_tmp = os.path.join(temp_dir, f"reel_music{out_ext or '.mp4'}")
            log_fn(f"🎵 Applying music: {os.path.basename(music['path'])}")
            music_track.apply_music(
                staged, music["path"], music_tmp,
                mode=music.get("mode", "replace"),
                music_volume=float(music.get("volume", 0.8)),
                log_fn=log_fn,
            )
            staged = music_tmp

        # Everything succeeded: promote the staged reel to the real output. Copy
        # (not rename) so it survives temp_dir living on a different volume than
        # the destination; the temp copy is cleaned up in finally.
        _check_cancel(cancel_check)
        shutil.copyfile(staged, output)
        log_fn(f"✅ Combined video saved: {output}")
        return output
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
