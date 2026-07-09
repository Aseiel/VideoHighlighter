import os
import subprocess

from modules.app_paths import ffmpeg_exe
from modules.encoder_select import encoder_chain


def cut_video(video_path, start_time, end_time, output_path):
    """Cut [start_time, end_time] out of video_path into output_path,
    re-encoding with the fastest available hardware encoder (HEVC for VR /
    high-res sources, H.264 otherwise) and falling back through the chain to
    CPU libx264. Pixel format is normalized to yuv420p so 10-bit VR sources
    don't break the encoders."""
    duration = end_time - start_time
    last_err = "unknown error"
    for enc, vargs in encoder_chain(video_path):
        cmd = [
            ffmpeg_exe(), "-y", "-v", "error",
            "-ss", str(start_time),   # fast seek before decoding
            "-i", video_path,
            "-t", str(duration),
            "-fflags", "+genpts",
            "-map", "0:v:0",
            "-map", "0:a:0?",
            "-vf", "format=yuv420p",  # normalize (VR is often 10-bit)
        ] + vargs + [
            "-c:a", "aac",            # re-encode audio
            "-b:a", "128k",
            "-af", "aresample=async=1:first_pts=0",
            "-movflags", "+faststart",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(output_path):
            print(f"Clip saved: {output_path} [{enc}]")
            return
        last_err = (result.stderr or "").strip()[-500:] or "unknown error"
        print(f"⚠️ cut_video {enc} failed (rc={result.returncode}); "
              + ("trying next encoder" if enc != "libx264" else "no fallback left"))
    raise RuntimeError(f"cut_video failed for {output_path}: {last_err}")
