import subprocess

def cut_video(video_path, start_time, end_time, output_path):
    duration = end_time - start_time
    subprocess.run([
        "ffmpeg", "-y", "-v", "error",
        "-ss", str(start_time),   # fast seek before decoding
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264",        # re-encode video only for this segment
        "-preset", "ultrafast",   # fastest re-encode (bigger file, but temporary clips)
        "-crf", "18",             # near lossless (adjust 18–23 if needed)
        "-c:a", "aac",            # re-encode audio
        "-b:a", "128k",
        output_path
    ])
    print(f"Clip saved: {output_path}")
