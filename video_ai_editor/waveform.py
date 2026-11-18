import os
import tempfile
import subprocess
import wave
import numpy as np

from modules.system.app_paths import ffmpeg_exe

# Bins per second of the timeline's AUDIO WAVEFORM. It used to be a fixed 1000
# or 2000 points for the whole file: on a 36-minute video one bin was about a
# second, drawn as a line at the bin's centre, so a slap at 3:21.9 showed at
# 3:22.4 - half a second after the playhead that was on it. At 20 a bin is
# 50 ms, under the time between two frames at 24 fps.
BINS_PER_SECOND = 20
# A cached waveform coarser than this is extracted again when the viewer opens.
MIN_BINS_PER_SECOND = 10


def is_coarse(waveform, duration) -> bool:
    """True when ``waveform`` has fewer than ``MIN_BINS_PER_SECOND`` bins per
    second of ``duration`` (a cache from before ``BINS_PER_SECOND``)."""
    try:
        return bool(waveform) and float(duration) > 0 and             len(waveform) / float(duration) < MIN_BINS_PER_SECOND
    except (TypeError, ValueError):
        return False


def bin_audio(audio, rate, bins_per_second=BINS_PER_SECOND, num_points=None):
    """``(min, max, rms)`` per bin of int16 ``audio``, each in [-1, 1].

    Bins are ``rate / bins_per_second`` samples, or ``len(audio) //
    num_points`` when ``num_points`` is given; the last may be shorter.
    Rounded to 4 places: the viewer saves the list into the video's cache."""
    audio = np.asarray(audio)
    if audio.size == 0:
        return []
    if num_points:
        step = max(1, audio.size // int(num_points))
    else:
        step = max(1, int(round(rate / float(bins_per_second))))
    x = audio.astype(np.float32) / 32768.0
    full = (x.size // step) * step
    parts = [x[:full].reshape(-1, step)] if full else []
    out = []
    for block in parts:
        mins = block.min(axis=1)
        maxs = block.max(axis=1)
        rms = np.sqrt(np.mean(block.astype(np.float64) ** 2, axis=1))
        out.extend(zip(np.round(mins, 4).tolist(), np.round(maxs, 4).tolist(),
                       np.round(rms, 4).tolist()))
    tail = x[full:]
    if tail.size:
        out.append((round(float(tail.min()), 4), round(float(tail.max()), 4),
                    round(float(np.sqrt(np.mean(tail.astype(np.float64) ** 2))), 4)))
    return out


class WaveformVisualizer:
    """Extracts and stores waveform data for visualization"""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.waveform_data = None  # List of (min, max, rms) tuples
        self.duration = 0
        # Enough for a level display, and a third of the memory of 44.1 kHz
        # for a long film read whole.
        self.sample_rate = 16000
    
    def extract_waveform(self, num_points=None):
        import os, tempfile, subprocess, wave
        import numpy as np

        fd, wav_file = tempfile.mkstemp(suffix=".wav")
        os.close(fd)  # IMPORTANT: don't keep the file handle open

        try:
            print(f"🎵 Extracting audio from: {self.video_path}")

            cmd = [
                ffmpeg_exe(),
                "-y",
                "-fflags", "+genpts",
                "-i", self.video_path,
                "-map", "0:a:0",          # pick first audio stream explicitly
                "-vn",
                "-ac", "1",
                "-ar", str(self.sample_rate),
                "-af", "aresample=async=1:first_pts=0",
                "-c:a", "pcm_s16le",
                wav_file,
                "-hide_banner",
                "-loglevel", "error",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ FFmpeg failed")
                print("stderr:", result.stderr.strip())
                return None

            if not os.path.exists(wav_file) or os.path.getsize(wav_file) < 44:
                print("❌ WAV output missing/too small (likely no audio or ffmpeg write failed)")
                return None

            with wave.open(wav_file, "rb") as wf:
                rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16)

            if audio.size == 0 or rate <= 0:
                print("❌ No audio samples decoded")
                return None

            self.duration = audio.size / rate

            # num_points=None: BINS_PER_SECOND, whatever the length.
            waveform = bin_audio(audio, rate, num_points=num_points)

            self.waveform_data = waveform
            print(f"✅ Waveform extracted: {len(waveform)} points, duration={self.duration:.2f}s")
            return waveform

        except Exception as e:
            print(f"❌ Waveform extraction error: {e}")
            import traceback; traceback.print_exc()
            return None

        finally:
            try:
                if os.path.exists(wav_file):
                    os.remove(wav_file)
            except:
                pass
