import os
import tempfile
import subprocess
import wave
import numpy as np

class WaveformVisualizer:
    """Extracts and stores waveform data for visualization"""
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.waveform_data = None  # List of (min_val, max_val) tuples
        self.duration = 0
        self.sample_rate = 44100
    
    def extract_waveform(self, num_points=1000):
        import os, tempfile, subprocess, wave
        import numpy as np

        fd, wav_file = tempfile.mkstemp(suffix=".wav")
        os.close(fd)  # IMPORTANT: don't keep the file handle open

        try:
            print(f"🎵 Extracting audio from: {self.video_path}")

            cmd = [
                "ffmpeg",
                "-y",
                "-i", self.video_path,
                "-map", "0:a:0",          # pick first audio stream explicitly
                "-vn",
                "-ac", "1",
                "-ar", str(self.sample_rate),
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

            step = max(1, audio.size // num_points)
            waveform = []
            for i in range(0, audio.size, step):
                chunk = audio[i:i + step]
                if chunk.size:
                    waveform.append((float(chunk.min()) / 32768.0, float(chunk.max()) / 32768.0))

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
