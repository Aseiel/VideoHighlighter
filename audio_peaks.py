import shutil
import subprocess
import tempfile
import numpy as np
import wave
from tqdm import tqdm

def extract_audio_peaks(video_path, chunk_size=0.1, threshold_db=-20, cancel_flag=None):
    """Extract audio peaks with cancellation support"""
    
    # Check for cancellation at start
    if cancel_flag and cancel_flag.is_set():
        return []
    
    # Check if ffmpeg exists
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "❌ ffmpeg not found. Please install it via Chocolatey or download:\n"
            "   Chocolatey (recommended): https://chocolatey.org/install\n"
            "   Or download full build: https://www.gyan.dev/ffmpeg/builds/ffmpeg-git-full.7z\n"
            "   Unpack to C:\\ffmpeg and add C:\\ffmpeg\\bin to PATH\n"
        )

    # Check for cancellation before audio extraction
    if cancel_flag and cancel_flag.is_set():
        return []

    wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    
    try:
        # Run ffmpeg to extract audio
        subprocess.run([
            "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
            "-ar", "44100", "-ac", "1", wav_file, "-y"
        ], check=True)

        # Check for cancellation after audio extraction
        if cancel_flag and cancel_flag.is_set():
            return []

        # Read audio file
        wf = wave.open(wav_file, 'rb')
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        wf.close()

        # Check for cancellation before processing
        if cancel_flag and cancel_flag.is_set():
            return []

        samples_per_chunk = int(chunk_size * rate)
        total_chunks = len(audio) // samples_per_chunk
        peaks = []

        pbar = tqdm(total=total_chunks, desc="Audio peak detection")
        
        for i in range(0, len(audio), samples_per_chunk):
            # Check for cancellation every 100 chunks (roughly every 10 seconds of audio)
            if i % (samples_per_chunk * 100) == 0 and cancel_flag and cancel_flag.is_set():
                pbar.close()
                return peaks  # Return whatever peaks we found so far
                
            chunk = audio[i:i+samples_per_chunk]
            rms = np.sqrt(np.mean(chunk**2))
            if 20 * np.log10(rms + 1e-6) > threshold_db:
                peaks.append(i / rate)
            pbar.update(1)
            
        pbar.close()
        return peaks

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"❌ ffmpeg failed: {e}")
    finally:
        # Clean up temporary file
        try:
            import os
            if os.path.exists(wav_file):
                os.remove(wav_file)
        except:
            pass  # Ignore cleanup errors