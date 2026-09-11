import subprocess
import tempfile
import numpy as np
import wave
import sys
from tqdm import tqdm

from modules.app_paths import ffmpeg_exe

def safe_tqdm(*args, **kwargs):
    """
    Safely create tqdm progress bar, handling cases where stderr is None or doesn't have write method.
    This is particularly important for GUI applications and frozen executables.
    """
    # Check if stderr exists and has a write method
    if sys.stderr is None or not hasattr(sys.stderr, "write"):
        kwargs["disable"] = True
    
    # Ensure the file parameter is valid
    if "file" not in kwargs or kwargs.get("file") is None:
        # Use stdout if available and has write method, otherwise use stderr
        if sys.stdout is not None and hasattr(sys.stdout, "write"):
            kwargs["file"] = sys.stdout
        elif sys.stderr is not None and hasattr(sys.stderr, "write"):
            kwargs["file"] = sys.stderr
        else:
            # If both are invalid, disable the progress bar
            kwargs["disable"] = True
    
    return tqdm(*args, **kwargs)

def extract_waveform_data(video_path, num_points=1000):
    """Extract waveform amplitude data for visualization"""
    
    ffmpeg = ffmpeg_exe()

    wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    try:
        # Extract audio
        subprocess.run([
            ffmpeg, "-i", video_path, "-vn", "-acodec", "pcm_s16le",
            "-ar", "44100", "-ac", "1", wav_file, "-y",
            "-hide_banner", "-loglevel", "error"
        ], check=True, capture_output=True)
        
        # Read audio
        wf = wave.open(wav_file, 'rb')
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
        wf.close()
        
        duration = len(audio) / rate
        
        # Downsample for visualization
        step = max(1, len(audio) // num_points)
        waveform = []
        
        for i in range(0, len(audio), step):
            chunk = audio[i:i+step]
            if len(chunk) > 0:
                max_val = np.max(chunk) / 32768.0  # Normalize to [-1, 1]
                min_val = np.min(chunk) / 32768.0
                # RMS energy ~ perceived loudness (broadband events like explosions
                # read hot, not just sharp transients with a high peak).
                rms = float(np.sqrt(np.mean((chunk.astype(np.float64) / 32768.0) ** 2)))
                waveform.append((min_val, max_val, rms))

        return waveform
        
    finally:
        import os
        if os.path.exists(wav_file):
            os.remove(wav_file)

def peaks_in_chunk(chunk, threshold_linear):
    """Samples louder than the threshold and at least as loud as both
    neighbours. Returns (offsets within the chunk, magnitudes).

    **The widening to int32 is the point.** The samples arrive as int16, whose
    range is -32768..32767, so ``abs(-32768)`` does not fit and numpy wraps it
    back to -32768 with an "overflow encountered in scalar absolute" warning.
    The effect is not cosmetic: a sample at full negative scale reads as a
    *negative* magnitude, fails the threshold test, and is dropped — so the one
    thing this function exists to find, the loudest moment in clipped audio,
    was the one thing it could not see.

    Vectorised for a second reason. This runs per sample over the whole
    soundtrack — 274 million iterations for a 1.7 hour recording — and as a
    Python loop it took ten minutes of a run that has better things to do.
    """
    if len(chunk) < 3:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int32)
    magnitude = np.abs(np.asarray(chunk).astype(np.int32))
    inner = magnitude[1:-1]
    is_peak = ((inner > threshold_linear)
               & (inner >= magnitude[:-2])
               & (inner >= magnitude[2:]))
    offsets = np.nonzero(is_peak)[0] + 1
    return offsets, magnitude[offsets]


def extract_audio_peaks(video_path, threshold_db=-20, chunk_duration_ms=10, merge_distance_ms=50, cancel_flag=None):
    """Extract precise audio peaks with proper event detection"""
    
    # Check for cancellation at start
    if cancel_flag and cancel_flag.is_set():
        return []
    
    # Resolve ffmpeg (system PATH, else the bundled imageio-ffmpeg binary)
    ffmpeg = ffmpeg_exe()

    wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    try:
        # Run ffmpeg to extract audio
        subprocess.run([
            ffmpeg, "-i", video_path, "-vn", "-acodec", "pcm_s16le",
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

        # Convert threshold from dB to linear amplitude (for 16-bit PCM)
        threshold_linear = 10 ** (threshold_db / 20) * 32768.0
        
        # Calculate samples per analysis chunk
        samples_per_chunk = int((chunk_duration_ms / 1000.0) * rate)
        if samples_per_chunk < 1:
            samples_per_chunk = 1
        
        # Samples to merge events (convert ms to samples)
        merge_distance_samples = int((merge_distance_ms / 1000.0) * rate)
        
        # Store raw peaks with their amplitude
        raw_peaks = []
        
        # Create progress bar with safe handling for GUI/EXE environments
        total_chunks = len(audio) // samples_per_chunk
        pbar = safe_tqdm(
            total=total_chunks, 
            desc="Audio peak detection"
        )
        
        for chunk_start in range(0, len(audio), samples_per_chunk):
            # Check for cancellation every 100 chunks
            if chunk_start % (samples_per_chunk * 100) == 0 and cancel_flag and cancel_flag.is_set():
                pbar.close()
                break
                
            chunk_end = min(chunk_start + samples_per_chunk, len(audio))
            chunk = audio[chunk_start:chunk_end]
            
            if len(chunk) == 0:
                pbar.update(1)
                continue
            
            # Local maxima above the threshold, in one pass — see peaks_in_chunk
            # on why this is not a per-sample loop and not int16 arithmetic.
            offsets, magnitudes = peaks_in_chunk(chunk, threshold_linear)
            for offset, magnitude in zip(offsets, magnitudes):
                exact_time = (chunk_start + int(offset)) / rate
                raw_peaks.append((exact_time, int(magnitude)))
            
            pbar.update(1)
        
        pbar.close()
        
        if not raw_peaks:
            return []
        
        # Sort peaks by time
        raw_peaks.sort(key=lambda x: x[0])
        
        # Merge nearby peaks into events
        peaks = []
        current_event_start = raw_peaks[0][0]
        current_event_end = raw_peaks[0][0]
        current_max_amplitude = raw_peaks[0][1]
        
        for i in range(1, len(raw_peaks)):
            time_diff = raw_peaks[i][0] - raw_peaks[i-1][0]
            
            if time_diff * 1000 < merge_distance_ms:  # Convert to ms
                # Same event, extend end time
                current_event_end = raw_peaks[i][0]
                current_max_amplitude = max(current_max_amplitude, raw_peaks[i][1])
            else:
                # New event, save previous one
                event_center = (current_event_start + current_event_end) / 2
                peaks.append(round(event_center, 3))
                
                # Start new event
                current_event_start = raw_peaks[i][0]
                current_event_end = raw_peaks[i][0]
                current_max_amplitude = raw_peaks[i][1]
        
        # Add the last event
        event_center = (current_event_start + current_event_end) / 2
        peaks.append(round(event_center, 3))
        
        print(f"✓ Found {len(peaks)} audio peaks")
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