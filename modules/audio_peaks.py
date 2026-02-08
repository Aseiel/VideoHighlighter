import shutil
import subprocess
import tempfile
import numpy as np
import wave
from tqdm import tqdm

def extract_waveform_data(video_path, num_points=1000):
    """Extract waveform amplitude data for visualization"""
    
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("❌ ffmpeg not found")
    
    wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    
    try:
        # Extract audio
        subprocess.run([
            "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
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
                waveform.append((min_val, max_val))
        
        return waveform
        
    finally:
        import os
        if os.path.exists(wav_file):
            os.remove(wav_file)

def extract_audio_peaks(video_path, threshold_db=-20, chunk_duration_ms=10, merge_distance_ms=50, cancel_flag=None):
    """Extract precise audio peaks with proper event detection"""
    
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
        
        # Process audio in chunks with overlap for better detection
        pbar = tqdm(total=len(audio) // samples_per_chunk, desc="Audio peak detection")
        
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
            
            # Find local maxima within this chunk
            # Simple peak detection: find samples where amplitude exceeds threshold
            # and is greater than neighbors
            for i in range(1, len(chunk)-1):
                if abs(chunk[i]) > threshold_linear:
                    # Check if it's a local peak (greater than neighbors)
                    if abs(chunk[i]) >= abs(chunk[i-1]) and abs(chunk[i]) >= abs(chunk[i+1]):
                        # Calculate exact timestamp
                        exact_sample = chunk_start + i
                        exact_time = exact_sample / rate
                        amplitude = abs(chunk[i])
                        raw_peaks.append((exact_time, amplitude))
            
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