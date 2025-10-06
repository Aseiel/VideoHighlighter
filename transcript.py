import os
import whisper
import torch
from tqdm import tqdm
import re
import subprocess

def is_repetitive_hallucination(text, threshold=0.7):
    """Detect repetitive segments like 'ha ha ha'"""
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    words = clean_text.split()
    if len(words) < 3:
        return False
    word_counts = {}
    for w in words:
        word_counts[w] = word_counts.get(w, 0) + 1
    most_common_count = max(word_counts.values())
    repetition_ratio = most_common_count / len(words)
    return repetition_ratio > threshold

def is_valid_speech(text):
    """Check if segment is valid speech, not hallucination"""
    hallucination_patterns = [
        r'^(oh+,?\s*)+$',
        r'^(ah+,?\s*)+$',
        r'^(ha+,?\s*)+$',
        r'^(um+,?\s*)+$',
        r'^(uh+,?\s*)+$',
    ]
    clean_text = text.lower().strip()
    if len(clean_text) < 2:
        return False
    for pattern in hallucination_patterns:
        if re.match(pattern, clean_text):
            return False
    if is_repetitive_hallucination(text):
        return False
    return True

def split_audio(video_file, chunk_length=600):
    """
    Split audio into chunks using ffmpeg (default: 600s = 10 min).
    Returns list of chunk file paths.
    """
    # Get the directory where the video is located
    video_dir = os.path.dirname(os.path.abspath(video_file))
    base, _ = os.path.splitext(os.path.basename(video_file))
    
    out_pattern = os.path.join(video_dir, f"{base}_chunk_%03d.wav")

    # Remove old chunks if they exist (safety cleanup)
    for f in os.listdir(video_dir):
        if f.startswith(base + "_chunk_") and f.endswith(".wav"):
            try:
                os.remove(os.path.join(video_dir, f))
            except OSError:
                pass

    # Use ffmpeg to split into fixed-length WAV files
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_file,
        "-f", "segment", "-segment_time", str(chunk_length),
        "-c:a", "pcm_s16le", "-ar", "16000",
        out_pattern
    ], check=True)

    # Return full paths to chunks in the video's directory
    chunks = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
              if f.startswith(base + "_chunk_") and f.endswith(".wav")]
    
    return sorted(chunks)

def get_transcript_segments(video_file, model_name="small", progress_fn=None, log_fn=print, chunk_length=600, cleanup=True):
    """
    Transcribe video safely by splitting into chunks.
    - Uses Whisper for transcription
    - Filters hallucinations
    - Preserves timestamps with offsets
    - Shows progress via progress_fn
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_fn(f"Using device for Whisper: {device}")

    model = whisper.load_model(model_name, device=device)
    log_fn("Splitting video into chunks...")

    chunks = split_audio(video_file, chunk_length=chunk_length)
    log_fn(f"Created {len(chunks)} chunks")

    all_segments = []
    for idx, chunk in enumerate(chunks):
        # Progress indicator
        if progress_fn:
            progress_fn(
                int((idx / len(chunks)) * 60),
                100,
                "Transcription",
                f"Processing chunk {idx+1}/{len(chunks)}"
            )

        log_fn(f"➡️ Transcribing chunk {idx+1}/{len(chunks)}: {chunk}")

        result = model.transcribe(
            chunk,
            language="en",
            task="transcribe",
            temperature=0.0,  # No randomness
            beam_size=1,      # Greedy decoding
            best_of=1,        # Single attempt
            patience=1.0,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,  # Detect repetition
            logprob_threshold=-1.0,           # Filter low confidence
            no_speech_threshold=0.6,          # Silence detection
            verbose=False
        )

        # Offset for proper timestamps
        offset = idx * chunk_length
        for seg in result.get("segments", []):
            text = seg["text"].strip()
            if len(text) >= 3 and is_valid_speech(text):
                all_segments.append({
                    "start": float(seg["start"]) + offset,
                    "end": float(seg["end"]) + offset,
                    "text": text
                })

    if progress_fn:
        progress_fn(95, 100, "Transcription", "Complete")

    if cleanup:
        log_fn("🧹 Cleaning up chunk files...")
        for f in chunks:
            try:
                os.remove(f)
            except OSError:
                pass

    log_fn(f"✅ Transcript ready: {len(all_segments)} segments (from {len(chunks)} chunks)")
    return all_segments

def search_transcript_for_keywords(transcript_segments, keywords, context_seconds=5):
    """Search transcript for keywords and return matching segments with context"""
    if not keywords or not transcript_segments:
        return []
    
    # Normalize keywords to lowercase
    keywords = [kw.lower().strip() for kw in keywords if kw.strip()]
    if not keywords:
        return []
    
    matches = []
    
    for seg in transcript_segments:
        text_lower = seg["text"].lower()
        
        # Check if any keyword appears in this segment
        for keyword in keywords:
            if keyword in text_lower:
                # Find context - segments within context_seconds
                start_time = seg["start"] - context_seconds
                end_time = seg["end"] + context_seconds
                
                context_segments = [
                    s for s in transcript_segments 
                    if s["start"] >= start_time and s["end"] <= end_time
                ]
                
                matches.append({
                    "keyword": keyword,
                    "main_segment": seg,
                    "context_segments": context_segments,
                    "start": max(0, start_time),
                    "end": end_time
                })
                break  # Don't duplicate same segment for multiple keywords
    
    return matches