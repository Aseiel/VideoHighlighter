import os
import whisper
import torch
from tqdm import tqdm
import re

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

def get_transcript_segments(video_file, model_name="small", progress_fn=None, log_fn=print):
    """Transcribe video with Whisper, filter hallucinations, show progress"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_fn(f"Using device for Whisper: {device}")
    
    model = whisper.load_model(model_name, device=device)
    log_fn("Transcribing video with Whisper (this may take a while)...")
    
    if progress_fn:
        progress_fn(0, 100, "Transcription", "Starting Whisper...")
    
    result = model.transcribe(
        video_file,
        language="en",
        task="transcribe",
        temperature=0.0,  # No randomness
        beam_size=1,  # Greedy decoding only
        best_of=1,  # Single attempt
        patience=1.0,
        condition_on_previous_text=False,  # Don't use context
        compression_ratio_threshold=2.4,  # Detect repetition
        logprob_threshold=-1.0,  # Filter low confidence
        no_speech_threshold=0.6,  # Better silence detection
        verbose=False
    )
    
    segments = result.get("segments", [])
    valid_segments = []

    log_fn("Filtering transcript segments for hallucinations...")
    if progress_fn:
        progress_fn(70, 100, "Transcription", "Filtering segments...")
    
    for i, seg in enumerate(segments):
        # Progress indicator
        if progress_fn and i % 5 == 0:
            progress_fn(70 + (i / len(segments)) * 25, 100, "Transcription", f"Processing {i+1}/{len(segments)}")
        
        text = seg["text"].strip()
        if len(text) < 3 or not is_valid_speech(text):
            continue
        valid_segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": text
        })

    if progress_fn:
        progress_fn(95, 100, "Transcription", "Complete")
    
    log_fn(f"Transcript ready: {len(valid_segments)} segments (filtered from {len(segments)})")
    return valid_segments

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