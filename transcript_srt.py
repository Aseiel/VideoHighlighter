import os
import re
import time
import random
import whisper
import cv2
from typing import List, Dict, Optional
from googletrans import Translator

# --------------------------
# VIDEO & TIMESTAMP UTILITIES
# --------------------------

def get_video_fps(video_path: str) -> float:
    """Get video FPS for proper timing"""
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps if fps > 0 else 25.0
    except:
        print("⚠️ Could not detect FPS, using 25.0")
        return 25.0

def format_timestamp_srt(seconds: float) -> str:
    """Convert seconds to SRT timestamp format"""
    seconds = max(0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    milliseconds = min(999, max(0, milliseconds))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

# --------------------------
# TEXT CLEANING & VALIDATION
# --------------------------

def clean_subtitle_text(text: str, max_line_chars=50) -> str:
    """Clean and format subtitle text"""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'([.!?])([A-Za-zĄąĆćĘęŁłŃńÓóŚśŹźŻż])', r'\1 \2', text)
    text = re.sub(r'(,)([^\s])', r'\1 \2', text)
    text = re.sub(r'^[-\s]*', '', text)
    text = re.sub(r'[-\s]*$', '', text)
    if text:
        text = text[0].upper() + text[1:]

    # Split into lines
    sentences = re.split(r'([.!?])', text)
    lines, current = [], ""
    for i in range(0, len(sentences), 2):
        sentence = sentences[i].strip()
        if not sentence:
            continue
        punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
        full_sentence = (sentence + punctuation).strip()
        if len(current) + len(full_sentence) + 1 <= max_line_chars:
            current = (current + " " + full_sentence).strip()
        else:
            if current:
                lines.append(current)
            current = full_sentence
    if current:
        lines.append(current)
    return "\n".join(lines)

def is_repetitive_hallucination(text: str, threshold=0.7) -> bool:
    clean_text = re.sub(r'[^\w\s]', '', text.lower())
    words = clean_text.split()
    if len(words) < 3:
        return False
    counts = {w: words.count(w) for w in words}
    return max(counts.values()) / len(words) > threshold

def is_valid_speech(text: str) -> bool:
    patterns = [r'^(oh+,?\s*)+$', r'^(ah+,?\s*)+$', r'^(ha+,?\s*)+$', r'^(um+,?\s*)+$', r'^(uh+,?\s*)+$']
    clean_text = text.lower().strip()
    if len(clean_text) < 2:
        return False
    if any(re.match(p, clean_text) for p in patterns):
        return False
    if is_repetitive_hallucination(text):
        return False
    return True

def remove_exact_duplicates(segments: List[Dict]) -> List[Dict]:
    seen, unique_segments = set(), []
    for seg in segments:
        key = seg['text'].strip().lower()
        if key not in seen:
            seen.add(key)
            unique_segments.append(seg)
    return unique_segments

# --------------------------
# SEGMENT MERGING
# --------------------------

def smart_merge_segments(segments: List[Dict], gap_threshold=0.5, max_merge_duration=6.0) -> List[Dict]:
    """Conservative merging of speech segments"""
    if not segments:
        return segments
    merged = [segments[0]]
    for current in segments[1:]:
        last = merged[-1]
        gap = current['start'] - last['end']
        merged_duration = current['end'] - last['start']
        ends_with_sentence = last['text'].strip().endswith(('.', '!', '?'))
        should_merge = (
            gap < gap_threshold and
            merged_duration <= max_merge_duration and
            (last['end'] - last['start']) < 3.0 and
            (current['end'] - current['start']) < 3.0 and
            not ends_with_sentence
        )
        if should_merge:
            merged[-1] = {
                'start': last['start'],
                'end': current['end'],
                'text': last['text'] + " " + current['text']
            }
        else:
            merged.append(current)
    return merged

# --------------------------
# SRT & TRANSCRIPT GENERATION
# --------------------------

def split_long_subtitle(text: str, max_chars=80) -> List[str]:
    """
    Split subtitle text into balanced chunks.
    Avoids leaving 1-word segments.
    """
    if len(text) <= max_chars:
        return [text]

    # First pass: try punctuation-based splitting
    sentences = re.split(r'([.!?]+)', text)
    chunks, current = [], ""
    for i in range(0, len(sentences), 2):
        sentence = sentences[i].strip()
        punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
        full_sentence = (sentence + punctuation).strip()
        if not full_sentence:
            continue
        if len((current + " " + full_sentence).strip()) <= max_chars:
            current = (current + " " + full_sentence).strip()
        else:
            if current:
                chunks.append(current)
            current = full_sentence
    if current:
        chunks.append(current)

    # Second pass: rebalance overly long or unbalanced chunks
    final_lines = []
    for chunk in chunks:
        if len(chunk) > max_chars:
            words = chunk.split()
            mid = len(words) // 2
            left, right = " ".join(words[:mid]), " ".join(words[mid:])

            # If badly unbalanced, fall back to greedy slicing
            if len(right) < len(left) * 0.3 or len(left) > max_chars:
                current, parts = "", []
                for w in words:
                    if len(current + " " + w) <= max_chars:
                        current = (current + " " + w).strip()
                    else:
                        parts.append(current)
                        current = w
                if current:
                    parts.append(current)
                final_lines.extend(parts)
            else:
                final_lines.extend([left.strip(), right.strip()])
        else:
            final_lines.append(chunk.strip())

    return final_lines

def calculate_reading_time(text: str, chars_per_second=20) -> float:
    """
    Calculate minimum reading time based on character count.
    Standard: 15-20 characters per second for comfortable reading.
    """
    char_count = len(text)
    return char_count / chars_per_second

def adjust_subtitle_duration(start: float, end: float, text: str, 
                             min_duration=0.8, chars_per_second=20) -> tuple:
    """
    Adjust subtitle duration to meet reading time standards.
    Returns (adjusted_start, adjusted_end)
    """
    duration = end - start
    required_duration = max(min_duration, calculate_reading_time(text, chars_per_second))
    
    if duration < required_duration:
        # Extend duration to meet reading time
        end = start + required_duration
    
    return start, end

def create_srt_content(segments, max_chars=80, min_duration=0.8, chars_per_second=15):
    """
    Create SRT content with precise timestamps from original segments.
    Translation text is used, but timing is preserved.
    Each chunk gets proper duration based on reading speed.
    """
    srt_lines = []
    subtitle_number = 1

    for idx, seg in enumerate(segments):
        text = clean_subtitle_text(seg['text'])
        if not text or len(text) < 2 or not is_valid_speech(text):
            continue

        start_seconds = float(seg.get('start', 0))
        end_seconds = float(seg.get('end', start_seconds + 2))
        
        # Check when the next segment starts to avoid overlap
        next_segment_start = segments[idx + 1]['start'] if idx + 1 < len(segments) else None

        # Split long lines into chunks first
        chunks = split_long_subtitle(text, max_chars=max_chars)
        
        # Calculate total required duration for all chunks
        total_required_duration = sum(max(min_duration, len(chunk) / chars_per_second) for chunk in chunks)
        original_duration = end_seconds - start_seconds
        
        # Calculate available duration (don't extend into next segment)
        if next_segment_start is not None:
            max_available_duration = next_segment_start - start_seconds - 0.1  # 0.1s gap
            available_duration = max(original_duration, min(total_required_duration, max_available_duration))
        else:
            # Last segment, can extend as needed
            available_duration = max(original_duration, total_required_duration)
        
        current_start = start_seconds
        for i, chunk in enumerate(chunks):
            # Calculate duration for this chunk based on its length
            chunk_required_duration = max(min_duration, len(chunk) / chars_per_second)
            
            # Scale the chunk duration proportionally to fit within available time
            if total_required_duration > 0:
                chunk_duration = chunk_required_duration * (available_duration / total_required_duration)
            else:
                chunk_duration = min_duration
            
            chunk_end = current_start + chunk_duration
            
            srt_lines.append(str(subtitle_number))
            srt_lines.append(f"{format_timestamp_srt(current_start)} --> {format_timestamp_srt(chunk_end)}")
            srt_lines.append(chunk)
            srt_lines.append("")

            subtitle_number += 1
            current_start = chunk_end

    return "\n".join(srt_lines)

def create_enhanced_transcript(segments: List[Dict], pause_threshold=2.0) -> str:
    transcript_parts = []
    last_end = 0
    last_timestamp = 0
    for i, seg in enumerate(segments):
        current_start = seg['start']
        if i > 0:
            gap = current_start - last_end
            if gap >= pause_threshold:
                transcript_parts.append(f"\n\n[{gap:.1f}s pause]\n\n")
        show_timestamp = (
            i == 0 or
            (current_start - last_timestamp > 30) or
            (i > 0 and (current_start - segments[i-1]['end']) >= pause_threshold)
        )
        if show_timestamp:
            transcript_parts.append(f"[{current_start:.1f}s] ")
            last_timestamp = current_start
        transcript_parts.append(seg['text'])
        if not seg['text'].strip().endswith(('.', '!', '?')):
            transcript_parts.append(".")
        transcript_parts.append(" ")
        last_end = seg['end']
    return "".join(transcript_parts).strip()

# --------------------------
# TRANSLATION (LLM + FALLBACK)
# --------------------------

def get_llm_translator():
    """Try to detect local LLM availability, return backend name or None"""
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5,
                        encoding='utf-8', errors='replace')
        if result.returncode == 0:
            return "ollama"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return None

def translate_with_llm(text, source_lang="en", target_lang="pl", model="llama3"):
    """Translate a single text using local LLM via ollama"""
    import subprocess

    prompt = (
        f"Translate the following text from {source_lang} to {target_lang}. "
        f"Return ONLY the translated text, nothing else. "
        f"Keep the same tone and style. Do not add quotes or explanations.\n\n"
        f"{text}"
    )

    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True, text=True, timeout=60,
            encoding='utf-8', errors='replace'
        )
        if result.returncode == 0 and result.stdout.strip():
            translated = result.stdout.strip()
            # Remove common LLM artifacts
            translated = re.sub(r'^["\'](.*)["\']$', r'\1', translated)
            translated = re.sub(
                r'^(Here is the translation:|Translation:|Translated text:)\s*',
                '', translated, flags=re.IGNORECASE
            )
            return translated.strip()
    except Exception as e:
        print(f"⚠️ LLM translation failed: {e}")

    return None

def translate_batch_with_llm(texts, source_lang="en", target_lang="pl",
                             model="llama3", batch_size=10):
    """Translate multiple texts in batches for efficiency"""
    import subprocess

    results = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(f"  🦙 Batch {batch_num}/{total_batches} ({len(batch)} segments)...")

        numbered = "\n".join(f"{j+1}. {t}" for j, t in enumerate(batch))
        prompt = (
            f"Translate each numbered line from {source_lang} to {target_lang}. "
            f"Return ONLY the translations, one per line, keeping the same numbering. "
            f"Do not add explanations or extra text.\n\n{numbered}"
        )

        try:
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True, text=True, timeout=120,
                encoding='utf-8', errors='replace'
            )
            if result.returncode == 0 and result.stdout.strip():
                lines = result.stdout.strip().split("\n")
                parsed = []
                for line in lines:
                    cleaned = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
                    if cleaned:
                        parsed.append(cleaned)

                # If we got the right count, use batch result
                if len(parsed) == len(batch):
                    results.extend(parsed)
                    continue
                else:
                    print(f"  ⚠️ Batch returned {len(parsed)} lines, expected {len(batch)}, falling back to individual")
        except Exception as e:
            print(f"  ⚠️ Batch LLM translation failed: {e}")

        # Fallback: translate individually for this batch
        for j, text in enumerate(batch):
            translated = translate_with_llm(text, source_lang, target_lang, model)
            if translated:
                results.append(translated)
            else:
                print(f"  ⚠️ LLM failed for segment {i+j+1}, keeping original")
                results.append(text)

    return results

def safe_translate(translator, text, src, dest, retries=3, delay=1.0):
    """
    Try translating with googletrans with retries and exponential backoff.
    Falls back to original text if all retries fail.
    """
    for attempt in range(retries):
        try:
            result = translator.translate(text, src=src, dest=dest)
            return result.text
        except Exception as e:
            print(f"⚠️ Translation failed (attempt {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                sleep_time = delay * (2 ** attempt) + random.uniform(0, 0.5)
                time.sleep(sleep_time)
            else:
                return text  # fallback

def translate_segments(segments, source_lang="en", target_lang="pl"):
    """
    Translate subtitle segments.
    Strategy: try local LLM (llama via ollama) first for better quality,
    fall back to googletrans if LLM is unavailable.
    """
    if not segments:
        print("No segments to translate")
        return []

    if source_lang == target_lang:
        print("No translation needed (same language)")
        return segments

    print(f"⏳ Translating {len(segments)} segments from {source_lang} to {target_lang}...")

    # --- Try LLM first ---
    llm_backend = get_llm_translator()
    if llm_backend:
        print(f"🦙 Using local LLM ({llm_backend}) for translation (better quality)")
        texts = [seg["text"] for seg in segments]
        translated_texts = translate_batch_with_llm(texts, source_lang, target_lang)

        if len(translated_texts) == len(segments):
            translated_segments = []
            for seg, translated_text in zip(segments, translated_texts):
                translated_segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': translated_text
                })
            print(f"✅ Translated {len(translated_segments)} segments via LLM")
            return translated_segments
        else:
            print(f"⚠️ LLM returned {len(translated_texts)} translations for {len(segments)} segments, falling back")

    # --- Fallback: googletrans ---
    try:
        translator = Translator()
        print("🌐 Using googletrans for translation (LLM not available)")
    except Exception as e:
        print(f"❌ No translation backend available: {e}")
        print("   Install ollama + llama3 for better translations, or pip install googletrans==4.0.0-rc1")
        return segments

    translated_segments = []
    for i, seg in enumerate(segments):
        translated_text = safe_translate(translator, seg["text"], src=source_lang, dest=target_lang)
        translated_segments.append({
            'start': seg['start'],
            'end': seg['end'],
            'text': translated_text
        })
        print(f"  Translated {i+1}/{len(segments)}", end='\r')

    print(f"\n✅ Translated {len(translated_segments)} segments via googletrans")
    return translated_segments

# --------------------------
# FILE GENERATION
# --------------------------

def create_srt_file(segments, output_path, source_lang="en", target_lang=None):
    """Create SRT subtitle file from transcript segments with optional translation"""
    if target_lang and target_lang != source_lang:
        print(f"Translating subtitles from {source_lang} to {target_lang}...")
        segments = translate_segments(segments, source_lang, target_lang)

    srt_content = create_srt_content(segments)

    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write(srt_content)

    print(f"SRT file saved: {output_path}")    

def create_highlight_subtitles(original_segments: List[Dict], highlight_segments: List[tuple], 
                               output_path: str, source_lang="en", target_lang=None):
    """Create SRT subtitles for highlight video from original transcript segments"""
    highlight_subtitle_segments = []
    current_time_offset = 0.0
    for h_start, h_end in highlight_segments:
        overlapping_segments = []
        for seg in original_segments:
            s, e = seg['start'], seg['end']
            if s < h_end and e > h_start:
                overlap_start = max(s, h_start)
                overlap_end = min(e, h_end)
                if overlap_end - overlap_start > 0.5:
                    new_start = current_time_offset + (overlap_start - h_start)
                    new_end = current_time_offset + (overlap_end - h_start)
                    overlapping_segments.append({'start': new_start, 'end': new_end, 'text': seg['text']})
        highlight_subtitle_segments.extend(overlapping_segments)
        current_time_offset += (h_end - h_start)
    if target_lang:
        highlight_subtitle_segments = translate_segments(highlight_subtitle_segments, source_lang, target_lang)
    if highlight_subtitle_segments:
        srt_content = create_srt_content(highlight_subtitle_segments)
        with open(output_path, "w", encoding="utf-8-sig") as f:
            f.write(srt_content)
        print(f"✅ Highlight subtitles saved: {output_path}")
    else:
        print("⚠️ No subtitle segments overlap with highlights")