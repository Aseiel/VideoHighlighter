import os
import re
import time
from typing import List, Dict, Optional

# Optional imports
try:
    import whisper
except ImportError:
    whisper = None

try:
    from googletrans import Translator
except ImportError:
    Translator = None

import cv2

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
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r'([.!?]+)', text)
    lines, current_line = [], ""
    for i in range(0, len(sentences), 2):
        sentence = sentences[i]
        punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
        full_sentence = sentence + punctuation
        if len(current_line + full_sentence) <= max_chars:
            current_line += full_sentence
        else:
            if current_line:
                lines.append(current_line.strip())
            current_line = full_sentence
    if current_line:
        lines.append(current_line.strip())
    final_lines = []
    for line in lines:
        if len(line) <= max_chars:
            final_lines.append(line)
        else:
            words, current = line.split(), ""
            for word in words:
                if len(current + " " + word) <= max_chars:
                    current += (" " + word) if current else word
                else:
                    if current:
                        final_lines.append(current)
                    current = word
            if current:
                final_lines.append(current)
    return final_lines

def create_srt_content(segments, max_chars=80, min_duration=0.8):
    """
    Create SRT content with precise timestamps from original segments.
    Translation text is used, but timing is preserved.
    """
    srt_lines = []
    subtitle_number = 1

    for seg in segments:
        text = clean_subtitle_text(seg['text'])
        if not text or len(text) < 2 or not is_valid_speech(text):
            continue

        start_seconds = float(seg.get('start', 0))
        end_seconds = float(seg.get('end', start_seconds + 2))

        # Ensure minimum duration
        duration = end_seconds - start_seconds
        if duration < min_duration:
            end_seconds = start_seconds + min_duration

        # Split long lines into chunks without changing timing
        chunks = split_long_subtitle(text, max_chars=max_chars)
        if len(chunks) > 1:
            # Split duration evenly for chunks
            chunk_duration = (end_seconds - start_seconds) / len(chunks)
        else:
            chunk_duration = end_seconds - start_seconds

        current_start = start_seconds
        for chunk in chunks:
            current_end = current_start + chunk_duration
            srt_lines.append(str(subtitle_number))
            srt_lines.append(f"{format_timestamp_srt(current_start)} --> {format_timestamp_srt(current_end)}")
            srt_lines.append(chunk)
            srt_lines.append("")

            subtitle_number += 1
            current_start = current_end

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
# LEGACY FUNCTIONS
# --------------------------

translate_segments = None
if Translator:
    def translate_segments(segments, source_lang="en", target_lang="pl"):
        """
        Translate segments with per-segment logging like your manual loop.
        """
        if not segments:
            print("No segments to translate")
            return []

        if source_lang == target_lang:
            print("No translation needed (same language)")
            return segments

        translated_segments = []
        translator = Translator()
        print(f"⏳ Translating {len(segments)} segments from {source_lang} to {target_lang}...")

        for i, seg in enumerate(segments):
            try:
                print(f"Translating {i+1}/{len(segments)}...", end='\r')
                translated_text = translator.translate(seg["text"], src=source_lang, dest=target_lang).text
                translated_segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': translated_text
                })
            except Exception as e:
                print(f"Translation error for segment {i+1}: {e}")
                translated_segments.append({
                    'start': seg['start'],
                    'end': seg['end'],
                    'text': seg['text']
                })

        print(f"\n✅ Translated {len(translated_segments)} segments")
        return translated_segments


def create_highlight_subtitles(original_segments: List[Dict], highlight_segments: List[tuple], 
                               output_path: str, source_lang="en", target_lang=None):
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
    if target_lang and Translator:
        highlight_subtitle_segments = translate_segments(highlight_subtitle_segments, source_lang, target_lang)
    if highlight_subtitle_segments:
        srt_content = create_srt_content(highlight_subtitle_segments)
        with open(output_path, "w", encoding="utf-8-sig") as f:
            f.write(srt_content)
