"""
speaker_utils.py — Speaker diarization & gender estimation module

Uses Resemblyzer (GE2E LSTM) for speaker embeddings + sklearn clustering.
No HuggingFace token, no gated models, no torchaudio dependency.

Dependencies:
  pip install resemblyzer librosa scikit-learn

Optional (better VAD):
  pip install webrtcvad
"""

import os
import re
import numpy as np
from typing import List, Dict, Optional, Tuple


# ==================================================
# VOICE ACTIVITY DETECTION (VAD)
# ==================================================

def detect_speech_segments(audio_path: str, min_speech_duration: float = 0.5,
                            min_silence_duration: float = 0.3,
                            log_fn=print) -> Optional[List[Tuple[float, float]]]:
    """
    Detect speech segments in audio.
    Tries WebRTC VAD first (fast, reliable), falls back to energy-based.
    
    Returns list of (start_sec, end_sec) tuples for speech regions.
    """
    segments = _vad_webrtc(audio_path, min_speech_duration, min_silence_duration, log_fn)
    if segments is not None:
        return segments

    return _vad_energy(audio_path, min_speech_duration, min_silence_duration, log_fn)


def _vad_webrtc(audio_path: str, min_speech_duration: float,
                min_silence_duration: float,
                log_fn=print) -> Optional[List[Tuple[float, float]]]:
    """WebRTC VAD — lightweight, fast, no torch dependency."""
    try:
        import webrtcvad
        import wave
        import struct
    except ImportError:
        log_fn("  ℹ️ webrtcvad not installed, trying energy-based VAD")
        return None

    try:
        import librosa
        import soundfile as sf
    except ImportError:
        return None

    try:
        # Load and convert to 16kHz mono 16-bit PCM (webrtcvad requirement)
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # Convert to 16-bit PCM bytes
        pcm_data = (y * 32767).astype(np.int16).tobytes()
        
        vad = webrtcvad.Vad(2)  # Aggressiveness: 0-3 (2 = balanced)
        
        # Process in 30ms frames (webrtcvad requires 10/20/30ms)
        frame_duration_ms = 30
        frame_size = int(sr * frame_duration_ms / 1000)  # samples per frame
        frame_bytes = frame_size * 2  # 16-bit = 2 bytes per sample
        
        segments = []
        in_speech = False
        start = 0.0
        
        for i in range(0, len(pcm_data) - frame_bytes, frame_bytes):
            frame = pcm_data[i:i + frame_bytes]
            t = i / 2 / sr  # current time in seconds
            
            is_speech = vad.is_speech(frame, sr)
            
            if is_speech and not in_speech:
                start = t
                in_speech = True
            elif not is_speech and in_speech:
                if t - start >= min_speech_duration:
                    segments.append((start, t))
                in_speech = False
        
        if in_speech:
            end_t = len(y) / sr
            if end_t - start >= min_speech_duration:
                segments.append((start, end_t))
        
        # Merge segments with small gaps
        merged = []
        for seg in segments:
            if merged and (seg[0] - merged[-1][1]) < min_silence_duration:
                merged[-1] = (merged[-1][0], seg[1])
            else:
                merged.append(seg)
        
        log_fn(f"  🔊 WebRTC VAD: {len(merged)} speech segments detected")
        return merged

    except Exception as e:
        log_fn(f"  ⚠️ WebRTC VAD failed: {e}")
        return None


def _vad_energy(audio_path: str, min_speech_duration: float,
                min_silence_duration: float, log_fn=print) -> Optional[List[Tuple[float, float]]]:
    """Simple energy-based VAD fallback."""
    try:
        import librosa
    except ImportError:
        log_fn("⚠️ librosa not installed — cannot run VAD")
        return None

    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as e:
        log_fn(f"⚠️ Could not load audio: {e}")
        return None

    frame_length = int(0.025 * sr)
    hop_length = int(0.010 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    threshold = np.mean(rms) * 0.5
    is_speech = rms > threshold

    segments = []
    in_speech = False
    start = 0.0

    for i, speech in enumerate(is_speech):
        t = i * hop_length / sr
        if speech and not in_speech:
            start = t
            in_speech = True
        elif not speech and in_speech:
            if t - start >= min_speech_duration:
                segments.append((start, t))
            in_speech = False

    if in_speech:
        end_t = len(y) / sr
        if end_t - start >= min_speech_duration:
            segments.append((start, end_t))

    # Merge segments with small gaps
    merged = []
    for seg in segments:
        if merged and (seg[0] - merged[-1][1]) < min_silence_duration:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)

    log_fn(f"  🔊 Energy VAD: {len(merged)} speech segments detected")
    return merged


# ==================================================
# SPEAKER EMBEDDING EXTRACTION (Resemblyzer GE2E)
# ==================================================

def extract_speaker_embeddings(audio_path: str,
                                speech_segments: List[Tuple[float, float]],
                                min_chunk_duration: float = 0.8,
                                max_segments: int = 500,
                                log_fn=print) -> Optional[Tuple[np.ndarray, List[Tuple[float, float]]]]:
    """
    Extract speaker embeddings for each speech segment using
    Resemblyzer's GE2E LSTM encoder (~17MB model, auto-downloads).
    
    No HuggingFace token, no torchaudio, no gated models.
    
    Returns:
        (embeddings_array, valid_segments) or None if unavailable.
    """
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        log_fn("⚠️ resemblyzer not installed — cannot extract speaker embeddings")
        log_fn("   Install: pip install resemblyzer")
        return None

    try:
        import librosa
    except ImportError:
        log_fn("⚠️ librosa not installed — needed for audio loading")
        return None

    try:
        log_fn("  🧠 Loading speaker embedding model (GE2E)...")
        encoder = VoiceEncoder(device="cpu")
    except Exception as e:
        log_fn(f"⚠️ Could not load embedding model: {e}")
        return None

    try:
        # Load audio as float32 mono 16kHz (resemblyzer expects this)
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as e:
        log_fn(f"⚠️ Could not load audio: {e}")
        return None

    # Subsample if too many segments
    if len(speech_segments) > max_segments:
        log_fn(f"  ℹ️ Subsampling {len(speech_segments)} → {max_segments} segments")
        indices = np.linspace(0, len(speech_segments) - 1, max_segments, dtype=int)
        speech_segments = [speech_segments[i] for i in indices]

    embeddings = []
    valid_segments = []

    for start, end in speech_segments:
        start_sample = int(start * sr)
        end_sample = min(int(end * sr), len(y))
        chunk = y[start_sample:end_sample]

        # Resemblyzer needs at least ~0.8s for reliable embeddings
        if len(chunk) < int(sr * min_chunk_duration):
            continue

        try:
            # Preprocess (normalize, trim silence) then embed
            processed = preprocess_wav(chunk, source_sr=sr)
            if len(processed) < int(sr * 0.5):
                continue
            
            emb = encoder.embed_utterance(processed)
            
            if emb is not None and emb.ndim == 1 and len(emb) > 0:
                embeddings.append(emb)
                valid_segments.append((start, end))
        except Exception:
            continue

    if not embeddings:
        log_fn("⚠️ No valid embeddings extracted")
        return None

    log_fn(f"  ✅ Extracted {len(embeddings)} speaker embeddings")
    return np.array(embeddings), valid_segments


# ==================================================
# SPEAKER CLUSTERING
# ==================================================

def cluster_speakers(embeddings: np.ndarray,
                     num_speakers: Optional[int] = None,
                     min_speakers: int = 1,
                     max_speakers: int = 10,
                     log_fn=print) -> np.ndarray:
    """
    Cluster speaker embeddings to assign speaker IDs.
    Uses Agglomerative Clustering with cosine distance.
    
    If num_speakers is not known, automatically estimates via silhouette score.
    
    Returns:
        Array of cluster labels (one per embedding).
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize

    embeddings_norm = normalize(embeddings)

    if num_speakers is not None:
        log_fn(f"  🔢 Clustering into {num_speakers} speakers (user-specified)")
        clustering = AgglomerativeClustering(
            n_clusters=num_speakers,
            metric='cosine',
            linkage='average'
        )
        return clustering.fit_predict(embeddings_norm)

    # Auto-detect speaker count
    if len(embeddings) < 3:
        log_fn("  ℹ️ Too few segments for auto-detection, assuming 1 speaker")
        return np.zeros(len(embeddings), dtype=int)

    log_fn("  🔍 Auto-detecting number of speakers...")
    best_score = -1
    best_k = 2
    best_labels = None

    actual_max = min(max_speakers or 10, len(embeddings) - 1)
    actual_min = max(min_speakers or 1, 2)

    if actual_min > actual_max:
        return np.zeros(len(embeddings), dtype=int)

    for k in range(actual_min, actual_max + 1):
        try:
            clustering = AgglomerativeClustering(
                n_clusters=k,
                metric='cosine',
                linkage='average'
            )
            labels = clustering.fit_predict(embeddings_norm)
            score = silhouette_score(embeddings_norm, labels, metric='cosine')

            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels
        except Exception:
            continue

    if best_score < 0.1:
        log_fn(f"  ℹ️ Low clustering confidence (score={best_score:.2f}), "
               f"possibly 1 speaker — using {best_k} anyway")
    else:
        log_fn(f"  ✅ Detected {best_k} speakers (silhouette={best_score:.2f})")

    return best_labels if best_labels is not None else np.zeros(len(embeddings), dtype=int)


# ==================================================
# BUILD DIARIZATION TIMELINE
# ==================================================

def build_diarization_timeline(speech_segments: List[Tuple[float, float]],
                                labels: np.ndarray) -> List[Dict]:
    """
    Build diarization output from speech segments and cluster labels.
    Merges consecutive segments from same speaker if gap < 500ms.
    """
    timeline = []
    for (start, end), label in zip(speech_segments, labels):
        speaker_id = f"SPEAKER_{int(label):02d}"
        timeline.append({
            'start': round(start, 3),
            'end': round(end, 3),
            'speaker': speaker_id
        })

    # Merge consecutive same-speaker segments
    merged = []
    for seg in timeline:
        if merged and merged[-1]['speaker'] == seg['speaker']:
            gap = seg['start'] - merged[-1]['end']
            if gap < 0.5:
                merged[-1]['end'] = seg['end']
                continue
        merged.append(dict(seg))

    return merged


# ==================================================
# GENDER ESTIMATION — PITCH-BASED (librosa)
# ==================================================

def estimate_gender_by_pitch(audio_path: str, diarization: List[Dict],
                              f0_threshold: float = 165.0,
                              max_samples_per_speaker: int = 20,
                              log_fn=print) -> Dict[str, Dict]:
    """
    Estimate speaker gender using fundamental frequency (F0).
    
    Typical ranges:
        Male:   85–180 Hz  (median ~120 Hz)
        Female: 165–255 Hz (median ~210 Hz)
    
    Returns:
        Dict of speaker_id → {'gender': str, 'confidence': str, 'median_f0': float}
    """
    try:
        import librosa
    except ImportError:
        log_fn("⚠️ librosa not installed — skipping gender estimation")
        log_fn("   Install: pip install librosa")
        return {}

    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
    except Exception as e:
        log_fn(f"⚠️ Could not load audio for gender estimation: {e}")
        return {}

    speaker_ranges: Dict[str, List[Tuple[float, float]]] = {}
    for seg in diarization:
        spk = seg['speaker']
        if spk not in speaker_ranges:
            speaker_ranges[spk] = []
        speaker_ranges[spk].append((seg['start'], seg['end']))

    gender_map = {}

    for speaker, time_ranges in speaker_ranges.items():
        f0_values = []

        sampled = time_ranges[:max_samples_per_speaker]
        for start, end in sampled:
            start_sample = int(start * sr)
            end_sample = min(int(end * sr), len(y))
            chunk = y[start_sample:end_sample]

            if len(chunk) < int(sr * 0.3):
                continue

            try:
                f0, voiced_flag, _ = librosa.pyin(
                    chunk, fmin=50, fmax=500, sr=sr
                )
                if voiced_flag is not None:
                    voiced_f0 = f0[voiced_flag]
                else:
                    voiced_f0 = f0[~np.isnan(f0)]

                if len(voiced_f0) > 0:
                    f0_values.extend(voiced_f0.tolist())
            except Exception:
                continue

        if f0_values:
            median_f0 = float(np.median(f0_values))
            distance = abs(median_f0 - f0_threshold)
            relative_conf = distance / f0_threshold

            if relative_conf > 0.25:
                confidence = "high"
            elif relative_conf > 0.1:
                confidence = "medium"
            else:
                confidence = "low"

            gender = "female" if median_f0 > f0_threshold else "male"

            gender_map[speaker] = {
                'gender': gender,
                'confidence': confidence,
                'median_f0': round(median_f0, 1)
            }
            log_fn(f"  🎤 {speaker}: median F0={median_f0:.1f}Hz → {gender} "
                   f"(confidence: {confidence})")
        else:
            gender_map[speaker] = {
                'gender': 'unknown',
                'confidence': 'none',
                'median_f0': 0.0
            }
            log_fn(f"  🎤 {speaker}: insufficient voiced audio → unknown")

    return gender_map


# ==================================================
# ALIGNMENT: Whisper segments ↔ diarization turns
# ==================================================

def align_segments_with_speakers(whisper_segments: List[Dict],
                                  diarization: List[Dict],
                                  gender_map: Dict[str, Dict]) -> List[Dict]:
    """
    Tag each Whisper segment with speaker ID and gender by finding
    the diarization turn with maximum time overlap.
    """
    for w_seg in whisper_segments:
        best_speaker = None
        best_overlap = 0.0

        w_start = w_seg['start']
        w_end = w_seg['end']

        for d_seg in diarization:
            overlap_start = max(w_start, d_seg['start'])
            overlap_end = min(w_end, d_seg['end'])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d_seg['speaker']

        if best_speaker and best_overlap > 0:
            info = gender_map.get(best_speaker, {})
            w_seg['speaker'] = best_speaker
            w_seg['gender'] = info.get('gender', 'unknown')
            w_seg['gender_confidence'] = info.get('confidence', 'none')
        else:
            w_seg['speaker'] = 'UNKNOWN'
            w_seg['gender'] = 'unknown'
            w_seg['gender_confidence'] = 'none'

    return whisper_segments


# ==================================================
# AUDIO EXTRACTION HELPER
# ==================================================

def extract_full_audio(video_path: str, output_dir: Optional[str] = None,
                       log_fn=print) -> Optional[str]:
    """
    Extract full audio track from video as 16kHz mono WAV.
    """
    import subprocess

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(video_path))

    base = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(output_dir, f"{base}_diarize_audio.wav")

    try:
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            "-y", audio_path
        ], check=True)
        log_fn(f"🎵 Extracted audio: {audio_path}")
        return audio_path
    except Exception as e:
        log_fn(f"⚠️ Audio extraction failed: {e}")
        return None


# ==================================================
# MAIN ENTRY POINT
# ==================================================

def enrich_segments_with_speakers(video_path: str,
                                   whisper_segments: List[Dict],
                                   num_speakers: Optional[int] = None,
                                   min_speakers: int = 1,
                                   max_speakers: int = 10,
                                   cleanup_audio: bool = True,
                                   log_fn=print) -> List[Dict]:
    """
    Full pipeline: extract audio → VAD → embeddings → cluster → 
                   gender estimation → align with Whisper segments.
    
    Gracefully returns unmodified segments if any step fails.
    No HuggingFace token required. No torchaudio dependency.
    
    Args:
        video_path:        Path to video file
        whisper_segments:  Transcribed segments from Whisper
        num_speakers:      Exact speaker count if known (None = auto-detect)
        min_speakers:      Minimum speakers hint for auto-detection
        max_speakers:      Maximum speakers hint for auto-detection
        cleanup_audio:     Remove extracted audio file after processing
        log_fn:            Logging function
    
    Returns:
        Enriched segments with 'speaker', 'speaker_label', 'gender',
        'gender_confidence' fields added.
    """
    min_speakers = min_speakers or 1
    max_speakers = max_speakers or 10

    # Step 1: Extract audio
    audio_path = extract_full_audio(video_path, log_fn=log_fn)
    if not audio_path:
        log_fn("ℹ️ Proceeding without speaker identification")
        return whisper_segments

    try:
        # Step 2: Voice Activity Detection
        log_fn("🎙️ Running speaker diarization (Resemblyzer)...")
        speech_segments = detect_speech_segments(audio_path, log_fn=log_fn)
        if not speech_segments:
            log_fn("⚠️ No speech segments detected")
            return whisper_segments

        # Step 3: Extract speaker embeddings
        result = extract_speaker_embeddings(audio_path, speech_segments, log_fn=log_fn)
        if result is None:
            log_fn("ℹ️ Proceeding without speaker identification")
            return whisper_segments

        embeddings, valid_segments = result

        # Step 4: Cluster speakers
        labels = cluster_speakers(
            embeddings,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            log_fn=log_fn
        )

        # Step 5: Build diarization timeline
        diarization = build_diarization_timeline(valid_segments, labels)
        unique_speakers = set(s['speaker'] for s in diarization)
        log_fn(f"  📊 Diarization timeline: {len(diarization)} turns, "
               f"{len(unique_speakers)} speakers")

        # Step 6: Estimate gender
        gender_map = estimate_gender_by_pitch(audio_path, diarization, log_fn=log_fn)

        # Step 7: Align with Whisper segments
        enriched = align_segments_with_speakers(whisper_segments, diarization, gender_map)

        # Step 8: Create friendly speaker labels
        speaker_ids = sorted(set(
            s.get('speaker', '')
            for s in enriched
            if s.get('speaker', '') != 'UNKNOWN'
        ))
        speaker_labels = {}
        for i, spk in enumerate(speaker_ids):
            speaker_labels[spk] = f"Person {i + 1}"

        for seg in enriched:
            raw = seg.get('speaker', 'UNKNOWN')
            seg['speaker_label'] = speaker_labels.get(raw, 'Unknown')

        # Summary
        log_fn(f"\n  {'─' * 45}")
        log_fn(f"  Speaker Summary:")
        log_fn(f"  {'─' * 45}")
        for spk, label in speaker_labels.items():
            info = gender_map.get(spk, {})
            gender = info.get('gender', 'unknown')
            conf = info.get('confidence', 'none')
            f0 = info.get('median_f0', 0)
            seg_count = sum(1 for s in enriched if s.get('speaker') == spk)
            log_fn(f"  📋 {label}: {gender} (F0: {f0}Hz, conf: {conf}, segments: {seg_count})")
        log_fn(f"  {'─' * 45}\n")

        log_fn(f"✅ Speaker enrichment complete: {len(speaker_labels)} speakers identified")
        return enriched

    except ImportError as e:
        log_fn(f"⚠️ Missing dependency: {e}")
        log_fn("   Install: pip install resemblyzer librosa scikit-learn")
        log_fn("   Proceeding without speaker identification")
        return whisper_segments

    except Exception as e:
        log_fn(f"⚠️ Speaker diarization failed: {e}")
        log_fn("   Proceeding without speaker identification")
        return whisper_segments

    finally:
        if cleanup_audio and audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                log_fn(f"🧹 Cleaned up: {audio_path}")
            except OSError:
                pass