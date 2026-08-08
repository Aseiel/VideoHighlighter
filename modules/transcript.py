import os
import whisper
import torch
from tqdm import tqdm
import re
import contextlib
import subprocess
import time


class TranscriptionCancelled(RuntimeError):
    """Raised when a run is cancelled part-way through transcription.

    A `RuntimeError` on purpose: `pipeline.check_cancellation` already signals
    cancellation that way and the pipeline's transcript step already treats a
    RuntimeError as "stop the run", so this arrives as a cancel rather than as a
    failure that leaves the run going with an empty transcript.
    """


class _ForwardingBar:
    """Stands in for the tqdm bar Whisper drives, reporting to a callback.

    Whisper's decode loop is the only thing that knows how far into a chunk it
    is, and it publishes that to a tqdm bar — which, in the packaged
    ``--windowed`` build, writes to a stderr that goes nowhere. Wearing tqdm's
    shape here turns those updates into progress the GUI can show.
    """

    def __init__(self, total=None, on_frac=None, **_kwargs):
        self.total = total or 0
        self.n = 0
        self._on_frac = on_frac

    def update(self, n=1):
        self.n += n
        if self._on_frac and self.total:
            self._on_frac(max(0.0, min(1.0, self.n / self.total)))

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


@contextlib.contextmanager
def _whisper_progress(on_frac):
    """Route Whisper's own per-chunk progress to ``on_frac`` for the duration.

    Swaps the ``tqdm`` name in ``whisper.transcribe``'s namespace — not the
    attribute on the real tqdm module, which every other library shares. If
    Whisper's internals ever stop looking like this, the bar simply stays as
    coarse as it was before; transcription itself is untouched either way.

    The module has to come from ``import_module``: ``whisper/__init__.py`` does
    ``from .transcribe import transcribe``, so ``whisper.transcribe`` the
    *attribute* is the function, and ``import whisper.transcribe as x`` binds
    that function rather than the module holding the name we need.
    """
    if on_frac is None:
        yield
        return
    try:
        import importlib
        _wt = importlib.import_module("whisper.transcribe")
    except Exception:
        yield
        return

    real = getattr(_wt, "tqdm", None)
    if real is None:
        yield
        return

    class _Shim:
        @staticmethod
        def tqdm(*args, **kwargs):
            kwargs.pop("disable", None)
            return _ForwardingBar(*args, on_frac=on_frac, **kwargs)

    _wt.tqdm = _Shim
    try:
        yield
    finally:
        _wt.tqdm = real

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

def _remove_chunks(video_dir, base):
    """Delete the `<base>_chunk_NNN.wav` files, if any are lying around."""
    try:
        names = os.listdir(video_dir)
    except OSError:
        return
    for f in names:
        if f.startswith(base + "_chunk_") and f.endswith(".wav"):
            try:
                os.remove(os.path.join(video_dir, f))
            except OSError:
                pass


def split_audio(video_file, chunk_length=600, should_cancel=None):
    """
    Split audio into chunks using ffmpeg (default: 600s = 10 min).
    Returns list of chunk file paths.

    `should_cancel` is polled while ffmpeg runs: on a feature-length video this
    step alone is a minute or two, and a Cancel that is only noticed afterwards
    is a Cancel the user does not believe in.
    """
    # Get the directory where the video is located
    video_dir = os.path.dirname(os.path.abspath(video_file))
    base, _ = os.path.splitext(os.path.basename(video_file))
    
    out_pattern = os.path.join(video_dir, f"{base}_chunk_%03d.wav")

    # Remove old chunks if they exist (safety cleanup)
    _remove_chunks(video_dir, base)

    # Use ffmpeg to split into fixed-length WAV files
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_file,
        "-f", "segment", "-segment_time", str(chunk_length),
        "-c:a", "pcm_s16le", "-ar", "16000",
        out_pattern
    ]
    if should_cancel is None:
        subprocess.run(cmd, check=True)
    else:
        proc = subprocess.Popen(cmd)
        while proc.poll() is None:
            if should_cancel():
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                # Whatever it had written is a half-split video, not a result.
                _remove_chunks(video_dir, base)
                raise TranscriptionCancelled("Cancelled while splitting audio")
            time.sleep(0.1)
        if proc.returncode != 0:
            raise subprocess.CalledProcessError(proc.returncode, cmd)

    # Return full paths to chunks in the video's directory
    chunks = [os.path.join(video_dir, f) for f in os.listdir(video_dir) 
              if f.startswith(base + "_chunk_") and f.endswith(".wav")]
    
    return sorted(chunks)

def get_transcript_segments(video_file, model_name="small", progress_fn=None, log_fn=print,
                           chunk_length=600, cleanup=True, language="en",
                           enable_diarization=False, num_speakers=None,
                           min_speakers=None, max_speakers=None,
                           should_cancel=None):
    """
    Transcribe video safely by splitting into chunks.
    - Uses Whisper for transcription
    - Filters hallucinations
    - Preserves timestamps with offsets
    - Shows progress via progress_fn
    - language: Source language code (e.g., "en", "pl", "fr", "auto" for auto-detect)
    - should_cancel: optional predicate, polled throughout. Raises
      TranscriptionCancelled promptly rather than at the end of the run.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log_fn(f"Using device for Whisper: {device}")

    log_fn(f"🔤 Language parameter received: {language}")

    def abort_if_cancelled():
        """Cancel means stop now.

        Whisper decodes a chunk of up to ten minutes in one call and knows
        nothing about cancelling, so a flag checked only between chunks leaves
        the user watching a button they already pressed. This is called from
        inside the decode loop as well — the same per-window hook the progress
        bar rides on.
        """
        if should_cancel and should_cancel():
            raise TranscriptionCancelled("Transcription cancelled")

    abort_if_cancelled()

    # The bar's own share of the run: loading and splitting are quick, the
    # chunk loop is everything else. Reported against 100 so a caller that
    # forwards these straight to a progress bar shows one steady climb.
    if progress_fn:
        progress_fn(1, 100, "Transcription", f"Loading Whisper '{model_name}'...")

    model = whisper.load_model(model_name, device=device)
    log_fn("Splitting video into chunks...")
    abort_if_cancelled()

    if progress_fn:
        progress_fn(3, 100, "Transcription", "Splitting audio...")

    chunks = split_audio(video_file, chunk_length=chunk_length,
                         should_cancel=should_cancel)
    log_fn(f"Created {len(chunks)} chunks")

    CHUNKS_FROM, CHUNKS_TO = 5, 90

    def report(idx, within, note=""):
        """Position inside the chunk loop, as a whole-run percentage."""
        if not progress_fn or not chunks:
            return
        done = (idx + within) / len(chunks)
        detail = f"Chunk {idx+1}/{len(chunks)}"
        if note:
            detail += f" — {note}"
        progress_fn(
            int(CHUNKS_FROM + done * (CHUNKS_TO - CHUNKS_FROM)),
            100,
            "Transcription",
            detail,
        )

    def tick(frac, idx):
        """Called once per decoded window: move the bar, honour a cancel."""
        report(idx, frac, f"{int(frac * 100)}%")
        abort_if_cancelled()

    all_segments = []
    try:
        for idx, chunk in enumerate(chunks):
            report(idx, 0.0)
            abort_if_cancelled()

            log_fn(f"➡️ Transcribing chunk {idx+1}/{len(chunks)}: {chunk}")

            # Prepare transcription parameters
            transcribe_params = {
                "task": "transcribe",
                "temperature": 0.0,  # No randomness
                "beam_size": 1,      # Greedy decoding
                "best_of": 1,        # Single attempt
                "patience": 1.0,
                "condition_on_previous_text": False,
                "compression_ratio_threshold": 2.4,  # Detect repetition
                "logprob_threshold": -1.0,           # Filter low confidence
                "no_speech_threshold": 0.6,          # Silence detection
                "verbose": False
            }

            # Add language if not auto-detect
            if language != "auto":
                transcribe_params["language"] = language

            # Whisper walks the chunk 30 seconds at a time; that inner walk is the
            # only signal there is while a chunk (up to 10 minutes of audio) decodes.
            # It skips its own progress update on windows it judges to be silence,
            # so a quiet stretch still shows as a pause — the closing report below
            # makes sure the chunk ends where it should regardless.
            with _whisper_progress(lambda f, _i=idx: tick(f, _i)):
                result = model.transcribe(chunk, **transcribe_params)

            report(idx, 1.0)

            detected_lang = result.get('language', 'unknown')

            # Warn if language mismatch
            if detected_lang != language and language != "auto":
                log_fn(f"⚠️ WARNING: Expected '{language}' but Whisper detected '{detected_lang}'")
                log_fn(f"   This may indicate unclear audio or incorrect language setting")

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
    finally:
        # Also on cancel: the chunks are a copy of the audio, and leaving
        # gigabytes of .wav next to the video is not what "cancel" should mean.
        if cleanup:
            log_fn("🧹 Cleaning up chunk files...")
            for f in chunks:
                try:
                    os.remove(f)
                except OSError:
                    pass

    if progress_fn:
        progress_fn(95, 100, "Transcription", "Complete")

    log_fn(f"✅ Transcript ready: {len(all_segments)} segments (from {len(chunks)} chunks)")

    # Before diarization, which is a second pass over the audio and would
    # otherwise run in full after the user asked for none of it. (Checked here
    # rather than inside: the block below turns every exception into a warning
    # and carries on, which is right for a failed diarization and wrong for a
    # cancel.)
    abort_if_cancelled()

    # --- Optional: Speaker diarization & gender estimation ---
    if enable_diarization:
        try:
            from modules.speaker_utils import enrich_segments_with_speakers

            if progress_fn:
                progress_fn(96, 100, "Diarization", "Identifying speakers...")

            all_segments = enrich_segments_with_speakers(
                video_path=video_file,
                whisper_segments=all_segments,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                cleanup_audio=True,
                log_fn=log_fn
            )

            if progress_fn:
                progress_fn(98, 100, "Diarization", "Speaker identification complete")

        except ImportError as e:
            log_fn(f"⚠️ Diarization module not available: {e}")
            log_fn("   Install: pip install speechbrain torchaudio librosa scikit-learn")
            log_fn("   Proceeding without speaker identification")
        except Exception as e:
            log_fn(f"⚠️ Diarization failed: {e}")
            log_fn("   Proceeding without speaker identification")

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