"""
llm_module.py — Local LLM interface for VideoHighlighter.

Supports two backends:
  1. ollama   — requires `ollama` running locally (easiest setup)
  2. llama-cpp — requires `llama-cpp-python` + a GGUF model file

The module builds rich context from your video analysis cache so the LLM
can reason about detected objects, actions, transcript, scores, etc.

Usage:
    from llm_module import LLMModule, VideoSeekAnalyzer

    llm = LLMModule(backend="ollama", model="llama3.2")
    llm.load()
    
    # Analyze video every 1 second
    analyzer = VideoSeekAnalyzer("video.mp4", llm)
    results = analyzer.analyze_every_1_second()
"""

from __future__ import annotations

import base64
import math
import json
import os
import re
import time
import threading
from typing import Optional, Callable

# Try to import OpenCV for video analysis (optional)
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("⚠️ OpenCV not installed. VideoSeekAnalyzer will not work. Install with: pip install opencv-python")


# ---------------------------------------------------------------------------
# Cancellation token — allows external code to abort generation mid-stream
# ---------------------------------------------------------------------------
class CancellationToken:
    """Thread-safe cancellation flag that backends check during generation."""
    def __init__(self):
        self._cancelled = threading.Event()

    def cancel(self):
        self._cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def reset(self):
        self._cancelled.clear()


class GenerationCancelled(Exception):
    """Raised when generation is cancelled via CancellationToken."""
    pass


# ---------------------------------------------------------------------------
# Response sanitizer — strips leaked role tokens and self-conversation
# ---------------------------------------------------------------------------
# Patterns that indicate the model started role-playing a conversation
_SELF_TALK_PATTERNS = re.compile(
    r'(?:\n|^)\s*(?:'
    r'(?:USER|User|user|HUMAN|Human|human)\s*:\s*'   # "USER:" turn markers
    r'|(?:ASSISTANT|Assistant|assistant)\s*:\s*'       # "ASSISTANT:" markers
    r'|⏹\s*Stopping generation'                       # leaked stop tokens
    r'|\[SYSTEM INSTRUCTIONS\]'                        # leaked system wrapping
    r'|\[END INSTRUCTIONS\]'
    r'|=== VIDEO ANALYSIS DATA ==='                    # leaked context markers
    r'|=== END DATA ==='
    r'|=== TIMELINE'
    r')',
    re.IGNORECASE
)

# Stop sequences to prevent generation past the assistant's turn
STOP_SEQUENCES = [
    "\nUSER:", "\nUser:", "\nuser:",
    "\nHUMAN:", "\nHuman:", "\nhuman:",
    "\nASSISTANT:", "\nAssistant:",
    "\n## ", "\n===",  # Don't regenerate context markers
    "⏹",
]

def sanitize_response(text: str) -> str:
    """
    Clean up LLM output: strip self-conversation, leaked markers, and 
    truncate at the first sign of role-playing.
    """
    if not text:
        return text
    
    # Find the first occurrence of a self-talk pattern
    match = _SELF_TALK_PATTERNS.search(text)
    if match:
        # Truncate everything from the first leaked marker onward
        text = text[:match.start()].rstrip()
    
    # Also strip any trailing partial role markers that didn't fully match
    # e.g. the model outputting "USER" right at the end
    for marker in ["USER", "User", "ASSISTANT", "Assistant", "HUMAN", "Human"]:
        if text.rstrip().endswith(marker):
            text = text[:text.rfind(marker)].rstrip()
    
    # Strip trailing whitespace and dangling punctuation
    text = text.rstrip()
    
    return text


# ---------------------------------------------------------------------------
# Backend base
# ---------------------------------------------------------------------------
class _LLMBackend:
    """Abstract backend."""

    def load(self, **kwargs):
        raise NotImplementedError

    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024,
                 temperature: float = 0.7, stream_callback: Optional[Callable] = None,
                 images: list[str] | None = None,
                 cancellation_token: Optional[CancellationToken] = None) -> str:
        raise NotImplementedError

    def is_loaded(self) -> bool:
        return False

    def unload(self):
        pass

    @staticmethod
    def available() -> bool:
        return False


# ---------------------------------------------------------------------------
# Helper: check cancellation inside any streaming loop
# ---------------------------------------------------------------------------
def _check_cancel(cancel_token: Optional[CancellationToken]):
    """Raise GenerationCancelled if the token is set."""
    if cancel_token and cancel_token.is_cancelled:
        raise GenerationCancelled("Generation cancelled by user")


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------
class _OllamaBackend(_LLMBackend):
    """Talks to a local Ollama server (http://localhost:11434)."""

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._loaded = False

    @staticmethod
    def available() -> bool:
        try:
            import requests  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self, **kwargs):
        """Verify the Ollama server is reachable and the model exists."""
        import requests
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            matched = any(self.model in m for m in models)
            if not matched:
                raise RuntimeError(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Available: {models}\n"
                    f"Run: ollama pull {self.model}"
                )
            self._loaded = True
        except requests.ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama at "
                f"{self.base_url}. Is it running?\n"
                "Start with: ollama serve"
            )

    def is_loaded(self) -> bool:
        return self._loaded

    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024,
                temperature: float = 0.7, stream_callback: Optional[Callable] = None,
                images: list[str] | None = None,
                cancellation_token: Optional[CancellationToken] = None) -> str:
        import requests

        formatted_prompt = prompt

        payload = {
            "model": self.model,
            "prompt": formatted_prompt,
            "system": system,
            "stream": stream_callback is not None,
            "options": {
                "num_predict": max_tokens,
                "num_ctx": 2048,
                "temperature": temperature,
                "repeat_penalty": 1.3,
                "repeat_last_n": 128,
                "stop": STOP_SEQUENCES,
            },
        }

        if images:
            payload["images"] = images

        if stream_callback:
            full_text = []
            with requests.post(f"{self.base_url}/api/generate", json=payload,
                            stream=True, timeout=120) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    # Check cancellation on every token
                    _check_cancel(cancellation_token)
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        full_text.append(token)
                        current_text = "".join(full_text)
                        if _SELF_TALK_PATTERNS.search(current_text):
                            break
                        stream_callback(token)
                    if chunk.get("done", False):
                        break
            raw = "".join(full_text)
            return sanitize_response(raw)
        else:
            # Non-streaming: use streaming internally so we can cancel
            full_text = []
            payload["stream"] = True
            with requests.post(f"{self.base_url}/api/generate", json=payload,
                            stream=True, timeout=120) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    _check_cancel(cancellation_token)
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        full_text.append(token)
                        current_text = "".join(full_text)
                        if _SELF_TALK_PATTERNS.search(current_text):
                            break
                    if chunk.get("done", False):
                        break
            raw = "".join(full_text)
            return sanitize_response(raw)

    def unload(self):
        self._loaded = False


# ---------------------------------------------------------------------------
# llama-cpp-python backend with vision support
# ---------------------------------------------------------------------------
class _LlamaCppBackend(_LLMBackend):
    """Uses llama-cpp-python to run a GGUF model directly (no server)."""

    def __init__(self, model_path: str, mmproj_path: str = None, n_ctx: int = 4096, n_gpu_layers: int = -1):
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._model = None
        self._chat_handler = None

    @staticmethod
    def available() -> bool:
        try:
            from llama_cpp import Llama  # noqa: F401
            return True
        except ImportError:
            return False

    def load(self, **kwargs):
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"GGUF model not found: {self.model_path}")
        
        from llama_cpp import Llama
        
        if self.mmproj_path and os.path.exists(self.mmproj_path):
            print(f"📷 Loading vision model with mmproj: {self.mmproj_path}")
            
            try:
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                
                self._chat_handler = Llava15ChatHandler(
                    clip_model_path=self.mmproj_path,
                    verbose=False
                )
                
                print(f"✅ Created Llava15ChatHandler for vision support")
                
                self._model = Llama(
                    model_path=self.model_path,
                    chat_handler=self._chat_handler,
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=False,
                    n_threads=None,
                )
            except ImportError:
                print("⚠️ Llava15ChatHandler not available, falling back to basic vision")
                self._model = Llama(
                    model_path=self.model_path,
                    clip_model_path=self.mmproj_path,
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=False,
                    n_threads=None,
                )
        else:
            self._model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
                n_threads=None,
            )

    def is_loaded(self) -> bool:
        return self._model is not None

    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024,
                temperature: float = 0.7, stream_callback: Optional[Callable] = None,
                images: list[str] | None = None,
                cancellation_token: Optional[CancellationToken] = None) -> str:
        if not self._model:
            raise RuntimeError("Model not loaded. Call load() first.")

        if images and self.mmproj_path:
            return self._generate_vision(prompt, system, images, max_tokens, 
                                        temperature, stream_callback,
                                        cancellation_token)
        else:
            return self._generate_text(prompt, system, max_tokens, 
                                      temperature, stream_callback,
                                      cancellation_token)

    def _generate_text(self, prompt: str, system: str = "", max_tokens: int = 1024,
                      temperature: float = 0.7, stream_callback: Optional[Callable] = None,
                      cancellation_token: Optional[CancellationToken] = None) -> str:
        """Handle text-only generation with anti-hallucination measures."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        gen_kwargs = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "repeat_penalty": 1.3,
            "stop": STOP_SEQUENCES,
        }

        # ALWAYS use streaming for llama-cpp so we can check cancellation
        full_text = []
        for chunk in self._model.create_chat_completion(**gen_kwargs, stream=True):
            _check_cancel(cancellation_token)
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content", "")
            if token:
                full_text.append(token)
                joined = "".join(full_text)
                if _SELF_TALK_PATTERNS.search(joined):
                    break
                if stream_callback:
                    stream_callback(token)
        raw = "".join(full_text)
        return sanitize_response(raw)

    def _generate_vision(self, prompt: str, system: str = "", images: list[str] = None,
                        max_tokens: int = 1024, temperature: float = 0.7,
                        stream_callback: Optional[Callable] = None,
                        cancellation_token: Optional[CancellationToken] = None) -> str:
        """Handle vision generation with proper chat handler.
        
        Always streams internally so cancellation_token can interrupt
        even when no stream_callback is provided.
        """
        try:
            image_urls = []
            for img_b64 in images:
                if ',' in img_b64:
                    img_b64 = img_b64.split(',', 1)[1]
                image_urls.append(f"data:image/jpeg;base64,{img_b64}")
            
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            
            user_content = []
            user_content.append({"type": "text", "text": prompt})
            
            for img_url in image_urls:
                user_content.append({
                    "type": "image_url", 
                    "image_url": {"url": img_url}
                })
            
            messages.append({"role": "user", "content": user_content})
            
            print(f"📤 Sending vision request with {len(images)} images")

            gen_kwargs = {
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "repeat_penalty": 1.3,
                "stop": STOP_SEQUENCES,
            }

            # ALWAYS stream internally so we can check cancellation_token
            # between tokens. Previously the non-streaming path was a single
            # blocking C call with no way to interrupt.
            full_text = []
            for chunk in self._model.create_chat_completion(**gen_kwargs, stream=True):
                _check_cancel(cancellation_token)
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    token = delta.get("content", "")
                    if token:
                        full_text.append(token)
                        joined = "".join(full_text)
                        if _SELF_TALK_PATTERNS.search(joined):
                            break
                        if stream_callback:
                            stream_callback(token)
            raw = "".join(full_text)
            result = sanitize_response(raw)
            print(f"✅ Vision generation complete: {len(result)} chars")
            return result
                    
        except GenerationCancelled:
            # Re-raise so callers see the cancellation
            raw = "".join(full_text) if 'full_text' in dir() else ""
            return sanitize_response(raw)
        except Exception as e:
            print(f"❌ Vision generation error: {e}")
            import traceback
            traceback.print_exc()
            
            print("⚠️ Falling back to raw prompt format...")
            return self._generate_vision_fallback(prompt, system, images, max_tokens, 
                                                  temperature, stream_callback,
                                                  cancellation_token)

    def _generate_vision_fallback(self, prompt: str, system: str = "", images: list[str] = None,
                                 max_tokens: int = 1024, temperature: float = 0.7,
                                 stream_callback: Optional[Callable] = None,
                                 cancellation_token: Optional[CancellationToken] = None) -> str:
        """Fallback method using raw prompt formatting (LLaVA style)."""
        try:
            prompt_parts = []
            
            if system:
                prompt_parts.append(system)
                prompt_parts.append("")
            
            for _ in images:
                prompt_parts.append("<image>")
            
            prompt_parts.append("")
            prompt_parts.append(f"Question: {prompt}")
            prompt_parts.append("Answer:")
            
            full_prompt = "\n".join(prompt_parts)
            
            print(f"📤 Using fallback prompt format with {len(images)} images")
            
            image_bytes = []
            for img_b64 in images:
                if ',' in img_b64:
                    img_b64 = img_b64.split(',', 1)[1]
                image_bytes.append(base64.b64decode(img_b64))
            
            gen_kwargs = {
                "prompt": full_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "image_data": image_bytes,
                "repeat_penalty": 1.3,
                "stop": STOP_SEQUENCES + ["\nQuestion:", "\nQ:"],
            }

            # Always stream for cancellation support
            full_text = []
            for chunk in self._model.create_completion(**gen_kwargs, stream=True):
                _check_cancel(cancellation_token)
                token = chunk["choices"][0].get("text", "")
                if token:
                    full_text.append(token)
                    joined = "".join(full_text)
                    if _SELF_TALK_PATTERNS.search(joined):
                        break
                    if stream_callback:
                        stream_callback(token)
            raw = "".join(full_text)
            return sanitize_response(raw)
                
        except GenerationCancelled:
            raw = "".join(full_text) if 'full_text' in dir() else ""
            return sanitize_response(raw)
        except Exception as e:
            print(f"❌ Fallback also failed: {e}")
            return f"Error processing image: {e}"

    def unload(self):
        self._model = None
        self._chat_handler = None

# ---------------------------------------------------------------------------
# Context builder — turns analysis cache into LLM-readable text
# ---------------------------------------------------------------------------
class VideoContextBuilder:
    """
    Converts a VideoHighlighter analysis cache dict into a concise text
    summary that fits in the LLM context window.
    """

    @staticmethod
    def build(analysis_data: dict, video_path: str = "", max_items: int = 30) -> str:
        """
        Build a context string from analysis data.

        Args:
            analysis_data: The cache dict (same format as collect_analysis_data output)
            video_path: Optional video filename for reference
            max_items: Cap per section to avoid blowing up context

        Returns:
            A formatted context string
        """
        parts = []

        # --- Video metadata ---
        meta = analysis_data.get("video_metadata", {})
        duration = meta.get("duration", 0)
        fps = meta.get("fps", 0)
        if video_path:
            parts.append(f"## Video: {os.path.basename(video_path)}")
        parts.append(f"Duration: {int(duration)}s ({int(duration)//60}m{int(duration)%60:02d}s), FPS: {fps:.1f}")
        parts.append("")

        # --- Detected objects ---
        objects_raw = analysis_data.get("objects", [])
        if objects_raw:
            obj_counts: dict[str, int] = {}
            obj_timestamps: dict[str, list] = {}
            for entry in objects_raw:
                ts = entry.get("timestamp", 0)
                for obj_name in entry.get("objects", []):
                    obj_counts[obj_name] = obj_counts.get(obj_name, 0) + 1
                    obj_timestamps.setdefault(obj_name, []).append(ts)

            parts.append(f"## Detected Objects ({len(objects_raw)} seconds with detections)")
            for obj, count in sorted(obj_counts.items(), key=lambda x: -x[1]):
                timestamps = obj_timestamps[obj]
                sample_ts = timestamps[:5]
                ts_str = ", ".join(f"{t//60}:{t%60:02d}" for t in sample_ts)
                if len(timestamps) > 5:
                    ts_str += f" ... (+{len(timestamps)-5} more)"
                parts.append(f"  - {obj}: {count} detections (at {ts_str})")
            parts.append("")

        # --- Detected actions ---
        actions_raw = analysis_data.get("actions", [])
        if actions_raw:
            action_groups: dict[str, list] = {}
            for act in actions_raw:
                name = act.get("action_name", "unknown")
                action_groups.setdefault(name, []).append(act)

            parts.append(f"## Detected Actions ({len(actions_raw)} total detections)")
            for name, detections in sorted(action_groups.items(), key=lambda x: -len(x[1])):
                confidences = [d.get("confidence", 0) for d in detections]
                timestamps = [d.get("timestamp", 0) for d in detections]
                avg_conf = sum(confidences) / len(confidences) if confidences else 0
                max_conf = max(confidences) if confidences else 0
                sample_ts = sorted(timestamps)[:5]
                ts_str = ", ".join(f"{int(t)//60}:{int(t)%60:02d}" for t in sample_ts)

                parts.append(
                    f"  - {name}: {len(detections)} detections, "
                    f"avg conf={avg_conf:.2f}, max conf={max_conf:.2f} "
                    f"(at {ts_str}{'...' if len(timestamps) > 5 else ''})"
                )
            parts.append("")

        # --- Scenes ---
        scenes = analysis_data.get("scenes", [])
        if scenes:
            parts.append(f"## Scene Changes ({len(scenes)})")
            for sc in scenes[:max_items]:
                s, e = sc.get("start", 0), sc.get("end", 0)
                parts.append(f"  - {int(s)//60}:{int(s)%60:02d} → {int(e)//60}:{int(e)%60:02d}")
            if len(scenes) > max_items:
                parts.append(f"  ... and {len(scenes) - max_items} more")
            parts.append("")

        # --- Motion events/peaks ---
        motion_events = analysis_data.get("motion_events", [])
        motion_peaks = analysis_data.get("motion_peaks", [])
        if motion_events or motion_peaks:
            parts.append(f"## Motion: {len(motion_events)} events, {len(motion_peaks)} peaks")
            if motion_peaks:
                sample = sorted(motion_peaks)[:10]
                parts.append(f"  Peak timestamps: {', '.join(f'{int(t)//60}:{int(t)%60:02d}' for t in sample)}")
            parts.append("")

        # --- Audio peaks ---
        audio_data = analysis_data.get("audio", {})
        audio_peaks = audio_data.get("peaks", []) if isinstance(audio_data, dict) else analysis_data.get("audio_peaks", [])
        if audio_peaks:
            parts.append(f"## Audio Peaks ({len(audio_peaks)})")
            sample = sorted(audio_peaks)[:10]
            parts.append(f"  Timestamps: {', '.join(f'{int(t)//60}:{int(t)%60:02d}' for t in sample)}")
            parts.append("")

        # --- Transcript snippets ---
        transcript = analysis_data.get("transcript", {})
        segments = transcript.get("segments", [])
        lang = transcript.get("language", "unknown")
        if segments:
            parts.append(f"## Transcript ({len(segments)} segments, language: {lang})")
            show_count = min(max_items, len(segments))
            for seg in segments[:show_count]:
                start = seg.get("start", 0)
                text = seg.get("text", "").strip()
                if text:
                    parts.append(f"  [{int(start)//60}:{int(start)%60:02d}] {text[:120]}")
            if len(segments) > show_count:
                parts.append(f"  ... ({len(segments) - show_count} more segments)")
            parts.append("")

        # --- Keyword matches ---
        keyword_matches = analysis_data.get("keyword_matches", [])
        if keyword_matches:
            parts.append(f"## Keyword Matches ({len(keyword_matches)})")
            for km in keyword_matches[:max_items]:
                kw = km.get("keyword", "?")
                seg = km.get("main_segment", {})
                start = seg.get("start", 0)
                text = seg.get("text", "")[:80]
                parts.append(f"  - '{kw}' at {int(start)//60}:{int(start)%60:02d}: \"{text}\"")
            parts.append("")

        return "\n".join(parts)

    @staticmethod
    def build_action_learning_context(
        action_name: str,
        object_name: str,
        existing_labels: list[str],
        clip_analysis: list[dict] | None = None,
    ) -> str:
        """
        Build context specifically for the auto-learn pipeline (Step 5).
        """
        parts = [
            f"## Action Learning Task",
            f"Target action: '{action_name}'",
            f"Expected object: '{object_name}'",
            f"",
            f"The system is trying to learn to recognize '{action_name}' from video clips.",
            f"Currently known actions ({len(existing_labels)}): "
            + ", ".join(existing_labels[:20])
            + ("..." if len(existing_labels) > 20 else ""),
            "",
        ]

        if clip_analysis:
            parts.append("## Clip Analysis Results")
            for i, clip in enumerate(clip_analysis):
                parts.append(f"### Clip {i+1}")
                if "objects" in clip:
                    parts.append(f"  Objects detected: {', '.join(clip['objects'])}")
                if "actions" in clip:
                    for act in clip["actions"]:
                        parts.append(
                            f"  Action: {act.get('name', '?')} "
                            f"(confidence: {act.get('confidence', 0):.2f})"
                        )
                if "motion_level" in clip:
                    parts.append(f"  Motion level: {clip['motion_level']}")
                parts.append("")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main LLM Module
# ---------------------------------------------------------------------------
class LLMModule:
    """
    High-level interface to a local LLM for VideoHighlighter.

    Examples:
        # Ollama (easiest)
        llm = LLMModule(backend="ollama", model="llama3.2")
        llm.load()
        answer = llm.query("Summarize the video", analysis_data=cache)

        # llama-cpp-python (no server)
        llm = LLMModule(backend="llama-cpp",
                         model_path="/models/llama-3.2-3b.Q4_K_M.gguf")
        llm.load()
    """

    SYSTEM_PROMPT = (
        "You are a video analysis assistant. You have access to structured analysis data below.\n"
        "RULES:\n"
        "1. Answer the user's question using ONLY the provided data.\n"
        "2. Quote specific findings with MM:SS timestamps.\n"
        "3. If something is NOT in the data, say so and list what IS detected.\n"
        "4. Be concise — 2-4 sentences unless more detail is needed.\n"
        "5. NEVER generate fake conversation turns. NEVER write 'USER:' or 'ASSISTANT:'.\n"
        "6. NEVER repeat or echo the analysis data back. Just answer the question.\n"
        "7. Give ONE response and STOP."
    )

    SYSTEM_PROMPT_ACTION_LEARNING = (
        "You are an AI assistant helping to learn new action categories from video clips. "
        "Given detected objects, motion features, and existing action classifications, "
        "determine whether a video clip likely contains the target action. "
        "Respond with a confidence score (0.0 to 1.0) and brief reasoning. "
        "Format: SCORE: 0.XX\nREASON: ...\n"
        "Give ONE response and STOP. Do not generate any follow-up."
    )

    SYSTEM_PROMPT_TIMELINE = (
        "You are a video analysis assistant with timeline control.\n"
        "RULES:\n"
        "1. Use the analysis data to answer questions with MM:SS timestamps.\n"
        "2. You can control the timeline with [CMD:...] commands.\n"
        "   - seek time MUST be a number of seconds (float or int), NOT mm:ss.\n"
        "3. Be concise — 1-2 sentences, then the command.\n"
        "4. NEVER list all clips or echo the timeline state.\n"
        "5. NEVER generate fake conversation turns.\n"
        "6. Give ONE response and STOP."
    )

    SYSTEM_PROMPT_VISION = (
        "Describe what you see in the attached image.\n"
        "Focus on: people (count, poses, clothing, actions), objects, scene, colors, lighting.\n"
        "If analysis data is provided, use it as additional context.\n"
        "Be specific and detailed. Give ONE description and STOP.\n"
        "Do NOT generate follow-up questions or fake conversation."
    )

    SYSTEM_PROMPT_VISUAL_SEARCH = (
        "You are a visual search assistant. The user will ask if a specific thing "
        "is present in the image.\n"
        "RULES:\n"
        "1. Start your answer with YES or NO.\n"
        "2. Then give a ONE sentence explanation of what you see.\n"
        "3. STOP after that. Do NOT describe the full scene.\n"
        "4. Do NOT generate follow-up questions or fake conversation."
    )

    def __init__(
        self,
        backend: str = "ollama",
        model: str = "llama3.2",
        model_path: str = "",
        mmproj_path: str = None,
        base_url: str = "http://localhost:11434",
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        log_fn: Callable = print,
    ):
        self.backend_name = backend
        self.log_fn = log_fn
        self._backend: _LLMBackend

        if backend == "ollama":
            self._backend = _OllamaBackend(model=model, base_url=base_url)
        elif backend == "llama-cpp":
            self._backend = _LlamaCppBackend(
                model_path=model_path,
                mmproj_path=mmproj_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'ollama' or 'llama-cpp'.")

    def load(self):
        """Load/verify the model. Raises RuntimeError on failure."""
        self.log_fn(f"🤖 Loading LLM backend: {self.backend_name}")
        start = time.time()
        self._backend.load()
        elapsed = time.time() - start
        self.log_fn(f"✅ LLM ready ({elapsed:.1f}s)")

    def is_loaded(self) -> bool:
        return self._backend.is_loaded()

    def unload(self):
        self._backend.unload()
        self.log_fn("🤖 LLM unloaded")

    def query(
        self,
        user_message: str,
        analysis_data: dict | None = None,
        video_path: str = "",
        system_prompt: str | None = None,
        timeline_context: str = "",
        frame_base64: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        stream_callback: Optional[Callable[[str], None]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> str:
        """
        Send a query to the LLM with optional video analysis context.
        
        Args:
            cancellation_token: Optional CancellationToken that, when cancelled,
                will interrupt generation even mid-token for GGUF vision models.
        """
        if not self._backend.is_loaded():
            raise RuntimeError("LLM not loaded. Call load() first.")
        
        print(f"🔍 query() called:")
        print(f"   frame_base64: {'YES (' + str(len(frame_base64)) + ' chars)' if frame_base64 else 'NONE'}")
        print(f"   timeline_context: {'YES' if timeline_context else 'NONE'}")
        print(f"   analysis_data: {'YES' if analysis_data else 'NONE'}")

        # ===== VISION MODE =====
        if frame_base64:
            prompt_parts = []
            
            if timeline_context:
                prompt_parts.append(
                    "--- TIMELINE CONTEXT ---\n"
                    f"{timeline_context}\n"
                    "--- END ---\n"
                )
            
            if analysis_data:
                context = VideoContextBuilder.build(analysis_data, video_path)
                prompt_parts.append(
                    "--- VIDEO ANALYSIS DATA ---\n"
                    f"{context}\n"
                    "--- END ---\n"
                )
            
            prompt_parts.append(user_message)
            full_prompt = "\n".join(prompt_parts)
            
            system = system_prompt or self.SYSTEM_PROMPT_VISION
            
            return self._backend.generate(
                prompt=full_prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                stream_callback=stream_callback,
                images=[frame_base64],
                cancellation_token=cancellation_token,
            )
        
        # ===== TEXT MODE =====
        prompt_parts = []
        
        if analysis_data:
            context = VideoContextBuilder.build(analysis_data, video_path)
            prompt_parts.append(
                "--- VIDEO ANALYSIS DATA ---\n"
                f"{context}\n"
                "--- END ---\n"
            )
        
        if timeline_context:
            prompt_parts.append(
                "--- TIMELINE CONTROL ---\n"
                f"{timeline_context}\n"
                "--- END ---\n"
            )
        
        prompt_parts.append(user_message)
        full_prompt = "\n".join(prompt_parts)
        
        if system_prompt:
            system = system_prompt
        elif timeline_context:
            system = self.SYSTEM_PROMPT_TIMELINE
        else:
            system = self.SYSTEM_PROMPT
        
        return self._backend.generate(
            prompt=full_prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            stream_callback=stream_callback,
            cancellation_token=cancellation_token,
        )

    def verify_action_clip(
        self,
        action_name: str,
        object_name: str,
        clip_analysis: list[dict],
        existing_labels: list[str],
    ) -> tuple[float, str]:
        """
        Step 5 of auto-learn pipeline: ask LLM whether a clip contains the target action.
        """
        context = VideoContextBuilder.build_action_learning_context(
            action_name=action_name,
            object_name=object_name,
            existing_labels=existing_labels,
            clip_analysis=clip_analysis,
        )

        prompt = (
            f"{context}\n\n"
            f"Question: Based on the detected objects and action features above, "
            f"does this clip likely show the action '{action_name}'?\n"
            f"Respond with:\n"
            f"SCORE: <0.0 to 1.0>\n"
            f"REASON: <brief explanation>"
        )

        response = self._backend.generate(
            prompt=prompt,
            system=self.SYSTEM_PROMPT_ACTION_LEARNING,
            max_tokens=256,
            temperature=0.3,
        )

        score = 0.5
        reason = response.strip()
        for line in response.strip().split("\n"):
            line_clean = line.strip().upper()
            if line_clean.startswith("SCORE:"):
                try:
                    score = float(line_clean.split(":", 1)[1].strip())
                    score = max(0.0, min(1.0, score))
                except ValueError:
                    pass
            elif line_clean.startswith("REASON:"):
                reason = line.strip().split(":", 1)[1].strip()

        return score, reason

    def query_async(
        self,
        user_message: str,
        callback: Callable[[str], None],
        error_callback: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> threading.Thread:
        """Non-blocking query — runs in a background thread."""
        def _run():
            try:
                result = self.query(user_message, **kwargs)
                callback(result)
            except Exception as e:
                if error_callback:
                    error_callback(str(e))
                else:
                    self.log_fn(f"❌ LLM query error: {e}")

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread


# ---------------------------------------------------------------------------
# Convenience: check what's available
# ---------------------------------------------------------------------------
def get_available_backends() -> list[str]:
    """Return list of available backend names."""
    backends = []
    if _OllamaBackend.available():
        backends.append("ollama")
    if _LlamaCppBackend.available():
        backends.append("llama-cpp")
    return backends


def get_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Query Ollama for available models. Returns empty list on failure."""
    try:
        import requests
        resp = requests.get(f"{base_url}/api/tags", timeout=3)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Video Seek Analyzer
# ---------------------------------------------------------------------------
class VideoSeekAnalyzer:
    """
    Analyzes video frames at regular intervals using LLM vision.
    
    Requires OpenCV: pip install opencv-python
    
    Example:
        llm = LLMModule(backend="ollama", model="llava")
        llm.load()
        
        analyzer = VideoSeekAnalyzer("video.mp4", llm)
        results = analyzer.analyze_every_1_second()
        
        for r in results:
            print(f"[{r['timestamp_str']}] {r['analysis'][:100]}...")
        
        analyzer.close()
    """
    
    def __init__(self, video_path: str, llm: LLMModule, verbose: bool = False):
        self.verbose = verbose

        if not HAS_CV2:
            raise ImportError(
                "OpenCV (cv2) is required for VideoSeekAnalyzer. "
                "Install with: pip install opencv-python"
            )
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        self.video_path = video_path
        self.llm = llm
        self.cap = cv2.VideoCapture(video_path)
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.current_time = 0
        self.lock = threading.Lock()
        self.running = False
        self.analysis_cache = []
        
        print(f"📹 Video loaded: {os.path.basename(video_path)}")
        print(f"   Duration: {int(self.duration)//60}m{int(self.duration)%60:02d}s")
        print(f"   Resolution: {self.width}x{self.height}")
        print(f"   FPS: {self.fps:.2f}")
    
    def seek_to_time(self, timestamp_seconds: float):
        timestamp_seconds = max(0, min(timestamp_seconds, self.duration))
        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
        self.cap.read()  # flush
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def frame_to_base64(self, frame, quality: int = 100) -> str:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        return base64.b64encode(buffer).decode('utf-8')
    
    def analyze_current_frame(
        self, 
        user_query: str = "What do you see in this frame? Describe the scene, people, objects, and actions.",
        custom_prompt: str = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> dict:
        frame = self.seek_to_time(self.current_time)
        if frame is None:
            return {
                "error": "Could not read frame",
                "timestamp": self.current_time,
                "timestamp_str": f"{int(self.current_time)//60}:{int(self.current_time)%60:02d}"
            }
        
        frame_b64 = self.frame_to_base64(frame)
        system = custom_prompt if custom_prompt else None
        
        try:
            response = self.llm.query(
                user_message=user_query,
                frame_base64=frame_b64,
                system_prompt=system,
                temperature=0.3,
                max_tokens=500,
                cancellation_token=cancellation_token,
            )
            
            return {
                "timestamp": self.current_time,
                "timestamp_str": f"{int(self.current_time)//60}:{int(self.current_time)%60:02d}",
                "analysis": response,
                "frame_info": {
                    "width": self.width,
                    "height": self.height
                }
            }
        except GenerationCancelled:
            return {
                "cancelled": True,
                "timestamp": self.current_time,
                "timestamp_str": f"{int(self.current_time)//60}:{int(self.current_time)%60:02d}"
            }
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": self.current_time,
                "timestamp_str": f"{int(self.current_time)//60}:{int(self.current_time)%60:02d}"
            }
    
    def analyze_every_n_seconds(self, interval: float = 1.0, callback=None, save_to_file=None,
                                cancellation_token: Optional[CancellationToken] = None):
        results = []
        num_analyses = int(self.duration / interval) + 1
        timestamps = [i * interval for i in range(num_analyses)]
        
        print(f"\n📊 Seeking analysis every {interval}s ({len(timestamps)} frames)")
        print(f"   Video duration: {int(self.duration)//60}m{int(self.duration)%60:02d}s")
        
        progress_interval = max(1, len(timestamps) // 10)
        
        for i, timestamp in enumerate(timestamps):
            # Check cancellation between frames
            if cancellation_token and cancellation_token.is_cancelled:
                print(f"\n⏹ Analysis cancelled at {timestamp:.1f}s")
                break

            if timestamp > self.duration + 0.1:
                break
                
            self.current_time = timestamp
            frame = self.seek_to_time(timestamp)
            
            if frame is None:
                if i % progress_interval == 0:
                    print(f"⚠️ Could not read frame at {timestamp:.1f}s")
                continue
            
            frame_b64 = self.frame_to_base64(frame)
            
            try:
                response = self.llm.query(
                    user_message="What do you see in this frame? Describe the scene, people, objects, and actions.",
                    frame_base64=frame_b64,
                    temperature=0.3,
                    max_tokens=500,
                    cancellation_token=cancellation_token,
                )
                
                result = {
                    "timestamp": timestamp,
                    "timestamp_str": f"{int(timestamp)//60}:{int(timestamp)%60:02d}",
                    "analysis": response,
                    "frame_number": i
                }
                
                results.append(result)
                self.analysis_cache.append(result)
                
                if callback:
                    callback(result)
                
                if i % progress_interval == 0 or i == len(timestamps) - 1:
                    print(f"  [{i+1}/{len(timestamps)}] {timestamp:.1f}s: Analyzed")
                    
                if self.verbose and len(response) > 0 and i % progress_interval == 0:
                    preview = response[:50] + "..." if len(response) > 50 else response
                    print(f"     ↪ {preview}")
                
            except GenerationCancelled:
                print(f"\n⏹ Generation cancelled at {timestamp:.1f}s")
                break
            except Exception as e:
                if i % progress_interval == 0:
                    print(f"❌ Error at {timestamp:.1f}s: {e}")
                results.append({
                    "timestamp": timestamp,
                    "timestamp_str": f"{int(timestamp)//60}:{int(timestamp)%60:02d}",
                    "error": str(e),
                    "frame_number": i
                })
        
        print(f"\n✅ Done. {len(results)} frames analyzed successfully")
        
        if save_to_file:
            self.save_results(results, save_to_file)
        
        return results

    def analyze_with_seeking(self, interval: float = 1.0, target_description: str = "explosion",
                            max_seeks: int = 100,
                            cancellation_token: Optional[CancellationToken] = None):
        results = []
        start_time = 0.0
        current_time = start_time
        
        print(f"\n🔍 Seeking every {interval}s looking for: {target_description}")
        print("=" * 60)
        
        for seek_num in range(max_seeks):
            if cancellation_token and cancellation_token.is_cancelled:
                print(f"\n⏹ Search cancelled at seek #{seek_num}")
                break

            timestamp = current_time + (seek_num * interval)
            
            if timestamp >= self.duration:
                print(f"\n🏁 Reached end of video at {timestamp:.1f}s")
                break
            
            print(f"\n⏩ Seeking to {timestamp:.1f}s ({int(timestamp)//60}:{int(timestamp)%60:02d})")
            frame = self.seek_to_time(timestamp)
            
            if frame is None:
                print(f"⚠️ Could not read frame at {timestamp:.1f}s")
                continue
            
            frame_b64 = self.frame_to_base64(frame)
            
            try:
                response = self.llm.query(
                    user_message=f"Does this frame contain a {target_description}? Answer with YES or NO, and briefly explain what you see.",
                    frame_base64=frame_b64,
                    system_prompt=LLMModule.SYSTEM_PROMPT_VISUAL_SEARCH,
                    temperature=0.1,
                    max_tokens=150,
                    cancellation_token=cancellation_token,
                )
                
                result = {
                    "timestamp": timestamp,
                    "timestamp_str": f"{int(timestamp)//60}:{int(timestamp)%60:02d}",
                    "analysis": response,
                    "contains_target": "yes" in response.lower() or target_description.lower() in response.lower()
                }
                
                results.append(result)
                print(f"📝 Analysis: {response[:100]}...")
                
                if result["contains_target"]:
                    print(f"\n🎯 FOUND {target_description.upper()} at {timestamp:.1f}s!")
                    print(f"Full analysis: {response}")
                    break
                
            except GenerationCancelled:
                print(f"\n⏹ Search cancelled at {timestamp:.1f}s")
                break
            except Exception as e:
                print(f"❌ Error at {timestamp:.1f}s: {e}")
            
            time.sleep(0.5)
        
        print(f"\n✅ Seeking complete. Analyzed {len(results)} frames")
        return results

    def save_results(self, results: list, filepath: str):
        output = {
            "video_path": self.video_path,
            "video_info": {
                "duration": self.duration,
                "fps": self.fps,
                "width": self.width,
                "height": self.height,
                "total_frames": self.total_frames
            },
            "analysis_interval": 1,
            "total_analyses": len(results),
            "analyses": results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=1, ensure_ascii=False)
        
        print(f"💾 Results saved to: {filepath}")
    
    def search_analyses(self, query: str, results: list = None) -> list:
        search_results = []
        analyses = results if results is not None else self.analysis_cache
        query_lower = query.lower()
        
        for r in analyses:
            if "analysis" in r and query_lower in r["analysis"].lower():
                search_results.append({
                    "timestamp": r["timestamp"],
                    "timestamp_str": r["timestamp_str"],
                    "context": r["analysis"][:200] + "..." if len(r["analysis"]) > 200 else r["analysis"]
                })
        
        return search_results
    
    def interactive_mode(self, interval: int = 1):
        self.running = True
        
        def analysis_loop():
            consecutive_errors = 0
            while self.running:
                try:
                    with self.lock:
                        current_pos = self.current_time
                    
                    result = self.analyze_current_frame()
                    
                    if "error" in result:
                        consecutive_errors += 1
                        if consecutive_errors > 3:
                            print("\n❌ Too many errors, stopping analysis")
                            break
                    else:
                        consecutive_errors = 0
                        self.analysis_cache.append(result)
                        
                        print(f"\n{'='*60}")
                        print(f"📹 [{result['timestamp_str']}] Analysis:")
                        print(f"{'='*60}")
                        analysis = result['analysis']
                        if len(analysis) > 300:
                            print(analysis[:300] + "...")
                        else:
                            print(analysis)
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    print(f"\n⚠️ Analysis error: {e}")
                    time.sleep(interval)
        
        thread = threading.Thread(target=analysis_loop, daemon=True)
        thread.start()
        self._show_help()
        
        while self.running:
            try:
                cmd = input("\n🎮 Command: ").strip().lower()
                if not cmd:
                    continue
                
                parts = cmd.split()
                
                if parts[0] == 's' and len(parts) > 1:
                    self._handle_seek_command(parts[1])
                elif parts[0] == 'p':
                    print(f"📌 Current position: {int(self.current_time)//60}:{int(self.current_time)%60:02d}")
                elif parts[0] == 'f' and len(parts) > 1:
                    self._handle_forward_command(parts[1])
                elif parts[0] == 'b' and len(parts) > 1:
                    self._handle_backward_command(parts[1])
                elif parts[0] == 'h':
                    self._show_help()
                elif parts[0] == 'q':
                    print("\n👋 Stopping analysis...")
                    self.running = False
                    break
                else:
                    print("❌ Unknown command. Type 'h' for help.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted, stopping...")
                self.running = False
                break
            except Exception as e:
                print(f"❌ Command error: {e}")
    
    def _show_help(self):
        print("\n" + "="*50)
        print("🎬 Interactive Video Analysis Mode")
        print("="*50)
        print("Commands:")
        print("  s <seconds>   - Seek to timestamp (e.g., 's 30')")
        print("  s <mm:ss>     - Seek to timestamp (e.g., 's 1:30')")
        print("  p             - Show current position")
        print("  f <sec>       - Move forward N seconds")
        print("  b <sec>       - Move backward N seconds")
        print("  h             - Show this help")
        print("  q             - Quit")
        print("="*50)
    
    def _handle_seek_command(self, arg: str):
        try:
            if ':' in arg:
                minutes, seconds = map(int, arg.split(':'))
                seek_to = minutes * 60 + seconds
            else:
                seek_to = int(arg)
            
            seek_to = max(0, min(seek_to, int(self.duration)))
            with self.lock:
                self.current_time = seek_to
            
            if self.verbose:
                print(f"⏩ Seeking to {self.current_time//60}:{self.current_time%60:02d}")
        except ValueError:
            print("❌ Invalid time format. Use seconds or mm:ss")
    
    def _handle_forward_command(self, arg: str):
        try:
            forward = int(arg)
            with self.lock:
                self.current_time = min(self.current_time + forward, self.duration)
            print(f"⏩ Forward {forward}s to {int(self.current_time)//60}:{int(self.current_time)%60:02d}")
        except ValueError:
            print("❌ Invalid forward value")
    
    def _handle_backward_command(self, arg: str):
        try:
            backward = int(arg)
            with self.lock:
                self.current_time = max(self.current_time - backward, 0)
            print(f"⏪ Back {backward}s to {int(self.current_time)//60}:{int(self.current_time)%60:02d}")
        except ValueError:
            print("❌ Invalid backward value")
    
    def close(self):
        self.running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        print("📹 Video capture released")


# ---------------------------------------------------------------------------
# Example usage when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("🔍 LLM Module with Video Seek Analyzer")
    print("=" * 50)
    print("This module provides:")
    print("  - LLMModule: Interface to local LLMs")
    print("  - VideoSeekAnalyzer: Analyze video frames every 1 second")
    print("\nExample usage:")
    print("  from llm_module import LLMModule, VideoSeekAnalyzer")
    print("  llm = LLMModule(backend='ollama', model='llava')")
    print("  llm.load()")
    print("  analyzer = VideoSeekAnalyzer('video.mp4', llm)")
    print("  results = analyzer.analyze_every_1_second()")
    print("  analyzer.close()")