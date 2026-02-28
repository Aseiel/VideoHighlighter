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
# Backend base
# ---------------------------------------------------------------------------
class _LLMBackend:
    """Abstract backend."""

    def load(self, **kwargs):
        raise NotImplementedError

    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024,
                 temperature: float = 0.7, stream_callback: Optional[Callable] = None,
                 images: list[str] | None = None) -> str:
        raise NotImplementedError

    def is_loaded(self) -> bool:
        return False

    def unload(self):
        pass

    @staticmethod
    def available() -> bool:
        return False


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
            # Match with or without tag suffix
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
                images: list[str] | None = None) -> str:
        import requests

        # For vision models: DON'T embed system in prompt — it drowns out the image
        if images:
            formatted_prompt = prompt
        else:
            # Text-only: embed system in prompt for models that ignore system field
            formatted_prompt = (
                f"[SYSTEM INSTRUCTIONS]\n{system}\n[END INSTRUCTIONS]\n\n{prompt}"
                if system else prompt
            )

        payload = {
            "model": self.model,
            "prompt": formatted_prompt,
            "system": system,
            "stream": stream_callback is not None,
            "options": {
                "num_predict": max_tokens,
                "num_ctx": 2048,
                "temperature": temperature,
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
                    chunk = json.loads(line)
                    token = chunk.get("response", "")
                    if token:
                        full_text.append(token)
                        stream_callback(token)
                    if chunk.get("done", False):
                        break
            return "".join(full_text)
        else:
            resp = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "")

    def unload(self):
        self._loaded = False


# ---------------------------------------------------------------------------
# llama-cpp-python backend
# ---------------------------------------------------------------------------
class _LlamaCppBackend(_LLMBackend):
    """Uses llama-cpp-python to run a GGUF model directly (no server)."""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._model = None

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
        self._model = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )

    def is_loaded(self) -> bool:
        return self._model is not None

    def generate(self, prompt: str, system: str = "", max_tokens: int = 1024,
                temperature: float = 0.7, stream_callback: Optional[Callable] = None,
                images: list[str] | None = None) -> str:
        if not self._model:
            raise RuntimeError("Model not loaded. Call load() first.")

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Note: llama-cpp-python doesn't support images natively
        # Vision requires Ollama backend
        if images:
            messages[-1]["content"] = "[Image attached but not supported by llama-cpp backend. Use Ollama with a vision model.]\n\n" + prompt

        if stream_callback:
            full_text = []
            for chunk in self._model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            ):
                delta = chunk["choices"][0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    full_text.append(token)
                    stream_callback(token)
            return "".join(full_text)
        else:
            result = self._model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return result["choices"][0]["message"]["content"]

    def unload(self):
        self._model = None


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
            # Aggregate object counts
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
            # Group by action name
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
            # Show first and last few segments
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

        Used when verifying whether a clip matches the target action.
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
        "You are an AI video analysis assistant integrated into VideoHighlighter.\n"
        "CRITICAL RULES:\n"
        "1. You ALWAYS have access to video analysis data provided below the separator '=== VIDEO ANALYSIS DATA ==='.\n"
        "2. When the user asks about objects, actions, or events in the video, you MUST search the provided data and quote specific findings with timestamps.\n"
        "3. NEVER say you 'don't have access' or 'can't see the video' — you have the full analysis results.\n"
        "4. If an action or object is NOT in the data, say 'This was not detected in the analysis' and list what WAS detected instead.\n"
        "5. Use MM:SS timestamps. Be concise and specific.\n"
        "6. When asked about a specific action, search the '## Detected Actions' section and report matches, confidence scores, and timestamps.\n"
        "7. When asked about objects, search '## Detected Objects' and report what was found and when."
    )

    SYSTEM_PROMPT_ACTION_LEARNING = (
        "You are an AI assistant helping to learn new action categories from video clips. "
        "Given detected objects, motion features, and existing action classifications, "
        "determine whether a video clip likely contains the target action. "
        "Respond with a confidence score (0.0 to 1.0) and brief reasoning. "
        "Format: SCORE: 0.XX\nREASON: ..."
    )

    SYSTEM_PROMPT_TIMELINE = (
        "You are an AI video analysis assistant integrated into VideoHighlighter.\n"
        "CRITICAL RULES:\n"
        "1. You have video analysis data below '=== VIDEO ANALYSIS DATA ==='.\n"
        "2. SEARCH the data and quote findings with MM:SS timestamps.\n"
        "3. NEVER say you 'don't have access' or 'can't see the video'.\n"
        "4. You can CONTROL THE TIMELINE with [CMD:...] commands.\n"
        "   - seek time MUST be a number of seconds (float or int), NOT mm:ss.\n"
        "5. BE CONCISE. Say what you're doing in 1-2 sentences, then the command.\n"
        "6. NEVER list all clips or echo the timeline state back. The user can see it.\n"
        "7. NEVER repeat the '## Current Timeline State' section in your response.\n"
    )

    SYSTEM_PROMPT_VISION = (
        "You have VISION. An image is attached to this message.\n"
        "IGNORE all text data. ONLY describe what you SEE in the image.\n"
        "Describe: people (count, poses, clothing, actions), objects, background, colors.\n"
        "If you cannot see the image, say 'I cannot see the image'.\n"
        "Do NOT mention timestamps, detections, or analysis data.\n"
        "Do NOT make up actions from text — ONLY describe the visual content.\n"
    )

    def __init__(
        self,
        backend: str = "ollama",
        model: str = "llama3.2",
        model_path: str = "",
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
                model_path=model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers
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
    ) -> str:
        """
        Send a query to the LLM with optional video analysis context.

        Args:
            user_message:    The user's question or instruction
            analysis_data:   Cache dict from VideoHighlighter pipeline
            video_path:      Video filename (for context)
            system_prompt:   Override the default system prompt
            timeline_context: Extra context about timeline state + available commands
            frame_base64:    Base64-encoded image of current video frame (for vision models)
            max_tokens:      Max response length
            temperature:     Sampling temperature
            stream_callback: Called with each token for streaming UI updates

        Returns:
            The LLM's response text
        """
        if not self._backend.is_loaded():
            raise RuntimeError("LLM not loaded. Call load() first.")
        
        # DEBUG: trace vision path
        print(f"🔍 query() called:")
        print(f"   frame_base64: {'YES ' + str(len(frame_base64)) + ' chars' if frame_base64 else 'NONE'}")
        print(f"   timeline_context: {'YES' if timeline_context else 'NONE'}")
        print(f"   analysis_data: {'YES' if analysis_data else 'NONE'}")

        # Check if this is a vision-focused query (user asking about current frame)
        _vision_keywords = ("see", "look", "frame", "current", "what's happening", 
                        "describe this", "what is this", "what do you see")
        _is_vision_query = any(kw in user_message.lower() for kw in _vision_keywords)
        
        # ===== VISION MODE: When user explicitly asks about current frame =====
        if frame_base64 and _is_vision_query:
            # Even with timeline context, prioritize vision for frame-specific questions
            if timeline_context:
                # Include timeline commands but focus on the image
                system = self.SYSTEM_PROMPT_VISION + "\n\n" + (
                    "IMPORTANT: The user has also provided timeline context below. "
                    "FIRST describe what you see in the image, THEN if relevant, "
                    "you can include timeline commands."
                )
                
                prompt_parts = [
                    "=== TIMELINE CONTEXT (for reference) ===\n",
                    timeline_context,
                    "\n=== END TIMELINE ===\n\n",
                    "Now, based on the image attached to this message, describe what you see.\n",
                    user_message
                ]
                full_prompt = "\n".join(prompt_parts)
            else:
                # Pure vision mode
                system = self.SYSTEM_PROMPT_VISION
                full_prompt = user_message
            
            return self._backend.generate(
                prompt=full_prompt,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
                stream_callback=stream_callback,
                images=[frame_base64],
            )
        
        # ===== TEXT MODE: No frame or not a vision query =====
        prompt_parts = []
        
        if analysis_data:
            context = VideoContextBuilder.build(analysis_data, video_path)
            prompt_parts.append(
                "=== VIDEO ANALYSIS DATA ===\n"
                "This is the ACTUAL analysis output from processing the video. "
                "Use this data to answer the user's question. "
                "All timestamps, objects, actions, and transcript text below are REAL detections.\n\n"
                f"{context}\n"
                "=== END DATA ===\n\n"
            )
        
        if timeline_context and not _is_vision_query:
            # Only include timeline context if not asking about current frame
            prompt_parts.append(
                "=== TIMELINE CONTROL ===\n"
                f"{timeline_context}\n"
                "=== END TIMELINE ===\n\n"
            )
        
        prompt_parts.append(
            "Based on the data above, answer the following. "
            "If the user asks you to modify the timeline, include the appropriate [CMD:...] commands.\n\n"
        )
        prompt_parts.append(user_message)
        full_prompt = "\n".join(prompt_parts)
        
        # Choose appropriate system prompt
        if system_prompt:
            system = system_prompt
        elif timeline_context and not _is_vision_query:
            system = self.SYSTEM_PROMPT_TIMELINE
        else:
            system = self.SYSTEM_PROMPT
        
        return self._backend.generate(
            prompt=full_prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            stream_callback=stream_callback,
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

        Args:
            action_name:     e.g. "pick up cup"
            object_name:     e.g. "cup"
            clip_analysis:   List of per-clip dicts with keys: objects, actions, motion_level
            existing_labels: Current known action labels

        Returns:
            (confidence_score, reasoning)
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

        # Parse response
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
        """
        Non-blocking query — runs in a background thread.

        Args:
            user_message:   The question
            callback:       Called with final response text
            error_callback: Called with error message on failure
            **kwargs:       Passed to query()

        Returns:
            The background Thread object
        """
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
# Video Seek Analyzer - NEW: Analyze video frames every 1 seconds
# ---------------------------------------------------------------------------
class VideoSeekAnalyzer:
    """
    Analyzes video frames at regular intervals using LLM vision.
    
    This class provides functionality to:
    - Seek to specific timestamps in a video
    - Capture frames and convert to base64 for LLM vision
    - Analyze frames every N seconds (default 1)
    - Save analysis results to JSON
    
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
        """
        Initialize the video analyzer.
        
        Args:
            video_path: Path to the video file
            llm: Initialized LLMModule instance (must support vision)
        
        Raises:
            ImportError: If OpenCV is not installed
            FileNotFoundError: If video file doesn't exist
        """
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
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # State for interactive mode
        self.current_time = 0
        self.lock = threading.Lock()
        self.running = False
        self.analysis_cache = []
        
        print(f"📹 Video loaded: {os.path.basename(video_path)}")
        print(f"   Duration: {int(self.duration)//60}m{int(self.duration)%60:02d}s")
        print(f"   Resolution: {self.width}x{self.height}")
        print(f"   FPS: {self.fps:.2f}")
    
    def seek_to_time(self, timestamp_seconds: float):
        """
        Seek to a specific timestamp in the video.
        
        Args:
            timestamp_seconds: Time in seconds to seek to
        
        Returns:
            Frame as numpy array or None if failed
        """
        # Ensure timestamp is within bounds
        timestamp_seconds = max(0, min(timestamp_seconds, self.duration))
        
        # Calculate frame number and seek
        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp_seconds * 1000)
        self.cap.read()  # flush
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def frame_to_base64(self, frame, quality: int = 100) -> str:
        """
        Convert OpenCV frame to base64 string for LLM vision.
        
        Args:
            frame: OpenCV frame (numpy array)
            quality: JPEG quality (0-100)
        
        Returns:
            Base64 encoded JPEG image
        """
        # Encode frame as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_params)
        
        # Convert to base64
        return base64.b64encode(buffer).decode('utf-8')
    
    def analyze_current_frame(
        self, 
        user_query: str = "What do you see in this frame? Describe the scene, people, objects, and actions.",
        custom_prompt: str = None
    ) -> dict:
        """
        Analyze the current frame using LLM vision.
        
        Args:
            user_query: Question to ask about the frame
            custom_prompt: Override the default system prompt
        
        Returns:
            Dictionary with analysis results
        """
        # Seek to current time and get frame
        frame = self.seek_to_time(self.current_time)
        if frame is None:
            return {
                "error": "Could not read frame",
                "timestamp": self.current_time,
                "timestamp_str": f"{int(self.current_time)//60}:{int(self.current_time)%60:02d}"
            }
        
        # Convert frame to base64
        frame_b64 = self.frame_to_base64(frame)
        
        # Use custom prompt if provided
        system = custom_prompt if custom_prompt else None
        
        # Query LLM with vision
        try:
            response = self.llm.query(
                user_message=user_query,
                frame_base64=frame_b64,
                system_prompt=system,
                temperature=0.3,
                max_tokens=500
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
        except Exception as e:
            return {
                "error": str(e),
                "timestamp": self.current_time,
                "timestamp_str": f"{int(self.current_time)//60}:{int(self.current_time)%60:02d}"
            }
    
    def analyze_every_n_seconds(self, interval: float = 1.0, callback=None, save_to_file=None):
        """
        Properly seek to each timestamp instead of sequential reading.
        
        Args:
            interval: Seconds between analyses (e.g., 1.0 for every second)
            callback: Optional function to call with each result
            save_to_file: Optional path to save results JSON
        
        Returns:
            List of analysis results
        """
        results = []
        
        # Calculate all timestamps to analyze
        num_analyses = int(self.duration / interval) + 1
        timestamps = [i * interval for i in range(num_analyses)]
        
        print(f"\n📊 Seeking analysis every {interval}s ({len(timestamps)} frames)")
        print(f"   Video duration: {int(self.duration)//60}m{int(self.duration)%60:02d}s")
        
        # ONLY print progress every 10 frames to reduce spam
        progress_interval = max(1, len(timestamps) // 10)  # Show ~10 updates total
        
        for i, timestamp in enumerate(timestamps):
            # Skip if beyond video duration (with small epsilon for floating point)
            if timestamp > self.duration + 0.1:
                break
                
            # Update current time
            self.current_time = timestamp
            
            # Seek to exact timestamp
            frame = self.seek_to_time(timestamp)
            
            if frame is None:
                # Only print errors occasionally
                if i % progress_interval == 0:
                    print(f"⚠️ Could not read frame at {timestamp:.1f}s")
                continue
            
            # Convert frame to base64
            frame_b64 = self.frame_to_base64(frame)
            
            # Analyze frame
            try:
                response = self.llm.query(
                    user_message="What do you see in this frame? Describe the scene, people, objects, and actions.",
                    frame_base64=frame_b64,
                    temperature=0.3,
                    max_tokens=500,
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
                
                # ONLY print progress at interval, not every frame
                if i % progress_interval == 0 or i == len(timestamps) - 1:
                    print(f"  [{i+1}/{len(timestamps)}] {timestamp:.1f}s: Analyzed")
                    
                # Optional verbose mode - but limited
                if self.verbose and len(response) > 0 and i % progress_interval == 0:
                    preview = response[:50] + "..." if len(response) > 50 else response
                    print(f"     ↪ {preview}")
                
            except Exception as e:
                # Only print errors occasionally
                if i % progress_interval == 0:
                    print(f"❌ Error at {timestamp:.1f}s: {e}")
                # Add error result to maintain timeline
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

    def analyze_with_seeking(self, interval: float = 1.0, target_description: str = "explosion", max_seeks: int = 100):
        """
        Properly seek through video at specified intervals and analyze each frame.
        
        Args:
            interval: Seconds between seeks (1.0 or 2.0)
            target_description: What to look for (e.g., "explosion")
            max_seeks: Maximum number of seeks to perform
        
        Returns:
            List of analysis results
        """
        results = []
        
        # Start from beginning or current position
        start_time = 0.0
        current_time = start_time
        
        print(f"\n🔍 Seeking every {interval}s looking for: {target_description}")
        print("=" * 60)
        
        for seek_num in range(max_seeks):
            # Calculate next timestamp
            timestamp = current_time + (seek_num * interval)
            
            # Stop if we've reached the end of the video
            if timestamp >= self.duration:
                print(f"\n🏁 Reached end of video at {timestamp:.1f}s")
                break
            
            # SEEK to the exact timestamp
            print(f"\n⏩ Seeking to {timestamp:.1f}s ({int(timestamp)//60}:{int(timestamp)%60:02d})")
            frame = self.seek_to_time(timestamp)
            
            if frame is None:
                print(f"⚠️ Could not read frame at {timestamp:.1f}s")
                continue
            
            # Convert frame to base64
            frame_b64 = self.frame_to_base64(frame)
            
            # Analyze the frame for the target
            try:
                response = self.llm.query(
                    user_message=f"Does this frame contain a {target_description}? Answer with YES or NO, and briefly explain what you see.",
                    frame_base64=frame_b64,
                    temperature=0.1,  # Low temperature for consistent answers
                    max_tokens=150,
                )
                
                result = {
                    "timestamp": timestamp,
                    "timestamp_str": f"{int(timestamp)//60}:{int(timestamp)%60:02d}",
                    "analysis": response,
                    "contains_target": "yes" in response.lower() or "explosion" in response.lower()
                }
                
                results.append(result)
                
                # Display result
                print(f"📝 Analysis: {response[:100]}...")
                
                # Check if we found the target
                if result["contains_target"]:
                    print(f"\n🎯 FOUND {target_description.upper()} at {timestamp:.1f}s!")
                    print(f"Full analysis: {response}")
                    break
                
            except Exception as e:
                print(f"❌ Error at {timestamp:.1f}s: {e}")
            
            # Small pause to simulate real seeking
            time.sleep(0.5)
        
        print(f"\n✅ Seeking complete. Analyzed {len(results)} frames")
        return results

    def save_results(self, results: list, filepath: str):
        """
        Save analysis results to JSON file.
        
        Args:
            results: List of analysis dictionaries
            filepath: Path to save JSON file
        """
        output = {
            "video_path": self.video_path,
            "video_info": {
                "duration": self.duration,
                "fps": self.fps,
                "width": self.width,
                "height": self.height,
                "total_frames": self.total_frames
            },
            "analysis_interval": 1,  # Default interval
            "total_analyses": len(results),
            "analyses": results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=1, ensure_ascii=False)
        
        print(f"💾 Results saved to: {filepath}")
    
    def search_analyses(self, query: str, results: list = None) -> list:
        """
        Search through analysis results for specific content.
        
        Args:
            query: Text to search for in analyses
            results: List to search (uses cache if None)
        
        Returns:
            List of matching results with timestamps
        """
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
        """
        Interactive mode with real-time analysis and seeking commands.
        
        Commands:
            s <seconds> or s <mm:ss> - Seek to timestamp
            p - Show current position
            f <sec> - Move forward N seconds
            b <sec> - Move backward N seconds
            q - Quit
            h - Show help
        
        Args:
            interval: Seconds between automatic analyses
        """
        self.running = True
        
        def analysis_loop():
            """Background thread for continuous analysis"""
            consecutive_errors = 0
            
            while self.running:
                try:
                    with self.lock:
                        current_pos = self.current_time
                    
                    # Analyze current frame
                    result = self.analyze_current_frame()
                    
                    if "error" in result:
                        consecutive_errors += 1
                        if consecutive_errors > 3:
                            print("\n❌ Too many errors, stopping analysis")
                            break
                    else:
                        consecutive_errors = 0
                        self.analysis_cache.append(result)
                        
                        # Show result
                        print(f"\n{'='*60}")
                        print(f"📹 [{result['timestamp_str']}] Analysis:")
                        print(f"{'='*60}")
                        # Truncate long responses for display
                        analysis = result['analysis']
                        if len(analysis) > 300:
                            print(analysis[:300] + "...")
                        else:
                            print(analysis)
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    print(f"\n⚠️ Analysis error: {e}")
                    time.sleep(interval)
        
        # Start analysis thread
        thread = threading.Thread(target=analysis_loop, daemon=True)
        thread.start()
        
        # Help text
        self._show_help()
        
        # Command loop
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
        """Show interactive mode help"""
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
        """Handle seek command with various formats"""
        try:
            # Parse mm:ss format
            if ':' in arg:
                minutes, seconds = map(int, arg.split(':'))
                seek_to = minutes * 60 + seconds
            else:
                # Parse seconds
                seek_to = int(arg)
            
            # Validate and seek
            seek_to = max(0, min(seek_to, int(self.duration)))
            
            with self.lock:
                self.current_time = seek_to
            
            if self.verbose:
                print(f"⏩ Seeking to {self.current_time//60}:{self.current_time%60:02d}")
            
        except ValueError:
            print("❌ Invalid time format. Use seconds or mm:ss")
    
    def _handle_forward_command(self, arg: str):
        """Handle forward command"""
        try:
            forward = int(arg)
            with self.lock:
                self.current_time = min(self.current_time + forward, self.duration)
            print(f"⏩ Forward {forward}s to {int(self.current_time)//60}:{int(self.current_time)%60:02d}")
        except ValueError:
            print("❌ Invalid forward value")
    
    def _handle_backward_command(self, arg: str):
        """Handle backward command"""
        try:
            backward = int(arg)
            with self.lock:
                self.current_time = max(self.current_time - backward, 0)
            print(f"⏪ Back {backward}s to {int(self.current_time)//60}:{int(self.current_time)%60:02d}")
        except ValueError:
            print("❌ Invalid backward value")
    
    def close(self):
        """Release video capture resources"""
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