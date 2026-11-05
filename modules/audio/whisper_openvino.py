"""Whisper on an Intel GPU, through OpenVINO GenAI.

openai-whisper runs on CUDA and on Apple's Metal; an Intel card gets the
processor. Through torch's XPU build it is slower still than the processor
(measured 2026-10-22 on an Arc A750). OpenVINO GenAI's ``WhisperPipeline`` on
the same card is 5-6x faster than the processor, measured again on 2026-11-05
with the app's settings over three 10-minute clips (73.8 -> 13.1 s, 71.9 ->
13.5 s, 104.3 -> 20.4 s):

* **Clean speech comes out the same**, word for word and to the timestamp.
* **Noisy audio comes out differently, not clearly worse.** Both paths lock
  into repetition loops there (the processor's raw output had 123 looping
  segments on the noisiest clip, this one's 144). Once
  ``transcript_cleanup`` has taken those out they end up with about the same
  words (440 vs 426), placed a few seconds apart in places.
* What the pipeline lacks is openai-whisper's own guards: no fallback to a
  higher temperature when a window's text compresses too well, and no
  per-segment no-speech numbers. ``transcript_cleanup`` is what catches the
  loops instead, on this path as on the other. Re-decoding the windows that
  looked wrong on the processor was measured too and does not pay: on noisy
  audio it re-decodes most of them, which costs the speed, and the second
  noisy guess is no closer to the first.

The models are the OpenVINO project's own exports on Hugging Face
(``OpenVINO/whisper-<size>-fp16-ov``, Apache-2.0 / MIT like the Whisper
weights), downloaded once into the app's models folder on first use. The
runtime is ``openvino-genai`` (Apache-2.0), pinned to the ``openvino`` the app
already ships. Nothing here is required: without the package, the card or the
model, transcription runs where it ran before.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, Dict, List, Optional

SAMPLE_RATE = 16000
# Shorter than this holds no line, and the pipeline rejects very short audio.
MIN_SECONDS = 0.5

# The app's model names -> the OpenVINO project's exports of them. "large" is
# what openai-whisper means by it today: large-v3.
REPOS = {
    "tiny": "OpenVINO/whisper-tiny-fp16-ov",
    "base": "OpenVINO/whisper-base-fp16-ov",
    "small": "OpenVINO/whisper-small-fp16-ov",
    "medium": "OpenVINO/whisper-medium-fp16-ov",
    "large": "OpenVINO/whisper-large-v3-fp16-ov",
    "large-v3": "OpenVINO/whisper-large-v3-fp16-ov",
    "turbo": "OpenVINO/whisper-large-v3-turbo-fp16-ov",
    "large-v3-turbo": "OpenVINO/whisper-large-v3-turbo-fp16-ov",
}
# Files a usable export has; a folder missing one is a download that stopped.
REQUIRED_FILES = ("openvino_encoder_model.xml", "openvino_decoder_model.xml",
                  "openvino_tokenizer.xml", "openvino_detokenizer.xml",
                  "generation_config.json")
MODELS_SUBDIR = "whisper-ov"

# Set to 0 to keep Whisper off the Intel GPU whatever else is chosen.
ENV_VAR = "VH_WHISPER_OPENVINO"

# Until a chunk has been timed on this machine, how much faster than real time
# the bar assumes the card runs (measured: 30-46x on an A750).
_ASSUMED_SPEED = 30.0


def disabled_by_env() -> bool:
    return os.environ.get(ENV_VAR, "").strip().lower() in ("0", "false", "off", "no")


def genai_available() -> bool:
    """True when the openvino_genai package imports."""
    try:
        import openvino_genai  # noqa: F401
    except Exception:  # noqa: BLE001 - absent or broken: either way, not here
        return False
    return True


def intel_gpu() -> Optional[str]:
    """The OpenVINO device name of the Intel GPU to use (discrete first), or
    None. Same choice the frame encoder makes."""
    from modules.vision.frame_encoder import _openvino_gpu
    device, _discrete = _openvino_gpu()
    return device


def wanted(backend: Optional[str], cuda: bool) -> bool:
    """Whether the run's accelerator choice sends Whisper to the Intel GPU.

    Automatic leaves an NVIDIA card in charge (openai-whisper on CUDA has all
    of its own guards) and otherwise takes the Intel card. "Intel" asks for it
    outright; every other named choice keeps Whisper where it was.
    """
    from modules.system import compute_backend as cb
    if disabled_by_env():
        return False
    chosen = cb.normalise(backend) if backend else cb.AUTO
    if chosen == cb.INTEL:
        return True
    return chosen == cb.AUTO and not cuda


def models_root() -> str:
    from modules.system import app_paths
    return os.path.join(app_paths.user_data_dir(), "models", MODELS_SUBDIR)


def _complete(folder: str) -> bool:
    return all(os.path.isfile(os.path.join(folder, f)) for f in REQUIRED_FILES)


def model_dir(model_name: str, log_fn=print, download: bool = True) -> Optional[str]:
    """The folder holding the export for ``model_name``, downloading it the
    first time. None when this size has no export or it cannot be fetched."""
    repo = REPOS.get(str(model_name or "").strip().lower())
    if not repo:
        return None
    folder = os.path.join(models_root(), repo.split("/", 1)[1])
    if _complete(folder):
        return folder
    if not download:
        return None
    try:
        from modules.system import hf_mirror
        hf_mirror.apply()
        from huggingface_hub import snapshot_download
        log_fn(f"⬇️ First use on the Intel GPU: downloading Whisper '{model_name}' "
               f"for OpenVINO ({repo})…")
        snapshot_download(repo_id=repo, local_dir=folder)
    except Exception as e:  # noqa: BLE001 - no download, no GPU route
        log_fn(f"⚠️ Could not download {repo}: {type(e).__name__}: {e}")
        return None
    return folder if _complete(folder) else None


def _compile_cache() -> Optional[str]:
    """OpenVINO's compiled-kernel cache, shared with the frame encoder."""
    try:
        from modules.system import app_paths
        path = os.path.join(app_paths.user_data_dir(), "ov-cache")
        os.makedirs(path, exist_ok=True)
        return path
    except Exception:  # noqa: BLE001 - no cache only costs time
        return None


class Cancelled(RuntimeError):
    """The run was cancelled while a chunk was decoding."""


class OpenVINOWhisper:
    """One loaded ``WhisperPipeline``; ``transcribe`` mirrors the parts of
    openai-whisper's result the app reads."""

    def __init__(self, folder: str, device: str):
        import openvino_genai as ov_genai
        props = {}
        cache = _compile_cache()
        if cache:
            props["CACHE_DIR"] = cache
        self.device = device
        self.folder = folder
        self._pipe = ov_genai.WhisperPipeline(folder, device, **props)
        self._speed = _ASSUMED_SPEED
        self._lock = threading.Lock()

    def _generate(self, samples, language: Optional[str]):
        kwargs = dict(task="transcribe", return_timestamps=True)
        if language and language != "auto":
            kwargs["language"] = f"<|{language}|>"
        with self._lock:
            return self._pipe.generate(samples, **kwargs)

    def transcribe(self, audio, language: Optional[str] = "en",
                   on_frac: Optional[Callable[[float], None]] = None,
                   should_cancel: Optional[Callable[[], bool]] = None,
                   poll: float = 0.1) -> Dict:
        """Decode ``audio`` (a 16 kHz float array, as whisper.load_audio
        returns) into ``{"language": ..., "segments": [...]}``.

        The pipeline decodes a whole chunk in one call and takes no callback
        for long audio, so it runs on a helper thread while this one keeps the
        bar moving (by the speed measured on the previous chunk) and watches
        for a cancel. A cancel returns at once; the call already on the card
        finishes on its own (seconds) and its result is dropped. Nothing is
        killed.
        """
        import numpy as np

        samples = np.asarray(audio, dtype=np.float32).tolist()
        seconds = len(samples) / SAMPLE_RATE
        if seconds < MIN_SECONDS:
            # The last chunk of a video can be a sliver (16 samples, seen on a
            # 600.001 s file), and the pipeline refuses anything shorter than
            # its FFT window. There is no speech in it either way.
            if on_frac:
                on_frac(1.0)
            return {"language": language if language and language != "auto" else "unknown",
                    "segments": []}
        box: Dict = {}

        def work():
            try:
                box["result"] = self._generate(samples, language)
            except BaseException as e:  # noqa: BLE001 - handed to the caller
                box["error"] = e

        t0 = time.time()
        worker = threading.Thread(target=work, name="whisper-openvino", daemon=True)
        worker.start()
        expected = max(1.0, seconds / max(self._speed, 1.0))
        while worker.is_alive():
            worker.join(poll)
            if should_cancel and should_cancel():
                raise Cancelled("Transcription cancelled")
            if on_frac:
                on_frac(min(0.95, (time.time() - t0) / expected))
        if "error" in box:
            raise box["error"]
        took = time.time() - t0
        if took > 0.5:
            self._speed = seconds / took
        if on_frac:
            on_frac(1.0)
        return to_result(box["result"])


def to_result(decoded) -> Dict:
    """openai-whisper's result shape from a WhisperDecodedResults: language
    code and segments with start, end and text. No no-speech or log-prob
    numbers: the pipeline has none, and transcript_cleanup only judges
    segments that carry them."""
    segments: List[Dict] = []
    for chunk in (getattr(decoded, "chunks", None) or []):
        text = (getattr(chunk, "text", "") or "").strip()
        if not text:
            continue
        start = float(getattr(chunk, "start_ts", 0.0) or 0.0)
        end = float(getattr(chunk, "end_ts", start) or start)
        if end < start:          # an open-ended last chunk comes back as -1
            end = start
        segments.append({"start": start, "end": end, "text": text})
    language = (getattr(decoded, "language", "") or "").strip("<|>") or "unknown"
    return {"language": language, "segments": segments}


def load(model_name: str, backend: Optional[str], cuda: bool,
         log_fn=print) -> Optional[OpenVINOWhisper]:
    """The Intel-GPU Whisper for this run, or None (with a line in the log
    when it was wanted and could not be had)."""
    if not wanted(backend, cuda):
        return None
    if not genai_available():
        from modules.system import compute_backend as cb
        if cb.normalise(backend) == cb.INTEL:
            log_fn("⚠️ Whisper on the Intel GPU needs the openvino-genai package; "
                   "using the processor")
        return None
    device = intel_gpu()
    if not device:
        return None
    folder = model_dir(model_name, log_fn)
    if not folder:
        log_fn(f"⚠️ No OpenVINO export of Whisper '{model_name}'; using the processor")
        return None
    try:
        t = time.time()
        model = OpenVINOWhisper(folder, device)
        log_fn(f"✅ Whisper '{model_name}' on the Intel GPU ({device}, OpenVINO) "
               f"loaded in {time.time() - t:.1f}s")
        return model
    except Exception as e:  # noqa: BLE001 - any failure costs the GPU, not the transcript
        log_fn(f"⚠️ Whisper could not start on the Intel GPU ({type(e).__name__}: {e}); "
               f"using the processor")
        return None
