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
* Nor can it be told to forget the previous window, so a chunk is handed to
  it in independent pieces of under 30 s (``PIECE_SECONDS``).

The models are the OpenVINO project's own exports on Hugging Face
(``OpenVINO/whisper-<size>-fp16-ov``, Apache-2.0 / MIT like the Whisper
weights), downloaded once into the app's models folder on first use. The
runtime is ``openvino-genai`` (Apache-2.0), pinned to the ``openvino`` the app
already ships. Nothing here is required: without the package, the card or the
model, transcription runs where it ran before.
"""
from __future__ import annotations

import difflib
import os
import re
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


# Longest piece the pipeline is handed in one call. Given a whole 10-minute
# chunk it walks it 30 s at a time, and each window carries the last one's
# habits forward: measured 2026-11-16 on a 21-minute episode with the 'small'
# model, once a window came back lowercase without punctuation the next ones
# did too ("My name's Ash and I'm misty" where two people speak), and a window
# opening on music came back as six words stretched over 20 s, with the dialogue
# that followed it lost. openai-whisper has a switch for that carry-over
# (condition_on_previous_text, off in this app); the pipeline has none, and an
# initial prompt did not change it. Decoded as independent pieces of at most a
# window, the first 10 minutes of that episode gave 1400 words instead of 1231,
# 2 long unpunctuated lines instead of 11, the two speakers as two lines, and no
# stretched line - for 41 s of decoding instead of 32.
PIECE_SECONDS = 28.0
# Each piece ends at the quietest moment of its last few seconds, so a cut
# falls between words rather than through one.
CUT_SEARCH_SECONDS = 6.0
CUT_FRAME_SECONDS = 0.2


def quiet_cuts(audio, piece: float = PIECE_SECONDS, search: float = CUT_SEARCH_SECONDS,
               frame: float = CUT_FRAME_SECONDS) -> List[int]:
    """Sample positions that split ``audio`` into pieces of at most ``piece``
    seconds, each ending at the quietest ``frame`` of its last ``search``
    seconds. Starts at 0 and ends at ``len(audio)``."""
    import numpy as np

    audio = np.asarray(audio, dtype=np.float32)
    n = len(audio)
    span = int(piece * SAMPLE_RATE)
    width = max(1, int(frame * SAMPLE_RATE))
    step = max(1, width // 2)
    cuts = [0]
    while n - cuts[-1] > span:
        start = cuts[-1]
        lo = start + max(width, int((piece - search) * SAMPLE_RATE))
        hi = start + span
        best, quietest = hi, None
        for p in range(lo, hi - width + 1, step):
            energy = float(np.mean(audio[p:p + width] ** 2))
            if quietest is None or energy < quietest:
                best, quietest = p + width // 2, energy
        cuts.append(best)
    cuts.append(n)
    return cuts


# A piece whose decode stops this long before the piece does has left speech
# unread. Within one piece the decoder still carries a habit forward: measured
# 2026-11-17 on the same episode, a piece opening on the theme song stayed in
# the song and stopped at its last lyric, 8 s early, with "A Golden Rod
# Opportunity!" in those 8 s; another stopped after one line and left the
# rest of the song. Decoded again on its own, each tail came back as what the
# other transcripts of that episode have there (five tails in 10 minutes, five
# lines or runs of lyric back). An empty piece is not re-read: there the
# decoder heard nothing at all, not something and then stopped.
HOLE_SECONDS = 3.0
# The tail is decoded from the quietest moment just after the last line, not
# from the line's end: that end can sit inside the song's last note (64.88 for
# a song that ended at 65.2), and from there the same tail came back as a line
# nobody says; from any start in the silence after it, as the right one.
HOLE_SEARCH_SECONDS = 3.0
# Less than this left after the start is not worth a call.
HOLE_MIN_REST_SECONDS = 2.0


def hole_start(audio, last: int, end: int, search: float = HOLE_SEARCH_SECONDS,
               frame: float = CUT_FRAME_SECONDS) -> Optional[int]:
    """Sample to decode the unread tail ``audio[last:end]`` again from: the
    middle of its quietest ``frame`` within ``search`` seconds after ``last``.
    None when the tail is shorter than ``HOLE_SECONDS`` or leaves too little."""
    import numpy as np

    if (end - last) / SAMPLE_RATE < HOLE_SECONDS:
        return None
    width = max(1, int(frame * SAMPLE_RATE))
    hi = min(last + int(search * SAMPLE_RATE), end - int(HOLE_MIN_REST_SECONDS * SAMPLE_RATE))
    if hi - width < last:
        return None
    audio = np.asarray(audio, dtype=np.float32)
    best, quietest = last, None
    for p in range(last, hi - width + 1, max(1, width // 2)):
        energy = float(np.mean(audio[p:p + width] ** 2))
        if quietest is None or energy < quietest:
            best, quietest = p + width // 2, energy
    return best


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
        # Word timestamps are what put each line where it is said (see
        # ``to_result``). They are a property of the compiled pipeline, so an
        # export without alignment heads fails here and loads without them.
        try:
            self._pipe = ov_genai.WhisperPipeline(folder, device, word_timestamps=True, **props)
            self._words = True
        except Exception as e:  # noqa: BLE001 - line times are still Whisper's own
            print(f"whisper_openvino: no word timestamps for {folder}: {type(e).__name__}: {e}")
            self._pipe = ov_genai.WhisperPipeline(folder, device, **props)
            self._words = False
        self._speed = _ASSUMED_SPEED
        self._lock = threading.Lock()

    def _generate(self, samples, language: Optional[str]):
        kwargs = dict(task="transcribe", return_timestamps=True)
        if getattr(self, "_words", False):
            kwargs["word_timestamps"] = True
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

        Audio longer than one Whisper window is decoded as independent pieces
        cut at quiet moments (``quiet_cuts``), each timed from the start of
        ``audio``: see ``PIECE_SECONDS`` for why. A piece whose decode stopped
        well before its end has that tail decoded again (``HOLE_SECONDS``).

        The pipeline decodes a piece in one call and takes no callback, so it
        runs on a helper thread while this one keeps the bar moving (by the
        speed measured so far) and watches for a cancel. A cancel returns at
        once; the call already on the card finishes on its own (seconds) and
        its result is dropped. Nothing is killed.
        """
        import numpy as np

        audio = np.asarray(audio, dtype=np.float32)
        seconds = len(audio) / SAMPLE_RATE
        if seconds < MIN_SECONDS:
            # The last chunk of a video can be a sliver (16 samples, seen on a
            # 600.001 s file), and the pipeline refuses anything shorter than
            # its FFT window. There is no speech in it either way.
            if on_frac:
                on_frac(1.0)
            return {"language": language if language and language != "auto" else "unknown",
                    "segments": []}

        cuts = quiet_cuts(audio)
        segments: List[Dict] = []
        detected = None
        shown = [0.0]

        def piece(a, b):
            if should_cancel and should_cancel():
                raise Cancelled("Transcription cancelled")

            def piece_frac(f):
                # Never backwards: a tail decoded again starts inside its piece.
                shown[0] = max(shown[0], min(0.95, (a + f * (b - a)) / len(audio)))
                if on_frac:
                    on_frac(shown[0])

            got = to_result(self._decode(audio[a:b].tolist(), language, piece_frac,
                                         should_cancel, poll))
            offset = a / SAMPLE_RATE
            return got["language"], [dict(seg, start=seg["start"] + offset,
                                          end=seg["end"] + offset)
                                     for seg in got["segments"]]

        for a, b in zip(cuts, cuts[1:]):
            if (b - a) / SAMPLE_RATE < MIN_SECONDS:
                continue
            lang, got = piece(a, b)
            detected = detected or (lang if lang != "unknown" else None)
            segments.extend(got)
            if got:
                last = max(a, int(max(s["end"] for s in got) * SAMPLE_RATE))
                start = hole_start(audio, last, b)
                if start is not None:
                    segments.extend(piece(start, b)[1])
        if on_frac:
            on_frac(1.0)
        return {"language": detected or "unknown", "segments": segments}

    def _decode(self, samples, language, on_frac, should_cancel, poll):
        """One piece through the pipeline, on a helper thread; see
        ``transcribe``."""
        seconds = len(samples) / SAMPLE_RATE
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
            on_frac(min(0.95, (time.time() - t0) / expected))
        if "error" in box:
            raise box["error"]
        took = time.time() - t0
        if took > 0.5:
            self._speed = seconds / took
        return box["result"]


_TOKEN = re.compile(r"\w+(?:'\w+)*", re.UNICODE)

def _tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


def times_from_words(segments: List[Dict], words) -> int:
    """Give each segment the start and end of its own words, in place.
    Returns how many segments moved.

    The pipeline's line times are not where the lines are said. Over a long
    chunk they fell on a half-second grid, 0.1-0.5 s early (295.44 for a line
    said at 295.7); decoded as one piece (``PIECE_SECONDS``) they are worse:
    the first line starts where the piece does and each next one where the
    last ended, on whole seconds - measured 2026-11-17, "That sure was a close
    call" at 307.8 for words said from 315.0, and "This place is humongous" at
    73.6 for 80.8. The word times were right in both. So the words win however
    far they are from the line; an earlier limit of 1 s from the line's own
    times, set while those were only the grid's fraction off, threw exactly
    these lines' words away.

    The words carry the same text as the lines, plus any loop the lines were
    cleaned of, so they are matched to the lines token by token. A line keeps
    Whisper's times when its words do not all match, or when another spoken
    word falls among them: then the line was matched across speech that is not
    its own.
    """
    from modules.audio.segment_timing import bounds_from_words

    flat = []                                   # (token, word index)
    for i, w in enumerate(words or []):
        for t in _tokens(getattr(w, "word", "")):
            flat.append((t, i))
    if not flat or not segments:
        return 0
    line_tokens = [(t, k) for k, seg in enumerate(segments) for t in _tokens(seg["text"])]
    owner: Dict[int, List[int]] = {}
    matcher = difflib.SequenceMatcher(None, [t for t, _ in line_tokens],
                                      [t for t, _ in flat], autojunk=False)
    for a, b, size in matcher.get_matching_blocks():
        for n in range(size):
            owner.setdefault(line_tokens[a + n][1], []).append(flat[b + n][1])
    wanted = {}
    for _t, k in line_tokens:
        wanted[k] = wanted.get(k, 0) + 1
    moved = 0
    for k, seg in enumerate(segments):
        found = owner.get(k, [])
        if not found or len(found) != wanted.get(k):
            continue
        own = sorted(set(found))
        mine = set(own)
        if any(_tokens(getattr(words[i], "word", ""))
               for i in range(own[0], own[-1] + 1) if i not in mine):
            continue
        picked = [words[i] for i in own]
        bounds = bounds_from_words([(float(w.start_ts), float(w.end_ts), w.word, 1.0)
                                    for w in picked])
        if bounds is None:
            continue
        start, end = bounds
        if end <= start:
            continue
        seg["start"], seg["end"] = start, end
        moved += 1
    if moved:
        _no_overlaps(segments)
    return moved


def _no_overlaps(segments: List[Dict]) -> None:
    """One line ends before the next begins. Moving some lines and not others
    left a few overlapping their neighbour (by under a second, on one episode
    10 of 386): the earlier line ends where the next starts, and a next line
    that now starts before the earlier one (a repeated "Pikachu!" matched to the
    other copy) starts where the earlier one ends instead."""
    for prev, nxt in zip(segments, segments[1:]):
        if nxt["start"] >= prev["end"]:
            continue
        if nxt["start"] > prev["start"]:
            prev["end"] = nxt["start"]
        else:
            nxt["start"] = min(prev["end"], nxt["end"])


def to_result(decoded) -> Dict:
    """openai-whisper's result shape from a WhisperDecodedResults: language
    code and segments with start, end and text, timed by their words when the
    pipeline returned them (``times_from_words``). No no-speech or log-prob
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
    words = getattr(decoded, "words", None)
    if words:
        try:
            times_from_words(segments, list(words))
        except Exception as e:  # noqa: BLE001 - Whisper's own times still stand
            print(f"whisper_openvino: word times not applied: {type(e).__name__}: {e}")
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
