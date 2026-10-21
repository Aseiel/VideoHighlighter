"""CLIP search in the user's own language.

CLIP's text tower (openai/clip-vit-base-patch32) was trained on English. A
query typed in Chinese, Japanese or Polish comes out as a vector that matches
nothing in particular, so the ranking is close to random and nothing says why.

sentence-transformers' clip-ViT-B-32-multilingual-v1 (Apache-2.0) is a text
model trained to put a sentence in any of 50+ languages where CLIP ViT-B/32's
own text tower puts its English translation. Its vectors therefore land in the
same space as the image embeddings the app already stores: the frame index,
the cache and the image tower are unchanged, and only the query is encoded
differently.

Which encoder a query gets (``wanted``): this one whenever the interface is in
a language other than English, or the text has letters outside ASCII; CLIP's
own otherwise, because for English queries the original tower ranks a little
better than its multilingual student. When this model is not installed, the
query goes to CLIP's tower as before, and the log says once that it would rank
better with the multilingual model.

The model runs on ONNX Runtime's processor route. A query is a few words
encoded once, so the GPU would buy nothing, and every build ships ONNX Runtime.
Made by ``tools/export_clip_multilingual.py``; delivered as the
``models-clip-multilingual`` pack (bundled in the mac build).
"""
from __future__ import annotations

import json
import os
import sys
import threading
from typing import Optional, Sequence

import numpy as np

SOURCE_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
SOURCE_REVISION = "58edf8cada9e398793dca955574a48cbb7f18be2"
MODEL_DIRNAME = "clip-vit-b32-multilingual"
MODEL_FILE = "text.onnx"
TOKENIZER_FILE = "tokenizer.json"
META_FILE = "encoder.json"
FORMAT = 1

MAX_LENGTH = 128          # the source model's max_seq_length
DIMS = 512                # CLIP ViT-B/32's embedding size
PAD_ID = 0
PROBE_TEXT = "a photo of a dog"
PROBE_MIN_COSINE = 0.99

DIR_ENV = "VH_CLIP_MULTILINGUAL_DIR"


def _per_user_dir() -> str:
    """Same rule as modules/packs/pack_manager.per_user_dir."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "VideoHighlighter")


def model_dir_candidates() -> list:
    """Every folder the model may be in, most specific first: the override,
    beside the exe, the per-user folder (where a pack goes when the exe's
    folder is read-only), the bundle, and the source checkout."""
    out = []
    override = os.environ.get(DIR_ENV, "").strip()
    if override:
        out.append(override)
    bases = []
    if getattr(sys, "frozen", False):
        bases.append(os.path.dirname(sys.executable))
        bases.append(_per_user_dir())
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            bases.append(meipass)
    # llm/clip_multilingual.py -> the project root.
    bases.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return out + [os.path.join(b, "models", MODEL_DIRNAME) for b in bases]


def find_model_dir() -> Optional[str]:
    for folder in model_dir_candidates():
        if all(os.path.isfile(os.path.join(folder, f))
               for f in (MODEL_FILE, TOKENIZER_FILE, META_FILE)):
            return folder
    return None


def is_installed() -> bool:
    return find_model_dir() is not None


def read_meta(folder: str) -> dict:
    with open(os.path.join(folder, META_FILE), encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("format") != FORMAT:
        raise ValueError(f"format {meta.get('format')!r}, this build reads {FORMAT}")
    if len(meta.get("probe") or []) != DIMS:
        raise ValueError("its probe vector is missing or the wrong size")
    return meta


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def make_tokenizer(path: str):
    """The source model's tokenizer as the app runs it: cased WordPiece, cut
    to MAX_LENGTH, padded to the longest text in the batch."""
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(path)
    tok.enable_truncation(MAX_LENGTH)
    tok.enable_padding(pad_id=PAD_ID)
    return tok


def tokenize(tok, texts: Sequence[str]) -> tuple:
    """Texts -> (input_ids, attention_mask), int64 [N, L]."""
    rows = tok.encode_batch([str(t) for t in texts])
    ids = np.array([r.ids for r in rows], dtype=np.int64)
    mask = np.array([r.attention_mask for r in rows], dtype=np.int64)
    return ids, mask


class MultilingualText:
    """The text model, loaded and checked. ``encode`` is the whole API."""

    def __init__(self, folder: str):
        from modules.system import ort_directml
        self.folder = folder
        self.meta = read_meta(folder)
        self._tok = make_tokenizer(os.path.join(folder, TOKENIZER_FILE))
        self._session = ort_directml.session(
            os.path.join(folder, MODEL_FILE),
            providers_override=[ort_directml.CPU_PROVIDER])
        probe = self.encode([PROBE_TEXT])[0]
        cos = float(probe @ _unit(self.meta["probe"]))
        if cos < PROBE_MIN_COSINE:
            raise RuntimeError(f"does not match the reference (cosine {cos:.4f})")

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Unit-norm vectors in CLIP ViT-B/32's space -> float32 [N, 512]."""
        if len(texts) == 0:
            return np.zeros((0, DIMS), np.float32)
        ids, mask = tokenize(self._tok, texts)
        out = self._session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        return _unit(out)


_lock = threading.Lock()
_loaded: Optional[MultilingualText] = None
_failed = False
_said_missing = False


def load(log=print) -> Optional[MultilingualText]:
    """The model, loaded once per process, or None when it is not installed
    or does not load. Never raises."""
    global _loaded, _failed
    with _lock:
        if _loaded is not None or _failed:
            return _loaded
        folder = find_model_dir()
        if folder is None:
            return None
        try:
            _loaded = MultilingualText(folder)
            log(f"✅ Multilingual search text: {SOURCE_MODEL} ({folder})")
        except Exception as e:  # noqa: BLE001 - CLIP's own tower is the answer
            _failed = True
            log(f"⚠️ Multilingual search text in {folder} is unusable: {e}")
        return _loaded


def _ui_language() -> str:
    try:
        from modules.i18n import current_language
        return current_language()
    except Exception:  # noqa: BLE001
        return "en"


def _non_ascii_letters(texts: Sequence[str]) -> bool:
    return any(ch.isalpha() and ord(ch) > 127 for t in texts for ch in str(t))


def wanted(texts: Sequence[str], ui_language: Optional[str] = None) -> bool:
    """True when these texts should go to the multilingual model: the
    interface is not in English, or the text itself is not plain ASCII."""
    lang = ui_language if ui_language is not None else _ui_language()
    return lang != "en" or _non_ascii_letters(texts)


def encode_if_wanted(texts: Sequence[str], log=print) -> Optional[np.ndarray]:
    """Unit vectors from the multilingual model when ``wanted`` says these texts
    need it and it is installed; None means "use CLIP's own text tower"."""
    global _said_missing
    if not texts or not wanted(texts):
        return None
    model = load(log)
    if model is None:
        if not _said_missing and not _failed:
            _said_missing = True
            log("ℹ️ Search understands English best: the multilingual search "
                "model is not installed, so queries in other languages rank "
                "poorly (it is offered under Packs / Downloads).")
        return None
    return model.encode(list(texts))


def reset_for_tests() -> None:
    global _loaded, _failed, _said_missing
    with _lock:
        _loaded, _failed, _said_missing = None, False, False


if __name__ == "__main__":  # pragma: no cover - manual check
    m = load()
    if m is None:
        sys.exit("not installed")
    v = m.encode(sys.argv[1:] or [PROBE_TEXT])
    print(v.shape, np.round(v[:, :6], 4))
