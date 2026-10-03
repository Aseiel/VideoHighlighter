"""Samples and words as vectors, cached per project.

Everything that compares a sample with a class goes through an ``Embedder``:
``images(frames_bgr)`` and ``texts(strings)``, both returning unit rows. Two
real ones: the app's CLIP (``llm.clip_index.ClipEmbedder``: OpenVINO, or
torch on NVIDIA), and the frame encoder that taught action models are built
on (``modules.vision.frame_encoder``, SigLIP2), which sorts by examples
better but cannot read words until its text half ships. Tests hand in a
fake, which is why nothing above this module imports either.

A sample's vector is the mean of a few frames spread across it. One frame
would describe a single instant of a clip that is about motion; every frame
would cost a full decode for what four already say.
"""
from __future__ import annotations

import os
from typing import Callable, Optional, Protocol, Sequence

import numpy as np

CACHE_FILE = os.path.join("cache", "vectors.npz")


class Embedder(Protocol):
    model_id: str

    def images(self, frames_bgr: Sequence) -> np.ndarray: ...

    def texts(self, texts: Sequence[str]) -> np.ndarray: ...


class ClipBackend:
    """The app's CLIP, behind the ``Embedder`` interface. Loaded on first use."""

    def __init__(self, device: str = "AUTO"):
        from llm.clip_prefilter import MODEL_ID

        self._device = device
        self._clip = None
        # Known before loading: the vector cache is opened before the first
        # embedding, and a cache opened without a model name cannot tell that
        # its vectors came from another model.
        self.model_id = MODEL_ID

    def _load(self):
        if self._clip is None:
            from llm.clip_index import ClipEmbedder
            clip = ClipEmbedder(device=self._device)
            clip.load()
            self._clip = clip
            self.model_id = getattr(clip, "model_id", "") or self.model_id
        return self._clip

    def images(self, frames_bgr: Sequence) -> np.ndarray:
        out = []
        for start in range(0, len(frames_bgr), 16):
            out.append(self._load().embed_frames_bgr(list(frames_bgr[start:start + 16])))
        return np.concatenate(out) if out else np.zeros((0, 0), np.float32)

    def texts(self, texts: Sequence[str]) -> np.ndarray:
        return self._load().embed_texts(list(texts))


class FrameEncoderBackend:
    """The frame encoder behind the ``Embedder`` interface. Loaded on first use.

    Examples only: SigLIP2's text half is not part of the encoder download, so
    a class known only by its words needs CLIP. ``texts`` says so rather than
    returning vectors from another model's space.
    """

    def __init__(self, backend: Optional[str] = None):
        from modules.vision.frame_encoder import ENCODER_ID

        self._backend = backend
        self._encoder = None
        self.model_id = ENCODER_ID

    def _load(self):
        if self._encoder is None:
            from modules.vision import frame_encoder
            encoder = frame_encoder.load(self._backend)
            if encoder is None:
                raise RuntimeError("the frame encoder is not installed or cannot run here")
            self._encoder = encoder
        return self._encoder

    def images(self, frames_bgr: Sequence) -> np.ndarray:
        if len(frames_bgr) == 0:
            return np.zeros((0, 0), np.float32)
        return unit(self._load().encode_bgr(list(frames_bgr)))

    def texts(self, texts: Sequence[str]) -> np.ndarray:
        raise RuntimeError("the frame encoder sorts by examples; a class with no examples "
                           "yet needs CLIP (--embedder clip)")


EMBEDDERS = ("clip", "frame-encoder")


def make(name: str = "clip"):
    """The embedder called ``name`` (one of ``EMBEDDERS``)."""
    if name == "frame-encoder":
        return FrameEncoderBackend()
    if name == "clip":
        return ClipBackend()
    raise ValueError(f"unknown embedder {name!r}; one of {', '.join(EMBEDDERS)}")


def unit(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)


def read_frames(path: str, count: int, *, start: float = 0.0,
                duration: Optional[float] = None) -> list:
    """``count`` BGR frames spread evenly over a clip (or part of a file).

    Seeks by frame index rather than decoding everything: a five-second sample
    costs four seeks. Frames that cannot be read are skipped, so a damaged
    file yields fewer frames, not an exception.
    """
    import cv2

    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        first = int(start * fps)
        span = int((duration or 0) * fps) or max(total - first, 1)
        frames = []
        for k in range(count):
            index = first + int((k + 0.5) * span / count)
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
        return frames
    finally:
        cap.release()


class VectorCache:
    """Sample vectors on disk, so re-sorting after a new example is instant."""

    def __init__(self, root: str, model_id: str = ""):
        self.path = os.path.join(root, CACHE_FILE)
        self.model_id = model_id
        self._vectors: dict = {}
        self._dirty = False
        self._load()

    def _load(self):
        try:
            data = np.load(self.path, allow_pickle=False)
        except (OSError, ValueError):
            return
        stored = str(data["model_id"]) if "model_id" in data.files else ""
        if stored != self.model_id:
            # Another model's space, or unrecorded: a vector is only comparable
            # with vectors from the same model, so these are thrown away.
            return
        for key, vector in zip(data["ids"], data["vectors"]):
            self._vectors[str(key)] = vector

    def get(self, key: str) -> Optional[np.ndarray]:
        return self._vectors.get(key)

    def put(self, key: str, vector: np.ndarray) -> None:
        self._vectors[key] = np.asarray(vector, dtype=np.float32)
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        keys = sorted(self._vectors)
        tmp = self.path + ".tmp.npz"
        np.savez(tmp, ids=np.array(keys), model_id=np.array(self.model_id),
                 vectors=np.stack([self._vectors[k] for k in keys]) if keys
                 else np.zeros((0, 0), np.float32))
        os.replace(tmp, self.path)
        self._dirty = False


def sample_vectors(samples: Sequence, embedder: Embedder, cache: VectorCache,
                   frames_per_sample: int = 4,
                   frame_reader: Optional[Callable] = None,
                   progress: Optional[Callable] = None) -> dict:
    """``{sample id: unit vector}`` for every sample that has frames."""
    frame_reader = frame_reader or read_frames
    out = {}
    for i, sample in enumerate(samples):
        vector = cache.get(sample.id)
        if vector is None:
            frames = frame_reader(sample.path, frames_per_sample)
            if not frames:
                continue
            vector = unit(unit(embedder.images(frames)).mean(axis=0))
            cache.put(sample.id, vector)
        out[sample.id] = vector
        if progress:
            progress(i + 1, len(samples))
    cache.save()
    return out
