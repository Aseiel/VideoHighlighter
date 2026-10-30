"""SigLIP2 for visual search, as an experiment next to CLIP ("SigLIP2 only").

The same encoder that scores actions (modules/vision/frame_encoder.py, vision
and text halves) stands in for CLIP ViT-B/32 behind the interface the search
worker already uses (`llm.clip_index.ClipEmbedder`): one vector per frame,
memoised per video, and the query scored the way CLIP's is -- a softmax over
the prompt and the same generic negatives, at SigLIP2's own logit scale. Only
the encoder differs, so the two engines can be compared on the same video.

Nothing to download: it needs the installed frame encoder with its text half.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from llm.clip_index import l2_normalize
from llm.clip_prefilter import ClipFramePrefilter


class SiglipEmbedder:
    """Duck-types `ClipEmbedder` for the search worker."""

    DEFAULT_NEGATIVES = ClipFramePrefilter.DEFAULT_NEGATIVES

    def __init__(self, device: str = "AUTO"):
        from modules.vision import frame_encoder

        # `device` is CLIP's setting; the frame encoder follows compute.backend.
        self.model_id = frame_encoder.ENCODER_ID
        self.device = device
        self.dims = frame_encoder.DIMS
        self._vision = None
        self._text = None
        self._scale = 100.0
        self._labels: Optional[list[str]] = None

    @staticmethod
    def import_error(device: str = "AUTO") -> Optional[str]:
        """None if the frame encoder and its text half are installed, else why
        not -- in words the windowed exe can show."""
        try:
            from modules.vision import frame_encoder
        except Exception as e:  # noqa: BLE001
            return f"frame encoder import failed: {type(e).__name__}: {e}"
        folder = frame_encoder.find_model_dir()
        if folder is None:
            return (f"the {frame_encoder.ENCODER_ID} encoder is not installed "
                    f"(it comes with the action recognition pack)")
        if not frame_encoder.has_text(folder):
            return "the installed encoder has no text half (reinstall the encoder pack)"
        return None

    def load(self):
        from modules.vision import frame_encoder

        folder = frame_encoder.find_model_dir()
        self._vision = frame_encoder.load(model_dir=folder)
        if self._vision is None:
            raise RuntimeError("SigLIP2 vision encoder could not be loaded (see debug log)")
        self._text = frame_encoder.load_text(model_dir=folder)
        if self._text is None:
            raise RuntimeError("SigLIP2 text encoder could not be loaded (see debug log)")
        self._scale = frame_encoder.logit_scale(folder)
        self.device = self._vision.label
        print(f"🔧 SigLIP2 search: images on {self._vision.label}, "
              f"text on {self._text.label}, logit scale {self._scale:.1f}")

    @property
    def logit_scale(self) -> float:
        return self._scale

    def set_query(self, target: str, positive_template: str = "a photo of {}",
                  negatives: Optional[list[str]] = None):
        """Same prompt and negatives as CLIP's `set_query`, for a fair comparison."""
        positive = positive_template.format(target.strip())
        negs = list(negatives) if negatives else list(self.DEFAULT_NEGATIVES)
        self._labels = [positive] + negs

    def embed_query_labels(self) -> np.ndarray:
        if self._labels is None:
            raise RuntimeError("Call set_query() first.")
        return self.embed_texts(self._labels)

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Unit vectors for prompts, each as written (no template ensemble)."""
        if self._text is None:
            raise RuntimeError("Call load() first.")
        return l2_normalize(self._text._raw(list(texts)).astype(np.float32))

    def embed_frames_bgr(self, frames_bgr: Sequence) -> np.ndarray:
        if self._vision is None:
            raise RuntimeError("Call load() first.")
        return l2_normalize(self._vision.encode_bgr(list(frames_bgr)))
