"""Teach's two embedders: CLIP (words and examples) and the frame encoder
(examples only). Nothing is loaded: the frame encoder is replaced by a fake."""

from __future__ import annotations

import numpy as np
import pytest

from modules.teach import embed


class FakeEncoder:
    encoder_id = "fake"

    def encode_bgr(self, frames):
        return np.full((len(frames), 4), 3.0, np.float32)


def test_make_picks_by_name():
    assert isinstance(embed.make("clip"), embed.ClipBackend)
    assert isinstance(embed.make("frame-encoder"), embed.FrameEncoderBackend)
    with pytest.raises(ValueError):
        embed.make("other")


def test_frame_encoder_rows_are_unit_and_words_are_refused(monkeypatch):
    from modules.vision import frame_encoder
    monkeypatch.setattr(frame_encoder, "load", lambda backend=None, log=print: FakeEncoder())
    backend = embed.make("frame-encoder")
    assert backend.model_id == frame_encoder.ENCODER_ID
    rows = backend.images([np.zeros((8, 8, 3), np.uint8)] * 2)
    np.testing.assert_allclose(np.linalg.norm(rows, axis=1), 1.0, rtol=1e-6)
    with pytest.raises(RuntimeError, match="examples"):
        backend.texts(["anything"])


def test_frame_encoder_missing_says_so(monkeypatch):
    from modules.vision import frame_encoder
    monkeypatch.setattr(frame_encoder, "load", lambda backend=None, log=print: None)
    with pytest.raises(RuntimeError, match="not installed"):
        embed.make("frame-encoder").images([np.zeros((8, 8, 3), np.uint8)])
