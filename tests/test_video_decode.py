"""The CLIP index's decoder setting, and the sparse read it drives."""
import sys

import numpy as np
import pytest

from modules.media import video_decode


def test_the_setting_reads_what_people_write():
    assert video_decode.normalise("GPU") == "gpu"
    assert video_decode.normalise("hardware") == "gpu"
    assert video_decode.normalise("processor") == "cpu"
    assert video_decode.normalise("banana") is None
    assert video_decode.from_config({"compute": {"video_decode": "cpu"}}) == "cpu"
    assert video_decode.from_config({"compute": {"backend": "auto"}}) is None
    assert video_decode.from_config(None) is None


def test_the_default_is_the_gpu_and_an_exported_variable_wins(monkeypatch):
    monkeypatch.delenv(video_decode.ENV_VAR, raising=False)
    assert video_decode.configured() == "gpu"
    assert video_decode.apply({"compute": {"video_decode": "cpu"}}) == "cpu"
    assert video_decode.configured() == "cpu"
    # Somebody launching with the variable set is testing something.
    monkeypatch.setenv(video_decode.ENV_VAR, "gpu")
    assert video_decode.apply({"compute": {"video_decode": "cpu"}}) is None
    assert video_decode.configured() == "gpu"


@pytest.fixture
def cv2_real(monkeypatch):
    from tests.conftest import real_opencv
    cv2 = real_opencv()                 # the suite shims cv2; this needs real frames
    if cv2 is None:
        pytest.skip("OpenCV not installed")
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    return cv2


def _write_video(cv2, path, frames=37):
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(frames):
        out.write(np.full((48, 64, 3), (i * 6) % 256, dtype=np.uint8))
    out.release()


def test_a_file_the_card_cannot_decode_falls_back_and_says_so(cv2_real, tmp_path):
    path = str(tmp_path / "clip.avi")
    _write_video(cv2_real, path)        # Motion JPEG: no video decoder takes it
    cap, decoder = video_decode.open_sparse(path, "gpu")
    assert cap.isOpened()
    assert decoder in ("gpu", "cpu (no GPU decoder for this file)")
    cap.release()
    cap, decoder = video_decode.open_sparse(path, "cpu")
    assert decoder == "cpu"
    cap.release()


def test_the_index_samples_the_same_grid_with_either_decoder(cv2_real, tmp_path):
    from llm import clip_index

    path = str(tmp_path / "clip.avi")
    _write_video(cv2_real, path)

    class Embedder:
        model_id = "test-model"
        logit_scale = 100.0

        def embed_frames_bgr(self, frames):
            # One number per frame - its brightness - so the test can see
            # exactly which frames were converted and embedded.
            v = np.zeros((len(frames), 512), dtype=np.float32)
            v[:, 0] = [f.mean() for f in frames]
            return v

    got = {}
    for decode in ("cpu", "gpu"):
        idx = clip_index.build_index(path, interval=1.0, embedder=Embedder(), decode=decode)
        got[decode] = idx
        assert np.allclose(idx.timestamps, [0.0, 1.0, 2.0, 3.0])     # frames 0, 10, 20, 30
        assert idx.meta["decoder"].startswith(("cpu", "gpu"))
    brightness = got["cpu"].embeddings[:, 0]
    assert np.allclose(brightness, [0, 60, 120, 180], atol=3)        # the sampled frames, not their neighbours
