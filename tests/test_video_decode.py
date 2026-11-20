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
    # OpenCV used to say yes to the card here; the codec check now says no.
    assert decoder in ("cpu (no GPU decoder for MJPG)", "cpu (no GPU decoder for this file)")
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


# --- what OpenCV cannot tell us ------------------------------------------------
# It reports the hardware decoder it was asked for even when FFmpeg falls back
# to software (seen on a 4320x2160 H.264 film: "decoded on gpu", card at 1 %).

@pytest.mark.parametrize("codec,w,h,why", [
    ("h264", 4320, 2160, "H.264 4320x2160 is larger than GPU decoders take (4096)"),
    ("avc1", 2160, 4320, "H.264 2160x4320 is larger than GPU decoders take (4096)"),
    ("H264", 4096, 2048, None),
    ("hevc", 8192, 4096, None),
    ("hev1", 5760, 2880, None),
    ("MJPG", 640, 480, "no GPU decoder for MJPG"),
    ("", 9000, 9000, None),
])
def test_what_the_cards_will_not_decode(codec, w, h, why):
    assert video_decode.gpu_refusal(codec, w, h) == why


def test_fourcc_is_read_as_text():
    n = ord("h") | ord("2") << 8 | ord("6") << 16 | ord("4") << 24
    assert video_decode.fourcc_name(float(n)) == "h264"
    assert video_decode.fourcc_name(0) == ""


class _FakeCap:
    def __init__(self, props, opened=True):
        self.props, self.opened, self.released = props, opened, False

    def isOpened(self):  # noqa: N802 - OpenCV's name
        return self.opened

    def get(self, prop):
        return self.props.get(prop, 0)

    def release(self):
        self.released = True


def _fake_cv2(monkeypatch, codec, w, h):
    import types
    n = sum(ord(c) << (8 * i) for i, c in enumerate(codec))
    cv2 = types.SimpleNamespace(CAP_FFMPEG=1900, CAP_PROP_HW_ACCELERATION=50,
                                VIDEO_ACCELERATION_ANY=1, CAP_PROP_FOURCC=6,
                                CAP_PROP_FRAME_WIDTH=3, CAP_PROP_FRAME_HEIGHT=4, error=Exception)
    opened = []

    def capture(path, *args):
        cap = (_FakeCap({50: 2.0, 6: float(n), 3: w, 4: h}) if args else _FakeCap({}))
        opened.append((args != (), cap))
        return cap

    cv2.VideoCapture = capture
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    return opened


def test_a_film_the_card_would_decode_in_software_is_read_on_the_processor_and_says_why(
        monkeypatch):
    opened = _fake_cv2(monkeypatch, "h264", 4320, 2160)
    cap, decoder = video_decode.open_sparse("film.mp4", "gpu")
    assert decoder == "cpu (H.264 4320x2160 is larger than GPU decoders take (4096))"
    assert opened[0][0] and opened[0][1].released        # the hardware try was let go
    assert not opened[1][0] and cap is opened[1][1]       # a plain capture is returned


def test_a_film_the_card_takes_stays_on_the_card(monkeypatch):
    opened = _fake_cv2(monkeypatch, "hevc", 2160, 1080)
    cap, decoder = video_decode.open_sparse("proxy.mp4", "gpu")
    assert decoder == "gpu" and cap is opened[0][1] and len(opened) == 1


def test_the_pool_worker_judges_files_the_same_way():
    clip_worker = pytest.importorskip("modules.cluster.clip_worker")   # Pro only
    for args in (("h264", 4320, 2160), ("hevc", 8192, 4096), ("MJPG", 64, 48), ("avc1", 4096, 4096)):
        assert clip_worker.gpu_refusal(*args) == video_decode.gpu_refusal(*args)
