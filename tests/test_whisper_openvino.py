"""Whisper on an Intel GPU through OpenVINO GenAI.

When the route is taken, what the transcript gets back, that a cancel still
lands promptly, and that any failure, at load or part-way through, leaves the
run on the processor rather than without a transcript.
"""

from __future__ import annotations

import sys
import threading
import time
import types

import numpy as np
import pytest

from modules.audio import transcript as tr
from modules.audio import whisper_openvino as wov
from modules.system import compute_backend as cb


@pytest.fixture(autouse=True)
def _route_on(monkeypatch):
    monkeypatch.delenv(wov.ENV_VAR, raising=False)


# ── when ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("backend,cuda,expected", [
    (None, False, True),          # automatic, no NVIDIA card: the Arc
    (cb.AUTO, False, True),
    (None, True, False),          # automatic with CUDA: openai-whisper there
    (cb.INTEL, True, True),       # asked for by name
    (cb.INTEL, False, True),
    (cb.CPU, False, False),
    (cb.DIRECTML, False, False),
    (cb.CUDA, False, False),
])
def test_wanted(backend, cuda, expected):
    assert wov.wanted(backend, cuda) is expected


def test_env_turns_it_off(monkeypatch):
    monkeypatch.setenv(wov.ENV_VAR, "0")
    assert wov.wanted(cb.INTEL, False) is False


def test_every_ui_size_has_an_export():
    for name in ("tiny", "base", "small", "medium", "large"):
        assert wov.REPOS[name].startswith("OpenVINO/whisper-")


def test_unknown_size_has_no_model(tmp_path, monkeypatch):
    monkeypatch.setattr(wov, "models_root", lambda: str(tmp_path))
    assert wov.model_dir("enormous", download=False) is None


def test_an_incomplete_download_is_not_a_model(tmp_path, monkeypatch):
    monkeypatch.setattr(wov, "models_root", lambda: str(tmp_path))
    folder = tmp_path / "whisper-small-fp16-ov"
    folder.mkdir()
    (folder / "openvino_encoder_model.xml").write_text("x")
    assert wov.model_dir("small", download=False) is None
    for name in wov.REQUIRED_FILES:
        (folder / name).write_text("x")
    assert wov.model_dir("small", download=False) == str(folder)


def test_load_gives_up_quietly_without_the_package(monkeypatch):
    monkeypatch.setattr(wov, "genai_available", lambda: False)
    lines = []
    assert wov.load("small", None, cuda=False, log_fn=lines.append) is None
    assert lines == []                       # automatic: nothing to say
    assert wov.load("small", cb.INTEL, cuda=False, log_fn=lines.append) is None
    assert any("openvino-genai" in ln for ln in lines)


# ── what comes back ──────────────────────────────────────────────────────

def _chunk(text, start, end):
    return types.SimpleNamespace(text=text, start_ts=start, end_ts=end)


def test_to_result():
    decoded = types.SimpleNamespace(language="<|en|>", chunks=[
        _chunk(" Hello there.", 0.0, 1.5), _chunk("   ", 1.5, 2.0),
        _chunk("Open end", 28.0, -1.0)])
    out = wov.to_result(decoded)
    assert out["language"] == "en"
    assert out["segments"] == [
        {"start": 0.0, "end": 1.5, "text": "Hello there."},
        {"start": 28.0, "end": 28.0, "text": "Open end"}]


class _Pipe:
    """Stands in for openvino_genai.WhisperPipeline."""

    def __init__(self, delay=0.0, fail=False):
        self.delay, self.fail, self.calls = delay, fail, []

    def generate(self, samples, **kwargs):
        self.calls.append(kwargs)
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("device lost")
        return types.SimpleNamespace(language="en", chunks=[_chunk("a real sentence here", 0.0, 2.0)])


def _model(pipe):
    m = wov.OpenVINOWhisper.__new__(wov.OpenVINOWhisper)
    m._pipe, m.device, m.folder, m._speed = pipe, "GPU", "x", 30.0
    m._lock = threading.Lock()
    return m


def test_language_is_passed_as_a_token_and_auto_is_left_out():
    pipe = _Pipe()
    m = _model(pipe)
    m.transcribe(np.zeros(16000, np.float32), language="pl")
    m.transcribe(np.zeros(16000, np.float32), language="auto")
    assert pipe.calls[0]["language"] == "<|pl|>"
    assert "language" not in pipe.calls[1]
    assert all(c["return_timestamps"] for c in pipe.calls)


def test_a_sliver_of_audio_is_not_sent_to_the_card():
    pipe = _Pipe(fail=True)          # would raise, as the real one does
    out = _model(pipe).transcribe(np.zeros(16, np.float32), language="en")
    assert out == {"language": "en", "segments": []}
    assert pipe.calls == []


def test_cancel_returns_while_the_card_is_still_busy():
    m = _model(_Pipe(delay=2.0))
    t = time.time()
    with pytest.raises(wov.Cancelled):
        m.transcribe(np.zeros(16000 * 60, np.float32), should_cancel=lambda: True, poll=0.02)
    assert time.time() - t < 1.0


def test_the_bar_moves_while_a_chunk_decodes():
    seen = []
    _model(_Pipe(delay=0.3)).transcribe(np.zeros(16000 * 60, np.float32),
                                        on_frac=seen.append, poll=0.02)
    assert len(seen) > 3 and seen[-1] == 1.0 and seen == sorted(seen)


# ── inside a transcript ──────────────────────────────────────────────────

@pytest.fixture
def run(monkeypatch, tmp_path):
    """get_transcript_segments over two chunk files, with the Intel route
    handed a stand-in pipeline and openai-whisper a stand-in model."""
    state = {"cpu_loads": 0}

    def fake_split(video_file, chunk_length=600, should_cancel=None):
        paths = []
        for i in range(2):
            p = tmp_path / f"movie_chunk_{i:03d}.wav"
            p.write_bytes(b"RIFF")
            paths.append(str(p))
        return paths

    class CpuModel:
        def transcribe(self, chunk, **params):
            state.setdefault("cpu_chunks", []).append(chunk)
            return {"language": "en", "segments": [
                {"start": 0.0, "end": 2.0, "text": "said on the processor"}]}

    def load_model(*a, **k):
        state["cpu_loads"] += 1
        return CpuModel()

    mod = types.ModuleType("whisper.transcribe")
    mod.tqdm = None
    monkeypatch.setitem(sys.modules, "whisper.transcribe", mod)
    monkeypatch.setattr(tr, "split_audio", fake_split)
    monkeypatch.setattr(tr.torch.cuda, "is_available", lambda: False, raising=False)
    monkeypatch.setattr(tr.whisper, "load_model", load_model, raising=False)
    monkeypatch.setattr(tr.whisper, "load_audio", lambda p: np.zeros(16000 * 4, np.float32),
                        raising=False)

    def go(pipe=None, load_fails=False, **kwargs):
        def fake_load(model_name, backend, cuda, log_fn=print):
            if load_fails or pipe is None:
                return None
            return _model(pipe)
        monkeypatch.setattr(wov, "load", fake_load)
        lines = []
        segs = tr.get_transcript_segments(str(tmp_path / "movie.mp4"),
                                          log_fn=lines.append, **kwargs)
        return segs, lines

    state["go"] = go
    return state


def test_the_intel_route_needs_no_openai_whisper(run):
    pipe = _Pipe()
    segs, lines = run["go"](pipe)
    assert len(pipe.calls) == 2
    assert run["cpu_loads"] == 0
    assert [s["text"] for s in segs] == ["a real sentence here"] * 2
    assert any("Intel GPU" in ln for ln in lines)


def test_chunk_offsets_are_kept(run):
    segs, _ = run["go"](_Pipe(), chunk_length=600)
    assert segs[0]["start"] == 0.0
    assert any(s["start"] == 600.0 for s in segs)


def test_without_the_route_it_is_the_processor(run):
    segs, lines = run["go"](None)
    assert run["cpu_loads"] == 1
    assert {s["text"] for s in segs} == {"said on the processor"}


def test_a_failure_part_way_finishes_on_the_processor(run):
    segs, lines = run["go"](_Pipe(fail=True))
    assert run["cpu_loads"] == 1
    assert len(run["cpu_chunks"]) == 2
    assert any("failed on the Intel GPU" in ln for ln in lines)
    assert {s["text"] for s in segs} == {"said on the processor"}


def test_cancel_on_the_intel_route(run):
    pipe = _Pipe(delay=1.0)
    t = time.time()
    with pytest.raises(tr.TranscriptionCancelled):
        run["go"](pipe, should_cancel=lambda: len(pipe.calls) > 0)
    assert time.time() - t < 1.0
