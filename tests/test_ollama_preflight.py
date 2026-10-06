"""A run says at the start what Ollama will not be able to do at the end.

Narration and subtitle translation both need Ollama and both run last. Without
it, narration was skipped with one line an hour into the log, and subtitles were
written in the spoken language under the target language's name, with the
reason only in the debug log. These pin the checks that now run first.
"""
from __future__ import annotations

import subprocess

import pytest

from modules.audio import transcript_srt as srt
from modules.narration import llm_discovery, story_run


# --- subtitle translation ---------------------------------------------------

def test_no_ollama_means_untranslated_subtitles(monkeypatch):
    monkeypatch.setattr(srt, "_ollama_list", lambda: None)
    problem = srt.translation_problem()
    assert "not running" in problem and "untranslated" in problem


def test_a_missing_model_is_named_with_the_command_that_fetches_it(monkeypatch):
    # `ollama run` would pull it mid-run, inside a 120 s timeout per batch.
    monkeypatch.setattr(srt, "_ollama_list", lambda: ["mistral:latest"])
    assert "ollama pull llama3" in srt.translation_problem()


def test_a_pulled_model_is_fine_with_or_without_its_tag(monkeypatch):
    monkeypatch.setattr(srt, "_ollama_list", lambda: ["llama3:latest"])
    assert srt.translation_problem() is None
    assert srt.get_llm_translator() == "ollama"


def test_ollama_list_is_read_past_its_header(monkeypatch):
    out = ("NAME             ID              SIZE      MODIFIED\n"
           "llama3:latest    365c0bd3c000    4.7 GB    2 days ago\n"
           "qwen2.5vl:7b     5ced39dfa4ba    6.0 GB    3 weeks ago\n")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k:
                        subprocess.CompletedProcess(a, 0, stdout=out, stderr=""))
    assert srt._ollama_list() == ["llama3:latest", "qwen2.5vl:7b"]


def test_ollama_not_installed(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("ollama")
    monkeypatch.setattr(subprocess, "run", missing)
    assert srt._ollama_list() is None


def test_untranslatable_segments_come_back_as_the_same_list(monkeypatch):
    # The pipeline tells "not translated" apart by identity.
    monkeypatch.setattr(srt, "_ollama_list", lambda: None)
    segments = [{"start": 0.0, "end": 1.0, "text": "hello"}]
    assert srt.translate_segments(segments, "en", "pl") is segments


# --- narration --------------------------------------------------------------

def _models(monkeypatch, names):
    monkeypatch.setattr(llm_discovery, "ollama_models", lambda *a, **k: list(names))


def test_narration_without_ollama(monkeypatch):
    _models(monkeypatch, [])
    (problem,) = story_run.preflight({})
    assert "Ollama is not running" in problem and "Highlight Report" in problem


def test_narration_with_its_model_pulled(monkeypatch):
    _models(monkeypatch, ["qwen2.5vl:7b"])
    config = {"narration_model": {"backend": "ollama", "model": "qwen2.5vl:7b"}}
    assert story_run.preflight(config) == []


def test_narration_model_not_pulled(monkeypatch):
    _models(monkeypatch, ["mistral:latest"])
    config = {"narration_model": {"backend": "ollama", "model": "qwen2.5vl:7b"}}
    (problem,) = story_run.preflight(config)
    assert "ollama pull qwen2.5vl:7b" in problem


def test_clip_descriptions_need_a_model_that_sees(monkeypatch):
    _models(monkeypatch, ["llama3:latest"])
    (problem,) = story_run.preflight({})
    assert "no vision half" in problem
    assert story_run.preflight({"narrate_clips": False}) == []


@pytest.mark.parametrize("config", [
    {"write_highlight_report": False},
    {"narrate_chapters": False, "narrate_clips": False},
])
def test_no_warning_when_nothing_would_be_narrated(monkeypatch, config):
    _models(monkeypatch, [])
    assert story_run.preflight(config) == []


def test_a_missing_gguf_is_named(monkeypatch, tmp_path):
    config = {"narration_model": {"backend": "llama-cpp",
                                  "model": str(tmp_path / "gone.gguf")}}
    (problem,) = story_run.preflight(config)
    assert "gone.gguf" in problem


def test_a_probe_that_fails_says_nothing(monkeypatch):
    def broken(*a, **k):
        raise OSError("no network stack")
    monkeypatch.setattr(llm_discovery, "ollama_models", broken)
    assert story_run.preflight({}) == []


# --- the run log ------------------------------------------------------------

def test_the_run_states_both_at_the_start(monkeypatch):
    import pipeline

    _models(monkeypatch, [])
    monkeypatch.setattr(srt, "_ollama_list", lambda: None)
    config = {"create_subtitles": True, "use_transcript": True,
              "transcript_source_lang": "en", "target_lang": "pl"}
    problems = pipeline._ollama_preflight(config)
    assert any(p.startswith("Narration is on") for p in problems)
    assert any(p.startswith("Subtitles: Ollama is not running") for p in problems)


@pytest.mark.parametrize("config", [
    {"create_subtitles": True, "use_transcript": True,
     "transcript_source_lang": "pl", "target_lang": "pl"},
    {"create_subtitles": False, "use_transcript": True, "target_lang": "pl"},
    {"create_subtitles": True, "use_transcript": True, "target_lang": None},
])
def test_subtitles_in_the_spoken_language_need_no_ollama(config):
    import pipeline

    assert pipeline._wants_translated_subtitles(config) is None
