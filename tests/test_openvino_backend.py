"""The OpenVINO backend, and the reasoning it must not publish as a paragraph.

This backend exists because llama.cpp runs a vision tower on the CPU whatever
GPU backend it is given: measured on an Arc A750, 41 tok/s against 4.9 with a
picture in the context, and an image encode of about a second against
forty-six. `docs/INTEL-GPU.md` has the figures.

What is delicate is not the speed but the shape of what comes back. Ollama
hands a thinking model's reasoning over in a field of its own, so a caller that
reads `response` can never accidentally publish it. OpenVINO returns one
string, reasoning first, and the terminator is asymmetric - a closing
`</think>` with no opening tag, because the chat template opens the block in
the prompt. So `<think>.*?</think>` matches nothing, strips nothing, and puts
three thousand characters of a model talking to itself about rule four on the
page as the description of a clip. That is the failure these tests exist for.

Nothing here needs OpenVINO installed. The pipeline is faked, because what is
under test is this module's handling of what comes back rather than anyone
else's inference.
"""

from __future__ import annotations

import sys
import types

import pytest

from llm.llm_module import THINK_CLOSE, _OpenVINOBackend


class _FakePipeline:
    """Streams scripted chunks the way VLMPipeline streams real ones."""

    script: list = []

    def __init__(self, *args, **kwargs):
        self.chunks = list(_FakePipeline.script)
        self.configs = []

    def generate(self, prompt, **kwargs):
        self.configs.append(kwargs.get("generation_config"))
        streamer = kwargs.get("streamer")
        for chunk in self.chunks:
            if streamer and streamer(chunk):
                break
        return "unused: this module reads the stream, not the return value"


class _FakeConfig:
    def __init__(self):
        self.do_sample = False
        self.temperature = 1.0
        self.rng_seed = 0
        # The real default is uint64 max, i.e. unbounded. Anything that caps a
        # thinking model has to show up as a change from it.
        self.max_new_tokens = 18446744073709551615


def _install(monkeypatch, chunks):
    _FakePipeline.script = chunks
    stub = types.ModuleType("openvino_genai")
    stub.VLMPipeline = _FakePipeline
    stub.GenerationConfig = _FakeConfig
    monkeypatch.setitem(sys.modules, "openvino_genai", stub)


def _backend(monkeypatch, tmp_path, chunks, thinks):
    _install(monkeypatch, chunks)
    if thinks:
        (tmp_path / "chat_template.jinja").write_text(
            "{% if enable_thinking %}<think>{{ x }}</think>{% endif %}",
            encoding="utf-8")
    backend = _OpenVINOBackend(model_path=str(tmp_path), device="GPU")
    backend.load()
    return backend


REASONING = "Got it, let us tackle this. Rule 2 says no numbers. "
ANSWER = "A person walks across a room as the light shifts."


class TestSplittingReasoningFromTheAnswer:
    def test_only_what_follows_the_close_tag_is_the_paragraph(self, monkeypatch, tmp_path):
        b = _backend(monkeypatch, tmp_path,
                     [REASONING, THINK_CLOSE, "\n\n", ANSWER], thinks=True)
        assert b.generate("describe this") == ANSWER

    def test_the_reasoning_never_appears_in_the_paragraph(self, monkeypatch, tmp_path):
        """The whole point: that text is the model talking to itself."""
        b = _backend(monkeypatch, tmp_path,
                     [REASONING, THINK_CLOSE, ANSWER], thinks=True)
        assert "Rule 2" not in b.generate("describe this")

    def test_a_close_tag_inside_the_reasoning_does_not_split_early(self, monkeypatch, tmp_path):
        """Split on the last close, not the first."""
        b = _backend(monkeypatch, tmp_path,
                     ["thinking about " + THINK_CLOSE + " as a string, ",
                      "still thinking", THINK_CLOSE, ANSWER], thinks=True)
        assert b.generate("describe this") == ANSWER

    def test_a_model_that_does_not_think_returns_everything(self, monkeypatch, tmp_path):
        b = _backend(monkeypatch, tmp_path, [ANSWER], thinks=False)
        assert b.generate("describe this") == ANSWER

    def test_reasoning_with_no_answer_after_it_is_an_error(self, monkeypatch, tmp_path):
        """Better than returning empty: the caller would log a clip with no
        description and carry on, which is what this whole exercise began as."""
        b = _backend(monkeypatch, tmp_path, [REASONING], thinks=True)
        with pytest.raises(RuntimeError) as caught:
            b.generate("describe this")
        assert "without an answer" in str(caught.value)


class TestWhatReachesTheStreamCallback:
    def test_the_reasoning_is_not_streamed(self, monkeypatch, tmp_path):
        seen = []
        b = _backend(monkeypatch, tmp_path,
                     [REASONING, THINK_CLOSE, ANSWER], thinks=True)
        b.generate("describe this", stream_callback=seen.append)
        assert "Rule 2" not in "".join(seen)
        assert ANSWER in "".join(seen)

    def test_an_answer_split_across_chunks_arrives_whole(self, monkeypatch, tmp_path):
        seen = []
        b = _backend(monkeypatch, tmp_path,
                     [REASONING, THINK_CLOSE + " A person walks ",
                      "across a room."], thinks=True)
        b.generate("describe this", stream_callback=seen.append)
        assert "".join(seen).strip() == "A person walks across a room."

    def test_a_model_that_does_not_think_streams_from_the_first_chunk(self, monkeypatch, tmp_path):
        seen = []
        b = _backend(monkeypatch, tmp_path, ["A person ", "walks."], thinks=False)
        b.generate("describe this", stream_callback=seen.append)
        assert "".join(seen) == "A person walks."


class TestTheBudget:
    def test_a_thinking_model_is_left_uncapped(self, monkeypatch, tmp_path):
        """Same decision as the Ollama backend: a cap is spent on the reasoning
        first and the answer never arrives."""
        b = _backend(monkeypatch, tmp_path,
                     [REASONING, THINK_CLOSE, ANSWER], thinks=True)
        b.generate("describe this", max_tokens=200)
        assert b._pipe.configs[-1].max_new_tokens == _FakeConfig().max_new_tokens

    def test_a_model_that_does_not_think_honours_the_caller(self, monkeypatch, tmp_path):
        b = _backend(monkeypatch, tmp_path, [ANSWER], thinks=False)
        b.generate("describe this", max_tokens=200)
        assert b._pipe.configs[-1].max_new_tokens == 200


class TestLoading:
    def test_thinking_is_read_off_the_chat_template(self, monkeypatch, tmp_path):
        """Deterministic, and costs nothing - a wasted call was the alternative.

        Separate directories on purpose: a template written for the first case
        would otherwise still be sitting there for the second, and the test
        would pass whatever the code did.
        """
        thinking = tmp_path / "thinking"
        plain = tmp_path / "plain"
        thinking.mkdir()
        plain.mkdir()
        assert _backend(monkeypatch, thinking, [ANSWER], thinks=True)._thinks
        assert not _backend(monkeypatch, plain, [ANSWER], thinks=False)._thinks

    def test_a_missing_model_directory_says_what_kind_of_model_is_wanted(self, monkeypatch, tmp_path):
        """IR, not GGUF and not an Ollama tag. Obtaining one is a setup step, so
        this error is the only place a user finds that out."""
        _install(monkeypatch, [ANSWER])
        b = _OpenVINOBackend(model_path=str(tmp_path / "absent"))
        with pytest.raises(FileNotFoundError) as caught:
            b.load()
        assert "OpenVINO IR" in str(caught.value)

    def test_generating_before_loading_is_an_error(self, monkeypatch, tmp_path):
        _install(monkeypatch, [ANSWER])
        with pytest.raises(RuntimeError):
            _OpenVINOBackend(model_path=str(tmp_path)).generate("x")


class TestCancellation:
    def test_a_cancelled_run_stops_streaming(self, monkeypatch, tmp_path):
        class _Token:
            is_cancelled = True

        b = _backend(monkeypatch, tmp_path, ["one", "two", "three"], thinks=False)
        seen = []
        with pytest.raises(Exception):
            b.generate("describe this", stream_callback=seen.append,
                       cancellation_token=_Token())
        assert len(seen) <= 1


class TestWhatATextOnlyCallMustNotNeed:
    """Pillow and numpy turn a picture into a tensor and are needed for nothing
    else here. Importing them before checking whether there is a picture failed
    ten tests on CI, which installs neither - all of them tests that passed no
    image at all. The same shape as importing `requests` in order to patch it.
    """

    def test_no_images_means_no_imaging_libraries(self, monkeypatch, tmp_path):
        import sys as _sys
        for absent in ("PIL", "PIL.Image", "numpy"):
            monkeypatch.setitem(_sys.modules, absent, None)
        b = _backend(monkeypatch, tmp_path, [ANSWER], thinks=False)
        assert b.generate("describe this") == ANSWER

    def test_an_empty_image_list_is_the_same_as_none(self, monkeypatch, tmp_path):
        import sys as _sys
        monkeypatch.setitem(_sys.modules, "PIL", None)
        b = _backend(monkeypatch, tmp_path, [ANSWER], thinks=False)
        assert b.generate("describe this", images=[]) == ANSWER

    def test_a_list_of_empty_strings_is_too(self, monkeypatch, tmp_path):
        """`_read_one` filters these out, but nothing here should depend on it."""
        import sys as _sys
        monkeypatch.setitem(_sys.modules, "PIL", None)
        b = _backend(monkeypatch, tmp_path, [ANSWER], thinks=False)
        assert b.generate("describe this", images=["", None]) == ANSWER
