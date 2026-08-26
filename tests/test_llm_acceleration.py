"""Choosing the narration backend, and saying why in a sentence worth reading.

On Intel the two paths are far apart - measured on an Arc A750, 41 tok/s
against 8.85 with a picture in the context, and an image encode of about a
second against twelve - so the fast one should be taken without the user having
to know any of that.

The part these tests protect is the explaining rather than the choosing. The
fast path can be unavailable for four reasons and only one of them is a
problem: no Intel GPU is simply most machines, a missing `openvino-genai` is an
install, and a missing converted model is a setup step that costs an hour and
several gigabytes and is the one a user is most likely to have skipped without
knowing it was expected. A single "falling back to Ollama" would make all four
look the same, and the expensive one look like nothing.

Nothing here needs OpenVINO installed; the probes are faked, because what is
being tested is the decision rather than anyone else's inference.
"""

from __future__ import annotations

import pytest

from modules import llm_acceleration


@pytest.fixture
def ir_model(tmp_path):
    """A directory that looks like a converted model, because it has the file
    that only a converted model has."""
    def _make(name="Qwen3-VL-8B-Thinking-int4-ov"):
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        (d / llm_acceleration.IR_MARKER).write_text("<net/>", encoding="utf-8")
        return str(d)
    return _make


def _machine(monkeypatch, *, card=None, genai=False, search=()):
    monkeypatch.setattr(llm_acceleration, "intel_gpu", lambda: card)
    monkeypatch.setattr(llm_acceleration, "genai_available", lambda: genai)
    monkeypatch.setattr(llm_acceleration, "DEFAULT_MODEL_DIRS", tuple(search))


class TestWhenTheFastPathIsAvailable:
    def test_an_intel_machine_with_a_converted_model_uses_openvino(
            self, monkeypatch, ir_model, tmp_path):
        model = ir_model()
        _machine(monkeypatch, card="Intel(R) Arc(TM) A750 Graphics",
                 genai=True, search=(str(tmp_path),))
        got = llm_acceleration.decide(
            {}, {"backend": "ollama", "model": "qwen3-vl:8b"})
        assert got["backend"] == "openvino"
        assert got["model"] == model
        assert got["accelerated"] is True

    def test_the_configured_directory_wins_over_what_was_found(
            self, monkeypatch, ir_model, tmp_path):
        found = ir_model("some-other-model")
        wanted = ir_model("the-one-that-was-configured")
        _machine(monkeypatch, card="Intel Arc", genai=True, search=(str(tmp_path),))
        got = llm_acceleration.decide(
            {"openvino_model_dir": wanted}, {"backend": "ollama", "model": "qwen3-vl:8b"})
        assert got["model"] == wanted
        assert got["model"] != found

    def test_the_reason_names_the_card_and_the_model(
            self, monkeypatch, ir_model, tmp_path):
        ir_model()
        _machine(monkeypatch, card="Intel(R) Arc(TM) A750 Graphics",
                 genai=True, search=(str(tmp_path),))
        reason = llm_acceleration.decide(
            {}, {"backend": "ollama", "model": "qwen3-vl:8b"})["reason"]
        assert "A750" in reason
        assert "Qwen3-VL-8B-Thinking-int4-ov" in reason


class TestWhyTheFastPathWasNotTaken:
    """Four causes, four sentences. One generic fallback message would make the
    expensive one indistinguishable from the ordinary ones."""

    def test_no_intel_gpu_is_not_reported_as_a_problem(self, monkeypatch):
        _machine(monkeypatch, card=None, genai=True)
        got = llm_acceleration.decide({}, {})
        assert got["backend"] == "ollama"
        assert got["accelerated"] is False
        assert "No Intel GPU" in got["reason"]

    def test_a_missing_library_is_reported_as_an_install(self, monkeypatch):
        _machine(monkeypatch, card="Intel Arc A750", genai=False)
        reason = llm_acceleration.decide({}, {})["reason"]
        assert "openvino-genai is not installed" in reason
        assert "Intel Arc A750" in reason

    def test_a_missing_converted_model_says_it_is_a_conversion(self, monkeypatch, tmp_path):
        """The expensive case, and the one most likely to be silently skipped."""
        _machine(monkeypatch, card="Intel Arc A750", genai=True,
                 search=(str(tmp_path),))
        reason = llm_acceleration.decide({}, {})["reason"]
        assert "conversion" in reason
        assert "docs/INTEL-GPU.md" in reason

    def test_every_fallback_still_names_a_usable_backend(self, monkeypatch, tmp_path):
        for card, genai in ((None, True), ("Intel Arc", False), ("Intel Arc", True)):
            _machine(monkeypatch, card=card, genai=genai, search=(str(tmp_path),))
            got = llm_acceleration.decide({}, {})
            assert got["backend"] == "ollama"
            assert got["reason"]


class TestAnOllamaTagIsAChoiceOfModelNotOfEngine:
    """The first version of this treated `backend: ollama` as a deliberate
    rejection of OpenVINO. The model picker writes that for every Ollama tag,
    so the effect was that nobody was ever accelerated - the feature shipped
    and did nothing. A tag chooses a model; running the same model faster keeps
    the choice."""

    def test_an_ollama_tag_is_accelerated_when_the_same_model_is_converted(
            self, monkeypatch, ir_model, tmp_path):
        model = ir_model("Qwen3-VL-8B-Thinking-int4-ov")
        _machine(monkeypatch, card="Intel Arc A750", genai=True,
                 search=(str(tmp_path),))
        got = llm_acceleration.decide(
            {}, {"backend": "ollama", "model": "qwen3-vl:8b"})
        assert got["backend"] == "openvino"
        assert got["model"] == model

    def test_a_different_model_is_never_substituted_silently(
            self, monkeypatch, ir_model, tmp_path):
        """Changing the engine is a speed change. Changing the model is a
        change of answer, so it is offered rather than done."""
        ir_model("Qwen3-VL-8B-Thinking-int4-ov")
        _machine(monkeypatch, card="Intel Arc A750", genai=True,
                 search=(str(tmp_path),))
        got = llm_acceleration.decide(
            {}, {"backend": "ollama", "model": "qwen2.5vl:7b"})
        assert got["backend"] == "ollama"
        assert "qwen2.5vl:7b" in got["reason"]
        assert "Qwen3-VL-8B-Thinking-int4-ov" in got["reason"]

    def test_a_gguf_choice_is_left_alone(self, monkeypatch, ir_model, tmp_path):
        """Someone who went and found a file meant that file."""
        ir_model()
        _machine(monkeypatch, card="Intel Arc A750", genai=True,
                 search=(str(tmp_path),))
        got = llm_acceleration.decide(
            {}, {"backend": "llama-cpp", "model": "D:/models/a.gguf"})
        assert got["backend"] == "llama-cpp"
        assert got["accelerated"] is False

    def test_a_configured_directory_is_used_and_a_mismatch_is_declared(
            self, monkeypatch, ir_model, tmp_path):
        """Pointing at a directory is deliberate, so it wins - but if it is not
        the model the picker shows, the log has to say so."""
        wanted = ir_model("Qwen3-VL-8B-Thinking-int4-ov")
        _machine(monkeypatch, card="Intel Arc", genai=True, search=())
        got = llm_acceleration.decide(
            {"openvino_model_dir": wanted}, {"backend": "ollama", "model": "qwen2.5vl:7b"})
        assert got["backend"] == "openvino"
        assert got["model"] == wanted
        assert "not qwen2.5vl:7b" in got["reason"]

    def test_a_configured_directory_matching_the_tag_says_nothing_extra(
            self, monkeypatch, ir_model):
        wanted = ir_model("Qwen3-VL-8B-Thinking-int4-ov")
        _machine(monkeypatch, card="Intel Arc", genai=True, search=())
        got = llm_acceleration.decide(
            {"openvino_model_dir": wanted}, {"backend": "ollama", "model": "qwen3-vl:8b"})
        assert "not qwen3-vl:8b" not in got["reason"]

    def test_openvino_chosen_with_a_directory_that_is_not_one_says_so(
            self, monkeypatch, tmp_path):
        _machine(monkeypatch, card="Intel Arc", genai=True)
        got = llm_acceleration.decide(
            {}, {"backend": "openvino", "model": str(tmp_path / "not-a-model")})
        assert llm_acceleration.IR_MARKER in got["reason"]


class TestMatchingATagToAConvertedDirectory:
    def test_the_same_weights_in_two_formats_match(self):
        assert llm_acceleration.same_model(
            "qwen3-vl:8b", "D:/ov/Qwen3-VL-8B-Thinking-int4-ov")

    def test_a_different_model_does_not(self):
        assert not llm_acceleration.same_model(
            "qwen2.5vl:7b", "D:/ov/Qwen3-VL-8B-Thinking-int4-ov")

    def test_a_different_size_of_the_same_family_does_not(self):
        """4B and 8B are not interchangeable, and the names differ only there."""
        assert not llm_acceleration.same_model(
            "qwen3-vl:4b", "D:/ov/Qwen3-VL-8B-Instruct-int4-ov")

    def test_an_empty_tag_matches_nothing(self):
        assert not llm_acceleration.same_model("", "D:/ov/anything-ov")


class TestRecognisingAConvertedModel:
    def test_a_directory_without_the_marker_is_not_one(self, tmp_path):
        (tmp_path / "model.gguf").write_text("not IR", encoding="utf-8")
        assert not llm_acceleration._is_ir_dir(str(tmp_path))

    def test_a_missing_directory_is_not_one(self, tmp_path):
        assert not llm_acceleration._is_ir_dir(str(tmp_path / "absent"))

    def test_an_empty_path_is_not_one(self):
        assert not llm_acceleration._is_ir_dir("")

    def test_searching_a_directory_that_does_not_exist_is_quiet(
            self, monkeypatch, tmp_path):
        """Scanning happens at preflight on every run, including machines where
        none of these paths exist.

        The built-in search paths are cleared first: leaving them in makes this
        pass or fail on whether the machine running the suite happens to have
        converted a model, which is exactly the kind of test that goes green
        everywhere except CI.
        """
        monkeypatch.setattr(llm_acceleration, "DEFAULT_MODEL_DIRS", ())
        assert llm_acceleration.converted_models((str(tmp_path / "nope"),)) == []

    def test_a_real_directory_of_models_is_found(self, monkeypatch, tmp_path, ir_model):
        monkeypatch.setattr(llm_acceleration, "DEFAULT_MODEL_DIRS", ())
        made = ir_model()
        assert llm_acceleration.converted_models((str(tmp_path),)) == [made]


class TestChoosingBetweenVariantsOfTheSameWeights:
    """An Instruct and a Thinking build answer to one tag and are not
    interchangeable: one reasons before answering and one does not, which is
    the entire reason a thinking model gets chosen in the first place.

    `same_model` cannot separate them - their names differ only in a word it
    strips - so name order picked between them once and quietly turned a
    reasoning narrator into a plain one. The tie is broken on behaviour, from
    what each side already publishes.
    """

    def _pair(self, tmp_path):
        def make(name, reasons):
            d = tmp_path / name
            d.mkdir(parents=True, exist_ok=True)
            (d / llm_acceleration.IR_MARKER).write_text("<net/>", encoding="utf-8")
            (d / "chat_template.jinja").write_text(
                "<think>{{x}}</think>" if reasons else "{{x}}", encoding="utf-8")
            return str(d)
        # Instruct sorts first, so name order would take it.
        return (make("Qwen3-VL-8B-Instruct-int4-ov", False),
                make("Qwen3-VL-8B-Thinking-int4-ov", True))

    def test_a_reasoning_tag_gets_the_reasoning_build(self, monkeypatch, tmp_path):
        instruct, thinking = self._pair(tmp_path)
        _machine(monkeypatch, card="Intel Arc", genai=True, search=(str(tmp_path),))
        monkeypatch.setattr(llm_acceleration, "tag_reasons", lambda *a, **k: True)
        got = llm_acceleration.decide({}, {"backend": "ollama", "model": "qwen3-vl:8b"})
        assert got["model"] == thinking
        assert got["model"] != instruct

    def test_a_plain_tag_gets_the_plain_build(self, monkeypatch, tmp_path):
        instruct, thinking = self._pair(tmp_path)
        _machine(monkeypatch, card="Intel Arc", genai=True, search=(str(tmp_path),))
        monkeypatch.setattr(llm_acceleration, "tag_reasons", lambda *a, **k: False)
        got = llm_acceleration.decide({}, {"backend": "ollama", "model": "qwen3-vl:8b"})
        assert got["model"] == instruct

    def test_the_reason_says_why_that_build_won(self, monkeypatch, tmp_path):
        self._pair(tmp_path)
        _machine(monkeypatch, card="Intel Arc", genai=True, search=(str(tmp_path),))
        monkeypatch.setattr(llm_acceleration, "tag_reasons", lambda *a, **k: True)
        reason = llm_acceleration.decide(
            {}, {"backend": "ollama", "model": "qwen3-vl:8b"})["reason"]
        assert "reasons" in reason
        assert "Instruct" in reason          # names what it passed over

    def test_an_unreachable_server_falls_back_to_name_order_and_says_so(
            self, monkeypatch, tmp_path):
        """Best-effort: preflight already reports a missing server better than
        this could, so it must not become a second failure."""
        instruct, _ = self._pair(tmp_path)
        _machine(monkeypatch, card="Intel Arc", genai=True, search=(str(tmp_path),))
        monkeypatch.setattr(llm_acceleration, "tag_reasons", lambda *a, **k: None)
        got = llm_acceleration.decide({}, {"backend": "ollama", "model": "qwen3-vl:8b"})
        assert got["model"] == instruct
        assert "could tell them apart" in got["reason"]

    def test_one_match_needs_no_tie_break(self, monkeypatch, tmp_path):
        """The server is not asked when there is nothing to choose between."""
        asked = []
        d = tmp_path / "Qwen3-VL-8B-Thinking-int4-ov"
        d.mkdir()
        (d / llm_acceleration.IR_MARKER).write_text("<net/>", encoding="utf-8")
        _machine(monkeypatch, card="Intel Arc", genai=True, search=(str(tmp_path),))
        monkeypatch.setattr(llm_acceleration, "tag_reasons",
                            lambda *a, **k: asked.append(1))
        got = llm_acceleration.decide({}, {"backend": "ollama", "model": "qwen3-vl:8b"})
        assert got["model"] == str(d)
        assert asked == []


class TestReadingReasoningOffAConvertedModel:
    def test_a_template_with_think_tags_reasons(self, tmp_path):
        (tmp_path / "chat_template.jinja").write_text(
            "<think>{{x}}</think>", encoding="utf-8")
        assert llm_acceleration.dir_reasons(str(tmp_path))

    def test_a_template_without_them_does_not(self, tmp_path):
        (tmp_path / "chat_template.jinja").write_text("{{x}}", encoding="utf-8")
        assert not llm_acceleration.dir_reasons(str(tmp_path))

    def test_a_missing_template_is_not_an_error(self, tmp_path):
        assert not llm_acceleration.dir_reasons(str(tmp_path))
