"""Tests for asking the machine what models it already has.

Small module, and every one of its behaviours is there because the alternative
costs the user something visible:

* the Ollama list is cached, because the call costs a three-second timeout
  exactly when the server is *not* running — which is when somebody opens the
  dialog repeatedly wondering what is wrong;
* Refresh has to bypass that cache, or a model pulled while the dialog is open
  can never appear;
* nothing raises, because every caller's fallback is the typing that already
  worked and a dialog that fails to open is strictly worse;
* the GGUF shortlist is the chat panel's own list, read in the shape QSettings
  actually returns it in rather than the shape it was written in.
"""
from __future__ import annotations

import pytest

from modules import llm_discovery


class FakeSettings:
    """Enough QSettings for this module, without touching the real store."""

    def __init__(self, value=None):
        self.stored = {} if value is None else {llm_discovery.RECENT_GGUF_KEY: value}

    def value(self, key, default=None):
        return self.stored.get(key, default)

    def setValue(self, key, value):
        self.stored[key] = value


@pytest.fixture(autouse=True)
def clean_cache():
    llm_discovery.forget_ollama_models()
    yield
    llm_discovery.forget_ollama_models()


class TestOllamaModels:
    def test_the_server_is_asked_once(self, monkeypatch):
        import llm.llm_module as module

        calls = []
        monkeypatch.setattr(module, "get_ollama_models",
                            lambda url: calls.append(url) or ["a", "b"])
        assert llm_discovery.ollama_models() == ["a", "b"]
        assert llm_discovery.ollama_models() == ["a", "b"]
        assert len(calls) == 1

    def test_refresh_asks_again(self, monkeypatch):
        import llm.llm_module as module

        answers = [["first"], ["second"]]
        monkeypatch.setattr(module, "get_ollama_models",
                            lambda url: answers.pop(0))
        assert llm_discovery.ollama_models() == ["first"]
        assert llm_discovery.ollama_models(refresh=True) == ["second"]

    def test_a_server_that_is_not_there_is_an_empty_list(self, monkeypatch):
        import llm.llm_module as module

        def unreachable(url):
            raise OSError("connection refused")

        monkeypatch.setattr(module, "get_ollama_models", unreachable)
        assert llm_discovery.ollama_models() == []

    def test_the_absence_is_cached_too(self, monkeypatch):
        # The expensive case is the one worth caching: a dead server costs the
        # full timeout, and the dialog is opened again the moment it fails.
        import llm.llm_module as module

        calls = []
        monkeypatch.setattr(module, "get_ollama_models",
                            lambda url: calls.append(url) or [])
        llm_discovery.ollama_models()
        llm_discovery.ollama_models()
        assert len(calls) == 1

    def test_the_caller_cannot_mutate_the_cache(self, monkeypatch):
        import llm.llm_module as module

        monkeypatch.setattr(module, "get_ollama_models", lambda url: ["a"])
        llm_discovery.ollama_models().append("b")
        assert llm_discovery.ollama_models() == ["a"]


class TestRecentGguf:
    def test_files_that_exist_are_listed_newest_first(self, tmp_path):
        one = tmp_path / "one.gguf"
        two = tmp_path / "two.gguf"
        one.write_bytes(b"x")
        two.write_bytes(b"x")
        settings = FakeSettings([str(two), str(one)])
        assert llm_discovery.recent_gguf(settings) == [str(two), str(one)]

    def test_a_path_that_is_gone_is_not_offered(self, tmp_path):
        # Offering it would produce a load failure minutes later; the entry
        # stays stored, because a disconnected drive is not a mistake.
        here = tmp_path / "here.gguf"
        here.write_bytes(b"x")
        settings = FakeSettings([str(tmp_path / "gone.gguf"), str(here)])
        assert llm_discovery.recent_gguf(settings) == [str(here)]
        assert len(settings.stored[llm_discovery.RECENT_GGUF_KEY]) == 2

    def test_a_single_entry_comes_back_as_a_bare_string(self, tmp_path):
        # QSettings on Windows does this, and the naive read iterates the
        # characters of the path.
        one = tmp_path / "one.gguf"
        one.write_bytes(b"x")
        assert llm_discovery.recent_gguf(FakeSettings(str(one))) == [str(one)]

    def test_nothing_stored_is_an_empty_list(self):
        assert llm_discovery.recent_gguf(FakeSettings()) == []


class TestRememberGguf:
    def test_the_newest_goes_to_the_front_without_duplicating(self):
        settings = FakeSettings(["a", "b"])
        llm_discovery.remember_gguf("b", settings)
        assert settings.stored[llm_discovery.RECENT_GGUF_KEY] == ["b", "a"]

    def test_the_list_is_capped(self):
        settings = FakeSettings([str(n) for n in range(20)])
        llm_discovery.remember_gguf("new", settings)
        assert len(settings.stored[llm_discovery.RECENT_GGUF_KEY]) == \
            llm_discovery.MAX_RECENT_GGUF

    def test_an_empty_path_is_not_stored(self):
        settings = FakeSettings(["a"])
        llm_discovery.remember_gguf("   ", settings)
        assert settings.stored[llm_discovery.RECENT_GGUF_KEY] == ["a"]


def test_it_writes_where_the_chat_panel_reads():
    """The one thing a unit test cannot check by behaviour.

    The shortlist is only shared if both sides name the same store. If the chat
    panel's keys move, this test is what says so — the alternative is two lists
    that quietly disagree about which model is recent.
    """
    import re
    from pathlib import Path

    panel = Path(__file__).resolve().parents[1] / "llm" / "llm_chat_widget.py"
    source = panel.read_text(encoding="utf-8")
    assert f'SETTINGS_KEY = "{llm_discovery.SETTINGS_ORG}"' in source
    assert re.search(r'QSettings\(self\.SETTINGS_KEY, "%s"\)'
                     % llm_discovery.SETTINGS_APP, source)
    assert f'"{llm_discovery.RECENT_GGUF_KEY}"' in source
