"""Tests for deciding which Ollama server the app talks to.

The bug this module exists to prevent is not "the URL is wrong" — it is *half*
the app being pointed somewhere. The chat panel, the report dialog, the advisor
and the narration each carried their own ``http://localhost:11434`` default, so
a user with the server on another machine could point one of them at it and have
the rest silently keep asking a box with no models on it. The failure that
produces is "model 'qwen3-vl:8b' not found" for a model the user can see
running, which is about as misleading as an error gets.

So the tests here are about the *decision*: what wins over what, and what a user
can type into the field without having to know URL syntax.
"""
from __future__ import annotations

import pytest

from llm import ollama_host


class FakeSettings:
    """Enough QSettings for this module, without touching the real store."""

    def __init__(self, value=None):
        self.stored = {} if value is None else {ollama_host.HOST_KEY: value}

    def value(self, key, default=None):
        return self.stored.get(key, default)

    def setValue(self, key, value):
        self.stored[key] = value

    def remove(self, key):
        self.stored.pop(key, None)


@pytest.fixture(autouse=True)
def no_inherited_env(monkeypatch):
    """The developer's own OLLAMA_HOST must not decide these assertions."""
    for name in ollama_host.ENV_VARS:
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------
# What a user is allowed to type
# --------------------------------------------------------------------------
@pytest.mark.parametrize("typed,expected", [
    # The shape somebody reads off the other machine's screen.
    ("192.168.1.118", "http://192.168.1.118:11434"),
    ("192.168.1.118:11434", "http://192.168.1.118:11434"),
    ("box.lan", "http://box.lan:11434"),
    ("http://box.lan:11434/", "http://box.lan:11434"),
    # A non-standard port is a decision, and survives.
    ("box.lan:12345", "http://box.lan:12345"),
    # The whole endpoint, pasted out of a terminal or a curl command.
    ("http://box.lan:11434/api/generate", "http://box.lan:11434"),
    # A reverse proxy keeps its mount point and its port 443.
    ("https://ai.example.com/ollama/", "https://ai.example.com/ollama"),
    ("https://ai.example.com", "https://ai.example.com"),
    # Nothing typed is not a URL.
    ("", ""),
    (None, ""),
    ("   ", ""),
])
def test_normalise_accepts_what_people_write(typed, expected):
    assert ollama_host.normalise(typed) == expected


def test_a_bind_address_is_not_a_destination():
    """`OLLAMA_HOST=0.0.0.0` is how a server is *published*, not where it is.

    Somebody who made their server reachable set that variable on the server
    machine, and it is exactly the machine likely to be running this app too.
    Dialling the any-address connects to nothing on Windows; localhost is what
    they meant.
    """
    assert ollama_host.normalise("0.0.0.0:11434") == "http://localhost:11434"
    assert ollama_host.normalise("http://0.0.0.0:11434") == "http://localhost:11434"


# --------------------------------------------------------------------------
# What wins
# --------------------------------------------------------------------------
def test_default_is_where_it_always_was():
    assert ollama_host.resolve(settings=FakeSettings()) == ollama_host.DEFAULT_BASE_URL


def test_the_environment_is_read_when_nothing_was_set(monkeypatch):
    """A machine already configured for a remote server says so once."""
    monkeypatch.setenv("OLLAMA_HOST", "192.168.1.118")
    assert ollama_host.resolve(settings=FakeSettings()) == "http://192.168.1.118:11434"


def test_the_field_the_user_filled_in_beats_the_environment(monkeypatch):
    """Or the field would lie about where the run went.

    An inherited variable is not a decision the user can see; the host in the
    LLM panel is one they can, and it is the one shown next to the model list
    they picked from.
    """
    monkeypatch.setenv("OLLAMA_HOST", "http://inherited:11434")
    settings = FakeSettings("http://typed-in:11434")
    assert ollama_host.resolve(settings=settings) == "http://typed-in:11434"


def test_an_explicit_argument_beats_everything(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://inherited:11434")
    settings = FakeSettings("http://typed-in:11434")
    assert (ollama_host.resolve("box.lan", settings=settings)
            == "http://box.lan:11434")


def test_clearing_the_field_restores_the_default():
    """Emptying the box has to mean something, and this is the only thing it
    can sensibly mean: go back to this machine."""
    settings = FakeSettings("http://typed-in:11434")
    assert ollama_host.remember("", settings=settings) == ""
    assert ollama_host.HOST_KEY not in settings.stored
    assert ollama_host.resolve(settings=settings) == ollama_host.DEFAULT_BASE_URL


def test_remember_stores_the_normalised_form():
    """What is stored is what every other caller will resolve, so it is stored
    in the shape they need rather than the shape it was typed in."""
    settings = FakeSettings()
    assert ollama_host.remember("192.168.1.118", settings=settings) == \
        "http://192.168.1.118:11434"
    assert settings.stored[ollama_host.HOST_KEY] == "http://192.168.1.118:11434"


def test_a_broken_store_is_not_fatal():
    """Every caller's fallback is localhost, which is strictly better than a
    dialog that will not open."""
    class Exploding:
        def value(self, *a, **k):
            raise RuntimeError("no settings here")

    assert ollama_host.stored(Exploding()) == ""
    assert ollama_host.resolve(settings=Exploding()) == ollama_host.DEFAULT_BASE_URL


# --------------------------------------------------------------------------
# Which advice the failure gives
# --------------------------------------------------------------------------
def test_remote_is_told_apart_from_local():
    """The two failures have different first things to check: a local server
    is usually not started, a remote one is usually bound to its own
    localhost."""
    assert not ollama_host.is_remote("http://localhost:11434")
    assert not ollama_host.is_remote("127.0.0.1")
    assert ollama_host.is_remote("192.168.1.118")
    assert ollama_host.is_remote("https://ai.example.com")
