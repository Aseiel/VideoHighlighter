"""Tests for the Hugging Face mirror setting (modules/system/hf_mirror.py).

What's pinned down: nothing changes unless the user asked, an HF_ENDPOINT
already in the environment is never overruled, and a bad value is ignored
rather than stopping the app.
"""
from __future__ import annotations

import pytest

from modules.system import hf_mirror

_REAL_FROM_CONFIG = hf_mirror._from_config


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv(hf_mirror.ENV_VAR, raising=False)
    monkeypatch.setattr(hf_mirror, "_from_config", lambda: None)


def test_nothing_set_changes_nothing(monkeypatch):
    import os

    assert hf_mirror.apply() is None
    assert "HF_ENDPOINT" not in os.environ


def test_the_variable_sets_the_endpoint(monkeypatch):
    import os

    monkeypatch.setenv(hf_mirror.ENV_VAR, "hf-mirror.com/")
    assert hf_mirror.apply() == "https://hf-mirror.com"
    assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"


def test_config_is_used_when_the_variable_is_not(monkeypatch):
    monkeypatch.setattr(hf_mirror, "_from_config", lambda: "https://mirror.example")
    assert hf_mirror.apply() == "https://mirror.example"


def test_an_existing_hf_endpoint_wins(monkeypatch):
    import os

    monkeypatch.setenv("HF_ENDPOINT", "https://mine.example")
    monkeypatch.setenv(hf_mirror.ENV_VAR, "https://other.example")
    assert hf_mirror.apply() is None
    assert os.environ["HF_ENDPOINT"] == "https://mine.example"


@pytest.mark.parametrize("value", ["ftp://x", "https://", "   "])
def test_a_bad_value_is_ignored(monkeypatch, value):
    monkeypatch.setenv(hf_mirror.ENV_VAR, value)
    assert hf_mirror.apply() is None


def test_it_reads_config_yaml(tmp_path, monkeypatch):
    from modules.system import app_paths

    (tmp_path / "config.yaml").write_text("hf_endpoint: https://hf-mirror.com\n",
                                          encoding="utf-8")
    monkeypatch.setattr(hf_mirror, "_from_config", _REAL_FROM_CONFIG)
    monkeypatch.setattr(app_paths, "data_file", lambda name: str(tmp_path / name))
    assert hf_mirror.configured() == "https://hf-mirror.com"
