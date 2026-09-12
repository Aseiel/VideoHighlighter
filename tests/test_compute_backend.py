"""Choosing the accelerator by name, and carrying that choice to the work.

The app picks CUDA, then Intel, then DirectML, then the processor, and that is
right for almost everybody. What it lacked was a way to *ask* for one, which is
the only way to find out whether DirectML beats OpenVINO on a particular card.

Two rules shape everything here. The choice travels in the environment, because
detection runs in worker processes that inherit that and no Python state. And
it is a preference, not a command: a backend this machine does not have costs a
line in the log, never a run.
"""

from __future__ import annotations

import os

import pytest

from modules import compute_backend as backend
from modules import directml_device


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(backend.ENV_VAR, raising=False)
    monkeypatch.delenv(directml_device.MODE_ENV, raising=False)
    directml_device.set_mode(None)
    yield
    directml_device.set_mode(None)


class TestNamingABackend:
    @pytest.mark.parametrize("written,expected", [
        ("cuda", "cuda"), ("NVIDIA", "cuda"), ("gpu", "cuda"),
        ("intel", "intel"), ("OpenVINO", "intel"), ("xpu", "intel"),
        ("arc", "intel"),
        ("directml", "directml"), ("dml", "directml"), ("AMD", "directml"),
        ("cpu", "cpu"), ("processor", "cpu"),
        ("auto", "auto"), ("", "auto"), ("  Automatic ", "auto"),
    ])
    def test_the_names_people_write(self, written, expected):
        """Vendor names included: the setting is about hardware, and somebody
        typing "nvidia" into a config file is not making a mistake."""
        assert backend.normalise(written) == expected

    @pytest.mark.parametrize("written", [None, "metal", "rocm", "fastest"])
    def test_a_backend_this_app_does_not_have(self, written):
        assert backend.normalise(written) is None

    def test_every_choice_is_offered_once(self):
        offered = [name for name, _ in backend.CHOICES]

        assert offered == ["auto", "cuda", "intel", "directml", "cpu"]

    def test_the_labels_name_hardware_first(self):
        """"OpenVINO" is an implementation detail of the Intel path; the person
        choosing knows which card is in their machine."""
        labels = dict(backend.CHOICES)

        assert "NVIDIA" in labels["cuda"]
        assert "Intel" in labels["intel"]
        assert "AMD" in labels["directml"]


class TestReadingTheSetting:
    def test_it_comes_from_its_own_config_section(self):
        assert backend.from_config({"compute": {"backend": "directml"}}) == "directml"

    @pytest.mark.parametrize("config", [
        None, {}, {"compute": None}, {"compute": {}}, {"compute": "cuda"},
        {"backend": "cuda"},
    ])
    def test_a_config_without_it_says_nothing(self, config):
        assert backend.from_config(config) is None

    def test_the_environment_is_what_a_worker_process_reads(self, monkeypatch):
        monkeypatch.setenv(backend.ENV_VAR, "intel")

        assert backend.configured() == "intel"


class TestPublishingTheChoice:
    def test_applying_it_reaches_the_environment(self):
        backend.apply({"compute": {"backend": "directml"}}, log=lambda *_: None)

        assert os.environ[backend.ENV_VAR] == "directml"

    def test_an_exported_variable_outranks_the_config(self, monkeypatch):
        monkeypatch.setenv(backend.ENV_VAR, "cpu")

        assert backend.apply({"compute": {"backend": "cuda"}},
                             log=lambda *_: None) is None
        assert os.environ[backend.ENV_VAR] == "cpu"

    def test_nothing_configured_publishes_nothing(self):
        assert backend.apply({}, log=lambda *_: None) is None
        assert backend.ENV_VAR not in os.environ

    def test_changing_it_needs_no_restart(self):
        backend.set_now("cpu", log=lambda *_: None)

        assert backend.configured() == "cpu"

        backend.set_now("intel", log=lambda *_: None)

        assert backend.configured() == "intel"

    def test_an_unknown_name_changes_nothing(self):
        backend.set_now("rocm", log=lambda *_: None)

        assert backend.ENV_VAR not in os.environ


class TestItKeepsDirectMLsOwnSwitchInStep:
    """`VH_DIRECTML` predates this setting and other code still reads it. The
    two saying different things is worse than either."""

    def test_choosing_directml_turns_it_on(self):
        backend.set_now("directml", log=lambda *_: None)

        assert directml_device.mode() == directml_device.MODE_FORCE

    @pytest.mark.parametrize("choice", ["cuda", "intel", "cpu"])
    def test_choosing_another_backend_turns_it_off(self, choice):
        backend.set_now(choice, log=lambda *_: None)

        assert directml_device.mode() == directml_device.MODE_OFF

    def test_automatic_leaves_it_alone(self):
        backend.set_now("directml", log=lambda *_: None)
        backend.set_now("auto", log=lambda *_: None)

        assert directml_device.MODE_ENV not in os.environ
        assert directml_device.mode() == directml_device.MODE_AUTO
