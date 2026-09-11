"""The interface-size override, and the one rule that makes it work.

Qt reads ``QT_SCALE_FACTOR`` when the QApplication is constructed and never
looks again, so this has to run before that and nothing here may be expensive
or fragile: a malformed config must cost a log line, not a window that never
appears.
"""

from __future__ import annotations

import os

import pytest

from modules import ui_scale


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(ui_scale.ENV_VAR, raising=False)
    monkeypatch.delenv(ui_scale.QT_VAR, raising=False)
    # No config unless a test writes one.
    monkeypatch.setattr(ui_scale, "_from_config", lambda: None)


class TestReadingTheSetting:
    def test_the_environment_variable_is_used(self, monkeypatch):
        monkeypatch.setenv(ui_scale.ENV_VAR, "0.75")

        assert ui_scale.apply() == 0.75
        assert os.environ[ui_scale.QT_VAR] == "0.75"

    def test_a_comma_decimal_is_understood(self, monkeypatch):
        """Half of Europe types it that way, and float() does not."""
        monkeypatch.setenv(ui_scale.ENV_VAR, "0,8")

        assert ui_scale.apply() == 0.8

    def test_the_environment_beats_the_config(self, monkeypatch):
        monkeypatch.setattr(ui_scale, "_from_config", lambda: 0.5)
        monkeypatch.setenv(ui_scale.ENV_VAR, "1.25")

        assert ui_scale.apply() == 1.25

    def test_the_config_is_used_when_the_variable_is_absent(self, monkeypatch):
        monkeypatch.setattr(ui_scale, "_from_config", lambda: 0.6)

        assert ui_scale.apply() == 0.6

    def test_nothing_configured_sets_nothing(self):
        """The OS decides, exactly as before this existed."""
        assert ui_scale.apply() is None
        assert ui_scale.QT_VAR not in os.environ


class TestRefusingBadValues:
    """A bad number must not be the reason the app does not start."""

    @pytest.mark.parametrize("value", ["big", "", "   ", "1.2.3", "None"])
    def test_nonsense_is_ignored(self, monkeypatch, value):
        monkeypatch.setenv(ui_scale.ENV_VAR, value)

        assert ui_scale.apply() is None
        assert ui_scale.QT_VAR not in os.environ

    @pytest.mark.parametrize("value", ["0.1", "12", "-1"])
    def test_values_outside_the_usable_range_are_ignored(self, monkeypatch, value):
        monkeypatch.setenv(ui_scale.ENV_VAR, value)

        assert ui_scale.apply() is None

    def test_it_says_why(self, monkeypatch, capsys):
        monkeypatch.setenv(ui_scale.ENV_VAR, "enormous")

        ui_scale.apply()

        assert "not a number" in capsys.readouterr().out

    def test_an_unreadable_config_is_survivable(self, monkeypatch, tmp_path):
        """The real _from_config, pointed at a file that is not YAML."""
        broken = tmp_path / "config.yaml"
        broken.write_text("ui_scale: [unclosed\n", encoding="utf-8")
        monkeypatch.setattr(ui_scale, "_from_config",
                            ui_scale._from_config)          # the real one
        monkeypatch.setattr("modules.app_paths.data_file",
                            lambda name: str(broken))

        assert ui_scale.configured() is None


class TestNotFightingTheUser:
    def test_an_existing_qt_variable_wins(self, monkeypatch):
        """Somebody who set Qt's own variable by hand outranks a config file."""
        monkeypatch.setenv(ui_scale.QT_VAR, "2")
        monkeypatch.setenv(ui_scale.ENV_VAR, "0.75")

        assert ui_scale.apply() is None
        assert os.environ[ui_scale.QT_VAR] == "2"
