"""Tests for `modules.llm_models` — the list of models a report can be written with.

The property worth protecting is that a stored list can never break the feature
it configures. Settings survive upgrades, get hand-edited, and come back from
QSettings as a string on one machine and a list on another; anything unusable in
them has to be dropped on the way in rather than offered in a menu and failed
after the user has waited for a model to load.
"""

from __future__ import annotations

from modules.llm_models import active, label_for, migrate, parse, serialise


class TestParsing:
    def test_the_stored_json_round_trips(self):
        models = [{"backend": "ollama", "model": "llama3"},
                  {"backend": "llama-cpp", "model": "D:/m/a.gguf"}]
        assert parse(serialise(models)) == models

    def test_an_already_decoded_list_is_accepted_too(self):
        """QSettings hands back a string or a list depending on the machine."""
        models = [{"backend": "ollama", "model": "llama3"}]
        assert parse(models) == models

    def test_nothing_stored_is_not_an_error(self):
        assert parse(None) == [] and parse("") == [] and parse("{{{") == []

    def test_a_backend_that_cannot_be_built_is_dropped_on_the_way_in(self):
        """Offering it only moves the failure to after the user has waited."""
        assert parse([{"backend": "gpt-9", "model": "x"}]) == []

    def test_an_entry_without_a_model_is_dropped(self):
        assert parse([{"backend": "ollama", "model": "  "}]) == []

    def test_duplicates_collapse(self):
        one = {"backend": "ollama", "model": "llama3"}
        assert parse([one, dict(one)]) == [one]

    def test_a_vision_projector_is_kept_with_its_model(self):
        entry = {"backend": "llama-cpp", "model": "a.gguf", "mmproj": "b.gguf"}
        assert parse([entry])[0]["mmproj"] == "b.gguf"


class TestLabels:
    def test_a_gguf_is_named_by_its_file_not_its_path(self):
        """The directory almost never identifies it and never fits a menu."""
        entry = {"backend": "llama-cpp", "model": "D:/models/Some-Model-Q4_K_M.gguf"}
        assert label_for(entry) == "Some-Model-Q4_K_M"

    def test_an_ollama_tag_is_its_own_label(self):
        assert label_for({"backend": "ollama", "model": "llama3"}) == "llama3"

    def test_a_name_the_user_gave_it_wins(self):
        entry = {"backend": "ollama", "model": "llama3", "label": "the small one"}
        assert label_for(entry) == "the small one"

    def test_nothing_configured_still_has_something_to_show(self):
        assert label_for(None) == "no model"


class TestChoosing:
    MODELS = [{"backend": "ollama", "model": "llama3"},
              {"backend": "llama-cpp", "model": "D:/m/big.gguf"}]

    def test_the_last_pick_is_remembered_by_name(self):
        assert active(self.MODELS, "big") == self.MODELS[1]

    def test_a_pick_that_no_longer_exists_falls_back_rather_than_failing(self):
        """A removed model must not leave the report unable to run at all."""
        assert active(self.MODELS, "one I deleted") == self.MODELS[0]

    def test_no_models_means_no_choice(self):
        assert active([], "anything") is None


class TestMigration:
    def test_the_older_single_model_setting_becomes_the_first_entry(self):
        """Losing it would fail the first run after an update, silently."""
        models = migrate([], "llama-cpp", "D:/m/a.gguf")
        assert models == [{"backend": "llama-cpp", "model": "D:/m/a.gguf"}]

    def test_it_is_not_added_twice(self):
        one = {"backend": "ollama", "model": "llama3"}
        assert migrate([one], "ollama", "llama3") == [one]

    def test_nothing_configured_before_migrates_to_nothing(self):
        assert migrate([], None, None) == []
