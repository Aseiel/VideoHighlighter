"""Tests for the report's model dialog — the form, not the settings.

The dialog is handed a list and read back after it closes, and everything that
reaches outside the window is injected, which is what makes it testable at all.
Two groups of behaviour are worth pinning. The ones that made the old chained
prompts unusable — fields that mean nothing for the chosen backend, and no view
of what was already configured. And the ones that made the Ollama field worse
than the panel next door: a name that had to be typed from memory when the
server it is about to talk to holds the list.

Every dialog here is built with stub sources. A test that asks a real Ollama
server is a test that passes or fails depending on what is running on the
machine, and one that reaches a dead server pays its timeout every run.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

INSTALLED = ["llama3.2:latest", "qwen2.5:7b"]
USED_BEFORE = ["D:/models/one.gguf", "D:/models/two.gguf"]


@pytest.fixture(scope="module")
def app():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def remembered():
    return []


@pytest.fixture
def make(app, remembered):
    """A dialog whose view of the machine is whatever the test says it is."""
    from modules.ui.model_dialog import ModelDialog

    def build(models=None, chosen=None, installed=INSTALLED, recent=USED_BEFORE):
        return ModelDialog(
            models=models if models is not None
            else [{"backend": "ollama", "model": "llama3"}],
            chosen=chosen if chosen is not None else "llama3",
            list_models=lambda refresh=False: list(installed),
            list_recent=lambda: list(recent),
            remember=remembered.append)
    return build


@pytest.fixture
def dialog(make):
    return make()


def test_what_is_already_configured_is_visible(dialog):
    assert [dialog.list.item(i).text() for i in range(dialog.list.count())] \
        == ["llama3"]


def test_only_the_fields_that_mean_something_are_shown(dialog):
    """A GGUF path means nothing to Ollama, and a tag means nothing to a file."""
    assert dialog.tag_row.isVisibleTo(dialog)
    assert not dialog.gguf_row.isVisibleTo(dialog)
    assert not dialog.recent_row.isVisibleTo(dialog)

    dialog.backend.setCurrentIndex(1)          # llama-cpp
    assert dialog.gguf_row.isVisibleTo(dialog)
    assert dialog.mmproj_row.isVisibleTo(dialog)
    assert dialog.recent_row.isVisibleTo(dialog)
    assert not dialog.tag_row.isVisibleTo(dialog)
    assert not dialog.status_row.isVisibleTo(dialog)


def test_a_gguf_is_added_with_its_projector(dialog):
    dialog.backend.setCurrentIndex(1)
    dialog.gguf.setText("D:/models/big.gguf")
    dialog.mmproj.setText("D:/models/mmproj.gguf")
    dialog._add()
    assert dialog.models[-1] == {"backend": "llama-cpp",
                                 "model": "D:/models/big.gguf",
                                 "mmproj": "D:/models/mmproj.gguf"}


def test_adding_selects_what_was_just_added(dialog):
    """Nobody adds a model in order to keep using the previous one."""
    dialog.backend.setCurrentIndex(1)
    dialog.gguf.setText("D:/models/big.gguf")
    dialog._add()
    assert dialog.chosen == "big"


def test_the_form_empties_so_the_next_one_starts_clean(dialog):
    dialog.backend.setCurrentIndex(1)
    dialog.gguf.setText("D:/models/big.gguf")
    dialog.mmproj.setText("D:/models/mmproj.gguf")
    dialog._add()
    assert dialog.gguf.text() == "" and dialog.mmproj.text() == ""


def test_a_name_the_user_gave_it_is_what_the_menu_will_show(dialog):
    dialog.backend.setCurrentIndex(1)
    dialog.gguf.setText("D:/models/big.gguf")
    dialog.label.setText("the big one")
    dialog._add()
    assert dialog.chosen == "the big one"


def test_an_empty_form_adds_nothing(dialog):
    before = len(dialog.models)
    dialog._add()
    assert len(dialog.models) == before


def test_removing_the_selected_model_moves_the_choice_on(dialog):
    """Removing the active model must not leave the report with none."""
    dialog.backend.setCurrentIndex(1)
    dialog.gguf.setText("D:/models/big.gguf")
    dialog._add()
    dialog.list.setCurrentRow(1)
    dialog._remove_selected()
    assert [m["model"] for m in dialog.models] == ["llama3"]
    assert dialog.chosen == "llama3"


def test_removing_the_last_one_leaves_nothing_chosen(make):
    dialog = make()
    dialog.list.setCurrentRow(0)
    dialog._remove_selected()
    assert dialog.models == [] and dialog.chosen is None


class TestWhatTheMachineAlreadyHas:
    """The reason this stopped being free text.

    A tag typed from memory is indistinguishable from one that has not been
    pulled, and the difference only surfaces at the end of a generation the user
    waited a minute for. The server holds the list; the dialog asks it.
    """

    def test_the_installed_models_are_offered(self, dialog):
        assert [dialog.tag.itemText(i) for i in range(dialog.tag.count())] \
            == INSTALLED

    def test_a_name_can_still_be_typed(self, dialog):
        # A model being pulled in another window is a perfectly good thing to
        # name, and a dropdown that refused it would be worse than the field it
        # replaced.
        dialog.tag.setCurrentText("something:new")
        dialog._add()
        assert dialog.models[-1] == {"backend": "ollama",
                                     "model": "something:new"}

    def test_adding_does_not_empty_the_offered_list(self, dialog):
        # `clear()` on a combo drops the items with the text, and the next model
        # would have to be typed after all.
        dialog.tag.setCurrentText("something:new")
        dialog._add()
        assert dialog.tag.count() == len(INSTALLED)
        assert dialog.tag.currentText() == ""

    def test_refreshing_asks_again(self, app, remembered):
        from modules.ui.model_dialog import ModelDialog

        asked = []

        def listing(refresh=False):
            asked.append(refresh)
            return ["only:one"]

        dialog = ModelDialog(models=[], list_models=listing,
                             list_recent=list, remember=remembered.append)
        dialog._refresh_tags()
        # Cached on the way in, bypassed on the button -- otherwise a model
        # pulled while the dialog is open can never appear.
        assert asked == [False, True]

    def test_no_server_is_reported_as_ordinary(self, make):
        # Not an error: a user who has not started Ollama, or who only runs
        # GGUF files, is in a perfectly normal state.
        dialog = make(installed=[])
        assert "Refresh" in dialog.status.text()
        assert dialog.tag.isEditable()

    def test_a_listing_that_raises_leaves_the_field_usable(self, app,
                                                           remembered):
        from modules.ui.model_dialog import ModelDialog

        def broken(refresh=False):
            raise RuntimeError("no")

        dialog = ModelDialog(models=[], list_models=broken, list_recent=list,
                             remember=remembered.append)
        dialog.tag.setCurrentText("typed:anyway")
        dialog._add()
        assert dialog.models == [{"backend": "ollama", "model": "typed:anyway"}]


class TestTheGgufShortlist:
    def test_files_used_before_are_offered(self, dialog):
        assert [dialog.recent.itemData(i)
                for i in range(dialog.recent.count())] == USED_BEFORE

    def test_picking_one_fills_the_path(self, dialog):
        dialog.backend.setCurrentIndex(1)
        dialog.recent.setCurrentIndex(1)
        dialog._recent_chosen(1)
        assert dialog.gguf.text() == USED_BEFORE[1]

    def test_a_file_added_here_joins_the_shortlist(self, dialog, remembered):
        # The same list the chat panel reads, so a model picked in either place
        # is offered in both. Two private lists of one thing is how they end up
        # disagreeing about which model is recent.
        dialog.backend.setCurrentIndex(1)
        dialog.gguf.setText("D:/models/three.gguf")
        dialog._add()
        assert remembered == ["D:/models/three.gguf"]

    def test_an_ollama_entry_is_not_remembered_as_a_file(self, dialog,
                                                         remembered):
        dialog.tag.setCurrentText("llama3.2:latest")
        dialog._add()
        assert remembered == []

    def test_nothing_used_before_says_so_and_disables_the_row(self, make):
        dialog = make(recent=[])
        assert not dialog.recent.isEnabled()
        assert "Browse" in dialog.recent.itemText(0)
