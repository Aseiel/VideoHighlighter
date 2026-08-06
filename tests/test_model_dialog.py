"""Tests for the report's model dialog — the form, not the settings.

The dialog deliberately touches no settings: it is handed a list and read back
after it closes, which is what makes it testable at all. What is worth pinning
is the behaviour that made the old chained prompts unusable — fields that mean
nothing for the chosen backend, and no view of what was already configured.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def app():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(app):
    from modules.ui.model_dialog import ModelDialog
    return ModelDialog(models=[{"backend": "ollama", "model": "llama3"}],
                       chosen="llama3")


def test_what_is_already_configured_is_visible(dialog):
    assert [dialog.list.item(i).text() for i in range(dialog.list.count())] \
        == ["llama3"]


def test_only_the_fields_that_mean_something_are_shown(dialog):
    """A GGUF path means nothing to Ollama, and a tag means nothing to a file."""
    assert dialog.tag_row.isVisibleTo(dialog)
    assert not dialog.gguf_row.isVisibleTo(dialog)

    dialog.backend.setCurrentIndex(1)          # llama-cpp
    assert dialog.gguf_row.isVisibleTo(dialog)
    assert dialog.mmproj_row.isVisibleTo(dialog)
    assert not dialog.tag_row.isVisibleTo(dialog)


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


def test_removing_the_last_one_leaves_nothing_chosen(app):
    from modules.ui.model_dialog import ModelDialog
    dialog = ModelDialog(models=[{"backend": "ollama", "model": "llama3"}],
                         chosen="llama3")
    dialog.list.setCurrentRow(0)
    dialog._remove_selected()
    assert dialog.models == [] and dialog.chosen is None
