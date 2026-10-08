"""Which trained action model a run uses, and importing one made elsewhere.

What is pinned here: a run uses the newest model unless the user chose one
(Advanced > Action Recognition), none at all when they chose "None", and the
newest again when the chosen one is gone; the choice is read from the
environment the GUI publishes, else from config.yaml. Importing copies a model
on the shared encoder or a fine-tuned one already in the app's layout (only
its model files), installs a fine-tuning export with its tower stored as fp16,
never overwrites an installed model, and refuses a folder the app cannot use
without writing anything.

Class names are made up.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from modules.system import app_paths
from modules.vision import action_models as M
from modules.vision import action_siglip as A
from modules.vision import frame_encoder as fe
from tests.test_action_own_encoder import (  # noqa: F401  (fixture)
    CLASSES, _export, _tower_vector, head_meta, make_head, no_openvino, own_block)


@pytest.fixture
def models(tmp_path, monkeypatch):
    root = tmp_path / "actions"
    root.mkdir()
    monkeypatch.setattr(app_paths, "action_models_dir", lambda: str(root))
    monkeypatch.delenv(A.HEAD_DIR_ENV, raising=False)
    return root


def _frozen(folder, **over):
    return make_head(folder, head_meta(encoder=fe.ENCODER_ID, **over), tower=None)


def _installed(models, *names):
    """Frozen heads, oldest first, so the last is the newest."""
    for i, name in enumerate(names):
        _frozen(models / name)
        t = time.time() - 100 + i
        os.utime(models / name, (t, t))


# ── the choice ───────────────────────────────────────────────────────────────

def test_the_newest_model_is_used_unless_one_is_chosen(models):
    _installed(models, "older", "newer")
    assert [os.path.basename(h) for h in A.active_heads(fe.ENCODER_ID)] == ["newer", "older"]
    A.choose_model("older")
    assert [os.path.basename(h) for h in A.active_heads(fe.ENCODER_ID)] == ["older"]
    assert A.installed_head_classes()[0] == "older"


def test_none_means_actions_by_name(models, monkeypatch):
    _installed(models, "only")
    A.choose_model(A.NO_MODEL)
    assert A.active_heads(fe.ENCODER_ID) == []
    monkeypatch.setattr(fe, "load_actions", lambda folder=None: (["walking"], None))
    assert A.installed_head_classes() == (A.TEXT_SOURCE, ["walking"])


def test_a_chosen_model_that_is_gone_falls_back_to_the_newest(models):
    _installed(models, "older", "newer")
    A.choose_model("deleted-since")
    assert os.path.basename(A.active_heads(fe.ENCODER_ID)[0]) == "newer"


def test_the_saved_choice_is_read_when_nothing_was_published(models, tmp_path, monkeypatch):
    _installed(models, "older", "newer")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("advanced:\n  action_model: older\n", encoding="utf-8")
    monkeypatch.setattr(app_paths, "config_path", lambda name: str(cfg))
    monkeypatch.delenv(A.MODEL_ENV, raising=False)
    assert A.chosen_model() == "older"
    assert os.path.basename(A.active_heads(fe.ENCODER_ID)[0]) == "older"
    A.choose_model("")                     # the GUI's choice wins over the file
    assert A.chosen_model() == ""


def test_the_run_takes_the_chosen_model(models, monkeypatch):
    _installed(models, "older", "newer")
    A.choose_model("older")
    seen = []

    class Head:
        def __init__(self, folder):
            seen.append(os.path.basename(folder))
            raise RuntimeError("stop here")

    monkeypatch.setattr(A, "ActionHead", Head)
    with pytest.raises(RuntimeError, match="stop here"):
        A.run_action_detection_siglip("video.mp4", log=lambda m: None)
    assert seen == ["older"]


def test_installed_lists_every_model_with_its_kind(models):
    _installed(models, "frozen-one")
    make_head(models / "tuned-one", head_meta(own_encoder=own_block()))
    found = {m["name"]: m for m in M.installed()}
    assert set(found) == {"frozen-one", "tuned-one"}
    assert found["tuned-one"]["fine_tuned"] and not found["frozen-one"]["fine_tuned"]
    assert found["frozen-one"]["classes"] == CLASSES


# ── importing ────────────────────────────────────────────────────────────────

def test_a_model_on_the_shared_encoder_is_copied_without_anything_else(models, tmp_path):
    src = tmp_path / "from-a-friend"
    _frozen(src)
    (src / "notes.txt").write_text("not part of the model", encoding="utf-8")
    dest = M.import_model(str(src), log=lambda m: None)
    assert dest == str(models / "from-a-friend")
    assert sorted(os.listdir(dest)) == [A.HEAD_META, A.HEAD_MODEL]
    assert A.read_head_meta(dest)["classes"] == CLASSES


def test_an_installed_fine_tuned_model_is_copied_with_its_tower(models, tmp_path):
    src = tmp_path / "tuned"
    make_head(src, head_meta(own_encoder=own_block()))
    dest = M.import_model(str(src), log=lambda m: None)
    assert sorted(os.listdir(dest)) == [A.HEAD_META, A.HEAD_MODEL, "vision.onnx"]
    assert A.own_encoder(dest, A.read_head_meta(dest))


def test_a_fine_tuning_export_is_installed(models, tmp_path, no_openvino):
    pytest.importorskip("onnxruntime")
    dest = M.import_model(_export(tmp_path), name="tuned", log=lambda m: None)
    own = A.own_encoder(dest, A.read_head_meta(dest))
    assert fe._cosine(own["probe"], _tower_vector(fe.probe_pixels())[0]) > 0.99999


def test_an_import_never_overwrites_an_installed_model(models, tmp_path):
    _installed(models, "same")
    src = tmp_path / "same"
    _frozen(src, classes=["class-c", "class-d"])
    dest = M.import_model(str(src), log=lambda m: None)
    assert os.path.basename(dest) == "same-2"
    assert A.read_head_meta(str(models / "same"))["classes"] == CLASSES


def test_the_word_for_no_model_is_never_a_folder_name(models, tmp_path):
    src = tmp_path / "None"
    _frozen(src)
    assert os.path.basename(M.import_model(str(src), log=lambda m: None)) == "None-model"


@pytest.mark.parametrize("prepare, words", [
    (lambda f: make_head(f, head_meta(encoder="someone-elses-encoder"), tower=None),
     "an encoder this app does not have"),
    (lambda f: (f.mkdir(), (f / A.HEAD_META).write_text(
        json.dumps(head_meta(encoder=fe.ENCODER_ID)), encoding="utf-8")), "head.onnx is missing"),
    (lambda f: f.mkdir(), "no head.json"),
    (lambda f: make_head(f, head_meta(kind="something-else"), tower=None), "not an action model"),
    (lambda f: make_head(f, head_meta(own_encoder=own_block(preprocess="other"))), "input"),
])
def test_a_folder_the_app_cannot_use_is_refused_and_nothing_written(models, tmp_path,
                                                                   prepare, words):
    src = tmp_path / "candidate"
    prepare(src)
    with pytest.raises(ValueError, match=words):
        M.import_model(str(src), log=lambda m: None)
    assert os.listdir(models) == []


def test_a_model_already_installed_is_not_imported_again(models):
    _installed(models, "here")
    with pytest.raises(ValueError, match="already installed"):
        M.import_model(str(models / "here"), log=lambda m: None)
