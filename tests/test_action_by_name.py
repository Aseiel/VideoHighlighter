"""Actions by name: SigLIP2 with no trained model.

A window is matched against action names written as text. Every name on the
encoder's list (Kinetics-700) competes in every window, and a typed action
counts when the window looks more like it than like the rest, collecting the
listed names that contain it. These pin that arithmetic on synthetic vectors,
so nothing here needs the model.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from modules.vision import action_siglip as a
from modules.vision import frame_encoder as fe

REPO = Path(__file__).resolve().parent.parent
NAMES = ["robot dancing", "swimming", "cooking", "dancing macarena"]


def _vocab(dims=8):
    return np.eye(dims, dtype=np.float32)[:len(NAMES)]


@pytest.fixture
def encoder_dir(tmp_path, monkeypatch):
    """An encoder folder holding only the action list and encoder.json."""
    np.savez(tmp_path / fe.ACTIONS_FILE, labels=np.array(NAMES),
             vectors=_vocab().astype(np.float16))
    (tmp_path / fe.META_FILE).write_text(json.dumps({"logit_scale": 100.0}))
    monkeypatch.setattr(fe, "find_model_dir", lambda: str(tmp_path))
    return str(tmp_path)


def _window(index, dims=8, frames=4):
    v = np.zeros((1, frames, dims), np.float32)
    v[0, :, index] = 1.0
    return v


def test_nothing_typed_reports_every_listed_action(encoder_dir):
    actions = a.text_actions(None, folder=encoder_dir)
    assert actions.classes == NAMES and not actions.typed
    shares = actions.scores(_window(1))
    assert shares.shape == (1, len(NAMES))
    assert actions.detected(shares)[0].tolist() == [False, True, False, False]


def test_a_typed_action_collects_the_listed_names_that_contain_it(encoder_dir):
    vocab = np.concatenate([_vocab(), np.eye(8, dtype=np.float32)[[6]]])

    class Text:
        def encode(self, texts):
            assert texts == ["dancing"]
            return vocab[-1:]

    actions = a.text_actions(["Dancing"], folder=encoder_dir, text_encoder=Text())
    assert actions.classes == ["dancing"] and actions.typed
    # A window that looks like "robot dancing" counts as dancing.
    assert actions.detected(actions.scores(_window(0)))[0, 0]
    # One that looks like swimming does not.
    assert not actions.detected(actions.scores(_window(1)))[0, 0]


def test_a_listed_action_needs_no_text_tower(encoder_dir, monkeypatch):
    def refuse(**_):
        raise AssertionError("the text tower was loaded for a listed action")

    monkeypatch.setattr(fe, "load_text", refuse)
    actions = a.text_actions(["swimming"], folder=encoder_dir)
    assert actions.classes == ["swimming"]


def test_an_unlisted_action_without_a_text_tower_is_named_and_skipped(encoder_dir, monkeypatch):
    lines = []
    monkeypatch.setattr(fe, "load_text", lambda **_: None)
    actions = a.text_actions(["swimming", "kite surfing"], folder=encoder_dir,
                             log=lines.append)
    assert actions.classes == ["swimming"]
    assert any("kite surfing" in line for line in lines)


def test_a_window_that_matches_nothing_clearly_reports_nothing(encoder_dir):
    actions = a.text_actions(None, folder=encoder_dir)
    blur = np.full((1, 4, 8), 1.0, np.float32)       # equally like everything
    assert not actions.detected(actions.scores(blur)).any()


def test_no_trained_head_still_counts_as_available(encoder_dir, monkeypatch):
    monkeypatch.setattr(fe, "is_installed", lambda: True)
    monkeypatch.setattr(a, "find_heads", lambda *_: [])
    assert a.available()
    assert a.installed_head_classes() == (a.TEXT_SOURCE, NAMES)


def test_ensemble_is_a_unit_vector_between_its_parts():
    out = fe.ensemble([np.array([[2.0, 0.0]]), np.array([[0.0, 3.0]])])
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert np.allclose(out, [[2 ** -0.5, 2 ** -0.5]])


def test_the_shipped_action_list_is_kinetics_700():
    data = json.loads((REPO / "kinetics_700_labels.json").read_text(encoding="utf-8"))
    names = [data[k] for k in sorted(data, key=int)]
    assert len(names) == len(set(names)) == 700
    assert all(n.strip() == n and n for n in names)


@pytest.mark.skipif(not fe.has_text(), reason="no exported encoder with a text tower here")
def test_the_app_tokenizer_pads_and_ends_like_the_export():
    folder = fe.find_model_dir()
    ids = fe.tokenize(["Riding a bike", "x " * 300], f"{folder}/{fe.TEXT_TOKENIZER}")
    assert ids.shape == (2, fe.TEXT_LENGTH) and ids.dtype == np.int64
    first = ids[0].tolist()
    assert first[first.index(1) + 1:] == [fe.PAD_ID] * (fe.TEXT_LENGTH - first.index(1) - 1)
    assert ids[1, -1] == 1            # cut to 64, still ending in the end token
