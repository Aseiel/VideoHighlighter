"""The taught-action head and its trainer, on synthetic vectors.

What is pinned here: frames are sampled across the whole clip; the cache
refuses another encoder's vectors; held-out folds never share a source video;
the head learns a separable problem and its ONNX export says the same thing;
and a class is trusted only when enough held-out hits vouch for it.

The suite shims torch and scikit-learn (conftest), so the tests that need
them run in a child process, which gets the real ones.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from model_training.action_head import features as Fx
from model_training.action_head import train as T
from model_training.action_head import trust as H

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Frames and the cache
# ---------------------------------------------------------------------------

def test_frames_are_spread_across_the_whole_clip():
    assert Fx.sample_indices(150, 4) == [18, 56, 93, 131]
    assert Fx.sample_indices(3, 4) == [0, 1, 1, 2]        # short clip: repeats, never past the end
    assert Fx.sample_indices(0, 2) == [0, 0]


def test_cache_round_trip_and_refusal(tmp_path):
    path = str(tmp_path / "c.npz")
    cache = Fx.FeatureCache(path, "enc-a", 4, 8)
    cache.put("a|1|2", np.ones((4, 8)))
    cache.save()
    again = Fx.FeatureCache(path, "enc-a", 4, 8)
    assert "a|1|2" in again
    np.testing.assert_allclose(again["a|1|2"], 1)
    assert "a|1|2" not in Fx.FeatureCache(path, "enc-b", 4, 8)    # another encoder
    assert "a|1|2" not in Fx.FeatureCache(path, "enc-a", 8, 8)    # another frame count


def test_clip_key_follows_the_file_not_the_dataset_location(tmp_path):
    clip = tmp_path / "train" / "x" / "v_temp_clip_1.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"12345")
    key = Fx.clip_key(str(clip), str(tmp_path))
    assert key.startswith("train/x/v_temp_clip_1.mp4|5|")
    clip.write_bytes(b"123456")
    assert Fx.clip_key(str(clip), str(tmp_path)) != key


class FakeEncoder:
    encoder_id, label, dims = "enc-a", "test", 8

    def __init__(self):
        self.calls = 0

    def encode_bgr(self, frames):
        self.calls += 1
        return np.full((len(frames), self.dims), float(len(frames)), np.float32)


def test_encode_uses_the_cache_and_skips_unreadable(monkeypatch, tmp_path):
    clips = []
    for name in ("a", "b", "bad"):
        p = tmp_path / f"{name}.mp4"
        p.write_bytes(name.encode())
        clips.append(str(p))
    monkeypatch.setattr(Fx, "read_frames",
                        lambda p, k: [] if "bad" in p else [np.zeros((4, 4, 3), np.uint8)] * k)
    cache = Fx.FeatureCache(str(tmp_path / "c.npz"), "enc-a", 4, 8)
    enc = FakeEncoder()
    feats, ok = Fx.encode_clips(clips, str(tmp_path), enc, cache, log=lambda m: None)
    assert ok.tolist() == [True, True, False] and enc.calls == 2
    np.testing.assert_allclose(feats[0], 4)
    cache = Fx.FeatureCache(str(tmp_path / "c.npz"), "enc-a", 4, 8)
    Fx.encode_clips(clips[:2], str(tmp_path), enc, cache, log=lambda m: None)
    assert enc.calls == 2                                   # both came from the cache


# ---------------------------------------------------------------------------
# Selecting clips and splitting by source video
# ---------------------------------------------------------------------------

class Clip:
    def __init__(self, split, labels, group):
        self.split, self.labels, self.group, self.path = split, labels, group, ""


def test_select_leaves_out_pairs_small_classes_and_test():
    clips = ([Clip("train", ("a",), f"v{i}") for i in range(6)]
             + [Clip("val", ("b",), "v9")] * 5
             + [Clip("train", ("c",), "v1")] * 2
             + [Clip("train", ("a", "b"), "v3")] * 4
             + [Clip("test", ("a",), "v7")] * 3)
    kept, notes = T.select_clips(clips, min_clips=5)
    assert len(kept) == 11 and {c.labels[0] for c in kept} == {"a", "b"}
    assert any("two or more" in n for n in notes) and any("c (2)" in n for n in notes)


# ---------------------------------------------------------------------------
# The head and the folds (real torch and sklearn, in a child process)
# ---------------------------------------------------------------------------

def _run_real(body: str, tmp_path) -> None:
    script = textwrap.dedent("""
        import sys
        import numpy as np
        try:
            import torch
            import sklearn.model_selection
        except Exception:
            sys.exit(77)
        from model_training.action_head import head as H
        from model_training.action_head import train as T

        def separable(n=240, classes=3, frames=4, dims=16, seed=0):
            rng = np.random.default_rng(seed)
            centres = rng.normal(size=(classes, dims)) * 3
            y = np.arange(n) % classes
            x = centres[y][:, None, :] + rng.normal(size=(n, frames, dims))
            return x.astype(np.float32), y
    """) + textwrap.dedent(body)
    env = dict(os.environ, PYTHONPATH=_ROOT, PYTHONIOENCODING="utf-8")
    done = subprocess.run([sys.executable, "-c", script], cwd=str(tmp_path), env=env,
                          capture_output=True, text=True, timeout=600)
    if done.returncode == 77:
        pytest.skip("needs torch and scikit-learn")
    assert done.returncode == 0, done.stdout + done.stderr


def test_folds_never_share_a_source_video(tmp_path):
    _run_real("""
        groups = np.array([f"v{i // 10}" for i in range(300)])
        y = np.array([i // 10 % 3 for i in range(300)])
        splits = T.group_folds(y, groups, 5, seed=0)
        assert len(splits) == 5
        for tr, te in splits:
            assert not set(groups[tr]) & set(groups[te])
    """, tmp_path)


def test_head_learns_and_any_frame_count_works(tmp_path):
    _run_real("""
        x, y = separable()
        model = H.train_head(x[:180], y[:180], 3, steps=300)
        assert np.mean(H.predict_proba(model, x[180:]).argmax(1) == y[180:]) > 0.95
        assert H.predict_proba(model, x[180:, :2]).shape == (60, 3)   # 2 frames instead of 4
    """, tmp_path)


def test_standardisation_is_inside_the_model(tmp_path):
    _run_real("""
        x, y = separable()
        model = H.train_head(x * 50 + 7, y, 3, steps=200)
        np.testing.assert_allclose(model.mean.numpy(), (x * 50 + 7).reshape(-1, 16).mean(0),
                                   rtol=1e-4)
    """, tmp_path)


def test_onnx_export_matches(tmp_path):
    pytest.importorskip("onnxruntime")
    _run_real("""
        x, y = separable()
        model = H.train_head(x, y, 3, steps=100)
        H.export_onnx(model, "head.onnx", frames=4)
        session = H.load_onnx_session("head.onnx")
        np.testing.assert_allclose(H.onnx_proba(session, x[:10]),
                                   H.predict_proba(model, x[:10]), atol=1e-5)
        assert H.onnx_proba(session, x[:3, :2]).shape == (3, 3)      # free frame count
    """, tmp_path)


def test_class_weights_are_softened(tmp_path):
    _run_real("""
        w = H.class_weights(np.array([0] * 90 + [1] * 10), 2)
        assert abs(w[1] / w[0] - 3.0) < 1e-9                # sqrt of the 9x imbalance
    """, tmp_path)


# ---------------------------------------------------------------------------
# Trust
# ---------------------------------------------------------------------------

def test_wilson_bound_needs_numbers():
    assert H.wilson_low(3, 3) < 0.7 < H.wilson_low(30, 30)


def test_threshold_is_where_precision_still_holds():
    # Class 0: 40 confident right answers, then 20 wrong ones at lower confidence.
    conf = np.concatenate([np.linspace(0.99, 0.80, 40), np.linspace(0.6, 0.5, 20)])
    proba = np.stack([conf, 1 - conf], 1)
    y = np.array([0] * 40 + [1] * 20)
    th = H.trust_thresholds(proba, y, target=0.7)
    assert 0.5 < th[0] <= 0.80
    assert th[1] is None                                    # never predicted
    assert H.trusted_mask(proba, th).sum() >= 40


def test_a_class_right_by_luck_is_not_trusted():
    proba = np.array([[0.9, 0.1]] * 2 + [[0.2, 0.8]] * 50)
    y = np.array([0, 0] + [1] * 50)
    assert H.trust_thresholds(proba, y, target=0.7)[0] is None


# ---------------------------------------------------------------------------
# The test/ folder: single clips and the two-action confusion test
# ---------------------------------------------------------------------------

def test_score_test_two_actions_and_unknown_labels():
    classes = ["a", "b", "c"]
    clips = [Clip("test", ("a", "b"), "v1"), Clip("test", ("a", "c"), "v2"),
             Clip("test", ("a", "x"), "v3"), Clip("test", ("c",), "v4")]
    proba = np.array([[0.5, 0.4, 0.1],     # top two are exactly a, b
                      [0.1, 0.8, 0.1],     # top guess b is neither a nor c
                      [0.6, 0.2, 0.2],     # names a class the head does not have
                      [0.1, 0.1, 0.8]])    # single clip, right
    scores = T.score_test(proba, clips, classes, [0.3, 0.3, 0.3])
    assert scores["single"] == {"clips": 1, "accuracy": 1.0}
    two = scores["two_actions"]
    assert two["clips"] == 2 and two["left_out_unknown_class"] == 1
    assert two["top1_is_one_of_them"] == 0.5 and two["top2_are_both"] == 0.5
    assert two["trusted_into_one_of_them"] == 0.5


def test_score_test_skips_clips_no_head_could_score():
    clips = [Clip("test", ("a", "b"), "v1")]
    proba = np.full((1, 2), np.nan)
    assert T.score_test(proba, clips, ["a", "b"], [None, None]) == {}
