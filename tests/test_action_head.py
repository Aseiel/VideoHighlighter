"""The taught-action head and its trainer, on synthetic vectors.

What is pinned here: frames are sampled across the whole clip; the cache
refuses another encoder's vectors; held-out folds never share a source video;
the head learns a separable problem and its ONNX export says the same thing;
and an action is trusted only when enough held-out hits vouch for it; a
taught pair scores both of its actions.

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


def _clips():
    return ([Clip("train", ("a",), f"v{i}") for i in range(6)]
            + [Clip("val", ("b",), "v9")] * 5
            + [Clip("train", ("c",), "v1")] * 2
            + [Clip("train", ("a", "b"), "v3")] * 4
            + [Clip("train", ("a", "c"), "v4")] * 2
            + [Clip("test", ("a", "b"), "v7")] * 3)


def test_select_keeps_pairs_of_known_classes_and_drops_small_classes():
    kept, classes, notes = T.select_clips(_clips(), min_clips=5)
    assert classes == ["a", "b"]
    assert len(kept) == 6 + 5 + 4                         # a, b, and the a+b pairs
    assert any("teach them together" in n for n in notes)
    assert any("c (2)" in n for n in notes)
    assert any("not a class here" in n for n in notes)    # the a+c pairs


def test_select_takes_test_only_when_asked():
    kept, _, _ = T.select_clips(_clips(), min_clips=5, splits=("train", "val", "test"))
    assert sum(c.split == "test" for c in kept) == 3


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
        p = H.predict_proba(model, x[180:])
        assert np.mean(p.argmax(1) == y[180:]) > 0.95
        assert ((p >= 0) & (p <= 1)).all()
        assert H.predict_proba(model, x[180:, :2]).shape == (60, 3)   # 2 frames instead of 4
    """, tmp_path)


def test_a_taught_pair_scores_both_actions(tmp_path):
    """Clips showing classes 0 and 1 together, taught as such: both score
    high, and class 2 does not."""
    _run_real("""
        x, y = separable(n=300)
        pair_x = (x[y == 0][:40] + x[y == 1][:40]) / 2
        targets = np.zeros((len(y) + 40, 3), np.float32)
        targets[np.arange(len(y)), y] = 1
        targets[len(y):, :2] = 1
        model = H.train_head(np.concatenate([x, pair_x]), targets, 3, steps=600)
        test = (x[y == 0][40:60] + x[y == 1][40:60]) / 2
        p = H.predict_proba(model, test)
        assert np.mean([set(np.argsort(-r)[:2]) == {0, 1} for r in p]) > 0.9
        assert (p[:, 2] < p[:, :2].min(1)).mean() > 0.9
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
        w = H.class_weights(H.as_targets(np.array([0] * 90 + [1] * 10), 2))
        assert abs(w[1] / w[0] - 3.0) < 1e-9                # sqrt of the 9x imbalance
    """, tmp_path)


# ---------------------------------------------------------------------------
# Trust and scoring (numpy only)
# ---------------------------------------------------------------------------

def test_wilson_bound_needs_numbers():
    assert H.wilson_low(3, 3) < 0.7 < H.wilson_low(30, 30)


def test_threshold_is_where_precision_still_holds():
    # Action 0: 40 clips that show it score 0.99-0.80, 20 that do not score 0.6-0.5.
    s0 = np.concatenate([np.linspace(0.99, 0.80, 40), np.linspace(0.6, 0.5, 20)])
    scores = np.stack([s0, np.full(60, 0.01)], 1)
    targets = np.zeros((60, 2))
    targets[:40, 0] = 1
    th = H.trust_thresholds(scores, targets, target=0.7)
    assert 0.5 < th[0] <= 0.80
    assert th[1] is None                                    # never shown, never trusted
    assert H.detected(scores, th)[:, 0].sum() >= 40


def test_an_action_right_by_luck_is_not_trusted():
    scores = np.array([[0.9, 0.1]] * 2 + [[0.2, 0.8]] * 50)
    targets = np.array([[1, 0]] * 2 + [[0, 1]] * 50)
    assert H.trust_thresholds(scores, targets, target=0.7)[0] is None


def test_score_clips_single_and_two_actions():
    scores = np.array([[0.9, 0.8, 0.1],     # pair {0,1}: both top 2, both detected
                       [0.2, 0.9, 0.1],     # pair {0,2}: top is neither, nothing right detected
                       [0.1, 0.1, 0.8]])    # single {2}: right, detected
    sets = [{0, 1}, {0, 2}, {2}]
    s = H.score_clips(scores, sets, [0.5, 0.5, 0.5])
    assert s["single"]["accuracy"] == 1.0 and s["single"]["trusted_precision"] == 1.0
    two = s["two_actions"]
    assert two["clips"] == 2
    assert two["both_in_top2"] == 0.5 and two["top1_is_one_of_them"] == 0.5
    assert two["both_detected"] == 0.5
    assert two["wrong_detected"] == 0.5                     # the second clip's action 1


def test_hits_from_too_few_videos_do_not_vouch():
    # 30 right answers above 0.8, but all from two source videos.
    scores = np.concatenate([np.linspace(0.99, 0.81, 30), np.linspace(0.3, 0.1, 30)])[:, None]
    targets = np.concatenate([np.ones(30), np.zeros(30)])[:, None]
    two_videos = np.array(["v1", "v2"] * 15 + ["w"] * 30)
    many_videos = np.array([f"v{i}" for i in range(30)] + ["w"] * 30)
    assert H.trust_thresholds(scores, targets, groups=two_videos, min_videos=3) == [None]
    assert H.trust_thresholds(scores, targets, groups=many_videos, min_videos=3)[0] is not None


def test_a_taught_pair_gets_its_own_threshold():
    rng = np.random.default_rng(0)
    # 20 clips show actions 0 and 1 together; their scores are middling (0.4-0.6),
    # well under the single-action thresholds. 40 clips show neither.
    pair = np.column_stack([rng.uniform(0.4, 0.6, 20), rng.uniform(0.4, 0.6, 20)])
    rest = rng.uniform(0.0, 0.3, (40, 2))
    scores = np.concatenate([pair, rest])
    targets = np.zeros((60, 2))
    targets[:20] = 1
    groups = np.array([f"v{i}" for i in range(60)])
    assert H.taught_pairs(targets) == [(0, 1)]
    pt = H.pair_thresholds(scores, targets, [(0, 1)], groups=groups)
    assert pt[0] is not None and pt[0] <= 0.4
    found = H.detected(scores, [0.9, 0.9], [(0, 1)], pt)
    assert found[:20].all()                                 # every pair found
    wrong = found[20:].any(1).sum()
    assert 20 / (20 + wrong) >= 0.7                         # at the precision asked for
