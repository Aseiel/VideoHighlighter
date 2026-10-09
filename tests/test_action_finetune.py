"""Fine-tuning the image tower in the action trainer (``--finetune-blocks``).

What is pinned here: the app's preprocessing is unchanged by splitting it in
two; the frame cache stores the frames at the positions it says, decodes an
edited clip again, leaves out a clip it cannot read and refuses a full disk;
LP-FT trains only the top blocks, the tower's pooling and the head, at the
learning rates the recipe sets; a tower that is not the app's encoder is
refused before training; the saved folder is one the app loads, with an
encoder id of its own; the fine-tuned model is saved only when it beats the
frozen head; and the device rules (no DirectML, the processor only when asked).

The suite shims torch, OpenCV and transformers (conftest), so everything that
needs them runs in a child process with a tiny random SigLIP tower (3 blocks,
32 numbers wide, 256 px) on the processor. Class names are made up.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from model_training.action_head import train as T

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PRELUDE = """
import os, sys
import numpy as np
try:
    import cv2
    import torch
    import transformers
    import sklearn.model_selection
    import onnx, onnxruntime
except Exception:
    sys.exit(77)
from modules.vision import frame_encoder as fe

DIMS = 32
fe.DIMS = DIMS                      # the tiny tower's width stands in for 768

def tiny_tower(seed=0):
    from transformers import SiglipVisionConfig, SiglipVisionModel
    torch.manual_seed(seed)
    cfg = SiglipVisionConfig(hidden_size=DIMS, intermediate_size=64, num_hidden_layers=3,
                             num_attention_heads=2, image_size=256, patch_size=16)
    return SiglipVisionModel(cfg).vision_model.float().eval()

def write_clip(path, n=24, colour=None, size=(96, 64)):
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 12, size)
    for i in range(n):
        frame = np.zeros((size[1], size[0], 3), np.uint8)
        frame[:] = colour if colour is not None else (i * 10 % 256, 0, 0)
        w.write(frame)
    w.release()
"""


def _run(body: str, tmp_path, timeout=900, prefix: str = "") -> str:
    script = textwrap.dedent(_PRELUDE) + prefix + textwrap.dedent(body)
    env = dict(os.environ, PYTHONPATH=_ROOT, PYTHONIOENCODING="utf-8")
    done = subprocess.run([sys.executable, "-c", script], cwd=str(tmp_path), env=env,
                          capture_output=True, text=True, timeout=timeout)
    if done.returncode == 77:
        pytest.skip("needs torch, transformers, OpenCV, onnx and scikit-learn")
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def test_preprocess_is_scaling_of_prepare_frame(tmp_path):
    """The split leaves the app's preprocessing bit for bit as it was."""
    _run("""
        def old(frames):
            out = np.empty((len(frames), 3, 256, 256), np.float32)
            for i, frame in enumerate(frames):
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                h, w = frame.shape[:2]
                s = 384 / min(h, w)
                if s < 1:
                    frame = cv2.resize(frame, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
                frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_LINEAR)
                rgb = frame[:, :, ::-1].astype(np.float32)
                out[i] = (rgb / 255.0 - 0.5).transpose(2, 0, 1) / 0.5
            return out
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 256, (720, 1280, 3), dtype=np.uint8),
                  rng.integers(0, 256, (200, 300), dtype=np.uint8),
                  rng.integers(0, 256, (480, 640, 4), dtype=np.uint8)]
        assert np.array_equal(fe.preprocess(frames), old(frames))
        u8 = np.stack([fe.prepare_frame(f) for f in frames])
        assert u8.dtype == np.uint8 and u8.shape == (3, 256, 256, 3)
        assert np.array_equal(fe.scale_pixels(u8), fe.preprocess(frames))
        # The trainer's on-card scaling is the same numbers.
        from model_training.action_head import finetune as FT
        np.testing.assert_allclose(FT.to_device_pixels(u8, "cpu").numpy(),
                                   fe.scale_pixels(u8), atol=1e-6)
    """, tmp_path)


def test_frame_cache_slots_rebuild_and_unreadable(tmp_path):
    _run("""
        from model_training.action_head import features as Fx
        from model_training.action_head import frames as FR

        # Frame i is blue 10 * i (BGR when written, RGB when cached): the colour
        # says which frame a slot holds.
        os.makedirs("data")
        write_clip("data/a.mp4", n=24)
        write_clip("data/b.mp4", n=24, colour=(0, 200, 0))
        open("data/bad.mp4", "wb").write(b"not a video")
        paths = [os.path.abspath(f"data/{n}.mp4") for n in ("a", "b", "bad")]
        keys = [Fx.clip_key(p, "data") for p in paths]
        lines = []
        cache = FR.build(os.path.abspath("cache"), paths, keys, log=lines.append, workers=2)
        assert cache.ok.tolist() == [True, True, False], lines
        assert any("Could not read bad.mp4" in l for l in lines)
        blue = cache.read([cache.row(keys[0])])[0][:, 128, 128, 2].astype(int)
        want = [Fx.sample_indices(24, 4), Fx.sample_indices(24, 8)]
        assert np.all(np.abs(blue - 10 * np.array(want[0] + want[1])) <= 8), blue
        np.testing.assert_allclose(FR.SLOT_POSITIONS[:4], [0.125, 0.375, 0.625, 0.875])
        cache.close()

        # Everything there: nothing decoded.
        calls = []
        real = FR.decode_clip
        FR.decode_clip = lambda p: calls.append(p) or real(p)
        again = FR.build(os.path.abspath("cache"), paths[:2], keys[:2], log=lines.append)
        assert calls == [] and len(again) == 3
        again.close()

        # An edited clip is decoded again; the other is copied, not decoded.
        write_clip("data/b.mp4", n=30, colour=(0, 0, 200))
        keys[1] = Fx.clip_key(paths[1], "data")
        third = FR.build(os.path.abspath("cache"), paths[:2], keys[:2], log=lines.append)
        assert calls == [paths[1]], calls
        assert len(third) == 2 and third.ok.all()
        assert third.read([third.row(keys[1])])[0][0, 128, 128, 0] > 150   # the new colour (red)
        third.close()

        # A disk too small for the cache: refused with a sentence, nothing written.
        FR._free_bytes = lambda folder: 10
        lines.clear()
        assert FR.build(os.path.abspath("cache2"), paths[:2], ["x", "y"], log=lines.append) is None
        assert "GB free" in lines[-1] and not os.path.exists("cache2")
    """, tmp_path)


# ---------------------------------------------------------------------------
# LP-FT
# ---------------------------------------------------------------------------

def test_only_the_top_blocks_pooling_and_head_change(tmp_path):
    _run("""
        from model_training.action_head import finetune as FT
        from model_training.action_head import frames as FR
        from model_training.action_head import head as H

        class Cache:                                   # 8 clips of random frames
            rng = np.random.default_rng(0)
            data = rng.integers(0, 256, (8, FR.SLOTS, 256, 256, 3), dtype=np.uint8)
            def read(self, rows, slots=None):
                out = self.data[np.asarray(rows)]
                return out if slots is None else out[:, slots]

        base = tiny_tower()
        before = {k: v.clone() for k, v in base.state_dict().items()}
        head0 = H.ActionHead(DIMS, 2)
        head_before = {k: v.clone() for k, v in head0.state_dict().items()}
        targets = np.eye(2, dtype=np.float32)[np.arange(8) % 2]
        s = FT.Settings(blocks=1, epochs=2, lr=1e-2, head_lr=1e-2, batch=4)
        seen = []
        tower, head, res = FT.finetune(base, head0, Cache(), np.arange(8), targets, s, "cpu",
                                       log=lambda m: None,
                                       eval_fn=lambda t, h: seen.append(1) or (None, 0.5))
        assert len(res) == 2 and len(seen) == 2
        after = tower.vm.state_dict()
        changed = {k for k in before if not torch.equal(before[k], after[k])}
        assert changed, "nothing trained"
        for k in changed:
            assert (k.startswith("encoder.layers.2.") or k.startswith("post_layernorm.")
                    or k.startswith("head.")), k
        assert any(k.startswith("encoder.layers.2.") for k in changed)
        assert any(k.startswith("head.") for k in changed)
        assert all(torch.equal(before[k], v) for k, v in base.state_dict().items())  # a copy trained
        assert any(not torch.equal(head_before[k], v) for k, v in head.state_dict().items())

        # The learning rates follow the decay: 0 = the top block.
        t = FT.Tower(tiny_tower(), 3)
        groups = FT.param_groups(t, H.ActionHead(DIMS, 2),
                                 FT.Settings(blocks=3, lr=1.0, decay=0.5, head_lr=0.1))
        assert [g["lr"] for g in groups] == [0.25, 0.5, 1.0, 1.0, 0.1]
        assert FT.lr_factor(0, 100) == 0 and FT.lr_factor(6, 100) == 1.0
        assert FT.lr_factor(100, 100) < 1e-6

        # Views: 4 of the 12 slots, in time order.
        slots = FT.pick_slots(np.random.default_rng(0), 50)
        assert slots.shape == (50, 4)
        assert all(np.all(np.diff(FR.SLOT_POSITIONS[r]) > 0) for r in slots)
    """, tmp_path)


def test_start_check_refuses_another_tower(tmp_path):
    _run("""
        from model_training.action_head import finetune as FT
        mine = tiny_tower(0)
        probe = FT.tower_vector(mine, fe.probe_pixels())[0]
        assert FT.start_check(mine, probe, "cpu") > 0.9999
        try:
            FT.start_check(tiny_tower(1), probe, "cpu")
        except ValueError as e:
            assert "not the app's encoder" in str(e)
        else:
            raise AssertionError("another tower passed the start check")
    """, tmp_path)


def test_device_rules(tmp_path):
    _run("""
        from model_training.action_head import finetune as FT
        for name in ("directml", "dml", "privateuseone:0"):
            try:
                FT.resolve_device(name)
            except FT.DeviceError as e:
                assert "DirectML" in str(e)
            else:
                raise AssertionError(name)
        assert FT.resolve_device("cpu") == "cpu"
        torch.cuda.is_available = lambda: False
        if hasattr(torch, "xpu"):
            torch.xpu.is_available = lambda: False
        try:
            FT.resolve_device("auto")           # no card: never the processor unasked
        except FT.DeviceError as e:
            assert "--device cpu" in str(e)
        else:
            raise AssertionError("auto fell back to the processor")
        assert FT.autocast_dtype("cpu") is None
    """, tmp_path)


# ---------------------------------------------------------------------------
# The whole run: frozen first, fine-tuned only when it is better
# ---------------------------------------------------------------------------

_DATASET_AND_ENCODER = """
from model_training.action_head import finetune as FT
from model_training.action_head import train as T
from modules.vision import action_siglip as A

# Three classes told apart by colour, 4 source videos each.
data = os.path.abspath("data")
colours = {"alpha": (255, 0, 0), "beta": (0, 255, 0), "gamma": (0, 0, 255)}
for split, videos in (("train", range(4)), ("val", range(4, 5))):
    for cls, colour in colours.items():
        os.makedirs(os.path.join(data, split, cls))
        for v in videos:
            for i in range(2):
                write_clip(os.path.join(data, split, cls, f"{cls}{v}_temp_{i}.mp4"), n=16,
                           colour=colour)

shared = tiny_tower(0)

class Encoder:
    encoder_id, dims, label, folder = fe.ENCODER_ID, DIMS, "test", ""
    def encode_pixels(self, px):
        return FT.tower_vector(shared, np.asarray(px, np.float32))
    def encode_bgr(self, frames):
        return self.encode_pixels(fe.preprocess(frames))
    def close(self):
        pass

fe.load = lambda backend=None, log=print: Encoder()
FT.load_base_tower = lambda source, revision: tiny_tower(0)

def run(out, extra=()):
    lines = []
    code = T.main(["--data-path", data, "--out", out, "--steps", "60", "--folds", "3",
                   "--min-videos", "1", "--cache", os.path.abspath("feat.npz"),
                   "--frame-cache", os.path.abspath("frames"), "--device", "cpu",
                   "--finetune-blocks", "1", "--ft-epochs", "2", "--ft-batch", "4", *extra],
                  log=lines.append)
    return code, lines
"""


def test_the_fine_tuned_model_is_saved_when_it_is_better(tmp_path):
    _run("""
        real = T._finetune_folds
        def better(*a, **k):
            r = real(*a, **k)
            r["accuracy"] += 1.0
            return r
        T._finetune_folds = better
        out = os.path.abspath("models/tuned")
        code, lines = run(out)
        assert code == 0, "\\n".join(lines)
        assert T.SAVED_FINETUNED in lines, "\\n".join(lines)
        assert sorted(os.listdir(out)) == ["head.json", "head.onnx", "vision.onnx"]
        meta = A.read_head_meta(out)                 # the app's own reader
        assert meta["encoder"] != fe.ENCODER_ID and meta["encoder"].endswith("-finetuned")
        assert meta["own_encoder"]["preprocess"] == fe.ENCODER_ID
        assert meta["finetune"]["blocks"] == 1 and meta["finetune"]["epochs"] in (1, 2)
        assert "single" in meta["finetune"]["frozen_heldout"]
        assert not os.path.exists(out + ".export")   # the fp32 export is gone
        assert any("fine-tune fold 3/3" in l for l in lines)
        assert any("Frozen image model, on unseen videos" in l for l in lines)
        # Nothing in the model names a file or holds a frame.
        text = open(os.path.join(out, "head.json"), encoding="utf-8").read()
        assert ".mp4" not in text and "_temp_" not in text
    """, tmp_path, prefix=_DATASET_AND_ENCODER)


def test_the_frozen_head_is_saved_when_the_fine_tune_is_not_better(tmp_path):
    _run("""
        real = T._finetune_folds
        def worse(*a, **k):
            r = real(*a, **k)
            r["accuracy"] = -1.0
            return r
        T._finetune_folds = worse
        out = os.path.abspath("models/kept")
        os.makedirs(out)
        open(os.path.join(out, "vision.onnx"), "wb").write(b"an older fine-tuned tower")
        code, lines = run(out)
        assert code == 0, "\\n".join(lines)
        assert T.SAVED_FROZEN in lines
        assert any("did not beat the frozen one" in l for l in lines)
        assert sorted(os.listdir(out)) == ["head.json", "head.onnx"]   # the stale tower went
        meta = A.read_head_meta(out)
        assert meta["encoder"] == fe.ENCODER_ID and "own_encoder" not in meta
        assert meta["finetune_not_saved"]["blocks"] == 1

        # A second run reuses the frame cache: nothing decoded.
        code, lines = run(os.path.abspath("models/again"))
        assert code == 0 and any("already decoded" in l for l in lines), "\\n".join(lines)
    """, tmp_path, prefix=_DATASET_AND_ENCODER)


def test_frozen_runs_write_the_same_head_json_as_before(tmp_path):
    """Without --finetune-blocks the trainer is today's: the same head.json
    fields, nothing about fine-tuning, no frame cache."""
    _run("""
        out = os.path.abspath("models/frozen")
        lines = []
        code = T.main(["--data-path", data, "--out", out, "--steps", "60", "--folds", "3",
                       "--min-videos", "1", "--cache", os.path.abspath("feat.npz")],
                      log=lines.append)
        assert code == 0, "\\n".join(lines)
        meta = A.read_head_meta(out)
        assert sorted(meta) == sorted([
            "format", "kind", "encoder", "frames", "input", "output", "activation", "classes",
            "trust_thresholds", "pair_thresholds", "trust_precision", "trust_min_videos",
            "steps", "heldout", "test", "val_folder", "per_class", "created"]), sorted(meta)
        assert not os.path.exists("frames") and sorted(os.listdir(out)) == ["head.json", "head.onnx"]
    """, tmp_path, prefix=_DATASET_AND_ENCODER)


# ---------------------------------------------------------------------------
# Choosing the epoch (numpy only)
# ---------------------------------------------------------------------------

def test_the_epoch_is_the_best_mean_held_out_over_all_folds():
    """Fold 1 peaks at epoch 1 and fold 2 at epoch 3, but over both folds
    epoch 2 is best: the choice is over all held-out clips together."""
    targets = np.eye(2, dtype=np.float32)[[0, 1, 0, 1, 0, 1, 0, 1]]
    folds = [(np.array([4, 5, 6, 7]), np.array([0, 1, 2, 3])),
             (np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7]))]
    right = lambda n: np.array([[1, 0], [0, 1], [1, 0], [0, 1]][:n] +  # noqa: E731
                               [[0, 1], [1, 0], [0, 1], [1, 0]][n:], np.float32)
    # Clips right per epoch: fold 1 -> 4, 3, 0; fold 2 -> 0, 3, 4 (so 4, 6, 4 in all).
    per_fold = [[right(4), right(3), right(0)], [right(0), right(3), right(4)]]

    class FakeFT:
        calls = 0

        @staticmethod
        def finetune(base, head0, cache, rows, t, s, device, *, log, check, eval_fn):
            k = FakeFT.calls
            FakeFT.calls += 1
            return None, None, [((p, None), 0.0) for p in per_fold[k]]

        @staticmethod
        def empty_cache(device):
            pass

    class S:
        epochs, blocks = 3, 1

    out = T._finetune_folds(FakeFT, S, "cpu", None, [None, None], folds, None,
                            np.arange(8), np.zeros(0, int), targets,
                            np.array(["v1"] * 4 + ["v2"] * 4), np.zeros(0, str),
                            targets.sum(1) == 1, log=lambda m: None, should_stop=None)
    assert out["epochs"] == 2
    assert out["by_epoch"] == [0.5, 0.75, 0.5]
    np.testing.assert_array_equal(out["scores"][:4], right(3))
