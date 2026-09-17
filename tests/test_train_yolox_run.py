"""Tests for the in-process YOLOX training run.

Two halves, deliberately separated by cost.

The **arithmetic and file handling** need nothing installed: progress fractions,
time estimates, device choice, and the labels sidecar are all pure, and they
are what a progress bar and an export are built on.

The **wiring assertions** need torch and yolox, and are skipped without them.
They exist because two bugs in this module were invisible from the outside —
both produced a plausible-looking run that had silently done nothing — and
neither could have been caught by testing the pure parts.
"""

from __future__ import annotations

import json
import os

import pytest

from training.train_yolox_run import (
    DEFAULT_SIZE,
    Progress,
    SIZES,
    TrainingResult,
    default_workers,
    eta_seconds,
    read_class_names,
    resolve_device,
    write_labels_sidecar,
)


# ── time estimates ───────────────────────────────────────────────────────

def test_eta_extrapolates_linearly_from_what_has_run():
    # A quarter done in 10s => 30s left.
    assert eta_seconds(done=25, total=100, elapsed=10.0) == pytest.approx(30.0)


def test_eta_is_zero_before_there_is_evidence():
    # An estimate from nothing swings wildly in the first seconds, which reads
    # as a broken bar rather than an honest unknown.
    assert eta_seconds(done=0, total=100, elapsed=5.0) == 0.0
    assert eta_seconds(done=10, total=100, elapsed=0.0) == 0.0


def test_eta_is_zero_once_finished():
    assert eta_seconds(done=100, total=100, elapsed=42.0) == 0.0
    assert eta_seconds(done=101, total=100, elapsed=42.0) == 0.0


# ── progress ─────────────────────────────────────────────────────────────

def _progress(**kw):
    base = dict(epoch=1, total_epochs=10, step=1, steps_per_epoch=5,
                loss=1.0, elapsed=0.0, eta=0.0)
    base.update(kw)
    return Progress(**base)


def test_fraction_spans_epochs_not_just_the_current_one():
    # Halfway through epoch 6 of 10 is 55%, not 50% and not 10%.
    assert _progress(epoch=6, step=3).fraction == pytest.approx(0.56, abs=0.01)


def test_fraction_starts_above_zero_and_ends_at_one():
    assert _progress(epoch=1, step=1).fraction == pytest.approx(0.02)
    assert _progress(epoch=10, step=5).fraction == 1.0


def test_fraction_never_exceeds_one():
    # A loader that yields more batches than len() promised must not push a
    # progress bar past its end.
    assert _progress(epoch=10, step=99).fraction == 1.0


def test_fraction_survives_a_degenerate_loader():
    assert _progress(steps_per_epoch=0, total_epochs=0).fraction <= 1.0


# ── device choice ────────────────────────────────────────────────────────

def test_an_explicit_device_is_honoured_without_probing():
    assert resolve_device("CPU") == "cpu"
    assert resolve_device("cuda:1") == "cuda:1"


def test_auto_prefers_intel_over_nvidia(monkeypatch):
    """Intel is probed first on purpose — see the docstring.

    This is the only detector-training path in the app, and the machine it was
    written for has an Arc and no NVIDIA card. A helper that reaches for CUDA
    first answers "cpu" there and silently turns a twenty-minute run into an
    overnight one.
    """
    import types
    fake = types.SimpleNamespace(
        xpu=types.SimpleNamespace(is_available=lambda: True),
        cuda=types.SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake)
    assert resolve_device("AUTO") == "xpu"


def test_auto_falls_back_through_cuda_to_cpu(monkeypatch):
    import types
    import sys
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        xpu=types.SimpleNamespace(is_available=lambda: False),
        cuda=types.SimpleNamespace(is_available=lambda: True),
    ))
    assert resolve_device("AUTO") == "cuda"
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        xpu=types.SimpleNamespace(is_available=lambda: False),
        cuda=types.SimpleNamespace(is_available=lambda: False),
    ))
    assert resolve_device("AUTO") == "cpu"


def test_a_broken_device_probe_answers_cpu_rather_than_raising(monkeypatch):
    import types
    import sys

    def explode():
        raise RuntimeError("driver went away")

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        xpu=types.SimpleNamespace(is_available=explode),
        cuda=types.SimpleNamespace(is_available=explode),
    ))
    assert resolve_device("AUTO") == "cpu"


def test_windows_uses_no_dataloader_workers(monkeypatch):
    # No fork on Windows: every worker re-imports torch and yolox, and in a
    # frozen build re-runs the entry point.
    monkeypatch.setattr("platform.system", lambda: "Windows")
    assert default_workers() == 0
    monkeypatch.setattr("platform.system", lambda: "Linux")
    assert default_workers() > 0


# ── the labels sidecar ───────────────────────────────────────────────────

def test_class_names_come_back_in_category_id_order(tmp_path):
    # Not alphabetical: the model reports an index, and the sidecar is what
    # turns that index back into a name.
    ann = tmp_path / "train.json"
    ann.write_text(json.dumps({"categories": [
        {"id": 3, "name": "zulu"}, {"id": 1, "name": "yankee"},
        {"id": 2, "name": "xray"},
    ]}), encoding="utf-8")
    assert read_class_names(str(ann)) == ["yankee", "xray", "zulu"]


def test_the_sidecar_is_written_where_the_app_looks(tmp_path):
    path = tmp_path / "nested" / "labels.json"
    write_labels_sidecar(str(path), ["a", "b"])
    assert json.loads(path.read_text(encoding="utf-8")) == ["a", "b"]


def test_result_summary_is_ascii_only():
    """Consoles here run cp1250, where a stray em-dash raises and kills the run.

    Learned the hard way: an emoji in a progress print took down a training run
    at the first line of output.
    """
    result = TrainingResult(
        weights_path="w", labels_path="l", class_names=["a"],
        epochs_completed=3, best_val_loss=1.25, final_train_loss=1.5,
        seconds=90.0, device="xpu",
    )
    result.summary().encode("ascii")   # raises if a non-ASCII char crept in


# ── wiring that only a real torch can check ──────────────────────────────
#
# These run in a **subprocess**, not in this interpreter, and the reason is
# specific: `conftest` replaces `torch` with a MagicMock whenever it has not
# already been imported, and yolox reads `torch.__version__` at import time, so
# under the shim it raises and an `importorskip` guard sails straight past.
#
# The obvious fix — borrow the real module the way `conftest.real_opencv` does
# — cannot work for torch. Dropping it from `sys.modules` and importing it
# again re-runs its C extension's docstring registration, which raises
# `RuntimeError: function '_has_torch_function' already has a docstring` and
# then leaves the interpreter's torch in a state that breaks later tests too.
#
# A clean interpreter has neither problem, costs a few seconds, and keeps this
# file from touching machinery the rest of the suite depends on.

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _probe(body: str):
    """Run a snippet in a fresh interpreter; return the JSON it prints last."""
    import subprocess
    import sys

    code = "import json, warnings\nwarnings.filterwarnings('ignore')\n" + body
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=600)
    if done.returncode != 0:
        stderr = done.stderr or ""
        if "No module named" in stderr:
            pytest.skip(f"needs torch and yolox installed: {stderr.strip()[-200:]}")
        raise AssertionError(f"probe failed:\n{stderr[-2000:]}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_learning_rate_is_never_left_at_zero():
    """The bug that made this module look like it worked while doing nothing.

    ``Exp.get_optimizer`` builds the optimizer at ``warmup_lr``, which is 0
    whenever warmup is enabled — YOLOX's own trainer raises it from the
    scheduler on every iteration. Without that, every step runs at lr=0: the
    weights never move, the training loss still wanders because augmentation is
    random, and the only symptom is a validation loss identical to four decimal
    places.

    So this pins both halves: that the trap is still there (the optimizer does
    start at zero), and that the scheduler this module drives lifts it.
    """
    out = _probe("""
from training.train_yolox_run import _build_exp
exp = _build_exp("unused", num_classes=3, size="tiny",
                 image_size=(416, 416), batch_size=8, workers=0, epochs=6)
exp.basic_lr_per_img = 0.01 / 8
exp.get_model()                      # get_optimizer groups exp.model's params
opt = exp.get_optimizer(8)
sched = exp.get_lr_scheduler(exp.basic_lr_per_img * 8, 6)
print(json.dumps({"optimizer_lr": opt.param_groups[0]["lr"],
                  "first": sched.update_lr(1),
                  "last": sched.update_lr(36)}))
""")
    assert out["optimizer_lr"] == 0.0, (
        "YOLOX no longer starts at warmup_lr — the scheduler wiring in train() "
        "may now be redundant, but verify before removing it")
    assert out["first"] > 0.0, "the scheduler left the rate at zero"
    assert out["last"] > 0.0, "the rate collapsed to zero before the end"


def test_the_epoch_count_reaches_the_scheduler():
    """max_epoch shapes the cosine decay, so a stubbed value flattens it."""
    out = _probe("""
from training.train_yolox_run import _build_exp
exp = _build_exp("unused", num_classes=1, size="tiny",
                 image_size=(416, 416), batch_size=4, workers=0, epochs=42)
print(json.dumps({"max_epoch": exp.max_epoch, "warmup": exp.warmup_epochs}))
""")
    assert out["max_epoch"] == 42
    # Short runs must not spend a quarter of their life warming up.
    assert out["warmup"] <= 5


def test_validation_keeps_its_labels():
    """ValTransform zeroes the targets, so it must not reach the loss.

    YOLOX's ValTransform is built for inference: it returns ``np.zeros((1, 5))``
    for every image, because labels are the evaluator's business there. Feed
    that to the loss and every frame is scored against "nothing is here" — a
    number that looks plausible, barely moves, and measures nothing.
    """
    out = _probe("""
import numpy as np
from yolox.data import TrainTransform, ValTransform
image = np.zeros((416, 416, 3), dtype=np.uint8)
boxes = np.array([[10., 10., 60., 60., 0.]], dtype=np.float32)
_, dropped = ValTransform(legacy=False)(image, boxes.copy(), (416, 416))
_, kept = TrainTransform(max_labels=50, flip_prob=0.0, hsv_prob=0.0)(
    image, boxes.copy(), (416, 416))
print(json.dumps({"val_transform_keeps": bool(dropped.any()),
                  "ours_keeps": bool(kept.any())}))
""")
    assert out["val_transform_keeps"] is False, "ValTransform started keeping labels"
    assert out["ours_keeps"] is True, "the validation transform dropped the labels"


def test_every_offered_size_builds_a_model():
    out = _probe("""
from training.train_yolox_run import _build_exp, SIZES
sizes = {}
for size in SIZES:
    exp = _build_exp("unused", num_classes=3, size=size,
                     image_size=(416, 416), batch_size=2, workers=0, epochs=1)
    sizes[size] = sum(p.numel() for p in exp.get_model().parameters())
print(json.dumps(sizes))
""")
    assert set(out) == set(SIZES)
    assert all(count > 0 for count in out.values()), out
    # A mini model is the point: nano < tiny < s, and none of them large.
    assert out["nano"] < out["tiny"] < out["s"]


def test_an_unknown_size_is_refused_before_anything_expensive_happens():
    from training.train_yolox_run import train
    with pytest.raises(ValueError, match="size must be one of"):
        train(dataset_dir="nowhere", output_dir="nowhere", size="enormous")
