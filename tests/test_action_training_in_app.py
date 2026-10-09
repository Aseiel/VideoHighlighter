"""Train > Actions runs the action trainer inside the app, not as a child process.

It used to start ``sys.executable -m model_training.action_head.train``. From
source that is Python; in the packaged app it is the app itself, so pressing
"Train an action model" opened a second copy of the app, sat at 0 %, and the
copy rotated the first one's debug log away. What is pinned here: the worker
calls ``train.main`` in-process and never starts a process; the trainer's lines
reach ``log`` and drive the progress bar; a stop is honoured between steps and
saves nothing; a failure says why in a sentence.

Class names are made up.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# The trainer, called the way the app calls it (real torch, in a child process
# because the suite shims torch)
# ---------------------------------------------------------------------------

def test_trainer_runs_in_process_and_stops_when_asked(tmp_path):
    script = textwrap.dedent(r"""
        import os, sys
        import numpy as np
        try:
            import torch
            import sklearn.model_selection
        except Exception:
            sys.exit(77)
        from model_training.action_head import features as Fx
        from model_training.action_head import train as T
        from modules.vision import frame_encoder

        data = os.path.abspath("data")
        for split, per_video in (("train", 6), ("val", 2)):
            for cls in ("alpha", "beta", "gamma"):
                d = os.path.join(data, split, cls)
                os.makedirs(d)
                for v in range(5):
                    for i in range(per_video):
                        open(os.path.join(d, f"{split}{cls}{v}_temp_{i}.mp4"), "wb").close()

        class Encoder:
            encoder_id, dims, label = frame_encoder.ENCODER_ID, 16, "test"

        centres = np.random.default_rng(0).normal(size=(3, 16)) * 3
        def encode_clips(paths, root, encoder, cache, log=print, should_stop=None, **kw):
            rng = np.random.default_rng(1)
            y = [("alpha", "beta", "gamma").index(os.path.basename(os.path.dirname(p)))
                 for p in paths]
            x = centres[y][:, None, :] + rng.normal(size=(len(paths), 4, 16))
            return x.astype(np.float32), np.ones(len(paths), bool)

        frame_encoder.load = lambda backend=None, log=print: Encoder()
        Fx.encode_clips = encode_clips

        lines = []
        out = os.path.abspath("model")
        args = ["--data-path", data, "--out", out, "--steps", "60", "--folds", "3",
                "--min-videos", "1", "--cache", os.path.abspath("cache.npz")]
        code = T.main(args, log=lines.append)
        assert code == 0, lines
        assert os.path.isfile(os.path.join(out, "head.onnx"))
        assert os.path.isfile(os.path.join(out, "head.json"))
        assert any("fold 1/3" in l for l in lines), lines
        assert any("Saved" in l for l in lines), lines

        asked = []
        def stop():
            asked.append(1)
            return len(asked) > 2          # stop at the second fold
        out2 = os.path.abspath("stopped")
        code = T.main(args[:2] + ["--out", out2] + args[4:], log=lines.append,
                      should_stop=stop)
        assert code == T.STOPPED, code
        assert not os.path.exists(out2)
        assert "Stopped" in lines[-1], lines[-1]
    """)
    env = dict(os.environ, PYTHONPATH=_ROOT, PYTHONIOENCODING="utf-8")
    done = subprocess.run([sys.executable, "-c", script], cwd=str(tmp_path), env=env,
                          capture_output=True, text=True, timeout=600)
    if done.returncode == 77:
        pytest.skip("needs torch and scikit-learn")
    assert done.returncode == 0, done.stdout + done.stderr


# ---------------------------------------------------------------------------
# The GUI worker
# ---------------------------------------------------------------------------

@pytest.fixture
def worker_env(monkeypatch):
    pytest.importorskip("PySide6")
    # A QApplication, not a QCoreApplication: the suite shares one per
    # process, and widget tests that run later crash on a core-only one.
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from model_training.action_head import train
    from modules.ui import training_panel

    def no_process(*a, **k):
        raise AssertionError("Train > Actions must not start a process")
    monkeypatch.setattr(subprocess, "Popen", no_process)
    monkeypatch.setattr(subprocess, "run", no_process)
    return train, training_panel


def _run(worker):
    seen = {"progress": [], "finished": [], "error": []}
    worker.progress.connect(lambda p, m: seen["progress"].append((p, m)))
    worker.finished.connect(seen["finished"].append)
    worker.error.connect(seen["error"].append)
    worker.run()
    return seen


def test_worker_calls_the_trainer_in_process(worker_env, monkeypatch):
    train, panel = worker_env
    calls = []

    def fake_main(argv, *, log, should_stop):
        calls.append(argv)
        log("Encoding 20 clips (0 cached) with x on y")
        log("  10/20 clips, 5.0/s, about 0.1 min left")
        log("Scoring on unseen source videos (5 folds)")
        log("  750 steps")
        log("    fold 2/5: 0.500 on 10 single-action clips")
        log("    fold 5/5: 0.500 on 10 single-action clips")
        log("  1500 steps")
        log("    fold 1/5: 0.500 on 10 single-action clips")
        log("\nHeld out (whole source videos never seen), single action (40 clips): "
            "accuracy 0.600, balanced 0.500, top-3 0.900")
        return 0

    monkeypatch.setattr(train, "main", fake_main)
    worker = panel.ActionTrainingWorker(data_path="D:/somewhere/clips", name="clips")
    seen = _run(worker)
    assert calls == [["--data-path", "D:/somewhere/clips", "--name", "clips"]]
    assert seen["error"] == []
    assert seen["finished"] == ["recognised 60% of the clips from videos it had not seen"]
    percents = [p for p, _ in seen["progress"]]
    # Encoding fills 0-60; the folds of every training length it compares
    # share 60-95, so the bar never runs backwards between lengths.
    assert percents == [0, 30, 64, 71, 74, 100]
    assert seen["progress"][-2][1] == "Testing on videos it has not seen... 6 of 15"
    assert worker.out_dir.endswith(os.path.join("actions", "clips"))


def test_worker_fine_tune_adds_the_flag_and_its_phases(worker_env, monkeypatch):
    """With the image model checked: the flag goes to the trainer, the bar
    walks decode -> encode -> frozen folds -> fine-tune folds -> final run and
    never goes backwards, and the note says which model was saved."""
    train, panel = worker_env
    calls = []

    def fake_main(argv, *, log, should_stop):
        calls.append(argv)
        log("  50/100 clips decoded, 20.0/s, about 0.0 min left")
        log("  100/100 clips, 50.0/s, about 0.0 min left")
        log("Scoring on unseen source videos (5 folds)")
        log("  750 steps")
        log("    fold 5/5: 0.500 on 10 single-action clips")
        log("\nFine-tuning the top 4 image-model blocks on xpu, scored on unseen source "
            "videos (5 folds x 10 epochs)")
        log("  fine-tune fold 1/5: 80 clips to learn from, 20 held out")
        log("      epoch 5/10: loss 0.300, 60 s, held-out 0.600")
        log("  fine-tune fold 1/5: 0.600 after 10 epochs (best 0.600 at epoch 5), 600 s")
        log("  fine-tune fold 5/5: 80 clips to learn from, 20 held out")
        log("      epoch 10/10: loss 0.300, 60 s, held-out 0.600")
        log("\nFinal model: training the head on all 100 clips (750 steps), then "
            "fine-tuning with it (9 epochs)")
        log("      final epoch 9/9: loss 0.300, 60 s")
        log("\nHeld out (whole source videos never seen), single action (40 clips): "
            "accuracy 0.600, balanced 0.500, top-3 0.900")
        log(train.SAVED_FINETUNED)
        return 0

    monkeypatch.setattr(train, "main", fake_main)
    worker = panel.ActionTrainingWorker(data_path="d", name="n", finetune_blocks=4)
    seen = _run(worker)
    assert calls == [["--data-path", "d", "--name", "n", "--finetune-blocks", "4"]]
    percents = [p for p, _ in seen["progress"]]
    # decode 0-15, encode 15-25, frozen folds 25-35, fine-tune folds 35-90, final 90-99.
    assert percents == [0, 7, 25, 28, 35, 40, 90, 90, 99, 100], seen["progress"]
    assert "round 1 of 5, epoch 5 of 10" in seen["progress"][5][1]
    assert seen["finished"] == ["recognised 60% of the clips from videos it had not seen; "
                                "the image model was fine-tuned too"]

    def frozen_kept(argv, *, log, should_stop):
        log(train.SAVED_FROZEN)
        return 0

    monkeypatch.setattr(train, "main", frozen_kept)
    seen = _run(panel.ActionTrainingWorker(data_path="d", name="n", finetune_blocks=4))
    assert "small model was kept" in seen["finished"][0]


def test_image_model_checkbox_needs_a_card_pytorch_trains_on(worker_env):
    _, panel = worker_env
    section = panel.ActionTrainingSection()
    box = section.finetune_box
    assert not box.isEnabled() and not box.isChecked()          # off until the probe
    section._on_device_found("xpu", "Intel Arc A750")
    assert box.isEnabled() and "Intel Arc A750" in box.toolTip()
    box.setChecked(True)
    for device in ("privateuseone:0", "cpu"):                   # DirectML, processor
        section._on_device_found(device, "")
        assert not box.isEnabled() and not box.isChecked()
        assert "Intel Arc or NVIDIA" in box.toolTip()
    section._on_device_found("cuda", "NVIDIA")
    assert box.isEnabled() and not box.isChecked()              # never ticked for the user
    assert panel.can_finetune("cuda:0") and not panel.can_finetune("dml")
    section.close()


def test_worker_stop_is_passed_to_the_trainer(worker_env, monkeypatch):
    train, panel = worker_env

    def fake_main(argv, *, log, should_stop):
        assert not should_stop()
        worker.cancel()
        assert should_stop()
        return train.STOPPED

    monkeypatch.setattr(train, "main", fake_main)
    worker = panel.ActionTrainingWorker(data_path="d", name="n")
    seen = _run(worker)
    assert seen["error"] == ["Stopped."]
    assert seen["finished"] == []


def test_worker_failure_says_why(worker_env, monkeypatch):
    train, panel = worker_env

    def fake_main(argv, *, log, should_stop):
        log("❌ No class has enough single-action clips to train on")
        return 1

    monkeypatch.setattr(train, "main", fake_main)
    seen = _run(panel.ActionTrainingWorker(data_path="d", name="n"))
    assert seen["error"] == [
        "Training did not finish: No class has enough single-action clips to train on."]


# ---------------------------------------------------------------------------
# Train > From videos (a teach project's round) takes the same path
# ---------------------------------------------------------------------------

def test_a_teach_round_trains_actions_in_process(tmp_path, monkeypatch):
    import json

    from model_training.action_head import train as trainer
    from modules.teach import train as teach_train

    def no_process(*a, **k):
        raise AssertionError("a teach round must not start a process")
    monkeypatch.setattr(subprocess, "run", no_process)
    monkeypatch.setattr(subprocess, "Popen", no_process)

    seen = []

    def fake_main(argv, *, log, should_stop=None):
        seen.append(argv)
        log("    fold 1/5: 0.500 on 10 single-action clips")
        out = argv[argv.index("--out") + 1]
        os.makedirs(out)
        with open(os.path.join(out, "head.json"), "w") as fh:
            json.dump({"classes": ["alpha move"], "trust_thresholds": [0.5],
                       "heldout": {"single": {"balanced_accuracy": 0.5, "accuracy": 0.6}}}, fh)
        return 0

    monkeypatch.setattr(trainer, "main", fake_main)

    class Project:
        def path(self, name):
            return str(tmp_path / name)

    run_dir = tmp_path / "runs" / "001"
    run_dir.mkdir(parents=True)
    metrics = teach_train.train_actions(Project(), str(run_dir), epochs=1)
    assert seen and seen[0][:2] == ["--data-path", str(tmp_path / teach_train.DATASET_DIR)]
    assert metrics["accuracy"] == 0.6
    assert "fold 1/5" in (run_dir / "train.log").read_text(encoding="utf-8")
