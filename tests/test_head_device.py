"""Where a head trains: the graphics card when there is one, back on the CPU after.

Measured 2026-11-19 on an Arc A750 (1,900 clips, 1,500 steps): 36 vectors per
clip took 98 s on the processor and 13 s on the card. The head that comes back
is always on the CPU, where it is scored and exported.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch  # the suite's shim (conftest); real training runs in a child process

from model_training.action_head import head as H
from model_training.action_head import train as T

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cards(monkeypatch, xpu, cuda):
    if not hasattr(torch, "xpu"):
        monkeypatch.setattr(torch, "xpu", type("X", (), {"is_available": staticmethod(lambda: xpu)}),
                            raising=False)
    else:
        monkeypatch.setattr(torch.xpu, "is_available", lambda: xpu)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)


@pytest.mark.parametrize("xpu,cuda,want", [(True, True, "xpu"), (False, True, "cuda"),
                                            (False, False, "cpu")])
def test_auto_takes_intel_then_nvidia_then_the_processor(monkeypatch, xpu, cuda, want):
    _cards(monkeypatch, xpu, cuda)
    assert H.head_device() == want and H.head_device("auto") == want and H.head_device(None) == want


def test_a_device_named_that_is_not_there_is_refused(monkeypatch):
    _cards(monkeypatch, False, False)
    assert H.head_device("cpu") == "cpu"
    with pytest.raises(ValueError, match="cannot use the device"):
        H.head_device("xpu")
    with pytest.raises(ValueError, match="cannot use the device"):
        H.head_device("sideways")


def _data(n=24, tokens=3, dims=8):
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, tokens, dims)).astype(np.float32), np.arange(n) % 2


def _run_real(body: str, tmp_path) -> None:
    """``body`` with the real torch, in a child process (77 = no torch: skip)."""
    script = textwrap.dedent("""
        import sys
        import numpy as np
        try:
            import torch
        except Exception:
            sys.exit(77)
        from model_training.action_head import head as H
        rng = np.random.default_rng(0)
        x = rng.normal(size=(24, 3, 8)).astype(np.float32)
        y = np.arange(24) % 2
    """) + textwrap.dedent(body)
    env = dict(os.environ, PYTHONPATH=_ROOT, PYTHONIOENCODING="utf-8")
    done = subprocess.run([sys.executable, "-c", script], cwd=str(tmp_path), env=env,
                          capture_output=True, text=True, timeout=600)
    if done.returncode == 77:
        pytest.skip("needs torch (or, where said, an Intel graphics card)")
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_trained_head_comes_back_on_the_cpu(tmp_path):
    _run_real("""
        m = H.train_head(x, y, 2, steps=4, device="cpu")
        assert all(p.device.type == "cpu" for p in m.parameters())
        assert H.predict_proba(m, x).shape == (len(x), 2)
    """, tmp_path)


def test_a_head_trained_on_the_card_is_scored_on_the_cpu(tmp_path):
    _run_real("""
        if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
            sys.exit(77)
        m = H.train_head(x, y, 2, steps=4, device="xpu")
        assert all(p.device.type == "cpu" for p in m.parameters())
        assert np.isfinite(H.predict_proba(m, x)).all()
    """, tmp_path)


def test_the_trainer_trains_its_fold_heads_on_the_device_it_was_given(monkeypatch):
    seen = []

    def fake_train(x, targets, n, *, steps, seed, device=None):
        seen.append(device)
        return "model"

    monkeypatch.setattr(H, "train_head", fake_train)
    monkeypatch.setattr(H, "predict_proba", lambda m, x: np.full((len(x), 2), 0.5, np.float32))
    x, y = _data(8)
    targets = np.eye(2, dtype=np.float32)[y]
    splits = [(np.arange(4), np.arange(4, 8)), (np.arange(4, 8), np.arange(4))]
    T.out_of_fold(x, targets, np.arange(8), splits, 5, 0, lambda m: None, device="xpu")
    assert seen == ["xpu", "xpu"]


def test_the_trainer_refuses_a_device_it_cannot_use(monkeypatch):
    _cards(monkeypatch, False, False)
    logs = []
    assert T.main(["--data-path", "nowhere", "--device", "xpu"], log=logs.append) == 1
    assert any("cannot use the device" in m for m in logs)
