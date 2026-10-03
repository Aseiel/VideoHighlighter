"""The head that names a clip's actions from its frame vectors, and how it trains.

The recipe is the one measured on held-out source videos
(``docs/plans/2026-10-01-action-models-measured.md`` and
``2026-10-02-frame-encoder-runtime.md``), not a guess:

- **Per frame, then attention over frames.** Each frame vector is normalised,
  projected, and the frames are pooled with learned weights, so the head works
  for any number of frames and finds the one that shows the action.
- **A score per action, each on its own** (cosine against a unit-length
  class centre, scale 16, plus a learned bias, through a sigmoid). Scores do
  not share 100 %, so two actions can both be present. Measured: single-action
  accuracy is the same as with a softmax (0.534 either way), and taught
  two-action combinations get both actions into the top two on unseen videos
  (42 % against 0-2 % for one answer per clip).
- **Class weights at power 0.5.** Fully balanced weights pushed new footage
  into rare classes; none at all ignores them.
- **Frames dropped at random** in training (each with 1 in 4 odds, at least 2
  kept), label smoothing, AdamW with a one-cycle schedule.
- **Standardisation is part of the model.** The per-dimension mean and spread
  of the training vectors are stored in it, so the exported head takes raw
  encoder vectors.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = 384
EMBED = 128
DROPOUT = 0.3
SCALE = 16.0
BIAS_INIT = -8.0
FRAME_DROP = 0.25
BATCH = 128
LR = 2e-3
WEIGHT_DECAY = 0.05
LABEL_SMOOTHING = 0.05
CLASS_WEIGHT_POWER = 0.5
OPSET = 17


class ActionHead(nn.Module):
    def __init__(self, dims: int, n_classes: int, mean=None, std=None):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dims) if mean is None
                             else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.ones(dims) if std is None
                             else torch.as_tensor(std, dtype=torch.float32))
        self.proj = nn.Sequential(nn.LayerNorm(dims), nn.Dropout(DROPOUT), nn.Linear(dims, HIDDEN))
        self.mix = nn.Sequential(nn.GELU(), nn.Linear(HIDDEN, HIDDEN), nn.GELU())
        self.att = nn.Linear(HIDDEN, 1)
        self.out = nn.Linear(HIDDEN, EMBED)
        self.centres = nn.Parameter(torch.randn(n_classes, EMBED) * 0.02)
        self.bias = nn.Parameter(torch.tensor(BIAS_INIT))

    def embed(self, x):
        """[B, T, dims] encoder vectors -> [B, EMBED] unit-length clip embedding."""
        h = self.mix(self.proj((x - self.mean) / self.std))
        w = torch.softmax(self.att(h), dim=1)
        return F.normalize(self.out((w * h).sum(1)), dim=-1)

    def forward(self, x):
        """[B, T, dims] -> logits [B, classes]; a sigmoid gives each action's score."""
        return SCALE * self.embed(x) @ F.normalize(self.centres, dim=-1).T + self.bias


def as_targets(labels, n_classes: int) -> np.ndarray:
    """Labels -> multi-hot [N, classes] float32: either a multi-hot array
    already, or one class index per clip."""
    labels = np.asarray(labels)
    if labels.ndim == 2:
        return labels.astype(np.float32)
    out = np.zeros((len(labels), n_classes), np.float32)
    out[np.arange(len(labels)), labels.astype(int)] = 1
    return out


def class_weights(targets: np.ndarray, power: float = CLASS_WEIGHT_POWER) -> np.ndarray:
    """Per class, (clips / (classes x clips showing it)) ** power."""
    counts = targets.sum(0).clip(1)
    return (len(targets) / (targets.shape[1] * counts)) ** power


def train_head(x: np.ndarray, labels, n_classes: int, *, steps: int = 1500,
               seed: int = 0, power: float = CLASS_WEIGHT_POWER) -> ActionHead:
    """Train on ``x`` [N, T, dims] with ``labels``: one class index per clip,
    or multi-hot [N, classes] for clips showing several actions."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x = np.asarray(x, np.float32)
    targets = as_targets(labels, n_classes)
    flat = x.reshape(-1, x.shape[-1])
    model = ActionHead(x.shape[-1], n_classes, mean=flat.mean(0), std=flat.std(0) + 1e-5)
    opt = torch.optim.AdamW(model.parameters(), lr=LR / 2, weight_decay=WEIGHT_DECAY)
    xt, yt = torch.as_tensor(x), torch.as_tensor(targets)
    w = torch.as_tensor(class_weights(targets, power), dtype=torch.float32)
    # A clip's weight is the mean weight of the actions it shows.
    sample_w = (yt * w).sum(1) / yt.sum(1).clamp(min=1)
    smooth = yt * (1 - LABEL_SMOOTHING) + LABEL_SMOOTHING / n_classes
    per_epoch = (len(yt) + BATCH - 1) // BATCH
    epochs = max(1, steps // per_epoch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=epochs * per_epoch)
    for _ in range(epochs):
        model.train()
        for b in torch.as_tensor(rng.permutation(len(yt))).split(BATCH):
            xb = xt[b]
            keep = torch.as_tensor(rng.random(xb.shape[1]) > FRAME_DROP)
            if int(keep.sum()) >= 2:
                xb = xb[:, keep]
            loss = F.binary_cross_entropy_with_logits(model(xb), smooth[b], reduction="none")
            loss = (loss.sum(1) * sample_w[b]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return model.eval()


def predict_proba(model: ActionHead, x: np.ndarray, batch: int = 1024) -> np.ndarray:
    """Each action's score in [0, 1], independent of the others: [N, classes]."""
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(torch.sigmoid(model(torch.as_tensor(np.asarray(x[i:i + batch], np.float32)))))
    return torch.cat(out).numpy() if out else np.zeros((0, model.centres.shape[0]), np.float32)


def export_onnx(model: ActionHead, path: str, frames: int) -> None:
    """``features`` [N, T, dims] float32 -> ``logits`` [N, classes]; N and T free."""
    dims = model.mean.shape[0]
    dummy = torch.zeros(1, frames, dims)
    torch.onnx.export(model.eval(), (dummy,), path, input_names=["features"],
                      output_names=["logits"], opset_version=OPSET, dynamo=False,
                      dynamic_axes={"features": {0: "n", 1: "frames"}, "logits": {0: "n"}})


def load_onnx_session(path: str):
    """A CPU session for an exported head; the head is tiny, a GPU buys nothing."""
    import onnxruntime as ort
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


def onnx_proba(session, x: np.ndarray) -> np.ndarray:
    """Each action's score from an exported head (a sigmoid of its logits)."""
    logits = session.run(None, {"features": np.asarray(x, np.float32)})[0]
    return 1.0 / (1.0 + np.exp(-logits))
