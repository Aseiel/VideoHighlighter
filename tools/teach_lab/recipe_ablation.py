"""The Intel trainer's decoder and recipe, rebuilt on its own cached features,
then changed one thing at a time.

    python tools/teach_lab/recipe_ablation.py <checkpoint_dir of an intel trainer run> [--seeds 3]

Input is identical in every row (the trainer's feature caches: its crop, its
frames, its labels, its split). Rows:

  trainer recipe          EncoderMLP(16x512 -> 256 -> 128), AdamW 1e-4, batch 2,
                          cosine over 25 epochs, label smoothing 0.1, inverse-
                          frequency class weights, early stop on val loss
                          (patience 5) -- restoring the *last* epoch, as the
                          shallow state_dict().copy() does
  + best epoch restored   the same, keeping a real copy of the best epoch
  + standardised input    features scaled per dimension (train statistics)
  + class weights ^0.5    softened weights instead of inverse frequency
  + no early stop         a fixed 25 epochs, no peeking at validation
  head / logreg           the part on top swapped (decoder_ablation.py)

Note that early stopping picks its epoch on the very clips it is scored on, so
the trainer's rows are a little optimistic; "no early stop" is not.
"""
import argparse
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decoder_ablation import load  # noqa: E402

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model_training.intel.model import EncoderMLP  # noqa: E402


def run(xtr, ytr, xva, yva, *, restore_best, standardise, balance, early_stop, seed, epochs=25):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_cls = int(max(ytr.max(), yva.max())) + 1
    if standardise:
        mu = xtr.reshape(-1, xtr.shape[-1]).mean(0)
        sd = xtr.reshape(-1, xtr.shape[-1]).std(0) + 1e-5
        xtr, xva = (xtr - mu) / sd, (xva - mu) / sd
    xt, yt = torch.as_tensor(xtr, dtype=torch.float32), torch.as_tensor(ytr)
    xv, yv = torch.as_tensor(xva, dtype=torch.float32), torch.as_tensor(yva)
    m = EncoderMLP(feature_dim=xtr.shape[-1], hidden_dim=256, num_classes=n_cls,
                   sequence_length=xtr.shape[1], dropout=0.3)
    cnt = np.bincount(ytr, minlength=n_cls).clip(1)
    w = torch.as_tensor((len(ytr) / (n_cls * cnt)) ** balance, dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    best_loss, best_state, best_acc, patience = float("inf"), None, 0.0, 0
    for _ in range(epochs):
        m.train()
        for b in torch.randperm(len(yt)).split(2):
            out, _ = m(xt[b])
            loss = crit(out, yt[b])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        sched.step()
        m.eval()
        with torch.no_grad():
            out, _ = m(xv)
            vl = float(crit(out, yv))
            va = float((out.argmax(1) == yv).float().mean())
        if not early_stop:
            continue
        if vl < best_loss - 0.001:
            best_loss, best_acc, patience = vl, va, 0
            best_state = ({k: v.clone() for k, v in m.state_dict().items()} if restore_best
                          else m.state_dict().copy())          # the trainer's shallow copy
        else:
            patience += 1
            if patience >= 5:
                break
    if early_stop and best_state is not None:
        m.load_state_dict(best_state)
    m.eval()
    with torch.no_grad():
        return float((m(xv)[0].argmax(1) == yv).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    xtr, ytr, xva, yva = load(args.ckpt_dir)
    base = dict(restore_best=False, standardise=False, balance=1.0, early_stop=True)
    steps = [
        ("trainer recipe (as shipped)", {}),
        ("+ best epoch really restored", dict(restore_best=True)),
        ("+ standardised input", dict(restore_best=True, standardise=True)),
        ("+ class weights ^0.5", dict(restore_best=True, standardise=True, balance=0.5)),
        ("+ no early stop (fixed 25 epochs)", dict(standardise=True, balance=0.5, early_stop=False)),
        ("trainer recipe, only class weights ^0.5", dict(balance=0.5)),
        ("trainer recipe, only standardised input", dict(standardise=True)),
    ]
    print(f"trainer's features: train {xtr.shape}, val {xva.shape}; {args.seeds} seeds each\n")
    for name, change in steps:
        accs = [run(xtr, ytr, xva, yva, seed=s, **{**base, **change}) for s in range(args.seeds)]
        print(f"{name:42} {np.mean(accs):.3f}  (+-{np.std(accs):.3f})", flush=True)


if __name__ == "__main__":
    sys.exit(main())
