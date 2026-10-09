# Fine-tune only the top blocks, starting from the frozen head (LP-FT)

**Positive scenario.**
1. Train the usual head on frozen vectors.
2. From it, train the encoder's top blocks (4 of 12), its pooling head and
   the action head together.
3. Choose the training length on unseen videos.

**Why it helps:**
- The encoder's upper layers adapt to this footage.
- The lower layers, most of its general knowledge, stay as they were.
- Starting from a trained head means the first updates do not scramble the
  encoder.

**Where it shows most: sorting.** Top-1 rises 7 points. The share of clips
that are sorted with confidence rises 13 points, at the same precision.
That share is what decides how much of a new video gets a label without a
person checking it.

**Measured** (5 folds by source video, 2,695 clips, 38 classes,
`docs/plans/2026-10-08-overfitting-and-siglip2-fine-tune.md`):

| | top-1 | top-3 | sorted with confidence |
|---|---|---|---|
| frozen + head | 0.533 | 0.733 | 44 % at 74 % |
| top 4 blocks, lr 1e-4 | **0.603** | **0.781** | **57 % at 75 %** |
| top 8 blocks, lr 3e-5 | 0.586 | 0.769 | 52 % at 74 % |
| all 12 blocks, lr 2e-5 | 0.586 | 0.762 | 54 % at 75 % |

- Every fold of every setting beat the frozen head.
- Unfreezing more blocks found both actions of a pair more often (56 %
  against 46 %), but was not better on single actions.

**In the app, on a whole new video** (the top 4 model, video 001, 52 min,
never trained on; scored against a person's review of 1,203 windows,
`docs/plans/2026-10-08-fine-tuned-action-encoder.md`):

| | frozen head | fine-tuned |
|---|---|---|
| reviewed labels found | 19.1 % | **33.9 %** |
| precision | 55.3 % | 52.9 % |
| windows with any label | 476 | 760 |
| run time | 74 s | 72 s |

It found 1.8× as much at about the same precision and speed.

**How solid:**
- 5 folds, one seed.
- On a single 29-video split the gain was +2, not +7.
- One real video, checked against one person's review.

**The cost:**
- Training: about an hour on a GPU, against minutes. Fine-tuning is not
  part of the app's trainer yet.
- Using it: the model brings its own copy of the encoder (186 MB at fp16),
  because search and typed actions need the original. It runs instead of
  the shared encoder, not as well as it, so the time per window is the
  same. Install an export with `tools/install_action_model.py`.

**Related positive:** `frozen-encoder-first.md`, `trust-thresholds.md`.
**Related negative:** `../negative/training-whole-network.md`.
