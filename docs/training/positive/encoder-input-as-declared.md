# Feed each encoder exactly the input it declares

**Positive scenario.** Frames reach a pretrained encoder in the colour order,
value range, normalisation and resize its model card or IR metadata declares.
The same input is used in training and in analysis.

**Why it helps:**
- A pretrained encoder's knowledge is tied to the input it learned from.
- Matching that input gives the trained head the encoder's full signal.
- Matching training and analysis means the head sees at run time what it
  was taught on.

**Measured** (same 29 unseen videos, 445 clips,
`docs/plans/2026-10-01-action-models-measured.md`):
- The Intel action encoder's IR does its own mean subtraction and channel
  flip, so it expects BGR 0-255.
- Feeding it that instead of RGB normalised twice:

| | before | after |
|---|---|---|
| Intel trainer | 0.348 | 0.371 |
| small head on the same features | 0.365 | 0.407 |

- For SigLIP2, its own preprocessing (squash to 256²) against letterbox:
  - 0.582 against 0.535 on one split;
  - a tie on 5 folds (0.577 each).
  - Its own preprocessing is the default, because it is what the model was
    trained on.

**How solid:** the Intel gain held across every framing tried (+3-4 points
each). The SigLIP2 one-split gain did not hold on 5 folds.

**Where it lives:** `modules/vision/frame_encoder.py` `preprocess()` is used
both by the trainer (`model_training/action_head/features.py`) and by
analysis.

**Related negative:** `../negative/wrong-encoder-input.md`.
