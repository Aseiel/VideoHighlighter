# Start with the encoder frozen

**Positive scenario.**
- The pretrained network is not trained at all.
- Only a small head on its vectors is.
- This is the baseline every other training choice has to beat, on unseen
  videos.

**Why it helps:**
- The encoder keeps what it learned from far more footage than any one
  dataset has.
- The head cannot memorise much.
- It trains in minutes, so it can be repeated per fold and per change.

**Measured** (same 29 unseen videos,
`docs/plans/2026-10-08-overfitting-and-siglip2-fine-tune.md`):

| R3D-18 | unseen videos |
|---|---|
| every layer trained (old trainer) | 0.37 (95 % on training clips) |
| frozen + small head | **0.45** |

The same network, on the same clips, was 8 points better frozen.

**How solid:** one split. The direction matches every other frozen-vs-trained
comparison made here.

**Where it lives:** `model_training/action_head` (frozen SigLIP2 + head, the
app's trainer).

**Related positive:** `partial-fine-tune.md`, the step after this one.
**Related negative:** `../negative/training-whole-network.md`.
