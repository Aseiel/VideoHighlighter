# Input the encoder does not expect

**Negative scenario.** Frames reach a pretrained encoder in another colour
order, range or normalisation than it was trained on.

**Not overfitting by itself, but it wastes the encoder.**
- The encoder's pretrained knowledge is the main defence against
  memorising a small dataset.
- Feed it the wrong input and its vectors carry less of that knowledge.
- What gets trained on top has less real signal to learn from.

**How you notice:** accuracy noticeably below what the same encoder gets
elsewhere. Look for preprocessing applied twice, or colour channels in the
wrong order.

**Measured:**
- The Intel action encoder's IR expects BGR 0-255; it does its own mean
  subtraction and channel flip.
- It was fed RGB normalised twice, an almost flat image with red and blue
  swapped.
- Fixing only that gained 3-4 points
  (`docs/plans/2026-10-01-action-models-measured.md`).
- **Analysis used a third variant:** custom decoders were fed RGB
  normalised once. That matched neither what the encoder declares nor what
  training fed it.
  - A train/inference mismatch is a mistake of its own: the head runs on
    vectors unlike the ones it learned from.
  - It was never fixed in code, because Intel was retired instead. An old
    Intel model a user still has was trained and run with the wrong colours.

**Instead:** feed exactly what the model card or IR metadata declares
(`../dataset-positive.md`, Training 3).
