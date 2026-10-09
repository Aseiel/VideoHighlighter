# Combine encoders when sorting a dataset

**Positive scenario.** Sorting new footage into a dataset's classes uses
several strong encoders together:
- an image model;
- a self-supervised image model;
- a video model, for classes defined by movement.

**Why it helps:**
- Each encoder sees something the others miss.
- Most of all, a video model finds classes that only movement separates.
- Sorting is a one-off job on the dataset builder's GPU that produces folders
  for review, so its cost and its encoder choice do not affect anyone else.

**Measured** (same head, 5 folds by source video,
`docs/plans/2026-10-03-automatic-sorting.md`):

| features | held-out | a movement-defined class (recall) |
|---|---|---|
| SigLIP2 base/16 (shipped) | 0.522 | 0.12 |
| SigLIP2 so400m | 0.595 | 0.20 |
| **so400m + DINOv2-L + V-JEPA 2-L** | **0.627** | **0.48** |
| V-JEPA 2 alone (video) | 0.515 | 0.68 |

The same ranking showed up in a person's review of one unseen video sorted
both ways.

**How solid:** 5 folds, one seed, plus one video reviewed by eye.

**Before shipping:** DINOv2 and V-JEPA 2 licences are still to be confirmed
(`CLAUDE.md`: weights, code and training toolkit).

**Related negative:** `../negative/small-encoder-near-variants.md`.
