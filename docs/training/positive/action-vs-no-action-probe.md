# Teach "is there an action at all" separately

**Positive scenario.** A small classifier on the encoder's vectors is
taught crops that show an action against crops that show none (a leg, a
back, an empty corner). It is kept separate from which action it is.

**Why it helps:**
- An action head was never taught "nothing", so its top score is a poor
  judge of it.
- A dedicated yes/no classifier is.
- It could keep empty crops out of a dataset, or out of a sort.

**Measured** (one reviewed video, 10 held-out stretches of it,
`docs/plans/2026-10-04-adding-reviewed-footage.md`):

| | AUC |
|---|---|
| the action head's top score | 0.66 |
| logistic regression, action vs no-action | **0.80** |

At a cut that keeps 90 % of action crops, it drops 39 % of empty ones.

**How solid:** one video only. It needs "no action" examples from more
videos before it can gate crops (`../negative/new-actions-one-video.md`).
