# Soften class weights (power 0.5)

**Positive scenario.** Rare classes get extra weight in training, but only
the square root of the full balancing weight:
`(clips / (classes × clips of this class)) ** 0.5`.

**Why it helps:**
- Without weights, rare classes are ignored.
- With fully balanced weights, rare classes are over-rewarded, and new
  footage gets pushed into them, especially into a rare near-variant of a
  large class.
- Power 0.5 sits between the two.

**Measured:** +3.0 points held-out against inverse-frequency weights, on the
same features, three seeds (`recipe_ablation.py`,
`docs/plans/2026-10-01-action-models-measured.md`). The largest single gain
of the recipe changes.

**How solid:** three seeds, one split by video.

**Where it lives:** `model_training/action_head/head.py`
(`CLASS_WEIGHT_POWER`).

**Related negative:** `../negative/forced-best-guess-sorting.md`.
