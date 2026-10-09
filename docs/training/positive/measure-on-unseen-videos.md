# Measure on unseen videos before believing any improvement

**Positive scenario.** Every claim that a change "will improve results" is
settled by one number: accuracy on whole source videos the model never
trained on. Without that number it is a guess, however confident it
sounds, and whoever it comes from: a person, a tutorial or an AI assistant.

**Why it matters more than any single improvement:**
- Every other page in this folder was found, or proven, by this
  measurement.
- Every mistake in `../negative/` survived because it was missing.
- A leaky score agrees with almost any plan, because the change that helps
  memorising also raises the leaky score.

**Three questions to ask before accepting a change:**
1. **How will we measure it?** Which number, from which script.
2. **On which videos?** If the answer is not "source videos the model never
   saw", the claim is not tested yet.
3. **What result would prove it wrong?** A claim with no losing outcome is
   not a prediction.

**Measured:** what the honest number changed.

| model | score the old way (random clip split) | on unseen videos |
|---|---|---|
| R(2+1)D / R3D, old trainer | 0.532 | 0.37 |
| SigLIP2 fine-tuned | 0.805 | 0.60 |

- **Agreement between advisers is not evidence.** Several AI assistants
  agreed that the old recipe would improve. It did not, and the leaky score
  appeared to confirm them each time.
- **A 5-fold test is steadier than one split.** The same fine-tune gained 7
  points on 5 folds by video, and only 2 on a single 29-video split.

**How to do it here:**
- `model_training/action_head/train.py` scores on 5 folds by source video
  and prints held-out numbers only.
- To compare two models, train both on the same folds and compare class by
  class.

**Related negative:** `../negative/leaky-validation.md`,
`../negative/epoch-chosen-on-leaky-score.md`,
`../negative/wrong-source-video-rule.md`.
