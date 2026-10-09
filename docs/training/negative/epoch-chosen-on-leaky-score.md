# Choosing the best epoch or settings on a leaky score

**Negative scenario.** The training length, the "best model" checkpoint or
the settings are chosen by a validation score that shares source videos with
training (`leaky-validation.md`).

**How it causes overfitting:** it actively selects it.
- The longer a model trains on few videos, the better it remembers them.
- A leaky score keeps rising while that happens.
- So the checkpoint kept is the one that remembers the training videos best,
  the most overfitted one.

**How you notice:** the "best" epoch is late, training accuracy is near 100 %,
and new footage disappoints.

**Measured:**
- R3D-18 trained by the old trainer reached 95 % on training clips.
- The checkpoint its score picked: 0.37 on unseen videos.

**Instead:** choose length, checkpoint and settings on held-out source videos.
`model_training/action_head` picks its training length from 5 folds by video
(`../dataset-positive.md`, Training 1).
