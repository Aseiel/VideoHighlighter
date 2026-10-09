# Teach pairs of actions as `a_b` folders, with one score per action

**Positive scenario.**
- Clips showing two actions at once go in a folder named `a_b`.
- The head gives each action its own score (sigmoid), not one shared answer
  (softmax).

**Why it helps:**
- A clip can then say "both", and the head learns which actions occur
  together.
- Each taught pair also gets a trust threshold of its own, on the lower of
  its two scores.

**Measured** (unseen videos, `model_training/action_head/head.py`):

| | both actions in the top two |
|---|---|
| one answer per clip (softmax) | 0-2 % |
| one score per action, pairs taught | **42 %** |
| a pair that was never taught | 0-4 % |

Single-action accuracy is unchanged (0.534 either way).

**How solid:** measured on the dataset's `test/` pairs, with 5 folds by video.

**Related negative:** `../negative/untaught-combinations.md`.
