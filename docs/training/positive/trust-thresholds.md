# Sort only what is trusted

**Positive scenario.** Each class (and each taught pair) gets a threshold,
the lowest score at which:
- its precision on unseen videos reaches 0.7 at the Wilson 80 % lower
  bound;
- with hits from 3 or more source videos.

Below it, a crop is only a suggestion for a person.

**Why it helps:**
- Forced best guesses put every crop somewhere, including the crops the
  model does not know.
- The threshold separates "sure" from "guessing", per class, using evidence
  from videos the model never saw.
- The 3-video rule stops one video's near-copies from vouching for a class.

**Measured:**
- `sort_trusted.py` (`docs/plans/2026-10-01-action-models-measured.md`):
  66 % of held-out clips sorted, 77 % of them correctly.
- A rare near-variant class that had absorbed 130 crops of a large class
  got no folder.
- The shipped trainer: 44 % sorted at 74 % with the frozen head, and 57 % at
  75 % with the fine-tuned one.

**How solid:** thresholds come from 5-fold scores. They protect precision at
the cost of how much gets sorted.

**Where it lives:** `model_training/action_head/trust.py`.

**Related negative:** `../negative/forced-best-guess-sorting.md`.
