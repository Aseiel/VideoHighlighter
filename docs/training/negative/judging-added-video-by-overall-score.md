# Judging an added video by the trainer's overall held-out line

**Negative scenario.** A large reviewed video is added to the dataset, and the
change is judged by the trainer's overall held-out accuracy.

**Not overfitting, but a misleading score the other way:** a good change can
look bad and be thrown away.
- The new video becomes a held-out fold of its own.
- It is scored by heads that never saw it, and when it is large it dominates
  the average.

**How you notice:** the overall line drops after adding reviewed footage,
while the model has not got worse on other videos.

**Measured** (`docs/plans/2026-10-04-adding-reviewed-footage.md`):
- Adding one video of 2,484 crops moved the line from 0.52 to 0.46.
- Trusted actions went from 17 to 13.
- On the dataset's own videos, nothing got worse.

**Instead:** compare old and new models on the dataset's own videos, same
folds (`../dataset-positive.md`, Growing 5).
