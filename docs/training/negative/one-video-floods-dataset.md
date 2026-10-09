# One video flooding the training set

**Negative scenario.** Every reviewed crop of one new video goes into the
training set, so one video becomes a large share of the data.

**How it causes overfitting:** the model learns that video.
- A head learns what is common in its data.
- When one scene, one cast and one camera are half of it, it gets better
  at that video and worse at every other.

**How you notice:**
- Held-out accuracy on the *other* videos drops.
- Actions the flooding video has little of lose recall the most.

**Measured** (`docs/plans/2026-10-04-adding-reviewed-footage.md`):
- All 2,484 crops of one video (half the data): top-1 on other videos fell
  from 0.526 to 0.505.
- An action that video had 267 crops of lost recall.
- An action it had 2 crops of fell from 0.52 to 0.20.

**Instead:**
- Add the video as one source video.
- Add at most ~50 crops per action, preferring actions the dataset is thin
  on (`../dataset-positive.md`, Growing 1-3).
