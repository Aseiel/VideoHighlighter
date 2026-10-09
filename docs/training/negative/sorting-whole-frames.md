# Sorting whole frames instead of the cropper's clips

**Negative scenario.** New footage is sorted from whole frames, while the
dataset (and the head) were built from the cropper's clips.

**Not overfitting, but a mismatch between what was taught and what is
sorted.**
- A whole frame can hold several people and actions.
- The head was taught one action per clip.

**How you notice:** far fewer crops pass the trust thresholds, and more
land in the wrong class.

**Measured:** 16 % trusted from whole frames, against 43 % from the
cropper's clips, on the same video
(`docs/plans/2026-10-03-automatic-sorting.md`).

**Instead:** cut and crop new footage the same way the dataset was built
(`../dataset-positive.md`, Building 4).
