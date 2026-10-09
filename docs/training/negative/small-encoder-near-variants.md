# Sorting near-variant classes with a small encoder

**Negative scenario.** Automatic sorting uses the small shipped encoder on
classes that differ only in detail from a larger class.

**Not overfitting directly, but it feeds wrong labels into the dataset.**
- The small encoder merges near-variants into the larger class.
- Accepted without review, those labels blur the classes for every later
  model.

**How you notice:** a near-variant class gets almost nothing from a new
video, while the larger class gets crops that belong to it.

**Measured** (`docs/plans/2026-10-03-automatic-sorting.md`):
- For one class, the base encoder called half of the crops the strong sort
  put there its near-variant instead.
- For a mid-sized class, its top guess was right for 36 of 96.
- Held-out accuracy: three large encoders 0.627, base encoder 0.522.

**Instead:** sorting is a one-off on the dataset builder's GPU, so it can use
the larger encoders. Shared models stay on the shipped one
(`../dataset-positive.md`, Sorting 3).
