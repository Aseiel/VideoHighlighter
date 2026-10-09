# Forcing a best guess on every crop when sorting

**Negative scenario.** Automatic sorting puts every crop into the class with
the highest score, however low, and fully balanced class weights boost rare
classes.

**How it leads to overfitting:** the next model trains on the sort's mistakes.
- A rare class that is a near-variant of a large one absorbs crops of the
  large one.
- Accepted into the dataset, those crops teach the next model the same
  mistake.
- Each round of sort-and-train repeats the mistake with more conviction.

**How you notice:**
- Everything is sorted and nothing is flagged for review.
- A small class suddenly receives a large share of a new video.

**Measured:** a rare near-variant class absorbed 130 crops of the large class
on one video. With trust thresholds it got no folder, and 66 % of held-out
clips were sorted at 77 % precision
(`docs/plans/2026-10-01-action-models-measured.md`).

**Instead:**
- Sort only above each class's trust threshold, and leave the rest for a
  person.
- Soften class weights to power 0.5 (`../dataset-positive.md`, Sorting 1;
  Training 4).
