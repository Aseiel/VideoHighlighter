# Trusting a cluster of unsure crops to be one action

**Negative scenario.** Unsure crops are grouped by look, and a whole group is
named as one class without checking what holds it together.

**How it causes overfitting:** the group is often one video, not one action.
- Encoder vectors group by source video more than by class.
- Naming such a group adds one video's scene as the class: the
  `classes-from-too-few-videos.md` problem, created by the sort itself.

**How you notice:** most crops in a group come from one or two source videos.

**Measured:** k-means on the dataset's encoder vectors agreed more with the
source video than with the class: NMI 0.47 against 0.40.

**Instead:** before naming a group as a whole, check it spans several videos;
otherwise name its crops one by one (`../dataset-positive.md`, Sorting 2).
