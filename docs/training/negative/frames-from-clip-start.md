# Frames from only part of the clip

**Negative scenario.** The frames a model sees are taken from one stretch of
the clip, usually its start, instead of across the whole clip.

**How it causes overfitting:** the action is often not in what the model
sees.
- The label says "action X", but the frames show only the scene before it.
- The only thing left to connect the frames to the label is the scene, so
  the model learns the scene.

**How you notice:** a model that recognises the videos it was trained on and
misses the action in new ones.

**Measured:** the old trainers' `compute_frame_indices` took its 16 frames
from the first ~2 s of a 5 s clip
(`docs/plans/2026-10-01-action-models-measured.md`).

**Instead:** spread frames over the whole clip, frame `i` of `k` at
`(i + 0.5) / k` (`model_training/action_head/features.py`;
`../dataset-positive.md`, Training 2).
