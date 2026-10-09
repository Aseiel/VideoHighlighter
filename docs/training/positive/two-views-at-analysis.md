# Two frame views per window at analysis

**Positive scenario.** At analysis time, each window is read as two 4-frame
views (8 frames), and their scores are averaged.

**Why it helps:**
- Twice the frames, at offset positions, gives a second chance to catch a
  short action.
- Averaging smooths a single unlucky frame.

**Measured:** +1.7 points, at twice the encoder time
(`docs/plans/2026-10-03-automatic-sorting.md`).

**How solid:**
- Small, close to the noise between seeds.
- For actions typed by name, 8 frames found a subset of what 4 found
  (`docs/plans/2026-10-07-action-frames.md`). It is a precision setting, not
  a recall one.

**Where it lives:** the **Frames per window** setting (4 or 8). A taught
model reads the number of frames it was trained on.
