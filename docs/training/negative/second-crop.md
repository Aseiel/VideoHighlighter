# A second crop on top of the cropper's clips

**Negative scenario.** Training (or analysis) crops the cropper's clips again,
for example to a detected person box.

**Not overfitting, but it loses accuracy.**
- It cuts away context the cropper deliberately kept.
- The box jitters from frame to frame.

**How you notice:** a "tighter" input that scores lower than the clip as it
was.

**Measured:** 3-4 points lost, even when the crop kept proportions. The
person box covered 82 % of the frame on median anyway
(`docs/plans/2026-10-01-action-models-measured.md`).

**Instead:** the cropper is the crop. Use its clips whole
(`../dataset-positive.md`, Building 4).
