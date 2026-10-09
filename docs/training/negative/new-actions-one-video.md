# New actions that only one reviewed video has

**Negative scenario.** A reviewed video brings actions no other video in the
dataset has, and they are expected to work right away.

**How it causes overfitting:** by construction, each new action is that one
video.
- The model can only learn it together with that video's scene.
- There are no unseen videos to test it on, so the overfitting cannot even
  be measured.

**How you notice:** the action is learned but never scored, never trusted and
never reported.

**Measured:** three actions present only in one reviewed video (one of them
"no action", 577 crops) were learned and never reported
(`docs/plans/2026-10-04-adding-reviewed-footage.md`).

**Instead:** keep them, capped at ~50 crops. They start working once 2 more
videos have them (`../dataset-positive.md`, Growing 4).
