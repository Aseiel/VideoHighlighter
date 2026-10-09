# Validation that shares source videos with training

**Negative scenario.** The clips used to score the model come from source
videos that are also in training.

**How it causes overfitting:** it hides it, and then rewards it.
- Clips of one video share scene, people, light and camera.
- A model that memorises those scores well on the validation clips without
  having learned the action.
- Nothing in the score warns you, and everything chosen by that score is
  pushed towards memorising.

**How you notice:**
- Validation accuracy sits close to training accuracy.
- Then the model does poorly on a video it has never seen.

**Measured** (`docs/plans/2026-10-08-overfitting-and-siglip2-fine-tune.md`):
- Fine-tuned SigLIP2: 0.805 on a random clip split, 0.60 on unseen videos.
- The old 3D CNN trainer:
  - its R(2+1)D reported 0.532 on its random split;
  - the same trainer's R3D-18 scored 0.37 on unseen videos.

**Where it happens:**
- The old trainers re-split clips at random when `val/` was small.
- A dataset's own `val/` folder can share videos too: 159 of 166 clips did.

**Instead:** hold out whole source videos for every score
(`../dataset-positive.md`, Training 1).
