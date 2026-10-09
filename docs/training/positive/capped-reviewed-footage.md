# Add a reviewed video capped, and where the dataset is thin

**Positive scenario.** When a sorted-and-reviewed video joins the training
set:
- it is added as one source video;
- at most ~50 crops per action;
- favouring actions with fewer than ~200 clips in the dataset.

**Why it helps:**
- A thin action needs new scenes, new people and new light, and a moderate
  number of new examples gives that.
- The cap stops the new video from becoming what the model learns.

**Measured** (3 seeds, scored on the dataset's own unseen videos,
`docs/plans/2026-10-04-adding-reviewed-footage.md`):

| action (thin in the dataset) | before | everything added | capped, thin only |
|---|---|---|---|
| A | 0.24 | 0.43 | 0.31 |
| B | 0.78 | 0.88 | 0.88 |
| C | 0.41 | 0.49 | 0.51 |
| F (barely in the new video) | 0.52 | **0.20** | 0.56 |

- Overall top-1: 0.526 before, 0.505 with everything added, 0.521 capped.
- Capped, it keeps the gains for thin actions and protects the rest.

**How solid:** 3 seeds, one reviewed video.

**Related negative:** `../negative/one-video-floods-dataset.md`,
`../negative/new-actions-one-video.md`.
