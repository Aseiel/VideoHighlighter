# A stronger pretrained encoder

**Positive scenario.** Choose the encoder by its held-out score on this kind
of footage, with the same small head on top of each.

**Why it helps:**
- On this footage, what is in the frame matters more than how it moves.
- Image encoders trained on billions of image-text pairs separate people
  and poses far better than video models trained on Kinetics-400 at low
  resolution.
- It is the largest gain measured here, larger than any recipe fix.

**Measured** (frozen + the same head, same 29 unseen videos,
`docs/plans/2026-10-01-action-models-measured.md`):

| encoder | held-out | ms per window (A750) |
|---|---|---|
| Intel action-recognition-0001 | 0.42 | 23 |
| R3D-18 | 0.45 | 4.5 |
| R(2+1)D-18 | 0.50 | 6.9 |
| CLIP ViT-B/32 | 0.50 | 2.8 |
| DINOv2-L | 0.58 | 31 |
| **SigLIP2 base/16 (shipped)** | **0.555** | **10** |
| SigLIP2 so400m | 0.625 | 133 |

**Sorted with confidence** (same head and trust rule, 5 folds by source
video, 2,695 clips, `docs/plans/2026-10-08-overfitting-and-siglip2-fine-tune.md`):

| encoder, frozen + head | top-1 | sorted with confidence |
|---|---|---|
| R3D-18 | 0.409 | 11 % at 79 % |
| R(2+1)D-18 | 0.414 | 14 % at 77 % |
| Intel action-recognition-0001 (correct input) | 0.418 | 18 % at 76 % |
| **SigLIP2 base/16 (shipped)** | **0.533** | **44 % at 74 %** |
| SigLIP2 base/16, top 4 blocks fine-tuned | 0.603 | 57 % at 75 % |

For sorting the gap is wider than top-1 suggests. The video models are
sure about too few clips to be worth sorting with: 11-18 %, against 44 %.

**How solid:** one split, and the ranking held on 5 folds for the leaders.

**The trade-off:** speed. The base encoder ships because it runs on every
machine. so400m is about 9× the work per frame.

**Related positive:** `combined-encoders-for-sorting.md`.
