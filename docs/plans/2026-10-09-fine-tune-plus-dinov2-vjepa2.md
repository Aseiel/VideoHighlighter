# Fine-tuned SigLIP2 plus DINOv2 and V-JEPA 2: what each adds

Follows `2026-10-08-overfitting-and-siglip2-fine-tune.md`. There, fine-tuning
SigLIP2 base/16 lifted held-out top-1 from 0.533 to 0.603. Here the question
is what the two other models from `2026-10-03-automatic-sorting.md` add on top
of it:

- **DINOv2** (image),
- **V-JEPA 2** (video).

**Short answer:** fine-tuned SigLIP2 + DINOv2-L + V-JEPA 2 sorts **11 points
more clips with confidence** (57-58 % → 68-70 %) at the same precision. It
takes **~15 minutes more training** (~85 min instead of ~70). The price is at
analysis time: about 13× the encoder work per window.

## The comparison

Same dataset, clips, classes, folds and metrics as the fine-tune doc:

- 2,695 clips, 38 classes, 135 source videos.
- 5 folds by source video, `test/` pairs taught.
- Trust rule: the Wilson 80 % lower bound of held-out precision ≥ 0.7, with
  hits from 3 or more videos.

| model | top-1 | top-3 | sorted with confidence | train time, total | GPU ms / window |
|---|---|---|---|---|---|
| SigLIP2 base/16 frozen + head (the app until 2026-10-08) | 0.533 | 0.733 | 44 % at 74 % | ~5 min | 13.5 |
| frozen so400m + DINOv2-L + V-JEPA 2, one head (`v3`) | 0.605 | 0.781 | 60 % at 75 % | ~40 min | ~260 |
| SigLIP2 base/16 fine-tuned (top 4 blocks), alone | 0.603 | 0.781 | 57 % at 75 % | ~70 min | 13.5 |
| fine-tuned + DINOv2-B | 0.614 | 0.808 | 66 % at 75 % | ~73 min | 27 |
| fine-tuned + DINOv2-L | 0.625 | 0.820 | 68 % at 76 % | ~75 min | 44 |
| fine-tuned + V-JEPA 2 | 0.610 | 0.802 | 63 % at 75 % | ~83 min | 149 |
| **fine-tuned + DINOv2-L + V-JEPA 2** | **0.630-0.644** | **0.826** | **68-70 % at 76 %** | **~85 min** | **179** |

- Rows with DINOv2 are the mean of 3 head seeds (±0.6 points). The fine-tune
  itself is one seed.
- **"Sorted with confidence"** is the share of held-out clips whose score
  clears its action's trust threshold, with the precision those clips reach.
- **Train time** is the whole job on an Arc A750:
  - preparing inputs (frame cache, frozen vectors);
  - the 5-fold cross-validation that produces the scores above and the
    trust thresholds;
  - the final model on all clips.
- **GPU ms / window** is PyTorch fp16 on the A750: 4 frames per window, 16 for
  V-JEPA 2. The app's OpenVINO path is faster in absolute terms (SigLIP2 base
  ~8 ms), so read this column as ratios.

### Where the training time goes

| | prepare inputs | 5-fold CV | final model | total |
|---|---|---|---|---|
| frozen trio | ~35 min features | ~3 min | ~0.5 min | ~40 min |
| SigLIP2 fine-tuned | ~5 min (frame cache, frozen vectors) | 52 min (5 × ~10.5 min) | ~11 min | ~70 min |
| + DINOv2-L + V-JEPA 2 | + ~1 min DINOv2-L, ~8-10 min V-JEPA 2 | + ~5 min (two heads) | + ~1 min | ~85 min |

- **Measured from run logs:**
  - fine-tune folds and final epochs;
  - frame cache;
  - DINOv2 feature extraction;
  - head training.
- **Estimated:**
  - The trio's features, from ~0.8 s per clip noted when `ds_big.npz` was
    made. The 10-03 doc's 0.25 s per clip would make that row ~20 min.
  - The V-JEPA 2 extraction, from its 135 ms per window plus decoding.
- **The fine-tune's time is mostly cross-validation.** The final model alone
  takes 11 min. The folds can't simply be dropped, because they set the trust
  thresholds. Three folds or fewer epochs per fold would cut them.
- **DINOv2 and V-JEPA 2 stay frozen.** Only their small heads are trained, so
  they add minutes, not another fine-tune.

## What it says

- **Fine-tuning beats size.** Fine-tuned SigLIP2 base alone matches the frozen
  heavy trio (0.603 vs 0.605) at about 1/20 of its cost per window.
- **DINOv2 is the best value.**
  - Frozen DINOv2-L alone scores 0.57, and frozen DINOv2-B 0.54. Both are at
    or above frozen SigLIP2 base (0.533).
  - Next to the fine-tuned model, DINOv2-B adds 8 points of confident sorting
    at the same speed as SigLIP2 base.
  - DINOv2-L adds 10 points at about 2× its cost.
- **V-JEPA 2 is expensive for what it adds here.**
  - Alone it scores 0.49.
  - On top of fine-tuned + DINOv2-L it adds about 1.4 points of top-1, for
    ~10× a SigLIP2 pass.
  - It is still the model that recognises movement-defined actions
    (`2026-10-03-automatic-sorting.md`). That suits dataset sorting, where
    time is cheap.
- **Each model keeps its own head, and their scores are averaged.**
  - The fusion takes each model's head and averages the log-odds with equal
    weights. Nothing is tuned on held-out clips.
  - This beats one head on all the features joined together. For frozen
    SigLIP2 base, DINOv2-L and V-JEPA 2: 0.612 averaged vs 0.589 joined.
  - An app model with several encoders should be built this way: one head per
    encoder, scores averaged.

## Licences

Checked 2026-10-09; both are permissive for code and weights:

| model | code | weights (Hugging Face card) |
|---|---|---|
| DINOv2 | Apache-2.0 | Apache-2.0 (`facebook/dinov2-*`) |
| V-JEPA 2 | MIT (3 data-loader files Apache-2.0) | MIT (`facebook/vjepa2-vitl-fpc64-256`) |

SigLIP2 is Apache-2.0. A model trained on any of them may be shared.

## Open

- **Fine-tune DINOv2 too.**
  - Use the same LP-FT recipe on DINOv2-B (about an hour on the A750).
  - The question: can two base-size fine-tuned models match or beat the full
    set, at ~2× today's analysis cost instead of ~13×?
- **The app runs one encoder per action model today.** A fused model needs
  more:
  - a model folder that names several encoders, each with its own head;
  - the analysis pass running each one on the same crops.
- **Sharing a fused model.**
  - The hub's single `model.onnx` (option A in
    `2026-10-08-fine-tuned-action-encoder.md`) has a 250 MB cap.
  - SigLIP2 base + DINOv2-B already passes that at fp16 (~360 MB).
  - Decide this before building fused models.
- **Not yet measured:**
  - Seeds for the fine-tune.
  - The fused model on a new video in the app (as in section 3b of the
    fine-tune doc).

Scripts (not in the repo), in `D:\teach\tools\`:

- `fusion_eval.py`, `fusion_eval2.py` (heads on the GPU), `fusion_seeds.py`;
- `dino_ft_frames.py`, `time_encoders.py`.

Out-of-fold scores are in `D:\teach\fusion\`.
