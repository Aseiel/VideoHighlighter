# Why R3D and Intel overfitted, and what fine-tuning SigLIP2 adds

**Questions:**

1. R3D reached 95 % on its training clips and 37 % on new videos. Is that
   the dataset or the training script?
2. Trained properly, would R3D and Intel come close to SigLIP2?
3. Fine-tuning SigLIP2 itself, the way R3D was trained: is it better than
   the frozen encoder + head the app ships?
4. Are the R3D/Intel training scripts being replaced?

**Short answers:**

1. **Mostly the script.** The dataset limits every model. The old recipe
   made R3D memorise each video's scene, and its validation hid that.
2. **Better, but not SigLIP2's level.** Proper training lifts them from
   0.35-0.37 to 0.42-0.50 on unseen videos. SigLIP2 base gets 0.55 frozen
   and 0.57 fine-tuned on the same videos. A properly fine-tuned R(2+1)D
   was not measured (see the end).
3. **Yes, by 7 points on 5 folds** (0.533 → 0.603), and by more where it
   counts: clips sorted with confidence went from 44 % to 57 % at the same
   precision. On a new video in the app it found 1.8× as many of a
   person's labels (section 3b). It brings its own copy of the encoder,
   which the app can load since 2026-10-08.
4. **Already done.** 0.13.1 removed `model_training/intel`, `r3d` and
   `shared`; `model_training/action_head` (frozen SigLIP2 + head) is the
   trainer. The fine-tune is an experiment outside the app so far.

Same hand-sorted dataset as `2026-10-01-action-models-measured.md`: 2,711
clips from 135 source videos, cut by the app's cropper. Class names are the
dataset's and do not appear here. "Held-out" means scored on source videos
the model never saw (the source video is the clip name before
`_temp`/`_highlight`).

## 1. The overfitting: script, not dataset

**The evidence is one network trained two ways on the same clips.** On the
same 29 held-out videos (445 clips):

| | held-out |
|---|---|
| r3d_18, whole network trained by `model_training/r3d` | 0.37 (95 % on training clips) |
| r3d_18 frozen, small head trained on its features | 0.45 |
| r(2+1)d_18 frozen, small head | 0.50 |
| Intel encoder + decoder trained by `model_training/intel` | 0.35 |
| the same, with the input and framing fixes | 0.42 |
| Intel encoder frozen, small head, with the fixes | 0.44 |

Same data, same network: the old recipe cost R3D 8 points and Intel 7-9.

**What the old scripts did:**

1. **Validation leaked, so the problem never showed.** When `val/` was too
   small, `validate_and_split_dataset` re-split each class by *clip*, at
   random (about 20 %). Clips of one video landed on both sides, so
   validation rewarded remembering the scene. That number also picked the
   "best" epoch: the one that remembered best. The originals' best
   validation scores (Intel 0.431, R(2+1)D 0.532) are scene memory plus
   action. See the table in section 2.
2. **R3D trained all of its ~31 M weights** on what is effectively 135
   scenes. Clips from one video are near-copies, so memorising scene, people
   and light is easier than learning the action.
3. **Frames from the start of the clip only.** `compute_frame_indices` took
   its 16 frames from the first ~2 s of a 5 s clip, so the action was often
   not in the input at all, and the scene was all there was to learn.
4. **Intel only:**
   - Its encoder was fed RGB normalised twice instead of the BGR 0-255 its
     IR declares (3-4 points).
   - "Best model loaded" restored the last epoch (a shallow `state_dict`
     copy).

**What the dataset does:** it doesn't cause the overfitting, but it caps
how far any model gets.

- 135 source videos are 135 independent examples of a scene.
- Many actions appear in only a few videos.
- Adding many clips from one more video did not help other videos
  (`2026-10-04-adding-reviewed-footage.md`). More *distinct* videos is what
  raises every model, SigLIP2 included.

## 2. Old models against SigLIP2, like for like

**Videos never seen** (the old trainers' own 29-video split; R3D/Intel are
last week's retrains with the old scripts, `--split-by-source`):

| model | held-out |
|---|---|
| Intel, old recipe | 0.348 |
| Intel, input + framing fixed | 0.405 (trainer) / 0.443 (head) |
| r3d_18, old recipe | 0.371 (best epoch 0.387) |
| r(2+1)d_18 frozen + head | 0.50 |
| SigLIP2 base/16 frozen + head (the app today) | 0.553 |
| **SigLIP2 base/16 fine-tuned** (top 4 of 12 blocks) | **0.566-0.577** |

**Random split by clip** (how the old trainers scored, so these sit next to
the originals' own numbers; leaky by design):

| model | held-out |
|---|---|
| Intel original (`intel_finetuned_classifier_3d`) | 0.431 |
| R(2+1)D original (`r3d_finetuned`) | 0.532 |
| SigLIP2 frozen + head | 0.765 |
| **SigLIP2 fine-tuned** | **0.805** |

The originals cannot be scored on unseen videos: every source video in the
dataset was already in their training data. The class lists differ
slightly between rows (35 classes in the old trainers, 38 here after merging
near-identical folders), which moves a score by a point or two, not by 20.

**Why proper training does not close the gap:**

- **The best measured "proper" R3D/Intel number is the frozen + head row.**
  R(2+1)D gets 0.50 and Intel 0.44. That is already what the corrected
  recipe gives: split by video, frames across the whole clip, the right
  input, softened class weights.
- **The rest of the gap is pretraining.** R3D and Intel learned from
  Kinetics-400 at 112-224 px. SigLIP2 learned from billions of image-text
  pairs. On this footage, what is in the frame matters more than motion,
  and the broader model separates people and poses far better.
- **Fine-tuning helps when the starting point is already good.** Fully
  training R3D on 135 videos lost 8 points. Partially training SigLIP2's
  top blocks gained 2-7.

## 3. Fine-tuning SigLIP2 (overnight, 2026-10-08)

LP-FT ("linear probe, then fine-tune"; experiment script, not in the repo:
`D:\teach\tools\finetune_siglip.py`):

1. Train the usual head on frozen vectors.
2. From it, train the top N encoder blocks, the encoder's pooling head and
   the action head together.

Recipe:
- AdamW, layer-wise learning-rate decay 0.8, warm-up then cosine, bf16 on
  the Arc A750.
- Each clip is seen as 4 of 12 frames spread across it, with one crop, flip
  and colour change per clip.
- Scored on the trainer's own 5 folds by source video, with the same clips,
  classes, trust rule and metrics as `model_training/action_head/train.py`.
  The frozen head is retrained on each fold as the baseline.

2,695 clips, 38 classes, 135 videos, `test/` pairs taught:

| | top-1 | balanced | top-3 | sorted (trusted) | test pairs: both in top 2 | time |
|---|---|---|---|---|---|---|
| frozen + head (the app today) | 0.533 | 0.339 | 0.733 | 44 % at 74 % | 48 % | ~2 min |
| top 4 blocks, lr 3e-5 | 0.578 | 0.374 | 0.765 | 53 % at 74 % | 49 % | ~1 min/epoch |
| **top 4 blocks, lr 1e-4** | **0.603** | **0.376** | **0.781** | **57 % at 75 %** | 46 % | ~1 min/epoch |
| top 8 blocks, lr 3e-5 | 0.586 | 0.374 | 0.769 | 52 % at 74 % | 54 % | ~1.6 min/epoch |
| all 12 blocks, lr 2e-5 | 0.586 | 0.367 | 0.762 | 54 % at 75 % | 56 % | ~2 min/epoch |

- **Every fold of every setting beat the frozen head.** The last 2-3 epochs
  score within a point of each other, so choosing the epoch is not luck.
- **On the single 29-video split the gain is smaller** (+1.6 to +2.4). One
  split of 29 videos is noisier than 5 folds over all 135.
- **Trade-off:** the saved model (top 4, lr 1e-4) is best on single actions.
  Unfreezing 8-12 blocks is better at finding both actions of a pair, and
  in its trust test the saved model adds a wrong action to 19 % of pairs
  (13 % for the frozen pairs head).
- **It still memorises** (100 % on its own training clips), but held-out
  accuracy rose with it, unlike R3D's.

**Using it in the app:** the fine-tuned tower is no longer the shared
encoder that search and actions by name need, so the model carries its own
copy (186 MB at fp16) in its folder. The action pass runs that copy instead
of the shared encoder, so the time per window is unchanged: by name scores
whole frames and a head scores person crops, so they never shared a pass.
Design and code: `2026-10-08-fine-tuned-action-encoder.md`; install an export
with `tools/install_action_model.py`. The model is saved in
`D:\teach\ft\k4-lr1e-4\`.

## 3b. What it changes for sorting

**Sorting with confidence is where the fine-tune pays most.** A clip is
sorted with confidence when its score clears its action's trust threshold.
That threshold is set so that precision on unseen videos reaches 0.7 at the
Wilson 80 % lower bound, with hits from 3 or more videos. Everything below it
is only a suggestion for a person.

**On the dataset** (same clips, 5 folds, head trainer and trust rule for
every row; the old encoders frozen with the same head, Intel with the
correct input):

| encoder | top-1 | top-3 | sorted with confidence |
|---|---|---|---|
| r3d_18 (Kinetics-400) | 0.409 | 0.628 | 11 % at 79 % |
| r(2+1)d_18 (Kinetics-400) | 0.414 | 0.627 | 14 % at 77 % |
| Intel action-recognition-0001 | 0.418 | 0.632 | 18 % at 76 % |
| SigLIP2 base/16, frozen (the app until now) | 0.533 | 0.733 | 44 % at 74 % |
| **SigLIP2 base/16, top 4 blocks fine-tuned** | **0.603** | **0.781** | **57 % at 75 %** |

- Top-1 moves 12 points from the best old encoder to frozen SigLIP2, and 7
  more with the fine-tune. The confident share moves 26 and then 13: 2.4× and
  then 1.3×.
- The old models trained whole by their own scripts are not in the table:
  they have no out-of-fold predictions on these folds. On the 29-video
  split they were below their frozen + head versions (section 2), so their
  share would be lower still.

**On a new video, in the app** (video 001, 52 min, in neither model's
training; 1,203 five-second windows a person reviewed; labels mapped through
the training aliases; headless run on the Arc A750):

| | frozen head (`action-head-siglip2-pairs`) | fine-tuned |
|---|---|---|
| reviewed labels found (of 1,473 the heads know) | 19.1 % | **33.9 %** |
| of each head's trusted actions | 21.1 % | **38.7 %** |
| precision on reviewed windows | 55.3 % | 52.9 % |
| reviewed windows with any label | 476 | 760 |
| run time | 74 s | 72 s |

- 1.8× as many labels found, for 2.4 points of precision.
- Precision counts any label the review does not list for that window as
  wrong, so both precisions are a floor.
- One video and one reviewer: the direction matches the 5-fold result; the
  exact size is this video's.

## 4. The training scripts

On `main` since 0.13.1:

- **Removed:** `model_training/intel`, `model_training/r3d`,
  `model_training/shared` (only stale `__pycache__` folders may remain in an
  old checkout).
- **The trainer:** `model_training/action_head`, frozen SigLIP2 + head. The
  section 1 fixes are built in:
  - split by source video,
  - frames evenly across the whole clip,
  - the encoder's own preprocessing,
  - class weights at power 0.5,
  - training length chosen on unseen videos,
  - per-action trust thresholds.
- **Fine-tuning is not in the app's trainer.** The app runs a fine-tuned
  model, but training one still takes the experiment script. It needs a GPU
  with a PyTorch training route (XPU or CUDA) and about an hour instead of
  two minutes. If it is added, it belongs as an option of that trainer, not
  as a separate script.

## Open

- **A properly fine-tuned R(2+1)D was not measured:** split by video, frames
  across the clip, top blocks only, LP-FT. The frozen + head row (0.50) is
  its likely floor. Reaching 0.57 would take a bigger gain than SigLIP2 got
  from the same treatment. Measuring it would settle question 2.
- **Seeds:** each setting ran once (seed 0). The fold-by-fold wins make the
  direction safe; the exact sizes are ±1-2 points.
