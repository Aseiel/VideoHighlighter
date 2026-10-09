# Fine-tuning SigLIP2 in the action trainer

Status: built (2026-10-09) with the proposed answers to all four questions at
the end: the fine-tune is saved only when it beats the frozen head, the GUI
has the checkbox, the frame cache is kept, the GUI fine-tunes 4 blocks. Where
the code differs from this design, see "As built" at the end.

Follows `2026-10-08-overfitting-and-siglip2-fine-tune.md` (top 4 blocks
fine-tuned with the head: held-out top-1 0.533 -> 0.603, sorted with
confidence 44 % -> 57 %) and `2026-10-08-fine-tuned-action-encoder.md` (the
app runs such a model from its own `vision.onnx`). Training one still takes
the experiment scripts in `D:\teach\tools` (`finetune_siglip.py`,
`ft_frames.py`) plus `tools/install_action_model.py`. This moves the recipe
into `model_training/action_head` as an option of the existing trainer, and
its output is the folder the app loads, with no install step.

## 1. Command line

One switch turns it on; the rest have the measured values as defaults.

```
python -m model_training.action_head.train --data-path <dataset> --finetune-blocks 4
    [--ft-epochs 10] [--ft-lr 1e-4] [--ft-decay 0.8] [--ft-head-lr 2e-4]
    [--ft-batch 16] [--device auto|xpu|cuda|cpu] [--frame-cache <folder>]
```

- `--finetune-blocks 0` (the default) is today's trainer, byte for byte.
- Every other flag (`--folds`, `--steps`, `--aliases`, `--teach-test`,
  `--min-videos`, `--precision`, `--out`, `--name`) means what it means now.
- The experiment's `--split-file`, `--random-split`, `--folds-only`,
  `--skip-cv`, `--frame-drop` are left out: they were for comparisons with
  the old trainers, not for making a model.

## 2. What a run does

The frozen trainer runs first, unchanged, and the fine-tune builds on it:

1. **Select clips** as today (`select_clips`, aliases, `--teach-test`).
2. **Decode every clip once into a frame cache** (section 3), 12 frames per
   clip.
3. **Frozen vectors** from the cache's 4 scoring frames, through the shared
   encoder (`frame_encoder.load`, any route), into the existing
   `FeatureCache` with the same keys. So a later frozen run reuses them and
   nothing is decoded twice. The encoder is closed before torch starts.
4. **Frozen 5-fold scoring**, as today: picks `--steps` and is the baseline.
5. **LP-FT per fold**, starting from that fold's frozen head: top N blocks
   (layer-wise lr `lr * decay^depth`, 0 = top block), the encoder's
   `post_layernorm` + pooling head at `lr`, the action head at `head_lr`.
   AdamW (wd 0.05), 6 % warm-up then cosine, grad-norm clip 1.0, bf16
   autocast. The loss is the head's own loss: class weights ^0.5, label
   smoothing, a clip's weight is the mean of its actions'. Each epoch scores
   the fold's held-out videos.
6. **Chooses the epoch count**: the epoch with the best mean held-out top-1
   over the folds. This is what the measured model did: 10 epochs in CV,
   best 9, the final model trained for 9.
7. **Trust thresholds and every printed number** come from the fine-tuned
   out-of-fold scores at that epoch. The rule is the same
   (`trust.trust_thresholds` / `pair_thresholds`).
8. **Final model**: a frozen head on every clip, then LP-FT on every clip for
   the chosen epochs.
9. **Export** (section 6).
10. **Frozen against fine-tuned, side by side**, on the same folds: top-1,
    balanced, top-3, sorted with confidence, test pairs. See section 7 for
    which one is saved.

Per training batch, each clip is seen as 4 of its 12 cached frames, chosen at
random and put in time order. It gets one crop (scale 0.7-1, aspect ±15 %),
flip and brightness/contrast change, applied to all four frames. Scoring
always uses the 4 frames the app uses at (i + 0.5) / 4. All of this is the
experiment's code with the paths and globals taken out.

Code layout (all in `model_training/action_head`, kept identical in both
editions like the rest of it):

- `finetune.py`: the tower wrapper (bottom blocks under `no_grad`; gradient
  checkpointing above 6 blocks), augmentation, `finetune()`, `predict()`.
- `frames.py`: the frame cache.
- `tower.py`: the export and its checks (section 6).
- `train.py`: the flags and the steps above.

## 3. Frames: the app's preprocessing, decoded once

**Exactly `frame_encoder.preprocess`.** The experiment copied that pipeline
into `ft_frames.py`. Here `frame_encoder.preprocess` is split in two:

- `prepare_frame(bgr) -> uint8 RGB [256, 256, 3]` (384 short side area,
  squash to 256 bilinear),
- the scaling to [-1, 1],

and `preprocess` calls both. The cache stores the first half's output, and
training does the second half on the GPU (`x / 127.5 - 1` equals
`(x / 255 - 0.5) / 0.5`). A test pins that `preprocess` is unchanged.
`frame_encoder.py` is a shared file, so this is one small change to port to
Pro.

**The cache** (`frames.py`): one `frames.npy` memmap `[N, 12, 256, 256, 3]`
uint8, plus `index.json`. It is keyed by `features.clip_key` (relative path,
size, mtime), so an edited clip is decoded again. It lives next to the feature
cache (`<user data>/cache/action_head/<dataset digest>-frames12/`).

- Slots 0-3 are at (i + 0.5) / 4 (what scoring and the app use), slots 4-11
  at (i + 0.5) / 8. `features.read_frames` gets a sibling that takes a list of
  positions and still decodes front to back in one pass, with the same
  second-pass rule when the container's frame count is wrong.
- **Size: 2.4 MB per clip** (6.0 GB for the 2,695-clip dataset). Before
  decoding, the run prints the size and stops with a sentence if the disk has
  less than that plus 1 GB free.
- Decoding uses a process pool (`ft_frames.py` used 6 workers).
- The cache is kept after the run, so a second run with other settings starts
  in seconds. The log says where it is and how big it is.

## 4. Device

Training needs torch on a GPU:

- **auto**: XPU, then CUDA (the `resolve_device` order the object trainer
  uses).
- **DirectML is refused.** It has no bf16 autocast, and the transformers
  backward pass on it is untested. The sentence says so, and suggests
  training without `--finetune-blocks`.
- **CPU only when asked for** with `--device cpu` (tests, tiny sets). It
  first prints a time estimate from the first 10 batches.
- **bf16 autocast** where the device supports it (`torch.xpu` / `torch.cuda`
  `is_bf16_supported()`), fp32 otherwise.
- **Memory:** batch 16 x 4 frames through the top 4 blocks fit the A750's
  8 GB. Out of memory stops the run with a sentence that names
  `--finetune-blocks` / `--ft-batch`. Splitting batches with gradient
  accumulation would help smaller cards, but it is not in this first version.
- **Keep awake:** while fine-tuning on Windows, the trainer asks the OS not to
  sleep (`SetThreadExecutionState`, as the experiment did) and stops asking
  when it ends. A run takes about an hour, and the overnight run needed this.

**Time:** about 1 min per epoch per 2,700 clips on the A750. 5 folds x 10
epochs + 9 final epochs is about 1 h. The first run adds decoding (~10 min for
2,700 clips) and the weights download.

## 5. Starting weights, dependencies, licences

- **Starting weights:** `transformers` `from_pretrained(frame_encoder.SOURCE_MODEL,
  revision=frame_encoder.SOURCE_REVISION)`, using `.vision_model`. This is the
  pinned checkpoint the shared encoder was exported from. The first run
  downloads the whole checkpoint into the Hugging Face cache (1.5 GB, text
  tower included). After that it works offline.
- **Start check:** before training, the torch tower's vector for
  `probe_pixels()` must match the shared encoder's `encoder.json` probe
  (cosine >= 0.99; expected ~1.0). This proves the fine-tune starts from the
  app's encoder, at the same revision. Otherwise the run stops with a
  sentence.
- **Nothing new to install:** `torch` (BSD-3), `transformers` (Apache-2.0),
  `onnx` (Apache-2.0), `onnxruntime` (MIT), `openvino` (Apache-2.0) and
  `scikit-learn` (BSD-3) are all in `requirements.txt` already.
- **Weights:** SigLIP2 is Apache-2.0, so a fine-tuned tower can carry a
  permissive licence, as the hub rules require for later. Training code and
  toolkit are torch + transformers, both permissive.
- No YOLO or ultralytics anywhere in the path.

## 6. Export: the folder the app loads

Written to `--out` (default `models/actions/<name>`), through a staging
folder that replaces the old one only after every check passes (as
`install_action_model` does):

```
vision.onnx   fine-tuned tower, weights stored as fp16, computes fp32 (~186 MB)
head.onnx     the action head
head.json     as today, plus own_encoder and finetune
```

1. `torch.onnx.export` of the fp32 tower (`pixel_values [N,3,256,256] ->
   image_embeds [N,768]`, opset 17, as `tools/export_frame_encoder.py`) to a
   temporary file. ONNX Runtime CPU must match torch (worst cosine >= 0.9999).
2. `store_weights_fp16`. This moves from `tools/export_frame_encoder.py` to
   `tower.py`, and the tool imports it from there: `tools/` is for development
   and might not be in a built app.
3. The fp16 tower is checked against torch on ONNX Runtime CPU and OpenVINO
   CPU (>= 0.9999). Nothing is written if one fails.
4. `head.onnx` is exported and checked against torch (drift <= 1e-4, as
   today).
5. `head.json`:
   - `encoder`: `"siglip2-base-patch16-256-finetuned"`, never the shared id.
   - `own_encoder`: `{file: "vision.onnx", preprocess: frame_encoder.ENCODER_ID,
     dims: 768, probe: <torch fp32 vector for probe_pixels()>}`.
   - `finetune`: `{blocks, epochs, lr, decay, head_lr, batch, base, revision}`.
   - Everything today's `head.json` has: classes, thresholds, pairs, held-out,
     test, val, per class, plus the frozen baseline's held-out numbers on the
     same folds.
6. The app's own check, `action_siglip.read_head_meta` + `own_encoder`, runs
   on the staging folder before it goes live.

`tools/install_action_model.py` stays, as a thin wrapper over `tower.py`'s
checks, for exports made elsewhere (the overnight model). The trainer does not
need it.

## 7. Which model is saved

**Proposed:** the run saves the fine-tuned model only if its held-out top-1 on
the same folds beats the frozen head's. Otherwise it saves the frozen head
(2 MB, shared encoder) and says why. That is the teach skill's rule ("install
only if better"), and a 186 MB model that is no better is a cost with nothing
in return.

The log also shows what the trade-off costs. On the measured data the
fine-tuned model is better on single actions but adds a wrong action to more
taught pairs (19 % against 13 %).

## 8. GUI: Train > Actions

**Proposed: yes, as one checkbox, off by default.**

- "Also train the image model (graphics card, about an hour; the model is
  186 MB instead of 2 MB)".
- Enabled only when `_probe_training_device()` finds XPU or CUDA. The panel
  already runs this off the GUI thread for objects. Otherwise it is greyed
  out, with the reason as a tooltip.
- Checked, the worker adds `--finetune-blocks 4`. Progress reads the new lines
  as a third phase: decoding 0-15 %, encoding 15-25 %, frozen folds 25-35 %,
  fine-tune folds and epochs 35-90 %, final 90-100 %.
- The finished note names the model that was saved, frozen or fine-tuned
  (section 7).

**Not changed:** teach projects (`modules/teach/train.py`) and the teach skill.
They keep training frozen heads; adding the flag there is a later, separate
step.

## Not changed

- Search, teach-by-example, actions by name: shared encoder only.
- `model_hub/`: action models are not shareable yet (`USABLE_TASKS`), so no
  change (see the encoder design's section 5).
- Content: no class names in code, tests or docs. Tests use made-up classes.
- Nothing written holds a frame, crop or path. The frame cache stays in the
  user's own data folder and is never copied into a model folder.

## Edition sync

`model_training/action_head/*` and `modules/vision/frame_encoder.py` are kept
identical in Pro and need porting. `tools/export_frame_encoder.py` needs it
too, because `store_weights_fp16` moves. `training_panel.py` already differs
from Pro, so the checkbox goes into Pro's copy by hand.

## Tests

The suite shims torch, so these run in a child process with the real torch,
as `test_action_head.py` does. They use a tiny random SigLIP tower (2 layers,
hidden 32, 256 px, patch 16) on the CPU, and the 768 check is patched to the
tiny size.

- `preprocess` = scaling(`prepare_frame`), bit for bit.
- The frame cache: round trip, an edited clip decoded again, the slot
  positions, a clip that cannot be read left out.
- LP-FT: only the top N blocks, the pooling head and the action head change
  (the rest is bit-identical). The lr groups follow the decay. Epoch choice is
  the best mean held-out.
- The start check refuses a tower whose probe differs.
- The export: the folder passes `read_head_meta` / `own_encoder`, the fp16
  tower is within cosine, and the encoder id is not the shared one.
- `--finetune-blocks 0` writes the same `head.json` as today (apart from
  `created`).
- Device: DirectML refused, CPU only with `--device cpu`.
- Section 7: the frozen head is saved when the fine-tune does not beat it.
- GUI: the checkbox adds the flag; disabled without XPU/CUDA.

**Real:** retrain on the same dataset (aliases, `--teach-test`) with
`--finetune-blocks 4`. Expect the overnight numbers within seed noise: 0.603
top-1, 57 % sorted with confidence. Then run headless on video 001 and compare
with the installed overnight model (33.9 % of reviewed labels found).

## For the maintainer to decide

1. **Section 7**: save the fine-tune only when it beats the frozen head
   (proposed), or always save what was asked for?
2. **GUI checkbox now** (proposed), or command line only at first?
3. **Frame cache kept after the run** (proposed; 6 GB for the big dataset),
   or deleted at the end unless `--frame-cache` is given?
4. **Default `--finetune-blocks` in the GUI: 4** (best single-action
   accuracy, measured) or 8-12 (better at pairs, 1.6-2x slower)?

## As built (2026-10-09)

Everything above, with these differences:

- **`tower.py` only exports.** `store_weights_fp16` had already moved to
  `modules/vision/onnx_weights.py` (with the Import button), so
  `tools/export_frame_encoder.py` is unchanged. `tower.py` writes the fp32
  `vision.onnx` (checked against PyTorch, worst cosine >= 0.9999), `head.onnx`
  (checked on the tower's own vectors) and `head.json`. It then hands the
  folder to `action_models.install_export`, the code Import uses: fp16
  weights, the probe, checks on ONNX Runtime and OpenVINO CPU, and
  `read_head_meta`, all in staging. The fp32 export is deleted afterwards. As
  in `install_export`, the probe comes from ONNX Runtime on the fp32 export,
  not from PyTorch; the two agree to the cosine above.
- **The start check** compares the torch tower with the vector the shared
  encoder returns for `probe_pixels()` on the route it loaded on. That route
  already reproduced `encoder.json`'s probe when it loaded, so it is the same
  proof, and a test can run it without a real encoder.
- **Decoding runs on threads, not a process pool.** In the packaged app a
  process pool starts copies of the app (the bug b0bfd8e fixed for the
  trainer itself). OpenCV decodes outside the GIL, so this costs nothing:
  measured 19 clips/s, 2.5 min for the 2,695-clip dataset, not 10 min.
- **When a run needs clips the cache lacks**, the cache is rebuilt: the rows
  it has are copied and only the new or edited clips are decoded.
- **The fold heads are reused.** The frozen folds keep the head of every fold
  at the chosen length, and each one starts that fold's fine-tune.
- **`val/` as the dataset defines it** is not scored for the fine-tuned model:
  that would take one more LP-FT on `train/` only. The frozen head's score is
  kept in `head.json` as `finetune.frozen_val_folder`.
- **No time estimate before a processor run.** Each epoch's line already
  says how long it took.
- When the frozen head is saved because the fine-tune did not beat it,
  `head.json` records the attempt as `finetune_not_saved` (its accuracy, the
  frozen accuracy, accuracy by epoch, settings).

Tests: `tests/test_action_finetune.py` (9, a tiny random SigLIP tower on the
processor in a child process) and `tests/test_action_training_in_app.py`
(the flag, the progress phases, the checkbox gating).

**Real run** (the same dataset, aliases and `--teach-test` as the overnight
model, `--finetune-blocks 4`, Arc A750): 71 min in all; decoding 2.5 min,
about 60 s per epoch, 10 to 11 min per fold. On the same 5 folds:

| | frozen | fine-tuned (10 epochs) | overnight script |
|---|---|---|---|
| held-out top-1 | 0.521 | **0.593** | 0.603 (frozen 0.533) |
| balanced | 0.311 | 0.372 | |
| top-3 | 0.724 | 0.778 | 0.781 |
| sorted with confidence | 44 % at 74 % | 59 % at 74 % | 57 % |
| test pairs, both in top 2 | 47 % | 48 % | |

The gain over the frozen head is the overnight one (+0.07). The best epoch was
the 10th, so the final model trained for 10. The fp16 tower matches the fp32
export to a worst cosine of 0.9999998 on ONNX Runtime and OpenVINO CPU, and the
app loads it on OpenVINO GPU.

Not ported to Pro yet: `modules/vision/frame_encoder.py` and
`model_training/action_head/*` are shared files; the checkbox goes into Pro's
`training_panel.py` by hand.
