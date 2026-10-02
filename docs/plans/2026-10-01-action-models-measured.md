# Action recognition: retire Intel and R3D, move to SigLIP2

**Question (Szymek):** after the train/val label fix, which action model should
the app train and ship, and is it time to drop the old ones?

**Decision:** retire the Intel action-recognition model and R3D: analysis,
custom training and the built-in Kinetics-400 decoder. Replace them with a
frozen SigLIP2 image encoder plus a small trained head. Kinetics-700 becomes
the suggested action list, and the user can also type any action of their
own.

**Why they were here, and why they can go:** both were chosen because they
were light and easy to train, at a time when training anything else was out
of reach. They did their job. They proved, end to end, that the app's
OpenVINO path (Intel CPU and GPU) and its CUDA path work for video models,
along with the ONNX Runtime and Core ML routes added for R3D later. On held-out
videos, though, every variant of them stays 10-20 points below SigLIP2 base.
SigLIP2 base is faster per analysis window than the Intel encoder, and only
its small head is trained. R3D is faster still in raw compute, but it is the
least accurate of the lot once fine-tuned.

Everything below is measured on one hand-sorted dataset: 2,413 clips, 135
source videos, cut by the app's cropper. Class names are the dataset's and do
not appear here.

## How it was measured

**Held-out accuracy:** train on some source videos, then score clips from
source videos the model never saw, not even other clips from them. That is
the situation the app is in with new footage.

The old validation scores could not separate the models, for two reasons:

1. **Labels** (fixed in `3f0d9ec`). When val lacked a folder that train had,
   the auto-split relabelled some clips and dropped others.
2. **Source-video leak.** 159 of the 166 clips in the dataset's own `val/`
   come from source videos that are also in `train/`. When val is too small,
   the trainer's auto-split splits by clip, so the same leak happens there.
   Neighbouring clips share scene, people and light, so validation rewarded
   remembering the scene, and that score also chose the best epoch.

From here on, the source video is the clip name before the first
`_temp`/`_highlight` (the rule in `modules/teach/benchmark.py`). The
comparisons on a single split use the same 29 held-out videos (445 clips).

## The comparison

The same small head on top of each frozen encoder
(`tools/teach_lab/compare_split.py`), plus each model's own trainer:

| encoder | published | held-out, frozen + head | held-out, own trainer | ms per window (A750)¹ |
|---|---|---|---|---|
| Intel action-recognition-0001 | ~2019-2020 | 0.42 | 0.35 (0.42 with the fixes below) | 23 (OpenVINO GPU) |
| R3D-18 | 2017 paper, weights 2019 | 0.45 | 0.37 (whole network trained) | 4.5 (PyTorch XPU) |
| R(2+1)D-18 | 2017 paper, weights 2019 | 0.50 | - | 6.9 (PyTorch XPU) |
| CLIP ViT-B/32 (teach today) | 2021 | 0.50 | - | 2.8 (OpenVINO GPU) |
| V-JEPA 2 ViT-L | 2025 | 0.50 | - | 68 (PyTorch XPU) |
| DINOv2-L | 2023 | 0.58 | - | 31 (OpenVINO GPU) |
| **SigLIP2 base/16 @256** | 2025 | **0.555** | - | **10** (OpenVINO GPU) |
| SigLIP2 so400m/14 @384 | 2025 | 0.625 | - | 133 (OpenVINO GPU) |

¹ Model time only, random input, after warm-up (`tools/teach_lab/bench_speed.py`):
16 frames per window for Intel, the 3D CNNs and V-JEPA 2; 8 for the image encoders.

5-fold cross-validation over all 130 videos with 20+ clips in a class
(`eval_big.py`), to check the leaders beyond one split:

| head on | acc | balanced | top-3 |
|---|---|---|---|
| SigLIP2 base | 0.576 | 0.500 | 0.747 |
| SigLIP2 so400m | 0.602 | 0.525 | 0.779 |
| SigLIP2 so400m + DINOv2 | 0.639 | 0.584 | 0.779 |
| SigLIP2 so400m + DINOv2 + V-JEPA 2 | 0.656 | 0.579 | 0.804 |

What the numbers say:

- **The old video models were trained on Kinetics-400 at low resolution.** On
  this footage, what is in the frame matters more than how it moves, and the
  recent image models learned from far broader data.
- **Training more of the network made R3D worse.** Frozen r3d_18 with a head
  reached 0.45; the whole network fine-tuned reached 0.37, with 95 %
  training accuracy. With 135 source videos, fine-tuning memorises the
  training videos.
- **SigLIP2 base is the default:** 2.5 points under so400m on the 5-fold test,
  at a thirteenth of the time per window. so400m is the "best quality"
  option.
- **Mistakes, not accuracy:** going from 0.45 to 0.62 cuts wrong labels from
  55 to 38 in 100, about 30 % fewer.

## Open vocabulary: suggestions, not a substitute for teaching

SigLIP2 matches images to text, so any typed action can be scored without
training. Typing this dataset's own class names, with no training
(`tools/teach_lab/zero_shot.py`):

| | accuracy | balanced | top-3 |
|---|---|---|---|
| SigLIP2 base, text only | 0.104 | 0.163 | 0.277 |
| SigLIP2 so400m, text only | 0.162 | 0.250 | 0.332 |
| always the largest class | 0.139 | | |
| SigLIP2 base, taught head (5-fold) | 0.576 | 0.500 | 0.747 |

For categories that web image-text data covers poorly, text alone is at
chance. It works as a way to start and to search, and teaching from examples
does the rest. Everyday actions, the kind Kinetics lists, are what these
models learned from, so the Kinetics-700 suggestions should do much better
than this. That is **not measured yet**, and it should be before the release
notes promise anything.

## Lessons for the new pipeline

Measured on the Intel trainer, one change at a time, on the same 445 held-out
clips. The Intel code is being retired, so these are not fixed there. They
are the rules the SigLIP2 pipeline starts from.

**Input.** "Fixed head" is a small head trained on that run's cached
features, which is less noisy than the trainer's early-stopped number.

| change from the trainer as shipped | trainer | fixed head |
|---|---|---|
| as shipped: person box squashed to 224², first ~2 s, encoder input normalised twice | 0.348 | 0.365 |
| person box, proportions kept | 0.348 | 0.343 |
| whole clip frame, squashed | 0.335 | 0.371 |
| whole clip frame, proportions kept | 0.375 | 0.387 |
| 16 frames across the whole clip (person box squashed) | 0.299 | 0.339 |
| whole frame, proportions kept, across the whole clip | 0.355 | 0.394 |
| encoder fed what its IR declares (BGR, 0-255) | 0.371 | 0.407 |
| the right input + person box, proportions kept | 0.398 | 0.409 |
| the right input + whole frame, proportions kept + whole clip | **0.418** | **0.443** |

**Recipe** (the trainer's own features, three seeds each, `recipe_ablation.py`):

| change | held-out |
|---|---|
| trainer recipe, rebuilt | 0.318 |
| class weights at power 0.5 instead of inverse frequency | **+3.0** |
| a linear layer instead of the 2-layer MLP | +2 |
| best epoch actually restored | +0.3 |
| standardising the encoder features | −2.7 |

The rules that follow:

1. **Split by source video**, always, for validation and for picking an epoch.
2. **Feed each encoder exactly what its model card declares.**
   action-recognition-0001's IR subtracts the ImageNet mean and flips the
   channels itself (`mean_values`, `reverse_input_channels` in the IR), so it
   expects BGR 0-255. `model_training` normalised twice, which left the
   encoder an almost flat image with red and blue swapped (values −11 to +10
   before the IR subtracts ~120). Analysis fed custom decoders a third
   variant: ImageNet-normalised once.
3. **The cropper is the crop.** The app's cropper already cuts clips around
   the people, one action per clip; the person box covers 82 % of the frame
   on median. A second crop at training time costs 3-4 points even when it
   keeps proportions: it cuts context the cropper kept, and the box jitters.
   Feed the cropper's clip whole.
4. **Never squash.** Keep proportions and pad. *(2026-10-02: true for the
   Intel encoder only. SigLIP2 was trained on squashed input, and squashing
   ties letterboxing on the 5-fold test, so rule 2 decides: squash. See
   `2026-10-02-siglip2-search-pose-export.md`.)*
5. **Frames across the whole clip.** With the wrong input it added nothing.
   The best Intel run used it, but together with the input fix and the
   framing, so its own share is not isolated. Prefer it anyway: the first 2 s
   of a clip can miss its action.
6. **Soften class weights** (power 0.5). Fully balanced weights push new
   footage into rare classes.
7. **Trust per class.** Sort into a class only above the confidence where the
   Wilson 80 % lower bound of its held-out precision reaches the target
   (`sort_trusted.py`). In the run that motivated this, a rare class that is a
   near variant of a large one absorbed 130 crops of the large class. With the
   bound it gets no folder, and 66 % of held-out clips are sorted at 77 %
   precision.

Other defects found on the way, recorded for anyone reading the retired code:

- `model_training/shared/dataset.py`: `compute_frame_indices` takes its 16
  frames from the start of the clip, which is the first 2 s of a 5 s clip.
  `crop_roi(frame, None)` returns a centre patch of the output size, not the
  whole frame. The ROI cache key ignores the sampling strategy and stores
  paths as written, so `D:/x` and `D:\x` are different keys.
- `model_training/intel/train.py`: `best_state = model.state_dict().copy()`
  is a shallow copy, so "best model loaded" restores the last epoch.
- Training time is mostly decoding: each frame is fetched with a seek
  (`cap.set(CAP_PROP_POS_FRAMES)`), and the encoder runs one frame per call.
  In one run, person boxes took 16 min, encoding 10.5 min and training 1.5 min.

## Plan

1. **SigLIP2 base export:** ONNX and OpenVINO IR. Time it on the A750, on a
   processor-only machine, and on CUDA (GTX 1060). DirectML and Core ML
   follow the routes R3D already proved.
2. **Analysis backend:** 8 frames per window from the cropper's clips,
   letterboxed, one encoder pass shared by every head. *(2026-10-02: squashed
   to 256², the model's own preprocessing; pose regions measured and not
   worth it.)*
3. **Taught actions:** a small head trained on the user's examples (about a
   minute on the processor), with per-class trust thresholds and validation
   split by source video.
4. **Typed actions:** SigLIP2's text tower, with Kinetics-700 as suggestions
   (Kinetics-700-2020 class list, CC BY 4.0, credited). Measure it on
   everyday actions first.
5. **Teach** moves from CLIP ViT-B/32 to the same encoder.
6. **Remove** the Intel and R3D analysis paths, `model_training/intel`,
   `model_training/r3d` and `kinetics_400_labels.json`. The release note says
   that custom action models trained on them stop working.
