# SigLIP2 base: visual search, pose-guided regions, and the exported model

Follows `2026-10-01-action-models-measured.md` (retire Intel and R3D, move
action recognition to a frozen SigLIP2 base plus a small taught head). Three
questions from that plan, measured on the same hand-sorted dataset, with whole
source videos held out as before:

1. Should **visual search** (CLIP ViT-B/32 today, as is teach) move to SigLIP2 base too?
2. Do **pose-guided action regions** (RTMPose) beat feeding the cropper's clip whole?
3. What do the **ONNX and OpenVINO exports** of SigLIP2 base cost on an Arc A750 and on a processor?

Short answers: (1) yes, as the one shared encoder, but processor-only
machines pay about 6x CLIP's time per frame. (2) No: on cropper clips a region
view adds about a point, which is inside the noise, and costs a person
detector, a pose model and a second encoder pass. (3) The exports are faithful
(held-out accuracy unchanged) and run at about 1.2 ms a frame on the A750 and
90 ms a frame on a 6-core processor.

## 1. Visual search: CLIP ViT-B/32 vs SigLIP2 base

*(In this edition: text search and teach use the encoder. Search by example
is a Pro feature; its rows are measurements of the same encoder, kept here
because they decide the shared encoder.)*

Same clips, same 8 frames per clip (decoded at short side 256, spread evenly).
CLIP runs through the app's own loader (`llm.clip_index.ClipEmbedder`, bundled
OpenVINO IR); SigLIP2 through its model card's processor. 2,271 clips, 24
classes with 20+ clips, 130 videos, 5 folds by source video
(`tools/teach_lab/search_eval.py`). so400m is there for scale.

| test | CLIP B/32 | SigLIP2 base | SigLIP2 so400m |
|---|---|---|---|
| **text search**, class names typed, nothing trained: closed-set accuracy | 0.069 | 0.104 | 0.161 |
| text search: mean AP when every clip is ranked by the query (chance 0.042) | 0.121 | 0.148 | 0.212 |
| text search: precision of the top 20 | 0.173 | 0.183 | 0.294 |
| **search by example**, 1 example clip from another video: mean AP | 0.176 | 0.189 | 0.188 |
| search by example, 5 examples averaged: mean AP | 0.195 | **0.221** | 0.233 |
| search by example, 20 examples averaged: mean AP | 0.206 | 0.237 | 0.252 |
| search by example, 5 examples: precision of the top 20 | 0.169 | 0.188 | 0.197 |
| nearest class mean (teach's prototypes, one per class): accuracy | 0.339 | 0.346 | 0.365 |
| taught head, 5 folds: accuracy | 0.541 | **0.590** | 0.611 |
| taught head, the 29 held-out videos (423 clips) | 0.537 | 0.600 | 0.638 |
| vector per frame | 512 | 768 | 1152 |
| image tower, OpenVINO, A750, 32 frames a call: ms per frame | 0.36 | 1.26 | - |
| image tower, OpenVINO, Ryzen 5 5600: ms per frame | 14.5 | 90 | - |

"Search by example" is what the app does: the example frames' vectors are
averaged into one query, and the held-out videos' clips are ranked by cosine.

What it says:

- **SigLIP2 base is better at every test, but the size of the gain depends on
  the test.** Search by example improves by about 13 % relative mean AP with 5
  examples (0.195 to 0.221). A taught head gains 5-6 points. Text search on
  this dataset's own categories stays weak for both: these are categories that
  web image-text data covers poorly.
- **Everyday wording is not measured here.** The dataset has no everyday
  classes. For scale only: the published ImageNet zero-shot accuracy is about
  63 % for CLIP B/32 and in the high 70s for SigLIP2 B/16. Those are not this
  app's numbers.
- **One encoder for everything.** Actions, teach and search can share one
  image pass per frame and one cached vector per frame. Keeping CLIP for
  search would mean two models to download and two indexes per video, and an
  action head can never be reused as a search query.
- **The cost lands on processor-only machines.** On the A750 SigLIP2 base still
  encodes about 800 frames a second, so a one-hour video indexed at one frame a
  second takes about 5 s. On the Ryzen 5 5600 the same index takes about 5.4 min
  instead of CLIP's 52 s.

**Recommendation:** move visual search and teach to SigLIP2 base together with
actions, as one encoder and one per-frame vector cache. The open question is
the processor-only path: accept the 6x, sample search more sparsely there, or
keep CLIP as a processor-only search fallback. The last option means two
indexes, so it is the least attractive.

What changes in the search code when it moves:

- SigLIP scores with a **sigmoid**, `sigmoid(scale * cos + bias)` (scale 112.9,
  bias -16.8 for base), not CLIP's softmax over a positive and contrastive
  negatives. Each query becomes an absolute probability, so the negative-prompt
  scheme in `clip_prefilter` is no longer needed.
- Text is lower-cased and padded to 64 tokens ("max_length"), as SigLIP2 was
  trained. The tokenizer is Gemma's (256k tokens), which is why the text tower
  is large (below).
- Old CLIP indexes are invalidated by model id already (`ClipFrameIndex.matches`).

### Which encoder: every candidate on the same tests

Added the same day after a question about speed. SigLIP2 also comes as
**base/32 @256** (64 patches, Apache-2.0), which costs about what CLIP B/32
does. Split = the trainer's 29 held-out videos (445 clips, 35 classes), mean of
3 seeds. 5-fold = 2,271 clips, 24 classes. Window = 8 frames (16 for Intel and
R3D), model time only, OpenVINO unless noted. The A750 column here is from
the first runs and mixes setups (Intel one frame per call, r3d_18 in PyTorch,
the image encoders at 32 frames per call). The like-for-like timings are in
the two tables under "Old GPUs and processor-only machines" below.

| encoder | split, 3 seeds | 5-fold head | search by example, k=5 | A750 ms/window | Ryzen 5 5600 ms/window | text search |
|---|---|---|---|---|---|---|
| Intel action-recognition-0001 | 0.42 (head), 0.35 (its trainer) | - | - | 23 (one frame per call) | 215 | no |
| r3d_18 (PyTorch) | 0.45 | - | - | 4.5 | 214 (122 on OpenVINO) | no |
| CLIP ViT-B/32 (bundled today) | 0.506 | 0.541 | 0.195 | 2.4 | 116 | yes |
| **SigLIP2 base/32 @256** | 0.537 | 0.550 | 0.220 | 3.1 | 152 | yes |
| **SigLIP2 base/16 @256** | 0.558 (0.582 own squash) | 0.590 | 0.221 | 9.2 | 720 | yes |
| SigLIP2 so400m/14 @384 | 0.634 | 0.611 | 0.233 | 133 | not measured | yes |

- **base/32 matches base/16 at search and is about as fast as CLIP.** Search
  by example 0.220 vs 0.221; on the processor it is faster than Intel or R3D.
- **base/16 is ahead where it matters most: the trained head**, by 2-4 points.
  On the GPU every model here takes 4-12 ms per window, so speed there does
  not decide between them (below).
- **CLIP B/32 has no case left** besides already being bundled. SigLIP2
  base/32 is as good or better on every test at about the same cost.
- A head only works with the encoder it was trained on. A shared model has to
  name its encoder, and the app should pick one default for everyone, or
  shared models stop being interchangeable.

### Old GPUs and processor-only machines: keep Intel and R3D?

Raised for users on old NVIDIA cards without usable CUDA (e.g. a GTX 680).

- **DirectML runs these models.** DirectML needs a DirectX 12 GPU, and its
  stated minimum includes NVIDIA Kepler (GTX 600 series), AMD GCN 1 and Intel
  Haswell graphics. The Pro build already ships `onnxruntime-directml`, and R3D
  already uses that route (`modules/vision/r3d_onnx.py`). Both SigLIP2 ONNX
  exports run on it with output identical to the CPU's (worst cosine 1.00000,
  `tools/teach_lab/bench_dml.py`). Microsoft has put DirectML in maintenance
  mode (security fixes only; Windows ML on Windows 11 24H2+), so it works but
  is not going to get faster.
- **Measured on the A750 through DirectML** (not a GTX 680): base/32 31 ms and
  base/16 54 ms per 4-frame window. OpenVINO on the same card is about 10x
  faster, so DirectML is only for cards with no better route. **Not measured on
  a Kepler card.** By compute alone (a GTX 680 has about a fifth of the
  A750's fp32 throughput), expect very roughly 5x those numbers there.
- **Four frames per window lose nothing measurable.** Split, 3 seeds: base/32
  0.556 with 4 frames vs 0.537 with 8; base/16 (squashed) 0.576 vs 0.582. Cost
  scales with frames, so 4 frames halves it.

Processor, one window, Ryzen 5 5600, all measured. GFLOPs are counted from
each network's convolutions and matrix products:

| model | frames | GFLOPs | ms | GFLOP/s | held-out |
|---|---|---|---|---|---|
| Intel encoder, one frame per call (as the app runs it) | 16 | 117 | 207 | 567 | 0.42 |
| Intel encoder, 16 frames in one call | 16 | 117 | 170 | 691 | 0.42 |
| r3d_18, PyTorch | 16 | 81 | 215 | 378 | 0.45 |
| r3d_18, OpenVINO | 16 | 81 | 122 | 666 | 0.45 |
| **SigLIP2 base/32, OpenVINO** | 4 | 45 | **88** | 517 | **0.556** |
| SigLIP2 base/32, OpenVINO | 8 | 91 | 173 | 523 | 0.537 |
| SigLIP2 base/16, OpenVINO | 4 | 178 | 376 | 472 | 0.576 |
| SigLIP2 base/16, OpenVINO | 8 | 355 | 724 | 491 | 0.582 |

- **On a processor, speed is the work per window.** OpenVINO gets 470-690
  GFLOP/s from every one of these networks. The Intel encoder is not small per
  window: 16 frames through a ResNet-34-sized network at 224x224 is 117
  GFLOPs, more than r3d_18 (81) or SigLIP2 base/32 at 4 frames (45).
- **SigLIP2 base/32 at 4 frames is the fastest:** 1.4x faster than r3d_18 on
  OpenVINO, 2.4x faster than the Intel encoder as the app calls it, and 10+
  points more accurate than both. base/16 is 2-3x slower than either old model
  on a processor.

Arc A750, one window per call, all through OpenVINO:

| model | frames | ms per window |
|---|---|---|
| Intel encoder, one frame per call (as the app runs it) | 16 | 23.0 |
| Intel encoder, 16 frames in one call | 16 | 5.6 |
| r3d_18 | 16 | **4.0** |
| SigLIP2 base/32 | 4 | 5.1 |
| SigLIP2 base/32 | 8 | 6.9 |
| SigLIP2 base/16 | 4 | 8.0 |
| SigLIP2 base/16 | 8 | 12.4 |

- **On the GPU, r3d_18 is the fastest** and SigLIP2 base/32 is close to the
  Intel encoder fed properly. One window in 4-12 ms is far below the cost of
  decoding the video, so on a GPU speed does not separate these models;
  accuracy does. (Batching several windows per call lowers all of them; the
  Intel encoder's 23 ms is the app calling it one frame at a time.)
- **R3D is faster than the Intel encoder on both** the processor (122 vs
  170-207 ms) and the GPU (4.0 vs 5.6-23 ms).
- **Other processors: the order is not fixed, so measure it there**
  (`tools/teach_lab/bench_cpu.py`). Precision changes it even on this one.
  The Intel encoder ships an INT8 IR (`FP16-INT8`). On the Ryzen 5 5600 it
  takes 86-93 ms per window, 2.3x faster than the FP32 IR the app loads, and
  faster than r3d_18 (122-129 ms on OpenVINO, 215-246 ms in PyTorch, which is
  the app's processor path for R3D). INT8 did nothing for SigLIP2 here
  (base/32 at 4 frames: 89 ms FP16 IR, 96 ms INT8): this processor has no
  VNNI, and transformers gain less than CNNs from INT8 without it. Intel
  processors with VNNI or AMX should speed up INT8 for every model, by
  amounts that are not measured here. **So neither "R3D is faster" nor "Intel
  is faster" holds on every processor.** On this one, Intel INT8 and SigLIP2
  base/32 at 4 frames tie (86 vs 89 ms), and SigLIP2 is 13+ points more
  accurate. (The INT8 Intel encoder's accuracy was not measured.)
- Speed is no reason to keep Intel or R3D: on a processor the options land
  within about 2x of each other, and which is fastest depends on the processor
  and precision. On a GPU all are fast enough. The accuracy gap (10+ points)
  holds everywhere, so accuracy decides. The Intel
  encoder cannot use an NVIDIA or AMD card at all: OpenVINO's GPU plugin is
  Intel-only, so on a GTX 680 machine it already runs on the processor.

## 2. Pose-guided action regions vs the cropper's clip whole

The idea under test is `AdaptiveActionDetector.detect_action_region` from
`model_training/shared/detection.py`. YOLOX proposes people and RTMPose keeps
the up-to-two largest proposals that hold a body (as the cropper does). The
body part that moved between sampled frames picks upper, lower or full body,
and the padded extent of those keypoints becomes the region. A simpler
variant, "people", is the kept people's boxes merged with 10 % padding. Each
view goes through SigLIP2 base on 8 frames per clip, and the head is trained
on the action trainer's 29-video split (445 held-out clips;
`tools/teach_lab/pose_regions.py`, `compare_split.py --seeds 3`).

On these clips, 79 % of sampled frames have a body and 51 % have two. The
action region covers 80 % of the frame on median, and the people box 94 %
(a frame with nobody found counts as the whole frame). The cropper has already
cut around the people.

| input to SigLIP2 base | 3 seeds | mean |
|---|---|---|
| whole clip frame, squashed to 256² (the model card's preprocessing) | 0.587 0.584 0.575 | **0.582** |
| action region only | 0.571 0.551 0.566 | 0.563 |
| people box only | 0.604 0.578 0.557 | 0.580 |
| whole frame + action region (two views per frame) | 0.573 0.593 0.607 | 0.591 |
| whole frame + people box | 0.578 0.589 0.609 | 0.592 |
| whole frame + the pose vector (people, torso and relative layout, 166 numbers) | 0.557 0.607 0.580 | 0.581 |

The same views letterboxed instead of squashed: whole 0.535, region 0.545,
people 0.548, whole + region 0.558, whole + people 0.565.

- **The region alone loses about 2 points.** Whole + region gains about 1
  point, while one head's accuracy moves 1-4 points with its seed on 445
  clips. Nothing here beats the clip whole by more than the noise.
- **The cost is not small:** a person detector and RTMPose on every analysed
  frame, plus a second SigLIP2 pass for the region view. The scripts here are
  bound by video decoding, so they do not measure that cost well.
- So the 2026-10-01 rule stands: **the cropper is the crop.** Pose-guided
  regions might still help on footage that never went through the cropper
  (wide shots with small people). That is untested, and it is not the
  analysis input the plan uses.
- **Found on the way: RTMPose on the Arc GPU returns NaN.** With OpenVINO
  2026.4 at the GPU's default f16 precision, every keypoint score came back NaN
  (46 of 46 boxes tried). CPU, and GPU with `INFERENCE_PRECISION_HINT=f32`,
  agree. NaN never passes a threshold, so callers see "no body".
  `build_pose_estimator()` defaults to the GPU, and the cropper uses it. This
  needs its own fix in `modules/vision/pose_backend.py`. The benchmark ran at
  f32.

### Squash or letterbox: correcting rule 4 of 2026-10-01

Rule 4 ("never squash, keep proportions and pad") came from the Intel encoder.
For SigLIP2 it does not hold. Its processor squashes to 256x256, which is how
it was trained. On the 29-video split, squashing beat letterboxing (0.582 vs
0.535, all three seeds). On the 5-fold test the two tie: head 0.577 vs 0.577,
search by example with 5 examples 0.229 vs 0.217. So the rule is rule 2 again:
**give each encoder the preprocessing it was trained with.** For SigLIP2 that
is squashing, which is also the cheaper option. Small preprocessing
differences (decode size, resize filter) moved the split by about 2 points:
the model card's processor gave 0.558 and the squash here 0.582. Below about 3
points, a single split cannot tell two setups apart.

## 3. SigLIP2 base exported: ONNX and OpenVINO IR

`tools/teach_lab/export_siglip.py` writes both towers as ONNX (opset 17,
dynamic batch) and OpenVINO IR (FP16 weights), plus `siglip.json`: input size
256, mean/std 0.5, text length 64, lower-case, logit scale and bias.
`quantize_siglip.py` adds an INT8 image tower (NNCF post-training, 150 dataset
clips for calibration) and an INT8-weights text tower.

**Faithful.** Worst cosine against PyTorch on the same inputs: ONNX Runtime
1.000000. OpenVINO CPU 0.999999 and GPU 0.999998 (image), 1.000000 and
0.999999 (text). All finite on the GPU, unlike RTMPose. End to end
(`ir_features.py`: decode, 8 frames, OpenVINO on the GPU, then the head on the
29-video split):

| image tower | held-out accuracy, 3 seeds |
|---|---|
| PyTorch (same letterboxed input) | 0.535 |
| OpenVINO IR, FP16 weights, GPU | 0.539 |
| OpenVINO IR, INT8, GPU | 0.536 |

(Letterboxed because this check ran before the squash result; it compares
runtimes on identical input, which is the point.)

**Size on disk:**

| file | FP16 IR | INT8 IR | ONNX fp32 |
|---|---|---|---|
| image tower | 186 MB | 96 MB | 372 MB |
| text tower | 565 MB | 283 MB (weights) | 1,129 MB |

The text tower is mostly the 256k-token embedding table. Shipping only the
image tower plus precomputed text vectors (e.g. the Kinetics-700 suggestions)
would leave the text tower as an optional download for typed queries.

**Speed** (`bench_siglip_ir.py`; model time only, random input, after warm-up;
Arc A750, Ryzen 5 5600; OpenVINO 2026.4). A window is 8 frames:

| runtime | frames per call | ms per frame | ms per window | frames/s |
|---|---|---|---|---|
| OpenVINO GPU f16, latency | 1 | 4.76 | 38.1 | 210 |
| OpenVINO GPU f16, latency | 8 | 1.54 | 12.3 | 651 |
| OpenVINO GPU f16, latency | 32 | 1.26 | 10.1 | 794 |
| OpenVINO GPU f16, throughput (4 in flight) | 32 | 1.16 | 9.2 | 865 |
| OpenVINO GPU, INT8, throughput | 32 | 1.08 | 8.6 | 925 |
| OpenVINO CPU f32, latency | 8 | 108.9 | 871 | 9 |
| OpenVINO CPU f32, throughput | 32 | 90.0 | 720 | 11 |
| OpenVINO CPU, INT8, throughput | 32 | 80.3 | 643 | 12 |
| ONNX Runtime CPU fp32 | 8 | 144.5 | 1,156 | 7 |
| *CLIP B/32, OpenVINO GPU f16, throughput* | 32 | 0.30 | 2.4 | 3,371 |
| *CLIP B/32, OpenVINO CPU f32, throughput* | 32 | 14.5 | 116 | 69 |

Text tower, one typed query: 3.5 ms on the GPU, 25 ms on the processor.

- **GPU:** batch the frames. One frame per call costs four times as much per
  frame as 32. That matches yesterday's 10 ms a window.
- **INT8 halves the size but barely changes the speed** on this hardware: 7 % on
  the GPU, 11 % on the processor (the Ryzen 5 5600 has no VNNI instructions,
  which is where INT8 gains most on a processor). It is worth measuring on a processor that has VNNI or AMX before
  deciding. As a download-size option it is free: accuracy did not move.
- **Processor:** ONNX Runtime is 30-60 % slower than OpenVINO. One window costs
  0.7 s, so analysing every 5-s window of a one-hour video at 8 frames takes
  about 8.6 min on this processor. Fewer frames per window is the lever there
  (cost scales with frames). Measure it against accuracy before choosing it.
- The GTX 1060 (CUDA / ONNX Runtime) and DirectML runs are still to do.

## Plan, updated

1. ~~SigLIP2 base export~~: done for OpenVINO and ONNX. Next: time the ONNX on
   CUDA (GTX 1060).
2. **Analysis backend:** 8 frames per window from the cropper's clips, **squashed
   to 256² (the model's own preprocessing)**, frames batched, one encoder pass
   shared by every head. No pose regions. Ship the image tower in FP16, or INT8
   to halve the download.
3. **Taught actions:** unchanged.
4. **Typed actions and visual search:** the text tower (or precomputed text
   vectors), sigmoid scoring with the model's scale and bias.
5. **Teach and visual search** move from CLIP ViT-B/32 to the same encoder and
   the same per-frame cache. Decide the processor-only path first.
6. **Remove** Intel and R3D: unchanged.
7. Separately: fix RTMPose's f16 NaN on the Arc GPU in `pose_backend.py`.
