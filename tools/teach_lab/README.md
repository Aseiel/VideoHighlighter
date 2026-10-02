# teach_lab — measuring and discovering on top of a hand-sorted dataset

Dev tools, not part of the app. They answer two questions about a dataset a
person already sorted into `train/val/test/<class>/` folders (the layout
`python -m modules.teach import` reads):

1. **How well can each kind of feature sort the way that person sorted?**
   Every score holds out whole source videos, so nothing is matched against
   footage from its own scene.
2. **What is in a new video?** Person-focused crops of it are grouped into
   proposals for classes, and sorted by a classifier trained on the dataset.

The class names are the dataset's own; nothing here knows any.

## The pieces

| Script | Does |
|---|---|
| `ds_features.py` | CLIP (4 frames), pose and motion features of every dataset clip, with its classes, split and source video |
| `video_features.py` | Motion-aware features: CLIP on 8 frames (mean + max), torchvision R(2+1)D-18 (Kinetics-400), the Intel action-recognition-0001 encoder/decoder in `models/intel_action/` |
| `eval_grouping.py` | Scores feature mixes against the sorting: grouping without labels (raw and in the learned space), nearest neighbours, a linear classifier, and teach's CLIP prototypes |
| `tune.py` | Classifier settings and per-class recall / most-confused pairs for one mix |
| `discover.py` | Groups a folder of crops (KMeans on CLIP + pose + motion + the uncropped sample's pose), with group folders, contact sheets and `groups.html` |
| `apply_classifier.py` | Trains on the dataset, picks a confidence threshold on held-out videos, sorts a video's crops into class folders, smooths over neighbouring windows, and regroups the crops in the learned space |
| `diagnose_grouping.py` | Do the features group by source video or by class? And does removing each video's mean, or a space learned from the dataset, change that? |
| `big_features.py` | Stronger frozen features kept per frame: SigLIP2 so400m/14 @384 and DINOv2-L/14 on 8 frames (Apache-2.0), V-JEPA 2 ViT-L on 16 frames (MIT). About 0.8 s a clip on an Arc A750 |
| `eval_big.py` | Linear probes on those, a small head trained over the per-frame features (scored as a classifier and as grouping of held-out videos), and whether the head's similarity carries to classes it never saw |
| `regroup.py` | Trains that head on the whole dataset and groups a video's crops in its space; each group says which dataset classes its crops sit nearest to. `--raw` blends the backbones' own view back in |
| `class_report.py` | Per class: clips, source videos, held-out recall and precision, what it is mistaken for, and how many of a new video's crops land in it |
| `sort_trusted.py` | Sorts a video's crops only into classes that earned it: a per-class confidence threshold where the Wilson lower bound of held-out precision reaches a target; other classes are suggestions only |
| `sort_with_head.py` | Sorts clips with a head from `model_training.action_head.train`, through the app's own frame encoder: hard links into `<action>/`, `<a>_<b>/` (a trusted pair, or two trusted actions) or `_unsure/`, plus every clip's top five in `sorted.csv`; can compare with an earlier sort |
| `compare_split.py` | Scores the head on the action trainer's own `--split-by-source` split, with any frozen encoder (`--blocks`), next to the trainer's validation accuracy |
| `siglip_base_features.py`, `r3d_features.py` | SigLIP2 base/16 per frame, and frozen torchvision r3d_18, for `compare_split.py` |
| `bench_speed.py` | Model time per analysis window for every encoder above, torch XPU and OpenVINO GPU |
| `decoder_ablation.py`, `recipe_ablation.py` | Different decoders, and the Intel trainer's own recipe changed one thing at a time, on that trainer's cached features |
| `bench_app.py` | The app's action-recognition call, headless, on any code checkout (e.g. a release tag next to its exe): its stage summary and wall time, with and without the annotated video and person detection, and a decode-only floor |
| `zero_shot.py` | Open vocabulary without training: each clip matched to class names through SigLIP2's text tower |
| `clip_frames.py` | The app's CLIP ViT-B/32 (its own loader and IR) on the same 8 frames as `siglip_base_features.py`, per frame |
| `search_eval.py` | Visual search on several encoders, same clips: text queries, search by example (k averaged examples from other videos), nearest class mean, and the head |
| `pose_regions.py` | YOLOX + RTMPose action regions (the `AdaptiveActionDetector` idea) and people boxes per frame, with SigLIP2 base on the whole frame and on each region, squashed and letterboxed |
| `export_siglip.py`, `quantize_siglip.py` | SigLIP2's image and text towers to ONNX and OpenVINO IR, checked against PyTorch; INT8 versions (NNCF) checked on real frames |
| `bench_cpu.py` | Every action-model candidate on this machine's processor (Intel encoder FP32/FP16/INT8, r3d_18 PyTorch and OpenVINO, SigLIP2 base/32 and base/16 FP16 and INT8), one window each |
| `bench_dml.py` | The exported image towers on ONNX Runtime DirectML (old NVIDIA, AMD): output checked against the CPU, ms per window |
| `ir_features.py`, `bench_siglip_ir.py` | Per-frame features from an exported IR, run as the app would; and the IR's speed per device, batch size and hint, next to CLIP's |

Crops come from the app's cropper (`modules.crop.actions.main`); samples from
`python -m modules.teach ... from-dataset`.

## A run

```bash
python tools/teach_lab/ds_features.py <dataset> ds.npz
python tools/teach_lab/video_features.py ds_video.npz --dataset <dataset>
python tools/teach_lab/eval_grouping.py ds.npz --extra ds_video.npz

python tools/teach_lab/discover.py <crops> <groups> --samples <samples>
python tools/teach_lab/video_features.py crops_video.npz --folder <crops>
python tools/teach_lab/apply_classifier.py ds.npz <groups>/features.npz <sorted> --crops <crops> --ds-extra ds_video.npz --crop-extra crops_video.npz --groups <groups>/groups.csv --learned-groups 20
```

With the stronger backbones (weights download from Hugging Face on first use, about 7 GB):

```bash
python tools/teach_lab/big_features.py ds_big.npz --dataset <dataset>
python tools/teach_lab/eval_big.py ds.npz ds_video.npz ds_big.npz --out eval_big.json
python tools/teach_lab/big_features.py crops_big.npz --folder <crops>
python tools/teach_lab/regroup.py <crops> crops_big.npz <learned groups> --ds ds.npz --big ds_big.npz
```

Pose features need `models/yolox/` and `models/rtmpose/`
(`tools/get_yolox_model.py`, `tools/get_rtmpose_model.py`).

## What one dataset showed (2,271 clips, 24 classes of 20+ clips, 130 videos)

| Features | Linear classifier, held-out videos: acc / balanced / top-3 |
|---|---|
| teach's CLIP prototypes (3 per class) | 0.37 / 0.35 / – |
| CLIP, 4 frames | 0.47 / 0.48 / 0.75 |
| CLIP 8 frames + R(2+1)D + Intel encoder | **0.55 / 0.50 / 0.81** |

- Pose and frame-difference motion added nothing measurable on top.
- Grouping without labels reached about 0.5 purity in the raw features and
  0.58 in the classifier's learned space: crops form a continuum, so density
  clustering (HDBSCAN) finds one blob — KMeans is used instead.
- The source video has to be read from the name before the cutter's suffix.
  The reader's old default, a leading number, counted 1,154 videos where there
  were 136, and every "held-out video" score leaked; the name before the
  first `_temp`/`_highlight` is now the default (`modules/teach/benchmark.py`).
- Classes under 20 clips are left out of the scores; a class built from one
  clip attracts whatever looks a little like it.

## Stronger backbones and a learned similarity (same dataset)

What this means for the app's action model (Intel vs R3D vs a head on a
frozen image encoder) is in `docs/plans/2026-10-01-action-models-measured.md`;
visual search, pose regions and the SigLIP2 export in
`docs/plans/2026-10-02-siglip2-search-pose-export.md`.

Why grouping lagged: the features group clips by **source video** more than
by class (KMeans NMI with the video 0.47, with the class 0.40). Inside one
long video that is camera angle, framing and light. Subtracting each video's
mean makes it worse (class NMI 0.40 -> 0.27): in a sorted dataset a video's
mean carries its classes too.

| Held-out videos | acc / balanced / top-3 | grouping purity / NMI |
|---|---|---|
| CLIP 8 frames + R(2+1)D + Intel encoder, linear | 0.55 / 0.50 / 0.81 | 0.58 / 0.56 (learned space) |
| SigLIP2 + DINOv2 + V-JEPA 2, linear | 0.61 / 0.55 / 0.86 | – |
| all six, linear | 0.63 / 0.55 / 0.87 | – |
| **SigLIP2 + DINOv2 + V-JEPA 2, head over per-frame features** | **0.66 / 0.58 / 0.80** | **0.71 / 0.67** |

- On its own each new backbone is no better than the old mix; together they are.
- A contrastive term whose positives are the same class in another video did
  not help (0.65 / 0.68 purity); `regroup.py` leaves it off (`--con 0`).
- The learned space does **not** carry to classes the head never saw: 6
  classes held out of training, grouped from unseen videos, NMI 0.31-0.44
  learned vs 0.36-0.54 in the raw backbones. Content the dataset has no class
  for is better kept apart by the raw view -- hence `regroup.py --raw`.
