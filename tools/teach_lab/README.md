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
