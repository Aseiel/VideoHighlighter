# The frame encoder in the app: SigLIP2 base/16, one file, every route

Follows `2026-10-02-siglip2-search-pose-export.md`. Step 1 of the move from
Intel/R3D to a frozen encoder plus a taught head: the encoder runtime the
trainer and the analysis will share (`modules/vision/frame_encoder.py`,
identical in both editions).

**Decision (Szymek): SigLIP2 base/16 @256 is the one encoder for everyone.**
It is 2-4 points ahead of base/32 on the taught head, and on a GPU the two cost
about the same. A head only works with the encoder it was trained on, so one
default keeps shared models interchangeable. The price is the processor-only
path: about 4x base/32's time per frame.

## The file

`tools/export_frame_encoder.py` writes `models/siglip2-base-patch16-256/`:

- `vision.onnx`: the image tower with its weights **stored in fp16** and
  widened to fp32 by a Cast at load. 186 MB, half the fp32 export, and both
  runtimes read it, so there is no separate OpenVINO IR to ship.
- `encoder.json`: the id, the input description, the pinned model revision,
  and the vector PyTorch computes for a fixed probe image.

Measured on the Arc A750 and Ryzen 5 5600, 8 frames per call:

| runtime | fp32 ONNX (372 MB) | FP16 IR (186 MB) | fp16-weights ONNX (186 MB) |
|---|---|---|---|
| OpenVINO GPU, ms/frame | 1.44 | 1.43 | 1.43 |
| OpenVINO CPU, ms/frame | 73.9 | 77.9 | 79.0 |
| ONNX Runtime CPU, ms/frame | 155.7 | - | 154.3 |
| ONNX Runtime DirectML (A750), ms/frame | 12.0 | - | 11.8 |
| worst cosine vs PyTorch | 1.000000 | 0.999998-0.999999 | 0.999999-1.000000 |

Exporting twice gives byte-identical files. The model revision is pinned, so
a rebuilt app does not make the updater fetch the model again.

## Routes

`frame_encoder.load()` follows the `compute.backend` setting. Automatic is:

1. OpenVINO GPU on an Intel discrete GPU.
2. ONNX Runtime GPU: DirectML on Windows (NVIDIA, AMD, any DX12 card; the
   build has no CUDA provider for ONNX Runtime), Core ML on a Mac. This comes
   first when CUDA is present and the Intel GPU is integrated.
3. OpenVINO GPU on an integrated Intel GPU.
4. OpenVINO CPU, then ONNX Runtime CPU.

Each route must reproduce `encoder.json`'s probe vector (cosine 0.99 or more)
before it is used. A route that returns NaN, the wrong shape or a different
vector is skipped with a log line. All four routes were checked on this PC.

## Same accuracy through the app's runtime

The whole hand-sorted dataset (2,699 clips) was encoded through
`frame_encoder` (OpenVINO GPU, the app's own preprocessing) and compared with
the PyTorch features behind the 0.582 measurement:

- **Per clip, the worst frame's cosine:** min 0.99953, 1st percentile 0.99995,
  median 0.999996.
- **Held-out head**, same 29-video split, 3 seeds:

| features | held-out accuracy |
|---|---|
| PyTorch, squashed (the measurement) | 0.582 |
| app runtime, 8 frames per clip | 0.579 |
| app runtime, 4 frames per clip | 0.578 |

The differences are inside one seed's noise, and 4 frames cost nothing, as
measured before.

## Delivery

- **Free:** a fourth pack, `models-frame-encoder`, built by `build-packs.yaml`
  on a `rebuild_packs` run. It is published by `publish-packs.yaml` under a
  new tag (`packs-torch-<torch>-r<n>`), and `pack_manager` installs it on first
  use. The installer checkbox and the first-use prompt arrive with the first
  feature that needs the encoder.
- **Pro:** exported by `build-release.yaml` and bundled with `models/`.

## Step 2: the trainer, first run

`python -m model_training.action_head.train --data-path <dataset>` (identical
in both editions) runs the whole recipe:

1. Encodes each clip once: 4 frames across it, decoded front to back, cached.
2. Scores 5 folds by source video, which also pick the training length.
3. Sets per-class trust thresholds from those held-out predictions.
4. Trains the saved head on everything.

It writes `head.onnx` (`features [N, frames, 768]` -> logits, standardisation
inside) and `head.json` (encoder id, frames, classes, thresholds, held-out
scores; no paths, no frames).

On the hand-sorted dataset (`train/` + `val/` pooled; `test/` holds only
two-class folders and is left out). Encoding 2,430 clips took 3 min on the A750
(14 clips/s); the whole run took 8 min.

| classes kept | clips | videos | held-out acc | balanced | top-3 | trusted classes | sorted / right |
|---|---|---|---|---|---|---|---|
| 5+ clips (38) | 2,430 | 134 | 0.534 | 0.355 | 0.674 | 20 | 43 % / 78 % |
| 20+ clips (24) | 2,271 | 130 | 0.572 | 0.479 | 0.781 | 15 | 48 % / 76 % |

The 20+ row is like for like with the earlier 5-fold measurement (0.577 at 8
frames, squashed). The 14 classes with 5-19 clips cost about 4 points of
accuracy and most of the balanced score. Collecting more of them is the
cheapest gain; the trust thresholds already keep them from sorting new
footage wrongly. Training length barely matters: 750, 1,500 and 3,000 steps
gave 0.528, 0.526 and 0.534.

## Next

- Step 2, continued: the Training tab and teach run this trainer instead of
  `model_training.intel` / `model_training.r3d`.
- Step 3: analysis. One encoder pass per sampled frame, heads over 4-frame
  windows, gated by the trust thresholds.
- Step 4: remove Intel and R3D.
