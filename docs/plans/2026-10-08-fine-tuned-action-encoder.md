# Running a fine-tuned action model in the app

Status: approved 2026-10-08 and built (`frame_encoder.load_tower`,
`action_siglip.own_encoder` / `encoder_for`, `tools/install_action_model.py`,
`tests/test_action_own_encoder.py`). Follows
`2026-10-08-overfitting-and-siglip2-fine-tune.md`, where fine-tuning the top 4
blocks of SigLIP2 base/16 with the head raised held-out top-1 on unseen source
videos from 0.533 to 0.603.

## The problem

A taught head today is `head.onnx` + `head.json` on vectors from the shared
frame encoder (`modules/vision/frame_encoder.py`, id
`siglip2-base-patch16-256`). `action_siglip` only accepts a head whose
`"encoder"` is that id. A fine-tuned model changes the encoder itself, so its
head only makes sense on its own `vision.onnx`. Search, teach-by-example and
actions by name must keep using the shared encoder: their vectors are the ones
everybody's heads, text vectors and caches are measured against.

## 1. How a head folder declares its own encoder

The folder gains one file and `head.json` one block:

```
<action models>/<name>/
    head.onnx
    head.json
    vision.onnx        # the fine-tuned image tower, weights stored as fp16
```

```json
"encoder": "siglip2-base-patch16-256-finetuned",
"own_encoder": {
  "file": "vision.onnx",
  "preprocess": "siglip2-base-patch16-256",
  "dims": 768,
  "probe": [768 numbers]
}
```

- `preprocess` names the encoder whose input pipeline the tower was trained
  on. It must equal `frame_encoder.ENCODER_ID`, so the app feeds it exactly
  `frame_encoder.preprocess` (384 short side area, squash to 256, [-1, 1]).
  Anything else is refused with a sentence in the log: the app has one
  preprocessing, and a tower trained on another would get the wrong pixels.
- `probe` is the tower's vector for `frame_encoder.probe_pixels()`, as
  `encoder.json` carries for the shared encoder. Every route must reproduce it
  (cosine >= 0.99) before it is used, the same rule that caught RTMPose's NaNs
  on the Arc. Without a probe the folder is refused.
- `encoder` stays a free label; with `own_encoder` present it must *not* be
  the shared id (a tower that claims to be the shared encoder would let its
  head be fed shared vectors).
- A head without `own_encoder` behaves exactly as today.

`find_heads()` accepts a head when it is on the shared encoder **or** carries a
valid `own_encoder`. Newest-first and `VH_ACTION_HEAD_DIR` are unchanged.

## 2. Loading: OpenVINO / ONNX Runtime

No new runtime code. `frame_encoder.load()` is split so its route loop takes a
model path, a probe and an id instead of reading `encoder.json`:

- `frame_encoder.load()` - the shared encoder, as now.
- `frame_encoder.load_tower(path, probe, encoder_id)` - any tower with the
  shared architecture and preprocessing. Same `route_order` from
  `compute.backend`, same OpenVINO GPU compile cache, same probe check per
  route, same `FrameEncoder` (with `encoder_id` set to the head's id).

`action_siglip.encoder_for(head)` picks between the two; the run and
`modules/teach/sort.py` (scoring clips with a head) both use it, so both see
what the model will say in use. The head itself still runs on ONNX Runtime CPU
(2 MB, microseconds).

## 3. fp16 weight storage

The overnight export is 372 MB fp32. The shared encoder ships 186 MB because
`tools/export_frame_encoder.store_weights_fp16` stores large initializers as
fp16 with a Cast back to fp32 that both runtimes fold at load; compute stays
fp32. The same function is applied to the fine-tuned tower:

- new `tools/install_action_model.py <export folder> [--name N]`: copies
  `head.onnx`, writes `vision.onnx` fp16-stored, computes the probe on the
  fp32 tower with ONNX Runtime CPU (the trainer checked that against PyTorch:
  worst cosine 1.000000), checks the fp16 tower against it, and writes the
  folder into the managed action models folder.
- Measured on the overnight tower: 92 tensors stored as fp16, 372 -> 186 MB;
  vectors against fp32 on the probe and 20 dataset frames: worst cosine
  0.9999991 (ONNX Runtime CPU). Video 001 detections are compared in the real
  test below.
- Later, when fine-tuning moves into `model_training/action_head`, its export
  writes this layout directly and the tool becomes unnecessary.

## 4. Cost per window

**No second encoder pass.** Actions by name score whole frames; a trained head
scores person crops. They never shared a pass, and a run uses one or the
other (with a head installed, typed actions select among its classes). So with
a fine-tuned head the action pass runs the fine-tuned tower *instead of* the
shared one: same architecture, same cost (A750 ~8 ms per 4-frame window,
Ryzen 5600 ~376 ms).

What it does cost:

- **Disk**: 186 MB per fine-tuned model (a frozen head is 2 MB).
- **Memory**: ~372 MB when loaded (weights widen to fp32). The action pass
  does not load the shared encoder, so a run holds one tower at a time unless
  search runs in the same process.
- **First run**: one more OpenVINO GPU compile (seconds), then cached.

(The fine-tune note's "second encoder pass per window" is corrected here; it
would only apply to a future mode that scores typed actions *and* a head in
one run, which would be two passes even on a frozen head, because the inputs
differ.)

## 5. Sharing: model_hub

CLAUDE.md: a package is exactly `model.onnx` + `videohighlighter.json` +
`README.md` (+ `LICENSE`), and `model_hub/` stays identical to Pro. Today
`USABLE_TASKS = {"object_detection"}`: **no action model, frozen or
fine-tuned, can be shared through the hub yet.** So this change needs no
model_hub edit, and none is proposed.

Decision for when action sharing lands (recommended, not built now):

- **Option A (recommended): one graph.** The package's `model.onnx` is the
  tower and head merged, `pixels [N, frames, 3, 256, 256] -> logits`, fp16
  stored, ~188 MB (under the 250 MB cap). The file rules stay as they are,
  and it carries no material: weights and class names only. Installing splits
  nothing; the app can run the merged graph or keep it whole.
- Option B: allow an `encoder.onnx` next to `model.onnx`. Changes the layout
  rule in both editions and makes "exactly these files" a list per task.

A frozen head in the hub has its own question (it depends on an encoder the
package does not carry) and is out of scope here.

## Not changed

- Search, teach-by-example, actions by name: shared encoder only.
- Content: no class names anywhere in the repo; tests use made-up classes.
- Windows, crops, thresholds, pairs, the return value: unchanged.

## Test plan

- Unit: a head with `own_encoder` is found; refused without probe, with a
  foreign `preprocess`, with the shared id, or with a missing `vision.onnx`;
  `encoder_for` returns the head's tower; the run never loads the shared
  encoder for it; `load_tower` rejects a tower whose probe does not match.
- Real: install `D:\teach\ft\k4-lr1e-4`, run headless on video 001, compare
  with `action-head-siglip2-pairs`: detections, time, and against the user's
  hand review of 001.

## Results (2026-10-08)

Installed the overnight model with `tools/install_action_model.py`
(fp16-stored tower, 186 MB; worst cosine 0.9999997 against the export on ONNX
Runtime CPU and OpenVINO CPU), then ran headless on video 001 (52 min, 1257
windows, A750, OpenVINO GPU for both encoders and YOLOX). Neither head was
trained on 001. Scored against the maintainer's hand review of 001 (1203
reviewed 5-s windows, labels mapped through the training aliases):

| | frozen head (`action-head-siglip2-pairs`) | fine-tuned (own tower) |
|---|---|---|
| run time | 74 s | 72 s |
| detections / seconds labelled | 765 / 751 | 1567 / 1404 |
| reviewed labels found (of 1473 the heads know) | 19.1 % | **33.9 %** |
| of each head's trusted actions | 21.1 % | **38.7 %** |
| precision on reviewed windows | 55.3 % | 52.9 % |
| reviewed windows with any label | 476 | 760 |

Precision counts any label the review does not list for that window as
wrong, so both numbers are a floor.

Trusted share on the dataset, same clips, folds, head trainer and trust rule
(Wilson 80 % lower bound of held-out precision >= 0.7, hits from >= 3
videos), frozen encoder + head:

| encoder | top-1 | top-3 | clips sorted with confidence |
|---|---|---|---|
| Intel action-recognition-0001 (fixed input) | 0.418 | 0.632 | 18 % at 76 % |
| r3d_18 (Kinetics-400) | 0.409 | 0.628 | 11 % at 79 % |
| r(2+1)d_18 (Kinetics-400) | 0.414 | 0.627 | 14 % at 77 % |
| SigLIP2 base/16, frozen (the app today) | 0.533 | 0.733 | 44 % at 74 % |
| SigLIP2 base/16, top 4 blocks fine-tuned | 0.603 | 0.781 | 57 % at 75 % |

The old models trained whole by their own scripts were not scored this way.
They have no out-of-fold predictions on these folds. On the 29-video split
they were below their frozen + head versions (0.35-0.37 top-1).
