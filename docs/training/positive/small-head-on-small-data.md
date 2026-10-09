# A small head when the data is small

**Positive scenario.** On top of a frozen encoder, the trained part is kept
small: a linear layer, or a small projection with attention over frames.

**Why it helps:** every trainable weight is a chance to memorise. With about
135 source videos, a smaller head has less room to learn the scene, and
more pressure to learn what the classes share.

**Measured:** a linear layer instead of the old 2-layer MLP decoder gained
+2 points held-out, on the same Intel features, three seeds
(`docs/plans/2026-10-01-action-models-measured.md`).

**How solid:** three seeds, one split by video.

**Where it lives:** `model_training/action_head/head.py`: a per-frame
projection, attention pooling and one embedding per class; about 0.5 M
weights.

**Related positive:** `frozen-encoder-first.md`.
