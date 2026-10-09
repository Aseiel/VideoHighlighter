# Keep the cropper's whole clip, and read frames across all of it

**Positive scenario.**
- The model sees the cropper's clip as it is, without a second crop.
- Its frames are spread over the whole clip (frame `i` of `k` at
  `(i + 0.5) / k`), not taken from the start.

**Why it helps:**
- The cropper already cut one person or action per clip, and a second crop
  removes context it kept.
- An action can sit anywhere in a 5 s clip, so frames across all of it
  contain it.

**Measured** (Intel trainer, same 29 unseen videos,
`docs/plans/2026-10-01-action-models-measured.md`):

| input | trainer | small head |
|---|---|---|
| as shipped: person box squashed, first ~2 s, wrong colours | 0.348 | 0.365 |
| right colours + whole frame, proportions kept, frames across the clip | **0.418** | **0.443** |

**How solid:**
- The combined gain is 7-8 points.
- Frames across the clip measured *alone*, with the wrong colours still in
  place, gained nothing. Its own share is not isolated.
- It is kept because the first 2 s of a clip can miss its action entirely.

**Where it lives:** `model_training/action_head/features.py`
(`sample_indices`).

**Related negative:** `../negative/frames-from-clip-start.md`,
`../negative/second-crop.md`.
