# Sorting new footage automatically: what works so far

Follows `2026-10-02-frame-encoder-runtime.md` (the shipped encoder, SigLIP2
base/16, and the action-head trainer). Here the question is phase 2 of the
plan: sort a new video's clips into a dataset's classes without a person
doing it, and leave a person only what the machine cannot vouch for.

Class names are the dataset's and do not appear here. Five classes the
reviewer singled out are called A-E:

- A: rare, and defined by movement.
- B: large, with a near-variant.
- C: shares most of its look with B.
- D: B's near-variant.
- E: a mid-sized class.

## The pipeline

1. **Cut and crop.** The video is cut into 5-s samples, and the cropper
   (`modules/crop`) makes one clip per person in each.
2. **Encode and score.** Each crop goes through an encoder and a head trained
   on the hand-sorted dataset.
3. **Sort what is trusted.** A crop goes into a class folder only when its
   score passes that class's trust threshold. Thresholds are set on held-out
   videos, with hits required from 3+ source videos.
4. **Group the rest.** Unsure crops are grouped by look (k-means on the
   encoder's vector), so a person can name a whole group at once.

`tools/teach_lab/sort_with_head.py` runs steps 2-4 with the app's own encoder
and an exported head.

## One unseen video, judged by eye

2,780 crops from one video outside the dataset, sorted four ways, and
reviewed by the person who sorted the dataset:

| sort | features | verdict |
|---|---|---|
| v2 | CLIP B/32 (8 frames) + R(2+1)D-18 + the Intel action encoder; logistic regression | best for D: fewest wrong; also the most precise for C |
| v3 | SigLIP2 so400m + DINOv2-L + V-JEPA 2-L (8 frames each); attention head; best guess forced | best overall: A recognised, B found widely, E almost entirely and right |
| v3b | as v3, with per-class trust thresholds | as v3, fewer sorted |
| v4 | SigLIP2 base/16 (4 frames, the shipped encoder); multi-label head; trust from 3+ videos | merges near-variants into the larger class (A into D, B into D, C into B), and finds E only a few times |

Checked against v3's folders:

- Of the crops v3 put in A, v4's top guess was D for half.
- Of those in B, v4 called 91 of 338 D.
- Of those in C, v4 called 19 of 23 B.
- Of those in E, v4's top guess was E for only 36 of 96. Its trust threshold
  for E (0.90) then let 3 through.

## The same question on the dataset

Every feature set gets the same head (per-frame projection, attention over
frames, class weights at power 0.5) and the same 5 folds by source video:
2,430 single-action clips, 38 classes, 134 videos. One seed, trained on an
Arc A750.

| features | accuracy | balanced | A R/P | B R/P | C R/P | D R/P | E R/P |
|---|---|---|---|---|---|---|---|
| base/16 (shipped) | 0.522 | 0.335 | .12/.21 | .47/.53 | .67/.68 | .74/.62 | .62/.55 |
| so400m | 0.595 | 0.375 | .20/.28 | .61/.71 | .74/.72 | .77/.63 | .75/.66 |
| **so400m + DINOv2 + V-JEPA 2 (v3)** | **0.627** | **0.415** | .48/.43 | .64/.65 | **.80/.73** | .79/.66 | **.77/.73** |
| V-JEPA 2 alone (video) | 0.515 | 0.313 | **.68/.71** | .62/.58 | .62/.55 | .72/.62 | .55/.56 |
| base/16 + V-JEPA 2 | 0.584 | 0.381 | .48/.57 | .64/.63 | .69/.71 | .82/.66 | .68/.54 |
| base/16 + R(2+1)D | 0.559 | 0.339 | .36/.33 | .60/.62 | .60/.65 | .80/.66 | .82/.61 |
| CLIP + R(2+1)D + Intel (v2) | 0.540 | 0.340 | .32/.44 | .58/.63 | .64/.68 | .77/.61 | .55/.47 |
| v2 + DINOv2 + V-JEPA 2 | 0.598 | 0.377 | .76/.49 | .66/.64 | .65/.62 | .82/.66 | .66/.57 |
| all of them | 0.624 | 0.389 | .68/.57 | .67/.69 | .68/.69 | .81/.69 | .74/.69 |

R is recall: share of the class's clips named as it. P is precision: share
of clips named it that are it.

What it says:

- **The image encoder's size is most of the gap.** so400m alone is 7 points
  above base/16, and it is clearly better on B, C and E: the near-variant
  confusions the review found.
- **Movement is what defines A.** A video model (V-JEPA 2) finds A far better
  than any image model. With so400m and DINOv2 it adds 3 more points overall.
- **v3's three models are the best set** (0.627). Adding the two older
  motion models on top changes nothing (0.624).
- **v2's edge on D does not show here.** Its precision for D is the lowest
  (0.61). On the one video it was the most conservative sort, which looks
  cleaner by eye: fewer wrong, but fewer found. The two views measure
  different things, and the reviewer's verdict stands for that video.
- **Agreement with the review:** the dataset ranks the sets the way the review
  of v3 against v4 did.

## What this means for the app

Two jobs with different budgets:

- **Analysis inside the app** runs on every machine, processor-only ones
  included.
  - Shared models must stay interchangeable, so there is one shipped encoder:
    SigLIP2 base/16 (chosen 2026-10-02).
  - Nothing here changes that: so400m is about 9x the work per frame, which a
    processor cannot afford.
- **Preparing a dataset** (sorting new footage for a person to check) runs
  rarely, on the machine of whoever is building the dataset, and its output is
  folders a person reviews.
  - Quality is all that matters there, and the heavy set is affordable on a
    GPU: so400m, DINOv2-L and V-JEPA 2-L together cost about 0.25 s per 5-s
    clip on an A750, about 12 minutes for 2,780 crops.

So the proposal: dataset sorting uses the strongest set available (v3's),
and the models people train and share stay on the shipped encoder. Sorting
only proposes folders, so its encoder never has to match anyone else's.

Before any of the heavy models ships, even as an optional download:

- **Check every model's licence:** weights, code and training toolkit, per
  the licence rule in `CLAUDE.md`. SigLIP2 is Apache-2.0. DINOv2 and
  V-JEPA 2 are to be confirmed.
- **Check the export:** ONNX and OpenVINO, faithful to PyTorch, as was done
  for base/16.

## Also measured

- **Analysis input.** Whole frames sort far worse than the cropper's clips
  (16 % trusted against 43 %). One crop per person on the sampled frames is
  close to the cropper (40 %) at a fraction of its cost. Details are in
  `2026-10-02-frame-encoder-runtime.md`.
- **Two views per clip at analysis** (8 frames, two 4-frame views averaged):
  +1.7 points at twice the encoder time.
- **Training the head on a GPU** (Intel XPU through PyTorch): one fold of the
  v3 set takes 33 s. The trainer runs on the processor today; for large
  feature sets the GPU is the better default.
