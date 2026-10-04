# Adding a reviewed video to the training set: more clips is not better

Follows `2026-10-03-automatic-sorting.md`. A new video was cropped, sorted by
the head and then checked by hand: 2,484 crops, each in its right action
folder (pairs as `a_b`, and a "no action" folder for crops that show none).
The obvious next step is to train on them. This is what that does.

Class names are the dataset's and do not appear here; actions are letters.

## Setup

- **Dataset:** 2,711 hand-sorted clips, 136 source videos, about 20 clips per
  video.
- **The reviewed video:** 2,484 crops, all of one video. They are added as one
  source video (renamed `<video>_highlight_<crop>`, so the trainer's
  source-video rule groups them), never as 2,484 videos of one crop each:
  that would put its neighbouring seconds on both sides of every test.
- **The question:** does the head get better on *other* videos? The same 5
  folds over the dataset's own source videos, 3 seeds; per fold one head is
  trained on the other dataset videos with or without the new crops, and both
  are scored on the same held-out dataset clips. The new video is never
  scored, so nothing it teaches about itself counts.
- Frame encoder SigLIP2 base/16, 4 frames, the trainer's head
  (`model_training/action_head`), 1,500 steps. Features cached, so each
  variant costs only head training.

## Everything in: worse

| trained on | top-1 | top-3 |
|---|---|---|
| dataset only | 0.526 ± 0.010 | 0.726 |
| + all 2,484 crops | 0.505 ± 0.009 | 0.702 |
| + 100, spread over the video's actions | 0.519 ± 0.003 | 0.721 |
| + 300, spread over the video's actions | 0.522 ± 0.007 | 0.716 |

One video became half of the training set. A head learns what is common in
its data, and here that is one scene, one cast, one camera: it gets better at
that video and worse at the rest. Smaller samples of the same video are
neutral, within the seed noise.

## Only where the dataset is thin: helps those actions, if capped

"Rare" = an action with fewer than N clips in the dataset; a crop joins when
any of its actions is rare. The cap is per action folder of the new video.

| added | crops | top-1 | recall of rare actions (34, macro) |
|---|---|---|---|
| nothing | 0 | 0.526 ± 0.010 | 0.282 ± 0.014 |
| everything | 2,478 | 0.505 ± 0.009 | 0.270 ± 0.007 |
| rare < 100 | 1,218 | 0.519 ± 0.006 | 0.274 ± 0.006 |
| rare < 100, max 50 per action | 472 | 0.517 ± 0.006 | 0.281 ± 0.006 |
| rare < 200 | 1,337 | 0.517 ± 0.008 | 0.271 ± 0.009 |
| rare < 200, max 50 per action | 564 | 0.521 ± 0.005 | 0.284 ± 0.001 |

The averages barely move; the actions do. Recall of single actions on
held-out dataset videos, mean of 3 seeds (dataset clips / dataset videos,
crops the new video has):

| action | dataset | new video | nothing | everything | rare < 200, max 50 |
|---|---|---|---|---|---|
| A | 25 / 10 | 37 | 0.24 | 0.43 | 0.31 |
| B | 49 / 15 | 20 | 0.78 | 0.88 | 0.88 |
| C | 48 / 15 | 5 | 0.41 | 0.49 | 0.51 |
| D | 53 / 12 | 267 | 0.64 | 0.58 | 0.61 |
| E | 82 / 32 | 62 | 0.75 | 0.65 | 0.68 |
| F | 44 / 6 | 2 | 0.52 | 0.20 | 0.56 |
| G | 21 / 10 | 38 | 0.00 | 0.00 | 0.00 |

- **A thin action that gets a moderate number of new examples improves**
  (A, B, C: +7 to +10 points). New scene, new people, same action: that is
  what a thin action lacks.
- **A flood hurts the action itself** (D: 267 crops against 53 in the dataset).
- **Uncapped, it hurts actions the new video barely has** (F falls from 0.52
  to 0.20). The cap is what protects them.
- **Some actions do not move at all** (G stays at 0 with 38 new crops): one
  more video of an action the head cannot see is not enough.
- Actions with a handful of held-out clips are noise either way; the table
  keeps those with more.

## What only one video cannot do

- **New actions cannot be scored or trusted.** Three actions exist only in
  the new video (one is "no action", 577 crops). The trainer scores an action
  on videos it never saw and trusts it only with hits from 3+ videos, so these
  are learned but never reported. Each needs examples from more videos.
- **The trainer's own held-out numbers get worse when the video is added**
  (0.52 → 0.46 single-action top-1, 17 → 13 trusted actions). That is not
  the head getting worse: the new video is one held-out fold of 2,607 clips,
  scored by heads that never saw it, and it dominates the average. Compare on
  the dataset's videos, as above.

## The rule this gives

When a reviewed video goes into the training set:

1. **Add it as one source video.**
2. **Add from each action at most ~50 crops**, spread over the video, not
   everything that was reviewed.
3. **Prefer the actions the dataset is thin on**; an action with plenty of
   examples gains nothing from one more scene and can lose from a flood.
4. **Keep "no action" and new actions anyway**, capped the same way: they
   become usable once 3+ videos have them.
5. **Measure on the dataset's videos**, not with the trainer's overall
   held-out line, before replacing a head.

More *videos* is what helps; more crops of one video mostly is not.

## Side result: is there an action in this crop?

The same review answers a question for the cropper: can a crop that shows no
action (a leg, someone's back) be told apart from one that does?

- The head's top score alone: AUC 0.66 (it was never taught "no action").
- A logistic regression on the encoder's mean vector, taught the review's
  577 "no action" against 1,907 action crops, scored on 10 held-out stretches
  of the video: AUC 0.80. Keeping 90 % of the action crops drops 39 % of the
  no-action ones.

One video only; it needs "no action" from more videos before it can gate
crops (`2026-10-03-automatic-sorting.md`, the cropper's open questions).
