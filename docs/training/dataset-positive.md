# Positive scenario: a dataset and a sort that teach well

What to do when building, growing or automatically sorting a dataset for an
action model. Each rule says why, with the measurement behind it. The
mistakes these rules prevent, one page each with how to spot them, are in
`negative/`. The measured gain behind each rule has its own page in
`positive/`.

## Building the dataset

1. **Count source videos, not clips.** Clips of one video are near-copies:
   2,711 clips from 135 videos is about 135 independent examples of a scene.
   More distinct videos is what raises accuracy. More clips of a video
   already in the set does not (`2026-10-04-adding-reviewed-footage.md`).
2. **Get every class into 3 or more source videos.** A class is only trusted
   when its held-out hits come from 3+ videos. Below that it is learned but
   never reported, and with fewer than 5 single-action clips it is left out
   of training entirely.
3. **Name clips so their source video can be recovered:**
   `<video>_temp…` or `<video>_highlight…`. Then check that the number of
   groups the trainer reports is close to the number of real videos.
4. **Use the cropper's clips, whole.** The cropper (`modules/crop`) cuts one
   clip per person or action. It is the crop, and it keeps the dataset
   maintainable.
   - Sorting on the cropper's clips: 43 % of clips trusted.
   - Sorting on whole frames: 16 %.
   - A second crop at training time on top of it cost 3-4 points
     (`2026-10-01-action-models-measured.md`).
5. **Show two actions together in a folder named `a_b`.** The head scores
   each action on its own, so such clips teach that both can be present.
   Taught pairs get both actions into the top two on unseen videos about
   42 % of the time. Pairs it never saw are found 0-4 % of the time.

## Growing it with a reviewed video

After a new video has been sorted and checked by hand
(`2026-10-04-adding-reviewed-footage.md`):

1. **Add it as one source video:** rename its crops
   `<video>_highlight_<crop>`, never one "video" per crop.
2. **Take at most ~50 crops per action**, spread across the video.
3. **Prefer actions the dataset is thin on.** A thin action given a
   moderate number of new examples gained 7-10 points of recall.
4. **Keep "no action" and new actions too, capped the same way.** They
   become usable once 3+ videos have them.
5. **Judge the change on the dataset's own videos,** not on the trainer's
   overall held-out line. The new video is a held-out fold of its own and
   drags that average down even when nothing got worse.

## Sorting new footage automatically

1. **Sort only what is trusted.** A crop goes into a class folder only when
   its score passes that class's threshold. The threshold is the score at
   which held-out precision reaches 0.7 at the Wilson 80 % lower bound, with
   hits from 3+ source videos (`model_training/action_head/trust.py`).
   - Measured: 66 % of held-out clips sorted, 77 % of them correctly.
   - Everything else is a suggestion for a person.
2. **Group the unsure rest by look** (k-means on the encoder's vectors), so
   a person names a whole group at once.
3. **Use the strongest encoders available for sorting.** Sorting is a
   one-off job on the dataset builder's GPU, and its output is only a
   proposal.
   - Three large encoders together: 0.627 held-out.
   - The shipped base encoder alone: 0.522.
   - The large set separates near-variant classes that the base encoder
     merges (`2026-10-03-automatic-sorting.md`).
   - Models people train and share stay on the shipped encoder, so they
     work in every build.
4. **Have a person review the sorted folders,** then grow the dataset by the
   rules above.

## Training

1. **Hold out whole source videos,** always: for the score, for choosing the
   training length, and for every threshold. `model_training/action_head`
   uses 5 folds by source video.
2. **Spread frames across the whole clip,** at `(i + 0.5) / k`. An action can
   sit anywhere in a 5 s clip.
3. **Feed the encoder exactly the input its model card declares,** including
   colour order, range and resize.
4. **Soften class weights** (power 0.5). Fully balanced weights push new
   footage into rare classes; no weights ignore them.
5. **Start frozen, and fine-tune only what the data can carry.**
   - A frozen encoder with a small head is the baseline (minutes).
   - Training the encoder's top 4 of 12 blocks from that head (LP-FT) raised
     held-out top-1 from 0.533 to 0.603 on 5 folds. Every fold improved
     (`2026-10-08-overfitting-and-siglip2-fine-tune.md`).
   - For sorting it matters more: clips sorted with confidence went from
     44 % to 57 % at the same precision. On a new video it found 1.8× as
     many of a person's labels (`positive/partial-fine-tune.md`).
   - Choose its length on unseen videos too.
6. **Compare a new model with the old one on the same folds,** class by
   class, before replacing it.
