# A source-video rule that finds too many videos

**Negative scenario.** The rule that recovers each clip's source video from
its file name splits one real video into many "videos".

**How it causes overfitting:** every split by video quietly becomes a split by
clip.
- Clips of one real video land on both sides, so every score leaks
  (`leaky-validation.md`).
- The trust check ("hits from 3+ videos") can be passed by one video
  counted as three.

**How you notice:** the trainer reports far more source videos than you
have.

**Measured:** a rule of "the digits before the first `_`" found 1,154 videos
in a set of 136.

**Instead:**
- Name clips `<video>_temp…` or `<video>_highlight…`, the rule in
  `modules/teach/benchmark.py`.
- Check that the reported count matches the real number of videos
  (`../dataset-positive.md`, Building 3).
