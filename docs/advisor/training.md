# When the detector cannot see it at all

A label-based detector reports only what its vocabulary contains. If the class
you want is not in it, there is no threshold, weight, or rule that produces it —
you are scaling zero. This is the point at which settings stop being the answer.

Two routes out, and they cost very different amounts.

## 1. Teach a category from examples — minutes, no GPU

Point at a few frames that show what you mean. They are embedded with CLIP and
averaged into a prototype; scoring a new frame is then a dot product against
frames that are already indexed.

- No dataset, no training run, no GPU.
- Adding a category costs milliseconds; re-scoring an indexed video is instant.
- Works for anything visually distinctive, including things no detector has a
  class for.
- Gives you a *score*, not a box. If you need to know where in the frame it is,
  this is not enough.

**Try this first.** It answers most "the detector cannot see it" cases in the
time it takes to pick the frames, and if it fails it fails cheaply.

## 2. Train a class — hours, needs a GPU and labels

Worth it when you need boxes rather than scores, when the thing must be found
reliably enough to drive a rule, or when the prototype approach is not
discriminating well enough.

The loop:

1. **Collect frames where it actually fails.** Not more of what already works —
   more of the same teaches the model nothing it does not know. Pull frames
   from the conditions that break it.
2. **Label them** with `tools/labeler.py`.
3. **Train**, then export. Keep the training output — weights, curves, and the
   label previews rendered from your frames — out of the repository: it is
   large, and those previews are your footage.
4. **Re-run and compare** against the same video. The report's detection counts
   are the measurement.

## The part people get wrong

Detection failures are rarely uniform — a detector that "does not work" usually
works fine on most footage and collapses on a specific kind of scene. Low
contrast between subject and background, unusual scale, motion blur, and
uncommon angles are the usual causes (`thresholds.md` lists the full set).

**A training set that does not contain those conditions cannot fix them.** The
useful question is never "how many more examples do I need" but "which
conditions is it failing on, and do I have examples of those". Ten frames from
the failing condition beat a thousand more of what already works.

So before collecting anything: find several moments where it failed, look at
what they have in common, and collect *that*.

## Related

- `docs/DETECTION-GUIDE.md` — what each detector is and what it costs.
- `thresholds.md` — telling a silent detector from a blind one, which decides
  whether you need this page at all.
