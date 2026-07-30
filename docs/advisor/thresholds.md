# Thresholds, and why nothing scored

A detector reports something only when it is confident enough. Below that line
it saw the thing and said nothing. Two different failures look identical in the
report — a silent detector and an absent one both contribute zero points — so
it is worth separating them before changing any setting.

## Is it silent, or is it blind?

**Silent**: the detector knows the class and is finding it, but below the
threshold. Symptom: lowering the threshold produces detections, usually along
with false ones. Fixable with a setting.

**Blind**: the detector has no class for what you want. Symptom: lowering the
threshold to the floor produces nothing relevant, only unrelated classes. No
setting fixes this — see `training.md`.

The quickest test is to lower the threshold hard, once, and look at what
appears. If the answer is "more of the wrong things", the detector is blind.

## Conditions that suppress detections

Detection rates are not uniform across a video. Common causes of a detector
going quiet on part of your footage:

- **Low contrast between subject and background.** A pale subject against a
  pale background is the classic case; the detector has little edge signal to
  work with. Bright, evenly-lit, low-texture scenes are the usual offenders.
- **Scale.** Objects much smaller or much larger than the training examples
  are missed. Very wide shots and extreme close-ups both suffer.
- **Motion blur and compression.** Fast movement and low bitrate both destroy
  the fine detail the detector relies on.
- **Unusual angles.** Detectors trained on typical framing degrade sharply on
  overhead, tilted, or heavily distorted views — including uncorrected VR.

If detections drop out on a *particular kind of scene* rather than at random,
that is the signature of a training-data gap, not a threshold problem. Collect
examples under those exact conditions and extend the class — `training.md`.

## The highlight came out shorter than requested

In MAX mode the cut stops when it runs out of moments that scored anything, not
when it runs out of budget. A short result means few seconds scored at all.

Options, in order of preference:

1. **Lower the threshold** of whichever detector you are relying on, then check
   the report for moments that should not have been included.
2. **Add a signal.** Motion and audio fire on nearly any footage and will fill
   a cut when a semantic detector is sparse.
3. **Accept the shorter cut.** Padding it means including moments where nothing
   was detected — the length improves, the highlight does not.

EXACT mode behaves differently: it always reaches its target, filling with
zero-scoring seconds if it must. A cut that hits the exact length but feels
random is usually this.

## A note on raising thresholds

If the highlight is full of moments you did not want, raising the threshold is
the blunt fix and it costs recall everywhere. Prefer to work out what the
unwanted moments have in common first — the report groups tags per clip, and
the advisor flags a tag shared by clips you swapped away.
