# Signal weights

Every second of the video gets a score. The score is the sum of points from
each signal that fired at that second, and the highlight is simply the
highest-scoring seconds, expanded into clips. So the weight table *is* the
definition of "interesting" — there is no other taste in the system.

## The signals

| signal | fires when | good for |
|---|---|---|
| scene changes | the shot cuts | structure, chapters, montage |
| motion events | sustained movement starts | activity of any kind |
| motion peaks | a spike of movement | impacts, sudden action |
| audio peaks | loudness rises sharply | reactions, impacts, speech onset |
| transcript keywords | a searched word is spoken | anything spoken aloud |
| objects | the detector sees a class it knows | presence of a known thing |
| actions | the action model recognises a movement | what someone is doing |

## Symptoms and what they mean

**Every point came from one signal.** The other weights are zero, so nothing
else could ever influence the cut. This is the single most common cause of a
disappointing highlight, and it does not look like a misconfiguration in the
report — every clip simply scores the same, because they all scored on the same
one thing. Give one or two more signals a non-zero weight.

**Every clip scored identically.** The ranking had nothing to rank; which
moments were chosen was effectively arbitrary among the ties. You want at least
one signal that varies continuously, such as motion or audio, rather than only
signals that are present or absent.

**A signal only appears in near misses.** It fires, but is outvoted everywhere.
If those are the moments you wanted, raise its weight. The near-miss table in
the report gives the timestamps to check before changing anything.

## Starting points

There is no universal table, because "interesting" depends on the footage. But
as a shape:

- Weight the signal that best matches what you are looking for **highest**.
- Give **two or three** others a small weight so agreement can break ties.
- Leave the rest at zero. A weight on a signal that fires constantly is noise:
  if something is true of every second, it distinguishes nothing.

## The multi-signal boost

Moments where several signals agree get multiplied. This is what makes a cut
feel deliberate rather than random — a loud moment with visible motion and a
recognised object is usually a better clip than any one of those alone.

It only helps if signals can actually coincide. With one detector enabled the
boost can never fire, and the report says so when that happens. Either enable
another detector or lower the agreement requirement.

## What weights cannot fix

If the detector has no class for what you are looking for, no weight makes it
appear — you are scaling zero. See `training.md`.
