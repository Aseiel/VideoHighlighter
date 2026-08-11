# When every clip looks the same

Selection has no notion of variety. It takes the highest-scoring moments, and
if the highest-scoring moments all resemble each other, so does the cut. Each
clip can be individually correct while the highlight as a whole is repetitive.

This is a ranking property, not a bug: the score says how strongly a moment
matched, never how different it is from what has already been picked.

## Three ways out, cheapest first

**Swap individual clips.** Right-click a clip in the HIGHLIGHTS row of the
timeline and take the next best moment instead. This costs nothing — no
re-analysis, no re-encode — because the alternative is chosen from the score
that is already stored with the report. Repeated swaps keep offering new
moments and never repeat one. Undo puts the previous clip back.

Use this when the cut is *mostly* right and two or three clips are wrong.

**Raise coverage.** Spreading the cut across the video usually diversifies it
as a side effect, because different parts of a video tend to contain different
things. See `coverage.md`.

Use this when the repetition and the concentration are the same problem.

**Change what is being scored.** If one tag appears in nearly every clip, the
signal producing it is doing all the work. Lower its weight and give another
signal a say, so moments have to be interesting in more than one way to be
picked. See `weights.md`.

Use this when swapping just produces more of the same — that means the whole
ranking is dominated by one thing, and no amount of re-picking escapes it.

## What the report tells you

- The **overview strip** shows concentration at a glance.
- The **tag rows** under each clip show what they have in common. A tag on
  nearly every clip is the thing making the cut monotonous.
- **Scored well, but not included** lists the alternatives. If they look just
  like the clips that were included, the problem is the ranking, not the
  selection.
