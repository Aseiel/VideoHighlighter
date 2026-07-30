# Coverage — best moments, or the whole story

Selection ranks every second and takes the best until the length budget runs
out. That is the right behaviour when you want the highlights and nothing else.
It is the wrong behaviour when the action concentrates: if the strongest
moments all sit in ten minutes of a two-hour video, the entire cut comes from
those ten minutes, and the rest is analysed but never represented.

The **Best moments ↔ Full story** slider controls this.

## What it does

The video is divided into as many buckets as there are clips in the budget, and
each bucket is capped on how much of the cut it may supply.

- **0.0 (Best moments)** — the cap is the whole budget, which is no constraint.
  Pure ranking, the original behaviour.
- **1.0 (Full story)** — the cap is one bucket's fair share, so every part of
  the video contributes and the cut spans the whole thing.
- **In between** — interpolates.

The cut never gets shorter. Anything the capped pass cannot place is placed by
ranking alone, so raising coverage changes *where* clips come from, never how
much you get.

## When to raise it

- The report's overview strip shows all clips bunched together.
- You are summarising something with a structure — a match, a lecture, a
  session — where the ending matters even if it scored lower than the middle.
- You want to know what is in a long video, rather than see its best parts.

## When to leave it at zero

- You want the peaks and genuinely do not care about the rest.
- The video is short enough that concentration is not a problem.
- Scores vary a lot and you trust them. Forcing coverage means taking weaker
  moments from quiet stretches in place of stronger ones.

## The trade-off is real

Coverage buys representativeness with quality. At 1.0 with a short target and a
long video, each bucket contributes about one clip, and some of those clips come
from stretches where nothing much happened. If the result feels flat, come back
down to 0.5 — that keeps the strongest moment and spreads the remainder.

## Related

- Individual clips can be exchanged without changing any setting; see
  `variety.md`.
- If the cut is concentrated *because* only one signal fires, coverage treats
  the symptom. See `weights.md`.
