"""'More like these': re-rank a video by the matches the user said were right.

A text search ranks frames against a *word*. Where the model has only a vague
idea of that word, every frame of a video scores about the same and the order
is close to noise: measured on real footage, the frame-to-prompt cosine spanned
0.21-0.30 across a whole scene, and a moment the user was looking for sat
around rank 80 of 1080 with thirty shown. Rewording the prompt did not move it
reliably in either encoder.

The frames themselves are far easier to tell apart than the word is to match.
So once the search has turned up even one right answer, that answer is a better
query than the word was: average the frames the user confirmed, and rank the
video against that. On the same footage this put the missed moment first and
brought back the same handful of moments from two unrelated encoders.

Nothing is encoded here. The search already memoised one embedding per frame
(`llm.clip_index`), so this is arithmetic on that array and answers in
milliseconds.

Two details carry the result:

Centring. Frames of one video share most of their content, so their raw
cosines sit in a narrow band near 1 and the differences that matter are in the
last few percent. Subtracting the video's mean frame first leaves only what
distinguishes a frame from the rest of this video, which is the comparison the
user means.

Moments, not frames. A match lasts several seconds, so the best frames of a
ranking are mostly neighbours of each other: thirty frames were twelve places.
`moments` gives the ranking back as places, so a result count means that many
different things to look at.

Content-neutral, like the rest of search: it matches whatever it is shown, and
the examples are the user's own choices, made at runtime and never stored here.
"""
from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import numpy as np

from llm.clip_index import ClipFrameIndex, l2_normalize

# Frames this close to a confirmed match are taken as part of the same match and
# join the example. One frame is a thin example; its neighbours are the same
# thing from a slightly different instant, at no cost to the user.
EXAMPLE_SPREAD_SECONDS = 1.0

# The entry on a video's right-click menu, beside "Teach a model to find this".
MENU_TEXT = "👍  More like this moment in the whole video"

# Frames nearer than this belong to one moment.
MOMENT_GAP_SECONDS = 5.0


class Moment(NamedTuple):
    """A stretch of video the ranking picked out, and its best frame."""

    start: float
    end: float
    timestamp: float      # the best-scoring frame: where to seek
    score: float
    frames: int


def _ordered(index: ClipFrameIndex) -> np.ndarray:
    """Row order by time. A memo is filled in whatever order scans ran."""
    return np.argsort(index.timestamps, kind="stable")


def example_rows(index: ClipFrameIndex, timestamps: Sequence[float],
                 spread: float = EXAMPLE_SPREAD_SECONDS) -> list[int]:
    """Rows of `index` within `spread` seconds of any of `timestamps`."""
    ts = index.timestamps
    rows: set[int] = set()
    for t in timestamps:
        rows.update(np.where(np.abs(ts - float(t)) <= spread + 1e-6)[0].tolist())
    return sorted(rows)


def centred(index: ClipFrameIndex) -> np.ndarray:
    """Unit vectors for what sets each frame apart from this video's average."""
    emb = np.asarray(index.embeddings, dtype=np.float32)
    if not len(emb):
        return emb
    return l2_normalize(emb - emb.mean(axis=0, keepdims=True))


def score_like(index: ClipFrameIndex, timestamps: Sequence[float],
               spread: float = EXAMPLE_SPREAD_SECONDS) -> np.ndarray:
    """How much each indexed frame resembles the frames at `timestamps`.

    Returns [n] in [0, 1], in the index's own row order: the cosine between
    the frame and the examples' average, both centred on the video, with
    anything less alike than an average frame floored at 0. An example frame
    scores near 1; this is a likeness to *these examples*, not a probability,
    and it is comparable between frames of one ranking only.
    """
    rows = example_rows(index, timestamps, spread)
    if not rows:
        raise ValueError("None of the examples is in this video's index.")
    frames = centred(index)
    query = l2_normalize(frames[rows].mean(axis=0))
    return np.clip(frames @ query, 0.0, 1.0)


def moments(timestamps: Sequence[float], scores: Sequence[float], top_k: int,
            gap: float = MOMENT_GAP_SECONDS,
            min_score: Optional[float] = None) -> list[Moment]:
    """The `top_k` best places in a ranking, best first.

    Frames are taken best first. One within `gap` seconds of a moment already
    found joins it; otherwise it starts a new moment, until there are `top_k`.
    So no moment is made of frames weaker than the weakest moment's best frame,
    and a long match counts once however many of its frames rank high.

    `min_score` drops frames below it before any of that.
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    sc = np.asarray(scores, dtype=np.float64)
    if len(ts) != len(sc):
        raise ValueError(f"{len(ts)} timestamps vs {len(sc)} scores")
    if top_k <= 0 or not len(ts):
        return []

    found: list[list[int]] = []          # each: rows, best first
    for row in np.argsort(-sc, kind="stable"):
        if min_score is not None and sc[row] < min_score:
            break
        t = ts[row]
        home = next((m for m in found
                     if ts[m].min() - gap <= t <= ts[m].max() + gap), None)
        if home is not None:
            home.append(int(row))
        elif len(found) < top_k:
            found.append([int(row)])
        else:
            break

    # A frame can land between two moments and belong to both; fold those.
    found.sort(key=lambda m: ts[m].min())
    merged: list[list[int]] = []
    for m in found:
        if merged and ts[m].min() - ts[merged[-1]].max() <= gap:
            merged[-1].extend(m)
        else:
            merged.append(list(m))

    out = []
    for m in merged:
        best = max(m, key=lambda r: sc[r])
        out.append(Moment(float(ts[m].min()), float(ts[m].max()),
                          float(ts[best]), float(sc[best]), len(m)))
    out.sort(key=lambda m: -m.score)
    return out


def more_like(index: ClipFrameIndex, timestamps: Sequence[float], top_k: int,
              since: float = 0.0, min_score: Optional[float] = None,
              gap: float = MOMENT_GAP_SECONDS) -> list[Moment]:
    """The `top_k` moments of the indexed video most like the frames at
    `timestamps`, best first. `since` leaves out everything before that time,
    the way a search started part-way through the video does."""
    scores = score_like(index, timestamps)
    keep = np.where(index.timestamps >= since - 1e-6)[0]
    return moments(index.timestamps[keep], scores[keep], top_k,
                   gap=gap, min_score=min_score)
