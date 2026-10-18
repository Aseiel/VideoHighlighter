"""Edit-timeline clips that cover the same source footage.

Two clips overlap when their source ranges share time, and then that footage
plays twice in the cut. It is easy to do by accident -- drag in two signal bars
that cover the same moment, or add every bar in a row -- and occasionally done
on purpose (a replay), so the timeline points it out and offers to resolve it
rather than refusing it.

Clips are (start, end) source ranges in edit order, as EditTimelineScene keeps
them. Everything here is a pure function of that list.
"""
from __future__ import annotations

# Less shared time than this is not an overlap worth a word: a cut leaves its
# two halves touching at one instant, and float edges can graze.
MIN_OVERLAP = 0.25

# Trimming never leaves a sliver shorter than this behind.
MIN_PIECE = 0.5


def shared(a, b):
    """Seconds of source time two (start, end) ranges have in common."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def pairs(clips, min_seconds=MIN_OVERLAP):
    """Every overlapping pair as (i, j, seconds), i < j, in edit order."""
    out = []
    for i in range(len(clips)):
        for j in range(i + 1, len(clips)):
            s = shared(clips[i], clips[j])
            if s >= min_seconds:
                out.append((i, j, s))
    return out


def by_clip(clips, min_seconds=MIN_OVERLAP):
    """{index: [(other index, seconds), ...]} for every clip that overlaps."""
    out = {}
    for i, j, s in pairs(clips, min_seconds):
        out.setdefault(i, []).append((j, s))
        out.setdefault(j, []).append((i, s))
    return out


def groups(clips, pair_list):
    """Connected sets of overlapping clips, each a sorted index list.

    Connected rather than pairwise: A overlapping B and B overlapping C is one
    stretch of footage, and merging it is one clip, not two."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _s in pair_list:
        parent[find(i)] = find(j)
    members = {}
    for x in list(parent):
        members.setdefault(find(x), []).append(x)
    return sorted((sorted(m) for m in members.values() if len(m) > 1),
                  key=lambda m: m[0])


def merge(clips, group_list):
    """Each group becomes one clip spanning all of it, at the edit position of
    its first member; the other members are removed. Returns a new list."""
    replace, drop = {}, set()
    for g in group_list:
        replace[g[0]] = (min(clips[i][0] for i in g), max(clips[i][1] for i in g))
        drop.update(g[1:])
    return [replace.get(i, c) for i, c in enumerate(clips) if i not in drop]


def trim_away(clip, others, min_piece=MIN_PIECE):
    """The parts of `clip` that none of `others` already covers, in order.

    Empty when `clip` lies wholly inside the others: then trimming it means
    not keeping it at all. More than one piece when an other sits inside it."""
    pieces = [tuple(clip)]
    for o_start, o_end in others:
        nxt = []
        for p_start, p_end in pieces:
            if o_end <= p_start or o_start >= p_end:
                nxt.append((p_start, p_end))
                continue
            if o_start > p_start:
                nxt.append((p_start, o_start))
            if o_end < p_end:
                nxt.append((o_end, p_end))
        pieces = nxt
    return [p for p in pieces if p[1] - p[0] >= min_piece]


def trim(clips, new_indices):
    """Cut each of `new_indices` down to what the other clips don't cover.

    Clips are trimmed one at a time in edit order, each against everything
    that is left, so two new clips that overlap only each other keep the
    footage once rather than both losing it. Returns a new list; a trimmed
    clip's pieces take its place in the edit order."""
    out = [tuple(c) for c in clips]
    marked = [i in set(new_indices) for i in range(len(out))]
    i = 0
    while i < len(out):
        if not marked[i]:
            i += 1
            continue
        others = [c for k, c in enumerate(out) if k != i]
        pieces = trim_away(out[i], others)
        out[i:i + 1] = pieces
        marked[i:i + 1] = [False] * len(pieces)
        i += len(pieces)
    return out
