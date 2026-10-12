"""Group samples by look before anything is named, then name group by group.

The other way into a project. ``add-class`` starts from words; this starts
from the footage: cut it, let the samples fall into groups that look alike,
and say what each group is. Naming twenty groups is much quicker than
deciding a few hundred samples, and it needs no examples up front.

What a group is trusted with, and what not (``docs/training/negative/
clusters-are-videos.md``): samples that look alike often share a scene, a
camera or a video rather than an action. So naming a group decides only the
samples a person was **shown** and did not mark as wrong. The rest of the
group is left undecided and goes through ``sort`` and ``review`` like any
other sample, where the shown ones are now the class's examples.

    group [--k N]              group every undecided sample not in a decided group
    group --split ID [--k N]   a mixed group into smaller ones
    name-group ID --name X     the shown tiles are X (minus --wrong ones)
    name-group ID --nothing    the shown tiles are none of the classes
    name-group ID --mixed      more than one thing: review it sample by sample
    name-group ID --reopen     undo whatever was said about it

Tiles are the group's first ``shown`` members, most typical first, numbered
from 1, the same in ``group-sheet`` and in the app.
"""
from __future__ import annotations

import math
import os
from collections import Counter
from typing import Callable, Optional

import numpy as np

from modules.teach import embed as embed_mod
from modules.teach import scoring
from modules.teach.project import (
    ACCEPTED, ACTIONS, AUTO, MIXED, NAMED, NEGATIVE, NONE, NOTHING, OBJECTS, OPEN,
    PENDING, REJECTED, SPLIT, UNSURE, GroupSpec, Project,
)

# Tiles a person is shown per group: enough to see what holds it together,
# few enough to check at a glance.
SHOWN = 12
MAX_GROUPS = 24
# Groups smaller than this are outliers, shown after the real groups.
MIN_GROUP = 3
# What a "none of these" group trains as, for actions (``background_class``):
# without one, an action model has no way to say "none".
BACKGROUND = "background"
DECIDED_BY = "group"
# decided_by for samples someone moved by hand (``move_samples``).
MOVED_BY = "move"


def default_k(n: int) -> int:
    """About how many groups ``n`` samples fall into: ~100 samples (one
    eight-minute video) make 7, and it stays between 2 and 12."""
    return max(2, min(12, int(round(math.sqrt(max(n, 1) / 2)))))


def _by(group_id: int) -> str:
    return f"{DECIDED_BY}:{group_id}"


def _decided_by_group(sample, group_id: int) -> bool:
    return sample.decided_by == _by(group_id)


def groupable(project: Project) -> list:
    """Undecided, readable samples not in a group someone already decided."""
    settled = {g.id for g in project.groups if g.state in (NAMED, NOTHING, MIXED)}
    return [s for s in project.samples
            if s.verdict == PENDING and not s.unreadable and s.group not in settled]


def group_samples(project: Project, embedder, *, k: Optional[int] = None,
                  split: Optional[int] = None, frame_reader: Optional[Callable] = None,
                  progress: Optional[Callable] = None) -> dict:
    """Cluster samples by look; tightest groups first, typical members first.

    Without ``split``: every undecided sample outside a decided group, and
    the open groups they were in are replaced. With ``split``: the undecided
    members of that group, which becomes SPLIT.
    """
    if split is not None:
        parent = project.get_group(split)
        if parent is None:
            raise KeyError(f"no group {split}")
        if parent.state in (NAMED, NOTHING, SPLIT):
            raise ValueError(f"group {split} is {parent.state}; reopen it first")
        wanted = set(parent.members)
        pool = [s for s in project.samples if s.id in wanted and s.verdict == PENDING
                and not s.unreadable]
    else:
        parent = None
        pool = groupable(project)
    if not pool:
        raise ValueError("nothing to group: every sample is decided or in a decided "
                         "group (cut some footage first)")

    cache = embed_mod.VectorCache(project.root, getattr(embedder, "model_id", ""))
    vectors = embed_mod.sample_vectors(
        pool, embedder, cache, project.settings.frames_per_sample,
        frame_reader=frame_reader or embed_mod.read_frames, progress=progress)
    for sample in pool:
        sample.unreadable = sample.id not in vectors
    pool = [s for s in pool if s.id in vectors]
    if not pool:
        project.save()
        raise ValueError("no frames could be read from any sample")

    if k is None or k <= 0:
        k = min(3, len(pool)) if parent is not None else default_k(len(pool))
    k = max(1, min(int(k), MAX_GROUPS, len(pool)))
    stack = embed_mod.unit(np.stack([vectors[s.id] for s in pool]))
    centres = scoring.spherical_kmeans(stack, k) if k > 1 else stack.mean(axis=0)[None]
    centres = embed_mod.unit(centres)
    likeness = stack @ centres.T
    assign = likeness.argmax(axis=1)

    found = []
    for c in range(len(centres)):
        rows = np.flatnonzero(assign == c)
        if not len(rows):
            continue
        order = rows[np.argsort(-likeness[rows, c])]
        found.append((float(likeness[rows, c].mean()), [pool[i] for i in order]))
    # Tightest first, but a group of one or two is "tight" only by having
    # nothing in it to disagree: those outliers go last.
    found.sort(key=lambda f: (len(f[1]) < MIN_GROUP, -f[0]))

    if parent is not None:
        parent.state = SPLIT
    else:
        # Open groups hold nothing anyone said; their samples are regrouped.
        project.groups = [g for g in project.groups if g.state != OPEN]
    next_id = max((g.id for g in project.groups), default=0) + 1
    made = []
    for tightness, members in found:
        group = GroupSpec(id=next_id, members=[s.id for s in members],
                          tightness=round(tightness, 4),
                          parent=parent.id if parent is not None else -1)
        for sample in members:
            sample.group = group.id
        project.groups.append(group)
        made.append(describe(project, group))
        next_id += 1
    project.save()
    return {"grouped": len(pool), "groups": made, "k": k,
            **({"split": parent.id} if parent is not None else {})}


def describe(project: Project, group: GroupSpec) -> dict:
    """What the app and ``status`` show about one group."""
    by_id = {s.id: s for s in project.samples}
    members = [by_id[i] for i in group.members if i in by_id]
    videos = Counter(s.source for s in members)
    return {
        "id": group.id, "state": group.state, "name": group.name,
        "size": len(members), "tightness": group.tightness,
        "videos": len(videos),
        # Most of a group from one video, in a project with several, is the
        # warning sign that the group is a scene rather than a thing.
        "one_video_share": (round(videos.most_common(1)[0][1] / len(members), 3)
                            if members else 0.0),
        "pending": sum(1 for s in members if s.verdict == PENDING),
        "shown": len(group.shown), "wrong": len(group.wrong),
        "parent": group.parent,
    }


def tiles(project: Project, group_id: int, shown: int = SHOWN) -> list:
    """The samples shown for a group, in tile order (tile n = index n-1).

    Members still undecided or decided by this group, most typical first:
    a member decided elsewhere (a review, an example) is not this group's to
    say anything about.
    """
    group = project.get_group(group_id)
    if group is None:
        raise KeyError(f"no group {group_id}")
    by_id = {s.id: s for s in project.samples}
    out = []
    for sid in group.members:
        sample = by_id.get(sid)
        if sample is None or sample.unreadable:
            continue
        if sample.verdict == PENDING or _decided_by_group(sample, group.id):
            out.append(sample)
        if len(out) >= shown:
            break
    return out


def _tile_numbers(text, count: int) -> list:
    from modules.teach.review import parse_numbers
    numbers = parse_numbers(text) if isinstance(text, str) else [int(n) for n in text or ()]
    bad = [n for n in numbers if not 1 <= n <= count]
    if bad:
        raise ValueError(f"no tile {bad[0]}: this group shows tiles 1-{count}")
    return numbers


def _undo(project: Project, group: GroupSpec) -> int:
    undone = 0
    for sample in project.samples:
        if _decided_by_group(sample, group.id):
            sample.verdict, sample.label, sample.decided_by = PENDING, "", ""
            sample.shown_guess, sample.guess_from = "", ""
            undone += 1
    return undone


def _drop_unused_class(project: Project, name: str) -> bool:
    """Remove a class nothing holds any more (no samples, examples or groups)."""
    spec = project.get_class(name)
    if spec is None or spec.examples or project.accepted(name):
        return False
    if any(g.state == NAMED and g.name == name for g in project.groups):
        return False
    project.classes = [c for c in project.classes if c.name != name]
    for sample in project.samples:
        sample.scores.pop(name, None)
        if sample.proposed == name:
            sample.proposed = ""
    return True


def name_group(project: Project, group_id: int, *, name: str = "",
               nothing: bool = False, mixed: bool = False, reopen: bool = False,
               wrong="", shown: int = SHOWN, description: str = "") -> dict:
    """Say what a group is. Exactly one of ``name``, ``nothing``, ``mixed``,
    ``reopen``. ``wrong``: tile numbers that do not belong (left undecided)."""
    chosen = [bool(name.strip()), nothing, mixed, reopen]
    if sum(chosen) != 1:
        raise ValueError("say one of: --name <class>, --nothing, --mixed, --reopen")
    group = project.get_group(group_id)
    if group is None:
        raise KeyError(f"no group {group_id}")
    if group.state == SPLIT:
        raise ValueError(f"group {group_id} was split; name the groups it was split into")

    shown_samples = tiles(project, group_id, shown)
    previous = group.name
    undone = _undo(project, group)
    group.state, group.name, group.shown, group.wrong = OPEN, "", [], []
    result = {"group": group_id, "undone": undone}

    if reopen or mixed:
        group.state = MIXED if mixed else OPEN
        if previous:
            result["class_removed"] = _drop_unused_class(project, previous)
        project.save()
        result["state"] = group.state
        return result

    wrong_n = set(_tile_numbers(wrong, len(shown_samples)))
    if len(wrong_n) == len(shown_samples):
        raise ValueError("every tile was marked wrong; call it mixed instead")
    if nothing:
        verdict, label, guess = NEGATIVE, "", NONE
        if project.task == ACTIONS and not project.settings.background_class:
            # "None of these" is only trained on through a background class;
            # saying a group is nothing means it should be.
            project.settings.background_class = BACKGROUND
            result["background_class"] = BACKGROUND
    else:
        from modules.teach.naming import normalize_name
        label = normalize_name(name)
        if project.get_class(label) is None:
            project.add_class(label, description)
            result["class_added"] = label
        elif description and not project.get_class(label).description:
            project.get_class(label).description = description.strip()
        verdict, guess = ACCEPTED, label

    accepted = 0
    for n, sample in enumerate(shown_samples, 1):
        group.shown.append(sample.id)
        if n in wrong_n:
            group.wrong.append(sample.id)
            continue
        project.decide(sample, verdict, label, by=_by(group.id), guess=guess,
                       guess_from=DECIDED_BY)
        accepted += 1
    group.state = NOTHING if nothing else NAMED
    group.name = label
    if previous and previous != label:
        result["class_removed"] = _drop_unused_class(project, previous)
    project.save()
    result.update({"state": group.state, "name": label, "decided": accepted,
                   "left_for_review": len(wrong_n)
                   + sum(1 for s in project.samples if s.group == group.id
                         and s.verdict == PENDING and s.id not in group.shown)})
    return result


def _guess_before(project: Project, sample) -> tuple:
    """``(guess, from)`` a sample showed before someone moved it: what
    auto-accept said, or the sort's proposal. Kept to measure the sorting."""
    if sample.decided_by == AUTO:
        return (sample.label if sample.verdict == ACCEPTED else NONE), "auto"
    if sample.proposed in project.class_names() or sample.proposed == NONE:
        return sample.proposed, sample.proposed_by or "sort"
    return UNSURE, "sort"


def move_samples(project: Project, sample_ids, *, to_class: str = "",
                 nothing: bool = False, reject: bool = False, undecided: bool = False,
                 to_group: Optional[int] = None, new_group: bool = False,
                 description: str = "") -> dict:
    """Put samples somewhere by hand: a class, "none of these", unusable, back
    to undecided, or another group. Exactly one destination.

    A class, nothing and unusable are a person's decision, like a verdict in
    review. A group only changes where the samples are shown. Moving a sample
    a group decided out of that group's class counts against the group.
    """
    chosen = [bool(to_class.strip()), nothing, reject, undecided, to_group is not None,
              new_group]
    if sum(chosen) != 1:
        raise ValueError("say one of: --to-class <class>, --nothing, --reject, "
                         "--undecided, --to-group N, --new-group")
    ids = list(dict.fromkeys(i.strip() for i in sample_ids if i and i.strip()))
    samples = [project.get_sample(i) for i in ids]
    missing = [i for i, s in zip(ids, samples) if s is None]
    if missing:
        raise KeyError(f"no sample {missing[0]}")
    if not samples:
        raise ValueError("no samples to move")
    result = {"moved": len(samples)}

    if to_group is not None or new_group:
        if new_group:
            target = GroupSpec(id=max((g.id for g in project.groups), default=0) + 1)
            project.groups.append(target)
            result["group_added"] = target.id
        else:
            target = project.get_group(to_group)
            if target is None:
                raise KeyError(f"no group {to_group}")
            if target.state == SPLIT:
                raise ValueError(f"group {to_group} was split; move to one of its parts")
        moving = {s.id for s in samples}
        for g in project.groups:
            if g is not target:
                g.members = [m for m in g.members if m not in moving]
        target.members += [s.id for s in samples if s.id not in target.members]
        for sample in samples:
            sample.group = target.id
        result["group"] = target.id
        project.save()
        return result

    label = ""
    if undecided:
        verdict = PENDING
    elif reject:
        verdict = REJECTED
    elif nothing:
        verdict = NEGATIVE
        if project.task == ACTIONS and not project.settings.background_class:
            project.settings.background_class = BACKGROUND
            result["background_class"] = BACKGROUND
    else:
        from modules.teach.naming import normalize_name
        label = normalize_name(to_class)
        if project.get_class(label) is None:
            project.add_class(label, description)
            result["class_added"] = label
        verdict = ACCEPTED

    for sample in samples:
        if sample.decided_by.startswith(DECIDED_BY + ":"):
            # A group said this; someone now says otherwise.
            g = project.get_group(int(sample.decided_by.split(":", 1)[1]))
            if (g is not None and sample.id in g.shown and sample.id not in g.wrong
                    and (verdict != sample.verdict or label != sample.label)):
                g.wrong.append(sample.id)
        if verdict == PENDING:
            sample.verdict, sample.label, sample.decided_by = PENDING, "", ""
            sample.shown_guess, sample.guess_from = "", ""
            continue
        guess, guess_from = _guess_before(project, sample)
        project.decide(sample, verdict, label, by=MOVED_BY, guess=guess,
                       guess_from=guess_from)
    project.save()
    result.update({"verdict": verdict, **({"label": label} if label else {})})
    return result


def summary(project: Project) -> dict:
    """Groups by state, and which still need someone to say what they are."""
    live = [g for g in project.groups if g.state != SPLIT]
    by_state = Counter(g.state for g in live)
    by_id = {s.id: s for s in project.samples}
    mixed_pending = sum(1 for g in live if g.state == MIXED
                        for sid in g.members
                        if sid in by_id and by_id[sid].verdict == PENDING)
    return {"total": len(live), "open": by_state.get(OPEN, 0),
            "named": by_state.get(NAMED, 0), "nothing": by_state.get(NOTHING, 0),
            "mixed": by_state.get(MIXED, 0), "mixed_pending": mixed_pending,
            "groups": [describe(project, g) for g in live]}


def render_group_sheet(project: Project, group_id: int, shown: int = SHOWN,
                       frame_reader: Optional[Callable] = None) -> dict:
    """A contact sheet of a group's tiles, for whoever names it from an image."""
    from modules.teach import review

    frame_reader = frame_reader or embed_mod.read_frames
    samples = tiles(project, group_id, shown)
    if not samples:
        return {"group": group_id, "image": "", "tiles": []}
    frames_per_tile = 1 if project.task == OBJECTS else 3
    pictures = [review._tile(frame_reader(s.path, frames_per_tile), review.TILE_HEIGHT)
                for s in samples]
    captions = [os.path.basename(s.path) for s in samples]
    path = project.path(review.REVIEW_DIR, f"group-{group_id:03d}.jpg")
    review.render_sheet(pictures, captions, 2 if frames_per_tile > 1 else 4, path,
                        header=f"group {group_id}: say what it is")
    return {"group": group_id, "image": path,
            "tiles": [{"n": n, "sample": s.id} for n, s in enumerate(samples, 1)]}
