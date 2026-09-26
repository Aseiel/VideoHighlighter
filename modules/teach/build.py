"""Accepted samples -> the dataset layout training already reads.

* ``actions``: ``dataset/train/<class>/*.mp4`` and ``dataset/val/<class>/*.mp4``,
  what ``model_training.r3d`` and ``model_training.intel`` load.
* ``objects``: the COCO layout ``training.train_yolox_run`` loads, built by
  ``modules.vision.label_store.build_dataset`` from ``labels.json``.

**The held-out set is frozen.** A sample's split is decided once, the first
time it is built into a dataset, and never changes. Every round is scored on
the same held-out samples, so "round 3 beats round 2" compares like with like;
folding reviewed samples into training round after round would feel productive
and destroy the only measurement there is.

**Neighbours travel together.** Samples next to each other in one source are
near-identical, so they go to the same side (``label_store.segments``):
otherwise validation scores the model on pictures it trained on.

**Enough validation per class.** The action trainers re-split a class with
fewer than two validation clips or under 20% of its clips held out — which
would un-freeze the held-out set behind our back — so each class is given at
least that much.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import zlib
from typing import Optional

from modules.teach.project import ACCEPTED, ACTIONS, NEGATIVE, TRAIN, VAL, Project

DATASET_DIR = "dataset"
# Neither train nor validation: see assign_splits.
SKIP = "skip"
BUILT_FILE = "built.json"
MIN_VAL_PER_CLASS = 2


def _stable(key: str) -> int:
    return zlib.crc32(key.encode("utf-8"))


def assign_splits(project: Project, val_fraction: Optional[float] = None) -> dict:
    """Give every accepted (and negative) sample without a split one, for good.

    Only samples a person decided are ever held out: the held-out set is what
    says whether a model is any good, and a label nobody checked cannot say
    that. Auto-accepted samples train. One that sits in the same stretch of
    footage as a held-out sample is left out of both (``SKIP``): in training it
    would be a near-copy of a question the model is later scored on. If a person
    checks it later, it joins its stretch in validation.
    """
    from modules.vision.label_store import segments

    fraction = project.settings.val_fraction if val_fraction is None else val_fraction
    groups_of = {}
    for name in project.class_names() + [NEGATIVE]:
        members = ([s for s in project.samples if s.verdict == NEGATIVE]
                   if name == NEGATIVE else project.accepted(name))
        if not members:
            continue
        for sample in members:
            if sample.split == SKIP and sample.is_human:
                sample.split = VAL
        human = [s for s in members if s.is_human]
        need = max(MIN_VAL_PER_CLASS, math.ceil(fraction * len(human)))
        have = sum(1 for s in members if s.split == VAL)
        fresh = [s for s in members if not s.split]
        # Whole segments, in an order fixed by their content rather than by
        # when they were added, so a rebuild makes the same choice.
        groups = sorted(segments(fresh), key=lambda g: _stable(g[0].id))
        for group in groups:
            checked = [s for s in group if s.is_human]
            side = VAL if (checked and have < need) else TRAIN
            for sample in group:
                sample.split = side if (side == TRAIN or sample.is_human) else SKIP
            if side == VAL:
                have += len(checked)
        groups_of[name] = {k: sum(1 for s in members if s.split == v)
                           for k, v in (("train", TRAIN), ("val", VAL), ("skipped", SKIP))}
    project.save()
    return groups_of


def _place(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build_actions(project: Project) -> dict:
    splits = assign_splits(project)
    root = project.path(DATASET_DIR)
    if os.path.isdir(root):
        shutil.rmtree(root)
    os.makedirs(root)
    written = 0
    for sample in project.accepted():
        if sample.split not in (TRAIN, VAL):
            continue
        # The person-focused crops when the cropper made them: those are what
        # the model will be shown, and each is its own training clip.
        files = sample.focus_paths or [sample.path]
        for i, src in enumerate(files):
            if not os.path.exists(src):
                continue
            suffix = f"_{i}" if len(files) > 1 else ""
            dst = os.path.join(root, sample.split, sample.label,
                               f"{sample.id}{suffix}{os.path.splitext(src)[1]}")
            _place(src, dst)
            written += 1
    return {"dataset": root, "clips": written, "splits": splits}


def build_objects(project: Project, progress=None) -> dict:
    from modules.teach.boxes import store
    from modules.vision.label_store import build_dataset

    root = project.path(DATASET_DIR)
    if os.path.isdir(root):
        shutil.rmtree(root)
    summary = build_dataset(store(project), root, val_fraction=project.settings.val_fraction,
                            progress=progress)
    summary["dataset"] = root
    return summary


def dataset_signature(project: Project) -> str:
    """What a dataset built now would contain, as a short hash.

    ``build`` stores it and each training round records it, so "has this been
    built / trained on?" is a comparison rather than a guess from timestamps.
    """
    parts = sorted(f"{s.id}:{s.label}:{s.split}:{s.verdict}" for s in project.samples
                   if s.verdict in (ACCEPTED, NEGATIVE))
    if project.task != ACTIONS:
        from modules.teach.boxes import store
        parts += sorted(f"{b.video}@{b.time}:{b.class_name}:{b.verdict}:{b.box}"
                        for b in store(project).boxes if b.verdict in (ACCEPTED, NEGATIVE))
    return f"{zlib.crc32(chr(10).join(parts).encode('utf-8')):08x}-{len(parts)}"


def built_signature(project: Project) -> str:
    try:
        with open(project.path(DATASET_DIR, BUILT_FILE), "r", encoding="utf-8") as handle:
            return json.load(handle).get("signature", "")
    except (OSError, ValueError):
        return ""


def build(project: Project) -> dict:
    result = build_actions(project) if project.task == ACTIONS else build_objects(project)
    # After assign_splits, which is part of what the signature covers.
    result["signature"] = dataset_signature(project)
    with open(project.path(DATASET_DIR, BUILT_FILE), "w", encoding="utf-8") as handle:
        json.dump({"signature": result["signature"]}, handle)
    return result
