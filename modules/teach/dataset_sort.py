"""Sort new footage by a dataset someone already sorted by hand.

A hand-sorted dataset (``<dataset>/train/<class>/*.mp4``, and ``val/``) is a
project's worth of examples before the project exists. ``add_dataset`` makes
it exactly that: every folder a class, named as the folder is — the name a
model trained on the same dataset reports, so the two can be compared — and
its clips accepted examples, referenced where they are, never copied. New
footage is then added, cut, and sorted against them the usual way, and
auto-accept, review and training all work as for any project.

``lay_out`` shows the result per video: its samples as hard links in
``by-class/<video>/<class>/``, named by where they start in the video (those
no class clearly won in ``_unsure/<best guess>/``), and a ``timeline.csv`` of
every sample's guess. The folders are a view, rebuilt each
time; the verdicts in ``samples.json`` are the record, as everywhere else.
"""
from __future__ import annotations

import csv
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from typing import Optional, Sequence

from modules.teach import benchmark
from modules.teach.project import (
    ACCEPTED, NEGATIVE, NONE, PENDING, REJECTED, UNSURE, ClassSpec, Project, Sample,
)

LAYOUT_DIR = "by-class"
TIMELINE = "timeline.csv"
DATASET_SOURCE = "dataset"
# A class prototype is the mean of at most this many examples
# (``scoring.MAX_EXAMPLES``); embedding more costs time and changes nothing.
DEFAULT_MAX_EXAMPLES = 200


def add_dataset(project: Project, clips: Sequence[benchmark.Clip], *,
                max_examples: int = DEFAULT_MAX_EXAMPLES, rng_seed: int = 0) -> dict:
    """Classes and examples from a dataset's single-class train and val clips.

    Safe to repeat: classes and examples already there are kept, and a class
    gets new examples only up to ``max_examples``.
    """
    pool = defaultdict(list)
    for clip in clips:
        if clip.split in benchmark.POOL_SPLITS and len(clip.labels) == 1:
            pool[clip.labels[0]].append(clip)
    if not pool:
        raise ValueError("no single-class clips in the dataset's train or val")
    known = {s.id for s in project.samples}
    rng = random.Random(rng_seed)
    added = {}
    for name in sorted(pool):
        spec = project.get_class(name)
        if spec is None:
            # The folder's own name, unchecked: it is what a model trained on
            # this dataset calls the class, and renaming it here would make
            # the two disagree about every sample.
            spec = ClassSpec(name=name)
            project.classes.append(spec)
        members = sorted(pool[name], key=lambda c: (os.path.basename(c.path), c.path))
        rng.shuffle(members)
        room = max(0, max_examples - len(spec.examples))
        count = 0
        for clip in members:
            if count >= room:
                break
            sid = benchmark.sample_id(clip.path)
            if sid in known:
                continue
            sample = Sample(id=sid, source=DATASET_SOURCE, path=clip.path,
                            start=0.0, duration=0.0)
            project.samples.append(sample)
            project.decide(sample, ACCEPTED, name, by="example")
            spec.examples.append(sid)
            known.add(sid)
            count += 1
        added[name] = count
    return {"classes": len(project.classes), "examples_added": added,
            "examples": sum(len(c.examples) for c in project.classes)}


def _stamp(seconds: float) -> str:
    whole = int(seconds)
    return f"{whole // 3600:02d}h{whole % 3600 // 60:02d}m{whole % 60:02d}s"


def _folder_name(text: str) -> str:
    return re.sub(r'[<>:"/\\|?*]+', "_", text).strip(" .") or "video"


def state_of(sample: Sample) -> tuple:
    """``(folder, how)``: where a sample belongs now, and who put it there."""
    if sample.verdict == ACCEPTED:
        return sample.label, "auto" if sample.is_auto else "checked"
    if sample.verdict == NEGATIVE:
        return NONE, "auto" if sample.is_auto else "checked"
    if sample.verdict == REJECTED:
        return "_reject", "checked"
    return sample.proposed or UNSURE, "guess"


def _link(src: str, dst: str) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def lay_out(project: Project, source_ids: Optional[Sequence[str]] = None) -> dict:
    """``by-class/<video>/<class>/`` and ``timeline.csv`` for each video."""
    wanted = set(source_ids) if source_ids else {s.id for s in project.sources}
    by_source = defaultdict(list)
    for sample in project.samples:
        if sample.source in wanted:
            by_source[sample.source].append(sample)
    out = {}
    for source in project.sources:
        if source.id not in wanted:
            continue
        stem = os.path.splitext(os.path.basename(source.path))[0]
        root = project.path(LAYOUT_DIR, _folder_name(f"{source.id} {stem}")[:80])
        if os.path.isdir(root):
            shutil.rmtree(root)
        os.makedirs(root)
        samples = sorted(by_source[source.id], key=lambda s: s.start)
        tally, how_tally = Counter(), Counter()
        rows = []
        for sample in samples:
            folder, how = state_of(sample)
            tally[folder] += 1
            how_tally[how] += 1
            ranked = sorted(sample.scores.items(), key=lambda kv: kv[1], reverse=True)
            best = ranked[0] if ranked else ("", 0.0)
            second = ranked[1] if len(ranked) > 1 else ("", 0.0)
            rows.append({
                "start": round(sample.start, 2), "end": round(sample.start + sample.duration, 2),
                "at": _stamp(sample.start), "class": folder, "how": how,
                "best": best[0], "score": round(best[1], 3),
                "second": second[0], "second_score": round(second[1], 3),
                "model": sample.model_proposed,
                "model_confidence": round(sample.model_confidence, 3),
                "sample": sample.id,
            })
            if os.path.exists(sample.path):
                target = os.path.join(root, _folder_name(folder))
                if folder == UNSURE and best[0]:
                    # Not proposed, but still best guessed as something: kept
                    # by that guess, so checking them is mostly confirming.
                    target = os.path.join(target, _folder_name(best[0]))
                os.makedirs(target, exist_ok=True)
                name = f"{_stamp(sample.start)}_{best[1]:.2f}_{how}.mp4"
                _link(sample.path, os.path.join(target, name))
        with open(os.path.join(root, TIMELINE), "w", newline="", encoding="utf-8") as handle:
            fields = list(rows[0]) if rows else ["start"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        out[source.id] = {
            "video": source.path, "folder": root, "samples": len(samples),
            "by_class": dict(tally.most_common()), "decided": dict(how_tally),
        }
    return out


def pending_of(project: Project, source_ids: Sequence[str]) -> int:
    wanted = set(source_ids)
    return sum(1 for s in project.samples if s.source in wanted and s.verdict == PENDING)
