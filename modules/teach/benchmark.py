"""Measure the teaching loop against a dataset someone already sorted by hand.

A hand-sorted dataset is the one thing that can say how well the automatic
parts work, because its answers are already known. Two questions, one each:

* **How much would a person still have to look at?** ``simulate`` replays the
  loop the app runs — a few examples per class, then ``sort``, auto-accept and
  review sheets, re-sorting after each — with the folders answering every
  sheet in place of the person. Every tile it shows is one a person would
  have had to look at; every sample auto-accepted and never shown went into
  the dataset unchecked, and the folders say whether it was right.
* **How good is a trained model?** ``model_test`` runs one on the ``val`` and
  ``test`` clips and scores it the way the folders define: one class per clip,
  or several (a folder named ``a_b`` shows *a* and *b* at once).

The layout read is the one the trainers read, one folder per class inside
each split::

    <dataset>/train/<class>/*.mp4
    <dataset>/val/<class>/*.mp4
    <dataset>/test/<class>_<class>/*.mp4

Folders starting with ``_`` are left out, and so are folders with no video
directly inside (annotations kept next to the clips, say). An aliases file,
``{"folder or class name": "class name" | ""}``, fixes names without touching
the dataset: a misspelled folder, a finer class a model does not have mapped
to the class it does, or ``""`` to leave a class out.

Nothing is copied or moved, and nothing here knows what any class is: names
come from the folders and stay in the output.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from modules.teach.cut import VIDEO_EXTENSIONS
from modules.teach.project import (
    ACCEPTED, ACTIONS, MIN_TO_TRAIN, PENDING, ClassSpec, Project, Sample,
)

SPLITS = ("train", "val", "test")
# Where a pool of clips to sort is drawn from; test stays unseen.
POOL_SPLITS = ("train", "val")
PAIR_SEPARATOR = "_"
# The app names cut clips "<video>_..._clip_<n>", so a leading number is the
# video a clip came from. Clips of one video share a scene, a cast and a
# camera, and are much easier to sort together than apart.
DEFAULT_GROUP = r"^(\d+)_"
# Bytes hashed to tell identical files apart; only files of equal size are read.
FINGERPRINT_BYTES = 256 * 1024
SIMULATION_DIR = "simulation"
MARKER = ".evaluate"
SHOWN_EXAMPLES = 20


@dataclass
class Clip:
    path: str
    split: str
    folder: str
    labels: tuple
    group: str

    @property
    def key(self) -> str:
        return " + ".join(self.labels)


# ---------------------------------------------------------------------------
# Reading the dataset
# ---------------------------------------------------------------------------

def load_aliases(path: Optional[str]) -> dict:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not all(isinstance(v, str) for v in data.values()):
        raise ValueError(f"{path}: aliases are one JSON object of name -> name (or \"\")")
    return {str(k).strip(): v.strip() for k, v in data.items()}


def labels_of(folder: str, aliases: Optional[dict] = None) -> tuple:
    """The classes a folder shows: ``a_b`` is both, after aliases.

    The whole folder name is looked up first, so a folder whose name does not
    split cleanly can be given its classes outright (``"a b c": "a b_c"``).
    """
    aliases = aliases or {}
    name = aliases.get(folder.strip(), folder)
    out = []
    for part in name.split(PAIR_SEPARATOR):
        part = part.strip()
        if not part:
            continue
        part = aliases.get(part, part)
        if part and part not in out:
            out.append(part)
    return tuple(out)


def group_of(filename: str, pattern: str = DEFAULT_GROUP) -> str:
    """Which video a clip came from, from its file name."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    match = re.search(pattern, stem) if pattern else None
    if match:
        return match.group(1) if match.groups() else match.group(0)
    return stem


def _videos(folder: str) -> list:
    return sorted(os.path.join(folder, n) for n in os.listdir(folder)
                  if n.lower().endswith(VIDEO_EXTENSIONS)
                  and os.path.isfile(os.path.join(folder, n)))


def _misnamed(folder: str) -> list:
    """Files named like a video with something after the extension
    (``a.mp4 (1)``): no trainer reads them, and nothing says so."""
    out = []
    for name in os.listdir(folder):
        lower = name.lower()
        if lower.endswith(VIDEO_EXTENSIONS):
            continue
        if any(ext + " " in lower or ext + "(" in lower or ext + "." in lower
               for ext in VIDEO_EXTENSIONS):
            out.append(name)
    return sorted(out)


def read_dataset(root: str, aliases: Optional[dict] = None,
                 group_pattern: str = DEFAULT_GROUP) -> dict:
    """``{"clips": [Clip], "skipped": [...], "splits": [...], "misnamed": [...]}``."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"no dataset folder {root}")
    clips, skipped, splits, misnamed = [], [], [], []
    for split in SPLITS:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        splits.append(split)
        for folder in sorted(os.listdir(split_dir)):
            path = os.path.join(split_dir, folder)
            if not os.path.isdir(path):
                continue
            where = f"{split}/{folder}"
            misnamed.extend(f"{where}/{n}" for n in _misnamed(path))
            if folder.startswith(("_", ".")):
                skipped.append({"folder": where, "why": "starts with _"})
                continue
            videos = _videos(path)
            if not videos:
                skipped.append({"folder": where, "why": "no video directly inside"})
                continue
            labels = labels_of(folder, aliases)
            if not labels:
                skipped.append({"folder": where, "why": "left out by aliases",
                                "clips": len(videos)})
                continue
            for video in videos:
                clips.append(Clip(os.path.abspath(video), split, folder, labels,
                                  group_of(video, group_pattern)))
    if not splits:
        raise FileNotFoundError(f"{root} has none of {', '.join(SPLITS)}")
    return {"clips": clips, "skipped": skipped, "splits": splits, "misnamed": misnamed}


def duplicates(clips: Sequence[Clip]) -> list:
    """Groups of byte-identical files (by size, then a hash of their start)."""
    by_size = defaultdict(list)
    for clip in clips:
        try:
            by_size[os.path.getsize(clip.path)].append(clip)
        except OSError:
            continue
    out = []
    for same_size in by_size.values():
        if len(same_size) < 2:
            continue
        by_hash = defaultdict(list)
        for clip in same_size:
            try:
                with open(clip.path, "rb") as handle:
                    by_hash[hashlib.sha1(handle.read(FINGERPRINT_BYTES)).hexdigest()].append(clip)
            except OSError:
                continue
        out.extend(group for group in by_hash.values() if len(group) > 1)
    return out


def report(data: dict, root: str = "", min_train: int = MIN_TO_TRAIN) -> dict:
    """What the dataset holds, and what would skew a measurement of it."""
    clips = data["clips"]
    counts = {split: dict(sorted(Counter(c.key for c in clips if c.split == split).items()))
              for split in data["splits"]}
    train_single = Counter(c.labels[0] for c in clips
                           if c.split == "train" and len(c.labels) == 1)
    known = set(train_single)

    not_in_train = {}
    for split in data["splits"]:
        if split == "train":
            continue
        missing = sorted({label for c in clips if c.split == split
                          for label in c.labels if label not in known})
        if missing:
            not_in_train[split] = missing

    split_sets = defaultdict(set)
    for clip in clips:
        split_sets[clip.group].add(clip.split)
    shared = Counter(" & ".join(sorted(s)) for s in split_sets.values() if len(s) > 1)

    same = duplicates(clips)

    def rel(path):
        return os.path.relpath(path, root) if root else path

    return {
        "clips": len(clips),
        "per_split": {split: sum(v.values()) for split, v in counts.items()},
        "classes": len(known),
        "counts": counts,
        "below_minimum": {k: v for k, v in sorted(train_single.items())
                          if v < min_train},
        "minimum": min_train,
        "not_in_train": not_in_train,
        "missing_from_val": sorted(known - {c.labels[0] for c in clips
                                            if c.split == "val" and len(c.labels) == 1})
        if "val" in data["splits"] else [],
        "multi_class_clips": {split: sum(1 for c in clips if c.split == split
                                         and len(c.labels) > 1)
                              for split in data["splits"]},
        "videos": len(split_sets),
        "videos_in_several_splits": dict(shared),
        "duplicates": {
            "groups": len(same),
            "extra_copies": sum(len(g) - 1 for g in same),
            "across_splits": sum(1 for g in same if len({c.split for c in g}) > 1),
            "across_classes": sum(1 for g in same if len({c.key for c in g}) > 1),
            "examples": [[rel(c.path) for c in g] for g in same[:SHOWN_EXAMPLES]],
        },
        "skipped": data["skipped"],
        "misnamed": {"files": len(data.get("misnamed", [])),
                     "examples": data.get("misnamed", [])[:SHOWN_EXAMPLES]},
    }


# ---------------------------------------------------------------------------
# The loop, answered by the folders
# ---------------------------------------------------------------------------

def sample_id(path: str) -> str:
    """Stable across runs, so the CLIP vectors cached by one run serve the next."""
    return "c" + hashlib.sha1(os.path.normcase(os.path.abspath(path))
                              .encode("utf-8")).hexdigest()[:16]


def _fresh_project(work_dir: str) -> Project:
    """A scratch project in ``work_dir/simulation``; its vector cache survives."""
    root = os.path.join(work_dir, SIMULATION_DIR)
    if os.path.isdir(root) and os.listdir(root) and not os.path.exists(
            os.path.join(root, MARKER)):
        raise FileExistsError(f"{root} exists and was not made by evaluate; "
                              "choose another --project")
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, MARKER), "w", encoding="utf-8") as handle:
        handle.write("scratch project of `evaluate`: safe to delete\n")
    for name in os.listdir(root):
        if name not in (MARKER, "cache"):
            path = os.path.join(root, name)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
    project = Project(root)
    project.task = ACTIONS
    return project


def _ready(project: Project) -> bool:
    """Nothing left undecided, and every auto-accepting class spot-checked enough."""
    from modules.teach import autolabel

    if any(s.verdict == PENDING and s.scores for s in project.samples):
        return False
    return all(autolabel.audits_needed(project, n) == 0 for n in project.class_names())


def simulate(clips: Sequence[Clip], embedder, work_dir: str, *, seeds: int = 5,
             sheet_size: int = 24, max_sheets: Optional[int] = None, rng_seed: int = 0,
             settings: Optional[dict] = None,
             frame_reader: Optional[Callable] = None,
             progress: Optional[Callable] = None) -> dict:
    """Replay the teaching loop over ``train`` + ``val``, the folders answering.

    Each class starts with ``seeds`` examples, as if someone had picked them.
    Then, until nothing is left to decide: sort (which auto-accepts), draw a
    sheet, answer every tile from the folders — confirming or overturning
    spot checks the way a person would — and sort again. ``settings``
    overrides project settings, to compare them on the same dataset.
    """
    from modules.teach import autolabel, review
    from modules.teach.sort import sort_project

    say = progress or (lambda message: None)
    pool = [c for c in clips if c.split in POOL_SPLITS and len(c.labels) == 1]
    if not pool:
        raise ValueError("no single-class clips in train or val to sort")
    project = _fresh_project(work_dir)
    for key, value in (settings or {}).items():
        if not hasattr(project.settings, key):
            raise KeyError(f"no setting {key!r}")
        setattr(project.settings, key, value)
    names = sorted({c.labels[0] for c in pool})
    project.classes = [ClassSpec(name=n) for n in names]
    truth, group = {}, {}
    for clip in pool:
        sid = sample_id(clip.path)
        if sid in truth:
            continue
        truth[sid], group[sid] = clip.labels[0], clip.group
        project.samples.append(Sample(id=sid, source=clip.group, path=clip.path,
                                      start=0.0, duration=0.0))

    rng = random.Random(rng_seed)
    seeded_groups = set()
    by_class = defaultdict(list)
    for sample in project.samples:
        by_class[truth[sample.id]].append(sample)
    for name in names:
        # By file name, not id: the ids hash the full path, and the same
        # dataset should get the same examples wherever it is kept.
        members = sorted(by_class[name], key=lambda s: (os.path.basename(s.path), s.path))
        rng.shuffle(members)
        for sample in members[:seeds]:
            project.decide(sample, ACCEPTED, name, by="example")
            project.get_class(name).examples.append(sample.id)
            seeded_groups.add(sample.source)
    project.save()

    def embedded(i, n):
        if i == n or i % 200 == 0:
            say(f"evaluate: CLIP vectors {i}/{n}")

    say(f"evaluate: {len(project.samples)} clips, {len(names)} classes; sorting")
    sort_project(project, embedder, frame_reader=frame_reader, progress=embedded)
    first = _first_sort(project, truth)

    shown = audits = overturned = 0
    shown_by_class = Counter()
    curve = []
    sheet = 0
    while not _ready(project):
        if max_sheets is not None and sheet >= max_sheets:
            break
        batch = review.pick_batch(project, size=sheet_size, seed=sheet)
        if not batch:
            break
        sheet += 1
        for sample in batch:
            answer = truth[sample.id]
            shown += 1
            shown_by_class[answer] += 1
            if sample.is_auto:
                audits += 1
                overturned += sample.label != answer
            project.decide(sample, ACCEPTED, answer, by=f"sheet:{sheet}")
        autolabel.apply(project)
        sort_project(project, embedder, frame_reader=frame_reader)
        auto = [s for s in project.samples if s.is_auto]
        curve.append({
            "sheet": sheet, "looked_at": shown + sum(len(c.examples) for c in project.classes),
            "auto": len(auto), "auto_wrong": sum(1 for s in auto if s.label != truth[s.id]),
            "pending": sum(1 for s in project.samples if s.verdict == PENDING),
        })
        if sheet % 10 == 0:
            say(f"evaluate: sheet {sheet}: {curve[-1]}")

    return _summary(project, truth, group, seeded_groups, first, curve,
                    shown, shown_by_class, audits, overturned, sheet, seeds)


def _first_sort(project: Project, truth: dict) -> dict:
    """How right the very first sort is, before anyone reviews anything."""
    names = set(project.class_names())
    pending = [s for s in project.samples if s.verdict == PENDING or s.is_auto]
    proposed = [s for s in pending if s.proposed in names]
    right = sum(1 for s in proposed if s.proposed == truth[s.id])
    return {"clips": len(pending), "proposed": len(proposed), "proposed_right": right,
            "unsure": len(pending) - len(proposed),
            "auto_accepted": sum(1 for s in pending if s.is_auto),
            "auto_right": sum(1 for s in pending if s.is_auto and s.label == truth[s.id])}


def _summary(project, truth, group, seeded_groups, first, curve, shown,
             shown_by_class, audits, overturned, sheets, seeds) -> dict:
    from modules.teach import autolabel

    total = len(project.samples)
    examples = sum(len(c.examples) for c in project.classes)
    auto = [s for s in project.samples if s.is_auto]
    wrong = [s for s in auto if s.label != truth[s.id]]
    pending = [s for s in project.samples if s.verdict == PENDING]

    def split_by_video(samples):
        seen = [s for s in samples if group[s.id] in seeded_groups]
        return {"from_example_videos": len(seen),
                "from_other_videos": len(samples) - len(seen)}

    per_class = {}
    for spec in project.classes:
        mine = [s for s in project.samples if truth[s.id] == spec.name]
        auto_as = [s for s in auto if s.label == spec.name]
        state = autolabel.state(project, spec.name)
        per_class[spec.name] = {
            "clips": len(mine),
            "examples": len(spec.examples),
            "looked_at": shown_by_class.get(spec.name, 0),
            "auto_accepted": len(auto_as),
            "auto_wrong": sum(1 for s in auto_as if truth[s.id] != spec.name),
            "left_undecided": sum(1 for s in pending if truth[s.id] == spec.name),
            "auto_accept": "on" if state["on"] else state["why"],
        }
    confusions = Counter(f"{truth[s.id]} -> {s.label}" for s in wrong)
    looked_at = examples + shown
    return {
        "clips": total,
        "classes": len(project.classes),
        "settings": {k: getattr(project.settings, k) for k in (
            "gate", "margin", "floor", "auto_gate", "auto_margin", "auto_min_checked",
            "auto_max_error", "prototypes_per_class", "frames_per_sample")},
        "examples_per_class": seeds,
        "sheets": sheets,
        "looked_at": looked_at,
        "looked_at_share": round(looked_at / total, 3) if total else 0.0,
        "auto_accepted_unchecked": len(auto),
        "auto_wrong": len(wrong),
        "auto_error_rate": round(len(wrong) / len(auto), 3) if auto else 0.0,
        "auto_wrong_by_video": split_by_video(wrong),
        "auto_accepted_by_video": split_by_video(auto),
        "left_undecided": len(pending),
        "spot_checks": {"shown": audits, "overturned": overturned},
        "first_sort": first,
        "per_class": per_class,
        "auto_confusions": dict(confusions.most_common(SHOWN_EXAMPLES)),
        "curve": curve,
    }


# ---------------------------------------------------------------------------
# New footage, sorted by everything else
# ---------------------------------------------------------------------------

def sort_test(clips: Sequence[Clip], embedder, work_dir: str, *,
              holdout: float = 0.2, max_examples: int = 200, rng_seed: int = 0,
              settings: Optional[dict] = None, frame_reader: Optional[Callable] = None,
              progress: Optional[Callable] = None) -> dict:
    """``from-dataset`` on footage it has not seen, scored by the folders.

    A share of the videos (by ``Clip.group``) is held out whole; the rest of
    train and val are the examples, as ``from-dataset`` makes them; the
    held-out clips are sorted once, with auto-accept, as a new video is. So
    the numbers say what sorting a new video with this dataset gets right,
    decides alone, and leaves to a person, before any review.
    """
    from modules.teach.dataset_sort import add_dataset
    from modules.teach.sort import sort_project

    say = progress or (lambda message: None)
    pool = [c for c in clips if c.split in POOL_SPLITS and len(c.labels) == 1]
    groups = sorted({c.group for c in pool})
    if len(groups) < 2:
        raise ValueError("clips come from one video: nothing to hold out "
                         "(is --group finding the video in the file names?)")
    rng = random.Random(rng_seed)
    held = set(rng.sample(groups, max(1, round(len(groups) * holdout))))
    examples = [c for c in pool if c.group not in held]
    unseen = [c for c in pool if c.group in held]

    project = _fresh_project(work_dir)
    for key, value in (settings or {}).items():
        if not hasattr(project.settings, key):
            raise KeyError(f"no setting {key!r}")
        setattr(project.settings, key, value)
    add_dataset(project, examples, max_examples=max_examples, rng_seed=rng_seed)
    truth = {}
    for clip in unseen:
        sid = sample_id(clip.path)
        if sid in truth or project.get_sample(sid) is not None:
            continue
        truth[sid] = clip.labels[0]
        project.samples.append(Sample(id=sid, source=clip.group, path=clip.path,
                                      start=0.0, duration=0.0))
    project.save()

    def embedded(i, n):
        if i == n or i % 200 == 0:
            say(f"evaluate: CLIP vectors {i}/{n}")

    say(f"evaluate: sorting {len(truth)} clips of {len(held)} held-out videos "
        f"by {len(examples)} clips of the rest")
    sort_project(project, embedder, frame_reader=frame_reader, progress=embedded)

    names = set(project.class_names())
    tested = [s for s in project.samples if s.id in truth]
    per_class = defaultdict(lambda: {"clips": 0, "best_right": 0, "proposed": 0,
                                     "proposed_right": 0, "auto": 0, "auto_wrong": 0})
    confusions = Counter()
    for sample in tested:
        want = truth[sample.id]
        row = per_class[want]
        row["clips"] += 1
        best = max(sample.scores, key=sample.scores.get) if sample.scores else ""
        if best == want:
            row["best_right"] += 1
        elif best:
            confusions[f"{want} -> {best}"] += 1
        guess = sample.label if sample.is_auto else sample.proposed
        if guess in names:
            row["proposed"] += 1
            row["proposed_right"] += guess == want
        if sample.is_auto:
            row["auto"] += 1
            row["auto_wrong"] += sample.label != want

    def total(key):
        return sum(r[key] for r in per_class.values())

    n = len(tested)
    return {
        "held_out_videos": len(held), "clips": n,
        "examples": sum(len(c.examples) for c in project.classes),
        "best_guess_right": round(total("best_right") / n, 3) if n else None,
        "proposed": total("proposed"), "proposed_right": total("proposed_right"),
        "left_to_check": n - total("auto"),
        "auto_accepted": total("auto"), "auto_wrong": total("auto_wrong"),
        "classes_without_examples": sorted({truth[s.id] for s in tested} - names),
        "per_class": {k: dict(v) for k, v in sorted(per_class.items())},
        "confusions": dict(confusions.most_common(SHOWN_EXAMPLES)),
    }


# ---------------------------------------------------------------------------
# A trained model on val and test
# ---------------------------------------------------------------------------

def model_test(clips: Sequence[Clip], scorer: Callable,
               progress: Optional[Callable] = None) -> dict:
    """Score ``scorer(path) -> {label: probability}`` on val and test clips.

    One-class clips: is the top guess right. Several-class clips: are all of
    them the top guesses (``all_on_top``: top 2 for a pair), is the top guess
    one of them (``top_is_one``), and are all of them in the top 3.
    """
    say = progress or (lambda message: None)
    single = defaultdict(lambda: {"clips": 0, "right": 0})
    multi = defaultdict(lambda: {"clips": 0, "all_on_top": 0, "top_is_one": 0,
                                 "all_in_top3": 0})
    confusions = Counter()
    unreadable = 0
    unknown = Counter()
    known = set()
    targets = [c for c in clips if c.split in ("val", "test")]
    for i, clip in enumerate(targets, 1):
        probs = scorer(clip.path)
        if i % 100 == 0 or i == len(targets):
            say(f"evaluate: model on {i}/{len(targets)} clips")
        if not probs:
            unreadable += 1
            continue
        known.update(probs)
        missing = [label for label in clip.labels if label not in probs]
        if missing:
            for label in missing:
                unknown[label] += 1
            continue
        ranked = sorted(probs, key=probs.get, reverse=True)
        if len(clip.labels) == 1:
            row = single[(clip.split, clip.labels[0])]
            row["clips"] += 1
            if ranked[0] == clip.labels[0]:
                row["right"] += 1
            else:
                confusions[f"{clip.labels[0]} -> {ranked[0]}"] += 1
        else:
            want = set(clip.labels)
            row = multi[clip.key]
            row["clips"] += 1
            row["all_on_top"] += set(ranked[:len(want)]) == want
            row["top_is_one"] += ranked[0] in want
            row["all_in_top3"] += want <= set(ranked[:3])

    def per_split(split):
        rows = {label: dict(v) for (s, label), v in sorted(single.items()) if s == split}
        n = sum(r["clips"] for r in rows.values())
        right = sum(r["right"] for r in rows.values())
        recalls = [r["right"] / r["clips"] for r in rows.values() if r["clips"]]
        return {"clips": n, "accuracy": round(right / n, 3) if n else None,
                "balanced_accuracy": round(sum(recalls) / len(recalls), 3)
                if recalls else None, "per_class": rows}

    pairs = {k: dict(v) for k, v in sorted(multi.items())}
    n_multi = sum(r["clips"] for r in pairs.values())

    def share(key):
        return round(sum(r[key] for r in pairs.values()) / n_multi, 3) if n_multi else None

    return {
        "val": per_split("val"),
        "test_single": per_split("test"),
        "test_multi": {"clips": n_multi, "all_on_top": share("all_on_top"),
                       "top_is_one": share("top_is_one"),
                       "all_in_top3": share("all_in_top3"), "per_combination": pairs},
        "confusions": dict(confusions.most_common(SHOWN_EXAMPLES)),
        "unreadable": unreadable,
        "classes_the_model_lacks": dict(unknown),
        "model_classes": len(known),
    }
