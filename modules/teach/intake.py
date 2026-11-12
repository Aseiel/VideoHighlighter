"""Add a reviewed video to a hand-sorted dataset without letting it take over.

A video that was sorted and then checked by hand is the best new material a
dataset can get, and the easiest to add wrongly. All of it in, and one scene,
one cast and one camera become a large part of the training set: the model
gets better at that video and worse at every other
(``docs/plans/2026-10-04-adding-reviewed-footage.md``). The rule measured
there is what ``plan`` applies:

* **one source video**: every crop is named ``<video>_highlight_<crop>``, so
  held-out folds keep the whole video on one side;
* **only where the dataset is thin**: a folder joins when one of its classes
  has fewer than ``thin_under`` clips in the dataset (a class it has none of
  counts as thin);
* **capped**: at most ``cap`` crops per folder, spread evenly over the video.

``plan`` only reads and says what would happen, folder by folder, with the
reason for everything left out. ``apply`` links the chosen crops into
``<dataset>/train/`` (a hard link; a copy across drives) and records exactly
what it added in ``<dataset>/.intake/<video>.json``, so ``undo`` can take the
same files out again and nothing else. The review folder is never changed.

Whether the addition helped is a separate question with its own command
(``model_training.action_head.compare``): it is answered on the dataset's
other videos, not by looking at what went in.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from collections import Counter, defaultdict
from typing import Optional, Sequence

from modules.teach import benchmark
from modules.teach.cut import VIDEO_EXTENSIONS

MANIFEST_DIR = ".intake"
TRAIN = "train"
DEFAULT_THIN_UNDER = 200
DEFAULT_CAP = 50
MARK = "_highlight_"
# Where a crop starts in its video, as the sorter names it: "<video>__<ms>...".
_WHEN = re.compile(r"__(\d+)")

# One video as a share of the training clips, above which the plan says so.
# Measured: about 20% (capped) changed nothing on other videos, 52% (everything)
# made them worse; nothing in between was measured, so this warns early.
LARGE_SHARE = 0.25

ENOUGH = "the dataset has enough"
LEFT_OUT = "left out by aliases"
EMPTY = "no clips"
EXCLUDED = "left out on request"


def manifest_path(dataset: str, video: str) -> str:
    return os.path.join(dataset, MANIFEST_DIR, f"{video}.json")


def read_manifest(dataset: str, video: str) -> Optional[dict]:
    path = manifest_path(dataset, video)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _check_name(video: str) -> None:
    if not video or video != video.strip() or any(c in video for c in '\\/:*?"<>|'):
        raise ValueError(f"{video!r} is not a usable video name")
    if benchmark.group_of(f"{video}{MARK}x.mp4") != video:
        raise ValueError(f"{video!r} cannot name a source video: the part before "
                         f"'_temp' or '_highlight' is what a clip's video is read from")


def _review_folders(review: str) -> dict:
    """``{folder: [clip paths]}``: each top folder with every clip under it
    (a sub-folder a reviewer made inside a class still shows that class)."""
    if not os.path.isdir(review):
        raise FileNotFoundError(f"no review folder {review}")
    out = {}
    for folder in sorted(os.listdir(review)):
        top = os.path.join(review, folder)
        if not os.path.isdir(top) or folder.startswith(("_", ".")):
            continue
        clips = []
        for where, _dirs, files in os.walk(top):
            clips.extend(os.path.join(where, f) for f in files
                         if f.lower().endswith(VIDEO_EXTENSIONS))
        out[folder] = clips
    return out


def _in_order(paths: list) -> list:
    """Clips in the order they occur in the video (by the time in the name,
    then the name), so an even spread is a spread over the video."""
    def key(path):
        name = os.path.basename(path)
        match = _WHEN.search(name)
        return (int(match.group(1)) if match else 0, name)
    return sorted(paths, key=key)


def spread(paths: list, cap: Optional[int]) -> list:
    """At most ``cap`` of ``paths``, evenly across them."""
    if not cap or len(paths) <= cap:
        return list(paths)
    step = len(paths) / cap
    return [paths[int(i * step)] for i in range(cap)]


def plan(dataset: str, review: str, video: str, *,
         thin_under: Optional[int] = DEFAULT_THIN_UNDER, cap: Optional[int] = DEFAULT_CAP,
         aliases: Optional[dict] = None, exclude: Sequence[str] = ()) -> dict:
    """What adding ``review`` to ``dataset`` as the video ``video`` would do.

    Reads only. ``thin_under`` 0 or None takes every folder, ``cap`` 0 or None
    every crop: the unguarded merge, kept so its cost can be shown, and
    flagged in ``warnings``. ``exclude`` names review folders to leave out,
    for the ones a comparison showed to cost.
    """
    _check_name(video)
    data = benchmark.read_dataset(dataset, aliases)
    before = read_manifest(dataset, video)
    mine = {os.path.normcase(os.path.join(os.path.abspath(dataset), rel))
            for rel in (before or {}).get("added", [])}
    theirs = [c for c in data["clips"]
              if c.group == video and os.path.normcase(c.path) not in mine]
    if theirs:
        raise ValueError(f"the dataset already has {len(theirs)} clips of a video named "
                         f"{video!r}; give the reviewed video another name")
    own = [c for c in data["clips"] if os.path.normcase(c.path) not in mine]
    clips_of, videos_of = Counter(), defaultdict(set)
    for clip in own:
        for label in clip.labels:
            clips_of[label] += 1
            videos_of[label].add(clip.group)
    train_before = sum(1 for c in own if c.split == TRAIN)

    excluded = set(exclude)
    rows, total, new_classes = [], 0, set()
    for folder, found in _review_folders(review).items():
        labels = benchmark.labels_of(folder, aliases)
        row = {"folder": folder, "classes": list(labels), "found": len(found),
               "take": 0, "why": "", "paths": [],
               "dataset_clips": {lb: clips_of[lb] for lb in labels},
               "dataset_videos": {lb: len(videos_of[lb]) for lb in labels}}
        rows.append(row)
        if folder in excluded:
            row["why"] = EXCLUDED
        elif not labels:
            row["why"] = LEFT_OUT
        elif not found:
            row["why"] = EMPTY
        elif thin_under and not any(clips_of[lb] < thin_under for lb in labels):
            row["why"] = ENOUGH
        else:
            row["paths"] = spread(_in_order(found), cap)
            row["take"] = len(row["paths"])
            total += row["take"]
            new_classes.update(lb for lb in labels if not clips_of[lb])
    share = total / (train_before + total) if total else 0.0
    warnings = []
    if not thin_under or not cap:
        warnings.append("No limit on what is added: one video that becomes a large part of "
                        "a dataset has measured worse on every other video.")
    if share > LARGE_SHARE:
        warnings.append(f"This one video would be {share:.0%} of the training clips.")
    if new_classes:
        warnings.append("Only in this video, so learned but not reported until 3 or more "
                        "videos show them: "
                        + ", ".join(sorted(new_classes)))
    return {"dataset": os.path.abspath(dataset), "review": os.path.abspath(review),
            "video": video, "thin_under": thin_under or 0, "cap": cap or 0,
            "folders": rows, "take": total,
            "found": sum(r["found"] for r in rows),
            "skipped": sum(r["found"] for r in rows if not r["take"]),
            "train_clips_before": train_before, "share_of_train": round(share, 4),
            "new_classes": sorted(new_classes), "warnings": warnings,
            "already_added": len(mine)}


def _place(src: str, dst: str) -> str:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
        return "link"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def apply(planned: dict) -> dict:
    """Put a plan's crops into ``<dataset>/train/`` and record them.

    Safe to repeat: a crop already there is left as it is, and the record
    keeps everything this video has ever added, so ``undo`` removes it all.
    """
    dataset, video = planned["dataset"], planned["video"]
    before = read_manifest(dataset, video) or {}
    added = list(before.get("added", []))
    known = set(added)
    new, how = 0, Counter()
    for row in planned["folders"]:
        for src in row["paths"]:
            rel = "/".join((TRAIN, row["folder"], f"{video}{MARK}{os.path.basename(src)}"))
            dst = os.path.join(dataset, *rel.split("/"))
            if not os.path.exists(dst):
                how[_place(src, dst)] += 1
                new += 1
            if rel not in known:
                known.add(rel)
                added.append(rel)
    record = {"format": 1, "video": video, "review": planned["review"],
              "thin_under": planned["thin_under"], "cap": planned["cap"],
              "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "added": added,
              "left_out": {r["folder"]: {"found": r["found"], "why": r["why"]}
                           for r in planned["folders"] if not r["take"]}}
    path = manifest_path(dataset, video)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1)
    os.replace(tmp, path)
    return {"video": video, "added": new, "in_dataset": len(added), "linked": how["link"],
            "copied": how["copy"], "record": path}


def undo(dataset: str, video: str) -> dict:
    """Take out what ``apply`` put in for ``video``: those files, and only those."""
    record = read_manifest(dataset, video)
    if record is None:
        raise FileNotFoundError(f"nothing recorded as added for {video!r} in {dataset}")
    removed, folders = 0, set()
    for rel in record.get("added", []):
        path = os.path.join(dataset, *rel.split("/"))
        if os.path.isfile(path):
            os.remove(path)
            removed += 1
        folders.add(os.path.dirname(path))
    for folder in folders:
        if os.path.isdir(folder) and not os.listdir(folder):
            os.rmdir(folder)
    os.remove(manifest_path(dataset, video))
    return {"video": video, "removed": removed}


def drop(dataset: str, video: str, folders: Sequence[str]) -> dict:
    """Take out the crops ``apply`` put into these class folders for ``video``,
    and keep the rest: what a comparison by parts says to do."""
    record = read_manifest(dataset, video)
    if record is None:
        raise FileNotFoundError(f"nothing recorded as added for {video!r} in {dataset}")
    wanted = set(folders)
    kept, removed, emptied = [], Counter(), set()
    for rel in record.get("added", []):
        folder = rel.split("/")[1]
        if folder not in wanted:
            kept.append(rel)
            continue
        path = os.path.join(dataset, *rel.split("/"))
        if os.path.isfile(path):
            os.remove(path)
        removed[folder] += 1
        emptied.add(os.path.dirname(path))
    for folder in emptied:
        if os.path.isdir(folder) and not os.listdir(folder):
            os.rmdir(folder)
    record["added"] = kept
    record.setdefault("dropped", {}).update(
        {name: {"clips": count, "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
         for name, count in removed.items()})
    path = manifest_path(dataset, video)
    with open(path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1)
    os.replace(path + ".tmp", path)
    return {"video": video, "removed": sum(removed.values()), "by_folder": dict(removed),
            "in_dataset": len(kept), "not_found": sorted(wanted - set(removed))}


def added_videos(dataset: str) -> list:
    """The videos this dataset got through ``apply``, by name."""
    folder = os.path.join(dataset, MANIFEST_DIR)
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.splitext(n)[0] for n in os.listdir(folder) if n.endswith(".json"))


def summary(planned: dict, paths: bool = False) -> dict:
    """A plan without its file lists: what a person or an agent reads.
    ``paths`` keeps each folder's chosen crops, for a window that shows them."""
    out = {k: v for k, v in planned.items() if k != "folders"}
    out["folders"] = [{k: v for k, v in row.items() if paths or k != "paths"}
                      for row in planned["folders"]]
    return out
