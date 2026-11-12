"""Is a hand-sorted dataset healthy, and what would improve it most?

``health`` answers this for a project being built in the app; this answers
it for a dataset that already exists as folders
(``<dataset>/train/<class>/*.mp4``), which is what a trainer reads. The
rules are the same ones, each from something that went wrong in practice
(``docs/training/negative``):

* a class needs clips from 3 or more source videos, or the model learns the
  scene with it and nothing can measure that;
* no one video should be most of a class, or of the dataset;
* a clip whose name gives no source video counts as a video of its own, and
  held-out scores quietly leak;
* a video with clips in ``train`` and in ``val`` or ``test`` makes those
  folders' scores easier than new footage will be;
* the same file twice, above all under two classes, teaches a contradiction.

``check`` reads file names only and takes a moment. ``near_copies`` needs the
encoder's vectors (the trainer's cache) and finds what names cannot: clips
that are almost the same picture but sit in different videos or classes.

``wanted`` turns the findings round: the classes that more *videos* would
help most, thinnest first. That list is what to look for next.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional, Sequence

import numpy as np

from modules.teach import benchmark
from modules.teach.health import MAX_ONE_VIDEO

OK, WARN, BAD = "ok", "warn", "bad"
MIN_VIDEOS = 3
# Clips below which "one video is most of it" says nothing a count does not.
FLOOD_MIN_CLIPS = 5
# Cosine similarity of two clips' mean vectors above which they are the same
# moment re-cut or re-encoded rather than two looks at one scene.
NEAR_COPY = 0.985
SHOWN = 20


def _finding(level, what, text, fix, items=()):
    return {"level": level, "what": what, "text": text, "fix": fix,
            "items": list(items)[:SHOWN], "count": len(items)}


def classes_table(clips: Sequence[benchmark.Clip]) -> list:
    """Per class: clips, source videos, and the share its largest video has."""
    by_class = defaultdict(Counter)
    for clip in clips:
        for label in clip.labels:
            by_class[label][clip.group] += 1
    out = []
    for name in sorted(by_class):
        videos = by_class[name]
        total = sum(videos.values())
        out.append({"class": name, "clips": total, "videos": len(videos),
                    "largest_video_share": round(max(videos.values()) / total, 3)})
    return out


def wanted(table: Sequence[dict], min_videos: int = MIN_VIDEOS) -> list:
    """Classes that more source videos would help most, thinnest first."""
    rows = []
    for row in table:
        short = max(0, min_videos - row["videos"])
        flooded = row["clips"] >= FLOOD_MIN_CLIPS and row["largest_video_share"] > MAX_ONE_VIDEO
        if short or flooded:
            rows.append({"class": row["class"], "videos": row["videos"],
                         "clips": row["clips"], "videos_needed": short,
                         "why": (f"needs {short} more source video{'s' if short != 1 else ''}"
                                 if short else
                                 f"one video is {row['largest_video_share']:.0%} of it")})
    return sorted(rows, key=lambda r: (-r["videos_needed"], r["clips"]))


def check(dataset: str, aliases: Optional[dict] = None,
          group_pattern: str = benchmark.DEFAULT_GROUP) -> dict:
    """Findings from the folders and file names alone."""
    data = benchmark.read_dataset(dataset, aliases, group_pattern)
    clips = data["clips"]
    train = [c for c in clips if c.split == "train"]
    table = classes_table(train)
    findings = []

    thin = [r["class"] for r in table if r["videos"] < MIN_VIDEOS]
    if thin:
        findings.append(_finding(
            WARN, "few_videos",
            f"{len(thin)} of {len(table)} classes come from fewer than {MIN_VIDEOS} "
            f"source videos, so they can be learned but never trusted.",
            "Add clips of these classes from other videos.", thin))
    flooded = [f"{r['class']} ({r['largest_video_share']:.0%})" for r in table
               if r["clips"] >= FLOOD_MIN_CLIPS and r["largest_video_share"] > MAX_ONE_VIDEO]
    if flooded:
        findings.append(_finding(
            WARN, "one_video_floods_class",
            f"In {len(flooded)} classes one video is more than {MAX_ONE_VIDEO:.0%} of the clips.",
            "Add other videos of the class, or take fewer clips from the large one.", flooded))
    videos = Counter(c.group for c in train)
    if videos:
        name, most = videos.most_common(1)[0]
        if len(videos) > 1 and most / len(train) > 0.25:
            findings.append(_finding(
                BAD, "one_video_floods_dataset",
                f"One video is {most / len(train):.0%} of the training clips.",
                "Cap what is taken from it (add-reviewed does), then compare-added.", [name]))

    ungrouped = data.get("ungrouped", 0)
    if ungrouped:
        findings.append(_finding(
            BAD if ungrouped > len(clips) / 2 else WARN, "no_source_video",
            f"{ungrouped} clips have names that give no source video; each counts as a "
            f"video of its own, so clips of one film can sit on both sides of a held-out test.",
            "Rename them '<video>_highlight_<n>' (or pass a --group pattern that fits).",
            [f"{c.split}/{c.folder}/{_base(c.path)}" for c in clips
             if c.group == _stem(c.path)]))

    in_split = defaultdict(set)
    for clip in clips:
        in_split[clip.split].add(clip.group)
    for split in ("val", "test"):
        shared = in_split.get(split, set()) & in_split.get("train", set())
        if shared:
            findings.append(_finding(
                WARN, f"{split}_shares_videos_with_train",
                f"{len(shared)} of {len(in_split[split])} videos in {split}/ also have clips in "
                f"train/, so a score on {split}/ is easier than new footage will be.",
                "Trust the trainer's held-out-video score instead, or move whole videos.",
                sorted(shared)))

    same = benchmark.duplicates(clips)
    conflict = [g for g in same if len({c.key for c in g}) > 1]
    if conflict:
        findings.append(_finding(
            BAD, "same_file_two_classes",
            f"{len(conflict)} files are in the dataset under two different classes.",
            "Keep each in one class (or in an 'a_b' folder if it shows both).",
            [" | ".join(f"{c.split}/{c.folder}/{_base(c.path)}" for c in g) for g in conflict]))
    elif same:
        findings.append(_finding(
            WARN, "same_file_twice", f"{len(same)} files are in the dataset more than once.",
            "Remove the extra copies.",
            [" | ".join(f"{c.split}/{c.folder}/{_base(c.path)}" for c in g) for g in same]))

    level = BAD if any(f["level"] == BAD for f in findings) else WARN if findings else OK
    return {"dataset": dataset, "level": level,
            "clips": len(clips), "train_clips": len(train),
            "classes": len(table), "videos": len({c.group for c in clips}),
            "findings": findings, "wanted": wanted(table), "classes_table": table}


def _base(path: str) -> str:
    import os
    return os.path.basename(path)


def _stem(path: str) -> str:
    import os
    return os.path.splitext(os.path.basename(path))[0]


def near_copies(vectors: np.ndarray, clips: Sequence[benchmark.Clip],
                threshold: float = NEAR_COPY, limit: int = 200) -> dict:
    """Pairs of clips that are almost the same picture where that matters:
    in **different classes** (one of the two labels is wrong, or the clip
    shows both), or in **different source videos** (the same footage under two
    names, which leaks across a held-out test). Pairs inside one video and
    class are what consecutive clips look like, and are not reported.

    ``vectors`` is [N, dims] or [N, frames, dims], in the order of ``clips``.
    """
    v = np.asarray(vectors, np.float32)
    if v.ndim == 3:
        v = v.mean(1)
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
    labels = [c.key for c in clips]
    groups = [c.group for c in clips]
    classes, videos = [], []
    for start in range(0, len(v), 512):
        sims = v[start:start + 512] @ v.T
        for i, j in zip(*np.nonzero(sims >= threshold)):
            i += start
            if j <= i:
                continue
            pair = {"a": _where(clips[i]), "b": _where(clips[j]),
                    "similarity": round(float(sims[i - start, j]), 4)}
            if labels[i] != labels[j]:
                classes.append(pair)
            elif groups[i] != groups[j]:
                videos.append(pair)
    classes.sort(key=lambda p: -p["similarity"])
    videos.sort(key=lambda p: -p["similarity"])
    return {"threshold": threshold,
            "different_classes": {"count": len(classes), "pairs": classes[:limit]},
            "different_videos": {"count": len(videos), "pairs": videos[:limit]}}


def _where(clip: benchmark.Clip) -> str:
    return f"{clip.split}/{clip.folder}/{_base(clip.path)}"
