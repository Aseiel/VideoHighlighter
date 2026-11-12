"""Which analysis signals a video already has saved.

The main window's per-signal scan buttons run one detector over the listed
videos and fold the result into each video's cache. Nothing on screen said
whether that had already happened, so the only way to find out was to press the
button and wait — or to open the timeline and look for the row. This module
answers it from the cache files, so the button can say "saved" beside itself.

"Saved" here means *a result is on disk*, not "a scan ran". A scan that found
nothing leaves an empty list, and so does a full run with that signal switched
off, and the two cannot be told apart afterwards. Reporting only what is
actually there is the claim the files can support.

Pure: no Qt, and the readers are passed in so the logic is testable without a
cache directory.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

# The kinds the main window can scan, in the order it lists them.
KINDS = ("motion", "audio", "transcript", "objects", "actions", "composition")


def kinds_in(cache: dict) -> frozenset:
    """The signal kinds one cache dict holds a result for."""
    cache = cache or {}
    found = set()

    if (cache.get("scenes") or cache.get("motion_events")
            or cache.get("motion_peaks") or cache.get("motion_curve")):
        found.add("motion")

    audio = cache.get("audio") if isinstance(cache.get("audio"), dict) else {}
    if (cache.get("audio_peaks") or audio.get("peaks") or audio.get("waveform")
            or audio.get("loudness_levels")):
        found.add("audio")

    transcript = cache.get("transcript")
    if isinstance(transcript, dict) and transcript.get("segments"):
        found.add("transcript")

    if cache.get("objects") or cache.get("object_bboxes"):
        found.add("objects")

    if cache.get("actions_all") or cache.get("actions"):
        found.add("actions")

    # Written by every composition run, even one whose rules never fired, so
    # this one kind can tell "applied, nothing matched" from "never applied".
    if "composed_event_names" in cache:
        found.add("composition")

    return frozenset(found)


# stamp of a video's cache files -> the kinds found in them. A cache can be
# tens of megabytes of boxes; parsing it again because the file list was
# touched would be paying that for an answer that has not changed.
_memo: Dict[tuple, frozenset] = {}


def _stamp(files: Iterable) -> tuple:
    out = []
    for path in files:
        try:
            st = path.stat()
            out.append((str(path), st.st_mtime_ns, st.st_size))
        except OSError:
            continue
    return tuple(out)


def saved_kinds(video_path: str, *,
                find_files: Optional[Callable] = None,
                load: Optional[Callable] = None) -> frozenset:
    """The kinds saved for one video, across every cache file it owns.

    A video can own several cache files (one per parameter signature, plus a
    legacy one) and an on-demand run writes into all of them, but an older full
    run may have left a signal in only one — so the answer is the union.
    """
    if find_files is None:
        def find_files(vp):
            from modules.report.analysis_ondemand import _cache_files
            return _cache_files(vp)[1]
    if load is None:
        def load(path):
            import json
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)

    try:
        files = list(find_files(video_path) or [])
    except Exception:
        return frozenset()          # a missing video has nothing saved
    if not files:
        return frozenset()

    key = _stamp(files)
    hit = _memo.get(key)
    if hit is not None:
        return hit

    found = set()
    for path in files:
        try:
            found |= kinds_in(load(path))
        except Exception:
            continue                # unreadable / half-written: says nothing
    result = frozenset(found)
    _memo[key] = result
    return result


def count_saved(video_paths: List[str], **readers) -> Dict[str, int]:
    """``{kind: how many of these videos have it saved}`` for every kind."""
    counts = {kind: 0 for kind in KINDS}
    for vp in video_paths or []:
        for kind in saved_kinds(vp, **readers):
            if kind in counts:
                counts[kind] += 1
    return counts


def describe(saved: int, total: int) -> str:
    """The few words shown beside a scan button for one kind."""
    if total <= 0:
        return ""
    if saved <= 0:
        return "nothing saved yet"
    if saved >= total:
        return "saved" if total == 1 else f"saved for all {total}"
    return f"saved for {saved} of {total}"
