"""
manual_avoid.py — user-specified forbidden ranges for the highlight pipeline.

Background
==========
The existing AVOID path in `pipeline.py` already supports a `forbidden_ranges`
list — `[(start_sec, end_sec), ...]` — which feeds two downstream behaviours:

  1. Score-zeroing (`pipeline.py` ~L1409): every second falling inside a
     forbidden range gets its highlight score set to 0.0.
  2. Segment subtraction (`subtract_forbidden`, now in
     `modules.pipeline_helpers`): forbidden ranges are cut out of any
     chosen highlight segment, with sliver-removal for sub-`min_keep`
     fragments.

Today that list is only populated automatically by `compute_forbidden.py`
from face-identity detections ("avoid this specific person"). There is no
way for a user to mark a region of the timeline as "do not include this".

This module is the data layer for the user-driven path. UI is intentionally
out of scope here: the future right-click "Avoid this range" context menu in
the signal timeline viewer will call into these helpers, but this module knows
nothing about Qt.

Public API
==========

    parse_ranges(raw) -> list[(start, end)]
        Accept a permissive input format (list of tuples, list of dicts,
        list of "MM:SS-MM:SS" strings) and return a clean,
        validated, sorted, non-overlapping list ready to feed
        `subtract_forbidden`.

    merge_overlapping(ranges, gap_tolerance=0.0) -> list[(start, end)]
        Collapse overlapping or near-touching ranges.

    combine(auto_ranges, manual_ranges) -> list[(start, end)]
        Merge automatic (face-identity) forbidden ranges with user-supplied
        manual ranges. Result is the union with overlaps collapsed.

Why a dedicated module?
=======================
The combine step matters even at the data level. Without it, score-zeroing
could double-process overlapping ranges (harmless but wasteful) and the
debug log line "located N forbidden range(s)" would be misleading. Doing
the merge in a tested helper keeps the orchestration in `pipeline.py` as
mechanical as possible.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Sequence, Tuple

Range = Tuple[float, float]


# ---------------------------------------------------------------------------
# Permissive input parsing
# ---------------------------------------------------------------------------
_TIME_TOKEN = re.compile(
    r"^\s*(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)\s*$"   # [HH:]MM:SS[.ms]
    r"|"
    r"^\s*(\d+(?:\.\d+)?)\s*$"                    # plain seconds
)


def _parse_time_token(token: str) -> float:
    """Parse "MM:SS", "HH:MM:SS", or "S" into seconds.

    Raises ValueError on anything else so callers know the input is dirty.
    """
    if isinstance(token, (int, float)):
        return float(token)
    if not isinstance(token, str):
        raise ValueError(f"expected str or number, got {type(token).__name__}")
    m = _TIME_TOKEN.match(token)
    if not m:
        raise ValueError(f"unrecognised time token: {token!r}")
    if m.group(4) is not None:
        return float(m.group(4))
    hh = int(m.group(1)) if m.group(1) else 0
    mm = int(m.group(2))
    ss = float(m.group(3))
    return hh * 3600 + mm * 60 + ss


def parse_ranges(raw: Any) -> List[Range]:
    """Normalise a permissive input to a clean list of `(start, end)` floats.

    Accepted shapes for individual entries:
      - `(start, end)` tuple/list of numbers or time tokens.
      - `{"start": ..., "end": ...}` dict.
      - `"MM:SS-MM:SS"` or `"S-S"` dash-separated string.

    Cleaning rules:
      - end > start (entries where end <= start are dropped silently).
      - Negative starts are clamped to 0.
      - The output is sorted by `start`.
      - Overlapping ranges are NOT merged here — call `merge_overlapping`
        for that. parse_ranges only validates and orders.

    Returns `[]` for falsy or unrecognised input rather than raising;
    individual malformed entries within an otherwise valid list raise
    ValueError. The asymmetry is deliberate: "no input" is fine, "bad input"
    is a bug we want surfaced.
    """
    if not raw:
        return []

    entries: List[Range] = []
    for item in raw:
        if isinstance(item, dict):
            start_raw = item.get("start")
            end_raw = item.get("end")
            if start_raw is None or end_raw is None:
                raise ValueError(f"range dict missing start/end: {item!r}")
            start = _parse_time_token(start_raw)
            end = _parse_time_token(end_raw)
        elif isinstance(item, str):
            # Dash-separated: support both `-` and `→` for friendlier paste-from-doc.
            for sep in ("-", "→", "—", "to"):
                if sep in item:
                    a, _, b = item.partition(sep)
                    start = _parse_time_token(a)
                    end = _parse_time_token(b)
                    break
            else:
                raise ValueError(f"string range missing separator: {item!r}")
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            start = _parse_time_token(item[0])
            end = _parse_time_token(item[1])
        else:
            raise ValueError(f"unrecognised range entry: {item!r}")

        start = max(0.0, float(start))
        end = float(end)
        if end > start:
            entries.append((start, end))

    entries.sort(key=lambda r: r[0])
    return entries


# ---------------------------------------------------------------------------
# Overlap collapse
# ---------------------------------------------------------------------------
def merge_overlapping(ranges: Sequence[Range],
                      gap_tolerance: float = 0.0) -> List[Range]:
    """Collapse overlapping or near-touching ranges into the minimal cover.

    `gap_tolerance` bridges ranges separated by at most that many seconds.
    Default is 0.0 (strictly touching counts; a 0.1s gap stays separate).
    Useful values:
      - 0.0  — pure mathematical union.
      - 0.5  — bridge tracker flicker.
      - 2.0  — match `compute_forbidden._merge_seconds` default for
               consistent behaviour when manual + auto ranges combine.

    Input must be sorted (call `parse_ranges` first or sort by `start`).
    Mixed input is sorted defensively.
    """
    if not ranges:
        return []
    sorted_ranges = sorted(ranges, key=lambda r: r[0])
    out: List[Range] = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        last_start, last_end = out[-1]
        if start <= last_end + gap_tolerance:
            out[-1] = (last_start, max(last_end, end))
        else:
            out.append((start, end))
    return out


# ---------------------------------------------------------------------------
# Combine automatic + manual
# ---------------------------------------------------------------------------
def combine(auto_ranges: Iterable[Range],
            manual_ranges: Iterable[Range],
            gap_tolerance: float = 0.0) -> List[Range]:
    """Union of automatic (face-identity) ranges and manual user ranges.

    Both inputs are normalised through `merge_overlapping` before the union,
    so the call is idempotent: `combine(a, []) == merge_overlapping(a)`.

    The returned list is the canonical form to feed to
    `pipeline.py`'s scoring zeroing pass and to `subtract_forbidden`.
    """
    return merge_overlapping(list(auto_ranges) + list(manual_ranges),
                             gap_tolerance=gap_tolerance)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
# Ranges marked in the timeline viewer used to live only in that window's
# in-memory scene: main.py read them back by reaching into the live widget
# (_get_manual_avoid_ranges). That only works because the viewer runs in the
# same process. Any other consumer — the web UI's sidecar, a second session,
# or the same session after the viewer is closed — had no way to see them.
#
# Storing them on disk, keyed by video path, makes the ranges the source of
# truth rather than a widget's private state. The in-process reader still wins
# when the live window is open, so the Qt behaviour is unchanged.

_STORE_FILENAME = "manual_avoid.json"


def store_path() -> str:
    """Location of the shared ranges file (next to the other user data)."""
    import os

    from modules.app_paths import user_data_dir

    return os.path.join(user_data_dir(), "cache", _STORE_FILENAME)


def _load_store() -> dict:
    import json
    import os

    path = store_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        # A corrupt store must never block a run; treat it as empty.
        return {}


def load_ranges(video_path: str) -> List[Range]:
    """Manual avoid ranges saved for ``video_path`` (empty when none)."""
    raw = _load_store().get(str(video_path)) or []
    try:
        return merge_overlapping(parse_ranges(raw))
    except ValueError:
        return []


def save_ranges(video_path: str, ranges: Iterable[Range]) -> str:
    """Persist ``ranges`` for ``video_path``. Returns the store path.

    Saving an empty list removes the entry rather than storing [], keeping the
    file free of dead keys as users clear ranges.
    """
    import json
    import os

    store = _load_store()
    clean = merge_overlapping(parse_ranges(list(ranges)))
    key = str(video_path)
    if clean:
        store[key] = [[float(a), float(b)] for a, b in clean]
    else:
        store.pop(key, None)

    path = store_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(store, fh, indent=2)
    return path