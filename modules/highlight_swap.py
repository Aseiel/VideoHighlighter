"""Replace a clip the user does not want with the best one that lost to it.

Re-choosing a moment needs the per-second score, which used to exist only inside
a pipeline run. The report now persists it (``curves.score_per_second``), so this
module can offer an alternative from a finished cut with no video file, no
detector and no re-analysis — a sort over an array that is already on disk.

State lives in :class:`SwapSession` because swapping is a conversation, not a
call: the moments already turned down have to accumulate, or pressing the button
twice offers the same clip again. The UI holds one session per cut and asks it
for the current segments; rendering stays a separate, explicit step, so trying
alternatives never costs an encode.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Sequence

import numpy as np

from modules.highlight_report import score_from_report, segments_from_report
from modules.highlight_select import swap_segment

# What the pipeline appends to the output's base name.
REPORT_SUFFIX = "_why.json"


def report_path_for(video_path: str,
                    output_path: Optional[str] = None) -> Optional[str]:
    """Locate the report for a cut, or ``None`` if it was never written.

    The pipeline names it after the output file when there is one and after the
    source video otherwise, so both are tried in that order.
    """
    bases = []
    if output_path:
        bases.append(os.path.splitext(output_path)[0])
    if video_path:
        bases.append(os.path.splitext(video_path)[0])
    for base in bases:
        candidate = f"{base}{REPORT_SUFFIX}"
        if os.path.exists(candidate):
            return candidate
    return None


class SwapSession:
    """One cut's worth of "give me a different clip".

    ``segments`` is the current list at all times; ``rejected`` grows with every
    swap so the next one offers something new.
    """

    def __init__(self,
                 score: np.ndarray,
                 segments: Sequence[tuple],
                 *,
                 video_duration: float,
                 clip_time: int,
                 duration_mode: str = "MAX"):
        self.score = np.asarray(score, dtype=float)
        self.segments = [(float(s), float(e)) for s, e in segments]
        self.video_duration = float(video_duration)
        self.clip_time = int(clip_time)
        self.duration_mode = str(duration_mode)
        self.rejected: list[tuple[float, float]] = []

    @classmethod
    def from_report(cls, report) -> "SwapSession":
        """Build a session from a report record or the path to one."""
        if isinstance(report, (str, os.PathLike)):
            with open(report, encoding="utf-8") as fh:
                report = json.load(fh)
        settings = report.get("settings") or {}
        clip_time = int(settings.get("clip_time") or 0)
        if clip_time <= 0:
            # Auto-segmentation produced variable-length clips; there is no
            # configured window to rebuild one from, so use what the cut itself
            # shows a clip is worth here.
            durations = [e["end"] - e["start"] for e in report.get("segments", [])]
            clip_time = int(round(sum(durations) / len(durations))) if durations else 10
        return cls(
            score_from_report(report),
            segments_from_report(report),
            video_duration=float((report.get("video") or {}).get("duration") or 0.0),
            clip_time=clip_time,
            duration_mode=str(settings.get("duration_mode") or "MAX"),
        )

    @property
    def usable(self) -> bool:
        """Whether this session can offer anything.

        A report written before the score was persisted loads fine and swaps
        nothing, which would look like a broken button. The caller checks this
        and says so instead.
        """
        return bool(len(self.score)) and bool(self.segments)

    def swap(self, index: int) -> bool:
        """Replace the clip at ``index``. False when nothing else is available."""
        replaced = self.segments[index]
        swapped = swap_segment(
            self.score,
            segments=self.segments,
            index=index,
            video_duration=self.video_duration,
            clip_time=self.clip_time,
            duration_mode=self.duration_mode,
            rejected=self.rejected,
        )
        if swapped is None:
            return False
        self.rejected.append(replaced)
        self.segments = swapped
        return True

    def undo(self) -> bool:
        """Put the most recently rejected moment back, in place of what replaced it.

        Trying alternatives is only comfortable if the first one can be recovered
        — otherwise a swap is a decision rather than a look.
        """
        if not self.rejected:
            return False
        restored = self.rejected.pop()
        # Whatever occupies the slot the restored clip would overlap is the one
        # that took its place.
        remaining = [
            seg for seg in self.segments
            if not (seg[0] < restored[1] and seg[1] > restored[0])
        ]
        if len(remaining) == len(self.segments) and self.segments:
            remaining = self.segments[:-1]
        self.segments = sorted(remaining + [restored], key=lambda seg: seg[0])
        return True
