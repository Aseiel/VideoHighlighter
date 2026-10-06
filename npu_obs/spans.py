"""Turn per-sample detections into time spans: "class X was on screen 12.5-31.0 s".

A detector sampled a few times a second flickers — a box drops out for one
sample and comes back. A span therefore opens only after ``min_hits`` samples
in a row see the class, and closes only once it has been missing for longer
than ``max_gap`` seconds. Both the live loop (which reports spans as they open
and close) and the recording scan (which wants the finished list) use this.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Span:
    label: str
    start: float          # seconds, time of the first sample that saw it
    end: float            # seconds, time of the last sample that saw it
    peak: float = 0.0     # highest confidence seen
    hits: int = 0         # samples that saw it

    def to_dict(self) -> dict:
        return {"label": self.label, "start": round(self.start, 3),
                "end": round(self.end, 3), "peak": round(self.peak, 3),
                "hits": self.hits}


@dataclass
class _Track:
    first: float
    last: float
    peak: float
    hits: int
    streak: int
    open: bool = False


@dataclass
class SpanTracker:
    min_hits: int = 2
    max_gap: float = 1.5
    _tracks: dict[str, _Track] = field(default_factory=dict)
    closed: list[Span] = field(default_factory=list)

    def update(self, t: float, seen: dict[str, float]) -> list[tuple[str, Span]]:
        """Feed one sample: ``seen`` maps label -> best confidence at time ``t``.

        Returns the events this sample caused, as ("start" | "end", span).
        """
        events: list[tuple[str, Span]] = []
        for label, conf in seen.items():
            tr = self._tracks.get(label)
            if tr is None or t - tr.last > self.max_gap:
                if tr is not None and tr.open:
                    events.append(("end", self._close(label, tr)))
                tr = self._tracks[label] = _Track(t, t, conf, 0, 0)
            tr.last = t
            tr.peak = max(tr.peak, conf)
            tr.hits += 1
            tr.streak += 1
            if not tr.open and tr.streak >= self.min_hits:
                tr.open = True
                events.append(("start", self._span(label, tr)))
        for label, tr in list(self._tracks.items()):
            if label in seen:
                continue
            tr.streak = 0
            if t - tr.last > self.max_gap:
                if tr.open:
                    events.append(("end", self._close(label, tr)))
                del self._tracks[label]
        return events

    def finish(self) -> list[tuple[str, Span]]:
        """Close every open span (end of stream) and return the "end" events."""
        events = [("end", self._close(label, tr))
                  for label, tr in self._tracks.items() if tr.open]
        self._tracks.clear()
        return events

    def _span(self, label: str, tr: _Track) -> Span:
        return Span(label, tr.first, tr.last, tr.peak, tr.hits)

    def _close(self, label: str, tr: _Track) -> Span:
        span = self._span(label, tr)
        self.closed.append(span)
        return span


def summarize(spans: list[Span]) -> dict[str, dict]:
    """Per label: how many spans, total seconds on screen, best confidence."""
    out: dict[str, dict] = {}
    for s in spans:
        row = out.setdefault(s.label, {"spans": 0, "seconds": 0.0, "peak": 0.0})
        row["spans"] += 1
        row["seconds"] += s.end - s.start
        row["peak"] = max(row["peak"], s.peak)
    for row in out.values():
        row["seconds"] = round(row["seconds"], 2)
        row["peak"] = round(row["peak"], 3)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]["seconds"]))
