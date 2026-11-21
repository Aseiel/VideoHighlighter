"""What each condition of a composed event actually measured, per bar.

A rule says "at most this" or "at least that"; the bar it draws says only that
the rule held. The numbers in between - how deep, how loud, how sure the face
reading was - are what make a bar something to agree or disagree with. This
keeps them: for every run of seconds an event fired, each signal condition's
readings over that run (lowest, highest, average; or the labels seen), next to
the threshold it was held to.

Stored by ``run_composition`` as ``composed_evidence`` in the analysis cache:
``{event_name: [{"start", "end", "signals": [{signal, need, min, max, mean,
seconds, labels}]}]}``. Nothing here knows what any signal or event is about.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Mapping, Optional

# Seconds between two hits that still count as one run (the timeline draws
# runs the same way, so a bar and its evidence cover the same seconds).
RUN_GAP = 1


def runs(seconds: Iterable[int], gap: int = RUN_GAP) -> List[tuple]:
    """``[(start, end_inclusive), ...]`` of consecutive seconds."""
    out: List[tuple] = []
    for s in sorted(set(int(x) for x in seconds)):
        if out and s - out[-1][1] <= gap:
            out[-1] = (out[-1][0], s)
        else:
            out.append((s, s))
    return out


def _value(source, sec: int):
    if source is None:
        return None
    if isinstance(source, Mapping):
        return source.get(sec, source.get(str(sec)))
    try:
        return source[sec] if 0 <= sec < len(source) else None
    except TypeError:
        return None


def need_text(cond: Mapping) -> str:
    lo, hi = cond.get("min"), cond.get("max")
    if lo is not None and hi is not None:
        return f"{lo:g}-{hi:g}"
    if lo is not None:
        return f">= {lo:g}"
    if hi is not None:
        return f"<= {hi:g}"
    values = cond.get("any_of") or cond.get("equals")
    if values:
        return "is " + (values if isinstance(values, str) else " | ".join(map(str, values)))
    return ""


def measure(cond: Mapping, signals: Mapping, start: int, end: int) -> dict:
    """One condition's readings over ``start``-``end`` (inclusive), widened by
    its ``within_secs`` the way the engine widens it."""
    name = str(cond.get("signal") or "")
    pad = int(round(float(cond.get("within_secs") or 0)))
    source = signals.get(name)
    nums, labels = [], {}
    for sec in range(start - pad, end + pad + 1):
        v = _value(source, sec)
        if v is None:
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            nums.append(float(v))
        else:
            labels[str(v)] = labels.get(str(v), 0) + 1
    row = {"signal": name, "need": need_text(cond), "seconds": len(nums) + sum(labels.values())}
    if nums:
        row.update(min=round(min(nums), 4), max=round(max(nums), 4),
                   mean=round(sum(nums) / len(nums), 4))
    if labels:
        row["labels"] = dict(sorted(labels.items(), key=lambda kv: -kv[1]))
    return row


def evidence(events: Iterable[Mapping], composed: Mapping, signals: Mapping) -> Dict[str, list]:
    """``composed`` is the engine's ``{second: [event names]}``; ``events`` the
    rules file's event list. Events without signal conditions are left out -
    their reason is the spatial rule, which the tooltip already states."""
    by_event: Dict[str, list] = {}
    for sec, names in (composed or {}).items():
        for n in names or []:
            by_event.setdefault(str(n), []).append(int(sec))
    out: Dict[str, list] = {}
    for event in events or []:
        name = str(event.get("name") or "")
        conds = event.get("signals") or []
        if not conds or name not in by_event:
            continue
        out[name] = [{"start": a, "end": b + 1,
                      "signals": [measure(c, signals, a, b) for c in conds]}
                     for a, b in runs(by_event[name])]
    return out


def for_bar(evidence_map: Optional[Mapping], event_name: str, start: float,
            end: float) -> Optional[dict]:
    """The stored run that overlaps a bar most, matching the event name without
    regard to case (the timeline shows names normalised)."""
    if not evidence_map:
        return None
    key = next((k for k in evidence_map if k.lower() == str(event_name).lower()), None)
    best, best_overlap = None, 0.0
    for run in (evidence_map.get(key) or []) if key else []:
        overlap = min(end, run["end"]) - max(start, run["start"])
        if overlap > best_overlap:
            best, best_overlap = run, overlap
    return best


def _clock(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}:{sec // 60 % 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"


def _n(v: float) -> str:
    """A reading as people read one: 72, 1.67, 0.29 - not 71.5833."""
    v = float(v)
    text = f"{v:.0f}" if abs(v) >= 10 else f"{v:.2f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def describe(run: Mapping) -> List[str]:
    """The tooltip lines for one run."""
    from modules.rules.rule_explain import signal_name

    lines = [f"Measured here ({_clock(run['start'])}-{_clock(run['end'])}):"]
    for row in run.get("signals") or []:
        name = row["signal"]
        need = f" (needs {row['need']})" if row.get("need") else ""
        if "max" in row:
            if name.startswith("past_line:"):
                source, region = name.split(":")[1:3]
                text = (f"{source} past the line on {region}: deepest {row['max']:.0%} "
                        f"of its length, {row['min']:.0%}-{row['max']:.0%} over "
                        f"{row['seconds']} s")
                need = re.sub(r"(\d+(?:\.\d+)?)", lambda m: f"{float(m.group(1)):.0%}", need)
            elif row["min"] == row["max"]:
                text = f"{signal_name(name)}: {_n(row['max'])}"
            else:
                text = (f"{signal_name(name)}: {_n(row['min'])} to {_n(row['max'])}, "
                        f"average {_n(row['mean'])}")
        elif row.get("labels"):
            text = f"{signal_name(name)}: " + ", ".join(
                f"{k} {v}s" for k, v in list(row["labels"].items())[:4])
        else:
            text = f"{signal_name(name)}: no reading"
        lines.append("  " + text + need)
    return lines
