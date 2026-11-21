"""What a composed event checked, in words, for the bar's tooltip.

A rule is named by whoever wrote it, and the name is all a timeline bar shows.
A week later the name no longer says which signals were asked for or at what
thresholds, and the bar cannot be questioned. This turns the rule back into
the conditions it was made of, so the reason for a bar is on the bar.

The words come from the rule file as it is now: an event drawn before the rule
was edited is explained by the current rule. Class names, descriptions and
event names are the user's and are shown as written.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

_CACHE: Dict[str, tuple] = {}

# Readable names for the measured signals. Anything else is shown as written.
_SIGNAL_NAMES = {
    "vocal_effort": "vocal effort",
    "vocal_density": "vocal density",
    "vocal_density_pct": "vocal density (percentile in the file)",
    "vocal_density_local": "vocal density (percentile, 10 min around)",
    "vocal_brightness": "vocal brightness",
    "vocal_onset": "vocal onset",
    "waveform_peak_density": "waveform peak density",
    "waveform_peak_density_pct": "waveform peak density (percentile)",
    "waveform_peak_density_local": "waveform peak density (local percentile)",
    "audio_level": "audio level",
    "motion_level": "motion level",
    "motion_rhythm": "motion rhythm",
    "motion_beat_hz": "motion beat (Hz)",
    "expression": "expression",
    "said": "said",
    "voice": "voice",
    "question": "question asked by",
}


def _num(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return str(int(f)) if f == int(f) else f"{f:g}"


def signal_name(name: str) -> str:
    """A signal name as a person would read it."""
    name = str(name or "")
    kind, _, rest = name.partition(":")
    args = rest.split(":") if rest else []
    if kind == "face_text" and rest:
        return f'face looks like "{rest}" (percentile)'
    if kind.startswith("said") and kind != "said":
        return f"said by {kind.split('_', 1)[-1]}"
    if kind == "said" and rest:
        return f"said by {rest}"
    if kind == "past_line" and len(args) == 2:
        return f"how far {args[0]} is past the line on {args[1]} (0 none, 1 all)"
    if kind == "line_crossings" and len(args) == 2:
        return f"{args[0]} passes into {args[1]} per second"
    if kind == "track_dwell" and len(args) == 2:
        return f"seconds {args[0]} has stayed on {args[1]}"
    if kind == "track_speed" and len(args) == 1:
        return f"{args[0]} speed (frame heights/s)"
    if kind == "track_direction" and len(args) == 1:
        return f"{args[0]} direction"
    return _SIGNAL_NAMES.get(name, name)


def _signal_line(cond: dict) -> str:
    parts = [signal_name(cond.get("signal"))]
    if cond.get("min") is not None and cond.get("max") is not None:
        parts.append(f"{_num(cond['min'])}-{_num(cond['max'])}")
    elif cond.get("min") is not None:
        parts.append(f">= {_num(cond['min'])}")
    elif cond.get("max") is not None:
        parts.append(f"<= {_num(cond['max'])}")
    values = cond.get("any_of") or cond.get("equals")
    if values:
        text = values if isinstance(values, str) else " | ".join(map(str, values))
        parts.append(f"is {text if len(text) <= 60 else text[:57] + '...'}")
    if cond.get("sustained_secs"):
        parts.append(f"for {_num(cond['sustained_secs'])} s in a row")
    if cond.get("within_secs"):
        parts.append(f"within {_num(cond['within_secs'])} s")
    line = " ".join(parts)
    return line + " (required)" if cond.get("required") else line


def _spatial_line(rule: dict) -> str:
    relation = str(rule.get("relation") or "inside")
    lo = rule.get("min_count")
    hi = rule.get("max_count")
    if hi is not None and int(hi) == 0:
        count = "none"
    elif hi is None or int(hi) >= 999:
        count = f"at least {int(lo if lo is not None else 1)}"
    else:
        count = f"{int(lo or 0)}-{int(hi)}"
    line = f"{rule.get('source')} {relation} {rule.get('region')}: {count}"
    if relation == "overlaps" and rule.get("min_overlap") is not None:
        line += f", {float(rule['min_overlap']):.0%} of it inside"
    if relation == "touches" and rule.get("max_gap"):
        line += f", gap up to {_num(rule['max_gap'])}"
    if rule.get("outline"):
        line += " (by outline)"
    return line


def explain(event: dict) -> List[str]:
    """The lines that say what ``event`` (one entry of a rules file) checks."""
    if not isinstance(event, dict):
        return []
    lines: List[str] = []
    for rule in event.get("rules") or []:
        lines.append("  " + _spatial_line(rule))
    signals = event.get("signals") or []
    need = int(event.get("min_matching") or 0)
    if signals:
        if need and need < len(signals):
            lines.append(f"  any {need} of these {len(signals)}:")
        elif lines:
            lines.append("  and all of these:")
        for cond in signals:
            lines.append("    " + _signal_line(cond))
    extra = []
    if event.get("min_duration_secs"):
        extra.append(f"lasting at least {_num(event['min_duration_secs'])} s")
    if event.get("max_duration_secs"):
        extra.append(f"at most {_num(event['max_duration_secs'])} s")
    if event.get("ignore_start_secs"):
        extra.append(f"not in the first {_num(event['ignore_start_secs'])} s")
    if extra:
        lines.append("  " + ", ".join(extra))
    return lines


def _load(path: str) -> list:
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return []
    cached = _CACHE.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    try:
        import yaml
        with open(path, encoding="utf-8") as fh:
            events = (yaml.safe_load(fh) or {}).get("events") or []
    except Exception:  # noqa: BLE001 - a broken file explains nothing, quietly
        events = []
    _CACHE[path] = (mtime, events)
    return events


def default_rules_path() -> Optional[str]:
    try:
        from modules.system.app_paths import composition_rules_path, user_data_dir
        return composition_rules_path() or os.path.join(user_data_dir(), "composition_rules.yaml")
    except Exception:  # noqa: BLE001
        return None


def explain_event(name: str, rules_path: Optional[str] = None) -> List[str]:
    """``explain`` for the event called ``name`` in the rules file; ``[]`` when
    the file or the event is not there."""
    path = rules_path or default_rules_path()
    if not path:
        return []
    for event in _load(path):
        if str(event.get("name")) == str(name):
            body = explain(event)
            return [f"Why \"{event.get('label') or name}\":"] + body if body else []
    return []
