"""Composition rules over MCP: read them, change one event, apply them to a
video, and read what they found - with the transcript that rules about speech
are written against.

The open timeline window follows its video's files
(``video_ai_editor/live_reload.py``): a rule changed here is re-applied there,
and events applied here show there, without reopening anything.

A rule is changed one event at a time, and the file is rewritten with
``ruamel.yaml`` when it is installed so the rest of it - comments included,
which in a rules file are often the measurements behind a threshold - stays
as it was. Without it the file is backed up once beside itself and written
plainly. Every change is checked by loading the whole file into the engine
first; a rule the engine cannot read is refused and nothing is written.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Sequence

def _rules_path() -> str:
    from modules.system.app_paths import composition_rules_path, user_data_dir
    return composition_rules_path() or os.path.join(user_data_dir(), "composition_rules.yaml")


def _read(path: str):
    """``(data, ruamel_yaml_or_None)``; data keeps comments when ruamel is here."""
    try:
        from ruamel.yaml import YAML
        y = YAML()
        y.preserve_quotes = True
        y.width = 4096
        with open(path, encoding="utf-8") as fh:
            return (y.load(fh) or {}), y
    except ImportError:
        import yaml
        with open(path, encoding="utf-8") as fh:
            return (yaml.safe_load(fh) or {}), None
    except FileNotFoundError:
        return {}, None


def _plain(obj):
    """A ruamel tree as plain dicts and lists, for answers."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    # ruamel keeps scalars as its own subclasses (to write them back as they
    # were); plain YAML and JSON need the base types.
    for base in (bool, int, float, str):
        if isinstance(obj, base):
            return base(obj)
    return obj


def _check(data) -> List[str]:
    """Load ``data`` into the engine; its event names, or ValueError."""
    import yaml
    from video_ai_editor.composition_engine import CompositionEngine
    fd, tmp = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            yaml.safe_dump(_plain(data), fh, allow_unicode=True, sort_keys=False)
        try:
            return list(CompositionEngine(tmp).event_names)
        except Exception as e:  # noqa: BLE001 - said to the caller
            raise ValueError(f"The engine cannot read these rules: {type(e).__name__}: {e}")
    finally:
        os.remove(tmp)


def _write(path: str, data, y) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    if y is not None:
        with open(tmp, "w", encoding="utf-8") as fh:
            y.dump(data, fh)
    else:
        import yaml
        backup = path + ".bak"
        if os.path.exists(path) and not os.path.exists(backup):
            shutil.copy2(path, backup)
        with open(tmp, "w", encoding="utf-8") as fh:
            yaml.safe_dump(_plain(data), fh, allow_unicode=True, sort_keys=False)
    os.replace(tmp, path)


def get_rules() -> dict:
    path = _rules_path()
    data, _y = _read(path)
    return {"path": path, "events": _plain((data or {}).get("events") or [])}


def set_rule(event: Dict[str, Any]) -> dict:
    """Add ``event`` or replace the event of the same name."""
    if not isinstance(event, dict) or not str(event.get("name") or "").strip():
        raise ValueError("An event needs a name.")
    if not event.get("signals") and not event.get("rules"):
        raise ValueError("An event needs `signals` or `rules`.")
    path = _rules_path()
    data, y = _read(path)
    if not isinstance(data, dict):
        data = {}
    events = data.get("events")
    if events is None:
        data["events"] = events = []
    name = str(event["name"]).strip()
    at = next((i for i, ev in enumerate(events) if str(ev.get("name")) == name), None)
    if at is None:
        events.append(dict(event))
        action = "added"
    else:
        events[at] = dict(event)
        action = "replaced"
    names = _check(data)
    _write(path, data, y)
    return {"path": path, "event": name, "action": action, "events": names}


def remove_rule(name: str) -> dict:
    path = _rules_path()
    data, y = _read(path)
    events = (data or {}).get("events") or []
    keep = [ev for ev in events if str(ev.get("name")) != str(name)]
    if len(keep) == len(events):
        raise ValueError(f"No event named {name!r}.")
    events[:] = keep
    _check(data)
    _write(path, data, y)
    return {"path": path, "removed": name, "events": [str(ev.get("name")) for ev in keep]}


def _stretches(seconds: Sequence[int]) -> List[dict]:
    out, run = [], []
    for s in sorted(set(int(x) for x in seconds)):
        if run and s - run[-1] > 1:
            out.append({"start": float(run[0]), "end": float(run[-1] + 1)})
            run = []
        run.append(s)
    if run:
        out.append({"start": float(run[0]), "end": float(run[-1] + 1)})
    return out


def events_in(cache: dict, names: Optional[Sequence[str]] = None) -> Dict[str, List[dict]]:
    """``{event: [{start, end}]}`` from a video's cache."""
    composed = list(cache.get("composed_event_names") or [])
    wanted = [n for n in (names or composed) if n]
    by_name: Dict[str, List[int]] = {n: [] for n in wanted}
    for entry in cache.get("objects") or []:
        for n in entry.get("objects") or []:
            if n in by_name:
                by_name[n].append(int(entry.get("timestamp", 0)))
    return {n: _stretches(secs) for n, secs in by_name.items()}


def get_events(video_path: str, names: Optional[Sequence[str]] = None) -> dict:
    from mcp_server.tools import _existing_file
    from modules.report.analysis_ondemand import read_cache
    path = _existing_file(video_path)
    found = events_in(read_cache(path) or {}, names)
    return {"video_path": path, "events": found,
            "seconds": {n: sum(r["end"] - r["start"] for r in rs) for n, rs in found.items()}}


def get_transcript(video_path: str, start: float = 0.0, end: Optional[float] = None) -> dict:
    from mcp_server.tools import _existing_file
    from modules.report.analysis_ondemand import read_cache
    path = _existing_file(video_path)
    transcript = (read_cache(path) or {}).get("transcript") or {}
    try:
        # Pro-only module, by name: a static import is a promise this shared
        # file cannot keep in the free edition (see tools._edition_module).
        from mcp_server.tools import _edition_module
        voice_pitch = _edition_module("modules.audio.voice_pitch")
        labels = voice_pitch.load(voice_pitch.cache_path_for(path))
        voice_of = lambda s: voice_pitch.voice_of(s, labels)  # noqa: E731
    except Exception:  # noqa: BLE001
        voice_of = lambda s: None  # noqa: E731
    lines = []
    for s in transcript.get("segments") or []:
        a, b = float(s.get("start", 0)), float(s.get("end", 0))
        if b < start or (end is not None and a > end):
            continue
        row = {"start": round(a, 2), "end": round(b, 2), "text": s.get("text", "")}
        if s.get("translation"):
            row["translation"] = s["translation"]
        v = voice_of(s)
        if v:
            row["voice"] = v
        lines.append(row)
    return {"video_path": path, "language": transcript.get("language"), "lines": lines}


def start_apply_rules(jobs, video_path: str) -> dict:
    from mcp_server.jobs import Job
    from mcp_server.tools import _existing_file
    path = _existing_file(video_path)

    def run(job: Job) -> dict:
        from modules.report.analysis_ondemand import merge_into_cache, run_composition
        patch = run_composition(path, log=job.log, cancel=job.cancel_flag)
        job.check_cancel()
        merge_into_cache(path, patch, log=job.log)
        found = events_in(patch)
        return {"video_path": path, "events": found,
                "seconds": {n: sum(r["end"] - r["start"] for r in rs) for n, rs in found.items()}}

    job = jobs.submit("apply_rules", {"video_path": path}, run)
    return {"job_id": job.id, "status": job.status}


def register(mcp, jobs, read_only, starts) -> None:
    """Add these tools to the FastMCP app (``server.build_server``)."""

    @mcp.tool(name="get_rules", annotations=read_only)
    def get_rules_tool() -> dict:
        """The composition rules file: its path and every event, as written."""
        return get_rules()

    @mcp.tool(name="set_rule", annotations=starts)
    def set_rule_tool(event: dict) -> dict:
        """Add an event to the composition rules, or replace the event with the
        same name. The engine reads the whole file first and refuses a rule it
        cannot read. The open timeline window re-applies the rules by itself."""
        return set_rule(event)

    @mcp.tool(name="remove_rule", annotations=starts)
    def remove_rule_tool(name: str) -> dict:
        """Remove one event from the composition rules by name."""
        return remove_rule(name)

    @mcp.tool(name="start_apply_rules", annotations=starts)
    def start_apply_rules_tool(video_path: str) -> dict:
        """Apply the composition rules to one video and save the events in its
        cache. Returns a job_id; poll get_job. Seconds when the signals the
        rules read are already measured, a minute or two the first time."""
        return start_apply_rules(jobs, video_path)

    @mcp.tool(name="get_events", annotations=read_only)
    def get_events_tool(video_path: str, names: Optional[list[str]] = None) -> dict:
        """Where each composed event holds in a video, as [start, end] seconds,
        and its total seconds. names: only these events (default: all)."""
        return get_events(video_path, names)

    @mcp.tool(name="get_transcript", annotations=read_only)
    def get_transcript_tool(video_path: str, start: float = 0.0,
                            end: Optional[float] = None) -> dict:
        """A video's transcript lines between start and end (s), each with its
        voice (male/female/both, by pitch) and translation when it has them."""
        return get_transcript(video_path, start, end)
