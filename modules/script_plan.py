"""
script_plan.py — the film the user asked for, written where the cutter can read it.

Why this exists
===============
The selector scores every second and greedily takes the best of them
(``modules.highlight_select``). That answers "which were the strongest
moments", which is the right question for a highlights reel and the wrong one
for a film. It cannot express intent — *open with ten seconds of establishing
shots, then three action beats of about eight seconds each, close on something
calm* — because what is being described is a **sequence of different requests**,
not one request with better numbers. No arrangement of sliders says it.

A script says it in one file. Each beat carries its own length, its own match
terms and its own ordering; :func:`compile_directives` flattens the beats into
one selection request per clip. The selector therefore keeps knowing nothing
about scripts, and the script keeps knowing nothing about scoring — the two
meet at :class:`CutDirective` and nowhere else.

Being a file is half the value. A slider setting exists only in the moment it
is dragged; a script is reviewable, diffable and re-runnable. Change one beat's
duration, run again, compare the two cuts, keep the better file.

Why unknown keys are fatal
==========================
Writing ``durations:`` instead of ``duration:`` is a thirty-second mistake that
costs an hour when the parser ignores it: the run completes, the cut is wrong
in a way that looks like the *engine* misbehaving, and the file that would
explain it reads as correct. So every key is checked against a known set and an
unrecognised one stops the parse, with the line number and the nearest legal
spelling. The same reasoning covers duplicate keys, unknown ``order`` values,
and a scalar where a list belongs: anything a person could plausibly typo is
rejected loudly rather than dropped quietly.

:func:`validate_script` is the other half of that split. Parsing answers "can
this be executed at all" and its failures must stop the run; validation answers
"will executing it disappoint", which is a judgement — and a judgement must
never be the reason a render does not happen.

The format
==========
YAML, because the repo already carries PyYAML and ``config.yaml`` established
the vocabulary. Hand-writable is the requirement it is designed against::

    title: My Film
    music: D:\\music\\track.mp3     # optional
    snap_to_beat: true              # optional
    total_duration: 180             # optional overall target, seconds
    beats:
      - name: Establishing
        duration: 12                # or [8, 15] for a min/max range
        match:
          objects: []               # user-supplied terms, all optional
          actions: []
          keywords: []
        sources: []                 # optional: only cut from these files
        order: chronological        # chronological | best_first (default)
      - name: Action
        repeat: 3                   # this beat contributes 3 clips
        duration: [6, 10]

Every match term is the user's own, typed at runtime. This module ships no
category lists, no presets and no vocabulary of its own: it matches whatever it
is given and holds no opinion about what that is.

Public API
==========
    parse_script(text_or_dict)  -> Script      # raises ScriptError
    load_script(path)           -> Script
    save_script(script, path)   -> str         # returns the path
    validate_script(script)     -> list[str]   # non-fatal warnings
    compile_directives(script)  -> list[CutDirective]
    example_script()            -> str         # commented starter template
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Any, NoReturn

# Every key this format understands. Membership is checked, not assumed — see
# the module docstring for why a stray key is an error rather than a shrug.
_TOP_KEYS = ("title", "music", "snap_to_beat", "total_duration", "beats")
_BEAT_KEYS = ("name", "duration", "repeat", "match", "sources", "order")
_MATCH_KEYS = ("objects", "actions", "keywords")

# chronological: the beat's clips play in the order they were shot, which is
# what makes an establishing or closing sequence read as one place rather than
# a montage. best_first: highest scoring moment first, the reel behaviour.
_ORDERS = ("best_first", "chronological")

_DEFAULT_ORDER = "best_first"
_DEFAULT_TITLE = "Untitled"


class ScriptError(ValueError):
    """A script that cannot be executed as written.

    ValueError rather than a new exception root: a caller already guarding a
    config load against ValueError catches this too, and a script is config.
    Every message names what is wrong and, when the script came from text,
    which line to look at.
    """


@dataclass
class Beat:
    """One movement of the film: how long, what it should hold, in what order.

    ``min_duration``/``max_duration`` are always both set even when the script
    wrote a single number — a fixed length is a range whose ends agree. The
    selector then has one shape to handle instead of two, and no downstream
    code has to ask which kind of duration it was given.
    """

    name: str
    min_duration: float
    max_duration: float
    repeat: int = 1
    objects: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    order: str = _DEFAULT_ORDER

    @property
    def has_match_terms(self) -> bool:
        """Whether this beat asks for anything in particular.

        A beat with no terms is legal and useful — it means "the best of
        whatever is here" — but it is also what a half-written beat looks like,
        so :func:`validate_script` mentions it.
        """
        return bool(self.objects or self.actions or self.keywords)


@dataclass
class Script:
    """A parsed script: the whole film, in the order it plays."""

    title: str
    beats: list[Beat]
    music: str = ""
    snap_to_beat: bool = False
    total_duration: float = 0.0

    @property
    def target_duration(self) -> float:
        """The budget to hand the selector, in seconds.

        An explicit ``total_duration`` wins outright: the user asked for a film
        of that length and the beats describe how to spend it. Without one the
        budget is the sum of the beat *maxima* — the most the script asks for —
        so the selector is never short of the length the script describes and
        the shortfall, if any, comes from the footage rather than the arithmetic.
        """
        if self.total_duration > 0:
            return float(self.total_duration)
        return float(sum(b.max_duration * b.repeat for b in self.beats))

    @property
    def clip_count(self) -> int:
        """How many clips the finished film contains — one per repeat."""
        return sum(int(b.repeat) for b in self.beats)


@dataclass
class CutDirective:
    """One clip's worth of instruction: the unit the engine actually acts on.

    A beat with ``repeat: 3`` is not a special case for the selector to learn;
    it is three requests that happen to share their constraints. Flattening
    happens here so that selection, reporting and the UI all iterate a flat
    list of clips in film order and none of them has to know the script format.
    """

    beat_name: str
    index: int                      # 0-based within its beat: clip 3 of 3 is 2
    min_duration: float
    max_duration: float
    objects: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    order: str = _DEFAULT_ORDER

    @property
    def has_match_terms(self) -> bool:
        """False when this clip should be filled from the overall score."""
        return bool(self.objects or self.actions or self.keywords)

    def accepts_source(self, path: str) -> bool:
        """May this clip be cut from ``path``?

        An empty ``sources`` list allows everything: a restriction nobody wrote
        is not one. The comparison is on the base name and case-insensitive,
        because the script names files the way the user sees them in the folder
        ("GX010123.MP4") while the engine carries full paths — and on Windows
        the same file arrives spelled either way on different days.
        """
        if not self.sources:
            return True
        name = _base_name(path)
        return any(_base_name(s) == name for s in self.sources)


def _base_name(path: str) -> str:
    """The file name out of a path written for either operating system.

    ``os.path.basename`` splits on the *host's* separator, so a Windows path
    handed to a process running on Linux comes back whole and two spellings of
    the same file stop matching. A script is a portable document — written on
    the machine that holds the footage, run wherever the engine happens to be,
    including CI — so both separators are treated as one here rather than
    leaving the answer to depend on which machine asked.
    """
    return str(path).replace("\\", "/").rsplit("/", 1)[-1].casefold()


# ---------------------------------------------------------------------------
# Source positions
# ---------------------------------------------------------------------------

class _Marked(dict):
    """A parsed mapping that remembers the lines it came from.

    PyYAML discards source positions once a document is built, so a semantic
    complaint ("no such key", "duration must be positive") can say what is
    wrong but not where. On a sixty-line script with eight similar-looking
    beats that is the difference between a ten-second fix and a hunt.
    """

    __slots__ = ("line", "key_lines")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.line: int = 0
        self.key_lines: dict[str, int] = {}


_LOADER: Any = None


def _loader():
    """The SafeLoader subclass that produces :class:`_Marked` mappings.

    Built on first use and cached: yaml is a deferred import (this module has
    to stay importable in a minimal environment), so the class cannot be
    declared at module scope.
    """
    global _LOADER
    if _LOADER is not None:
        return _LOADER

    import yaml

    class _LineLoader(yaml.SafeLoader):
        def construct_yaml_map(self, node):
            data = _Marked()
            yield data
            seen: dict[str, int] = {}
            for key_node, _value_node in node.value:
                key = str(getattr(key_node, "value", ""))
                line = key_node.start_mark.line + 1
                if key in seen:
                    # PyYAML keeps the last of a repeated key and says nothing.
                    # That is the silent-drop failure this format exists to
                    # refuse, so it is reported like any other typo.
                    raise ScriptError(
                        f"{key!r} is set twice (lines {seen[key]} and {line}); "
                        "the first one would be silently ignored"
                    )
                seen[key] = line
            data.line = node.start_mark.line + 1
            data.key_lines = seen
            data.update(self.construct_mapping(node))

    _LineLoader.add_constructor(
        "tag:yaml.org,2002:map", _LineLoader.construct_yaml_map
    )
    _LOADER = _LineLoader
    return _LOADER


def _where(mapping: Any = None, key: str | None = None) -> str:
    """`` (line 14)`` when the position is known, and nothing when it is not.

    A script handed over as a dict has no lines, and a message that invented
    one would be worse than a message without one.
    """
    lines = getattr(mapping, "key_lines", None) or {}
    line = lines.get(key) if key else None
    if not line:
        line = int(getattr(mapping, "line", 0) or 0)
    return f" (line {line})" if line else ""


def _fail(message: str, mapping: Any = None, key: str | None = None) -> NoReturn:
    raise ScriptError(f"{message}{_where(mapping, key)}")


def _describe(value: Any) -> str:
    """How to name a wrong value in an error message, from the user's side."""
    if value is None:
        return "nothing"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Mapping):
        return "a mapping"
    if isinstance(value, (list, tuple)):
        return "a list"
    return repr(value)


# ---------------------------------------------------------------------------
# Field readers — each one rejects what it cannot honour
# ---------------------------------------------------------------------------

def _text(value: Any, what: str, mapping: Any = None, key: str | None = None) -> str:
    """A single-line string. Absent reads as empty; a list or mapping is an error."""
    if value is None:
        return ""
    if isinstance(value, (Mapping, list, tuple)):
        _fail(f"{what} must be text, got {_describe(value)}", mapping, key)
    return str(value).strip()


def _number(value: Any, what: str, mapping: Any = None,
            key: str | None = None) -> float:
    """A number that can actually be a length. ``True`` is rejected despite
    being an int in Python — a duration of ``yes`` is a typo, not a length.

    ``.nan`` and ``.inf`` are refused here rather than by each caller because
    they defeat every range check written downstream: ``nan <= 0`` and
    ``nan > high`` are both false, so a NaN would walk through
    :func:`_durations` untouched and reach the engine as a clip length, and an
    infinite total is a budget nothing can spend. ``1.0e400`` is infinity by
    another spelling and a four-hundred-digit integer is not a float at all, so
    the guard is on the converted number and not on how it was written — both
    would otherwise leave as an OverflowError, which callers guarding
    :class:`ScriptError` do not catch.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{what} must be a number, got {_describe(value)}", mapping, key)
    try:
        number = float(value)
    except (OverflowError, ValueError):
        _fail(f"{what} is too large to be a number of seconds", mapping, key)
    if not math.isfinite(number):
        _fail(f"{what} must be a finite number of seconds, got {number:g}",
              mapping, key)
    return number


def _term_list(value: Any, what: str, mapping: Any = None,
               key: str | None = None) -> list[str]:
    """A list of user-supplied terms.

    A bare string is refused rather than wrapped in a list. ``objects: one``
    reads as a single term to a person and would work if it were accepted,
    which is exactly why it must not be: the day it grows into ``objects: one,
    two`` it silently becomes one nonsense term instead of two real ones.
    """
    if value is None:
        return []
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        _fail(f"{what} must be a list, got {_describe(value)} — "
              f"write '{what.split('.')[-1]}: [one, two]'", mapping, key)
    terms: list[str] = []
    for item in value:
        if isinstance(item, (Mapping, list, tuple)):
            _fail(f"{what} has an entry that is not a term: {_describe(item)}",
                  mapping, key)
        text = "" if item is None else str(item).strip()
        if not text:
            # A dash with nothing after it — an interrupted edit. It matches
            # nothing, and a script that quietly matches nothing looks from the
            # outside exactly like a script that is working.
            _fail(f"{what} has an empty entry", mapping, key)
        terms.append(text)
    return terms


def _reject_unknown(mapping: Mapping, allowed: tuple[str, ...], what: str) -> None:
    """Stop on the first key this format does not understand.

    The message carries the nearest legal spelling when there is one, because
    the whole point of failing here is to end the search immediately.

    A key that is a real key at the wrong level is answered with the level and
    not with a spelling, and that test comes first: ``actions`` is a legal match
    term and is also two edits from ``duration``, so asking the spelling checker
    first sends the one person who knows exactly what they wrote off to look at
    an unrelated key. Reaching here at all means the name is not in ``allowed``,
    so when the mapping being checked *is* the match block its own keys have
    already been let through and this branch cannot misfire.
    """
    for key in mapping:
        name = str(key)
        if name in allowed:
            continue
        if name in _MATCH_KEYS:
            hint = " — match terms belong under 'match:'"
        else:
            close = get_close_matches(name, allowed, n=1, cutoff=0.6)
            hint = (f" — did you mean {close[0]!r}?" if close
                    else f" — known {what} keys: {', '.join(allowed)}")
        _fail(f"unknown {what} key {name!r}{hint}", mapping, name)


def _durations(value: Any, name: str, mapping: Any) -> tuple[float, float]:
    """``12`` and ``[8, 15]`` both collapse to a (min, max) pair.

    A fixed length is a range whose ends agree, so the two spellings stop being
    different the moment they are parsed and nothing downstream branches on it.
    """
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            _fail(f"beat {name!r}: a duration range is [min, max], "
                  f"got {len(value)} value(s)", mapping, "duration")
        low = _number(value[0], f"beat {name!r} minimum duration", mapping, "duration")
        high = _number(value[1], f"beat {name!r} maximum duration", mapping, "duration")
    else:
        low = high = _number(value, f"beat {name!r} duration", mapping, "duration")

    if low <= 0 or high <= 0:
        _fail(f"beat {name!r}: duration must be greater than zero seconds — "
              "a beat with no length contributes nothing", mapping, "duration")
    if low > high:
        _fail(f"beat {name!r}: duration range runs {low:g}..{high:g} — "
              "the minimum must not exceed the maximum", mapping, "duration")
    return low, high


def _repeat(value: Any, name: str, mapping: Any) -> int:
    """How many clips this beat contributes. Whole numbers only: half a clip
    is not a thing the cutter can produce, and rounding one silently would
    change the length of the film."""
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(f"beat {name!r}: repeat must be a whole number of clips, "
              f"got {_describe(value)}", mapping, "repeat")
    if value < 1:
        _fail(f"beat {name!r}: repeat must be at least 1 — delete the beat "
              "instead of asking for none of it", mapping, "repeat")
    return int(value)


def _order(value: Any, name: str, mapping: Any) -> str:
    """One of :data:`_ORDERS`. A closed vocabulary, so a misspelling is caught
    here rather than becoming a silent fall back to the default."""
    text = _text(value, f"beat {name!r} order", mapping, "order").lower()
    if not text:
        return _DEFAULT_ORDER
    if text not in _ORDERS:
        _fail(f"beat {name!r}: unknown order {text!r} — expected one of "
              f"{', '.join(_ORDERS)}", mapping, "order")
    return text


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_beat(raw: Any, position: int, parent: Any) -> Beat:
    """One entry of the ``beats:`` list."""
    if not isinstance(raw, Mapping):
        _fail(f"beat {position} must be a mapping with at least a name and a "
              f"duration, got {_describe(raw)}", parent, "beats")

    _reject_unknown(raw, _BEAT_KEYS, "beat")

    name = _text(raw.get("name"), f"beat {position} name", raw, "name")
    if not name:
        _fail(f"beat {position} has no name — every beat needs one so the "
              "finished cut can be reported against the script", raw)

    if "duration" not in raw:
        _fail(f"beat {name!r} has no duration — say how long it should run, "
              "e.g. 'duration: 8' or 'duration: [6, 10]'", raw)
    low, high = _durations(raw.get("duration"), name, raw)

    repeat = _repeat(raw.get("repeat", 1), name, raw)

    match = raw.get("match")
    if match is None:
        match = {}
    if not isinstance(match, Mapping):
        _fail(f"beat {name!r}: match must be a mapping of "
              f"{'/'.join(_MATCH_KEYS)} lists, got {_describe(match)}",
              raw, "match")
    _reject_unknown(match, _MATCH_KEYS, "match")

    return Beat(
        name=name,
        min_duration=low,
        max_duration=high,
        repeat=repeat,
        objects=_term_list(match.get("objects"), "match.objects", match, "objects"),
        actions=_term_list(match.get("actions"), "match.actions", match, "actions"),
        keywords=_term_list(match.get("keywords"), "match.keywords", match, "keywords"),
        sources=_term_list(raw.get("sources"), "sources", raw, "sources"),
        order=_order(raw.get("order", _DEFAULT_ORDER), name, raw),
    )


def parse_script(text_or_dict: Any) -> Script:
    """Read a script from YAML text or an already-loaded mapping.

    Raises :class:`ScriptError` for anything that cannot be executed as
    written, with the offending line when the source was text. Nothing is
    coerced into working: see the module docstring for why silence is the one
    failure mode this format refuses.
    """
    if isinstance(text_or_dict, os.PathLike):
        raise ScriptError(
            "parse_script() takes script text or a mapping; "
            "use load_script() for a path"
        )

    if isinstance(text_or_dict, Mapping):
        data: Any = text_or_dict
    elif isinstance(text_or_dict, (str, bytes)):
        import yaml
        try:
            data = yaml.load(text_or_dict, Loader=_loader())
        except yaml.YAMLError as exc:
            mark = getattr(exc, "problem_mark", None)
            where = (f" (line {mark.line + 1}, column {mark.column + 1})"
                     if mark is not None else "")
            problem = getattr(exc, "problem", None) or str(exc)
            raise ScriptError(f"script is not valid YAML{where}: {problem}") from exc
    else:
        raise ScriptError(
            f"cannot read a script from {type(text_or_dict).__name__}")

    if data is None:
        raise ScriptError("script is empty — start from example_script()")
    if not isinstance(data, Mapping):
        raise ScriptError(
            f"a script is a mapping with a 'beats:' list, got {_describe(data)}")

    _reject_unknown(data, _TOP_KEYS, "top level")

    beats_raw = data.get("beats")
    if beats_raw is None:
        _fail("a script needs a 'beats:' list — without beats there is "
              "nothing to cut", data, "beats")
    if not isinstance(beats_raw, (list, tuple)) or isinstance(beats_raw, (str, bytes)):
        _fail(f"'beats' must be a list of beats, got {_describe(beats_raw)}",
              data, "beats")
    if not beats_raw:
        _fail("a script needs at least one beat", data, "beats")

    beats = [_parse_beat(raw, i, data) for i, raw in enumerate(beats_raw, start=1)]

    snap = data.get("snap_to_beat", False)
    if snap is None:
        snap = False
    if not isinstance(snap, bool):
        _fail(f"snap_to_beat must be true or false, got {_describe(snap)}",
              data, "snap_to_beat")

    raw_total = data.get("total_duration")
    total = (0.0 if raw_total is None
             else _number(raw_total, "total_duration", data, "total_duration"))
    if total < 0:
        _fail("total_duration must be a positive number of seconds",
              data, "total_duration")

    return Script(
        title=_text(data.get("title"), "title", data, "title") or _DEFAULT_TITLE,
        beats=beats,
        music=_text(data.get("music"), "music", data, "music"),
        snap_to_beat=bool(snap),
        total_duration=total,
    )


def load_script(path: str, log_fn=print) -> Script:
    """Read and parse the script at ``path``.

    Unreadable files raise :class:`ScriptError` rather than OSError so a caller
    running "load this script and cut it" has one exception type to guard, and
    the message is prefixed with the file name — several scripts are usually
    open at once and "unknown beat key" alone does not say which one.

    "Unreadable" includes the wrong encoding, which is why the decode error is
    caught alongside the OS one. Notepad's "Unicode" option writes UTF-16 and a
    Polish or Czech title typed in a legacy editor arrives as cp1250; both raise
    UnicodeDecodeError, which is a ValueError rather than an OSError and would
    otherwise leave here as the one failure that breaks the guarantee above —
    the wrong type, and the only load failure whose message never names the file.

    Warnings from :func:`validate_script` are logged here because this is the
    moment the user just handed the file over; they never stop the load.
    """
    name = os.path.basename(str(path))
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as exc:
        raise ScriptError(f"cannot read script {name}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ScriptError(
            f"cannot read script {name}: it is not UTF-8 text "
            f"({exc.reason}, byte {exc.start}) — re-save it as UTF-8"
        ) from exc

    try:
        script = parse_script(text)
    except ScriptError as exc:
        raise ScriptError(f"{name}: {exc}") from exc

    log_fn(f"📜 Script {script.title!r}: {len(script.beats)} beat(s), "
           f"{script.clip_count} clip(s), target {script.target_duration:.0f}s")
    for warning in validate_script(script):
        log_fn(f"⚠️ {warning}")
    return script


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

class _FlowList(list):
    """A list to be written on one line.

    ``duration:`` and the term lists are short and are read as a unit, so they
    belong on one line; the beats are not and do not. PyYAML's global
    ``default_flow_style`` cannot express that — ``False`` splits every range
    across three lines, ``None`` collapses a whole beat into ``{name: A,
    duration: 8}`` — so the choice is made per list instead, here.
    """


_DUMPER: Any = None


def _dumper():
    """The SafeDumper subclass that knows about :class:`_FlowList`.

    Cached, and built on first use for the same reason as the loader: yaml is
    a deferred import.
    """
    global _DUMPER
    if _DUMPER is not None:
        return _DUMPER

    import yaml

    class _ScriptDumper(yaml.SafeDumper):
        pass

    _ScriptDumper.add_representer(
        _FlowList,
        lambda dumper, data: dumper.represent_sequence(
            "tag:yaml.org,2002:seq", data, flow_style=True),
    )
    _DUMPER = _ScriptDumper
    return _DUMPER


def _plain(value: float) -> Any:
    """Whole seconds as ``12``, not ``12.0``. The file is read by people.

    Only the spelling changes — the value does not. Rounding a fraction here to
    keep the line short would break the loop this format exists for: save, edit
    one beat, load, and the other beats have quietly moved. Below half a
    millisecond it is worse than lossy, because the value rounds to ``0.0`` and
    the parser then refuses a file this function just wrote. Python's repr is
    already the shortest text that reads back as the same float, so a fraction
    written in full costs a few characters and nothing else.

    A non-finite length cannot be written at all: ``.inf`` and ``.nan`` are what
    the parser rejects on the way back in, so emitting them would produce a file
    the tool cannot open. It is reported as a :class:`ScriptError` because that
    is the exception :func:`save_script`'s callers guard.
    """
    number = float(value)
    if not math.isfinite(number):
        raise ScriptError(
            f"cannot save a duration of {number:g} — a script holds finite "
            "seconds, and this file would not load again")
    return int(number) if number.is_integer() else number


def _to_dict(script: Script) -> dict:
    """The script as plain data, defaults omitted.

    A saved script has to look like one a person would write. Restating
    ``repeat: 1`` and ``order: best_first`` on every beat buries the two lines
    that actually differ, and makes a diff between two versions of a film
    unreadable — so only fields carrying information are written.
    """
    data: dict[str, Any] = {"title": script.title}
    if script.music:
        data["music"] = script.music
    if script.snap_to_beat:
        data["snap_to_beat"] = True
    if script.total_duration > 0:
        data["total_duration"] = _plain(script.total_duration)

    beats: list[dict[str, Any]] = []
    for beat in script.beats:
        item: dict[str, Any] = {"name": beat.name}
        if beat.min_duration == beat.max_duration:
            item["duration"] = _plain(beat.min_duration)
        else:
            item["duration"] = _FlowList(
                [_plain(beat.min_duration), _plain(beat.max_duration)])
        if beat.repeat != 1:
            item["repeat"] = int(beat.repeat)
        match = {key: _FlowList(values) for key, values in (
            ("objects", beat.objects),
            ("actions", beat.actions),
            ("keywords", beat.keywords),
        ) if values}
        if match:
            item["match"] = match
        if beat.sources:
            item["sources"] = _FlowList(beat.sources)
        if beat.order != _DEFAULT_ORDER:
            item["order"] = beat.order
        beats.append(item)

    data["beats"] = beats
    return data


def save_script(script: Script, path: str) -> str:
    """Write ``script`` to ``path`` as YAML and return the path.

    The output is meant to be edited by hand afterwards — that is the whole
    loop this format exists for — so the beats stay a block list while ranges
    and term lists stay on one line each (see :class:`_FlowList`), and every
    duration is written at full precision so loading the file back returns the
    script that was saved.

    A script assembled in code rather than parsed can hold a length no file can
    express; that raises :class:`ScriptError` here (see :func:`_plain`) rather
    than writing a script the parser would refuse.
    """
    import yaml

    directory = os.path.dirname(str(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    text = yaml.dump(_to_dict(script), Dumper=_dumper(), sort_keys=False,
                     allow_unicode=True, default_flow_style=False)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return str(path)


# ---------------------------------------------------------------------------
# Advice and compilation
# ---------------------------------------------------------------------------

def validate_script(script: Script) -> list[str]:
    """Everything legal about a script that is probably not what was meant.

    Returned rather than raised, and returned as sentences: these are things a
    user might have done on purpose. A beat with no match terms is a perfectly
    good way to say "the best of whatever is here", and the day it stops a
    render is the day someone edits the script to shut the tool up.

    The music check is the only line here that touches the disk — a path typed
    weeks before the file was copied across is common enough to be worth it.
    """
    warnings: list[str] = []

    seen: dict[str, str] = {}
    for beat in script.beats:
        key = beat.name.casefold()
        if key in seen:
            warnings.append(
                f"two beats are called {beat.name!r} — the cut report cannot "
                "tell them apart")
        seen[key] = beat.name
        if not beat.has_match_terms and not beat.sources:
            warnings.append(
                f"beat {beat.name!r} matches nothing configured — it will be "
                "filled from the overall score")

    if script.snap_to_beat and not script.music:
        warnings.append(
            "snap_to_beat is on but no music is set — there are no beats to "
            "snap to")

    if script.music and not os.path.exists(script.music):
        warnings.append(f"music file not found: {script.music}")

    if script.total_duration > 0:
        floor = sum(b.min_duration * b.repeat for b in script.beats)
        ceiling = sum(b.max_duration * b.repeat for b in script.beats)
        if floor > script.total_duration:
            warnings.append(
                f"the beats ask for at least {floor:.0f}s but total_duration "
                f"is {script.total_duration:.0f}s — some beats will be cut short")
        elif ceiling < script.total_duration:
            warnings.append(
                f"the beats supply at most {ceiling:.0f}s but total_duration "
                f"is {script.total_duration:.0f}s — the film will come up short")

    return warnings


def compile_directives(script: Script) -> list[CutDirective]:
    """Flatten the script into one selection request per clip, in film order.

    This is the bridge to the engine: after this call nothing needs to know
    what a repeat is, what YAML is, or that a script existed at all.

    The term lists are copied per directive. A consumer that filters one
    clip's terms in place — narrowing to the objects a particular source file
    actually contains, say — must not reach back and change its siblings'.
    """
    directives: list[CutDirective] = []
    for beat in script.beats:
        for index in range(max(0, int(beat.repeat))):
            directives.append(CutDirective(
                beat_name=beat.name,
                index=index,
                min_duration=beat.min_duration,
                max_duration=beat.max_duration,
                objects=list(beat.objects),
                actions=list(beat.actions),
                keywords=list(beat.keywords),
                sources=list(beat.sources),
                order=beat.order,
            ))
    return directives


def example_script() -> str:
    """A commented template to start a script from.

    Every match list is empty on purpose. The terms that matter are the user's
    own — whatever their detection settings are configured to find — and a
    starter list of "good" ones would make this repo hold an opinion about
    content, which it does not have and must not ship. The beat names are
    ordinary filming vocabulary describing shape, not subject.
    """
    return """\
# A script says what the film should contain, beat by beat, and the cutter
# honours it. Edit, run, compare, keep the better file.
#
# Required: a title, and beats that each have a name and a duration.
# Everything else below is optional and can be deleted.

title: Untitled
# music: D:\\music\\track.mp3    # a music bed for the finished film
# snap_to_beat: true            # land the cuts on that music's beats
# total_duration: 180           # overall target in seconds. Left out, the
#                               # target is the sum of the beats below.

beats:
  - name: Establishing
    duration: 12                # exactly 12 seconds
    order: chronological        # chronological | best_first (the default)
    match:
      # Your own terms — whatever your detection settings look for. Empty
      # lists mean "anything", and the beat is filled by score alone.
      objects: []
      actions: []
      keywords: []

  - name: Action
    repeat: 3                   # this beat contributes 3 separate clips
    duration: [6, 10]           # each one between 6 and 10 seconds
    match:
      objects: []
      actions: []

  - name: Calm
    duration: [8, 14]
    sources: []                 # only cut from these files, e.g.
                                # ["GX010123.MP4"]. Empty means all of them.
"""
