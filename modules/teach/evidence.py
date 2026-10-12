"""Objects found in each sample, and rules that name a class from them.

An image encoder sees a sample whole. Two classes that differ in how many of
something there are, or in what touches what, look nearly alike to it; a
detector counts. ``detect`` runs the dataset's detector over a few frames of
every sample and keeps what it found (``detections.json``). ``rules.yaml`` in
the dataset folder, in the composition engine's format, says which
combination is which class. An event's class is its ``class:`` key, else its
``name`` or ``label`` when that is one of the classes (``nothing`` is "none
of these"), so a rules file written for the timeline can be reused as it is.
When two classes' rules both hold, the one with the higher ``priority:``
wins (by default, the one with more conditions: the more specific claim).

A rule's guess is a proposal like the sort's, and is measured the same way:
review records it as a guess ``from: rule``, Health shows how often it was
right, and only a rule people have checked often enough, and found right,
decides samples alone (``rule_auto_min_checked`` / ``rule_auto_precision``),
under the same spot checks as every auto-accept.

The detector is the user's own: any model ``detection_backend.create_detector``
loads, named in ``detector_model`` and ``detector_classes``. It is used where
it is, never copied or bundled, and nothing is trained on its weights: its
detections only propose labels a person confirms.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter
from typing import Callable, Optional

from modules.teach import embed as embed_mod
from modules.teach.project import ACCEPTED, AUTO, NEGATIVE, NONE, PENDING, Project

DETECTIONS_FILE = "detections.json"
RULES_FILE = "rules.yaml"
RULE = "rule"
# An event named this is the rule for "none of these".
NOTHING_EVENT = "nothing"

RULES_TEMPLATE = """\
# Rules for this dataset: which objects, in which combination, make a class.
#
# An event is the rule for the class in its `class:` key, or for the class
# its `name` or `label` matches (`nothing` is "none of these"). It fires on a
# sample when its conditions hold for at least `rule_min_share` of the sample's
# seconds (a dataset setting, 0.4 by default). When two classes' rules fire,
# the higher `priority:` wins; by default the one with more conditions. Its
# guess goes to review like any other, and Health shows how often it was right.
#
# A condition counts `source` objects that stand in a `relation` (inside,
# overlaps or touches) to a `region` object, and needs the count to fall in
# [min_count, max_count]. Conditions in one event must all hold together.
# Names are the detector's class names (detector_classes).
#
# events:
#   - name: <class>                 # one of the dataset's classes
#     class: <class>                # optional: when the name is something else
#     priority: 1                   # optional: wins over a lower one that also fires
#     min_share: 0.2                # optional: this rule's own rule_min_share
#     rules:
#       - {source: <object>, region: <object>, relation: touches, min_count: 1, max_count: 1}
#   - name: <another class>
#     rules:
#       - {source: <object>, region: <object>, relation: touches, min_count: 2}

events: []
"""


# --- the detector -----------------------------------------------------------------

def class_names(spec: str) -> list:
    """The detector's class names: a JSON list (or ``{"names": [...]}``), a text
    file with one per line, or the names themselves, comma-separated."""
    spec = (spec or "").strip()
    if spec and os.path.isfile(spec):
        with open(spec, "r", encoding="utf-8") as handle:
            text = handle.read()
        try:
            data = json.loads(text)
        except ValueError:
            return [line.strip() for line in text.splitlines() if line.strip()]
        if isinstance(data, dict):
            data = data.get("names") or data.get("classes") or []
            if isinstance(data, dict):            # {"0": "a", "1": "b"}
                data = [data[k] for k in sorted(data, key=lambda k: int(k))]
        return [str(n) for n in data]
    return [n.strip() for n in spec.split(",") if n.strip()]


def model_signature(path: str) -> str:
    """Which model the cached detections came from: a new file means new ones."""
    try:
        stat = os.stat(path)
    except OSError:
        return ""
    return f"{os.path.abspath(path)}|{stat.st_size}|{int(stat.st_mtime)}"


def make_detector(project: Project):
    """The dataset's detector, or ``None`` when none is set."""
    settings = project.settings
    if not settings.detector_model:
        return None
    if not os.path.exists(settings.detector_model):
        raise FileNotFoundError(f"detector_model {settings.detector_model!r} is not there")
    names = class_names(settings.detector_classes)
    if not names:
        raise ValueError("set detector_classes: the detector's class names (a labels "
                         "file, or names separated by commas)")
    from modules.vision.detection_backend import create_detector
    return create_detector(settings.detector_model, names, device="GPU")


# --- what it found ---------------------------------------------------------------

def load(project: Project) -> dict:
    try:
        with open(project.path(DETECTIONS_FILE), "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {"model": "", "samples": {}}
    data.setdefault("samples", {})
    return data


def _save(project: Project, data: dict) -> None:
    from modules.teach.project import _write_json
    _write_json(project.path(DETECTIONS_FILE), data)


def todo(project: Project) -> list:
    """Readable samples the current detector has not looked at yet."""
    if not project.settings.detector_model:
        return []
    data = load(project)
    if data.get("model") != model_signature(project.settings.detector_model):
        seen = set()
    else:
        seen = set(data["samples"])
    return [s for s in project.samples if not s.unreadable and s.id not in seen]


def detect_samples(project: Project, detector=None, *,
                   frame_reader: Optional[Callable] = None,
                   progress: Optional[Callable] = None) -> dict:
    """Run the detector over ``detector_frames`` frames of each sample not yet seen."""
    detector = detector or make_detector(project)
    if detector is None:
        raise ValueError("no detector set: set detector_model and detector_classes")
    read = frame_reader or embed_mod.read_frames
    data = load(project)
    signature = model_signature(project.settings.detector_model)
    if data.get("model") != signature:
        data = {"model": signature, "samples": {}}
    samples = todo(project)
    count = max(1, int(project.settings.detector_frames))
    found = Counter()
    for i, sample in enumerate(samples, 1):
        frames = read(sample.path, count)
        entries = []
        for k, frame in enumerate(frames):
            height, width = frame.shape[:2]
            dets = detector.detect(frame)
            entries.append({
                "timestamp": round((k + 0.5) * (sample.duration or 5.0) / len(frames), 3),
                "objects": [d.class_name for d in dets],
                "bboxes": [[d.x1 / width, d.y1 / height, (d.x2 - d.x1) / width,
                            (d.y2 - d.y1) / height] for d in dets],
                "confidences": [round(float(d.confidence), 4) for d in dets],
            })
            found.update(set(d.class_name for d in dets))
        data["samples"][sample.id] = entries
        if progress:
            progress(i, len(samples))
        if i % 50 == 0:
            _save(project, data)
    _save(project, data)
    return {"detected": len(samples), "frames_with": dict(found),
            "samples_total": len(data["samples"])}


# --- rules -------------------------------------------------------------------------

def rules_path(project: Project) -> str:
    return project.path(RULES_FILE)


def init_rules(project: Project) -> str:
    path = rules_path(project)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(RULES_TEMPLATE)
    return path


def _class_of(event: str, names: list) -> str:
    if event.strip().lower() == NOTHING_EVENT:
        return NONE
    from modules.teach.naming import normalize_name
    clean = normalize_name(event)
    return clean if clean in names else ""


def rule_events(project: Project, names=None) -> dict:
    """``{event name: (class or NONE, priority, min share or None)}`` for the
    rules file's events
    that are a class's rule. Signal-only events (no ``rules:``) are left out:
    a sample has no per-second signals to test them on."""
    import yaml
    path = rules_path(project)
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    names = project.class_names() if names is None else names
    out = {}
    for ev in raw.get("events") or []:
        if not ev.get("rules") or not ev.get("name"):
            continue
        wanted = ev.get("class")
        cls = (_class_of(str(wanted), names) if wanted
               else _class_of(str(ev["name"]), names)
               or _class_of(str(ev.get("label") or ""), names))
        if cls:
            share = ev.get("min_share")
            out[str(ev["name"])] = (cls, float(ev.get("priority", len(ev["rules"]))),
                                    None if share is None else float(share))
    return out


def rule_classes(project: Project) -> list:
    """The classes the rules file names in ``class:`` keys, existing or not."""
    import yaml
    path = rules_path(project)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    from modules.teach.naming import normalize_name
    return list(dict.fromkeys(
        normalize_name(str(ev["class"])) for ev in raw.get("events") or []
        if ev.get("rules") and ev.get("class") and ev.get("enabled", True)
        and str(ev["class"]).strip().lower() != NOTHING_EVENT))


def rule_guesses(project: Project) -> dict:
    """``{sample id: (class or NONE, share of its seconds)}`` for every sample
    a class's rule fired on. When several classes' rules fire, the highest
    priority wins; a tie between classes is no guess (review settles it)."""
    path = rules_path(project)
    if not os.path.exists(path):
        return {}
    from video_ai_editor.composition_engine import CompositionEngine
    engine = CompositionEngine(path)
    names = project.class_names()
    events = rule_events(project, names)
    if not events:
        return {}
    data = load(project)
    by_id = {s.id: s for s in project.samples}
    need = float(project.settings.rule_min_share)
    out = {}
    for sid, entries in data["samples"].items():
        sample = by_id.get(sid)
        if sample is None or not entries:
            continue
        seconds = max(1, math.ceil(sample.duration or 5.0))
        fired, _ = engine.run(entries, duration=float(sample.duration or 5.0))
        per_class, priority, needs = Counter(), {}, {}
        for sec, names_fired in fired.items():
            if 0 <= int(sec) < seconds:
                for event in set(names_fired):
                    if event in events:
                        cls, rank, own = events[event]
                        per_class[cls] += 1
                        priority[cls] = max(priority.get(cls, rank), rank)
                        needs[cls] = min(needs.get(cls, need), need if own is None else own)
        held = {c: n / seconds for c, n in per_class.items()
                if n / seconds >= needs.get(c, need)}
        if not held:
            continue
        top = max(priority[c] for c in held)
        best = [c for c in held if priority[c] == top]
        if len(best) == 1:
            out[sid] = (best[0], round(min(held[best[0]], 1.0), 3))
    return out


def rule_record(project: Project) -> dict:
    """Per class: how often a person agreed with what its rule guessed."""
    out = {}
    for s in project.samples:
        if s.guess_from != RULE or not s.is_human or not s.shown_guess:
            continue
        row = out.setdefault(s.shown_guess, {"checked": 0, "right": 0})
        row["checked"] += 1
        right = (s.verdict == NEGATIVE if s.shown_guess == NONE
                 else s.verdict == ACCEPTED and s.label == s.shown_guess)
        row["right"] += right
    return out


def trusted(project: Project, cls: str, record: Optional[dict] = None) -> bool:
    """Has this rule earned deciding alone: checked often enough, and right?"""
    row = (record if record is not None else rule_record(project)).get(cls)
    settings = project.settings
    return bool(row and row["checked"] >= settings.rule_auto_min_checked
                and row["right"] / row["checked"] >= settings.rule_auto_precision)


def apply_rules(project: Project) -> dict:
    """Put each rule's guess on the undecided samples it fired on, and let a
    trusted rule decide them. Called by ``sort`` after it has proposed."""
    guesses = rule_guesses(project)
    record = rule_record(project)
    names = set(project.class_names())
    proposed, decided, agreed = Counter(), Counter(), 0
    for sample in project.samples:
        sample.rule_guess = ""
        hit = guesses.get(sample.id)
        if hit is None:
            continue
        cls, _share = hit
        sample.rule_guess = cls
        if sample.verdict != PENDING:
            continue
        agreed += sample.proposed == cls
        sample.proposed, sample.proposed_by = cls, RULE
        # Not the encoder's confidence: auto-accept by scores must not take it.
        sample.margin = 0.0
        proposed[cls] += 1
        if (cls in names or cls == NONE) and trusted(project, cls, record):
            project.decide(sample, ACCEPTED if cls != NONE else NEGATIVE,
                           cls if cls != NONE else "", by=AUTO)
            decided[cls] += 1
    project.save()
    return {"proposed": dict(proposed), "agreed_with_sort": agreed,
            "decided": dict(decided), "record": record}
