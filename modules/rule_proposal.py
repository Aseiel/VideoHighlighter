"""Turn "nothing was watching for that" into a rule that would watch for it.

:mod:`modules.vocabulary_gap` can say that a stretch talks about something no
class covers. The next question is always the same -- *so what would I add?* --
and answering it takes exactly the thing the rest of this report refuses to do:
deciding what a word means and which detections would stand for it. That is a
judgement, so it is a model's job, and it is the user's decision whether to keep
it.

Hence the shape here. A model proposes; this module *validates*; a human
approves; only then is anything written. Each of those three is load-bearing:

**Validation is not politeness.** The failure that costs the user most is a rule
naming a class the detector never emits. It parses, it loads, it fires on
nothing, and the only symptom is a re-run -- minutes to hours -- that changes
nothing. So a proposal referencing an unobserved class is rejected here rather
than written and discovered later.

**Approval is required.** The composition rules are the user's own file, in the
user's own vocabulary, deliberately outside version control. Nothing in this
module writes without being called with a proposal a person has seen; there is
no "apply automatically" path to turn on later, and adding one would be a change
of kind rather than of degree.

**What was proposed is recorded.** :func:`apply` leaves a note beside the video
saying which rule was added and which spoken claim it was meant to test. The
next run's report picks that note up, so the run that finally has the signal can
say whether it fired -- which is the only reason any of this was worth doing.
The alternative is a user who adds a rule, waits, and then has to remember what
question they were asking.

Nothing in this file knows any subject matter. It is handed a list of class
names the detector produced in this video, and every rule it will accept is
built from those.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional, Sequence

# What the model is allowed to answer with. Deliberately a schema and not prose:
# the reply is going into a config file, and a rule that has to be parsed out of
# a paragraph is one that will one day be parsed wrong.
PROPOSAL_SYSTEM_PROMPT = (
    "You design rules for a composition engine that labels moments in a video "
    "by how detected boxes are arranged.\n"
    "A rule fires when, for each of its conditions, the number of `source` "
    "boxes whose centre falls inside a `region` box is between min_count and "
    "max_count. All conditions must hold at once.\n"
    "Rules:\n"
    "1. Reply with one JSON object and nothing else. No prose, no code fence, "
    "no explanation outside the JSON.\n"
    "2. Use ONLY class names from the provided list, exactly as spelled. A rule "
    "naming any other class can never fire and is worthless.\n"
    "3. Shape: {\"name\": snake_case, \"label\": \"Short Title\", \"why\": "
    "\"one sentence on what firing would show\", \"rules\": [{\"source\": "
    "class, \"region\": class, \"min_count\": int, \"max_count\": int}]}\n"
    "4. Keep it to one or two conditions. A rule with more is a rule that never "
    "fires.\n"
    "5. `name` must not be one already in use.\n"
    "6. If the claim cannot be expressed as an arrangement of the available "
    "classes, reply {\"name\": null, \"why\": \"<why not>\"} instead of "
    "inventing a class. That is a useful answer, not a failure."
)

# No upper limit, in the engine's own convention.
UNBOUNDED = 999

# A name has to survive being a YAML key and a Python identifier-ish token,
# because it becomes a tag in the report and a key in the record.
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,39}$")

# Where the note about what a rule was meant to test lives. Beside the video
# rather than in the rules file: the rules file is configuration the user edits
# by hand, and a tool writing bookkeeping into it is how hand-edits get lost.
CHECKS_SUFFIX = "_checks.json"


@dataclass
class Proposal:
    """One rule a model suggested, already checked against the real classes."""
    name: str
    label: str
    why: str
    rules: list
    claim: str = ""                   # the spoken line this is meant to test
    claim_at: Optional[float] = None  # where that line was said
    model: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    def as_event(self) -> dict:
        """The engine's own shape, ready to append to the rules file."""
        return {"name": self.name, "label": self.label,
                "rules": [dict(r) for r in self.rules],
                "window_secs": 0.75, "persist_secs": 0.5}

    def as_yaml(self) -> str:
        """What the user is shown before deciding. Their file's own style."""
        import yaml
        return yaml.safe_dump({"events": [self.as_event()]},
                              sort_keys=False, allow_unicode=True)


def existing_rules(rules_path: Optional[str]) -> list:
    """Event dicts already in the file, or ``[]`` if there is no file yet."""
    if not rules_path or not os.path.exists(rules_path):
        return []
    import yaml
    try:
        with open(rules_path, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
    except Exception as exc:                       # pragma: no cover - defensive
        print(f"⚠️ Could not read composition rules: {exc}")
        return []
    return list(loaded.get("events") or [])


def build_prompt(claim: str,
                 classes: Sequence[str],
                 existing: Sequence[Mapping] = (),
                 gaps: Sequence[Mapping] = ()) -> str:
    """What the model is given: the claim, the classes, and what already exists.

    The existing rules are included so the model can see the conventions of the
    file it is extending, and so it does not propose one that is already there
    under another name -- which is the most common wasted suggestion.
    """
    parts = ["## The claim to test",
             claim.strip() or "(none given)",
             "",
             "## Classes this detector actually produced in this video",
             ", ".join(sorted(classes)) or "(none)",
             ""]
    if gaps:
        parts += ["## Words said in this video that no class covers",
                  ", ".join(f"{g['word']} (×{g['count']})" for g in gaps[:8]),
                  ""]
    if existing:
        parts += ["## Rules already defined, which you must not duplicate"]
        for event in existing:
            conditions = "; ".join(
                f"{r.get('min_count', 1)}–"
                f"{'any' if int(r.get('max_count', UNBOUNDED)) >= UNBOUNDED else r.get('max_count')}"
                f" {r.get('source')} inside {r.get('region')}"
                for r in (event.get("rules") or []))
            parts.append(f"- {event.get('name')}: {conditions}")
        parts.append("")
    parts += ["## Task",
              "Propose one rule whose firing would be evidence for or against "
              "the claim. JSON only."]
    return "\n".join(parts)


def parse(reply: str,
          classes: Sequence[str],
          existing: Sequence[Mapping] = ()) -> Optional[Proposal]:
    """Validate a model's reply into a Proposal, or ``None`` with a reason logged.

    Every rejection here is a rule that would have cost a re-run to discover was
    useless, so the checks are strict on purpose and the reason is printed.
    """
    raw = _json_object(reply)
    if raw is None:
        print("⚠️ Rule proposal: no JSON object in the reply")
        return None
    if not raw.get("name"):
        # The model's own refusal. Passed back as a reason rather than an error:
        # "this cannot be expressed with these classes" is the correct answer to
        # many claims, and hiding it would send the user looking for a bug.
        print(f"ℹ️ Rule proposal declined: {raw.get('why') or 'no reason given'}")
        return None

    name = str(raw.get("name") or "").strip().lower()
    if not NAME_RE.match(name):
        print(f"⚠️ Rule proposal: '{name}' is not a usable rule name")
        return None
    if name in {str(e.get("name") or "").lower() for e in (existing or [])}:
        print(f"⚠️ Rule proposal: '{name}' already exists")
        return None

    known = {str(c) for c in classes}
    conditions = []
    for rule in (raw.get("rules") or []):
        source = str(rule.get("source") or "").strip()
        region = str(rule.get("region") or "").strip()
        # The check this module exists for. A rule naming a class the detector
        # never emitted parses, loads, and fires on nothing.
        if source not in known or region not in known:
            print(f"⚠️ Rule proposal names a class this video has no "
                  f"detections for: {source!r}/{region!r}")
            return None
        try:
            low = max(1, int(rule.get("min_count", 1)))
            high = int(rule.get("max_count", UNBOUNDED))
        except (TypeError, ValueError):
            print("⚠️ Rule proposal: counts are not whole numbers")
            return None
        if high < low:
            print("⚠️ Rule proposal: max_count below min_count")
            return None
        conditions.append({"source": source, "region": region,
                           "min_count": low, "max_count": high})
    if not conditions:
        print("⚠️ Rule proposal: no conditions")
        return None

    return Proposal(name=name,
                    label=str(raw.get("label") or name.replace("_", " ").title()),
                    why=str(raw.get("why") or "").strip(),
                    rules=conditions)


def _json_object(reply: str) -> Optional[dict]:
    """The first JSON object in a reply, fences and chatter tolerated.

    Models are asked for bare JSON and mostly comply; the ones that do not wrap
    it in a fence or a sentence, and throwing that away would fail a proposal
    that was actually correct.
    """
    text = str(reply or "").strip()
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth, index = 0, start
        while index < len(text):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        found = json.loads(text[start:index + 1])
                    except ValueError:
                        break
                    if isinstance(found, dict):
                        return found
                    break
            index += 1
        start = text.find("{", start + 1)
    return None


def propose(claim: str,
            classes: Sequence[str],
            *,
            llm,
            existing: Sequence[Mapping] = (),
            gaps: Sequence[Mapping] = (),
            claim_at: Optional[float] = None,
            model_name: str = "",
            max_tokens: int = 320) -> Optional[Proposal]:
    """Ask a model for a rule that would test ``claim``. ``None`` if it cannot."""
    if llm is None or not classes:
        return None
    from modules.advisor import _generate

    prompt = build_prompt(claim, classes, existing, gaps)
    try:
        reply = _generate(llm, prompt, PROPOSAL_SYSTEM_PROMPT, max_tokens, 0.2)
    except Exception as exc:
        print(f"⚠️ Rule proposal failed: {exc}")
        return None
    proposal = parse(reply, classes, existing)
    if proposal is not None:
        proposal.claim = str(claim or "").strip()
        proposal.claim_at = claim_at
        proposal.model = str(model_name or "")
    return proposal


# ---------------------------------------------------------------------------
# Writing, which only ever happens on a proposal a person has seen
# ---------------------------------------------------------------------------
def apply(rules_path: str, proposal: Proposal,
          video_path: Optional[str] = None) -> str:
    """Append the rule to the user's file and note what it was meant to test.

    The existing file is copied to ``.bak`` first and rewritten from parsed YAML
    rather than appended to as text: a half-written rule is a rules file that no
    longer loads, and the failure would surface on the next run as every
    composed event silently vanishing.

    Returns the path written. Raises rather than half-succeeding.
    """
    import yaml

    existing = existing_rules(rules_path)
    if any(str(e.get("name") or "").lower() == proposal.name for e in existing):
        raise ValueError(f"'{proposal.name}' is already in {rules_path}")

    if os.path.exists(rules_path):
        shutil.copy2(rules_path, rules_path + ".bak")
    events = existing + [proposal.as_event()]
    tmp = rules_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        yaml.safe_dump({"events": events}, fh, sort_keys=False,
                       allow_unicode=True)
    os.replace(tmp, rules_path)

    if video_path:
        record_check(video_path, proposal)
    return rules_path


def checks_path_for(video_path: str) -> str:
    """Where the note beside a video lives."""
    base, _ext = os.path.splitext(str(video_path))
    return base + CHECKS_SUFFIX


def record_check(video_path: str, proposal: Proposal) -> str:
    """Remember that this rule was added to test this claim.

    Kept beside the video rather than in the report, because the report is
    rewritten by the next run and this has to survive it -- surviving it is the
    entire point.
    """
    path = checks_path_for(video_path)
    checks = load_checks(video_path)
    checks = [c for c in checks if c.get("rule") != proposal.name]
    checks.append({
        "rule": proposal.name,
        "label": proposal.label,
        "claim": proposal.claim,
        "claim_at": proposal.claim_at,
        "why": proposal.why,
        "model": proposal.model,
        "added": _dt.datetime.now().isoformat(timespec="seconds"),
    })
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(checks, fh, indent=1, ensure_ascii=False)
    return path


def load_checks(video_path: str) -> list:
    """Notes left by earlier runs. ``[]`` when there are none."""
    path = checks_path_for(video_path)
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            loaded = json.load(fh)
    except Exception as exc:                       # pragma: no cover - defensive
        print(f"⚠️ Could not read pending checks: {exc}")
        return []
    return list(loaded) if isinstance(loaded, list) else []


def settle_checks(checks: Sequence[Mapping],
                  object_detections: Optional[Mapping] = None,
                  composed_event_names: Sequence[str] = ()) -> list:
    """Say, for each remembered check, whether its rule fired this time.

    Three outcomes and they are not the same. *Fired* is evidence about the
    claim. *Never fired* is evidence too -- weaker, because a rule can be right
    and still miss. *Not in the rules* means the rule is not being evaluated at
    all, which is not evidence about anything and must never be shown as if it
    were.
    """
    known = {str(n) for n in (composed_event_names or ())}
    seconds: dict = {}
    for sec, names in (object_detections or {}).items():
        for name in (names or []):
            seconds.setdefault(str(name), []).append(int(sec))

    out = []
    for check in (checks or []):
        rule = str(check.get("rule") or "")
        row = dict(check)
        hits = sorted(seconds.get(rule) or [])
        if rule not in known:
            row["state"] = "not_in_rules"
        elif hits:
            row["state"] = "fired"
            row["seconds"] = len(hits)
            row["first"] = hits[0]
            row["last"] = hits[-1]
        else:
            row["state"] = "never_fired"
        out.append(row)
    return out
