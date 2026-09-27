"""Decide the obvious samples without a review, and keep checking that it is right.

Reviewing every sample is what made teaching slow. Most samples are not in
doubt: once a class has a handful of checked examples, a sample that looks as
much like them as they look like each other, and like nothing else, is that
class. Auto-accept decides those, and review is left with the doubtful ones.

Three things keep it honest, because a label nobody looked at is only as good
as the rule that made it:

* **It starts late.** A class is auto-accepted only once it has
  ``auto_min_checked`` samples a person accepted (examples count). Before that,
  its prototype is the class *name*, and CLIP's reading of a name is not
  something to train on unchecked.
* **It is spot-checked.** Every review sheet carries some auto-accepted
  samples (``review.pick_batch``). What auto said is kept on the sample
  (``auto_label``), so each check is scored: agreed or overturned. If more
  than ``auto_max_error`` of a class's checks are overturned, auto-accept stops
  for that class, and everything it accepted there unchecked goes back to
  the review queue.
* **It never judges the model.** The held-out set that scores every round is
  drawn only from samples a person decided (``build.assign_splits``), and
  class prototypes are built only from those (``sort.build_prototypes``), so
  an error cannot teach the sorter to repeat itself.

Negatives are auto-decided the same way: once a person has marked enough
"none of these", a sample below the footage's ordinary level for every class
is one more.
"""
from __future__ import annotations

import math

from modules.teach.project import (
    ACCEPTED, AUTO, NEGATIVE, NONE, PENDING, Project,
)

# A class needs this many spot checks before its error rate is trusted to
# switch it off; fewer, and one unlucky check would.
MIN_CHECKS_TO_JUDGE = 5
# Share of a class's auto-accepted samples that must be spot-checked before a
# round is trained on them, and never fewer than this many.
AUDIT_SHARE = 0.1
MIN_AUDITS = 3


def checked(project: Project, name: str) -> int:
    """Samples of ``name`` a person decided (examples included)."""
    if name == NONE:
        return sum(1 for s in project.samples if s.verdict == NEGATIVE and s.is_human)
    return sum(1 for s in project.accepted(name) if s.is_human)


def spot_checks(project: Project, name: str) -> tuple:
    """``(checked, overturned)`` among samples auto-accepted as ``name``."""
    audited = [s for s in project.samples if s.auto_label == name and s.is_human]
    if name == NONE:
        wrong = [s for s in audited if s.verdict != NEGATIVE]
    else:
        wrong = [s for s in audited if s.verdict != ACCEPTED or s.label != name]
    return len(audited), len(wrong)


def audits_needed(project: Project, name: str) -> int:
    """How many more spot checks ``name`` needs before training on it."""
    auto = sum(1 for s in project.samples if s.auto_label == name)
    if not auto:
        return 0
    done, _ = spot_checks(project, name)
    want = min(auto, max(MIN_AUDITS, math.ceil(AUDIT_SHARE * auto)))
    return max(0, want - done)


def state(project: Project, name: str) -> dict:
    """Is auto-accept on for ``name``, and why not if it is off."""
    settings = project.settings
    done, wrong = spot_checks(project, name)
    rate = (wrong / done) if done else 0.0
    if not settings.auto_accept:
        return {"on": False, "why": "auto_accept is off", "checks": done, "overturned": wrong}
    if done >= MIN_CHECKS_TO_JUDGE and rate > settings.auto_max_error:
        return {"on": False, "why": f"spot checks overturned {wrong} of {done}",
                "checks": done, "overturned": wrong, "tripped": True}
    have = checked(project, name)
    if have < settings.auto_min_checked:
        return {"on": False, "why": f"{have} of {settings.auto_min_checked} checked "
                                    "samples so far", "checks": done, "overturned": wrong}
    return {"on": True, "why": "", "checks": done, "overturned": wrong}


def apply(project: Project) -> dict:
    """Auto-decide what is obvious; take back what a tripped class decided."""
    settings = project.settings
    names = project.class_names()
    states = {n: state(project, n) for n in names + [NONE]}

    reverted = 0
    for sample in project.samples:
        if not sample.is_auto:
            continue
        key = sample.label if sample.verdict == ACCEPTED else NONE
        if states.get(key, {}).get("tripped"):
            sample.verdict, sample.label, sample.decided_by = PENDING, "", ""
            sample.auto_label = ""          # nobody checked it; nothing to score
            reverted += 1

    decided = {}
    for sample in project.samples:
        if sample.verdict != PENDING or not sample.scores:
            continue
        guess = sample.proposed
        # The last round's model disagreeing is exactly what review is for.
        if sample.model_proposed and sample.model_proposed != guess:
            continue
        if guess in names and states[guess]["on"]:
            if (sample.scores.get(guess, 0.0) >= settings.auto_gate
                    and sample.margin >= settings.auto_margin):
                project.decide(sample, ACCEPTED, guess, by=AUTO)
                decided[guess] = decided.get(guess, 0) + 1
        elif guess == NONE and states[NONE]["on"]:
            # Below this footage's ordinary level for every class.
            if max(sample.scores.values(), default=1.0) <= 0.0:
                project.decide(sample, NEGATIVE, by=AUTO)
                decided[NONE] = decided.get(NONE, 0) + 1

    project.save()
    return {"accepted": decided, "reverted": reverted,
            "classes": {n: {k: v for k, v in st.items() if k != "tripped"}
                        for n, st in states.items()}}
