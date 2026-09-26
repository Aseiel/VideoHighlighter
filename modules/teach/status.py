"""Where a project stands, and the one thing to do next.

The whole pipeline is steps with files between them, so at any moment exactly
one step is the useful next one. ``status`` works it out and hands back the
command for it. A person reads it as a to-do line; an agent runs the command,
calls ``status`` again, and repeats: that loop is all an LLM needs to take a
project from "I want to find X" to a trained model, stopping only where a
judgement is needed (naming, reviewing).

Each step is described with ``who``: ``"auto"`` steps can run unattended;
``"judge"`` steps need someone to look (a person, or an agent that can see
the contact sheet), because every safeguard downstream rests on it.
"""
from __future__ import annotations

import shlex

from modules.teach.project import (
    MIN_TO_TRAIN, OBJECTS, PENDING, Project,
)


def _cmd(project: Project, *args) -> str:
    return " ".join(["python -m modules.teach", "--project",
                     shlex.quote(project.root)] + [shlex.quote(str(a)) for a in args])


def _step(project, who: str, why: str, *args) -> dict:
    return {"who": who, "why": why, "command": _cmd(project, *args) if args else "",
            "args": list(args)}


def next_step(project: Project) -> dict:
    counts = project.counts()
    names = project.class_names()
    pending = [s for s in project.samples if s.verdict == PENDING]
    scored = [s for s in project.samples if s.scores]

    if not names:
        return _step(project, "judge", "Say what to find: add a class (name + a few "
                     "words describing it). `check-name` first if unsure what to call it.",
                     "add-class", "<name>", "--description", "<what it looks like>")
    if not project.sources:
        return _step(project, "judge", "Give it footage: a few videos where the classes "
                     "appear. A URL is downloaded; a file is used in place.",
                     "add-video", "<path or url>")
    if any(not s.cut for s in project.sources):
        return _step(project, "auto", "Cut the new footage into samples.", "cut")
    if (project.task == "actions" and project.settings.focus
            and any(not s.focus_tried for s in project.samples)):
        return _step(project, "auto", "Crop samples to the people in them.", "focus")
    unsorted = [s for s in project.samples if not s.scores]
    if not scored or unsorted:
        return _step(project, "auto", "Score every sample against every class.", "sort")

    no_examples = [n for n in names if counts[n]["accepted"] == 0 and not
                   project.get_class(n).examples]
    if no_examples and len(pending) and all(counts[n]["accepted"] == 0 for n in names):
        # Words alone sort weakly. One reviewed sheet turns into examples, and
        # every sort after it uses them.
        return _step(project, "judge", "Check the first guesses: accepted samples become "
                     "examples and every later sort sharpens. Or add example clips you "
                     "already have with `add-example`.", "review")

    short = [n for n in names if counts[n]["accepted"] < counts[n]["target"]]
    ready = [n for n in names if counts[n]["accepted"] >= MIN_TO_TRAIN]
    if short and pending:
        worst = min(short, key=lambda n: counts[n]["accepted"] / max(counts[n]["target"], 1))
        c = counts[worst]
        if project.rounds and len(ready) == len(names):
            pass    # enough to train again; reviewing more is optional below
        else:
            return _step(project, "judge",
                         f"Review guesses: {worst!r} has {c['accepted']} of "
                         f"{c['target']}. Re-run `sort` after a few sheets so accepted "
                         "samples sharpen the next guesses.", "review")

    if project.task == OBJECTS:
        from modules.teach.boxes import labeler_worklist, store
        labels = store(project)
        if labels.pending():
            return _step(project, "judge", f"Check {len(labels.pending())} proposed boxes.",
                         "boxes", "review")
        todo = labeler_worklist(project)
        if any(not s.boxes_tried for s in project.accepted()):
            return _step(project, "auto", "Propose boxes on accepted samples.",
                         "boxes", "propose")
        if len(todo) > len(project.accepted()) // 2:
            return _step(project, "judge", f"{len(todo)} accepted samples have no good box. "
                         "Draw them in tools/labeler.py, then import the exports.",
                         "boxes", "worklist")

    if len(ready) < len(names):
        missing = [n for n in names if n not in ready]
        return _step(project, "judge", f"Need at least {MIN_TO_TRAIN} accepted samples of "
                     f"{', '.join(repr(n) for n in missing)}. Add footage where they "
                     "appear more, or review further.", "add-video", "<path or url>")

    from modules.teach import autolabel
    audits = {n: autolabel.audits_needed(project, n) for n in names + ["_none"]}
    owed = {n: k for n, k in audits.items() if k}
    if owed:
        listed = ", ".join(f"{k} of {n!r}" for n, k in owed.items())
        return _step(project, "judge", f"Spot-check what was auto-accepted before training "
                     f"on it ({listed}); review sheets include them.", "review")

    from modules.teach.build import built_signature, dataset_signature

    signature = dataset_signature(project)
    if built_signature(project) != signature:
        return _step(project, "auto", "Build the dataset from what was accepted.", "build")
    if not any(r.get("dataset") == signature for r in project.rounds):
        return _step(project, "auto", "Train a round (GPU minutes to hours; runs "
                     "unattended, and installs the model only if it beats the last).",
                     "train")
    return _step(project, "judge", "Trained on everything accepted. To improve it: add a "
                 "video it has not seen, then cut, sort and review; its mistakes there "
                 "are the most useful labels there are.", "add-video", "<path or url>")


def report(project: Project) -> dict:
    counts = project.counts()
    verdicts = {}
    for sample in project.samples:
        verdicts[sample.verdict] = verdicts.get(sample.verdict, 0) + 1
    last = project.rounds[-1] if project.rounds else None
    return {
        "project": project.root,
        "name": project.name,
        "task": project.task,
        "classes": counts,
        "sources": len(project.sources),
        "samples": len(project.samples),
        "verdicts": verdicts,
        "rounds": len(project.rounds),
        "last_round": ({k: last[k] for k in ("round", "metrics", "installed",
                                             "better_than_installed") if k in last}
                       if last else None),
        "next": next_step(project),
    }
