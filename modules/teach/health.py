"""Is this dataset healthy enough to train on, and how well did the sorting do?

``status`` says what to do next; this says how good what is there already is.
Each rule comes from something that went wrong in practice
(``docs/training/negative``), and each check says what fixes it:

* every group was given a name, and nothing it put in a class went unseen;
* what auto-accept decided was spot-checked (``autolabel``);
* each class has enough samples (5 or the trainer leaves it out; under ~20
  it draws wrong guesses on new footage);
* each class comes from 3+ videos, or the model learns the scene with it
  and nothing can measure that (``classes-from-too-few-videos.md``);
* no one video is most of a class (``one-video-floods-dataset.md``);
* for actions, a "none of these" class, or the model can never say "none".

``sorting`` is the other half: of the samples a person decided, how often
the guess they were shown was right, by what made the guess (a group, the
sort, auto-accept) and by class, and what the wrong ones really were.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from modules.teach import autolabel
from modules.teach import group as group_mod
from modules.teach.project import (
    ACCEPTED, ACTIONS, NAMED, NEGATIVE, NONE, NOTHING, PENDING, REJECTED, UNSURE, Project,
)

OK, WARN, BAD = "ok", "warn", "bad"

# The trainer leaves out a class with fewer clips than this.
TRAINER_MIN = 5
# Under this many, a class drew many wrong guesses on new footage.
SOLID = 20
# A class is trusted only from hits in this many videos.
MIN_VIDEOS = 3
# One video holding more than this share of a class is the class.
MAX_ONE_VIDEO = 0.6
# Largest class over smallest: beyond this the small one is drowned out.
MAX_IMBALANCE = 5.0

READY = "ready"                    # train, and trust the held-out score
NEEDS_VIDEOS = "needs_videos"      # trains, but only on one or two videos' worth
NOT_READY = "not_ready"            # something has to be done first


def _check(name: str, level: str, detail: str, fix: str = "", cls: str = "") -> dict:
    return {"name": name, "level": level, "detail": detail, "fix": fix,
            **({"class": cls} if cls else {})}


def _videos(samples) -> Counter:
    return Counter(s.source for s in samples)


def class_rows(project: Project) -> dict:
    """Per class: samples, who decided them, videos, and what is still waiting."""
    rows = {}
    for spec in project.classes:
        accepted = project.accepted(spec.name)
        videos = _videos(accepted)
        rows[spec.name] = {
            "samples": len(accepted),
            "by_person": sum(1 for s in accepted if s.is_human),
            "by_auto": sum(1 for s in accepted if not s.is_human),
            "videos": len(videos),
            "one_video_share": (round(videos.most_common(1)[0][1] / len(accepted), 3)
                                if accepted else 0.0),
            "waiting": sum(1 for s in project.samples
                           if s.verdict == PENDING and s.proposed == spec.name),
            "spot_checks_owed": autolabel.audits_needed(project, spec.name),
            "target": spec.target,
        }
    negatives = [s for s in project.samples if s.verdict == NEGATIVE]
    rows[NONE] = {
        "samples": len(negatives),
        "by_person": sum(1 for s in negatives if s.is_human),
        "by_auto": sum(1 for s in negatives if not s.is_human),
        "videos": len(_videos(negatives)),
        "trained_as": project.settings.background_class if project.task == ACTIONS else "",
        "spot_checks_owed": autolabel.audits_needed(project, NONE),
    }
    return rows


def checks(project: Project) -> list:
    out = []
    readable = [s for s in project.samples if not s.unreadable]
    if not project.sources:
        return [_check("footage", BAD, "No videos yet.", "Add a video.")]
    if not readable:
        return [_check("footage", BAD, "No samples yet.", "Cut the videos into samples.")]

    # --- groups ---------------------------------------------------------------
    groups = group_mod.summary(project)
    if groups["total"]:
        if groups["open"]:
            out.append(_check("groups named", BAD,
                              f"{groups['open']} of {groups['total']} groups have no name yet.",
                              "Name each group, or mark it 'nothing' or 'mixed'."))
        else:
            out.append(_check("groups named", OK,
                              f"All {groups['total']} groups named "
                              f"({groups['named']} as classes, {groups['nothing']} as "
                              f"nothing, {groups['mixed']} mixed)."))
        if groups["mixed_pending"]:
            out.append(_check("mixed groups", WARN,
                              f"{groups['mixed_pending']} samples from mixed groups are "
                              "not checked yet.",
                              "Split them, or check them one by one in review."))

    # --- review ---------------------------------------------------------------
    rows = class_rows(project)
    decided = [s for s in readable if s.verdict != PENDING]
    by_person = sum(1 for s in decided if s.is_human)
    pending = len(readable) - len(decided)
    detail = (f"{by_person} of {len(readable)} samples decided by you, "
              f"{len(decided) - by_person} by auto-accept, {pending} not decided.")
    if not decided:
        out.append(_check("review", BAD, detail, "Name groups or check guesses."))
    else:
        out.append(_check("review", OK if by_person else WARN, detail,
                          "" if by_person else "Check some guesses yourself."))
    owed = {n: r["spot_checks_owed"] for n, r in rows.items() if r.get("spot_checks_owed")}
    if owed:
        out.append(_check("spot checks", BAD,
                          "Auto-accepted samples nobody has spot-checked yet: "
                          + ", ".join(f"{k} of " + ("'none of these'" if n == NONE
                                                     else repr(n))
                                      for n, k in owed.items()) + ".",
                          "Check guesses: review batches include them."))

    # --- classes --------------------------------------------------------------
    names = project.class_names()
    if not names:
        out.append(_check("classes", BAD, "Nothing is named yet.",
                          "Name a group, or add a class."))
    trainable = []
    for name in names:
        r = rows[name]
        n = r["samples"]
        if n < TRAINER_MIN:
            out.append(_check("size", BAD, f"{n} samples; training leaves out a class with "
                              f"fewer than {TRAINER_MIN}.",
                              "Check more guesses for it, or add footage where it appears.",
                              name))
        elif n < SOLID:
            out.append(_check("size", WARN, f"{n} samples; under {SOLID} a class gets many "
                              "wrong guesses on new footage.",
                              "Check more guesses for it, or add footage where it appears.",
                              name))
        else:
            out.append(_check("size", OK, f"{n} samples.", "", name))
        if n >= TRAINER_MIN:
            trainable.append(name)
        if not n:
            continue
        if r["videos"] < MIN_VIDEOS:
            more = MIN_VIDEOS - r["videos"]
            out.append(_check("videos", WARN,
                              f"From {r['videos']} video{'s' if r['videos'] != 1 else ''}: "
                              "the model cannot tell it from that footage's scene, and "
                              "nothing can measure how it does on new footage.",
                              f"Add {more} more video{'s' if more != 1 else ''} showing it.",
                              name))
        elif r["one_video_share"] > MAX_ONE_VIDEO:
            out.append(_check("videos", WARN,
                              f"{r['one_video_share']:.0%} of it comes from one video.",
                              "Add more from the other videos; it learns that one video.",
                              name))
        else:
            out.append(_check("videos", OK, f"From {r['videos']} videos.", "", name))

    if project.task == ACTIONS and names:
        none = rows[NONE]
        if not project.settings.background_class:
            out.append(_check("none of these", WARN,
                              "No 'none of these' class: the model will name one of your "
                              "classes for everything it sees.",
                              "Mark a group or some samples as 'nothing'."))
        elif none["samples"] < TRAINER_MIN:
            out.append(_check("none of these", WARN,
                              f"{none['samples']} 'none of these' samples.",
                              "Mark more groups or samples as 'nothing'."))
        else:
            out.append(_check("none of these", OK, f"{none['samples']} samples."))
        if project.settings.background_class and none["samples"] >= TRAINER_MIN:
            trainable.append(NONE)

    if names and len(trainable) < 2:
        out.append(_check("classes", BAD, "Training needs at least two classes with "
                          f"{TRAINER_MIN}+ samples each ('none of these' counts).",
                          "Name another group, or mark some samples as 'nothing'."))
    sizes = [rows[n]["samples"] for n in names if rows[n]["samples"] >= TRAINER_MIN]
    if len(sizes) >= 2 and max(sizes) > MAX_IMBALANCE * min(sizes):
        small = min((n for n in names if rows[n]["samples"] >= TRAINER_MIN),
                    key=lambda n: rows[n]["samples"])
        out.append(_check("balance", WARN,
                          f"The largest class has {max(sizes)} samples, {small!r} only "
                          f"{min(sizes)}.",
                          f"Check more guesses for {small!r}; training softens the rest."))
    return out


def verdict(found: list) -> str:
    if any(c["level"] == BAD for c in found):
        return NOT_READY
    if any(c["name"] == "videos" and c["level"] == WARN for c in found):
        return NEEDS_VIDEOS
    return READY


SAYS = {
    READY: "Healthy: ready to train.",
    NEEDS_VIDEOS: "Labelled, and it will train, but some classes come from too few "
                  "videos to work on new footage. Add videos that show them.",
    NOT_READY: "Not ready to train yet.",
}


def _right(sample, guess: str) -> bool:
    if guess == NONE:
        return sample.verdict == NEGATIVE
    return sample.verdict == ACCEPTED and sample.label == guess


def sorting(project: Project) -> dict:
    """How often the machine's guess was what a person then decided.

    Only samples a person decided while shown a guess count. A guess of a
    class or "none of these" is right or wrong; a rejected sample (unusable)
    and an "unsure" guess are counted apart, being neither.
    """
    seen = [s for s in project.samples if s.is_human and s.shown_guess]
    by_source = defaultdict(lambda: {"shown": 0, "right": 0, "wrong": 0, "rejected": 0,
                                     "unsure": 0})
    by_class = defaultdict(lambda: {"guessed": 0, "right": 0, "rejected": 0,
                                    "really": Counter()})
    # Groups: every tile a person was shown is a guess of the group's name,
    # and those said not to belong were the wrong ones -- they stay undecided,
    # so they are counted from the group, not from the samples.
    by_id = {s.id: s for s in project.samples}
    groups = []
    for g in project.groups:
        if not g.shown or g.state not in (NAMED, NOTHING):
            continue
        guess, row = g.name or NONE, by_source[group_mod.DECIDED_BY]
        wrong = set(g.wrong)
        for sid in g.shown:
            row["shown"] += 1
            by_class[guess]["guessed"] += 1
            if sid not in wrong:
                row["right"] += 1
                by_class[guess]["right"] += 1
                continue
            row["wrong"] += 1
            later = by_id.get(sid)
            if later is not None and later.is_human:
                by_class[guess]["really"][later.label if later.verdict == ACCEPTED
                                          else NONE if later.verdict == NEGATIVE
                                          else "rejected"] += 1
        groups.append({"id": g.id, "name": guess, "shown": len(g.shown),
                       "wrong": len(g.wrong),
                       "accuracy": round(1 - len(g.wrong) / len(g.shown), 3)})

    for sample in seen:
        if sample.guess_from == group_mod.DECIDED_BY:
            continue                        # counted from its group above
        guess, row = sample.shown_guess, by_source[sample.guess_from or "sort"]
        row["shown"] += 1
        if guess == UNSURE or not guess:
            row["unsure"] += 1
            continue
        if sample.verdict == REJECTED:
            row["rejected"] += 1
            by_class[guess]["rejected"] += 1
            by_class[guess]["guessed"] += 1
            continue
        c = by_class[guess]
        c["guessed"] += 1
        if _right(sample, guess):
            row["right"] += 1
            c["right"] += 1
        else:
            row["wrong"] += 1
            c["really"][sample.label if sample.verdict == ACCEPTED else NONE] += 1

    def rate(right, wrong):
        return round(right / (right + wrong), 3) if right + wrong else None

    sources = {k: {**v, "accuracy": rate(v["right"], v["wrong"])}
               for k, v in by_source.items()}
    classes = {k: {"guessed": v["guessed"], "right": v["right"],
                   "rejected": v["rejected"],
                   "accuracy": rate(v["right"], v["guessed"] - v["right"] - v["rejected"]),
                   "really": dict(v["really"].most_common())}
               for k, v in by_class.items()}
    shown = sum(g["shown"] for g in groups)
    wrong = sum(g["wrong"] for g in groups)
    # Fixed later: what auto-accept decided and a spot check overturned.
    audits = {}
    for name in project.class_names() + [NONE]:
        done, overturned = autolabel.spot_checks(project, name)
        if done:
            audits[name] = {"checked": done, "overturned": overturned}
    total_right = sum(v["right"] for v in by_source.values())
    total_wrong = sum(v["wrong"] for v in by_source.values())
    return {
        "checked": len(seen),
        "accuracy": rate(total_right, total_wrong),
        "by_source": sources,
        "by_class": classes,
        "groups": {"shown": shown, "wrong": wrong,
                   "accuracy": round(1 - wrong / shown, 3) if shown else None,
                   "each": groups},
        "spot_checks": audits,
    }


def report(project: Project) -> dict:
    found = checks(project)
    state = verdict(found)
    return {"verdict": state, "says": SAYS[state], "checks": found,
            "classes": class_rows(project), "groups": group_mod.summary(project),
            "sorting": sorting(project)}
