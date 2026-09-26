"""``python -m modules.teach`` — every step of teaching a model, one command each.

Each command prints exactly one JSON object on stdout, and everything else
(progress, library chatter) goes to stderr, so a script or an agent can parse
the answer without guessing. ``status`` names the next command to run.

    python -m modules.teach --project jumps init --task actions
    python -m modules.teach --project jumps add-class "<name>" --description "..."
    python -m modules.teach --project jumps add-video clip1.mp4 clip2.mp4
    python -m modules.teach --project jumps status

``--project`` is a folder, or a bare name for a folder under the app's user
data (``<user data>/teach/<name>``).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

from modules.teach import project as project_mod
from modules.teach.project import ACCEPTED, Project, Sample


def make_embedder():
    """The real CLIP. Tests replace this."""
    from modules.teach.embed import ClipBackend
    return ClipBackend()


def make_detector():
    """The stock object detector, for proposing boxes. Tests replace this."""
    from modules.vision.detection_backend import build_object_detector
    detector, _ = build_object_detector("coco", default_prefer="large",
                                        log=lambda *a: print(*a, file=sys.stderr))
    if detector is None:
        raise RuntimeError("no object detector installed")
    return detector


def resolve_root(value: str) -> str:
    if os.sep in value or (os.altsep and os.altsep in value) or value.startswith("."):
        return os.path.abspath(value)
    return os.path.join(project_mod.projects_root(), project_mod.slugify(value))


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_init(args, root):
    project = Project.create(root, args.task, name=args.name or "")
    if args.clip_seconds:
        project.settings.clip_seconds = project.settings.stride_seconds = args.clip_seconds
    if args.focus:
        project.settings.focus = True
    project.save()
    return {"created": project.root, "task": project.task,
            "settings": vars(project.settings)}


def cmd_add_class(args, project):
    spec = project.add_class(args.name, args.description or "",
                             target=args.target or project_mod.DEFAULT_TARGET)
    project.save()
    from modules.teach.naming import check_name
    advice = [p.to_json() for p in check_name(
        spec.name, [n for n in project.class_names() if n != spec.name], project.task)]
    return {"added": spec.name, "advice": advice, "classes": project.class_names()}


def cmd_rename_class(args, project):
    project.rename_class(args.old, args.new)
    project.save()
    return {"classes": project.class_names()}


def cmd_check_name(args, project):
    from modules.teach.naming import check_name, normalize_name
    name = normalize_name(args.name)
    problems = check_name(name, project.class_names(), project.task)
    return {"name": name, "ok": not any(p.blocking for p in problems),
            "problems": [p.to_json() for p in problems]}


def _vectors_for(project, embedder, clips=(), sample_ids=(), class_name=""):
    from modules.teach import embed as embed_mod

    samples = []
    for sid in sample_ids:
        sample = project.get_sample(sid)
        if sample is None:
            raise KeyError(f"no sample {sid}")
        samples.append(sample)
    if class_name:
        spec = project.get_class(class_name)
        if spec is None:
            raise KeyError(f"no class {class_name!r}")
        ids = list(dict.fromkeys(spec.examples + [s.id for s in project.accepted(class_name)]))
        samples += [project.get_sample(i) for i in ids if project.get_sample(i)]
    for i, clip in enumerate(clips):
        samples.append(Sample(id=f"clip:{os.path.abspath(clip)}", source="", path=clip,
                              start=0.0, duration=0.0))
    cache = embed_mod.VectorCache(project.root, getattr(embedder, "model_id", ""))
    vectors = embed_mod.sample_vectors(samples, embedder, cache,
                                       project.settings.frames_per_sample)
    return [vectors[s.id] for s in samples if s.id in vectors]


def cmd_suggest_names(args, project):
    import numpy as np

    from modules.teach import naming

    embedder = make_embedder()
    vectors = _vectors_for(project, embedder, args.clip or (), args.sample or (),
                           args.cls or "")
    if not vectors:
        raise ValueError("give example clips (--clip), samples (--sample) or a class "
                         "with accepted samples (--class)")
    labels = naming.load_vocabulary(project.task)
    label_vectors = embedder.texts([naming.PROMPTS[project.task].format(x) for x in labels])
    result = naming.suggest_names(np.stack(vectors), labels, label_vectors, args.top)
    result["examples"] = len(vectors)
    result["advice"] = (
        "A 'good' fit can be reused as the name. Otherwise name it yourself, in the "
        "style of the stock labels, and add a description. A low consistency or a "
        "split means the examples may be two different things.")
    return result


def cmd_add_video(args, project):
    added = []
    for item in args.items:
        if item.startswith(("http://", "https://")):
            from downloader import download_video
            ok, path, meta = download_video(item, project.path("videos"),
                                            log_fn=lambda *a: print(*a, file=sys.stderr))
            if not ok or not path:
                raise RuntimeError(f"could not download {item}")
            source = project.add_source(path, url=item)
        else:
            if not os.path.exists(item):
                raise FileNotFoundError(item)
            if os.path.isdir(item):
                from modules.teach.cut import VIDEO_EXTENSIONS
                paths = sorted(os.path.join(item, n) for n in os.listdir(item)
                               if n.lower().endswith(VIDEO_EXTENSIONS))
                for path in paths:
                    added.append(project.add_source(path).id)
                continue
            source = project.add_source(item)
        added.append(source.id)
    project.save()
    return {"sources": added, "total": len(project.sources)}


def cmd_add_example(args, project):
    """Clips you already have, or samples, shown as examples of a class."""
    from modules.teach.cut import probe_duration

    spec = project.get_class(args.cls)
    if spec is None:
        raise KeyError(f"no class {args.cls!r}")
    ids = []
    for sid in args.sample or ():
        sample = project.get_sample(sid)
        if sample is None:
            raise KeyError(f"no sample {sid}")
        project.decide(sample, ACCEPTED, spec.name, by="example")
        ids.append(sid)
    for clip in args.clip or ():
        if not os.path.exists(clip):
            raise FileNotFoundError(clip)
        source = project.add_source(clip)
        source.cut = True           # an example is already the right length
        sid = f"{source.id}__example"
        if project.get_sample(sid) is None:
            project.samples.append(Sample(id=sid, source=source.id,
                                          path=os.path.abspath(clip), start=0.0,
                                          duration=probe_duration(clip)))
        project.decide(project.get_sample(sid), ACCEPTED, spec.name, by="example")
        ids.append(sid)
    for sid in ids:
        if sid not in spec.examples:
            spec.examples.append(sid)
    project.save()
    return {"class": spec.name, "examples": spec.examples}


def cmd_cut(args, project):
    from modules.teach.cut import cut_project
    return cut_project(project, progress=lambda s, i, n: print(
        f"cut {s}: {i}/{n}", file=sys.stderr) if i == n or i % 20 == 0 else None)


def cmd_focus(args, project):
    from modules.teach.focus import focus_project
    return focus_project(project)


def cmd_sort(args, project):
    from modules.teach import sort
    classifier = None
    if args.model_xml:
        base = os.path.splitext(args.model_xml)[0]
        classifier = sort.sorter_classifier(args.model_xml, base + ".bin",
                                            args.model_mapping or base + ".json")
    result = sort.sort_project(project, make_embedder(), model_classifier=classifier,
                               progress=lambda i, n: print(f"embedded {i}/{n}",
                                                           file=sys.stderr)
                               if i == n or i % 50 == 0 else None)
    if args.folders:
        result["folders"] = sort.lay_out_folders(project)
    return result


def cmd_folders(args, project):
    from modules.teach import review, sort
    if args.read:
        return review.from_folders(project, args.confirm or ())
    return sort.lay_out_folders(project)


def cmd_review(args, project):
    from modules.teach import review
    record = review.next_sheet(project, size=args.size, class_name=args.cls or None)
    if not record:
        return {"sheet": None, "message": "nothing waiting for review"}
    record["how"] = (f"Look at {record['image']}. Then: verdict --sheet {record['sheet']} "
                     "--accept 1-5,7 --reject 6 --negative 9 --relabel 8=<class>, "
                     "or --accept-rest to take every unmentioned guess as right.")
    return record


def cmd_verdict(args, project):
    from modules.teach import review
    return review.apply_verdicts(project, args.sheet, accept=args.accept or "",
                                 reject=args.reject or "", negative=args.negative or "",
                                 relabel=args.relabel or (), accept_rest=args.accept_rest)


def cmd_boxes(args, project):
    from modules.teach import boxes
    if args.action == "propose":
        return boxes.propose(project, make_detector(), make_embedder())
    if args.action == "review":
        record = boxes.next_sheet(project, size=args.size)
        if record:
            record["how"] = (f"Look at {record['image']}. Then: boxes verdict --sheet "
                             f"{record['sheet']} --accept 1-4 --reject 5, or --accept-rest.")
        return record or {"sheet": None, "message": "no boxes waiting"}
    if args.action == "verdict":
        return boxes.apply_verdicts(project, args.sheet, accept=args.accept or "",
                                    reject=args.reject or "", accept_rest=args.accept_rest)
    if args.action == "worklist":
        items = boxes.labeler_worklist(project)
        return {"to_label": items,
                "how": "Open each path in `python tools/labeler.py`, mark the thing, "
                       "export, then: boxes import <export.json> ..."}
    if args.action == "import":
        return boxes.import_labeler(project, args.files, accept=args.accept_all)
    raise ValueError(args.action)


def cmd_build(args, project):
    from modules.teach.build import build
    return build(project)


def cmd_train(args, project):
    from modules.teach.train import train_round
    return train_round(project, epochs=args.epochs, install_policy=args.install)


def cmd_status(args, project):
    from modules.teach.status import report
    return report(project)


def cmd_set(args, project):
    known = project_mod.Settings.__dataclass_fields__
    changed = {}
    for pair in args.pairs:
        key, _, value = pair.partition("=")
        if key not in known:
            raise KeyError(f"no setting {key!r}; settings: {sorted(known)}")
        current = getattr(project.settings, key)
        if isinstance(current, bool):
            new = value.lower() in ("1", "true", "yes", "on")
        else:
            new = type(current)(value)
        setattr(project.settings, key, new)
        changed[key] = new
    project.save()
    return {"settings": vars(project.settings), "changed": changed}


# ---------------------------------------------------------------------------

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m modules.teach",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--project", required=True,
                   help="project folder, or a name under the app's user data")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init", help="start a project")
    s.add_argument("--task", required=True, choices=project_mod.TASKS)
    s.add_argument("--name")
    s.add_argument("--clip-seconds", type=float, dest="clip_seconds")
    s.add_argument("--focus", action="store_true",
                   help="actions: crop samples to the people in them (modules/crop)")

    s = sub.add_parser("add-class", help="something to find")
    s.add_argument("name")
    s.add_argument("--description")
    s.add_argument("--target", type=int)

    s = sub.add_parser("rename-class")
    s.add_argument("old")
    s.add_argument("new")

    s = sub.add_parser("check-name", help="is this a good class name?")
    s.add_argument("name")

    s = sub.add_parser("suggest-names", help="stock labels that describe some examples")
    s.add_argument("--clip", action="append")
    s.add_argument("--sample", action="append")
    s.add_argument("--class", dest="cls")
    s.add_argument("--top", type=int, default=5)

    s = sub.add_parser("add-video", help="footage: files, folders or URLs")
    s.add_argument("items", nargs="+")

    s = sub.add_parser("add-example", help="clips or samples that show a class")
    s.add_argument("--class", dest="cls", required=True)
    s.add_argument("--clip", action="append")
    s.add_argument("--sample", action="append")

    sub.add_parser("cut", help="cut footage into samples")
    sub.add_parser("focus", help="actions: person-focused crops of each sample")

    s = sub.add_parser("sort", help="score and propose a class for every sample")
    s.add_argument("--folders", action="store_true", help="also lay out sorted/ folders")
    s.add_argument("--model-xml", dest="model_xml",
                   help="an Intel-encoder decoder IR for sorter.py to propose with")
    s.add_argument("--model-mapping", dest="model_mapping")

    s = sub.add_parser("folders", help="lay out sorted/ folders, or --read them back")
    s.add_argument("--read", action="store_true")
    s.add_argument("--confirm", action="append",
                   help="class folders whose unmoved files are confirmed; 'all' for every one")

    s = sub.add_parser("review", help="draw the next contact sheet")
    s.add_argument("--size", type=int, default=24)
    s.add_argument("--class", dest="cls")

    s = sub.add_parser("verdict", help="record what a contact sheet shows")
    s.add_argument("--sheet", type=int, required=True)
    s.add_argument("--accept")
    s.add_argument("--reject")
    s.add_argument("--negative")
    s.add_argument("--relabel", action="append", help="N=<class>, repeatable")
    s.add_argument("--accept-rest", action="store_true", dest="accept_rest")

    s = sub.add_parser("boxes", help="object projects: boxes on accepted samples")
    s.add_argument("action", choices=["propose", "review", "verdict", "worklist", "import"])
    s.add_argument("files", nargs="*")
    s.add_argument("--sheet", type=int)
    s.add_argument("--size", type=int, default=20)
    s.add_argument("--accept")
    s.add_argument("--reject")
    s.add_argument("--accept-rest", action="store_true", dest="accept_rest")
    s.add_argument("--accept-all", action="store_true", dest="accept_all",
                   help="import: the labeller's points are already checked")

    sub.add_parser("build", help="write the dataset")

    s = sub.add_parser("train", help="train a round")
    s.add_argument("--epochs", type=int)
    s.add_argument("--install", choices=["if-better", "always", "never"],
                   default="if-better")

    sub.add_parser("status", help="where it stands, and the next command")

    s = sub.add_parser("set", help="change settings: key=value ...")
    s.add_argument("pairs", nargs="+")
    return p


COMMANDS = {
    "add-class": cmd_add_class, "rename-class": cmd_rename_class,
    "check-name": cmd_check_name, "suggest-names": cmd_suggest_names,
    "add-video": cmd_add_video, "add-example": cmd_add_example, "cut": cmd_cut,
    "focus": cmd_focus, "sort": cmd_sort, "folders": cmd_folders,
    "review": cmd_review, "verdict": cmd_verdict, "boxes": cmd_boxes,
    "build": cmd_build, "train": cmd_train, "status": cmd_status, "set": cmd_set,
}


def run(argv=None) -> tuple:
    """``(exit code, result dict)``; what ``main`` prints."""
    args = parser().parse_args(argv)
    root = resolve_root(args.project)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if args.command == "init":
                result = cmd_init(args, root)
            else:
                if not os.path.exists(os.path.join(root, project_mod.PROJECT_FILE)):
                    raise FileNotFoundError(f"no project at {root}; run init first")
                project = Project.load(root)
                result = COMMANDS[args.command](args, project)
                if args.command not in ("status",):
                    from modules.teach.status import next_step
                    result = dict(result or {})
                    result.setdefault("next", next_step(Project.load(root)))
        code = 1 if (result or {}).get("errors") else 0
        return code, result
    except (KeyError, ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
        return 2, {"error": f"{type(exc).__name__}: {message}"}


def main(argv=None) -> int:
    code, result = run(argv)
    print(json.dumps(result, indent=2, default=str))
    return code
