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


def round_detector(project):
    """The installed round's detector, to propose boxes from round 2 on."""
    from modules.teach.train import installed_detector_files

    found = installed_detector_files(project)
    if found is None or not os.path.exists(found["xml"]):
        return None
    try:
        from modules.vision.detection_backend import create_detector
        return create_detector(found["xml"], found["classes"])
    except Exception as exc:        # fall back to the stock detector alone
        print(f"teach: round {found['round']} detector unusable: {exc}", file=sys.stderr)
        return None


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
    elif not args.no_model:
        # From round 2 the project's own model proposes too, and review asks
        # first about where it and CLIP disagree.
        classifier = sort.round_classifier(project)
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
    if args.window:
        from modules.teach.review_window import open_window
        open_window(project.root, size=args.size, class_name=args.cls or None)
        return {"window": "closed", "counts": Project.load(project.root).counts()}
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
        return boxes.propose(project, make_detector(), make_embedder(),
                             model_detector=round_detector(project))
    if args.action == "review":
        if args.window:
            from modules.teach.review_window import open_window
            open_window(project.root, size=args.size, boxes=True)
            labels = boxes.store(Project.load(project.root))
            return {"window": "closed", "accepted": len(labels.accepted()),
                    "pending": len(labels.pending())}
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


def run_auto(root: str, *, train: bool = False, max_steps: int = 20) -> dict:
    """Run every unattended step in turn; stop where someone has to look.

    Each step is exactly what ``status`` names, run through this same CLI, so
    ``auto`` can never do anything a person could not do by hand. It stops at
    a ``judge`` step, at training unless ``train`` is set, at a failure, or if
    a step leaves ``status`` asking for it again.
    """
    from modules.teach.status import next_step

    done = []
    for _ in range(max_steps):
        step = next_step(Project.load(root))
        args = step.get("args") or []
        if step["who"] != "auto" or not args:
            return {"ran": done, "stopped_at": step}
        if args[0] == "train" and not train:
            return {"ran": done, "stopped_at": step,
                    "message": "Ready to train. Run `train`, or `auto --train` to "
                               "include it (GPU minutes to hours)."}
        if done and done[-1]["args"] == args:
            return {"ran": done, "stopped_at": step,
                    "error": f"`{' '.join(args)}` ran but is still the next step"}
        print(f"auto: {' '.join(args)}", file=sys.stderr)
        code, result = run(["--project", root, *args])
        result = dict(result or {})
        result.pop("next", None)
        done.append({"args": args, "exit": code,
                     "result": {k: v for k, v in result.items()
                                if k not in ("items", "prototypes")}})
        if code != 0:
            return {"ran": done, "stopped_at": step,
                    "error": result.get("error") or result.get("errors")}
    return {"ran": done, "stopped_at": next_step(Project.load(root)),
            "message": f"stopped after {max_steps} steps"}


def cmd_auto(args, project):
    return run_auto(project.root, train=args.train)


def cmd_quick(args, root):
    """Project, classes, examples and footage in one go, then ``auto``.

    ``--examples`` is a folder with one subfolder per class, named after what
    it shows, holding a few clips of it: the folder names become the classes
    and the clips their first examples. Everything already there is kept, so
    running it again with more examples or videos just adds them.
    """
    from modules.teach import doctor
    from modules.teach.cut import VIDEO_EXTENSIONS
    from modules.teach.naming import check_name, normalize_name

    # Before creating anything: a missing ffmpeg or CLIP found here costs
    # seconds; found after the footage is added, it costs a confusing error.
    if not args.skip_checks:
        doctor.require(root)

    if os.path.exists(os.path.join(root, project_mod.PROJECT_FILE)):
        project = Project.load(root)
    else:
        if not args.task:
            raise ValueError("a new project needs --task actions or --task objects")
        project = Project.create(root, args.task)
    if args.focus:
        project.settings.focus = True

    classes, problems = {}, []
    if args.examples:
        for folder in sorted(os.listdir(args.examples)):
            path = os.path.join(args.examples, folder)
            if not os.path.isdir(path) or folder.startswith((".", "_")):
                continue
            clips = sorted(os.path.join(path, n) for n in os.listdir(path)
                           if n.lower().endswith(VIDEO_EXTENSIONS))
            name = normalize_name(folder)
            if project.get_class(name) is None:
                blocking = [p for p in check_name(name, project.class_names(), project.task)
                            if p.blocking]
                if blocking:
                    problems.append(f"{folder!r}: " + "; ".join(p.message for p in blocking))
                    continue
                project.add_class(name)
            classes[name] = clips
    if problems:
        raise ValueError("rename these example folders: " + " | ".join(problems))
    if not project.classes:
        raise ValueError("no classes: give --examples <folder with one subfolder per class>")
    project.save()

    from types import SimpleNamespace
    added = {}
    for name, clips in classes.items():
        if clips:
            cmd_add_example(SimpleNamespace(cls=name, clip=clips, sample=None), project)
            added[name] = len(clips)
    if args.videos:
        cmd_add_video(SimpleNamespace(items=args.videos), project)
    project.save()
    return {"project": project.root, "classes": project.class_names(),
            "examples_added": added, "sources": len(project.sources),
            **run_auto(project.root, train=args.train)}


def cmd_share(args, project):
    from dataclasses import asdict

    from modules.teach.share import NotShareable, share_draft
    try:
        onnx, draft = share_draft(project)
    except NotShareable as exc:
        raise ValueError(str(exc)) from None
    return {"model": onnx, "draft": asdict(draft),
            "how": "Training -> From videos -> Share... opens the publish wizard with "
                   "this filled in; name, description, category and the checklist "
                   "are yours to complete."}


def cmd_doctor(args, root):
    from modules.teach import doctor
    return doctor.run(root)


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
    s.add_argument("--no-model", action="store_true", dest="no_model",
                   help="do not use the project's trained model as a second opinion")

    s = sub.add_parser("folders", help="lay out sorted/ folders, or --read them back")
    s.add_argument("--read", action="store_true")
    s.add_argument("--confirm", action="append",
                   help="class folders whose unmoved files are confirmed; 'all' for every one")

    s = sub.add_parser("review", help="draw the next contact sheet")
    s.add_argument("--size", type=int, default=24)
    s.add_argument("--window", action="store_true",
                   help="review by clicking, in a window, instead of a sheet image")
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
    s.add_argument("--window", action="store_true",
                   help="review: by clicking, in a window, instead of a sheet image")

    sub.add_parser("build", help="write the dataset")

    s = sub.add_parser("train", help="train a round")
    s.add_argument("--epochs", type=int)
    s.add_argument("--install", choices=["if-better", "always", "never"],
                   default="if-better")

    sub.add_parser("status", help="where it stands, and the next command")

    s = sub.add_parser("auto", help="run every unattended step until one needs a look")
    s.add_argument("--train", action="store_true", help="include training")

    s = sub.add_parser("quick", help="examples folder + videos -> as far as it can go alone")
    s.add_argument("--task", choices=project_mod.TASKS,
                   help="needed when the project does not exist yet")
    s.add_argument("--examples", help="folder with one subfolder of clips per class")
    s.add_argument("--videos", nargs="+", default=[], help="files, folders or URLs")
    s.add_argument("--focus", action="store_true")
    s.add_argument("--train", action="store_true")
    s.add_argument("--skip-checks", action="store_true", dest="skip_checks",
                   help="do not run `doctor` first")

    sub.add_parser("doctor", help="is this machine ready? (seconds; nothing is loaded)")
    sub.add_parser("share", help="the installed detector, drafted for the model hub")

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
    "auto": cmd_auto, "share": cmd_share,
}


def run(argv=None) -> tuple:
    """``(exit code, result dict)``; what ``main`` prints."""
    import io

    complaint = io.StringIO()
    try:
        with contextlib.redirect_stderr(complaint):
            args = parser().parse_args(argv)
    except SystemExit as exc:
        if not exc.code:                    # --help: argparse printed it, as asked
            sys.stderr.write(complaint.getvalue())
            raise
        # Bad arguments: an answer like any other, not an exit, so the panel
        # and ``auto`` (which call this in-process) get told instead of dying.
        lines = [ln for ln in complaint.getvalue().splitlines() if ln.strip()]
        return 2, {"error": (lines[-1].split(" error: ", 1)[-1] if lines
                            else "bad arguments"),
                   "usage": "\n".join(lines[:-1])}
    root = resolve_root(args.project)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            if args.command == "init":
                result = cmd_init(args, root)
            elif args.command == "doctor":
                result = cmd_doctor(args, root)
            elif args.command == "quick":
                result = cmd_quick(args, root)
                result.setdefault("next", result.get("stopped_at"))
            else:
                if not os.path.exists(os.path.join(root, project_mod.PROJECT_FILE)):
                    raise FileNotFoundError(f"no project at {root}; run init first")
                project = Project.load(root)
                result = COMMANDS[args.command](args, project)
                if args.command not in ("status", "auto"):
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
