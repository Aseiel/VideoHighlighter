"""A teaching project on disk: what to find, where, and every decision so far.

Layout, all under one folder the user owns (never the repo)::

    project.json        task, classes, settings, sources, training rounds
    samples.json        one row per cut sample: scores, proposal, verdict, split
    labels.json         object boxes (modules.vision.label_store format)
    videos/             sources fetched from a URL (local files stay in place)
    samples/            cut samples, <source>__<start ms>.mp4
    focus/              the cropper's person-focused versions (actions only)
    sorted/<class>/     samples as folders, for sorting by hand
    review/             contact sheets and what each tile is
    dataset/            what training reads
    runs/<n>/           each training round

The two JSON files are the state. Everything else can be rebuilt from them and
the sources, which is what makes every step safe to re-run and a project safe
to leave half-done.

**Two tasks.** ``actions`` teaches a clip classifier (R3D, which the app's
action recognition loads); a sample's label is the whole clip. ``objects``
teaches a detector (YOLOX); a sample is a place to look, and the labels are
boxes on its frames.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

FORMAT = 1
PROJECT_FILE = "project.json"
SAMPLES_FILE = "samples.json"
LABELS_FILE = "labels.json"

ACTIONS = "actions"
OBJECTS = "objects"
TASKS = (ACTIONS, OBJECTS)

# Sample verdicts. PENDING has not been looked at; the rest were decided.
PENDING = "pending"
ACCEPTED = "accepted"      # shows ``label``
REJECTED = "rejected"      # not usable (wrong, unclear, a bad cut) — never trained on
NEGATIVE = "negative"      # shows none of the classes: object frames with no boxes,
                           # and for actions the `background_class`, when one is set
VERDICTS = (PENDING, ACCEPTED, REJECTED, NEGATIVE)

# decided_by for auto-accepted samples.
AUTO = "auto"
# decided_by for samples accepted because their found box was accepted
# without a question (``cutoff``); spot-checked through the boxes, not here.
AUTO_BOX = "auto-box"

# Proposal when no class stood out, and when every class scored low.
UNSURE = "_unsure"
NONE = "_none"

# The sample set that measures every round. Assigned once, never moved.
TRAIN = "train"
VAL = "val"

# How many accepted samples per class a round aims for. A hundred gave good
# results by hand (docs/CUSTOM-MODEL-TRAINING.md has the measured range), and
# twenty is the least that trains anything worth measuring.
DEFAULT_TARGET = 100
MIN_TO_TRAIN = 20


@dataclass
class ClassSpec:
    name: str
    description: str = ""          # more words for CLIP, e.g. what it looks like
    examples: list = field(default_factory=list)   # sample ids shown as examples
    target: int = DEFAULT_TARGET


@dataclass
class Source:
    id: str
    path: str
    url: str = ""
    duration: float = 0.0
    cut: bool = False


@dataclass
class Sample:
    id: str
    source: str
    path: str
    start: float
    duration: float
    focus_paths: list = field(default_factory=list)
    focus_tried: bool = False      # the cropper has seen it (it may have made nothing)
    boxes_tried: bool = False      # boxes were proposed for it (maybe none found)
    unreadable: bool = False       # no frames could be decoded: never sorted, never asked about
    scores: dict = field(default_factory=dict)       # class -> calibrated score
    proposed: str = ""                               # class, UNSURE or NONE
    margin: float = 0.0
    model_proposed: str = ""                         # the last round's model says
    model_confidence: float = 0.0
    verdict: str = PENDING
    label: str = ""
    decided_by: str = ""                             # "example", "sheet", "folders", "auto"
    split: str = ""
    # What auto-accept said, kept after a person re-checks it: the record the
    # spot checks are scored from (``autolabel``).
    auto_label: str = ""

    @property
    def is_decided(self) -> bool:
        return self.verdict != PENDING

    @property
    def is_auto(self) -> bool:
        """Decided by auto-accept and not looked at by anyone since."""
        return self.decided_by == AUTO

    @property
    def is_human(self) -> bool:
        """Decided by someone looking, not by a rule: only these build
        prototypes and the held-out set."""
        return self.is_decided and self.decided_by not in (AUTO, AUTO_BOX)

    # label_store.segments() groups by these two names.
    @property
    def video(self) -> str:
        return self.source

    @property
    def time(self) -> float:
        return self.start


@dataclass
class Settings:
    clip_seconds: float = 5.0
    stride_seconds: float = 5.0      # == clip_seconds: back to back, no overlap
    frames_per_sample: int = 4       # embedded per sample for sorting
    focus: bool = False              # run the person-focused cropper (actions)
    gate: float = 0.5                # calibrated score to propose a class
    margin: float = 0.1              # ... and its lead over the runner-up
    floor: float = 0.2               # below this for every class: NONE
    # Centres per class prototype (``scoring``): 1 is the examples' mean; more
    # suits a class shown in a few different ways.
    prototypes_per_class: int = 1
    # How samples are scored (``sort``): "prototypes" compares them with each
    # class's centres; "linear" trains a layer on the samples a person sorted
    # (``linear``), and needs every class to have ``linear_min_examples`` --
    # until then the sort uses prototypes. Its proposals are gated by the
    # probability at which held-out videos were right ``linear_precision`` of
    # the time, in place of ``gate``, and auto-accept by the one at which they
    # were right ``linear_auto_precision`` of the time, in place of
    # ``auto_gate``. Both are measured on the examples' own videos, and a new
    # video comes out lower: on one dataset, 0.8 here gave proposals right 65%
    # of the time on held-out videos (0.7 gave 52%, 0.85 gave 86% of far fewer).
    scorer: str = "prototypes"
    linear_min_examples: int = 5
    linear_precision: float = 0.8
    linear_auto_precision: float = 0.9
    val_fraction: float = 0.2
    boxes_per_sample: int = 3        # frames labelled per accepted object sample
    epochs: int = 30
    # Auto-accept (``autolabel``): confident guesses decided without a review,
    # once a class has enough checked samples to be judged by, and while spot
    # checks keep agreeing with it.
    auto_accept: bool = True
    auto_gate: float = 0.9           # calibrated score: ~as close as the examples are
    auto_margin: float = 0.3         # and clearly ahead of every other class
    auto_min_checked: int = 5        # checked samples a class needs first
    auto_max_error: float = 0.2      # spot checks overturning more: off for that class
    # Actions only: the class "none of these" samples are trained as. An action
    # classifier without one has no way to say "none", so it names one of the
    # classes for everything; with one, it can decline. Empty = not trained on.
    background_class: str = ""
    # Improve this project unattended while the app is idle (``background``).
    # Set by teaching from the player; the From videos tab has a switch.
    background: bool = False


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return slug[:48] or "project"


class Project:
    """Load, change and save one project. Not thread-safe; one writer."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.name = os.path.basename(self.root)
        self.task = ACTIONS
        self.created = 0.0
        self.classes: list = []
        self.sources: list = []
        self.settings = Settings()
        self.rounds: list = []
        self.samples: list = []

    # --- paths -------------------------------------------------------------

    def path(self, *parts) -> str:
        return os.path.join(self.root, *parts)

    @property
    def exists(self) -> bool:
        return os.path.exists(self.path(PROJECT_FILE))

    # --- persistence -------------------------------------------------------

    @classmethod
    def create(cls, root: str, task: str, name: str = "") -> "Project":
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, not {task!r}")
        project = cls(root)
        if project.exists:
            raise FileExistsError(f"{project.root} already holds a project")
        project.task = task
        project.name = name or project.name
        project.created = time.time()
        os.makedirs(project.root, exist_ok=True)
        project.save()
        return project

    @classmethod
    def load(cls, root: str) -> "Project":
        project = cls(root)
        with open(project.path(PROJECT_FILE), "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("format") != FORMAT:
            raise ValueError(f"unsupported project format {data.get('format')!r}")
        project.name = data.get("name") or project.name
        project.task = data["task"]
        project.created = float(data.get("created", 0.0))
        project.classes = [ClassSpec(**c) for c in data.get("classes", [])]
        project.sources = [Source(**s) for s in data.get("sources", [])]
        known = Settings.__dataclass_fields__
        project.settings = Settings(**{k: v for k, v in data.get("settings", {}).items()
                                       if k in known})
        project.rounds = list(data.get("rounds", []))
        try:
            with open(project.path(SAMPLES_FILE), "r", encoding="utf-8") as handle:
                project.samples = [Sample(**s) for s in json.load(handle)]
        except FileNotFoundError:
            project.samples = []
        return project

    def save(self) -> None:
        data = {
            "format": FORMAT,
            "name": self.name,
            "task": self.task,
            "created": self.created,
            "classes": [asdict(c) for c in self.classes],
            "sources": [asdict(s) for s in self.sources],
            "settings": asdict(self.settings),
            "rounds": self.rounds,
        }
        _write_json(self.path(PROJECT_FILE), data)
        _write_json(self.path(SAMPLES_FILE), [asdict(s) for s in self.samples])

    # --- classes -----------------------------------------------------------

    def get_class(self, name: str) -> Optional[ClassSpec]:
        for spec in self.classes:
            if spec.name == name:
                return spec
        return None

    def class_names(self) -> list:
        return [c.name for c in self.classes]

    def add_class(self, name: str, description: str = "",
                  target: int = DEFAULT_TARGET) -> ClassSpec:
        from modules.teach.naming import check_name, normalize_name

        clean = normalize_name(name)
        problems = [p for p in check_name(clean, self.class_names()) if p.blocking]
        if problems:
            raise ValueError("; ".join(p.message for p in problems))
        spec = ClassSpec(name=clean, description=description.strip(), target=int(target))
        self.classes.append(spec)
        return spec

    def rename_class(self, old: str, new: str) -> None:
        from modules.teach.naming import check_name, normalize_name

        spec = self.get_class(old)
        if spec is None:
            raise KeyError(old)
        clean = normalize_name(new)
        others = [n for n in self.class_names() if n != old]
        problems = [p for p in check_name(clean, others) if p.blocking]
        if problems:
            raise ValueError("; ".join(p.message for p in problems))
        spec.name = clean
        # Everywhere the name is recorded, or it quietly splits in two: the
        # spot-check record, the last model's guesses, and object boxes.
        for sample in self.samples:
            for attr in ("label", "proposed", "auto_label", "model_proposed"):
                if getattr(sample, attr) == old:
                    setattr(sample, attr, clean)
            if old in sample.scores:
                sample.scores[clean] = sample.scores.pop(old)
        if os.path.exists(self.path(LABELS_FILE)):
            from modules.vision.label_store import LabelStore
            labels = LabelStore(self.path(LABELS_FILE)).load()
            renamed = 0
            for box in labels.boxes:
                if box.class_name == old:
                    box.class_name = clean
                    renamed += 1
            if renamed:
                labels.save()

    # --- sources and samples -----------------------------------------------

    def add_source(self, path: str, url: str = "") -> Source:
        path = os.path.abspath(path)
        for source in self.sources:
            if os.path.normcase(source.path) == os.path.normcase(path):
                return source
        source = Source(id=f"v{len(self.sources) + 1:03d}", path=path, url=url)
        self.sources.append(source)
        return source

    def get_source(self, source_id: str) -> Optional[Source]:
        for source in self.sources:
            if source.id == source_id:
                return source
        return None

    def get_sample(self, sample_id: str) -> Optional[Sample]:
        for sample in self.samples:
            if sample.id == sample_id:
                return sample
        return None

    def decide(self, sample: Sample, verdict: str, label: str = "",
               by: str = "") -> None:
        """Record a verdict. ACCEPTED needs a class; the others clear it."""
        if verdict not in VERDICTS:
            raise ValueError(f"verdict must be one of {VERDICTS}")
        if verdict == ACCEPTED:
            label = label or sample.proposed
            if self.get_class(label) is None:
                raise ValueError(f"{sample.id}: no class {label!r} to accept it as")
            sample.label = label
        else:
            sample.label = ""
        sample.verdict = verdict
        sample.decided_by = by
        if by == AUTO:
            sample.auto_label = label if verdict == ACCEPTED else NONE

    def accepted(self, class_name: Optional[str] = None) -> list:
        return [s for s in self.samples if s.verdict == ACCEPTED
                and (class_name is None or s.label == class_name)]

    def counts(self) -> dict:
        """Per class: accepted, examples, proposed and still pending."""
        out = {}
        for spec in self.classes:
            out[spec.name] = {
                "accepted": len(self.accepted(spec.name)),
                "examples": len(spec.examples),
                "pending_proposed": sum(1 for s in self.samples
                                        if s.verdict == PENDING and s.proposed == spec.name),
                "target": spec.target,
            }
        return out


def _write_json(path: str, data) -> None:
    """Write via a temp file, so an interrupted save never leaves half a file."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=1)
        handle.write("\n")
    os.replace(tmp, path)


def projects_root() -> str:
    """Where projects live by default: the app's user data folder."""
    from modules.system.app_paths import user_data_dir
    return os.path.join(user_data_dir(), "teach")
