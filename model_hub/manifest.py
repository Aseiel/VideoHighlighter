"""videohighlighter.json - the manifest every community model must ship.

The manifest is what makes a model *usable in VideoHighlighter*: it tells the
app what the model detects, how to feed it frames and how to read its output.
Only plain JSON is parsed here; nothing in a model package is ever executed.

Two rules are enforced here rather than trusted to the publisher:

* **Detectors are YOLOX.** The only detector output format is ``yolox``
  (``[1, anchors, 5 + labels]``, raw grid, BGR 0-255 input) — what the app's
  own training produces. The transposed ``[1, 4 + labels, anchors]`` layout
  belongs to an AGPL toolkit, and a model trained with it inherits that
  licence; a shared model must be usable by anyone, in any build.
* **Licences are permissive.** A model with a copyleft or unknown licence
  cannot be used in every build, which defeats sharing it.

Content-neutral like the rest of the app: ``category`` is a slug the publisher
chooses (``animals/horses``, ``games/some-title``); no list of categories lives
in code.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_NAME = "videohighlighter.json"
SCHEMA_VERSION = 1
HUB_TAG = "videohighlighter"

TASKS = {
    # task id: (human name, output formats the app knows how to decode)
    "object_detection": ("Object detection", {"yolox"}),
    "image_classification": ("Frame classification", {"logits", "probabilities"}),
    "action_recognition": ("Action recognition", {"logits", "probabilities"}),
}
# What this version of the app can actually put on a timeline. The schema
# describes more, so packages made for later versions still validate.
USABLE_TASKS = {"object_detection"}

LAYOUTS = {"NCHW", "NHWC"}
COLOR_ORDERS = {"RGB", "BGR"}
NORMALIZATIONS = {"0-1", "0-255", "imagenet", "minus1-1"}

# SPDX-style ids as Hugging Face spells them. Permissive only, on purpose.
LICENSES = ("apache-2.0", "mit", "cc-by-4.0", "cc0-1.0")
DEFAULT_LICENSE = "apache-2.0"

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
CATEGORY_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*(?:/[a-z0-9]+(?:-[a-z0-9]+)*){0,2}$")
CATEGORY_TAG_PREFIX = "vh-cat-"

# Numbers the app measured, shown on the card and the site. Anything else in
# ``metrics`` is refused, so the field cannot carry text, paths or blobs.
METRIC_KEYS = {
    "heldout_found": "found on held-out frames",
    "heldout_expected": "things on held-out frames",
    "false_alarms": "wrong guesses on held-out frames",
    "train_frames": "frames trained on",
    "rounds": "rounds of training",
    "videos": "videos",
}

COMPLIANCE_ITEMS = {
    "terms_checked": "I checked the terms of every game or video source I trained on, "
                     "and none of them forbids AI/ML training.",
    "own_footage": "I trained only on footage I have the right to use (e.g. my own recordings).",
    "no_training_data": "The package contains no clips, screenshots, audio or other training data.",
    "non_generative": "The model only detects or classifies; it does not generate content.",
}


class ManifestError(ValueError):
    """Raised with a list of human-readable problems."""

    def __init__(self, problems: list[str]):
        super().__init__("\n".join(problems))
        self.problems = problems


@dataclass
class InputSpec:
    width: int = 416
    height: int = 416
    channels: int = 3
    layout: str = "NCHW"
    color: str = "BGR"
    normalize: str = "0-255"
    frames: int = 1  # >1 only for action_recognition clips


@dataclass
class Manifest:
    name: str
    display_name: str
    description: str
    task: str
    labels: list[str]
    author: str
    license: str
    category: str
    input: InputSpec = field(default_factory=InputSpec)
    output_format: str = "yolox"
    confidence_threshold: float = 0.3
    model_file: str = "model.onnx"
    format: str = "onnx"
    version: str = "1.0.0"
    min_app_version: str = "0.11.0"
    game: str = ""          # descriptive only, e.g. "Game X" - no logos/trademarks
    content_type: str = ""  # e.g. "gameplay", "sports", "music video"
    metrics: dict[str, float] = field(default_factory=dict)
    compliance: dict[str, bool] = field(default_factory=dict)
    sha256: dict[str, str] = field(default_factory=dict)  # filled by package.build()
    schema_version: int = SCHEMA_VERSION

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, folder: str | Path) -> Path:
        path = Path(folder) / MANIFEST_NAME
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        if not isinstance(data, dict):
            raise ManifestError(["Manifest must be a JSON object."])
        known = {f for f in cls.__dataclass_fields__}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ManifestError([f"Unknown manifest fields: {', '.join(unknown)}"])
        data = dict(data)
        inp = data.pop("input", {}) or {}
        if not isinstance(inp, dict):
            raise ManifestError(["'input' must be an object."])
        in_known = set(InputSpec.__dataclass_fields__)
        bad_in = sorted(set(inp) - in_known)
        if bad_in:
            raise ManifestError([f"Unknown input fields: {', '.join(bad_in)}"])
        try:
            m = cls(input=InputSpec(**inp), **data)
        except TypeError as exc:  # missing required fields
            raise ManifestError([f"Manifest is incomplete: {exc}"]) from exc
        m.validate()
        return m

    @classmethod
    def load(cls, folder: str | Path) -> "Manifest":
        path = Path(folder) / MANIFEST_NAME
        if not path.is_file():
            raise ManifestError([f"{MANIFEST_NAME} is missing."])
        if path.stat().st_size > 256_000:
            raise ManifestError([f"{MANIFEST_NAME} is too large."])
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ManifestError([f"{MANIFEST_NAME} is not valid JSON: {exc}"]) from exc
        return cls.from_dict(data)

    # ------------------------------------------------------------ validation
    def problems(self, require_compliance: bool = True) -> list[str]:
        p: list[str] = []
        if self.schema_version != SCHEMA_VERSION:
            p.append(f"Unsupported schema_version {self.schema_version} (expected {SCHEMA_VERSION}).")
        if not SLUG_RE.match(self.name or ""):
            p.append("name must be 3-64 characters: lowercase letters, digits and hyphens.")
        if not (self.display_name or "").strip() or len(self.display_name) > 80:
            p.append("display_name is required (max 80 characters).")
        if not (self.description or "").strip() or len(self.description) > 1000:
            p.append("description is required (max 1000 characters).")
        if not (self.author or "").strip() or len(self.author) > 80:
            p.append("author is required (max 80 characters).")
        if self.license not in LICENSES:
            p.append(f"license must be one of: {', '.join(LICENSES)} "
                     "(shared models must be usable by anyone).")
        if not CATEGORY_RE.match(self.category or ""):
            p.append("category must look like 'animals/horses': lowercase words and "
                     "hyphens, up to three levels.")
        if self.task not in TASKS:
            p.append(f"task must be one of: {', '.join(TASKS)}.")
        elif self.output_format not in TASKS[self.task][1]:
            allowed = ", ".join(sorted(TASKS[self.task][1]))
            p.append(f"output_format for {self.task} must be one of: {allowed}.")
        if self.format != "onnx" or not self.model_file.endswith(".onnx") or "/" in self.model_file \
                or "\\" in self.model_file:
            p.append("Only a single ONNX file in the package root is supported (format 'onnx').")
        labels = self.labels if isinstance(self.labels, list) else []
        if not labels or not all(isinstance(x, str) and x.strip() for x in labels):
            p.append("labels must be a non-empty list of names.")
        elif len(set(labels)) != len(labels):
            p.append("labels must be unique.")
        elif len(labels) > 1000:
            p.append("At most 1000 labels are supported.")
        try:
            if not 0.0 < float(self.confidence_threshold) < 1.0:
                p.append("confidence_threshold must be between 0 and 1.")
        except (TypeError, ValueError):
            p.append("confidence_threshold must be a number.")
        for v, key in ((self.version, "version"), (self.min_app_version, "min_app_version")):
            if not VERSION_RE.match(v or ""):
                p.append(f"{key} must look like 1.2.3.")
        for text, key in ((self.game, "game"), (self.content_type, "content_type")):
            if len(text or "") > 80:
                p.append(f"{key} is longer than 80 characters.")
        p.extend(_metric_problems(self.metrics))

        i = self.input
        if not (16 <= i.width <= 2048 and 16 <= i.height <= 2048):
            p.append("input width/height must be between 16 and 2048.")
        if i.channels not in (1, 3):
            p.append("input channels must be 1 or 3.")
        if i.layout not in LAYOUTS:
            p.append(f"input layout must be one of: {', '.join(sorted(LAYOUTS))}.")
        if i.color not in COLOR_ORDERS:
            p.append(f"input color must be one of: {', '.join(sorted(COLOR_ORDERS))}.")
        if i.normalize not in NORMALIZATIONS:
            p.append(f"input normalize must be one of: {', '.join(sorted(NORMALIZATIONS))}.")
        if self.output_format == "yolox" and (i.layout, i.color, i.normalize, i.channels) != \
                ("NCHW", "BGR", "0-255", 3):
            p.append("A YOLOX detector takes NCHW, BGR, 0-255, 3-channel input.")
        if self.task == "action_recognition":
            if not 2 <= i.frames <= 128:
                p.append("action_recognition needs input frames between 2 and 128.")
        elif i.frames != 1:
            p.append("input frames must be 1 unless task is action_recognition.")

        if require_compliance:
            missing = [k for k in COMPLIANCE_ITEMS if self.compliance.get(k) is not True]
            if missing:
                p.append("Confirm every publishing checklist item: " + ", ".join(missing) + ".")
        return p

    def validate(self, require_compliance: bool = False) -> None:
        problems = self.problems(require_compliance=require_compliance)
        if problems:
            raise ManifestError(problems)

    # -------------------------------------------------------------- helpers
    def expected_input_shape(self) -> list[int]:
        i = self.input
        if self.task == "action_recognition":
            # N, T, C, H, W  or  N, T, H, W, C
            return ([1, i.frames, i.channels, i.height, i.width] if i.layout == "NCHW"
                    else [1, i.frames, i.height, i.width, i.channels])
        return ([1, i.channels, i.height, i.width] if i.layout == "NCHW"
                else [1, i.height, i.width, i.channels])

    def hub_tags(self) -> list[str]:
        tags = [HUB_TAG, "onnx", f"vh-{self.task.replace('_', '-')}"]
        tags += category_tags(self.category)
        if self.content_type:
            tags.append("vh-" + re.sub(r"[^a-z0-9]+", "-", self.content_type.lower()).strip("-"))
        return tags

    def found_sentence(self) -> str:
        """"Found 11 of 11 on frames it never trained on" — or empty."""
        found, expected = self.metrics.get("heldout_found"), self.metrics.get("heldout_expected")
        if isinstance(found, (int, float)) and isinstance(expected, (int, float)) and expected > 0:
            return f"Found {int(found)} of {int(expected)} on frames it never trained on"
        return ""


def _metric_problems(metrics: Any) -> list[str]:
    if not isinstance(metrics, dict):
        return ["metrics must be an object of numbers."]
    p = []
    unknown = sorted(set(metrics) - set(METRIC_KEYS))
    if unknown:
        p.append(f"Unknown metrics: {', '.join(unknown)}")
    for k, v in metrics.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            p.append(f"metric {k} must be a non-negative number.")
    return p


def category_tags(category: str) -> list[str]:
    """``animals/horses`` → ``vh-cat-animals``, ``vh-cat-animals--horses``.

    One tag per level, so the site can filter on any level with a plain tag
    match. ``--`` separates levels because a slug segment never contains it.
    """
    if not CATEGORY_RE.match(category or ""):
        return []
    parts = category.split("/")
    return [CATEGORY_TAG_PREFIX + "--".join(parts[:i]) for i in range(1, len(parts) + 1)]


def category_from_tags(tags: list[str]) -> str:
    """The deepest category a model's tags name, as a slug path, or ``""``."""
    best = ""
    for t in tags or []:
        if t.startswith(CATEGORY_TAG_PREFIX):
            path = t[len(CATEGORY_TAG_PREFIX):].replace("--", "/")
            if CATEGORY_RE.match(path) and path.count("/") >= best.count("/") and len(path) > len(best):
                best = path
    return best
