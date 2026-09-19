"""Build and check model packages.

A package is a folder with exactly these files:

    model.onnx            the model (single file, no external data, standard ops)
    videohighlighter.json manifest (see manifest.py)
    README.md             model card (generated if missing)
    LICENSE               optional

Anything else - clips, screenshots, audio, archives, pickles - is rejected.
The same checks run before upload *and* after download, so a model that
reaches the user's pipeline has always passed them.

**What leaves the machine is the model, never the material it was made from.**
Training on footage and handing that footage to other people are different
acts, and the second is where the copyright risk is. So the package has no
place for media at all.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .manifest import MANIFEST_NAME, InputSpec, Manifest, ManifestError

ALLOWED_FILES = {"model.onnx", MANIFEST_NAME, "README.md", "LICENSE", ".gitattributes"}
UPLOAD_PATTERNS = ["model.onnx", MANIFEST_NAME, "README.md", "LICENSE"]
MAX_MODEL_BYTES = 250 * 1024 * 1024
MAX_TEXT_BYTES = 512 * 1024
ONNX_MAGIC_HINT = b"\x08"  # ModelProto starts with field 1 (ir_version, varint)


@dataclass
class CheckReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def text(self) -> str:
        lines = [f"✔ {m}" for m in self.info]
        lines += [f"⚠ {m}" for m in self.warnings]
        lines += [f"✖ {m}" for m in self.errors]
        return "\n".join(lines)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- file layout
def check_layout(folder: Path, report: CheckReport, extra_allowed: frozenset[str] = frozenset()) -> None:
    if not folder.is_dir():
        report.errors.append(f"{folder} is not a folder.")
        return
    for p in folder.rglob("*"):
        rel = p.relative_to(folder)
        if any(part.startswith(".git") and part != ".gitattributes" for part in rel.parts):
            continue  # local git metadata is never uploaded
        if p.is_symlink():
            report.errors.append(f"Symbolic links are not allowed: {rel}")
        elif p.is_dir():
            report.errors.append(f"Subfolders are not allowed: {rel}/")
        elif str(rel) not in ALLOWED_FILES | extra_allowed:
            report.errors.append(
                f"Not allowed in a model package: {rel} "
                "(only model.onnx, videohighlighter.json, README.md and LICENSE; "
                "never include clips, images, audio or training data)")
    model = folder / "model.onnx"
    if not model.is_file():
        report.errors.append("model.onnx is missing.")
    elif model.stat().st_size > MAX_MODEL_BYTES:
        report.errors.append(f"model.onnx is larger than {MAX_MODEL_BYTES // 2**20} MB.")
    elif model.stat().st_size < 64:
        report.errors.append("model.onnx is empty or truncated.")
    for name in ("README.md", "LICENSE", MANIFEST_NAME):
        f = folder / name
        if f.is_file() and f.stat().st_size > MAX_TEXT_BYTES:
            report.errors.append(f"{name} is too large.")


# ------------------------------------------------------------ onnx checks
def _dims(value_info) -> list:
    return [d if isinstance(d, int) else None for d in value_info.shape]


def _shape_matches(actual: list, expected: list[int]) -> bool:
    if len(actual) != len(expected):
        return False
    return all(a is None or a == e or (i == 0) for i, (a, e) in enumerate(zip(actual, expected)))


def detector_layout(shape) -> str:
    """``"yolox"``, ``"transposed"`` or ``"unknown"`` for a detector output shape.

    YOLOX emits ``[1, anchors, 5 + labels]``; the transposed
    ``[1, 4 + labels, anchors]`` layout is the AGPL toolkit's. The anchor axis
    always dwarfs the channel axis, which is what tells them apart — the same
    rule ``modules.vision.detection_backend.create_detector`` uses to pick a decoder.
    """
    if len(shape) != 3:
        return "unknown"
    return "yolox" if shape[1] > shape[2] else "transposed"


def check_onnx(folder: Path, manifest: Manifest, report: CheckReport) -> None:
    """Load the model with ONNX Runtime and run one dummy inference.

    ONNX Runtime only executes the graph's standard operators, so loading an
    ONNX file cannot run arbitrary code (unlike .pt/.pkl pickles).
    """
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        report.errors.append("onnxruntime is required to check models.")
        return

    model_path = folder / manifest.model_file
    with open(model_path, "rb") as fh:
        head = fh.read(1)
    if head != ONNX_MAGIC_HINT:
        report.warnings.append("model.onnx has an unusual header; continuing with a full load.")

    # Reject external-data models: all weights must live inside model.onnx.
    try:
        import onnx
        proto = onnx.load(str(model_path), load_external_data=False)
        from onnx.external_data_helper import uses_external_data
        if any(uses_external_data(t) for t in proto.graph.initializer):
            report.errors.append("model.onnx references external weight files; export as one file.")
            return
        custom = sorted({n.domain for n in proto.graph.node
                         if n.domain not in ("", "ai.onnx", "ai.onnx.ml", "com.microsoft")})
        if custom:
            report.errors.append(f"Custom operator domains are not supported: {', '.join(custom)}")
            return
        report.info.append(f"ONNX opset {max((o.version for o in proto.opset_import if o.domain in ('', 'ai.onnx')), default='?')}")
        del proto
    except ImportError:
        report.warnings.append("Package 'onnx' not installed; skipped external-data check.")
    except Exception as exc:  # noqa: BLE001 - any parse failure is a hard error
        report.errors.append(f"model.onnx could not be parsed: {exc}")
        return

    so = ort.SessionOptions()
    so.log_severity_level = 3
    try:
        sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"ONNX Runtime cannot load the model: {exc}")
        return

    inputs = sess.get_inputs()
    if len(inputs) != 1:
        report.errors.append(f"Model must have exactly one input (has {len(inputs)}).")
        return
    inp = inputs[0]
    expected = manifest.expected_input_shape()
    actual = _dims(inp)
    if not _shape_matches(actual, expected):
        report.errors.append(
            f"Input shape {actual} does not match the manifest {expected} "
            f"(check width/height/channels/layout{'/frames' if manifest.task == 'action_recognition' else ''}).")
        return
    if inp.type not in ("tensor(float)", "tensor(float16)", "tensor(uint8)"):
        report.errors.append(f"Unsupported input type {inp.type}.")
        return
    report.info.append(f"Input {inp.name} {expected} {inp.type}")

    dtype = {"tensor(float)": np.float32, "tensor(float16)": np.float16,
             "tensor(uint8)": np.uint8}[inp.type]
    dummy = np.zeros(expected, dtype=dtype)
    try:
        outputs = sess.run(None, {inp.name: dummy})
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"Test inference failed: {exc}")
        return
    report.info.append("Test inference ran on CPU")

    n = len(manifest.labels)
    out = outputs[0]
    if manifest.task == "object_detection":
        layout = detector_layout(out.shape)
        if layout == "transposed":
            report.errors.append(
                f"Detector output {list(out.shape)} is in the transposed [1, 4 + labels, N] "
                "layout of an AGPL training toolkit. Only detectors trained in "
                "VideoHighlighter (YOLOX) can be shared: a model trained with that "
                "toolkit carries its licence.")
            return
        if layout != "yolox" or out.shape[2] != 5 + n:
            report.errors.append(
                f"Detector output {list(out.shape)} does not match the YOLOX format "
                f"[1, N, {5 + n}] for {n} labels.")
            return
    else:
        if out.ndim != 2 or out.shape[1] != n:
            report.errors.append(
                f"Output {list(out.shape)} does not match {n} labels (expected [1, {n}]).")
            return
    report.info.append(f"Output {list(out.shape)} matches {n} label(s)")


# ------------------------------------------------------------------ public api
def check_package(folder: str | Path, require_compliance: bool = True,
                  verify_hashes: bool = True,
                  extra_allowed: frozenset[str] = frozenset()) -> tuple[CheckReport, Manifest | None]:
    folder = Path(folder)
    report = CheckReport()
    check_layout(folder, report, extra_allowed)
    try:
        manifest = Manifest.load(folder)
    except ManifestError as exc:
        report.errors.extend(exc.problems)
        return report, None
    report.errors.extend(manifest.problems(require_compliance=require_compliance))

    model = folder / manifest.model_file
    if verify_hashes and model.is_file():
        expected = manifest.sha256.get(manifest.model_file)
        if not expected:
            report.errors.append("Manifest has no sha256 for model.onnx; rebuild the package.")
        elif sha256_file(model) != expected:
            report.errors.append("model.onnx does not match the sha256 in the manifest.")
        else:
            report.info.append("Checksum verified")

    if report.ok:
        check_onnx(folder, manifest, report)
    return report, manifest


def build_package(model_file: str | Path, manifest: Manifest, out_dir: str | Path,
                  license_file: str | Path | None = None) -> tuple[Path, CheckReport]:
    """Create a clean package folder from a trained model + filled-in manifest."""
    from .card import render_model_card

    model_file = Path(model_file)
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; choose an empty folder.")
    if model_file.suffix.lower() != ".onnx":
        raise ValueError("Only .onnx models can be published. Export your model to ONNX first.")
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_file, out / "model.onnx")
    if license_file:
        shutil.copyfile(license_file, out / "LICENSE")

    manifest.model_file = "model.onnx"
    manifest.format = "onnx"
    manifest.sha256 = {"model.onnx": sha256_file(out / "model.onnx")}
    manifest.save(out)
    (out / "README.md").write_text(render_model_card(manifest), encoding="utf-8")

    report, _ = check_package(out, require_compliance=True)
    return out, report


def draft_for_trained_detector(onnx_path: str | Path, author: str = "",
                               metrics: dict | None = None) -> Manifest:
    """A manifest for a detector the Train tab just exported, with every
    technical field already right.

    ``training.export_yolox`` writes ``labels.json`` and ``<name>.meta.json``
    beside the ONNX. From those, the task, labels, input size, colour order,
    normalisation and output format are known — a person sharing a model they
    just trained should only have to say what it is, where it belongs, and tick
    the checklist. Name, description and category are left for them.
    """
    onnx_path = Path(onnx_path)
    folder = onnx_path.parent
    labels: list[str] = []
    labels_file = folder / "labels.json"
    if labels_file.is_file():
        data = json.loads(labels_file.read_text(encoding="utf-8"))
        if isinstance(data, list):
            labels = [str(x) for x in data]
    width = height = 416
    meta_file = folder / f"{onnx_path.stem}.meta.json"
    if meta_file.is_file():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        size = meta.get("input_size") or [416, 416]
        height, width = int(size[0]), int(size[1])
        labels = labels or [str(x) for x in meta.get("class_names", [])]
    return Manifest(
        name="", display_name="", description="",
        task="object_detection", labels=labels, author=author,
        license="apache-2.0", category="",
        input=InputSpec(width=width, height=height, channels=3, layout="NCHW",
                        color="BGR", normalize="0-255"),
        output_format="yolox", confidence_threshold=0.3,
        metrics={k: v for k, v in (metrics or {}).items() if v is not None},
    )
