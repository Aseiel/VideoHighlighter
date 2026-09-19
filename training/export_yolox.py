"""Turn a trained checkpoint into a model the app can actually load.

``train_yolox_run.train`` leaves a ``.pth``, which is a PyTorch object and
nothing the detector backend knows how to open. This module takes it the rest
of the way: ONNX, then OpenVINO IR, then into ``models/custom/`` with the
sidecar that gives the classes their names.

Three details decide whether the result works, and each one fails silently when
it is wrong.

**The decode must not be baked in.** ``YoloxOpenVINODetector._decode`` applies
the grid and stride arithmetic itself, so the export has to emit raw grid
predictions — ``[1, n_anchors, 5 + num_classes]`` of ``(cx, cy, w, h, obj,
*classes)`` still in grid units. YOLOX's head decodes in inference mode by
default, and an export made that way has the *same shape*: nothing raises, the
IR loads, and every box lands in the wrong place. Hence
``head.decode_in_inference = False`` before tracing, and the shape check after.

**The names live beside the model, not in it.** A raw-grid export carries no
class-name metadata at all, so ``modules.vision.detection_backend.names_from_model``
finds nothing. Without ``labels.json`` next to the IR the app logs "No class
names for custom model" and quietly falls back to the 80-class detector — which
reads as a bad model rather than a missing file.

**The input size travels with the model.** The IR records it, but a caller that
guesses wrong letterboxes to the wrong shape and gets plausible nonsense. It is
written into the sidecar directory as well so nothing has to infer it.

Standalone::

    python -m training.export_yolox path/to/yolox_tiny.pth
    python -m training.export_yolox path/to/yolox_tiny.pth --dest models/custom
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from typing import Optional, Sequence

# Opset 11 is what YOLOX's own export uses and what the released ONNX models
# were made with, so it is the version the decoder has been exercised against.
ONNX_OPSET = 11

# Names matched to the official exports. Nothing in this app looks them up by
# name, but a mismatch makes an exported model harder to compare against a
# reference one when something is wrong.
INPUT_NAME = "images"
OUTPUT_NAME = "output"

# Where the app looks for a user's own model.
DEFAULT_DEST = os.path.join("models", "custom")


@dataclass
class ExportResult:
    """Where everything ended up."""

    xml_path: str
    bin_path: str
    onnx_path: str
    labels_path: str
    class_names: list
    input_size: tuple
    anchors: int

    def summary(self) -> str:
        h, w = self.input_size
        return (f"{len(self.class_names)} class(es) at {w}x{h}, "
                f"{self.anchors} anchors -> {os.path.basename(self.xml_path)}")


def load_checkpoint(checkpoint_path: str) -> dict:
    """Read a checkpoint written by :func:`training.train_yolox_run.train`.

    ``weights_only=False`` is required for a checkpoint carrying its own class
    names and geometry rather than bare tensors. That is safe here and only
    here: this file was produced by this application, on this machine, minutes
    ago. Never widen it to checkpoints from elsewhere — unpickling arbitrary
    files executes arbitrary code.
    """
    import torch

    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" not in data:
        raise ValueError(
            f"{checkpoint_path} has no 'model' key — is it a training checkpoint?")
    return data


def export_onnx(checkpoint_path: str, onnx_path: str,
                class_names: Optional[Sequence] = None,
                size: Optional[str] = None,
                input_size: Optional[tuple] = None) -> tuple:
    """Trace the trained network to ONNX, raw-grid. → (path, class_names, size, anchors)

    Everything not passed is read from the checkpoint, so a normal call needs
    only the two paths.
    """
    import torch

    from training.train_yolox_run import _build_exp

    checkpoint = load_checkpoint(checkpoint_path)
    class_names = list(class_names or checkpoint.get("class_names") or [])
    if not class_names:
        raise ValueError("No class names in the checkpoint; pass class_names=")
    size = size or checkpoint.get("size") or "tiny"
    input_size = tuple(input_size or checkpoint.get("image_size") or (416, 416))

    exp = _build_exp("unused", num_classes=len(class_names), size=size,
                     image_size=input_size, batch_size=1, workers=0, epochs=1)
    model = exp.get_model()
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # The line this whole module exists around. Leave it True and the export
    # still traces, still loads, still produces boxes — in the wrong places.
    model.head.decode_in_inference = False

    dummy = torch.zeros(1, 3, input_size[0], input_size[1])
    os.makedirs(os.path.dirname(os.path.abspath(onnx_path)), exist_ok=True)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        dynamo=False,      # the tracer the released exports were made with
    )

    with torch.no_grad():
        traced_out = model(dummy)
    anchors = int(traced_out.shape[1])
    _check_raw_grid(traced_out, len(class_names), input_size)
    return onnx_path, class_names, size, anchors


def _check_raw_grid(output, num_classes: int, input_size: tuple) -> None:
    """Fail loudly if the export is not the layout the decoder expects.

    Both halves matter. The width proves the class count lines up with the
    sidecar about to be written; the anchor count proves the strides are the
    (8, 16, 32) the decoder assumes, which is the part a changed input size or
    a decoded export would quietly break.
    """
    if output.ndim != 3:
        raise ValueError(f"expected [1, anchors, 5+C], got shape {tuple(output.shape)}")
    width = int(output.shape[2])
    if width != 5 + num_classes:
        raise ValueError(
            f"output width {width} does not match {num_classes} classes "
            f"(expected {5 + num_classes}: cx, cy, w, h, obj, *classes)")
    h, w = input_size
    expected = sum((h // s) * (w // s) for s in (8, 16, 32))
    if int(output.shape[1]) != expected:
        raise ValueError(
            f"{output.shape[1]} anchors, expected {expected} for a {w}x{h} input "
            f"at strides (8, 16, 32) — the decoder's grid would not line up")


def convert_to_ir(onnx_path: str, xml_path: str) -> str:
    """ONNX → OpenVINO IR, fp16, matching ``tools/get_yolox_model.py``."""
    import openvino as ov

    os.makedirs(os.path.dirname(os.path.abspath(xml_path)), exist_ok=True)
    model = ov.convert_model(str(onnx_path))
    ov.save_model(model, str(xml_path), compress_to_fp16=True)
    return xml_path


def install(checkpoint_path: str,
            dest_dir: str = DEFAULT_DEST,
            name: Optional[str] = None,
            keep_onnx: bool = True) -> ExportResult:
    """The whole chain: checkpoint → ONNX → IR → ``dest_dir`` with its sidecar.

    This is the one call the GUI makes when a training run finishes.
    """
    checkpoint = load_checkpoint(checkpoint_path)
    size = checkpoint.get("size") or "tiny"
    name = name or f"yolox_{size}_custom"
    # Each model gets its own directory. The sidecar's name is fixed
    # (``labels.json``, because that is what the app reads), so installing a
    # second model into a shared directory silently overwrites the first one's
    # class names — leaving a model that loads and reports every detection
    # under somebody else's labels. Found the hard way: a test run replaced the
    # sidecar of an unrelated model already sitting there.
    dest_dir = os.path.join(os.path.abspath(dest_dir), name)
    os.makedirs(dest_dir, exist_ok=True)

    onnx_path = os.path.join(dest_dir, f"{name}.onnx")
    onnx_path, class_names, size, anchors = export_onnx(checkpoint_path, onnx_path)

    xml_path = os.path.join(dest_dir, f"{name}.xml")
    convert_to_ir(onnx_path, xml_path)
    bin_path = os.path.splitext(xml_path)[0] + ".bin"

    input_size = tuple(checkpoint.get("image_size") or (416, 416))
    labels_path = write_sidecar(dest_dir, class_names, input_size, name)

    if not keep_onnx:
        try:
            os.remove(onnx_path)
        except OSError:
            pass

    result = ExportResult(
        xml_path=xml_path, bin_path=bin_path, onnx_path=onnx_path,
        labels_path=labels_path, class_names=class_names,
        input_size=input_size, anchors=anchors,
    )
    print(f"[export] {result.summary()}")
    return result


def write_sidecar(dest_dir: str, class_names: Sequence,
                  input_size: tuple, name: str) -> str:
    """Write ``labels.json``, plus the geometry, beside the IR.

    ``labels.json`` keeps the bare list the app already reads — this is not the
    place to invent a richer format, since the reader predates it. The extra
    detail goes in a second file, so a caller that wants the input size can
    have it without anything having to parse the IR.
    """
    labels_path = os.path.join(dest_dir, "labels.json")
    with open(labels_path, "w", encoding="utf-8") as fh:
        json.dump([str(n) for n in class_names], fh, indent=2)

    with open(os.path.join(dest_dir, f"{name}.meta.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "class_names": [str(n) for n in class_names],
            "input_size": [int(input_size[0]), int(input_size[1])],
            "decode_in_inference": False,
            "strides": [8, 16, 32],
        }, fh, indent=2)
    return labels_path


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Export a trained YOLOX checkpoint for the app")
    parser.add_argument("checkpoint", help="the .pth written by the training run")
    parser.add_argument("--dest", default=DEFAULT_DEST,
                        help=f"where to install (default: {DEFAULT_DEST})")
    parser.add_argument("--name", default=None, help="base name for the files")
    parser.add_argument("--no-onnx", action="store_true",
                        help="delete the intermediate ONNX once converted")
    args = parser.parse_args(argv)

    result = install(args.checkpoint, dest_dir=args.dest, name=args.name,
                     keep_onnx=not args.no_onnx)
    print(f"  IR     : {result.xml_path}")
    print(f"  labels : {result.labels_path}")
    print(f"  classes: {result.class_names}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
