"""Fetch the official RTMPose keypoint models (Apache-2.0) and convert them to IR.

The app has had no pose estimation since the AGPL YOLO package was dropped: YOLOX
detects, and has no keypoint head. RTMPose fills that hole without reopening the
licensing question — mmpose is Apache-2.0, and the exports below are OpenMMLab's
own released SDK bundles, so nothing here is trained or exported with an AGPL
toolkit (see the hard licensing gate in CLAUDE.md).

Downloads OpenMMLab's ONNX SDK zip, lifts ``end2end.onnx`` out of it, and
converts to ``models/rtmpose/rtmpose_<size>.xml`` — where
``modules.vision.pose_backend.find_default_rtmpose_ir`` looks. Mirrors
``modules/vision/yolox_models.py`` deliberately: same cache-then-convert shape, same
"cheap to call again" contract, so the two model installers stay readable as a
pair.

WHICH VARIANT, AND WHY IT MATTERS
    The ``body7`` models are the 17-keypoint COCO layout. That is not a
    preference — the cropper indexes keypoints positionally (``[9, 10, 11, 12,
    15, 16]`` for wrists, hips and ankles in modules/crop/pose.py), so a model
    with a different keypoint order returns numbers that are silently wrong
    rather than absent. A 26-keypoint halpe or 133-keypoint wholebody export
    will load and produce garbage. Keep the layout, or fix every index with it.

TOP-DOWN, WHICH IS THE WHOLE DESIGN
    RTMPose takes ONE person box and returns that person's keypoints. It cannot
    find someone the detector missed. That is why modules/crop/config.py drops
    DETECTOR_SCORE_FLOOR to 0.05: YOLOX proposes generously, RTMPose decides
    which proposals have a real skeleton behind them.
"""
from __future__ import annotations

import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Iterable

RELEASE_BASE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk"

# size -> (SDK zip filename, COCO AP). All are body7 / 17-keypoint / 256x192.
SIZES = {
    "s": ("rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip", 72.2),
    "m": ("rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip", 75.3),
    "l": ("rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.zip", 76.3),
}

# One size by default, unlike YOLOX: pose runs on a sparse sample of frames for
# counting and activity, not on every frame of the write loop, so the accurate
# model is affordable and the fast one has no job to do.
DEFAULT_SIZES = ("m",)

# The input the exports were made at, (w, h) — and the SimCC decode constants
# from the bundle's own pipeline.json. pose_backend reads these rather than
# hardcoding them, so a future export at 384x288 needs one edit, here.
INPUT_SIZE = (192, 256)
SIMCC_SPLIT_RATIO = 2.0
MEAN = (123.675, 116.28, 103.53)
STD = (58.395, 57.12, 57.375)
BBOX_PADDING = 1.25

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "rtmpose"
CACHE_DIR = MODEL_DIR / "onnx"


def ir_path(size: str) -> Path:
    return MODEL_DIR / f"rtmpose_{size}.xml"


def _download(url: str, dest: Path, log: Callable[[str], None]) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    log(f"⬇️ Downloading {dest.name}…")

    def _hook(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            done = min(blocks * block_size, total)
            sys.stdout.write(
                f"\r  {done // (1 << 20)}MB / {total // (1 << 20)}MB "
                f"({done * 100 // total}%)")
            sys.stdout.flush()

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp, reporthook=_hook)
    print()
    tmp.replace(dest)


def _extract_onnx(zip_path: Path, dest: Path, log: Callable[[str], None]) -> None:
    """Lift end2end.onnx out of the SDK bundle.

    The zip also carries pipeline.json, detail.json and two sample renders. We
    keep only the graph: the constants those files describe are mirrored above,
    and an unused 44KB JPEG in models/ is the kind of thing that ends up in a
    release bundle by accident.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return
    with zipfile.ZipFile(zip_path) as z:
        member = next((n for n in z.namelist() if n.endswith("end2end.onnx")), None)
        if member is None:
            raise ValueError(f"{zip_path.name} contains no end2end.onnx — layout changed?")
        log(f"📦 Extracting {member.rsplit('/', 1)[-1]} → {dest.name}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with z.open(member) as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)


def _convert(onnx_path: Path, xml_path: Path, log: Callable[[str], None]) -> None:
    import openvino as ov  # lazy

    log(f"⚙️ Converting {onnx_path.name} → {xml_path.name}")
    model = ov.convert_model(str(onnx_path))
    ov.save_model(model, str(xml_path), compress_to_fp16=True)


def install(sizes: Iterable[str] = DEFAULT_SIZES,
            log: Callable[[str], None] = print) -> list[str]:
    """Make sure each requested size exists as IR. Returns the .xml paths.

    Already-converted sizes are skipped, so this is cheap to call again.
    Raises ``ValueError`` for an unknown size; network and conversion errors
    propagate so the caller can say what went wrong.
    """
    sizes = [s.lower() for s in sizes]
    unknown = [s for s in sizes if s not in SIZES]
    if unknown:
        raise ValueError(f"Unknown RTMPose size(s): {unknown}. Choose from: {', '.join(SIZES)}")
    out = []
    for size in sizes:
        xml = ir_path(size)
        if not xml.exists():
            zip_name, _ap = SIZES[size]
            zip_path = CACHE_DIR / zip_name
            onnx_path = CACHE_DIR / f"rtmpose_{size}.onnx"
            _download(f"{RELEASE_BASE}/{zip_name}", zip_path, log)
            _extract_onnx(zip_path, onnx_path, log)
            _convert(onnx_path, xml, log)
        out.append(str(xml))
    return out
