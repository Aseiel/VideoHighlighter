"""Fetch the stock YOLOX detectors (Apache-2.0) and convert them to OpenVINO IR.

The official Megvii release ONNX exports are downloaded and converted into
``models/yolox/yolox_<size>.xml``, which is where
``modules.vision.detection_backend.find_default_yolox_ir`` looks.

Both the CLI (``tools/get_yolox_model.py``, run by the release build) and the
app's first-run path (``build_object_detector(auto_install=True)``) call this,
so a source checkout works without a separate setup step.

The exports are made WITHOUT ``decode_in_inference``: they emit raw grid
predictions, which is exactly what ``YoloxOpenVINODetector._decode`` expects.
Do not swap in an export with decoding baked in or detections will be garbage.

Licensing: YOLOX code and released weights are Apache-2.0
(github.com/Megvii-BaseDetection/YOLOX). Nothing here touches an AGPL detector.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path
from typing import Callable, Iterable

RELEASE_BASE = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
)

# size -> (onnx filename, input resolution the export was made at)
SIZES = {
    "nano": ("yolox_nano.onnx", 416),
    "tiny": ("yolox_tiny.onnx", 416),
    "s": ("yolox_s.onnx", 640),
    "m": ("yolox_m.onnx", 640),
    "l": ("yolox_l.onnx", 640),
    "x": ("yolox_x.onnx", 640),
}

# What a fresh install gets: tiny for the live/person loops, s for offline
# object detection. Big enough to be useful, small enough to download quickly.
DEFAULT_SIZES = ("tiny", "s")

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "yolox"
CACHE_DIR = MODEL_DIR / "onnx"


def ir_path(size: str) -> Path:
    return MODEL_DIR / f"yolox_{size}.xml"


def find_onnx(prefer: str = "large") -> str | None:
    """The downloaded ONNX export for ONNX Runtime, mirroring
    ``detection_backend.find_default_yolox_ir``: ``prefer="small"`` picks the
    fastest installed size, anything else the most accurate."""
    present = [size for size, (name, _res) in SIZES.items() if (CACHE_DIR / name).exists()]
    if not present:
        return None
    chosen = present[0] if prefer == "small" else present[-1]
    return str(CACHE_DIR / SIZES[chosen][0])


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
        raise ValueError(f"Unknown YOLOX size(s): {unknown}. Choose from: {', '.join(SIZES)}")
    out = []
    for size in sizes:
        xml = ir_path(size)
        if not xml.exists():
            onnx_name, _resolution = SIZES[size]
            onnx_path = CACHE_DIR / onnx_name
            _download(f"{RELEASE_BASE}/{onnx_name}", onnx_path, log)
            _convert(onnx_path, xml, log)
        out.append(str(xml))
    return out
