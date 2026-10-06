"""Object detection on an Intel NPU, leaving the CPU and GPU to everything else.

WHY THIS EXISTS
    Core Ultra processors carry an NPU ("Intel AI Boost") that OpenVINO exposes
    as the device ``"NPU"``. Running the YOLOX detector there costs about 1 ms of
    processor time per frame; run on the processor itself the same inference
    costs 30-260 ms of it depending on model size, and on a GPU it competes with
    whatever that GPU is rendering or encoding — a game being recorded, say.

    Nothing about the model changes. Pre-processing, decoding and NMS are
    ``detection_backend.YoloxOpenVINODetector``'s, inherited unchanged, and the
    NPU's results match the processor's: the same boxes, with confidences within
    about 0.01 (the NPU computes in FP16). What this module adds is what the
    plain detector has no reason to care about:

    - **No silent fallback.** The base detector quietly retries on the CPU when
      a device refuses the model, so "running on the NPU" could never be known.
      Here ``.device`` is where it actually runs, and whether to fall back at
      all is the caller's choice (``fallback=``).
    - **Static shapes.** The NPU compiles static shapes only, so a model
      exported with a dynamic batch or size is pinned to 1x3xHxW first.
    - **Compiled-model cache.** The compile is cached under the user data dir,
      so the next start skips most of it.

    YOLOX layout only — the same rule ``model_hub/`` holds shared detectors to.

USAGE
    from modules.vision.npu_detector import build_npu_detector
    det = build_npu_detector()                 # stock YOLOX, smallest installed
    det = build_npu_detector("models/custom/my_model/my_model.xml")
    print(det.device)                          # "NPU", or the fallback it took
    for d in det.detect(frame_bgr): ...
"""
from __future__ import annotations

import os
from typing import Callable, Sequence

from modules.vision.detection_backend import (
    DEFAULT_MODEL_DIR,
    INPUT_SIZE,
    NMS_THR,
    SCORE_THR,
    STRIDES,
    YoloxOpenVINODetector,
    find_default_yolox_ir,
    load_class_names,
)

NPU = "NPU"
COCO_LABELS = "yolo_objects_labels.json"


def npu_device(core=None) -> str | None:
    """The OpenVINO name of this machine's NPU ("NPU"), or None when there is
    none — or when OpenVINO itself is missing, which amounts to the same."""
    try:
        if core is None:
            import openvino as ov  # lazy
            core = ov.Core()
        return next((d for d in core.available_devices
                     if d == NPU or d.startswith(NPU + ".")), None)
    except Exception:
        return None


def npu_name(core=None) -> str | None:
    """Human-readable NPU name ("Intel(R) AI Boost"), or None without one."""
    try:
        if core is None:
            import openvino as ov  # lazy
            core = ov.Core()
        dev = npu_device(core)
        return str(core.get_property(dev, "FULL_DEVICE_NAME")) if dev else None
    except Exception:
        return None


def default_cache_dir() -> str:
    from modules.system.app_paths import user_data_dir  # lazy
    return os.path.join(user_data_dir(), "cache", "openvino")


def expected_anchors(input_size: tuple[int, int]) -> int:
    """Rows a raw-grid YOLOX export emits for an input of (h, w)."""
    h, w = input_size
    return sum((h // s) * (w // s) for s in STRIDES)


class NpuYoloxDetector(YoloxOpenVINODetector):
    """YOLOX on the NPU. Implements the Detector protocol.

    Raises ``ValueError`` for a model this decoder would misread (wrong layout,
    class count that disagrees with ``class_names``) and ``RuntimeError`` when
    neither ``device`` nor any ``fallback`` device will compile it.
    """

    def __init__(self, ov_model_xml: str, class_names: Sequence[str],
                 device: str = NPU, fallback: Sequence[str] = ("CPU",),
                 score_thr: float = SCORE_THR, nms_thr: float = NMS_THR,
                 cache_dir: str | None = None,
                 log: Callable[[str], None] = print):
        import openvino as ov  # lazy
        self.model_path = str(ov_model_xml)
        self.class_names = list(class_names)
        self.score_thr = float(score_thr)
        self.nms_thr = float(nms_thr)
        core = ov.Core()
        model = core.read_model(str(ov_model_xml))
        self.input_size = _pin_static_input(model)
        _check_yolox_output(model, len(self.class_names), self.input_size)
        model = _uint8_hwc_input(model)

        config = {"PERFORMANCE_HINT": "LATENCY"}
        cache = default_cache_dir() if cache_dir is None else cache_dir
        if cache:
            try:
                os.makedirs(cache, exist_ok=True)
                config["CACHE_DIR"] = cache
            except OSError as e:
                print(f"[npu] compile cache disabled ({cache}): {e}")

        errors = []
        for dev in dict.fromkeys([device, *fallback]):  # de-dup, keep order
            try:
                self._compiled = core.compile_model(model, dev, config)
                break
            except Exception as e:  # noqa: BLE001 - every plugin raises its own type
                errors.append(f"{dev}: {str(e).strip().splitlines()[0] if str(e).strip() else e!r}")
        else:
            raise RuntimeError("No device would compile the detector — " + "; ".join(errors))
        if errors:
            log(f"⚠️ {device} could not run the detector ({errors[0]}); using {dev} instead")
        self.device = _execution_devices(self._compiled, dev)
        self._out = self._compiled.output(0)

    def _preprocess(self, frame_bgr):
        """The base letterbox, kept in uint8: the same pixels reach the network
        (the cast now happens on the device, see ``_uint8_hwc_input``), for about
        a quarter of the processor time on a 1440p frame."""
        import cv2  # lazy
        in_h, in_w = self.input_size
        h0, w0 = frame_bgr.shape[:2]
        r = min(in_h / h0, in_w / w0)
        nh, nw = int(round(h0 * r)), int(round(w0 * r))
        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        padded = cv2.copyMakeBorder(resized, 0, in_h - nh, 0, in_w - nw,
                                    cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return padded[None], r


def build_npu_detector(model_path: str = "", class_names: Sequence[str] | None = None,
                       prefer: str = "small", device: str = NPU,
                       fallback: Sequence[str] = ("CPU",),
                       score_thr: float = SCORE_THR, nms_thr: float = NMS_THR,
                       cache_dir: str | None = None,
                       log: Callable[[str], None] = print) -> NpuYoloxDetector:
    """An NPU detector for the stock YOLOX model, or for ``model_path``.

    With no ``model_path`` the stock model under models/yolox/ is used —
    ``prefer="small"`` the fastest installed size, anything else the most
    accurate — with COCO class names. A custom model (.xml or .onnx) takes its
    names from ``class_names``, else its embedded metadata, else the
    ``labels.json`` beside it, the same order the app uses.
    """
    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Detector model not found: {model_path}")
        from modules.system.app_paths import object_model_names  # lazy
        names = list(class_names or object_model_names(model_path))
        if not names and _is_stock_model(model_path):
            names = load_class_names(COCO_LABELS)
    else:
        model_path = find_default_yolox_ir(prefer=prefer) or ""
        if not model_path:
            raise FileNotFoundError(
                "No YOLOX model under models/yolox/ — run: python tools/get_yolox_model.py")
        names = list(class_names or load_class_names(COCO_LABELS))
    if not names:
        raise ValueError(f"No class names for {os.path.basename(model_path)} "
                         "(no embedded names and no labels.json beside it)")
    return NpuYoloxDetector(model_path, names, device=device, fallback=fallback,
                            score_thr=score_thr, nms_thr=nms_thr,
                            cache_dir=cache_dir, log=log)


def _is_stock_model(model_path: str) -> bool:
    """A file tools/get_yolox_model.py installed (COCO-trained, no names inside)."""
    try:
        path = os.path.normcase(os.path.realpath(model_path))
        root = os.path.normcase(os.path.realpath(DEFAULT_MODEL_DIR))
        return os.path.commonpath([path, root]) == root
    except ValueError:  # different drives
        return False


def _pin_static_input(model) -> tuple[int, int]:
    """Make the input 1x3xHxW static (the NPU accepts nothing else) and return
    (h, w). Dynamic H/W take the module default."""
    shape = model.input(0).partial_shape
    if shape.rank.is_dynamic or shape.rank.get_length() != 4:
        raise ValueError(f"Expected a 4-D image input, got {shape}")

    def _dim(i: int, default: int) -> int:
        return shape[i].get_length() if shape[i].is_static else default

    h, w = _dim(2, INPUT_SIZE[0]), _dim(3, INPUT_SIZE[1])
    if not shape.is_static or _dim(0, 1) != 1:
        model.reshape([1, 3, h, w])
    return h, w


def _uint8_hwc_input(model):
    """Take the frame as uint8 NHWC and let the device cast it to float32 and
    transpose it to NCHW, instead of the processor doing both per frame."""
    from openvino import Layout, Type  # lazy
    from openvino.preprocess import PrePostProcessor
    ppp = PrePostProcessor(model)
    ppp.input().tensor().set_element_type(Type.u8).set_layout(Layout("NHWC"))
    ppp.input().model().set_layout(Layout("NCHW"))
    ppp.input().preprocess().convert_element_type(Type.f32)
    return ppp.build()


def _check_yolox_output(model, num_classes: int, input_size: tuple[int, int]) -> None:
    """Refuse an output this decoder would turn into confident nonsense."""
    shape = model.output(0).partial_shape
    if shape.rank.is_dynamic or shape.rank.get_length() != 3:
        raise ValueError(f"Expected a [1, anchors, 5+classes] output, got {shape}")
    anchors, channels = shape[1], shape[2]
    if not (anchors.is_static and channels.is_static):
        return  # can't tell before compiling; trust it as the base detector does
    a, c = anchors.get_length(), channels.get_length()
    if a < c:
        raise ValueError(
            f"Output {shape} is channels-first ([1, 4+classes, anchors]) — not a "
            "YOLOX export. Only YOLOX-layout detectors run here.")
    if a != expected_anchors(input_size):
        raise ValueError(
            f"Output has {a} rows; a raw-grid YOLOX export at {input_size} has "
            f"{expected_anchors(input_size)}. Was it exported with decoding baked in, "
            "or with other strides?")
    if c != 5 + num_classes:
        raise ValueError(
            f"Model predicts {c - 5} classes but {num_classes} names were given — "
            "wrong labels file?")


def _execution_devices(compiled, requested: str) -> str:
    """Where a compiled model really runs ("NPU", "GPU.1", "CPU")."""
    try:
        devs = compiled.get_property("EXECUTION_DEVICES")
        if isinstance(devs, str):
            return devs
        return ",".join(str(d) for d in devs) or requested
    except Exception:
        return requested


if __name__ == "__main__":
    import sys
    import time

    from modules.system.debug_console import force_utf8_stdio
    force_utf8_stdio()
    import numpy as np

    print(f"NPU: {npu_name() or 'none found'}")
    try:
        det = build_npu_detector(sys.argv[1] if len(sys.argv) > 1 else "")
    except Exception as e:
        print(f"❌ {e}")
        raise SystemExit(1)
    frame = np.full((720, 1280, 3), 114, dtype=np.uint8)
    det.detect(frame)  # first call allocates
    t0 = time.perf_counter()
    for _ in range(20):
        det.detect(frame)
    ms = (time.perf_counter() - t0) / 20 * 1000
    print(f"✅ {det.device}  input {det.input_size}  {ms:.1f} ms/frame (incl. pre/post)")
