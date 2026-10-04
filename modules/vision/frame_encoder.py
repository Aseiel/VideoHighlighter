"""The frozen image encoder that taught action models are built on: SigLIP2
base/16 at 256 px, run on whichever accelerator this machine has.

Why one encoder, and why this one
---------------------------------
A taught action model is a small head on top of a frozen encoder's vectors, so
a head only works with the encoder it was trained on. Models people make and
share have to be interchangeable, so the app has exactly one encoder, named by
:data:`ENCODER_ID`. A head records that id, and a head with another id is
refused, never silently fed the wrong vectors. Base/16 was chosen over base/32
for accuracy (2-4 points more on held-out videos; measured in
``docs/plans/2026-10-02-siglip2-search-pose-export.md``), at the price of being
about 4x slower on a processor.

The preprocessing is part of the id
-----------------------------------
:func:`preprocess` is exactly the pipeline that was measured best: shrink to
a short side of 384 with an area filter (only when larger), squash to 256x256
bilinear, RGB, scaled to [-1, 1]. SigLIP2 was trained on squashed input, and
small differences here (resize filter, decode size) moved held-out accuracy by
about 2 points. Changing it changes every vector, so it means a new
:data:`ENCODER_ID`, the same as changing the model.

The model file
--------------
``models/<ENCODER_ID>/vision.onnx`` plus ``encoder.json``, written by
``tools/export_frame_encoder.py`` and delivered as a download (a pack in the
free edition, part of the build in Pro), never kept in git. The ONNX stores
its weights in fp16 (186 MB) and widens them to fp32 at load. Both runtimes
read the same file: OpenVINO reads ONNX directly and runs it as fast as its own
IR (measured), and ONNX Runtime needs ONNX anyway.

Routes
------
OpenVINO drives Intel GPUs and is the fastest processor path. ONNX Runtime
drives everything else that has a GPU: DirectML on Windows (any DX12 card,
NVIDIA included, since the build has no CUDA provider for ONNX Runtime) and
Core ML on a Mac. :func:`route_order` turns the user's ``compute.backend``
choice into an order, and :func:`load` takes the first route that proves
itself.

**Every route proves itself before it is used.** ``encoder.json`` carries the
vector PyTorch computes for :func:`probe_pixels`, and each route must
reproduce it. A route that returns NaN (as RTMPose did on the Arc GPU at f16)
or plausible-looking garbage is skipped with a line in the log, and the next
route is tried.

Nothing here raises to its caller: :func:`load` returns None when there is no
model or no route, and the caller says so.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Callable, Optional, Sequence

import numpy as np

ENCODER_ID = "siglip2-base-patch16-256"
SOURCE_MODEL = "google/siglip2-base-patch16-256"
# The Hugging Face commit the export is made from. Pinned so every build
# exports the same bytes: the updater re-downloads any file whose hash moved.
SOURCE_REVISION = "3f9f96cb90da5dbc758b01813f2f6f1aee24c1ab"
MODEL_DIRNAME = ENCODER_ID
MODEL_FILE = "vision.onnx"
META_FILE = "encoder.json"
# Bump when encoder.json gains something an older app cannot do without.
FORMAT = 1

INPUT_SIZE = 256
DECODE_SHORT = 384
DIMS = 768

# A folder holding vision.onnx + encoder.json, for a source checkout or a test.
DIR_ENV = "VH_FRAME_ENCODER_DIR"

# Frames per call. On a GPU, one frame per call costs about 4x more per frame
# than 32; on a processor the gain flattens out by 8 and bigger batches only
# hold more memory.
GPU_BATCH = 32
CPU_BATCH = 8

# How close a route's probe vector must be to PyTorch's. Faithful routes
# measure 0.99999+; a broken one lands far below.
PROBE_MIN_COSINE = 0.99

OPENVINO_GPU = "openvino-gpu"
ONNX_GPU = "onnxruntime-gpu"
OPENVINO_CPU = "openvino-cpu"
ONNX_CPU = "onnxruntime-cpu"
_GPU_ROUTES = (OPENVINO_GPU, ONNX_GPU)

LogFn = Callable[[str], None]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(frames_bgr: Sequence[np.ndarray]) -> np.ndarray:
    """BGR frames of any size -> float32 [N, 3, 256, 256] in [-1, 1], RGB."""
    import cv2

    out = np.empty((len(frames_bgr), 3, INPUT_SIZE, INPUT_SIZE), np.float32)
    for i, frame in enumerate(frames_bgr):
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        h, w = frame.shape[:2]
        scale = DECODE_SHORT / min(h, w)
        if scale < 1:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        frame = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        rgb = frame[:, :, ::-1].astype(np.float32)
        out[i] = (rgb / 255.0 - 0.5).transpose(2, 0, 1) / 0.5
    return out


def probe_pixels() -> np.ndarray:
    """A fixed, image-like input [1, 3, 256, 256] whose vector encoder.json
    records. Built from arithmetic, not a random generator, so every numpy
    version produces the same pixels."""
    y, x = np.mgrid[0:INPUT_SIZE, 0:INPUT_SIZE].astype(np.float32) / (INPUT_SIZE - 1)
    r = np.sin(6.0 * x + 2.0 * y)
    g = np.cos(9.0 * x * y + 1.0) * 0.8
    b = 2.0 * np.abs(((4.0 * x + 3.0 * y) % 1.0) - 0.5) * 2.0 - 1.0
    return np.stack([r, g, b])[None].astype(np.float32)


def _cosine(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Where the model is
# ---------------------------------------------------------------------------

def _per_user_dir() -> str:
    """Same rule as modules/packs/pack_manager.per_user_dir."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "VideoHighlighter")


def model_dir_candidates() -> list:
    """Every folder the model may be in, most specific first: the override,
    beside the exe, the per-user folder (where a download goes when the exe's
    folder is read-only), the bundle, and the source checkout."""
    out = []
    override = os.environ.get(DIR_ENV, "").strip()
    if override:
        out.append(override)
    bases = []
    if getattr(sys, "frozen", False):
        bases.append(os.path.dirname(sys.executable))
        bases.append(_per_user_dir())
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            bases.append(meipass)
    # modules/vision/frame_encoder.py -> the project root.
    bases.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    out += [os.path.join(b, "models", MODEL_DIRNAME) for b in bases]
    return out


def find_model_dir() -> Optional[str]:
    """The first candidate folder holding both files, or None."""
    for folder in model_dir_candidates():
        if (os.path.isfile(os.path.join(folder, MODEL_FILE))
                and os.path.isfile(os.path.join(folder, META_FILE))):
            return folder
    return None


def is_installed() -> bool:
    return find_model_dir() is not None


def read_meta(folder: str) -> dict:
    """encoder.json, checked against what this app was built for. Raises
    ValueError with a sentence for the log when it does not match."""
    with open(os.path.join(folder, META_FILE), encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("id") != ENCODER_ID:
        raise ValueError(f"{META_FILE} is for {meta.get('id')!r}, this app uses {ENCODER_ID!r}")
    if int(meta.get("format", 0)) > FORMAT:
        raise ValueError(f"{META_FILE} format {meta.get('format')} is newer than this app ({FORMAT})")
    if int(meta.get("dims", 0)) != DIMS or len(meta.get("probe", [])) != DIMS:
        raise ValueError(f"{META_FILE} does not describe {DIMS}-number vectors")
    return meta


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _openvino_gpu() -> tuple:
    """(device, is_discrete) of the Intel GPU OpenVINO can drive, discrete
    first; (None, False) when there is none. OpenVINO's GPU plugin is
    Intel-only, so any GPU it lists is Intel."""
    try:
        import openvino as ov
        core = ov.Core()
        gpus = [d for d in core.available_devices if d == "GPU" or d.startswith("GPU.")]
    except Exception:  # noqa: BLE001 - no OpenVINO means no such route
        return None, False
    found = []
    for dev in gpus:
        try:
            kind = str(core.get_property(dev, "DEVICE_TYPE"))
        except Exception:  # noqa: BLE001
            kind = ""
        found.append((dev, "DISCRETE" in kind.upper()))
    found.sort(key=lambda d: not d[1])
    return found[0] if found else (None, False)


def _cuda_present() -> bool:
    """True when torch, already imported by the app, sees an NVIDIA card. Never
    imports torch itself: that costs seconds and is not this module's call."""
    torch = sys.modules.get("torch")
    try:
        return bool(torch is not None and torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def route_order(backend: Optional[str], *, intel_gpu: bool, intel_discrete: bool,
                cuda: bool) -> list:
    """The routes to try, best first, for a ``compute.backend`` choice.

    Automatic follows the app's own order (CUDA, then Intel, then DirectML):
    an Arc card runs OpenVINO; an NVIDIA card next to an integrated Intel GPU
    runs DirectML, because the NVIDIA card is the faster one; an integrated
    Intel GPU alone runs OpenVINO, which drives it far better than DirectML. A
    named choice puts its route first, and the processor routes always close
    the list, so a choice this machine cannot honour still runs.
    """
    from modules.system import compute_backend as cb

    cpu = [OPENVINO_CPU, ONNX_CPU]
    chosen = cb.normalise(backend) if backend else None
    if chosen == cb.CPU:
        return cpu
    if chosen == cb.INTEL:
        return [OPENVINO_GPU] + cpu
    if chosen in (cb.CUDA, cb.DIRECTML, cb.APPLE):
        return [ONNX_GPU] + cpu
    if intel_gpu and (intel_discrete or not cuda):
        return [OPENVINO_GPU, ONNX_GPU] + cpu
    return [ONNX_GPU, OPENVINO_GPU] + cpu


def route_label(route: str) -> str:
    if route == ONNX_GPU:
        from modules.system import ort_directml
        name = "Core ML" if ort_directml.gpu_provider() == ort_directml.COREML_PROVIDER else "DirectML"
        return f"ONNX Runtime {name}"
    return {OPENVINO_GPU: "OpenVINO GPU", OPENVINO_CPU: "OpenVINO CPU",
            ONNX_CPU: "ONNX Runtime CPU"}.get(route, route)


def _compile_cache_dir() -> Optional[str]:
    """Where OpenVINO keeps compiled GPU kernels, so the compile (seconds) is
    paid once per machine rather than once per run."""
    try:
        from modules.system import app_paths
        path = os.path.join(app_paths.user_data_dir(), "ov-cache")
        os.makedirs(path, exist_ok=True)
        return path
    except Exception:  # noqa: BLE001 - no cache only costs time
        return None


def openvino_config(device: str) -> dict:
    """Compile options for an OpenVINO device.

    A processor computes in fp32. OpenVINO's CPU plugin otherwise picks its
    own precision per machine: fp16 on ARM (Apple silicon, where the vectors
    measured 0.9977 against PyTorch's while ONNX Runtime gave 1.000000) and
    bf16 on Xeons with AMX. A head trained on one machine's vectors has to read
    the same vectors on every other, so the processor route may not drift.
    """
    return {"INFERENCE_PRECISION_HINT": "f32"} if device == "CPU" else {}


class _OpenVINORunner:
    def __init__(self, model_path: str, device: str):
        import openvino as ov
        core = ov.Core()
        cache = _compile_cache_dir() if device.startswith("GPU") else None
        if cache:
            core.set_property({"CACHE_DIR": cache})
        self._net = core.compile_model(model_path, device, openvino_config(device))

    def run(self, pixels: np.ndarray) -> np.ndarray:
        return np.array(self._net(pixels)[0], dtype=np.float32, copy=True)


class _OnnxRunner:
    def __init__(self, model_path: str, gpu: bool):
        from modules.system import ort_directml
        if gpu:
            if not ort_directml.available():
                raise RuntimeError(ort_directml.unavailable_reason() or "no GPU provider")
            self._session = ort_directml.session(model_path)
            got = ort_directml.session_backend(self._session)
            if not ort_directml.is_gpu_provider(got):
                raise RuntimeError(f"ONNX Runtime put the model on {got}")
        else:
            self._session = ort_directml.session(
                model_path, providers_override=[ort_directml.CPU_PROVIDER])
        self._input = self._session.get_inputs()[0].name

    def run(self, pixels: np.ndarray) -> np.ndarray:
        return np.asarray(self._session.run(None, {self._input: pixels})[0], dtype=np.float32)


def _open_route(route: str, model_path: str, intel_device: Optional[str]):
    if route == OPENVINO_GPU:
        if not intel_device:
            raise RuntimeError("no Intel GPU")
        return _OpenVINORunner(model_path, intel_device)
    if route == OPENVINO_CPU:
        return _OpenVINORunner(model_path, "CPU")
    if route == ONNX_GPU:
        return _OnnxRunner(model_path, gpu=True)
    if route == ONNX_CPU:
        return _OnnxRunner(model_path, gpu=False)
    raise ValueError(f"unknown route {route!r}")


# ---------------------------------------------------------------------------
# The encoder
# ---------------------------------------------------------------------------

class FrameEncoder:
    """A loaded encoder on one route. ``encode_bgr`` is the whole API."""

    encoder_id = ENCODER_ID
    dims = DIMS

    def __init__(self, runner, route: str, folder: str, batch: Optional[int] = None):
        self._runner = runner
        self.route = route
        self.label = route_label(route)
        self.folder = folder
        self.batch = int(batch or (GPU_BATCH if route in _GPU_ROUTES else CPU_BATCH))

    @property
    def on_gpu(self) -> bool:
        return self.route in _GPU_ROUTES

    def encode_pixels(self, pixels: np.ndarray) -> np.ndarray:
        """Preprocessed [N, 3, 256, 256] -> float32 [N, 768], in order."""
        pixels = np.ascontiguousarray(pixels, dtype=np.float32)
        if len(pixels) == 0:
            return np.zeros((0, DIMS), np.float32)
        parts = [self._runner.run(pixels[i:i + self.batch])
                 for i in range(0, len(pixels), self.batch)]
        return np.concatenate(parts).astype(np.float32, copy=False)

    def encode_bgr(self, frames_bgr: Sequence[np.ndarray]) -> np.ndarray:
        """BGR frames of any size -> float32 [N, 768], one vector per frame,
        preprocessed a batch at a time so a long list never sits in memory as
        pixels."""
        if len(frames_bgr) == 0:
            return np.zeros((0, DIMS), np.float32)
        parts = [self._runner.run(preprocess(frames_bgr[i:i + self.batch]))
                 for i in range(0, len(frames_bgr), self.batch)]
        return np.concatenate(parts).astype(np.float32, copy=False)

    def close(self):
        self._runner = None


def _check_route(runner, meta: dict) -> None:
    """Raise unless ``runner`` reproduces encoder.json's probe vector."""
    out = runner.run(probe_pixels())
    if out.shape != (1, DIMS):
        raise RuntimeError(f"returned shape {tuple(out.shape)}, expected (1, {DIMS})")
    if not np.isfinite(out).all():
        raise RuntimeError("returned non-finite numbers")
    cos = _cosine(out[0], meta["probe"])
    if cos < PROBE_MIN_COSINE:
        raise RuntimeError(f"does not match the reference (cosine {cos:.4f})")


def load(backend: Optional[str] = None, log: LogFn = print,
         model_dir: Optional[str] = None) -> Optional[FrameEncoder]:
    """The encoder on the best route that works here, or None.

    ``backend`` is a ``compute.backend`` value; None reads the user's setting.
    """
    folder = model_dir or find_model_dir()
    if folder is None:
        log(f"⚠️ Frame encoder {ENCODER_ID} is not installed "
            f"(looked in: {'; '.join(model_dir_candidates())})")
        return None
    try:
        meta = read_meta(folder)
    except Exception as e:  # noqa: BLE001
        log(f"⚠️ Frame encoder in {folder} is unusable: {e}")
        return None

    if backend is None:
        try:
            from modules.system import compute_backend
            backend = compute_backend.configured()
        except Exception:  # noqa: BLE001
            backend = None
    intel_device, discrete = _openvino_gpu()
    order = route_order(backend, intel_gpu=intel_device is not None,
                        intel_discrete=discrete, cuda=_cuda_present())
    model_path = os.path.join(folder, MODEL_FILE)
    for route in order:
        try:
            runner = _open_route(route, model_path, intel_device)
            _check_route(runner, meta)
        except Exception as e:  # noqa: BLE001 - the next route is the answer
            print(f"ℹ️ Frame encoder: {route_label(route)} skipped ({type(e).__name__}: {e})")
            continue
        encoder = FrameEncoder(runner, route, folder)
        log(f"✅ Frame encoder: {ENCODER_ID} on {encoder.label}")
        return encoder
    log(f"⚠️ Frame encoder: no route could run {ENCODER_ID} here "
        f"(tried {', '.join(route_label(r) for r in order)})")
    return None
