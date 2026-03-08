"""
modules/device_utils.py
=======================
Centralized device detection and resolution for the highlight pipeline.

One source of truth for all device strings passed to:
  - YOLO / Ultralytics  (.pt models and OpenVINO models)
  - OpenVINO            (action recognition encoder/decoder)
  - PyTorch / R3D       (action recognition CUDA model)
  - Motion detection
"""

import os

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Primary entry point — call this once at the top of pipeline.py
# ---------------------------------------------------------------------------

def detect_best_device(log_fn=print):
    """
    Detect the best available hardware and return a DeviceInfo with
    pre-resolved device strings for every consumer in the pipeline.

    Priority: CUDA > Intel XPU > CPU

    Fields on the returned DeviceInfo:
        .yolo_pt_device    str   device for YOLO .pt models        "cuda:0" | "cpu"
        .yolo_ov_device    str   device for YOLO OpenVINO models    "cpu"
        .openvino_device   str   device hint for OpenVINO Core      "GPU" | "CPU" | "AUTO"
        .pytorch_device    str   device for PyTorch / R3D           "cuda" | "cpu"
        .motion_device     str   device for motion detection        "cuda:0" | "cpu"
        .use_openvino_yolo bool  True → load OpenVINO YOLO model
        .gpu_available     bool  True if any GPU was found
        .backend_name      str   human-readable label for logging
    """
    # ---- NVIDIA CUDA -------------------------------------------------------
    if _TORCH_AVAILABLE:
        try:
            if torch.cuda.is_available():
                count = torch.cuda.device_count()
                log_fn(f"✅ CUDA available: {count} device(s)")
                for i in range(count):
                    try:
                        name = torch.cuda.get_device_name(i)
                        vram = torch.cuda.get_device_properties(i).total_mem / (1024 ** 3)
                        log_fn(f"   Device {i}: {name} ({vram:.1f} GB VRAM)")
                    except Exception:
                        pass
                return DeviceInfo(
                    yolo_pt_device="cuda:0",
                    yolo_ov_device="cpu",
                    openvino_device="AUTO",
                    pytorch_device="cuda",
                    motion_device="cuda:0",
                    use_openvino_yolo=False,
                    gpu_available=True,
                    backend_name="CUDA",
                )
        except Exception as e:
            log_fn(f"⚠️ CUDA check failed: {e}")

    # ---- Intel XPU ---------------------------------------------------------
    if _TORCH_AVAILABLE:
        try:
            import intel_extension_for_pytorch as ipex  # noqa: F401
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                count = torch.xpu.device_count()
                log_fn(f"✅ Intel XPU available: {count} device(s)")
                for i in range(count):
                    try:
                        name = torch.xpu.get_device_name(i)
                        log_fn(f"   Device {i}: {name}")
                    except Exception:
                        pass
                # XPU: use OpenVINO GPU for YOLO + action recognition.
                # PyTorch R3D stays on CPU — torchvision has no XPU backend.
                return DeviceInfo(
                    yolo_pt_device="cpu",
                    yolo_ov_device="cpu",
                    openvino_device="GPU",
                    pytorch_device="cpu",
                    motion_device="cpu",
                    use_openvino_yolo=True,
                    gpu_available=True,
                    backend_name="Intel XPU (OpenVINO)",
                )
        except ImportError:
            log_fn("ℹ️ Intel Extension for PyTorch not installed — XPU unavailable")
        except Exception as e:
            log_fn(f"⚠️ XPU check failed: {e}")

    # ---- CPU fallback -------------------------------------------------------
    log_fn("ℹ️ No GPU found — using CPU")
    return DeviceInfo(
        yolo_pt_device="cpu",
        yolo_ov_device="cpu",
        openvino_device="CPU",
        pytorch_device="cpu",
        motion_device="cpu",
        use_openvino_yolo=True,
        gpu_available=False,
        backend_name="CPU",
    )


# ---------------------------------------------------------------------------
# Safety net — use in worker processes or anywhere a raw device string
# arrives (e.g. via multiprocessing, CLI arg, or old cached config).
# ---------------------------------------------------------------------------

def resolve_yolo_device(requested: str) -> str:
    """
    Validate a raw device string and return something YOLO/Ultralytics accepts.

    "xpu:0", "mps", "npu", etc. → "cpu"
    "cuda" / "cuda:N"           → "cuda:N" if CUDA is available, else "cpu"
    "cpu"                       → "cpu"
    """
    if not requested or requested == "cpu":
        return "cpu"

    if requested.startswith("cuda") or requested.isdigit():
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            return requested
        _warn(f"CUDA requested ('{requested}') but torch.cuda.is_available() is False. "
              f"Falling back to CPU.")
        return "cpu"

    # xpu:0, mps, npu, or anything else Ultralytics doesn't understand
    _warn(f"Unrecognized YOLO device '{requested}'. Falling back to CPU.")
    return "cpu"


resolve_device = resolve_yolo_device


# ---------------------------------------------------------------------------
# DeviceInfo value object
# ---------------------------------------------------------------------------

class DeviceInfo:
    __slots__ = (
        "yolo_pt_device", "yolo_ov_device", "openvino_device",
        "pytorch_device", "motion_device",
        "use_openvino_yolo", "gpu_available", "backend_name",
    )

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return (
            f"DeviceInfo(backend={self.backend_name!r}, "
            f"yolo_pt={self.yolo_pt_device!r}, "
            f"openvino={self.openvino_device!r}, "
            f"pytorch={self.pytorch_device!r})"
        )


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _warn(msg: str):
    print(f"⚠️ [device_utils] {msg}")
    print(f"   CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(not set)')}")
