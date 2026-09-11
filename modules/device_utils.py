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

# Experimental AMD support. Guarded like torch above so a checkout missing the
# module (or a frozen build that did not bundle it) still resolves devices —
# DirectML is the one backend here that is allowed to be absent by design.
try:
    from modules import directml_device as _dml
except Exception:  # noqa: BLE001
    _dml = None

# The other DirectML runtime. ONNX Runtime's provider needs no particular torch,
# so unlike `directml_device` it survives into a packaged build — which is the
# only reason an AMD user running the exe can have a GPU at all. Detection is
# what it drives; see modules/onnx_detector.py.
try:
    from modules import ort_directml as _ort_dml
except Exception:  # noqa: BLE001
    _ort_dml = None


# ---------------------------------------------------------------------------
# Primary entry point — call this once at the top of pipeline.py
# ---------------------------------------------------------------------------

def detect_best_device(log_fn=print):
    """
    Detect the best available hardware and return a DeviceInfo with
    pre-resolved device strings for every consumer in the pipeline.

    Priority: CUDA > Intel XPU > Intel/OpenVINO > DirectML (AMD) > CPU

    DirectML sits last on purpose. It is the slowest of the accelerated paths
    and has the narrowest operator coverage, so it is worth having only where
    the alternative is the CPU — which on an AMD box is exactly the situation.
    `VH_DIRECTML=force` moves it to the front, for testing it on a machine that
    has something better.

    Fields on the returned DeviceInfo:
        .yolo_pt_device    str   device for YOLO .pt models        "cuda:0" | "cpu"
        .yolo_ov_device    str   device for YOLO OpenVINO models    "cpu"
        .openvino_device   str   device hint for OpenVINO Core      "GPU" | "CPU" | "AUTO"
        .pytorch_device    str   device for PyTorch / R3D           "cuda" | "cpu"
        .motion_device     str   device for motion detection        "cuda:0" | "cpu"
        .dml_device        str   DirectML device, or None           "privateuseone:0"
        .use_openvino_yolo bool  True → load OpenVINO YOLO model
        .gpu_available     bool  True if any GPU was found
        .backend_name      str   human-readable label for logging
    """
    # ---- DirectML, when explicitly forced ahead of everything else ----------
    if _dml is not None and _dml.forced():
        forced = _directml_info(log_fn)
        if forced is not None:
            return forced
        log_fn(f"⚠️ {_dml.MODE_ENV}=force but DirectML is unusable: "
               f"{_dml.unavailable_reason()}")

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

    # ---- Intel XPU (Arc dGPU / Xe iGPU) --------------------------------------
    # torch 2.5+ '+xpu' builds expose torch.xpu natively — no ipex import needed.
    if _TORCH_AVAILABLE and hasattr(torch, "xpu"):
        try:
            if torch.xpu.is_available():
                count = torch.xpu.device_count()
                log_fn(f"✅ Intel XPU available: {count} device(s)")
                for i in range(count):
                    try:
                        log_fn(f"   Device {i}: {torch.xpu.get_device_name(i)}")
                    except Exception:
                        pass
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
        except Exception as e:
            log_fn(f"⚠️ XPU check failed: {e}")

    # ---- Intel GPU via OpenVINO (no torch xpu build needed) -----------------
    # The frozen exe ships a CUDA torch (the release build installs the cu124
    # wheel). torch.xpu still *exists* on it — the attribute is there in any
    # build — but reports is_available() False, so the XPU branch above never
    # fires in the packaged app, even on an Arc machine. OpenVINO can still
    # drive the GPU, so probe it directly and use it for OpenVINO consumers
    # (YOLO OV model + action recognition).
    try:
        from openvino import Core
        _ov_devices = Core().available_devices
        if any(d == "GPU" or d.startswith("GPU.") for d in _ov_devices):
            log_fn(f"✅ Intel GPU available via OpenVINO: {_ov_devices}")
            return DeviceInfo(
                yolo_pt_device="cpu",
                yolo_ov_device="cpu",
                openvino_device="GPU",
                pytorch_device="cpu",
                motion_device="cpu",
                use_openvino_yolo=True,
                gpu_available=True,
                backend_name="Intel GPU (OpenVINO)",
            )
    except Exception as e:
        log_fn(f"⚠️ OpenVINO GPU probe failed: {e}")

    # ---- DirectML (AMD, and anything else with a DX12 driver) ---------------
    # Reached only when neither CUDA nor an Intel path was found, so this can
    # never take work away from a faster backend — it only rescues machines
    # that would otherwise run everything on the processor.
    if _dml is not None:
        info = _directml_info(log_fn)
        if info is not None:
            return info
        reason = _dml.unavailable_reason()
        if reason and _dml.enabled():
            log_fn(f"ℹ️ DirectML not used for torch models: {reason}")

    # ---- DirectML for detection only, through ONNX Runtime ------------------
    # Reached when torch has no DirectML but ONNX Runtime does, which is every
    # packaged build: `torch-directml` pins an exact torch and so can never be
    # bundled beside the CUDA one, while `onnxruntime-directml` depends on no
    # torch at all. Detection is the heaviest per-frame stage, so this is the
    # larger half of the win even though it moves fewer models.
    info = _onnx_dml_info(log_fn)
    if info is not None:
        return info

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


def _directml_info(log_fn=print):
    """DeviceInfo for a usable DirectML device, or None.

    **`pytorch_device` carries the DirectML string**, which is what routes the
    R3D action-recognition model onto an AMD card. R3D is a 3D CNN, and 3D
    convolution is the part of DirectML's operator coverage least likely to
    hold up — so this is not taken on trust: `R3DModelWrapper._warmup()` runs a
    real forward pass at load and moves the model to the CPU if the backend
    cannot execute it. That turns the risk into a slow run with one explanatory
    line, instead of an "operator not implemented" an hour into a job.

    Every other consumer of this field asks `== "cuda"`, and all of them still
    correctly answer no. The Intel action encoder/decoder is deliberately left
    on OpenVINO: it is small enough that moving it would buy nothing, and it
    has no ONNX/torch form DirectML could run anyway.

    Object detection is *not* covered by this. YOLO runs through Ultralytics
    here, which has no DirectML backend, so `yolo_pt_device` stays "cpu" and
    `resolve_yolo_device` answers a DirectML request with "cpu".

    `gpu_available` is True and `backend_name` says AMD, which is what
    `modules/encoder_select.py` reads to prefer the AMF video encoders — a win
    that lands even when no model ever touches DirectML.
    """
    if _dml is None or not _dml.enabled():
        return None
    p = _dml.probe()
    if not p.available:
        return None
    device = p.device_string()
    log_fn(f"✅ {_dml.describe()}")
    log_fn(f"   torch device: {device} (experimental — see docs/AMD-GPU.md)")
    return DeviceInfo(
        yolo_pt_device="cpu",
        yolo_ov_device="cpu",
        # Not "GPU": OpenVINO's GPU plugin is Intel-only, so asking for it on
        # an AMD box buys a failed plugin load instead of acceleration.
        openvino_device="CPU",
        pytorch_device=device,
        motion_device="cpu",
        dml_device=device,
        use_openvino_yolo=True,
        gpu_available=True,
        # A source install can have both runtimes. If ONNX Runtime is one of
        # them, detection goes to the GPU too rather than staying on the CPU
        # because Ultralytics has no DirectML backend.
        onnx_dml_yolo=(_ort_dml is not None and _ort_dml.available()),
        backend_name="DirectML (AMD/DX12)",
    )


def _onnx_dml_info(log_fn=print):
    """DeviceInfo for a machine whose GPU only ONNX Runtime can reach, or None.

    Everything torch touches stays on the processor — that is the honest answer
    when there is no torch build for this card. What changes is `onnx_dml_yolo`:
    the object detector may load an ONNX export and run it on the GPU, which is
    where most of a run's per-frame time goes.

    `gpu_available` is True for the same reason it is on the Intel/OpenVINO
    branch above: a GPU *is* doing work, just not through torch, and
    `modules/encoder_select.py` reads the backend name to prefer the AMF video
    encoders on an AMD box.
    """
    if _ort_dml is None or not _ort_dml.available():
        return None
    probe = _ort_dml.probe()
    log_fn(f"✅ DirectML via ONNX Runtime {probe.version or ''}".rstrip())
    log_fn("   object detection only — torch models stay on the CPU "
           "(see docs/AMD-GPU.md)")
    return DeviceInfo(
        yolo_pt_device="cpu",
        yolo_ov_device="cpu",
        # OpenVINO's GPU plugin is Intel-only; asking for it here buys a failed
        # plugin load rather than acceleration.
        openvino_device="CPU",
        pytorch_device="cpu",
        motion_device="cpu",
        use_openvino_yolo=True,
        gpu_available=True,
        onnx_dml_yolo=True,
        backend_name="DirectML (ONNX Runtime)",
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
    "dml" / "privateuseone:N"   → "cpu" (Ultralytics has no DirectML backend)
    "cpu"                       → "cpu"
    """
    if not requested or requested == "cpu":
        return "cpu"

    # DirectML is answered with the CPU here, deliberately. This function is the
    # *detector's* device and its whole contract is that the value it returns is
    # safe to use — and Ultralytics does not accept "privateuseone:0", so passing
    # one on would trade a slow run for a failed one. A DirectML string can reach
    # here from a stale config, a CLI flag, or a worker process.
    #
    # The consumers that can use DirectML (R3D action recognition, the CLIP
    # prefilter) reach it through DeviceInfo.dml_device, never through this.
    if _dml is not None and _dml.is_directml(requested):
        _warn(f"DirectML requested for the detector ({requested!r}), which has no "
              f"DirectML path — YOLO runs through Ultralytics. Using CPU. Action "
              f"recognition and visual search do use DirectML; see docs/AMD-GPU.md.")
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
        "pytorch_device", "motion_device", "dml_device",
        "use_openvino_yolo", "gpu_available", "backend_name",
        "onnx_dml_yolo",
    )

    # Every slot gets a default. __slots__ leaves an unset attribute *missing*
    # rather than None, so a field added later (dml_device was) would raise
    # AttributeError on every DeviceInfo built by the branches that predate it.
    _DEFAULTS = {"dml_device": None, "onnx_dml_yolo": False}

    def __init__(self, **kwargs):
        for k, v in self._DEFAULTS.items():
            setattr(self, k, v)
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
