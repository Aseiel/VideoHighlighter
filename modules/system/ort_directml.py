"""ONNX Runtime's DirectML execution provider — the GPU path a packaged build
can actually ship.

``torch-directml`` cannot be bundled. It pins an exact torch (2.4.1 for the
current release) and pip satisfies that by *replacing* whatever torch is
installed, so a build carrying it could not also carry the CUDA torch the
NVIDIA path needs — one process, one torch. That is why
``modules/system/directml_device.py`` is opt-in and source-only, and why an AMD user
running the exe gets the processor.

ONNX Runtime has no such coupling. ``onnxruntime-directml`` declares no torch
dependency at all, so the same build can hold CUDA torch *and* a DirectML-
capable runtime beside it, and route a model to whichever one the machine can
use. The model format is the common denominator, not the framework — the same
arrangement AnimeJaNai ships (TensorRT for NVIDIA, DirectML for AMD/Intel, one
release, ONNX models).

What this module is
-------------------
The provider probe and the session factory, and nothing else. It answers "can
this machine run an ONNX model on its GPU, and why not" in the same shape
``directml_device`` answers it for torch, so a caller can ask both without
learning two idioms. :mod:`modules.vision.onnx_detector` is the first consumer.

**One onnxruntime, ever.** ``onnxruntime``, ``onnxruntime-gpu`` and
``onnxruntime-directml`` all install the same ``onnxruntime`` package and
overwrite each other. Only the DirectML one belongs in ``requirements.txt``:
NVIDIA already has torch+CUDA and Intel has OpenVINO, so a second accelerated
ORT build would buy nothing and break the one that matters.

The user's switch is shared with the torch backend: ``VH_DIRECTML=off`` turns
off DirectML, whichever runtime would have provided it.

On macOS the same probe and factory hand out Apple's Core ML provider instead
(:mod:`modules.system.ort_coreml`), which runs a model on the Apple GPU or
Neural Engine. The plain ``onnxruntime`` wheel carries it there, the models and
their pre- and post-processing are the same, and so is every caller: "can ONNX
Runtime reach the GPU here" has one answer per platform, and this module gives
it. ``VH_COREML=off`` is the Mac's switch.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Sequence

from modules.system import directml_device as _dml
from modules.system import ort_coreml as _coreml

# ORT's name for the provider. A literal here is fine — unlike torch's backend
# name, this string is part of ONNX Runtime's public API and is what
# get_available_providers() returns.
PROVIDER = "DmlExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"
COREML_PROVIDER = _coreml.PROVIDER

# Every provider that means "the model is on a GPU", whichever platform put it
# there. What a live session got is compared against this, never against one
# name, so a Mac's Core ML session counts as the GPU it is.
GPU_PROVIDERS = (PROVIDER, COREML_PROVIDER)

# Adapter index, for a machine with more than one DX12 card. Shares the numbering
# DirectML itself uses, so `tools/check_directml.py` output applies here too.
DEVICE_ENV = "VH_DIRECTML_DEVICE"

_probe_cache = None            # Optional[ProviderProbe]


class ProviderProbe:
    """What one look at ONNX Runtime found.

    ``reason`` is populated whenever ``available`` is False and is meant to be
    shown to a user: "no onnxruntime at all" and "this onnxruntime was built
    without DirectML" are the same outcome and completely different problems.
    """

    __slots__ = ("available", "reason", "version", "providers", "provider",
                 "adapter_index", "adapter_name")

    def __init__(self, available=False, reason=None, version=None, providers=(),
                 provider=None, adapter_index=None, adapter_name=None):
        self.available = available
        self.reason = reason
        self.version = version
        self.providers = tuple(providers)
        # The GPU provider this probe was about: DirectML, or Core ML on a Mac.
        self.provider = provider
        # The graphics card DirectML will bind, when the adapters could be
        # listed. None on a Mac, and wherever DXGI could not be asked.
        self.adapter_index = adapter_index
        self.adapter_name = adapter_name

    def __repr__(self):  # pragma: no cover - diagnostics only
        state = "available" if self.available else f"unavailable ({self.reason})"
        return f"<ProviderProbe {state} ort={self.version}>"


def _import_onnxruntime():
    """Imported lazily and never at module scope: ORT loads native libraries,
    and this module is imported by the device probe on machines that will never
    touch it."""
    import onnxruntime  # noqa: PLC0415 - deliberate, see docstring
    return onnxruntime


def gpu_provider() -> str:
    """The GPU provider ONNX Runtime can offer on this platform.

    One per platform, because each platform's wheel carries exactly one:
    ``onnxruntime-directml`` on Windows, the plain wheel with Core ML on macOS.
    """
    return COREML_PROVIDER if sys.platform == "darwin" else PROVIDER


def is_gpu_provider(name) -> bool:
    """True when a session that got provider ``name`` runs on a GPU."""
    return name in GPU_PROVIDERS


def device_id() -> int:
    """Which DX12 adapter to bind: the user's pick, else the first graphics
    card the probe found, else 0."""
    raw = str(os.environ.get(DEVICE_ENV, "")).strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    found = probe().adapter_index
    return found if found is not None else 0


# What ONNX Runtime itself refuses to bind DirectML to (its IsSoftwareAdapter):
# an adapter flagged as software, or Microsoft's Basic Render Driver.
_DXGI_ADAPTER_FLAG_SOFTWARE = 2
_MICROSOFT_VENDOR_ID = 0x1414
_BASIC_RENDER_DEVICE_ID = 0x8C
_DXGI_ERROR_NOT_FOUND = 0x887A0002 - (1 << 32)   # as a signed HRESULT


def _dxgi_adapters():
    """Every display adapter DXGI lists, as ``(index, name, is_hardware)``, or
    None when DXGI could not be asked.

    ``onnxruntime-directml`` reports ``DmlExecutionProvider`` whether or not
    the machine has a graphics card, so the provider list alone said "GPU" on
    a VM or a CI runner whose only adapter is Microsoft's software renderer.
    Every session there then failed to bind it and fell back to the processor,
    one "EP Error" per worker. The index is DXGI's, which is the numbering
    DirectML's ``device_id`` uses.
    """
    if sys.platform != "win32":
        return None
    import ctypes  # noqa: PLC0415 - Windows only

    class _GUID(ctypes.Structure):
        _fields_ = [("Data1", ctypes.c_uint32), ("Data2", ctypes.c_uint16),
                    ("Data3", ctypes.c_uint16), ("Data4", ctypes.c_ubyte * 8)]

    class _LUID(ctypes.Structure):
        _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]

    class _AdapterDesc1(ctypes.Structure):          # DXGI_ADAPTER_DESC1
        _fields_ = [("Description", ctypes.c_wchar * 128),
                    ("VendorId", ctypes.c_uint32), ("DeviceId", ctypes.c_uint32),
                    ("SubSysId", ctypes.c_uint32), ("Revision", ctypes.c_uint32),
                    ("DedicatedVideoMemory", ctypes.c_size_t),
                    ("DedicatedSystemMemory", ctypes.c_size_t),
                    ("SharedSystemMemory", ctypes.c_size_t),
                    ("AdapterLuid", _LUID), ("Flags", ctypes.c_uint32)]

    def call(obj, slot, *args, argtypes=()):
        """COM method ``slot`` of ``obj``'s vtable."""
        vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
        method = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtable[slot])
        return method(obj, *args)

    # Vtable slots: IUnknown::Release is 2, IDXGIFactory1::EnumAdapters1 is 12,
    # IDXGIAdapter1::GetDesc1 is 10.
    RELEASE, ENUM_ADAPTERS1, GET_DESC1 = 2, 12, 10
    iid_factory1 = _GUID(0x770AAE78, 0xF26F, 0x4DBA,
                         (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
    try:
        create = ctypes.WinDLL("dxgi").CreateDXGIFactory1
        create.restype = ctypes.c_long
        create.argtypes = [ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
        factory = ctypes.c_void_p()
        hr = create(ctypes.byref(iid_factory1), ctypes.byref(factory))
        if hr < 0:
            raise OSError(f"CreateDXGIFactory1 failed (0x{hr & 0xFFFFFFFF:08X})")
        adapters = []
        try:
            for index in range(64):
                adapter = ctypes.c_void_p()
                hr = call(factory, ENUM_ADAPTERS1, index, ctypes.byref(adapter),
                          argtypes=(ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)))
                if hr == _DXGI_ERROR_NOT_FOUND:
                    break
                if hr < 0:
                    raise OSError(f"EnumAdapters1({index}) failed (0x{hr & 0xFFFFFFFF:08X})")
                desc = _AdapterDesc1()
                try:
                    hr = call(adapter, GET_DESC1, ctypes.byref(desc),
                              argtypes=(ctypes.POINTER(_AdapterDesc1),))
                finally:
                    call(adapter, RELEASE)
                if hr < 0:
                    raise OSError(f"GetDesc1({index}) failed (0x{hr & 0xFFFFFFFF:08X})")
                software = (desc.Flags & _DXGI_ADAPTER_FLAG_SOFTWARE
                            or (desc.VendorId == _MICROSOFT_VENDOR_ID
                                and desc.DeviceId == _BASIC_RENDER_DEVICE_ID)
                            or _dml._is_software_adapter(desc.Description))
                adapters.append((index, desc.Description, not software))
        finally:
            call(factory, RELEASE)
        return adapters
    except Exception as e:  # noqa: BLE001 - not knowing is answered by the provider list
        print(f"⚠️ Could not list display adapters: {type(e).__name__}: {e}")
        return None


def probe(refresh: bool = False) -> ProviderProbe:
    """Can ONNX Runtime run a model on this machine's GPU?

    Cached, because the answer cannot change inside a process and the import
    costs real time. ``refresh=True`` is for tests and for a settings UI that
    just changed the mode.
    """
    global _probe_cache
    if _probe_cache is not None and not refresh:
        return _probe_cache

    wanted = gpu_provider()

    # One switch for both DirectML runtimes: a user who turned DirectML off
    # means off, not "off for torch and on for ONNX". The Mac has its own,
    # because DirectML's switch saying "off" there would be about nothing.
    if wanted == COREML_PROVIDER:
        if not _coreml.enabled():
            _probe_cache = ProviderProbe(
                reason=f"disabled ({_coreml.MODE_ENV}=off)", provider=wanted)
            return _probe_cache
    elif not _dml.enabled():
        _probe_cache = ProviderProbe(reason=f"disabled ({_dml.MODE_ENV}=off)",
                                     provider=wanted)
        return _probe_cache

    try:
        ort = _import_onnxruntime()
    except Exception as e:  # noqa: BLE001 - a missing package is a normal answer
        _probe_cache = ProviderProbe(
            reason=f"onnxruntime is not installed ({type(e).__name__}: {e})",
            provider=wanted)
        return _probe_cache

    version = getattr(ort, "__version__", None)
    try:
        providers = tuple(ort.get_available_providers())
    except Exception as e:  # noqa: BLE001
        _probe_cache = ProviderProbe(
            reason=f"onnxruntime could not list its providers ({e})",
            version=version, provider=wanted)
        return _probe_cache

    if wanted not in providers:
        # The plain `onnxruntime` wheel reports CPU only. Saying which build is
        # installed is the difference between a fixable message and a shrug.
        if wanted == COREML_PROVIDER:
            reason = ("this onnxruntime build has no Core ML provider "
                      f"(has: {', '.join(providers) or 'none'})")
        else:
            reason = ("this onnxruntime build has no DirectML provider "
                      f"(has: {', '.join(providers) or 'none'}) — "
                      "onnxruntime-directml is the one that does")
        _probe_cache = ProviderProbe(reason=reason, version=version,
                                     providers=providers, provider=wanted)
        return _probe_cache

    # The DirectML build lists its provider on any Windows machine, graphics
    # card or not; whether there is one to bind is DXGI's to say. When DXGI
    # cannot be asked, the provider list is all there is to go on.
    adapter_index = adapter_name = None
    if wanted == PROVIDER:
        adapters = _dxgi_adapters()
        if adapters is not None:
            hardware = [(i, name) for i, name, real in adapters if real]
            if not hardware:
                listed = ", ".join(name for _, name, _ in adapters) or "none"
                _probe_cache = ProviderProbe(
                    reason=f"no graphics card DirectML can use (adapters: {listed})",
                    version=version, providers=providers, provider=wanted)
                return _probe_cache
            adapter_index, adapter_name = hardware[0]

    _probe_cache = ProviderProbe(available=True, version=version,
                                 providers=providers, provider=wanted,
                                 adapter_index=adapter_index,
                                 adapter_name=adapter_name)
    return _probe_cache


def available() -> bool:
    """True when an ONNX model can be run on the GPU here."""
    return probe().available


def unavailable_reason() -> Optional[str]:
    """Why it cannot, in a sentence, or None when it can."""
    p = probe()
    return None if p.available else p.reason


def providers() -> Sequence:
    """The provider list to hand :func:`session`, best first.

    Always ends in CPU. A DirectML or Core ML session that cannot place one
    operator falls back per-node rather than failing the run, which is the
    behaviour worth having on a backend with partial operator coverage.
    """
    p = probe()
    if not p.available:
        return [CPU_PROVIDER]
    if p.provider == COREML_PROVIDER:
        return [(COREML_PROVIDER, _coreml.provider_options()), CPU_PROVIDER]
    return [(PROVIDER, {"device_id": device_id()}), CPU_PROVIDER]


def session(model_path, *, providers_override=None):
    """An ``InferenceSession`` for ``model_path`` on the best provider here.

    Raises whatever ONNX Runtime raises: a model that will not load is a real
    error, and the caller (which knows what it was loading) is better placed to
    decide whether to fall back than a swallowed exception here.
    """
    ort = _import_onnxruntime()
    return ort.InferenceSession(
        str(model_path),
        providers=list(providers_override if providers_override is not None
                       else providers()))


def session_backend(sess) -> str:
    """Which provider a live session actually got, for logging.

    Asked of the session rather than assumed from the request: ORT silently
    drops a provider it cannot initialise, and a run that quietly landed on the
    processor should say so.
    """
    try:
        got = list(sess.get_providers())
    except Exception:  # noqa: BLE001 - diagnostics must not raise
        return "unknown"
    return got[0] if got else "unknown"


def reset_probe_cache():
    """Forget the cached probe. For tests, and for a mode change at runtime."""
    global _probe_cache
    _probe_cache = None
