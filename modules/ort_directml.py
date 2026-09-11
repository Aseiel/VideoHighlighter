"""ONNX Runtime's DirectML execution provider — the GPU path a packaged build
can actually ship.

``torch-directml`` cannot be bundled. It pins an exact torch (2.4.1 for the
current release) and pip satisfies that by *replacing* whatever torch is
installed, so a build carrying it could not also carry the CUDA torch the
NVIDIA path needs — one process, one torch. That is why
``modules/directml_device.py`` is opt-in and source-only, and why an AMD user
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
learning two idioms. :mod:`modules.onnx_detector` is the first consumer.

**One onnxruntime, ever.** ``onnxruntime``, ``onnxruntime-gpu`` and
``onnxruntime-directml`` all install the same ``onnxruntime`` package and
overwrite each other. Only the DirectML one belongs in ``requirements.txt``:
NVIDIA already has torch+CUDA and Intel has OpenVINO, so a second accelerated
ORT build would buy nothing and break the one that matters.

The user's switch is shared with the torch backend: ``VH_DIRECTML=off`` turns
off DirectML, whichever runtime would have provided it.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

from modules import directml_device as _dml

# ORT's name for the provider. A literal here is fine — unlike torch's backend
# name, this string is part of ONNX Runtime's public API and is what
# get_available_providers() returns.
PROVIDER = "DmlExecutionProvider"
CPU_PROVIDER = "CPUExecutionProvider"

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

    __slots__ = ("available", "reason", "version", "providers")

    def __init__(self, available=False, reason=None, version=None, providers=()):
        self.available = available
        self.reason = reason
        self.version = version
        self.providers = tuple(providers)

    def __repr__(self):  # pragma: no cover - diagnostics only
        state = "available" if self.available else f"unavailable ({self.reason})"
        return f"<ProviderProbe {state} ort={self.version}>"


def _import_onnxruntime():
    """Imported lazily and never at module scope: ORT loads native libraries,
    and this module is imported by the device probe on machines that will never
    touch it."""
    import onnxruntime  # noqa: PLC0415 - deliberate, see docstring
    return onnxruntime


def device_id() -> int:
    """Which DX12 adapter to bind. 0 unless the user picked another."""
    try:
        return max(0, int(str(os.environ.get(DEVICE_ENV, "")).strip() or 0))
    except (TypeError, ValueError):
        return 0


def probe(refresh: bool = False) -> ProviderProbe:
    """Can ONNX Runtime run a model on this machine's GPU?

    Cached, because the answer cannot change inside a process and the import
    costs real time. ``refresh=True`` is for tests and for a settings UI that
    just changed the mode.
    """
    global _probe_cache
    if _probe_cache is not None and not refresh:
        return _probe_cache

    # One switch for both DirectML runtimes: a user who turned DirectML off
    # means off, not "off for torch and on for ONNX".
    if not _dml.enabled():
        _probe_cache = ProviderProbe(reason=f"disabled ({_dml.MODE_ENV}=off)")
        return _probe_cache

    try:
        ort = _import_onnxruntime()
    except Exception as e:  # noqa: BLE001 - a missing package is a normal answer
        _probe_cache = ProviderProbe(
            reason=f"onnxruntime is not installed ({type(e).__name__}: {e})")
        return _probe_cache

    version = getattr(ort, "__version__", None)
    try:
        providers = tuple(ort.get_available_providers())
    except Exception as e:  # noqa: BLE001
        _probe_cache = ProviderProbe(
            reason=f"onnxruntime could not list its providers ({e})",
            version=version)
        return _probe_cache

    if PROVIDER not in providers:
        # The plain `onnxruntime` wheel reports CPU only. Saying which build is
        # installed is the difference between a fixable message and a shrug.
        _probe_cache = ProviderProbe(
            reason=("this onnxruntime build has no DirectML provider "
                    f"(has: {', '.join(providers) or 'none'}) — "
                    "onnxruntime-directml is the one that does"),
            version=version, providers=providers)
        return _probe_cache

    _probe_cache = ProviderProbe(available=True, version=version,
                                 providers=providers)
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

    Always ends in CPU. A DirectML session that cannot place one operator falls
    back per-node rather than failing the run, which is the behaviour worth
    having on a backend with partial operator coverage.
    """
    if not available():
        return [CPU_PROVIDER]
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
