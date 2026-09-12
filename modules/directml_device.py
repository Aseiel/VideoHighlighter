"""
modules/directml_device.py
==========================
Experimental DirectML backend, so an AMD card is not automatically a CPU-only
machine.

Why DirectML and not ROCm
-------------------------
ROCm is the vendor's own stack and it is faster, but it is Linux-first and AMD
dropped consumer GPUs from it early: Polaris (RX 470/570/580) has had no
support since ROCm 4.5, and the Windows HIP SDK reports "no ROCm-capable
device detected" on exactly the cards most likely to be sitting in a machine
that would benefit. ZLUDA (CUDA-on-AMD translation) is the other option and is
a dependency maze. Requiring either would mean telling an AMD user to install
Linux to use this app, which is not a real answer.

DirectML is Microsoft's compute backend on top of DirectX 12, so it works on
anything with a DX12 driver -- every AMD GPU since Polaris, plus Intel and
NVIDIA -- on stock Windows, with no vendor SDK. It is slower than CUDA/ROCm and
its operator coverage is a subset of torch's, which is why everything here is
switchable and every consumer keeps its existing fallback.

Licensing: ``torch-directml`` is MIT and is **not** in ``requirements.txt``
and **not** in the frozen build. A user who wants it installs it themselves.
That is deliberate -- see "The install is destructive" below -- and it also
keeps the shipped bundle free of a dependency we have not audited for
redistribution.

The install is destructive, which is why this is opt-in
-------------------------------------------------------
``torch-directml`` pins an exact ``torch`` version and pip will happily satisfy
that by *replacing* whatever torch is installed. On a machine with a ``+xpu``
or ``+cu124`` build that silently removes Arc or CUDA support, and the app then
looks broken for an unrelated reason. So DirectML must never be a hard
requirement, must never be installed as a side effect of anything, and belongs
in its own virtualenv on a machine that also has another accelerator.

The version strings are also a trap: there is no ``torch-directml>=1.13``.
Releases are dated dev builds (``0.2.5.dev240914``), so a requirement pin with
a normal-looking floor resolves to nothing at all.

What this module does and does not decide
-----------------------------------------
It answers "is there a usable DirectML device, and what is torch's name for
it". It does not decide policy: :func:`mode` reads the opt-in, and each caller
decides whether DirectML is appropriate for *its* model. Nothing here imports
torch at module scope, so it stays free to be imported during preflight.

Never raises. A probe that throws is a probe that costs somebody their GPU.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

# Opt-in switch. "auto" (the default) uses DirectML only when no CUDA and no
# Intel path was found -- i.e. only where the alternative is the CPU, so it can
# never displace a faster backend. "force" puts it ahead of everything, which is
# how you test it on a machine that also has another card. "off" disables it.
MODE_ENV = "VH_DIRECTML"
MODE_OFF, MODE_AUTO, MODE_FORCE = "off", "auto", "force"
_MODES = {
    "0": MODE_OFF, "off": MODE_OFF, "false": MODE_OFF, "no": MODE_OFF,
    "auto": MODE_AUTO, "": MODE_AUTO,
    "1": MODE_FORCE, "on": MODE_FORCE, "force": MODE_FORCE,
    "true": MODE_FORCE, "yes": MODE_FORCE,
}

# fp16 is off by default on this backend. DirectML implements half precision
# unevenly across operators, and a model that silently falls back to fp32 for
# one layer pays a conversion on every call instead of saving anything. The
# models this app runs on DirectML are small enough that fp32 fits, so the
# default is the one that works; set this to opt in and measure.
FP16_ENV = "VH_DIRECTML_FP16"

# Torch's name for the DirectML backend since torch-directml 0.2. Only a
# fallback for the string: the real one is read off the device object the
# package hands back, because 0.1.x called it "dml" and a future build may
# call it something else again.
_FALLBACK_BACKEND = "privateuseone"

# What a user might reasonably type in a device field or leave in a config file.
_ALIASES = ("dml", "directml", "privateuseone")

# DirectML enumerates *every* DX12 adapter, and some of them are not graphics
# cards. Microsoft's software renderer (WARP) is a CPU implementation of D3D12
# that will happily accept a model and run it slower than the CPU path it just
# replaced — with no error and no way to tell from outside. So adapters whose
# name matches one of these are skipped when picking a device, and a machine
# whose *only* adapter is one of them is reported as having no DirectML at all.
_SOFTWARE_MARKERS = ("microsoft basic", "basic render", "software adapter",
                     "software renderer", "warp")


def _is_software_adapter(name) -> bool:
    lowered = str(name or "").lower()
    return any(marker in lowered for marker in _SOFTWARE_MARKERS)

_probe_cache = None            # Optional[DirectMLProbe]
_mode_override = None          # Optional[str]


# ---------------------------------------------------------------------------
# The opt-in
# ---------------------------------------------------------------------------

def mode() -> str:
    """``"off"`` | ``"auto"`` | ``"force"``.

    A process-level override (:func:`set_mode`, for a settings UI) wins over
    the environment variable, which is what a user sets to try this without a
    rebuild. Anything unrecognised means "auto" rather than an error, because
    the cost of a typo here should be the default behaviour, not a crash.
    """
    if _mode_override is not None:
        return _mode_override
    return _MODES.get(str(os.environ.get(MODE_ENV, "")).strip().lower(), MODE_AUTO)


def set_mode(value: Optional[str]) -> str:
    """Override :func:`mode` for this process. ``None`` restores the env var.

    Clears the probe cache: "off" must take effect for a caller that already
    asked, or the switch does not do what the label says.
    """
    global _mode_override, _probe_cache
    _mode_override = None if value is None else _MODES.get(
        str(value).strip().lower(), MODE_AUTO)
    _probe_cache = None
    return mode()


def enabled() -> bool:
    """True unless the user switched DirectML off."""
    return mode() != MODE_OFF


def forced() -> bool:
    """True when DirectML should be tried ahead of CUDA/Intel."""
    return mode() == MODE_FORCE


def prefer_float16() -> bool:
    """Whether to run DirectML models in fp16. False unless opted in -- see
    :data:`FP16_ENV`."""
    return str(os.environ.get(FP16_ENV, "")).strip().lower() in (
        "1", "on", "true", "yes")


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------

def _missing_package_reason(exc) -> str:
    """Why the package is not here, in words the reader can act on.

    A packaged build never has it and never can (see the module docstring), so
    telling that user it "is not installed" reads as an instruction they have no
    way to follow: there is no pip inside an exe.

    It deliberately says nothing about what ONNX Runtime is doing. This module
    cannot see that, and the caller that can
    (`device_utils._any_directml_info`) now withholds this line entirely
    whenever the other runtime is about to announce itself — so by the time a
    user reads this, DirectML really is absent and a claim about a second
    runtime would only be wrong.
    """
    if getattr(sys, "frozen", False):
        return ("a packaged build cannot carry torch-directml — it pins an "
                "exact torch and this build ships the CUDA one. Running from "
                "source adds it: see docs/AMD-GPU.md")
    return f"torch-directml is not installed ({type(exc).__name__}: {exc})"


class DirectMLProbe:
    """What one look at this machine's DirectML support found.

    ``reason`` is populated whenever ``available`` is False and is meant to be
    shown to a user: "no DX12 GPU" and "torch-directml is not installed" are
    the same outcome and completely different problems.
    """

    __slots__ = ("available", "device_count", "names", "backend", "version",
                 "reason", "preferred_index", "software_only")

    def __init__(self, available=False, device_count=0, names=(),
                 backend=_FALLBACK_BACKEND, version=None, reason=None):
        self.available = bool(available)
        self.device_count = int(device_count)
        self.names = list(names)
        self.backend = backend
        self.version = version
        self.reason = reason

        # Which adapter to actually use. Not always 0: on a machine where the
        # software renderer enumerates first, index 0 is the slowest device
        # present, and taking it silently is the whole failure this guards.
        real = [i for i, n in enumerate(self.names)
                if not _is_software_adapter(n)]
        self.software_only = bool(self.names) and not real
        self.preferred_index = real[0] if real else 0

    def device_string(self, index: Optional[int] = None) -> Optional[str]:
        """Torch's string for an adapter. ``None`` means the preferred one."""
        if index is None:
            index = self.preferred_index
        if not self.available or not (0 <= index < max(self.device_count, 1)):
            return None
        return f"{self.backend}:{index}"

    def name(self, index: Optional[int] = None) -> Optional[str]:
        if index is None:
            index = self.preferred_index
        return self.names[index] if 0 <= index < len(self.names) else None

    def __repr__(self):
        if not self.available:
            return f"DirectMLProbe(available=False, reason={self.reason!r})"
        return (f"DirectMLProbe(backend={self.backend!r}, "
                f"devices={self.device_count}, names={self.names!r}, "
                f"preferred={self.preferred_index})")


def _import_torch_directml():
    """The single seam between this module and the hardware.

    Tests monkeypatch this; everything else in the module goes through it, so
    an AMD box can be simulated from a machine that has never had one.
    """
    import torch_directml
    return torch_directml


def probe(refresh: bool = False) -> DirectMLProbe:
    """Look for a DirectML device. Cached; never raises.

    Cached because ``import torch_directml`` pulls in torch and enumerates DX12
    adapters, which is far too expensive to repeat per frame -- and because the
    answer cannot change while the process runs.
    """
    global _probe_cache
    if _probe_cache is not None and not refresh:
        return _probe_cache

    if not enabled():
        _probe_cache = DirectMLProbe(reason=f"disabled ({MODE_ENV}=off)")
        return _probe_cache

    try:
        dml = _import_torch_directml()
    except Exception as e:  # noqa: BLE001 -- a missing package is a normal answer
        _probe_cache = DirectMLProbe(reason=_missing_package_reason(e))
        return _probe_cache

    try:
        if not bool(dml.is_available()):
            _probe_cache = DirectMLProbe(
                reason="torch-directml is installed but reports no DirectML "
                       "device (no DX12 GPU, or the display driver is too old)")
            return _probe_cache

        count = int(getattr(dml, "device_count", lambda: 1)())
        names = []
        for i in range(max(count, 1)):
            try:
                names.append(str(dml.device_name(i)))
            except Exception:  # noqa: BLE001 -- a nameless adapter is still one
                names.append(f"DirectML device {i}")

        # Read the backend name off the device object rather than assuming it:
        # 0.1.x called it "dml", 0.2.x calls it "privateuseone".
        backend = _FALLBACK_BACKEND
        try:
            backend = str(dml.device(0)).split(":")[0] or _FALLBACK_BACKEND
        except Exception:  # noqa: BLE001
            pass

        found = DirectMLProbe(
            available=True, device_count=max(count, 1), names=names,
            backend=backend,
            version=str(getattr(dml, "__version__", "") or "") or None,
        )

        # A box whose only DX12 adapter is a software renderer has no GPU, and
        # saying otherwise would be worse than saying nothing: the app would
        # route models onto a CPU implementation of D3D12 and run them slower
        # than the CPU path they came from, with nothing anywhere reporting a
        # fault. `force` still allows it, because testing the plumbing on a
        # machine with no real adapter is a legitimate thing to want.
        if found.software_only and not forced():
            _probe_cache = DirectMLProbe(
                reason=f"the only DirectML adapter is a software renderer "
                       f"({found.names[0]}), which is slower than the CPU — "
                       f"install the graphics driver, or set "
                       f"{MODE_ENV}=force to use it anyway")
            return _probe_cache

        _probe_cache = found
    except Exception as e:  # noqa: BLE001 -- a broken driver must not break us
        _probe_cache = DirectMLProbe(
            reason=f"DirectML probe failed ({type(e).__name__}: {e})")
    return _probe_cache


def is_available() -> bool:
    """True if a DirectML device can be used right now."""
    return probe().available


def unavailable_reason() -> Optional[str]:
    """Why DirectML is not being used, in a sentence, or None if it is."""
    p = probe()
    return None if p.available else p.reason


def device_string(index: Optional[int] = None) -> Optional[str]:
    """Torch's device string for a DirectML adapter (``"privateuseone:0"``),
    or None if there is none. ``index=None`` means the preferred adapter, which
    is not always 0 — see :data:`_SOFTWARE_MARKERS`.

    Prefer this over a literal: the backend name is read from the installed
    package, not assumed.
    """
    return probe().device_string(index)


def adapter_names() -> list:
    """Every DirectML adapter, named. Empty when DirectML is unusable."""
    return list(probe().names)


# ---------------------------------------------------------------------------
# Using the device
# ---------------------------------------------------------------------------

def is_directml(spec) -> bool:
    """True if ``spec`` names the DirectML backend, in any spelling a user or an
    old config might contain (``dml``, ``directml``, ``privateuseone:1``)."""
    if spec is None:
        return False
    head = str(spec).strip().lower().split(":")[0]
    return head in _ALIASES


def normalize(spec) -> Optional[str]:
    """A user's spelling of a DirectML device -> the string torch accepts, or
    None if this machine cannot provide one.

    ``"dml"`` -> ``"privateuseone:0"``, ``"DirectML:1"`` -> ``"privateuseone:1"``.
    An ordinal past the last adapter comes back as None rather than as a string
    that fails later inside a model load, where the message would be useless.
    """
    if not is_directml(spec):
        return None
    p = probe()
    if not p.available:
        return None
    _, _, tail = str(spec).strip().lower().partition(":")
    try:
        # No ordinal means "whichever one the app would pick", not "device 0".
        index = int(tail) if tail else None
    except ValueError:
        index = None
    return p.device_string(index)


def ensure_backend(spec=None) -> bool:
    """Make sure torch's DirectML backend is registered before a device string
    is used, and report whether it is.

    This is the footgun the rest of the app must not have to remember: a bare
    ``tensor.to("privateuseone:0")`` raises unless ``torch_directml`` has been
    imported *in this process*, because that import is what registers the
    backend. A device string surviving into a worker process, or coming back
    out of a config file, is exactly the case where it will not have been. Call
    this immediately before the first ``.to()``.

    Passing a non-DirectML ``spec`` returns True and does nothing, so a caller
    can call it unconditionally.
    """
    if spec is not None and not is_directml(spec):
        return True
    return probe().available


def torch_device(index: Optional[int] = None):
    """The ``torch.device`` object for a DirectML adapter, or None.
    ``index=None`` means the preferred adapter.

    The object rather than the string, for callers that would rather not think
    about backend registration at all.
    """
    p = probe()
    if not p.available:
        return None
    if index is None:
        index = p.preferred_index
    try:
        return _import_torch_directml().device(index)
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ [directml] could not build device {index}: {e}")
        return None


def describe() -> Optional[str]:
    """One line naming the DirectML hardware, for a log or a UI, or None.

    The adapter name matters more here than on other backends: DirectML
    enumerates *every* DX12 adapter, including the integrated one and
    Microsoft's software renderer, so "it found a device" is not the same as
    "it found your graphics card".
    """
    p = probe()
    if not p.available:
        return None
    if not p.names:
        names = f"{p.device_count} device(s)"
    else:
        # Mark the one that will be used. On a two-adapter machine "it found a
        # device" and "it found the right device" are different statements, and
        # this is the line somebody reads to tell them apart.
        names = ", ".join(f"{n} [using]" if i == p.preferred_index else n
                          for i, n in enumerate(p.names))
    version = f" (torch-directml {p.version})" if p.version else ""
    return f"DirectML: {names}{version}"
