"""Which accelerator to use, chosen by name and carried to every process.

The app picks a backend on its own — CUDA, then Intel, then DirectML, then the
processor — and that order is right for almost everybody. It is not testable by
the person holding the machine, though: to find out whether DirectML beats
OpenVINO on a particular card you have to be able to *ask* for it, and until
now the only way to ask was an environment variable that a shortcut-launched
app never sees.

So: a named choice, saved in ``config.yaml`` under ``compute.backend``, and
published into the environment at startup.

**The environment rather than a Python variable**, because object detection can
run in worker *processes*. They re-execute the app's entry point and inherit
the environment and nothing else, so a process-level override would apply to
the window and quietly not to the work.

**A preference rather than a command.** A backend that is not available on this
machine logs why and falls back to the automatic order. A config file copied
between machines, or a card that was swapped out, should cost a line in the log
rather than a run that refuses to start.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from modules.system import directml_device

CONFIG_SECTION = "compute"
CONFIG_KEY = "backend"
ENV_VAR = "VH_BACKEND"

AUTO = "auto"
CUDA = "cuda"
INTEL = "intel"
DIRECTML = "directml"
APPLE = "apple"
CPU = "cpu"

# Every backend the app knows, in the order the settings screen lists them —
# which of them it lists depends on the platform, see `choices()`. Each label names
# the hardware first, because that is what somebody choosing knows about their
# own machine — "OpenVINO" is an implementation detail of the Intel path, and
# which of the two Intel paths answers depends on how torch was built.
CHOICES = (
    (AUTO, "Automatic — the fastest backend this machine has"),
    (CUDA, "NVIDIA (CUDA)"),
    (INTEL, "Intel (OpenVINO)"),
    (DIRECTML, "AMD / any DX12 card (DirectML)"),
    (APPLE, "Apple GPU (Core ML)"),
    (CPU, "Processor only"),
)

# What people write when they mean one of these. Vendor names included: the
# setting is about hardware, and somebody typing "nvidia" into a config file is
# not making a mistake worth punishing.
_ALIASES = {
    "": AUTO, "auto": AUTO, "automatic": AUTO, "default": AUTO, "best": AUTO,
    "cuda": CUDA, "nvidia": CUDA, "gpu": CUDA,
    "intel": INTEL, "openvino": INTEL, "ov": INTEL, "xpu": INTEL, "arc": INTEL,
    "directml": DIRECTML, "dml": DIRECTML, "amd": DIRECTML, "dx12": DIRECTML,
    "apple": APPLE, "mac": APPLE, "coreml": APPLE, "metal": APPLE, "mps": APPLE,
    "cpu": CPU, "processor": CPU, "none": CPU,
}


# A Mac has one accelerator and a PC never has it: CUDA, Intel's GPU plugin and
# DirectML do not exist on macOS, and Core ML exists nowhere else. Offering the
# others anyway is a menu of choices that can only fall back.
_MAC_ONLY = (APPLE,)
_NOT_ON_MAC = (CUDA, INTEL, DIRECTML)


def choices(platform: Optional[str] = None) -> tuple:
    """The (backend, label) pairs this platform can use, in display order.

    ``platform`` is a ``sys.platform`` value; the running one when omitted.
    """
    platform = platform or sys.platform
    skip = _NOT_ON_MAC if platform == "darwin" else _MAC_ONLY
    return tuple((name, label) for name, label in CHOICES if name not in skip)


def offered(backend, platform: Optional[str] = None) -> bool:
    """True when ``backend`` can be chosen on this platform."""
    name = normalise(backend)
    return name is not None and name in dict(choices(platform))


def normalise(value) -> Optional[str]:
    """One of the known backends, or None when the value names nothing."""
    if value is None:
        return None
    return _ALIASES.get(str(value).strip().lower())


def label_for(backend) -> str:
    """The settings-screen label for a backend, for logging it back."""
    return dict(CHOICES).get(normalise(backend) or AUTO, str(backend))


def from_config(config: dict | None) -> Optional[str]:
    """The backend a config dict asks for, or None when it does not ask."""
    if not isinstance(config, dict):
        return None
    section = config.get(CONFIG_SECTION)
    if not isinstance(section, dict):
        return None
    return normalise(section.get(CONFIG_KEY))


def configured() -> Optional[str]:
    """What this process should prefer: the environment, else nothing.

    Read by `device_utils.detect_best_device` on every probe, including in
    worker processes that never loaded a config file.
    """
    return normalise(os.environ.get(ENV_VAR))


def apply(config: dict | None, log=print) -> Optional[str]:
    """Publish the configured backend into the environment. Returns what it set.

    A variable already exported is left alone: somebody who set it before
    launching is testing something, and a config file written weeks ago should
    not overrule them.
    """
    if os.environ.get(ENV_VAR):
        return None
    backend = from_config(config)
    if backend is None:
        return None
    if not offered(backend):
        # A config carried over from another kind of machine — "cuda" from a
        # PC on a Mac. The settings screen cannot show it, so publishing it
        # would leave a run choosing something the user cannot see.
        log(f"ℹ️ Compute backend {label_for(backend)!r} from the settings does "
            f"not exist on this platform — using automatic")
        return None
    return _publish(backend, log, note="")


def set_now(value, log=print) -> Optional[str]:
    """Change the backend for this process *and* the ones it starts.

    What the settings combo calls. Both halves are needed: the environment is
    what a worker process inherits, and clearing DirectML's probe cache is what
    makes the change visible to anything that already asked in this process.
    """
    backend = normalise(value)
    if backend is None or not offered(backend):
        return None
    return _publish(backend, log, note=" (applies to the next run)")


def _publish(backend: str, log, note: str) -> str:
    os.environ[ENV_VAR] = backend

    # DirectML's own switch predates this one and other code still reads it —
    # keep the two saying the same thing rather than leaving a user with a
    # backend chosen here and DirectML disabled there.
    if backend == DIRECTML:
        directml_device.set_mode(directml_device.MODE_FORCE)
        os.environ[directml_device.MODE_ENV] = directml_device.MODE_FORCE
    elif backend in (CUDA, INTEL, APPLE, CPU):
        directml_device.set_mode(directml_device.MODE_OFF)
        os.environ[directml_device.MODE_ENV] = directml_device.MODE_OFF
    else:
        directml_device.set_mode(None)
        os.environ.pop(directml_device.MODE_ENV, None)

    if backend != AUTO:
        log(f"🎛️ Compute backend: {label_for(backend)}{note}")
    return backend
