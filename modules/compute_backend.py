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
from typing import Optional

from modules import directml_device

CONFIG_SECTION = "compute"
CONFIG_KEY = "backend"
ENV_VAR = "VH_BACKEND"

AUTO = "auto"
CUDA = "cuda"
INTEL = "intel"
DIRECTML = "directml"
CPU = "cpu"

# What the settings screen offers, in the order it offers them. Each label names
# the hardware first, because that is what somebody choosing knows about their
# own machine — "OpenVINO" is an implementation detail of the Intel path, and
# which of the two Intel paths answers depends on how torch was built.
CHOICES = (
    (AUTO, "Automatic — the fastest backend this machine has"),
    (CUDA, "NVIDIA (CUDA)"),
    (INTEL, "Intel (OpenVINO)"),
    (DIRECTML, "AMD / any DX12 card (DirectML)"),
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
    "cpu": CPU, "processor": CPU, "none": CPU,
}


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
    return _publish(backend, log, note="")


def set_now(value, log=print) -> Optional[str]:
    """Change the backend for this process *and* the ones it starts.

    What the settings combo calls. Both halves are needed: the environment is
    what a worker process inherits, and clearing DirectML's probe cache is what
    makes the change visible to anything that already asked in this process.
    """
    backend = normalise(value)
    if backend is None:
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
    elif backend in (CUDA, INTEL, CPU):
        directml_device.set_mode(directml_device.MODE_OFF)
        os.environ[directml_device.MODE_ENV] = directml_device.MODE_OFF
    else:
        directml_device.set_mode(None)
        os.environ.pop(directml_device.MODE_ENV, None)

    if backend != AUTO:
        log(f"🎛️ Compute backend: {label_for(backend)}{note}")
    return backend
