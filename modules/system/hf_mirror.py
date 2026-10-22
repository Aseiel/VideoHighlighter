"""Point Hugging Face downloads at a mirror, where huggingface.co is out of reach.

huggingface.co is blocked in some places, mainland China among them, and the
app fetches from it by name: models people share on the hub (model_hub), and
model weights that are not installed locally. ``huggingface_hub`` decides where
to go from ``HF_ENDPOINT``, read once, when it is first imported. So this sets
it at startup, before anything imports it.

Where the value comes from, first match winning:

``HF_ENDPOINT``
    huggingface_hub's own variable, already set: left exactly as it is.
``VH_HF_ENDPOINT``
    For trying a mirror without touching a file.
``hf_endpoint`` in ``config.yaml``
    For keeping it, e.g. ``hf_endpoint: https://hf-mirror.com``.

Absent from all three, nothing is set and downloads go to huggingface.co as
before. A mirror serves the same repos at the same paths, but what it sends is
its own word, so this is never switched on for anyone: the user picks a mirror
they trust. Packs do not come through here; they have their own mirror and are
checked against pinned hashes (modules/packs/pack_manager.py).
"""

from __future__ import annotations

import os
from typing import Optional

ENV_VAR = "VH_HF_ENDPOINT"
CONFIG_KEY = "hf_endpoint"
HF_VAR = "HF_ENDPOINT"


def _clean(value) -> Optional[str]:
    """A base URL to hand huggingface_hub, or None. A bad value is not worth
    refusing to start over: the reader gets a line in the log and huggingface.co."""
    text = str(value or "").strip()
    if not text:
        return None
    if "://" not in text:
        text = "https://" + text.lstrip("/")
    text = text.rstrip("/")
    scheme, _, rest = text.partition("://")
    if scheme.lower() not in ("http", "https") or not rest:
        print(f"hf_mirror: {ENV_VAR}/{CONFIG_KEY}: {value!r} is not an http(s) address; ignoring")
        return None
    return text


def _from_config() -> Optional[str]:
    """``hf_endpoint`` from config.yaml, read defensively, as ui_scale reads
    its own key: this runs before most of the app is imported."""
    try:
        import yaml

        from modules.system.app_paths import data_file

        path = data_file("config.yaml")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception as e:  # noqa: BLE001 - see docstring
        print(f"hf_mirror: could not read {CONFIG_KEY} from config.yaml: "
              f"{type(e).__name__}: {e}")
        return None
    if isinstance(data, dict):
        return _clean(data.get(CONFIG_KEY))
    return None


def configured() -> Optional[str]:
    """The mirror the user asked for, or None for huggingface.co."""
    from_env = _clean(os.environ.get(ENV_VAR))
    return from_env if from_env is not None else _from_config()


def apply() -> Optional[str]:
    """Set ``HF_ENDPOINT`` if a mirror was configured. Returns what it set.

    **Call before anything imports huggingface_hub** (transformers does, and
    so does model_hub); it reads the variable once, at that moment. Child
    processes inherit it with the rest of the environment.
    """
    if os.environ.get(HF_VAR):
        return None
    endpoint = configured()
    if endpoint is None:
        return None
    os.environ[HF_VAR] = endpoint
    # Plain text: the sidecar can run with no UTF-8 console behind it.
    print(f"hf_mirror: Hugging Face downloads go through {endpoint}")
    return endpoint
