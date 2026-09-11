"""Let the user set the interface size, before Qt decides it for them.

Qt takes the display's scale factor from the OS, and Windows picks that from
the monitor's size and pixel density. On a 55" 4K panel used as a monitor it
lands at 150% or 200%, which is right for a television at three metres and
enormous for an app at desk distance. Nothing in the app could override it: the
scale factor is read once, when the QApplication is constructed, and after that
nothing can change it.

So this runs *before* that, and sets ``QT_SCALE_FACTOR`` — the multiplier Qt
applies on top of whatever the OS said. 0.75 on a 200% display gives an
interface at 150%, and every widget, font and icon follows, because it is the
same mechanism the display scale itself uses.

Where the value comes from, first match winning:

``VH_UI_SCALE``
    An environment variable, for trying a number without touching a file.
``ui_scale`` in ``config.yaml``
    For keeping it.

Absent from both, nothing is set and Qt behaves exactly as before.
"""

from __future__ import annotations

import os
from typing import Optional

ENV_VAR = "VH_UI_SCALE"
CONFIG_KEY = "ui_scale"
QT_VAR = "QT_SCALE_FACTOR"

# Below the floor the interface stops being usable and starts being a puzzle;
# above the ceiling a maximised window no longer fits its own minimum size. Both
# are generous — the useful range on a large 4K panel is about 0.6 to 1.0.
MIN_SCALE, MAX_SCALE = 0.4, 4.0


def _clean(value) -> Optional[float]:
    """A usable multiplier, or None. A bad value is not worth refusing to start
    over: the reader gets a line in the log and the OS default."""
    if value is None or str(value).strip() == "":
        return None
    try:
        scale = float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        print(f"⚠️ {ENV_VAR}/{CONFIG_KEY}: {value!r} is not a number — ignoring")
        return None
    if not (MIN_SCALE <= scale <= MAX_SCALE):
        print(f"⚠️ interface scale {scale} is outside "
              f"{MIN_SCALE}–{MAX_SCALE} — ignoring")
        return None
    return scale


def _from_config() -> Optional[float]:
    """``ui_scale`` from config.yaml, read defensively.

    Imported and parsed here rather than through the app's own config loader:
    this runs before Qt exists and before most of the app is imported, and a
    malformed config must not be the reason the window never appears.
    """
    try:
        import yaml

        from modules.app_paths import data_file

        path = data_file("config.yaml")
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception as e:  # noqa: BLE001 - see docstring
        print(f"⚠️ could not read {CONFIG_KEY} from config.yaml: "
              f"{type(e).__name__}: {e}")
        return None
    if isinstance(data, dict):
        return _clean(data.get(CONFIG_KEY))
    return None


def configured() -> Optional[float]:
    """The scale the user asked for, or None for "whatever the OS says"."""
    from_env = _clean(os.environ.get(ENV_VAR))
    return from_env if from_env is not None else _from_config()


def apply() -> Optional[float]:
    """Set ``QT_SCALE_FACTOR`` if a scale was configured. Returns what it set.

    **Call before constructing the QApplication.** Qt reads the variable once,
    at that moment; setting it afterwards changes nothing and looks like the
    setting is broken.

    An existing ``QT_SCALE_FACTOR`` in the environment is left alone — somebody
    who set Qt's own variable by hand outranks the config file.
    """
    if os.environ.get(QT_VAR):
        return None
    scale = configured()
    if scale is None:
        return None
    os.environ[QT_VAR] = repr(float(scale))
    print(f"🔍 Interface scale: {scale}× (on top of the display's own)")
    return scale
