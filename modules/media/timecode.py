"""Seconds to a clock string and back, one format for every display.

The viewer used to format time in a dozen places, each with its own idea:
the ruler said ``1:09:05`` past an hour while the player clock said
``69:05`` for the same moment. Under an hour everything reads ``MM:SS``
(the ruler's form); from an hour up it is ``H:MM:SS``, never minutes
counted past sixty.

``parse_clock`` reads back what a person types to jump somewhere, in any
form they are likely to use: ``21:31``, ``1:09:05``, ``4145`` (seconds),
``47:50.5``, ``1h9m5s``.
"""
from __future__ import annotations

import re
from typing import Optional


def clock(seconds: float, decimals: int = 0) -> str:
    """``MM:SS`` under an hour, ``H:MM:SS`` from an hour up.

    ``decimals`` adds fractions of a second (``MM:SS.d``). Rounding happens
    once, on the whole value, so 59.96 s at one decimal is ``01:00.0`` and
    never ``00:60.0``.
    """
    decimals = max(0, int(decimals))
    total = round(max(0.0, float(seconds or 0.0)), decimals)
    whole = int(total)
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    if decimals:
        secs = f"{s + (total - whole):0{3 + decimals}.{decimals}f}"
    else:
        secs = f"{s:02d}"
    return f"{h:d}:{m:02d}:{secs}" if h else f"{m:02d}:{secs}"


_UNITS = re.compile(r"^(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?$")


def parse_clock(text: str) -> Optional[float]:
    """Seconds from a typed time, or None when it is not one.

    Accepts ``SS``, ``M:SS``, ``H:MM:SS`` (any part may carry a fraction on
    the last field), and ``1h9m5s``-style units. A comma works as the decimal
    point. Minutes or seconds past 59 are taken as written - ``69:05`` is
    4145 s - because that is what someone copying an old display meant.
    """
    t = (text or "").strip().lower().replace(",", ".").replace(" ", "")
    if not t:
        return None
    if ":" in t:
        parts = t.split(":")
        if len(parts) > 3 or any(p == "" for p in parts):
            return None
        try:
            values = [float(p) for p in parts]
        except ValueError:
            return None
        if any(v < 0 for v in values) or any(not p.isdigit() for p in parts[:-1]):
            return None
        total = 0.0
        for v in values:
            total = total * 60 + v
        return total
    m = _UNITS.match(t)
    if m and any(m.groups()):
        h, mi, s = (float(g) if g else 0.0 for g in m.groups())
        return h * 3600 + mi * 60 + s
    try:
        value = float(t)
    except ValueError:
        return None
    return value if value >= 0 else None
