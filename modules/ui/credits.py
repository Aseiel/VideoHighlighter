"""People outside the project whose requests and reports improved it.

Shown in About and listed in CREDITS.md. One list, byte-identical in both
editions, so a name added here reaches both without a second edit of either
About tab.

An entry says what changed because of the person, with what was measured
when the change was made, so a thank-you is also a record. Add one only with
the person's agreement to be named.
"""
from __future__ import annotations

from typing import NamedTuple, Tuple


class Credit(NamedTuple):
    name: str
    what: Tuple[str, ...]


CREDITS: Tuple[Credit, ...] = (
    Credit("Scott Napier", (
        "Shots: asked for a frame-accurate shot list for subtitling. "
        "The result was the list exported as EDL, found by a learned "
        "shot-boundary model: 436 shots in a 20-minute episode, in 46 s.",
        "Subtitles: asked for subtitles timed to shot changes. On the same "
        "episode, lines starting on a cut went from 17 to 115, and lines "
        "running into the next shot from 55 to 12.",
    )),
)


def html() -> str:
    """The list for a rich-text label."""
    from html import escape
    parts = []
    for credit in CREDITS:
        items = "".join(f"<li>{escape(w)}</li>" for w in credit.what)
        parts.append(f"<b>{escape(credit.name)}</b><ul style='margin:2px 0 6px 0'>"
                     f"{items}</ul>")
    return "".join(parts)
