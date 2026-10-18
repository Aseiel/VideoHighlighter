"""Structural checks on catalog entries.

A translation may reword anything, but some parts of the English are not
words: format fields, HTML tags, line breaks, a shortcut-safe "&&". When a
translation drops or mangles one, the UI shows broken markup, crashes on
.format(), or underlines a letter -- worse than showing the English. These
checks find such entries so they can be fixed or left to fall back.
"""
from __future__ import annotations

import re

_FIELD = re.compile(r"\{[^{}]*\}")
_TAG = re.compile(r"</?[a-zA-Z][^<>]*>")
_LONE_AMP = re.compile(r"(?<!&)&(?!&)")


def problems(msgid: str, text: str) -> list[str]:
    """What is structurally wrong with `text` as a translation of `msgid`;
    empty when nothing is."""
    out = []
    if not text.strip():
        out.append("empty")
        return out
    if sorted(_FIELD.findall(msgid)) != sorted(_FIELD.findall(text)):
        out.append("format fields differ")
    if len(_TAG.findall(msgid)) != len(_TAG.findall(text)):
        out.append("HTML tags differ")
    if msgid.count("\n") != text.count("\n"):
        out.append("line breaks differ")
    if len(_LONE_AMP.findall(text)) > len(_LONE_AMP.findall(msgid)):
        out.append("single & (a Qt shortcut marker)")
    return out
