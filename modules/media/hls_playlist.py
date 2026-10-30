"""Read an HLS playlist well enough to fetch it segment by segment.

For the one case nothing else can download: a stream that only answers the
browser session that loaded its page. The session fetches each piece itself;
this says which pieces, in what order. No network here.
"""
from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field
from typing import List, Optional

_ATTR = re.compile(r'([A-Z0-9-]+)=("[^"]*"|[^,]*)')


def is_playlist(url: str, body: Optional[bytes | str] = None) -> bool:
    """An HLS playlist, by its URL or by its first bytes."""
    path = urllib.parse.urlparse(url).path.lower()
    if path.endswith(".m3u8"):
        return True
    if body is None:
        return False
    head = body[:16] if isinstance(body, str) else body[:16].decode("latin-1")
    return head.lstrip("﻿").startswith("#EXTM3U")


def _attrs(line: str) -> dict:
    _tag, _sep, rest = line.partition(":")
    return {k: v.strip('"') for k, v in _ATTR.findall(rest)}


def best_variant(text: str, base_url: str) -> Optional[str]:
    """The highest-bandwidth stream of a master playlist; None for a media one."""
    best, best_bw = None, -1
    lines = [l.strip() for l in text.splitlines()]
    for i, line in enumerate(lines):
        if not line.startswith("#EXT-X-STREAM-INF"):
            continue
        uri = next((l for l in lines[i + 1:] if l and not l.startswith("#")), None)
        if uri is None:
            continue
        try:
            bw = int(_attrs(line).get("BANDWIDTH", "0"))
        except ValueError:
            bw = 0
        if bw > best_bw:
            best, best_bw = urllib.parse.urljoin(base_url, uri), bw
    return best


@dataclass
class MediaPlaylist:
    segments: List[str] = field(default_factory=list)
    init: Optional[str] = None          # fMP4 init segment (#EXT-X-MAP)
    encrypted: bool = False             # #EXT-X-KEY other than NONE
    seconds: float = 0.0


def media_playlist(text: str, base_url: str) -> MediaPlaylist:
    """Segments of a media playlist, absolute, in order."""
    out = MediaPlaylist()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#EXT-X-KEY"):
            if _attrs(line).get("METHOD", "NONE").upper() != "NONE":
                out.encrypted = True
        elif line.startswith("#EXT-X-MAP") and out.init is None:
            uri = _attrs(line).get("URI")
            if uri:
                out.init = urllib.parse.urljoin(base_url, uri)
        elif line.startswith("#EXTINF"):
            try:
                out.seconds += float(line.split(":", 1)[1].split(",", 1)[0])
            except (IndexError, ValueError):
                pass
        elif not line.startswith("#"):
            out.segments.append(urllib.parse.urljoin(base_url, line))
    return out
