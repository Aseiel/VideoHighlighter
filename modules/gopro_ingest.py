"""
gopro_ingest.py — find a GoPro card, copy its footage off, and describe what
arrived.

Why this exists
===============
The engine takes a list of video paths. Getting footage from a camera to that
list is, in practice, the step that goes wrong: cards mount on a drive letter
that changes between sessions, GoPro's filenames sort into the wrong order, a
long recording arrives as several files that are really one take, and a copy
interrupted halfway leaves a truncated file that ffmpeg will happily half-read
later. This module handles those four problems and nothing else. It does not
analyse, cut, or render.

Card detection
==============
A GoPro card is identified by structure, not by drive letter or volume label:
``DCIM/1xxGOPRO/`` must exist. ``MISC/version.txt`` is JSON the camera writes
(camera type, firmware, serial); when present it is recorded as provenance and
used to name the destination folder. The label is deliberately *not* trusted —
a reused card can still say "WININSTALL".

Filename convention
===================
GoPro names a file ``GXccnnnn.MP4``: two codec letters (``GH`` = AVC,
``GX`` = HEVC), a two-digit *chapter* number, then a four-digit *file* number.
The chapter sits before the file number, so a plain alphabetical listing
interleaves separate recordings::

    GH010527  GH010528  GH020527  GH020528     <- what sorting gives you
    GH010527  GH020527  GH010528  GH020528     <- actual recording order

Sorting therefore keys on ``(file_number, chapter)``, never on the name. Files
sharing a file number are chapters of one continuous recording and are grouped
into a single :class:`Take`; the camera splits on a size limit, so a take's
chapters are contiguous in time and belong together.

Copy safety
===========
Copies go to a ``.part`` temporary file and are renamed into place only after
the byte count matches the source, so an interrupted run can never leave a
short file that looks complete. A file already at the destination with the
right size is skipped, which makes re-running the ingest cheap and idempotent
(the card is usually still inserted after a failure). ``verify="hash"`` adds a
BLAKE2b comparison for callers who want to pay for it; the default size check
is what catches the realistic failure — a truncated transfer.

Nothing is ever deleted from the card. Freeing space is a decision this module
does not make.

Public API
==========
    find_gopro_cards()                 -> list[GoProCard]
    scan_card(card)                    -> list[Take]
    ingest(card, dest_root, ...)       -> IngestResult
    write_manifest(result, path)       -> str
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import string
import time
from dataclasses import dataclass, field, asdict

# GXccnnnn.MP4 — codec letters, chapter, file number. THM/LRV companions share
# the stem but are previews, not footage, and are matched separately.
_GOPRO_NAME = re.compile(r"^(G[HXL])(\d{2})(\d{4})\.MP4$", re.IGNORECASE)
_MEDIA_DIR = re.compile(r"^\d{3}GOPRO$", re.IGNORECASE)

_CODEC_LABEL = {"GH": "AVC", "GX": "HEVC", "GL": "LRV"}

# Read/write in 8 MiB blocks: large enough that per-call overhead disappears on
# 4 GB files, small enough to keep progress callbacks responsive.
_CHUNK = 8 * 1024 * 1024


@dataclass
class GoProCard:
    """A mounted card that looks like a GoPro's."""
    root: str                      # e.g. "G:\\"
    media_dirs: list[str] = field(default_factory=list)
    camera_type: str = ""          # "HERO13 Black" when MISC/version.txt is readable
    firmware: str = ""
    serial: str = ""
    file_count: int = 0
    total_bytes: int = 0

    @property
    def label(self) -> str:
        """Human name for the card, falling back to the mount point when the
        camera never wrote a version file."""
        return self.camera_type or f"GoPro card ({self.root})"


@dataclass
class Clip:
    """One .MP4 on the card."""
    path: str
    name: str
    codec: str                     # "AVC" / "HEVC"
    chapter: int
    file_number: int
    size: int
    mtime: float

    @property
    def sort_key(self) -> tuple[int, int]:
        return (self.file_number, self.chapter)


@dataclass
class Take:
    """One continuous recording: every chapter sharing a file number, in order."""
    file_number: int
    clips: list[Clip] = field(default_factory=list)

    @property
    def size(self) -> int:
        return sum(c.size for c in self.clips)

    @property
    def started_at(self) -> float:
        return min((c.mtime for c in self.clips), default=0.0)

    @property
    def is_chaptered(self) -> bool:
        return len(self.clips) > 1


@dataclass
class CopiedFile:
    source: str
    dest: str
    size: int
    skipped: bool = False          # already present with the right size
    duration: float = 0.0          # seconds, 0.0 when unprobed/unprobeable
    width: int = 0
    height: int = 0
    fps: float = 0.0
    rotation: int = 0


@dataclass
class IngestResult:
    card: GoProCard
    dest_root: str
    files: list[CopiedFile] = field(default_factory=list)
    takes: list[list[str]] = field(default_factory=list)   # dest paths per take
    copied_bytes: int = 0
    skipped_bytes: int = 0
    seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def paths(self) -> list[str]:
        """Destination paths in recording order — what the engine consumes."""
        return [f.dest for f in self.files]


def _read_version_file(root: str) -> dict:
    """Parse ``MISC/version.txt``. Returns {} when absent or malformed — a card
    with unreadable metadata is still a perfectly usable card."""
    path = os.path.join(root, "MISC", "version.txt")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return json.loads(fh.read())
    except (OSError, ValueError):
        return {}


def _media_dirs(root: str) -> list[str]:
    """Absolute paths of ``DCIM/1xxGOPRO`` folders, sorted."""
    dcim = os.path.join(root, "DCIM")
    try:
        names = sorted(os.listdir(dcim))
    except OSError:
        return []
    return [os.path.join(dcim, n) for n in names
            if _MEDIA_DIR.match(n) and os.path.isdir(os.path.join(dcim, n))]


def _candidate_roots() -> list[str]:
    """Mount points worth probing.

    Windows has no /media to walk, so every drive letter is tried; the
    ``DCIM`` check below is what actually decides. On POSIX the usual removable
    mount parents are scanned instead of every path on the system.
    """
    if os.name == "nt":
        return [f"{letter}:\\" for letter in string.ascii_uppercase]
    roots = []
    for parent in ("/media", "/run/media", "/mnt", "/Volumes"):
        try:
            for entry in sorted(os.listdir(parent)):
                full = os.path.join(parent, entry)
                if os.path.isdir(full):
                    roots.append(full)
                    # /media/<user>/<label> nests one level deeper.
                    try:
                        roots.extend(os.path.join(full, sub)
                                     for sub in sorted(os.listdir(full))
                                     if os.path.isdir(os.path.join(full, sub)))
                    except OSError:
                        pass
        except OSError:
            continue
    return roots


def find_gopro_cards(extra_roots: list[str] | None = None,
                     scan_mounts: bool = True) -> list[GoProCard]:
    """Every mounted volume whose layout says "GoPro card".

    Detection is structural (``DCIM/1xxGOPRO`` present and holding at least one
    parseable .MP4), so it survives a stale volume label and an unfamiliar
    drive letter. ``extra_roots`` lets a caller add a path the scan would not
    reach — a card image mounted somewhere unusual, or a folder in a test.

    ``scan_mounts=False`` restricts the search to ``extra_roots``, so a caller
    can look at one specific folder without touching the machine's real drives.
    Tests use this: probing every drive letter would otherwise pick up whatever
    card is physically in the developer's reader.
    """
    cards: list[GoProCard] = []
    seen: set[str] = set()
    roots = (list(_candidate_roots()) if scan_mounts else []) + list(extra_roots or [])
    for root in roots:
        key = os.path.normcase(os.path.abspath(root))
        if key in seen:
            continue
        seen.add(key)
        dirs = _media_dirs(root)
        if not dirs:
            continue
        clips = [c for d in dirs for c in _scan_media_dir(d)]
        if not clips:
            continue
        info = _read_version_file(root)
        cards.append(GoProCard(
            root=root,
            media_dirs=dirs,
            camera_type=str(info.get("camera type", "") or ""),
            firmware=str(info.get("firmware version", "") or ""),
            serial=str(info.get("camera serial number", "") or ""),
            file_count=len(clips),
            total_bytes=sum(c.size for c in clips),
        ))
    return cards


def _scan_media_dir(directory: str) -> list[Clip]:
    """Parseable .MP4 files in one ``1xxGOPRO`` folder.

    Files that do not match the GoPro pattern are ignored rather than guessed
    at: an unrecognised name has no reliable chapter or take, and inventing one
    would silently reorder someone's footage. ``GL`` (low-res proxy) files are
    excluded too — they are previews of clips already being copied.
    """
    out: list[Clip] = []
    try:
        names = os.listdir(directory)
    except OSError:
        return out
    for name in names:
        match = _GOPRO_NAME.match(name)
        if not match:
            continue
        prefix, chapter, number = match.group(1).upper(), match.group(2), match.group(3)
        if prefix == "GL":
            continue
        path = os.path.join(directory, name)
        try:
            stat = os.stat(path)
        except OSError:
            continue
        out.append(Clip(
            path=path,
            name=name,
            codec=_CODEC_LABEL.get(prefix, prefix),
            chapter=int(chapter),
            file_number=int(number),
            size=stat.st_size,
            mtime=stat.st_mtime,
        ))
    return out


def scan_card(card: GoProCard) -> list[Take]:
    """Group a card's footage into takes, in recording order.

    Chapters of one recording share a file number and are ordered by chapter;
    takes are ordered by file number. See the module docstring for why sorting
    on the filename is wrong.
    """
    clips = [c for d in card.media_dirs for c in _scan_media_dir(d)]
    grouped: dict[int, Take] = {}
    for clip in sorted(clips, key=lambda c: c.sort_key):
        grouped.setdefault(clip.file_number, Take(file_number=clip.file_number)).clips.append(clip)
    return [grouped[n] for n in sorted(grouped)]


def suggest_folder_name(card: GoProCard, takes: list[Take], prefix: str = "") -> str:
    """A destination folder name like ``2026-08-08_HERO13-Black``.

    Dated from the earliest recording rather than "now", so re-ingesting a card
    weeks later still files the footage under the day it was shot.
    """
    stamp = min((t.started_at for t in takes if t.started_at), default=time.time())
    day = time.strftime("%Y-%m-%d", time.localtime(stamp))
    camera = re.sub(r"[^A-Za-z0-9]+", "-", card.camera_type).strip("-")
    parts = [p for p in (prefix.strip(), day, camera) if p]
    return "_".join(parts)


def _hash_file(path: str) -> str:
    """BLAKE2b of a file. Chosen over SHA-256 purely for speed: this runs over
    gigabytes and the comparison is an integrity check, not a security one."""
    digest = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_one(src: str, dest: str, size: int, verify: str,
              progress_fn=None, cancel_check=None) -> bool:
    """Copy ``src`` to ``dest`` atomically. Returns True when bytes were moved,
    False when an intact copy was already there.

    Raises RuntimeError on a size or hash mismatch, and CopyCancelled when
    ``cancel_check`` asks to stop. The partial file is removed in both cases:
    a ``.part`` left behind would be re-copied next run anyway, and a stale one
    is just clutter.
    """
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        if verify != "hash" or _hash_file(dest) == _hash_file(src):
            return False

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    copied = 0
    try:
        with open(src, "rb") as fin, open(part, "wb") as fout:
            while True:
                if cancel_check is not None and cancel_check():
                    raise CopyCancelled(os.path.basename(src))
                block = fin.read(_CHUNK)
                if not block:
                    break
                fout.write(block)
                copied += len(block)
                if progress_fn is not None:
                    progress_fn(len(block))
            # Force the bytes out before the rename claims the file is whole;
            # without this a power loss can leave a correctly-named short file.
            fout.flush()
            os.fsync(fout.fileno())
    except BaseException:
        _quiet_remove(part)
        raise

    actual = os.path.getsize(part)
    if actual != size:
        _quiet_remove(part)
        raise RuntimeError(
            f"{os.path.basename(src)}: copied {actual} bytes, expected {size}")
    if verify == "hash" and _hash_file(part) != _hash_file(src):
        _quiet_remove(part)
        raise RuntimeError(f"{os.path.basename(src)}: hash mismatch after copy")

    os.replace(part, dest)
    shutil.copystat(src, dest, follow_symlinks=False)
    return True


def _quiet_remove(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


class CopyCancelled(RuntimeError):
    """Raised between blocks when cancel_check() says stop. The in-flight
    ``.part`` file is deleted, so a cancelled ingest leaves only whole files."""


def _probe(path: str) -> dict:
    """Metadata for a copied file, or zeros when ffprobe is unavailable.

    Deferred import and swallowed errors on purpose: the copy is the job here,
    and a missing ffprobe must not turn a successful ingest into a failure. A
    zero duration is visibly wrong downstream, which is the honest outcome.
    """
    try:
        from modules.video_probe import probe_video
        return probe_video(path)
    except Exception:
        return {}


def ingest(card: GoProCard, dest_root: str, *, folder_name: str = "",
           verify: str = "size", probe: bool = True, log_fn=print,
           progress_fn=None, cancel_check=None) -> IngestResult:
    """Copy every take off ``card`` into ``dest_root/folder_name``.

    ``progress_fn(done_bytes, total_bytes, filename)`` is called as bytes move,
    which is the only progress signal that means anything when one file can be
    500 MB. ``verify`` is "size" (default) or "hash". ``probe`` fills in
    duration/resolution per file for the manifest.

    Files that fail individually are recorded in ``result.errors`` and the run
    continues: one unreadable clip on a card is a bad sector, not a reason to
    abandon the other twenty. Cancellation, by contrast, stops everything.
    """
    started = time.time()
    takes = scan_card(card)
    if not folder_name:
        folder_name = suggest_folder_name(card, takes)
    dest_dir = os.path.join(dest_root, folder_name)
    os.makedirs(dest_dir, exist_ok=True)

    total = sum(t.size for t in takes)
    result = IngestResult(card=card, dest_root=dest_dir)
    done = 0

    log_fn(f"📥 Ingesting {len(takes)} take(s), {_gb(total)} from {card.label} -> {dest_dir}")

    for take in takes:
        take_dests: list[str] = []
        for clip in take.clips:
            dest = os.path.join(dest_dir, clip.name)

            def bump(n: int, _dest=dest) -> None:
                nonlocal done
                done += n
                if progress_fn is not None:
                    progress_fn(done, total, os.path.basename(_dest))

            try:
                moved = _copy_one(clip.path, dest, clip.size, verify,
                                  progress_fn=bump, cancel_check=cancel_check)
            except CopyCancelled:
                result.seconds = time.time() - started
                log_fn("⏹️ Ingest cancelled")
                raise
            except (OSError, RuntimeError) as exc:
                result.errors.append(f"{clip.name}: {exc}")
                log_fn(f"⚠️ {clip.name}: {exc}")
                # Bytes for this file never landed; keep the progress total
                # honest so the bar still reaches 100%.
                done += clip.size
                if progress_fn is not None:
                    progress_fn(done, total, clip.name)
                continue

            if moved:
                result.copied_bytes += clip.size
            else:
                result.skipped_bytes += clip.size
                done += clip.size
                if progress_fn is not None:
                    progress_fn(done, total, clip.name)
                log_fn(f"↩️ {clip.name} already copied, skipping")

            meta = _probe(dest) if probe else {}
            result.files.append(CopiedFile(
                source=clip.path, dest=dest, size=clip.size, skipped=not moved,
                duration=float(meta.get("duration") or 0.0),
                width=int(meta.get("width") or 0),
                height=int(meta.get("height") or 0),
                fps=float(meta.get("fps") or 0.0),
                rotation=int(meta.get("rotation") or 0),
            ))
            take_dests.append(dest)

        if take_dests:
            result.takes.append(take_dests)

    result.seconds = time.time() - started
    log_fn(f"✅ Ingest done: {_gb(result.copied_bytes)} copied, "
           f"{_gb(result.skipped_bytes)} already present, "
           f"{len(result.errors)} error(s) in {result.seconds:.0f}s")
    return result


def _gb(n: int) -> str:
    return f"{n / (1024 ** 3):.2f} GB"


def write_manifest(result: IngestResult, path: str = "") -> str:
    """Write ``ingest.json`` describing what landed, and return its path.

    The manifest is the handoff to the rest of the pipeline (and the record of
    where footage came from once the card is reused). Takes are listed as
    groups of destination paths so a chaptered recording stays recognisable as
    one thing after ingest.
    """
    path = path or os.path.join(result.dest_root, "ingest.json")
    payload = {
        "version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "card": asdict(result.card),
        "dest_root": result.dest_root,
        "seconds": round(result.seconds, 1),
        "copied_bytes": result.copied_bytes,
        "skipped_bytes": result.skipped_bytes,
        "errors": result.errors,
        "takes": result.takes,
        "files": [asdict(f) for f in result.files],
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def read_manifest(path: str) -> dict:
    """Load a manifest written by :func:`write_manifest`. Raises on unreadable
    or non-v1 files — a caller acting on a manifest it cannot parse would
    silently process the wrong footage."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("version") != 1:
        raise ValueError(f"unsupported manifest version: {data.get('version')!r}")
    return data
