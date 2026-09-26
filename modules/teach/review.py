"""Check the sorter's guesses: which to look at, how, and what was decided.

Every safeguard in the loop rests on someone having looked, and this is where
they look. Two ways, and both write the same verdicts:

**Contact sheets.** ``next_sheet`` picks a batch, draws it as one image —
numbered tiles, each a sample's start, middle and end (actions) or one frame
(objects), captioned with the guess — and records which tile is which sample.
``apply_verdicts`` then takes "accept 1-5,7; reject 6; 8 is <class>". One
glance per tile, which is what makes a hundred samples a few minutes. It is
also the form an LLM that can see images can check: it reads the sheet and
answers in the same words.

**Folders.** ``sort.lay_out_folders`` puts each sample in the folder of its
guess; a person drags the wrong ones elsewhere, and ``from_folders`` reads
back where everything ended up. The manual process, kept.

What goes on a sheet (``pick_batch``), most useful first, after
docs/CUSTOM-MODEL-TRAINING.md's rules:

1. where the last model and CLIP **disagree** — one of them is wrong;
2. the samples nearest the **decision line** — unsure, or a narrow lead;
3. candidate **negatives** — "none of these" is what stops a model firing on
   everything, and CLIP finds them for free;
4. a slice of **confident** guesses, never zero — the only check on both
   engines being confidently wrong together.

Classes furthest from their target come first within each group.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from typing import Callable, Optional, Sequence

from modules.teach import embed as embed_mod
from modules.teach.project import (
    ACCEPTED, NEGATIVE, NONE, OBJECTS, PENDING, REJECTED, UNSURE, Project,
)

REVIEW_DIR = "review"
DEFAULT_BATCH = 24
CONFIDENT_SHARE = 0.1
NEGATIVE_SHARE = 0.15
AUDIT_SHARE = 0.15

TILE_HEIGHT = 150
CAPTION_HEIGHT = 22


# ---------------------------------------------------------------------------
# What to ask about
# ---------------------------------------------------------------------------

def _top(sample) -> tuple:
    if not sample.scores:
        return ("", 0.0)
    name = max(sample.scores, key=sample.scores.get)
    return name, float(sample.scores[name])


def pick_batch(project: Project, size: int = DEFAULT_BATCH,
               class_name: Optional[str] = None, seed: int = 0) -> list:
    """The next samples worth a person's glance, most useful first."""
    pending = [s for s in project.samples if s.verdict == PENDING and s.scores]
    has_auto = any(s.is_auto for s in project.samples)
    if class_name:
        pending = [s for s in pending
                   if s.proposed == class_name or _top(s)[0] == class_name
                   or s.model_proposed == class_name]
    if not pending and not has_auto:
        return []

    counts = project.counts()

    def need(name: str) -> float:
        c = counts.get(name)
        return (c["accepted"] / max(c["target"], 1)) if c else 1.0

    names = set(project.class_names())
    gate = project.settings.gate

    disagree = [s for s in pending if s.model_proposed and s.proposed in names
                and s.model_proposed != s.proposed]
    rest = [s for s in pending if s not in disagree]
    unsure = sorted((s for s in rest if s.proposed == UNSURE),
                    key=lambda s: (need(_top(s)[0]), abs(_top(s)[1] - gate)))
    narrow = sorted((s for s in rest if s.proposed in names),
                    key=lambda s: (need(s.proposed), s.margin))
    negatives = sorted((s for s in rest if s.proposed == NONE),
                       key=lambda s: -_top(s)[1])     # the closest calls first
    confident = sorted((s for s in rest if s.proposed in names),
                       key=lambda s: -s.margin)

    # Spot checks of auto-accepted samples, most-needed class first.
    from modules.teach import autolabel
    audit_pool = sorted((s for s in project.samples if s.is_auto
                         and (not class_name or s.label == class_name)),
                        key=lambda s: (-autolabel.audits_needed(
                            project, s.label if s.verdict == ACCEPTED else NONE),
                            s.margin))
    n_audit = min(len(audit_pool), max(1, int(size * AUDIT_SHARE))) if audit_pool else 0

    n_confident = max(1, int(size * CONFIDENT_SHARE)) if confident else 0
    n_negative = min(len(negatives), max(1, int(size * NEGATIVE_SHARE))) if negatives else 0

    chosen, seen = [], set()

    def take(pool, limit):
        for s in pool:
            if len(chosen) >= limit:
                return
            if s.id not in seen:
                chosen.append(s)
                seen.add(s.id)

    take(disagree, size)
    main_limit = size - n_confident - n_negative - n_audit
    # Interleave the two kinds of boundary case so neither starves the other.
    merged = [x for pair in zip(unsure, narrow) for x in pair]
    merged += unsure[len(narrow):] + narrow[len(unsure):]
    take(merged, max(main_limit, len(chosen)))
    take(negatives, len(chosen) + n_negative)
    rng = random.Random(seed)
    pool = [s for s in confident[: max(n_confident * 5, 10)] if s.id not in seen]
    rng.shuffle(pool)
    take(pool, len(chosen) + n_confident)
    take(audit_pool, len(chosen) + n_audit)
    # Anything left over fills the batch, so a small project is not stuck at
    # half a sheet while samples wait.
    take(merged + negatives + confident + audit_pool, size)
    return chosen[:size]


# ---------------------------------------------------------------------------
# The sheet
# ---------------------------------------------------------------------------

def guess_of(sample) -> str:
    """What a tile asks to be confirmed: an auto-accepted sample's decision,
    or an undecided one's proposal."""
    if sample.is_auto:
        return sample.label if sample.verdict == ACCEPTED else NONE
    return sample.proposed


def caption(sample) -> str:
    if sample.is_auto:
        return f"auto: {guess_of(sample)} (spot check)"
    if sample.proposed == UNSURE:
        name, score = _top(sample)
        text = f"{name}? ({score:.2f})" if name else "unsure"
    elif sample.proposed == NONE:
        text = "none of these"
    else:
        text = f"{sample.proposed} ({sample.scores.get(sample.proposed, 0):.2f})"
    if sample.model_proposed and sample.model_proposed != sample.proposed:
        text += f" | model: {sample.model_proposed}"
    return text


def _tile(frames: list, height: int):
    from PIL import Image
    import cv2

    pictures = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        width = max(1, int(image.width * height / max(image.height, 1)))
        pictures.append(image.resize((width, height)))
    if not pictures:
        return Image.new("RGB", (height * 16 // 9, height), (40, 40, 40))
    total = sum(p.width for p in pictures) + 2 * (len(pictures) - 1)
    strip = Image.new("RGB", (total, height), (0, 0, 0))
    x = 0
    for picture in pictures:
        strip.paste(picture, (x, 0))
        x += picture.width + 2
    return strip


def _font(size: int):
    """A readable default font: Pillow's scalable one where it has it (10.1+)."""
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def render_sheet(tiles: Sequence, captions: Sequence[str], columns: int, path: str,
                 header: str = "") -> str:
    """Numbered tiles (1-based) in a grid, each captioned, saved as JPEG."""
    from PIL import Image, ImageDraw

    if not tiles:
        raise ValueError("nothing to draw")
    cell_w = max(t.width for t in tiles)
    cell_h = max(t.height for t in tiles) + CAPTION_HEIGHT
    rows = (len(tiles) + columns - 1) // columns
    top = CAPTION_HEIGHT if header else 0
    sheet = Image.new("RGB", (columns * (cell_w + 6) + 6, top + rows * (cell_h + 6) + 6),
                      (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    font = _font(15)
    if header:
        draw.text((8, 3), header, fill=(230, 230, 230), font=font)
    for i, (tile, text) in enumerate(zip(tiles, captions)):
        col, row = i % columns, i // columns
        x = 6 + col * (cell_w + 6)
        y = top + 6 + row * (cell_h + 6)
        sheet.paste(tile, (x, y))
        draw.rectangle([x, y + tile.height, x + cell_w, y + tile.height + CAPTION_HEIGHT],
                       fill=(0, 0, 0))
        draw.text((x + 4, y + tile.height + 3), f"{i + 1}  {text}", fill=(255, 255, 255),
                  font=font)
        draw.rectangle([x, y, x + 30, y + 20], fill=(255, 210, 0))
        draw.text((x + 4, y + 2), str(i + 1), fill=(0, 0, 0), font=font)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    sheet.save(path, quality=85)
    return path


def _next_sheet_number(review_dir: str) -> int:
    numbers = [int(m.group(1)) for name in (os.listdir(review_dir)
                                            if os.path.isdir(review_dir) else [])
               for m in [re.match(r"sheet-(\d+)\.json$", name)] if m]
    return max(numbers, default=0) + 1


def next_sheet(project: Project, size: int = DEFAULT_BATCH,
               class_name: Optional[str] = None,
               frame_reader: Optional[Callable] = None,
               renderer: Optional[Callable] = None,
               keep_tiles: bool = False) -> dict:
    """Pick a batch and draw it. Returns what the sheet holds, or ``{}``."""
    frame_reader = frame_reader or embed_mod.read_frames
    renderer = renderer or render_sheet
    batch = pick_batch(project, size, class_name)
    if not batch:
        return {}
    review_dir = project.path(REVIEW_DIR)
    os.makedirs(review_dir, exist_ok=True)
    number = _next_sheet_number(review_dir)
    frames_per_tile = 1 if project.task == OBJECTS else 3
    tiles, captions, items = [], [], []
    for n, sample in enumerate(batch, 1):
        tiles.append(_tile(frame_reader(sample.path, frames_per_tile), TILE_HEIGHT))
        captions.append(caption(sample))
        items.append({"n": n, "sample": sample.id, "proposed": guess_of(sample),
                      "caption": captions[-1]})
    image = os.path.join(review_dir, f"sheet-{number:04d}.jpg")
    columns = 2 if frames_per_tile > 1 else 4
    header = (f"sheet {number}: {project.name} ({project.task}) - classes: "
              + ", ".join(project.class_names()))
    renderer(tiles, captions, columns, image, header)
    record = {"sheet": number, "image": image, "created": time.time(),
              "items": items, "applied": False}
    with open(os.path.join(review_dir, f"sheet-{number:04d}.json"), "w",
              encoding="utf-8") as handle:
        json.dump(record, handle, indent=1)
    if keep_tiles:
        # The pictures themselves, for the review window; not saved.
        record["tiles"] = tiles
    return record


def load_sheet(project: Project, number: int) -> dict:
    path = project.path(REVIEW_DIR, f"sheet-{number:04d}.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def parse_numbers(text: str) -> list:
    """``"1-3, 7"`` -> ``[1, 2, 3, 7]``."""
    out = []
    for part in re.split(r"[,\s]+", (text or "").strip()):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def apply_verdicts(project: Project, number: int, *, accept: str = "",
                   reject: str = "", negative: str = "",
                   relabel: Sequence[str] = (), accept_rest: bool = False,
                   by: str = "sheet") -> dict:
    """Record what a sheet was judged to show.

    ``accept``: tiles whose guess is right. ``reject``: unusable (unclear, a
    bad cut, the wrong thing but not one of the classes either).
    ``negative``: shows none of the classes. ``relabel``: ``"8=<class>"``,
    the guess was wrong and this is what it is. ``accept_rest``: every tile
    not mentioned is taken as guessed — class guesses accepted, "none of
    these" guesses made negatives, unsure ones left for later.
    """
    sheet = load_sheet(project, number)
    by_n = {item["n"]: item for item in sheet["items"]}
    decided, errors = {}, []

    def mark(n, verdict, label=""):
        if n not in by_n:
            errors.append(f"sheet {number} has no tile {n}")
            return
        if n in decided:
            errors.append(f"tile {n} was given two verdicts")
            return
        decided[n] = (verdict, label)

    for n in parse_numbers(accept):
        # Accepting a "none of these" guess confirms it: that is a negative,
        # and refusing the whole sheet over it punished the natural reading
        # of "accept 1-24" = "every guess on this sheet is right".
        guessed_none = n in by_n and by_n[n]["proposed"] == NONE
        mark(n, NEGATIVE if guessed_none else ACCEPTED)
    for n in parse_numbers(reject):
        mark(n, REJECTED)
    for n in parse_numbers(negative):
        mark(n, NEGATIVE)
    for entry in relabel:
        left, _, name = str(entry).partition("=")
        name = name.strip()
        if not name:
            errors.append(f"relabel {entry!r}: say which class, like 8=<class>")
            continue
        for n in parse_numbers(left):
            mark(n, ACCEPTED, name)
    if errors:
        return {"applied": 0, "errors": errors}

    names = set(project.class_names())
    if accept_rest:
        for n, item in by_n.items():
            if n in decided:
                continue
            if item["proposed"] in names:
                decided[n] = (ACCEPTED, "")
            elif item["proposed"] == NONE:
                decided[n] = (NEGATIVE, "")

    for n, (verdict, label) in sorted(decided.items()):
        sample = project.get_sample(by_n[n]["sample"])
        if sample is None:
            errors.append(f"tile {n}: sample {by_n[n]['sample']} is gone")
            continue
        label = label or (by_n[n]["proposed"] if verdict == ACCEPTED else "")
        if verdict == ACCEPTED and label not in names:
            errors.append(f"tile {n} was a guess of {sample.proposed!r}; "
                          "say which class it is (relabel) instead of accepting")
            continue
        try:
            project.decide(sample, verdict, label, by=f"{by}:{number}")
        except ValueError as exc:
            errors.append(f"tile {n}: {exc}")
    if errors:
        return {"applied": 0, "errors": errors}

    from modules.teach import autolabel
    auto = autolabel.apply(project)
    sheet["applied"] = True
    with open(project.path(REVIEW_DIR, f"sheet-{number:04d}.json"), "w",
              encoding="utf-8") as handle:
        json.dump(sheet, handle, indent=1)
    project.save()
    tally = {}
    for verdict, label in decided.values():
        key = label or verdict
        tally[key] = tally.get(key, 0) + 1
    return {"applied": len(decided), "decisions": tally, "errors": [],
            "auto": {"accepted": auto["accepted"], "reverted": auto["reverted"]}}


def from_folders(project: Project, confirm: Sequence[str] = ()) -> dict:
    """Read verdicts back from ``sorted/`` after files were moved by hand.

    A file **moved** to another folder is a decision wherever it went: a class
    folder accepts it as that class, ``_none`` makes it a negative,
    ``_reject`` rejects it, ``_unsure`` sends it back to undecided. A file
    **left** where it was placed is only a decision for the class folders
    named in ``confirm`` (``["all"]`` for every class) — someone who sorted
    one folder has not vouched for the others.
    """
    from modules.teach.sort import PLACED_FILE, REJECT_DIR, SORTED_DIR

    root = project.path(SORTED_DIR)
    try:
        with open(os.path.join(root, PLACED_FILE), "r", encoding="utf-8") as handle:
            placed = json.load(handle)
    except (OSError, ValueError):
        return {"applied": 0, "errors": ["no sorted/ folders yet; run `folders` first"]}

    names = set(project.class_names())
    confirmed = names if "all" in confirm else set(confirm)
    unknown = [c for c in confirmed if c not in names]
    tally, errors = {}, [f"no class {c!r}" for c in unknown]

    for folder in sorted(os.listdir(root)):
        path = os.path.join(root, folder)
        if not os.path.isdir(path):
            continue
        for name in sorted(os.listdir(path)):
            sample = project.get_sample(os.path.splitext(name)[0])
            if sample is None:
                continue
            moved = placed.get(name) != folder
            if folder in names and (moved or folder in confirmed):
                verdict, label = ACCEPTED, folder
            elif folder == NONE and (moved or "_none" in confirm or "all" in confirm):
                verdict, label = NEGATIVE, ""
            elif folder == REJECT_DIR and moved:
                verdict, label = REJECTED, ""
            elif folder == UNSURE and moved:
                verdict, label = PENDING, ""
            else:
                continue
            if verdict == PENDING:
                sample.verdict, sample.label, sample.decided_by = PENDING, "", ""
            else:
                project.decide(sample, verdict, label, by="folders")
            key = label or verdict
            tally[key] = tally.get(key, 0) + 1
    project.save()
    from modules.teach import autolabel
    autolabel.apply(project)
    return {"applied": sum(tally.values()), "decisions": tally, "errors": errors}
