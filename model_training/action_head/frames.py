"""Every clip decoded once into a frame cache, for fine-tuning the image tower.

A fine-tune reads each clip's frames ten times or more (every epoch of every
fold), so decoding them each time would cost more than the training. They are
decoded once into one uint8 memmap, ``[clips, 12, 256, 256, 3]``, 2.4 MB per
clip:

- **Slots 0-3** are at ``(i + 0.5) / 4`` of the clip: the frames the frozen
  trainer encodes and the app scores, so held-out scoring sees what the app
  will see.
- **Slots 4-11** are at ``(i + 0.5) / 8``: more views of the clip for
  training, which takes 4 of the 12 at random.

**Exactly the app's pixels.** A frame is stored as
``frame_encoder.prepare_frame`` makes it, the first half of the app's own
preprocessing; :func:`pixels` is the second half. So what the tower is
trained on is what it will be fed.

**Keyed like the feature cache** (``features.clip_key``: path in the dataset,
size, modification time), so an edited clip is decoded again. A cache that
holds every clip a run needs is used as it is; otherwise it is rebuilt,
copying the rows it already has. It lives in the user's data folder and is
kept after a run, so a second run starts in seconds. Nothing here is ever
copied into a model folder: a model carries the model, not the material.

Decoding runs on threads, not processes: OpenCV decodes outside the GIL, and
a process pool in the packaged app would start copies of the app.
"""
from __future__ import annotations

import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Sequence

import numpy as np

FORMAT = 1
SLOTS = 12
SCORING = 4                         # slots 0-3: the frames the app scores
SLOT_SETS = (SCORING, SLOTS - SCORING)
FRAMES_FILE = "frames.npy"
INDEX_FILE = "index.json"
SPARE_BYTES = 1 << 30               # free disk left over after the cache is written

# Where each slot sits in its clip, as a share of the clip's length.
SLOT_POSITIONS = np.array([(i + 0.5) / k for k in SLOT_SETS for i in range(k)])


def size_px() -> int:
    from modules.vision import frame_encoder
    return frame_encoder.INPUT_SIZE


def bytes_per_clip() -> int:
    return SLOTS * size_px() * size_px() * 3


def pixels(rgb_u8: np.ndarray) -> np.ndarray:
    """Cached frames -> the encoder's input; ``frame_encoder.scale_pixels``
    over any leading dimensions: uint8 [..., 256, 256, 3] -> float32
    [..., 3, 256, 256]."""
    from modules.vision import frame_encoder
    rgb_u8 = np.asarray(rgb_u8)
    lead = rgb_u8.shape[:-3]
    flat = frame_encoder.scale_pixels(rgb_u8.reshape((-1,) + rgb_u8.shape[-3:]))
    return flat.reshape(lead + flat.shape[1:])


def decode_clip(path: str) -> Optional[np.ndarray]:
    """One clip -> uint8 [12, 256, 256, 3] RGB in slot order, or None when it
    cannot be read."""
    from model_training.action_head import features
    from modules.vision import frame_encoder
    try:
        frames = features.read_frame_sets(path, SLOT_SETS, convert=frame_encoder.prepare_frame)
    except Exception as e:  # noqa: BLE001 - one bad clip is left out, not fatal
        print(f"⚠️ {os.path.basename(path)}: {e}")
        return None
    return np.stack(frames) if len(frames) == SLOTS else None


class FrameCache:
    """The cached frames of a run's clips: ``rows(keys)`` says where each one
    is, ``read(rows)`` returns them."""

    def __init__(self, folder: str, keys: Sequence[str], ok: Sequence[bool], frames):
        self.folder = folder
        self.keys = list(keys)
        self.ok = np.asarray(ok, bool)
        self.frames = frames                    # memmap [N, 12, S, S, 3]
        self._row = {k: i for i, k in enumerate(self.keys)}

    def __len__(self):
        return len(self.keys)

    def row(self, key: str) -> int:
        return self._row[key]

    def read(self, rows, slots=None) -> np.ndarray:
        """Frames of ``rows`` in that order (read from disk in file order,
        which is far faster on a memmap), all slots or only ``slots``."""
        rows = np.asarray(rows)
        order = np.argsort(rows, kind="stable")
        out = np.asarray(self.frames[rows[order]])
        out = out[np.argsort(order, kind="stable")]
        return out if slots is None else out[:, slots]

    @property
    def nbytes(self) -> int:
        return len(self.keys) * bytes_per_clip()

    def close(self):
        mm = getattr(self.frames, "_mmap", None)
        self.frames = None
        if mm is not None:
            try:
                mm.close()
            except Exception:  # noqa: BLE001
                pass


def _open(folder: str) -> Optional[FrameCache]:
    try:
        with open(os.path.join(folder, INDEX_FILE), encoding="utf-8") as fh:
            index = json.load(fh)
        if (index.get("format") != FORMAT or index.get("slots") != SLOTS
                or index.get("size") != size_px()):
            return None
        frames = np.load(os.path.join(folder, FRAMES_FILE), mmap_mode="r")
        if frames.shape != (len(index["keys"]), SLOTS, size_px(), size_px(), 3):
            return None
        return FrameCache(folder, index["keys"], index["ok"], frames)
    except Exception:  # noqa: BLE001 - missing or unreadable: built again
        return None


def _free_bytes(folder: str) -> int:
    probe = folder
    while probe and not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return shutil.disk_usage(probe or ".").free


def default_workers() -> int:
    return max(1, min(6, (os.cpu_count() or 2) - 1))


def build(folder: str, paths: Sequence[str], keys: Sequence[str],
          log: Callable[[str], None] = print, check: Callable[[], None] = lambda: None,
          workers: Optional[int] = None) -> Optional[FrameCache]:
    """The cache for these clips (``keys[i]`` is ``paths[i]``'s
    ``features.clip_key``), decoding only the clips it does not hold yet.
    Returns None, with a sentence in ``log``, when the disk is too full.
    ``check`` is called between clips and may raise to stop; a stopped build
    leaves the old cache as it was."""
    keys = list(keys)
    old = _open(folder)
    if old is not None and all(k in old._row for k in keys):
        log(f"Frame cache: all {len(keys)} clips already decoded ({folder})")
        return old

    need = len(keys) * bytes_per_clip()
    free = _free_bytes(folder)
    if free < need + SPARE_BYTES:
        log(f"❌ The frame cache needs {need / 1e9:.1f} GB plus 1 GB to spare, and the disk "
            f"holding {folder} has {free / 1e9:.1f} GB free. Free some space, or point "
            f"--frame-cache at another disk.")
        if old is not None:
            old.close()
        return None

    os.makedirs(folder, exist_ok=True)
    tmp = os.path.join(folder, FRAMES_FILE + ".building.npy")
    s = size_px()
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                   shape=(len(keys), SLOTS, s, s, 3))
    ok = [False] * len(keys)
    todo = []
    for i, key in enumerate(keys):
        j = old._row.get(key) if old is not None else None
        if j is not None and old.ok[j]:
            mm[i] = old.frames[j]
            ok[i] = True
        else:
            todo.append(i)
    if old is not None:
        old.close()
    log(f"Decoding {len(todo)} clips into the frame cache ({len(keys) - len(todo)} already "
        f"there; {need / 1e9:.1f} GB in {folder})")
    started, failed = time.time(), []
    workers = workers or default_workers()
    try:
        with ThreadPoolExecutor(workers) as pool:
            done = 0
            # A few clips per worker at a time, so stopping waits for those only.
            chunk = workers * 4
            for c in range(0, len(todo), chunk):
                check()
                part = todo[c:c + chunk]
                for i, frames in zip(part, pool.map(decode_clip, [paths[i] for i in part])):
                    done += 1
                    if frames is None:
                        failed.append(paths[i])
                    else:
                        mm[i] = frames
                        ok[i] = True
                    if done % 100 == 0 or done == len(todo):
                        rate = done / max(time.time() - started, 1e-6)
                        log(f"  {done}/{len(todo)} clips decoded, {rate:.1f}/s, about "
                            f"{(len(todo) - done) / rate / 60:.1f} min left")
        mm.flush()
    except BaseException:
        del mm
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    del mm
    for p in failed[:10]:
        log(f"⚠️ Could not read {os.path.basename(p)}; left out")
    if len(failed) > 10:
        log(f"⚠️ ... and {len(failed) - 10} more clips that could not be read")
    final = os.path.join(folder, FRAMES_FILE)
    index = os.path.join(folder, INDEX_FILE)
    if os.path.exists(index):
        os.remove(index)                 # never an index that names the wrong rows
    os.replace(tmp, final)
    with open(index + ".tmp", "w", encoding="utf-8") as fh:
        json.dump({"format": FORMAT, "slots": SLOTS, "size": s, "keys": keys, "ok": ok}, fh)
    os.replace(index + ".tmp", index)
    cache = _open(folder)
    log(f"Frame cache: {sum(ok)} clips, {need / 1e9:.1f} GB, kept in {folder} for the next run")
    return cache
