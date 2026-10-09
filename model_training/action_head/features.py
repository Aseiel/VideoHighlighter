"""Clip -> frame encoder vectors, with a cache so a clip is encoded once.

**Frames evenly across the whole clip.** A clip's action can sit anywhere in
it; the old trainer took its frames from the first two seconds. Frame ``i`` of
``k`` is the one at ``(i + 0.5) / k`` of the clip, the sampling the accuracy
was measured with.

**Decoded in one forward pass.** Seeking (``CAP_PROP_POS_FRAMES``) was most
of the old trainer's time. Here every frame is grabbed in order and only the
chosen ones are converted, so a clip costs one sequential decode.

**The cache is keyed by the clip and the encoder.** A key is the clip's path
relative to the dataset, its size and its modification time, so an edited or
replaced clip is encoded again and a moved dataset is not. A cache from
another encoder or frame count is ignored rather than mixed in.
"""
from __future__ import annotations

import os
import time
from typing import Callable, Optional, Sequence

import numpy as np

CACHE_FORMAT = 1


def sample_indices(n_frames: int, k: int) -> list:
    """``k`` frame numbers spread evenly over ``n_frames`` (centres of k equal parts)."""
    n_frames = max(1, int(n_frames))
    return [min(n_frames - 1, int((i + 0.5) * n_frames / k)) for i in range(k)]


def read_frames(path: str, k: int) -> list:
    """``k`` BGR frames evenly across the clip, decoded front to back, or []
    when the file cannot be read. A frame count the container gets wrong
    costs a second pass with the true count, never a frame from the wrong
    place."""
    return read_frame_sets(path, (k,))


def read_frame_sets(path: str, counts: Sequence[int], convert=None) -> list:
    """For each ``k`` in ``counts``, ``k`` frames evenly across the clip, all
    in one front-to-back pass and concatenated in that order (``(4, 8)`` gives
    12 frames: 4 across the clip, then 8). ``convert`` is applied to each
    decoded frame once. [] when the file cannot be read."""
    import cv2

    def one_pass(targets):
        cap = cv2.VideoCapture(path)
        wanted, out, index = sorted(set(targets)), {}, 0
        try:
            while wanted and index <= wanted[-1]:
                if not cap.grab():
                    break
                if index in wanted:
                    ok, frame = cap.retrieve()
                    if ok:
                        out[index] = frame if convert is None else convert(frame)
                index += 1
        finally:
            cap.release()
        return out, index

    def targets_for(n):
        return [t for k in counts for t in sample_indices(n, k)]

    cap = cv2.VideoCapture(path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if count <= 0:
        count = 1_000_000                  # unknown: count it on the way
    targets = targets_for(count)
    got, decoded = one_pass(targets)
    if len(got) < len(set(targets)):
        if decoded <= 0:
            return []
        targets = targets_for(decoded)
        got, _ = one_pass(targets)
        if len(got) < len(set(targets)):
            return []
    return [got[t] for t in targets]


def clip_key(path: str, root: str) -> str:
    st = os.stat(path)
    rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root)).replace("\\", "/")
    return f"{rel}|{st.st_size}|{int(st.st_mtime)}"


class FeatureCache:
    """``key -> float16 [frames, dims]`` in one .npz, for one encoder and frame count."""

    def __init__(self, path: str, encoder_id: str, frames: int, dims: int):
        self.path, self.encoder_id, self.frames, self.dims = path, encoder_id, frames, dims
        self.items: dict = {}
        if os.path.isfile(path):
            try:
                z = np.load(path, allow_pickle=False)
                if (int(z["format"]) == CACHE_FORMAT and str(z["encoder"]) == encoder_id
                        and int(z["frames"]) == frames and z["features"].shape[2:] == (dims,)):
                    self.items = dict(zip(z["keys"].tolist(), z["features"]))
            except Exception as e:  # noqa: BLE001 - a bad cache is rebuilt, not fatal
                print(f"⚠️ Feature cache {path} unreadable ({e}); encoding again")

    def __contains__(self, key):
        return key in self.items

    def __getitem__(self, key):
        return self.items[key]

    def put(self, key: str, features: np.ndarray):
        self.items[key] = np.asarray(features, np.float16)

    def save(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        keys = sorted(self.items)
        feats = (np.stack([self.items[k] for k in keys]) if keys
                 else np.zeros((0, self.frames, self.dims), np.float16))
        tmp = self.path + ".tmp.npz"
        np.savez(tmp, format=CACHE_FORMAT, encoder=self.encoder_id, frames=self.frames,
                 keys=np.array(keys, dtype=str), features=feats)
        os.replace(tmp, self.path)


def encode_clips(paths: Sequence[str], root: str, encoder, cache: FeatureCache,
                 log: Callable[[str], None] = print, save_every: int = 200,
                 should_stop: Optional[Callable[[], bool]] = None) -> tuple:
    """Vectors for every clip: ``(features [N, frames, dims] float32, ok [N] bool)``.

    A clip that cannot be decoded is ``ok=False`` with zeros, and the caller
    leaves it out. The cache is saved as it goes, so a stopped run resumes.
    """
    feats = np.zeros((len(paths), cache.frames, cache.dims), np.float32)
    ok = np.zeros(len(paths), bool)
    todo = []
    for i, p in enumerate(paths):
        key = clip_key(p, root)
        if key in cache:
            feats[i], ok[i] = cache[key], True
        else:
            todo.append((i, p, key))
    if todo:
        log(f"Encoding {len(todo)} clips ({len(paths) - len(todo)} cached) "
            f"with {encoder.encoder_id} on {encoder.label}")
    start, failed = time.time(), []
    for n, (i, p, key) in enumerate(todo, 1):
        if should_stop and should_stop():
            break
        frames = read_frames(p, cache.frames)
        if not frames:
            failed.append(p)
            continue
        v = encoder.encode_bgr(frames)
        cache.put(key, v)
        feats[i], ok[i] = v, True
        if n % save_every == 0 or n == len(todo):
            cache.save()
            rate = n / max(time.time() - start, 1e-6)
            log(f"  {n}/{len(todo)} clips, {rate:.1f}/s, about "
                f"{(len(todo) - n) / rate / 60:.1f} min left")
    if todo:
        cache.save()
    for p in failed[:10]:
        log(f"⚠️ Could not read {os.path.basename(p)}; left out")
    if len(failed) > 10:
        log(f"⚠️ ... and {len(failed) - 10} more clips that could not be read")
    return feats, ok
