"""Scan a video's faces once, then reuse the answer.

Detection and classification are cheap per frame and expensive per video, and
the result is small — a label and a confidence per second. So this runs the
sweep once, writes it beside the other caches, and everything downstream reads
from that: the highlight pipeline scoring a signal, a timeline row to navigate,
or the command line asking "where is this expression" without touching the
highlighter at all.

Running it standalone is a first-class use, not a debug aid. Wanting only the
moments matching one expression — and a highlight built from nothing else — is
a different question from "find the interesting parts", and it should not
require configuring a weight table to ask.

Decoding, detection and classification all arrive as callables. The sweep is
then testable with arrays, and a caller with frames already in hand (the
pipeline, which decodes for motion anyway) can hand them over instead of paying
for a second pass.

Standalone usage:

    python -m modules.face_scan --video "D:\\clips\\a.mp4" --interval 1
    python -m modules.face_scan --video "D:\\clips\\a.mp4" --label happy --top 20
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from modules.face_emotions import EMOTION_LABELS, emotions_by_second
from modules.face_examples import DEFAULT_PAD, MIN_DETECTION_SCORE, crop_face

SCHEMA = 1

# One sample a second. Expressions last longer than that, and a finer interval
# multiplies the sweep's cost for detail nothing downstream can use — the score
# array is per-second.
DEFAULT_INTERVAL = 1.0


def cache_path_for(video_path: str, cache_dir: str = "./cache") -> str:
    """Where this video's face scan lives."""
    stem = os.path.splitext(os.path.basename(str(video_path)))[0]
    return os.path.join(cache_dir, f"{stem}_faces.json")


def iter_video_frames(video_path: str,
                      interval: float = DEFAULT_INTERVAL,
                      *,
                      cancel_fn: Optional[Callable] = None,
                      progress_fn: Optional[Callable] = None) -> Iterable:
    """Yield ``(timestamp, frame_bgr)`` every ``interval`` seconds.

    Seeks rather than decoding every frame: at one sample a second, decoding all
    of them would spend the entire budget on frames that are thrown away.
    """
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise IOError(f"could not open {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        duration = frames / fps if fps else 0.0
        second = 0.0
        while duration <= 0 or second < duration:
            if cancel_fn is not None and cancel_fn():
                return
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(second * fps))
            ok, frame = capture.read()
            if not ok:
                return
            yield float(second), frame
            if progress_fn is not None and duration > 0:
                progress_fn(second, duration)
            second += interval
    finally:
        capture.release()


def scan(frames: Iterable,
         *,
         detect_fn: Callable,
         classify_fn: Callable,
         pad: float = DEFAULT_PAD,
         min_det_score: float = MIN_DETECTION_SCORE,
         max_faces_per_frame: int = 4,
         min_confidence: float = 0.5) -> dict:
    """Faces per second, reduced to the clearest expression at each.

    ``classify_fn(crops)`` returns one row of class probabilities per crop.
    Frames with no readable face simply do not appear — an absent second is
    "no face", which is different from "a face showing nothing".
    """
    seconds: dict = {}
    for timestamp, frame in frames:
        faces = detect_fn(frame) or []
        faces = sorted(faces, key=lambda f: -float(f.get("det_score") or 0.0))
        crops, kept = [], []
        for face in faces[:max_faces_per_frame]:
            if float(face.get("det_score") or 0.0) < min_det_score:
                continue
            crop = crop_face(frame, face.get("bbox") or (0, 0, 0, 0), pad)
            if crop is None:
                continue
            crops.append(crop)
            kept.append(face)
        if not crops:
            continue

        probabilities = classify_fn(crops)
        if probabilities is None or not len(probabilities):
            continue

        # emotions_by_second wants objects carrying a timestamp; every crop in
        # this frame shares one.
        holders = [type("_C", (), {"timestamp": float(timestamp)})()
                   for _ in crops]
        best = emotions_by_second(holders, probabilities,
                                  min_confidence=min_confidence)
        if not best:
            continue
        label, confidence = next(iter(best.values()))
        seconds[int(timestamp)] = {
            "label": label,
            "confidence": round(float(confidence), 4),
            "faces": len(crops),
        }
    return seconds


def best_by_second(seconds: Mapping) -> dict:
    """``{second: (label, confidence)}`` — the shape the signal builder wants."""
    return {int(sec): (str(v.get("label") or ""), float(v.get("confidence") or 0.0))
            for sec, v in (seconds or {}).items()}


def moments_for(seconds: Mapping,
                label: str,
                *,
                min_confidence: float = 0.0) -> list:
    """Every second matching one expression, strongest first.

    This is the standalone question — "where is this expression" — answered
    without a weight table, a highlight, or a run of the pipeline.
    """
    wanted = str(label).lower()
    hits = [
        (int(sec), float(v.get("confidence") or 0.0))
        for sec, v in (seconds or {}).items()
        if str(v.get("label") or "").lower() == wanted
        and float(v.get("confidence") or 0.0) >= min_confidence
    ]
    hits.sort(key=lambda pair: -pair[1])
    return hits


def label_counts(seconds: Mapping) -> dict:
    """How many seconds each expression accounted for."""
    counts = {label: 0 for label in EMOTION_LABELS}
    for value in (seconds or {}).values():
        label = str(value.get("label") or "")
        if label:
            counts[label] = counts.get(label, 0) + 1
    return counts


def save(seconds: Mapping, path: str, *, video_path: str = "",
         interval: float = DEFAULT_INTERVAL) -> bool:
    """Write the scan, replacing atomically so a crash cannot truncate it."""
    try:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        payload = {
            "schema": SCHEMA,
            "video": str(video_path),
            "interval": float(interval),
            "scanned_at": time.time(),
            "seconds": {str(k): v for k, v in (seconds or {}).items()},
        }
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
        os.replace(temporary, path)
        return True
    except OSError as exc:
        print(f"⚠️ Could not save face scan: {exc}")
        return False


def load(path: str) -> Optional[dict]:
    """The scan's per-second entries, or ``None`` if there isn't one to read."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError) as exc:
        print(f"⚠️ Could not read face scan: {exc}")
        return None
    return {int(k): v for k, v in (payload.get("seconds") or {}).items()}


def scan_video(video_path: str,
               *,
               interval: float = DEFAULT_INTERVAL,
               cache_dir: str = "./cache",
               use_cache: bool = True,
               detector=None,
               classifier=None,
               cancel_fn: Optional[Callable] = None,
               progress_fn: Optional[Callable] = None,
               log_fn: Callable = print) -> dict:
    """Scan a file end to end, reusing a previous scan when there is one.

    Returns ``{}`` — after saying why — when the expression model is not
    installed, rather than pretending every face was neutral.
    """
    path = cache_path_for(video_path, cache_dir)
    if use_cache:
        cached = load(path)
        if cached is not None:
            log_fn(f"ℹ Using cached face scan ({len(cached)} second(s))")
            return cached

    from video_ai_editor.face_identity import FaceIdentityBank
    from modules.face_emotions import EmotionClassifier

    detector = detector or FaceIdentityBank(
        db_path=os.path.join(cache_dir, "face_db.json"))
    classifier = classifier or EmotionClassifier()
    if not classifier.load():
        return {}

    started = time.time()
    frames = iter_video_frames(video_path, interval, cancel_fn=cancel_fn,
                               progress_fn=progress_fn)
    seconds = scan(frames, detect_fn=detector.detect_faces,
                   classify_fn=classifier.classify)
    log_fn(f"✅ Face scan: {len(seconds)} second(s) with a readable expression "
           f"in {time.time() - started:.1f}s")
    save(seconds, path, video_path=video_path, interval=interval)
    return seconds


def _main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m modules.face_scan",
        description="Find expressions in a video, with or without a highlight.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument("--cache-dir", default="./cache")
    parser.add_argument("--rescan", action="store_true",
                        help="ignore any cached scan")
    parser.add_argument("--label", choices=EMOTION_LABELS,
                        help="list the moments matching this expression")
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args(argv)

    seconds = scan_video(args.video, interval=args.interval,
                         cache_dir=args.cache_dir, use_cache=not args.rescan)
    if not seconds:
        return 1

    def stamp(sec):
        return f"{int(sec) // 60}:{int(sec) % 60:02d}"

    if args.label:
        hits = moments_for(seconds, args.label,
                           min_confidence=args.min_confidence)
        print(f"\n{len(hits)} second(s) of '{args.label}':")
        for sec, confidence in hits[:args.top]:
            print(f"  {stamp(sec):>8}   {confidence:.2f}")
    else:
        print("\nseconds by expression:")
        for label, count in sorted(label_counts(seconds).items(),
                                   key=lambda kv: -kv[1]):
            print(f"  {label:<9} {count:>5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
