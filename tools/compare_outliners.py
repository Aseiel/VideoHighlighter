"""Draw two outliners' outlines side by side on real frames, to choose between them.

    python tools/compare_outliners.py <video> --class person --frames 16
    python tools/compare_outliners.py <video> --class person --a grabcut --b sam

Takes detections of ``--class`` from the video's analysis cache (run object
detection first), picks frames spread across the video, and writes one image:
each tile is a frame with outliner A's outline in yellow and B's in cyan over
the detector's box (grey). The question it answers is the one that decides
whether SAM is worth shipping (docs/plans/2026-09-26-composition-outlines.md):
on *this* footage, which outline sits on the thing? Also prints each
outliner's time per object, and how often the two agree (IoU of their shapes).

Dev tool; nothing in the app imports it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402


def pick(bboxes: list, class_name: str, count: int) -> list:
    """``(timestamp, box)`` for up to ``count`` detections, spread over the video."""
    found = []
    for entry in bboxes or []:
        names = entry.get("objects") or []
        boxes = entry.get("bboxes") or []
        for i, name in enumerate(names):
            if name == class_name and i < len(boxes):
                found.append((float(entry.get("timestamp", 0)), list(boxes[i])))
                break
    if len(found) <= count:
        return found
    step = len(found) / count
    return [found[int(k * step)] for k in range(count)]


def shape_iou(a, b, size=(256, 256)) -> float:
    """IoU of two normalised outlines, drawn on a grid."""
    import cv2

    if not a or not b:
        return 0.0
    w, h = size
    ma, mb = np.zeros((h, w), np.uint8), np.zeros((h, w), np.uint8)
    cv2.fillPoly(ma, [np.array([[x * w, y * h] for x, y in a], np.int32)], 1)
    cv2.fillPoly(mb, [np.array([[x * w, y * h] for x, y in b], np.int32)], 1)
    union = (ma | mb).sum()
    return float((ma & mb).sum() / union) if union else 0.0


def draw(frame, box, outline_a, outline_b):
    import cv2

    out = frame.copy()
    h, w = out.shape[:2]
    x, y, bw, bh = box
    thick = max(2, w // 400)
    cv2.rectangle(out, (int(x * w), int(y * h)), (int((x + bw) * w), int((y + bh) * h)),
                  (160, 160, 160), thick)
    for outline, colour in ((outline_a, (0, 220, 255)), (outline_b, (255, 220, 0))):
        if outline:
            pts = np.array([[px * w, py * h] for px, py in outline], np.int32)
            cv2.polylines(out, [pts], True, colour, thick + 1)
    return out


def compare(video: str, picks: list, outliner_a, outliner_b, out_path: str,
            frame_reader=None, renderer=None) -> dict:
    from modules.teach.review import _tile, render_sheet
    from modules.vision.outlines import read_frames_at

    frame_reader = frame_reader or read_frames_at
    renderer = renderer or render_sheet
    tiles, captions, ious = [], [], []
    spent = {"a": 0.0, "b": 0.0}
    stamps = [ts for ts, _ in picks]
    for (ts, box), (_, frame) in zip(picks, frame_reader(video, stamps)):
        if frame is None:
            continue
        h, w = frame.shape[:2]
        px = [(box[0] * w, box[1] * h, (box[0] + box[2]) * w, (box[1] + box[3]) * h)]
        t0 = time.perf_counter()
        (a,) = outliner_a.outline(frame, px)
        t1 = time.perf_counter()
        (b,) = outliner_b.outline(frame, px)
        t2 = time.perf_counter()
        spent["a"] += t1 - t0
        spent["b"] += t2 - t1
        iou = shape_iou(a, b)
        ious.append(iou)
        tiles.append(_tile([draw(frame, box, a, b)], 240))
        captions.append(f"{ts:.1f}s  agree {iou:.2f}"
                        + ("  A: none" if not a else "") + ("  B: none" if not b else ""))
    if not tiles:
        raise SystemExit("no frames could be read for those detections")
    name_a = getattr(outliner_a, "name", "A")
    name_b = getattr(outliner_b, "name", "B")
    if name_a == name_b:              # one outliner against itself: keep both timings
        name_a, name_b = f"{name_a}_a", f"{name_b}_b"
    renderer(tiles, captions, 4, out_path,
             f"yellow = {name_a}   cyan = {name_b}   grey = detector box")
    n = len(tiles)
    return {"image": out_path, "frames": n,
            f"{name_a}_ms_per_object": round(1000 * spent["a"] / n, 1),
            f"{name_b}_ms_per_object": round(1000 * spent["b"] / n, 1),
            "mean_agreement": round(float(np.mean(ious)), 3)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("video")
    parser.add_argument("--class", dest="cls", required=True)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--a", default="grabcut", choices=("grabcut", "sam"))
    parser.add_argument("--b", default="sam", choices=("grabcut", "sam"))
    parser.add_argument("--out", default="outliner-comparison.jpg")
    args = parser.parse_args(argv)

    from modules.report.analysis_ondemand import read_cache
    from modules.vision.outlines import make_outliner

    bboxes = (read_cache(args.video) or {}).get("object_bboxes") or []
    picks = pick(bboxes, args.cls, args.frames)
    if not picks:
        print(f"No {args.cls!r} detections cached for this video; run object detection first.")
        return 1
    result = compare(args.video, picks, make_outliner(args.a), make_outliner(args.b),
                     args.out)
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
