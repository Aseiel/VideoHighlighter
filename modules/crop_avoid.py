"""
crop_avoid.py — exclusion cropper ("avoid this person").

The inverse of crop_actions.py. Focus cropping asks "where is the action?" and
frames toward it. Avoid cropping asks "where is the person I must NOT show?" and
frames away from them, keeping as much of everyone else as a single rectangle can.

CONTRACT — deliberately knows nothing about faces or identities:
    The chooser takes plain rectangles. The avoided person arrives as
    `forbidden_boxes_by_frame[frame_idx] = [(x1,y1,x2,y2), ...]`. Whoever calls
    this (the pipeline) is responsible for turning bank.avoided_ids() + the
    identity-tagged cache into those rectangles. That keeps this module trivially
    unit-testable with hand-made boxes and immune to how recognition works.

GEOMETRY (a crop is ONE axis-aligned rectangle — no hole in the middle):
    The avoided person occupies a horizontal slab [fx1, fx2]. Removing it leaves
    up to two usable bands: left [0, fx1] and right [fx2, W]. We pick the band
    that keeps the most of everyone else.
      - avoided at an edge      -> one big band on the far side. Clean win.
      - avoided in the middle   -> two bands; we take the better one and sacrifice
                                   the smaller side (two-crop output is a future
                                   extension, see note at choose_avoid_window).
      - avoided glued to a keeper-> that keeper's box straddles the slab; it can't
                                   be kept without showing the avoided person. We
                                   don't fake it — that keeper is simply lost to
                                   the crop (cropping can't separate shared pixels;
                                   only inpainting could).
      - no usable band at all   -> status "impossible"; we hold the last good
                                   window rather than reveal them.

THE EXCLUSION GUARANTEE:
    Smoothing and any box math can drift a window sideways, and a naive median
    across a left<->right flip would land the window right on top of the avoided
    person. So the *final* step every frame is clamp_out_of_forbidden(): whatever
    the window is after smoothing, we shrink its offending edge until it no longer
    overlaps the forbidden slab. That single clamp is the source of truth for "they
    are not in frame," regardless of what smoothing did.

WHY NOT reuse expand_box / safe_crop from crop_core here:
    Both of those grow a box horizontally (margin expansion, min-width enforcement,
    aspect fixes). For focus that's harmless. For avoid it's dangerous — horizontal
    growth can re-cross the forbidden slab and put the avoided person back in shot.
    So avoid computes its final window explicitly and clamps it; the only crop_core
    pieces it reuses are the genuinely policy-neutral ones: pad_to_size (letterbox),
    calculate_iou (to tell the avoided YOLO box apart from keepers), and
    MultiSmoother (temporal stability, with the clamp as a safety net).
"""

import os
import cv2
import numpy as np

from crop_core import pad_to_size, calculate_iou, MultiSmoother


# ── config ──────────────────────────────────────────────────────────────────
PERSON_CONF = 0.12              # YOLO person confidence for "keepers"
FORBIDDEN_MATCH_IOU = 0.5       # a YOLO person this close to a forbidden box IS the avoided one
MIN_BAND_RATIO = 0.18           # a usable band must be at least this fraction of frame width
VERTICAL_PAD_RATIO = 0.08       # padding above/below kept people
SMOOTHING_WINDOW = 12
CALIBRATION_SAMPLES = 30
PADDING_COLOR = (0, 0, 0)
DEFAULT_TARGET = (854, 480)


# ── pure window chooser (unit-testable with synthetic boxes) ──────────────────
def choose_avoid_window(frame_w, frame_h, keep_boxes, forbidden_boxes,
                        min_band_ratio=MIN_BAND_RATIO,
                        vertical_pad_ratio=VERTICAL_PAD_RATIO):
    """
    Decide a crop rectangle that excludes the forbidden slab.

    Args:
        frame_w, frame_h : frame size
        keep_boxes       : [(x1,y1,x2,y2), ...] people we'd like to keep
        forbidden_boxes  : [(x1,y1,x2,y2), ...] the avoided person this frame
        min_band_ratio   : reject a side band narrower than this * frame_w

    Returns:
        (window, status, side)
        window : (x1,y1,x2,y2) or None
        status : "clear" | "excluded" | "impossible"
        side   : "left" | "right" | None     (used for smoothing continuity)
    """
    # No one to avoid this frame -> frame the keepers (or the whole frame).
    if not forbidden_boxes:
        if keep_boxes:
            x1 = min(b[0] for b in keep_boxes)
            y1 = min(b[1] for b in keep_boxes)
            x2 = max(b[2] for b in keep_boxes)
            y2 = max(b[3] for b in keep_boxes)
            return _clampbox((x1, y1, x2, y2), frame_w, frame_h), "clear", None
        return (0, 0, frame_w, frame_h), "clear", None

    # Forbidden horizontal slab = union of forbidden boxes' x-extents.
    fx1 = max(0, min(b[0] for b in forbidden_boxes))
    fx2 = min(frame_w, max(b[2] for b in forbidden_boxes))

    left_w = fx1 - 0
    right_w = frame_w - fx2
    min_band = min_band_ratio * frame_w

    # A keeper belongs to a band if its centre is on that side of the slab.
    def cx(b):
        return (b[0] + b[2]) / 2.0
    left_keep = [b for b in keep_boxes if cx(b) <= fx1]
    right_keep = [b for b in keep_boxes if cx(b) >= fx2]
    # (keepers whose centre is inside [fx1, fx2] straddle the avoided person and
    #  are unavoidably lost — cropping can't separate them.)

    def coverage(boxes):
        return sum((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)

    candidates = []
    if left_w >= min_band:
        candidates.append(("left", 0, fx1, coverage(left_keep), left_w, left_keep))
    if right_w >= min_band:
        candidates.append(("right", fx2, frame_w, coverage(right_keep), right_w, right_keep))

    if not candidates:
        return None, "impossible", None

    # Prefer the band that keeps the most people-area; tie-break on width.
    side, bx1, bx2, _score, _w, band_keep = max(candidates, key=lambda c: (c[3], c[4]))

    # Vertical extent: span of kept people in that band (padded), else full height.
    if band_keep:
        pad = int(vertical_pad_ratio * frame_h)
        by1 = max(0, min(b[1] for b in band_keep) - pad)
        by2 = min(frame_h, max(b[3] for b in band_keep) + pad)
    else:
        by1, by2 = 0, frame_h

    return (int(bx1), int(by1), int(bx2), int(by2)), "excluded", side
    # FUTURE: when both bands are usable AND both hold keepers, this is the
    # "avoided person in the middle" case where two output crops (left + right)
    # beat sacrificing a side. That needs the runner to manage N writers; left as
    # an extension so v1 stays single-output.


def clamp_out_of_forbidden(window, forbidden_boxes, frame_w):
    """
    The exclusion guarantee. Shrink `window` so it cannot overlap the forbidden
    horizontal slab, no matter what smoothing did to it. Returns None if the
    window collapses (nothing left to show on the chosen side).
    """
    if window is None or not forbidden_boxes:
        return window

    fx1 = max(0, min(b[0] for b in forbidden_boxes))
    fx2 = min(frame_w, max(b[2] for b in forbidden_boxes))

    x1, y1, x2, y2 = window
    overlaps = x1 < fx2 and x2 > fx1
    if overlaps:
        centre = (x1 + x2) / 2.0
        slab_centre = (fx1 + fx2) / 2.0
        if centre <= slab_centre:
            x2 = min(x2, fx1)     # we're the left band -> cap right edge
        else:
            x1 = max(x1, fx2)     # we're the right band -> push left edge

    if x2 - x1 < 2:
        return None
    return (int(x1), int(y1), int(x2), int(y2))


def _clampbox(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w))
    y1 = max(0, min(int(y1), h))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    return (x1, y1, x2, y2)


# ── per-frame keeper detection ────────────────────────────────────────────────
def _detect_keepers(rgb, yolo_model, forbidden_boxes, conf=PERSON_CONF):
    """YOLO persons this frame, minus the avoided person (matched by IoU)."""
    result = yolo_model.predict(rgb, conf=conf, classes=[0], verbose=False)
    persons = []
    for r in result:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            persons.append((x1, y1, x2, y2))

    if not forbidden_boxes:
        return persons

    keepers = []
    for p in persons:
        is_forbidden = any(calculate_iou(p, f) >= FORBIDDEN_MATCH_IOU for f in forbidden_boxes)
        if not is_forbidden:
            keepers.append(p)
    return keepers


# ── lightweight output-size calibration (avoid-specific; no focus tracker) ─────
def _calibrate_avoid(video_path, yolo_model, forbidden_boxes_by_frame, samples=CALIBRATION_SAMPLES):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0:
        cap.release()
        return DEFAULT_TARGET

    idxs = [int((i / samples) * total) for i in range(samples)]
    sizes = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        forb = forbidden_boxes_by_frame.get(fi, [])
        keep = _detect_keepers(rgb, yolo_model, forb)
        win, status, _ = choose_avoid_window(w, h, keep, forb)
        if win is not None:
            sizes.append((win[2] - win[0], win[3] - win[1]))
    cap.release()

    if not sizes:
        return DEFAULT_TARGET

    tw = int(np.percentile([s[0] for s in sizes], 80))
    th = int(np.percentile([s[1] for s in sizes], 80))
    # even dimensions, sane floor
    tw = max(320, tw - (tw % 2))
    th = max(320, th - (th % 2))
    return (tw, th)


# ── main runner ─────────────────────────────────────────────────────────────
def process_video_avoiding(input_path, output_folder, yolo_model,
                           forbidden_boxes_by_frame, output_name=None,
                           min_band_ratio=MIN_BAND_RATIO):
    """
    Render one crop that keeps the avoided person out of frame.

    Args:
        forbidden_boxes_by_frame : {frame_idx: [(x1,y1,x2,y2), ...]}; the avoided
                                   person's box(es) per frame. Missing/empty frame
                                   = nobody to avoid there.

    Returns:
        dict with the output path and an honest per-frame breakdown:
        {"output": path, "frames": n,
         "clear": k, "excluded": k, "impossible": k, "dropped": k}
    """
    os.makedirs(output_folder, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]
    out_path = os.path.join(output_folder, output_name or f"{base}_avoided.mp4")

    target = _calibrate_avoid(input_path, yolo_model, forbidden_boxes_by_frame)
    print(f"🚫 Avoid crop target size: {target[0]}x{target[1]}")

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, target)

    smoother = MultiSmoother(num_actions=1, window_size=SMOOTHING_WINDOW)
    last_window = None
    last_side = None
    stats = {"clear": 0, "excluded": 0, "impossible": 0, "dropped": 0}

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        forb = forbidden_boxes_by_frame.get(frame_idx, [])
        keep = _detect_keepers(rgb, yolo_model, forb)

        window, status, side = choose_avoid_window(W, H, keep, forb, min_band_ratio)

        if window is None:
            # impossible this frame: hold last good window, but re-clamp it against
            # the CURRENT forbidden slab so the avoided person still can't sneak in.
            stats["impossible"] += 1
            final = clamp_out_of_forbidden(last_window, forb, W) if last_window else None
            if final is None:
                writer.write(np.full((target[1], target[0], 3), PADDING_COLOR, dtype=np.uint8))
                stats["dropped"] += 1
                frame_idx += 1
                continue
        else:
            stats[status] += 1
            # smooth within a side; reset history on a left<->right flip so the
            # median never blends across the forbidden slab.
            if side != last_side:
                smoother.histories[0].clear()
                last_side = side
            sm = smoother.smooth(window)
            smoothed = sm[0] if sm else window
            # the guarantee: clamp out of the forbidden slab after smoothing.
            final = clamp_out_of_forbidden(smoothed, forb, W)
            if final is None:
                final = window  # smoothing+clamp collapsed -> trust the raw choice
            last_window = final

        x1, y1, x2, y2 = _clampbox(final, W, H)
        if x2 <= x1 or y2 <= y1:
            writer.write(np.full((target[1], target[0], 3), PADDING_COLOR, dtype=np.uint8))
            stats["dropped"] += 1
        else:
            crop = frame[y1:y2, x1:x2]
            writer.write(pad_to_size(crop, target, PADDING_COLOR))

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"  frame {frame_idx}/{total}")

    cap.release()
    writer.release()

    n = max(1, frame_idx)
    print(f"✅ Avoid crop done: {os.path.basename(out_path)}")
    print(f"   excluded {stats['excluded']} | clear {stats['clear']} | "
          f"impossible {stats['impossible']} | dropped {stats['dropped']} "
          f"({stats['excluded'] * 100 // n}% cleanly excluded)")

    return {"output": out_path, "frames": frame_idx, **stats}


# ── smoke test: verify the geometry with synthetic boxes, no video needed ─────
if __name__ == "__main__":
    W, H = 1000, 500

    def run(name, keep, forb):
        win, status, side = choose_avoid_window(W, H, keep, forb)
        clamped = clamp_out_of_forbidden(win, forb, W) if win else None
        print(f"{name:28s} -> status={status:10s} side={side} "
              f"window={win} clamped={clamped}")

    # avoided at the left edge: should crop the big right band
    run("avoided-left-edge", keep=[(700, 100, 850, 450)], forb=[(0, 100, 200, 450)])
    # avoided at the right edge: should crop the big left band
    run("avoided-right-edge", keep=[(150, 100, 300, 450)], forb=[(800, 100, 1000, 450)])
    # avoided in the middle, keepers both sides: picks the richer side
    run("avoided-middle", keep=[(80, 100, 230, 450), (770, 100, 920, 450)],
        forb=[(450, 100, 600, 450)])
    # avoided fills the frame: impossible -> None
    run("avoided-fills-frame", keep=[(400, 100, 600, 450)], forb=[(50, 50, 950, 480)])
    # nobody to avoid: frames the keepers
    run("no-forbidden", keep=[(300, 100, 700, 450)], forb=[])
    # exclusion guarantee: a window overlapping the slab gets clamped back out
    print("clamp check        ->",
          clamp_out_of_forbidden((0, 0, 700, 500), [(450, 0, 600, 500)], W))