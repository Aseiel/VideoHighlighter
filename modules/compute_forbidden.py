"""
compute_forbidden.py — turn "avoid these identities" into the two feeds the
pipeline needs.

Wraps identity_tagging.tag_video_with_identities (the offline track + face pass)
and filters its per-frame, identity-tagged person boxes down to just the avoided
identities. Knows nothing about cropping or scoring — it only answers
"where and when is the avoided person?" in two shapes:

    forbidden_ranges          : [(start_sec, end_sec), ...]        -> skip method
    forbidden_boxes_by_frame  : {frame_idx: [(x1,y1,x2,y2), ...]}  -> crop method (pixels)

Notes that come straight from identity_tagging's output contract:
  - Its bboxes are NORMALISED [x, y, w, h] (0..1), so we denormalise to pixels.
  - It only emits entries for frames that had person boxes; identity is already
    propagated to every frame of a track, so a present person is dense, not just
    on the sampled face frames.
  - Tracking needs YOLO .track(), which wants a .pt model. An OpenVINO detect
    model won't track — so we load our own yolo_model_path here instead of
    reusing the pipeline's detection model.

Import-path assumption: this file sits next to pipeline.py (root), and the
identity modules live in the video_ai_editor package. Adjust the import below if
your layout differs.
"""

from __future__ import annotations
import cv2


def _merge_seconds(seconds, merge_gap=2.0):
    """Set of int seconds -> merged (start, end) ranges, bridging gaps <= merge_gap."""
    if not seconds:
        return []
    s = sorted(seconds)
    ranges = []
    start = prev = s[0]
    for cur in s[1:]:
        if cur - prev <= merge_gap:
            prev = cur
        else:
            ranges.append((float(start), float(prev + 1)))   # +1: cover the whole second
            start = prev = cur
    ranges.append((float(start), float(prev + 1)))
    return ranges


def compute_forbidden(video_path, bank, avoid_ids, fps,
                      yolo_model_path="yolo11s.pt",
                      face_every=10, vid_stride=1, merge_gap=2.0,
                      log_fn=print, cancel_flag=None):
    """
    Returns (forbidden_ranges, forbidden_boxes_by_frame).

    forbidden_boxes_by_frame is keyed by ORIGINAL frame index of `video_path`.
    The crop step remaps that to clip-local indices per highlight clip.
    """
    avoid = set(avoid_ids or [])
    if not avoid:
        return [], {}

    from video_ai_editor.identity_tagging import tag_video_with_identities

    # frame size, to denormalise [x,y,w,h] (0..1) back to pixels
    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if W == 0 or H == 0:
        log_fn("⚠️ compute_forbidden: could not read frame size — skipping exclusion")
        return [], {}

    def _progress(i, msg):
        if cancel_flag is not None and getattr(cancel_flag, "is_set", lambda: False)():
            raise RuntimeError("cancelled during identity tagging")
        if i % 150 == 0:
            log_fn(f"🚫 Avoid: {msg}")

    entries = tag_video_with_identities(
        video_path, bank,
        yolo_model_path=yolo_model_path,
        face_every=face_every,
        vid_stride=vid_stride,
        save_bank=False,                 # don't persist incidental enrollments from analysis
        progress_cb=_progress,
    )

    forbidden_seconds = set()
    forbidden_boxes_by_frame = {}

    for e in entries:
        ts = e["timestamp"]
        frame_idx = int(round(ts * fps))
        boxes_here = []
        for bbox, iid in zip(e["bboxes"], e["identity_ids"]):
            if iid in avoid:
                x, y, bw, bh = bbox                       # normalised x,y,w,h
                x1, y1 = int(x * W), int(y * H)
                x2, y2 = int((x + bw) * W), int((y + bh) * H)
                if x2 > x1 and y2 > y1:
                    boxes_here.append((x1, y1, x2, y2))
        if boxes_here:
            forbidden_seconds.add(int(ts))
            forbidden_boxes_by_frame.setdefault(frame_idx, []).extend(boxes_here)

    forbidden_ranges = _merge_seconds(forbidden_seconds, merge_gap=merge_gap)
    log_fn(f"🚫 Avoid: avoided identity present in {len(forbidden_boxes_by_frame)} frame(s), "
           f"{len(forbidden_ranges)} merged range(s)")
    return forbidden_ranges, forbidden_boxes_by_frame


# ── manual run: python compute_forbidden.py <video> <face_db.json> <avoid_id> [...] ──
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python compute_forbidden.py <video> <face_db.json> <avoid_id> [avoid_id...]")
        sys.exit(1)

    from video_ai_editor.face_identity import FaceIdentityBank

    video, db = sys.argv[1], sys.argv[2]
    avoid_ids = sys.argv[3:]

    bank = FaceIdentityBank(db_path=db)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    ranges, boxes = compute_forbidden(video, bank, avoid_ids, fps)
    print(f"\nforbidden_ranges ({len(ranges)}):")
    for a, b in ranges[:20]:
        print(f"  {a:.1f}s – {b:.1f}s")
    print(f"forbidden_boxes_by_frame: {len(boxes)} frame(s)")