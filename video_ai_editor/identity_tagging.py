"""
Identity Tagging Pass — attach a persistent person identity to every box.

This is the OFFLINE pass (runs during analysis, writes to cache). It does NOT
run during playback. The live overlay just reads the identity_id this produces.

Pipeline:
    1. YOLO tracks people across the video  -> each person box gets a track_id
       (cheap, frame-to-frame continuity via ByteTrack).
    2. On sampled frames, FaceIdentityBank recognises faces and matches each
       face to the person box it sits inside -> a vote for that track's identity.
    3. Each track's identity is decided by majority vote of its face matches,
       then propagated to EVERY frame of that track — including the frames where
       no face was visible (back turned, profile). That's the whole point:
       the face is the anchor, the track_id carries it.

Output is a list of per-timestamp entries in the same shape your
realtime_overlay.OverlayScene already reads (objects / bboxes / confidences),
with three extra parallel arrays added: track_ids, identity_ids, identity_names.

Usage:
    from video_ai_editor.face_identity import FaceIdentityBank
    from video_ai_editor.identity_tagging import tag_video_with_identities

    bank = FaceIdentityBank(db_path="./cache/face_db.json",
                            providers=["OpenVINOExecutionProvider",
                                       "CPUExecutionProvider"])
    object_bboxes = tag_video_with_identities("clip.mp4", bank)
    bank.save()

    # then merge into your cache and persist (your existing cache writer):
    cache_data["object_bboxes"] = object_bboxes

Performance:
    - YOLO tracking runs every (strided) frame — that's what keeps track_ids
      stable. Use `vid_stride` to skip frames if you need it faster.
    - Face recognition runs only every `face_every` frames. It does NOT need to
      run often: a handful of good face reads per track is enough, the vote
      settles, and track_id does the rest.
"""

from __future__ import annotations

import os
from collections import defaultdict, Counter
from typing import Optional

import numpy as np


def tag_video_with_identities(
    video_path: str,
    bank,                                   # FaceIdentityBank
    yolo_model_path: str = "yolo11s.pt",
    person_conf: float = 0.25,
    face_every: int = 10,                   # run face recognition every N processed frames
    vid_stride: int = 1,                    # skip frames in tracking (1 = every frame)
    tracker: str = "bytetrack.yaml",
    min_votes: int = 1,                     # min face matches before trusting an identity
    save_bank: bool = False,
    model=None,                             # pre-built YOLO model (reused if provided)
    device=None,                            # tracking device: 'xpu' | 0 (cuda) | 'cpu'
    progress_cb=None,                       # optional: progress_cb(frame_idx, message)
    
) -> list[dict]:
    """
    Run tracking + face identity over a video and return identity-tagged
    per-timestamp box entries (object_bboxes shape).

    Each returned entry:
        {
            "timestamp": float,
            "objects":        ["person", ...],
            "bboxes":         [[x, y, w, h], ...],   # normalised 0..1
            "confidences":    [float, ...],
            "track_ids":      [int|None, ...],
            "identity_ids":   [str|None, ...],
            "identity_names": [str|None, ...],
        }
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("ultralytics is required: pip install ultralytics") from e

    import cv2

    # fps for timestamps
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    if model is None:
        model = YOLO(yolo_model_path)

    # PASS 1 — track people, collect boxes, vote on identities per track
    # ------------------------------------------------------------------
    # Per-frame raw records (identity filled in pass 2)
    frame_records: list[dict] = []          # {timestamp, boxes:[{...}]}
    track_votes: dict[int, Counter] = defaultdict(Counter)

    results = model.track(
        source=video_path,
        persist=True,
        classes=[0],                        # person only
        conf=person_conf,
        stream=True,
        vid_stride=vid_stride,
        tracker=tracker,
        device=device,
        verbose=False,
    )

    processed = 0
    for r in results:
        real_frame = processed * vid_stride
        timestamp = real_frame / fps if fps else float(real_frame)

        frame_bgr = r.orig_img                 # BGR numpy
        h, w = frame_bgr.shape[:2]

        # gather this frame's person boxes (pixel coords + track id)
        boxes_px = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = (r.boxes.conf.cpu().numpy()
                     if r.boxes.conf is not None else np.ones(len(xyxy)))
            ids = (r.boxes.id.cpu().numpy().astype(int)
                   if r.boxes.id is not None else [None] * len(xyxy))
            for (x1, y1, x2, y2), conf, tid in zip(xyxy, confs, ids):
                boxes_px.append({
                    "px": (int(x1), int(y1), int(x2), int(y2)),
                    "conf": float(conf),
                    "track_id": (int(tid) if tid is not None else None),
                })

        # face recognition only on sampled frames
        do_faces = (processed % face_every == 0)
        if do_faces and boxes_px:
            faces = bank.detect_faces(frame_bgr)
            if faces:
                for b in boxes_px:
                    face = bank.best_face_for_box(faces, b["px"])
                    if face is None:
                        continue
                    thumb = _crop(frame_bgr, face["bbox"])
                    iid = bank.assign(face["embedding"], thumbnail=thumb,
                                      det_score=face["det_score"])
                    if b["track_id"] is not None:
                        track_votes[b["track_id"]][iid] += 1

        # store normalised boxes for this frame (identity filled later)
        norm_boxes = []
        for b in boxes_px:
            x1, y1, x2, y2 = b["px"]
            norm_boxes.append({
                "bbox": [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
                "conf": b["conf"],
                "track_id": b["track_id"],
            })
        frame_records.append({"timestamp": timestamp, "boxes": norm_boxes})

        processed += 1
        if progress_cb and processed % 30 == 0:
            progress_cb(processed, f"Tracking + identity… frame {real_frame}")

    # PASS 2 — resolve each track's identity by majority vote
    # ------------------------------------------------------------------
    track_identity: dict[int, str] = {}
    for tid, votes in track_votes.items():
        winner, n = votes.most_common(1)[0]
        if n >= min_votes:
            track_identity[tid] = winner

    print(f"🪪 Resolved {len(track_identity)} track→identity mappings "
          f"from {len(track_votes)} tracks "
          f"({len(bank)} identities in bank)")

    # PASS 3 — emit object_bboxes shape with identity filled in
    # ------------------------------------------------------------------
    object_bboxes: list[dict] = []
    for rec in frame_records:
        if not rec["boxes"]:
            continue
        objects, bboxes, confs = [], [], []
        track_ids, identity_ids, identity_names = [], [], []
        for b in rec["boxes"]:
            tid = b["track_id"]
            iid = track_identity.get(tid) if tid is not None else None
            objects.append("person")
            bboxes.append(b["bbox"])
            confs.append(b["conf"])
            track_ids.append(tid)
            identity_ids.append(iid)
            identity_names.append(bank.name_for(iid) if iid else None)
        object_bboxes.append({
            "timestamp": rec["timestamp"],
            "objects": objects,
            "bboxes": bboxes,
            "confidences": confs,
            "track_ids": track_ids,
            "identity_ids": identity_ids,
            "identity_names": identity_names,
        })

    if save_bank:
        bank.save()

    return object_bboxes


# ──────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────

def _crop(frame_bgr: np.ndarray, bbox) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    h, w = frame_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame_bgr[y1:y2, x1:x2].copy()


# ──────────────────────────────────────────────────────────────────
# smoke test:  python identity_tagging.py <video> [face_db.json]
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python identity_tagging.py <video> [face_db.json]")
        sys.exit(1)

    from face_identity import FaceIdentityBank

    video = sys.argv[1]
    db = sys.argv[2] if len(sys.argv) > 2 else None

    bank = FaceIdentityBank(db_path=db)
    entries = tag_video_with_identities(
        video, bank,
        face_every=10,
        progress_cb=lambda i, m: print(f"  [{i}] {m}"),
    )

    # summary
    seen_ids = set()
    for e in entries:
        for iid in e["identity_ids"]:
            if iid:
                seen_ids.add(iid)
    print(f"\nFrames with boxes: {len(entries)}")
    print(f"Distinct identities seen in video: {len(seen_ids)}")
    for ident in bank.all_identities():
        print(f"  {ident['id'][:8]}  name={ident['name']}  seen={ident['count']}")

    if db:
        bank.save()
    # dump a tiny preview
    if entries:
        print("\nFirst entry preview:")
        print(json.dumps(entries[0], indent=2)[:600])