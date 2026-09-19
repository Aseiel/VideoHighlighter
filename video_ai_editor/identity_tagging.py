"""
Identity Tagging Pass — attach a persistent person identity to every box.

This is the OFFLINE pass (runs during analysis, writes to cache). It does NOT
run during playback. The live overlay just reads the identity_id this produces.

Pipeline:
    1. A person tracker supplies boxes across the video -> each person box gets
       a track_id.
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
    - Tracking runs every (strided) frame. Use `vid_stride` to skip frames if you
      need it faster.
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
    yolo_model_path: str | None = None,     # unused; the tracker resolves its own model
    person_conf: float = 0.25,
    face_every: int = 10,                   # run face recognition every N processed frames
    vid_stride: int = 1,                    # skip frames in tracking (1 = every frame)
    tracker: str = "iou",                   # unused; tracking is IoU association
    min_votes: int = 1,                     # min face matches before trusting an identity
    save_bank: bool = False,
    model=None,                             # pre-built tracker model (reused if provided)
    device=None,                            # OpenVINO device for the tracker ('GPU' | 'CPU')
    progress_cb=None,                       # optional: progress_cb(frame_idx, message)
    avoid_match_threshold: float = 0.38,    # direct-match threshold for AVOIDED ids
                                            # (looser than bank.sim_threshold on purpose)
) -> list[dict]:
    """
    Run tracking + face identity over a video and return identity-tagged
    per-timestamp box entries (object_bboxes shape).

    AVOID OVERRIDE: every detected face is also matched DIRECTLY against the
    embeddings of avoided identities (bank.avoided_ids()) with a relaxed
    threshold. A direct hit overrides the track's majority vote — for
    exclusion, "appears at all" must win over "is the most common face".
    """
    import cv2

    # fps for timestamps
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    if model is None:
        from modules.segments.compute_forbidden import build_tracking_model
        model = build_tracking_model("n", device=device or "GPU")
    if model is None:
        print("⚠️ Identity tracking skipped — no person tracker available")
        if save_bank:
            bank.save()
        return []

    # ── direct-match galleries for avoided identities ────────────────
    avoid_galleries = []
    try:
        for av_iid in bank.avoided_ids():
            ident = bank._id_index.get(av_iid)
            if ident is not None and ident["embeddings"].size:
                avoid_galleries.append((av_iid, ident["embeddings"]))
    except Exception:
        pass
    if avoid_galleries:
        print(f"🚫 Direct avoid-match active for {len(avoid_galleries)} identit(ies), "
              f"threshold={avoid_match_threshold}")

    # PASS 1 — track people, collect boxes, vote on identities per track
    # ------------------------------------------------------------------
    frame_records: list[dict] = []          # {timestamp, boxes:[{...}]}
    track_votes: dict[int, Counter] = defaultdict(Counter)
    avoid_track_votes: dict[int, Counter] = defaultdict(Counter)

    if hasattr(model, "iter_frames"):
        frame_iter = model.iter_frames(video_path, vid_stride=vid_stride, person_conf=person_conf)
        processed = 0
        for frame_bgr, real_frame, tracked in frame_iter:
            timestamp = real_frame / fps if fps else float(real_frame)
            h, w = frame_bgr.shape[:2]
            boxes_px = [{
                "px": (p.x1, p.y1, p.x2, p.y2),
                "conf": p.conf,
                "track_id": p.track_id,
                "force_iid": None,
            } for p in tracked]
            _process_identity_frame(
                frame_bgr, boxes_px, processed, face_every, bank,
                track_votes, avoid_track_votes, avoid_galleries, avoid_match_threshold,
            )
            norm_boxes = []
            for b in boxes_px:
                x1, y1, x2, y2 = b["px"]
                norm_boxes.append({
                    "bbox": [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
                    "conf": b["conf"],
                    "track_id": b["track_id"],
                    "force_iid": b["force_iid"],
                })
            frame_records.append({"timestamp": timestamp, "boxes": norm_boxes})
            processed += 1
            if progress_cb and processed % 30 == 0:
                progress_cb(processed, f"Tracking + identity… frame {real_frame}")
    else:
        print("⚠️ Identity tracking: unknown tracker model type")
        if save_bank:
            bank.save()
        return []

    # PASS 2 — resolve each track's identity by majority vote
    # ------------------------------------------------------------------
    track_identity: dict[int, str] = {}
    for tid, votes in track_votes.items():
        winner, n = votes.most_common(1)[0]
        if n >= min_votes:
            track_identity[tid] = winner

    # AVOID OVERRIDE — any direct avoid hit on a track wins over the vote.
    forced = 0
    for tid, votes in avoid_track_votes.items():
        av_iid, _ = votes.most_common(1)[0]
        if track_identity.get(tid) != av_iid:
            forced += 1
        track_identity[tid] = av_iid

    print(f"🪪 Resolved {len(track_identity)} track→identity mappings "
          f"from {len(track_votes)} tracks "
          f"({len(bank)} identities in bank)"
          + (f" — 🚫 {forced} track(s) forced to avoided identity" if forced else ""))

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
            iid = b.get("force_iid") or (track_identity.get(tid) if tid is not None else None)
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

def _process_identity_frame(frame_bgr, boxes_px, processed, face_every, bank,
                            track_votes, avoid_track_votes, avoid_galleries,
                            avoid_match_threshold):
    """Run face recognition on sampled frames and vote on track identities."""
    do_faces = (processed % face_every == 0)
    if not do_faces or not boxes_px:
        return
    faces = bank.detect_faces(frame_bgr)
    if not faces:
        return
    for b in boxes_px:
        face = bank.best_face_for_box(faces, b["px"])
        if face is None:
            continue
        thumb = _crop(frame_bgr, face["bbox"])
        iid = bank.assign(face["embedding"], thumbnail=thumb,
                          det_score=face["det_score"])
        if b["track_id"] is not None:
            track_votes[b["track_id"]][iid] += 1
        for av_iid, gal in avoid_galleries:
            sim = float(np.max(gal @ face["embedding"]))
            if sim >= avoid_match_threshold:
                b["force_iid"] = av_iid
                if b["track_id"] is not None:
                    avoid_track_votes[b["track_id"]][av_iid] += 1
                break


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