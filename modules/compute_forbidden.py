"""
compute_forbidden.py — turn "avoid these identities" into the two feeds the
pipeline needs, with a per-video cache so the expensive tagging pass runs ONCE.

    forbidden_ranges          : [(start_sec, end_sec), ...]        -> skip method
    forbidden_boxes_by_frame  : {frame_idx: [(x1,y1,x2,y2), ...]}  -> crop method (pixels)

Device: Intel Arc is driven through OpenVINO (Ultralytics device='intel:gpu') —
Ultralytics does not accept torch's 'xpu' string. NVIDIA uses the .pt model on CUDA;
plain CPU uses the .pt model. InsightFace stays on CPU. Result cached per video.
"""

from __future__ import annotations
import os
import json
import hashlib
import cv2


def build_tracking_model(model_size="n", log_fn=print):
    """YOLO tracking model on the best device.
      Intel Arc  -> OpenVINO model (Ultralytics device='intel:gpu')
      NVIDIA     -> .pt on CUDA
      CPU        -> .pt on CPU
    The inference device itself is applied at track() time (see track_device)."""
    from ultralytics import YOLO
    import torch

    # Intel GPU (Arc): Ultralytics runs Intel GPUs via OpenVINO, not torch xpu.
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        ov_folder = f"yolo11{model_size}_openvino_model/"
        if not os.path.exists(ov_folder):
            log_fn(f"⚙️ Exporting yolo11{model_size} → OpenVINO for Intel GPU (one-time)…")
            YOLO(f"yolo11{model_size}.pt").export(format="openvino")
        log_fn("✅ Tracking model: OpenVINO (Intel GPU / Arc)")
        return YOLO(ov_folder, task="detect")

    if torch.cuda.is_available():
        log_fn(f"✅ Tracking model: CUDA yolo11{model_size}.pt")
        return YOLO(f"yolo11{model_size}.pt")

    log_fn(f"✅ Tracking model: CPU yolo11{model_size}.pt")
    return YOLO(f"yolo11{model_size}.pt")


def track_device():
    """Ultralytics device for the track() call: 'intel:gpu' | 0 (cuda) | 'cpu'."""
    import torch
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "intel:gpu"     # OpenVINO GPU plugin → the Arc
    if torch.cuda.is_available():
        return 0
    return "cpu"


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
            ranges.append((float(start), float(prev + 1)))
            start = prev = cur
    ranges.append((float(start), float(prev + 1)))
    return ranges


# ── cache ─────────────────────────────────────────────────────────────────────
def _cache_key(video_path, avoid_ids, model_size, face_every, vid_stride):
    try:
        st = os.stat(video_path)
        stat_sig = f"{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        stat_sig = "nostat"
    sig = "|".join([
        os.path.abspath(video_path), stat_sig,
        ",".join(sorted(avoid_ids)), str(model_size), str(face_every), str(vid_stride),
    ])
    return hashlib.md5(sig.encode("utf-8")).hexdigest()


def _cache_load(path, log_fn):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ranges = [tuple(r) for r in data.get("forbidden_ranges", [])]
        boxes = {int(k): [tuple(b) for b in v]
                 for k, v in data.get("forbidden_boxes_by_frame", {}).items()}
        log_fn(f"🚫 Avoid: loaded cached tagging ({len(boxes)} frame(s), {len(ranges)} range(s))")
        return ranges, boxes
    except Exception:
        return None


def _cache_save(path, ranges, boxes, log_fn):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "forbidden_ranges": [list(r) for r in ranges],
            "forbidden_boxes_by_frame": {str(k): [list(b) for b in v] for k, v in boxes.items()},
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception as e:
        log_fn(f"⚠️ Avoid: could not write cache: {e}")


def compute_forbidden(video_path, bank, avoid_ids, fps,
                      yolo_model=None, model_size="n",
                      face_every=15, vid_stride=3, merge_gap=2.0,
                      use_cache=True, cache_dir="./cache",
                      log_fn=print, cancel_flag=None):
    """Returns (forbidden_ranges, forbidden_boxes_by_frame). Cached per video."""
    avoid = set(avoid_ids or [])
    if not avoid:
        return [], {}

    cache_path = os.path.join(cache_dir, "avoid",
                              _cache_key(video_path, avoid, model_size, face_every, vid_stride) + ".json")
    if use_cache and os.path.exists(cache_path):
        hit = _cache_load(cache_path, log_fn)
        if hit is not None:
            return hit

    from video_ai_editor.identity_tagging import tag_video_with_identities

    cap = cv2.VideoCapture(video_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if W == 0 or H == 0:
        log_fn("⚠️ compute_forbidden: could not read frame size — skipping exclusion")
        return [], {}

    if yolo_model is None:
        yolo_model = build_tracking_model(model_size, log_fn=log_fn)

    def _progress(i, msg):
        if cancel_flag is not None and getattr(cancel_flag, "is_set", lambda: False)():
            raise RuntimeError("cancelled during identity tagging")
        if i % 150 == 0:
            log_fn(f"🚫 Avoid: {msg}")

    dev = track_device()
    log_fn(f"🚫 Avoid: tracking on device={dev}")
    entries = tag_video_with_identities(
        video_path, bank,
        model=yolo_model,
        device=dev,
        face_every=face_every,
        vid_stride=vid_stride,
        save_bank=False,
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

    if use_cache:
        _cache_save(cache_path, forbidden_ranges, forbidden_boxes_by_frame, log_fn)

    return forbidden_ranges, forbidden_boxes_by_frame


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