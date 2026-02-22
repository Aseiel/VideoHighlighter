"""
Shared Visualization Utilities
================================

Training sample visualization: annotated videos showing
person detection, ROI, pose skeletons, and focus region.
"""

import cv2
import random
import numpy as np
from tqdm import tqdm

from .detection import (
    PersonTracker,
    SmartActionDetector,
    AdaptiveActionDetector,
    SmoothedROIDetector,
    PoseExtractor,
    merge_boxes,
    get_yolo_people_model,
)


# Skeleton connections for YOLOv11 pose (17 keypoints)
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # face
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),                 # torso
    (11, 13), (13, 15), (12, 14), (14, 16),     # legs
]


def visualize_training_sample(video_path, label, pose_extractor,
                               adaptive_detector, output_path="training_sample.mp4",
                               sample_rate=5, debug=False, visualize_skeletons=False):
    """Create annotated video showing detection pipeline for a training sample."""
    print(f"\n🎬 Visualization: {video_path}")
    print(f"   Label: {label} | sample_rate: {sample_rate}")

    adaptive_detector.debug = debug
    yolo_people = get_yolo_people_model()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open: {video_path}")
        return False

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Try codecs
    video_writer = None
    for codec, ext in [("mp4v", ".mp4"), ("avc1", ".mp4"), ("XVID", ".avi")]:
        try:
            out = output_path.rsplit(".", 1)[0] + ext
            fourcc = cv2.VideoWriter_fourcc(*codec)
            vw = cv2.VideoWriter(out, fourcc, fps, (width, height + 100))
            if vw.isOpened():
                video_writer = vw
                output_path = out
                break
            vw.release()
        except Exception:
            continue

    if video_writer is None:
        print("❌ Could not init video writer")
        cap.release()
        return False

    tracker = PersonTracker(iou_threshold=0.3, max_lost_frames=10)
    action_det = SmartActionDetector()
    roi_smoother = SmoothedROIDetector(window_size=3, base_alpha=0.5, adaptive=True)

    color_palette = [(0, 255, 0), (255, 0, 255), (255, 255, 0), (0, 255, 255)]
    track_colors = {}

    last_tracked = []
    last_roi = None
    last_poses = []
    focus_region = "full_body"
    frame_count = 0
    ok_frames = 0

    pbar = tqdm(total=total_frames, desc="Visualizing")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            display = frame.copy()

            if frame_count % sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                last_tracked = action_det.detect_with_tracking(
                    frame_rgb, yolo_people, tracker, max_people=2
                )
                boxes = (
                    [b for _, b in last_tracked]
                    if last_tracked and isinstance(last_tracked[0], tuple)
                    else last_tracked
                )

                if pose_extractor and boxes:
                    roi, focus_region = adaptive_detector.detect_action_region(
                        frame_rgb, boxes, pose_extractor, max_poses=2
                    )
                else:
                    roi = merge_boxes(boxes) if boxes else None
                    focus_region = "full_body"

                last_roi = roi_smoother.update(roi)

                # Optional skeleton
                if visualize_skeletons and pose_extractor and boxes:
                    last_poses = _match_poses(frame_rgb, boxes, pose_extractor)
                else:
                    last_poses = []

            # Draw person boxes
            for item in last_tracked:
                if isinstance(item, tuple) and len(item) == 2:
                    tid, (x1, y1, x2, y2) = item
                    if tid not in track_colors:
                        track_colors[tid] = color_palette[len(track_colors) % len(color_palette)]
                    c = track_colors[tid]
                    cv2.rectangle(display, (x1, y1), (x2, y2), c, 3)
                    cv2.putText(display, f"Person {tid}", (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2)

            # Draw ROI
            if last_roi:
                rx1, ry1, rx2, ry2 = last_roi
                roi_color = {
                    "upper_body": (0, 255, 255),
                    "lower_body": (0, 255, 0),
                }.get(focus_region, (255, 0, 0))
                cv2.rectangle(display, (rx1, ry1), (rx2, ry2), roi_color, 4)

            # Draw skeleton
            for kpts in last_poses:
                for i1, i2 in POSE_CONNECTIONS:
                    if kpts[i1, 2] > 0.3 and kpts[i2, 2] > 0.3:
                        p1 = (int(kpts[i1, 0]), int(kpts[i1, 1]))
                        p2 = (int(kpts[i2, 0]), int(kpts[i2, 1]))
                        cv2.line(display, p1, p2, (0, 255, 255), 2)
                for kp in kpts:
                    if kp[2] > 0.3:
                        cv2.circle(display, (int(kp[0]), int(kp[1])), 4, (0, 0, 255), -1)

            # Label bar
            bar = np.zeros((100, width, 3), dtype=np.uint8)
            cv2.putText(bar, f"ACTION: {label}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(bar, f"Frame: {frame_count}/{total_frames}", (width - 250, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            focus_color = {"upper_body": (0, 255, 255), "lower_body": (0, 255, 0)}.get(
                focus_region, (255, 255, 255)
            )
            cv2.putText(bar, f"People: {len(last_tracked)} | Focus: {focus_region}",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, focus_color, 2)

            video_writer.write(np.vstack([bar, display]))
            ok_frames += 1

        except Exception as e:
            print(f"⚠️  Frame {frame_count}: {e}")

        frame_count += 1
        pbar.update(1)

    cap.release()
    video_writer.release()
    pbar.close()

    pct = (ok_frames / frame_count * 100) if frame_count else 0
    print(f"✅ {ok_frames}/{frame_count} frames ({pct:.1f}%) → {output_path}")
    return ok_frames > 0


def create_sample_visualizations(dataset, pose_extractor, num_samples=2,
                                  sample_rate=5, visualize_skeletons=False):
    """Create visualisation videos for random training samples."""
    if not dataset.samples:
        print("❌ No samples")
        return

    indices = random.sample(
        range(len(dataset.samples)), min(num_samples, len(dataset.samples))
    )
    det = AdaptiveActionDetector()

    for i, idx in enumerate(indices):
        vp, lbl_idx = dataset.samples[idx]
        name = dataset.idx_to_label[lbl_idx]
        out = f"sample_{i + 1}_{name.replace(' ', '_')}.mp4"
        visualize_training_sample(
            vp, name, pose_extractor, det, out,
            sample_rate=sample_rate, visualize_skeletons=visualize_skeletons,
        )


# ---- helper ----
def _match_poses(frame_rgb, boxes, pose_extractor, max_poses=2):
    """Match detected poses to person boxes."""
    results = pose_extractor.model.predict(frame_rgb, conf=0.3, verbose=False)
    if not results or results[0].keypoints is None:
        return []
    all_kpts = results[0].keypoints.data.cpu().numpy()
    matched = []
    for box in boxes[:max_poses]:
        ax1, ay1, ax2, ay2 = box
        best_idx, best_s = None, 0
        for idx, kpts in enumerate(all_kpts):
            vis = kpts[kpts[:, 2] > 0.3]
            if len(vis) >= 5:
                c = vis[:, :2].mean(axis=0)
                if ax1 <= c[0] <= ax2 and ay1 <= c[1] <= ay2:
                    if len(vis) > best_s:
                        best_s = len(vis)
                        best_idx = idx
        if best_idx is not None:
            matched.append(all_kpts[best_idx])
    return matched
