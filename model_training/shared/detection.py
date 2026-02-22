"""
Shared Detection Utilities
===========================

Person detection, tracking, adaptive action region detection, ROI smoothing,
and pose estimation. Used by both Intel and R3D training pipelines.
"""

import cv2
import numpy as np
from collections import deque

# =============================
# Optional YOLO import
# =============================
YOLO_AVAILABLE = False
_yolo_people_model = None  # Lazy-loaded singleton

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass


def get_yolo_people_model(model_name="yolo11n.pt"):
    """Lazy-load YOLO people detection model (singleton)."""
    global _yolo_people_model
    if _yolo_people_model is None:
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed — run: pip install ultralytics")
        _yolo_people_model = YOLO(model_name)
    return _yolo_people_model


# =============================
# Box Utilities
# =============================
def merge_boxes(boxes):
    """Merge multiple bounding boxes into one enclosing box."""
    if len(boxes) == 0:
        return None
    if len(boxes) == 1:
        return boxes[0]
    x1_min = min(b[0] for b in boxes)
    y1_min = min(b[1] for b in boxes)
    x2_max = max(b[2] for b in boxes)
    y2_max = max(b[3] for b in boxes)
    return (x1_min, y1_min, x2_max, y2_max)


def compute_iou(box1, box2):
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    x1_i = max(box1[0], box2[0])
    y1_i = max(box1[1], box2[1])
    x2_i = min(box1[2], box2[2])
    y2_i = min(box1[3], box2[3])
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def crop_roi(frame, roi, output_size):
    """Crop frame to ROI at high resolution, then resize to output_size (H, W)."""
    if roi is None:
        h, w = frame.shape[:2]
        th, tw = output_size
        y = max(0, h // 2 - th // 2)
        x = max(0, w // 2 - tw // 2)
        crop = frame[y:y + th, x:x + tw]
        return cv2.resize(crop, output_size)

    x1, y1, x2, y2 = roi
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        th, tw = output_size
        y = max(0, h // 2 - th // 2)
        x = max(0, w // 2 - tw // 2)
        crop = frame[y:y + th, x:x + tw]
        return cv2.resize(crop, output_size)

    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, output_size)


# =============================
# Person Tracker (IoU-based)
# =============================
class PersonTracker:
    """Simple IoU-based person tracker."""

    def __init__(self, iou_threshold=0.3, max_lost_frames=10):
        self.tracks = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames

    def update(self, detected_boxes):
        if len(detected_boxes) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['lost_frames'] += 1
                if self.tracks[track_id]['lost_frames'] > self.max_lost_frames:
                    del self.tracks[track_id]
            return []

        matched_tracks = set()
        matched_detections = set()

        for det_idx, det_box in enumerate(detected_boxes):
            best_iou = 0
            best_track_id = None
            for track_id, track_data in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                iou = compute_iou(det_box, track_data['box'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
            if best_track_id is not None:
                self.tracks[best_track_id]['box'] = det_box
                self.tracks[best_track_id]['lost_frames'] = 0
                matched_tracks.add(best_track_id)
                matched_detections.add(det_idx)
            else:
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {'box': det_box, 'lost_frames': 0}
                matched_detections.add(det_idx)

        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.tracks[track_id]['lost_frames'] += 1
                if self.tracks[track_id]['lost_frames'] > self.max_lost_frames:
                    del self.tracks[track_id]

        sorted_tracks = sorted(self.tracks.items(), key=lambda x: x[0])
        return [(tid, data['box']) for tid, data in sorted_tracks]

    def reset(self):
        self.tracks.clear()
        self.next_id = 0


# =============================
# Smart Action Detector
# =============================
class SmartActionDetector:
    """
    Detects people most likely performing the action using interaction
    and relative motion patterns.
    """

    def __init__(self, sticky_frames=15, sticky_weight=0.5, debug=False):
        self.prev_frame_data = None
        self.frame_count = 0
        self.selection_history = deque(maxlen=sticky_frames)
        self.sticky_weight = sticky_weight
        self.locked_pair = None
        self.lock_strength = 0
        self.locked_track_ids = set()
        self.debug = debug

    # ----- core detection -----
    def detect(self, frame, detector, max_people=2, allow_dynamic_group=True):
        h, w = frame.shape[:2]
        center_x, center_y = w / 2, h / 2

        result = detector.predict(frame, conf=0.40, classes=[0], verbose=False)
        current_detections = []

        for r in result:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf)
                bcx = (x1 + x2) / 2
                bcy = (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)

                dist = np.sqrt((bcx - center_x) ** 2 + (bcy - center_y) ** 2)
                max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
                center_score = 1 - (dist / max_dist)
                size_score = min(area / (h * w * 0.3), 1.0)

                motion_score = 0
                motion_vector = (0, 0)
                if self.prev_frame_data and self.frame_count > 0:
                    best_motion = 0
                    best_vec = (0, 0)
                    for prev in self.prev_frame_data:
                        iou = compute_iou((x1, y1, x2, y2), prev['box'])
                        if iou > 0.3:
                            pcx, pcy = prev['center']
                            pos_change = np.sqrt((bcx - pcx) ** 2 + (bcy - pcy) ** 2)
                            sz_change = abs(area - prev['area']) / max(prev['area'], 1)
                            m = pos_change / 50.0 + sz_change * 2.0
                            if m > best_motion:
                                best_motion = m
                                best_vec = (bcx - pcx, bcy - pcy)
                    motion_score = min(best_motion, 1.0)
                    motion_vector = best_vec

                temporal_score = 0
                for prev_sel in self.selection_history:
                    for prev_box in prev_sel:
                        if compute_iou((x1, y1, x2, y2), prev_box) > 0.5:
                            temporal_score = 1.0
                            break
                    if temporal_score > 0:
                        break

                current_detections.append({
                    'box': (x1, y1, x2, y2), 'center': (bcx, bcy), 'area': area,
                    'conf': conf, 'motion': motion_score, 'motion_vector': motion_vector,
                    'center_prox': center_score, 'size': size_score, 'temporal': temporal_score,
                })

        # interaction + motion coherence
        frame_diag = np.sqrt(w ** 2 + h ** 2)
        for i, det in enumerate(current_detections):
            interaction_score = 0
            max_coherence = 0
            for j, other in enumerate(current_detections):
                if i == j:
                    continue
                dx = det['center'][0] - other['center'][0]
                dy = det['center'][1] - other['center'][1]
                norm_dist = np.sqrt(dx ** 2 + dy ** 2) / frame_diag
                proximity = max(0, 1 - norm_dist * 3)
                overlap = compute_iou(det['box'], other['box']) * 2.0

                coherence = 0
                if self.prev_frame_data:
                    my_mag = np.sqrt(det['motion_vector'][0] ** 2 + det['motion_vector'][1] ** 2)
                    ot_mag = np.sqrt(other['motion_vector'][0] ** 2 + other['motion_vector'][1] ** 2)
                    diff = abs(my_mag - ot_mag)
                    if diff > 5 and norm_dist < 0.35:
                        coherence = min(diff / 40.0, 1.0) * 1.8
                    elif my_mag > 3 and ot_mag > 3:
                        dot = (det['motion_vector'][0] * other['motion_vector'][0]
                               + det['motion_vector'][1] * other['motion_vector'][1])
                        cos_sim = dot / (my_mag * ot_mag)
                        if abs(cos_sim) > 0.6:
                            coherence = 0.9 * abs(cos_sim)
                    if norm_dist < 0.25 and (my_mag > 3 or ot_mag > 3):
                        coherence = max(coherence, min((my_mag + ot_mag) / 60.0, 1.0) * 1.2)

                pair_int = proximity * 0.4 + overlap * 0.3 + coherence * 0.3
                interaction_score = max(interaction_score, pair_int)
                max_coherence = max(max_coherence, coherence)

            det['interaction'] = min(interaction_score, 1.0)
            det['motion_coherence'] = max_coherence

        # final score
        for det in current_detections:
            lock_bonus = 0
            if self.locked_pair and self.selection_history:
                for lb in self.selection_history[-1]:
                    if compute_iou(det['box'], lb) > 0.4:
                        lock_bonus = 0.5 * (self.lock_strength / 10.0)
                        break
            det['score'] = (
                det['conf'] * 0.07 + det['center_prox'] * 0.08 + det['size'] * 0.06
                + det['motion'] * 0.14 + det['interaction'] * 0.25
                + det['motion_coherence'] * 0.18 + det['temporal'] * 0.22 + lock_bonus
            )

        self.prev_frame_data = current_detections
        self.frame_count += 1

        # dynamic group
        if allow_dynamic_group:
            high_int = [d for d in current_detections if d['interaction'] > 0.4]
            if 2 <= len(high_int) <= 4:
                high_int.sort(key=lambda x: x['score'], reverse=True)
                boxes = [d['box'] for d in high_int]
                self.selection_history.append(boxes)
                return boxes

        # pair selection
        if max_people and len(current_detections) >= max_people:
            best_pair, best_ps = None, -1

            # check locked pair
            if self.locked_pair and self.selection_history and len(self.selection_history[-1]) >= 2:
                locked_matches = []
                for det in current_detections:
                    for lb in self.selection_history[-1][:2]:
                        if compute_iou(det['box'], lb) > 0.25:
                            locked_matches.append(det)
                            break
                if len(locked_matches) >= 2:
                    di, dj = locked_matches[0], locked_matches[1]
                    mi = np.sqrt(di['motion_vector'][0] ** 2 + di['motion_vector'][1] ** 2)
                    mj = np.sqrt(dj['motion_vector'][0] ** 2 + dj['motion_vector'][1] ** 2)
                    comp = 0.35 if abs(mi - mj) > 5 else 0
                    avg_coh = (di['motion_coherence'] + dj['motion_coherence']) / 2
                    lps = ((di['score'] + dj['score']) / 2
                           + di['interaction'] + dj['interaction'] + comp + avg_coh * 0.5 + 0.6)
                    if lps > 0.3:
                        self.lock_strength = min(self.lock_strength + 1, 10)
                        boxes = [di['box'], dj['box']]
                        self.selection_history.append(boxes)
                        self.locked_pair = boxes
                        return boxes

            for i in range(len(current_detections)):
                for j in range(i + 1, len(current_detections)):
                    di, dj = current_detections[i], current_detections[j]
                    mi = np.sqrt(di['motion_vector'][0] ** 2 + di['motion_vector'][1] ** 2)
                    mj = np.sqrt(dj['motion_vector'][0] ** 2 + dj['motion_vector'][1] ** 2)
                    comp = 0.35 if abs(mi - mj) > 5 else 0
                    temp = (di['temporal'] + dj['temporal']) * 0.40
                    avg_coh = (di['motion_coherence'] + dj['motion_coherence']) / 2
                    ps = ((di['score'] + dj['score']) / 2
                          + di['interaction'] + dj['interaction'] + temp + comp + avg_coh * 0.5)
                    if ps > best_ps:
                        best_ps = ps
                        best_pair = (i, j)

            if best_pair and best_ps > 0.5:
                ii, jj = best_pair
                boxes = [current_detections[ii]['box'], current_detections[jj]['box']]
                is_new = True
                if self.locked_pair:
                    match_count = sum(
                        1 for sb in boxes
                        for lb in self.locked_pair
                        if compute_iou(sb, lb) > 0.3
                    )
                    if match_count >= 2:
                        is_new = False
                if is_new:
                    if best_ps > 0.9:
                        self.locked_pair = boxes
                        self.lock_strength = 1
                    else:
                        if self.locked_pair and self.selection_history:
                            self.lock_strength = max(self.lock_strength - 1, 0)
                            if self.lock_strength > 0:
                                return self.selection_history[-1][:2]
                        self.locked_pair = boxes
                        self.lock_strength = 1
                else:
                    self.lock_strength = min(self.lock_strength + 1, 10)
                    self.locked_pair = boxes
                self.selection_history.append(boxes)
                return boxes

        # fallback
        sorted_dets = sorted(current_detections, key=lambda x: x['score'], reverse=True)
        boxes = [d['box'] for d in sorted_dets[:max_people]] if max_people else [d['box'] for d in sorted_dets]
        self.selection_history.append(boxes)
        return boxes

    # ----- tracking wrapper -----
    def detect_with_tracking(self, frame, detector, tracker, max_people=2):
        if len(self.locked_track_ids) >= 2:
            all_boxes = self.detect(frame, detector, max_people=None)
            tracked = tracker.update(all_boxes)
            locked = [(tid, box) for tid, box in tracked if tid in self.locked_track_ids]
            if len(locked) >= max_people:
                return locked[:max_people]

        action_boxes = self.detect(frame, detector, max_people=None)
        tracked = tracker.update(action_boxes)
        if len(tracked) <= max_people:
            if len(tracked) >= 2:
                self.locked_track_ids = {tid for tid, _ in tracked[:max_people]}
            return tracked

        scored = []
        for tid, box in tracked:
            if self.prev_frame_data:
                for det in self.prev_frame_data:
                    if compute_iou(box, det['box']) > 0.5:
                        scored.append((det['score'], tid, box))
                        break
        scored.sort(reverse=True)
        top = [(tid, box) for _, tid, box in scored[:max_people]]
        if len(top) >= 2:
            self.locked_track_ids = {tid for tid, _ in top}
        return top

    def reset(self):
        self.prev_frame_data = None
        self.frame_count = 0
        self.selection_history.clear()
        self.locked_pair = None
        self.lock_strength = 0
        self.locked_track_ids = set()


# =============================
# Adaptive Action Region Detector
# =============================
class AdaptiveActionDetector:
    """Detects WHERE the action is happening based on motion and pose analysis."""

    def __init__(self, motion_threshold=5.0, debug=False):
        self.motion_threshold = motion_threshold
        self.prev_poses = None
        self.debug = debug

    def detect_action_region(self, frame, person_boxes, pose_extractor, max_poses=2):
        h, w = frame.shape[:2]
        poses = self._get_matched_poses(frame, person_boxes, pose_extractor, max_poses)

        if len(poses) == 0:
            return self._merge_boxes(person_boxes), 'lower_body'

        visibility = self._check_body_visibility(poses)
        motion = self._analyze_motion_regions(poses)
        focus = self._determine_focus_region(poses, motion, person_boxes, visibility)
        roi = self._adaptive_crop(focus, poses, w, h)
        return roi, focus

    # ----- internal helpers -----
    def _get_matched_poses(self, frame, person_boxes, pose_extractor, max_poses):
        frame_rgb = frame if frame.shape[2] == 3 and frame.dtype == np.uint8 else frame
        # Try to handle BGR input
        try:
            results = pose_extractor.model.predict(frame_rgb, conf=0.15, verbose=False)
        except Exception:
            return []

        if len(results) == 0 or results[0].keypoints is None:
            return []

        all_kpts = results[0].keypoints.data.cpu().numpy()
        matched = []
        for box in person_boxes[:max_poses]:
            ax1, ay1, ax2, ay2 = box
            best_idx, best_score = None, 0
            for idx, kpts in enumerate(all_kpts):
                if np.sum(kpts[:, 2] > 0.15) >= 2:
                    visible = kpts[kpts[:, 2] > 0.15]
                    center = visible[:, :2].mean(axis=0)
                    if ax1 <= center[0] <= ax2 and ay1 <= center[1] <= ay2:
                        s = len(visible)
                        if s > best_score:
                            best_score = s
                            best_idx = idx
            if best_idx is not None:
                matched.append(all_kpts[best_idx])
                if len(matched) >= max_poses:
                    break
        return matched

    def _check_body_visibility(self, poses):
        vis = {'has_upper': False, 'has_lower': False, 'has_hips': False,
               'has_feet': False, 'has_hands': False}
        for p in poses:
            if p[11, 2] > 0.15 or p[12, 2] > 0.15:
                vis['has_hips'] = vis['has_lower'] = True
            if p[15, 2] > 0.15 or p[16, 2] > 0.15:
                vis['has_feet'] = vis['has_lower'] = True
            if p[13, 2] > 0.15 or p[14, 2] > 0.15:
                vis['has_lower'] = True
            if p[5, 2] > 0.15 or p[6, 2] > 0.15:
                vis['has_upper'] = True
            if p[9, 2] > 0.15 or p[10, 2] > 0.15:
                vis['has_hands'] = vis['has_upper'] = True
        return vis

    def _analyze_motion_regions(self, current_poses):
        regions = {'head': 0, 'upper_body': 0, 'lower_body': 0, 'hands': 0, 'feet': 0}
        if self.prev_poses is None or len(self.prev_poses) != len(current_poses):
            self.prev_poses = current_poses
            if current_poses:
                for p in current_poses:
                    lower_vis = sum(1 for i in [11, 12, 13, 14, 15, 16] if p[i, 2] > 0.15)
                    upper_vis = sum(1 for i in [5, 6, 7, 8, 9, 10] if p[i, 2] > 0.15)
                    if lower_vis >= upper_vis:
                        regions['lower_body'] += 2.0
                        regions['feet'] += 1.5
                    else:
                        regions['upper_body'] += 2.0
            return regions

        mapping = {
            'head': [0, 1, 2, 3, 4], 'upper_body': [5, 6, 7, 8, 9, 10],
            'lower_body': [11, 12, 13, 14, 15, 16], 'hands': [9, 10], 'feet': [15, 16],
        }
        for prev, curr in zip(self.prev_poses, current_poses):
            for name, indices in mapping.items():
                rm, vp = 0, 0
                for idx in indices:
                    if prev[idx, 2] > 0.15 and curr[idx, 2] > 0.15:
                        m = np.sqrt((curr[idx, 0] - prev[idx, 0]) ** 2
                                    + (curr[idx, 1] - prev[idx, 1]) ** 2)
                        if name in ('lower_body', 'feet'):
                            m *= 1.3
                        rm += m
                        vp += 1
                if vp > 0:
                    regions[name] += rm / vp

        self.prev_poses = current_poses
        return regions

    def _determine_focus_region(self, poses, motion, boxes, vis):
        if not poses:
            return 'lower_body'

        hands = motion.get('hands', 0)
        feet = motion.get('feet', 0)
        head = motion.get('head', 0)
        upper = motion.get('upper_body', 0)
        lower = motion.get('lower_body', 0)

        total_upper = upper + hands + head
        total_lower = lower + feet
        base_thr = self.motion_threshold * 0.3

        # full body check
        if total_upper > base_thr * 0.7 and total_lower > base_thr * 0.6:
            ratio = min(total_upper, total_lower) / max(total_upper, total_lower)
            if ratio > 0.5:
                return 'full_body'
            return 'upper_body' if total_upper > total_lower * 1.3 else 'lower_body'

        # priority checks
        if hands > base_thr * 1.5:
            return 'upper_body'
        if feet > base_thr * 0.6 or lower > base_thr * 0.6:
            return 'lower_body'
        if (vis['has_hips'] or vis['has_feet']) and (lower > 0 or feet > 0):
            return 'lower_body'
        if head > base_thr:
            return 'upper_body'
        if upper > base_thr * 0.7:
            return 'upper_body'
        if vis['has_hips'] or vis['has_feet']:
            return 'lower_body'
        return 'full_body'

    def _adaptive_crop(self, focus, poses, fw, fh):
        indices_map = {
            'upper_body': list(range(0, 11)),
            'lower_body': list(range(11, 17)),
            'full_body': list(range(0, 17)),
        }
        indices = indices_map.get(focus, list(range(17)))
        pts = []
        for p in poses:
            for idx in indices:
                if p[idx, 2] > 0.2:
                    pts.append(p[idx, :2])
        if not pts:
            return None

        pts = np.array(pts)
        xmin, ymin = pts.min(axis=0)
        xmax, ymax = pts.max(axis=0)

        pad_configs = {
            'upper_body': (0.4, 0.6, 0.3),
            'lower_body': (0.4, 0.3, 0.6),
            'full_body': (0.3, 0.4, 0.4),
        }
        px, pyt, pyb = pad_configs.get(focus, (0.3, 0.4, 0.4))
        dx, dy = xmax - xmin, ymax - ymin

        x1 = max(0, int(xmin - dx * px))
        y1 = max(0, int(ymin - dy * pyt))
        x2 = min(fw, int(xmax + dx * px))
        y2 = min(fh, int(ymax + dy * pyb))

        # enforce minimums
        if (x2 - x1) < fw * 0.3:
            cx = (x1 + x2) // 2
            x1 = max(0, int(cx - fw * 0.15))
            x2 = min(fw, int(cx + fw * 0.15))
        if (y2 - y1) < fh * 0.4:
            cy = (y1 + y2) // 2
            y1 = max(0, int(cy - fh * 0.2))
            y2 = min(fh, int(cy + fh * 0.2))

        return (x1, y1, x2, y2)

    def _merge_boxes(self, boxes):
        return merge_boxes(boxes)

    def reset(self):
        self.prev_poses = None


# =============================
# Smoothed ROI Detector
# =============================
class SmoothedROIDetector:
    """Temporal smoothing for ROI with adaptive alpha based on motion speed."""

    def __init__(self, window_size=5, base_alpha=0.5, adaptive=True, debug=False):
        self.window_size = window_size
        self.base_alpha = base_alpha
        self.alpha = base_alpha
        self.adaptive = adaptive
        self.debug = debug
        self.roi_history = deque(maxlen=window_size)
        self.smoothed_roi = None
        self.prev_roi = None

    def update(self, current_roi):
        if current_roi is None:
            return tuple(self.smoothed_roi.astype(int)) if self.smoothed_roi is not None else None

        self.roi_history.append(current_roi)

        if self.adaptive and self.prev_roi is not None:
            self.alpha = self._calc_alpha(current_roi, self.prev_roi)
        else:
            self.alpha = self.base_alpha

        if self.smoothed_roi is None:
            self.smoothed_roi = np.array(current_roi, dtype=np.float32)
        else:
            self.smoothed_roi = (self.alpha * np.array(current_roi, dtype=np.float32)
                                 + (1 - self.alpha) * self.smoothed_roi)

        self.prev_roi = current_roi
        return tuple(self.smoothed_roi.astype(int))

    def _calc_alpha(self, curr, prev):
        c, p = np.array(curr, dtype=np.float32), np.array(prev, dtype=np.float32)
        cc = np.array([(c[0] + c[2]) / 2, (c[1] + c[3]) / 2])
        pc = np.array([(p[0] + p[2]) / 2, (p[1] + p[3]) / 2])
        disp = np.linalg.norm(cc - pc) / 224.0
        cs = (c[2] - c[0]) * (c[3] - c[1])
        ps = (p[2] - p[0]) * (p[3] - p[1])
        sc = abs(cs - ps) / max(ps, 1)
        ms = disp + sc * 0.5

        if ms < 0.02:
            return 0.25
        elif ms < 0.05:
            return 0.30
        elif ms < 0.08:
            return 0.40
        elif ms < 0.12:
            return 0.50
        elif ms < 0.18:
            return 0.60
        elif ms < 0.25:
            return 0.65
        return 0.70

    def reset(self):
        self.roi_history.clear()
        self.smoothed_roi = None
        self.prev_roi = None
        self.alpha = self.base_alpha


# =============================
# Pose Extractor
# =============================
class PoseExtractor:
    """YOLO11 pose estimation for spatial cropping guidance."""

    def __init__(self, model_name="yolo11n-pose.pt", conf_threshold=0.3):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics not installed")
        print(f"🦴 Loading pose estimation model: {model_name}")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold

    def get_all_keypoints_for_visualization(self, frame):
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        if len(results) == 0 or results[0].keypoints is None:
            return []
        all_kpts = results[0].keypoints.data.cpu().numpy()
        return [k for k in all_kpts if np.sum(k[:, 2] > 0.3) >= 5]
