import cv2
from ultralytics import YOLO
from collections import deque
import numpy as np
import os
import glob
import shutil
import time
from collections import Counter

# ===== ENHANCED CONFIG =====
INPUT_FOLDER = "input_videos"
OUTPUT_FOLDER = "output_videos"
MIN_PEOPLE_REQUIRED = 2
MAX_PEOPLE = 3
PEOPLE_SAMPLE_FRAMES = 30
STICKY_FRAMES = 15
SMOOTHING_WINDOW = 8
CALIBRATION_FRAMES = 30
PADDING_COLOR = (0, 0, 0)
BOX_EXPANSION = 0.50
ACTION_LOCK_FRAMES = 60
MAX_MISSING_FRAMES = 30
OVERLAP_MARGIN = 20

# ROI Detection Settings
USE_ROI_DETECTION = True  # Enable ROI-based action detection
USE_POSE_FOR_ROI = True   # Use pose estimation for better ROI
ROI_CONFIDENCE_THRESHOLD = 0.15  # Lower threshold for pose detection
MIN_POSE_KEYPOINTS = 2    # Minimum keypoints to consider a pose
ROI_SMOOTHING = True      # Smooth ROI over time
ROI_SMOOTH_WINDOW = 5     # Smoothing window size

# Pose Settings (only if USE_POSE_FOR_ROI is True)
USE_POSE_ESTIMATION = True
USE_POSE_CENTERING = False  # Disable this when using ROI detection
POSE_CONFIDENCE_THRESHOLD = 0.3
MIN_EXPANSION = 0.50
MAX_EXPANSION = 0.85
# ===========================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def calculate_iou(box1, box2):
    """Calculate Intersection over Union"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    return intersection / (area1 + area2 - intersection)

def analyze_pose_activity(keypoints, box):
    """
    Analyze pose keypoints to determine activity level.
    Returns a score from 0-1 indicating how "active" the pose is.
    """
    if keypoints is None or len(keypoints) == 0:
        return 0.0
    
    activity_score = 0.0
    
    # YOLO pose keypoints: 0-nose, 1-left_eye, 2-right_eye, 3-left_ear, 4-right_ear,
    # 5-left_shoulder, 6-right_shoulder, 7-left_elbow, 8-right_elbow,
    # 9-left_wrist, 10-right_wrist, 11-left_hip, 12-right_hip,
    # 13-left_knee, 14-right_knee, 15-left_ankle, 16-right_ankle
    
    # Get keypoints with proper numpy array handling
    left_shoulder = keypoints[5] if len(keypoints) > 5 else None
    right_shoulder = keypoints[6] if len(keypoints) > 6 else None
    left_wrist = keypoints[9] if len(keypoints) > 9 else None
    right_wrist = keypoints[10] if len(keypoints) > 10 else None
    
    box_x1, box_y1, box_x2, box_y2 = box
    box_width = box_x2 - box_x1
    box_height = box_y2 - box_y1
    
    # Check arms extension (away from body)
    arms_extended = 0
    
    # Check left shoulder and wrist
    if left_shoulder is not None and left_wrist is not None:
        # left_shoulder and left_wrist are numpy arrays [x, y, confidence]
        left_shoulder_conf = left_shoulder[2] if len(left_shoulder) > 2 else 0
        left_wrist_conf = left_wrist[2] if len(left_wrist) > 2 else 0
        
        if left_shoulder_conf > 0.3 and left_wrist_conf > 0.3:
            arm_span = abs(left_wrist[0] - left_shoulder[0])
            if arm_span > box_width * 0.3:
                arms_extended += 1
    
    # Check right shoulder and wrist
    if right_shoulder is not None and right_wrist is not None:
        right_shoulder_conf = right_shoulder[2] if len(right_shoulder) > 2 else 0
        right_wrist_conf = right_wrist[2] if len(right_wrist) > 2 else 0
        
        if right_shoulder_conf > 0.3 and right_wrist_conf > 0.3:
            arm_span = abs(right_wrist[0] - right_shoulder[0])
            if arm_span > box_width * 0.3:
                arms_extended += 1
    
    # Check arms raised (above shoulders)
    arms_raised = 0
    
    if left_shoulder is not None and left_wrist is not None:
        left_shoulder_conf = left_shoulder[2] if len(left_shoulder) > 2 else 0
        left_wrist_conf = left_wrist[2] if len(left_wrist) > 2 else 0
        
        if left_shoulder_conf > 0.3 and left_wrist_conf > 0.3:
            if left_wrist[1] < left_shoulder[1] - box_height * 0.1:
                arms_raised += 1
    
    if right_shoulder is not None and right_wrist is not None:
        right_shoulder_conf = right_shoulder[2] if len(right_shoulder) > 2 else 0
        right_wrist_conf = right_wrist[2] if len(right_wrist) > 2 else 0
        
        if right_shoulder_conf > 0.3 and right_wrist_conf > 0.3:
            if right_wrist[1] < right_shoulder[1] - box_height * 0.1:
                arms_raised += 1
    
    # Combine scores
    activity_score = (
        (arms_extended / 2.0) * 0.4 +
        (arms_raised / 2.0) * 0.3 +
        0.3  # Base activity score
    )
    
    return min(activity_score, 1.0)

def get_pose_center_target(keypoints, box):
    """
    Calculate optimal crop center based on pose keypoints.
    """
    if keypoints is None or len(keypoints) < 17:
        return None
    
    box_x1, box_y1, box_x2, box_y2 = box
    box_width = box_x2 - box_x1
    box_height = box_y2 - box_y1
    
    # Get reliable keypoints
    reliable_kps = []
    for i in range(17):
        if keypoints[i][2] > POSE_CONFIDENCE_THRESHOLD:
            reliable_kps.append(keypoints[i])
    
    if len(reliable_kps) < MIN_POSE_KEYPOINTS:
        return None
    
    # Calculate center of all reliable keypoints
    kp_center_x = np.mean([kp[0] for kp in reliable_kps])
    kp_center_y = np.mean([kp[1] for kp in reliable_kps])
    
    # Weight important keypoints higher
    important_indices = [5, 6, 11, 12, 0]  # shoulders, hips, nose
    important_kps = []
    
    for idx in important_indices:
        if idx < len(keypoints) and keypoints[idx][2] > POSE_CONFIDENCE_THRESHOLD:
            important_kps.append(keypoints[idx])
    
    if important_kps:
        important_center_x = np.mean([kp[0] for kp in important_kps])
        important_center_y = np.mean([kp[1] for kp in important_kps])
        target_x = 0.7 * important_center_x + 0.3 * kp_center_x
        target_y = 0.7 * important_center_y + 0.3 * kp_center_y
    else:
        target_x = kp_center_x
        target_y = kp_center_y
    
    # Constrain within box
    target_x = max(box_x1 + box_width * 0.2, min(box_x2 - box_width * 0.2, target_x))
    target_y = max(box_y1 + box_height * 0.2, min(box_y2 - box_height * 0.2, target_y))
    
    return (target_x, target_y)

class ROIDetector:
    """
    ROI Detector that focuses on action regions using pose and motion analysis.
    """
    def __init__(self, debug=False):
        self.debug = debug
        self.prev_poses = None
        self.roi_history = deque(maxlen=ROI_SMOOTH_WINDOW)
        
    def detect_action_roi(self, frame, person_boxes, pose_model=None, max_people=2):
        """
        Detect ROI where action is happening.
        Returns (roi_box, focus_region)
        """
        h, w = frame.shape[:2]
        
        # If no person boxes, return None
        if not person_boxes or len(person_boxes) == 0:
            if self.debug:
                print("   ⚠️  No person boxes -> no ROI")
            return None, 'full_body'
        
        # Get poses if pose model is available
        current_poses = []
        if pose_model and USE_POSE_FOR_ROI:
            current_poses = self._get_matched_poses(frame, person_boxes, pose_model, max_people)
        
        # If poses detected, use pose-based ROI
        if len(current_poses) > 0:
            roi = self._get_pose_based_roi(current_poses, person_boxes, w, h)
            focus_region = self._determine_focus_region(current_poses)
        else:
            # Fallback: merge person boxes
            roi = self._merge_boxes(person_boxes)
            focus_region = 'full_body'
            
        # Smooth ROI if enabled
        if ROI_SMOOTHING and roi:
            self.roi_history.append(roi)
            if len(self.roi_history) >= 3:
                roi = self._smooth_roi()
        
        if self.debug:
            print(f"   ROI: {roi}, Focus: {focus_region}, Poses: {len(current_poses)}")
        
        return roi, focus_region
    
    def _get_matched_poses(self, frame, person_boxes, pose_model, max_poses):
        """Get poses only for detected people with low thresholds"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_model.predict(frame_rgb, conf=ROI_CONFIDENCE_THRESHOLD, verbose=False)
        
        if len(results) == 0 or results[0].keypoints is None:
            return []
        
        all_keypoints = results[0].keypoints.data.cpu().numpy()
        matched_poses = []
        
        for action_box in person_boxes[:max_poses]:
            ax1, ay1, ax2, ay2 = action_box
            
            best_match_idx = None
            best_match_score = 0
            
            for idx, kpts in enumerate(all_keypoints):
                # Very permissive: accept poses with just 2 visible keypoints
                if np.sum(kpts[:, 2] > ROI_CONFIDENCE_THRESHOLD) >= MIN_POSE_KEYPOINTS:
                    visible_kpts = kpts[kpts[:, 2] > ROI_CONFIDENCE_THRESHOLD]
                    pose_center = visible_kpts[:, :2].mean(axis=0)
                    
                    # Check if pose center is within person box
                    if ax1 <= pose_center[0] <= ax2 and ay1 <= pose_center[1] <= ay2:
                        score = len(visible_kpts)
                        if score > best_match_score:
                            best_match_score = score
                            best_match_idx = idx
            
            if best_match_idx is not None:
                matched_poses.append(all_keypoints[best_match_idx])
                if len(matched_poses) >= max_poses:
                    break
        
        return matched_poses
    
    def _get_pose_based_roi(self, poses, person_boxes, frame_width, frame_height):
        """Get ROI based on pose keypoints"""
        all_points = []
        
        for pose in poses:
            # Use all keypoints for ROI
            for idx in range(17):
                if pose[idx, 2] > ROI_CONFIDENCE_THRESHOLD:
                    all_points.append(pose[idx, :2])
        
        if len(all_points) == 0:
            # Fallback to person boxes
            return self._merge_boxes(person_boxes)
        
        all_points = np.array(all_points)
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        
        # Add padding based on pose spread
        width = x_max - x_min
        height = y_max - y_min
        padding_x = width * 0.4
        padding_y = height * 0.4
        
        x1 = max(0, int(x_min - padding_x))
        y1 = max(0, int(y_min - padding_y))
        x2 = min(frame_width, int(x_max + padding_x))
        y2 = min(frame_height, int(y_max + padding_y))
        
        # Ensure minimum size
        min_width = frame_width * 0.2
        min_height = frame_height * 0.3
        
        if (x2 - x1) < min_width:
            center_x = (x1 + x2) // 2
            x1 = max(0, int(center_x - min_width // 2))
            x2 = min(frame_width, int(center_x + min_width // 2))
        
        if (y2 - y1) < min_height:
            center_y = (y1 + y2) // 2
            y1 = max(0, int(center_y - min_height // 2))
            y2 = min(frame_height, int(center_y + min_height // 2))
        
        return (x1, y1, x2, y2)
    
    def _determine_focus_region(self, poses):
        """Determine focus region based on visible keypoints"""
        if len(poses) == 0:
            return 'full_body'
        
        upper_count = 0
        lower_count = 0
        
        for pose in poses:
            # Count upper body keypoints (0-10)
            for idx in range(0, 11):
                if pose[idx, 2] > ROI_CONFIDENCE_THRESHOLD:
                    upper_count += 1
            
            # Count lower body keypoints (11-16)
            for idx in range(11, 17):
                if pose[idx, 2] > ROI_CONFIDENCE_THRESHOLD:
                    lower_count += 1
        
        if upper_count > lower_count * 1.5:
            return 'upper_body'
        elif lower_count > upper_count * 1.5:
            return 'lower_body'
        else:
            return 'full_body'
    
    def _merge_boxes(self, boxes):
        """Merge multiple boxes into one"""
        if len(boxes) == 0:
            return None
        if len(boxes) == 1:
            return boxes[0]
        
        x1_min = min(b[0] for b in boxes)
        y1_min = min(b[1] for b in boxes)
        x2_max = max(b[2] for b in boxes)
        y2_max = max(b[3] for b in boxes)
        
        # Add some padding around merged boxes
        width = x2_max - x1_min
        height = y2_max - y1_min
        padding_x = width * 0.1
        padding_y = height * 0.1
        
        return (
            max(0, int(x1_min - padding_x)),
            max(0, int(y1_min - padding_y)),
            int(x2_max + padding_x),
            int(y2_max + padding_y)
        )
    
    def _smooth_roi(self):
        """Smooth ROI over history"""
        if len(self.roi_history) == 0:
            return None
        
        # Calculate median box
        x1s = [b[0] for b in self.roi_history]
        y1s = [b[1] for b in self.roi_history]
        x2s = [b[2] for b in self.roi_history]
        y2s = [b[3] for b in self.roi_history]
        
        return (
            int(np.median(x1s)),
            int(np.median(y1s)),
            int(np.median(x2s)),
            int(np.median(y2s))
        )
    
    def reset(self):
        """Reset detector state"""
        self.prev_poses = None
        self.roi_history.clear()

def calculate_motion_expansion(current_box, history, base_margin=0.50, pose_activity=0.0):
    """
    Calculate adaptive expansion based on movement history AND pose activity.
    """
    if not history or len(history) < 3:
        return max(base_margin, MIN_EXPANSION + (pose_activity * 0.3))
    
    recent_boxes = list(history)[-10:]
    
    if len(recent_boxes) < 2:
        return max(base_margin, MIN_EXPANSION + (pose_activity * 0.3))
    
    # Calculate max displacement
    max_dx = 0
    max_dy = 0
    
    for i in range(len(recent_boxes) - 1):
        x1_curr, y1_curr, x2_curr, y2_curr = recent_boxes[i]
        x1_next, y1_next, x2_next, y2_next = recent_boxes[i + 1]
        
        cx_curr = (x1_curr + x2_curr) / 2
        cy_curr = (y1_curr + y2_curr) / 2
        cx_next = (x1_next + x2_next) / 2
        cy_next = (y1_next + y2_next) / 2
        
        dx = abs(cx_next - cx_curr)
        dy = abs(cy_next - cy_curr)
        
        max_dx = max(max_dx, dx)
        max_dy = max(max_dy, dy)
    
    # Check box size changes
    box_widths = [b[2] - b[0] for b in recent_boxes]
    box_heights = [b[3] - b[1] for b in recent_boxes]
    
    width_variance = max(box_widths) - min(box_widths)
    height_variance = max(box_heights) - min(box_heights)
    
    # Current box size
    curr_w = current_box[2] - current_box[0]
    curr_h = current_box[3] - current_box[1]
    
    # Calculate adaptive margin
    motion_factor_x = max_dx / max(curr_w, 1)
    motion_factor_y = max_dy / max(curr_h, 1)
    size_factor_x = width_variance / max(curr_w, 1)
    size_factor_y = height_variance / max(curr_h, 1)
    
    # Combine factors
    motion_factor = max(
        motion_factor_x,
        motion_factor_y,
        size_factor_x,
        size_factor_y
    )
    
    adaptive_margin = base_margin + (motion_factor * 2.5) + (pose_activity * 0.35)
    
    # Enforce minimum and maximum
    adaptive_margin = max(adaptive_margin, MIN_EXPANSION)
    adaptive_margin = min(adaptive_margin, MAX_EXPANSION)
    
    return adaptive_margin

def is_video_already_processed(video_path, output_folder):
    """Check if a video has already been processed"""
    filename = os.path.basename(video_path)
    base_name = os.path.splitext(filename)[0]
    
    original_in_output = os.path.join(output_folder, filename)
    if os.path.exists(original_in_output):
        return True, "original"
    
    two_crop_patterns = ["left", "right"]
    two_crop_exists = []
    for position in two_crop_patterns:
        crop_filename = f"{base_name}_cropped_{position}.mp4"
        crop_path = os.path.join(output_folder, crop_filename)
        if os.path.exists(crop_path):
            two_crop_exists.append(position)
    
    three_crop_patterns = ["left", "middle", "right"]
    three_crop_exists = []
    for position in three_crop_patterns:
        crop_filename = f"{base_name}_cropped_{position}.mp4"
        crop_path = os.path.join(output_folder, crop_filename)
        if os.path.exists(crop_path):
            three_crop_exists.append(position)
    
    if len(three_crop_exists) == 3:
        return True, "3-crop"
    elif len(two_crop_exists) == 2:
        return True, "2-crop"
    elif len(two_crop_exists) > 0 or len(three_crop_exists) > 0:
        return True, f"partial ({len(two_crop_exists) + len(three_crop_exists)} crops)"
    
    return False, None

def determine_crop_count(people_count):
    """Determine how many crops to create based on people count"""
    if people_count >= 3:
        return 3
    elif people_count == 2:
        return 2
    else:
        return 0

def get_crop_positions(crop_count):
    """Get position names for the given crop count"""
    if crop_count == 3:
        return ["left", "middle", "right"]
    elif crop_count == 2:
        return ["left", "right"]
    else:
        return []

def count_people_in_video(video_path, yolo_model, sample_frames=30):
    """Count people in video"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return 0
    
    frame_indices = []
    if total_frames <= sample_frames:
        frame_indices = list(range(total_frames))
    else:
        for i in range(0, sample_frames):
            pos = int((i / sample_frames) * total_frames)
            frame_indices.append(pos)
    
    people_counts = []
    
    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = yolo_model.predict(rgb, conf=0.4, classes=[0], verbose=False)
        
        frame_count = 0
        h, w = frame.shape[:2]
        frame_area = h * w
        
        for r in result:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf)
                
                box_w, box_h = x2 - x1, y2 - y1
                area = box_w * box_h
                
                if area / frame_area < 0.003:
                    continue
                
                aspect = box_w / max(box_h, 1)
                if aspect < 0.2 or aspect > 5:
                    continue
                
                frame_count += 1
        
        people_counts.append(frame_count)
        
        if idx % 10 == 0:
            print(f"  Frame {idx+1}/{len(frame_indices)}: {frame_count} people")
    
    cap.release()
    
    if not people_counts:
        return 0
    
    print(f"  Raw counts: {people_counts}")
    
    counts_array = np.array(people_counts)
    mean_count = np.mean(counts_array)
    median_count = np.median(counts_array)
    
    counter = Counter(people_counts)
    most_common = counter.most_common(3)
    
    print(f"  Statistics: mean={mean_count:.1f}, median={median_count}")
    print(f"  Most common: {most_common}")
    
    candidate_counts = [median_count]
    for count, freq in most_common:
        if freq >= len(people_counts) * 0.25:
            candidate_counts.append(count)
    
    final_count = int(max(candidate_counts))
    
    max_seen = max(people_counts)
    if max_seen >= 2 and final_count < 2:
        print(f"  ⚠️  Overriding: saw {max_seen} people in at least one frame")
        final_count = max_seen
    
    return final_count

def prevent_overlap(boxes, frame_width, margin=OVERLAP_MARGIN):
    """Adjust boxes to prevent overlap while maintaining left-middle-right order."""
    if not boxes or len(boxes) < 2:
        return boxes
    
    adjusted = []
    
    for i, box in enumerate(boxes):
        if box is None:
            adjusted.append(None)
            continue
            
        x1, y1, x2, y2 = box
        
        if i > 0 and adjusted[i-1] is not None:
            prev_x2 = adjusted[i-1][2]
            if x1 < prev_x2 + margin:
                shift = (prev_x2 + margin) - x1
                x1 += shift
                x2 += shift
        
        if i < len(boxes) - 1 and boxes[i+1] is not None:
            next_x1 = boxes[i+1][0]
            if x2 > next_x1 - margin:
                x2 = next_x1 - margin
        
        if x2 <= x1:
            x2 = x1 + 100
        
        x1 = max(0, x1)
        x2 = min(frame_width, x2)
        
        adjusted.append((int(x1), int(y1), int(x2), int(y2)))
    
    return adjusted

class MultiActionTracker:
    def __init__(self, max_actions=3):
        self.max_actions = max_actions
        self.locked_actions = [None] * max_actions
        self.actions_confirmed = [False] * max_actions
        self.histories = [deque(maxlen=30) for _ in range(max_actions)]
        self.confidences = [0] * max_actions
        self.missing_counters = [0] * max_actions

    def update(self, boxes, frame_shape, frame_idx, crop_count=3):
        """Update tracker with boxes, crop_count determines active slots"""
        h, w = frame_shape[:2]
        
        active_indices = []
        if crop_count == 3:
            active_indices = [0, 1, 2]
        elif crop_count == 2:
            active_indices = [0, 2]
        
        if len(boxes) > crop_count:
            boxes_sorted = sorted(boxes, key=lambda b: (b[0] + b[2]) / 2)
            
            if crop_count == 2:
                if len(boxes_sorted) >= 2:
                    boxes = [boxes_sorted[0], boxes_sorted[-1]]
                else:
                    boxes = boxes_sorted
            else:
                boxes = boxes_sorted[:crop_count]
        else:
            boxes_sorted = sorted(boxes, key=lambda b: (b[0] + b[2]) / 2)
            boxes = boxes_sorted
        
        boxes = prevent_overlap(boxes, w)
        
        for i, action_idx in enumerate(active_indices):
            if i < len(boxes):
                box = boxes[i]
                box = self._fine_tune_box(box, action_idx, (h, w))
                self.histories[action_idx].append(box)
                self.confidences[action_idx] = min(self.confidences[action_idx] + 1, 10)
                self.missing_counters[action_idx] = 0
            else:
                self.missing_counters[action_idx] += 1
        
        for action_idx in active_indices:
            if (not self.actions_confirmed[action_idx] and 
                self.confidences[action_idx] >= 8 and 
                len(self.histories[action_idx]) >= 15):
                self.locked_actions[action_idx] = self._get_optimal_box(
                    self.histories[action_idx], action_idx, (h, w))
                self.actions_confirmed[action_idx] = True
                print(f"🎯 Locked Action {action_idx+1} at position {action_idx}")
        
        return self._get_current_regions(h, w, crop_count)

    def _fine_tune_box(self, box, action_idx, frame_shape):
        h, w = frame_shape
        x1, y1, x2, y2 = box
        
        if action_idx == 0:
            y_center = (y1 + y2) / 2
            ideal_y_center = h * 0.5
            y_adjust = (ideal_y_center - y_center) * 0.2
            x_adjust = -5
        elif action_idx == 1:
            y_center = (y1 + y2) / 2
            ideal_y_center = h * 0.5
            y_adjust = (ideal_y_center - y_center) * 0.2
            x_center = (x1 + x2) / 2
            ideal_x_center = w * 0.5
            x_adjust = (ideal_x_center - x_center) * 0.1
        else:
            y_center = (y1 + y2) / 2
            ideal_y_center = h * 0.5
            y_adjust = (ideal_y_center - y_center) * 0.2
            x_adjust = 5
        
        x1 = int(max(0, x1 + x_adjust))
        y1 = int(max(0, y1 + y_adjust))
        x2 = int(min(w, x2 + x_adjust))
        y2 = int(min(h, y2 + y_adjust))
        
        return (x1, y1, x2, y2)

    def _get_optimal_box(self, history, action_idx, frame_shape):
        h, w = frame_shape
        
        median_box = self._get_median_box(history)
        if median_box is None:
            return None
        
        x1, y1, x2, y2 = median_box
        box_h = y2 - y1
        box_w = x2 - x1
        
        ideal_y = max(0, (h - box_h) // 2)
        y1 = int(ideal_y)
        y2 = int(y1 + box_h)
        
        if action_idx == 0:
            if x1 < w * 0.1:
                x1 = int(w * 0.05)
                x2 = int(x1 + box_w)
        elif action_idx == 1:
            ideal_x = max(0, (w - box_w) // 2)
            x1 = int(ideal_x)
            x2 = int(x1 + box_w)
        else:
            if x2 > w * 0.9:
                x2 = int(w * 0.95)
                x1 = int(x2 - box_w)
        
        return (int(x1), int(y1), int(x2), int(y2))

    def _get_median_box(self, boxes):
        if not boxes:
            return None
        x1s = [b[0] for b in boxes]
        y1s = [b[1] for b in boxes]
        x2s = [b[2] for b in boxes]
        y2s = [b[3] for b in boxes]
        return (
            int(np.median(x1s)),
            int(np.median(y1s)),
            int(np.median(x2s)),
            int(np.median(y2s))
        )

    def _get_current_regions(self, h, w, crop_count=3):
        """Get current regions based on crop count"""
        regions = []
        
        if crop_count == 3:
            indices = [0, 1, 2]
        elif crop_count == 2:
            indices = [0, 2]
        else:
            return []
        
        for action_idx in indices:
            if self.actions_confirmed[action_idx] and self.locked_actions[action_idx]:
                regions.append(self.locked_actions[action_idx])
            elif self.histories[action_idx]:
                regions.append(self._get_median_box(self.histories[action_idx]))
            else:
                regions.append(None)
        
        return prevent_overlap(regions, w)

class MultiActionDetector:
    def __init__(self, max_actions=3, use_roi_detection=True):
        self.max_actions = max_actions
        self.tracker = MultiActionTracker(max_actions)
        self.last_good_actions = [None] * max_actions
        self.missing_counters = [0] * max_actions
        self.frame_idx = 0
        self.motion_histories = [deque(maxlen=10) for _ in range(max_actions)]
        self.pose_activities = [0.0] * max_actions
        self.use_roi_detection = use_roi_detection
        
        # Initialize ROI detector if enabled
        self.roi_detector = ROIDetector(debug=False) if use_roi_detection else None

    def detect(self, frame, detector, crop_count=3, pose_model=None, roi_detector=None):
        """Detect actions with ROI-based focusing"""
        self.frame_idx += 1
        h, w = frame.shape[:2]
        
        # Get detection boxes
        result = detector.predict(frame, conf=0.5, classes=[0], verbose=False)
        boxes = []
        for r in result:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf)
                box_w, box_h = x2 - x1, y2 - y1
                area = box_w * box_h
                frame_area = h * w
                min_ratio = 0.02
                max_ratio = 0.5
                aspect = box_w / box_h if box_h > 0 else 1
                
                if (min_ratio < area / frame_area < max_ratio and 
                    0.5 < aspect < 2.0):
                    boxes.append((x1, y1, x2, y2))
        
        # Get ROI if ROI detection is enabled
        action_roi = None
        if self.use_roi_detection and self.roi_detector and len(boxes) > 0:
            action_roi, focus_region = self.roi_detector.detect_action_roi(
                frame, boxes, pose_model, max_people=crop_count
            )
            
            # If ROI is detected, use it to filter and adjust boxes
            if action_roi:
                # Filter boxes to only include those that overlap with ROI
                filtered_boxes = []
                for box in boxes:
                    if calculate_iou(box, action_roi) > 0.1:  # At least 10% overlap
                        filtered_boxes.append(box)
                
                if len(filtered_boxes) > 0:
                    boxes = filtered_boxes
        
        # Update tracker with boxes
        actions = self.tracker.update(boxes, (h, w), self.frame_idx, crop_count)
        
        # If ROI is available and we have actions, adjust actions to be within ROI
        if action_roi and len(actions) > 0:
            adjusted_actions = []
            roi_x1, roi_y1, roi_x2, roi_y2 = action_roi
            
            for i, action in enumerate(actions):
                if action is None:
                    adjusted_actions.append(None)
                    continue
                
                ax1, ay1, ax2, ay2 = action
                
                # Constrain action box within ROI
                ax1 = max(roi_x1, ax1)
                ay1 = max(roi_y1, ay1)
                ax2 = min(roi_x2, ax2)
                ay2 = min(roi_y2, ay2)
                
                # Ensure valid box
                if ax2 > ax1 and ay2 > ay1:
                    adjusted_actions.append((ax1, ay1, ax2, ay2))
                else:
                    # If box becomes invalid, use ROI center
                    center_x = (roi_x1 + roi_x2) // 2
                    center_y = (roi_y1 + roi_y2) // 2
                    size = min(roi_x2 - roi_x1, roi_y2 - roi_y1) // 2
                    adjusted_actions.append((
                        max(0, center_x - size),
                        max(0, center_y - size),
                        min(w, center_x + size),
                        min(h, center_y + size)
                    ))
            
            actions = adjusted_actions
        
        # Get pose data if available (for activity scoring)
        pose_data = {}
        if pose_model and USE_POSE_ESTIMATION:
            try:
                pose_result = pose_model.predict(frame, conf=0.3, verbose=False)
                for pr in pose_result:
                    if hasattr(pr, 'keypoints') and pr.keypoints is not None:
                        for idx, (kp, box) in enumerate(zip(pr.keypoints.data, pr.boxes.xyxy)):
                            x1, y1, x2, y2 = map(int, box)
                            pose_data[(x1, y1, x2, y2)] = kp.cpu().numpy()
            except Exception as e:
                print(f"⚠️ Pose estimation failed: {e}")
        
        # Calculate pose activity for each action
        for i, action in enumerate(actions):
            if action is not None:
                # Find matching pose data
                activity = 0.0
                if pose_data:
                    best_match = None
                    best_overlap = 0
                    
                    for pose_box, keypoints in pose_data.items():
                        overlap = calculate_iou(action, pose_box)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_match = keypoints
                    
                    if best_match is not None and best_overlap > 0.3:
                        activity = analyze_pose_activity(best_match, action)
                
                self.pose_activities[i] = activity
                self.motion_histories[i].append(action)
                self.last_good_actions[i] = action
                self.missing_counters[i] = 0
            else:
                self.missing_counters[i] += 1
        
        # Fill in missing actions with fallbacks
        final_actions = []
        active_indices = [0, 1, 2] if crop_count == 3 else [0, 2]
        
        for i, action_idx in enumerate(active_indices):
            if i < len(actions) and actions[i] is not None:
                final_actions.append(actions[i])
            else:
                final_actions.append(self._get_fallback(action_idx, (h, w)))
                self.pose_activities[action_idx] = 0.0
        
        return prevent_overlap(final_actions, w)

    def _get_fallback(self, action_idx, frame_shape):
        h, w = frame_shape
        
        if (self.last_good_actions[action_idx] is not None and 
            self.missing_counters[action_idx] < MAX_MISSING_FRAMES):
            return self.last_good_actions[action_idx]
        
        default_size = int(min(h, w) // 3)
        
        if action_idx == 0:
            return (int(w//8), int(h//2 - default_size//2),
                    int(w//8 + default_size), int(h//2 + default_size//2))
        elif action_idx == 1:
            return (int(w//2 - default_size//2), int(h//2 - default_size//2),
                    int(w//2 + default_size//2), int(h//2 + default_size//2))
        else:
            return (int(w*7//8 - default_size), int(h//2 - default_size//2),
                    int(w*7//8), int(h//2 + default_size//2))

class MultiSmoother:
    def __init__(self, num_actions=3, window_size=8):
        self.num_actions = num_actions
        self.histories = [deque(maxlen=window_size) for _ in range(num_actions)]

    def smooth(self, *boxes):
        smoothed = []
        for i in range(self.num_actions):
            if i < len(boxes) and boxes[i] is not None:
                box = (int(boxes[i][0]), int(boxes[i][1]), 
                       int(boxes[i][2]), int(boxes[i][3]))
                self.histories[i].append(box)
                
                if len(self.histories[i]) >= 3:
                    smoothed.append(self._median_box(self.histories[i]))
                elif self.histories[i]:
                    smoothed.append(self.histories[i][-1])
                else:
                    smoothed.append(boxes[i] if i < len(boxes) else None)
        
        if smoothed:
            max_x = max([b[2] for b in smoothed if b is not None], default=1920)
            smoothed = prevent_overlap(smoothed, max_x)
        
        return smoothed

    def _median_box(self, history):
        if not history:
            return None
        boxes = list(history)
        x1s = [b[0] for b in boxes]
        y1s = [b[1] for b in boxes]
        x2s = [b[2] for b in boxes]
        y2s = [b[3] for b in boxes]
        return (
            int(np.median(x1s)),
            int(np.median(y1s)),
            int(np.median(x2s)),
            int(np.median(y2s))
        )

def safe_crop(frame, box, action_idx=0, default_scale=0.25):
    if box is None:
        h, w = frame.shape[:2]
        size = int(min(h, w) * default_scale)
        if action_idx == 0:
            x1 = int(w//4 - size//2)
        elif action_idx == 1:
            x1 = int(w//2 - size//2)
        else:
            x1 = int(w*3//4 - size//2)
        y1 = int(h//2 - size//2)
        box = (x1, y1, x1 + size, y1 + size)
    
    x1, y1, x2, y2 = map(int, box)
    h, w = frame.shape[:2]
    
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    min_size = 150
    if (x2 - x1) < min_size:
        cx = (x1 + x2) // 2
        x1 = int(max(0, cx - min_size//2))
        x2 = int(min(w, cx + min_size//2))
    if (y2 - y1) < min_size:
        cy = (y1 + y2) // 2
        y1 = int(max(0, cy - min_size//2))
        y2 = int(min(h, cy + min_size//2))
    
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    if x2 <= x1 or y2 <= y1:
        size = int(min(h, w) * default_scale)
        x1 = int(w//2 - size//2)
        y1 = int(h//2 - size//2)
        return frame[y1:y1+size, x1:x1+size]
    
    return frame[y1:y2, x1:x2]

def expand_box(box, frame_shape, action_idx=0, margin=0.2):
    if box is None:
        return None
    
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = map(int, box)
    bw, bh = x2 - x1, y2 - y1
    
    if action_idx == 1:
        ew = int(bw * margin * 0.9)
        eh = int(bh * margin)
    else:
        ew = int(bw * margin)
        eh = int(bh * margin)
    
    max_expand = min(bw, bh) * 0.45
    ew = min(ew, int(max_expand))
    eh = min(eh, int(max_expand))
    
    x1 = int(max(0, x1 - ew))
    y1 = int(max(0, y1 - eh))
    x2 = int(min(w, x2 + ew))
    y2 = int(min(h, y2 + eh))
    
    return (x1, y1, x2, y2)

def pad_to_size(crop, target_size, color=(0, 0, 0)):
    target_w, target_h = target_size
    
    if crop.size == 0:
        return np.full((target_h, target_w, 3), color, dtype=np.uint8)
    
    h, w = crop.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(crop, (new_w, new_h))
    
    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    
    return canvas

def get_multi_calibration(video_path, detector, num_frames=40, crop_count=3):
    """Calibration that adapts to crop count"""
    cap = cv2.VideoCapture(video_path)
    all_sizes = []
    tracker = MultiActionTracker(max_actions=3)
    
    for idx in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = detector.predict(rgb, conf=0.5, classes=[0], verbose=False)
        
        boxes = []
        for r in result:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                boxes.append((x1, y1, x2, y2))
        
        actions = tracker.update(boxes, frame.shape, idx, crop_count)
        
        for i, action in enumerate(actions):
            if action:
                w, h = action[2]-action[0], action[3]-action[1]
                all_sizes.append((w, h))
    
    cap.release()
    
    if all_sizes:
        target_w = int(np.percentile([w for w, h in all_sizes], 80))
        target_h = int(np.percentile([h for w, h in all_sizes], 80))
        
        aspect = target_w / target_h
        if aspect > 1.6:
            target_w = int(target_h * 1.4)
        elif aspect < 0.7:
            target_h = int(target_w * 1.4)
        
        target_w = max(target_w, 400)
        target_h = max(target_h, 400)
        return (target_w, target_h)
    
    return (480, 480)

def process_video_with_dynamic_crops(input_path, output_folder, yolo_model, crop_count):
    """Process video with dynamic number of crops (2 or 3) with ROI detection"""
    positions = get_crop_positions(crop_count)
    position_text = f"{crop_count}-crop ({' & '.join(positions)})"
    
    print(f"\n🎬 Processing {position_text} (ROI-based action detection): {os.path.basename(input_path)}")
    
    # Load pose model if ROI detection with pose is enabled
    pose_model = None
    if USE_ROI_DETECTION and USE_POSE_FOR_ROI:
        try:
            print("🧍 Loading pose estimation model for ROI detection...")
            pose_model = YOLO("yolo11n-pose.pt")
            print("✅ Pose model loaded for ROI detection")
        except Exception as e:
            print(f"⚠️ Could not load pose model: {e}")
            print("   Continuing without pose-based ROI detection")
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    
    output_files = []
    for position in positions:
        output_name = f"{base_name}_cropped_{position}.mp4"
        output_path = os.path.join(output_folder, output_name)
        output_files.append(output_path)
    
    print(f"🔍 Getting calibration for {crop_count} actions...")
    TARGET_SIZE = get_multi_calibration(input_path, yolo_model, CALIBRATION_FRAMES, crop_count)
    print(f"✅ Target size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use ROI-based detector
    detector = MultiActionDetector(max_actions=MAX_PEOPLE, use_roi_detection=USE_ROI_DETECTION)
    smoother = MultiSmoother(num_actions=MAX_PEOPLE, window_size=SMOOTHING_WINDOW)
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writers = []
    for output_path in output_files:
        writer = cv2.VideoWriter(output_path, fourcc, fps, TARGET_SIZE)
        writers.append(writer)
    
    frame_count = 0
    print(f"📹 Processing with synchronized {position_text} (ROI detection: {'ON' if USE_ROI_DETECTION else 'OFF'})...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect actions with ROI-based detector
        actions = detector.detect(rgb, yolo_model, crop_count, pose_model, detector.roi_detector)
        
        expanded_actions = []
        action_indices = []
        if crop_count == 3:
            action_indices = [0, 1, 2]
        elif crop_count == 2:
            action_indices = [0, 2]
        
        for i, action_idx in enumerate(action_indices):
            if i < len(actions) and actions[i] is not None:
                # Get pose activity for this action
                pose_activity = detector.pose_activities[action_idx]
                
                # Calculate adaptive margin
                adaptive_margin = calculate_motion_expansion(
                    actions[i], 
                    detector.motion_histories[action_idx],
                    base_margin=BOX_EXPANSION,
                    pose_activity=pose_activity
                )
                
                # Debug output
                if frame_count % 100 == 0 and i == 0:
                    print(f"   Frame {frame_count}: margin={adaptive_margin:.2f}, pose_activity={pose_activity:.2f}")
                
                expanded = expand_box(
                    actions[i], 
                    frame.shape, 
                    action_idx=action_idx, 
                    margin=adaptive_margin
                )
                expanded_actions.append(expanded)
        
        h, w = frame.shape[:2]
        expanded_actions = prevent_overlap(expanded_actions, w)
        
        smoothed_actions = smoother.smooth(*expanded_actions)
        
        for i in range(crop_count):
            if i < len(smoothed_actions):
                action_idx = action_indices[i] if i < len(action_indices) else i
                crop = safe_crop(frame, smoothed_actions[i], action_idx=action_idx, default_scale=0.25)
                padded = pad_to_size(crop, TARGET_SIZE, PADDING_COLOR)
                writers[i].write(padded)
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f" Frame {frame_count}/{total_frames}")
    
    cap.release()
    for writer in writers:
        writer.release()
    
    print(f"✅ {position_text} processing complete for {os.path.basename(input_path)}!")
    print(f" Frames processed: {frame_count}")
    for i, output_path in enumerate(output_files):
        position = positions[i]
        print(f" Output {position}: {os.path.basename(output_path)}")
    
    return output_files

def copy_video_to_output(input_path, output_folder):
    filename = os.path.basename(input_path)
    output_path = os.path.join(output_folder, filename)
    
    try:
        shutil.copy2(input_path, output_path)
        print(f"📋 Copied: {filename} (no processing)")
        return output_path
    except Exception as e:
        print(f"❌ Error copying {filename}: {e}")
        return None

def main():
    print("🚀 Starting SMART batch video processing (ROI-based action detection)...")
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Config: base_expansion={BOX_EXPANSION}, min={MIN_EXPANSION}, max={MAX_EXPANSION}")
    print(f"ROI detection: {'ENABLED' if USE_ROI_DETECTION else 'DISABLED'}")
    print(f"Pose for ROI: {'ENABLED' if USE_POSE_FOR_ROI else 'DISABLED'}")
    print(f"ROI smoothing: {'ENABLED' if ROI_SMOOTHING else 'DISABLED'}")
    
    print("📦 Loading YOLO model...")
    yolo = YOLO("yolo11n.pt")
    
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv', '*.wmv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    
    if not video_files:
        print(f"❌ No video files found in {INPUT_FOLDER}")
        return
    
    print(f"📁 Found {len(video_files)} video(s) to process\n")
    
    all_handled_videos = []
    skipped_videos = []
    
    for i, video_path in enumerate(video_files, 1):
        filename = os.path.basename(video_path)
        
        already_processed, processing_type = is_video_already_processed(video_path, OUTPUT_FOLDER)
        if already_processed:
            print(f"⏭️ [{i}/{len(video_files)}] Skipping {filename} (already processed as {processing_type})")
            skipped_videos.append((filename, processing_type))
            continue
        
        print(f"🔍 [{i}/{len(video_files)}] Investigating {filename}...")
        
        start_time = time.time()
        people_count = count_people_in_video(video_path, yolo, PEOPLE_SAMPLE_FRAMES)
        elapsed = time.time() - start_time
        
        print(f"   👥 Detected {people_count} person(s) in {elapsed:.1f}s")
        
        crop_count = determine_crop_count(people_count)
        
        if crop_count >= MIN_PEOPLE_REQUIRED:
            if crop_count == 3:
                print(f"   ✅ Processing {filename} with 3-crop (3+ people detected)")
            elif crop_count == 2:
                print(f"   ✅ Processing {filename} with 2-crop (2 people detected)")
            
            process_video_with_dynamic_crops(video_path, OUTPUT_FOLDER, yolo, crop_count)
        else:
            print(f"   📋 Copying {filename} as-is ({people_count} person(s))")
            copy_video_to_output(video_path, OUTPUT_FOLDER)
        
        all_handled_videos.append(video_path)
    
    print("\n" + "="*60)
    print("📊 PROCESSING SUMMARY")
    print("="*60)
    
    if skipped_videos:
        print(f"⏭️ Skipped {len(skipped_videos)} video(s):")
        for filename, processing_type in skipped_videos:
            print(f"   - {filename} ({processing_type})")
    
    if all_handled_videos:
        print(f"✅ Processed {len(all_handled_videos)} video(s)")
    
    if all_handled_videos:
        print("\n" + "="*50)
        response = input("❓ Do you want to delete the original videos? (y/n): ").strip().lower()
        
        if response in ['y', 'yes']:
            deleted_count = 0
            for original_path in all_handled_videos:
                try:
                    os.remove(original_path)
                    print(f"🗑️ Deleted: {os.path.basename(original_path)}")
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ Error deleting {original_path}: {e}")
            print(f"\n✅ Deleted {deleted_count} original video(s)")
        else:
            print("📁 Original videos kept intact")
    else:
        print("📁 No new videos were processed (all were already done)")
    
    print("\n🎉 Batch processing complete!")

if __name__ == "__main__":
    main()