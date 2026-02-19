import cv2
from ultralytics import YOLO
from collections import deque
from pathlib import Path
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
PEOPLE_SAMPLE_FRAMES = 40

# IMPROVED: Lower confidence thresholds for better partial person detection
PERSON_DETECTION_CONF = 0.10  # Lowered from 0.4 - catches partial people
PERSON_DETECTION_CONF_ZONES = 0.12  # Even lower for zone analysis
MIN_PERSON_AREA_RATIO = 0.0005  # Lowered from 0.003 - allows smaller people

STICKY_FRAMES = 30
SMOOTHING_WINDOW = 15
CALIBRATION_FRAMES = 30
PADDING_COLOR = (0, 0, 0)
BOX_EXPANSION = 0.20  # For good detections
FALLBACK_BOX_EXPANSION = 0.50  # For fallback boxes - SEPARATE!
ACTION_LOCK_FRAMES = 90
MAX_MISSING_FRAMES = 45
OVERLAP_MARGIN = 15

# Minimum box dimensions (as percentage of frame)
MIN_BOX_WIDTH_RATIO = 0.35
MIN_BOX_HEIGHT_RATIO = 0.40
MAX_ASPECT_RATIO = 1.4
MIN_ASPECT_RATIO = 0.6

# ROI Detection Settings
USE_ROI_DETECTION = True
USE_POSE_FOR_ROI = True
ROI_CONFIDENCE_THRESHOLD = 0.15
MIN_POSE_KEYPOINTS = 2
ROI_SMOOTHING = True
ROI_SMOOTH_WINDOW = 8

# Pose Settings
USE_POSE_ESTIMATION = True
USE_POSE_CENTERING = False
POSE_CONFIDENCE_THRESHOLD = 0.3
MIN_EXPANSION = 0.1
MAX_EXPANSION = 0.15

PERSON_DETECTION_CONF_TRACKING = 0.12 # can affect window size!

# ===== DEBUG VISUALIZATION CONFIG =====
DEBUG_MODE = True  # Set to True to enable debug visualization
DEBUG_SAMPLES = 4  # Number of sample frames to visualize
DEBUG_OUTPUT_FOLDER = "debug_visualizations"  # Folder for debug images

# Video debug settings
DEBUG_CREATE_VIDEOS = True  # Create full debug videos
DEBUG_VIDEO_FOLDER = "debug_videos"  # Folder for debug videos
DEBUG_VIDEO_SIDE_BY_SIDE = False  # Show original + debug side-by-side
DEBUG_SHOW_METRICS = True  # Show tracking metrics overlay

# ===== ENHANCED PEOPLE DETECTION =====
# Detect partial people and interaction zones
USE_PARTIAL_PERSON_DETECTION = True
PARTIAL_PERSON_MIN_AREA_RATIO = 0.0005  # Even smaller for legs-only, torsos, etc.
INTERACTION_ZONE_EXPANSION = 0.3  # Expand detection zones to catch nearby partial people
POSE_KEYPOINT_CLUSTER_DETECTION = True  # Detect people by keypoint clusters
MIN_KEYPOINT_CLUSTER_SIZE = 2  # Minimum keypoints to count as a person
KEYPOINT_CLUSTER_RADIUS = 120  # Pixels - how close keypoints must be

# People counting adjustment
PEOPLE_COUNT_CONFIDENCE_BOOST = True  # Use multiple detection methods
COMBINE_BBOX_AND_POSE_COUNTS = True  # Merge bbox and pose detections
ADJACENCY_BONUS = True  # If 2 detected, check if 3rd is likely nearby


# COHERENCE DETECTION SETTINGS
USE_COHERENCE_DETECTION = True
COHERENCE_THRESHOLD_HIGH = 0.6  # Above this = same action
COHERENCE_THRESHOLD_LOW = 0.4   # Below this = different actions
MIN_COHERENCE_SAMPLES = 5

# Jump Resistance Settings
MAX_JUMP_RATIO = 0.18           # Max jump = 18% of frame width per frame
JUMP_RESISTANCE_MIN_HISTORY = 4  # Enforce jump limits after 4 frames of history

# ===== POSE VALIDATION FOR LOW-CONFIDENCE DETECTIONS =====
# Below this confidence, a bbox detection MUST have matching pose keypoints
# to be counted as a real person. Above this, trust the detection as-is.
POSE_VALIDATION_CONF_THRESHOLD = 0.25
POSE_VALIDATION_MIN_KEYPOINTS = 1       # Minimum keypoints inside bbox to validate
POSE_VALIDATION_KEYPOINT_CONF = 0.15    # Minimum keypoint confidence for validation
POSE_VALIDATION_IOU_THRESHOLD = 0.15    # Min IoU between bbox and pose bbox to match


# ======================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


if DEBUG_MODE and DEBUG_OUTPUT_FOLDER:
    os.makedirs(DEBUG_OUTPUT_FOLDER, exist_ok=True)


# ===== POSE VALIDATION HELPER =====
def get_pose_keypoints_for_frame(rgb_frame, pose_model, conf=0.15):
    """
    Run pose estimation on a frame and return all detected keypoints + bboxes.
    Returns list of dicts: [{'keypoints': array, 'bbox': (x1,y1,x2,y2), 'num_visible': int}, ...]
    """
    if pose_model is None:
        return []

    try:
        pose_result = pose_model.predict(rgb_frame, conf=conf, verbose=False)
        poses = []

        for pr in pose_result:
            if not hasattr(pr, 'keypoints') or pr.keypoints is None:
                continue
            for kp, box in zip(pr.keypoints.data, pr.boxes.xyxy):
                kp_np = kp.cpu().numpy()
                bx1, by1, bx2, by2 = map(int, box)

                # Count visible keypoints
                num_visible = 0
                for k in kp_np:
                    if len(k) >= 3 and k[2] > POSE_VALIDATION_KEYPOINT_CONF:
                        num_visible += 1

                poses.append({
                    'keypoints': kp_np,
                    'bbox': (bx1, by1, bx2, by2),
                    'num_visible': num_visible
                })

        return poses
    except Exception as e:
        return []


def bbox_has_pose_support(bbox, poses, min_keypoints=POSE_VALIDATION_MIN_KEYPOINTS):
    """
    Check if a bounding box has pose keypoint support.
    
    STRICT: The pose bbox CENTER must be inside the detection bbox.
    This prevents a nearby person's stray keypoints from validating
    a non-person object (e.g. stuffed animals, pillows).
    
    Args:
        bbox: (x1, y1, x2, y2)
        poses: list from get_pose_keypoints_for_frame()
        min_keypoints: minimum visible keypoints in the matching pose
    
    Returns:
        True if the bbox is validated by pose data, False otherwise
    """
    if not poses:
        # No pose data available - can't validate, so give benefit of doubt
        return True

    bx1, by1, bx2, by2 = bbox

    for pose in poses:
        # Pose bbox center must be INSIDE the detection bbox
        pcx = (pose['bbox'][0] + pose['bbox'][2]) / 2
        pcy = (pose['bbox'][1] + pose['bbox'][3]) / 2
        if bx1 <= pcx <= bx2 and by1 <= pcy <= by2:
            if pose['num_visible'] >= min_keypoints:
                return True

    return False


def validate_detections_with_pose(detections, poses, conf_threshold=POSE_VALIDATION_CONF_THRESHOLD):
    """
    Filter detections: low-confidence ones must have pose support.
    
    Args:
        detections: list of dicts with 'box' and 'conf' keys
                    (or list of tuples: (x1, y1, x2, y2, conf, ...))
        poses: list from get_pose_keypoints_for_frame()
        conf_threshold: below this conf, require pose validation
    
    Returns:
        filtered list (same format as input)
    """
    if not detections:
        return detections

    validated = []

    for det in detections:
        # Handle both dict and tuple formats
        if isinstance(det, dict):
            conf = det.get('conf', 1.0)
            box = det.get('box', None)
        elif isinstance(det, (list, tuple)) and len(det) >= 5:
            box = (det[0], det[1], det[2], det[3])
            conf = det[4]
        else:
            validated.append(det)
            continue

        # High confidence → keep without validation
        if conf >= conf_threshold:
            validated.append(det)
            continue

        # Low confidence → require pose support
        if box and bbox_has_pose_support(box, poses):
            validated.append(det)
        # else: filtered out (no pose support for low-conf detection)

    return validated
# ===== END NEW POSE VALIDATION HELPER =====


def create_debug_video_writer(video_path, output_folder, fps, frame_shape):
    """
    Create a video writer for debug visualization
    """
    base_name = Path(video_path).stem
    debug_filename = f"{base_name}_debug_tracking.mp4"
    debug_path = Path(output_folder) / debug_filename
    
    h, w = frame_shape[:2]
    
    # If side-by-side, double the width
    if DEBUG_VIDEO_SIDE_BY_SIDE:
        output_width = w * 2
        output_height = h
    else:
        output_width = w
        output_height = h
    
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(str(debug_path), fourcc, fps, (output_width, output_height))
    
    return writer, debug_path

def create_enhanced_debug_frame(frame, frame_idx, yolo_boxes, expanded_boxes, 
                               smoothed_boxes, final_boxes, action_statuses, 
                               positions, detector, debug_info=None, people_info=None):
    """
    Enhanced debug visualization with comprehensive tracking info and people count
    """
    h, w = frame.shape[:2]
    
    # Create visualization frame
    vis_frame = frame.copy()
    
    # Color scheme
    colors = {
        'yolo': (0, 0, 255),        # RED
        'expanded': (0, 255, 255),  # YELLOW
        'smoothed': (0, 255, 0),    # GREEN
        'good_track': (255, 0, 0),  # BLUE
        'fallback': (255, 0, 255),  # MAGENTA
        'white': (255, 255, 255),
        'black': (0, 0, 0)
    }
    
    # 1. Draw YOLO detections (thin, red)
    for i, box in enumerate(yolo_boxes):
        if box:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), colors['yolo'], 1)
            cv2.putText(vis_frame, f"Y{i}", (x1, max(15, y1-5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors['yolo'], 1)
    
    # 2. Draw expanded boxes (dashed yellow)
    for i, box in enumerate(expanded_boxes):
        if box:
            x1, y1, x2, y2 = map(int, box)
            draw_dashed_rectangle(vis_frame, (x1, y1), (x2, y2), colors['expanded'], 2)
    
    # 3. Draw smoothed boxes (thin green)
    for i, box in enumerate(smoothed_boxes):
        if box:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), colors['smoothed'], 2)
    
    # 4. Draw final crops with status-based colors (thick)
    for i, (box, status) in enumerate(zip(final_boxes, action_statuses)):
        if box:
            x1, y1, x2, y2 = map(int, box)
            
            # Choose color
            if status in ["FRESH_DETECTION", "TRACKED-good"]:
                color = colors['good_track']
                label_bg = (180, 0, 0)
            else:
                color = colors['fallback']
                label_bg = (180, 0, 180)
            
            # Draw thick rectangle
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw corner markers
            corner_len = 20
            cv2.line(vis_frame, (x1, y1), (x1+corner_len, y1), color, 4)
            cv2.line(vis_frame, (x1, y1), (x1, y1+corner_len), color, 4)
            cv2.line(vis_frame, (x2, y1), (x2-corner_len, y1), color, 4)
            cv2.line(vis_frame, (x2, y1), (x2, y1+corner_len), color, 4)
            cv2.line(vis_frame, (x1, y2), (x1+corner_len, y2), color, 4)
            cv2.line(vis_frame, (x1, y2), (x1, y2-corner_len), color, 4)
            cv2.line(vis_frame, (x2, y2), (x2-corner_len, y2), color, 4)
            cv2.line(vis_frame, (x2, y2), (x2, y2-corner_len), color, 4)
            
            # Position label
            position = positions[i] if i < len(positions) else f"P{i}"
            label = f"{position.upper()}"
            
            # Background for label
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(vis_frame, 
                         (x1, y1-30), 
                         (x1+label_size[0]+10, y1-5), 
                         label_bg, -1)
            cv2.putText(vis_frame, label, (x1+5, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors['white'], 2)
            
            # Center crosshair
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(vis_frame, (center_x, center_y), 8, color, 2)
            cv2.line(vis_frame, (center_x-15, center_y), (center_x+15, center_y), color, 2)
            cv2.line(vis_frame, (center_x, center_y-15), (center_x, center_y+15), color, 2)
    
    # Add comprehensive info overlay INCLUDING PEOPLE COUNT
    if DEBUG_SHOW_METRICS:
        vis_frame = add_metrics_overlay(vis_frame, frame_idx, action_statuses, 
                                       positions, detector, debug_info, people_info)
    
    return vis_frame

def draw_dashed_rectangle(img, pt1, pt2, color, thickness=1, gap=10):
    """Draw a dashed rectangle"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Top edge
    for x in range(x1, x2, gap*2):
        cv2.line(img, (x, y1), (min(x+gap, x2), y1), color, thickness)
    # Bottom edge
    for x in range(x1, x2, gap*2):
        cv2.line(img, (x, y2), (min(x+gap, x2), y2), color, thickness)
    # Left edge
    for y in range(y1, y2, gap*2):
        cv2.line(img, (x1, y), (x1, min(y+gap, y2)), color, thickness)
    # Right edge
    for y in range(y1, y2, gap*2):
        cv2.line(img, (x2, y), (x2, min(y+gap, y2)), color, thickness)


def add_metrics_overlay(frame, frame_idx, action_statuses, positions, detector, debug_info, people_info=None):
    """Add comprehensive metrics overlay including people count"""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Create semi-transparent background for metrics panel
    panel_height = 200  # Increased height for people count info
    cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Frame info
    cv2.putText(frame, f"Frame: {frame_idx}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Start text cursor under frame line
    y_pos = 60

    # Add people count info if available
    # Add people count info if available
    if people_info:
        final_count = people_info.get('final_count', 'N/A')
        count_ok = (isinstance(final_count, (int, float)) and final_count >= 2)

        # Video-level estimate
        cv2.putText(frame, f"People (video estimate): {final_count}", (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 255) if count_ok else (255, 255, 255), 2)
        y_pos += 35

        # Per-frame detections
        current_detected = people_info.get('current_frame_detected', None)
        if current_detected is not None:
            cv2.putText(frame, f"People (this frame): {current_detected}", (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_pos += 35

        stats = people_info.get('stats', {})
        if stats:
            stats_texts = [
                f"Mean: {stats.get('mean', 0):.1f}",
                f"Median: {stats.get('median', 0):.1f}",
                f"Max: {stats.get('max', 0)}"
            ]
            for text in stats_texts:
                cv2.putText(frame, text, (20, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                y_pos += 25

        if 'crop_strategy' in people_info:
            strategy_text = f"Strategy: {people_info['crop_strategy']}"
            cv2.putText(frame, strategy_text, (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_pos += 30

    # Original debug info (now always safe)
    if debug_info:
        for key, value in debug_info.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_pos += 25

    # Tracking status for each position (right side)
    status_x = w - 350
    status_y = 30
    cv2.putText(frame, "TRACKING STATUS:", (status_x, status_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    for i, (status, pos) in enumerate(zip(action_statuses, positions)):
        y = status_y + 30 + (i * 35)

        if "DETECTION" in status or "good" in status:
            status_color = (0, 255, 0)
            indicator = "●"
        elif "FALLBACK" in status or "poor" in status:
            status_color = (0, 165, 255)
            indicator = "◐"
        else:
            status_color = (0, 0, 255)
            indicator = "○"

        text = f"{indicator} {pos.upper()}: {status.replace('_', ' ')}"
        cv2.putText(frame, text, (status_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)

        if i < len(detector.missing_counters):
            missing = detector.missing_counters[i]
            if missing > 0:
                cv2.putText(frame, f"({missing}f)", (status_x + 240, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

    # Legend at bottom
    legend_y = h - 120
    cv2.rectangle(overlay, (10, legend_y - 10), (w - 10, h - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, "LEGEND:", (20, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    legend_items = [
        ("RED", (0, 0, 255), "YOLO"),
        ("YELLOW", (0, 255, 255), "Expanded"),
        ("GREEN", (0, 255, 0), "Smoothed"),
        ("BLUE", (255, 0, 0), "Good Track"),
        ("MAGENTA", (255, 0, 255), "Fallback"),
    ]

    x_offset = 120
    for label, color, desc in legend_items:
        cv2.rectangle(frame, (x_offset, legend_y - 12), (x_offset + 25, legend_y + 5), color, -1)
        cv2.putText(frame, desc, (x_offset + 30, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        x_offset += 120

    return frame

def create_side_by_side_frame(original, debug):
    """Create side-by-side comparison"""
    h, w = original.shape[:2]
    
    # Add labels
    cv2.putText(original, "ORIGINAL", (20, h-20), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(debug, "DEBUG VIEW", (20, h-20), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Concatenate horizontally
    combined = np.hstack([original, debug])
    
    return combined

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0



def analyze_pose_activity(keypoints, box):
    """Analyze pose keypoints to determine activity level."""
    if keypoints is None or len(keypoints) == 0:
        return 0.0

    activity_score = 0.0

    left_shoulder = keypoints[5] if len(keypoints) > 5 else None
    right_shoulder = keypoints[6] if len(keypoints) > 6 else None
    left_wrist = keypoints[9] if len(keypoints) > 9 else None
    right_wrist = keypoints[10] if len(keypoints) > 10 else None

    box_x1, box_y1, box_x2, box_y2 = box
    box_width = box_x2 - box_x1
    box_height = box_y2 - box_y1

    arms_extended = 0

    if left_shoulder is not None and left_wrist is not None:
        left_shoulder_conf = left_shoulder[2] if len(left_shoulder) > 2 else 0
        left_wrist_conf = left_wrist[2] if len(left_wrist) > 2 else 0

        if left_shoulder_conf > 0.3 and left_wrist_conf > 0.3:
            arm_span = abs(left_wrist[0] - left_shoulder[0])
            if arm_span > box_width * 0.3:
                arms_extended += 1

    if right_shoulder is not None and right_wrist is not None:
        right_shoulder_conf = right_shoulder[2] if len(right_shoulder) > 2 else 0
        right_wrist_conf = right_wrist[2] if len(right_wrist) > 2 else 0

        if right_shoulder_conf > 0.3 and right_wrist_conf > 0.3:
            arm_span = abs(right_wrist[0] - right_shoulder[0])
            if arm_span > box_width * 0.3:
                arms_extended += 1

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

    activity_score = (
        (arms_extended / 2.0) * 0.4 +
        (arms_raised / 2.0) * 0.3 +
        0.3
    )

    return min(activity_score, 1.0)

# ===== ACTIVITY-BASED ZONE ANALYSIS FUNCTIONS =====

def analyze_region_activity(video_path, yolo_model, pose_model, sample_frames=20):
    """
    Analyze actual activity in different regions of the video.
    Hybrid approach: Handles both corner people AND multi-person scenes.
    Merges overlapping detections to prevent false splits in close-ups.
    
    UPDATED: Uses pose validation to filter false positives (stuffed animals etc.)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define zones with overlap to catch corner people
    zone_width = frame_width / 3
    zones = {
        'left': (0, zone_width * 1.2),
        'center': (zone_width * 0.8, 2.2 * zone_width),
        'right': (1.8 * zone_width, frame_width)
    }

    zone_activity = {'left': [], 'center': [], 'right': []}
    zone_people_count = {'left': [], 'center': [], 'right': []}

    sample_indices = [int((i / sample_frames) * total_frames) for i in range(sample_frames)]

    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ===== Get pose data FIRST for validation =====
        frame_poses = get_pose_keypoints_for_frame(rgb, pose_model, conf=0.15)

        # Use lower confidence for zone analysis
        result = yolo_model.predict(rgb, conf=PERSON_DETECTION_CONF_ZONES, classes=[0], verbose=False)
        raw_boxes = []
        corner_boxes = []
        
        for r in result:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                box_w, box_h = x2 - x1, y2 - y1
                area = box_w * box_h
                frame_area = frame_width * frame_height
                aspect = box_w / max(box_h, 1)
                conf = float(b.conf)

                # Check if person is in a corner
                in_corner = is_in_corner(x1, y1, x2, y2, frame_width, frame_height)
                
                # TWO PATHS: Corner detection vs Regular detection
                if in_corner:
                    if area / frame_area >= MIN_PERSON_AREA_RATIO * 0.3:  # Lowered from 0.5x
                        if aspect >= 0.05 and aspect <= 12:  # Wider range for distorted partials
                            if conf > 0.20:  # Lowered from 0.25
                                # ===== Pose validation for low-conf corner detections =====
                                if conf < POSE_VALIDATION_CONF_THRESHOLD:
                                    if not bbox_has_pose_support((x1, y1, x2, y2), frame_poses):
                                        continue  # Skip: no pose support for low-conf detection
                                raw_boxes.append((x1, y1, x2, y2, conf, True))
                else:
                    # REGULAR PATH: Standard thresholds for multi-person scenes
                    if area / frame_area < MIN_PERSON_AREA_RATIO:
                        continue
                    if aspect < 0.08 or aspect > 10:
                        continue
                    # ===== Pose validation for low-conf regular detections =====
                    if conf < POSE_VALIDATION_CONF_THRESHOLD:
                        if not bbox_has_pose_support((x1, y1, x2, y2), frame_poses):
                            continue  # Skip: no pose support for low-conf detection
                    raw_boxes.append((x1, y1, x2, y2, conf, False))  # False = not corner

        # Merge overlapping boxes (removes face+hand false splits)
        boxes = merge_overlapping_boxes(raw_boxes, iou_threshold=0.25)
        corner_boxes = [box for box, is_corner in boxes if is_corner]
        boxes = [box for box, _ in boxes]

        # Get pose data for activity scoring (reuse frame_poses)
        pose_data = {}
        for pose in frame_poses:
            pose_data[pose['bbox']] = pose['keypoints']

        # Analyze each zone
        for zone_name, (zone_start, zone_end) in zones.items():
            zone_boxes = []
            zone_activities = []

            for box in boxes:
                box_center_x = (box[0] + box[2]) / 2
                box_width = box[2] - box[0]
                box_left = box[0]
                box_right = box[2]

                # Calculate overlap with zone
                overlap_start = max(zone_start, box_left)
                overlap_end = min(zone_end, box_right)
                overlap = max(0, overlap_end - overlap_start)

                # ADAPTIVE THRESHOLD: More lenient for corner boxes
                is_corner_box = box in corner_boxes
                overlap_threshold = 0.2 if is_corner_box else 0.3
                
                if overlap > box_width * overlap_threshold:
                    zone_boxes.append(box)

                    # Calculate activity score
                    activity_score = 0.5
                    for pose_box, keypoints in pose_data.items():
                        if calculate_iou(box, pose_box) > 0.2:
                            activity_score = analyze_pose_activity(keypoints, box)
                            break

                    zone_activities.append(activity_score)

            # Store results
            zone_people_count[zone_name].append(len(zone_boxes))
            if zone_activities:
                zone_activity[zone_name].append(np.mean(zone_activities))
            else:
                zone_activity[zone_name].append(0.0)

    cap.release()

    # Calculate aggregate scores
    zone_scores = {}
    for zone_name in zones.keys():
        avg_people = np.mean(zone_people_count[zone_name]) if zone_people_count[zone_name] else 0
        avg_activity = np.mean(zone_activity[zone_name]) if zone_activity[zone_name] else 0
        zone_scores[zone_name] = avg_people * avg_activity

    return zone_scores, zone_people_count, zone_activity


def is_in_corner(x1, y1, x2, y2, frame_width, frame_height, margin=0.15):
    """
    Check if a bounding box is in any corner of the frame.
    """
    left_edge = x1 < frame_width * margin
    right_edge = x2 > frame_width * (1 - margin)
    top_edge = y1 < frame_height * margin
    bottom_edge = y2 > frame_height * (1 - margin)
    
    # Check all four corners
    in_top_left = left_edge and top_edge
    in_top_right = right_edge and top_edge
    in_bottom_left = left_edge and bottom_edge
    in_bottom_right = right_edge and bottom_edge
    
    return in_top_left or in_top_right or in_bottom_left or in_bottom_right

def merge_overlapping_boxes(raw_boxes, iou_threshold=0.3):
    """
        Merge overlapping bounding boxes to prevent detecting face+hand as separate people.
    Uses greedy NMS approach.
    
    Args:
        raw_boxes: List of (x1, y1, x2, y2, confidence, is_corner)
        iou_threshold: IoU threshold for merging (0.3 means 30% overlap triggers merge)
    
    Returns:
        List of merged boxes as ((x1, y1, x2, y2), is_corner)
    """
    if not raw_boxes:
        return []
    
    # Sort by confidence (highest first)
    raw_boxes = sorted(raw_boxes, key=lambda x: x[4], reverse=True)
    
    merged = []
    used = [False] * len(raw_boxes)
    
    for i, (x1_i, y1_i, x2_i, y2_i, conf_i, is_corner_i) in enumerate(raw_boxes):
        if used[i]:
            continue
            
        # Start with this box
        merge_group = [(x1_i, y1_i, x2_i, y2_i, conf_i, is_corner_i)]
        used[i] = True
        
        # Find all boxes that overlap with this one
        for j, (x1_j, y1_j, x2_j, y2_j, conf_j, is_corner_j) in enumerate(raw_boxes):
            if used[j] or i == j:
                continue
                
            iou = calculate_iou((x1_i, y1_i, x2_i, y2_i), (x1_j, y1_j, x2_j, y2_j))
            
            # If significant overlap, merge them
            if iou > iou_threshold:
                merge_group.append((x1_j, y1_j, x2_j, y2_j, conf_j, is_corner_j))
                used[j] = True
        
        # Create merged bounding box
        x1_merged = min(box[0] for box in merge_group)
        y1_merged = min(box[1] for box in merge_group)
        x2_merged = max(box[2] for box in merge_group)
        y2_merged = max(box[3] for box in merge_group)
        
        is_corner_merged = any(box[5] for box in merge_group)
        
        merged.append(((x1_merged, y1_merged, x2_merged, y2_merged), is_corner_merged))
    
    return merged


def pick_best_zones_by_presence(zone_people, zone_activity, k=2):
    # Score = avg_people + small weight on avg_activity
    # Tie-breaker preference: center > right > left
    preference = {"center": 2, "right": 1, "left": 0}

    scores = []
    for z in ["left", "center", "right"]:
        avg_p = float(np.mean(zone_people[z])) if zone_people[z] else 0.0
        avg_a = float(np.mean(zone_activity[z])) if zone_activity[z] else 0.0
        score = avg_p + 0.25 * avg_a
        scores.append((score, preference[z], z, avg_p, avg_a))

    # sort by score desc, then preference desc
    scores.sort(key=lambda x: (x[0], x[1]), reverse=True)
    picked = [s[2] for s in scores[:k]]
    return picked, scores

def analyze_action_coherence(video_path, yolo_model, pose_model, sample_frames=15):
    """
    Analyze whether multiple people are performing coherent actions.
    Returns (similarity_score, interaction_score, combined_score)
    similarity_score = 0.0-1.0 (similar poses)
    interaction_score = 0.0-1.0 (interacting across zones)
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    similarity_scores = []
    interaction_scores = []
    sample_indices = [int((i / sample_frames) * total_frames) for i in range(sample_frames)]
    
    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get people detections
        result = yolo_model.predict(rgb, conf=PERSON_DETECTION_CONF_ZONES, classes=[0], verbose=False)
        people_boxes = []
        
        for r in result:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                box_w, box_h = x2 - x1, y2 - y1
                area = box_w * box_h
                frame_area = frame.shape[0] * frame.shape[1]
                
                if area / frame_area >= MIN_PERSON_AREA_RATIO:
                    people_boxes.append((x1, y1, x2, y2))
        
        # If not exactly 2 people, skip
        if len(people_boxes) != 2:
            continue
        
        # Get pose data for both people
        pose_data = {}
        if pose_model:
            pose_result = pose_model.predict(rgb, conf=0.3, verbose=False)
            for pr in pose_result:
                if hasattr(pr, 'keypoints') and pr.keypoints is not None:
                    for idx, (kp, box) in enumerate(zip(pr.keypoints.data, pr.boxes.xyxy)):
                        x1, y1, x2, y2 = map(int, box)
                        pose_data[(x1, y1, x2, y2)] = kp.cpu().numpy()
        
        # If we have pose data for both, analyze
        if len(pose_data) >= 2:
            # Find which pose matches which person box
            person_poses = []
            matched_boxes = []

            # For each person box, find the pose with best IoU
            for person_box in people_boxes:
                best_pose = None
                best_iou = 0
                best_box = None
                
                for pose_box, keypoints in pose_data.items():
                    iou = calculate_iou(person_box, pose_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_pose = keypoints
                        best_box = pose_box
                
                # Accept if IoU > 0.1 (more lenient)
                if best_pose is not None and best_iou > 0.1:
                    person_poses.append(best_pose)
                    matched_boxes.append(best_box)
                else:
                    # If no good pose match, still add None but keep the box
                    person_poses.append(None)
                    matched_boxes.append(person_box)
            
            # If we have poses for both people
            if len(person_poses) == 2:
                # Calculate pose similarity (original)
                similarity = calculate_pose_coherence(person_poses[0], person_poses[1])
                similarity_scores.append(similarity)
                
                # Calculate interaction score
                interaction = calculate_interaction_score(
                    person_poses[0], person_poses[1], 
                    matched_boxes[0], matched_boxes[1],
                    frame.shape
                )
                interaction_scores.append(interaction)
    
    cap.release()
    
    # Calculate averages
    avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.5
    avg_interaction = np.mean(interaction_scores) if interaction_scores else 0.0
    
    # Combined score (weighted)
    # If interaction is high, they're doing something together even if poses differ
    combined_score = max(avg_similarity * 0.4, avg_interaction * 0.8)
    
    return avg_similarity, avg_interaction, combined_score

def calculate_pose_coherence(pose1, pose2, threshold=0.3):
    """
    Calculate how similar two poses are (0.0-1.0).
    Higher = more similar/coordinated actions.
    """
    # Keypoint indices for key body parts
    KEY_INDICES = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # nose, shoulders, elbows, wrists, hips, knees, ankles
    
    similar_count = 0
    total_compared = 0
    
    for idx in KEY_INDICES:
        if idx < len(pose1) and idx < len(pose2):
            conf1 = pose1[idx][2] if len(pose1[idx]) > 2 else 0
            conf2 = pose2[idx][2] if len(pose2[idx]) > 2 else 0
            
            # Only compare if both keypoints are confident
            if conf1 > threshold and conf2 > threshold:
                # Get positions
                x1, y1 = pose1[idx][0], pose1[idx][1]
                x2, y2 = pose2[idx][0], pose2[idx][1]
                
                # Normalize by body size (approximate)
                # Use shoulder width as reference
                shoulder_width1 = 0
                if len(pose1) > 6:
                    if pose1[5][2] > threshold and pose1[6][2] > threshold:
                        shoulder_width1 = abs(pose1[5][0] - pose1[6][0])
                
                shoulder_width2 = 0
                if len(pose2) > 6:
                    if pose2[5][2] > threshold and pose2[6][2] > threshold:
                        shoulder_width2 = abs(pose2[5][0] - pose2[6][0])
                
                ref_shoulder = max(shoulder_width1, shoulder_width2, 50)  # Minimum reference
                
                # Calculate normalized distance between same body parts
                dx = abs(x1 - x2) / ref_shoulder
                dy = abs(y1 - y2) / ref_shoulder
                
                # If body parts are close (similar positions), they might be coordinated
                if dx < 0.5 and dy < 0.5:
                    similar_count += 1
                
                total_compared += 1
    
    if total_compared > 0:
        return similar_count / total_compared
    return 0.0

def calculate_interaction_score(pose1, pose2, box1, box2, frame_shape):
    """
    Calculate how much two people are interacting (0.0-1.0)
    More robust version that works even with partial pose data
    """
    h, w = frame_shape[:2]
    
    # Calculate centers
    cx1 = (box1[0] + box1[2]) / 2
    cy1 = (box1[1] + box1[3]) / 2
    cx2 = (box2[0] + box2[2]) / 2
    cy2 = (box2[1] + box2[3]) / 2
    
    # 1. Proximity score (closer = more likely interacting)
    distance = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
    box_widths = (box1[2] - box1[0] + box2[2] - box2[0]) / 2
    max_interact_distance = box_widths * 2.5  # Within 2.5 body widths
    proximity = max(0, 1 - (distance / max_interact_distance))
    
    interaction_score = proximity * 0.4  # Base from proximity
    
    # 2. Check vertical alignment (people standing near each other)
    vertical_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    avg_height = ((box1[3] - box1[1]) + (box2[3] - box2[1])) / 2
    if avg_height > 0:
        vertical_alignment = vertical_overlap / avg_height
        interaction_score += vertical_alignment * 0.2
    
    # 3. If we have pose data, check for directional movement
    if pose1 is not None and pose2 is not None and len(pose1) >= 17 and len(pose2) >= 17:
        
        # Get wrist positions if available
        wrists = []
        for pose in [pose1, pose2]:
            left_wrist = pose[9] if len(pose) > 9 else None
            right_wrist = pose[10] if len(pose) > 10 else None
            
            wrist_positions = []
            if left_wrist is not None and left_wrist[2] > 0.2:
                wrist_positions.append((left_wrist[0], left_wrist[1]))
            if right_wrist is not None and right_wrist[2] > 0.2:
                wrist_positions.append((right_wrist[0], right_wrist[1]))
            wrists.append(wrist_positions)
        
        # Check if any wrist from person 1 is close to person 2's center
        for wrist in wrists[0]:
            wx, wy = wrist
            dist_to_p2 = np.sqrt((wx - cx2)**2 + (wy - cy2)**2)
            if dist_to_p2 < box_widths:
                # Wrist is near the other person - STRONG interaction!
                interaction_score += 0.4
                break
        
        # Check if any wrist from person 2 is close to person 1's center
        for wrist in wrists[1]:
            wx, wy = wrist
            dist_to_p1 = np.sqrt((wx - cx1)**2 + (wy - cy1)**2)
            if dist_to_p1 < box_widths:
                interaction_score += 0.4
                break
        
        # Check for extended limbs (potential kick/punch)
        limb_indices = [(5, 7, 9), (6, 8, 10)]  # shoulder, elbow, wrist
        
        for person_idx, pose in enumerate([pose1, pose2]):
            other_center = (cx2, cy2) if person_idx == 0 else (cx1, cy1)
            
            for shoulder_idx, elbow_idx, wrist_idx in limb_indices:
                if (shoulder_idx < len(pose) and elbow_idx < len(pose) and wrist_idx < len(pose)):
                    shoulder = pose[shoulder_idx]
                    wrist = pose[wrist_idx]
                    
                    if shoulder[2] > 0.2 and wrist[2] > 0.2:
                        # Calculate limb extension
                        dx = wrist[0] - shoulder[0]
                        dy = wrist[1] - shoulder[1]
                        limb_length = np.sqrt(dx**2 + dy**2)
                        
                        # Vector from shoulder to other person
                        tox = other_center[0] - shoulder[0]
                        toy = other_center[1] - shoulder[1]
                        to_length = np.sqrt(tox**2 + toy**2)
                        
                        if limb_length > 30 and to_length > 0:
                            # Check if limb points toward other person
                            dot = (dx * tox + dy * toy) / (limb_length * to_length)
                            if dot > 0.3:  # Pointing somewhat toward them
                                interaction_score += dot * 0.3
    
    # Cap at 1.0
    return min(interaction_score, 1.0)


def calculate_movement_synchrony(person_boxes_history, max_frames=20):
    """
    Calculate if two people move in sync over time.
    Returns sync score (0.0-1.0).
    """
    if len(person_boxes_history) < 2:
        return 0.5
    
    # Get recent movement vectors
    movements = []
    for i in range(min(len(person_boxes_history), max_frames)):
        if i >= len(person_boxes_history[0]) or i >= len(person_boxes_history[1]):
            continue
        
        box1 = person_boxes_history[0][-i-1] if len(person_boxes_history[0]) > 0 else None
        box2 = person_boxes_history[1][-i-1] if len(person_boxes_history[1]) > 0 else None
        
        if box1 and box2:
            # Calculate centers
            cx1 = (box1[0] + box1[2]) / 2
            cy1 = (box1[1] + box1[3]) / 2
            cx2 = (box2[0] + box2[2]) / 2
            cy2 = (box2[1] + box2[3]) / 2
            
            movements.append((cx1, cy1, cx2, cy2))
    
    if len(movements) < 5:
        return 0.5
    
    # Calculate correlation of movements
    x1_movements = [movements[i][0] - movements[i-1][0] for i in range(1, len(movements))]
    y1_movements = [movements[i][1] - movements[i-1][1] for i in range(1, len(movements))]
    x2_movements = [movements[i][2] - movements[i-1][2] for i in range(1, len(movements))]
    y2_movements = [movements[i][3] - movements[i-1][3] for i in range(1, len(movements))]
    
    # Normalize
    if len(x1_movements) > 1:
        x_corr = np.corrcoef(x1_movements, x2_movements)[0, 1]
        y_corr = np.corrcoef(y1_movements, y2_movements)[0, 1]
        
        # Handle NaN
        x_corr = 0 if np.isnan(x_corr) else max(0, x_corr)
        y_corr = 0 if np.isnan(y_corr) else max(0, y_corr)
        
        return (x_corr + y_corr) / 2
    return 0.5


def determine_smart_crop_strategy_v2(video_path, yolo_model, pose_model=None, sample_frames=20, people_count=0):
    """
    ACTION-AWARE cropping: Focus on where actions happen, not just people.
    Returns: (crop_count, positions_to_use, strategy_description)
    
    Takes people_count as input to make intelligent decisions about 2-person videos
    """
    # Helper function to sort positions spatially (left to right)
    def sort_positions(positions):
        spatial_order = {"left": 0, "center": 1, "middle": 1, "right": 2}
        return sorted(positions, key=lambda p: spatial_order.get(p, 1))
    
    # ===== HANDLE SINGLE PERSON EXPLICITLY =====
    if people_count == 1:
        print(f"   👤 Single person detected - NO CROP (would split body parts)")
        return 0, [], "single-person-no-crop"
    # ================================================

    
    print(f"   🔍 Analyzing ACTION zones (not just people)...")
    
    # Get activity analysis
    zone_scores, zone_people, zone_activity = analyze_region_activity(
        video_path, yolo_model, pose_model, sample_frames
    )
    
    print(f"   📊 ACTION Zone analysis:")
    
    # Calculate ACTION metrics (not people metrics)
    zone_action_potential = {}
    
    for zone in ['left', 'center', 'right']:
        if zone_activity[zone]:
            # Key metrics for action cropping:
            # 1. Maximum activity level (peak action)
            max_activity = max(zone_activity[zone])
            
            # 2. Percentage of frames with significant action (> 0.6)
            high_action_frames = sum(1 for activity in zone_activity[zone] if activity > 0.6)
            action_consistency = high_action_frames / len(zone_activity[zone])
            
            # 3. Action density (activity * people)
            avg_people = np.mean(zone_people[zone]) if zone_people[zone] else 0
            avg_activity = np.mean(zone_activity[zone])
            action_density = avg_people * avg_activity
            
            zone_action_potential[zone] = {
                'max_activity': max_activity,
                'action_consistency': action_consistency,
                'action_density': action_density,
                'avg_activity': avg_activity,
                'avg_people': avg_people,
                'has_action': action_consistency > 0.15 or max_activity > 0.55
            }
            
            print(f"      {zone.capitalize()}:")
            print(f"        Max activity: {max_activity:.2f}")
            print(f"        Action consistency: {action_consistency:.0%}")
            print(f"        Action density: {action_density:.2f}")
            print(f"        Avg people: {avg_people:.1f}")
            print(f"        Has action: {'✓' if zone_action_potential[zone]['has_action'] else '✗'}")
    
    # Count zones with significant action
    action_zones = [zone for zone in ['left', 'center', 'right'] 
                    if zone in zone_action_potential and zone_action_potential[zone]['has_action']]
    
    print(f"   🎯 Zones with action: {len(action_zones)} ({action_zones})")
    
    # ===== IMPROVED 2-PERSON LOGIC =====
    # Only apply strict 2-person logic if we're CONFIDENT it's exactly 2 people
    # (not 3+ with some partially visible)
    if people_count == 2:
        print(f"   👥 2-person video detected - checking action coherence...")
        
        # Calculate coherence and interaction scores
        similarity_score = 0.5
        interaction_score = 0.0
        combined_score = 0.5

        if pose_model:
            similarity_score, interaction_score, combined_score = analyze_action_coherence(
                video_path, yolo_model, pose_model, sample_frames=10
            )
            print(f"   🧘 Pose similarity: {similarity_score:.2f} (0=different, 1=same)")
            print(f"   🤝 Interaction score: {interaction_score:.2f} (0=no interaction, 1=strong interaction)")
            print(f"   🎯 Combined action score: {combined_score:.2f}")
            
            # Use combined score for decision making
            pose_coherence = combined_score  # Replace old variable
        
        # Determine zones with people
        zones_with_people = []
        for zone in ['left', 'center', 'right']:
            if zone in zone_action_potential:
                avg_people = zone_action_potential[zone]['avg_people']
                if avg_people >= 0.7:
                    zones_with_people.append(zone)
        
        print(f"   📍 Zones with people: {zones_with_people}")
        
        # ===== DECISION LOGIC =====
        
        # Case A: Both people in SAME zone (likely collaborative action)
        if len(zones_with_people) == 1:
            zone = zones_with_people[0]
            print(f"   👥 Both people in {zone} zone")
            
            # Check if they're doing the same action
            if pose_coherence > 0.6:
                print(f"   🤝 High coherence ({pose_coherence:.2f}) - COLLABORATIVE action")
                print(f"   ➡️ Cropping single {zone} zone (showing both together)")
                return 1, [zone], f"two-person-collaborative-{zone}"
            else:
                print(f"   🏃 Low coherence ({pose_coherence:.2f}) - DIFFERENT actions in same zone")
                if zone == 'center':
                    print(f"   ➡️ Center zone with different actions - cropping left+right")
                    return 2, ['left', 'right'], "two-person-different-center"
                else:
                    print(f"   ➡️ Side zone with different actions - no crop (too cramped)")
                    return 0, [], "two-person-cramped-different"
        
        # Case B: People in DIFFERENT zones
        elif len(zones_with_people) >= 2:
            print(f"   ↔️ People in different zones")
            
            # Check action separation
            action_zones = [z for z in zones_with_people 
                          if zone_action_potential[z]['has_action']]
            
            print(f"   🎯 Action zones: {action_zones}")
            
            # If both zones have action AND low coherence -> separate crops
            if len(action_zones) >= 2 and pose_coherence < 0.4:
                action_zones = sort_positions(action_zones[:2])
                print(f"   🎯 Low coherence + action in both zones -> SEPARATE crops")
                return 2, action_zones, "two-person-separate-actions"
            
            # If high coherence but in different zones -> could be reaching across
            elif pose_coherence > 0.6:
                # Find the main action zone
                main_zone = max(action_zones, 
                              key=lambda z: zone_action_potential[z]['action_density'], 
                              default='center')
                print(f"   🤝 High coherence across zones -> MAIN action in {main_zone}")
                return 1, [main_zone], "two-person-coherent-across-zones"
            
            else:
                # Default: crop both if separated
                zones_with_people = sort_positions(zones_with_people[:2])
                print(f"   ⚖️ Default: crop separated people")
                return 2, zones_with_people, "two-person-default-separated"
        
        # Case C: Can't determine clearly
        else:
            print(f"   ❓ Ambiguous 2-person scenario")
            best2, dbg = pick_best_zones_by_presence(zone_people, zone_activity, k=2)
            best2 = sort_positions(best2)
            print(f"   📋 Fallback to presence-based: {best2}")
            return 2, best2, f"two-person-fallback-{best2[0]}-{best2[1]}"
    
    # ===== ORIGINAL LOGIC FOR 3+ PEOPLE (or when 2-person check skipped) =====
    
    # Case 1: No significant action anywhere
    if len(action_zones) == 0:
        best2, dbg = pick_best_zones_by_presence(zone_people, zone_activity, k=2)
        best2 = sort_positions(best2)  # ✅ SORT!
        print(f"   📋 No clear action - using presence-based zones: {best2}")
        return 2, best2, f"no-action-presence-{best2[0]}-{best2[1]}"
    
    # Case 2: Single action zone
    elif len(action_zones) == 1:
        zone = action_zones[0]
        print(f"   🎯 Single action zone: {zone}")
        return 1, [zone], f"single-action-{zone}"
    
    # Case 3: Two action zones
    elif len(action_zones) == 2:
        action_zones = sort_positions(action_zones)  # ✅ SORT!
        print(f"   🎯 Two action zones: {action_zones}")
        
        # If actions are in left+center or center+right, they might be connected
        if set(action_zones) == {'left', 'center'}:
            # Check if these are separate actions or one continuous action
            left_density = zone_action_potential['left']['action_density']
            center_density = zone_action_potential['center']['action_density']
            
            # If center has much higher density, might be main action
            if center_density > left_density * 2:
                print(f"   🎯 Center dominant - cropping center only")
                return 1, ['center'], "center-dominant-action"
            else:
                return 2, action_zones, "left-center-actions"
                
        elif set(action_zones) == {'center', 'right'}:
            center_density = zone_action_potential['center']['action_density']
            right_density = zone_action_potential['right']['action_density']
            
            if center_density > right_density * 2:
                return 1, ['center'], "center-dominant-action"
            else:
                return 2, action_zones, "center-right-actions"
        
        else:  # left + right
            return 2, action_zones, "left-right-actions"
    
    # Case 4: Three action zones
    else:  # all 3 zones have action
        action_zones = sort_positions(action_zones)  # ✅ SORT!
        print(f"   🎯 Three action zones detected")
        
        # Check if center is the main action hub
        center_density = zone_action_potential['center']['action_density']
        left_density = zone_action_potential['left']['action_density']
        right_density = zone_action_potential['right']['action_density']
        
        # If center has significantly more action than sides
        if center_density > (left_density + right_density) * 0.8:
            print(f"   🎯 Center is action hub - cropping center only")
            return 1, ['center'], "center-action-hub"
        
        # If sides have more action than center
        elif (left_density + right_density) > center_density * 1.5:
            print(f"   🎯 Sides have more action - cropping left+right")
            return 2, ['left', 'right'], "side-actions-dominant"
        
        # Otherwise, crop all three
        else:
            print(f"   🎯 Balanced action across zones - cropping all three")
            return 3, ['left', 'center', 'right'], "balanced-three-zone-actions"
                
def get_pose_center_target(keypoints, box):
    """
    Calculate optimal crop center based on pose keypoints.
    """
    if keypoints is None or len(keypoints) < 17:
        return None

    box_x1, box_y1, box_x2, box_y2 = box
    box_width = box_x2 - box_x1
    box_height = box_y2 - box_y1

    reliable_kps = []
    for i in range(17):
        if keypoints[i][2] > POSE_CONFIDENCE_THRESHOLD:
            reliable_kps.append(keypoints[i])

    if len(reliable_kps) < MIN_POSE_KEYPOINTS:
        return None

    kp_center_x = np.mean([kp[0] for kp in reliable_kps])
    kp_center_y = np.mean([kp[1] for kp in reliable_kps])

    important_indices = [5, 6, 11, 12, 0]
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

        if not person_boxes or len(person_boxes) == 0:
            if self.debug:
                print("   ⚠️  No person boxes -> no ROI")
            return None, 'full_body'

        current_poses = []
        if pose_model and USE_POSE_FOR_ROI:
            current_poses = self._get_matched_poses(frame, person_boxes, pose_model, max_people)

        if len(current_poses) > 0:
            roi = self._get_pose_based_roi(current_poses, person_boxes, w, h)
            focus_region = self._determine_focus_region(current_poses)
        else:
            roi = self._merge_boxes(person_boxes)
            focus_region = 'full_body'

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
                if np.sum(kpts[:, 2] > ROI_CONFIDENCE_THRESHOLD) >= MIN_POSE_KEYPOINTS:
                    visible_kpts = kpts[kpts[:, 2] > ROI_CONFIDENCE_THRESHOLD]
                    pose_center = visible_kpts[:, :2].mean(axis=0)

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

    def _smart_merge_boxes(self, boxes, frame_width, frame_height):
        """Merge boxes intelligently considering interaction zones"""
        if len(boxes) == 0:
            return None

        if len(boxes) == 1:
            # Expand single box for better framing
            x1, y1, x2, y2 = boxes[0]
            width = x2 - x1
            height = y2 - y1

            # Add more padding for single person
            padding_x = width * 0.3
            padding_y = height * 0.4

            return (
                max(0, int(x1 - padding_x)),
                max(0, int(y1 - padding_y)),
                min(frame_width, int(x2 + padding_x)),
                min(frame_height, int(y2 + padding_y))
            )

        # Merge multiple boxes
        x1_min = min(b[0] for b in boxes)
        y1_min = min(b[1] for b in boxes)
        x2_max = max(b[2] for b in boxes)
        y2_max = max(b[3] for b in boxes)

        width = x2_max - x1_min
        height = y2_max - y1_min

        # Adaptive padding based on box distribution
        box_centers = [(b[0]+b[2])/2 for b in boxes]
        spread = max(box_centers) - min(box_centers)

        if spread < frame_width * 0.3:
            # Boxes are close together - tighter padding
            padding_x = width * 0.2
            padding_y = height * 0.25
        else:
            # Boxes are spread out - generous padding
            padding_x = width * 0.15
            padding_y = height * 0.2

        merged_box = (
            max(0, int(x1_min - padding_x)),
            max(0, int(y1_min - padding_y)),
            min(frame_width, int(x2_max + padding_x)),
            min(frame_height, int(y2_max + padding_y))
        )

        # Ensure minimum size
        merged_width = merged_box[2] - merged_box[0]
        merged_height = merged_box[3] - merged_box[1]

        if merged_width < frame_width * 0.25 or merged_height < frame_height * 0.3:
            center_x = (merged_box[0] + merged_box[2]) // 2
            center_y = (merged_box[1] + merged_box[3]) // 2

            min_width = max(merged_width, frame_width * 0.25)
            min_height = max(merged_height, frame_height * 0.3)

            return (
                max(0, int(center_x - min_width // 2)),
                max(0, int(center_y - min_height // 2)),
                min(frame_width, int(center_x + min_width // 2)),
                min(frame_height, int(center_y + min_height // 2))
            )

        return merged_box

    def _get_pose_based_roi(self, poses, person_boxes, frame_width, frame_height):
        """Get ROI based on pose keypoints"""
        all_points = []
        
        for pose in poses:
            for idx in range(17):
                if pose[idx, 2] > ROI_CONFIDENCE_THRESHOLD:
                    point = pose[idx, :2]
                    all_points.append(point)
        
        if len(all_points) == 0:
            return self._smart_merge_boxes(person_boxes, frame_width, frame_height)
        
        all_points_array = np.array(all_points)
        x_min, y_min = np.min(all_points_array, axis=0)
        x_max, y_max = np.max(all_points_array, axis=0)
        
        width = x_max - x_min
        height = y_max - y_min
        
        # Use reasonable padding
        padding_x = width * 0.20
        padding_y = height * 0.25
        
        x1 = max(0, int(x_min - padding_x))
        y1 = max(0, int(y_min - padding_y))
        x2 = min(frame_width, int(x_max + padding_x))
        y2 = min(frame_height, int(y_max + padding_y))
        
        # Ensure minimum size (but not excessive)
        min_width = frame_width * 0.25
        min_height = frame_height * 0.25
        
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
            for idx in range(0, 11):
                if pose[idx, 2] > ROI_CONFIDENCE_THRESHOLD:
                    upper_count += 1

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

def calculate_motion_expansion(current_box, history, base_margin=0.20, pose_activity=0.0):
    """
    Calculate adaptive expansion based on movement history AND pose activity.
    Much more conservative expansion.
    """
    if not history or len(history) < 3:
        # Return base margin or small minimum
        return max(base_margin, MIN_EXPANSION + (pose_activity * 0.05))  # Reduced from 0.3 to 0.05

    recent_boxes = list(history)[-10:]

    if len(recent_boxes) < 2:
        return max(base_margin, MIN_EXPANSION + (pose_activity * 0.05))

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

    box_widths = [b[2] - b[0] for b in recent_boxes]
    box_heights = [b[3] - b[1] for b in recent_boxes]

    width_variance = max(box_widths) - min(box_widths)
    height_variance = max(box_heights) - min(box_heights)

    curr_w = current_box[2] - current_box[0]
    curr_h = current_box[3] - current_box[1]

    motion_factor_x = max_dx / max(curr_w, 1)
    motion_factor_y = max_dy / max(curr_h, 1)
    size_factor_x = width_variance / max(curr_w, 1)
    size_factor_y = height_variance / max(curr_h, 1)

    motion_factor = max(
        motion_factor_x,
        motion_factor_y,
        size_factor_x,
        size_factor_y
    )

    # CAP IT! Motion can't be more than 1.0 (100% of box size)
    motion_factor = min(motion_factor, 1.0)

    adaptive_margin = base_margin + (motion_factor * 0.5) + (pose_activity * 0.1)

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

def count_people_in_video(video_path, yolo_model, pose_model=None, sample_frames=30, return_details=False):
    """
    ENHANCED people counting with:
    1. Partial person detection (legs only, torsos, etc.)
    2. Pose keypoint clustering
    3. Interaction zone analysis
    4. Multi-method fusion
    5. Pose validation for low-confidence detections
    
    Returns more accurate count even when people are partially visible or overlapping.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        if return_details:
            return {'final_count': 0, 'raw_counts': [], 'stats': {}, 'method': 'no-frames'}
        return 0

    frame_indices = []
    if total_frames <= sample_frames:
        frame_indices = list(range(total_frames))
    else:
        for i in range(0, sample_frames):
            pos = int((i / sample_frames) * total_frames)
            frame_indices.append(pos)

    # Track counts from different methods
    bbox_counts = []
    pose_counts = []
    combined_counts = []
    frame_details = []
    pose_filtered_counts = []  # track how many were filtered

    for idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        h, w = frame.shape[:2]
        frame_area = h * w
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ===== Get pose data FIRST for validation =====
        frame_poses = get_pose_keypoints_for_frame(rgb, pose_model, conf=0.15)

        # ===== METHOD 1: BOUNDING BOX DETECTION (Enhanced) =====
        result = yolo_model.predict(rgb, conf=PERSON_DETECTION_CONF, classes=[0], verbose=False)
        
        bbox_detections = []
        partial_detections = []
        filtered_by_pose = 0  # counter
        
        for r in result:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf)
                box_w, box_h = x2 - x1, y2 - y1
                area = box_w * box_h
                aspect = box_w / max(box_h, 1)

                # Standard detection
                if area / frame_area >= MIN_PERSON_AREA_RATIO:
                    if aspect >= 0.15 and aspect <= 6:
                        # ===== Pose validation for low-conf detections =====
                        if conf < POSE_VALIDATION_CONF_THRESHOLD:
                            if not bbox_has_pose_support((x1, y1, x2, y2), frame_poses):
                                filtered_by_pose += 1
                                continue  # Skip: no pose support
                        
                        bbox_detections.append({
                            'box': (x1, y1, x2, y2),
                            'conf': conf,
                            'area': area,
                            'type': 'full'
                        })
                        continue

                # ENHANCED: Partial person detection
                if USE_PARTIAL_PERSON_DETECTION:
                    if area / frame_area >= PARTIAL_PERSON_MIN_AREA_RATIO:
                        is_partial = False
                        partial_type = None
                        
                        # Very tall thin boxes = legs only
                        if aspect >= 0.1 and aspect <= 0.4 and box_h > h * 0.2:
                            is_partial = True
                            partial_type = 'legs'
                        # Wide short boxes = torso/sitting
                        elif aspect >= 1.2 and aspect <= 4 and box_w > w * 0.1:
                            is_partial = True
                            partial_type = 'torso'
                        
                        if is_partial:
                            # ===== Pose validation for partial detections too =====
                            if conf < POSE_VALIDATION_CONF_THRESHOLD:
                                if not bbox_has_pose_support((x1, y1, x2, y2), frame_poses):
                                    filtered_by_pose += 1
                                    continue  # Skip: no pose support for partial
                            
                            partial_detections.append({
                                'box': (x1, y1, x2, y2),
                                'conf': conf,
                                'area': area,
                                'type': partial_type
                            })

        pose_filtered_counts.append(filtered_by_pose)

        # Merge overlapping full detections
        merged_bbox = merge_overlapping_boxes(
            [(d['box'][0], d['box'][1], d['box'][2], d['box'][3], d['conf'], False) 
             for d in bbox_detections],
            iou_threshold=0.3
        )
        bbox_count = len(merged_bbox)

        # ===== METHOD 2: POSE KEYPOINT CLUSTERING =====
        pose_count = 0
        keypoint_clusters = []
        
        if pose_model and POSE_KEYPOINT_CLUSTER_DETECTION:
            try:
                pose_result = pose_model.predict(rgb, conf=0.2, verbose=False)
                
                if len(pose_result) > 0 and hasattr(pose_result[0], 'keypoints'):
                    all_keypoints = pose_result[0].keypoints.data.cpu().numpy()
                    
                    # Cluster keypoints by proximity
                    keypoint_clusters = cluster_keypoints_by_person(
                        all_keypoints, 
                        min_keypoints=MIN_KEYPOINT_CLUSTER_SIZE,
                        radius=KEYPOINT_CLUSTER_RADIUS
                    )
                    pose_count = len(keypoint_clusters)
                    
            except Exception as e:
                print(f"⚠️ Pose detection failed: {e}")

        # ===== METHOD 3: COMBINED ANALYSIS =====
        combined_count = bbox_count
        
        if COMBINE_BBOX_AND_POSE_COUNTS:
            # Use the MAXIMUM of bbox and pose counts as baseline
            combined_count = max(bbox_count, pose_count)
            
            # If we have partial detections, check if they represent additional people
            if len(partial_detections) > 0:
                # Check if partial detections are near existing full detections
                additional_people = count_additional_from_partials(
                    merged_bbox, partial_detections, w, h
                )
                combined_count += additional_people

            # ADJACENCY BONUS: If we detected 2 people but there are signs of a 3rd
            if combined_count == 2 and ADJACENCY_BONUS:
                if has_evidence_of_third_person(
                    merged_bbox, partial_detections, keypoint_clusters, w, h
                ):
                    print(f"  🔍 Frame {idx}: Adjacency bonus - likely 3rd person")
                    combined_count = 3

        bbox_counts.append(bbox_count)
        pose_counts.append(pose_count)
        combined_counts.append(combined_count)
        
        frame_details.append({
            'frame_idx': int(frame_idx),
            'bbox_count': int(bbox_count),
            'pose_count': int(pose_count),
            'combined_count': int(combined_count),
            'partial_detections': len(partial_detections),
            'filtered_by_pose': filtered_by_pose
        })

        if idx % 10 == 0:
            filter_info = f", filtered={filtered_by_pose}" if filtered_by_pose > 0 else ""
            print(f"  Frame {idx+1}/{len(frame_indices)}: bbox={bbox_count}, pose={pose_count}, combined={combined_count}{filter_info}")

    cap.release()

    if not combined_counts:
        if return_details:
            return {'final_count': 0, 'raw_counts': [], 'stats': {}, 'method': 'no-detections'}
        return 0

    # ===== FINAL COUNT DETERMINATION =====
    total_filtered = sum(pose_filtered_counts)
    print(f"  📊 Detection summary:")
    print(f"     BBox counts: {bbox_counts}")
    print(f"     Pose counts: {pose_counts}")
    print(f"     Combined counts: {combined_counts}")
    if total_filtered > 0:
        print(f"     🔬 Pose validation filtered {total_filtered} false positives across all frames")

    # Use combined counts as primary method
    counts_array = np.array(combined_counts)
    mean_count = np.mean(counts_array)
    median_count = np.median(counts_array)
    max_count = max(combined_counts)

    counter = Counter(combined_counts)
    most_common = counter.most_common(3)

    print(f"  Statistics: mean={mean_count:.1f}, median={median_count}, max={max_count}")
    print(f"  Most common: {most_common}")

    # Decision logic: favor higher counts when evidence is strong
    candidate_counts = []
    
    # Add mean if reasonable
    if mean_count >= 2:
        candidate_counts.append(mean_count)
    
    # Add median
    candidate_counts.append(median_count)
    
    # Add mode if it appears frequently enough
    for count, freq in most_common:
        if freq >= len(combined_counts) * 0.15:  # Lower threshold - 15%
            candidate_counts.append(count)
    
    # Add max if it appears in at least 20% of frames
    max_freq = combined_counts.count(max_count) / len(combined_counts)
    if max_freq >= 0.2:
        candidate_counts.append(max_count)

    final_count = int(round(max(candidate_counts)))

    # Override logic: if max_count is significantly higher and appears FREQUENTLY enough
    if max_count >= 3:
        bbox_3plus = sum(1 for c in bbox_counts if c >= 3)
        pose_3plus = sum(1 for c in pose_counts if c >= 3)
        combined_3plus = sum(1 for c in combined_counts if c >= 3)
        total = len(combined_counts)
        
        bbox_3_pct = bbox_3plus / total if total > 0 else 0
        pose_3_pct = pose_3plus / total if total > 0 else 0
        combined_3_pct = combined_3plus / total if total > 0 else 0
        
        print(f"  📊 3+ frequency: bbox={bbox_3_pct:.0%} ({bbox_3plus}/{total}), pose={pose_3_pct:.0%} ({pose_3plus}/{total}), combined={combined_3_pct:.0%} ({combined_3plus}/{total})")
        
        # Require 3+ to appear in at least 15% of frames from ANY method
        if bbox_3_pct >= 0.15 or pose_3_pct >= 0.15 or combined_3_pct >= 0.15:
            print(f"  ⚠️ Strong evidence of 3: sufficient frequency in samples")
            final_count = max(final_count, 3)
        else:
            print(f"  ℹ️ Max=3 but too rare ({combined_3_pct:.0%}) - not overriding")

    # Special case: if mean is 2.3+ and max is 3+, likely 3 people
    # Also require 3+ in at least 10% of frames
    combined_3_freq = sum(1 for c in combined_counts if c >= 3) / len(combined_counts) if combined_counts else 0
    if mean_count >= 2.3 and max_count >= 3 and final_count < 3 and combined_3_freq >= 0.10:
        print(f"  ⚠️ Overriding to 3: mean={mean_count:.1f}, max={max_count}, freq={combined_3_freq:.0%}")
        final_count = 3

    if return_details:
        return {
            'final_count': final_count,
            'raw_counts': combined_counts,
            'bbox_counts': bbox_counts,
            'pose_counts': pose_counts,
            'stats': {
                'mean': float(mean_count),
                'median': float(median_count),
                'max': int(max_count),
                'most_common': most_common
            },
            'frame_details': frame_details,
            'method': 'enhanced-multi-method-pose-validated',
            'total_pose_filtered': total_filtered
        }
    
    return final_count


def cluster_keypoints_by_person(all_keypoints, min_keypoints=3, radius=100):
    """
    Cluster pose keypoints into separate people based on spatial proximity.
    Handles cases where bbox detection misses someone but pose keypoints are visible.
    """
    clusters = []
    used_keypoints = set()
    
    for person_idx, keypoints in enumerate(all_keypoints):
        # Get confident keypoints
        confident_kps = []
        for kp_idx, kp in enumerate(keypoints):
            if kp[2] > 0.3:  # Confidence threshold
                confident_kps.append((kp[0], kp[1], kp_idx))
        
        if len(confident_kps) < min_keypoints:
            continue
        
        # Check if this cluster overlaps with existing clusters
        is_new_person = True
        for cluster in clusters:
            # Check distance to cluster centroid
            cluster_center = np.mean([[kp[0], kp[1]] for kp in cluster['keypoints']], axis=0)
            person_center = np.mean([[kp[0], kp[1]] for kp in confident_kps], axis=0)
            
            distance = np.linalg.norm(cluster_center - person_center)
            
            if distance < radius:
                # Merge into existing cluster
                cluster['keypoints'].extend(confident_kps)
                is_new_person = False
                break
        
        if is_new_person:
            clusters.append({
                'keypoints': confident_kps,
                'center': np.mean([[kp[0], kp[1]] for kp in confident_kps], axis=0)
            })
    
    return clusters


def count_additional_from_partials(full_detections, partial_detections, frame_w, frame_h):
    """
    Count how many additional people are represented by partial detections.
    Only count partials that are NOT overlapping with full detections.
    """
    if not partial_detections:
        return 0
    
    additional = 0
    
    for partial in partial_detections:
        px1, py1, px2, py2 = partial['box']
        
        # Check if this partial overlaps with any full detection
        overlaps_with_full = False
        for full_box, _ in full_detections:
            fx1, fy1, fx2, fy2 = full_box
            
            # Calculate IoU
            iou = calculate_iou(partial['box'], full_box)
            
            if iou > 0.1:  # 10% overlap
                overlaps_with_full = True
                break
        
        # If it doesn't overlap, it might be an additional person
        if not overlaps_with_full:
            # Additional heuristics:
            # - Legs-only detections at bottom of frame
            # - Torso detections that are substantial
            
            if partial['type'] == 'legs':
                # Legs should be in bottom 60% of frame
                if py1 > frame_h * 0.4:
                    additional += 1
            elif partial['type'] == 'torso':
                # Torso should be substantial
                area = (px2 - px1) * (py2 - py1)
                if area > (frame_w * frame_h) * 0.02:
                    additional += 1
    
    # Cap at 1 additional person from partials to avoid over-counting
    return min(additional, 1)


def has_evidence_of_third_person(full_detections, partial_detections, keypoint_clusters, frame_w, frame_h):
    """
    IMPROVED: More aggressive 3rd person detection
    """
    if len(full_detections) != 2:
        return False
    
    # Check 1: Partial detections (✅ IMPROVED distance threshold)
    if partial_detections:
        for partial in partial_detections:
            px1, py1, px2, py2 = partial['box']
            pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
            
            min_dist = float('inf')
            for full_box, _ in full_detections:
                fx1, fy1, fx2, fy2 = full_box
                fcx, fcy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
                
                dist = np.sqrt((pcx - fcx)**2 + (pcy - fcy)**2)
                min_dist = min(min_dist, dist)
            
            # ✅ IMPROVED: Lower threshold (20% vs 25%)
            if min_dist > frame_w * 0.20:  # Was 0.25
                return True
    
    # Check 2: Keypoint clusters
    if len(keypoint_clusters) > 2:
        return True
    
    # Check 3: Spatial arrangement (✅ IMPROVED middle zone detection)
    (box1, _), (box2, _) = full_detections
    x1_center = (box1[0] + box1[2]) / 2
    x2_center = (box2[0] + box2[2]) / 2
    
    if abs(x1_center - x2_center) > frame_w * 0.5:
        # ✅ IMPROVED: Wider middle zone
        middle_zone = (min(x1_center, x2_center) + abs(x1_center - x2_center) * 0.20,  # Was 0.25
                      min(x1_center, x2_center) + abs(x1_center - x2_center) * 0.80)  # Was 0.75
        
        for partial in partial_detections:
            px1, px2 = partial['box'][0], partial['box'][2]
            pcx = (px1 + px2) / 2
            
            if middle_zone[0] < pcx < middle_zone[1]:
                return True
    
    return False



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
        self.last_centers = [None] * max_actions

    def update(self, boxes, frame_shape, frame_idx, crop_count=3, positions=None):
        """
        Update tracker with boxes.

        KEY RULE:
        - positions defines which slots are active and in what OUTPUT order.
        - mapping: left->0, middle/center->1, right->2
        - returns regions in the SAME ORDER as `positions`
        """
        h, w = frame_shape[:2]

        # 1) Normalize positions
        if positions:
            positions = ["middle" if p == "center" else p for p in positions]
        else:
            positions = ["left", "middle", "right"] if crop_count == 3 else (["left", "right"] if crop_count == 2 else ["middle"])

        pos_to_idx = {"left": 0, "middle": 1, "right": 2}
        active_actions_indicies = [pos_to_idx[p] for p in positions if p in pos_to_idx]

        if not active_actions_indicies:
            active_actions_indicies = [0, 1, 2] if crop_count == 3 else ([0, 2] if crop_count == 2 else [1])
            positions = ["left", "middle", "right"] if crop_count == 3 else (["left", "right"] if crop_count == 2 else ["middle"])

        # 2) Build zone targets
        slot_targets = []
        for p in positions:
            if p == "left":
                slot_targets.append(w * (1/6))
            elif p == "middle":
                slot_targets.append(w * 0.5)
            else:
                slot_targets.append(w * (5/6))

        # 3) Classify slots: established vs new
        max_jump_px = w * MAX_JUMP_RATIO
        slot_established = [False] * len(positions)
        slot_last_cx = [None] * len(positions)

        for slot_i, action_idx in enumerate(active_actions_indicies):
            has_history = len(self.histories[action_idx]) >= JUMP_RESISTANCE_MIN_HISTORY
            if has_history and self.last_centers[action_idx] is not None:
                slot_established[slot_i] = True
                slot_last_cx[slot_i] = self.last_centers[action_idx]

        # 4) TWO-PHASE ASSIGNMENT
        assigned_per_slot = [None] * len(positions)
        used_boxes = set()

        # Phase 1: Established slots grab only NEARBY detections
        if boxes:
            for slot_i in range(len(positions)):
                if not slot_established[slot_i]:
                    continue
                last_cx = slot_last_cx[slot_i]
                best_box = None
                best_dist = float('inf')
                best_box_idx = -1

                for box_idx, box in enumerate(boxes):
                    if box_idx in used_boxes:
                        continue
                    cx = (box[0] + box[2]) / 2.0
                    dist = abs(cx - last_cx)
                    if dist <= max_jump_px and dist < best_dist:
                        best_dist = dist
                        best_box = box
                        best_box_idx = box_idx

                if best_box is not None:
                    assigned_per_slot[slot_i] = best_box
                    used_boxes.add(best_box_idx)

        # Phase 2: Unestablished slots use zone-target matching
        if boxes:
            for slot_i in range(len(positions)):
                if slot_established[slot_i]:
                    continue
                if assigned_per_slot[slot_i] is not None:
                    continue
                best_box = None
                best_dist = float('inf')
                best_box_idx = -1

                for box_idx, box in enumerate(boxes):
                    if box_idx in used_boxes:
                        continue
                    cx = (box[0] + box[2]) / 2.0
                    dist = abs(cx - slot_targets[slot_i])
                    if dist < best_dist:
                        best_dist = dist
                        best_box = box
                        best_box_idx = box_idx

                if best_box is not None:
                    assigned_per_slot[slot_i] = best_box
                    used_boxes.add(best_box_idx)

        # Map into action-index boxes
        action_boxes = [None] * self.max_actions
        for slot_i, action_idx in enumerate(active_actions_indicies):
            action_boxes[action_idx] = assigned_per_slot[slot_i]

        # 5) Prevent overlap
        active_boxes_in_output_order = [action_boxes[idx] for idx in active_actions_indicies]
        active_boxes_in_output_order = prevent_overlap(active_boxes_in_output_order, w)
        for slot_i, action_idx in enumerate(active_actions_indicies):
            action_boxes[action_idx] = active_boxes_in_output_order[slot_i]

        # 6) Update tracking state
        for action_idx in active_actions_indicies:
            if action_boxes[action_idx] is not None:
                box = self._fine_tune_box(action_boxes[action_idx], action_idx, (h, w))
                self.histories[action_idx].append(box)
                self.confidences[action_idx] = min(self.confidences[action_idx] + 1, 10)
                self.missing_counters[action_idx] = 0
                self.last_centers[action_idx] = (box[0] + box[2]) / 2.0
            else:
                self.missing_counters[action_idx] += 1

        # Lock actions when confirmed
        for action_idx in active_actions_indicies:
            if (not self.actions_confirmed[action_idx] and
                self.confidences[action_idx] >= 8 and
                len(self.histories[action_idx]) >= 15):
                self.locked_actions[action_idx] = self._get_optimal_box(
                    self.histories[action_idx], action_idx, (h, w)
                )
                self.actions_confirmed[action_idx] = True
                print(f"🎯 Locked Action idx={action_idx} ({positions[active_actions_indicies.index(action_idx)]})")

        # 7) Return current regions
        regions = []
        for action_idx in active_actions_indicies:
            if self.actions_confirmed[action_idx] and self.locked_actions[action_idx]:
                regions.append(self.locked_actions[action_idx])
            elif self.histories[action_idx]:
                regions.append(self._get_median_box(self.histories[action_idx]))
            else:
                regions.append(None)

        return prevent_overlap(regions, w)

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

    def _get_current_regions(self, h, w, crop_count=3, positions=None):
        regions = []
        if positions:
            positions = ["middle" if p == "center" else p for p in positions]
            pos_to_idx = {"left": 0, "middle": 1, "right": 2}
            indices = [pos_to_idx[p] for p in positions if p in pos_to_idx]
        else:
            indices = [0, 1, 2] if crop_count == 3 else [0, 2]
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
        self.roi_detector = ROIDetector(debug=False) if use_roi_detection else None

        # Initialize ROI detector if enabled
        self.roi_detector = ROIDetector(debug=False) if use_roi_detection else None

    def detect(self, frame, detector, crop_count=3, pose_model=None, positions=None):
        """
        Detect actions with ROI-based focusing.

        UPDATED:
        - Keeps torso-only and legs-only detections (partial people).
        - Uses separate shape/size heuristics to avoid garbage boxes.
        - Allows lower confidence for partials without flooding full detections.
        """
        self.frame_idx += 1
        h, w = frame.shape[:2]

        # 1) Normalize positions
        if positions:
            positions = ["middle" if p == "center" else p for p in positions]
        else:
            positions = (
                ["left", "middle", "right"] if crop_count == 3
                else (["left", "right"] if crop_count == 2 else ["middle"])
            )

        pos_to_idx = {"left": 0, "middle": 1, "right": 2}
        active_actions_indicies = [pos_to_idx[p] for p in positions if p in pos_to_idx]

        if not active_actions_indicies:
            active_actions_indicies = [0, 1, 2] if crop_count == 3 else ([0, 2] if crop_count == 2 else [1])
            positions = ["left", "middle", "right"] if crop_count == 3 else (["left", "right"] if crop_count == 2 else ["middle"])

        # 2) YOLO detections
        TRACK_CONF = PERSON_DETECTION_CONF_TRACKING
        result = detector.predict(frame, conf=TRACK_CONF, classes=[0], verbose=False)

        boxes = []
        frame_area = float(h * w)

        CONF_FULL = max(0.10, TRACK_CONF)
        CONF_PARTIAL = max(0.06, TRACK_CONF)

        def touches_border(x1, y1, x2, y2, margin=0.03):
            return (
                x1 < w * margin or x2 > w * (1 - margin) or
                y1 < h * margin or y2 > h * (1 - margin)
            )

        for r in result:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf)
                box_w = max(1, x2 - x1)
                box_h = max(1, y2 - y1)
                area = box_w * box_h
                area_ratio = area / frame_area
                aspect = box_w / float(box_h)
                border = touches_border(x1, y1, x2, y2)

                is_fullish = (
                    area_ratio > 0.015 and
                    0.35 < aspect < 3.5 and
                    box_h > h * 0.20
                )
                is_legs = (
                    area_ratio > 0.0015 and
                    aspect < 0.45 and
                    box_h > h * 0.20
                )
                is_torso = (
                    area_ratio > 0.0020 and
                    aspect > 1.2 and
                    box_w > w * 0.10 and
                    box_h > h * 0.12
                )
                accept_partial = (is_legs or is_torso)

                keep = False
                if is_fullish and conf >= CONF_FULL:
                    keep = True
                elif accept_partial and conf >= CONF_PARTIAL:
                    keep = True

                if keep and area_ratio < 0.70:
                    boxes.append((x1, y1, x2, y2))

        # 3) ROI detection: Don't filter boxes, only use ROI as hint
        #    The tracker's jump resistance handles assignment correctly.
        #    ROI filtering was removing boxes from established regions.
        action_roi = None
        if self.use_roi_detection and self.roi_detector and len(boxes) > 0:
            action_roi, focus_region = self.roi_detector.detect_action_roi(
                frame, boxes, pose_model, max_people=crop_count
            )

        # 4) Update tracker (with jump resistance built in)
        actions = self.tracker.update(
            boxes,
            (h, w),
            self.frame_idx,
            crop_count,
            positions=positions
        )

        # 5) Pose activity / tracking stats
        pose_data = {}
        if pose_model and USE_POSE_ESTIMATION:
            try:
                pose_result = pose_model.predict(frame, conf=0.3, verbose=False)
                for pr in pose_result:
                    if hasattr(pr, 'keypoints') and pr.keypoints is not None:
                        for kp, box in zip(pr.keypoints.data, pr.boxes.xyxy):
                            px1, py1, px2, py2 = map(int, box)
                            pose_data[(px1, py1, px2, py2)] = kp.cpu().numpy()
            except Exception as e:
                print(f"⚠️ Pose estimation failed: {e}")

        for slot_i, action_idx in enumerate(active_actions_indicies):
            action = actions[slot_i] if slot_i < len(actions) else None

            if action is not None:
                activity = 0.0
                if pose_data:
                    best_match = None
                    best_overlap = 0.0
                    for pose_box, keypoints in pose_data.items():
                        overlap = calculate_iou(action, pose_box)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_match = keypoints
                    if best_match is not None and best_overlap > 0.2:
                        activity = analyze_pose_activity(best_match, action)

                self.pose_activities[action_idx] = activity
                self.motion_histories[action_idx].append(action)
                self.last_good_actions[action_idx] = action
                self.missing_counters[action_idx] = 0
            else:
                self.missing_counters[action_idx] += 1

        # 6) Fill missing actions with fallbacks
        final_actions = []
        for slot_i, action_idx in enumerate(active_actions_indicies):
            if slot_i < len(actions) and actions[slot_i] is not None:
                final_actions.append(actions[slot_i])
            else:
                final_actions.append(self._get_fallback(
                    action_idx,
                    (h, w),
                    positions=positions,
                    crop_count=crop_count
                ))
                self.pose_activities[action_idx] = 0.0

        return prevent_overlap(final_actions, w)


    def _get_fallback(self, action_idx, frame_shape, positions=None, crop_count=3):
        """
        fallback: Use last_good_actions if available.
        If no history exists, return None (black frame) — don't invent positions.
        """
        # ONLY option: use last good tracked position
        if self.last_good_actions[action_idx] is not None:
            return self.last_good_actions[action_idx]

        # No history = nothing to show. Return None → black frame until real detection.
        return None

    def _is_head_focused(self, position):
        return True

    def _get_legacy_fallback(self, action_idx, frame_shape):
        h, w = frame_shape
        default_size = int(min(h, w) // 2.5)
        vertical_offset = int(h * 0.45)
        if action_idx == 0:
            return (int(w//8), vertical_offset, int(w//8 + default_size), vertical_offset + default_size)
        elif action_idx == 1:
            return (int(w//2 - default_size//2), vertical_offset, int(w//2 + default_size//2), vertical_offset + default_size)
        else:
            return (int(w*7//8 - default_size), vertical_offset, int(w*7//8), vertical_offset + default_size)

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
    
    # REDUCE MINIMUM DIMENSIONS for head-focused crops
    # Check if this looks like a head crop (based on aspect ratio)
    original_box_w = x2 - x1
    original_box_h = y2 - y1
    
    # If it's a portrait-oriented box, it might be head-focused
    is_portrait = original_box_h > original_box_w * 1.2
    
    if is_portrait:
        # For head crops, allow smaller minimums
        min_width = int(w * 0.25)  # Reduced from 0.30
        min_height = int(h * 0.28)  # Reduced from 0.30
    else:
        # For regular crops, use standard minimums
        min_width = int(w * 0.30)
        min_height = int(h * 0.30)
    
    box_w = x2 - x1
    box_h = y2 - y1
    
    # Only enforce minimums if box is REALLY small
    if box_w < min_width and box_w < w * 0.2:  # Only expand if very small
        cx = (x1 + x2) // 2
        x1 = int(max(0, cx - min_width//2))
        x2 = int(min(w, cx + min_width//2))
        box_w = x2 - x1
    
    if box_h < min_height and box_h < h * 0.2:  # Only expand if very small
        cy = (y1 + y2) // 2
        y1 = int(max(0, cy - min_height//2))
        y2 = int(min(h, cy + min_height//2))
        box_h = y2 - y1
    
    # Make aspect ratio requirements more lenient
    aspect_ratio = box_w / max(box_h, 1)
    
    if aspect_ratio > 2.0:  # Too wide
        target_h = int(box_w / 1.2)  # More lenient than 1.5
        cy = (y1 + y2) // 2
        y1 = int(max(0, cy - target_h//2))
        y2 = int(min(h, cy + target_h//2))
    elif aspect_ratio < 0.4:  # More lenient than 0.5 (allow taller boxes)
        target_w = int(box_h * 0.7)  # More lenient than 0.8
        cx = (x1 + x2) // 2
        x1 = int(max(0, cx - target_w//2))
        x2 = int(min(w, cx + target_w//2))
    
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Final validation
    if x2 <= x1 or y2 <= y1:
        size = int(min(h, w) * default_scale)
        x1 = int(w//2 - size//2)
        y1 = int(h//2 - size//2)
        return frame[y1:y1+size, x1:x1+size]
    
    return frame[y1:y2, x1:x2]

def expand_box(box, frame_shape, frame_count, action_idx=0, margin=0.2, is_fallback=False):
    if box is None:
        return None
    
    # ✅ ACTUALLY EXPAND THE GIVEN BOX
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = map(int, box)
    
    # ✅ Use FALLBACK_BOX_EXPANSION if is_fallback is True
    if is_fallback:
        margin = FALLBACK_BOX_EXPANSION
    
    # DEBUG
    if frame_count % 30 == 0 and is_fallback:
        print(f"DEBUG expand_box: Frame {frame_count}, idx={action_idx}, margin={margin:.2f}")
        print(f"  Original: ({x1}, {y1}, {x2}, {y2}) size: {x2-x1}x{y2-y1}")
    
    # Calculate expansion amounts
    bw, bh = x2 - x1, y2 - y1
    ew = int(bw * margin)
    eh = int(bh * margin)
    
    # DEBUG - Show raw expansion before clipping
    if frame_count % 30 == 0 and is_fallback:
        print(f"  Raw expansion: {ew} horizontal, {eh} vertical")
    
    # Cap expansion to ensure box stays within frame
    max_left_expansion = x1  # Can't go more left than 0
    max_right_expansion = w - x2  # Can't go more right than w
    
    # For boxes near edges, distribute expansion more evenly
    if action_idx == 0:  # Left crop
        # Left box: more expansion on right side, less on left
        left_exp = min(ew // 3, max_left_expansion)
        right_exp = min(ew, max_right_expansion)
        if frame_count % 30 == 0 and is_fallback:
            print(f"  Left crop: left_exp={left_exp}, right_exp={right_exp}")
    elif action_idx == 2:  # Right crop
        # Right box: more expansion on left side, less on right
        left_exp = min(ew, max_left_expansion)
        right_exp = min(ew // 3, max_right_expansion)
        if frame_count % 30 == 0 and is_fallback:
            print(f"  Right crop: left_exp={left_exp}, right_exp={right_exp}")
            print(f"  Distance to right edge: {w - x2}px, max_right_expansion={max_right_expansion}")
    else:  # Middle crop
        # Middle box: equal expansion both sides
        left_exp = min(ew // 2, max_left_expansion)
        right_exp = min(ew // 2, max_right_expansion)
        if frame_count % 30 == 0 and is_fallback:
            print(f"  Middle crop: left_exp={left_exp}, right_exp={right_exp}")
    
    # Same for vertical expansion
    max_top_expansion = y1
    max_bottom_expansion = h - y2
    top_exp = min(eh // 2, max_top_expansion)
    bottom_exp = min(eh // 2, max_bottom_expansion)
    
    # Apply the expansion
    x1_new = int(max(0, x1 - left_exp))
    x2_new = int(min(w, x2 + right_exp))
    y1_new = int(max(0, y1 - top_exp))
    y2_new = int(min(h, y2 + bottom_exp))
    
    # DEBUG - Show after expansion
    if frame_count % 30 == 0 and is_fallback:
        print(f"  After expansion: ({x1_new}, {y1_new}, {x2_new}, {y2_new}) size: {x2_new-x1_new}x{y2_new-y1_new}")
    
    # Ensure minimum size for all boxes
    min_width = int(w * 0.3)
    min_height = int(h * 0.3)
    
    if (x2_new - x1_new) < min_width:
        center_x = (x1_new + x2_new) // 2
        x1_new = max(0, center_x - min_width // 2)
        x2_new = min(w, center_x + min_width // 2)
        if frame_count % 30 == 0 and is_fallback:
            print(f"  Adjusted width to minimum: {min_width}")
    
    if (y2_new - y1_new) < min_height:
        center_y = (y1_new + y2_new) // 2
        y1_new = max(0, center_y - min_height // 2)
        y2_new = min(h, center_y + min_height // 2)
        if frame_count % 30 == 0 and is_fallback:
            print(f"  Adjusted height to minimum: {min_height}")
    
    return (x1_new, y1_new, x2_new, y2_new)
    
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
    """Calibration that adapts to crop count and rounds to standard resolutions"""
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

    # Standard resolutions in the 480-720 range
    STANDARD_RESOLUTIONS = [
        (480, 480),    # Square
        (640, 480),    # 4:3
        (720, 480),    # 3:2 (DV NTSC)
        (640, 360),    # 16:9 (360p)
        (854, 480),    # 16:9 (480p)
        (720, 540),    # 4:3
        (720, 720),    # Square HD
        (960, 540),    # qHD
    ]
    
    # Rounding tolerance (within 15% of target size)
    TOLERANCE = 0.15

    if all_sizes:
        # Calculate target size based on percentiles
        target_w = int(np.percentile([w for w, h in all_sizes], 80))
        target_h = int(np.percentile([h for w, h in all_sizes], 80))

        # Calculate aspect ratio
        aspect = target_w / max(target_h, 1)
        
        print(f"🔧 Raw calibration: {target_w}x{target_h} (aspect: {aspect:.2f})")
        
        # Find the closest standard resolution
        best_res = None
        best_score = float('inf')
        
        for std_w, std_h in STANDARD_RESOLUTIONS:
            # Check if within aspect ratio tolerance
            std_aspect = std_w / std_h
            aspect_diff = abs(aspect - std_aspect)
            
            if aspect_diff > 0.2:  # Skip if aspect ratio is too different
                continue
            
            # Calculate size difference score
            size_score = abs(target_w - std_w) / target_w + abs(target_h - std_h) / target_h
            
            # Prioritize resolutions within tolerance
            if (abs(target_w - std_w) / target_w <= TOLERANCE and 
                abs(target_h - std_h) / target_h <= TOLERANCE):
                size_score *= 0.5  # Prefer resolutions within tolerance
            
            if size_score < best_score:
                best_score = size_score
                best_res = (std_w, std_h)
        
        # If no good match found, use the calculated size but round to nearest standard
        if best_res is None:
            print(f"⚠️ No standard resolution match found for {target_w}x{target_h}")
            
            # Round to nearest standard width/height separately
            std_widths = [480, 640, 720, 854, 960]
            std_heights = [360, 480, 540, 720]
            
            # Find closest standard width
            closest_w = min(std_widths, key=lambda x: abs(x - target_w))
            
            # Find closest standard height that maintains reasonable aspect
            target_aspect = target_w / target_h
            best_h = None
            best_aspect_diff = float('inf')
            
            for h in std_heights:
                aspect = closest_w / h
                aspect_diff = abs(aspect - target_aspect)
                if aspect_diff < best_aspect_diff:
                    best_aspect_diff = aspect_diff
                    best_h = h
            
            best_res = (closest_w, best_h)
        
        target_w, target_h = best_res
        
        # Ensure minimum size
        target_w = max(target_w, 400)
        target_h = max(target_h, 400)
        
        print(f"✅ Rounded to standard: {target_w}x{target_h}")
        return (target_w, target_h)

    # Fallback to standard 480p if no detection
    print("⚠️ No detections for calibration, using default 480p")
    return (854, 480)

def is_currently_using_tracked_box(detector, action_idx: int) -> bool:
    """MAIN logic: True if currently re-using a previously tracked (missing) box"""
    if action_idx >= len(detector.missing_counters):
        return True
    return detector.missing_counters[action_idx] > 0


def has_good_tracking_quality(detector, action_idx: int) -> bool:
    """FALLBACK/best logic: True if track has sufficient history and isn't stale"""
    if action_idx >= len(detector.motion_histories):
        return False
    history = detector.motion_histories[action_idx]
    missing = detector.missing_counters[action_idx]
    return len(history) >= 5 and missing < 10


def visualize_crop_process(frame, frame_idx, yolo_boxes, expanded_boxes, smoothed_boxes, 
                          final_boxes, action_statuses, positions, debug_info=None):
    """
    Create debug visualization showing the crop process step by step.
    """
    # Create a copy of the frame for visualization
    vis_frame = frame.copy()
    
    # 1. Draw original YOLO detections (RED)
    for i, box in enumerate(yolo_boxes):
        if box:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(vis_frame, f"YOLO {i}", (x1, max(20, y1-5)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # 2. Draw expanded boxes (YELLOW)
    for i, box in enumerate(expanded_boxes):
        if box:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(vis_frame, f"Exp {i}", (x1, max(40, y1-25)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # 3. Draw smoothed boxes (GREEN)
    for i, box in enumerate(smoothed_boxes):
        if box:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Smoothed {i}", (x1, max(60, y1-45)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 4. Draw final crop regions with status-based colors
    for i, (box, status) in enumerate(zip(final_boxes, action_statuses)):
        if box:
            x1, y1, x2, y2 = map(int, box)
            
            # Choose color based on tracking status
            if status in ["FRESH_DETECTION", "TRACKED-good"]:
                color = (255, 0, 0)  # BLUE for good tracking
            elif status in ["PURE_FALLBACK", "FRESH_FALLBACK", "TRACKED-poor"]:
                color = (255, 0, 255)  # MAGENTA for fallback
            else:
                color = (255, 255, 255)  # WHITE for unknown
            
            # Draw thicker box for final crop
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)
            
            # Add position and status text
            position = positions[i] if i < len(positions) else f"Pos{i}"
            status_text = status.replace("_", " ")
            cv2.putText(vis_frame, f"{position}: {status_text}", 
                       (x1, max(80, y1-65)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.circle(vis_frame, (center_x, center_y), 5, color, -1)
    
    # Add frame info overlay
    h, w = frame.shape[:2]
    info_y = 30
    
    # Create semi-transparent overlay for text
    overlay = vis_frame.copy()
    cv2.rectangle(overlay, (10, 10), (w-10, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)
    
    # Add debug information
    cv2.putText(vis_frame, f"Frame: {frame_idx}", (20, info_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if debug_info:
        for j, (key, value) in enumerate(debug_info.items()):
            cv2.putText(vis_frame, f"{key}: {value}", (20, info_y + 30 + j*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add legend
    legend_y = h - 150
    legend_items = [
        ("RED", (0, 0, 255), "YOLO Detection"),
        ("YELLOW", (0, 255, 255), "Expanded Box"),
        ("GREEN", (0, 255, 0), "Smoothed Box"),
        ("BLUE", (255, 0, 0), "Good Tracking"),
        ("MAGENTA", (255, 0, 255), "Fallback"),
    ]
    
    for i, (label, color, desc) in enumerate(legend_items):
        cv2.rectangle(vis_frame, (20, legend_y + i*25 - 15), (50, legend_y + i*25 + 5), color, -1)
        cv2.putText(vis_frame, f"{desc}", (60, legend_y + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis_frame


def process_video_with_dynamic_crops(input_path, output_folder, yolo_model, crop_count, 
                                    positions_override=None, people_info=None):
    """Process video with dynamic number of crops (2 or 3) with ROI detection and debug visualization"""
    # Use override positions if provided, otherwise use default
    if positions_override:
        positions = positions_override
        # Map 'center' to 'middle' for consistency with existing code
        positions = ['middle' if pos == 'center' else pos for pos in positions]
    else:
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
    
    # Create debug folder for this video if debug mode is enabled
    debug_video_folder = None
    if DEBUG_MODE:
        debug_video_folder = os.path.join(DEBUG_OUTPUT_FOLDER, base_name)
        os.makedirs(debug_video_folder, exist_ok=True)
        print(f"📊 Debug visualization enabled: {debug_video_folder}")

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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writers = []
    for output_path in output_files:
        writer = cv2.VideoWriter(output_path, fourcc, fps, TARGET_SIZE)
        writers.append(writer)

    # Create debug video writer
    debug_writer = None
    debug_path = None

    frame_count = 0
    debug_sample_count = 0
    print(f"📹 Processing with synchronized {position_text} (ROI detection: {'ON' if USE_ROI_DETECTION else 'OFF'})...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Initialize debug video writer on first frame
        if DEBUG_MODE and DEBUG_CREATE_VIDEOS and debug_writer is None:
            h, w = frame.shape[:2]
            os.makedirs(DEBUG_VIDEO_FOLDER, exist_ok=True)
            debug_writer, debug_path = create_debug_video_writer(
                input_path, DEBUG_VIDEO_FOLDER, fps, (h, w)
            )
            print(f"📹 Creating debug video: {os.path.basename(debug_path)}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get YOLO detections for debug visualization
        yolo_boxes = []
        need_frame_people_count = DEBUG_MODE and (DEBUG_CREATE_VIDEOS or DEBUG_SHOW_METRICS)

        if need_frame_people_count:
            result = yolo_model.predict(rgb, conf=PERSON_DETECTION_CONF_TRACKING, classes=[0], verbose=False)
            for r in result:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
                    area = box_w * box_h
                    frame_area = frame.shape[0] * frame.shape[1]
                    area_ratio = area / frame_area
                    aspect = box_w / float(box_h)

                    # Match the SAME filters as detect() so debug shows what tracker sees
                    is_fullish = (area_ratio > 0.015 and 0.35 < aspect < 3.5 and box_h > frame.shape[0] * 0.20)
                    is_legs = (area_ratio > 0.0015 and aspect < 0.45 and box_h > frame.shape[0] * 0.20)
                    is_torso = (area_ratio > 0.0020 and aspect > 1.2 and box_w > frame.shape[1] * 0.10 and box_h > frame.shape[0] * 0.12)

                    if (is_fullish or is_legs or is_torso) and area_ratio < 0.70:
                        yolo_boxes.append((x1, y1, x2, y2))

        # Build per-frame people_info for overlay (don't mutate shared dict)
        frame_people_info = None
        if people_info is not None and need_frame_people_count:
            frame_people_info = dict(people_info)
            frame_people_info["current_frame_detected"] = len(yolo_boxes)


        
        # Detect actions with ROI-based detector
        actions = detector.detect(rgb, yolo_model, crop_count=crop_count, pose_model=pose_model, positions=positions)

        expanded_actions = []
        action_indices = []
        action_statuses = []  # Track status for each action
        
        # Determine action indices OUTSIDE the loop
        if crop_count == 3:
            action_indices = [0, 1, 2]
        elif crop_count == 2:
            # Map positions to indices: left=0, center=1, right=2
            if positions == ['left', 'right']:
                action_indices = [0, 2]
            elif positions in (['left', 'center'], ['left', 'middle']):
                action_indices = [0, 1]
            elif positions in (['center', 'right'], ['middle', 'right']):
                action_indices = [1, 2]
            else:
                action_indices = [0, 2]
        
        # Main processing loop
        for i, action_idx in enumerate(action_indices):
            if i < len(actions) and actions[i] is not None:
                current_box = actions[i]
                missing = detector.missing_counters[action_idx] if action_idx < len(detector.missing_counters) else 0
                hist_len = len(detector.motion_histories[action_idx]) if action_idx < len(detector.motion_histories) else 0
                pose_activity = detector.pose_activities[action_idx] if action_idx < len(detector.pose_activities) else 0.0
                history = detector.motion_histories[action_idx] if action_idx < len(detector.motion_histories) else deque(maxlen=10)
                
                # Determine status based on missing counter
                if missing == 0:
                    # We have a fresh detection from YOLO!
                    status = "FRESH_DETECTION"
                    use_fallback_expansion = False
                    
                    # Use small expansion for fresh detections
                    adaptive_margin = calculate_motion_expansion(
                        current_box, history, 
                        base_margin=BOX_EXPANSION,  # 0.20
                        pose_activity=pose_activity
                    )
                    
                elif missing > 0 and missing <= 8 and hist_len >= 5:
                    # TRACKED with good history
                    status = "TRACKED-good"
                    use_fallback_expansion = False
                    adaptive_margin = BOX_EXPANSION  # 0.20
                    
                else:
                    # POOR tracking or stale (missing > 8 or low history)
                    status = "TRACKED-poor"
                    use_fallback_expansion = True
                    adaptive_margin = FALLBACK_BOX_EXPANSION  # 0.50
                    
            else:
                # ✅ PURE FALLBACK - no box from detector at all
                status = "PURE_FALLBACK"
                current_box = detector._get_fallback(
                    action_idx, 
                    frame.shape[:2],
                    positions=positions,
                    crop_count=crop_count
                )
                adaptive_margin = FALLBACK_BOX_EXPANSION  # 0.50
                use_fallback_expansion = True
                
                # Create dummy history for the fallback box
                history = deque(maxlen=10)
                history.append(current_box)

            # Store status for visualization
            action_statuses.append(status)
            
            # ──── Actually expand the box (or None if no detection) ────
            if current_box is not None:
                expanded = expand_box(
                    current_box,
                    frame.shape,
                    frame_count,
                    action_idx=action_idx,
                    margin=adaptive_margin,
                    is_fallback=use_fallback_expansion
                )
                expanded_actions.append(expanded)
            else:
                expanded_actions.append(None)
            
        # This is now outside the loop, as it should be
        h, w = frame.shape[:2]
        expanded_actions = prevent_overlap(expanded_actions, w)
        
        smoothed_actions = smoother.smooth(*expanded_actions)
                
        # Write debug frame with people_info
        if DEBUG_MODE and DEBUG_CREATE_VIDEOS and debug_writer:
            debug_info = {
                "Crop Count": crop_count,
                "Positions": ", ".join(positions),
                "Target": f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}"
            }
            
            debug_frame = create_enhanced_debug_frame(
                frame, frame_count, yolo_boxes, expanded_actions,
                smoothed_actions, smoothed_actions, action_statuses,
                positions, detector, debug_info, frame_people_info
            )
            
            if DEBUG_VIDEO_SIDE_BY_SIDE:
                combined_frame = create_side_by_side_frame(frame.copy(), debug_frame)
                debug_writer.write(combined_frame)
            else:
                debug_writer.write(debug_frame)

        # Save debug visualization for first few samples
        if DEBUG_MODE and debug_sample_count < DEBUG_SAMPLES and frame_count % 30 == 0:
            # Create debug info dictionary
            debug_info = {
                "Crop Count": crop_count,
                "Positions": ", ".join(positions),
                "Frame": f"{frame_count}/{total_frames}",
                "Target Size": f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}"
            }
            
            # Create visualization - pass all required parameters
            vis_frame = visualize_crop_process(
                frame, frame_count, yolo_boxes, expanded_actions, 
                smoothed_actions, smoothed_actions, action_statuses, 
                positions, debug_info
            )
            
            # Save debug image
            debug_filename = f"{base_name}_frame_{frame_count:06d}_debug.jpg"
            debug_path = os.path.join(debug_video_folder, debug_filename)
            cv2.imwrite(debug_path, vis_frame)
            
            print(f"📸 Saved debug visualization: {debug_filename}")
            debug_sample_count += 1
            
            # Also save individual crops for reference
            for i, crop_box in enumerate(smoothed_actions):
                if crop_box is not None and i < len(positions):
                    crop = safe_crop(frame, crop_box, action_idx=i, default_scale=0.25)
                    if crop is not None and crop.size > 0:
                        crop_filename = f"{base_name}_frame_{frame_count:06d}_crop_{positions[i]}.jpg"
                        crop_path = os.path.join(debug_video_folder, crop_filename)
                        cv2.imwrite(crop_path, crop)
        
        # Process each crop
        for i in range(crop_count):
            if i < len(smoothed_actions) and smoothed_actions[i] is not None:
                action_idx = action_indices[i] if i < len(action_indices) else i
                crop = safe_crop(frame, smoothed_actions[i], action_idx=action_idx, default_scale=0.25)
                
                # ✅ ADDED: Check if crop is valid before processing
                if crop is None or crop.size == 0:
                    print(f"⚠️ Frame {frame_count}: Empty crop for idx={action_idx}")
                    padded = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=np.uint8)
                else:
                    padded = pad_to_size(crop, TARGET_SIZE, PADDING_COLOR)
                
                writers[i].write(padded)
            else:
                # No crop available - write black frame with warning
                if frame_count % 60 == 0:
                    print(f"⚠️ Frame {frame_count}: No smoothed action for crop {i}")
                padded = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0], 3), dtype=np.uint8)
                writers[i].write(padded)
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f" Frame {frame_count}/{total_frames}")
    
    cap.release()
    for writer in writers:
        writer.release()

    # Release debug writer
    if DEBUG_MODE and DEBUG_CREATE_VIDEOS and debug_writer:
        debug_writer.release()
        print(f"✅ Debug video saved: {os.path.basename(debug_path)}")

    
    print(f"✅ {position_text} processing complete for {os.path.basename(input_path)}!")
    print(f" Frames processed: {frame_count}")
    
    if DEBUG_MODE and debug_video_folder:
        print(f"📊 Debug visualizations saved to: {debug_video_folder}")
        print(f"📸 Debug samples captured: {debug_sample_count}")
    
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
    print("🚀 Starting SMART batch video processing (Activity-Based Zone Detection)...")
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"ROI detection: {'ENABLED' if USE_ROI_DETECTION else 'DISABLED'}")
    print(f"Smart crop strategy: ENABLED ✨ (Activity-aware, fully automatic)")
    print(f"Pose validation: ENABLED 🔬 (conf < {POSE_VALIDATION_CONF_THRESHOLD} requires pose keypoints)")
    
    if DEBUG_MODE:
        print(f"🔍 DEBUG MODE ENABLED - Visualizing {DEBUG_SAMPLES} samples per video")
        print(f"📁 Debug output folder: {DEBUG_OUTPUT_FOLDER}")
        print("🎨 Visualization colors:")
        print("   RED (0, 0, 255) - Original YOLO detections")
        print("   YELLOW (0, 255, 255) - Expanded boxes")
        print("   GREEN (0, 255, 0) - Smoothed boxes")
        print("   BLUE (255, 0, 0) - Final crop regions (good tracking)")
        print("   MAGENTA (255, 0, 255) - Final crop regions (fallback mode)")
        print("-" * 60)

    print("📦 Loading YOLO model...")
    yolo = YOLO("yolo11n.pt")
    print("✅ YOLO model loaded")

    # Load pose model for activity analysis
    pose_model = None
    if USE_POSE_ESTIMATION or USE_ROI_DETECTION:
        try:
            print("🧍 Loading pose estimation model...")
            pose_model = YOLO("yolo11n-pose.pt")
            print("✅ Pose model loaded")
        except Exception as e:
            print(f"⚠️ Could not load pose model: {e}")
            print("   Continuing without pose estimation")

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

        # ✨ Enhanced people counting
        start_time = time.time()
        people_info = count_people_in_video(
            video_path, yolo, pose_model, 
            sample_frames=PEOPLE_SAMPLE_FRAMES, 
            return_details=True
        )
        people_count = people_info['final_count']
        elapsed = time.time() - start_time
        
        print(f"   👥 Detected {people_count} person(s) in {elapsed:.1f}s")
        print(f"      Method: {people_info['method']}")
        
        # Show detection breakdown if available
        if 'bbox_counts' in people_info and 'pose_counts' in people_info:
            bbox_avg = np.mean(people_info['bbox_counts'])
            pose_avg = np.mean(people_info['pose_counts'])
            print(f"      Avg BBox: {bbox_avg:.1f}, Avg Pose: {pose_avg:.1f}")
        
        # Show pose validation stats
        if 'total_pose_filtered' in people_info and people_info['total_pose_filtered'] > 0:
            print(f"      🔬 Pose validation filtered {people_info['total_pose_filtered']} false positives")

        # STEP 2: Determine crop strategy
        crop_count = 0
        positions = []
        strategy = ""

        if people_count >= MIN_PEOPLE_REQUIRED:
            if people_count >= 4:
                print(f"   👥👥 4+ people detected - analyzing distribution...")

                # Get zone analysis for distribution
                zone_scores, zone_people, zone_activity = analyze_region_activity(
                    video_path, yolo, pose_model, sample_frames=25
                )

                # Calculate average people per zone
                left_avg = np.mean(zone_people['left']) if zone_people['left'] else 0
                center_avg = np.mean(zone_people['center']) if zone_people['center'] else 0
                right_avg = np.mean(zone_people['right']) if zone_people['right'] else 0

                print(f"   📊 Zone distribution: Left={left_avg:.1f}, Center={center_avg:.1f}, Right={right_avg:.1f}")

                # Count zones with significant people presence
                zones_with_people = []
                if left_avg >= 1.0:
                    zones_with_people.append('left')
                if center_avg >= 1.0:
                    zones_with_people.append('center')
                if right_avg >= 1.0:
                    zones_with_people.append('right')

                print(f"   📍 Zones with people (avg >= 1.0): {zones_with_people}")

                # Decision logic for 4+ people
                if len(zones_with_people) >= 3:
                    # People in all 3 zones → use all 3 crops
                    print(f"   🎯 People in all 3 zones → 3 crops")
                    crop_count = 3
                    positions = ['left', 'center', 'right']
                    strategy = "4plus-all-three-zones"

                elif len(zones_with_people) == 2:
                    # People in 2 zones → crop those 2
                    crop_count = 2
                    positions = zones_with_people
                    strategy = f"4plus-two-zones-{'-'.join(zones_with_people)}"
                    print(f"   🎯 People in 2 zones → {positions}")

                elif len(zones_with_people) == 1:
                    # All concentrated in one zone
                    crop_count = 1
                    positions = zones_with_people
                    strategy = f"4plus-concentrated-{zones_with_people[0]}"
                    print(f"   🎯 All concentrated in {zones_with_people[0]}")

                else:
                    # Fallback: use all 3 to be safe
                    print(f"   ⚖️ Can't determine distribution → 3 crops to be safe")
                    crop_count = 3
                    positions = ['left', 'center', 'right']
                    strategy = "4plus-fallback-all-three"

            else:
                # For 2-3 people, use the existing smart strategy
                # PASS people_count to enable smart 2-person logic
                crop_count, positions, strategy = determine_smart_crop_strategy_v2(
                    video_path, yolo, pose_model, sample_frames=20, people_count=people_count
                )
                print(f"   ✅ Strategy: {crop_count}-crop ({strategy})")
                print(f"      Positions: {positions}")
        else:
            # Not enough people
            crop_count = 0
            positions = []
            strategy = f"insufficient-people-{people_count}"
            print(f"   📋 Not enough people for cropping")

        # Add crop strategy to people_info for debugging
        people_info['crop_strategy'] = strategy
        people_info['crop_count'] = crop_count
        people_info['positions'] = positions

        # STEP 3: Process or copy based on strategy
        # Fully automatic - if crop_count is 0, copy; otherwise crop
        if crop_count >= MIN_PEOPLE_REQUIRED and len(positions) >= MIN_PEOPLE_REQUIRED:
            print(f"   🎬 Processing with {crop_count}-crop: {positions}")
            
            # Modify process_video_with_dynamic_crops to accept people_info
            output_files = process_video_with_dynamic_crops(
                video_path,
                OUTPUT_FOLDER,
                yolo,
                crop_count,
                positions_override=positions,
                people_info=people_info,   # ✅ ADD THIS
            )

            
            # Also save people count info to debug folder
            if DEBUG_MODE:
                debug_folder = os.path.join(DEBUG_OUTPUT_FOLDER, os.path.splitext(filename)[0])
                os.makedirs(debug_folder, exist_ok=True)
                
                # Save people count info as JSON
                people_info_path = os.path.join(debug_folder, "people_count_info.json")
                import json
                
                # Convert numpy types to Python native types for JSON serialization
                def convert_to_serializable(obj):
                    if isinstance(obj, (np.integer, np.floating)):
                        return obj.item()
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, list):
                        return [convert_to_serializable(item) for item in obj]
                    elif isinstance(obj, dict):
                        return {key: convert_to_serializable(value) for key, value in obj.items()}
                    else:
                        return obj
                
                serializable_info = convert_to_serializable(people_info)
                
                with open(people_info_path, 'w') as f:
                    json.dump(serializable_info, f, indent=2)
                
                print(f"   💾 Saved people count info to: {os.path.basename(people_info_path)}")
        else:
            reason = "strategy" if crop_count == 0 else "people count"
            print(f"   📋 Copying {filename} as-is (reason: {reason}, strategy: {strategy})")
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
        
        if DEBUG_MODE:
            print(f"🔍 Debug visualizations saved to: {DEBUG_OUTPUT_FOLDER}")
            for video_path in all_handled_videos:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                debug_folder = os.path.join(DEBUG_OUTPUT_FOLDER, base_name)
                if os.path.exists(debug_folder):
                    debug_files = glob.glob(os.path.join(debug_folder, "*_debug.jpg"))
                    print(f"   {base_name}: {len(debug_files)} debug images")

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
    print("Script starting...")
    try:
        main()
    except Exception as e:
        print(f"\n❌ SCRIPT CRASHED: {e}")
        import traceback
        traceback.print_exc()