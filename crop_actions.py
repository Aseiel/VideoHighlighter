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
PEOPLE_SAMPLE_FRAMES = 30

# IMPROVED: Lower confidence thresholds for better partial person detection
PERSON_DETECTION_CONF = 0.25  # Lowered from 0.4 - catches partial people
PERSON_DETECTION_CONF_ZONES = 0.20  # Even lower for zone analysis
MIN_PERSON_AREA_RATIO = 0.001  # Lowered from 0.003 - allows smaller people

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

PERSON_DETECTION_CONF_TRACKING = 0.15 # can affect window size!

# ===== DEBUG VISUALIZATION CONFIG =====
DEBUG_MODE = True  # Set to True to enable debug visualization
DEBUG_SAMPLES = 4  # Number of sample frames to visualize
DEBUG_OUTPUT_FOLDER = "debug_visualizations"  # Folder for debug images

# NEW: Video debug settings
DEBUG_CREATE_VIDEOS = True  # Create full debug videos
DEBUG_VIDEO_FOLDER = "debug_videos"  # Folder for debug videos
DEBUG_VIDEO_SIDE_BY_SIDE = False  # Show original + debug side-by-side
DEBUG_SHOW_METRICS = True  # Show tracking metrics overlay
# ======================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


if DEBUG_MODE and DEBUG_OUTPUT_FOLDER:
    os.makedirs(DEBUG_OUTPUT_FOLDER, exist_ok=True)

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
                               positions, detector, debug_info=None):
    """
    Enhanced debug visualization with comprehensive tracking info
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
    
    # Add comprehensive info overlay
    if DEBUG_SHOW_METRICS:
        vis_frame = add_metrics_overlay(vis_frame, frame_idx, action_statuses, 
                                       positions, detector, debug_info)
    
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


def add_metrics_overlay(frame, frame_idx, action_statuses, positions, detector, debug_info):
    """Add comprehensive metrics overlay"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Create semi-transparent background for metrics panel
    panel_height = 180
    cv2.rectangle(overlay, (0, 0), (w, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Frame info
    cv2.putText(frame, f"Frame: {frame_idx}", (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    if debug_info:
        y_pos = 60
        for key, value in debug_info.items():
            text = f"{key}: {value}"
            cv2.putText(frame, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_pos += 25
    
    # Tracking status for each position
    status_x = w - 350
    status_y = 30
    cv2.putText(frame, "TRACKING STATUS:", (status_x, status_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    for i, (status, pos) in enumerate(zip(action_statuses, positions)):
        y = status_y + 30 + (i * 35)
        
        # Status indicator color
        if "DETECTION" in status or "good" in status:
            status_color = (0, 255, 0)  # Green
            indicator = "●"
        elif "FALLBACK" in status or "poor" in status:
            status_color = (0, 165, 255)  # Orange
            indicator = "◐"
        else:
            status_color = (0, 0, 255)  # Red
            indicator = "○"
        
        # Draw status
        text = f"{indicator} {pos.upper()}: {status.replace('_', ' ')}"
        cv2.putText(frame, text, (status_x, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1)
        
        # Missing frames counter
        if i < len(detector.missing_counters):
            missing = detector.missing_counters[i]
            if missing > 0:
                cv2.putText(frame, f"({missing}f)", (status_x + 240, y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
    
    # Legend at bottom
    legend_y = h - 120
    cv2.rectangle(overlay, (10, legend_y-10), (w-10, h-10), (0, 0, 0), -1)
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
        cv2.rectangle(frame, (x_offset, legend_y-12), (x_offset+25, legend_y+5), color, -1)
        cv2.putText(frame, desc, (x_offset+30, legend_y), 
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
    """
    Calculate Intersection over Union (IoU) between two boxes.
    """
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

        # IMPROVED: Use lower confidence for zone analysis
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
                    # CORNER PATH: More lenient thresholds
                    if area / frame_area >= MIN_PERSON_AREA_RATIO * 0.5:
                        if aspect >= 0.12 and aspect <= 8:
                            if conf > 0.25:
                                raw_boxes.append((x1, y1, x2, y2, conf, True))  # True = is_corner
                else:
                    # REGULAR PATH: Standard thresholds for multi-person scenes
                    if area / frame_area < MIN_PERSON_AREA_RATIO:
                        continue
                    if aspect < 0.15 or aspect > 6:
                        continue
                    raw_boxes.append((x1, y1, x2, y2, conf, False))  # False = not corner

        # CRITICAL FIX: Merge overlapping boxes (removes face+hand false splits)
        boxes = merge_overlapping_boxes(raw_boxes, iou_threshold=0.3)
        corner_boxes = [box for box, is_corner in boxes if is_corner]
        boxes = [box for box, _ in boxes]

        # Get pose data for activity scoring
        pose_data = {}
        if pose_model:
            try:
                pose_result = pose_model.predict(rgb, conf=0.2, verbose=False)
                for pr in pose_result:
                    if hasattr(pr, 'keypoints') and pr.keypoints is not None:
                        for idx, (kp, box) in enumerate(zip(pr.keypoints.data, pr.boxes.xyxy)):
                            x1, y1, x2, y2 = map(int, box)
                            pose_data[(x1, y1, x2, y2)] = kp.cpu().numpy()
            except:
                pass

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
        
        # Create merged bounding box (takes the union of all boxes in group)
        x1_merged = min(box[0] for box in merge_group)
        y1_merged = min(box[1] for box in merge_group)
        x2_merged = max(box[2] for box in merge_group)
        y2_merged = max(box[3] for box in merge_group)
        
        # Keep corner status if ANY box in group is a corner box
        is_corner_merged = any(box[5] for box in merge_group)
        
        merged.append(((x1_merged, y1_merged, x2_merged, y2_merged), is_corner_merged))
    
    return merged


def determine_smart_crop_strategy_v2(video_path, yolo_model, pose_model=None, sample_frames=20, people_count=0):
    """
    ACTION-AWARE cropping: Focus on where actions happen, not just people.
    Returns: (crop_count, positions_to_use, strategy_description)
    
    FIXED: Takes people_count as input to make intelligent decisions about 2-person videos
    """
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
                'has_action': action_consistency > 0.3 or max_activity > 0.7
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
        print(f"   👥 2-person video detected - checking confidence and separation...")
        
        # Calculate total average people across all zones
        total_avg_people = sum(
            zone_action_potential[z]['avg_people'] 
            for z in ['left', 'center', 'right'] 
            if z in zone_action_potential
        )
        
        print(f"   📊 Total avg people across zones: {total_avg_people:.1f}")
        
        # If total average is close to 3 or more, might be 3 people with partial visibility
        # In that case, skip the strict 2-person logic
        if total_avg_people >= 2.5:
            print(f"   ⚠️ Total avg ({total_avg_people:.1f}) suggests possibly 3+ people with partial visibility")
            print(f"   ➡️ Skipping strict 2-person check, using normal action-based logic")
        else:
            # Confident it's actually 2 people - apply strict separation check
            print(f"   ✓ Confident this is actually 2 people (total avg: {total_avg_people:.1f})")
            
            # For 2 people, only crop if they're clearly in DIFFERENT zones
            zones_with_people = []
            for zone in ['left', 'center', 'right']:
                if zone in zone_action_potential:
                    avg_people = zone_action_potential[zone]['avg_people']
                    if avg_people >= 0.7:  # Lowered from 0.8 for better detection
                        zones_with_people.append(zone)
            
            print(f"   📍 Zones with people: {zones_with_people}")
            
            # If both people are in different zones AND there's action in both zones
            if len(zones_with_people) >= 2 and len(action_zones) >= 2:
                # Check if action zones match people zones
                people_action_overlap = [z for z in zones_with_people if z in action_zones]
                
                if len(people_action_overlap) >= 2:
                    print(f"   ✅ 2 people in separate zones with action - WILL CROP")
                    return 2, people_action_overlap[:2], "two-person-separated-actions"
                else:
                    print(f"   📋 2 people but not enough action separation - NO CROP")
                    return 0, [], "two-person-no-action-separation"
            else:
                print(f"   📋 2 people not separated enough - NO CROP")
                return 0, [], "two-person-not-separated"
    
    # ===== ORIGINAL LOGIC FOR 3+ PEOPLE (or when 2-person check skipped) =====
    
    # Case 1: No significant action anywhere
    if len(action_zones) == 0:
        print(f"   📋 No clear action - using default left+right")
        return 2, ['left', 'right'], "no-action-default"
    
    # Case 2: Single action zone
    elif len(action_zones) == 1:
        zone = action_zones[0]
        print(f"   🎯 Single action zone: {zone}")
        return 1, [zone], f"single-action-{zone}"
    
    # Case 3: Two action zones
    elif len(action_zones) == 2:
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
                return 2, ['left', 'center'], "left-center-actions"
                
        elif set(action_zones) == {'center', 'right'}:
            center_density = zone_action_potential['center']['action_density']
            right_density = zone_action_potential['right']['action_density']
            
            if center_density > right_density * 2:
                return 1, ['center'], "center-dominant-action"
            else:
                return 2, ['center', 'right'], "center-right-actions"
        
        else:  # left + right
            return 2, ['left', 'right'], "left-right-actions"
    
    # Case 4: Three action zones
    else:  # all 3 zones have action
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
        """Get ROI based on pose keypoints - FIXED VERSION"""
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
        
        # FIXED: Use reasonable padding
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
    FIXED VERSION: Much more conservative expansion.
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

    # DEBUG: Uncomment to see what's happening
    # print(f"DEBUG calculate_motion_expansion: base={base_margin:.2f}, motion_factor={motion_factor:.3f}, pose={pose_activity:.2f}, result={adaptive_margin:.2f}")

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
    """
    Count people in video with IMPROVED detection for partial people.
    """
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

        # IMPROVED: Lower confidence threshold
        result = yolo_model.predict(rgb, conf=PERSON_DETECTION_CONF, classes=[0], verbose=False)

        frame_count = 0
        h, w = frame.shape[:2]
        frame_area = h * w

        for r in result:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf)

                box_w, box_h = x2 - x1, y2 - y1
                area = box_w * box_h

                # IMPROVED: Lower minimum area
                if area / frame_area < MIN_PERSON_AREA_RATIO:
                    continue

                # IMPROVED: More lenient aspect ratio for partial people
                aspect = box_w / max(box_h, 1)
                if aspect < 0.15 or aspect > 6:
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

    # IMPROVED: Use mean instead of median for better 3-person detection
    candidate_counts = [mean_count, median_count]
    for count, freq in most_common:
        if freq >= len(people_counts) * 0.20:  # Lowered from 0.25
            candidate_counts.append(count)

    final_count = int(round(max(candidate_counts)))

    max_seen = max(people_counts)
    if max_seen >= 2 and final_count < 2:
        print(f"  ⚠️  Overriding: saw {max_seen} people in at least one frame")
        final_count = max_seen

    # IMPROVED: Better 3-person detection
    if mean_count >= 2.4 and max_seen >= 3 and final_count < 3:
        print(f"  ⚠️  Overriding to 3: mean={mean_count:.1f}, max={max_seen}")
        final_count = 3

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
        """
        Update tracker with boxes, crop_count determines active slots.
        
        FIXED: Properly maps zone assignments to action indices
        """
        h, w = frame_shape[:2]

        # Define active indices based on crop count
        active_indices = []
        if crop_count == 3:
            active_indices = [0, 1, 2]
        elif crop_count == 2:
            active_indices = [0, 2]
        elif crop_count == 1:
            active_indices = [0]
        
        # ========================================
        # ZONE-BASED ASSIGNMENT
        # ========================================
        
        # Define zones based on crop count
        if crop_count == 3:
            # Left, Middle, Right zones
            zone_width = w / 3
            zones = [
                (0, zone_width),                    # Left zone
                (zone_width, 2 * zone_width),       # Middle zone  
                (2 * zone_width, w)                 # Right zone
            ]
            zone_names = ['left', 'middle', 'right']
            zone_to_action = {0: 0, 1: 1, 2: 2}  # Direct mapping
            
        elif crop_count == 2:
            # Left and Right zones (no middle)
            half_width = w / 2
            zones = [
                (0, half_width),                    # Left zone
                (half_width, w)                     # Right zone
            ]
            zone_names = ['left', 'right']
            zone_to_action = {0: 0, 1: 2}  # Zone 0→Action 0, Zone 1→Action 2
            
        else:  # crop_count == 1
            zones = [(0, w)]
            zone_names = ['full']
            zone_to_action = {0: 0}
        
        # Assign each detected box to its nearest zone
        assigned_boxes = [None] * len(zones)
        
        if len(boxes) > 0:
            for box in boxes:
                box_center_x = (box[0] + box[2]) / 2
                
                # Find which zone this box belongs to
                best_zone_idx = 0
                min_distance = float('inf')
                
                for zone_idx, (zone_start, zone_end) in enumerate(zones):
                    zone_center = (zone_start + zone_end) / 2
                    distance = abs(box_center_x - zone_center)
                    
                    # Check if box center is actually IN this zone (preference)
                    in_zone = zone_start <= box_center_x <= zone_end
                    
                    if in_zone:
                        # If in zone, prioritize it
                        distance = distance * 0.5  # Make it more attractive
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_zone_idx = zone_idx
                
                # Assign box to the best zone
                # If zone already has a box, keep the one closer to zone center
                if assigned_boxes[best_zone_idx] is None:
                    assigned_boxes[best_zone_idx] = box
                else:
                    # Compare which box is better for this zone
                    existing_box = assigned_boxes[best_zone_idx]
                    existing_center_x = (existing_box[0] + existing_box[2]) / 2
                    new_center_x = box_center_x
                    
                    zone_start, zone_end = zones[best_zone_idx]
                    zone_center = (zone_start + zone_end) / 2
                    
                    existing_dist = abs(existing_center_x - zone_center)
                    new_dist = abs(new_center_x - zone_center)
                    
                    # Keep the box closer to zone center
                    if new_dist < existing_dist:
                        assigned_boxes[best_zone_idx] = box
        
        # Debug: Show zone assignments
        if frame_idx % 60 == 0:
            print(f"\n🎯 Frame {frame_idx} Zone Assignments:")
            for idx, (box, zone_name) in enumerate(zip(assigned_boxes, zone_names)):
                if box:
                    box_center_x = (box[0] + box[2]) / 2
                    print(f"   {zone_name.capitalize()}: box at x={box_center_x:.0f}")
                else:
                    print(f"   {zone_name.capitalize()}: NO DETECTION")
        
        # ========================================
        # CRITICAL FIX: Map zone boxes to action indices
        # ========================================
        action_boxes = [None] * self.max_actions
        for zone_idx, box in enumerate(assigned_boxes):
            action_idx = zone_to_action[zone_idx]
            action_boxes[action_idx] = box
        
        # Use the properly mapped boxes
        boxes = action_boxes
        
        # ========================================
        # Prevent overlap
        # ========================================
        # Only prevent overlap for active boxes
        active_boxes = [boxes[i] for i in active_indices]
        active_boxes = prevent_overlap(active_boxes, w)
        
        # Put them back in the correct positions
        for i, action_idx in enumerate(active_indices):
            if i < len(active_boxes):
                boxes[action_idx] = active_boxes[i]
        
        # ========================================
        # Update tracking for each active action
        # ========================================
        for action_idx in active_indices:
            if boxes[action_idx] is not None:
                box = boxes[action_idx]
                box = self._fine_tune_box(box, action_idx, (h, w))
                self.histories[action_idx].append(box)
                self.confidences[action_idx] = min(self.confidences[action_idx] + 1, 10)
                self.missing_counters[action_idx] = 0
            else:
                self.missing_counters[action_idx] += 1

        # ========================================
        # Lock actions when confirmed
        # ========================================
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

        # print(f"\nDEBUG: Frame {self.frame_idx}, crop_count={crop_count}")

        # Get detection boxes
        result = detector.predict(frame, conf=PERSON_DETECTION_CONF_TRACKING, classes=[0], verbose=False)
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

    def _get_fallback(self, action_idx, frame_shape, positions=None, crop_count=3):
        """
        Improved fallback that respects the actual crop positions strategy.
        
        Args:
            action_idx: Which action slot (0, 1, 2)
            frame_shape: (h, w) tuple
            positions: Actual positions being used (e.g., ['left', 'right'])
            crop_count: How many crops total
        """
        h, w = frame_shape
        
        print(f"⚠️ FALLBACK USED for action {action_idx}! missing_counter={self.missing_counters[action_idx]}")
        print(f"   Positions: {positions}, Crop count: {crop_count}")

        # If we have positions info, use it to map action_idx to actual position
        actual_position = None
        if positions and action_idx < len(positions):
            actual_position = positions[action_idx]
        
        # Determine vertical position based on whether it's likely head/upper body
        default_size = int(min(h, w) // 2.5)
        
        # Head-focused crops should be higher in frame
        if actual_position and self._is_head_focused(actual_position):
            vertical_offset = int(h * 0.35)  # Higher for heads
        else:
            vertical_offset = int(h * 0.45)  # Standard
        
        # Map position to actual screen location
        if actual_position == 'left':
            return (
                int(w//8), 
                vertical_offset,
                int(w//8 + default_size), 
                vertical_offset + default_size
            )
        elif actual_position == 'center' or actual_position == 'middle':
            return (
                int(w//2 - default_size//2), 
                vertical_offset,
                int(w//2 + default_size//2), 
                vertical_offset + default_size
            )
        elif actual_position == 'right':
            return (
                int(w*7//8 - default_size), 
                vertical_offset,
                int(w*7//8), 
                vertical_offset + default_size
            )
        
        # Fallback to old logic if no position info
        print(f"⚠️ No position info for fallback idx={action_idx}")
        return self._get_legacy_fallback(action_idx, frame_shape)

    def _is_head_focused(self, position):
        """Check if this crop should focus on head/upper body."""
        # You might want to track this per position
        # For now, assume all are head-focused
        return True

    def _get_legacy_fallback(self, action_idx, frame_shape):
        """Original fallback logic for compatibility."""
        h, w = frame_shape
        default_size = int(min(h, w) // 2.5)
        vertical_offset = int(h * 0.45)
        
        if action_idx == 0:
            return (int(w//8), vertical_offset, int(w//8 + default_size), 
                    vertical_offset + default_size)
        elif action_idx == 1:
            return (int(w//2 - default_size//2), vertical_offset,
                    int(w//2 + default_size//2), vertical_offset + default_size)
        else:
            return (int(w*7//8 - default_size), vertical_offset,
                    int(w*7//8), vertical_offset + default_size)

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
        print(f"DEBUG: expand_box called with None for action_idx={action_idx}")
        # Only create a new box if the input is None
        h, w = frame_shape[:2]
        default_size = int(min(h, w) // 2.5)
        vertical_offset = int(h * 0.45)
        
        if action_idx == 0:
            return (
                int(w//8), 
                vertical_offset,
                int(w//8 + default_size), 
                vertical_offset + default_size
            )
        elif action_idx == 1:
            return (
                int(w//2 - default_size//2), 
                vertical_offset,
                int(w//2 + default_size//2), 
                vertical_offset + default_size
            )
        else:
            return (
                int(w*7//8 - default_size), 
                vertical_offset,
                int(w*7//8), 
                vertical_offset + default_size
            )
    
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
    
    Args:
        frame: Original frame
        frame_idx: Frame number
        yolo_boxes: Original YOLO detections (RED)
        expanded_boxes: Expanded boxes after margin application (YELLOW)
        smoothed_boxes: Smoothed boxes after temporal filtering (GREEN)
        final_boxes: Final crop regions (BLUE for good tracking, MAGENTA for fallback)
        action_statuses: Status for each action (for coloring)
        positions: Position names (left, middle, right)
        debug_info: Additional debug information
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


def process_video_with_dynamic_crops(input_path, output_folder, yolo_model, crop_count, positions_override=None):
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

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writers = []
    for output_path in output_files:
        writer = cv2.VideoWriter(output_path, fourcc, fps, TARGET_SIZE)
        writers.append(writer)

    # NEW: Create debug video writer
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
        if DEBUG_MODE and debug_sample_count < DEBUG_SAMPLES:
            result = yolo_model.predict(rgb, conf=PERSON_DETECTION_CONF_TRACKING, classes=[0], verbose=False)
            for r in result:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    box_w, box_h = x2 - x1, y2 - y1
                    area = box_w * box_h
                    frame_area = frame.shape[0] * frame.shape[1]
                    aspect = box_w / box_h if box_h > 0 else 1
                    
                    if (0.02 < area / frame_area < 0.5 and 0.5 < aspect < 2.0):
                        yolo_boxes.append((x1, y1, x2, y2))
        
        # Detect actions with ROI-based detector
        actions = detector.detect(rgb, yolo_model, crop_count, pose_model, detector.roi_detector)
        print(f"DEBUG Frame {frame_count}: actions returned = {actions}")
        print(f"  missing_counters = {detector.missing_counters}")
        print(f"  last_good_actions = {detector.last_good_actions}")


        expanded_actions = []
        action_indices = []
        action_statuses = []  # Track status for each action
        
        # ✅ FIXED: Determine action indices OUTSIDE the loop
        if crop_count == 3:
            action_indices = [0, 1, 2]
        elif crop_count == 2:
            # Map positions to indices: left=0, center=1, right=2
            if positions == ['left', 'right']:
                action_indices = [0, 2]
            elif positions == ['left', 'center']:
                action_indices = [0, 1]
            elif positions == ['center', 'right']:
                action_indices = [1, 2]
            else:
                action_indices = [0, 2]  # Default fallback
        
        # ✅ FIXED: Main processing loop
        for i, action_idx in enumerate(action_indices):
            if i < len(actions) and actions[i] is not None:
                current_box = actions[i]
                missing = detector.missing_counters[action_idx] if action_idx < len(detector.missing_counters) else 0
                hist_len = len(detector.motion_histories[action_idx]) if action_idx < len(detector.motion_histories) else 0
                pose_activity = detector.pose_activities[action_idx] if action_idx < len(detector.pose_activities) else 0.0
                history = detector.motion_histories[action_idx] if action_idx < len(detector.motion_histories) else deque(maxlen=10)
                
                # ✅ FIXED LOGIC: Determine status based on missing counter
                # missing == 0 means we have a REAL FRESH detection from YOLO
                # missing > 0 means we're using a tracked/fallback box
                
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
            
            # ──── Debug print (only occasionally) ────
            if frame_count % 45 == 0:
                miss = detector.missing_counters[action_idx] if action_idx < len(detector.missing_counters) else -1
                hist = len(detector.motion_histories[action_idx]) if action_idx < len(detector.motion_histories) else 0
                print(f"  F {frame_count} | idx={action_idx} | miss={miss:2d} | hist={hist:2d} | {status:16} | margin={adaptive_margin:.3f}")

            # ──── Actually expand the box ────
            expanded = expand_box(
                current_box,
                frame.shape,
                frame_count,
                action_idx=action_idx,
                margin=adaptive_margin,
                is_fallback=use_fallback_expansion
            )
            expanded_actions.append(expanded)
            
        # ✅ FIXED: This is now outside the loop, as it should be
        h, w = frame.shape[:2]
        expanded_actions = prevent_overlap(expanded_actions, w)
        
        smoothed_actions = smoother.smooth(*expanded_actions)
                
        # NEW: Write debug frame
        if DEBUG_MODE and DEBUG_CREATE_VIDEOS and debug_writer:
            debug_info = {
                "Crop Count": crop_count,
                "Positions": ", ".join(positions),
                "Target": f"{TARGET_SIZE[0]}x{TARGET_SIZE[1]}"
            }
            
            debug_frame = create_enhanced_debug_frame(
                frame, frame_count, yolo_boxes, expanded_actions,
                smoothed_actions, smoothed_actions, action_statuses,
                positions, detector, debug_info
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

    # NEW: Release debug writer
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

        # STEP 1: Count people (quick filter)
        start_time = time.time()
        people_count = count_people_in_video(video_path, yolo, PEOPLE_SAMPLE_FRAMES)
        elapsed = time.time() - start_time
        print(f"   👥 Detected {people_count} person(s) in {elapsed:.1f}s")

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

                # Decision logic for 4+ people scenarios
                if left_avg >= 2.5 and right_avg >= 1.5:
                    print(f"   ↔️ Scenario: ~{left_avg:.0f} left, ~{right_avg:.0f} right - using left+right crops")
                    crop_count = 2
                    positions = ['left', 'right']
                    strategy = "4plus-left-right-split"

                elif left_avg >= 3 and left_avg > center_avg + right_avg:
                    print(f"   ⬅️ Left concentration ({left_avg:.1f} people) - single left crop")
                    crop_count = 1
                    positions = ['left']
                    strategy = "4plus-left-dominant"

                elif right_avg >= 3 and right_avg > center_avg + left_avg:
                    print(f"   ➡️ Right concentration ({right_avg:.1f} people) - single right crop")
                    crop_count = 1
                    positions = ['right']
                    strategy = "4plus-right-dominant"

                elif left_avg >= 2 and right_avg >= 2:
                    print(f"   ↔️ People on both sides - left+right crops")
                    crop_count = 2
                    positions = ['left', 'right']
                    strategy = "4plus-both-sides"

                elif center_avg >= 3:
                    print(f"   ⬆️ Center concentration ({center_avg:.1f} people) - center crop")
                    crop_count = 1
                    positions = ['center']
                    strategy = "4plus-center-dominant"

                else:
                    print(f"   ⚖️ Even distribution - default to left+right crops")
                    crop_count = 2
                    positions = ['left', 'right']
                    strategy = "4plus-even-distribution"

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

        # STEP 3: Process or copy based on strategy
        # Fully automatic - if crop_count is 0, copy; otherwise crop
        if crop_count >= MIN_PEOPLE_REQUIRED and len(positions) >= MIN_PEOPLE_REQUIRED:
            print(f"   🎬 Processing with {crop_count}-crop: {positions}")
            process_video_with_dynamic_crops(
                video_path, OUTPUT_FOLDER, yolo, crop_count, 
                positions_override=positions
            )
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