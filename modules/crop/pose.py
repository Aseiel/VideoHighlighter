"""
pose.py — keypoint estimation and everything derived from it.

DORMANT AS OF THIS SPLIT. Every function here needs a pose model, and the app
has not had one since the AGPL YOLO package was dropped: YOLOX is a detector
with no keypoint head, and the shim in modules/vision/detection_backend.py
says so explicitly (`_PredictResult.keypoints = None`). So `pose_model` arrives
as None at every call site, `get_pose_keypoints_for_frame` returns [], and the
guards downstream (`if pose_model:`, `if not poses: return True`) turn the whole
layer into a no-op rather than an error.

Collected into one module so that is VISIBLE. Threaded through five files it
looked like working code; here the dormancy is the first thing you read.

WHAT THE APP LOSES WHILE THIS SLEEPS
    - count_people_in_video()'s second opinion. Its 1 -> 2 upgrade requires
      `bbox_2plus >= 0.30 AND pose_2plus >= 0.30`, so with pose flat the gate can
      never open: the counter can only ever agree with the boxes.
    - the same for the 3-person promotion's `both_agree` half.
    - activity scoring, which feeds crop_core.expand_box(pose_activity=...) and
      the ROI centering in track.py. Both currently see a constant 0.0.

REVIVING IT
    RTMPose (mmpose, Apache-2.0) exported to OpenVINO IR — no new runtime
    dependency, openvino is already required, and it clears the licensing gate in
    CLAUDE.md where the AGPL package did not. It is top-down: it takes person
    boxes and
    returns keypoints for each, so it does NOT restore the keypoint-cluster path
    that looked for people the detector missed. What replaces that is a generous
    detector plus confirmation — YOLOX proposing at a low score_thr and RTMPose
    deciding which proposals have a real skeleton behind them.

    The seam is get_pose_keypoints_for_frame(): keep its return shape
    ({'keypoints', 'bbox', 'num_visible'} per person) and nothing else in the
    package needs to change.

FUNCTIONS WITH NO CALLER AT ALL (dead before pose was dropped, not because of it):
    validate_detections_with_pose, analyze_action_coherence,
    calculate_pose_coherence, calculate_interaction_score,
    calculate_movement_synchrony, get_pose_center_target.
    Kept deliberately — they are the analysis the coherence knobs in config.py
    were written for, and re-wiring them is cheaper than rewriting them.
"""
import cv2
import numpy as np

from modules.crop.core import calculate_iou
from modules.crop.config import (
    MIN_PERSON_AREA_RATIO,
    MIN_POSE_KEYPOINTS,
    PERSON_DETECTION_CONF_ZONES,
    POSE_CONFIDENCE_THRESHOLD,
    POSE_VALIDATION_CONF_THRESHOLD,
    POSE_VALIDATION_KEYPOINT_CONF,
    POSE_VALIDATION_MIN_KEYPOINTS,
)


def get_pose_keypoints_for_frame(rgb_frame, pose_model, conf=0.15, person_boxes=None):
    """
    Run pose estimation on a frame and return all detected keypoints + bboxes.
    Returns list of dicts: [{'keypoints': array, 'bbox': (x1,y1,x2,y2), 'num_visible': int}, ...]

    TOP-DOWN: `person_boxes` is required, because RTMPose estimates keypoints
    for a box rather than searching the frame. Callers already have the boxes —
    they detect on the same frame — so this is a reordering, not extra work.
    Without boxes there is nothing to estimate and the answer is honestly empty.

    The return shape predates the backend swap and is kept deliberately: every
    consumer here indexes `pose['keypoints'][i][2]` for visibility and reads
    `pose['bbox']`, so the adapter absorbs the change and nothing downstream
    moves. `pose_model` is whatever build_pose_estimator() handed back, or None
    — which stays a supported, silent no-op.
    """
    if pose_model is None or person_boxes is None or len(person_boxes) == 0:
        return []

    try:
        # The estimator works in BGR (it normalises with the model's own RGB
        # mean/std after converting); callers here hold RGB because that is what
        # the YOLOX shim wants. Convert once rather than threading two frames.
        bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        poses = []
        for pose in pose_model.estimate(bgr, person_boxes):
            kp_np = pose.keypoints
            if kp_np is None or len(kp_np) == 0:
                continue

            # Count visible keypoints
            num_visible = 0
            for k in kp_np:
                if len(k) >= 3 and k[2] > POSE_VALIDATION_KEYPOINT_CONF:
                    num_visible += 1

            poses.append({
                'keypoints': kp_np,
                'bbox': tuple(int(v) for v in pose.bbox),
                'num_visible': num_visible
            })

        return poses
    except Exception as e:
        print(f"⚠️ Pose estimation failed: {e}")
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


def analyze_pose_activity(keypoints, box):
    """Analyze pose keypoints to determine activity level"""
    if keypoints is None or len(keypoints) == 0:
        return 0.0

    activity_score = 0.0
    box_x1, box_y1, box_x2, box_y2 = box
    box_width = box_x2 - box_x1
    box_height = box_y2 - box_y1

    # ARM MOVEMENTS
    arms_extended = 0
    arms_raised = 0
    
    # LEG MOVEMENTS
    legs_apart = 0
    knees_bent = 0
    kicking = 0
    hip_movement = 0

    # Check arms
    left_shoulder = keypoints[5] if len(keypoints) > 5 else None
    right_shoulder = keypoints[6] if len(keypoints) > 6 else None
    left_wrist = keypoints[9] if len(keypoints) > 9 else None
    right_wrist = keypoints[10] if len(keypoints) > 10 else None
    
    # Leg keypoints
    left_hip = keypoints[11] if len(keypoints) > 11 else None
    right_hip = keypoints[12] if len(keypoints) > 12 else None
    left_knee = keypoints[13] if len(keypoints) > 13 else None
    right_knee = keypoints[14] if len(keypoints) > 14 else None
    left_ankle = keypoints[15] if len(keypoints) > 15 else None
    right_ankle = keypoints[16] if len(keypoints) > 16 else None

    # ===== ARM DETECTION =====
    if left_shoulder is not None and left_wrist is not None:
        left_shoulder_conf = left_shoulder[2] if len(left_shoulder) > 2 else 0
        left_wrist_conf = left_wrist[2] if len(left_wrist) > 2 else 0
        if left_shoulder_conf > 0.3 and left_wrist_conf > 0.3:
            arm_span = abs(left_wrist[0] - left_shoulder[0])
            if arm_span > box_width * 0.3:
                arms_extended += 1
            if left_wrist[1] < left_shoulder[1] - box_height * 0.1:
                arms_raised += 1

    if right_shoulder is not None and right_wrist is not None:
        right_shoulder_conf = right_shoulder[2] if len(right_shoulder) > 2 else 0
        right_wrist_conf = right_wrist[2] if len(right_wrist) > 2 else 0
        if right_shoulder_conf > 0.3 and right_wrist_conf > 0.3:
            arm_span = abs(right_wrist[0] - right_shoulder[0])
            if arm_span > box_width * 0.3:
                arms_extended += 1
            if right_wrist[1] < right_shoulder[1] - box_height * 0.1:
                arms_raised += 1

    # ===== FIXED HIP/LEG DETECTION =====
    
    # 1. HIP MOVEMENT - Check if hips are visible (this was missing!)
    if left_hip is not None and right_hip is not None:
        left_hip_conf = left_hip[2] if len(left_hip) > 2 else 0
        right_hip_conf = right_hip[2] if len(right_hip) > 2 else 0
        
        if left_hip_conf > 0.3 and right_hip_conf > 0.3:
            # Check hip width (wider stance = more activity)
            hip_width = abs(left_hip[0] - right_hip[0])
            if hip_width > box_width * 0.3:
                hip_movement += 1
                
            # Check hip height relative to box (squatting/crouching)
            avg_hip_y = (left_hip[1] + right_hip[1]) / 2
            box_center_y = (box_y1 + box_y2) / 2
            
            # If hips are lower than box center, likely squatting/crouching
            if avg_hip_y > box_center_y + box_height * 0.1:
                hip_movement += 2  # Strong indicator of activity
    
    # 2. Legs spread apart
    if left_ankle is not None and right_ankle is not None:
        left_ankle_conf = left_ankle[2] if len(left_ankle) > 2 else 0
        right_ankle_conf = right_ankle[2] if len(right_ankle) > 2 else 0
        if left_ankle_conf > 0.2 and right_ankle_conf > 0.2:
            leg_span = abs(left_ankle[0] - right_ankle[0])
            if leg_span > box_width * 0.25:
                legs_apart += 1
    
    # 3. Knees bent
    if left_hip is not None and left_knee is not None and left_ankle is not None:
        left_hip_conf = left_hip[2] if len(left_hip) > 2 else 0
        left_knee_conf = left_knee[2] if len(left_knee) > 2 else 0
        left_ankle_conf = left_ankle[2] if len(left_ankle) > 2 else 0
        if left_hip_conf > 0.2 and left_knee_conf > 0.2 and left_ankle_conf > 0.2:
            # Vector from hip to knee
            hip_knee_x = left_knee[0] - left_hip[0]
            hip_knee_y = left_knee[1] - left_hip[1]
            # Vector from knee to ankle
            knee_ankle_x = left_ankle[0] - left_knee[0]
            knee_ankle_y = left_ankle[1] - left_knee[1]
            
            dot = hip_knee_x * knee_ankle_x + hip_knee_y * knee_ankle_y
            hip_knee_len = np.sqrt(hip_knee_x**2 + hip_knee_y**2)
            knee_ankle_len = np.sqrt(knee_ankle_x**2 + knee_ankle_y**2)
            
            if hip_knee_len > 0 and knee_ankle_len > 0:
                cos_angle = dot / (hip_knee_len * knee_ankle_len)
                angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi
                if angle < 160:
                    knees_bent += 1
    
    # 4. Kicking motion
    if left_hip is not None and left_knee is not None and right_hip is not None and right_knee is not None:
        left_hip_conf = left_hip[2] if len(left_hip) > 2 else 0
        right_hip_conf = right_hip[2] if len(right_hip) > 2 else 0
        left_knee_conf = left_knee[2] if len(left_knee) > 2 else 0
        right_knee_conf = right_knee[2] if len(right_knee) > 2 else 0
        
        if left_hip_conf > 0.2 and right_hip_conf > 0.2 and left_knee_conf > 0.2 and right_knee_conf > 0.2:
            # Check if one knee is much higher than the other
            knee_height_diff = abs(left_knee[1] - right_knee[1])
            if knee_height_diff > box_height * 0.08:
                kicking += 1
                
            # Also check if one knee is far forward/back
            knee_depth_diff = abs(left_knee[0] - right_knee[0])
            if knee_depth_diff > box_width * 0.15:
                kicking += 1

    # ===== ADD HIP ROTATION DETECTION =====
    if left_hip is not None and right_hip is not None:
        left_hip_conf = left_hip[2] if len(left_hip) > 2 else 0
        right_hip_conf = right_hip[2] if len(right_hip) > 2 else 0
        
        if left_hip_conf > 0.2 and right_hip_conf > 0.2:
            # Check if hips are rotated (different Y positions)
            hip_y_diff = abs(left_hip[1] - right_hip[1])
            if hip_y_diff > box_height * 0.05:  # Slight rotation
                hip_movement += 1

    # ===== COMBINED SCORE - REBALANCED =====
    # FIXED: Give more weight to hip/leg movements
    arm_score = (arms_extended / 2.0) * 0.15 + (arms_raised / 2.0) * 0.15
    leg_score = (legs_apart) * 0.15 + (knees_bent) * 0.20 + (kicking) * 0.20 + (hip_movement) * 0.15
    
    activity_score = arm_score + leg_score
    
    # Add small baseline for any movement
    if activity_score > 0:
        activity_score += 0.05

    return min(activity_score, 1.0)


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
            if len(person_poses) == 2 and person_poses[0] is not None and person_poses[1] is not None:
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
