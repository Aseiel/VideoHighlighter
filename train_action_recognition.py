import os
import glob
import cv2
import json
import random
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from openvino.runtime import Core
from tqdm import tqdm
from collections import deque

# =============================
# Adaptive Action Region Detection
# =============================
class AdaptiveActionDetector:
    """Detects WHERE the action is happening based on motion and interaction patterns"""
    
    def __init__(self, motion_threshold=5.0, debug=False):
        self.motion_threshold = motion_threshold
        self.prev_poses = None
        self.debug = debug

    
    def _check_body_visibility(self, poses):
        """Check which body parts are visible in the poses"""
        visibility = {
            'has_upper': False,
            'has_lower': False,
            'has_hips': False,
            'has_feet': False,
            'has_hands': False
        }
        
        for pose in poses:
            # Hips: keypoints 11, 12
            if pose[11, 2] > 0.15 or pose[12, 2] > 0.15:
                visibility['has_hips'] = True
                visibility['has_lower'] = True
            
            # Feet: keypoints 15, 16
            if pose[15, 2] > 0.15 or pose[16, 2] > 0.15:
                visibility['has_feet'] = True
                visibility['has_lower'] = True
            
            # Knees: 13, 14 also count as lower body
            if pose[13, 2] > 0.15 or pose[14, 2] > 0.15:
                visibility['has_lower'] = True
            
            # Shoulders: 5, 6
            if pose[5, 2] > 0.15 or pose[6, 2] > 0.15:
                visibility['has_upper'] = True
            
            # Hands: 9, 10
            if pose[9, 2] > 0.15 or pose[10, 2] > 0.15:
                visibility['has_hands'] = True
                visibility['has_upper'] = True
        
        return visibility
    
    def detect_action_region(self, frame, person_boxes, pose_extractor, max_poses=2):
        """Smart detection of WHERE action is happening"""
        h, w = frame.shape[:2]
        
        current_poses = self._get_matched_poses(frame, person_boxes, pose_extractor, max_poses)
        
        if len(current_poses) == 0:
            if self.debug:
                print("   ⚠️  No poses detected -> defaulting to LOWER BODY (hip-based)")
            # When no poses detected, assume lower body action
            return self._merge_boxes(person_boxes), 'lower_body'
        
        # Check what body parts are visible
        body_part_visibility = self._check_body_visibility(current_poses)
        motion_regions = self._analyze_motion_regions(current_poses)
        focus_region = self._determine_focus_region(current_poses, motion_regions, person_boxes, body_part_visibility)
        final_roi = self._adaptive_crop(focus_region, current_poses, w, h)
        
        return final_roi, focus_region
    
    def _get_matched_poses(self, frame, person_boxes, pose_extractor, max_poses):
        """Get poses only for the detected action people - WITH LOWER THRESHOLDS"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_extractor.model.predict(frame_rgb, conf=0.15, verbose=False)  # Lower confidence!
        
        if len(results) == 0 or results[0].keypoints is None:
            return []
        
        all_keypoints = results[0].keypoints.data.cpu().numpy()
        matched_poses = []
        
        for action_box in person_boxes[:max_poses]:
            ax1, ay1, ax2, ay2 = action_box
            
            best_match_idx = None
            best_match_score = 0
            
            for idx, kpts in enumerate(all_keypoints):
                # Accept poses with even just 2 visible keypoints (very permissive!)
                if np.sum(kpts[:, 2] > 0.15) >= 2:  # Lower from 4 keypoints and 0.2 confidence
                    visible_kpts = kpts[kpts[:, 2] > 0.15]
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
    
    def _analyze_motion_regions(self, current_poses):
        """Analyze motion with BETTER lower body sensitivity"""
        motion_regions = {
            'head': 0, 'upper_body': 0, 'lower_body': 0, 'hands': 0, 'feet': 0
        }
        
        # FIRST FRAME: No motion history yet
        if self.prev_poses is None or len(self.prev_poses) != len(current_poses):
            self.prev_poses = current_poses
            # For first frame, estimate likely action region based on pose structure
            if len(current_poses) > 0:
                body_visibility = self._check_body_visibility(current_poses)
                
                # Check body part positions to guess action type
                for pose in current_poses:
                    # If lower body keypoints are visible, give initial boost
                    lower_visible = sum(1 for idx in [11, 12, 13, 14, 15, 16] if pose[idx, 2] > 0.15)
                    upper_visible = sum(1 for idx in [5, 6, 7, 8, 9, 10] if pose[idx, 2] > 0.15)
                    
                    # Initial bias based on visibility
                    if lower_visible >= upper_visible:
                        motion_regions['lower_body'] += 2.0
                        motion_regions['feet'] += 1.5
                        if self.debug:
                            print("   🎬 FIRST FRAME: Lower body visible, mild LOWER BODY bias")
                    elif body_visibility.get('has_hips', False):
                        motion_regions['lower_body'] += 1.5
                        if self.debug:
                            print("   🎬 FIRST FRAME: Hips visible, mild LOWER BODY bias")
                    else:
                        motion_regions['upper_body'] += 2.0
                        if self.debug:
                            print("   🎬 FIRST FRAME: Upper body dominant")

            return motion_regions
        
        region_mapping = {
            'head': [0, 1, 2, 3, 4],
            'upper_body': [5, 6, 7, 8, 9, 10],
            'lower_body': [11, 12, 13, 14, 15, 16],
            'hands': [9, 10],
            'feet': [15, 16]
        }
        
        total_motion = 0
        motion_count = 0
        
        for prev_pose, curr_pose in zip(self.prev_poses, current_poses):
            for region_name, indices in region_mapping.items():
                region_motion = 0
                valid_points = 0
                
                for idx in indices:
                    if (idx < len(prev_pose) and idx < len(curr_pose) and 
                        prev_pose[idx, 2] > 0.15 and curr_pose[idx, 2] > 0.15):  # Lower confidence needed
                        
                        motion = np.sqrt(
                            (curr_pose[idx, 0] - prev_pose[idx, 0])**2 +
                            (curr_pose[idx, 1] - prev_pose[idx, 1])**2
                        )
                        
                        # BOOST lower body motion more aggressively
                        if region_name in ['lower_body', 'feet']:
                            motion = motion * 1.3
                        else:
                            motion = motion * 1.0
                        
                        region_motion += motion
                        valid_points += 1
                        total_motion += motion
                        motion_count += 1
                
                if valid_points > 0:
                    motion_regions[region_name] += region_motion / valid_points
        
        self.prev_poses = current_poses
        return motion_regions
    
    def _determine_focus_region(self, poses, motion_regions, person_boxes, body_visibility):
        """Determine which body region to focus on - WITH FULL BODY DETECTION"""
        if len(poses) == 0:
            return 'lower_body'  # Default to lower body when no poses
        
        # Get motion values
        hands_motion = motion_regions.get('hands', 0)
        feet_motion = motion_regions.get('feet', 0) 
        head_motion = motion_regions.get('head', 0)
        upper_body_motion = motion_regions.get('upper_body', 0)
        lower_body_motion = motion_regions.get('lower_body', 0)
        
        # Check if this is likely first frame analysis (motion from visibility bias)
        total_motion = sum(motion_regions.values())
        is_first_frame_bias = (total_motion > 0 and 
                            all(v < 1.0 or v > 2.0 for v in motion_regions.values() if v > 0))
        
        if is_first_frame_bias:
            if self.debug:
                print("   🎬 FIRST FRAME DECISION based on visibility")
            # First frame: decide purely on visibility
            if lower_body_motion > upper_body_motion:
                if self.debug:
                    print("   🦵 First frame: Lower body visibility bias -> LOWER BODY")
                return 'lower_body'
            elif body_visibility.get('has_hips', False):
                if self.debug:
                    print("   🦴 First frame: Hips visible -> LOWER BODY")
            elif body_visibility.get('has_feet', False):
                if self.debug:
                    print("   🦶 First frame: Feet visible -> LOWER BODY")
                return 'lower_body'
        
        # NO MOTION DETECTED
        if not motion_regions or all(value == 0 for value in motion_regions.values()):
            # If no motion but we can see body parts, decide based on visibility
            if body_visibility['has_hips'] or body_visibility['has_feet']:
                if self.debug:
                    print("   🦴 Hips/feet visible, no motion -> LOWER BODY default")
                return 'lower_body'
            elif body_visibility['has_upper']:
                if self.debug:
                    print("   🦴 Upper body visible, no motion -> UPPER BODY default")
                return 'upper_body'
            return 'full_body'
        
        # MUCH MORE SENSITIVE thresholds for lower body
        base_threshold = self.motion_threshold * 0.3
        
        # Debug output
        if self.debug:
            debug_info = {k: f"{v:.2f}" for k, v in motion_regions.items()}
            visibility_info = ', '.join([k.replace('has_', '') for k, v in body_visibility.items() if v])
            print(f"   Motion: {debug_info}, Visible: {visibility_info}, Threshold: {base_threshold:.2f}")
        
        # ============================================
        # FULL BODY DETECTION
        # ============================================
        # Calculate total motion for upper and lower body
        total_upper = upper_body_motion + hands_motion + head_motion
        total_lower = lower_body_motion + feet_motion
        
        # Check if BOTH regions have significant motion
        upper_active = total_upper > base_threshold * 0.7
        lower_active = total_lower > base_threshold * 0.6
        
        if upper_active and lower_active:
            # Both regions are moving - check if motion is balanced
            motion_ratio = min(total_upper, total_lower) / max(total_upper, total_lower)
            
            # If motion is relatively balanced (within 2x), it's a full body action
            if motion_ratio > 0.5:
                if self.debug:
                    print(f"   🧍 FULL BODY detected - balanced motion (upper={total_upper:.2f}, lower={total_lower:.2f}, ratio={motion_ratio:.2f})")
                return 'full_body'
            elif total_upper > total_lower * 1.3:
                if self.debug:
                    print(f"   💪 Both moving but UPPER dominant (upper={total_upper:.2f}, lower={total_lower:.2f})")
                return 'upper_body'
            elif total_lower > total_upper * 1.3:
                if self.debug:
                    print(f"   🦵 Both moving but LOWER dominant (upper={total_upper:.2f}, lower={total_lower:.2f})")
                return 'lower_body'
            else:
                if self.debug:
                    print(f"   🧍 FULL BODY detected - both regions active (upper={total_upper:.2f}, lower={total_lower:.2f})")
                return 'full_body'
        
        # Check visibility for full body actions
        has_full_body_visible = (body_visibility.get('has_upper', False) and 
                                (body_visibility.get('has_hips', False) or body_visibility.get('has_lower', False)))
        
        if has_full_body_visible and total_motion > base_threshold:
            # If we can see both upper and lower body, and there's general motion
            if total_upper > 0 and total_lower > 0:
                print(f"   🧍 FULL BODY detected - visibility + distributed motion")
                return 'full_body'
        
        # ============================================
        # EXISTING PRIORITY ORDER
        # ============================================
        # SPECIAL CASE: If hips are visible but upper body isn't, favor lower body
        if body_visibility['has_hips'] and not body_visibility['has_upper']:
            if self.debug:
                print("   🦴 Hips visible, no upper body -> LOWER BODY bias")
            if lower_body_motion > 0 or feet_motion > 0:
                if self.debug:
                    print("   🦵 ANY lower motion with hip visibility -> LOWER BODY")
                return 'lower_body'
        
        # EVEN MORE AGGRESSIVE: If lower body visible at all, lower the threshold further
        if body_visibility.get('has_hips', False) or body_visibility.get('has_feet', False):
            lower_body_threshold = base_threshold * 0.6
            if self.debug:
                print(f"   🦴 Lower body parts visible, using moderate threshold: {lower_body_threshold:.2f}")
            
            if lower_body_motion > lower_body_threshold or feet_motion > lower_body_threshold * 0.7:
                if self.debug:
                    print("   🦵 LOWER BODY detected with hip/feet visibility")
                return 'lower_body'

        # PRIORITY ORDER with LOWER thresholds for lower body
        # 1. Hands (clear hand actions)
        if hands_motion > base_threshold * 1.5:
            if self.debug:
                print("   👐 Hand motion detected -> UPPER BODY")
            return 'upper_body'
        
        # 2. Feet (explicit foot motion)
        if feet_motion > base_threshold * 0.6:
            if self.debug:
                print("   🦶 Foot motion detected -> LOWER BODY")
            return 'lower_body'
        
        # 3. Lower body region (leg movement)
        if lower_body_motion > base_threshold * 0.6:
            if self.debug:
                print("   🦵 Lower body motion detected -> LOWER BODY")
            return 'lower_body'
        
        # 4. Head motion
        if head_motion > base_threshold * 1.0:
            if self.debug:
                print("   👤 Head motion detected -> UPPER BODY")
            return 'upper_body'
        
        # 5. Upper body (torso, arms)
        if upper_body_motion > base_threshold * 0.7:
            if self.debug:
                print("   💪 Upper body motion detected -> UPPER BODY")
            return 'upper_body'
        
        # 6. ANY lower body hint + hip visibility
        if (lower_body_motion > 0 or feet_motion > 0) and body_visibility['has_hips']:
            if self.debug:
                print(f"   🦴 Lower motion with visible hips -> LOWER BODY")
            return 'lower_body'
        
        # 7. ANY lower body motion (even tiny)
        if lower_body_motion > 0 or feet_motion > 0:
            total_lower = lower_body_motion + feet_motion
            total_upper = upper_body_motion + hands_motion + head_motion
            
            if total_lower > 0 and total_lower >= total_upper * 0.3:
                if self.debug:
                    print(f"   🦵 Subtle lower motion (lower={total_lower:.2f}, upper={total_upper:.2f}) -> LOWER BODY")
                return 'lower_body'
        
        # 8. Final fallback based on visibility
        if body_visibility['has_hips'] or body_visibility['has_feet']:
            if self.debug:
                print("   🦴 Defaulting to LOWER BODY (hips/feet visible)")
            return 'lower_body'
        
        if self.debug:
            print("   🔄 No clear focus -> FULL BODY")
        return 'full_body'
    
    def _adaptive_crop(self, focus_region, poses, frame_width, frame_height):
        """Apply different cropping strategies based on focus region"""
        all_points = []
        
        for pose in poses:
            if focus_region == 'upper_body':
                indices = list(range(0, 11))
            elif focus_region == 'lower_body':
                indices = list(range(11, 17))
            else:
                indices = list(range(0, 17))
            
            for idx in indices:
                if pose[idx, 2] > 0.2:
                    all_points.append(pose[idx, :2])
        
        if len(all_points) == 0:
            return None
        
        all_points = np.array(all_points)
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        
        if focus_region == 'upper_body':
            padding_x = (x_max - x_min) * 0.4
            padding_y_top = (y_max - y_min) * 0.6
            padding_y_bottom = (y_max - y_min) * 0.3
        elif focus_region == 'lower_body':
            padding_x = (x_max - x_min) * 0.4
            padding_y_top = (y_max - y_min) * 0.3
            padding_y_bottom = (y_max - y_min) * 0.6
        else:
            padding_x = (x_max - x_min) * 0.3
            padding_y_top = (y_max - y_min) * 0.4
            padding_y_bottom = (y_max - y_min) * 0.4
        
        x1 = max(0, int(x_min - padding_x))
        y1 = max(0, int(y_min - padding_y_top))
        x2 = min(frame_width, int(x_max + padding_x))
        y2 = min(frame_height, int(y_max + padding_y_bottom))
        
        min_width = frame_width * 0.3
        min_height = frame_height * 0.4
        
        if (x2 - x1) < min_width:
            center_x = (x1 + x2) // 2
            x1 = max(0, int(center_x - min_width // 2))
            x2 = min(frame_width, int(center_x + min_width // 2))
        
        if (y2 - y1) < min_height:
            center_y = (y1 + y2) // 2
            y1 = max(0, int(center_y - min_height // 2))
            y2 = min(frame_height, int(center_y + min_height // 2))
        
        return (x1, y1, x2, y2)
    
    def _merge_boxes(self, boxes):
        if len(boxes) == 0:
            return None
        if len(boxes) == 1:
            return boxes[0]
        
        x1_min = min(b[0] for b in boxes)
        y1_min = min(b[1] for b in boxes)
        x2_max = max(b[2] for b in boxes)
        y2_max = max(b[3] for b in boxes)
        
        return (x1_min, y1_min, x2_max, y2_max)
    
    def debug_motion_analysis(self, frame, person_boxes, pose_extractor):
        """Debug method - only called when debug_motion_analysis is enabled"""
        if not self.debug:
            return
            
        current_poses = self._get_matched_poses(frame, person_boxes, pose_extractor, 2)
        
        if len(current_poses) == 0:
            print("   ❌ No poses detected -> Will default to LOWER BODY")
            return
        
        body_visibility = self._check_body_visibility(current_poses)
        motion_regions = self._analyze_motion_regions(current_poses)
        focus_region = self._determine_focus_region(current_poses, motion_regions, person_boxes, body_visibility)
        
        print(f"   🔍 DEBUG: {len(current_poses)} poses, Focus: {focus_region}")
        print(f"   📊 Motion: {motion_regions}")
        
        upper_body_kpts = 0
        lower_body_kpts = 0
        hip_kpts = 0
        for pose in current_poses:
            for idx in [5, 6, 7, 8, 9, 10]:
                if pose[idx, 2] > 0.15:
                    upper_body_kpts += 1
            for idx in [11, 12, 13, 14, 15, 16]:
                if pose[idx, 2] > 0.15:
                    lower_body_kpts += 1
            for idx in [11, 12]:
                if pose[idx, 2] > 0.15:
                    hip_kpts += 1
        
        print(f"   👤 Keypoints: upper={upper_body_kpts}, lower={lower_body_kpts}, hips={hip_kpts}")
        print(f"   🦴 Visibility: {body_visibility}")



    def reset(self):
        """Reset motion history"""
        self.prev_poses = None

# =============================
# Smart Action-Based Person Detection
# =============================
class SmartActionDetector:
    """Detects people most likely performing the action using interaction + relative motion patterns"""
    def __init__(self, sticky_frames=15, sticky_weight=0.5, debug=False):
        self.prev_frame_data = None
        self.frame_count = 0
        self.selected_people = []  # People selected in recent frames
        self.selection_history = deque(maxlen=sticky_frames)  # Track selections
        self.sticky_weight = sticky_weight  # Weight for previous selections
        self.locked_pair = None  # Lock onto a specific pair once detected
        self.lock_strength = 0  # How strongly we're locked onto current pair
        self.locked_track_ids = set()  # Track IDs of locked people (for tracking mode)
        self.debug = debug
        
    def detect(self, frame, detector, max_people=2, allow_dynamic_group=True):
        """
        Detect people most likely to be performing actions based on INTERACTION + RELATIVE MOTION
        """
        h, w = frame.shape[:2]
        center_x, center_y = w / 2, h / 2
        
        result = detector.predict(frame, conf=0.40, classes=[0], verbose=False)
        current_detections = []
        
        for r in result:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf)
                
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)
                
                # Score 1: Center proximity (0-1, higher = closer to center)
                dist = np.sqrt((box_center_x - center_x)**2 + (box_center_y - center_y)**2)
                max_dist = np.sqrt(center_x**2 + center_y**2)
                center_score = 1 - (dist / max_dist)
                
                # Score 2: Size relevance (larger boxes are often more important)
                frame_area = h * w
                size_score = min(area / (frame_area * 0.3), 1.0)
                
                # Score 3: Motion with direction tracking
                motion_score = 0
                motion_vector = (0, 0)  # Track motion direction
                if self.prev_frame_data and self.frame_count > 0:
                    best_match_motion = 0
                    best_motion_vector = (0, 0)
                    for prev_box in self.prev_frame_data:
                        iou = self._iou((x1, y1, x2, y2), prev_box['box'])
                        if iou > 0.3:
                            prev_cx, prev_cy = prev_box['center']
                            position_change = np.sqrt((box_center_x - prev_cx)**2 + 
                                                     (box_center_y - prev_cy)**2)
                            
                            prev_area = prev_box['area']
                            size_change = abs(area - prev_area) / max(prev_area, 1)
                            
                            motion = position_change / 50.0 + size_change * 2.0
                            if motion > best_match_motion:
                                best_match_motion = motion
                                best_motion_vector = (box_center_x - prev_cx, box_center_y - prev_cy)
                    
                    motion_score = min(best_match_motion, 1.0)
                    motion_vector = best_motion_vector
                
                # Score 4: Temporal consistency (was this person selected before?)
                temporal_score = 0
                if len(self.selection_history) > 0:
                    for prev_selection in self.selection_history:
                        for prev_box in prev_selection:
                            if self._iou((x1, y1, x2, y2), prev_box) > 0.5:
                                temporal_score = 1.0
                                break
                        if temporal_score > 0:
                            break
                
                current_detections.append({
                    'box': (x1, y1, x2, y2),
                    'center': (box_center_x, box_center_y),
                    'area': area,
                    'conf': conf,
                    'motion': motion_score,
                    'motion_vector': motion_vector,
                    'center_prox': center_score,
                    'size': size_score,
                    'temporal': temporal_score
                })
        
        # Calculate interaction scores with RELATIVE MOTION COHERENCE
        for i, det in enumerate(current_detections):
            interaction_score = 0
            max_pair_motion_coherence = 0
            
            for j, other_det in enumerate(current_detections):
                if i == j:
                    continue
                
                # Spatial proximity
                dx = det['center'][0] - other_det['center'][0]
                dy = det['center'][1] - other_det['center'][1]
                distance = np.sqrt(dx**2 + dy**2)
                
                frame_diag = np.sqrt(w**2 + h**2)
                norm_distance = distance / frame_diag
                
                proximity_score = max(0, 1 - norm_distance * 3)
                
                # Overlap
                iou = self._iou(det['box'], other_det['box'])
                overlap_score = iou * 2.0
                
                # RELATIVE MOTION COHERENCE
                motion_coherence = 0
                if self.prev_frame_data and len(det['motion_vector']) == 2 and len(other_det['motion_vector']) == 2:
                    my_motion_mag = np.sqrt(det['motion_vector'][0]**2 + det['motion_vector'][1]**2)
                    other_motion_mag = np.sqrt(other_det['motion_vector'][0]**2 + other_det['motion_vector'][1]**2)
                    
                    # Case 1: One person moving, other static OR both have some motion
                    motion_diff = abs(my_motion_mag - other_motion_mag)
                    if motion_diff > 5:
                        static_moving_bonus = min(motion_diff / 40.0, 1.0)
                        
                        if norm_distance < 0.35:
                            motion_coherence = static_moving_bonus * 1.8
                    
                    # Case 2: Both moving (coordinated action)
                    elif my_motion_mag > 3 and other_motion_mag > 3:
                        if my_motion_mag > 0 and other_motion_mag > 0:
                            dot_product = (det['motion_vector'][0] * other_det['motion_vector'][0] + 
                                         det['motion_vector'][1] * other_det['motion_vector'][1])
                            cos_sim = dot_product / (my_motion_mag * other_motion_mag)
                            
                            if abs(cos_sim) > 0.6:
                                motion_coherence = 0.9 * abs(cos_sim)
                    
                    # Case 3: QUICK BURST motion (like slapping)
                    if norm_distance < 0.25 and (my_motion_mag > 3 or other_motion_mag > 3):
                        quick_action_score = min((my_motion_mag + other_motion_mag) / 60.0, 1.0)
                        motion_coherence = max(motion_coherence, quick_action_score * 1.2)
                
                # Combine factors with MOTION COHERENCE BOOST
                pair_interaction = (
                    proximity_score * 0.4 + 
                    overlap_score * 0.3 + 
                    motion_coherence * 0.3
                )
                
                interaction_score = max(interaction_score, pair_interaction)
                max_pair_motion_coherence = max(max_pair_motion_coherence, motion_coherence)
            
            det['interaction'] = min(interaction_score, 1.0)
            det['motion_coherence'] = max_pair_motion_coherence
        
        # Calculate final action score
        for det in current_detections:
            # Check if this detection matches our locked pair
            lock_bonus = 0
            if self.locked_pair is not None and len(self.selection_history) > 0:
                last_selection = self.selection_history[-1]
                for locked_box in last_selection:
                    if self._iou(det['box'], locked_box) > 0.4:
                        lock_bonus = 0.5 * (self.lock_strength / 10.0)
                        break
            
            action_score = (
                det['conf'] * 0.07 +
                det['center_prox'] * 0.08 +
                det['size'] * 0.06 +
                det['motion'] * 0.14 +
                det['interaction'] * 0.25 +
                det['motion_coherence'] * 0.18 +
                det['temporal'] * 0.22 +
                lock_bonus
            )
            det['score'] = action_score
        
        # Update history
        self.prev_frame_data = current_detections
        self.frame_count += 1
        
        # DYNAMIC GROUP DETECTION
        if allow_dynamic_group:
            high_interaction = [d for d in current_detections if d['interaction'] > 0.4]
            
            if 2 <= len(high_interaction) <= 4:
                high_interaction.sort(key=lambda x: x['score'], reverse=True)
                selected_boxes = [d['box'] for d in high_interaction]
                self.selection_history.append(selected_boxes)
                return selected_boxes
        
        # SMART PAIR SELECTION
        if max_people and len(current_detections) >= max_people:
            best_pair = None
            best_pair_score = -1
            
            # Check locked pair first
            if self.locked_pair is not None and len(self.selection_history) > 0:
                last_selection = self.selection_history[-1]
                if len(last_selection) >= 2:
                    locked_matches = []
                    for det in current_detections:
                        for locked_box in last_selection[:2]:
                            if self._iou(det['box'], locked_box) > 0.25:
                                locked_matches.append(det)
                                break
                    
                    if len(locked_matches) >= 2:
                        det_i, det_j = locked_matches[0], locked_matches[1]
                        
                        motion_i_mag = np.sqrt(det_i['motion_vector'][0]**2 + det_i['motion_vector'][1]**2)
                        motion_j_mag = np.sqrt(det_j['motion_vector'][0]**2 + det_j['motion_vector'][1]**2)
                        complementary_motion_bonus = 0
                        if abs(motion_i_mag - motion_j_mag) > 5:
                            complementary_motion_bonus = 0.35
                        
                        avg_motion_coherence = (det_i['motion_coherence'] + det_j['motion_coherence']) / 2
                        
                        locked_pair_score = (
                            (det_i['score'] + det_j['score']) / 2 + 
                            det_i['interaction'] + det_j['interaction'] +
                            complementary_motion_bonus +
                            avg_motion_coherence * 0.5 +
                            0.6
                        )
                        
                        if locked_pair_score > 0.3:
                            self.lock_strength = min(self.lock_strength + 1, 10)
                            selected_boxes = [det_i['box'], det_j['box']]
                            self.selection_history.append(selected_boxes)
                            self.locked_pair = selected_boxes
                            return selected_boxes
            
            # Search for best pair
            for i in range(len(current_detections)):
                for j in range(i + 1, len(current_detections)):
                    det_i = current_detections[i]
                    det_j = current_detections[j]
                    
                    motion_i_mag = np.sqrt(det_i['motion_vector'][0]**2 + det_i['motion_vector'][1]**2)
                    motion_j_mag = np.sqrt(det_j['motion_vector'][0]**2 + det_j['motion_vector'][1]**2)
                    
                    complementary_motion_bonus = 0
                    if abs(motion_i_mag - motion_j_mag) > 5:
                        complementary_motion_bonus = 0.35
                    
                    temporal_boost = (det_i['temporal'] + det_j['temporal']) * 0.40
                    
                    avg_motion_coherence = (det_i['motion_coherence'] + det_j['motion_coherence']) / 2
                    
                    pair_score = (
                        (det_i['score'] + det_j['score']) / 2 + 
                        det_i['interaction'] + det_j['interaction'] +
                        temporal_boost +
                        complementary_motion_bonus +
                        avg_motion_coherence * 0.5
                    )
                    
                    if pair_score > best_pair_score:
                        best_pair_score = pair_score
                        best_pair = (i, j)
            
            if best_pair is not None and best_pair_score > 0.5:
                idx_i, idx_j = best_pair
                selected_boxes = [current_detections[idx_i]['box'], current_detections[idx_j]['box']]
                
                is_new_pair = True
                if self.locked_pair is not None:
                    iou_match_count = 0
                    for sel_box in selected_boxes:
                        for locked_box in self.locked_pair:
                            if self._iou(sel_box, locked_box) > 0.3:
                                iou_match_count += 1
                                break
                    if iou_match_count >= 2:
                        is_new_pair = False
                
                if is_new_pair:
                    if best_pair_score > 0.9:
                        print(f"   🔄 Switching to new pair (score: {best_pair_score:.2f})")
                        self.locked_pair = selected_boxes
                        self.lock_strength = 1
                    else:
                        if self.locked_pair is not None and len(self.selection_history) > 0:
                            self.lock_strength = max(self.lock_strength - 1, 0)
                            if self.lock_strength > 0:
                                print(f"   🔒 Keeping locked pair (new score: {best_pair_score:.2f} < 0.9)")
                                return self.selection_history[-1][:2]
                        self.locked_pair = selected_boxes
                        self.lock_strength = 1
                else:
                    self.lock_strength = min(self.lock_strength + 1, 10)
                    self.locked_pair = selected_boxes
                
                self.selection_history.append(selected_boxes)
                return selected_boxes
        
        # Fallback
        sorted_detections = sorted(current_detections, 
                                  key=lambda x: x['score'], 
                                  reverse=True)
        
        if max_people is None:
            selected_boxes = [d['box'] for d in sorted_detections]
        else:
            selected_boxes = [d['box'] for d in sorted_detections[:max_people]]
        
        self.selection_history.append(selected_boxes)
        return selected_boxes
    
    def detect_with_tracking(self, frame, detector, tracker, max_people=2):
        """Detect action people and maintain tracking IDs"""
        if len(self.locked_track_ids) >= 2:
            all_action_boxes = self.detect(frame, detector, max_people=None)
            tracked = tracker.update(all_action_boxes)
            
            locked_tracks = []
            for track_id, box in tracked:
                if track_id in self.locked_track_ids:
                    locked_tracks.append((track_id, box))
            
            if len(locked_tracks) >= max_people:
                return locked_tracks[:max_people]
            
            if len(locked_tracks) > 0 and len(locked_tracks) < max_people and self.locked_pair is not None:
                for locked_box in self.locked_pair:
                    found = False
                    for track_id, box in tracked:
                        if track_id not in [t[0] for t in locked_tracks]:
                            if self._iou(box, locked_box) > 0.3:
                                locked_tracks.append((track_id, box))
                                self.locked_track_ids.add(track_id)
                                found = True
                                break
                    if found and len(locked_tracks) >= max_people:
                        break
                
                if len(locked_tracks) >= max_people:
                    return locked_tracks[:max_people]
        
        action_boxes = self.detect(frame, detector, max_people=None)
        tracked = tracker.update(action_boxes)
        
        if len(tracked) <= max_people:
            if len(tracked) >= 2:
                self.locked_track_ids = {track_id for track_id, _ in tracked[:max_people]}
            return tracked
        
        scored_tracks = []
        for track_id, box in tracked:
            for det in self.prev_frame_data:
                if self._iou(box, det['box']) > 0.5:
                    scored_tracks.append((det['score'], track_id, box))
                    break
        
        scored_tracks.sort(reverse=True)
        top_tracks = [(tid, box) for _, tid, box in scored_tracks[:max_people]]
        
        if len(top_tracks) >= 2:
            self.locked_track_ids = {track_id for track_id, _ in top_tracks}
        
        return top_tracks
    
    def _iou(self, box1, box2):
        """Compute IoU between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0.0
    
    def reset(self):
        """Reset detector state"""
        self.prev_frame_data = None
        self.frame_count = 0
        self.selection_history.clear()
        self.locked_pair = None
        self.lock_strength = 0
        self.locked_track_ids = set()

# =============================
# Smoothed ROI Detection
# =============================
class SmoothedROIDetector:
    """
    Temporal smoothing for ROI detection with adaptive alpha based on motion speed.
    
    Features:
    - Adaptive alpha: automatically adjusts smoothing based on ROI motion
    - Slow motion/static scenes: Heavy smoothing (alpha=0.3) for stability
    - Fast actions: Light smoothing (alpha=0.7) for responsiveness
    - Camera shake resistance: Small jitters get filtered with heavy smoothing
    """
    def __init__(self, window_size=5, base_alpha=0.5, adaptive=True, debug=False):
        """
        Args:
            window_size: Number of historical ROIs to keep
            base_alpha: Default alpha when adaptive is disabled (0=full smoothing, 1=no smoothing)
            adaptive: Enable adaptive alpha based on motion
            debug: Print alpha values for debugging
        """
        self.window_size = window_size
        self.base_alpha = base_alpha
        self.alpha = base_alpha
        self.adaptive = adaptive
        self.debug = debug
        
        self.roi_history = deque(maxlen=window_size)
        self.smoothed_roi = None
        self.prev_roi = None
        self.alpha_history = deque(maxlen=10)  # Track alpha values for debugging
        
    def update(self, current_roi):
        """
        Update with new ROI and return smoothed result.
        
        Args:
            current_roi: Tuple of (x1, y1, x2, y2) or None
            
        Returns:
            Smoothed ROI as (x1, y1, x2, y2) tuple or None
        """
        if current_roi is None:
            if self.smoothed_roi is not None:
                return tuple(self.smoothed_roi.astype(int))
            return None
        
        self.roi_history.append(current_roi)
        
        if len(self.roi_history) == 0:
            return None
        
        # Calculate adaptive alpha based on motion
        if self.adaptive and self.prev_roi is not None:
            self.alpha = self._calculate_adaptive_alpha(current_roi, self.prev_roi)
            self.alpha_history.append(self.alpha)
            
            if self.debug:
                avg_alpha = sum(self.alpha_history) / len(self.alpha_history)
                print(f"   🎚️  Adaptive alpha: {self.alpha:.2f} (avg: {avg_alpha:.2f})")
        else:
            self.alpha = self.base_alpha
            
        # Initialize or update smoothed ROI
        if self.smoothed_roi is None:
            self.smoothed_roi = np.array(current_roi, dtype=np.float32)
        else:
            current = np.array(current_roi, dtype=np.float32)
            self.smoothed_roi = self.alpha * current + (1 - self.alpha) * self.smoothed_roi
        
        self.prev_roi = current_roi
        return tuple(self.smoothed_roi.astype(int))
    
    def _calculate_adaptive_alpha(self, current_roi, prev_roi):
        """
        Calculate adaptive alpha based on ROI motion magnitude.
        
        Logic:
        - Measures ROI center displacement and size change
        - Fast motion → higher alpha (0.6-0.7) for responsiveness
        - Slow motion → lower alpha (0.3-0.4) for stability
        - Camera shake (very small motion) → lowest alpha (0.2-0.3) for filtering
        
        Args:
            current_roi: Current ROI (x1, y1, x2, y2)
            prev_roi: Previous ROI (x1, y1, x2, y2)
            
        Returns:
            Alpha value between 0.2 and 0.7
        """
        curr = np.array(current_roi, dtype=np.float32)
        prev = np.array(prev_roi, dtype=np.float32)
        
        # Calculate ROI center displacement (normalized by typical image size)
        curr_center = np.array([(curr[0] + curr[2]) / 2, (curr[1] + curr[3]) / 2])
        prev_center = np.array([(prev[0] + prev[2]) / 2, (prev[1] + prev[3]) / 2])
        displacement = np.linalg.norm(curr_center - prev_center)
        normalized_displacement = displacement / 224.0  # Normalize by crop size
        
        # Calculate ROI size change (relative)
        curr_size = (curr[2] - curr[0]) * (curr[3] - curr[1])
        prev_size = (prev[2] - prev[0]) * (prev[3] - prev[1])
        size_change = abs(curr_size - prev_size) / max(prev_size, 1)
        
        # Combine metrics into motion score
        # Displacement is primary indicator, size change is secondary
        motion_score = normalized_displacement + (size_change * 0.5)
        
        # Map motion_score to alpha with smooth transitions
        # Very low motion (likely jitter/noise)
        if motion_score < 0.02:
            alpha = 0.25  # Maximum smoothing - filter jitter
        # Slow motion (walking, slow gestures)
        elif motion_score < 0.05:
            alpha = 0.30  # Heavy smoothing for stability
        # Low-moderate motion
        elif motion_score < 0.08:
            alpha = 0.40  # Moderate smoothing
        # Moderate motion (normal actions)
        elif motion_score < 0.12:
            alpha = 0.50  # Balanced (base alpha)
        # Fast motion (quick gestures)
        elif motion_score < 0.18:
            alpha = 0.60  # Light smoothing, more responsive
        # Very fast motion (rapid actions like punches, kicks)
        elif motion_score < 0.25:
            alpha = 0.65  # Highly responsive
        # Extreme motion (very rapid actions)
        else:
            alpha = 0.70  # Maximum responsiveness
        
        if self.debug:
            print(f"   📊 Motion score: {motion_score:.4f} (disp={normalized_displacement:.4f}, size={size_change:.4f}) → alpha={alpha:.2f}")
        
        return alpha
    
    def get_stats(self):
        """Get statistics about smoothing behavior"""
        if len(self.alpha_history) == 0:
            return None
        
        return {
            'avg_alpha': sum(self.alpha_history) / len(self.alpha_history),
            'min_alpha': min(self.alpha_history),
            'max_alpha': max(self.alpha_history),
            'current_alpha': self.alpha,
            'history_size': len(self.roi_history)
        }
    
    def reset(self):
        """Reset all state"""
        self.roi_history.clear()
        self.smoothed_roi = None
        self.prev_roi = None
        self.alpha = self.base_alpha
        self.alpha_history.clear()

# =============================
# Pose Estimation for Spatial Guidance
# =============================
class PoseExtractor:
    """Extract YOLOv11 pose keypoints to guide spatial cropping"""
    def __init__(self, model_name="yolo11n-pose.pt", conf_threshold=0.3):
        print(f"🦴 Loading pose estimation model: {model_name}")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.num_keypoints = 17
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
    def get_all_keypoints_for_visualization(self, frame):
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        
        if len(results) == 0 or results[0].keypoints is None:
            return []
        
        all_keypoints = results[0].keypoints.data.cpu().numpy()
        return [kpts for kpts in all_keypoints if np.sum(kpts[:, 2] > 0.3) >= 5]

# =============================
# Person Detection with Tracking
# =============================
yolo_people = YOLO("yolo11n.pt")

class PersonTracker:
    """Simple IoU-based person tracker"""
    def __init__(self, iou_threshold=0.3, max_lost_frames=10):
        self.tracks = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
    
    def _compute_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
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
                
                iou = self._compute_iou(det_box, track_data['box'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                self.tracks[best_track_id]['box'] = det_box
                self.tracks[best_track_id]['lost_frames'] = 0
                matched_tracks.add(best_track_id)
                matched_detections.add(det_idx)
            else:
                new_track_id = self.next_id
                self.next_id += 1
                self.tracks[new_track_id] = {'box': det_box, 'lost_frames': 0}
                matched_detections.add(det_idx)
        
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.tracks[track_id]['lost_frames'] += 1
                if self.tracks[track_id]['lost_frames'] > self.max_lost_frames:
                    del self.tracks[track_id]
        
        sorted_tracks = sorted(self.tracks.items(), key=lambda x: x[0])
        
        return [(track_id, data['box']) for track_id, data in sorted_tracks]
    
    def reset(self):
        self.tracks = {}
        self.next_id = 0

def merge_boxes(boxes):
    if len(boxes) == 0:
        return None
    if len(boxes) == 1:
        return boxes[0]

    x1_min = min(b[0] for b in boxes)
    y1_min = min(b[1] for b in boxes)
    x2_max = max(b[2] for b in boxes)
    y2_max = max(b[3] for b in boxes)
    
    return (x1_min, y1_min, x2_max, y2_max)

def crop_roi(frame, roi, output_size):
    """Crop frame to ROI at high resolution, then resize"""
    if roi is None:
        h, w, _ = frame.shape
        th, tw = output_size
        y = max(0, h // 2 - th // 2)
        x = max(0, w // 2 - tw // 2)
        crop = frame[y:y+th, x:x+tw]
        return cv2.resize(crop, output_size)

    x1, y1, x2, y2 = roi
    h, w, _ = frame.shape
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        h, w, _ = frame.shape
        th, tw = output_size
        y = max(0, h // 2 - th // 2)
        x = max(0, w // 2 - tw // 2)
        crop = frame[y:y+th, x:x+tw]
        return cv2.resize(crop, output_size)

    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, output_size)

# =============================
# Video Sample Visualization
# =============================
def visualize_training_sample(video_path, label, pose_extractor, adaptive_detector, 
                             output_path="training_sample.mp4", sample_rate=5, debug=False):
    """Creates a video sample visualization showing action detection"""
    print(f"\n🎬 Creating training sample visualization for: {video_path}")
    print(f"   Action label: {label}")
    print(f"   Detection sample rate: every {sample_rate} frame(s)")
    
    # Set debug mode for the detector if passed
    adaptive_detector.debug = debug

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0:
        fps = 30
    
    print(f"   Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Try codecs
    codecs = [('mp4v', '.mp4'), ('avc1', '.mp4'), ('XVID', '.avi'), ('MJPG', '.avi')]
    video_writer = None
    
    for codec, ext in codecs:
        try:
            output_path_with_ext = output_path.replace('.mp4', ext).replace('.avi', ext)
            fourcc = cv2.VideoWriter_fourcc(*codec)
            video_writer = cv2.VideoWriter(output_path_with_ext, fourcc, fps, (width, height + 100))
            if video_writer.isOpened():
                output_path = output_path_with_ext
                print(f"   Using codec: {codec}")
                break
            else:
                video_writer = None
        except:
            continue
    
    if video_writer is None:
        print("❌ Could not initialize video writer")
        cap.release()
        return False
    
    frame_count = 0
    successful_frames = 0
    pbar = tqdm(total=total_frames, desc="Creating visualization")
    
    person_tracker = PersonTracker(iou_threshold=0.3, max_lost_frames=10)
    action_detector = SmartActionDetector()
    roi_smoother = SmoothedROIDetector(
    window_size=3, 
    base_alpha=0.5,
    adaptive=True,  # Enable adaptive smoothing
    debug=False
    )

    
    track_colors = {}
    color_palette = [(0, 255, 0), (255, 0, 255), (255, 255, 0), (0, 255, 255)]
    
    last_tracked_people = []
    last_action_roi = None
    last_poses = []
    focus_region = "full_body"
    
    # Pose visualization connections
    pose_connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Face
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        try:
            display_frame = frame.copy()
            
            if frame_count % sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Use SMART ACTION DETECTION to find ACTION PEOPLE ONLY
                last_tracked_people = action_detector.detect_with_tracking(
                    frame_rgb, yolo_people, person_tracker, max_people=2
                )
                
                # Extract boxes from ACTION people only
                if last_tracked_people and isinstance(last_tracked_people[0], tuple):
                    boxes_only = [box for _, box in last_tracked_people]
                else:
                    boxes_only = last_tracked_people
                
                # Get ACTION-FOCUSED ROI using ADAPTIVE detection
                if pose_extractor is not None and len(boxes_only) > 0:
                    # Get ROI and focus region TOGETHER to keep them in sync
                    action_roi, focus_region = adaptive_detector.detect_action_region(
                        frame_rgb, boxes_only, pose_extractor, max_poses=2
                    )
                    
                    print(f"   Frame {frame_count}: Focus region = {focus_region}")
                    
                    # Debug motion analysis
                    adaptive_detector.debug_motion_analysis(frame_rgb, boxes_only, pose_extractor)
                    
                    # Get poses for visualization
                    if CONFIG.get('visualize_skeletons', False):
                        last_poses = []
                        results = pose_extractor.model.predict(frame_rgb, conf=pose_extractor.conf_threshold, verbose=False)
                        
                        if len(results) > 0 and results[0].keypoints is not None:
                            all_keypoints = results[0].keypoints.data.cpu().numpy()
                            
                            for action_box in boxes_only[:2]:
                                ax1, ay1, ax2, ay2 = action_box
                                
                                best_match_idx = None
                                best_match_score = 0
                                
                                for idx, kpts in enumerate(all_keypoints):
                                    if np.sum(kpts[:, 2] > 0.3) >= 5:
                                        visible_kpts = kpts[kpts[:, 2] > 0.3]
                                        pose_center = visible_kpts[:, :2].mean(axis=0)
                                        
                                        if ax1 <= pose_center[0] <= ax2 and ay1 <= pose_center[1] <= ay2:
                                            score = len(visible_kpts)
                                            if score > best_match_score:
                                                best_match_score = score
                                                best_match_idx = idx
                                
                                if best_match_idx is not None:
                                    last_poses.append(all_keypoints[best_match_idx])
                                    
                                    if len(last_poses) >= 2:
                                        break
                    else:
                        last_poses = []
                else:
                    action_roi = merge_boxes(boxes_only)
                    last_poses = []
                    focus_region = "full_body"
                    print(f"   Frame {frame_count}: No pose detection, using FULL BODY")
                
                last_action_roi = roi_smoother.update(action_roi)

            if debug and CONFIG.get('debug_motion_analysis', False):
                adaptive_detector.debug_motion_analysis(frame_rgb, boxes_only, pose_extractor)
            else:
                # Print only the focus region without detailed motion analysis
                if pose_extractor is not None and len(boxes_only) > 0:
                    action_roi, focus_region = adaptive_detector.detect_action_region(
                        frame_rgb, boxes_only, pose_extractor, max_poses=2
                    )
                    if not debug:  # Only print summary if not in debug mode
                        print(f"   Frame {frame_count}: Focus region = {focus_region}")   
            
            # Draw tracked person boxes with ACTION SCORE indicators
            for i, item in enumerate(last_tracked_people):
                if isinstance(item, tuple):
                    track_id, (x1, y1, x2, y2) = item
                    if track_id not in track_colors:
                        track_colors[track_id] = color_palette[len(track_colors) % len(color_palette)]
                    color = track_colors[track_id]
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
                    cv2.putText(display_frame, f"ACTION Person {track_id}", (x1, max(20, y1-10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                else:
                    x1, y1, x2, y2 = item
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw ACTION ROI (BLUE - thicker)
            if last_action_roi:
                x1, y1, x2, y2 = last_action_roi
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
                
                # Color code based on focus region
                if focus_region == 'upper_body':
                    color = (0, 255, 255)  # Yellow for upper body
                    region_text = "UPPER BODY ACTION"
                elif focus_region == 'lower_body':
                    color = (0, 255, 0)    # Green for lower body  
                    region_text = "LOWER BODY ACTION"
                else:
                    color = (255, 0, 0)    # Blue for full body
                    region_text = "FULL BODY ACTION"
            
            # Draw pose keypoints and skeleton
            for pose_kpts in last_poses:
                for conn in pose_connections:
                    pt1_idx, pt2_idx = conn
                    if pose_kpts[pt1_idx, 2] > 0.3 and pose_kpts[pt2_idx, 2] > 0.3:
                        pt1 = (int(pose_kpts[pt1_idx, 0]), int(pose_kpts[pt1_idx, 1]))
                        pt2 = (int(pose_kpts[pt2_idx, 0]), int(pose_kpts[pt2_idx, 1]))
                        cv2.line(display_frame, pt1, pt2, (0, 255, 255), 2)
                
                for kpt in pose_kpts:
                    if kpt[2] > 0.3:
                        cv2.circle(display_frame, (int(kpt[0]), int(kpt[1])), 4, (0, 0, 255), -1)
            
            # Create enhanced label area with debug info
            label_area = np.zeros((100, width, 3), dtype=np.uint8)
            cv2.putText(label_area, f"ACTION: {label}", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(label_area, f"Frame: {frame_count}/{total_frames}", (width-250, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            num_people = len(last_tracked_people)
            skeleton_status = "ON" if CONFIG.get('visualize_skeletons', False) else "OFF"
            
            # Color code the focus region text
            if focus_region == 'upper_body':
                focus_color = (0, 255, 255)  # Yellow
            elif focus_region == 'lower_body':
                focus_color = (0, 255, 0)    # Green
            else:
                focus_color = (255, 255, 255)  # White
            
            cv2.putText(label_area, f"Action People: {num_people} | Focus: {focus_region}", 
                       (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, focus_color, 2)
            
            cv2.putText(label_area, "Detection: Adaptive Action Region", 
                       (width-450, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            combined_frame = np.vstack([label_area, display_frame])
            video_writer.write(combined_frame)
            successful_frames += 1
            
        except Exception as e:
            print(f"⚠️  Error processing frame {frame_count}: {e}")
        
        frame_count += 1
        pbar.update(1)
    
    cap.release()
    video_writer.release()
    pbar.close()
    
    success_rate = (successful_frames / frame_count) * 100 if frame_count > 0 else 0
    print(f"✅ Visualization: {successful_frames}/{frame_count} frames ({success_rate:.1f}%)")
    print(f"✅ Output: {output_path}")
    
    return successful_frames > 0

def create_sample_visualizations(dataset, pose_extractor, num_samples=2):
    """Create visualizations for random samples"""
    print(f"\n📹 Creating {num_samples} sample visualizations...")
    
    if len(dataset.samples) == 0:
        print("❌ No samples found")
        return
    
    selected_indices = random.sample(range(len(dataset.samples)), min(num_samples, len(dataset.samples)))
    
    adaptive_detector = AdaptiveActionDetector()
    
    for i, idx in enumerate(selected_indices):
        video_path, label_idx = dataset.samples[idx]
        label_name = dataset.idx_to_label[label_idx]
        output_filename = f"sample_{i+1}_{label_name.replace(' ', '_')}.mp4"
        visualize_training_sample(
            video_path, 
            label_name,
            pose_extractor,
            adaptive_detector,
            output_filename,
            sample_rate=CONFIG.get('visualization_sample_rate', 5)
        )

# =============================
# Configuration
# =============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

CONFIG = {
    "data_path": "dataset",
    "batch_size": 2,
    "base_epochs": 25,
    "base_learning_rate": 1e-4,
    "finetune_learning_rate": 1e-5,
    "max_finetune_epochs": 15,
    "early_stopping_patience": 5,
    "min_delta": 0.001,
    "use_class_weights": True,
    "augmentation_prob": 0.3,
    "sequence_length": 16,
    "crop_size": (224, 224),
    "model_save_path": "intel_finetuned_classifier_3d.pth",
    "checkpoint_path": r"D:\movie_highlighter\checkpoints\checkpoint_latest.pth",
    "save_checkpoint_every": 5,
    "checkpoint_dir": "checkpoints",
    "min_train_per_action": 5,
    "min_val_per_action": 2,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "create_visualizations": True,
    "num_visualization_samples": 4,
    "visualization_sample_rate": 5,
    "use_roi_smoothing": True,
    "use_adaptive_cropping": True,  # Enable adaptive action region detection
    "motion_threshold": 2.0,        # Lower threshold for better detection of body part involved
    "use_pose_guided_crop": True,
    "pose_model": "yolo11n-pose.pt",
    "pose_conf_threshold": 0.3,
    "visualize_skeletons": False,
    "max_action_people": 2,
    "allow_dynamic_group": True,
    "sticky_frames": 10,
    "interaction_threshold": 0.4,
    "sampling_strategy": "temporal_stride",
    "default_stride": 4,
    "min_stride": 3,
    "max_stride": 8,
    "debug_mode": False,  # Set to True to enable verbose debug output
    "debug_motion_analysis": False,  # Separate flag for motion analysis debug
    "use_roi_smoothing": True,
    "adaptive_smoothing": True,  # Enable adaptive alpha
    "smoothing_base_alpha": 0.5,  # Base alpha when adaptive is off
    "smoothing_window_size": 5,  # History window
    "debug_smoothing": False,  # Set True to see alpha values
    "min_production_accuracy": 0.3,
}


BASE_DIR = os.getcwd()
ENCODER_XML = os.path.join(BASE_DIR, "models/intel_action/encoder/FP32/action-recognition-0001-encoder.xml")
ENCODER_BIN = os.path.join(BASE_DIR, "models/intel_action/encoder/FP32/action-recognition-0001-encoder.bin")

# =============================
# Video Loading with Adaptive Action Detection
# =============================
def load_video_normalized(path, pose_extractor=None, is_training=True, verbose=False, debug=False):
    """Load video with adaptive action-based person detection"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return []

    # Deterministic frame sampling
    sampling_strategy = CONFIG.get("sampling_strategy", "temporal_stride")
    
    if sampling_strategy == "temporal_stride":
        stride = CONFIG.get("default_stride", 4)
        stride = max(CONFIG.get("min_stride", 3), 
                    min(CONFIG.get("max_stride", 8), 
                        total_frames // (CONFIG["sequence_length"] * 2)))
        
        # Use consistent sampling - remove random offset
        indices = np.arange(0, min(CONFIG["sequence_length"] * stride, total_frames), stride)
        indices = indices[:CONFIG["sequence_length"]]
    else:
        indices = np.linspace(0, total_frames - 1, CONFIG["sequence_length"]).astype(int)
    
    output_frames = []
    crop_size = CONFIG["crop_size"]
    
    # Create NEW detectors for EACH video to prevent state leakage
    action_detector = SmartActionDetector(debug=debug)
    adaptive_detector = AdaptiveActionDetector(
        motion_threshold=CONFIG.get("motion_threshold", 5.0),
        debug=debug
    )
    roi_smoother = SmoothedROIDetector(
        window_size=5, 
        base_alpha=CONFIG.get("smoothing_base_alpha", 0.5),
        adaptive=CONFIG.get("adaptive_smoothing", True),
        debug=CONFIG.get("debug_smoothing", False) or debug
    ) if CONFIG["use_roi_smoothing"] else None

    person_tracker = PersonTracker(iou_threshold=0.3, max_lost_frames=5)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()
        if not success:
            # Handle missing frames better - repeat last valid frame
            if len(output_frames) > 0:
                output_frames.append(output_frames[-1].copy())
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect action people
        tracked = action_detector.detect_with_tracking(
            frame, yolo_people, person_tracker, 
            max_people=CONFIG.get('max_action_people', 2)
        )
        
        if tracked and isinstance(tracked[0], tuple):
            people = [box for _, box in tracked]
        else:
            people = tracked

        # Get ROI
        if CONFIG.get('use_adaptive_cropping', False) and pose_extractor is not None and len(people) > 0:
            roi, focus_region = adaptive_detector.detect_action_region(
                frame, people, pose_extractor, max_poses=2
            )
        elif len(people) > 0:
            roi = merge_boxes(people)
        else:
            roi = None
        
        # Smooth the ROI
        if roi_smoother and roi is not None:
            roi = roi_smoother.update(roi)
        
        # Crop and resize
        frame = crop_roi(frame, roi, crop_size)
        
        # More conservative augmentation
        if is_training and random.random() < CONFIG.get('augmentation_prob', 0.2):
            # Only apply ONE augmentation per frame
            aug_choice = random.random()
            
            if aug_choice < 0.33:
                # Brightness
                brightness_factor = random.uniform(0.85, 1.15)  # Less aggressive
                frame = np.clip(frame * brightness_factor, 0, 255).astype(np.uint8)
            elif aug_choice < 0.66:
                # Contrast
                contrast_factor = random.uniform(0.85, 1.15)  # Less aggressive
                mean = frame.mean(axis=(0, 1), keepdims=True)
                frame = np.clip((frame - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
            else:
                # Horizontal flip (only for symmetric actions)
                frame = np.fliplr(frame)
        
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        mean = np.array(CONFIG["mean"], dtype=np.float32)
        std = np.array(CONFIG["std"], dtype=np.float32)
        frame = (frame - mean) / std
        
        output_frames.append(frame)

    cap.release()
    
    # Better padding strategy
    if len(output_frames) == 0:
        return []
    
    if len(output_frames) < CONFIG["sequence_length"]:
        # Repeat the entire sequence cyclically instead of just padding with last frame
        while len(output_frames) < CONFIG["sequence_length"]:
            remaining = CONFIG["sequence_length"] - len(output_frames)
            to_add = min(remaining, len(output_frames))
            output_frames.extend(output_frames[:to_add])

    return np.stack(output_frames[:CONFIG["sequence_length"]], axis=0)

# =============================
# Dataset
# =============================
class VideoDataset(Dataset):
    def __init__(self, root, sequence_length=16, pose_extractor=None, is_training=True):
        self.samples = []  # Use only one attribute for samples
        self.sequence_length = sequence_length
        self.pose_extractor = pose_extractor
        self.is_training = is_training

        if not os.path.exists(root):
            print(f"Warning: Dataset path {root} does not exist")
            self.labels = []
            self.label_to_idx = {}
            self.idx_to_label = {}
            return

        class_folders = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.label_to_idx = {label: idx for idx, label in enumerate(class_folders)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.labels = class_folders
        
        print(f"📊 Detected {len(self.labels)} action classes:")
        for label, idx in self.label_to_idx.items():
            print(f"  {idx}: {label}")

        video_count = 0
        for label in self.labels:
            label_path = os.path.join(root, label)
            video_files = glob.glob(os.path.join(label_path, "*.mp4")) + \
                          glob.glob(os.path.join(label_path, "*.avi")) + \
                          glob.glob(os.path.join(label_path, "*.mov"))
            for video_path in video_files:
                self.samples.append((video_path, self.label_to_idx[label]))
                video_count += 1

        print(f"✅ Found {video_count} videos")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        frames = load_video_normalized(
            video_path, 
            pose_extractor=self.pose_extractor if CONFIG.get('use_pose_guided_crop') or CONFIG.get('use_adaptive_cropping') else None,
            is_training=self.is_training,
            verbose=False
        )
        
        if len(frames) == 0:
            frames = np.zeros((self.sequence_length, CONFIG["crop_size"][0], CONFIG["crop_size"][1], 3), dtype=np.float32)
        
        frames = np.transpose(frames, (0, 3, 1, 2))
        return torch.tensor(frames, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def get_label_mapping(self):
        return self.label_to_idx, self.idx_to_label

# =============================
# Dataset validation function
# =============================
def validate_and_split_dataset(train_dataset, val_dataset):
    """Check dataset and auto-split if validation is insufficient"""
    from sklearn.model_selection import train_test_split
    
    min_train = CONFIG['min_train_per_action']
    min_val = CONFIG['min_val_per_action']
    
    print(f"\n📊 Validating dataset size...")
    print(f"  Minimum training videos per action: {min_train}")
    print(f"  Minimum validation videos per action: {min_val}")
    
    # Count videos per action in BOTH sets
    train_counts = {}
    for video_path, label in train_dataset.samples:
        action = train_dataset.idx_to_label[label]
        if action not in train_counts:
            train_counts[action] = []
        train_counts[action].append((video_path, label))
    
    val_counts = {}
    for video_path, label in val_dataset.samples:
        action = val_dataset.idx_to_label[label]
        if action not in val_counts:
            val_counts[action] = []
        val_counts[action].append((video_path, label))
    
    # Determine which actions are valid and which need splitting
    valid_actions = []
    new_train_samples = []
    new_val_samples = []
    
    for action in train_dataset.labels:
        train_vids = train_counts.get(action, [])
        val_vids = val_counts.get(action, [])
        
        train_count = len(train_vids)
        val_count = len(val_vids)
        total_count = train_count + val_count
        
        # Check if action has enough TOTAL samples
        if train_count < min_train:
            print(f"  ❌ Action '{action}': Only {train_count} train videos (need {min_train}) - SKIPPED")
            continue
        
        # Check if we need to split for validation
        if val_count < min_val:
            # Check if we have enough TOTAL samples to split
            if total_count < min_train + min_val:
                print(f"  ⚠️  Action '{action}': {total_count} total videos (need {min_train + min_val}) - SKIPPED")
                continue
            
            # AUTO-SPLIT: We have enough total, but validation is insufficient
            print(f"  🔄 Action '{action}': {train_count} train, {val_count} val → AUTO-SPLITTING")
            
            all_videos = train_vids + val_vids
            
            # Calculate split ratio to ensure minimum validation samples
            val_ratio = max(min_val / total_count, 0.2)  # At least min_val or 20%
            val_ratio = min(val_ratio, 0.3)  # Max 30%
            
            new_train, new_val = train_test_split(
                all_videos,
                test_size=val_ratio,
                random_state=42
            )
            
            new_train_samples.extend(new_train)
            new_val_samples.extend(new_val)
            valid_actions.append(action)
            
            print(f"     ✓ Split into: {len(new_train)} train, {len(new_val)} val")
        else:
            # Action already has enough samples in both sets
            print(f"  ✅ Action '{action}': {train_count} train, {val_count} val - OK")
            new_train_samples.extend(train_vids)
            new_val_samples.extend(val_vids)
            valid_actions.append(action)
    
    # Check if we have any valid actions
    if len(valid_actions) == 0:
        print("\n❌ No actions meet minimum requirements. Please collect more videos.")
        return False, [], [], []
    
    # Summary
    print(f"\n✅ Dataset validation complete!")
    print(f"  Valid actions: {len(valid_actions)}/{len(train_dataset.labels)}")
    print(f"  Final training samples: {len(new_train_samples)}")
    print(f"  Final validation samples: {len(new_val_samples)}")
    
    if len(new_val_samples) == 0:
        print(f"\n⚠️  WARNING: No validation samples after filtering!")
        print(f"   Training will proceed but validation metrics will be unreliable.")
    
    return True, valid_actions, new_train_samples, new_val_samples


# =============================
# Class Weight Computation
# =============================
def compute_class_weights(train_dataset):
    """Compute inverse frequency weights for balanced training"""
    # Count samples per class using the current dataset's samples
    label_counts = {}
    for _, label in train_dataset.samples:  # Use .samples, not .video_samples
        label_counts[label] = label_counts.get(label, 0) + 1
    
    total_samples = len(train_dataset)
    num_classes = len(train_dataset.labels)  # Use the actual number of classes
    
    # Create weights for all current classes
    weights = []
    for class_idx in range(num_classes):
        count = label_counts.get(class_idx, 1)  # Default to 1 to avoid division by zero
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    print(f"\n⚖️  Class weights computed (inverse frequency):")
    print(f"   Total samples: {total_samples}")
    print(f"   Number of classes: {num_classes}")
    
    for idx, weight in enumerate(weights):
        # Safely get class name
        if idx in train_dataset.idx_to_label:
            class_name = train_dataset.idx_to_label[idx]
        else:
            class_name = f"Class_{idx}"
        count = label_counts.get(idx, 0)
        print(f"   {class_name}: {count} samples, weight: {weight:.4f}")
    
    # Verify we have the right number of weights
    if len(weights) != num_classes:
        print(f"⚠️  WARNING: Expected {num_classes} weights, got {len(weights)}")
        print(f"   Truncating to {num_classes} weights...")
        weights = weights[:num_classes]
    
    return torch.FloatTensor(weights)

def print_class_distribution(dataset, dataset_name="Dataset"):
    """Print class distribution statistics"""
    label_counts = {}
    for _, label in dataset.video_samples:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\n📊 {dataset_name} class distribution:")
    total = sum(label_counts.values())
    for label_idx in sorted(label_counts.keys()):
        count = label_counts[label_idx]
        percentage = (count / total) * 100
        class_name = dataset.idx_to_label[label_idx]
        print(f"   {class_name}: {count} videos ({percentage:.1f}%)")
    
    counts = list(label_counts.values())
    if len(counts) > 1:
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count
        if imbalance_ratio > 2.0:
            print(f"   ⚠️  Class imbalance detected! Ratio: {imbalance_ratio:.2f}x")
            print(f"   💡 Class weighting is ENABLED to handle this")
        else:
            print(f"   ✅ Classes are relatively balanced (ratio: {imbalance_ratio:.2f}x)")

# =============================
# Intel Feature Extractor
# =============================
class IntelFeatureExtractor:
    def __init__(self, encoder_xml, encoder_bin):
        self.ie = Core()
        self.encoder_model = self.ie.read_model(model=encoder_xml, weights=encoder_bin)
        self.encoder = self.ie.compile_model(self.encoder_model, device_name="CPU")
        
        input_tensor = self.encoder.inputs[0]
        self.input_name = input_tensor.get_any_name()
        self.input_shape = list(input_tensor.get_shape())
        print(f"Encoder input: {self.input_name}, shape: {self.input_shape}")

    def encode(self, frames_batch):
        """
        frames_batch: torch tensor or numpy array of shape (B, T, C, H, W)
        Returns: torch.FloatTensor of encoded features shape (B, T, feat_dim)
        """
        if isinstance(frames_batch, torch.Tensor):
            frames_batch = frames_batch.cpu().numpy()

        B, T, C, H, W = frames_batch.shape
        feats = []
        
        for batch_idx in range(B):
            batch_feats = []
            for time_idx in range(T):
                frame = frames_batch[batch_idx, time_idx]
                frame_batch = np.expand_dims(frame, axis=0)
                frame_batch = self._preprocess_batch(frame_batch)
                
                out = self.encoder([frame_batch])
                
                try:
                    output_node = self.encoder.output(0)
                    feat = out[output_node]
                except Exception:
                    feat = list(out.values())[0]
                
                if feat.ndim > 1:
                    feat = feat.reshape(feat.shape[0], -1)
                batch_feats.append(feat)
            
            batch_feats = np.concatenate(batch_feats, axis=0)
            feats.append(batch_feats)
        
        feats = np.stack(feats, axis=0)
        return torch.tensor(feats, dtype=torch.float32)

    def _preprocess_batch(self, batch_frames):
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        batch_frames = batch_frames * 255.0
        batch_frames = (batch_frames / 255.0 - mean) / std
        return batch_frames.astype(np.float32)

# =============================
# Model
# =============================
class EncoderLSTM(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256, num_classes=31, 
                 num_layers=2, dropout=0.3):
        """
        Enhanced classifier with:
        - 2-layer bidirectional LSTM
        - Attention mechanism
        - Layer normalization
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 2-layer bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(hidden_dim * 2)  # *2 for bidirectional
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Second normalization after attention
        self.ln2 = nn.LayerNorm(hidden_dim * 2)
        
        # Dropout before classification
        self.dropout = nn.Dropout(dropout)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence"""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 (helps with gradient flow)
                if len(param.shape) >= 1:
                    n = param.shape[0]
                    param.data[n//4:n//2].fill_(1.0)  # Forget gate
        
        for layer in [self.attention, self.classifier]:
            for module in layer:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        module.bias.data.fill_(0.01)
    
    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, feature_dim)
        Returns: (batch_size, num_classes), attention_weights
        """
        batch_size, seq_len, _ = x.shape
        
        # LSTM with 2 layers
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Layer normalization
        lstm_out = self.ln1(lstm_out)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.nn.functional.softmax(attention_weights, dim=1)
        
        # Context vector: weighted sum of LSTM outputs
        context = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Second normalization
        context = self.ln2(context)
        
        # Dropout
        context = self.dropout(context)
        
        # Classification
        logits = self.classifier(context)
        
        return logits, attention_weights.squeeze(-1)

# =============================
# Checkpoint Management
# =============================
def save_checkpoint(model, optimizer, epoch, best_val_acc, label_to_idx, idx_to_label, 
                   feature_dim, checkpoint_path, best_val_loss=None):
    """Save training checkpoint"""
    os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else '.', exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss if best_val_loss is not None else float('inf'),
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'feature_dim': feature_dim,
        'sequence_length': CONFIG['sequence_length'],
        'num_classes': len(label_to_idx),
        'config': CONFIG.copy()
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path} (epoch {epoch+1})")

def load_checkpoint(checkpoint_path, model, optimizer=None, strict=True):
    """Load training checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'], weights_only=False)
    
    # Check if number of classes matches
    saved_num_classes = checkpoint.get('num_classes', 0)
    current_num_classes = model.fc.out_features
    
    if saved_num_classes != current_num_classes:
        print(f"⚠️  Class mismatch detected:")
        print(f"   Checkpoint: {saved_num_classes} classes")
        print(f"   Current model: {current_num_classes} classes")
        print(f"   Loading shared weights only (transfer learning mode)")
        
        # Load everything except the final classification layer
        state_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()
        
        # Filter out fc layer weights
        pretrained_dict = {k: v for k, v in state_dict.items() 
                          if k in model_dict and 'fc' not in k and v.shape == model_dict[k].shape}
        
        # Update current model dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        print(f"   ✅ Loaded {len(pretrained_dict)} shared layers")
        print(f"   🆕 Final classification layer randomly initialized for {current_num_classes} classes")
        
        # Don't load optimizer state when doing transfer learning
        return {
            'epoch': -1,  # Start from epoch 0
            'best_val_acc': 0.0,
            'best_val_loss': float('inf'),
            'label_to_idx': checkpoint.get('label_to_idx', {}),
            'idx_to_label': checkpoint.get('idx_to_label', {}),
            'transfer_learning': True
        }
    
    # Normal loading when classes match
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f"⚠️  Could not load optimizer state: {e}")
    
    print(f"✅ Checkpoint loaded successfully!")
    print(f"   Resuming from epoch {checkpoint['epoch'] + 1}")
    print(f"   Best validation accuracy: {checkpoint.get('best_val_acc', 0):.4f}")
    
    return checkpoint

# =============================
# Training & Validation
# =============================
def validate_classifier(encoder, model, val_loader, device, criterion):
    """Validate model on validation set"""
    model.eval()
    total_correct = 0
    total_samples = 0
    running_loss = 0.0
    
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for frames, labels in val_loader:
            frames, labels = frames.to(device), labels.to(device)
            feats = encoder.encode(frames.cpu()).to(device)
            
            # Enhanced model returns (outputs, attention_weights)
            outputs, attention_weights = model(feats)
            loss = criterion(outputs, labels)
            
            preds = outputs.argmax(1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            running_loss += loss.item() * labels.size(0)
            
            # Per-class accuracy
            for label, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                label = int(label)
                class_total[label] = class_total.get(label, 0) + 1
                if label == pred:
                    class_correct[label] = class_correct.get(label, 0) + 1
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    avg_loss = running_loss / total_samples if total_samples > 0 else float('inf')
    
    per_class_acc = {}
    for label in class_total:
        per_class_acc[label] = class_correct.get(label, 0) / class_total[label] if class_total[label] > 0 else 0
    
    return avg_loss, accuracy, per_class_acc

def train_classifier(encoder, train_loader, val_loader, num_classes, label_to_idx, idx_to_label):
    device = torch.device(CONFIG["device"])
    
    with torch.no_grad():
        sample_frames, _ = next(iter(train_loader))
        dummy_feats = encoder.encode(sample_frames[0:1].cpu())
        feature_dim = dummy_feats.shape[-1]
    
    print(f"Feature dimension: {feature_dim}")
    
    model = EncoderLSTM(
        feature_dim=feature_dim, 
        hidden_dim=256,  # Reduced from 512 for 2-layer architecture
        num_classes=num_classes,
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Enhanced Model Architecture:")
    print(f"  - Input: (B, {CONFIG['sequence_length']}, {feature_dim})")
    print(f"  - 2-layer BiLSTM: {256} hidden units, bidirectional")
    print(f"  - Attention: Tanh-based")
    print(f"  - Parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"  - Estimated size: ~{total_params * 4 / 1e6:.1f} MB")
    
    # Class weights and criterion
    if CONFIG.get('use_class_weights', True):
        class_weights = compute_class_weights(train_loader.dataset).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        print("\n⚠️  Training without class weights")
    
    is_resuming = CONFIG.get('checkpoint_path') and os.path.exists(CONFIG['checkpoint_path'])
    
    if is_resuming:
        lr = CONFIG['finetune_learning_rate']
        print(f"🔄 Resume detected: using finetune LR {lr}")

    else:
        lr = CONFIG['base_learning_rate']
        print(f"🆕 Training from scratch: using base LR {lr}")
    
    # Optimizer with gradient clipping
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr,
        weight_decay=1e-4  # Increased for 2 layers
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=CONFIG.get('base_epochs', 25),
        eta_min=1e-6
    )
    
    start_epoch = 0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_model_state = None

    if is_resuming:
        checkpoint = load_checkpoint(CONFIG['checkpoint_path'], model, optimizer)
        if checkpoint:
            if checkpoint.get('transfer_learning', False):
                # Transfer learning mode - start fresh with lower learning rate
                start_epoch = 0
                best_val_acc = 0.0
                best_val_loss = float('inf')
                print("   🔄 Transfer learning: using pretrained features, training new classifier")
            else:
                # Normal resume
                start_epoch = checkpoint.get('epoch', 0) + 1
                best_val_acc = checkpoint.get('best_val_acc', 0.0)
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        else:
            print("⚠️  Failed to load checkpoint — starting from scratch")

    
    if is_resuming:
        max_epochs = start_epoch + CONFIG.get('max_finetune_epochs', 15)
        print(f"   Fine-tuning mode: starting at epoch {start_epoch}, will run up to epoch {max_epochs}")
    else:
        max_epochs = CONFIG.get('base_epochs', 25)
        print(f"   Fresh training mode: will run up to epoch {max_epochs}")
    
    patience_counter = 0

    for epoch in range(start_epoch, max_epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
        for frames, labels in pbar:
            frames, labels = frames.to(device), labels.to(device)
            
            with torch.no_grad():
                feats = encoder.encode(frames.cpu())
            feats = feats.to(device)
            
            # Handle attention outputs
            outputs, attention_weights = model(feats)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (important for 2-layer LSTM)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            preds = outputs.argmax(1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            running_loss += loss.item() * frames.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{total_correct/total_samples:.4f}'
            })
        
        # Update learning rate
        scheduler.step()
        
        train_loss = running_loss / total_samples if total_samples > 0 else float('inf')
        train_acc = total_correct / total_samples if total_samples > 0 else 0.0

        print(f"\nEpoch {epoch+1}/{max_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        val_loss = float('inf')
        val_acc = 0.0
        per_class_acc = {}
        if len(val_loader) > 0:
            # Use enhanced validation function
            val_loss, val_acc, per_class_acc = validate_classifier(
                encoder, model, val_loader, device, criterion
            )
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            if per_class_acc:
                print(f"  Per-class validation accuracy:")
                for label_idx in sorted(per_class_acc.keys()):
                    class_name = idx_to_label[label_idx]
                    acc = per_class_acc[label_idx]
                    status = "⚠️" if acc == 0.0 else "✓"
                    print(f"    {status} {class_name}: {acc:.4f}")

        improved = val_loss < (best_val_loss - CONFIG.get('min_delta', 0.001))
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print("   ⭐ Validation loss improved — saving best model state and resetting patience.")
        else:
            patience_counter += 1
            print(f"   No improvement ({patience_counter}/{CONFIG['early_stopping_patience']})")
            if patience_counter >= CONFIG['early_stopping_patience']:
                print("\n🛑 Early stopping triggered")
                break

        if CONFIG.get('save_checkpoint_every') and (epoch + 1) % CONFIG['save_checkpoint_every'] == 0:
            checkpoint_name = f"checkpoint_epoch_{epoch+1}.pth"
            checkpoint_path = os.path.join(CONFIG.get('checkpoint_dir', '.'), checkpoint_name)
            save_checkpoint(model, optimizer, epoch, best_val_acc, 
                          label_to_idx, idx_to_label, feature_dim, checkpoint_path, best_val_loss)
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Loaded best model (val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})")

    # Create wrapped model
    wrapped_model = ActionRecognitionModel(
        model=model,
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label,
        feature_dim=feature_dim,
        sequence_length=CONFIG['sequence_length']
    )
    
    # Save using the wrapper
    wrapped_model.save(CONFIG['model_save_path'])
    
    return wrapped_model

# =============================
# Action Recognition Model Wrapper
# =============================
class ActionRecognitionModel:
    """Wrapper class for the trained model with metadata"""
    def __init__(self, model, label_to_idx, idx_to_label, feature_dim, sequence_length):
        self.model = model
        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
    
    def save(self, path):
        """Save model and metadata"""
        torch.save(self.model.state_dict(), path)
        mapping_path = path.replace('.pth', '_mapping.json')
        mapping_data = {
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'feature_dim': self.feature_dim,
            'sequence_length': self.sequence_length,
            'model_type': 'EncoderLSTM',
            'hidden_dim': 256,
            'num_layers': 2
        }
        with open(mapping_path, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        print(f"✅ Model saved: {path}")
        print(f"✅ Mapping saved: {mapping_path}")
    
    @classmethod
    def load(cls, path, device='cpu'):
        """Load model and metadata"""
        mapping_path = path.replace('.pth', '_mapping.json')
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)
        
        model = EncoderLSTM(
            feature_dim=mapping_data['feature_dim'],
            hidden_dim=mapping_data.get('hidden_dim', 256),
            num_classes=len(mapping_data['label_to_idx']),
            num_layers=mapping_data.get('num_layers', 2)
        )
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        
        return cls(
            model=model,
            label_to_idx=mapping_data['label_to_idx'],
            idx_to_label={int(k): v for k, v in mapping_data['idx_to_label'].items()},
            feature_dim=mapping_data['feature_dim'],
            sequence_length=mapping_data['sequence_length']
        )

# =============================
# Post-Training Label Filtering
# =============================
def create_production_model(encoder, trained_model, train_loader, val_loader, 
                           device, min_val_accuracy=0.3):
    """
    Create production-ready model by removing classes that don't meet minimum accuracy.
    
    Args:
        encoder: Feature extractor
        trained_model: The ActionRecognitionModel after training
        train_loader, val_loader: Data loaders
        device: torch device
        min_val_accuracy: Minimum validation accuracy required (default: 30%)
    
    Returns:
        Filtered ActionRecognitionModel with only reliable classes
    """
    
    print(f"\n🔍 CREATING PRODUCTION MODEL")
    print(f"   Minimum required validation accuracy: {min_val_accuracy:.1%}")
    print("=" * 80)
    
    # Evaluate per-class accuracy
    trained_model.model.eval()
    class_correct = {}
    class_total = {}
    class_predictions = {}  # For debugging
    
    with torch.no_grad():
        for frames, labels in val_loader:
            frames, labels = frames.to(device), labels.to(device)
            
            # Encode frames properly
            feats = encoder.encode(frames.cpu()).to(device)
            
            # Model returns (outputs, attention_weights)
            outputs, attention_weights = trained_model.model(feats)
            preds = outputs.argmax(1)
            
            for label, pred in zip(labels, preds):
                label_item = label.item()
                pred_item = pred.item()
                
                if label_item not in class_total:
                    class_total[label_item] = 0
                    class_correct[label_item] = 0
                    class_predictions[label_item] = []
                
                class_total[label_item] += 1
                if label == pred:
                    class_correct[label_item] += 1
                
                class_predictions[label_item].append(pred_item)
    
    # Determine which classes to keep
    classes_to_keep = []
    classes_to_remove = []
    
    print(f"\n📊 Per-Class Validation Analysis:")
    print(f"{'Action':<30} {'Accuracy':<12} {'Samples':<10} {'Decision'}")
    print("-" * 80)
    
    for class_idx in sorted(class_total.keys()):
        class_name = trained_model.idx_to_label[class_idx]
        correct = class_correct.get(class_idx, 0)
        total = class_total[class_idx]
        accuracy = correct / total if total > 0 else 0
        
        # Decision criteria
        if accuracy >= min_val_accuracy:
            decision = "✅ KEEP"
            classes_to_keep.append(class_idx)
        else:
            decision = f"❌ REMOVE (too low)"
            classes_to_remove.append(class_idx)
            
            # Show what it's confused with
            if class_idx in class_predictions:
                from collections import Counter
                preds = class_predictions[class_idx]
                most_common = Counter(preds).most_common(2)
                confused_with = [trained_model.idx_to_label[idx] for idx, _ in most_common if idx != class_idx]
                if confused_with:
                    decision += f" (confused with: {', '.join(confused_with[:2])})"
        
        status = "✓" if accuracy >= min_val_accuracy else "⚠️"
        print(f"{class_name:<30} {accuracy:<12.4f} {total:<10} {decision}")
    
    # Summary
    print("\n" + "=" * 80)
    print(f"📊 SUMMARY:")
    print(f"   Total classes: {len(class_total)}")
    print(f"   Classes meeting threshold: {len(classes_to_keep)} ✅")
    print(f"   Classes below threshold: {len(classes_to_remove)} ❌")
    
    if len(classes_to_remove) > 0:
        print(f"\n🗑️  Classes to be removed from production model:")
        for idx in classes_to_remove:
            name = trained_model.idx_to_label[idx]
            acc = class_correct.get(idx, 0) / class_total[idx] if class_total.get(idx, 0) > 0 else 0
            print(f"      • {name} ({acc:.1%} accuracy)")
    
    # Create new model with only reliable classes
    if len(classes_to_remove) == 0:
        print(f"\n✅ All classes meet minimum accuracy! No filtering needed.")
        return trained_model
    
    print(f"\n🔨 Creating filtered production model...")
    
    # Create new label mappings
    new_label_to_idx = {}
    new_idx_to_label = {}
    old_to_new_idx = {}
    
    for new_idx, old_idx in enumerate(sorted(classes_to_keep)):
        old_label = trained_model.idx_to_label[old_idx]
        new_label_to_idx[old_label] = new_idx
        new_idx_to_label[new_idx] = old_label
        old_to_new_idx[old_idx] = new_idx
    
    # Create new model architecture
    new_model = EncoderLSTM(
        feature_dim=trained_model.feature_dim,
        hidden_dim=256,
        num_classes=len(classes_to_keep),
        num_layers=2,
        dropout=0.3
    ).to(device)
    
    # Copy weights for kept classes
    old_state = trained_model.model.state_dict()
    new_state = new_model.state_dict()
    
    # Copy LSTM weights (they're class-agnostic)
    for key in old_state.keys():
        if 'lstm' in key or 'ln1' in key or 'ln2' in key or 'attention' in key:
            new_state[key] = old_state[key]
    
    # Copy classifier weights for kept classes only
    old_fc_weight = old_state['classifier.3.weight']  # Final layer
    old_fc_bias = old_state['classifier.3.bias']
    
    new_fc_weight = torch.zeros(len(classes_to_keep), old_fc_weight.shape[1])
    new_fc_bias = torch.zeros(len(classes_to_keep))
    
    for old_idx in classes_to_keep:
        new_idx = old_to_new_idx[old_idx]
        new_fc_weight[new_idx] = old_fc_weight[old_idx]
        new_fc_bias[new_idx] = old_fc_bias[old_idx]
    
    new_state['classifier.3.weight'] = new_fc_weight
    new_state['classifier.3.bias'] = new_fc_bias
    
    # Copy other classifier layers
    for key in ['classifier.0.weight', 'classifier.0.bias']:
        if key in old_state:
            new_state[key] = old_state[key]
    
    new_model.load_state_dict(new_state)
    
    # Create new ActionRecognitionModel
    production_model = ActionRecognitionModel(
        model=new_model,
        label_to_idx=new_label_to_idx,
        idx_to_label=new_idx_to_label,
        feature_dim=trained_model.feature_dim,
        sequence_length=trained_model.sequence_length
    )
    
    print(f"\n✅ Production model created!")
    print(f"   Classes: {len(new_label_to_idx)} (removed {len(classes_to_remove)})")
    print(f"   Labels: {list(new_label_to_idx.keys())}")
    
    return production_model

# =============================
# Main
# =============================
if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)
    print("✓ Random seed set to 42 for reproducibility\n")
    
    # Check if model files exist
    if not os.path.exists(ENCODER_XML) or not os.path.exists(ENCODER_BIN):
        print(f"Error: Intel model files not found at:")
        print(f"  XML: {ENCODER_XML}")
        print(f"  BIN: {ENCODER_BIN}")
        print("Please download the model using the OpenVINO Model Downloader")
        exit(1)

    train_dataset = VideoDataset(os.path.join(CONFIG['data_path'], "train"))
    val_dataset = VideoDataset(os.path.join(CONFIG['data_path'], "val"))
    
    if len(train_dataset) == 0:
        print("No training samples found! Please check your dataset structure.")
        exit(1)
        
    print(f"\n📁 Initial Dataset Information:")
    print(f"  Training samples:   {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Classes detected: {train_dataset.labels}")
    
    # 🔧 VALIDATE AND AUTO-SPLIT
    is_valid, valid_actions, new_train_samples, new_val_samples = validate_and_split_dataset(
        train_dataset, 
        val_dataset
    )
    
    if not is_valid:
        print("\n❌ Training aborted due to insufficient data.")
        exit(1)
    
    # 🔧 UPDATE DATASETS WITH FILTERED/SPLIT SAMPLES
    print(f"\n🔄 Updating datasets with validated data...")
    
    # Create new label mapping for filtered actions
    new_label_to_idx = {action: idx for idx, action in enumerate(valid_actions)}
    
    # Rebuild samples with correct labels
    def map_samples(samples, idx_to_label):
        mapped = []
        for video_path, old_label in samples:
            action_name = idx_to_label.get(old_label)
            if action_name in new_label_to_idx:
                new_label = new_label_to_idx[action_name]
                mapped.append((video_path, new_label))
        return mapped
    
    # Get original mappings BEFORE updating
    train_idx_to_label_original = train_dataset.idx_to_label.copy()
    val_idx_to_label_original = val_dataset.idx_to_label.copy()
    
    # Map samples
    train_dataset.samples = map_samples(new_train_samples, train_idx_to_label_original)
    val_dataset.samples = map_samples(new_val_samples, val_idx_to_label_original)
    
    # Update dataset properties
    new_idx_to_label = {idx: action for action, idx in new_label_to_idx.items()}
    
    train_dataset.labels = valid_actions
    train_dataset.label_to_idx = new_label_to_idx.copy()
    train_dataset.idx_to_label = new_idx_to_label.copy()
    
    val_dataset.labels = valid_actions
    val_dataset.label_to_idx = new_label_to_idx.copy()
    val_dataset.idx_to_label = new_idx_to_label.copy()
    
    print(f"\n✅ Final dataset after filtering and splitting:")
    print(f"  Training samples: {len(train_dataset.samples)}")
    print(f"  Validation samples: {len(val_dataset.samples)}")
    print(f"  Classes: {valid_actions}")
    
    # Print detailed breakdown - SAFE VERSION
    print(f"\n📊 Per-class breakdown:")
    for action in valid_actions:
        if action in new_label_to_idx:
            label_idx = new_label_to_idx[action]
            train_count = sum(1 for _, label in train_dataset.samples if label == label_idx)
            val_count = sum(1 for _, label in val_dataset.samples if label == label_idx)
            print(f"  {action}: {train_count} train, {val_count} val")
    
    # Get label mappings
    label_to_idx, idx_to_label = train_dataset.get_label_mapping()
    print(f"\n📝 Final label mapping:")
    for label, idx in sorted(label_to_idx.items(), key=lambda x: x[1]):
        print(f"  {idx}: {label}")
    print()
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    # Initialize encoder
    encoder = IntelFeatureExtractor(ENCODER_XML, ENCODER_BIN)

    # Initialize pose extractor if visualization is enabled
    pose_extractor = None
    if CONFIG.get('create_visualizations', False):
        print("\n🦴 Initializing pose extractor for visualizations...")
        pose_extractor = PoseExtractor(
            model_name=CONFIG.get('pose_model', 'yolo11n-pose.pt'),
            conf_threshold=CONFIG.get('pose_conf_threshold', 0.3)
        )

    # Create sample visualizations BEFORE training
    if CONFIG.get('create_visualizations', False) and pose_extractor is not None:
        create_sample_visualizations(
            train_dataset, 
            pose_extractor,
            num_samples=CONFIG.get('num_visualization_samples', 2)
        )

    # Train
    print(f"\n🚀 Starting training...\n")
    action_model = train_classifier(
        encoder, 
        train_loader, 
        val_loader, 
        num_classes=len(valid_actions),
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label
    )
    
    # Final validation
    if len(val_loader) > 0:
        print(f"\n📊 Final Validation:")
        device = torch.device("cpu")
        
        # Create criterion for final validation
        if CONFIG.get('use_class_weights', True):
            class_weights = compute_class_weights(train_loader.dataset).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        final_val_loss, final_val_acc, final_per_class_acc = validate_classifier(
            encoder, action_model.model, val_loader, device, criterion
        )
        
        print(f"  Final Validation Loss: {final_val_loss:.4f}")
        print(f"  Final Validation Accuracy: {final_val_acc:.4f}")
        
        if final_per_class_acc:
            print(f"\n  Final per-class validation accuracy:")
            for label_idx in sorted(final_per_class_acc.keys()):
                class_name = idx_to_label[label_idx]
                acc = final_per_class_acc[label_idx]
                status = "⚠️" if acc == 0.0 else "✓"
                print(f"    {status} {class_name}: {acc:.4f}")
        
        # =====================================================================
        # FILTER ZERO-ACCURACY CLASSES FROM BASE MODEL
        # Classes with 0% val accuracy are undetectable — keeping them in the
        # model just adds noise and hurts predictions for other classes.
        # =====================================================================
        zero_acc_classes = [idx for idx, acc in final_per_class_acc.items() if acc == 0.0]
        
        if zero_acc_classes:
            zero_names = [idx_to_label[idx] for idx in zero_acc_classes]
            print(f"\n🗑️  Removing {len(zero_acc_classes)} zero-accuracy classes from BASE model:")
            for name in zero_names:
                print(f"      • {name} (0% validation accuracy — undetectable)")
            
            # Filter via create_production_model with a tiny threshold
            # This keeps anything with >0% accuracy, removing only truly dead classes
            action_model = create_production_model(
                encoder, action_model, train_loader, val_loader, device,
                min_val_accuracy=0.001  # Effectively removes only 0% classes
            )
            
            # Overwrite the base model save with filtered version
            action_model.save(CONFIG['model_save_path'])
            
            # Update label mappings so production model step uses filtered set
            label_to_idx = action_model.label_to_idx
            idx_to_label = action_model.idx_to_label
            
            print(f"   ✅ Base model updated: {len(label_to_idx)} classes remaining")
        else:
            print(f"\n   ✅ All classes have >0% accuracy — no filtering needed for base model")

    # 🔧 CREATE PRODUCTION MODEL (stricter filtering on top of base)
    production_model = create_production_model(
        encoder, 
        action_model, 
        train_loader, 
        val_loader, 
        device,
        min_val_accuracy=CONFIG.get('min_production_accuracy', 0.3)  # 30% minimum
    )
        
    # Save production model
    production_path = CONFIG['model_save_path'].replace('.pth', '_production.pth')
    production_model.save(production_path)
   
    print(f"\n✅ Training completed! Model and labels saved.")
    print(f"✓ Base model: {CONFIG['model_save_path']} ({len(action_model.label_to_idx)} classes)")
    print(f"✓ Production model: {production_path} ({len(production_model.label_to_idx)} classes)")
    print(f"\n💡 Summary:")
    print(f"  - Trained on {len(valid_actions)} actions")
    print(f"  - Used {len(train_dataset.samples)} training videos")
    print(f"  - Used {len(val_dataset.samples)} validation videos")
    print(f"  - Auto-split was applied where validation was insufficient")
    print(f"  - Base model: removed classes with 0% val accuracy")
    print(f"  - Production model: removed classes below {CONFIG.get('min_production_accuracy', 0.3):.0%} val accuracy")
