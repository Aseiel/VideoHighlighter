"""
track.py — following the action across frames, once the strategy is fixed.

The runtime half of the cropper: given a crop count and positions from zones.py,
decide where each crop window sits on every frame, and keep it there smoothly.

    ROIDetector          per-frame region of interest for one slot
    MultiActionTracker   identity/continuity across frames, jump resistance
    MultiActionDetector  the per-frame entry point the writer loop calls
    get_multi_calibration  warm-up pass that seeds the slots before writing

Jump resistance (MAX_JUMP_RATIO, JUMP_RESISTANCE_MIN_HISTORY) is the reason this
is stateful rather than a pure function of the current frame: a detector blink
must not throw the window across the frame and back.
"""
import cv2
import numpy as np
from collections import deque

from modules.crop.core import calculate_iou, prevent_overlap
from modules.crop.pose import analyze_pose_activity, get_pose_keypoints_for_frame
from modules.crop.config import (
    JUMP_RESISTANCE_MIN_HISTORY,
    MAX_EXPANSION,
    MAX_JUMP_RATIO,
    MIN_EXPANSION,
    MIN_POSE_KEYPOINTS,
    PERSON_DETECTION_CONF_TRACKING,
    ROI_CONFIDENCE_THRESHOLD,
    ROI_SMOOTHING,
    ROI_SMOOTH_WINDOW,
    USE_POSE_ESTIMATION,
    USE_POSE_FOR_ROI,
)


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
        self.frame_width = w
        self.frame_height = h

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
        """Poses for the detected people, in box order.

        Top-down estimation makes this nearly trivial: RTMPose is asked about
        each box, so a returned skeleton IS that box's skeleton. The old
        whole-frame model needed the matching this method is named for —
        estimate every skeleton in the frame, then work out which box each one
        belonged to by testing whether its centroid fell inside. That matching
        was a source of cross-assignment whenever two people overlapped, and it
        is simply gone: the correspondence is now an index.

        Skeletons too sparse to be useful are dropped rather than returned
        empty, because the caller treats a non-empty list as "pose is usable
        here" and falls back to plain box merging otherwise.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        poses = get_pose_keypoints_for_frame(
            frame_rgb, pose_model, conf=ROI_CONFIDENCE_THRESHOLD,
            person_boxes=list(person_boxes)[:max_poses],
        )

        matched_poses = []
        for pose in poses:
            kpts = pose['keypoints']
            if np.sum(kpts[:, 2] > ROI_CONFIDENCE_THRESHOLD) >= MIN_POSE_KEYPOINTS:
                matched_poses.append(kpts)
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
        """
        Get ROI based on pose keypoints - ALL keypoints are equal, no weighting
        """
        all_points = []
        
        for pose in poses:
            # Add ALL visible keypoints with equal weight
            for idx in range(17):
                if pose[idx, 2] > ROI_CONFIDENCE_THRESHOLD:
                    point = pose[idx, :2]
                    all_points.append(point)  # Each point added ONCE, no weighting
        
        if len(all_points) == 0:
            # If we have history, use last good ROI
            if len(self.roi_history) > 0:
                return self.roi_history[-1]
            # Otherwise fall back to box merge
            return self._smart_merge_boxes(person_boxes, frame_width, frame_height)
        
        all_points_array = np.array(all_points)
        
        # Use min/max to capture full body - this is the problem!
        # This always captures from head to toe
        x_min = np.min(all_points_array[:, 0])
        y_min = np.min(all_points_array[:, 1])
        x_max = np.max(all_points_array[:, 0])
        y_max = np.max(all_points_array[:, 1])
        
        # This includes head (y_min) and feet (y_max) - that's why hips aren't focused
        
        width = x_max - x_min
        height = y_max - y_min
        
        # Padding
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
        """Determine focus region based on visible keypoints - BETTER HIP/LOWER BODY DETECTION"""
        if len(poses) == 0:
            return 'full_body'

        # Count keypoints in different regions
        head_count = 0      # Nose, eyes, ears (0-4)
        upper_count = 0     # Shoulders, elbows (5-8)
        core_count = 0      # Wrists, hips (9-12) - THIS IS THE HIP AREA!
        lower_count = 0     # Knees, ankles (13-16)

        for pose in poses:
            for idx in range(17):
                if pose[idx, 2] > ROI_CONFIDENCE_THRESHOLD:
                    if idx <= 4:  # Head
                        head_count += 1
                    elif 5 <= idx <= 8:  # Upper body
                        upper_count += 1
                    elif 9 <= idx <= 12:  # Core/Hip area (wrists + hips)
                        core_count += 1
                    elif 13 <= idx <= 16:  # Lower body
                        lower_count += 1

        # Calculate total visible keypoints
        total = head_count + upper_count + core_count + lower_count
        
        if total == 0:
            return 'full_body'
        
        # Calculate percentages
        head_pct = head_count / total
        upper_pct = upper_count / total
        core_pct = core_count / total
        lower_pct = lower_count / total
        
        # Decision logic focusing on core/hip area
        if core_pct > 0.4:  # Many core/hip keypoints visible
            return 'core_body'  # New region type!
        elif lower_pct > upper_pct * 1.5:
            return 'lower_body'
        elif upper_pct > lower_pct * 1.5:
            return 'upper_body'
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
        """Smooth ROI over history with jump resistance"""
        if len(self.roi_history) == 0:
            return None
        
        # Calculate movement between last ROI and current
        if len(self.roi_history) >= 2:
            last_roi = self.roi_history[-1]
            current_roi = self.roi_history[-2]  # Actually previous
            
            # If jump too big, reject it
            last_center = ((last_roi[0] + last_roi[2])//2, (last_roi[1] + last_roi[3])//2)
            curr_center = ((current_roi[0] + current_roi[2])//2, (current_roi[1] + current_roi[3])//2)
            
            jump_distance = np.sqrt((curr_center[0] - last_center[0])**2 + 
                                (curr_center[1] - last_center[1])**2)
            
            if jump_distance > MAX_JUMP_RATIO * self.frame_width:
                # Reject jump, use last good ROI
                return last_roi
        
        # Normal smoothing
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

        if not hasattr(self, 'initialized'):
            self.initialized = False
        
        if not self.initialized and w > 0 and h > 0:
            # Create default boxes for each position
            default_width = int(w * 0.3)
            default_height = int(h * 0.5)
            default_y = int((h - default_height) / 2)  # ← THIS IS THE CULPRIT!
            
            for action_idx in range(self.max_actions):
                if action_idx == 0:  # left
                    default_x = int(w * 0.1)
                elif action_idx == 1:  # middle
                    default_x = int((w - default_width) / 2)
                else:  # right
                    default_x = int(w * 0.7)
                    
                default_box = (default_x, default_y, 
                            default_x + default_width, 
                            default_y + default_height)
                
                # Add to history so prediction can work immediately
                for _ in range(3):  # Add 3 copies to build history
                    self.histories[action_idx].append(default_box)
                self.last_centers[action_idx] = (default_x + default_width/2)
                
                # Also set as last_good for fallback
                if hasattr(self, 'last_good_actions'):
                    self.last_good_actions[action_idx] = default_box
            
            self.initialized = True

            if frame_idx < 10:  # Only print once at beginning
                print(f"🎯 Initialized default boxes for all {self.max_actions} positions")

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

        for slot_i, action_idx in enumerate(active_actions_indicies):
            if assigned_per_slot[slot_i] is None:
                # This slot has no detection - predict from history
                if len(self.histories[action_idx]) >= 3:
                    # Predict where this person should be based on movement
                    last_positions = list(self.histories[action_idx])[-3:]
                    if len(last_positions) >= 2:
                        # Get last box and previous box
                        last_box = last_positions[-1]
                        prev_box = last_positions[-2]
                        
                        # Calculate centers
                        last_center = (last_box[0] + last_box[2]) / 2
                        prev_center = (prev_box[0] + prev_box[2]) / 2
                        
                        # Calculate movement
                        movement = last_center - prev_center
                        
                        # Predict next center
                        predicted_center = last_center + movement
                        
                        # Use last box dimensions
                        box_width = last_box[2] - last_box[0]
                        box_height = last_box[3] - last_box[1]
                        
                        # Create predicted box (keep same y position)
                        predicted_box = (
                            int(max(0, predicted_center - box_width/2)),
                            last_box[1],
                            int(min(w, predicted_center + box_width/2)),
                            last_box[3]
                        )
                        
                        assigned_per_slot[slot_i] = predicted_box
                        
                        if frame_idx % 30 == 0:  # Print occasionally
                            print(f"🔮 Predicted position for {positions[slot_i]} (no detection)")

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
        
        # Only apply gentle horizontal positioning for left/right
        if action_idx == 0:  # left
            if x1 < w * 0.1:  # Keep it from hugging the edge
                x1 = int(w * 0.05)
                x2 = int(x1 + box_w)
        elif action_idx == 1:  # center
            # Optional: very gentle centering pull, but keep y position
            ideal_x = max(0, (w - box_w) // 2)
            # Blend between current and center (70% current, 30% center)
            x1 = int(x1 * 0.7 + ideal_x * 0.3)
            x2 = int(x1 + box_w)
        else:  # right
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
                is_hip_region = (
                    area_ratio > 0.01 and
                    0.6 < aspect < 1.4 and
                    box_h > h * 0.15 and
                    box_h < h * 0.4
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
        if pose_model and USE_POSE_ESTIMATION and boxes:
            # Keyed by box so the activity lookup below can match a slot's
            # action rectangle to a skeleton by IoU. Top-down means every key
            # here is a box the detector actually proposed, rather than a
            # skeleton's own bounding box that may not correspond to any of
            # them — so that IoU match now always has a perfect candidate.
            for pose in get_pose_keypoints_for_frame(
                frame, pose_model, conf=0.3, person_boxes=boxes
            ):
                pose_data[tuple(pose['bbox'])] = pose['keypoints']

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
