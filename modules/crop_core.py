"""
crop_core.py — shared cropping primitives.

Pure geometry + render helpers used by BOTH the focus cropper (crop_actions.py)
and the avoid cropper (crop_avoid.py, coming next). Nothing in here knows about
zones, people-counting, identities, or "focus vs avoid" — it only turns a chosen
crop rectangle into pixels on disk, and provides the box math both policies need.

Extracted verbatim from crop_actions.py (no behaviour change) so the smoothing /
expansion / overlap logic exists in exactly one place. Fix a jump-resistance or
padding bug here and both croppers get it.

What deliberately did NOT move here:
  - get_multi_calibration(): uses MultiActionTracker (focus-specific), stays in
    crop_actions.py.
  - the zone brain (determine_smart_crop_strategy_v2, analyze_region_activity,
    count_people_in_video, MultiActionDetector / MultiActionTracker, ROIDetector):
    that's "where is the action" = the focus objective. Stays in crop_actions.py.
"""

import cv2
import numpy as np
from collections import deque


# ── geometry constants the helpers below depend on ───────────────────────────
# (moved here with the functions that use them; crop_actions.py imports these
#  back from crop_core so its own references keep working.)
OVERLAP_MARGIN = 5
FALLBACK_BOX_EXPANSION = 0.50   # expansion for fallback boxes (poor/lost track)
MIN_BOX_WIDTH_RATIO = 0.35
MIN_BOX_HEIGHT_RATIO = 0.40


# ── IoU ───────────────────────────────────────────────────────────────────────
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


# ── overlap prevention ─────────────────────────────────────────────────────────
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


# ── box expansion ──────────────────────────────────────────────────────────────
def expand_box(box, frame_shape, frame_count, action_idx=0, margin=0.2,
               is_fallback=False, pose_activity=0.0):
    if box is None:
        return None

    h, w = frame_shape[:2]
    x1, y1, x2, y2 = map(int, box)

    if is_fallback:
        margin = FALLBACK_BOX_EXPANSION

    bw, bh = x2 - x1, y2 - y1
    ew = int(bw * margin)
    eh = int(bh * margin)

    # VERTICAL EXPANSION - More aggressive for lower body activity
    # If pose activity indicates leg/hip movement, add extra vertical padding
    leg_activity_bonus = 0
    if pose_activity > 0.3:  # Significant activity detected
        leg_activity_bonus = int(bh * 0.15)  # Add 15% more vertical space

    # Calculate desired vertical padding
    target_top_pad = eh
    target_bottom_pad = eh + leg_activity_bonus  # More space at bottom for legs

    # Apply vertical expansion
    y1_new = max(0, y1 - target_top_pad)
    y2_new = min(h, y2 + target_bottom_pad)

    # HORIZONTAL expansion
    left_exp = min(ew, x1)
    right_exp = min(ew, w - x2)
    x1_new = max(0, x1 - left_exp)
    x2_new = min(w, x2 + right_exp)

    # Ensure minimum size
    min_width = int(w * MIN_BOX_WIDTH_RATIO)
    min_height = int(h * MIN_BOX_HEIGHT_RATIO)

    # Check if we need to expand further
    current_width = x2_new - x1_new
    current_height = y2_new - y1_new

    # If still too small, expand from center
    if current_height < min_height:
        center_y = (y1_new + y2_new) // 2
        y1_new = max(0, center_y - min_height // 2)
        y2_new = min(h, center_y + min_height // 2)

    if current_width < min_width:
        center_x = (x1_new + x2_new) // 2
        x1_new = max(0, center_x - min_width // 2)
        x2_new = min(w, center_x + min_width // 2)

    return (x1_new, y1_new, x2_new, y2_new)


# ── pad a crop to the target output size (letterbox) ────────────────────────────
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


# ── safe crop with region-aware vertical recentering ────────────────────────────
def safe_crop(frame, box, action_idx=0, default_scale=0.25, focus_region='full_body'):
    """
    Safe cropping with region awareness - NOW WITH CORE/HIP FOCUS

    Args:
        frame: Input frame
        box: (x1, y1, x2, y2) bounding box
        action_idx: Index of action (0=left, 1=center, 2=right)
        default_scale: Default crop scale if box is None
        focus_region: 'full_body', 'upper_body', 'lower_body', or 'core_body'

    Returns:
        Cropped frame region
    """
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

    # Clamp to frame boundaries
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Store original dimensions
    original_box_w = x2 - x1
    original_box_h = y2 - y1

    # Skip if box is invalid
    if original_box_w <= 0 or original_box_h <= 0:
        # Return a default center crop
        size = int(min(h, w) * 0.3)
        x1 = w//2 - size//2
        y1 = h//2 - size//2
        return frame[y1:y1+size, x1:x1+size]

    # === ADJUST VERTICAL POSITION BASED ON FOCUS REGION ===

    if focus_region == 'core_body':
        # Core/hip focus - aim for region around 60-70% down from top
        new_center_y = int(y1 + original_box_h * 0.65)

        half_height = original_box_h // 2
        y1_new = max(0, new_center_y - half_height)
        y2_new = min(h, new_center_y + half_height)

        if y2_new > y1_new:
            y1, y2 = y1_new, y2_new

            if (y2 - y1) < original_box_h * 0.6:
                y1 = max(0, new_center_y - original_box_h // 2)
                y2 = min(h, new_center_y + original_box_h // 2)

    elif focus_region == 'lower_body':
        shift_y = int(original_box_h * 0.15)
        y1_new = min(h - original_box_h, y1 + shift_y)
        y2_new = y1_new + original_box_h

        if y2_new <= h and y1_new >= 0:
            y1, y2 = y1_new, y2_new

    elif focus_region == 'upper_body':
        shift_y = int(original_box_h * 0.10)
        y1_new = max(0, y1 - shift_y)
        y2_new = y1_new + original_box_h

        if y2_new <= h:
            y1, y2 = y1_new, y2_new

    # === ENSURE MINIMUM SIZE ===
    box_w = x2 - x1
    box_h = y2 - y1

    min_width = int(w * 0.25)
    min_height = int(h * 0.25)

    if box_w < min_width and box_w < w * 0.2:
        cx = (x1 + x2) // 2
        x1 = int(max(0, cx - min_width // 2))
        x2 = int(min(w, cx + min_width // 2))
        box_w = x2 - x1

    if box_h < min_height and box_h < h * 0.2:
        cy = (y1 + y2) // 2
        y1 = int(max(0, cy - min_height // 2))
        y2 = int(min(h, cy + min_height // 2))
        box_h = y2 - y1

    # === ENFORCE REASONABLE ASPECT RATIO ===
    aspect_ratio = box_w / max(box_h, 1)

    if aspect_ratio > 2.2:
        target_h = int(box_w / 1.5)
        cy = (y1 + y2) // 2
        y1_new = max(0, cy - target_h // 2)
        y2_new = min(h, cy + target_h // 2)

        if y2_new - y1_new > box_h * 0.5:
            y1, y2 = y1_new, y2_new

    elif aspect_ratio < 0.4:
        target_w = int(box_h * 0.7)
        cx = (x1 + x2) // 2
        x1_new = max(0, cx - target_w // 2)
        x2_new = min(w, cx + target_w // 2)

        if x2_new - x1_new > box_w * 0.5:
            x1, x2 = x1_new, x2_new

    # Final clamp to frame boundaries
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))

    # Final validation
    if x2 <= x1 or y2 <= y1:
        size = int(min(h, w) * default_scale)
        x1 = w//2 - size//2
        y1 = h//2 - size//2
        return frame[y1:y1+size, x1:x1+size]

    return frame[y1:y2, x1:x2]


# ── temporal smoothing of crop windows ──────────────────────────────────────────
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