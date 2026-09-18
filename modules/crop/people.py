"""
people.py — how many people are in this video?

Answers one question, per sampled frame and then over the whole sample: the
count that decides how many crops the video gets. Everything here is counting;
where they are and how to follow them is zones.py and track.py.

THE TWO-SIGNAL DESIGN, AND WHY IT IS CURRENTLY ONE SIGNAL
    count_people_in_video() was written to combine independent evidence — boxes
    from the detector, skeletons from pose — precisely because each fails in a way
    the other does not: boxes merge when people stand close, skeletons stay
    separate; skeletons vanish at small scale, boxes survive. The upgrade gates
    encode that (`bbox_2plus >= 0.30 AND pose_2plus >= 0.30`).

    With pose dormant (see pose.py) the second signal is flat zero, so those gates
    cannot open and the count is whatever the boxes say. Read the gates as
    "currently unreachable", not as "tuned too strictly".

THE OTHER HALF OF THAT, WHICH IS A PLAIN BUG
    The partial-person path deliberately runs at low confidence — legs-only and
    torso-only boxes are weak detections by nature, hence PERSON_DETECTION_CONF
    = 0.10 and "Lowered from 0.4 - catches partial people" in config.py. But the
    detector is constructed with score_thr=0.40 and filters inside OpenVINO,
    before any `conf=` argument is looked at, so nothing below 0.40 ever arrives
    and the whole partial path is starved. See actions.py, where the detector is
    built.
"""
import cv2
import numpy as np
from collections import Counter

from modules.crop.core import calculate_iou
from modules.crop.pose import (
    bbox_has_pose_support,
    cluster_keypoints_by_person,
    get_pose_keypoints_for_frame,
)
from modules.crop.config import (
    ADJACENCY_BONUS,
    COMBINE_BBOX_AND_POSE_COUNTS,
    KEYPOINT_CLUSTER_RADIUS,
    MIN_KEYPOINT_CLUSTER_SIZE,
    MIN_PERSON_AREA_RATIO,
    PARTIAL_PERSON_MIN_AREA_RATIO,
    PERSON_DETECTION_CONF,
    POSE_KEYPOINT_CLUSTER_DETECTION,
    POSE_VALIDATION_CONF_THRESHOLD,
    USE_PARTIAL_PERSON_DETECTION,
)


def detect_single_person_split(merged_boxes, frame_poses, frame_w, frame_h):
    """
    Detect when 2 merged bboxes are actually parts of the SAME person
    
    Returns: (corrected_count, was_split_detected, split_reason)
    """
    if len(merged_boxes) != 2:
        return len(merged_boxes), False, ""
    
    box1, _ = merged_boxes[0] if isinstance(merged_boxes[0], tuple) and len(merged_boxes[0]) == 2 else (merged_boxes[0], False)
    box2, _ = merged_boxes[1] if isinstance(merged_boxes[1], tuple) and len(merged_boxes[1]) == 2 else (merged_boxes[1], False)
    
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Sort by vertical position (top box first)
    if y1_1 > y1_2:
        box1, box2 = box2, box1
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
    
    cx1 = (x1_1 + x2_1) / 2
    cx2 = (x1_2 + x2_2) / 2
    w1 = x2_1 - x1_1
    w2 = x2_2 - x1_2
    h1 = y2_1 - y1_1
    h2 = y2_2 - y1_2
    
    # CHECK 1: Vertically stacked boxes (torso on top, legs on bottom)
    horizontal_overlap = abs(cx1 - cx2) < max(w1, w2) * 0.7
    vertical_gap = y1_2 - y2_1  # Gap between bottom of top box and top of bottom box
    vertically_adjacent = -h1 * 0.3 < vertical_gap < frame_h * 0.15  # Allow small gap or overlap
    
    if horizontal_overlap and vertically_adjacent:
        # Additional check: combined height should be reasonable for one person
        combined_h = y2_2 - y1_1
        if combined_h < frame_h * 0.95 and combined_h > frame_h * 0.3:
            # Check pose: should only have 1 pose skeleton
            pose_count = len(frame_poses) if frame_poses else 0
            if pose_count <= 1:
                return 1, True, "vertically-stacked-single-pose"
    
    # CHECK 2: High IoU overlap (>0.35) - boxes mostly covering same area
    iou = 0.0
    ix1 = max(x1_1, x1_2)
    iy1 = max(y1_1, y1_2)
    ix2 = min(x2_1, x2_2)
    iy2 = min(y2_1, y2_2)
    if ix2 > ix1 and iy2 > iy1:
        intersection = (ix2 - ix1) * (iy2 - iy1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        iou = intersection / union if union > 0 else 0
    
    if iou > 0.35:
        return 1, True, f"high-iou-overlap-{iou:.2f}"
    
    # CHECK 3: One box contains the other (containment)
    contained = (x1_1 <= x1_2 and y1_1 <= y1_2 and x2_1 >= x2_2 and y2_1 >= y2_2) or \
                (x1_2 <= x1_1 and y1_2 <= y1_1 and x2_2 >= x2_1 and y2_2 >= y2_1)
    if contained:
        return 1, True, "containment"
    
    return 2, False, ""


def merge_overlapping_boxes(raw_boxes, iou_threshold=0.5):
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

        # ===== METHOD 1: BOUNDING BOX DETECTION (Enhanced) =====
        result = yolo_model.predict(rgb, conf=PERSON_DETECTION_CONF, classes=[0], verbose=False)

        # ===== Pose, on the boxes just found =====
        # Detection now runs first. It used to be the other way round, from when
        # pose searched the whole frame independently; RTMPose is top-down, so
        # the boxes are its input rather than a thing it second-guesses. What
        # validates a weak box below is whether a skeleton was found INSIDE it.
        frame_poses = get_pose_keypoints_for_frame(
            rgb, pose_model, conf=0.15,
            person_boxes=[tuple(map(int, b.xyxy[0])) for r in result for b in r.boxes],
        )
        
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
            iou_threshold=0.5
        )
        bbox_count = len(merged_bbox)

        # ===== SPLIT DETECTION =====
        if bbox_count == 2:
            split_corrected_count, split_was_detected, split_reason = detect_single_person_split(
                merged_bbox, frame_poses, w, h
            )
            if split_was_detected:
                bbox_count = split_corrected_count
                if idx % 10 == 0:
                    print(f"  🔬 Frame {idx}: Split detected! 2→{split_corrected_count} ({split_reason})")

        # ===== METHOD 2: POSE SKELETON COUNTING =====
        pose_count = 0
        raw_pose_count = 0
        keypoint_clusters = []
        
        if pose_model and POSE_KEYPOINT_CLUSTER_DETECTION and frame_poses:
            # No second inference: frame_poses already holds one skeleton per
            # detected box, from the top-down pass above. Under the old
            # whole-frame model this ran the network a second time at a lower
            # threshold to find skeletons the first pass missed; top-down has
            # nothing extra to find, so counting what we have is both cheaper
            # and honest about what the signal now is.
            all_keypoints = [fp['keypoints'] for fp in frame_poses]

            # METHOD 2a: RAW skeleton count — boxes with a real skeleton in them
            for kpts in all_keypoints:
                visible_count = sum(1 for k in kpts if len(k) >= 3 and k[2] > 0.15)
                if visible_count >= MIN_KEYPOINT_CLUSTER_SIZE:
                    raw_pose_count += 1

            # METHOD 2b: Cluster-based count (backup, more conservative)
            keypoint_clusters = cluster_keypoints_by_person(
                all_keypoints,
                min_keypoints=MIN_KEYPOINT_CLUSTER_SIZE,
                radius=KEYPOINT_CLUSTER_RADIUS
            )
            cluster_count = len(keypoint_clusters)

            # Use the HIGHER of raw vs clustered
            pose_count = max(raw_pose_count, cluster_count)

            if raw_pose_count != cluster_count:
                print(f"    🔬 Pose: raw_skeletons={raw_pose_count}, clustered={cluster_count} → using {pose_count}")

        # ===== METHOD 3: COMBINED ANALYSIS =====
        combined_count = bbox_count
        
        if COMBINE_BBOX_AND_POSE_COUNTS:
            # For dense scenes, raw pose count is most reliable
            # because pose skeletons are separated even when bboxes overlap
            if raw_pose_count > bbox_count:
                combined_count = raw_pose_count
            else:
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

    # Decision logic: use MODE-FIRST approach (more robust than max)
    # Previously used max(candidate_counts) which favored occasional split detections
    
    # Mode-first: use the most frequent count if it dominates
    mode_count = most_common[0][0]
    mode_freq = most_common[0][1] / len(combined_counts)
    
    print(f"  📊 Mode: {mode_count} (appears {mode_freq:.0%} of frames)")
    
    if mode_freq >= 0.40:
        # Strong mode - trust it
        final_count = mode_count
        print(f"  ✅ Using mode (strong): {final_count}")
    else:
        # No strong mode - use median (robust to outliers)
        final_count = int(round(median_count))
        print(f"  ✅ Using median (no strong mode): {final_count}")
    
    # Upgrade to higher count ONLY with strong multi-method agreement
    # For upgrading 1→2: require BOTH bbox AND pose to find 2+ in ≥30% of frames
    if final_count == 1 and max_count >= 2:
        bbox_2plus = sum(1 for c in bbox_counts if c >= 2) / len(bbox_counts) if bbox_counts else 0
        pose_2plus = sum(1 for c in pose_counts if c >= 2) / len(pose_counts) if pose_counts else 0
        if bbox_2plus >= 0.30 and pose_2plus >= 0.30:
            final_count = 2
            print(f"  ⬆️ Upgraded 1→2: bbox_2+={bbox_2plus:.0%}, pose_2+={pose_2plus:.0%}")
        else:
            print(f"  ℹ️ Staying at 1: bbox_2+={bbox_2plus:.0%}, pose_2+={pose_2plus:.0%} (need both ≥30%)")

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
        
        # Require 3+ to appear in at least 20% of frames from BOTH bbox AND pose methods
        # OR combined ≥35% (stricter than before)
        both_agree = bbox_3_pct >= 0.20 and pose_3_pct >= 0.20
        combined_strong = combined_3_pct >= 0.35
        
        if both_agree or combined_strong:
            print(f"  ⚠️ Strong evidence of 3: bbox={bbox_3_pct:.0%}, pose={pose_3_pct:.0%}, combined={combined_3_pct:.0%}")
            final_count = max(final_count, 3)
        else:
            print(f"  ℹ️ Max=3 but insufficient agreement: bbox={bbox_3_pct:.0%}, pose={pose_3_pct:.0%}, combined={combined_3_pct:.0%}")

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
