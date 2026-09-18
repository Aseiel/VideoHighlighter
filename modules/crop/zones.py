"""
zones.py — where is the action, and therefore how should the frame be split?

Samples the video, scores horizontal bands by how much is happening in them, and
turns that into a crop strategy: how many crops, and which positions they take.
This is the "focus" objective named in core.py — it decides the shape of the
output before a single frame is written.

Counting lives in people.py, per-frame following in track.py. This module runs
once per video, up front, on a sparse sample.
"""
import cv2
import numpy as np

from modules.crop.core import calculate_iou
from modules.crop.people import merge_overlapping_boxes
from modules.crop.pose import analyze_pose_activity, get_pose_keypoints_for_frame
from modules.crop.config import PERSON_DETECTION_CONF_ZONES


def analyze_region_activity(video_path, yolo_model, pose_model, sample_frames=20):
    """
    Analyze actual activity in different regions of the video.
    NOW RETURNS: zone_scores, zone_people, zone_activity, zone_positions
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define zones
    zone_width = frame_width / 3
    zones = {
        'left': (0, zone_width * 1.2),
        'center': (zone_width * 0.8, 2.2 * zone_width),
        'right': (1.8 * zone_width, frame_width)
    }

    zone_activity = {'left': [], 'center': [], 'right': []}
    zone_people_count = {'left': [], 'center': [], 'right': []}
    zone_positions = {'left': [], 'center': [], 'right': []}  # NEW: Store actual positions of action

    sample_indices = [int((i / sample_frames) * total_frames) for i in range(sample_frames)]

    for frame_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get detections
        result = yolo_model.predict(rgb, conf=PERSON_DETECTION_CONF_ZONES, classes=[0], verbose=False)

        # Get pose data, on those boxes — RTMPose is top-down (see crop/pose.py)
        frame_poses = get_pose_keypoints_for_frame(
            rgb, pose_model, conf=0.15,
            person_boxes=[tuple(map(int, b.xyxy[0])) for r in result for b in r.boxes],
        )
        raw_boxes = []
        
        # [existing detection code...]
        
        boxes = merge_overlapping_boxes(raw_boxes, iou_threshold=0.45)

        # Get pose data for activity scoring
        pose_data = {}
        for pose in frame_poses:
            pose_data[pose['bbox']] = pose['keypoints']

        # Analyze each zone
        for zone_name, (zone_start, zone_end) in zones.items():
            zone_boxes = []
            zone_activities = []
            zone_action_points = []  # NEW: Store actual action coordinates

            for box in boxes:
                box_center_x = (box[0] + box[2]) / 2
                box_width = box[2] - box[0]
                box_left = box[0]
                box_right = box[2]

                # Calculate overlap with zone
                overlap_start = max(zone_start, box_left)
                overlap_end = min(zone_end, box_right)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > box_width * 0.3:
                    zone_boxes.append(box)

                    # Calculate activity score AND collect active keypoints
                    activity_score = 0.0
                    active_points = []
                    
                    for pose_box, keypoints in pose_data.items():
                        if calculate_iou(box, pose_box) > 0.2:
                            # Get active keypoints (hands, feet, hips)
                            for idx in [9, 10, 11, 12, 15, 16]:  # wrists, hips, ankles
                                if idx < len(keypoints) and keypoints[idx][2] > 0.3:
                                    x, y = keypoints[idx][0], keypoints[idx][1]
                                    if zone_start <= x <= zone_end:  # In this zone
                                        active_points.append((x, y))

                            
                            pose_activity = analyze_pose_activity(keypoints, box)
                            activity_score = max(activity_score, pose_activity)

                    zone_activities.append(activity_score)
                    
                    # If there's significant activity, store the points
                    if activity_score > 0.2 and active_points:
                        zone_action_points.extend(active_points)

            # Store results
            zone_people_count[zone_name].append(len(zone_boxes))
            if zone_activities:
                zone_activity[zone_name].append(np.mean(zone_activities))
            else:
                zone_activity[zone_name].append(0.0)
            
            # NEW: Store action positions for this zone
            if zone_action_points:
                zone_positions[zone_name].append(zone_action_points)

    cap.release()

    # Calculate aggregate scores
    zone_scores = {}
    for zone_name in zones.keys():
        avg_people = np.mean(zone_people_count[zone_name]) if zone_people_count[zone_name] else 0
        avg_activity = np.mean(zone_activity[zone_name]) if zone_activity[zone_name] else 0
        zone_scores[zone_name] = avg_people * avg_activity

    return zone_scores, zone_people_count, zone_activity, zone_positions  # NEW: Return positions too


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


def determine_smart_crop_strategy_v2(video_path, yolo_model, pose_model=None, sample_frames=20, 
                                    people_count=0, bbox_counts=None, pose_counts=None):
    """
    ACTION-AWARE cropping: Focus on where actions happen, not just people.
    Returns: (crop_count, positions_to_use, strategy_description)
    
    Takes people_count as input to make intelligent decisions about 2-person videos
    Also uses bbox_counts and pose_counts for corner case detection
    """
    # Initialize empty lists if not provided
    if bbox_counts is None:
        bbox_counts = []
    if pose_counts is None:
        pose_counts = []
    
    # Initialize action_hotspots at the VERY BEGINNING
    action_hotspots = {}
    
    # Helper function to sort positions spatially (left to right)
    def sort_positions(positions):
        spatial_order = {"left": 0, "center": 1, "middle": 1, "right": 2}
        return sorted(positions, key=lambda p: spatial_order.get(p, 1))
    
    # ===== CORNER CASE DETECTION: Dense/Packed People =====
    # Pattern: YOLO sees 1-2, Pose sees 3-4, ALL zones have action
    if bbox_counts and pose_counts:
        yolo_max = max(bbox_counts) if bbox_counts else 0
        pose_max = max(pose_counts) if pose_counts else 0
        
        # Only do this check if we have both types of data
        if yolo_max <= 2 and pose_max >= 3:
            print(f"   🚨 POTENTIAL CORNER CASE: YOLO max={yolo_max}, Pose max={pose_max}")
            print(f"      Checking zone activity...")
            
            # Get zone analysis to confirm
            zone_scores, zone_people, zone_activity, zone_positions = analyze_region_activity(
                video_path, yolo_model, pose_model, sample_frames=15
            )
            
            # Update action_hotspots from zone_positions
            zones_to_check = ['left', 'center', 'right']
            for zone in zones_to_check:
                if zone in zone_positions and zone_positions[zone]:
                    all_points = []
                    for frame_points in zone_positions[zone]:
                        all_points.extend(frame_points)
                    
                    if all_points:
                        points_array = np.array(all_points)
                        hot_x = np.mean(points_array[:, 0])
                        hot_y = np.mean(points_array[:, 1])
                        spread_x = np.std(points_array[:, 0])
                        spread_y = np.std(points_array[:, 1])
                        
                        action_hotspots[zone] = {
                            'center': (hot_x, hot_y),
                            'spread': (spread_x, spread_y),
                            'num_points': len(all_points)
                        }
            
            # Check if all zones have action
            zone_action_potential = {}
            all_zones_active = True
            
            for zone in ['left', 'center', 'right']:
                if zone_activity[zone]:
                    max_activity = max(zone_activity[zone])
                    high_action_frames = sum(1 for activity in zone_activity[zone] if activity > 0.15)
                    action_consistency = high_action_frames / len(zone_activity[zone])
                    has_action = action_consistency >= 0.15 or max_activity >= 0.25
                    
                    zone_action_potential[zone] = {
                        'max_activity': max_activity,
                        'action_consistency': action_consistency,
                        'has_action': has_action
                    }
                    
                    if not has_action:
                        all_zones_active = False
                        print(f"      {zone.capitalize()} has no action")
                else:
                    all_zones_active = False
            
            if all_zones_active:
                print(f"   🚨 CORNER CASE CONFIRMED: Dense crowd with activity in all zones")
                print(f"      YOLO max: {yolo_max}, Pose max: {pose_max}")
                print(f"      All zones have action - people packed together")
                print(f"   ➡️ Using 3 crops to capture all activity")
                return 3, ['left', 'center', 'right'], "corner-case-dense-crowd", action_hotspots
            else:
                print(f"   ℹ️ Not a corner case - zones lack consistent action")
    
    # ===== HANDLE SINGLE PERSON EXPLICITLY =====
    if people_count == 1:
        print(f"   👤 Single person detected - NO CROP (would split body parts)")
        return 0, [], "single-person-no-crop", action_hotspots
    
    print(f"   🔍 Analyzing ACTION zones (not just people)...")
    
    # Get activity analysis
    zone_scores, zone_people, zone_activity, zone_positions = analyze_region_activity(
        video_path, yolo_model, pose_model, sample_frames
    )
    
    # ===== Calculate action hotspots for each zone =====
    zones_to_check = ['left', 'center', 'right']
    
    for zone in zones_to_check:
        if zone in zone_positions and zone_positions[zone]:
            # Collect all action points from this zone
            all_points = []
            for frame_points in zone_positions[zone]:
                all_points.extend(frame_points)
            
            if all_points:
                points_array = np.array(all_points)
                # Calculate the centroid of action
                hot_x = np.mean(points_array[:, 0])
                hot_y = np.mean(points_array[:, 1])
                
                # Also calculate spread to determine if we need wide or tight crop
                spread_x = np.std(points_array[:, 0])
                spread_y = np.std(points_array[:, 1])
                
                action_hotspots[zone] = {
                    'center': (hot_x, hot_y),
                    'spread': (spread_x, spread_y),
                    'num_points': len(all_points)
                }
                
                print(f"   🔥 {zone} action hot spot: x={hot_x:.0f}, y={hot_y:.0f}, spread={spread_x:.0f}")
    
    # ===== Calculate zone action potential =====
    print(f"   📊 ACTION Zone analysis:")
    
    # Calculate ACTION metrics (not people metrics)
    zone_action_potential = {}
    
    for zone in ['left', 'center', 'right']:
        if zone_activity[zone]:
            # Key metrics for action cropping:
            # 1. Maximum activity level (peak action)
            max_activity = max(zone_activity[zone])
            
            # 2. Percentage of frames with real action (> 0.15 = actual limb movement)
            high_action_frames = sum(1 for activity in zone_activity[zone] if activity > 0.15)
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
                'has_action': action_consistency >= 0.15 or max_activity >= 0.25
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
    
    # ===== NEW: Use hotspots to refine decisions =====
    # Calculate hotspot scores for each zone
    hotspot_scores = {}
    for zone in ['left', 'center', 'right']:
        if zone in action_hotspots:
            hotspot = action_hotspots[zone]
            # Score based on:
            # 1. Number of action points (more = more action)
            # 2. Tightness of cluster (lower spread = focused action)
            num_points = hotspot['num_points']
            spread = (hotspot['spread'][0] + hotspot['spread'][1]) / 2
            
            # Higher score for more points and tighter clusters
            if num_points > 0:
                hotspot_scores[zone] = num_points / (spread + 50)  # +50 to avoid division by zero
                print(f"   🔥 {zone} hotspot score: {hotspot_scores[zone]:.3f} ({num_points} points, spread={spread:.0f})")
            else:
                hotspot_scores[zone] = 0
        else:
            hotspot_scores[zone] = 0
    
    # Combine activity scores with hotspot scores
    combined_scores = {}
    for zone in ['left', 'center', 'right']:
        activity_score = zone_action_potential.get(zone, {}).get('action_density', 0)
        hotspot_score = hotspot_scores.get(zone, 0)
        
        # Weighted combination (70% activity, 30% hotspot clustering)
        combined_scores[zone] = activity_score * 0.7 + hotspot_score * 0.3
        
        if combined_scores[zone] > 0:
            print(f"   📊 {zone} combined: {combined_scores[zone]:.3f} (activity={activity_score:.3f}, hotspot={hotspot_score:.3f})")
    
    # Sort zones by combined score
    sorted_zones = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    top_zones = [z[0] for z in sorted_zones if z[1] > 0]
    
    # ===== HOTSPOT-BASED DECISION LOGIC =====
    
    # Case 1: Clear winner (top score much higher than others)
    if len(sorted_zones) >= 2 and sorted_zones[0][1] > sorted_zones[1][1] * 1.5 and sorted_zones[0][1] > 0.1:
        top_zone = sorted_zones[0][0]
        print(f"   🎯 Clear hotspot winner: {top_zone} (score {sorted_zones[0][1]:.3f})")
        
        # Check if single crop is enough or if we need multiple
        if people_count >= 3 and sorted_zones[1][1] > 0.05:
            # Second zone still has significant action
            top_two = [z[0] for z in sorted_zones[:2]]
            print(f"   🎯 But second zone also active → 2 crops: {top_two}")
            return 2, sort_positions(top_two), f"hotspot-2-{top_two[0]}-{top_two[1]}", action_hotspots
        else:
            return 1, [top_zone], f"hotspot-single-{top_zone}", action_hotspots
    
    # Case 2: Two strong hotspots
    elif len(sorted_zones) >= 2 and sorted_zones[1][1] > 0.1:
        top_two = [z[0] for z in sorted_zones[:2]]
        print(f"   🎯 Two strong hotspots: {top_two}")
        return 2, sort_positions(top_two), f"hotspot-two-{top_two[0]}-{top_two[1]}", action_hotspots
    
    # Case 3: All zones have hotspots
    elif len([z for z in combined_scores if combined_scores[z] > 0.05]) >= 3:
        print(f"   🎯 All zones active - using 3 crops")
        return 3, ['left', 'center', 'right'], "hotspot-all-three", action_hotspots
    
    # ===== FALLBACK TO ORIGINAL LOGIC =====
    # Only use this if hotspot-based decisions didn't apply
    
    # Case: No significant action anywhere
    if len(action_zones) == 0:
        # Fallback to presence-based
        def pick_best_zones_by_presence(zone_people, zone_activity, k=2):
            scores = []
            for z in ["left", "center", "right"]:
                avg_p = float(np.mean(zone_people[z])) if zone_people[z] else 0.0
                avg_a = float(np.mean(zone_activity[z])) if zone_activity[z] else 0.0
                score = avg_p + 0.25 * avg_a
                scores.append((score, z))
            scores.sort(key=lambda x: x[0], reverse=True)
            return [s[1] for s in scores[:k]]
        
        best2 = pick_best_zones_by_presence(zone_people, zone_activity, k=2)
        best2 = sort_positions(best2)
        print(f"   📋 No clear action - using presence-based zones: {best2}")
        return 2, best2, f"no-action-presence-{best2[0]}-{best2[1]}", action_hotspots
    
    # Case: Single action zone
    elif len(action_zones) == 1:
        zone = action_zones[0]
        print(f"   🎯 Single action zone: {zone}")
        return 1, [zone], f"single-action-{zone}", action_hotspots
    
    # Case: Two action zones
    elif len(action_zones) == 2:
        action_zones = sort_positions(action_zones)
        print(f"   🎯 Two action zones: {action_zones}")
        return 2, action_zones, f"two-action-{action_zones[0]}-{action_zones[1]}", action_hotspots
    
    # Case: Three action zones
    else:
        print(f"   🎯 Three action zones detected")
        return 3, ['left', 'center', 'right'], "three-action-zones", action_hotspots


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
