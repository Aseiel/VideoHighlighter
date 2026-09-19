"""
actions.py — the focus cropper: orchestration and the batch CLI.

The top of the package. Owns the order of operations (count -> strategy ->
calibrate -> write) and the batch driver that walks INPUT_FOLDER. Every decision
it makes is delegated: people.py counts, zones.py picks the strategy, track.py
follows, debug.py draws, core.py turns a rectangle into pixels.

Run it directly:  python -m modules.crop.actions
"""
import cv2
import numpy as np
import os
import glob
import shutil
import time
from collections import deque

from modules.crop.core import (
    FALLBACK_BOX_EXPANSION,
    MultiSmoother,
    expand_box,
    pad_to_size,
    prevent_overlap,
    safe_crop,
)
from modules.crop.debug import (
    create_debug_video_writer,
    create_enhanced_debug_frame,
    create_side_by_side_frame,
    visualize_crop_process,
)
from modules.crop.people import count_people_in_video
from modules.crop.track import (
    MultiActionDetector,
    calculate_motion_expansion,
    get_multi_calibration,
)
from modules.crop.zones import (
    analyze_region_activity,
    determine_smart_crop_strategy_v2,
    get_crop_positions,
)
from modules.crop.config import (
    BOX_EXPANSION,
    DETECTOR_SCORE_FLOOR,
    CALIBRATION_FRAMES,
    DEBUG_CREATE_VIDEOS,
    DEBUG_MODE,
    DEBUG_OUTPUT_FOLDER,
    DEBUG_SAMPLES,
    DEBUG_SHOW_METRICS,
    DEBUG_VIDEO_FOLDER,
    DEBUG_VIDEO_SIDE_BY_SIDE,
    INPUT_FOLDER,
    MAX_PEOPLE,
    MIN_PEOPLE_REQUIRED,
    OUTPUT_FOLDER,
    PADDING_COLOR,
    PEOPLE_SAMPLE_FRAMES,
    PERSON_DETECTION_CONF,
    PERSON_DETECTION_CONF_TRACKING,
    POSE_VALIDATION_CONF_THRESHOLD,
    SMOOTHING_WINDOW,
    USE_POSE_ESTIMATION,
    USE_POSE_FOR_ROI,
    USE_ROI_DETECTION,
)


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


def process_video_with_dynamic_crops(input_path, output_folder, yolo_model, crop_count, 
                                    positions_override=None, people_info=None, action_hotspots=None):
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
        from modules.vision.pose_backend import build_pose_estimator
        pose_model = build_pose_estimator()
        if pose_model is None:
            # Still a supported state: every pose check tolerates None and
            # falls back to person boxes. Say so rather than failing, because
            # the IR is an optional download (tools/get_rtmpose_model.py).
            print("ℹ️ Pose-based ROI detection unavailable — using person boxes")

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
                        base_margin=BOX_EXPANSION,
                        pose_activity=pose_activity
                    )
                    
                elif missing > 0 and missing <= 8 and hist_len >= 5:
                    # TRACKED with good history
                    status = "TRACKED-good"
                    use_fallback_expansion = False
                    adaptive_margin = BOX_EXPANSION
                    
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
                    is_fallback=use_fallback_expansion,
                    pose_activity=pose_activity
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
    # Created here rather than at import time. This module used to run
    # os.makedirs() at module level, so merely importing it — from a test, a
    # REPL, or another module — littered the current working directory with
    # output_videos/ and debug_visualizations/.
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    if DEBUG_MODE and DEBUG_OUTPUT_FOLDER:
        os.makedirs(DEBUG_OUTPUT_FOLDER, exist_ok=True)

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

    print("📦 Loading person detector (YOLOX)...")
    from modules.vision.detection_backend import YoloxPeopleDetector
    # score_thr, NOT the per-call conf= argument, is what YOLOX filters on:
    # YoloxOpenVINODetector applies `cls_scores > self.score_thr` inside
    # inference, and the shim's conf= only filters what survived that. Its
    # default is 0.40, so every threshold in config.py below 0.40 —
    # PERSON_DETECTION_CONF = 0.10, the zone and tracking variants, the whole
    # partial-person path — was being starved of boxes it was written to see.
    # Build the detector permissively and let each call site's conf= do the
    # filtering it already thinks it is doing.
    yolo = YoloxPeopleDetector(score_thr=DETECTOR_SCORE_FLOOR)
    print("✅ Person detector loaded")

    # Load pose model for activity analysis
    pose_model = None
    if USE_POSE_ESTIMATION or USE_ROI_DETECTION:
        print("📦 Loading pose estimator (RTMPose)...")
        from modules.vision.pose_backend import build_pose_estimator
        pose_model = build_pose_estimator(auto_install=True)
        if pose_model is None:
            print("ℹ️ Pose estimation unavailable — continuing with person boxes")
        else:
            print("✅ Pose estimator loaded")

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

        # ===== CORNER CASE OVERRIDE - MODIFIED CONDITION =====
        bbox_counts = people_info.get('bbox_counts', [])
        pose_counts = people_info.get('pose_counts', [])

        if bbox_counts and pose_counts:
            low_yolo_frames = sum(1 for c in bbox_counts if c <= 1) / len(bbox_counts) if bbox_counts else 0
            pose_max = max(pose_counts) if pose_counts else 0
            
            print(f"   📊 Corner check: Low YOLO frames={low_yolo_frames:.0%}, Pose max={pose_max}")
            
        if low_yolo_frames >= 0.7 and pose_max >= 3 and people_count <= 2:
            print(f"   🚨 CORNER CASE DETECTED: YOLO sees 0-1, Pose sees up to {pose_max}")
            
            # SIMPLIFIED ZONE CHECK - just check if pose exists at all
            if pose_max >= 3:  # If pose detected 3+ skeletons, that's enough evidence
                print(f"   ✅ Pose detected {pose_max} skeletons - Overriding without zone check!")
                
                # Use pose count to determine crop count
                crop_count = min(pose_max, 3)  # 3 crops max
                positions = ['left', 'center', 'right'][:crop_count]
                strategy = f"corner-case-pose-{pose_max}"
                
                people_info['crop_strategy'] = strategy
                people_info['crop_count'] = crop_count
                people_info['positions'] = positions
                
                print(f"   🎬 Processing with {crop_count}-crop (pose-based override)")
                output_files = process_video_with_dynamic_crops(
                    video_path,
                    OUTPUT_FOLDER,
                    yolo,
                    crop_count,
                    positions_override=positions,
                    people_info=people_info,
                )
                
                all_handled_videos.append(video_path)
                continue
            else:
                print(f"   ℹ️ Pose only detected {pose_max} skeletons, not enough for override")
                    # ===== END CORNER CASE OVERRIDE =====

        # STEP 2: Determine crop strategy
        crop_count = 0
        positions = []
        strategy = ""

        if people_count >= MIN_PEOPLE_REQUIRED:
            if people_count >= 4:
                print(f"   👥👥 4+ people detected - analyzing distribution...")

                # Get zone analysis for distribution
                zone_scores, zone_people, zone_activity, zone_positions = analyze_region_activity(
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
                crop_count, positions, strategy, action_hotspots = determine_smart_crop_strategy_v2(
                    video_path, yolo, pose_model, 
                    sample_frames=20, 
                    people_count=people_count,
                    bbox_counts=people_info.get('bbox_counts', []),
                    pose_counts=people_info.get('pose_counts', [])
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
                people_info=people_info,
            )
        else:
            reason = "strategy" if crop_count == 0 else "people count"
            print(f"   📋 Copying {filename} as-is (reason: {reason}, strategy: {strategy})")
            
            # Copy the video
            copy_video_to_output(video_path, OUTPUT_FOLDER)
            
            # ==== CREATE DEBUG VIDEO FOR SKIPPED VIDEO (if debug mode is enabled) ====
            if DEBUG_MODE and DEBUG_CREATE_VIDEOS:
                print(f"   🎥 Creating debug video for skipped video...")
                
                # Create a dummy detector with the attributes that create_enhanced_debug_frame expects
                class DummyDetector:
                    def __init__(self):
                        self.missing_counters = [0]  # For the "SKIPPED" status
                        self.motion_histories = [deque(maxlen=10)]
                        self.pose_activities = [0.0]
                
                dummy_detector = DummyDetector()
                
                # Create debug video
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                debug_video_folder = os.path.join(DEBUG_VIDEO_FOLDER, "skipped")
                os.makedirs(debug_video_folder, exist_ok=True)
                
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                ret, first_frame = cap.read()
                if ret:
                    h, w = first_frame.shape[:2]
                    debug_filename = f"{base_name}_skipped_{strategy}_debug.mp4"
                    debug_path = os.path.join(debug_video_folder, debug_filename)
                    
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    if DEBUG_VIDEO_SIDE_BY_SIDE:
                        output_width = w * 2
                        output_height = h
                    else:
                        output_width = w
                        output_height = h
                    
                    debug_writer = cv2.VideoWriter(str(debug_path), fourcc, fps, (output_width, output_height))
                    
                    # Sample every 30 frames to keep file size manageable
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_count = 0
                    sample_interval = 30
                    
                    print(f"      Debug video: {debug_filename}")
                    
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        if frame_count % sample_interval == 0:
                            # Get YOLO detections
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            result = yolo.predict(rgb, conf=PERSON_DETECTION_CONF, classes=[0], verbose=False)
                            
                            yolo_boxes = []
                            for r in result:
                                for b in r.boxes:
                                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                                    yolo_boxes.append((x1, y1, x2, y2))
                            
                            # Create debug info
                            debug_info = {
                                "Status": "SKIPPED",
                                "Reason": f"{reason} ({strategy})",
                                "Detections": len(yolo_boxes),
                                "People Count": people_info.get('final_count', 'N/A')
                            }
                            
                            # Use existing debug frame function with dummy detector
                            debug_frame = create_enhanced_debug_frame(
                                frame=frame,
                                frame_idx=frame_count,
                                yolo_boxes=yolo_boxes,
                                expanded_boxes=[],
                                smoothed_boxes=[],
                                final_boxes=[],
                                action_statuses=["SKIPPED"],
                                positions=["full"],
                                detector=dummy_detector,  # Now passing a valid detector
                                debug_info=debug_info,
                                people_info=people_info
                            )
                            
                            if DEBUG_VIDEO_SIDE_BY_SIDE:
                                combined_frame = create_side_by_side_frame(frame.copy(), debug_frame)
                                debug_writer.write(combined_frame)
                            else:
                                debug_writer.write(debug_frame)
                        
                        frame_count += 1
                    
                    debug_writer.release()
                    print(f"      ✅ Debug video saved")
                
                cap.release()


            
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
