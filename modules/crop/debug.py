"""
debug.py — visualisation only. No policy, nothing reads back into the cropper.

Draws what the other modules decided: raw detections, expanded boxes, smoothed
boxes, final crop windows, and the metrics overlay. Split out first because it
is the one group with no inbound edges — nothing here is consulted by tracking,
counting or zone analysis, so it can be silenced or replaced wholesale without
touching a decision.

COLOUR CONVENTION (shared by the stills and the debug videos):
    RED     raw detections            YELLOW  expanded boxes
    GREEN   smoothed boxes            BLUE    final crop, good tracking
    MAGENTA final crop, fallback mode
"""
import cv2
import numpy as np
from pathlib import Path

from modules.crop.config import DEBUG_SHOW_METRICS, DEBUG_VIDEO_SIDE_BY_SIDE


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
    panel_height = 240  # Increased height for pipeline detection info
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

        # Show per-frame pipeline info if available
        pipeline = people_info.get('pipeline_info', None)
        if pipeline:
            pipe_text = f"Pipeline: Raw={pipeline.get('raw', '?')} Merged={pipeline.get('merged', '?')} Split={pipeline.get('split_corrected', '?')}"
            cv2.putText(frame, pipe_text, (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
            y_pos += 25
            split_reason = pipeline.get('split_reason', '')
            if split_reason:
                cv2.putText(frame, f"Split fix: {split_reason}", (20, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
                y_pos += 25

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
