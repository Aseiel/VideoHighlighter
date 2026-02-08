import cv2
import os
import csv
from ultralytics import YOLO
from tqdm import tqdm
from multiprocessing import Process, Manager
import time
import numpy as np

# ---------------- CONFIG ----------------
NUM_WORKERS = 4
FRAME_SKIP = 5
openvino_model_folder = "yolo11n_openvino_model/"  # Default, will be overridden
highlight_objects = []  # Add your objects of interest here, e.g., ["person", "car"]

# Bounding box visualization settings
BBOX_COLORS = {
    'person': (0, 255, 0),      # Green
    'car': (255, 0, 0),          # Blue
    'dog': (0, 165, 255),        # Orange
    'cat': (147, 20, 255),       # Purple
    'default': (0, 255, 255)     # Yellow
}
BBOX_THICKNESS = 2
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# ---------------- Progress Monitor ----------------
def progress_monitor(progress_queue, total_frames, progress_fn):
    """Monitor progress from worker processes and call progress_fn"""
    processed_frames = 0
    
    while True:
        # Get progress updates from queue
        item = progress_queue.get()
        # Check for sentinel value to stop
        if item is None:
            break
            
        processed_frames += item
        progress = min(processed_frames / total_frames, 0.99) # Cap at 99% until complete
        # Call the progress function with current progress and status
        progress_fn(progress, f"Processing frames: {processed_frames}/{total_frames}")

# ---------------- Utilities ----------------
def detect_objects_in_frame(frame, model, objects_of_interest, draw_boxes=False):
    """
    Detect objects in frame and optionally draw bounding boxes
    
    Args:
        frame: Input frame
        model: YOLO model
        objects_of_interest: List of object classes to detect
        draw_boxes: If True, draw bounding boxes on the frame
    
    Returns:
        tuple: (list of detected object names, annotated frame if draw_boxes=True else None)
    """
    objs = []
    annotated_frame = frame.copy() if draw_boxes else None
    
    try:
        results = model(frame, verbose=False, imgsz=640)
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0])
                    
                    if conf > 0.5 and cls_name in objects_of_interest:
                        objs.append(cls_name)
                        
                        if draw_boxes:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            
                            # Choose color for this object class
                            color = BBOX_COLORS.get(cls_name, BBOX_COLORS['default'])
                            
                            # Draw rectangle
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, BBOX_THICKNESS)
                            
                            # Prepare label text
                            label = f"{cls_name} {conf:.2f}"
                            
                            # Get text size for background
                            (text_width, text_height), baseline = cv2.getTextSize(
                                label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
                            )
                            
                            # Draw background rectangle for text
                            cv2.rectangle(
                                annotated_frame,
                                (x1, y1 - text_height - baseline - 5),
                                (x1 + text_width, y1),
                                color,
                                -1  # Filled
                            )
                            
                            # Draw text
                            cv2.putText(
                                annotated_frame,
                                label,
                                (x1, y1 - baseline - 2),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                FONT_SCALE,
                                (255, 255, 255),  # White text
                                FONT_THICKNESS
                            )
    except Exception as e:
        print(f"⚠️ Error in detection: {e}")
    
    return objs, annotated_frame

def get_video_segments(video_path, num_segments):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    frames_per_segment = total_frames // num_segments
    segments = []
    for i in range(num_segments):
        start = i * frames_per_segment
        end = (i + 1) * frames_per_segment if i < num_segments - 1 else total_frames
        segments.append((start, end))
    return segments, fps, total_frames

# ---------------- Worker ----------------
def worker_process(video_path, start_frame, end_frame, objects_of_interest, return_dict, worker_id, fps, 
                  model_path, openvino_folder=None, progress_queue=None, draw_boxes=False, annotated_output_path=None):
    """
    Worker process for object detection
    
    Args:
        model_path: Path to YOLO model (.pt file)
        openvino_folder: Path to OpenVINO model folder (optional, will use if exists)
        draw_boxes: If True, create annotated video output
        annotated_output_path: Path for annotated video (worker will append _workerN.mp4)
    """
    # Try to load OpenVINO model first, fall back to PT model
    if openvino_folder and os.path.exists(openvino_folder):
        try:
            model = YOLO(openvino_folder, task="detect")
            print(f"Worker {worker_id}: Loaded OpenVINO model from {openvino_folder}")
        except Exception as e:
            print(f"Worker {worker_id}: Failed to load OpenVINO model, falling back to PT: {e}")
            model = YOLO(model_path)
    else:
        model = YOLO(model_path)
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Setup video writer if drawing boxes
    video_writer = None
    if draw_boxes and annotated_output_path:
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create worker-specific output path
        base, ext = os.path.splitext(annotated_output_path)
        worker_output = f"{base}_worker{worker_id}{ext}"
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(worker_output, fourcc, fps, (frame_width, frame_height))
        return_dict[f'worker_{worker_id}_video'] = worker_output

    sec_objects = {}
    frame_idx = start_frame
    processed_frames = 0

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
            
        should_detect = (frame_idx % FRAME_SKIP == 0)
        
        if should_detect:
            # Detect objects (and optionally get annotated frame)
            objs, annotated_frame = detect_objects_in_frame(frame, model, objects_of_interest, draw_boxes)
            
            if objs:
                sec = int(frame_idx / fps)
                sec_objects.setdefault(sec, []).extend(objs)
            
            # Write annotated frame if enabled
            if draw_boxes and video_writer and annotated_frame is not None:
                video_writer.write(annotated_frame)
        else:
            # For frames we skip detection, still write original frame if creating annotated video
            if draw_boxes and video_writer:
                video_writer.write(frame)
        
        frame_idx += 1
        processed_frames += 1

        # Report progress if progress_queue is provided
        if progress_queue is not None:
            progress_queue.put(1) # Signal that one frame was processed

    cap.release()
    if video_writer:
        video_writer.release()
    
    return_dict[worker_id] = sec_objects

# ---------------- Main function for module ----------------
def run_object_detection(video_path, highlight_objects, frame_skip=5, csv_file="objects_log.csv", 
                        progress_fn=None, draw_boxes=False, annotated_output=None,
                        yolo_model_size="n", yolo_pt_path=None, openvino_model_folder=None):
    """
    Run object detection on video
    
    Args:
        video_path: Path to input video
        highlight_objects: List of object classes to detect
        frame_skip: Process every Nth frame
        csv_file: Output CSV file path
        progress_fn: Progress callback function
        draw_boxes: If True, create annotated video with bounding boxes
        annotated_output: Path for annotated video output (only used if draw_boxes=True)
        yolo_model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
        yolo_pt_path: Custom path to YOLO .pt file (optional, overrides default)
        openvino_model_folder: Custom path to OpenVINO model folder (optional, overrides default)
    
    Returns:
        dict: Dictionary of {second: [objects]} detections
    """
    if not os.path.exists(video_path):
        error_msg = f"⚠️ Video not found: {video_path}"
        print(error_msg)
        if progress_fn:
            progress_fn(1.0, error_msg)
        return {}

    if not highlight_objects:
        error_msg = "⚠️ No objects specified in highlight_objects list!"
        print(error_msg)
        if progress_fn:
            progress_fn(1.0, error_msg)
        return {}

    # Update global FRAME_SKIP with the parameter value
    global FRAME_SKIP
    FRAME_SKIP = frame_skip

    # Determine model paths based on parameters
    if yolo_pt_path and os.path.exists(yolo_pt_path):
        # Use custom PT path if provided and exists
        model_path = yolo_pt_path
        print(f"🎯 Using custom YOLO model: {model_path}")
    else:
        # Use default based on model size
        model_path = f"yolo11{yolo_model_size}.pt"
        print(f"🎯 Using YOLO model: {model_path} (size: {yolo_model_size})")
    
    # Determine OpenVINO folder
    if openvino_model_folder and os.path.exists(openvino_model_folder):
        openvino_folder = openvino_model_folder
        print(f"🎯 Using OpenVINO model folder: {openvino_folder}")
    else:
        # Use default OpenVINO folder name based on model size
        openvino_folder = f"yolo11{yolo_model_size}_openvino_model/"
        print(f"🎯 Using default OpenVINO folder: {openvino_folder}")

    segments, fps, total_frames = get_video_segments(video_path, NUM_WORKERS)
    manager = Manager()
    return_dict = manager.dict()
    
    progress_queue = manager.Queue() if progress_fn else None
    processes = []

    print(f"🎬 Processing video with {NUM_WORKERS} workers, FPS: {fps:.2f}")
    print(f"🔍 Looking for: {highlight_objects}")
    if draw_boxes:
        print(f"🎨 Bounding box visualization enabled")

    # Start progress monitoring in a separate process if progress_fn is provided
    progress_process = None
    if progress_fn and progress_queue:
        progress_process = Process(
            target=progress_monitor,
            args=(progress_queue, total_frames, progress_fn)
        )
        progress_process.start()
        progress_fn(0.0, "Starting object detection workers...")

    # Determine annotated output path for workers
    worker_annotated_path = None
    if draw_boxes and annotated_output:
        worker_annotated_path = annotated_output

    # Start worker processes
    for i, seg in enumerate(segments):
        p = Process(
            target=worker_process,
            args=(video_path, seg[0], seg[1], highlight_objects, return_dict, i, fps, 
                  model_path, openvino_folder, progress_queue, draw_boxes, worker_annotated_path)
        )
        p.start()
        processes.append(p)

    # Wait for all worker processes to complete
    for p in processes:
        p.join()

    # Stop progress monitoring
    if progress_queue:
        progress_queue.put(None)
    if progress_process:
        progress_process.join()

    if progress_fn:
        progress_fn(0.95, "Merging detection results...")

    # Merge results from all workers
    all_frame_objects = []
    final_objects = {}
    worker_videos = []
    
    for key, value in return_dict.items():
        if isinstance(key, str) and key.startswith('worker_') and key.endswith('_video'):
            # This is a worker video path
            worker_videos.append(value)
        else:
            # This is detection data
            for sec, objs in value.items():
                final_objects.setdefault(sec, []).extend(objs)
                timestamp_str = f"{sec//60:02d}:{sec%60:02d}"
                for obj_name in objs:
                    all_frame_objects.append([timestamp_str, None, obj_name, None, sec])

    # Merge worker videos if bounding boxes were drawn
    if draw_boxes and worker_videos and annotated_output:
        print(f"🎬 Merging {len(worker_videos)} annotated video segments...")
        merge_worker_videos(worker_videos, annotated_output, fps)

    # Write CSV
    if all_frame_objects:
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_mmss", "frame_id", "label", "confidence", "timestamp_seconds"])
            writer.writerows(all_frame_objects)
        print(f"✅ CSV created: {csv_file} with {len(all_frame_objects)} detections")
    else:
        print("❌ No objects detected - CSV file not created")

    total_detections = sum(len(v) for v in final_objects.values())
    print(f"✅ Total seconds with objects: {len(final_objects)}, total detections: {total_detections}")

    # Final progress update
    if progress_fn:
        progress_fn(1.0, f"Completed: {total_detections} detections found")

    return final_objects

def merge_worker_videos(worker_videos, output_path, fps):
    """Merge multiple worker video segments into single annotated video"""
    if not worker_videos:
        return
    
    try:
        # Get video properties from first worker video
        cap = cv2.VideoCapture(worker_videos[0])
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Create output writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Concatenate all worker videos
        for worker_video in sorted(worker_videos):
            if os.path.exists(worker_video):
                cap = cv2.VideoCapture(worker_video)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                cap.release()
                
                # Clean up worker video
                try:
                    os.remove(worker_video)
                except:
                    pass
        
        out.release()
        print(f"✅ Annotated video saved: {output_path}")
        
    except Exception as e:
        print(f"⚠️ Error merging annotated videos: {e}")

# ---------------- Standalone execution ----------------
if __name__ == "__main__":
    def example_progress_fn(progress, status):
        bar_length = 40
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f'\rProgress: |{bar}| {progress:.1%} - {status}', end='', flush=True)
        if progress >= 1.0:
            print()
    
    test_video = "test_video.mp4"
    test_objects = ["person", "car", "dog"]
    
    if os.path.exists(test_video):
        run_object_detection(
            video_path=test_video, 
            highlight_objects=test_objects, 
            frame_skip=5,
            csv_file="objects_log.csv",
            progress_fn=example_progress_fn,
            draw_boxes=True,
            annotated_output="test_video_objects_annotated.mp4",
            yolo_model_size="n",  # or "s", "m", "l", "x"
            yolo_pt_path=None,    # Optional custom path
            openvino_model_folder=None  # Optional custom OpenVINO folder
        )
    else:
        print(f"Test video {test_video} not found.")