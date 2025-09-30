import cv2
import os
import csv
from ultralytics import YOLO
from tqdm import tqdm
from multiprocessing import Process, Manager
import time

# ---------------- CONFIG ----------------
NUM_WORKERS = 4
FRAME_SKIP = 5
openvino_model_folder = "yolo11n_openvino_model/"
highlight_objects = []  # Add your objects of interest here, e.g., ["person", "car"]
model_path_default = "yolo11n.pt"

# ---------------- Progress Monitor (MUST be at module level) ----------------
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
        progress = min(processed_frames / total_frames, 0.99)  # Cap at 99% until complete
        
        # Call the progress function with current progress and status
        progress_fn(progress, f"Processing frames: {processed_frames}/{total_frames}")

# ---------------- Utilities ----------------
def detect_objects_in_frame(frame, model, objects_of_interest):
    objs = []
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
    except Exception as e:
        print(f"⚠️ Error: {e}")
    return objs

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
def worker_process(video_path, start_frame, end_frame, objects_of_interest, return_dict, worker_id, fps, progress_queue=None):
    model = YOLO(model_path_default)
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    total_frames = end_frame - start_frame

    sec_objects = {}
    frame_idx = start_frame
    processed_frames = 0

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_SKIP == 0:
            objs = detect_objects_in_frame(frame, model, objects_of_interest)
            if objs:
                sec = int(frame_idx / fps)
                sec_objects.setdefault(sec, []).extend(objs)
        frame_idx += 1
        processed_frames += 1
        
        # Report progress if progress_queue is provided
        if progress_queue is not None:
            progress_queue.put(1)  # Signal that one frame was processed

    cap.release()
    return_dict[worker_id] = sec_objects

# ---------------- Main function for module ----------------
def run_object_detection(video_path, highlight_objects, frame_skip=5, csv_file="objects_log.csv", progress_fn=None):
    if not os.path.exists(video_path):
        error_msg = f"⚠️ Video not found: {video_path}"
        print(error_msg)
        if progress_fn:
            progress_fn(1.0, error_msg)
        return {}

    if not highlight_objects:
        error_msg = "⚠️ No objects specified in highlight_objects list!"
        print(error_msg)
        print("💡 Example: highlight_objects = ['person', 'car', 'dog']")
        if progress_fn:
            progress_fn(1.0, error_msg)
        return {}

    # Update global FRAME_SKIP with the parameter value
    global FRAME_SKIP
    FRAME_SKIP = frame_skip

    segments, fps, total_frames = get_video_segments(video_path, NUM_WORKERS)
    manager = Manager()
    return_dict = manager.dict()
    
    # Create progress queue if progress function is provided
    progress_queue = manager.Queue() if progress_fn else None
    processes = []

    print(f"🎬 Processing video with {NUM_WORKERS} workers, FPS: {fps:.2f}")
    print(f"🔍 Looking for: {highlight_objects}")

    # Start progress monitoring in a separate process if progress_fn is provided
    progress_process = None
    if progress_fn and progress_queue:
        progress_process = Process(
            target=progress_monitor,
            args=(progress_queue, total_frames, progress_fn)
        )
        progress_process.start()
        if progress_fn:
            progress_fn(0.0, "Starting object detection workers...")

    # Start worker processes
    for i, seg in enumerate(segments):
        p = Process(
            target=worker_process,
            args=(video_path, seg[0], seg[1], highlight_objects, return_dict, i, fps, progress_queue)
        )
        p.start()
        processes.append(p)

    # Wait for all worker processes to complete
    for p in processes:
        p.join()

    # Signal progress monitoring to stop
    if progress_queue:
        progress_queue.put(None)  # Sentinel value to stop progress monitoring
    
    if progress_process:
        progress_process.join()

    if progress_fn:
        progress_fn(0.95, "Merging detection results...")

    # Merge results from all workers
    all_frame_objects = []
    final_objects = {}
    for sec_objs in return_dict.values():
        for sec, objs in sec_objs.items():
            final_objects.setdefault(sec, []).extend(objs)
            timestamp_str = f"{sec//60:02d}:{sec%60:02d}"
            for obj_name in objs:
                all_frame_objects.append([timestamp_str, None, obj_name, None, sec])

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

# ---------------- Standalone execution ----------------
if __name__ == "__main__":
    # Example progress function for testing (MUST be at module level)
    def example_progress_fn(progress, status):
        """Example progress function for testing"""
        # Create a simple text-based progress bar
        bar_length = 40
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        print(f'\rProgress: |{bar}| {progress:.1%} - {status}', end='', flush=True)
        if progress >= 1.0:
            print()  # New line when complete
    
    # Test with sample video and objects
    test_video = "test_video.mp4"
    test_objects = ["person", "car", "dog"]
    
    if os.path.exists(test_video):
        run_object_detection(
            test_video, 
            test_objects, 
            csv_file="objects_log.csv",
            progress_fn=example_progress_fn
        )
    else:
        print(f"Test video {test_video} not found. Using dummy detection.")
        # Create a dummy progress demonstration
        def dummy_progress(progress, status):
            print(f"Progress: {progress:.1%} - {status}")
            time.sleep(0.1)
        
        # Simulate progress without actual video processing
        for i in range(10):
            dummy_progress(i/10, f"Simulating progress {i+1}/10")
        dummy_progress(1.0, "Simulation complete")