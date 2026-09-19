import cv2
import os
import csv
from tqdm import tqdm
from multiprocessing import Process, Manager
import time
import numpy as np

from modules.system.device_utils import resolve_device

# ---------------- CONFIG ----------------
NUM_WORKERS = 4
FRAME_SKIP = 5
openvino_model_folder = None  # legacy setting, unused: the detector is YOLOX IR
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

# Default confidence threshold
DEFAULT_CONFIDENCE_THRESHOLD = 0.3

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
def seconds_to_mmss(sec):
    """Convert seconds to mm:ss format"""
    minutes, seconds = divmod(int(sec), 60)
    return f"{minutes:02d}:{seconds:02d}"


def detect_objects_in_frame(frame, model, objects_of_interest, draw_boxes=False,
                            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD, device="cpu"):
    """
    Detect objects in frame and optionally draw bounding boxes
    
    Args:
        frame: Input frame
        model: a Detector (modules.vision.detection_backend) — anything with .detect(frame)
        objects_of_interest: List of object classes to detect
        draw_boxes: If True, draw bounding boxes on the frame
        confidence_threshold: Minimum confidence to accept a detection
        device: Device for inference
    
    Returns:
        tuple: (list of detected object names, annotated frame if draw_boxes=True else None, bbox_data)
    """
    objs = []
    bbox_data = []  # collect [x1, y1, x2, y2, confidence] per detection
    annotated_frame = frame.copy() if draw_boxes else None
    
    try:
        for detection in model.detect(frame):
            cls_name = str(detection.class_name)
            conf = float(detection.confidence)
            if conf <= confidence_threshold or cls_name not in objects_of_interest:
                continue
            objs.append(cls_name)
            x1, y1 = int(detection.x1), int(detection.y1)
            x2, y2 = int(detection.x2), int(detection.y2)
            # Store raw pixel coords for cache
            bbox_data.append((x1, y1, x2, y2, conf))

            if draw_boxes:
                color = BBOX_COLORS.get(cls_name, BBOX_COLORS['default'])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, BBOX_THICKNESS)
                label = f"{cls_name} {conf:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
                )
                # Filled background behind the label text
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
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
    
    return objs, annotated_frame, bbox_data

def directml_detector(prefer="large", log=print):
    """The stock YOLOX detector on ONNX Runtime's DirectML provider, or None.

    OpenVINO only accelerates Intel GPUs; on an AMD or NVIDIA card it runs the
    detector on the processor. Where ``device_utils`` reports that ONNX Runtime
    can reach the GPU (``onnx_dml_yolo``), the same YOLOX export runs there
    instead. Any failure returns None and the caller uses OpenVINO.
    """
    try:
        import sys
        from modules.system import ort_directml
        from modules.vision import yolox_models
        from modules.vision.detection_backend import YoloxOnnxRuntimeDetector, load_class_names
        onnx_path = yolox_models.find_onnx(prefer)
        if not onnx_path and not getattr(sys, "frozen", False):
            yolox_models.install(log=log)
            onnx_path = yolox_models.find_onnx(prefer)
        names = load_class_names("yolo_objects_labels.json")
        if not onnx_path or not names:
            return None
        session = ort_directml.session(onnx_path)
        detector = YoloxOnnxRuntimeDetector(session, names)
        log(f"✅ Object detector: YOLOX on {ort_directml.session_backend(session)} "
            f"({os.path.basename(onnx_path)})")
        return detector
    except Exception as e:
        log(f"⚠️ DirectML detector unavailable, using OpenVINO: {e}")
        return None


def _wants_directml(log=print):
    try:
        from modules.system.device_utils import detect_best_device
        return bool(getattr(detect_best_device(log_fn=lambda *_a: None), "onnx_dml_yolo", False))
    except Exception:
        return False


def load_detector(model_path=None, model_size="n", device="AUTO", log=print):
    """The YOLOX object detector for a size, or a user's own model when
    ``model_path`` points at an .onnx/.xml. Returns None when nothing is usable.

    Fetches the stock models on first use from a source checkout, the way the
    previous detector downloaded its weights on demand.
    """
    from modules.vision.detection_backend import build_object_detector
    custom = bool(model_path) and os.path.exists(str(model_path)) and         str(model_path).lower().endswith((".onnx", ".xml"))
    prefer = "small" if str(model_size).lower() in ("n", "nano", "tiny") else "large"
    if not custom and _wants_directml():
        detector = directml_detector(prefer, log=log)
        if detector is not None:
            return detector
    detector, _names = build_object_detector(
        mode="custom" if custom else "coco",
        custom_model_xml=str(model_path) if custom else "",
        device=device,
        default_prefer=prefer,
        log=log, auto_install=True,
    )
    return detector


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
                  model_path, openvino_folder=None, progress_queue=None, draw_boxes=False, 
                  annotated_output_path=None, device="cpu", confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Worker process for object detection
    """
    model = load_detector(model_path, device="AUTO")
    if model is None:
        print(f"Worker {worker_id}: no object detector available")
        return
    print(f"Worker {worker_id}: loaded {type(model).__name__}")

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Setup video writer if drawing boxes
    video_writer = None
    if draw_boxes and annotated_output_path:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        base, ext = os.path.splitext(annotated_output_path)
        worker_output = f"{base}_worker{worker_id}{ext}"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(worker_output, fourcc, fps, (frame_width, frame_height))
        return_dict[f'worker_{worker_id}_video'] = worker_output

    sec_objects = {}
    sec_bboxes = {}
    frame_idx = start_frame
    processed_frames = 0

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
            
        should_detect = (frame_idx % FRAME_SKIP == 0)
        
        if should_detect:
            objs, annotated_frame, bbox_data = detect_objects_in_frame(
                frame, model, objects_of_interest, draw_boxes,
                confidence_threshold=confidence_threshold, device=device
            )
            
            if objs:
                sec = int(frame_idx / fps)
                sec_objects.setdefault(sec, []).extend(objs)
                frame_h, frame_w = frame.shape[:2]
                for i, name in enumerate(objs):
                    if i < len(bbox_data):
                        x1, y1, x2, y2, conf = bbox_data[i]
                        sec_bboxes.setdefault(sec, []).append({
                            'class': name,
                            'bbox': [x1 / frame_w, y1 / frame_h,
                                    (x2 - x1) / frame_w, (y2 - y1) / frame_h],
                            'confidence': conf,
                        })

            if draw_boxes and video_writer and annotated_frame is not None:
                video_writer.write(annotated_frame)
        else:
            if draw_boxes and video_writer:
                video_writer.write(frame)
        
        frame_idx += 1
        processed_frames += 1

        if progress_queue is not None:
            progress_queue.put(1)

    cap.release()
    if video_writer:
        video_writer.release()
    
    return_dict[worker_id] = sec_objects
    return_dict[f'bboxes_{worker_id}'] = dict(sec_bboxes)


# ---------------- Single-threaded detection (used by pipeline) ----------------
def run_object_detection_single(video_path, model, highlight_objects, log_fn=print,
                                progress_fn=None, frame_skip=5, cancel_flag=None,
                                csv_output="object_log.csv", draw_boxes=False,
                                annotated_output=None, device="cpu",
                                confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
                                preview_fn=None):
    """
    Single-threaded object detection with progress tracking and cancellation support.
    Used by the pipeline for integrated processing with a pre-loaded model.

    Args:
        video_path: Path to input video
        model: Pre-loaded Detector instance
        highlight_objects: List of object classes to detect
        log_fn: Logging function
        progress_fn: Progress callback (current, total, task, details)
        frame_skip: Process every Nth frame
        cancel_flag: threading.Event for cancellation
        csv_output: Path for optional CSV log
        draw_boxes: If True, create annotated video
        annotated_output: Path for annotated video output
        device: Device string for inference
        confidence_threshold: Minimum confidence to accept a detection

    Returns:
        tuple: (sec_objects dict, object_bboxes_cache list)
    """
    if model is None:
        log_fn("⚠️ No object detector available, skipping object detection")
        return {}, []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_fn(f"❌ Failed to open video: {video_path}")
        return {}, []

    fps_local = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames_local = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_seconds = int(total_frames_local / fps_local) if fps_local else 0

    if total_seconds <= 0:
        log_fn("⚠️ Could not determine video duration")
        cap.release()
        return {}, []

    # Setup video writer if drawing boxes
    video_writer = None
    if draw_boxes and annotated_output:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(annotated_output, fourcc, fps_local, (frame_width, frame_height))
        log_fn(f"🎨 Creating object detection annotated video: {annotated_output}")

    if progress_fn:
        progress_fn(0, total_seconds, "Object Detection",
                     f"Analyzing {seconds_to_mmss(total_seconds)} of video (conf≥{confidence_threshold})")

    sec_objects = {}
    sec_bboxes = {}
    frame_idx = 0
    current_second = -1
    objects_found = 0
    _last_preview_t = 0.0  # wall-clock throttle for the live preview
    _preview_failed = False

    try:
        while True:
            if cancel_flag and cancel_flag.is_set():
                log_fn("⏹️ Object detection cancelled")
                break

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                sec = int(frame_idx / fps_local)
                if sec > current_second:
                    if cancel_flag and cancel_flag.is_set():
                        log_fn("⏹️ Object detection cancelled")
                        break

                    if progress_fn:
                        progress_fn(sec, total_seconds, "Object Detection",
                                    f"Found {objects_found} objects so far ({seconds_to_mmss(sec)})")
                    current_second = sec

                    try:
                        objs, annotated_frame, bbox_data = detect_objects_in_frame(
                            frame, model, highlight_objects, draw_boxes,
                            confidence_threshold=confidence_threshold, device=device
                        )

                        if objs:
                            sec_objects.setdefault(sec, []).extend(objs)
                            objects_found += len(objs)

                            # Build bbox cache entries
                            frame_h, frame_w = frame.shape[:2]
                            for i, name in enumerate(objs):
                                if i < len(bbox_data):
                                    x1, y1, x2, y2, conf = bbox_data[i]
                                    sec_bboxes.setdefault(sec, []).append({
                                        'class': name,
                                        'bbox': [x1 / frame_w, y1 / frame_h,
                                                 (x2 - x1) / frame_w, (y2 - y1) / frame_h],
                                        'confidence': conf,
                                    })

                        # Write annotated frame if enabled
                        if video_writer and draw_boxes and annotated_frame is not None:
                            video_writer.write(annotated_frame)
                        elif video_writer and draw_boxes:
                            video_writer.write(frame)

                        # ── Live detection preview ──
                        # Send a downscaled frame + normalised boxes to the GUI.
                        # Throttled to ~8 fps wall-clock so it never slows
                        # detection (which is the real bottleneck anyway).
                        if preview_fn is not None:
                            now = time.time()
                            if now - _last_preview_t >= 0.12:
                                _last_preview_t = now
                                try:
                                    fh, fw = frame.shape[:2]
                                    target_w = 480
                                    scale = target_w / fw if fw > target_w else 1.0
                                    small = cv2.resize(
                                        frame, (int(fw * scale), int(fh * scale)),
                                        interpolation=cv2.INTER_AREA
                                    ) if scale != 1.0 else frame.copy()
                                    boxes = []
                                    for i, name in enumerate(objs or []):
                                        if i < len(bbox_data):
                                            x1, y1, x2, y2, conf = bbox_data[i]
                                            boxes.append((name,
                                                          x1 / fw, y1 / fh,
                                                          (x2 - x1) / fw, (y2 - y1) / fh,
                                                          float(conf)))
                                    preview_fn(small, boxes, sec)
                                except Exception as e:
                                    # Once, not per frame — see the same guard
                                    # in action_recognition: a silently dropped
                                    # preview frame is indistinguishable from a
                                    # preview nobody ever fed.
                                    if not _preview_failed:
                                        _preview_failed = True
                                        log_fn(f"⚠️ Live preview frame failed "
                                               f"(reported once per run): {e}")

                    except Exception as e:
                        log_fn(f"⚠️ Error in object detection at frame {frame_idx}: {e}")

            frame_idx += 1

    except Exception as e:
        log_fn(f"❌ Object detection error: {e}")
    finally:
        cap.release()
        if video_writer:
            video_writer.release()
            if draw_boxes:
                log_fn(f"✅ Object detection annotated video saved: {annotated_output}")

        if not (cancel_flag and cancel_flag.is_set()):
            if progress_fn:
                progress_fn(total_seconds, total_seconds, "Object Detection",
                            f"Complete - {objects_found} objects found")
            log_fn(f"✅ Object detection complete: {objects_found} total objects detected")

        # Optional CSV output
        if csv_output:
            try:
                with open(csv_output, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(["timestamp_mmss", "timestamp_seconds", "Objects"])
                    for sec, objs in sorted(sec_objects.items()):
                        writer.writerow([seconds_to_mmss(sec), sec, ";".join(objs)])
                log_fn(f"✅ CSV saved to {csv_output}")
            except Exception as e:
                log_fn(f"⚠️ Failed to save CSV: {e}")

    # Build cache-ready bbox list
    object_bboxes_cache = []
    for sec in sorted(sec_bboxes.keys()):
        entries = sec_bboxes[sec]
        object_bboxes_cache.append({
            'timestamp': float(sec),
            'objects': [e['class'] for e in entries],
            'bboxes': [e['bbox'] for e in entries],
            'confidences': [e['confidence'] for e in entries],
        })

    return sec_objects, object_bboxes_cache


# ---------------- Multi-process detection (standalone) ----------------
def run_object_detection(video_path, highlight_objects, frame_skip=5, csv_file="objects_log.csv", 
                        progress_fn=None, draw_boxes=False, annotated_output=None,
                        yolo_model_size="n", yolo_pt_path=None, openvino_model_folder=None,
                        device="cpu", cancel_flag=None, log_fn=print,
                        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Run object detection on video using multiple worker processes.
    
    Args:
        video_path: Path to input video
        highlight_objects: List of object classes to detect
        frame_skip: Process every Nth frame
        csv_file: Output CSV file path
        progress_fn: Progress callback function
        draw_boxes: If True, create annotated video with bounding boxes
        annotated_output: Path for annotated video output (only used if draw_boxes=True)
        yolo_model_size: detector size ('n' picks the small model, others the large)
        yolo_pt_path: path to a custom .onnx/.xml detector (optional, overrides default)
        openvino_model_folder: unused, kept for callers
        device: Device for inference
        cancel_flag: threading.Event for cancellation support
        log_fn: Logging function
        confidence_threshold: Minimum confidence to accept a detection
    
    Returns:
        tuple: (dict of {second: [objects]}, list of bbox cache entries)
    """
    device = resolve_device(device)
    if not os.path.exists(video_path):
        error_msg = f"⚠️ Video not found: {video_path}"
        log_fn(error_msg)
        if progress_fn:
            progress_fn(1.0, error_msg)
        return {}, []

    if not highlight_objects:
        error_msg = "⚠️ No objects specified in highlight_objects list!"
        log_fn(error_msg)
        if progress_fn:
            progress_fn(1.0, error_msg)
        return {}, []

    # Update global FRAME_SKIP with the parameter value
    global FRAME_SKIP
    FRAME_SKIP = frame_skip

    # A custom .onnx/.xml detector when one is given, else the stock YOLOX
    model_path = yolo_pt_path if yolo_pt_path and os.path.exists(yolo_pt_path) else None
    if model_path:
        log_fn(f"🎯 Using custom model: {model_path}")
    else:
        log_fn(f"🎯 Using YOLOX detector (size: {yolo_model_size})")
    openvino_folder = None
    # Fetch the stock models once here, not in four workers at the same time
    if load_detector(model_path, yolo_model_size, log=log_fn) is None:
        log_fn("⚠️ No object detector available, skipping object detection")
        return {}, []

    log_fn(f"🔍 Confidence threshold: {confidence_threshold}")

    segments, fps, total_frames = get_video_segments(video_path, NUM_WORKERS)
    manager = Manager()
    return_dict = manager.dict()
    
    progress_queue = manager.Queue() if progress_fn else None
    processes = []

    log_fn(f"🎬 Processing video with {NUM_WORKERS} workers, FPS: {fps:.2f}")
    log_fn(f"🔍 Looking for: {highlight_objects}")
    if draw_boxes:
        log_fn(f"🎨 Bounding box visualization enabled")

    # Start progress monitoring
    progress_process = None
    if progress_fn and progress_queue:
        progress_process = Process(
            target=progress_monitor,
            args=(progress_queue, total_frames, progress_fn)
        )
        progress_process.start()
        progress_fn(0.0, "Starting object detection workers...")

    worker_annotated_path = None
    if draw_boxes and annotated_output:
        worker_annotated_path = annotated_output

    # Start worker processes
    for i, seg in enumerate(segments):
        p = Process(
            target=worker_process,
            args=(video_path, seg[0], seg[1], highlight_objects, return_dict, i, fps, 
                  model_path, openvino_folder, progress_queue, draw_boxes, worker_annotated_path,
                  device, confidence_threshold)
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
    all_bboxes = {}

    for key, value in return_dict.items():
        if isinstance(key, str) and key.startswith('worker_') and key.endswith('_video'):
            worker_videos.append(value)
        elif isinstance(key, str) and key.startswith('bboxes_'):
            for sec, entries in value.items():
                all_bboxes.setdefault(sec, []).extend(entries)
        else:
            for sec, objs in value.items():
                final_objects.setdefault(sec, []).extend(objs)
                timestamp_str = f"{sec//60:02d}:{sec%60:02d}"
                for obj_name in objs:
                    all_frame_objects.append([timestamp_str, None, obj_name, None, sec])

    # Build cache-ready bbox list
    object_bboxes_cache = []
    for sec in sorted(all_bboxes.keys()):
        entries = all_bboxes[sec]
        object_bboxes_cache.append({
            'timestamp': float(sec),
            'objects': [e['class'] for e in entries],
            'bboxes': [e['bbox'] for e in entries],
            'confidences': [e['confidence'] for e in entries],
        })

    # Merge worker videos if bounding boxes were drawn
    if draw_boxes and worker_videos and annotated_output:
        log_fn(f"🎬 Merging {len(worker_videos)} annotated video segments...")
        merge_worker_videos(worker_videos, annotated_output, fps)

    # Write CSV
    if all_frame_objects:
        with open(csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_mmss", "frame_id", "label", "confidence", "timestamp_seconds"])
            writer.writerows(all_frame_objects)
        log_fn(f"✅ CSV created: {csv_file} with {len(all_frame_objects)} detections")
    else:
        log_fn("❌ No objects detected - CSV file not created")

    total_detections = sum(len(v) for v in final_objects.values())
    log_fn(f"✅ Total seconds with objects: {len(final_objects)}, total detections: {total_detections}")

    if progress_fn:
        progress_fn(1.0, f"Completed: {total_detections} detections found")

    return final_objects, object_bboxes_cache

def merge_worker_videos(worker_videos, output_path, fps):
    """Merge multiple worker video segments into single annotated video"""
    if not worker_videos:
        return
    
    try:
        cap = cv2.VideoCapture(worker_videos[0])
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        for worker_video in sorted(worker_videos):
            if os.path.exists(worker_video):
                cap = cv2.VideoCapture(worker_video)
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out.write(frame)
                cap.release()
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
            yolo_model_size="n",
        )
    else:
        print(f"Test video {test_video} not found.")