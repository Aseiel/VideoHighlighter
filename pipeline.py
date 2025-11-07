# pipeline.py
import os
import time
import subprocess
from collections import defaultdict
import numpy as np
import torch
import warnings
import yaml
import csv
import cv2
from tqdm import tqdm

# modules
from motion_scene_detect_optimized import detect_scenes_motion_optimized
from action_recognition import run_action_detection, load_models
from audio_peaks import extract_audio_peaks
from video_cutter import cut_video
from ultralytics import YOLO

# Keep warnings about CUDA quiet
warnings.filterwarnings("ignore", message="torch.cuda")

class ProgressTracker:
    """Simple progress tracker that works with or without GUI callback"""
    def __init__(self, progress_fn=None, log_fn=print):
        self.progress_fn = progress_fn
        self.log_fn = log_fn
        
    def update_progress(self, current, total, task_name, details=""):
        """Update progress if callback is available"""
        if self.progress_fn:
            try:
                self.progress_fn(current, total, task_name, details)
            except:
                pass  # Ignore callback errors

# Transcript modules (optional)
try:
    from transcript import get_transcript_segments, search_transcript_for_keywords
    from transcript_srt import create_highlight_subtitles, create_enhanced_transcript, create_srt_file, translate_segments
    TRANSCRIPT_AVAILABLE = True
except ImportError:
    TRANSCRIPT_AVAILABLE = False
    print("⚠ Warning: Transcript modules not available. Transcript features disabled.")

def seconds_to_mmss(sec):
    """Convert seconds to mm:ss format"""
    minutes, seconds = divmod(int(sec), 60)
    return f"{minutes:02d}:{seconds:02d}"

def check_cancellation(cancel_flag, log_fn, step_name="operation"):
    """Check if cancellation was requested and raise exception if so"""
    if cancel_flag and cancel_flag.is_set():
        log_fn(f"⏹️ Cancelled during {step_name}")
        raise RuntimeError(f"Operation cancelled during {step_name}")


def check_xpu_availability(log_fn=print):
    """Check if Intel XPU is available and return (available, device_str)."""
    try:
        import intel_extension_for_pytorch as ipex
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device_count = torch.xpu.device_count()
            log_fn(f"✅ Intel XPU available: {device_count} device(s)")
            for i in range(device_count):
                try:
                    device_name = torch.xpu.get_device_name(i)
                    log_fn(f"   Device {i}: {device_name}")
                except Exception:
                    pass
            return True, "xpu:0"
        else:
            log_fn("❌ Intel XPU not available")
            return False, "cpu"
    except ImportError:
        log_fn("❌ Intel Extension for PyTorch not installed")
        log_fn("   Install with: pip install intel_extension_for_pytorch")
        return False, "cpu"
    except Exception as e:
        log_fn(f"❌ XPU initialization error: {e}")
        return False, "cpu"

def detect_objects_with_progress(video_path, model, highlight_objects, log_fn=print,
                                 progress_fn=None, frame_skip=5, cancel_flag=None,
                                 csv_output="object_log.csv", draw_boxes=False,
                                 annotated_output=None):
    """Object detection with progress tracking, cancellation support, and optional CSV export in mm:ss format"""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log_fn(f"❌ Failed to open video: {video_path}")
        return {}  # or raise an exception
    
    # Get fps for video writer
    fps_local = cap.get(cv2.CAP_PROP_FPS)

    # Setup video writer if drawing boxes
    video_writer = None
    if draw_boxes and annotated_output:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(annotated_output, fourcc, fps_local, (frame_width, frame_height))
        log_fn(f"🎨 Creating object detection annotated video: {annotated_output}")

    # Bounding box visualization settings
    BBOX_COLORS = {
        'person': (0, 255, 0),
        'car': (255, 0, 0),
        'dog': (0, 165, 255),
        'cat': (147, 20, 255),
        'default': (0, 255, 255)
    }
    BBOX_THICKNESS = 2
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2

    if model is None:
        log_fn("⚠️ No YOLO model available, skipping object detection")
        return {}

    def seconds_to_mmss(sec):
        minutes, seconds = divmod(int(sec), 60)
        return f"{minutes:02d}:{seconds:02d}"

    cap = cv2.VideoCapture(video_path)
    fps_local = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames_local = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    total_seconds = int(total_frames_local / fps_local) if fps_local else 0

    if total_seconds <= 0:
        log_fn("⚠️ Could not determine video duration")
        cap.release()
        return {}

    # Create progress tracker
    progress = ProgressTracker(progress_fn, log_fn)
    progress.update_progress(0, total_seconds, "Object Detection",
                             f"Analyzing {seconds_to_mmss(total_seconds)} of video")

    sec_objects = {}
    frame_idx = 0
    current_second = -1
    objects_found = 0

    try:
        while True:
            if cancel_flag and cancel_flag.is_set():
                log_fn("⏹️ Object detection cancelled")
                break

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip == 0:
                sec = int(frame_idx / fps_local)  # keep as integer for calculations
                if sec > current_second:
                    if cancel_flag and cancel_flag.is_set():
                        log_fn("⏹️ Object detection cancelled")
                        break

                    progress.update_progress(sec, total_seconds, "Object Detection",
                                             f"Found {objects_found} objects so far ({seconds_to_mmss(sec)})")
                    current_second = sec

                    try:
                        results = model(frame, verbose=False, imgsz=640)
                        objs = []
                        annotated_frame = frame.copy() if draw_boxes else None
                        
                        for result in results:
                            if result.boxes is not None:
                                for box in result.boxes:
                                    cls_id = int(box.cls[0])
                                    cls_name = model.names[cls_id]
                                    conf = float(box.conf[0])
                                    if conf > 0.3 and cls_name in highlight_objects:
                                        objs.append(cls_name)
                                        
                                        # Draw bounding box if enabled
                                        if draw_boxes and annotated_frame is not None:
                                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                                            color = BBOX_COLORS.get(cls_name, BBOX_COLORS['default'])
                                            
                                            # Draw rectangle
                                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, BBOX_THICKNESS)
                                            
                                            # Prepare label
                                            label = f"{cls_name} {conf:.2f}"
                                            (text_width, text_height), baseline = cv2.getTextSize(
                                                label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS
                                            )
                                            
                                            # Draw background for text
                                            cv2.rectangle(
                                                annotated_frame,
                                                (x1, y1 - text_height - baseline - 5),
                                                (x1 + text_width, y1),
                                                color,
                                                -1
                                            )
                                            
                                            # Draw text
                                            cv2.putText(
                                                annotated_frame,
                                                label,
                                                (x1, y1 - baseline - 2),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                FONT_SCALE,
                                                (255, 255, 255),
                                                FONT_THICKNESS
                                            )

                        if objs:
                            sec_objects.setdefault(sec, []).extend(objs)
                            objects_found += len(objs)
                        
                        # Write annotated frame if enabled
                        if video_writer and draw_boxes and annotated_frame is not None:
                            video_writer.write(annotated_frame)

                        if objs:
                            sec_objects.setdefault(sec, []).extend(objs)  # keep key as int
                            objects_found += len(objs)

                        else:
                        # For frames we skip detection, still write original frame if creating annotated video
                            if video_writer and draw_boxes:
                                video_writer.write(frame)

                    except Exception as e:
                        log_fn(f"⚠️ Error in object detection at frame: {e}")

            frame_idx += 1

    except Exception as e:
        log_fn(f"❌ Object detection error: {e}")
    finally:
        cap.release()
        if video_writer:  # NEW
            video_writer.release()  # NEW
            if draw_boxes:  # NEW
                log_fn(f"✅ Object detection annotated video saved: {annotated_output}")  # NEW


        if not (cancel_flag and cancel_flag.is_set()):
            progress.update_progress(total_seconds, total_seconds, "Object Detection",
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

    return sec_objects


def run_highlighter(video_path, sample_rate=5, gui_config: dict = None, 
                    log_fn=print, progress_fn=None, cancel_flag=None):
    """
    Process single video or multiple videos for highlight generation.
    
    Args:
        video_path: str for single video OR list of str for multiple videos
        sample_rate: Frame sampling rate
        gui_config: Configuration dictionary
        log_fn: Logging function
        progress_fn: Progress callback function
        cancel_flag: Threading event for cancellation
    
    Returns:
        str (single output path) or list of tuples [(input_path, output_path), ...]
    """
    
    # ========== MULTI-FILE BATCH PROCESSING ==========
    if isinstance(video_path, (list, tuple)):
        results = []
        total_videos = len(video_path)
        progress = ProgressTracker(progress_fn, log_fn)
        
        for idx, single_video_path in enumerate(video_path, 1):
            log_fn(f"\n{'='*60}")
            log_fn(f"📹 Processing video {idx}/{total_videos}: {os.path.basename(single_video_path)}")
            log_fn(f"{'='*60}\n")
            
            # Check cancellation
            if cancel_flag and cancel_flag.is_set():
                log_fn("⏹️ Batch processing cancelled")
                break
            
            # Update batch progress
            batch_progress = int((idx - 1) / total_videos * 100)
            progress.update_progress(batch_progress, 100, "Batch Processing", 
                                   f"Video {idx}/{total_videos}")
            
            # Auto-generate output filename
            video_gui_config = gui_config.copy() if gui_config else {}
            if "output_file" not in video_gui_config or video_gui_config["output_file"] == "highlight.mp4":
                base_name = os.path.splitext(single_video_path)[0]
                video_gui_config["output_file"] = f"{base_name}_highlight.mp4"
            else:
                base_name = os.path.splitext(video_gui_config["output_file"])[0]
                ext = os.path.splitext(video_gui_config["output_file"])[1]
                video_gui_config["output_file"] = f"{base_name}_{idx}{ext}"
            
            # Recursive call for single video
            try:
                result = run_highlighter(
                    video_path=single_video_path,
                    sample_rate=sample_rate,
                    gui_config=video_gui_config,
                    log_fn=log_fn,
                    progress_fn=progress_fn,
                    cancel_flag=cancel_flag
                )
                results.append((single_video_path, result))
                
                if result:
                    log_fn(f"✅ Completed {idx}/{total_videos}: {os.path.basename(result)}")
                else:
                    log_fn(f"⚠️ Failed {idx}/{total_videos}: {os.path.basename(single_video_path)}")
            except Exception as e:
                log_fn(f"❌ Error processing {single_video_path}: {e}")
                results.append((single_video_path, None))
        
        # Summary
        log_fn(f"\n{'='*60}")
        log_fn(f"📊 BATCH PROCESSING SUMMARY")
        log_fn(f"{'='*60}")
        successful = sum(1 for _, r in results if r is not None)
        log_fn(f"Total: {total_videos} | ✅ Success: {successful} | ❌ Failed: {total_videos - successful}")
        
        for input_path, output_path in results:
            status = "✅" if output_path else "❌"
            log_fn(f"  {status} {os.path.basename(input_path)}")
        
        progress.update_progress(100, 100, "Batch Processing", 
                               f"Complete: {successful}/{total_videos}")
        return results
    
    # ========== SINGLE FILE PROCESSING ==========
    gui_config = gui_config or {}
    log = log_fn
    
    # Create progress tracker
    progress = ProgressTracker(progress_fn, log_fn)

    try:
        # --- Load config defaults (from config.yaml) ---
        config = {}
        cfg_path = "config.yaml"
        if os.path.exists(cfg_path):
            try:
                check_cancellation(cancel_flag, log, "config loading")
                with open(cfg_path, "r") as f:
                    config = yaml.safe_load(f) or {}
                log("✅ Loaded config.yaml")
            except RuntimeError:
                return None
            except Exception as e:
                log(f"⚠ Failed to read config.yaml: {e}")
        else:
            log("⚠ config.yaml not found — using defaults and GUI overrides")

        # Check cancellation after config load
        check_cancellation(cancel_flag, log, "initialization")


        # Merge CLI/gui-style values with defaults
        OUTPUT_FILE = gui_config.get("output_file") or config.get("video", {}).get("output", "highlight.mp4")
        MAX_DURATION = gui_config.get("max_duration") or config.get("highlights", {}).get("max_duration", 420)
        EXACT_DURATION = gui_config.get("exact_duration") or config.get("highlights", {}).get("exact_duration", None)
        CLIP_TIME = gui_config.get("clip_time") or config.get("highlights", {}).get("clip_time", 10)
        KEEP_TEMP = gui_config.get("keep_temp", config.get("highlights", {}).get("keep_temp", False))

        # Transcript settings
        USE_TRANSCRIPT = gui_config.get("use_transcript", False) and TRANSCRIPT_AVAILABLE
        TRANSCRIPT_MODEL = gui_config.get("transcript_model", "base")
        SEARCH_KEYWORDS = gui_config.get("search_keywords", [])
        CREATE_SUBTITLES = gui_config.get("create_subtitles", False)
        TRANSCRIPT_ONLY = gui_config.get("transcript_only", False)
        TRANSCRIPT_POINTS = int(gui_config.get("transcript_points", 0))
        SOURCE_LANG = gui_config.get("source_lang", "en")
        TARGET_LANG = gui_config.get("target_lang", None)

        target_duration = EXACT_DURATION if EXACT_DURATION else MAX_DURATION
        duration_mode = "EXACT" if EXACT_DURATION else "MAX"
        log(f"🎯 Mode: {duration_mode} duration of {target_duration} seconds ({target_duration/60:.1f} minutes)")

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Input video not found at path: {video_path}")

        # Initial progress
        progress.update_progress(0, 100, "Pipeline", "Initializing...")
        check_cancellation(cancel_flag, log, "setup")

        # Device check
        motion_device = "cpu"
        xpu_available, yolo_device = check_xpu_availability(log_fn=log)
        if not xpu_available:
            log("⚠️ Falling back to CPU for YOLO inference")
            yolo_device = "cpu"
        log(f"🎯 YOLO device: {yolo_device}")

        # Get video info
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        video_duration = total_frames / fps if fps else 0
        cap.release()
        log(f"🎬 Video duration: {video_duration:.2f}s, FPS: {fps}, total frames: {total_frames}")

        check_cancellation(cancel_flag, log, "video info extraction")

        # --- Transcript processing ---
        transcript_segments = []
        keyword_matches = []
        USE_TRANSCRIPT = gui_config.get("use_transcript", False) and TRANSCRIPT_AVAILABLE
        SEARCH_KEYWORDS = gui_config.get("search_keywords", [])

        if USE_TRANSCRIPT:
            progress.update_progress(5, 100, "Pipeline", "Processing transcript...")
            log("🔹 Step 0.5: Processing transcript...")
            try:
                check_cancellation(cancel_flag, log, "transcript processing")
                # Note: You'll need to modify your transcript functions to accept cancel_flag
                transcript_segments = get_transcript_segments(video_path, model_name=gui_config.get("transcript_model", "medium"), progress_fn=progress_fn, log_fn=log)
                
                check_cancellation(cancel_flag, log, "transcript processing")
                
                # Save transcript
                base_name = os.path.splitext(video_path)[0]
                transcript_file = f"{base_name}_transcript.txt"
                transcript_text = create_enhanced_transcript(transcript_segments)
                with open(transcript_file, "w", encoding="utf-8") as f:
                    f.write(transcript_text)
                log(f"✅ Transcript saved: {transcript_file}")
            except RuntimeError:
                return None
            except Exception as e:
                log(f"⚠ Transcript processing failed: {e}")
                transcript_segments = []

            if SEARCH_KEYWORDS and transcript_segments:
                check_cancellation(cancel_flag, log, "keyword search")
                log(f"🔹 Searching transcript for keywords: {SEARCH_KEYWORDS}")
                keyword_matches = search_transcript_for_keywords(transcript_segments, SEARCH_KEYWORDS, context_seconds=CLIP_TIME//2)
                log(f"✅ Found {len(keyword_matches)} keyword matches")
        else:
            log("ℹ Transcript processing disabled")

        check_cancellation(cancel_flag, log, "transcript phase")

        start_time = time.time()

        # --- 1+2 Detect scenes + motion + peaks with live progress ---
        progress.update_progress(10, 100, "Pipeline", "Detecting motion and scenes...")

        # Check if we should skip motion detection based on GUI config
        scene_points = gui_config.get("scene_points", 0)
        motion_event_points = gui_config.get("motion_event_points", 0) 
        motion_peak_points = gui_config.get("motion_peak_points", 0)

        # Skip motion detection if all motion-related points are 0
        if scene_points == 0 and motion_event_points == 0 and motion_peak_points == 0:
            log("ℹ️ Skipping motion detection (all scene/motion points set to 0)")
            scenes, motion_events, motion_peaks = [], [], []
            progress.update_progress(25, 100, "Pipeline", "Motion detection skipped - no motion scoring enabled")
        else:
            log("🔹 Step 1+2: Detecting scenes, motion events, and motion peaks (this may take time)...")

            scenes, motion_events, motion_peaks = [], [], []

            try:
                check_cancellation(cancel_flag, log, "motion detection")
                
                # Call the actual motion detection function with video path
                result = detect_scenes_motion_optimized(
                    video_path,
                    scene_threshold=70.0,
                    motion_threshold=100.0,
                    spike_factor=1.2,
                    freeze_seconds=4,
                    freeze_factor=0.8,
                    device=motion_device,
                    cancel_flag=cancel_flag
                )
                
                # Unpack the results
                if result and len(result) == 3:
                    scenes, motion_events, motion_peaks = result
                    log(f"✅ Motion detection results: {len(scenes)} scenes, {len(motion_events)} motion events, {len(motion_peaks)} motion peaks")
                else:
                    log(f"⚠️ Unexpected motion detection result format: {result}")
                    
            except RuntimeError:
                return None
            except Exception as e:
                log(f"❌ Motion detection failed: {e}")
                import traceback
                log(f"Full error: {traceback.format_exc()}")

            # Add progress update after motion detection
            progress.update_progress(25, 100, "Pipeline", f"Motion detection complete: {len(scenes)} scenes, {len(motion_events)} events, {len(motion_peaks)} peaks")

        check_cancellation(cancel_flag, log, "motion detection completion")


        # Add progress update after motion detection
        progress.update_progress(25, 100, "Pipeline", f"Motion detection complete: {len(scenes)} scenes, {len(motion_events)} events, {len(motion_peaks)} peaks")
        check_cancellation(cancel_flag, log, "motion detection completion")

        # 3 Audio peaks
        progress.update_progress(30, 100, "Pipeline", "Analyzing audio...")
        log("🔹 Step 3: Detecting audio peaks...")
        try:
            check_cancellation(cancel_flag, log, "audio peak detection")
            # Note: You may need to modify extract_audio_peaks to accept cancel_flag
            audio_peaks = extract_audio_peaks(video_path, cancel_flag=cancel_flag)
            log(f"✅ Audio peak detection done: {len(audio_peaks)} peaks")
        except RuntimeError:
            return None

        # 4 Object detection setup
        progress.update_progress(40, 100, "Pipeline", "Setting up object detection...")
        check_cancellation(cancel_flag, log, "object detection setup")
        
        # Get list of objects to highlight from GUI or config
        highlight_objects = gui_config.get("highlight_objects", config.get("highlight_objects", []))
        openvino_model_folder = gui_config.get("openvino_model_folder", "yolo11n_openvino_model/")

        # Check OpenVINO devices (best-effort)
        try:
            from openvino.runtime import Core
            ie = Core()
            log(f"🔹 OpenVINO available devices: {ie.available_devices}")
        except Exception as e:
            log(f"⚠️ OpenVINO device check failed: {e}")

        # Export model to OpenVINO if missing
        if not os.path.exists(openvino_model_folder):
            try:
                check_cancellation(cancel_flag, log, "YOLO model export")
                log("⚠️ OpenVINO folder not found. Exporting YOLO model (requires yolo11n.pt)...")
                yolo_model_export = YOLO(gui_config.get("yolo_pt_path", "yolo11n.pt"))
                export_result = yolo_model_export.export(format="openvino")
                log(f"✅ Model exported to: {export_result}")
            except RuntimeError:
                return None
            except Exception as e:
                log(f"❌ YOLO export to OpenVINO failed: {e}")

        # Load YOLO model
        try:
            check_cancellation(cancel_flag, log, "YOLO model loading")
            yolo_model = YOLO(openvino_model_folder, task="detect")
            log(f"🎯 YOLO OpenVINO model loaded successfully")
        except RuntimeError:
            return None
        except Exception as e:
            log(f"❌ Failed to load YOLO model: {e}")
            yolo_model = None

        # --- Object detection ---
        highlight_objects = gui_config.get("highlight_objects", config.get("highlight_objects", []))

        if not highlight_objects:
            log("ℹ Skipping object detection (no objects to highlight)")
            object_detections = {}
        else:
            frame_skip_for_obj = gui_config.get("object_frame_skip", CLIP_TIME if CLIP_TIME > 0 else 5)
            
            # Get bounding box settings
            draw_object_boxes = gui_config.get("draw_object_boxes", False)
            object_annotated_path = None
            if draw_object_boxes:
                video_basename = os.path.splitext(os.path.basename(video_path))[0]
                temp_folder = os.path.dirname(video_path) or "."
                object_annotated_path = os.path.join(temp_folder, f"{video_basename}_objects_annotated.mp4")
                log(f"🎨 Object bounding boxes enabled, output: {object_annotated_path}")

            # Run detection with cancellation support and bounding boxes
            object_detections = detect_objects_with_progress(
                video_path,
                yolo_model,
                highlight_objects,
                log_fn=log_fn,
                progress_fn=progress_fn,
                frame_skip=frame_skip_for_obj,
                cancel_flag=cancel_flag,
                draw_boxes=draw_object_boxes,
                annotated_output=object_annotated_path
            )


        print("Detections per second:", len(object_detections))

        def group_consecutive_adaptive(actions, max_gap=1.3, jump_threshold=0.45):
            """
            Groups consecutive actions of the same type if:
            - time gap <= max_gap
            - confidence change between frames <= jump_threshold
            """
            if not actions:
                return []

            # Ensure consistent format first
            normalized_actions = []
            for action in actions:
                if len(action) == 4:
                    timestamp, frame_id, score, action_name = action
                    normalized_actions.append((timestamp, frame_id, -1, score, action_name))
                else:
                    normalized_actions.append(action)
            
            actions = sorted(normalized_actions, key=lambda x: x[0])
            
            # grouping logic with consistent 5-element format
            groups = []
            current = [actions[0]]
            
            for i in range(1, len(actions)):
                prev = actions[i-1]
                curr = actions[i]
                
                # Now all actions are 5-element: (timestamp, frame_id, action_id, score, action_name)
                prev_timestamp, _, _, prev_score, prev_action = prev
                curr_timestamp, _, _, curr_score, curr_action = curr
                
                same_action = curr_action == prev_action
                time_gap = curr_timestamp - prev_timestamp
                close_in_time = time_gap <= max_gap
                conf_change = abs(curr_score - prev_score)
                conf_stable = conf_change <= jump_threshold
                
                if same_action and close_in_time and conf_stable:
                    current.append(curr)
                else:
                    groups.append(current)
                    current = [curr]
            
            if current:
                groups.append(current)
            
            # Collapse groups
            result = []
            for g in groups:
                timestamps = [x[0] for x in g]
                start = min(timestamps)
                end = max(timestamps)
                duration = max(0.5, end - start)
                avg_conf = sum(x[3] for x in g) / len(g)  # score is at index 3
                action_name = g[0][4]  # action_name is at index 4
                
                result.append((start, end, duration, avg_conf, action_name))
            
            return result

        selected_sequences = []


        # --- Action recognition with grouping ---
        action_detections = []
        sample_rate = gui_config.get("sample_rate", 5)

        interesting_actions = gui_config.get("interesting_actions", [])
        if interesting_actions:
            try:
                # NEW: Get action label settings
                draw_action_labels = gui_config.get("draw_action_labels", False)
                action_annotated_path = None
                if draw_action_labels:
                    video_basename = os.path.splitext(os.path.basename(video_path))[0]
                    temp_folder = os.path.dirname(video_path) or "."
                    action_annotated_path = os.path.join(temp_folder, f"{video_basename}_actions_annotated.mp4")
                    log(f"🎨 Action labels enabled, output: {action_annotated_path}")
                
                all_action_detections = run_action_detection(
                    video_path,
                    interesting_actions=interesting_actions,
                    progress_callback=progress.update_progress,
                    cancel_flag=cancel_flag,
                    sample_rate=sample_rate,
                    draw_labels=draw_action_labels,  # NEW
                    annotated_output=action_annotated_path  # NEW
                )


                check_cancellation(cancel_flag, log, "action recognition processing")

                if all_action_detections:
                    print("DEBUG: First 10 individual actions as returned:")
                    for i, action in enumerate(all_action_detections[:10]):
                        # Handle both 4 and 5 element formats
                        if len(action) == 4:
                            timestamp, frame_id, score, action_name = action
                            print(f"  {i+1}. {timestamp:.1f}s -> {action_name} (confidence: {score:.3f})")
                        else:
                            timestamp, frame_id, action_id, score, action_name = action
                            print(f"  {i+1}. {timestamp:.1f}s -> {action_name} (confidence: {score:.3f})")

                # Ensure consistent data format (convert to 5-element tuples)
                normalized_detections = []
                for detection in all_action_detections:
                    if len(detection) == 4:
                        # Convert (timestamp, frame_id, score, action_name) to (timestamp, frame_id, -1, score, action_name)
                        timestamp, frame_id, score, action_name = detection
                        normalized_detections.append((timestamp, frame_id, -1, score, action_name))
                    else:
                        normalized_detections.append(detection)
                
                # 1️⃣ Group consecutive actions chronologically
                grouped_actions = group_consecutive_adaptive(all_action_detections, max_gap=1)

                print(f"\nDEBUG: Grouped {len(all_action_detections)} individual actions into {len(grouped_actions)} sequences")

                # 2️⃣ Sort by confidence (best first) and cap total duration
                # Cap at 2-3x your target to give scoring system good options without hanging
                MAX_ACTION_DURATION = target_duration * 3  # 3x target = plenty of options
                selected_sequences = []
                total_duration = 0

                sorted_sequences = sorted(grouped_actions, key=lambda x: x[3], reverse=True)

                for sequence in sorted_sequences:
                    start_time, end_time, duration, confidence, action_name = sequence
                    
                    # Stop when we hit duration cap
                    if total_duration >= MAX_ACTION_DURATION:
                        break
                    
                    selected_sequences.append(sequence)
                    total_duration += duration
                    
                    # Log ALL sequences
                    print(f"DEBUG: Added sequence {action_name} at {seconds_to_mmss(start_time)}-{seconds_to_mmss(end_time)} "
                        f"({duration:.1f}s duration, confidence: {confidence:.3f}) - Total: {seconds_to_mmss(total_duration)})")

                print(f"\nDEBUG: Selected {len(selected_sequences)} sequences totaling {total_duration:.1f}s")

                # 3️⃣ Convert back to individual action format for pipeline compatibility
                action_detections = []
                for start_time, end_time, duration, confidence, action_name in selected_sequences:
                    # pick best detection in this group
                    detections_in_group = [
                        a for a in all_action_detections
                        if a[4] == action_name and start_time <= a[0] <= end_time
                    ]
                    if detections_in_group:
                        best_detection = max(detections_in_group, key=lambda a: a[3])  # highest confidence
                        action_detections.append(best_detection)

                # Sort chronologically for pipeline
                action_detections = sorted(action_detections, key=lambda x: x[0])
                log(f"✅ Action recognition: {len(action_detections)} action sequences selected (total duration: {total_duration:.1f}s)")


            except Exception as e:
                log(f"⚠ Action recognition failed: {e}")
                import traceback
                log(f"Full error: {traceback.format_exc()}")
                            
        # 6 Compute scores per second
        progress.update_progress(80, 100, "Pipeline", "Computing scores...")
        check_cancellation(cancel_flag, log, "score computation")
        
        score = np.zeros(int(video_duration) + 1)
        scene_score = np.zeros_like(score)
        motion_event_score = np.zeros_like(score)
        motion_peak_score = np.zeros_like(score)
        audio_score = np.zeros_like(score)
        keyword_score = np.zeros_like(score)
        beginning_score = np.zeros_like(score)
        ending_score = np.zeros_like(score)
        object_score = np.zeros_like(score)
        action_score = np.zeros(int(video_duration) + 1)

        # Scoring configuration: prefer gui overrides, else config.yaml, else defaults
        SCENE_POINTS = gui_config.get("scene_points", config.get("scene_points", 0))
        MOTION_EVENT_POINTS = gui_config.get("motion_event_points", config.get("motion_event_points", 0))
        MOTION_PEAK_POINTS = gui_config.get("motion_peak_points", config.get("motion_peak_points", 3))
        AUDIO_PEAK_POINTS = gui_config.get("audio_peak_points", config.get("audio_peak_points", 0))
        KEYWORD_POINTS = gui_config.get("keyword_points", config.get("keyword_points", 2))
        BEGINNING_POINTS = gui_config.get("beginning_points", config.get("beginning_points", 0))
        ENDING_POINTS = gui_config.get("ending_points", config.get("ending_points", 0))
        MULTI_SIGNAL_BOOST = gui_config.get("multi_signal_boost", config.get("multi_signal_boost", 1.2))
        MIN_SIGNALS_FOR_BOOST = gui_config.get("min_signals_for_boost", config.get("min_signals_for_boost", 2))
        OBJECT_POINTS = gui_config.get("object_points", config.get("object_points", 10))
        ACTION_POINTS = gui_config.get("action_points", config.get("action_points", 10))
        
        # Fill scores using the detected signals
        for start, end in scenes:
            idx = int(start)
            if idx < len(score):
                scene_score[idx] += SCENE_POINTS
        for t in motion_events:
            idx = int(t)
            if idx < len(score):
                motion_event_score[idx] += MOTION_EVENT_POINTS
        for t in motion_peaks:
            idx = int(t)
            if idx < len(score):
                motion_peak_score[idx] += MOTION_PEAK_POINTS
        for t in audio_peaks:
            idx = int(t)
            if idx < len(score):
                audio_score[idx] += AUDIO_PEAK_POINTS

        # object scoring
        total_detections = sum(len(objs) for objs in object_detections.values())
        detection_summary = {}
        for sec, objs in object_detections.items():
            for obj in objs:
                detection_summary[obj] = detection_summary.get(obj, 0) + 1
                if obj in highlight_objects and sec < len(object_score):
                    object_score[sec] += OBJECT_POINTS

        # action scoring (group by seconds)
        detections_by_sec = defaultdict(list)
        for (timestamp_secs, frame_id, action_id, sc, action_name) in action_detections:
            sec = int(timestamp_secs)
            detections_by_sec[sec].append((action_name, sc))

        # Get the require_objects flag
        actions_require_objects = gui_config.get("actions_require_objects", False)
        OBJECT_TOLERANCE = 10
        BASE_ACTION_POINTS = ACTION_POINTS

        # action scoring (group by seconds)
        detections_by_sec = defaultdict(list)
        for (timestamp_secs, frame_id, action_id, sc, action_name) in action_detections:
            sec = int(timestamp_secs)
            detections_by_sec[sec].append((action_name, sc))

        # Calculate confidence percentiles to scale relative to video
        if detections_by_sec:
            all_confidences = [max(conf for _, conf in actions) for actions in detections_by_sec.values()]
            confidence_90th = np.percentile(all_confidences, 90)  # 90th percentile as "high confidence"
            confidence_50th = np.percentile(all_confidences, 50)  # 50th percentile as "average"
            
            log(f"📊 Action confidence stats: 50th={confidence_50th:.2f}, 90th={confidence_90th:.2f}")

        for sec, actions in detections_by_sec.items():
            if sec < len(action_score):
                if not actions_require_objects or any(abs(obj_sec - sec) <= OBJECT_TOLERANCE for obj_sec in object_detections):
                    max_confidence = max(conf for _, conf in actions)
                    
                    # Scale points relative to this video's confidence distribution
                    if max_confidence >= confidence_90th:
                        # Top 10% confidence: bonus points
                        action_score[sec] += ACTION_POINTS * 1.5
                    elif max_confidence >= confidence_50th:
                        # Above average: normal points
                        action_score[sec] += ACTION_POINTS
                    else:
                        # Below average: reduced points
                        action_score[sec] += ACTION_POINTS * 0.5

        log(f"✅ Object detection summary: {total_detections} detections")

        # Beginning & ending boost
        for i in range(min(int(video_duration), 60)):
            beginning_score[i] += BEGINNING_POINTS
        for i in range(max(0, int(video_duration) - 120), int(video_duration)):
            ending_score[i] += ENDING_POINTS

        # Sum signals
        score = (scene_score + motion_event_score + motion_peak_score + audio_score +
                 keyword_score + beginning_score + ending_score + object_score + action_score)

        # Multi-signal boost
        motion_set = set(int(t) for t in motion_events)
        motion_peaks_set = set(int(t) for t in motion_peaks)
        audio_set = set(int(t) for t in audio_peaks)
        keyword_set = set()
        object_set = set(object_detections.keys())
        action_set = set(detections_by_sec.keys())

        for i in range(len(score)):
            signals = sum([
                i in motion_set,
                i in motion_peaks_set,
                i in audio_set,
                i in keyword_set,
                i in object_set,
                i in action_set
            ])
            if signals >= MIN_SIGNALS_FOR_BOOST:
                score[i] *= MULTI_SIGNAL_BOOST

        progress.update_progress(80, 100, "Score Calculation", "Score computation complete")
        check_cancellation(cancel_flag, log, "score computation completion")

        # -------------------------
        # DEBUG: score breakdown
        # -------------------------
        max_score = max(score)
        min_score = min(score)
        avg_score = np.mean(score)
        ending_start = max(0, video_duration - 120)
        ending_scores = score[int(ending_start):] if ending_start < len(score) else []

        print(f"\n=== SCORE DISTRIBUTION ===")
        print(f"Max score: {max_score:.1f}")
        print(f"Min score: {min_score:.1f}")
        print(f"Average score: {avg_score:.1f}")
        print(f"Score range: {max_score - min_score:.1f}")
        print(f"Average ending score: {np.mean(ending_scores) if len(ending_scores) > 0 else 0:.2f}")

        # Top 10 scoring seconds
        top_indices = np.argsort(score)[-10:][::-1]
        print(f"\n=== TOP 10 SCORING MOMENTS ===")
        for i, idx in enumerate(top_indices):
            timestamp = f"{idx//60:02d}:{idx%60:02d}"
            print(f"{i+1}. Second {idx} ({timestamp}): {score[idx]:.1f} points")

        # Module-level flag to ensure logging happens only once per video
        if 'segments_logged' not in globals():
            globals()['segments_logged'] = False

        # Only use scored seconds depending on mode
        if duration_mode == "EXACT":
            candidate_indices = np.arange(len(score))  # allow seconds with 0 point score 
        else:  # "MAX"
            candidate_indices = np.where(score > 0)[0]

        # Get scores for candidate indices
        candidate_scores = score[candidate_indices]

        # Get confidence values for tie-breaking
        candidate_confidences = np.zeros(len(candidate_indices))
        for idx, sec in enumerate(candidate_indices):
            if sec in detections_by_sec:
                candidate_confidences[idx] = max(conf for _, conf in detections_by_sec[sec])

        # Sort by score descending, then by confidence descending for ties
        sorted_indices = np.lexsort((-candidate_confidences, -candidate_scores))  # negative for descending
        top_indices_all = candidate_indices[sorted_indices]

        # DEBUG: Print top 20 actions moments being considered
        print(f"\n=== TOP 20 ACTION MOMENTS BEING CONSIDERED ===")
        for i, sec in enumerate(top_indices_all[:20]):
            timestamp = f"{sec//60:02d}:{sec%60:02d}"
            
            # Get confidence for this second
            confidence = 0.0
            if sec in detections_by_sec:
                confidence = max(conf for _, conf in detections_by_sec[sec])
            
            print(f"{i+1}. Second {sec} ({timestamp}): {score[sec]:.1f} points, confidence: {confidence:.3f}")
            
        segments = []
        used_seconds = set()

        for sec in top_indices_all:
            if sec in used_seconds:
                continue

            start = max(0, sec - CLIP_TIME // 2)
            end = min(video_duration, start + CLIP_TIME)

            # Adjust start/end to ensure full CLIP_TIME
            if end - start < CLIP_TIME and end < video_duration:
                end = min(video_duration, start + CLIP_TIME)
            if end - start < CLIP_TIME and start > 0:
                start = max(0, end - CLIP_TIME)

            # Skip if any second is already used
            if any(s in used_seconds for s in range(int(start), int(end))):
                continue

            # Adjust segment to not exceed target duration ---
            current_duration = sum(e - s for s, e in segments)
            remaining = target_duration - current_duration
            if remaining <= 0:
                break
            if end - start > remaining:
                end = start + remaining

            segments.append((start, end))
            for s in range(int(start), int(end)):
                used_seconds.add(s)

            # Break based on duration mode
            current_duration = sum(e - s for s, e in segments)
            if duration_mode == "EXACT" and current_duration >= EXACT_DURATION:
                break
            elif duration_mode == "MAX" and current_duration >= MAX_DURATION:
                break

        # Sort segments by start time
        segments.sort(key=lambda x: x[0])

        print("\n🔍 FINAL HIGHLIGHT BREAKDOWN:")
        print(f"Total segments: {len(segments)}")
        total_final_duration = sum(e - s for s, e in segments)
        print(f"Total highlight duration: {total_final_duration:.1f}s")

        # Show the actual selected segments with BETTER confidence information
        print(f"\nACTUAL SELECTED SEGMENTS (PEAK CONFIDENCE):")
        for i, (seg_start, seg_end) in enumerate(segments):
            seg_duration = seg_end - seg_start
            
            # Find the PEAK confidence in this segment (not average)
            peak_confidence = 0
            high_confidence_moments = []
            
            for action_seq in selected_sequences:
                action_start, action_end, action_duration, action_conf, action_name = action_seq
                overlap_start = max(action_start, seg_start)
                overlap_end = min(action_end, seg_end)
                # FIX: Use >= instead of > to include single-moment actions
                if overlap_end >= overlap_start and action_conf > peak_confidence:
                    peak_confidence = action_conf
                if action_conf > 5.0:  # Track high-confidence moments
                    high_confidence_moments.append((action_conf, f"{seconds_to_mmss(action_start)}-{seconds_to_mmss(action_end)}"))
            
            # Sort high-confidence moments
            high_confidence_moments.sort(reverse=True)
            
            if peak_confidence > 0:
                confidence_str = f"PEAK: {peak_confidence:.1f}"
                if high_confidence_moments:
                    confidence_str += f" | {len(high_confidence_moments)} high-conf moments"
                    if len(high_confidence_moments) <= 3:  # Show top 3 if not too many
                        for conf, range_str in high_confidence_moments[:3]:
                            confidence_str += f" | {range_str}({conf:.1f})"
            else:
                confidence_str = "no high-confidence actions"
            
            print(f"  Segment {i+1}: {seconds_to_mmss(seg_start)}-{seconds_to_mmss(seg_end)} ({seg_duration:.1f}s) - {confidence_str}")

        # Check which action sequences made it into the final highlight (SIGNIFICANTLY included)
        action_sequences_in_highlight = []
        for action_seq in selected_sequences:
            action_start, action_end, action_duration, action_conf, action_name = action_seq
            # Check if this action sequence is SIGNIFICANTLY included (not just 0s overlap)
            for seg_start, seg_end in segments:
                overlap_start = max(action_start, seg_start)
                overlap_end = min(action_end, seg_end)
                overlap_duration = overlap_end - overlap_start
                
                # FIX: Use >= 0 instead of > 0 to include single-moment actions
                if overlap_duration >= 0:
                    included_ratio = overlap_duration / action_duration
                    action_sequences_in_highlight.append({
                        'action_name': action_name,
                        'original_range': f"{seconds_to_mmss(action_start)}-{seconds_to_mmss(action_end)}",
                        'highlight_range': f"{seconds_to_mmss(overlap_start)}-{seconds_to_mmss(overlap_end)}", 
                        'duration': overlap_duration,
                        'confidence': action_conf,
                        'included_ratio': included_ratio
                    })
                    break

        print(f"\nACTION SEQUENCES INCLUDED IN HIGHLIGHT (≥1s):")
        if action_sequences_in_highlight:
            # Sort by confidence to see what actually made it
            action_sequences_in_highlight.sort(key=lambda x: x['confidence'], reverse=True)
            
            for action in action_sequences_in_highlight:
                ratio_percent = action['included_ratio'] * 100
                print(f"  {action['action_name']}: {action['highlight_range']} "
                    f"({action['duration']:.1f}s, {ratio_percent:.0f}% of original, conf: {action['confidence']:.3f})")
        else:
            print("  No action sequences significantly included in final highlight")
            
        total_action_duration = sum(a['duration'] for a in action_sequences_in_highlight)
        if total_final_duration > 0:
            action_percentage = (total_action_duration / total_final_duration) * 100
            print(f"Total action content in highlight: {total_action_duration:.1f}s ({action_percentage:.1f}% of total)")
        else:
            print(f"Total action content in highlight: {total_action_duration:.1f}s (no highlight segments)")

        # Also show high-confidence sequences that didn't make it
        print(f"\nTOP 10 HIGH-CONFIDENCE ACTION SEQUENCES EXCLUDED:")
        high_conf_excluded = []
        for action_seq in selected_sequences:
            action_start, action_end, action_duration, action_conf, action_name = action_seq
            included = False
            for seg_start, seg_end in segments:
                overlap_start = max(action_start, seg_start)
                overlap_end = min(action_end, seg_end)
                # FIX: Use >= 1.0 instead of > 1.0 to be consistent
                if overlap_end - overlap_start >= 1.0:  # At least 1s included
                    included = True
                    break
            if not included and action_conf > 6.0:  # Only show high confidence excluded
                high_conf_excluded.append((action_conf, action_name, f"{seconds_to_mmss(action_start)}-{seconds_to_mmss(action_end)}"))

        # Show top 10 excluded by confidence
        for conf, name, range_str in sorted(high_conf_excluded, reverse=True)[:10]:
            print(f"  {name}: {range_str} (conf: {conf:.3f})")



        # Compute total duration once
        total_duration = sum(e - s for s, e in segments)

        # Log final segments exactly once, even if target not reached
        if not globals()['segments_logged']:
            log(f"\n🎯 Final segments selected: {len(segments)}, total {total_duration:.1f}s (target {target_duration}s)")
            globals()['segments_logged'] = True

        print(f"\n=== DETAILED DEBUG FOR TOP MOMENTS ===")
        for idx in top_indices[:10]:
            minutes = idx // 60
            seconds = idx % 60
            timestamp = f"{minutes:02d}:{seconds:02d}"
            
            # Calculate pre-boost total
            pre_boost_total = (scene_score[idx] + motion_event_score[idx] + 
                            motion_peak_score[idx] + audio_score[idx] + 
                            keyword_score[idx] + object_score[idx] + action_score[idx])
            
            print(f"\nTime {timestamp} ({idx} sec): {score[idx]:.1f} total points")
            print(f"  Scene: {scene_score[idx]:.1f}")
            print(f"  Motion events: {motion_event_score[idx]:.1f}")
            print(f"  Motion peaks: {motion_peak_score[idx]:.1f}")
            print(f"  Audio: {audio_score[idx]:.1f}")
            print(f"  Keywords: {keyword_score[idx]:.1f}")
            print(f"  Objects: {object_score[idx]:.1f}")
            print(f"  Actions: {action_score[idx]:.1f}")
            print(f"  Subtotal (before boost): {pre_boost_total:.1f}")

            # 🔍 Show which objects were detected at this second
            if idx in object_detections:
                print(f"    Objects detected: {object_detections[idx]}")

            # 🔍 Show which actions were detected at this second
            if idx in detections_by_sec:
                detected_actions = [f"{name} ({score:.2f})" for name, score in detections_by_sec[idx]]
                print(f"    Actions detected: {', '.join(detected_actions)}")
                
                actions_require_objects = gui_config.get("actions_require_objects", False)
                if actions_require_objects:
                    if idx in object_detections:
                        # Show actual points added (includes confidence multiplier)
                        actual_points = action_score[idx]
                        max_confidence = max(conf for _, conf in detections_by_sec[idx])
                        
                        if max_confidence >= confidence_90th:
                            tier = "BONUS (≥90th percentile)"
                        elif max_confidence >= confidence_50th:
                            tier = "NORMAL (≥50th percentile)"
                        else:
                            tier = "REDUCED (<50th percentile)"
                        
                        print(f"    ✓ Action scored (objects present): +{actual_points:.1f} points [{tier}, conf={max_confidence:.2f}]")
                    else:
                        print(f"    ✗ Action NOT scored (no objects detected)")
                else:
                    # Show actual points added (includes confidence multiplier)
                    actual_points = action_score[idx]
                    max_confidence = max(conf for _, conf in detections_by_sec[idx])
                    
                    if max_confidence >= confidence_90th:
                        tier = "BONUS (≥90th percentile)"
                    elif max_confidence >= confidence_50th:
                        tier = "NORMAL (≥50th percentile)"
                    else:
                        tier = "REDUCED (<50th percentile)"
                    
                    print(f"    ➕ Added {actual_points:.1f} action points [{tier}, conf={max_confidence:.2f}]")
                        
            # Count signals
            signals = sum([
                motion_event_score[idx] > 0,
                motion_peak_score[idx] > 0,
                audio_score[idx] > 0,
                keyword_score[idx] > 0,
                object_score[idx] > 0,
                idx in detections_by_sec
            ])
            
            if signals >= MIN_SIGNALS_FOR_BOOST:
                boost_amount = score[idx] - pre_boost_total
                print(f"  ⚡ Multi-signal boost: {signals} signals detected")
                print(f"     Multiplier: x{MULTI_SIGNAL_BOOST}")
                print(f"     Boost added: +{boost_amount:.1f} points")
                print(f"     Final score: {score[idx]:.1f}")





        check_cancellation(cancel_flag, log, "segment selection")

        # Cut and concatenate
        progress.update_progress(90, 100, "Pipeline", "Creating highlight video...")
        log("🔹 Step 7: Cutting video segments...")
        try:
            if len(segments) == 0:
                log("⚠️ No segments selected — nothing to cut.")
            elif len(segments) == 1:
                check_cancellation(cancel_flag, log, "video cutting")
                cut_video(video_path, segments[0][0], segments[0][1], OUTPUT_FILE)
            else:
                temp_clips = []
                # Get the directory of the output file to save temp clips in the same location
                output_dir = os.path.dirname(OUTPUT_FILE)
                video_base_name = os.path.splitext(os.path.basename(video_path))[0]
                
                for i, (s, e) in enumerate(segments):
                    check_cancellation(cancel_flag, log, f"video cutting clip {i+1}")
                    # Include the directory path for temp files
                    temp_name = os.path.join(output_dir, f"{video_base_name}_temp_clip_{i}.mp4")
                    cut_video(video_path, s, e, temp_name)
                    temp_clips.append(temp_name)
                    # Update progress for each clip
                    progress.update_progress(90 + (i+1) * 5 // len(segments), 100, "Pipeline", f"Cut clip {i+1}/{len(segments)}")
                
                check_cancellation(cancel_flag, log, "video concatenation")
                
                concat_file = os.path.join(output_dir, "concat_list.txt")
                with open(concat_file, "w") as f:
                    for t in temp_clips:
                        f.write(f"file '{t}'\n")
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                                "-i", concat_file, "-c", "copy", OUTPUT_FILE], check=True)
                if not KEEP_TEMP:
                    for t in temp_clips:
                        try:
                            os.remove(t)
                        except Exception:
                            pass
                    try:
                        os.remove(concat_file)
                    except Exception:
                        pass
            log(f"✅ Highlight saved: {OUTPUT_FILE}, duration {total_duration:.1f}s")
        except RuntimeError:
            return None
        except Exception as e:
            log(f"⚠️ Error during cutting/concatenation: {e}")
            raise



        # Create matching subtitles for highlight video OR full video
        if CREATE_SUBTITLES and USE_TRANSCRIPT and transcript_segments:
            try:
                base_name = os.path.splitext(OUTPUT_FILE)[0]

                # Always create full subtitles
                progress.update_progress(95, 100, "Pipeline", "Creating full-video subtitles...")
                log("Creating subtitles for the full video...")
                full_srt = f"{os.path.splitext(video_path)[0]}_{TARGET_LANG}.srt"
                if TARGET_LANG and TARGET_LANG != SOURCE_LANG:
                    translated = translate_segments(transcript_segments, target_lang=TARGET_LANG)
                    create_srt_file(translated, full_srt)
                else:
                    full_srt = f"{os.path.splitext(video_path)[0]}_{SOURCE_LANG}.srt"
                    create_srt_file(transcript_segments, full_srt)
                log(f"Full-video subtitles created: {full_srt}")

                # Create highlight subtitles if we have segments
                if segments:
                    progress.update_progress(95, 100, "Pipeline", "Creating highlight subtitles...")
                    log("Creating subtitles that match highlight timing...")
                    if TARGET_LANG and TARGET_LANG != SOURCE_LANG:
                        highlight_srt_file = f"{base_name}_{TARGET_LANG}.srt"
                        create_highlight_subtitles(
                            original_segments=transcript_segments,
                            highlight_segments=segments,
                            output_path=highlight_srt_file,
                            source_lang=SOURCE_LANG,
                            target_lang=TARGET_LANG
                        )
                    else:
                        highlight_srt_file = f"{base_name}_{SOURCE_LANG}.srt"
                        create_highlight_subtitles(
                            original_segments=transcript_segments,
                            highlight_segments=segments,
                            output_path=highlight_srt_file,
                            source_lang=SOURCE_LANG,
                            target_lang=None
                        )
                    log(f"Highlight subtitles created: {highlight_srt_file}")

            except Exception as e:
                log(f"Error creating subtitles: {e}")


        # Final progress
        progress.update_progress(100, 100, "Pipeline", "Complete!")

        # End timer
        end_time = time.time()
        elapsed = end_time - start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        log(f"⏱️ Processing time: {minutes}m {seconds}s")

        # Clean up XPU memory if used
        try:
            if xpu_available and "xpu" in yolo_device:
                try:
                    torch.xpu.empty_cache()
                    log("✅ XPU memory cleaned up")
                except Exception:
                    pass
        except Exception:
            pass

        return OUTPUT_FILE

    except RuntimeError as e:
        # This handles our cancellation exceptions
        log(f"⏹️ Pipeline cancelled: {e}")
        return None
    except Exception as e:
        log(f"❌ Pipeline failed: {e}")
        import traceback
        log(f"Full error: {traceback.format_exc()}")
        return None