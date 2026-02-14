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
from ultralytics import YOLO

from action_recognition import run_action_detection, load_models
# modules
from modules.audio_peaks import extract_audio_peaks
from modules.motion_scene_detect_optimized import detect_scenes_motion_optimized
from modules.video_cache import VideoAnalysisCache, CachedAnalysisData
from modules.video_cutter import cut_video

# Keep warnings about CUDA quiet
warnings.filterwarnings("ignore", message="torch.cuda")

def build_analysis_cache_params(gui_config: dict, config: dict, sample_rate: int, video_duration: float):
    # “Analysis params” = anything that changes the computed analysis artifacts
    # Keep values JSON-serializable and stable (sort lists)
    highlight_objects = gui_config.get("highlight_objects", config.get("highlight_objects", [])) or []
    interesting_actions = gui_config.get("interesting_actions", []) or []
    search_keywords = gui_config.get("search_keywords", []) or []

    # Time range settings (if enabled)
    use_time_range = bool(gui_config.get("use_time_range", False))
    range_start = int(gui_config.get("range_start", 0) or 0)
    range_end = gui_config.get("range_end", None)
    range_end = int(range_end) if range_end is not None else None

    # YOLO settings
    yolo_model_size = str(gui_config.get("yolo_model_size") or "n").lower()
    openvino_model_folder = gui_config.get("openvino_model_folder", f"yolo11{yolo_model_size}_openvino_model/")
    yolo_pt_path = gui_config.get("yolo_pt_path", f"yolo11{yolo_model_size}.pt")

    params = {
        # bump this when you change the meaning/format of cached analysis
        "analysis_cache_schema": "analysis_v2",

        # core toggles
        "use_transcript": bool(gui_config.get("use_transcript", False)),
        "transcript_model": str(gui_config.get("transcript_model", "medium")),
        "search_keywords": sorted([str(k).lower() for k in search_keywords]),

        "highlight_objects": sorted([str(o) for o in highlight_objects]),
        "interesting_actions": sorted([str(a) for a in interesting_actions]),

        # object/action sampling knobs
        "object_frame_skip": int(gui_config.get("object_frame_skip", gui_config.get("clip_time", 10) or 10)),
        "sample_rate": int(sample_rate),

        # action detector knobs used in your call
        "action_use_person_detection": True,
        "action_max_people": int(gui_config.get("action_max_people", 2) or 2),

        # yolo identity
        "yolo_model_size": yolo_model_size,
        "yolo_pt_path": str(yolo_pt_path),
        "openvino_model_folder": str(openvino_model_folder),

        # time-range
        "use_time_range": use_time_range,
        "range_start": range_start if use_time_range else 0,
        "range_end": range_end if use_time_range else None,

        # optional: points affect scoring, not analysis — but if you cache “analysis only”
        # you can omit scoring params. If you cache waveforms/peaks based on thresholds,
        # include them.
        "scene_threshold": float(gui_config.get("scene_threshold", 70.0)),
        "motion_threshold": float(gui_config.get("motion_threshold", 100.0)),
        "spike_factor": float(gui_config.get("spike_factor", 1.2)),
        "freeze_seconds": float(gui_config.get("freeze_seconds", 4)),
        "freeze_factor": float(gui_config.get("freeze_factor", 0.8)),
    }
    return params

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
        # Try to import but don't fail if not available
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
                                 annotated_output=None, yolo_model_size="n",
                                 yolo_pt_path=None, openvino_model_folder=None):
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
        if video_writer:
            video_writer.release()
            if draw_boxes:
                log_fn(f"✅ Object detection annotated video saved: {annotated_output}")

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


def collect_analysis_data(video_path, video_duration, fps, transcript_segments, 
                         object_detections, action_detections, scenes, 
                         motion_events, motion_peaks, audio_peaks, source_lang="en",
                         waveform_data=None, keyword_segments_only=False, 
                         search_keywords=None, keyword_matches=None):  # Add this parameter
    """
    Collect all analysis results into a structured dictionary for caching.

    Args:
        keyword_segments_only: If True and search_keywords provided, only cache segments containing keywords
        search_keywords: List of keywords to filter transcript segments
        keyword_matches: Pre-computed keyword matches to cache
        waveform_data: Optional waveform data for timeline visualization
    """
    # Filter transcript segments if we're only caching keyword-relevant parts
    filtered_transcript_segments = transcript_segments
    if keyword_segments_only and search_keywords and transcript_segments:
        # Create a set of keywords for faster lookup
        keyword_set = {kw.lower() for kw in search_keywords}
        filtered_transcript_segments = []
        
        for segment in transcript_segments:
            segment_text = segment.get("text", "").lower()
            # Check if any keyword is in the segment text
            if any(keyword in segment_text for keyword in keyword_set):
                filtered_transcript_segments.append(segment)
    
    # Ensure action_detections is in a cacheable format
    actions_for_cache = []
    for detection in action_detections:
        if len(detection) >= 5:
            timestamp, frame_id, action_id, score, action_name = detection[:5]
            actions_for_cache.append({
                "timestamp": float(timestamp),
                "frame_id": int(frame_id),
                "action_id": int(action_id),
                "confidence": float(score),
                "action_name": str(action_name)
            })
    
    # Convert numpy arrays/lists to Python native types
    motion_events_clean = [float(t) for t in motion_events]
    motion_peaks_clean = [float(t) for t in motion_peaks]
    audio_peaks_clean = [float(t) for t in audio_peaks]
    
    analysis_data = {
        "video_metadata": {
            "duration": float(video_duration),
            "fps": float(fps),
            "resolution": "unknown",
            "total_frames": int(video_duration * fps),
            "file_size": int(os.path.getsize(video_path)) if os.path.exists(video_path) else 0
        },
        "transcript": {
            "segments": filtered_transcript_segments if keyword_segments_only else transcript_segments,
            "language": source_lang,
            "cached_full_transcript": not keyword_segments_only,
            "keyword_filtered": keyword_segments_only
        },
        "keyword_matches": keyword_matches or [],
        "objects": [
            {
                "timestamp": int(sec),
                "objects": [str(obj) for obj in objs],
                "count": len(objs)
            }
            for sec, objs in object_detections.items()
        ],
        "actions": actions_for_cache,
        "scenes": [
            {"start": float(start), "end": float(end)}
            for start, end in scenes
        ],
        "motion_events": motion_events_clean,
        "motion_peaks": motion_peaks_clean,
        "pipeline_version": "1.0",
        "cache_flags": {
            "keyword_segments_only": keyword_segments_only,
            "search_keywords": search_keywords if keyword_segments_only else None
        }
    }
    
    # Add audio data (including waveform for timeline viewer)
    # Store in a structured way for easy access
    analysis_data["audio"] = {
        "peaks": audio_peaks_clean,
        "waveform": waveform_data
    }
    
    # Also keep legacy key for backward compatibility
    analysis_data["audio_peaks"] = audio_peaks_clean
    
    return analysis_data

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
            base_name = os.path.splitext(single_video_path)[0]  # Always use current video's name
            video_gui_config["output_file"] = f"{base_name}_highlight.mp4"
                        
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
        TRANSCRIPT_SOURCE_LANG = gui_config.get("transcript_source_lang", "en")
        SEARCH_KEYWORDS = gui_config.get("search_keywords", [])
        CREATE_SUBTITLES = gui_config.get("create_subtitles", False)
        TRANSCRIPT_ONLY = gui_config.get("transcript_only", False)
        TRANSCRIPT_POINTS = int(gui_config.get("transcript_points", 0))
        SOURCE_LANG = gui_config.get("source_lang", "en")  # For subtitles
        TARGET_LANG = gui_config.get("target_lang", None)  # For subtitles

        keyword_matches = []

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

        # --- Time Range Processing ---
        USE_TIME_RANGE = gui_config.get("use_time_range", False)
        RANGE_START = int(gui_config.get("range_start", 0))
        RANGE_END = gui_config.get("range_end", None)
        if RANGE_END is not None:
            RANGE_END = int(RANGE_END)

        # Store original video path and duration for later use
        original_video_path = video_path
        original_video_duration = video_duration
        processed_video_path = video_path
        temp_trimmed_video = None

        if USE_TIME_RANGE:
            if RANGE_END is None or RANGE_END == 0:
                RANGE_END = video_duration
            
            # Validate range
            if RANGE_START >= RANGE_END:
                log(f"⚠️ Invalid time range: start ({RANGE_START}s) >= end ({RANGE_END}s)")
                return None
            
            if RANGE_START >= video_duration:
                log(f"⚠️ Start time ({RANGE_START}s) exceeds video duration ({video_duration:.1f}s)")
                return None
            
            # Clamp end time to video duration
            RANGE_END = int(min(RANGE_END, video_duration))
            range_duration = RANGE_END - RANGE_START
            
            log(f"🎯 Processing time range: {RANGE_START//60}:{RANGE_START%60:02d} to {RANGE_END//60}:{RANGE_END%60:02d}")
            log(f"   Range duration: {range_duration//60}:{int(range_duration%60):02d} ({range_duration:.1f}s)")
            log(f"   Skipping: {RANGE_START:.1f}s at start, {video_duration - RANGE_END:.1f}s at end")
            
            # Create temporary trimmed video
            progress.update_progress(5, 100, "Pipeline", "Trimming video to selected range...")
            
            video_base_name = os.path.splitext(os.path.basename(video_path))[0]
            temp_folder = os.path.dirname(video_path) or "."
            temp_trimmed_video = os.path.join(temp_folder, f"{video_base_name}_temp_trimmed.mp4")
            
            try:
                check_cancellation(cancel_flag, log, "video trimming")
                
                # Use FFmpeg to trim the video (fast, no re-encoding)
                log(f"   Using FFmpeg to extract range...")
                subprocess.run([
                    "ffmpeg", "-y", "-v", "error",
                    "-ss", str(RANGE_START),
                    "-to", str(RANGE_END),
                    "-i", video_path,
                    "-c", "copy",  # Copy streams without re-encoding for speed
                    temp_trimmed_video
                ], check=True)
                
                log(f"✅ Video trimmed to: {temp_trimmed_video}")
                processed_video_path = temp_trimmed_video
                
                # Update video_duration for the rest of the pipeline
                video_duration = range_duration
                
                # Update video info for the trimmed video
                cap = cv2.VideoCapture(processed_video_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                cap.release()
                log(f"📊 Trimmed video: {video_duration:.2f}s, FPS: {fps}, frames: {total_frames}")
                
            except subprocess.CalledProcessError as e:
                log(f"⚠️ FFmpeg trimming with copy failed, trying with re-encoding...")
                try:
                    # Fallback: re-encode if copy fails
                    subprocess.run([
                        "ffmpeg", "-y", "-v", "error",
                        "-ss", str(RANGE_START),
                        "-to", str(RANGE_END),
                        "-i", video_path,
                        temp_trimmed_video
                    ], check=True)
                    log(f"✅ Video trimmed (re-encoded) to: {temp_trimmed_video}")
                    processed_video_path = temp_trimmed_video
                    video_duration = range_duration
                    
                    # Update video info
                    cap = cv2.VideoCapture(processed_video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    cap.release()
                    log(f"📊 Trimmed video: {video_duration:.2f}s, FPS: {fps}, frames: {total_frames}")
                except Exception as e2:
                    log(f"❌ Failed to trim video: {e2}")
                    return None
            except RuntimeError:
                return None
        else:
            log("ℹ️ Processing full video")

        # ========== CACHE CHECK ==========
        # Goal:
        # - Cache MUST be invalidated automatically when settings change (objects/actions/transcript/time-range/etc.)
        # - Use VideoAnalysisCache signature-based files via load(..., params=analysis_params)
        # - Maintain backward compatibility
        # - Ensure timeline viewer gets all necessary data

        # Build analysis parameters that affect cache signature
        analysis_params = build_analysis_cache_params(
            gui_config=gui_config,
            config=config,
            sample_rate=sample_rate,
            video_duration=video_duration
        )

        # Initialize cache controls
        use_cache = gui_config.get("use_cache", True)
        force_reprocess = gui_config.get("force_reprocess", False)

        # Initialize variables that might come from cache
        transcript_segments = []
        object_detections = {}
        action_detections = []
        scenes = []
        motion_events = []
        motion_peaks = []
        audio_peaks = []
        waveform_data = None  # For timeline viewer
        using_cache = False

        # Try to load from cache if enabled
        if use_cache and not force_reprocess:
            cache = VideoAnalysisCache(cache_dir=gui_config.get("cache_dir", "./cache"))
            try:
                start_time_cache = time.time()
                # Use signature-based loading
                cached_data = cache.load(processed_video_path, params=analysis_params)
                load_time = time.time() - start_time_cache
                
                if cached_data:
                    # Verify it's for the same video (check duration, etc.)
                    cache_video_duration = cached_data.get("video_metadata", {}).get("duration", 0)
                    if abs(cache_video_duration - video_duration) < 1.0:  # Within 1 second
                        # Check if the cache matches our current keyword requirements
                        cache_keyword_filtered = cached_data.get("transcript", {}).get("keyword_filtered", False)
                        cache_search_keywords = cached_data.get("cache_flags", {}).get("search_keywords", [])
                        cache_language = cached_data.get("transcript", {}).get("language", "en")  # Add this line
                        
                        # We can use cached data if:
                        # 1. We don't need transcript at all (not using transcript)
                        # 2. Cache has full transcript and we need full transcript
                        # 3. Cache has keyword-filtered transcript and we need keyword-filtered with same keywords
                        current_keywords = SEARCH_KEYWORDS if USE_TRANSCRIPT else []
                        
                        cache_compatible = False
                        if not USE_TRANSCRIPT:
                            cache_compatible = True
                        elif not cache_keyword_filtered:
                            # Cache has full transcript - ONLY COMPATIBLE IF LANGUAGES MATCH
                            if cache_language == TRANSCRIPT_SOURCE_LANG:
                                cache_compatible = True
                            else:
                                log(f"⚠️ Cache language mismatch: cached '{cache_language}' vs requested '{TRANSCRIPT_SOURCE_LANG}'")
                        elif cache_keyword_filtered and current_keywords:
                            # Check if cache has the keywords we need and language matches
                            cached_keywords_set = set([kw.lower() for kw in (cache_search_keywords or [])])
                            current_keywords_set = set([kw.lower() for kw in current_keywords])
                            if cached_keywords_set.issuperset(current_keywords_set) and cache_language == TRANSCRIPT_SOURCE_LANG:
                                cache_compatible = True
                            else:
                                log(f"⚠️ Cache incompatible: language mismatch or keywords not matching")
                        
                        if cache_compatible:
                            log(f"✅ Loaded from cache ({load_time:.2f}s) [signature match]")
                            
                            # Extract data from cache - Ensure all data is loaded
                            transcript_segments = cached_data.get("transcript", {}).get("segments", [])
                            object_detections_raw = cached_data.get("objects", [])
                            action_detections_raw = cached_data.get("actions", [])
                            scenes_raw = cached_data.get("scenes", [])
                            motion_events = cached_data.get("motion_events", [])
                            motion_peaks = cached_data.get("motion_peaks", [])
                            keyword_matches = cached_data.get("keyword_matches", [])


                            # Get audio data - handle both new and old formats
                            audio_block = cached_data.get("audio") or {}
                            if isinstance(audio_block, dict) and "peaks" in audio_block:
                                audio_peaks = audio_block.get("peaks", [])
                                waveform_data = audio_block.get("waveform")
                            else:
                                # Legacy format
                                audio_peaks = cached_data.get("audio_peaks", [])
                                waveform_data = cached_data.get("waveform") or cached_data.get("waveform_data")
                            
                            # Convert to pipeline format
                            object_detections = {}
                            for obj in object_detections_raw:
                                sec = int(obj.get("timestamp", 0))
                                object_detections[sec] = obj.get("objects", [])
                            
                            # Convert action detections to proper format
                            action_detections = []
                            if action_detections_raw:
                                for action in action_detections_raw:
                                    # Handle both 5-element and 6-element formats
                                    if len(action) >= 5:
                                        action_detections.append((
                                            action.get("timestamp", 0),
                                            action.get("frame_id", 0),
                                            action.get("action_id", -1),
                                            action.get("confidence", 0),
                                            action.get("action_name", "")
                                        ))
                            
                            scenes = [(s.get("start", 0), s.get("end", 0)) for s in scenes_raw]
                            
                            # Extract keyword matches from cache
                            keyword_matches = cached_data.get("keyword_matches", [])

                            # Mark that we're using cached data
                            using_cache = True
                            cache_status = "full" if not cache_keyword_filtered else f"keyword-filtered ({len(cache_search_keywords or [])} keywords)"
                            log(f"✅ Loaded from cache: {len(transcript_segments)} transcript segments ({cache_status}), "
                                f"{len(object_detections)} object seconds, {len(action_detections)} actions, "
                                f"{len(scenes)} scenes, {len(motion_events)} motion events, {len(motion_peaks)} motion peaks, "
                                f"{len(audio_peaks)} audio peaks")
                        else:
                            log(f"⚠️ Cache incompatible: cached with {'keyword-filtered' if cache_keyword_filtered else 'full'} transcript, "
                                f"need {'keyword-filtered' if current_keywords else 'full'} transcript")
                            cached_data = None
                    else:
                        log(f"⚠️ Cache duration mismatch: {cache_video_duration}s vs {video_duration}s")
                        cached_data = None
            except Exception as e:
                log(f"⚠️ Cache load error: {e}")
                cached_data = None
        else:
            log("ℹ️ Cache disabled or forced reprocess")

        # Ensure using_cache is properly set
        using_cache = 'cached_data' in locals() and cached_data is not None
        # ========== END CACHE CHECK ==========

        # --- Transcript processing ---
        if not using_cache:
                # Original transcript processing code
                progress.update_progress(5, 100, "Pipeline", "Processing transcript...")
                log("🔹 Step 0.5: Processing transcript...")
                try:
                    check_cancellation(cancel_flag, log, "transcript processing")
                    transcript_segments = get_transcript_segments(
                        processed_video_path, 
                        model_name=TRANSCRIPT_MODEL, 
                        progress_fn=progress_fn, 
                        log_fn=log,
                        language=TRANSCRIPT_SOURCE_LANG
                    )

                    
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
                    
                    # 🆕 ADD THIS DEBUG BLOCK:
                    if keyword_matches:
                        log(f"\n📊 KEYWORD MATCH DETAILS:")
                        for i, match in enumerate(keyword_matches[:10]):  # Show first 10
                            main_seg = match["main_segment"]
                            keyword = match.get("keyword", "unknown")
                            start_sec = int(main_seg["start"])
                            end_sec = int(main_seg["end"])
                            text = main_seg.get("text", "")[:50]  # First 50 chars
                            log(f"   Match {i+1}: '{keyword}' at {start_sec}-{end_sec}s")
                            log(f"            Text: \"{text}...\"")
                    else:
                        log(f"⚠️ No keyword matches found!")
                        log(f"   Searched for: {SEARCH_KEYWORDS}")
                        log(f"   In {len(transcript_segments)} transcript segments")
                else:
                    keyword_matches = []

        else:
            log("ℹ️ Using cached transcript")
            # transcript_segments and keyword_matches already loaded from cache
            
            if keyword_matches:
                log(f"✅ Loaded {len(keyword_matches)} keyword matches from cache")
            else:
                log("ℹ️ No keyword matches in cache (none were found or no keywords specified)")

        check_cancellation(cancel_flag, log, "transcript phase")

        start_time = time.time()

        # --- 1+2 Detect scenes + motion + peaks with live progress ---
        if not using_cache:
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
                        processed_video_path,
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
        else:
            log("ℹ️ Using cached motion analysis")
            progress.update_progress(25, 100, "Pipeline", "Loaded cached motion analysis")

        check_cancellation(cancel_flag, log, "motion detection completion")

        # 3 Audio peaks
        # - If audio_peak_points == 0: skip *peaks* but still compute waveform (for timeline viewer)
        # - If using cache: load peaks + waveform from cache (support both new and legacy key layouts)

        audio_peaks = audio_peaks if 'audio_peaks' in locals() else []
        waveform_data = None

        def _get_cached_waveform(cached):
            if not cached:
                return None
            # New preferred layout: {"audio": {"waveform": ...}}
            audio_block = cached.get("audio") or {}
            if isinstance(audio_block, dict) and "waveform" in audio_block:
                return audio_block.get("waveform")
            # Legacy layouts
            return cached.get("waveform") or cached.get("waveform_data")

        def _get_cached_audio_peaks(cached):
            if not cached:
                return []
            # New preferred layout: {"audio": {"peaks": [...]}}
            audio_block = cached.get("audio") or {}
            if isinstance(audio_block, dict) and "peaks" in audio_block:
                return audio_block.get("peaks") or []
            # Legacy layout: top-level "audio_peaks"
            return cached.get("audio_peaks") or []

        if using_cache:
            log("ℹ️ Using cached audio data")
            audio_peaks = _get_cached_audio_peaks(cached_data)
            waveform_data = _get_cached_waveform(cached_data)

            # If waveform wasn't cached in older runs, compute it now (cheap) so timeline works
            if waveform_data is None:
                try:
                    from modules.audio_peaks import extract_waveform_data
                    waveform_data = extract_waveform_data(processed_video_path, num_points=1000)
                    log("✅ Waveform computed (was missing in cache)")
                except Exception as e:
                    log(f"⚠️ Failed to compute waveform: {e}")

        else:
            # Check if we should skip audio detection based on GUI config
            audio_peak_points = gui_config.get("audio_peak_points", 0)

            # Always try to compute waveform for the timeline viewer
            try:
                from modules.audio_peaks import extract_waveform_data
                waveform_data = extract_waveform_data(processed_video_path, num_points=1000)
            except Exception as e:
                log(f"⚠️ Waveform extraction failed: {e}")
                waveform_data = None

            if audio_peak_points == 0:
                log("ℹ️ Skipping audio peak detection (audio_peak_points set to 0)")
                audio_peaks = []
                progress.update_progress(
                    30, 100, "Pipeline",
                    "Audio peaks skipped (no audio scoring) — waveform computed for timeline"
                )
            else:
                progress.update_progress(30, 100, "Pipeline", "Analyzing audio...")
                log("🔹 Step 3: Detecting audio peaks...")
                try:
                    check_cancellation(cancel_flag, log, "audio peak detection")
                    audio_peaks = extract_audio_peaks(processed_video_path, cancel_flag=cancel_flag)
                    log(f"✅ Audio peak detection done: {len(audio_peaks)} peaks")
                except RuntimeError:
                    return None

        # 4 Object detection setup
        progress.update_progress(40, 100, "Pipeline", "Setting up object detection...")
        check_cancellation(cancel_flag, log, "object detection setup")

        # Get list of objects to highlight from GUI or config
        highlight_objects = gui_config.get("highlight_objects", config.get("highlight_objects", []))

        yolo_model_size = str(gui_config.get("yolo_model_size") or "n").lower()
        openvino_model_folder = gui_config.get(
            "openvino_model_folder",
            f"yolo11{yolo_model_size}_openvino_model/"
        )
        yolo_pt_path = gui_config.get("yolo_pt_path", f"yolo11{yolo_model_size}.pt")


        # Also update the default PT path based on model size
        default_pt_path = f"yolo11{yolo_model_size}.pt"
        log(f"🎯 YOLO model size: {yolo_model_size} (using {default_pt_path})")

        # Check OpenVINO devices (best-effort)
        try:
            from openvino.runtime import Core
            ie = Core()
            log(f"🔹 OpenVINO available devices: {ie.available_devices}")
        except ImportError:
            log("ℹ️ OpenVINO not available")
        except Exception as e:
            log(f"⚠️ OpenVINO device check failed: {e}")

        # Export model to OpenVINO if missing
        if not os.path.exists(openvino_model_folder):
            try:
                check_cancellation(cancel_flag, log, "YOLO model export")
                log(f"⚠️ OpenVINO folder not found. Exporting YOLO model (requires {default_pt_path})...")
                
                # Use the PT path from config, or fall back to default based on model size
                yolo_pt_path = gui_config.get("yolo_pt_path", default_pt_path)
                yolo_model_export = YOLO(yolo_pt_path)
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
        if not using_cache:
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
                    processed_video_path,
                    yolo_model,
                    highlight_objects,
                    log_fn=log_fn,
                    progress_fn=progress_fn,
                    frame_skip=frame_skip_for_obj,
                    cancel_flag=cancel_flag,
                    draw_boxes=draw_object_boxes,
                    annotated_output=object_annotated_path,
                    yolo_model_size=yolo_model_size,
                    yolo_pt_path=yolo_pt_path,
                    openvino_model_folder=openvino_model_folder
                )

                log(f"✅ Object detection complete: {len(object_detections)} seconds with objects")
        else:
            log("ℹ️ Using cached object detections")

        print("Detections per second:", len(object_detections))

        def group_consecutive_adaptive(actions, max_gap=1.3, jump_threshold=0.01):
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
        interesting_actions = gui_config.get("interesting_actions", [])

        if not using_cache and interesting_actions:
            try:
                # Get action label settings
                draw_action_labels = gui_config.get("draw_action_labels", False)
                action_annotated_path = None
                if draw_action_labels:
                    video_basename = os.path.splitext(os.path.basename(video_path))[0]
                    temp_folder = os.path.dirname(video_path) or "."
                    action_annotated_path = os.path.join(temp_folder, f"{video_basename}_actions_annotated.mp4")
                    log(f"🎨 Action labels enabled, output: {action_annotated_path}")
                
                # Call action detection with include_model_type=False for backward compatibility
                all_action_detections = run_action_detection(
                    video_path=processed_video_path,
                    sample_rate=sample_rate,
                    debug=False,
                    interesting_actions=interesting_actions,
                    progress_callback=progress.update_progress,
                    cancel_flag=cancel_flag,
                    draw_bboxes=True,
                    annotated_output=action_annotated_path,
                    use_person_detection=True,
                    max_people=2,
                    include_model_type=False
                )

                check_cancellation(cancel_flag, log, "action recognition processing")

                if all_action_detections:
                    log(f"✅ Action detection complete: {len(all_action_detections)} detections")
                    
                    # DEBUG: Print format of returned data
                    if len(all_action_detections) > 0:
                        first_detection = all_action_detections[0]
                        log(f"DEBUG: Detection format - {len(first_detection)} elements: {first_detection}")
                    
                    # NORMALIZE: Ensure all detections are 5-element tuples
                    normalized_detections = []
                    for detection in all_action_detections:
                        if len(detection) == 5:
                            # Already correct format: (timestamp, frame_id, action_id, score, action_name)
                            normalized_detections.append(detection)
                        elif len(detection) == 4:
                            # Old format: (timestamp, frame_id, score, action_name)
                            timestamp, frame_id, score, action_name = detection
                            normalized_detections.append((timestamp, frame_id, -1, score, action_name))
                        elif len(detection) == 6:
                            # New format with model_type: (timestamp, frame_id, action_id, score, action_name, model_type)
                            timestamp, frame_id, action_id, score, action_name, model_type = detection
                            normalized_detections.append((timestamp, frame_id, action_id, score, action_name))
                        else:
                            log(f"⚠️ Unexpected detection format with {len(detection)} elements: {detection}")
                            continue
                    
                    all_action_detections = normalized_detections
                    log(f"✅ Normalized {len(all_action_detections)} detections to 5-element format")

                    # 1️⃣ Group consecutive actions chronologically - GROUP EACH ACTION TYPE SEPARATELY
                    sequences_by_action = defaultdict(list)

                    # First, separate actions by type
                    for timestamp, frame_id, action_id, score, action_name in all_action_detections:
                        sequences_by_action[action_name].append((timestamp, frame_id, action_id, score, action_name))

                    # Now group each action type independently
                    grouped_by_action = {}
                    for action_name, action_list in sequences_by_action.items():
                        grouped_by_action[action_name] = group_consecutive_adaptive(
                            action_list, 
                            max_gap=1.3, 
                            jump_threshold=0.01
                        )
                        log(f"DEBUG: {action_name}: {len(action_list)} detections → {len(grouped_by_action[action_name])} sequences")

                    # 2️⃣ Select best sequences FROM EACH action with per-action quota
                    MAX_ACTION_DURATION = target_duration * 3
                    selected_sequences = []

                    # Calculate quota per action (distribute duration fairly)
                    num_actions = len(grouped_by_action)
                    quota_per_action = MAX_ACTION_DURATION / num_actions if num_actions > 0 else 0

                    log(f"DEBUG: Allocating {quota_per_action:.1f}s per action type ({num_actions} types)")

                    # Select best sequences from EACH action independently
                    for action_name, action_sequences in grouped_by_action.items():
                        # Sort this action's sequences by confidence
                        sorted_action_seqs = sorted(action_sequences, key=lambda x: x[3], reverse=True)
                        
                        action_duration = 0
                        for sequence in sorted_action_seqs:
                            start_time, end_time, duration, confidence, action_name = sequence
                            
                            # Stop when this action hits its quota
                            if action_duration >= quota_per_action:
                                break
                            
                            selected_sequences.append(sequence)
                            action_duration += duration
                            
                            log(f"DEBUG: Selected {action_name} at {seconds_to_mmss(start_time)}-{seconds_to_mmss(end_time)} "
                                f"({duration:.1f}s, conf: {confidence:.3f}) - Action total: {action_duration:.1f}s/{quota_per_action:.1f}s")

                    log(f"\nDEBUG: Selected {len(selected_sequences)} sequences from {num_actions} action types")

                    # 3️⃣ Convert back to individual action format for pipeline compatibility
                    action_detections = []
                    for start_time, end_time, duration, confidence, action_name in selected_sequences:
                        # Find best detection in this group
                        detections_in_group = [
                            det for det in all_action_detections
                            if det[4] == action_name and start_time <= det[0] <= end_time
                        ]
                        if detections_in_group:
                            best_detection = max(detections_in_group, key=lambda a: a[3])  # highest confidence
                            action_detections.append(best_detection)

                    # Calculate total duration from selected sequences
                    total_duration = sum(duration for _, _, duration, _, _ in selected_sequences)

                    # Sort chronologically for pipeline
                    action_detections = sorted(action_detections, key=lambda x: x[0])
                    log(f"✅ Action recognition: {len(action_detections)} action sequences selected (total duration: {total_duration:.1f}s)")

            except Exception as e:
                log(f"⚠ Action recognition failed: {e}")
                import traceback
                log(f"Full error: {traceback.format_exc()}")
                action_detections = []
        elif using_cache:
            log("ℹ️ Using cached action detections")
            # action_detections already loaded from cache - ensure it's in 5-element format
            if action_detections and len(action_detections) > 0:
                first_det = action_detections[0]
                if len(first_det) == 6:
                    # Convert from 6-element to 5-element format
                    action_detections = [
                        (timestamp, frame_id, action_id, score, action_name)
                        for timestamp, frame_id, action_id, score, action_name, _ in action_detections
                    ]
                    log(f"✅ Converted cached detections from 6-element to 5-element format")
        elif not interesting_actions:
            log("ℹ️ No interesting actions specified, skipping action recognition")
            action_detections = []

        # ========== SAVE TO CACHE IF NOT USING CACHE ==========
        if not using_cache and use_cache and not (cancel_flag and cancel_flag.is_set()):
            try:
                # Determine if we should cache only keyword segments
                keyword_segments_only = bool(SEARCH_KEYWORDS and USE_TRANSCRIPT)
                
                # Collect analysis data with keyword filtering if needed
                analysis_data = collect_analysis_data(
                    video_path=processed_video_path,
                    video_duration=float(video_duration),  # Ensure float
                    fps=float(fps),  # Ensure float
                    transcript_segments=transcript_segments,
                    object_detections=object_detections,
                    action_detections=action_detections,
                    scenes=scenes,
                    motion_events=[float(t) for t in motion_events],  # Convert numpy floats
                    motion_peaks=[float(t) for t in motion_peaks],  # Convert numpy floats
                    audio_peaks=[float(t) for t in audio_peaks],  # Convert numpy floats
                    source_lang=TRANSCRIPT_SOURCE_LANG,
                    waveform_data=waveform_data,
                    keyword_segments_only=keyword_segments_only,
                    search_keywords=SEARCH_KEYWORDS if keyword_segments_only else None,
                    keyword_matches=keyword_matches
                )
                
                # Add analysis parameters for future validation
                analysis_data["analysis_parameters"] = analysis_params
                
                # Save to cache with signature-based naming
                cache = VideoAnalysisCache(cache_dir=gui_config.get("cache_dir", "./cache"))
                cache.save(processed_video_path, analysis_data, params=analysis_params)
                
                if keyword_segments_only:
                    log(f"✅ Analysis results cached (keyword-filtered: {len(analysis_data['transcript']['segments'])} segments, language: {TRANSCRIPT_SOURCE_LANG})")
                else:
                    log(f"✅ Analysis results cached (full transcript: {len(analysis_data['transcript']['segments'])} segments, language: {TRANSCRIPT_SOURCE_LANG})")
                
            except Exception as e:
                log(f"⚠️ Failed to save cache: {e}")
                import traceback
                log(f"Full error: {traceback.format_exc()}")
        # ========== END CACHE SAVE ==========

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
        keyword_set = set()
        if keyword_matches:
            for match in keyword_matches:
                main_seg = match["main_segment"]
                start_sec = int(main_seg["start"])
                end_sec = int(main_seg["end"])
                for sec in range(start_sec, end_sec + 1):
                    keyword_set.add(sec)

        
        # Fill scores using the detected signals
        for start, end in scenes:
            idx = int(round(start))
            if 0 <= idx < len(score):
                scene_score[idx] += SCENE_POINTS

        for t in motion_events:
            idx = int(round(t))
            if 0 <= idx < len(score):
                motion_event_score[idx] += MOTION_EVENT_POINTS

        for t in motion_peaks:
            idx = int(round(t))
            if 0 <= idx < len(score):
                motion_peak_score[idx] += MOTION_PEAK_POINTS

        for t in audio_peaks:
            idx = int(round(t))
            if 0 <= idx < len(score):
                audio_score[idx] += AUDIO_PEAK_POINTS

        for sec in keyword_set:
            if 0 <= sec < len(keyword_score):
                keyword_score[sec] += KEYWORD_POINTS

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

        # Calculate confidence percentiles PER ACTION TYPE
        action_type_confidences = defaultdict(list)
        for sec, actions in detections_by_sec.items():
            for action_name, confidence in actions:
                action_type_confidences[action_name].append(confidence)

        # Calculate percentiles for each action type
        action_type_percentiles = {}
        for action_name, confidences in action_type_confidences.items():
            if len(confidences) > 0:
                action_type_percentiles[action_name] = {
                    '50th': np.percentile(confidences, 50),
                    '90th': np.percentile(confidences, 90)
                }
                log(f"📊 {action_name} confidence stats: 50th={action_type_percentiles[action_name]['50th']:.2f}, 90th={action_type_percentiles[action_name]['90th']:.2f}")

        # Now score each second with action-type-specific percentiles
        for sec, actions in detections_by_sec.items():
            if sec < len(action_score):
                if not actions_require_objects or any(abs(obj_sec - sec) <= OBJECT_TOLERANCE for obj_sec in object_detections):
                    # Find the HIGHEST confidence action in this second
                    max_confidence = 0
                    best_action_name = None
                    
                    for action_name, confidence in actions:
                        if confidence > max_confidence:
                            max_confidence = confidence
                            best_action_name = action_name
                    
                    # Score ONLY ONCE per second using the best action
                    if best_action_name and max_confidence > 0:
                        percentiles = action_type_percentiles.get(best_action_name, {})
                        confidence_90th = percentiles.get('90th', 0)
                        confidence_50th = percentiles.get('50th', 0)
                        
                        if max_confidence >= confidence_90th:
                            action_score[sec] += ACTION_POINTS * 1.5
                        elif max_confidence >= confidence_50th:
                            action_score[sec] += ACTION_POINTS
                        else:
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

        # ========== SAVE HIGHLIGHT SEGMENTS TO CACHE ==========
        if segments and use_cache and not (cancel_flag and cancel_flag.is_set()):
            try:
                # Prepare parameters for cache
                highlight_parameters = {
                    'max_duration': MAX_DURATION,
                    'exact_duration': EXACT_DURATION if EXACT_DURATION else None,
                    'clip_time': CLIP_TIME,
                    'highlight_objects': highlight_objects,
                    'interesting_actions': interesting_actions,
                    'scene_points': SCENE_POINTS,
                    'motion_event_points': MOTION_EVENT_POINTS,
                    'motion_peak_points': MOTION_PEAK_POINTS,
                    'audio_peak_points': AUDIO_PEAK_POINTS,
                    'keyword_points': KEYWORD_POINTS,
                    'object_points': OBJECT_POINTS,
                    'action_points': ACTION_POINTS
                }
                
                # Create segments metadata with scores - CONVERT NUMPY TYPES TO PYTHON NATIVE
                segments_metadata = []
                for start, end in segments:
                    duration = end - start
                    
                    # Calculate average score in this segment - CONVERT to Python float
                    avg_score = 0.0
                    if start < len(score) and end < len(score):
                        segment_indices = range(int(start), min(int(end) + 1, len(score)))
                        if segment_indices:
                            # Explicitly convert numpy float to Python float
                            avg_score = float(np.mean([score[i] for i in segment_indices]))
                    
                    # Determine primary reason
                    primary_reason = "multiple_signals"
                    if start in object_detections:
                        primary_reason = "objects"
                    elif start in detections_by_sec:
                        primary_reason = "actions"
                    elif start in motion_peaks_set:
                        primary_reason = "motion_peaks"
                    elif start in audio_set:
                        primary_reason = "audio_peaks"
                    
                    # Make sure all values are Python native types
                    segments_metadata.append({
                        'score': float(avg_score) if avg_score != 0 else 0.0,
                        'signals': {
                            'objects': 1.0 if start in object_detections else 0.0,
                            'actions': 1.0 if start in detections_by_sec else 0.0,
                            'motion': 1.0 if start in motion_peaks_set else 0.0,
                            'audio': 1.0 if start in audio_set else 0.0
                        },
                        'primary_reason': str(primary_reason)
                    })
                
                # Convert score_info values to Python native types
                score_info_python = {
                    'total_score': float(np.sum(score)),
                    'max_score': float(np.max(score)),
                    'avg_score': float(np.mean(score))
                }
                
                # Save to highlight cache
                cache = VideoAnalysisCache(cache_dir=gui_config.get("cache_dir", "./cache"))
                success = cache.save_highlight_segments(
                    processed_video_path,
                    highlight_parameters,
                    segments,
                    segments_metadata,
                    score_info_python,  # Use the converted version
                    analysis_params=analysis_params
                )
                
                if success:
                    log(f"✅ Saved {len(segments)} highlight segments to cache")
                else:
                    log("⚠️ Failed to save highlight segments to cache")
                    
            except Exception as e:
                log(f"⚠️ Error saving highlight cache: {e}")
                import traceback
                log(f"Full error: {traceback.format_exc()}")
        # ========== END HIGHLIGHT CACHE SAVE ==========


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
                cut_video(processed_video_path, segments[0][0], segments[0][1], OUTPUT_FILE)
            else:
                temp_clips = []
                # Get the directory of the output file to save temp clips in the same location
                output_dir = os.path.dirname(OUTPUT_FILE)
                video_base_name = os.path.splitext(os.path.basename(processed_video_path))[0]
                
                # Sanitize the base name to avoid issues with special characters
                import re
                video_base_name = re.sub(r"['\"]", "", video_base_name)
                video_base_name = re.sub(r"[@#$%^&*()]", "_", video_base_name)
                
                for i, (s, e) in enumerate(segments):
                    check_cancellation(cancel_flag, log, f"video cutting clip {i+1}")
                    # Include the directory path for temp files
                    temp_name = os.path.join(output_dir, f"{video_base_name}_temp_clip_{i}.mp4")
                    log(f"  Creating temp clip: {temp_name}")
                    cut_video(processed_video_path, s, e, temp_name)
                    
                    # Verify the file was created
                    if not os.path.exists(temp_name):
                        raise Exception(f"Failed to create temp clip: {temp_name}")
                    
                    temp_clips.append(temp_name)
                    # Update progress for each clip
                    progress.update_progress(90 + (i+1) * 5 // len(segments), 100, "Pipeline", f"Cut clip {i+1}/{len(segments)}")
                
                check_cancellation(cancel_flag, log, "video concatenation")
                
                concat_file = os.path.join(output_dir, "concat_list.txt")
                log(f"📝 Writing concat file: {concat_file}")
                with open(concat_file, "w", encoding='utf-8') as f:
                    for t in temp_clips:
                        # Use absolute path and convert to forward slashes
                        abs_path = os.path.abspath(t).replace('\\', '/')
                        f.write(f"file '{abs_path}'\n")
                
                # DEBUG: Print concat file contents
                log("📋 Concat file contents:")
                with open(concat_file, "r", encoding='utf-8') as f:
                    log(f.read())
                
                # Normalize concat file path
                concat_file_normalized = concat_file.replace('\\', '/')
                
                # Sanitize OUTPUT_FILE name too
                output_filename = os.path.basename(OUTPUT_FILE)
                output_filename_clean = re.sub(r"['\"]", "", output_filename)
                output_filename_clean = re.sub(r"[@#$%^&*()]", "_", output_filename_clean)
                OUTPUT_FILE_CLEAN = os.path.join(output_dir, output_filename_clean)
                
                log(f"🎬 Running FFmpeg concatenation to: {OUTPUT_FILE_CLEAN}")
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                                "-i", concat_file_normalized, "-c", "copy", OUTPUT_FILE_CLEAN], check=True)
                
                # Update OUTPUT_FILE to the cleaned version
                OUTPUT_FILE = OUTPUT_FILE_CLEAN
                
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

        # Clean up temporary trimmed video if it was created
        if temp_trimmed_video and os.path.exists(temp_trimmed_video):
            try:
                os.remove(temp_trimmed_video)
                log(f"🧹 Cleaned up temporary trimmed video")
            except Exception as e:
                log(f"⚠️ Could not remove temporary file: {e}")

        # ========== TIMELINE VISUALIZATION ==========
        if gui_config.get("create_timeline_viewer", False):
            try:
                from signal_timeline_viewer import show_timeline_viewer
                log("🎨 Launching Signal Timeline Viewer...")
                
                # Create analysis_data if not already created for cache
                if 'analysis_data' not in locals() or analysis_data is None:
                    analysis_data = collect_analysis_data(
                        video_path=processed_video_path,
                        video_duration=video_duration,
                        fps=fps,
                        transcript_segments=transcript_segments,
                        object_detections=object_detections,
                        action_detections=action_detections,
                        scenes=scenes,
                        motion_events=motion_events,
                        motion_peaks=motion_peaks,
                        audio_peaks=audio_peaks,
                        source_lang=SOURCE_LANG,
                        waveform_data=waveform_data
                    )
                
                # Launch in separate thread/process so it doesn't block
                import threading
                timeline_thread = threading.Thread(
                    target=show_timeline_viewer,
                    args=(processed_video_path, analysis_data),
                    daemon=True
                )
                timeline_thread.start()
            except Exception as e:
                log(f"⚠️ Timeline viewer failed: {e}")
        # ============================================

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