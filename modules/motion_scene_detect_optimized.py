import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import queue
import threading

from modules.video_regions import crop_frame_for_analysis, normalize_analysis_region

def detect_scenes_motion_optimized(video_path,
                               scene_threshold=50.0,
                               motion_threshold=25.0,
                               min_area=500,  # Critical parameter
                               frame_skip=5,
                               downscale_factor=2,
                               spike_factor=1.2,
                               freeze_seconds=4,
                               freeze_factor=0.8,
                               device="xpu",
                               debug=True,
                               batch_size=8,
                               prefetch_frames=16,
                               cancel_flag=None,
                               progress_callback=None,
                               analysis_region=None):  # Added cancellation support
    """
    Hybrid approach combining speed optimizations with accurate motion detection:
    - Fast GPU batching for initial processing
    - Contour-based min_area filtering for accurate motion detection
    - Optimized memory management
    - Cancellation support
    """
    
    # Check for cancellation at start
    if cancel_flag and cancel_flag.is_set():
        return [], [], []
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0  # OpenCV can't always read fps (VR/8K/variable-fps); assume 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    video_duration = (total_frames / fps) if total_frames > 0 else 0.0
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 8)

    # Sampled-frame total for the GUI progress bar. 0 => unknown, which happens a
    # lot on VR/8K/streamed files where OpenCV can't report a frame count; in that
    # case we emit an indeterminate ("busy") update so the UI still shows activity
    # instead of a dead 0%. sampled_done is a monotonic counter of frames analysed.
    sampled_total = max(1, total_frames // frame_skip) if total_frames > 0 else 0
    sampled_done = 0

    def _report_progress():
        if not progress_callback:
            return
        if sampled_total:
            done = min(sampled_done, sampled_total)
            progress_callback(done, sampled_total, "Motion Detection",
                              f"Analyzed {done}/{sampled_total} sampled frames")
        else:
            progress_callback(sampled_done, 0, "Motion Detection",
                              f"Analyzed {sampled_done} sampled frames")
    analysis_region = normalize_analysis_region(analysis_region)
    
    scenes = []
    motion_events = []
    motion_peaks = []
    # Every sampled frame's difference from the one before it, kept rather than
    # thresholded on the spot. Deciding cuts after the pass is what lets the
    # threshold be checked against the video's own distribution when the
    # configured one turns out to be unreachable — see modules/scene_cuts.py.
    # One float per sampled frame: an hour at frame_skip=5 is ~43k of them.
    scene_diffs = []
    
    if device in ["xpu", "cuda"] or device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
    
    frame_queue = queue.Queue(maxsize=prefetch_frames)
    prev_gray_scene = None
    scene_start_sec = 0
    all_motion_data = []
    current_scene_id = 0
    freeze_frames = int(freeze_seconds * fps / frame_skip)
    
    # Flag to signal frame loader to stop
    stop_loading = threading.Event()

    if debug:
        print(f"=== HYBRID OPTIMIZED VIDEO PROCESSING ===")
        print(f"Device: {device}, min_area: {min_area}")
        print(f"Motion threshold: {motion_threshold}, Scene threshold: {scene_threshold}")

    def frame_loader():
        """Async frame loading with CPU downscaling and cancellation support"""
        frame_idx = 0
        ret, frame = cap.read()
        while ret and not stop_loading.is_set():
            # Check for cancellation every 100 frames
            if frame_idx % 100 == 0 and cancel_flag and cancel_flag.is_set():
                break
                
            if frame_idx % frame_skip == 0:
                frame = crop_frame_for_analysis(frame, analysis_region)

                # Downscale on CPU to reduce GPU memory usage
                if downscale_factor > 1:
                    height, width = frame.shape[:2]
                    new_height = height // downscale_factor
                    new_width = width // downscale_factor
                    frame = cv2.resize(frame, (new_width, new_height), 
                                     interpolation=cv2.INTER_AREA)
                
                # Retry until the consumer drains the queue, rather than
                # dropping the frame after a single 1s timeout.
                while not stop_loading.is_set():
                    if cancel_flag and cancel_flag.is_set():
                        break
                    try:
                        frame_queue.put((frame_idx, frame.copy()), timeout=1)
                        break
                    except queue.Full:
                        continue
                if stop_loading.is_set() or (cancel_flag and cancel_flag.is_set()):
                    break
            ret, frame = cap.read()
            frame_idx += 1
        # Sentinel: never block forever here, or this thread outlives the
        # release-guard in finally and the capture leaks every run.
        try:
            frame_queue.put((None, None), timeout=1)
        except queue.Full:
            pass

    def process_frame_batch_hybrid(frames_batch):
        """Fast GPU processing for scene detection with cancellation checks"""
        if not frames_batch:
            return []
        
        # Check for cancellation before processing
        if cancel_flag and cancel_flag.is_set():
            return []
        
        results = []
        
        for frame_idx, frame in frames_batch:
            # Check cancellation for long batches
            if cancel_flag and cancel_flag.is_set():
                break
                
            frame_tensor = torch.from_numpy(frame).float()
            if device != "cpu":
                frame_tensor = frame_tensor.to(device, non_blocking=True)

            # Convert to grayscale on GPU
            gray_weights = torch.tensor([0.114, 0.587, 0.299],
                                      device=device if device != "cpu" else None)
            gray_scene = torch.sum(frame_tensor * gray_weights, dim=2)

            # Keep only the single-channel grayscale plane, not the 3-channel
            # float32 frame. Downstream motion detection only ever diffs the
            # greyscale, but retaining frame_tensor held ~48MB per entry on a
            # 5.3K source — ~9.5GB across the 200-entry buffer, which exhausts
            # the heap mid-run and faults inside torch (0xC0000005).
            gray_np = gray_scene.cpu().numpy()
            del frame_tensor
            results.append((frame_idx, gray_np, gray_scene))

        return results

    def motion_detection_hybrid(frame_tuples):
        """Hybrid approach: GPU for initial processing, CPU for contour filtering with cancellation"""
        if len(frame_tuples) < 2:
            return []
        
        # Check for cancellation
        if cancel_flag and cancel_flag.is_set():
            return []
        
        motion_results = []
        
        # Process in smaller batches to avoid memory issues
        for i in range(0, len(frame_tuples) - 1, min(4, len(frame_tuples) - 1)):
            # Check cancellation at each batch
            if cancel_flag and cancel_flag.is_set():
                break
                
            batch_end = min(i + 4, len(frame_tuples) - 1)
            batch_pairs = frame_tuples[i:batch_end + 1]
            
            if len(batch_pairs) < 2:
                continue
                
            # GPU processing for difference calculation
            prev_frames = []
            curr_frames = []
            
            for j in range(len(batch_pairs) - 1):
                # Check cancellation in inner loop
                if cancel_flag and cancel_flag.is_set():
                    break
                    
                # Already single-channel greyscale (H, W) — the weighted RGB
                # sum happened once in process_frame_batch_hybrid.
                prev_tensor = batch_pairs[j][2]
                curr_tensor = batch_pairs[j + 1][2]

                if device != "cpu":
                    prev_tensor = prev_tensor.to(device, non_blocking=True)
                    curr_tensor = curr_tensor.to(device, non_blocking=True)

                prev_frames.append(prev_tensor)
                curr_frames.append(curr_tensor)

            if not prev_frames or (cancel_flag and cancel_flag.is_set()):
                continue

            # Batch difference on GPU — (N, H, W)
            prev_batch = torch.stack(prev_frames)
            curr_batch = torch.stack(curr_frames)

            gray_diff_batch = torch.abs(prev_batch - curr_batch)

            # Blur operation
            gray_diff_batch = gray_diff_batch.unsqueeze(1)
            blur_batch = F.avg_pool2d(gray_diff_batch, kernel_size=5, stride=1, padding=2)
            blur_batch = blur_batch.squeeze(1)
            
            # Threshold
            motion_mask = blur_batch > motion_threshold
            
            # Move to CPU for contour processing
            motion_mask_np = motion_mask.detach().cpu().numpy().astype(np.uint8) * 255
            
            # Apply min_area filtering with contours (CPU)
            for mask_idx in range(motion_mask_np.shape[0]):
                # Check cancellation during contour processing
                if cancel_flag and cancel_flag.is_set():
                    break
                    
                mask_frame = motion_mask_np[mask_idx]
                
                # Find contours
                contours, _ = cv2.findContours(mask_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Calculate total area of contours above min_area
                total_motion_area = sum(cv2.contourArea(c) for c in contours 
                                      if cv2.contourArea(c) > min_area)
                
                frame_idx = batch_pairs[mask_idx + 1][0]
                motion_results.append((frame_idx, total_motion_area))
            
            # Cleanup GPU memory
            del prev_batch, curr_batch, gray_diff_batch, blur_batch, motion_mask
            if device != "cpu":
                if device == "xpu":
                    torch.xpu.empty_cache()
                else:
                    torch.cuda.empty_cache()
        
        return motion_results

    loader_thread = None

    try:
        # Start async frame loading
        loader_thread = threading.Thread(target=frame_loader, daemon=True)
        loader_thread.start()

        pbar = tqdm(total=total_frames // frame_skip, desc="Hybrid processing")

        # Show the step on the GUI immediately (before the first batch completes).
        _report_progress()

        frame_buffer = []
        processed_frames = []
        
        while not (cancel_flag and cancel_flag.is_set()):
            try:
                frame_data = frame_queue.get(timeout=2)  # Reduced timeout for better responsiveness
                if frame_data[0] is None:
                    break
                    
                frame_buffer.append(frame_data)
                
                if len(frame_buffer) >= batch_size:
                    # Check cancellation before processing batch
                    if cancel_flag and cancel_flag.is_set():
                        break
                        
                    # Fast GPU processing for scene detection
                    batch_results = process_frame_batch_hybrid(frame_buffer)
                    if not batch_results:  # Empty results might indicate cancellation
                        break
                        
                    processed_frames.extend(batch_results)
                    frame_buffer.clear()
                    
                    # Scene detection
                    for frame_idx, gray_scene, frame_tensor in batch_results:
                        if cancel_flag and cancel_flag.is_set():
                            break
                            
                        if prev_gray_scene is not None:
                            scene_diffs.append((frame_idx, float(
                                np.abs(prev_gray_scene - gray_scene).mean())))
                        prev_gray_scene = gray_scene
                    
                    # Check cancellation before motion detection
                    if cancel_flag and cancel_flag.is_set():
                        break
                    
                    # Accurate motion detection with min_area filtering
                    if len(processed_frames) >= 2:
                        motion_results = motion_detection_hybrid(processed_frames[-batch_size-1:])
                        
                        for frame_idx, motion_level in motion_results:
                            if cancel_flag and cancel_flag.is_set():
                                break
                            all_motion_data.append((frame_idx, motion_level, current_scene_id))
                            if motion_level > 0:
                                motion_events.append(frame_idx / fps)
                    
                    # Memory management. Only the last batch_size+1 entries are
                    # ever read (see the slice passed to motion_detection_hybrid
                    # above), so retaining 200 just pinned frames nothing would
                    # look at again — costly on high-resolution sources.
                    keep = batch_size + 1
                    if len(processed_frames) > keep:
                        processed_frames = processed_frames[-keep:]
                    
                    pbar.update(len(batch_results))
                    sampled_done += len(batch_results)
                    _report_progress()

            except queue.Empty:
                # Check if we should continue waiting or if cancelled
                if cancel_flag and cancel_flag.is_set():
                    break
                continue
        
        # Signal frame loader to stop
        stop_loading.set()
        
        # Process remaining frames only if not cancelled
        if frame_buffer and not (cancel_flag and cancel_flag.is_set()):
            batch_results = process_frame_batch_hybrid(frame_buffer)
            processed_frames.extend(batch_results)
            
            # Scene detection for remaining
            for frame_idx, gray_scene, frame_tensor in batch_results:
                if cancel_flag and cancel_flag.is_set():
                    break
                if prev_gray_scene is not None:
                    scene_diffs.append((frame_idx, float(
                        np.abs(prev_gray_scene - gray_scene).mean())))
                prev_gray_scene = gray_scene
            
            # Motion detection for remaining
            if len(processed_frames) >= 2 and not (cancel_flag and cancel_flag.is_set()):
                motion_results = motion_detection_hybrid(processed_frames[-len(batch_results)*2:])
                for frame_idx, motion_level in motion_results:
                    if cancel_flag and cancel_flag.is_set():
                        break
                    all_motion_data.append((frame_idx, motion_level, current_scene_id))
                    if motion_level > 0:
                        motion_events.append(frame_idx / fps)
            
            pbar.update(len(batch_results))
            sampled_done += len(batch_results)
            if sampled_total:
                sampled_done = sampled_total  # final batch → report 100%
            _report_progress()
        
        # Decide the cuts now that the whole distribution is known. Doing it
        # here rather than per frame is what lets an unreachable threshold be
        # noticed at all: a fixed number describes the mastering, not the
        # content, and on graded footage 70 is never reached, which used to
        # come back as one scene covering the film. See modules/scene_cuts.py.
        if not (cancel_flag and cancel_flag.is_set()):
            from modules.scene_cuts import resolve, scenes_from_cuts

            # OpenCV cannot always report a frame count (VR/8K/streamed), so
            # fall back to the last frame actually read rather than trusting a
            # zero that would collapse every scene to nothing.
            if video_duration <= 0 and scene_diffs:
                video_duration = scene_diffs[-1][0] / fps

            diffs = [d for _idx, d in scene_diffs]
            minutes = video_duration / 60.0 if video_duration > 0 else 0.0
            # Shortest shot worth reporting, expressed in samples because that
            # is the lattice the differences live on. Half a second: below that
            # it is a dissolve or the frames either side of one cut, not a shot.
            min_gap = max(1, int(round(0.5 * fps / frame_skip)))
            picked, used_threshold, recalibrated = resolve(
                diffs, minutes, scene_threshold, min_gap=min_gap)
            cut_times = [scene_diffs[i][0] / fps for i in picked]
            scenes = scenes_from_cuts(cut_times, video_duration)

            if debug or recalibrated:
                rate = (len(scenes) / minutes) if minutes else 0.0
                if recalibrated:
                    print(f"🎬 Scene threshold {scene_threshold:.0f} found nothing "
                          f"in {minutes:.0f} min; recalibrated to "
                          f"{used_threshold:.1f} from this video's own frame "
                          f"differences -> {len(scenes)} scenes ({rate:.1f}/min)")
                else:
                    print(f"🎬 Scenes: {len(scenes)} at threshold "
                          f"{used_threshold:.0f} ({rate:.1f}/min)")

        pbar.close()
        
        # Wait for loader thread to finish
        if loader_thread.is_alive():
            loader_thread.join(timeout=2)
        
        # Spike+freeze detection (same as original) - skip if cancelled
        if not (cancel_flag and cancel_flag.is_set()):
            for scene_idx, (start_sec, end_sec) in enumerate(scenes):
                # Check cancellation during spike detection
                if cancel_flag and cancel_flag.is_set():
                    break
                    
                start_frame = int(start_sec * fps)
                end_frame = int(end_sec * fps)
                
                scene_motion = [(frame_idx, motion) for frame_idx, motion, scene_id in all_motion_data 
                               if start_frame <= frame_idx < end_frame]

                if not scene_motion or len(scene_motion) <= freeze_frames:
                    continue

                motion_array = np.array([m for _, m in scene_motion])
                avg_motion = np.mean(motion_array)
                spike_threshold = avg_motion * spike_factor
                freeze_threshold = avg_motion * freeze_factor

                spike_mask = motion_array > spike_threshold
                spike_indices = np.where(spike_mask)[0]
                
                for spike_idx in spike_indices:
                    # Check cancellation in inner loop
                    if cancel_flag and cancel_flag.is_set():
                        break
                        
                    if spike_idx + freeze_frames < len(scene_motion):
                        freeze_window = motion_array[spike_idx+1:spike_idx+1+freeze_frames]
                        if len(freeze_window) == freeze_frames:
                            avg_freeze = np.mean(freeze_window)
                            low_motion_count = np.sum(freeze_window < freeze_threshold)
                            freeze_ratio = low_motion_count / len(freeze_window)
                            
                            if (freeze_ratio >= 0.7 and 
                                avg_freeze < freeze_threshold and 
                                avg_freeze < motion_array[spike_idx] * 0.3):
                                
                                timestamp = scene_motion[spike_idx][0] / fps
                                motion_peaks.append(timestamp)

        if debug and not (cancel_flag and cancel_flag.is_set()):
            print(f"Scenes: {len(scenes)}, Motion events: {len(motion_events)}, Peaks: {len(motion_peaks)}")
        elif debug and cancel_flag and cancel_flag.is_set():
            print(f"Motion detection cancelled - partial results: Scenes: {len(scenes)}, Motion events: {len(motion_events)}, Peaks: {len(motion_peaks)}")

    except Exception as e:
        if debug:
            print(f"Motion detection error: {e}")
    finally:
        # Cleanup
        try:
            stop_loading.set()
            # The loader thread calls cap.read() in a loop; releasing the
            # VideoCapture out from under it frees the decoder while it is
            # mid-read, which faults inside OpenCV (0xC0000005). stop_loading
            # only takes effect between reads, so we must wait for the thread
            # to actually exit before releasing. Drain the queue so a loader
            # blocked in put() can reach its next stop_loading check.
            if loader_thread is not None:
                for _ in range(50):  # ~5s worst case
                    if not loader_thread.is_alive():
                        break
                    try:
                        while True:
                            frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    loader_thread.join(timeout=0.1)
                if loader_thread.is_alive():
                    # Never release a capture a live thread may still read from.
                    print("⚠️ Frame loader did not stop; leaking VideoCapture to avoid a crash")
                    cap = None
            if cap is not None:
                cap.release()
            if device != "cpu":
                if device == "xpu":
                    torch.xpu.empty_cache()
                else:
                    torch.cuda.empty_cache()
        except:
            pass

    return scenes, motion_events, motion_peaks