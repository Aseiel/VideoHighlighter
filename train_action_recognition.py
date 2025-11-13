import os
import glob
import cv2
import json
import random
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from openvino.runtime import Core
from tqdm import tqdm
from collections import deque

# =============================
# How to Use
# Put short 5-10s videos to dataset/train/{action_name} and dataset/val/{action_name}
# Usually You should put minimum 100+ videos for train and 20+ for val
# =============================

# =============================
# Smoothed ROI Detection
# =============================
class SmoothedROIDetector:
    """Temporal smoothing for ROI detection to reduce jitter"""
    def __init__(self, window_size=5, alpha=0.3):
        self.window_size = window_size
        self.alpha = alpha
        self.roi_history = deque(maxlen=window_size)
        self.smoothed_roi = None
        
    def update(self, current_roi):
        if current_roi is None:
            if self.smoothed_roi is not None:
                return tuple(self.smoothed_roi.astype(int))
            return None
        
        self.roi_history.append(current_roi)
        
        if len(self.roi_history) == 0:
            return None
            
        if self.smoothed_roi is None:
            self.smoothed_roi = np.array(current_roi, dtype=np.float32)
        else:
            current = np.array(current_roi, dtype=np.float32)
            self.smoothed_roi = self.alpha * current + (1 - self.alpha) * self.smoothed_roi
        
        return tuple(self.smoothed_roi.astype(int))
    
    def reset(self):
        self.roi_history.clear()
        self.smoothed_roi = None

# =============================
# Pose Estimation for Spatial Guidance
# =============================
class PoseExtractor:
    """Extract YOLOv11 pose keypoints to guide spatial cropping"""
    def __init__(self, model_name="yolo11n-pose.pt", conf_threshold=0.3):
        """
        Initialize pose extractor for spatial guidance
        Args:
            model_name: YOLOv11 pose model (n/s/m/l/x variants)
            conf_threshold: Confidence threshold for pose detection
        """
        print(f"🦴 Loading pose estimation model: {model_name}")
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        # COCO format: 17 keypoints
        self.num_keypoints = 17
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
    def get_action_region_from_poses(self, frame, person_boxes):
        """
        Use pose keypoints to identify WHERE the action is happening
        Focuses on torso/hip/pelvis region for intimate actions
        
        Args:
            frame: RGB frame (H, W, 3)
            person_boxes: List of (x1, y1, x2, y2) person bounding boxes
            
        Returns:
            action_roi: (x1, y1, x2, y2) focused on action region
        """
        h, w = frame.shape[:2]
        
        # Run pose detection on full frame
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        
        if len(results) == 0 or results[0].keypoints is None:
            # Fallback: use merged person boxes
            return self._merge_boxes(person_boxes)
        
        all_keypoints = results[0].keypoints.data.cpu().numpy()
        
        if len(all_keypoints) == 0:
            return self._merge_boxes(person_boxes)
        
        # Define "action region" keypoints (focus on torso/hips/pelvis)
        # COCO format indices:
        # 5: left_shoulder, 6: right_shoulder
        # 11: left_hip, 12: right_hip
        # 13: left_knee, 14: right_knee
        action_keypoint_indices = [5, 6, 11, 12, 13, 14]
        
        action_points = []
        
        # Collect all relevant keypoints from all detected people
        for kpts in all_keypoints:
            for idx in action_keypoint_indices:
                if kpts[idx, 2] > 0.3:  # confidence > 0.3
                    action_points.append(kpts[idx, :2])
        
        if len(action_points) == 0:
            return self._merge_boxes(person_boxes)
        
        action_points = np.array(action_points)
        
        # Find bounding box around action points
        x_min = np.min(action_points[:, 0])
        y_min = np.min(action_points[:, 1])
        x_max = np.max(action_points[:, 0])
        y_max = np.max(action_points[:, 1])
        
        # Add padding (30% extra on each side to capture context)
        width = x_max - x_min
        height = y_max - y_min
        padding_x = width * 0.3
        padding_y = height * 0.3
        
        x1 = max(0, int(x_min - padding_x))
        y1 = max(0, int(y_min - padding_y))
        x2 = min(w, int(x_max + padding_x))
        y2 = min(h, int(y_max + padding_y))
        
        return (x1, y1, x2, y2)
    
    def _merge_boxes(self, boxes):
        """Merge multiple bounding boxes into one"""
        if len(boxes) == 0:
            return None
        if len(boxes) == 1:
            return boxes[0]
        
        x1_min = min(b[0] for b in boxes)
        y1_min = min(b[1] for b in boxes)
        x2_max = max(b[2] for b in boxes)
        y2_max = max(b[3] for b in boxes)
        
        return (x1_min, y1_min, x2_max, y2_max)
    
    def get_all_keypoints_for_visualization(self, frame):
        """
        Get all detected keypoints for visualization purposes
        
        Returns:
            List of keypoint arrays (each is 17x3: x, y, conf)
        """
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        
        if len(results) == 0 or results[0].keypoints is None:
            return []
        
        all_keypoints = results[0].keypoints.data.cpu().numpy()
        return [kpts for kpts in all_keypoints if np.sum(kpts[:, 2] > 0.3) >= 5]

# =============================
# Person Detection with Tracking
# =============================
yolo_people = YOLO("yolo11n.pt")

class PersonTracker:
    """Simple IoU-based person tracker to maintain consistent IDs"""
    def __init__(self, iou_threshold=0.3, max_lost_frames=10):
        self.tracks = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_lost_frames = max_lost_frames
    
    def _compute_iou(self, box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def update(self, detected_boxes):
        if len(detected_boxes) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['lost_frames'] += 1
                if self.tracks[track_id]['lost_frames'] > self.max_lost_frames:
                    del self.tracks[track_id]
            return []
        
        matched_tracks = set()
        matched_detections = set()
        
        for det_idx, det_box in enumerate(detected_boxes):
            best_iou = 0
            best_track_id = None
            
            for track_id, track_data in self.tracks.items():
                if track_id in matched_tracks:
                    continue
                
                iou = self._compute_iou(det_box, track_data['box'])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_track_id = track_id
            
            if best_track_id is not None:
                self.tracks[best_track_id]['box'] = det_box
                self.tracks[best_track_id]['lost_frames'] = 0
                matched_tracks.add(best_track_id)
                matched_detections.add(det_idx)
            else:
                new_track_id = self.next_id
                self.next_id += 1
                self.tracks[new_track_id] = {'box': det_box, 'lost_frames': 0}
                matched_detections.add(det_idx)
        
        for track_id in list(self.tracks.keys()):
            if track_id not in matched_tracks:
                self.tracks[track_id]['lost_frames'] += 1
                if self.tracks[track_id]['lost_frames'] > self.max_lost_frames:
                    del self.tracks[track_id]
        
        sorted_tracks = sorted(self.tracks.items(), key=lambda x: x[0])
        
        return [(track_id, data['box']) for track_id, data in sorted_tracks]
    
    def reset(self):
        self.tracks = {}
        self.next_id = 0

def detect_all_people(frame, detector, tracker=None):
    """Detect all persons with optional tracking"""
    result = detector.predict(frame, conf=0.40, classes=[0], verbose=False)
    boxes = []

    for r in result:
        for b in r.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            score = float(b.conf)
            boxes.append((score, (x1, y1, x2, y2)))

    boxes = sorted(boxes, key=lambda x: x[0], reverse=True)
    detected_boxes = [b for _, b in boxes]
    
    if tracker is None:
        return detected_boxes
    else:
        tracked = tracker.update(detected_boxes)
        return tracked

def merge_boxes(boxes):
    if len(boxes) == 0:
        return None
    if len(boxes) == 1:
        return boxes[0]

    x1_min = min(b[0] for b in boxes)
    y1_min = min(b[1] for b in boxes)
    x2_max = max(b[2] for b in boxes)
    y2_max = max(b[3] for b in boxes)
    
    return (x1_min, y1_min, x2_max, y2_max)

def crop_roi(frame, roi, output_size):
    """Crop frame to ROI at high resolution, then resize to output_size"""
    if roi is None:
        h, w, _ = frame.shape
        th, tw = output_size
        y = max(0, h // 2 - th // 2)
        x = max(0, w // 2 - tw // 2)
        crop = frame[y:y+th, x:x+tw]
        return cv2.resize(crop, output_size)

    x1, y1, x2, y2 = roi
    h, w, _ = frame.shape
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        h, w, _ = frame.shape
        th, tw = output_size
        y = max(0, h // 2 - th // 2)
        x = max(0, w // 2 - tw // 2)
        crop = frame[y:y+th, x:x+tw]
        return cv2.resize(crop, output_size)

    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, output_size)

# =============================
# Improved Frame Sampling for Static Actions
# =============================
def improved_frame_sampling(video_path, sequence_length=16, sampling_strategy="temporal_stride"):
    """
    Multiple sampling strategies for better temporal understanding
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Choose sampling strategy based on video characteristics
    if sampling_strategy == "temporal_stride":
        # For static actions: sample every 4-6 frames to see subtle changes
        stride = max(4, total_frames // (sequence_length * 2))
        indices = np.linspace(0, total_frames - 1, sequence_length * stride)[::stride][:sequence_length]
        indices = indices.astype(int)
        
    elif sampling_strategy == "three_segment":
        # Divide video into beginning, middle, end - good for action progression
        segment_frames = sequence_length // 3
        remaining = sequence_length % 3
        
        # Beginning segment
        start_indices = np.linspace(0, total_frames // 3, segment_frames + remaining).astype(int)
        # Middle segment  
        mid_indices = np.linspace(total_frames // 3, 2 * total_frames // 3, segment_frames).astype(int)
        # End segment
        end_indices = np.linspace(2 * total_frames // 3, total_frames - 1, segment_frames).astype(int)
        
        indices = np.concatenate([start_indices, mid_indices, end_indices])
        
    elif sampling_strategy == "adaptive_motion":
        # Sample more frames during high-motion periods
        indices = adaptive_motion_sampling(cap, total_frames, sequence_length)
    
    else:  # uniform
        indices = np.linspace(0, total_frames - 1, sequence_length).astype(int)
    
    cap.release()
    return indices

def adaptive_motion_sampling(cap, total_frames, sequence_length):
    """Sample more frames during periods of high motion"""
    # Read sample frames to detect motion
    sample_interval = max(1, total_frames // 50)
    frame_diff = []
    prev_frame = None
    
    for i in range(0, total_frames, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray)
            frame_diff.append((i, diff.mean()))
        prev_frame = gray
    
    if not frame_diff:
        return np.linspace(0, total_frames - 1, sequence_length).astype(int)
    
    # Find high-motion regions
    frame_diff = np.array(frame_diff)
    motion_threshold = np.percentile(frame_diff[:, 1], 70)
    high_motion_frames = frame_diff[frame_diff[:, 1] > motion_threshold, 0]
    
    # Sample strategically: more frames in high-motion regions
    if len(high_motion_frames) > sequence_length // 2:
        # If lots of motion, sample evenly from high-motion frames
        high_motion_samples = np.random.choice(
            high_motion_frames, 
            size=min(sequence_length // 2, len(high_motion_frames)), 
            replace=False
        )
        remaining_samples = sequence_length - len(high_motion_samples)
        other_samples = np.linspace(0, total_frames - 1, remaining_samples * 2)[::2].astype(int)
        indices = np.concatenate([high_motion_samples, other_samples])
    else:
        # Default to temporal stride
        stride = max(4, total_frames // (sequence_length * 2))
        indices = np.linspace(0, total_frames - 1, sequence_length * stride)[::stride][:sequence_length].astype(int)
    
    return np.sort(indices)

# =============================
# Video Sample Visualization
# =============================
def visualize_training_sample(video_path, label, pose_extractor, output_path="training_sample.mp4", 
                              sample_rate=5):
    """Creates a video sample visualization with bounding boxes, pose keypoints, and action ROI"""
    print(f"\n🎬 Creating training sample visualization for: {video_path}")
    print(f"   Action label: {label}")
    print(f"   Detection sample rate: every {sample_rate} frame(s)")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return False
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps == 0:
        fps = 30
    
    print(f"   Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Try codecs
    codecs = [('mp4v', '.mp4'), ('avc1', '.mp4'), ('XVID', '.avi'), ('MJPG', '.avi')]
    video_writer = None
    
    for codec, ext in codecs:
        try:
            output_path_with_ext = output_path.replace('.mp4', ext).replace('.avi', ext)
            fourcc = cv2.VideoWriter_fourcc(*codec)
            video_writer = cv2.VideoWriter(output_path_with_ext, fourcc, fps, (width, height + 60))
            if video_writer.isOpened():
                output_path = output_path_with_ext
                print(f"   Using codec: {codec}")
                break
            else:
                video_writer = None
        except:
            continue
    
    if video_writer is None:
        print("❌ Could not initialize video writer")
        cap.release()
        return False
    
    frame_count = 0
    successful_frames = 0
    pbar = tqdm(total=total_frames, desc="Creating visualization")
    
    person_tracker = PersonTracker(iou_threshold=0.3, max_lost_frames=10)
    roi_smoother = SmoothedROIDetector(window_size=5, alpha=0.3)
    
    track_colors = {}
    color_palette = [(0, 255, 0), (255, 0, 255), (255, 255, 0), (0, 255, 255), 
                     (255, 128, 0), (128, 255, 0), (0, 128, 255), (255, 0, 128)]
    
    last_tracked_people = []
    last_action_roi = None
    last_poses = []
    
    # Pose visualization connections
    pose_connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Face
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Torso
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        try:
            display_frame = frame.copy()
            
            if frame_count % sample_rate == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Detect and track people
                last_tracked_people = detect_all_people(frame_rgb, yolo_people, person_tracker)
                
                # Extract boxes
                if last_tracked_people and isinstance(last_tracked_people[0], tuple):
                    boxes_only = [box for _, box in last_tracked_people]
                else:
                    boxes_only = last_tracked_people
                
                # Get ACTION-FOCUSED ROI using pose
                if pose_extractor is not None and len(boxes_only) > 0:
                    action_roi = pose_extractor.get_action_region_from_poses(frame_rgb, boxes_only)
                else:
                    action_roi = merge_boxes(boxes_only)
                
                # Smooth the action ROI
                last_action_roi = roi_smoother.update(action_roi)
                
                # Get keypoints for visualization
                if pose_extractor is not None:
                    last_poses = pose_extractor.get_all_keypoints_for_visualization(frame_rgb)
            
            # Draw tracked person boxes (GREEN)
            for i, item in enumerate(last_tracked_people):
                if isinstance(item, tuple):
                    track_id, (x1, y1, x2, y2) = item
                    if track_id not in track_colors:
                        track_colors[track_id] = color_palette[len(track_colors) % len(color_palette)]
                    color = track_colors[track_id]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, f"Person {track_id}", (x1, max(20, y1-10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    x1, y1, x2, y2 = item
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw ACTION ROI (BLUE - thicker)
            if last_action_roi:
                x1, y1, x2, y2 = last_action_roi
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 4)
                cv2.putText(display_frame, "ACTION CROP", (x1, max(45, y1-40)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)
            
            # Draw pose keypoints and skeleton
            for pose_kpts in last_poses:
                # Draw skeleton connections (CYAN)
                for conn in pose_connections:
                    pt1_idx, pt2_idx = conn
                    if pose_kpts[pt1_idx, 2] > 0.3 and pose_kpts[pt2_idx, 2] > 0.3:
                        pt1 = (int(pose_kpts[pt1_idx, 0]), int(pose_kpts[pt1_idx, 1]))
                        pt2 = (int(pose_kpts[pt2_idx, 0]), int(pose_kpts[pt2_idx, 1]))
                        cv2.line(display_frame, pt1, pt2, (0, 255, 255), 2)
                
                # Draw keypoints (RED)
                for kpt in pose_kpts:
                    if kpt[2] > 0.3:
                        cv2.circle(display_frame, (int(kpt[0]), int(kpt[1])), 4, (0, 0, 255), -1)
            
            # Create label area
            label_area = np.zeros((60, width, 3), dtype=np.uint8)
            cv2.putText(label_area, f"ACTION: {label}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(label_area, f"Frame: {frame_count}/{total_frames}", (width-250, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            num_people = len(last_tracked_people)
            cv2.putText(label_area, f"People: {num_people} | Poses: {len(last_poses)}", (width-550, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            combined_frame = np.vstack([label_area, display_frame])
            video_writer.write(combined_frame)
            successful_frames += 1
            
        except Exception as e:
            print(f"⚠️  Error processing frame {frame_count}: {e}")
        
        frame_count += 1
        pbar.update(1)
    
    cap.release()
    video_writer.release()
    pbar.close()
    
    success_rate = (successful_frames / frame_count) * 100 if frame_count > 0 else 0
    print(f"✅ Visualization: {successful_frames}/{frame_count} frames ({success_rate:.1f}%)")
    print(f"✅ Output: {output_path}")
    
    return successful_frames > 0

def create_sample_visualizations(dataset, pose_extractor, num_samples=2):
    """Create visualizations for random samples"""
    print(f"\n📹 Creating {num_samples} sample visualizations...")
    
    if len(dataset.samples) == 0:
        print("❌ No samples found")
        return
    
    selected_indices = random.sample(range(len(dataset.samples)), min(num_samples, len(dataset.samples)))
    
    for i, idx in enumerate(selected_indices):
        video_path, label_idx = dataset.samples[idx]
        label_name = dataset.idx_to_label[label_idx]
        output_filename = f"sample_{i+1}_{label_name.replace(' ', '_')}.mp4"
        visualize_training_sample(
            video_path, 
            label_name,
            pose_extractor,
            output_filename,
            sample_rate=CONFIG.get('visualization_sample_rate', 5)
        )

# =============================
# Configuration
# =============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

CONFIG = {
    "data_path": "dataset",
    "batch_size": 2,
    "base_epochs": 25,
    "base_learning_rate": 1e-4,
    "finetune_learning_rate": 1e-5,
    "max_finetune_epochs": 15,
    "early_stopping_patience": 5,
    "min_delta": 0.001,
    "use_class_weights": True,
    "augmentation_prob": 0.3,  # 30% chance of augmentation per frame
    "sequence_length": 16,
    "crop_size": (224, 224),  # Intel encoder requires 224x224
    "model_save_path": "intel_finetuned_classifier_3d.pth",
    "checkpoint_path": r"D:\movie_highlighter\checkpoints\checkpoint_latest.pth",
    "save_checkpoint_every": 5,
    "checkpoint_dir": "checkpoints",
    "min_train_per_action": 5,
    "min_val_per_action": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "create_visualizations": True,
    "num_visualization_samples": 2,
    "visualization_sample_rate": 5,
    "use_roi_smoothing": True,
    # Pose-guided cropping (NOT feature extraction)
    "use_pose_guided_crop": True,
    "pose_model": "yolo11n-pose.pt",
    "pose_conf_threshold": 0.3,
    # Improved frame sampling for static actions
    "sampling_strategy": "temporal_stride",  # "temporal_stride", "three_segment", "adaptive_motion"
    "default_stride": 4,  # For temporal_stride: sample every 4-6 frames
    "min_stride": 3,
    "max_stride": 8,
}

BASE_DIR = os.getcwd()
ENCODER_XML = os.path.join(BASE_DIR, "models/intel_action/encoder/FP32/action-recognition-0001-encoder.xml")
ENCODER_BIN = os.path.join(BASE_DIR, "models/intel_action/encoder/FP32/action-recognition-0001-encoder.bin")

# =============================
# Video Loading with Improved Frame Sampling
# =============================
def load_video_normalized(path, pose_extractor=None, is_training=True, verbose=False):
    """Load video with improved frame sampling for static actions"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return []

    # Use improved sampling strategy
    sampling_strategy = CONFIG.get("sampling_strategy", "temporal_stride")
    
    if sampling_strategy == "temporal_stride":
        stride = CONFIG.get("default_stride", 4)
        # Adjust stride based on video length
        stride = max(CONFIG.get("min_stride", 3), 
                    min(CONFIG.get("max_stride", 8), 
                        total_frames // (CONFIG["sequence_length"] * 2)))
        
        indices = np.linspace(0, total_frames - 1, CONFIG["sequence_length"] * stride)
        indices = indices[::stride][:CONFIG["sequence_length"]].astype(int)
        
    elif sampling_strategy == "three_segment":
        indices = improved_frame_sampling(path, CONFIG["sequence_length"], "three_segment")
        
    elif sampling_strategy == "adaptive_motion":
        indices = improved_frame_sampling(path, CONFIG["sequence_length"], "adaptive_motion")
        
    else:  # uniform (fallback)
        indices = np.linspace(0, total_frames - 1, CONFIG["sequence_length"]).astype(int)
    
    # Only print sampling info occasionally during training for debugging
    if verbose and random.random() < 0.01:  # Only print 1% of the time
        print(f"   Sampling: {len(indices)} frames with strategy '{sampling_strategy}' (stride: {stride if sampling_strategy == 'temporal_stride' else 'N/A'})")
    
    output_frames = []
    crop_size = CONFIG["crop_size"]
    
    roi_smoother = SmoothedROIDetector(window_size=5, alpha=0.3) if CONFIG["use_roi_smoothing"] else None
    person_tracker = PersonTracker(iou_threshold=0.3, max_lost_frames=5) if CONFIG["use_roi_smoothing"] else None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect and track people
        if person_tracker:
            tracked = detect_all_people(frame, yolo_people, person_tracker)
            if tracked and isinstance(tracked[0], tuple):
                people = [box for _, box in tracked]
            else:
                people = tracked
        else:
            people = detect_all_people(frame, yolo_people)
        
        # Get ACTION-FOCUSED ROI using pose guidance
        if pose_extractor is not None and len(people) > 0:
            roi = pose_extractor.get_action_region_from_poses(frame, people)
        else:
            # Fallback: merge all person boxes
            roi = merge_boxes(people)
        
        # Smooth the ROI
        if roi_smoother:
            roi = roi_smoother.update(roi)
        
        # Crop at HIGH resolution, then resize to 224x224
        # This preserves more detail than cropping directly at 224x224
        frame = crop_roi(frame, roi, crop_size)
        
        # Data augmentation for training only
        if is_training and random.random() < CONFIG.get('augmentation_prob', 0.3):
            # Random brightness adjustment (helps with lighting variations)
            brightness_factor = random.uniform(0.8, 1.2)
            frame = np.clip(frame * brightness_factor, 0, 255).astype(np.uint8)
            
            # Random contrast adjustment (helps model focus on color/texture changes)
            contrast_factor = random.uniform(0.8, 1.2)
            mean = frame.mean(axis=(0, 1), keepdims=True)
            frame = np.clip((frame - mean) * contrast_factor + mean, 0, 255).astype(np.uint8)
            
            # Random horizontal flip (50% chance)
            if random.random() < 0.5:
                frame = np.fliplr(frame)
        
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        mean = np.array(CONFIG["mean"], dtype=np.float32)
        std = np.array(CONFIG["std"], dtype=np.float32)
        frame = (frame - mean) / std
        
        output_frames.append(frame)

    cap.release()
    
    if len(output_frames) == 0:
        return []
    
    # Pad if fewer frames than sequence_length
    if len(output_frames) < CONFIG["sequence_length"]:
        last = output_frames[-1]
        while len(output_frames) < CONFIG["sequence_length"]:
            output_frames.append(last.copy())

    return np.stack(output_frames, axis=0)
# =============================
# Dataset
# =============================
class VideoDataset(Dataset):
    def __init__(self, root, sequence_length=16, pose_extractor=None, is_training=True):
        self.video_samples = []
        self.sequence_length = sequence_length
        self.pose_extractor = pose_extractor
        self.is_training = is_training

        if not os.path.exists(root):
            print(f"Warning: Dataset path {root} does not exist")
            self.labels = []
            self.label_to_idx = {}
            self.idx_to_label = {}
            self.samples = []
            return

        class_folders = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.label_to_idx = {label: idx for idx, label in enumerate(class_folders)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.labels = class_folders
        
        print(f"📊 Detected {len(self.labels)} action classes:")
        for label, idx in self.label_to_idx.items():
            print(f"  {idx}: {label}")

        video_count = 0
        for label in self.labels:
            label_path = os.path.join(root, label)
            video_files = glob.glob(os.path.join(label_path, "*.mp4")) + \
                          glob.glob(os.path.join(label_path, "*.avi")) + \
                          glob.glob(os.path.join(label_path, "*.mov"))
            for video_path in video_files:
                self.video_samples.append((video_path, self.label_to_idx[label]))
                video_count += 1

        print(f"✅ Found {video_count} videos")
        self.samples = self.video_samples

    def __len__(self):
        return len(self.video_samples)

    def __getitem__(self, idx):
        video_path, label = self.video_samples[idx]
        
        # Pass pose_extractor for guided cropping (not feature extraction)
        frames = load_video_normalized(
            video_path, 
            pose_extractor=self.pose_extractor if CONFIG.get('use_pose_guided_crop') else None,
            is_training=self.is_training,
            verbose=False  # Disable verbose output during training
        )
        
        if len(frames) == 0:
            frames = np.zeros((self.sequence_length, CONFIG["crop_size"][0], CONFIG["crop_size"][1], 3), dtype=np.float32)
        
        frames = np.transpose(frames, (0, 3, 1, 2))
        return torch.tensor(frames, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def get_label_mapping(self):
        return self.label_to_idx, self.idx_to_label

# =============================
# Class Weight Computation
# =============================
def compute_class_weights(train_dataset):
    """Compute inverse frequency weights for balanced training"""
    label_counts = {}
    for _, label in train_dataset.video_samples:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    total_samples = len(train_dataset)
    num_classes = len(label_counts)
    
    weights = []
    for class_idx in range(num_classes):
        count = label_counts.get(class_idx, 1)
        weight = total_samples / (num_classes * count)
        weights.append(weight)
    
    print(f"\n⚖️  Class weights computed (inverse frequency):")
    for idx, weight in enumerate(weights):
        class_name = train_dataset.idx_to_label[idx]
        count = label_counts.get(idx, 0)
        print(f"   {class_name}: {count} samples, weight: {weight:.4f}")
    
    return torch.FloatTensor(weights)

def print_class_distribution(dataset, dataset_name="Dataset"):
    """Print class distribution statistics"""
    label_counts = {}
    for _, label in dataset.video_samples:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print(f"\n📊 {dataset_name} class distribution:")
    total = sum(label_counts.values())
    for label_idx in sorted(label_counts.keys()):
        count = label_counts[label_idx]
        percentage = (count / total) * 100
        class_name = dataset.idx_to_label[label_idx]
        print(f"   {class_name}: {count} videos ({percentage:.1f}%)")
    
    counts = list(label_counts.values())
    if len(counts) > 1:
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count
        if imbalance_ratio > 2.0:
            print(f"   ⚠️  Class imbalance detected! Ratio: {imbalance_ratio:.2f}x")
            print(f"   💡 Class weighting is ENABLED to handle this")
        else:
            print(f"   ✅ Classes are relatively balanced (ratio: {imbalance_ratio:.2f}x)")

# =============================
# Intel Feature Extractor
# =============================
class IntelFeatureExtractor:
    def __init__(self, encoder_xml, encoder_bin):
        self.ie = Core()
        self.encoder_model = self.ie.read_model(model=encoder_xml, weights=encoder_bin)
        self.encoder = self.ie.compile_model(self.encoder_model, device_name="CPU")
        
        input_tensor = self.encoder.inputs[0]
        self.input_name = input_tensor.get_any_name()
        self.input_shape = list(input_tensor.get_shape())
        print(f"Encoder input: {self.input_name}, shape: {self.input_shape}")

    def encode(self, frames_batch):
        """
        frames_batch: torch tensor or numpy array of shape (B, T, C, H, W)
        Returns: torch.FloatTensor of encoded features shape (B, T, feat_dim)
        """
        if isinstance(frames_batch, torch.Tensor):
            frames_batch = frames_batch.cpu().numpy()

        B, T, C, H, W = frames_batch.shape
        feats = []
        
        for batch_idx in range(B):
            batch_feats = []
            for time_idx in range(T):
                frame = frames_batch[batch_idx, time_idx]
                frame_batch = np.expand_dims(frame, axis=0)
                frame_batch = self._preprocess_batch(frame_batch)
                
                out = self.encoder([frame_batch])
                
                try:
                    output_node = self.encoder.output(0)
                    feat = out[output_node]
                except Exception:
                    feat = list(out.values())[0]
                
                if feat.ndim > 1:
                    feat = feat.reshape(feat.shape[0], -1)
                batch_feats.append(feat)
            
            batch_feats = np.concatenate(batch_feats, axis=0)
            feats.append(batch_feats)
        
        feats = np.stack(feats, axis=0)
        return torch.tensor(feats, dtype=torch.float32)

    def _preprocess_batch(self, batch_frames):
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        batch_frames = batch_frames * 255.0
        batch_frames = (batch_frames / 255.0 - mean) / std
        return batch_frames.astype(np.float32)

# =============================
# Model
# =============================
class EncoderLSTM(nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=512, num_classes=2, bidirectional=True):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        
        # Adjust output dimension based on bidirectional
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(lstm_output_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the last time step
        return self.fc(out)


# =============================
# Checkpoint Management
# =============================
def save_checkpoint(model, optimizer, epoch, best_val_acc, label_to_idx, idx_to_label, 
                   feature_dim, checkpoint_path, best_val_loss=None):
    """Save training checkpoint"""
    os.makedirs(os.path.dirname(checkpoint_path) if os.path.dirname(checkpoint_path) else '.', exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'best_val_acc': best_val_acc,
        'best_val_loss': best_val_loss if best_val_loss is not None else float('inf'),
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'feature_dim': feature_dim,
        'sequence_length': CONFIG['sequence_length'],
        'num_classes': len(label_to_idx),
        'config': CONFIG.copy()
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path} (epoch {epoch+1})")

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load training checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=CONFIG['device'])
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and checkpoint.get('optimizer_state_dict') is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            print(f"⚠️  Could not load optimizer state: {e}")
    
    print(f"✅ Checkpoint loaded successfully!")
    print(f"   Resuming from epoch {checkpoint['epoch'] + 1}")
    print(f"   Best validation accuracy: {checkpoint.get('best_val_acc', 0):.4f}")
    
    return checkpoint

# =============================
# Training & Validation
# =============================
def validate_classifier(encoder, model, val_loader, device, criterion):
    """Validate model on validation set"""
    model.eval()
    total_correct = 0
    total_samples = 0
    running_loss = 0.0
    
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for frames, labels in val_loader:
            frames, labels = frames.to(device), labels.to(device)
            feats = encoder.encode(frames.cpu()).to(device)
            outputs = model(feats)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(1)
            
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            running_loss += loss.item() * labels.size(0)
            
            for label, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                if label not in class_total:
                    class_total[label] = 0
                    class_correct[label] = 0
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    avg_loss = running_loss / total_samples if total_samples > 0 else float('inf')
    
    per_class_acc = {}
    for label in class_total:
        per_class_acc[label] = class_correct[label] / class_total[label] if class_total[label] > 0 else 0
    
    return avg_loss, accuracy, per_class_acc

def train_classifier(encoder, train_loader, val_loader, num_classes, label_to_idx, idx_to_label):
    device = torch.device(CONFIG["device"])
    
    with torch.no_grad():
        sample_frames, _ = next(iter(train_loader))
        dummy_feats = encoder.encode(sample_frames[0:1].cpu())
        feature_dim = dummy_feats.shape[-1]
    
    print(f"Feature dimension: {feature_dim}")
    
    model = EncoderLSTM(feature_dim=feature_dim, hidden_dim=512, num_classes=num_classes, bidirectional=True).to(device)
    
    if CONFIG.get('use_class_weights', True):
        class_weights = compute_class_weights(train_loader.dataset).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
        print("\n⚠️  Training without class weights")
    
    is_resuming = CONFIG.get('checkpoint_path') and os.path.exists(CONFIG['checkpoint_path'])
    
    if is_resuming:
        lr = CONFIG['finetune_learning_rate']
        print(f"🔄 Resume detected: using finetune LR {lr}")
    else:
        lr = CONFIG['base_learning_rate']
        print(f"🆕 Training from scratch: using base LR {lr}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    start_epoch = 0
    best_val_acc = 0.0
    best_val_loss = float('inf')
    best_model_state = None
    checkpoint = None

    if is_resuming:
        checkpoint = load_checkpoint(CONFIG['checkpoint_path'], model, optimizer)
        if checkpoint:
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        else:
            print("⚠️  Failed to load checkpoint — starting from scratch with finetune LR.")
    
    if is_resuming:
        max_epochs = start_epoch + CONFIG.get('max_finetune_epochs', 15)
        print(f"   Fine-tuning mode: starting at epoch {start_epoch}, will run up to epoch {max_epochs}")
    else:
        max_epochs = CONFIG.get('base_epochs', 25)
        print(f"   Fresh training mode: will run up to epoch {max_epochs}")
    
    patience_counter = 0

    for epoch in range(start_epoch, max_epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
        for frames, labels in pbar:
            frames, labels = frames.to(device), labels.to(device)
            
            with torch.no_grad():
                feats = encoder.encode(frames.cpu())
            feats = feats.to(device)
            
            outputs = model(feats)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            preds = outputs.argmax(1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            running_loss += loss.item() * frames.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{total_correct/total_samples:.4f}'
            })
        
        train_loss = running_loss / total_samples if total_samples > 0 else float('inf')
        train_acc = total_correct / total_samples if total_samples > 0 else 0.0

        print(f"\nEpoch {epoch+1}/{max_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

        val_loss = float('inf')
        val_acc = 0.0
        per_class_acc = {}
        if len(val_loader) > 0:
            val_loss, val_acc, per_class_acc = validate_classifier(encoder, model, val_loader, device, criterion)
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
            # Only print per-class accuracy every 5 epochs to reduce clutter
            if per_class_acc and (epoch + 1) % 5 == 0:
                print(f"  Per-class accuracy:")
                for label_idx in sorted(per_class_acc.keys()):
                    class_name = idx_to_label[label_idx]
                    acc = per_class_acc[label_idx]
                    print(f"    {class_name}: {acc:.4f}")

        improved = val_loss < (best_val_loss - CONFIG.get('min_delta', 0.0))
        if improved:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print("   ⭐ Validation loss improved — saving best model state and resetting patience.")
        else:
            patience_counter += 1
            print(f"   No improvement in validation loss ({patience_counter}/{CONFIG['early_stopping_patience']})")
            if patience_counter >= CONFIG['early_stopping_patience']:
                print("\n🛑 Early stopping triggered — stopping training.")
                break

        if CONFIG.get('save_checkpoint_every') and (epoch + 1) % CONFIG['save_checkpoint_every'] == 0:
            checkpoint_name = f"checkpoint_epoch_{epoch+1}.pth"
            checkpoint_path = os.path.join(CONFIG.get('checkpoint_dir', '.'), checkpoint_name)
            save_checkpoint(model, optimizer, epoch, best_val_acc, 
                          label_to_idx, idx_to_label, feature_dim, checkpoint_path, best_val_loss)

        if CONFIG.get('checkpoint_dir'):
            latest_checkpoint = os.path.join(CONFIG['checkpoint_dir'], 'checkpoint_latest.pth')
            save_checkpoint(model, optimizer, epoch, best_val_acc, 
                          label_to_idx, idx_to_label, feature_dim, latest_checkpoint, best_val_loss)
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Loaded best model (val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})")

    torch.save(model.state_dict(), CONFIG['model_save_path'])
    mapping_path = CONFIG['model_save_path'].replace('.pth', '_mapping.json')
    mapping_data = {
        'label_to_idx': label_to_idx,
        'idx_to_label': idx_to_label,
        'feature_dim': feature_dim,
        'sequence_length': CONFIG['sequence_length'],
        'num_classes': num_classes
    }
    with open(mapping_path, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    return model

# =============================
# Main
# =============================
if __name__ == "__main__":
    set_seed(42)
    
    print("=" * 60)
    print("🎯 ACTION RECOGNITION TRAINING (IMPROVED FRAME SAMPLING)")
    print("=" * 60)
    
    if CONFIG.get('checkpoint_path'):
        print(f"\n📂 Config checkpoint path: {CONFIG['checkpoint_path']}")
    print(f"   Device: {CONFIG['device']}")
    print(f"   Batch size: {CONFIG['batch_size']}")
    print(f"   Pose-guided cropping: {'ENABLED' if CONFIG.get('use_pose_guided_crop') else 'DISABLED'}")
    print(f"   Class weighting: {'ENABLED' if CONFIG.get('use_class_weights') else 'DISABLED'}")
    print(f"   Early stopping patience: {CONFIG['early_stopping_patience']} epochs")
    print(f"   Frame sampling strategy: {CONFIG.get('sampling_strategy', 'temporal_stride')}")
    print(f"   Default stride: {CONFIG.get('default_stride', 4)} frames")
    print("=" * 60)
    
    # Initialize pose extractor for CROPPING (not feature extraction)
    pose_extractor = None
    if CONFIG.get('use_pose_guided_crop', False):
        print("\n🦴 Initializing pose-guided cropping...")
        pose_extractor = PoseExtractor(
            model_name=CONFIG.get('pose_model', 'yolo11n-pose.pt'),
            conf_threshold=CONFIG.get('pose_conf_threshold', 0.3)
        )
        print("   ✅ Pose will guide ACTION REGION detection for better cropping")
    else:
        print("\n⚠️  Pose-guided cropping DISABLED - using full person boxes")
    
    # Load datasets
    print("\n📁 Loading datasets...")
    train_dataset = VideoDataset(
        os.path.join(CONFIG['data_path'], "train"),
        pose_extractor=pose_extractor,
        is_training=True
    )
    val_dataset = VideoDataset(
        os.path.join(CONFIG['data_path'], "val"),
        pose_extractor=pose_extractor,
        is_training=False
    )
    
    print(f"\n📊 Dataset statistics:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Number of classes: {len(train_dataset.labels)}")
    
    print_class_distribution(train_dataset, "Training set")
    if len(val_dataset) > 0:
        print_class_distribution(val_dataset, "Validation set")
    
    # Create sample visualizations BEFORE training
    if CONFIG.get('create_visualizations', False) and pose_extractor is not None:
        create_sample_visualizations(
            train_dataset, 
            pose_extractor,
            num_samples=CONFIG.get('num_visualization_samples', 2)
        )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)
    
    # Initialize encoder
    print(f"\n🔧 Initializing Intel encoder...")
    encoder = IntelFeatureExtractor(ENCODER_XML, ENCODER_BIN)
    
    # Get label mappings
    label_to_idx, idx_to_label = train_dataset.get_label_mapping()
    
    # Train model
    print(f"\n🚀 Starting training...")
    model = train_classifier(encoder, train_loader, val_loader, 
                            len(train_dataset.labels), label_to_idx, idx_to_label)
    
    if model:
        print(f"\n✅ Training complete!")
        print(f"   Final model saved to: {CONFIG['model_save_path']}")
    else:
        print(f"\n❌ Training failed!")
    
    print("=" * 60)