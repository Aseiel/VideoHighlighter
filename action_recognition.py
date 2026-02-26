import cv2
import numpy as np
from pathlib import Path
from openvino.runtime import Core
import csv
import json
import threading
import queue
import time
import argparse
from collections import Counter, deque
from ultralytics import YOLO
import concurrent.futures
import os
import gc
import re

# =============================
# Optional PyTorch / CUDA imports
# =============================
TORCH_AVAILABLE = False
CUDA_AVAILABLE = False
try:
    import torch
    import torchvision.models.video as video_models
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
    print(f"✓ PyTorch available (version {torch.__version__})")
    if CUDA_AVAILABLE:
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ CUDA not available — R3D will run on CPU (slow)")
except ImportError:
    print("⚠️ PyTorch not installed — R3D/CUDA backend disabled")

# =============================
# Load labels - support Kinetics-400, custom, and R3D models
# =============================
BASE_DIR = Path(__file__).parent.resolve()

CUSTOM_MAPPING_PATH = BASE_DIR / "intel_finetuned_classifier_3d_mapping.json"
KINETICS_LABELS_PATH = BASE_DIR / "kinetics_400_labels.json"
R3D_CUSTOM_MAPPING_PATH = BASE_DIR / "r3d_finetuned_mapping.json"
R3D_CUSTOM_WEIGHTS_PATH = BASE_DIR / "r3d_finetuned.pth"


CUSTOM_LABELS = None
KINETICS_400_LABELS = None
R3D_CUSTOM_LABELS = None
R3D_CUSTOM_META = None


def load_label_mappings():
    global CUSTOM_LABELS, KINETICS_400_LABELS, R3D_CUSTOM_LABELS, R3D_CUSTOM_META

    if CUSTOM_MAPPING_PATH.exists():
        with open(CUSTOM_MAPPING_PATH, "r") as f:
            custom_data = json.load(f)
            CUSTOM_LABELS = custom_data.get('idx_to_label', {})
            CUSTOM_LABELS = {int(k): v for k, v in CUSTOM_LABELS.items()}
        print(f"✓ Loaded custom OpenVINO model labels: {len(CUSTOM_LABELS)} classes")
    else:
        CUSTOM_LABELS = None

    if KINETICS_LABELS_PATH.exists():
        with open(KINETICS_LABELS_PATH, "r") as f:
            KINETICS_400_LABELS = json.load(f)
        print(f"✓ Loaded Kinetics-400 labels: {len(KINETICS_400_LABELS)} classes")
    else:
        print("⚠️ Kinetics-400 labels not found")

    if R3D_CUSTOM_MAPPING_PATH.exists():
        with open(R3D_CUSTOM_MAPPING_PATH, "r") as f:
            r3d_data = json.load(f)
            R3D_CUSTOM_LABELS = r3d_data.get('idx_to_label', {})
            R3D_CUSTOM_LABELS = {int(k): v for k, v in R3D_CUSTOM_LABELS.items()}
            # Extract metadata (model_variant, num_classes, crop_size, etc.)
            R3D_CUSTOM_META = r3d_data.get('metadata', {})
            if not R3D_CUSTOM_META:
                # Fallback: build from top-level keys if metadata block is missing
                R3D_CUSTOM_META = {
                    'model_variant': r3d_data.get('model_variant'),
                    'num_classes': len(R3D_CUSTOM_LABELS),
                }
        print(f"✓ Loaded R3D custom labels: {len(R3D_CUSTOM_LABELS)} classes")
        variant = (R3D_CUSTOM_META or {}).get('model_variant', 'unknown')
        print(f"  Model variant: {variant}")
    else:
        R3D_CUSTOM_LABELS = None
        R3D_CUSTOM_META = None


load_label_mappings()



def get_action_name(action_id, model_type='custom'):
    if model_type == 'custom' and CUSTOM_LABELS and action_id in CUSTOM_LABELS:
        return CUSTOM_LABELS[action_id]
    elif model_type == 'r3d_custom' and R3D_CUSTOM_LABELS and action_id in R3D_CUSTOM_LABELS:
        return R3D_CUSTOM_LABELS[action_id]
    elif model_type in ('intel', 'r3d', 'cuda') and KINETICS_400_LABELS:
        return KINETICS_400_LABELS.get(str(action_id), f"action_{action_id}")
    else:
        return f"action_{action_id}"


def get_id_from_name(name):
    if CUSTOM_LABELS:
        for idx, action_name in CUSTOM_LABELS.items():
            if action_name.lower() == name.lower():
                return idx, 'custom'
    if KINETICS_400_LABELS:
        for k, v in KINETICS_400_LABELS.items():
            if v.lower() == name.lower():
                return int(k), 'intel'
    raise ValueError(f"Action '{name}' not found in any label set")

def get_all_ids_from_name(name):
    """Return ALL model matches for a given action name (for mixed mode).
    
    Supports tagged format: 'laughing [custom]' → only custom model
                           'laughing [intel]'  → only intel model  
                           'laughing'          → all matching models
    """
    import re
    # Parse optional [model] tag
    tag_match = re.match(r'^(.+?)\s*\[(custom|intel|cuda)\]\s*$', name.strip())
    if tag_match:
        clean_name = tag_match.group(1).strip()
        forced_model = tag_match.group(2)
    else:
        clean_name = name.strip()
        forced_model = None

    results = []
    if CUSTOM_LABELS and (forced_model is None or forced_model == 'custom'):
        for idx, action_name in CUSTOM_LABELS.items():
            if action_name.lower() == clean_name.lower():
                results.append((idx, 'custom'))
    if R3D_CUSTOM_LABELS and (forced_model is None or forced_model == 'r3d_custom'):
        for idx, action_name in R3D_CUSTOM_LABELS.items():
            if action_name.lower() == clean_name.lower():
                results.append((idx, 'r3d_custom'))
    if KINETICS_400_LABELS and (forced_model is None or forced_model == 'intel'):
        for k, v in KINETICS_400_LABELS.items():
            if v.lower() == clean_name.lower():
                results.append((int(k), 'intel'))
    if not results:
        raise ValueError(f"Action '{name}' not found in any label set")
    return results

def get_id_from_name_with_r3d(name):
    """Extended version that also checks R3D/CUDA model (uses Kinetics-400 labels)."""
    if CUSTOM_LABELS:
        for idx, action_name in CUSTOM_LABELS.items():
            if action_name.lower() == name.lower():
                return idx, 'custom'
    if KINETICS_400_LABELS:
        for k, v in KINETICS_400_LABELS.items():
            if v.lower() == name.lower():
                # Prefer 'cuda' model type if CUDA is available, else 'intel'
                return int(k), 'intel'
    raise ValueError(f"Action '{name}' not found in any label set")


# =============================
# Person Detection & Tracking
# =============================
class PersonTracker:
    """Simple IoU-based person tracker"""

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
        self.tracks.clear()
        self.next_id = 0


class SmartActionDetector:
    """Detects people most likely performing actions"""

    def __init__(self, sticky_frames=15):
        self.prev_frame_data = None
        self.frame_count = 0
        self.selection_history = deque(maxlen=sticky_frames)

    def detect(self, frame, detector, max_people=2):
        h, w = frame.shape[:2]
        center_x, center_y = w / 2, h / 2
        result = detector.predict(frame, conf=0.40, classes=[0], verbose=False)
        current_detections = []
        for r in result:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                conf = float(b.conf)
                box_center_x = (x1 + x2) / 2
                box_center_y = (y1 + y2) / 2
                area = (x2 - x1) * (y2 - y1)
                dist = np.sqrt((box_center_x - center_x) ** 2 + (box_center_y - center_y) ** 2)
                max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
                center_score = 1 - (dist / max_dist)
                frame_area = h * w
                size_score = min(area / (frame_area * 0.3), 1.0)
                motion_score = 0
                if self.prev_frame_data and self.frame_count > 0:
                    for prev_box in self.prev_frame_data:
                        iou = self._iou((x1, y1, x2, y2), prev_box['box'])
                        if iou > 0.3:
                            prev_cx, prev_cy = prev_box['center']
                            position_change = np.sqrt(
                                (box_center_x - prev_cx) ** 2 + (box_center_y - prev_cy) ** 2)
                            motion_score = min(position_change / 50.0, 1.0)
                            break
                temporal_score = 0
                if len(self.selection_history) > 0:
                    for prev_selection in self.selection_history:
                        for prev_box in prev_selection:
                            if self._iou((x1, y1, x2, y2), prev_box) > 0.5:
                                temporal_score = 1.0
                                break
                        if temporal_score > 0:
                            break
                current_detections.append({
                    'box': (x1, y1, x2, y2),
                    'center': (box_center_x, box_center_y),
                    'area': area,
                    'conf': conf,
                    'motion': motion_score,
                    'center_prox': center_score,
                    'size': size_score,
                    'temporal': temporal_score
                })
        for det in current_detections:
            action_score = (
                det['conf'] * 0.2 +
                det['center_prox'] * 0.2 +
                det['size'] * 0.2 +
                det['motion'] * 0.2 +
                det['temporal'] * 0.2
            )
            det['score'] = action_score
        self.prev_frame_data = current_detections
        self.frame_count += 1
        sorted_detections = sorted(current_detections, key=lambda x: x['score'], reverse=True)
        selected_boxes = [d['box'] for d in sorted_detections[:max_people]]
        self.selection_history.append(selected_boxes)
        return selected_boxes

    def detect_from_boxes(self, frame, boxes, max_people=2):
        if not boxes:
            return []
        h, w = frame.shape[:2]
        center_x, center_y = w / 2, h / 2
        current_detections = []
        for box in boxes:
            x1, y1, x2, y2 = box
            box_center_x = (x1 + x2) / 2
            box_center_y = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)
            dist = np.sqrt((box_center_x - center_x) ** 2 + (box_center_y - center_y) ** 2)
            max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
            center_score = 1 - (dist / max_dist)
            frame_area = h * w
            size_score = min(area / (frame_area * 0.3), 1.0)
            motion_score = 0
            if self.prev_frame_data and self.frame_count > 0:
                for prev_box in self.prev_frame_data:
                    iou = self._iou((x1, y1, x2, y2), prev_box['box'])
                    if iou > 0.3:
                        prev_cx, prev_cy = prev_box['center']
                        position_change = np.sqrt(
                            (box_center_x - prev_cx) ** 2 + (box_center_y - prev_cy) ** 2)
                        motion_score = min(position_change / 50.0, 1.0)
                        break
            temporal_score = 0
            if len(self.selection_history) > 0:
                for prev_selection in self.selection_history:
                    for prev_box in prev_selection:
                        if self._iou((x1, y1, x2, y2), prev_box) > 0.5:
                            temporal_score = 1.0
                            break
                    if temporal_score > 0:
                        break
            current_detections.append({
                'box': (x1, y1, x2, y2),
                'center': (box_center_x, box_center_y),
                'area': area,
                'conf': 0.5,
                'motion': motion_score,
                'center_prox': center_score,
                'size': size_score,
                'temporal': temporal_score
            })
        for det in current_detections:
            action_score = (
                det['conf'] * 0.2 +
                det['center_prox'] * 0.2 +
                det['size'] * 0.2 +
                det['motion'] * 0.2 +
                det['temporal'] * 0.2
            )
            det['score'] = action_score
        self.prev_frame_data = current_detections
        self.frame_count += 1
        sorted_detections = sorted(current_detections, key=lambda x: x['score'], reverse=True)
        selected_boxes = [d['box'] for d in sorted_detections[:max_people]]
        self.selection_history.append(selected_boxes)
        return selected_boxes

    def _iou(self, box1, box2):
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

    def cleanup(self):
        self.prev_frame_data = None
        self.selection_history.clear()


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


# =============================
# Model paths
# =============================
ENCODER_XML = BASE_DIR / "models/intel_action/encoder/FP32/action-recognition-0001-encoder.xml"
ENCODER_BIN = BASE_DIR / "models/intel_action/encoder/FP32/action-recognition-0001-encoder.bin"
SEQUENCE_LENGTH = 16

# R3D model config
R3D_INPUT_SIZE = 112  # R3D expects 112x112
R3D_CLIP_LENGTH = 16  # Same as SEQUENCE_LENGTH — convenient!
R3D_IMAGENET_MEAN = [0.43216, 0.394666, 0.37645]
R3D_IMAGENET_STD = [0.22803, 0.22145, 0.216989]


# =============================
# R3D CUDA Model Wrapper
# =============================
class R3DModelWrapper:
    """
    Wraps a torchvision R3D model for use alongside OpenVINO models.

    R3D is an end-to-end 3D CNN that takes a clip of frames (B, C, T, H, W)
    and outputs Kinetics-400 class probabilities directly — no separate
    encoder/decoder needed.

    Supported model variants:
      - r3d_18  (default, smallest, fastest)
      - mc3_18  (mixed convolution variant)
      - r2plus1d_18  (R(2+1)D — decomposed 3D convolutions, more accurate)
    """

    def __init__(self, model_name='r3d_18', device_str='cuda', half_precision=True,
                 custom_weights=None, custom_num_classes=None):
        """
        Args:
            model_name: One of 'r3d_18', 'mc3_18', 'r2plus1d_18'
            device_str: 'cuda' or 'cpu'
            half_precision: Use FP16 on CUDA for faster inference
            custom_weights: Path to .pth file with fine-tuned weights (optional)
            custom_num_classes: Number of classes in custom model (required if custom_weights)
        """
        self.model_name = model_name
        self.device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
        self.half = half_precision and self.device.type == 'cuda'
        self.num_classes = custom_num_classes or 400  # default Kinetics-400

        tag = " (custom)" if custom_weights else ""
        print(f"🚀 Loading {model_name}{tag} on {self.device} (FP16: {self.half})...")

        # Load pretrained model
        if model_name == 'r3d_18':
            self.model = video_models.r3d_18(weights='DEFAULT')
        elif model_name == 'mc3_18':
            self.model = video_models.mc3_18(weights='DEFAULT')
        elif model_name == 'r2plus1d_18':
            self.model = video_models.r2plus1d_18(weights='DEFAULT')
        else:
            raise ValueError(f"Unknown R3D variant: {model_name}")

        if custom_weights and custom_num_classes:
            in_features = self.model.fc.in_features
            self.model.fc = torch.nn.Sequential(
                torch.nn.Dropout(0.4),
                torch.nn.Linear(in_features, custom_num_classes),
            )
            state = torch.load(custom_weights, map_location='cpu', weights_only=True)
            # Handle ActionRecognitionModel wrapper format
            if 'model_state_dict' in state:
                self.model.load_state_dict(state['model_state_dict'])
            else:
                self.model.load_state_dict(state)
            print(f"   ✓ Loaded custom weights: {custom_num_classes} classes")

        self.model.eval()
        self.model.to(self.device)
        if self.half:
            self.model.half()


        # Pre-build normalization tensors on device for speed
        self.mean = torch.tensor(R3D_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1, 1)
        self.std = torch.tensor(R3D_IMAGENET_STD, device=self.device).view(1, 3, 1, 1, 1)
        if self.half:
            self.mean = self.mean.half()
            self.std = self.std.half()

        # Warm-up inference to trigger CUDA kernel compilation
        self._warmup()
        print(f"✓ {model_name} loaded and warmed up on {self.device}")

    def _warmup(self):
        """Run a dummy forward pass to warm up CUDA kernels."""
        dummy = torch.zeros(1, 3, R3D_CLIP_LENGTH, R3D_INPUT_SIZE, R3D_INPUT_SIZE,
                            device=self.device)
        if self.half:
            dummy = dummy.half()
        with torch.no_grad():
            _ = self.model(dummy)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

    def preprocess_clip(self, raw_frames, roi=None):
        """
        Preprocess a list of BGR numpy frames into a (1, 3, T, H, W) tensor.

        Args:
            raw_frames: List of numpy arrays (BGR, HWC) — exactly R3D_CLIP_LENGTH frames
            roi: Optional (x1, y1, x2, y2) to crop before resizing

        Returns:
            torch.Tensor on self.device, shape (1, 3, T, 112, 112), normalized
        """
        assert len(raw_frames) == R3D_CLIP_LENGTH, \
            f"Expected {R3D_CLIP_LENGTH} frames, got {len(raw_frames)}"

        processed = []
        for frame in raw_frames:
            f = frame
            if roi is not None:
                x1, y1, x2, y2 = roi
                f = f[y1:y2, x1:x2]

            # Resize to 112x112
            f = cv2.resize(f, (R3D_INPUT_SIZE, R3D_INPUT_SIZE),
                           interpolation=cv2.INTER_LINEAR)

            # BGR → RGB, HWC → CHW, scale to [0, 1]
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            f = f.astype(np.float32) / 255.0
            f = np.transpose(f, (2, 0, 1))  # (3, H, W)
            processed.append(f)

        # Stack: (T, 3, H, W) → (3, T, H, W) → (1, 3, T, H, W)
        clip = np.stack(processed, axis=0)            # (T, 3, H, W)
        clip = np.transpose(clip, (1, 0, 2, 3))       # (3, T, H, W)
        clip = np.expand_dims(clip, axis=0)            # (1, 3, T, H, W)

        tensor = torch.from_numpy(clip).to(self.device)
        if self.half:
            tensor = tensor.half()

        # Normalize with ImageNet stats
        tensor = (tensor - self.mean) / self.std
        return tensor

    @torch.no_grad()
    def predict(self, clip_tensor):
        """
        Run inference on a preprocessed clip tensor.

        Args:
            clip_tensor: (1, 3, T, 112, 112) tensor on device

        Returns:
            numpy array of raw logits (400,)
        """
        output = self.model(clip_tensor)
        return output.cpu().float().numpy().flatten()

    @torch.no_grad()
    def predict_from_frames(self, raw_frames, roi=None):
        """
        Convenience: preprocess + predict in one call.

        Args:
            raw_frames: List of BGR numpy frames (length = R3D_CLIP_LENGTH)
            roi: Optional crop region

        Returns:
            numpy array of raw logits (400,)
        """
        clip_tensor = self.preprocess_clip(raw_frames, roi=roi)
        return self.predict(clip_tensor)

    def cleanup(self):
        """Free GPU memory."""
        del self.model
        del self.mean
        del self.std
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        print(f"🧹 {self.model_name} cleaned up")


# =============================
# Async Inference Engine (OpenVINO)
# =============================
class AsyncBatchedInferenceEngine:
    def __init__(self, compiled_model, input_layer, output_layer, num_requests=2):
        self.compiled_model = compiled_model
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.num_requests = num_requests
        self.requests = [compiled_model.create_infer_request() for _ in range(num_requests)]
        self.current_request = 0
        self.total_inferences = 0
        self.start_time = time.time()

    def infer_async(self, frame):
        request = self.requests[self.current_request]
        self.current_request = (self.current_request + 1) % self.num_requests
        request.start_async({self.input_layer.any_name: frame})
        self.total_inferences += 1
        return request

    def wait_and_get(self, request):
        request.wait()
        result = request.get_tensor(self.output_layer).data
        return result

    def get_stats(self):
        elapsed = time.time() - self.start_time
        fps = self.total_inferences / elapsed if elapsed > 0 else 0
        return {
            'total_inferences': self.total_inferences,
            'elapsed_time': elapsed,
            'inference_fps': fps
        }

    def cleanup(self):
        self.requests.clear()


# =============================
# PARALLEL YOLO DETECTOR - OPTIMIZED
# =============================
class ParallelYOLODetector:
    """Parallel YOLO detection with frame skipping"""

    def __init__(self, model_name="yolo11n.pt", num_workers=2, skip_frames=4):
        self.model = YOLO(model_name)
        self.skip_frames = skip_frames
        self.frame_counter = 0
        self.last_detections = None
        self.detection_lock = threading.Lock()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)
        self.pending_future = None
        self._shutdown = False

    def detect_async(self, frame):
        if self._shutdown:
            return self.get_latest_detections()

        self.frame_counter += 1

        if self.skip_frames > 1 and self.frame_counter % self.skip_frames != 0:
            return self.get_latest_detections()

        if self.pending_future is not None and self.pending_future.done():
            try:
                self.pending_future.result()
            except Exception:
                pass
            self.pending_future = None

        if self.pending_future is None or self.pending_future.done():
            self.pending_future = self.executor.submit(self._detect_sync, frame.copy())

        return self.get_latest_detections()

    def _detect_sync(self, frame):
        try:
            results = self.model.predict(frame, conf=0.40, classes=[0],
                                         verbose=False, imgsz=640)
            boxes = []
            for r in results:
                for b in r.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    boxes.append((x1, y1, x2, y2))
            with self.detection_lock:
                self.last_detections = boxes
            return boxes
        except Exception as e:
            print(f"YOLO detection error: {e}")
            return []

    def get_latest_detections(self):
        with self.detection_lock:
            return self.last_detections.copy() if self.last_detections else []

    def shutdown(self):
        self._shutdown = True
        if self.pending_future and not self.pending_future.done():
            self.pending_future.cancel()
        self.pending_future = None
        self.executor.shutdown(wait=True, cancel_futures=True)
        with self.detection_lock:
            self.last_detections = None


# =============================
# Pipelined Preprocessing Pool
# =============================
class PreprocessPipeline:
    def __init__(self, num_workers=2):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_workers)

    def submit(self, frame, input_shape, roi=None):
        return self.executor.submit(preprocess_frame, frame, input_shape, roi)

    def shutdown(self):
        self.executor.shutdown(wait=False, cancel_futures=True)


# =============================
# Video Writer Thread
# =============================
class ThreadedVideoWriter:
    def __init__(self, output_path, fourcc, fps, frame_size):
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        self.queue = queue.Queue(maxsize=60)
        self._shutdown = False
        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.thread.start()

    def _write_loop(self):
        while not self._shutdown or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.5)
                self.writer.write(frame)
                self.queue.task_done()
            except queue.Empty:
                continue

    def write(self, frame):
        if not self._shutdown:
            try:
                self.queue.put_nowait(frame)
            except queue.Full:
                pass

    def release(self):
        self._shutdown = True
        self.thread.join(timeout=5.0)
        self.writer.release()


# =============================
# Load models - TRIPLE MODEL SUPPORT (custom + intel + r3d/cuda)
# =============================
def load_models(device="AUTO", openvino_threads=None,
                enable_r3d=True, r3d_model_name='r3d_18', r3d_half=True):
    """
    Load all available models.

    Returns:
        compiled_encoder, encoder_input, encoder_output,
        models_info dict, selected_device string, r3d_wrapper (or None)
    """
    ie = Core()
    available_devices = ie.available_devices
    print(f"Available OpenVINO devices: {available_devices}")

    if device == "AUTO":
        device_priority = ["GPU.1", "GPU.0", "GPU", "CPU"]
        selected_device = next((d for d in device_priority if d in available_devices), "CPU")
    else:
        selected_device = device if device in available_devices else "CPU"

    print(f"Using OpenVINO device: {selected_device}")

    if openvino_threads and selected_device == "CPU":
        ie.set_property("CPU", {"INFERENCE_NUM_THREADS": openvino_threads})
        print(f"✓ OpenVINO CPU threads set to {openvino_threads}")

    if not ENCODER_XML.exists() or not ENCODER_BIN.exists():
        raise FileNotFoundError(f"❌ Encoder model not found at {ENCODER_XML}")
    print("✓ Encoder model found")

    encoder_model = ie.read_model(model=ENCODER_XML, weights=ENCODER_BIN)
    compiled_encoder = ie.compile_model(model=encoder_model, device_name=selected_device)

    custom_decoder_xml = BASE_DIR / "action_classifier_3d.xml"
    custom_decoder_bin = BASE_DIR / "action_classifier_3d.bin"
    intel_decoder_xml = BASE_DIR / "models/intel_action/decoder/FP32/action-recognition-0001-decoder.xml"
    intel_decoder_bin = BASE_DIR / "models/intel_action/decoder/FP32/action-recognition-0001-decoder.bin"

    models_info = {'custom': None, 'intel': None, 'cuda': None}

    if custom_decoder_xml.exists() and custom_decoder_bin.exists():
        print("✓ Loading custom fine-tuned decoder model")
        custom_decoder_model = ie.read_model(model=custom_decoder_xml, weights=custom_decoder_bin)
        compiled_custom_decoder = ie.compile_model(model=custom_decoder_model, device_name=selected_device)
        models_info['custom'] = {
            'compiled': compiled_custom_decoder,
            'input': compiled_custom_decoder.input(0),
            'output': compiled_custom_decoder.output(0),
            'labels': CUSTOM_LABELS,
            'type': 'openvino'
        }
    else:
        print("⚠️ Custom decoder model not found")

    if intel_decoder_xml.exists() and intel_decoder_bin.exists():
        print("✓ Loading Intel Kinetics-400 decoder model")
        intel_decoder_model = ie.read_model(model=intel_decoder_xml, weights=intel_decoder_bin)
        compiled_intel_decoder = ie.compile_model(model=intel_decoder_model, device_name=selected_device)
        models_info['intel'] = {
            'compiled': compiled_intel_decoder,
            'input': compiled_intel_decoder.input(0),
            'output': compiled_intel_decoder.output(0),
            'labels': KINETICS_400_LABELS,
            'type': 'openvino'
        }
    else:
        print("⚠️ Intel decoder model not found")

    # ---- R3D / CUDA model (pretrained Kinetics-400) ----
    r3d_wrapper = None
    if enable_r3d and TORCH_AVAILABLE:
        try:
            r3d_device = 'cuda' if CUDA_AVAILABLE else 'cpu'
            r3d_wrapper = R3DModelWrapper(
                model_name=r3d_model_name,
                device_str=r3d_device,
                half_precision=r3d_half and CUDA_AVAILABLE
            )
            models_info['cuda'] = {
                'wrapper': r3d_wrapper,
                'labels': KINETICS_400_LABELS,
                'type': 'pytorch'
            }
        except Exception as e:
            print(f"⚠️ Failed to load R3D model: {e}")
            r3d_wrapper = None
    elif enable_r3d and not TORCH_AVAILABLE:
        print("⚠️ R3D requested but PyTorch is not installed — skipping")

    # ---- R3D Custom fine-tuned model ----
    if (enable_r3d and TORCH_AVAILABLE
            and R3D_CUSTOM_LABELS and R3D_CUSTOM_WEIGHTS_PATH.exists()):
        try:
            r3d_custom_device = 'cuda' if CUDA_AVAILABLE else 'cpu'
            # Use variant from mapping metadata, fall back to CLI r3d_model_name
            custom_variant = (R3D_CUSTOM_META or {}).get('model_variant') or r3d_model_name
            num_classes = len(R3D_CUSTOM_LABELS)

            r3d_custom_wrapper = R3DModelWrapper(
                model_name=custom_variant,
                device_str=r3d_custom_device,
                half_precision=r3d_half and CUDA_AVAILABLE,
                custom_weights=str(R3D_CUSTOM_WEIGHTS_PATH),
                custom_num_classes=num_classes,
            )
            models_info['r3d_custom'] = {
                'wrapper': r3d_custom_wrapper,
                'labels': R3D_CUSTOM_LABELS,
                'type': 'pytorch',
            }
        except Exception as e:
            print(f"⚠️ Failed to load R3D custom model: {e}")
    elif enable_r3d and R3D_CUSTOM_LABELS and not R3D_CUSTOM_WEIGHTS_PATH.exists():
        print(f"⚠️ R3D custom mapping found but weights missing: {R3D_CUSTOM_WEIGHTS_PATH}")

    if (models_info['custom'] is None and models_info['intel'] is None
            and models_info['cuda'] is None and models_info.get('r3d_custom') is None):
        raise FileNotFoundError("❌ No models found! Need at least one decoder or R3D.")

    print("✓ Model loading complete.")
    print(f"  - Custom model:  {'✓ Available' if models_info['custom'] else '✗ Not available'}")
    print(f"  - Intel model:   {'✓ Available' if models_info['intel'] else '✗ Not available'}")
    print(f"  - R3D/CUDA:      {'✓ Available' if models_info['cuda'] else '✗ Not available'}")
    print(f"  - R3D Custom:    {'✓ Available (' + str(len(R3D_CUSTOM_LABELS)) + ' classes)' if models_info.get('r3d_custom') else '✗ Not available'}")
    print()

    return (
        compiled_encoder, compiled_encoder.input(0), compiled_encoder.output(0),
        models_info,
        selected_device,
        r3d_wrapper
    )


# =============================
# Preprocess frame with ROI support (for OpenVINO encoder)
# =============================
def preprocess_frame(frame, input_shape, roi=None):
    frame_to_process = frame
    if roi is not None:
        x1, y1, x2, y2 = roi
        frame_to_process = frame[y1:y2, x1:x2]

    N, C, H, W = input_shape
    h, w = frame_to_process.shape[:2]

    if h > 1080 or w > 1920:
        scale_factor = min(720 / h, 1280 / w)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        frame_to_process = cv2.resize(frame_to_process, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = frame_to_process.shape[:2]

    scale = min(W / w, H / h)
    new_w, new_h = int(w * scale), int(h * scale)
    frame_resized = cv2.resize(frame_to_process, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_top = (H - new_h) // 2
    pad_bottom = H - new_h - pad_top
    pad_left = (W - new_w) // 2
    pad_right = W - new_w - pad_left

    frame_padded = cv2.copyMakeBorder(frame_resized, pad_top, pad_bottom, pad_left, pad_right,
                                      borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

    frame_padded = np.ascontiguousarray(frame_padded.transpose(2, 0, 1))
    result = np.expand_dims(frame_padded, axis=0).astype(np.float32)
    return result


# =============================
# Draw visualization
# =============================
def draw_detections_with_actions(frame, tracked_people, action_roi, detected_actions,
                                 focus_region="full_body"):
    annotated = frame.copy()
    h, w = frame.shape[:2]
    color_palette = [(0, 255, 0), (255, 0, 255), (255, 255, 0), (0, 255, 255)]

    for i, item in enumerate(tracked_people):
        if isinstance(item, tuple) and len(item) == 2:
            track_id, (x1, y1, x2, y2) = item
            color = color_palette[track_id % len(color_palette)]
        else:
            x1, y1, x2, y2 = item
            color = (0, 255, 0)
            track_id = i
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"Person {track_id}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - label_h - 5), (x1 + label_w, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    if action_roi:
        x1, y1, x2, y2 = action_roi
        if focus_region == 'upper_body':
            roi_color = (0, 255, 255)
            region_text = "UPPER BODY"
        elif focus_region == 'lower_body':
            roi_color = (0, 255, 0)
            region_text = "LOWER BODY"
        else:
            roi_color = (255, 0, 0)
            region_text = "FULL BODY"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), roi_color, 3)
        label = f"ACTION: {region_text}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y2 + 5), (x1 + label_w + 10, y2 + label_h + 15), roi_color,
                      -1)
        cv2.putText(annotated, label, (x1 + 5, y2 + label_h + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    if detected_actions:
        draw_action_panel(annotated, detected_actions, max_labels=3)

    return annotated


def draw_action_panel(frame, detected_actions, max_labels=3):
    if not detected_actions:
        return
    h, w = frame.shape[:2]
    top_actions = detected_actions[:max_labels]
    panel_height = 30 + (len(top_actions) * 35)
    panel_width = 400
    panel_x = w - panel_width - 10
    panel_y = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y),
                  (panel_x + panel_width, panel_y + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (panel_x, panel_y),
                  (panel_x + panel_width, panel_y + panel_height), (0, 255, 255), 2)
    cv2.putText(frame, "DETECTED ACTIONS",
                (panel_x + 10, panel_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    y_offset = panel_y + 50
    for i, item in enumerate(top_actions):
        if len(item) == 3:
            action_name, score, model_type = item
            if model_type == 'custom':
                text_color = (0, 255, 0)       # Green
            elif model_type == 'cuda':
                text_color = (0, 128, 255)      # Orange — R3D/CUDA
            else:
                text_color = (255, 165, 0)      # Blue-ish — Intel
        else:
            action_name, score = item
            text_color = (0, 255, 255)

        action_text = f"{i + 1}. {action_name}"
        score_text = f"{score:.2%}"
        cv2.putText(frame, action_text,
                    (panel_x + 10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        cv2.putText(frame, score_text,
                    (panel_x + panel_width - 70, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        bar_width = int((panel_width - 30) * score)
        cv2.rectangle(frame,
                      (panel_x + 10, y_offset + 5),
                      (panel_x + 10 + bar_width, y_offset + 10),
                      text_color, -1)
        y_offset += 35


# =============================
# Softmax
# =============================
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)


# =============================
# MAIN — Run action detection (OPTIMIZED + R3D CUDA)
# =============================
def run_action_detection(video_path, device="AUTO", sample_rate=5, log_file="action_log.csv",
                         debug=False, top_k=50, confidence_threshold=0.01, show_video=False,
                         num_requests=2, interesting_actions=None,
                         progress_callback=None, cancel_flag=None,
                         draw_bboxes=True, annotated_output=None,
                         use_person_detection=True, max_people=2,
                         yolo_workers=2, yolo_skip_frames=4, downscale_factor=0.5,
                         warm_up_seconds=2, include_model_type=False,
                         openvino_threads=None, preprocess_workers=2,
                         enable_r3d=True, r3d_model_name='r3d_18', r3d_half=True):
    """
    Run action recognition — OPTIMIZED version with R3D/CUDA support.

    Key features:
      1. Pipelined preprocessing (overlaps with inference via thread pool)
      2. Decoder result caching (one forward pass per model, not per action)
      3. Threaded video writer (non-blocking disk I/O)
      4. R3D end-to-end model on CUDA alongside OpenVINO encoder+decoder
      5. Raw frame ring buffer for R3D (stores BGR frames, not encoder features)
      6. Configurable OpenVINO thread count to share cores fairly
    """

    # ---- CPU thread budget ----
    cpu_count = os.cpu_count() or 4
    print(f"📊 CPU cores: {cpu_count} (threads: {cpu_count})")

    if openvino_threads is None:
        openvino_threads = max(2, cpu_count // 2)
    os.environ["OMP_NUM_THREADS"] = str(openvino_threads)
    os.environ["MKL_NUM_THREADS"] = str(openvino_threads)
    print(f"✅ OpenVINO threads: {openvino_threads} | "
          f"YOLO workers: {yolo_workers} | Preprocess workers: {preprocess_workers}")
    if enable_r3d:
        print(f"✅ R3D model: {r3d_model_name} | FP16: {r3d_half}")

    # ---- Parse interesting actions ----
    action_to_model = {}
    if interesting_actions is not None:
        interesting_actions_set = set([s.lower() for s in interesting_actions])
        for action_name in interesting_actions:
            try:
                all_matches = get_all_ids_from_name(action_name)
                # Use clean name (without tag) for the key
                clean_name = re.sub(r'\s*\[(custom|intel|cuda)\]\s*$', '', action_name).strip()
                for action_id, model_type in all_matches:
                    key = f"{clean_name.lower()}__{model_type}"
                    action_to_model[key] = (action_id, model_type)
                    print(f"📌 Action '{action_name}' found in {model_type} model (ID: {action_id})")
                    # If R3D is enabled and the action is from Kinetics-400, also map to cuda
                    if enable_r3d and model_type == 'intel':
                        cuda_key = f"{action_name.lower()}__cuda"
                        action_to_model[cuda_key] = (action_id, 'cuda')
                        print(f"   ↳ Also mapped to R3D/CUDA model (ID: {action_id})")
            except ValueError as e:
                print(f"⚠️ {e}")
                raise
    else:
        interesting_actions_set = None

    # ---- Load models ----
    (compiled_encoder, encoder_input, encoder_output,
     models_info, actual_device, r3d_wrapper) = \
        load_models(device, openvino_threads=openvino_threads,
                    enable_r3d=enable_r3d, r3d_model_name=r3d_model_name,
                    r3d_half=r3d_half)

    encoder_engine = AsyncBatchedInferenceEngine(
        compiled_encoder, encoder_input, encoder_output, num_requests=num_requests)

    # ---- Preprocessing pipeline ----
    preprocess_pool = PreprocessPipeline(num_workers=preprocess_workers)

    # ---- Person detection ----
    yolo_detector = None
    person_tracker = None
    action_detector = None

    if use_person_detection:
        print(f"🔍 Initializing parallel YOLO with {yolo_workers} workers "
              f"(skip: {yolo_skip_frames})...")
        yolo_detector = ParallelYOLODetector(
            model_name="yolo11n.pt",
            num_workers=yolo_workers,
            skip_frames=yolo_skip_frames
        )
        person_tracker = PersonTracker(iou_threshold=0.3, max_lost_frames=10)
        action_detector = SmartActionDetector(sticky_frames=15)

    # ---- Open video ----
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    expected_processed_frames = total_frames // sample_rate

    video_writer = None
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ---- R3D raw frame ring buffer ----
    # R3D needs raw BGR frames (not encoder features), so we keep a separate buffer
    has_any_r3d = r3d_wrapper is not None or models_info.get('r3d_custom') is not None
    raw_frame_buffer = deque(maxlen=R3D_CLIP_LENGTH) if has_any_r3d else None

    try:
        if draw_bboxes and annotated_output:
            if frame_height > 1080 and downscale_factor < 1.0:
                frame_width = int(frame_width * downscale_factor)
                frame_height = int(frame_height * downscale_factor)
                print(f"📏 Downscaling output to {frame_width}x{frame_height}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = ThreadedVideoWriter(annotated_output, fourcc, fps,
                                               (frame_width, frame_height))
            print(f"🎨 Creating annotated video (threaded writer): {annotated_output}")

        sequence_buffer = []
        all_actions = []
        prev_req = None
        prev_timestamp_secs = None
        prev_frame_id = None
        frame_id = 0
        processed_frames = 0
        detection_count = 0

        recent_detections = deque(maxlen=SEQUENCE_LENGTH)
        current_tracked_people = []
        current_action_roi = None
        current_focus_region = "full_body"

        start_time = time.time()
        last_gui_update = start_time

        yolo_time = 0
        preprocess_time = 0
        inference_time = 0
        r3d_time = 0
        draw_time = 0
        last_perf_print = start_time
        last_gc_time = start_time

        # =============================================
        # WARM-UP PHASE
        # =============================================
        print(f"\n🔥 WARM-UP: Pre-filling buffer with {warm_up_seconds}s of frames...")
        warm_up_frames_needed = min(int(fps * warm_up_seconds), SEQUENCE_LENGTH)
        warm_up_frame_count = 0

        while warm_up_frame_count < warm_up_frames_needed:
            ret, warm_up_frame = cap.read()
            if not ret:
                break

            if use_person_detection and yolo_detector:
                yolo_start = time.time()
                h, w = warm_up_frame.shape[:2]
                if h > 1080 or w > 1920:
                    processing_frame = cv2.resize(
                        warm_up_frame,
                        (int(w * downscale_factor), int(h * downscale_factor)),
                        interpolation=cv2.INTER_AREA)
                else:
                    processing_frame = warm_up_frame

                yolo_detector.detect_async(processing_frame)
                raw_boxes = yolo_detector.get_latest_detections()

                if raw_boxes:
                    if processing_frame.shape[:2] != warm_up_frame.shape[:2]:
                        scale_h = warm_up_frame.shape[0] / processing_frame.shape[0]
                        scale_w = warm_up_frame.shape[1] / processing_frame.shape[1]
                        raw_boxes = [
                            (int(x1 * scale_w), int(y1 * scale_h),
                             int(x2 * scale_w), int(y2 * scale_h))
                            for (x1, y1, x2, y2) in raw_boxes
                        ]
                    action_boxes = action_detector.detect_from_boxes(
                        warm_up_frame, raw_boxes, max_people=max_people)
                    tracked = person_tracker.update(action_boxes)
                    current_tracked_people = tracked
                    current_action_roi = merge_boxes(action_boxes) if action_boxes else None
                yolo_time += time.time() - yolo_start

            # Preprocess for OpenVINO encoder
            preprocess_start = time.time()
            processed_frame = preprocess_frame(
                warm_up_frame, encoder_input.shape,
                roi=current_action_roi if use_person_detection else None)
            preprocess_time += time.time() - preprocess_start

            inference_start = time.time()
            req = encoder_engine.infer_async(processed_frame)
            features = encoder_engine.wait_and_get(req)[0]
            features = np.reshape(features, (-1,))
            sequence_buffer.append(features)
            inference_time += time.time() - inference_start

            # Store raw frame for R3D
            if raw_frame_buffer is not None:
                raw_frame_buffer.append(warm_up_frame.copy())

            if video_writer and draw_bboxes:
                draw_start = time.time()
                warm_up_annotated = warm_up_frame.copy()
                h_a, w_a = warm_up_annotated.shape[:2]
                status_text = f"Initializing... ({warm_up_frame_count + 1}/{warm_up_frames_needed})"
                text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
                text_x = (w_a - text_size[0]) // 2
                text_y = (h_a + text_size[1]) // 2
                cv2.putText(warm_up_annotated, status_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                if use_person_detection:
                    warm_up_annotated = draw_detections_with_actions(
                        warm_up_annotated, current_tracked_people,
                        current_action_roi, [], current_focus_region)
                if (warm_up_annotated.shape[0] != frame_height or
                        warm_up_annotated.shape[1] != frame_width):
                    warm_up_annotated = cv2.resize(
                        warm_up_annotated, (frame_width, frame_height),
                        interpolation=cv2.INTER_LINEAR)
                video_writer.write(warm_up_annotated)
                draw_time += time.time() - draw_start

            warm_up_frame_count += 1
            frame_id += 1

            if progress_callback and time.time() - last_gui_update > 0.1:
                progress_msg = f"Warm-up: {warm_up_frame_count}/{warm_up_frames_needed} frames"
                progress_callback(warm_up_frame_count, warm_up_frames_needed, "Warm-up",
                                  progress_msg)
                last_gui_update = time.time()

        print(f"✅ Warm-up complete: Buffer has {len(sequence_buffer)}/{SEQUENCE_LENGTH} frames")
        if raw_frame_buffer is not None:
            print(f"   R3D raw buffer: {len(raw_frame_buffer)}/{R3D_CLIP_LENGTH} frames")

        # =============================================
        # MAIN PROCESSING LOOP
        # =============================================
        pending_preprocess_future = None

        with open(log_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp_mmss", "frame_id", "action_id", "action_name",
                             "score", "timestamp_seconds", "model_type"])

            while True:
                if cancel_flag and cancel_flag.is_set():
                    print("⚠️ Action detection canceled by user.")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_id += 1

                # ---- Store raw frame for R3D ----
                if raw_frame_buffer is not None:
                    raw_frame_buffer.append(frame.copy())

                # ---- Person detection ----
                if use_person_detection and yolo_detector:
                    yolo_start = time.time()
                    h, w = frame.shape[:2]
                    if h > 1080 or w > 1920:
                        processing_frame = cv2.resize(
                            frame,
                            (int(w * downscale_factor), int(h * downscale_factor)),
                            interpolation=cv2.INTER_AREA)
                    else:
                        processing_frame = frame

                    yolo_detector.detect_async(processing_frame)
                    raw_boxes = yolo_detector.get_latest_detections()

                    if raw_boxes:
                        if processing_frame.shape[:2] != frame.shape[:2]:
                            scale_h = frame.shape[0] / processing_frame.shape[0]
                            scale_w = frame.shape[1] / processing_frame.shape[1]
                            raw_boxes = [
                                (int(x1 * scale_w), int(y1 * scale_h),
                                 int(x2 * scale_w), int(y2 * scale_h))
                                for (x1, y1, x2, y2) in raw_boxes
                            ]
                        action_boxes = action_detector.detect_from_boxes(
                            frame, raw_boxes, max_people=max_people)
                        tracked = person_tracker.update(action_boxes)
                        current_tracked_people = tracked
                        current_action_roi = merge_boxes(action_boxes) if action_boxes else None
                    yolo_time += time.time() - yolo_start

                # ---- Write annotated frame ----
                if video_writer and draw_bboxes:
                    draw_start = time.time()
                    current_timestamp_secs = frame_id / fps
                    mins, secs = divmod(int(current_timestamp_secs), 60)
                    timestamp_str = f"{mins:02d}:{secs:02d}"

                    annotated = draw_detections_with_actions(
                        frame, current_tracked_people, current_action_roi,
                        list(recent_detections) if len(sequence_buffer) >= SEQUENCE_LENGTH else [],
                        current_focus_region)
                    cv2.putText(annotated, timestamp_str, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    if len(sequence_buffer) < SEQUENCE_LENGTH:
                        buffer_status = f"Buffer: {len(sequence_buffer)}/{SEQUENCE_LENGTH}"
                        cv2.putText(annotated, buffer_status, (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    if (annotated.shape[0] != frame_height or
                            annotated.shape[1] != frame_width):
                        annotated = cv2.resize(annotated, (frame_width, frame_height),
                                               interpolation=cv2.INTER_LINEAR)
                    video_writer.write(annotated)
                    draw_time += time.time() - draw_start

                # ---- Action recognition (sampled frames) ----
                if frame_id % sample_rate == 0:
                    timestamp_secs = frame_id / fps
                    mins, secs = divmod(int(timestamp_secs), 60)
                    timestamp_str = f"{mins:02d}:{secs:02d}"

                    # --- PIPELINE: collect previous preprocess result ---
                    if pending_preprocess_future is not None:
                        preprocess_start = time.time()
                        processed_frame = pending_preprocess_future.result()
                        preprocess_time += time.time() - preprocess_start
                    else:
                        preprocess_start = time.time()
                        processed_frame = preprocess_frame(
                            frame, encoder_input.shape,
                            roi=current_action_roi if use_person_detection else None)
                        preprocess_time += time.time() - preprocess_start

                    # Submit NEXT frame's preprocess now
                    pending_preprocess_future = preprocess_pool.submit(
                        frame, encoder_input.shape,
                        roi=current_action_roi if use_person_detection else None)

                    inference_start = time.time()
                    req = encoder_engine.infer_async(processed_frame)
                    inference_time += time.time() - inference_start

                    # ---- Process PREVIOUS request's results ----
                    if prev_req is not None and len(sequence_buffer) >= SEQUENCE_LENGTH:
                        features = encoder_engine.wait_and_get(prev_req)[0]
                        features = np.reshape(features, (-1,))
                        sequence_buffer.append(features.copy())

                        if len(sequence_buffer) > SEQUENCE_LENGTH:
                            sequence_buffer.pop(0)

                        if len(sequence_buffer) == SEQUENCE_LENGTH:
                            # OpenVINO decoder input
                            sequence_array = np.expand_dims(
                                np.stack(sequence_buffer, axis=0), axis=0)

                            frame_detections = []
                            use_timestamp_secs = prev_timestamp_secs
                            use_frame_id = prev_frame_id
                            use_mins, use_secs = divmod(int(use_timestamp_secs), 60)
                            use_timestamp_str = f"{use_mins:02d}:{use_secs:02d}"

                            # ==============================
                            # Run decoders (OpenVINO + R3D)
                            # ==============================
                            decoder_cache = {}
                            # -- R3D inference (pretrained + custom, if buffer is full) --
                            if (raw_frame_buffer is not None
                                    and len(raw_frame_buffer) == R3D_CLIP_LENGTH):
                                r3d_start = time.time()
                                r3d_roi = current_action_roi if use_person_detection else None
                                frames_list = list(raw_frame_buffer)

                                # Pretrained R3D (Kinetics-400)
                                if r3d_wrapper is not None:
                                    try:
                                        r3d_logits = r3d_wrapper.predict_from_frames(
                                            frames_list, roi=r3d_roi)
                                        decoder_cache['cuda'] = softmax(r3d_logits)
                                    except Exception as e:
                                        if debug:
                                            print(f"⚠️ R3D inference error: {e}")

                                # Custom R3D (fine-tuned)
                                r3d_custom_info = models_info.get('r3d_custom')
                                if r3d_custom_info is not None:
                                    try:
                                        r3d_custom_logits = r3d_custom_info['wrapper'].predict_from_frames(
                                            frames_list, roi=r3d_roi)
                                        decoder_cache['r3d_custom'] = softmax(r3d_custom_logits)
                                    except Exception as e:
                                        if debug:
                                            print(f"⚠️ R3D custom inference error: {e}")

                                r3d_time += time.time() - r3d_start

                            if interesting_actions_set:
                                # === Run needed decoders for all model mappings ===
                                for key, (action_id, model_type) in action_to_model.items():
                                    action_name = key.split("__")[0]
                                    model_data = models_info.get(model_type)
                                    if model_data is None:
                                        continue

                                    if model_type not in decoder_cache:
                                        if model_data.get('type') == 'openvino':
                                            predictions = model_data['compiled'](
                                                [sequence_array])[model_data['output']].flatten()
                                            decoder_cache[model_type] = softmax(predictions)

                                    probabilities = decoder_cache.get(model_type)
                                    if probabilities is None:
                                        continue
                                    if action_id >= len(probabilities):
                                        continue

                                    score = float(probabilities[action_id])
                                    if score >= confidence_threshold:
                                        writer.writerow([use_timestamp_str, use_frame_id,
                                                         action_id, action_name, score,
                                                         use_timestamp_secs, model_type])
                                        if include_model_type:
                                            all_actions.append((use_timestamp_secs, use_frame_id,
                                                                action_id, score, action_name,
                                                                model_type))
                                        else:
                                            all_actions.append((use_timestamp_secs, use_frame_id,
                                                                action_id, score, action_name))
                                        frame_detections.append((action_name, score, model_type))
                                        detection_count += 1
                                        if debug:
                                            print(f"{use_timestamp_str} -> {action_name} "
                                                  f"[{model_type}] (score:{score:.3f})")
                            else:
                                # === Scan all models (including R3D) ===
                                all_probabilities = {}
                                for model_type, model_data in models_info.items():
                                    if model_data is None:
                                        continue

                                    if model_type not in decoder_cache:
                                        if model_data.get('type') == 'openvino':
                                            predictions = model_data['compiled'](
                                                [sequence_array])[model_data['output']].flatten()
                                            decoder_cache[model_type] = softmax(predictions)
                                        # 'cuda' already handled above

                                    probabilities = decoder_cache.get(model_type)
                                    if probabilities is None:
                                        continue

                                    top_indices = np.argsort(probabilities)[-top_k:][::-1]
                                    for idx in top_indices:
                                        score = float(probabilities[idx])
                                        if score >= confidence_threshold:
                                            action_name = get_action_name(idx, model_type)
                                            key = (action_name, model_type)
                                            all_probabilities[key] = (idx, score, action_name,
                                                                      model_type)

                                sorted_results = sorted(all_probabilities.values(),
                                                        key=lambda x: x[1], reverse=True)
                                for idx, score, action_name, model_type in sorted_results[:top_k]:
                                    writer.writerow([use_timestamp_str, use_frame_id, idx,
                                                     action_name, score, use_timestamp_secs,
                                                     model_type])
                                    if include_model_type:
                                        all_actions.append((use_timestamp_secs, use_frame_id,
                                                            idx, score, action_name, model_type))
                                    else:
                                        all_actions.append((use_timestamp_secs, use_frame_id,
                                                            idx, score, action_name))
                                    frame_detections.append((action_name, score, model_type))
                                    detection_count += 1
                                    if debug:
                                        print(f"{use_timestamp_str} -> {action_name} "
                                              f"[{model_type}] (score:{score:.3f})")

                            if frame_detections:
                                frame_detections.sort(key=lambda x: x[1], reverse=True)
                                recent_detections.clear()
                                recent_detections.extend(frame_detections[:3])

                    prev_req = req
                    prev_timestamp_secs = timestamp_secs
                    prev_frame_id = frame_id
                    processed_frames += 1

                # ---- Periodic GC ----
                current_time = time.time()
                if current_time - last_gc_time > 15.0:
                    gc.collect()
                    last_gc_time = current_time

                # ---- Performance stats ----
                if current_time - last_perf_print > 5.0:
                    total_elapsed = current_time - start_time
                    r3d_info = f" | R3D: {r3d_time:.1f}s" if r3d_wrapper else ""
                    print(f"\n📊 Progress: {frame_id}/{total_frames} frames "
                          f"({frame_id / total_elapsed:.1f} fps) | "
                          f"YOLO: {yolo_time:.1f}s | "
                          f"Preprocess: {preprocess_time:.1f}s | "
                          f"Inference: {inference_time:.1f}s{r3d_info} | "
                          f"Draw: {draw_time:.1f}s")
                    last_perf_print = current_time

                if progress_callback and (current_time - last_gui_update > 0.1):
                    elapsed = current_time - start_time
                    processing_fps = processed_frames / elapsed if elapsed > 0 else 0
                    engine_stats = encoder_engine.get_stats()
                    progress_msg = (
                        f"Frame {processed_frames}/{expected_processed_frames} | "
                        f"Detections: {detection_count} | "
                        f"Processing: {processing_fps:.1f} FPS | "
                        f"Inference: {engine_stats['inference_fps']:.1f} FPS | "
                        f"Device: {actual_device}")
                    if r3d_wrapper:
                        progress_msg += f" + {r3d_model_name}/CUDA"
                    progress_callback(processed_frames, expected_processed_frames,
                                      "Action Recognition", progress_msg)
                    last_gui_update = current_time

            # ---- Flush last frame ----
            if prev_req is not None:
                features = encoder_engine.wait_and_get(prev_req)[0]
                features = np.reshape(features, (-1,))
                sequence_buffer.append(features.copy())

                if len(sequence_buffer) > SEQUENCE_LENGTH:
                    sequence_buffer.pop(0)

                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    sequence_array = np.expand_dims(np.stack(sequence_buffer, axis=0), axis=0)

                    use_timestamp_secs = prev_timestamp_secs
                    use_frame_id = prev_frame_id
                    use_mins, use_secs = divmod(int(use_timestamp_secs), 60)
                    use_timestamp_str = f"{use_mins:02d}:{use_secs:02d}"

                    frame_detections = []
                    decoder_cache = {}

                    # R3D final flush
                    if (r3d_wrapper is not None and raw_frame_buffer is not None
                            and len(raw_frame_buffer) == R3D_CLIP_LENGTH):
                        try:
                            r3d_roi = current_action_roi if use_person_detection else None
                            r3d_logits = r3d_wrapper.predict_from_frames(
                                list(raw_frame_buffer), roi=r3d_roi)
                            decoder_cache['cuda'] = softmax(r3d_logits)
                        except Exception as e:
                            if debug:
                                print(f"⚠️ R3D flush error: {e}")

                    if interesting_actions_set:
                        # === Run needed decoders for all model mappings ===
                        for key, (action_id, model_type) in action_to_model.items():
                            action_name = key.split("__")[0]
                            model_data = models_info.get(model_type)
                            if model_data is None:
                                continue

                            if model_type not in decoder_cache:
                                if model_data.get('type') == 'openvino':
                                    predictions = model_data['compiled'](
                                        [sequence_array])[model_data['output']].flatten()
                                    decoder_cache[model_type] = softmax(predictions)

                            probabilities = decoder_cache.get(model_type)
                            if probabilities is None:
                                continue
                            if action_id >= len(probabilities):
                                continue

                            score = float(probabilities[action_id])
                            if score >= confidence_threshold:
                                writer.writerow([use_timestamp_str, use_frame_id, action_id,
                                                 action_name, score, use_timestamp_secs,
                                                 model_type])
                                if include_model_type:
                                    all_actions.append((use_timestamp_secs, use_frame_id,
                                                        action_id, score, action_name,
                                                        model_type))
                                else:
                                    all_actions.append((use_timestamp_secs, use_frame_id,
                                                        action_id, score, action_name))
                                frame_detections.append((action_name, score, model_type))
                                detection_count += 1
                    else:
                        all_probabilities = {}
                        for model_type, model_data in models_info.items():
                            if model_data is None:
                                continue
                            if model_type not in decoder_cache:
                                if model_data.get('type') == 'openvino':
                                    predictions = model_data['compiled'](
                                        [sequence_array])[model_data['output']].flatten()
                                    decoder_cache[model_type] = softmax(predictions)
                            probabilities = decoder_cache.get(model_type)
                            if probabilities is None:
                                continue
                            top_indices = np.argsort(probabilities)[-top_k:][::-1]
                            for idx in top_indices:
                                score = float(probabilities[idx])
                                if score >= confidence_threshold:
                                    action_name = get_action_name(idx, model_type)
                                    key = (action_name, model_type)
                                    all_probabilities[key] = (idx, score, action_name,
                                                              model_type)
                        sorted_results = sorted(all_probabilities.values(),
                                                key=lambda x: x[1], reverse=True)
                        for idx, score, action_name, model_type in sorted_results[:top_k]:
                            writer.writerow([use_timestamp_str, use_frame_id, idx,
                                             action_name, score, use_timestamp_secs,
                                             model_type])
                            if include_model_type:
                                all_actions.append((use_timestamp_secs, use_frame_id,
                                                    idx, score, action_name, model_type))
                            else:
                                all_actions.append((use_timestamp_secs, use_frame_id,
                                                    idx, score, action_name))
                            frame_detections.append((action_name, score, model_type))
                            detection_count += 1

                    if frame_detections:
                        frame_detections.sort(key=lambda x: x[1], reverse=True)
                        recent_detections.clear()
                        recent_detections.extend(frame_detections[:3])

    finally:
        print("\n🧹 Cleaning up resources...")
        cap.release()
        if video_writer:
            video_writer.release()
            print(f"✅ Annotated video saved: {annotated_output}")
        if yolo_detector:
            yolo_detector.shutdown()
        if person_tracker:
            person_tracker.reset()
        if action_detector:
            action_detector.cleanup()
        if encoder_engine:
            encoder_engine.cleanup()
        if r3d_wrapper:
            r3d_wrapper.cleanup()
        r3d_custom_info = models_info.get('r3d_custom')
        if r3d_custom_info and r3d_custom_info.get('wrapper'):
            r3d_custom_info['wrapper'].cleanup()
        preprocess_pool.shutdown()
        sequence_buffer.clear()
        recent_detections.clear()
        if raw_frame_buffer is not None:
            raw_frame_buffer.clear()
        gc.collect()
        print("✅ Cleanup complete")

    if progress_callback:
        total_time = time.time() - start_time
        engine_stats = encoder_engine.get_stats()
        final_msg = (f"Complete! {detection_count} actions detected | "
                     f"Processed {processed_frames} frames in {total_time:.1f}s | "
                     f"Avg Processing: {processed_frames / total_time:.1f} FPS | "
                     f"Avg Inference: {engine_stats['inference_fps']:.1f} FPS")
        if r3d_wrapper:
            final_msg += f" | R3D time: {r3d_time:.1f}s"
        progress_callback(processed_frames, expected_processed_frames,
                          "Action Recognition Complete", final_msg)

    # ---- Performance summary ----
    print("\n" + "=" * 60)
    print("🏁 PERFORMANCE SUMMARY")
    print("=" * 60)
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.1f}s")
    print(f"Total frames: {frame_id}")
    print(f"Overall FPS: {frame_id / total_time:.1f}")
    print(f"YOLO time: {yolo_time:.1f}s ({yolo_time / total_time * 100:.1f}%)")
    print(f"Preprocess time: {preprocess_time:.1f}s ({preprocess_time / total_time * 100:.1f}%)")
    print(f"Inference time (OV): {inference_time:.1f}s ({inference_time / total_time * 100:.1f}%)")
    if r3d_wrapper:
        print(f"R3D/CUDA time: {r3d_time:.1f}s ({r3d_time / total_time * 100:.1f}%)")
    print(f"Draw time: {draw_time:.1f}s ({draw_time / total_time * 100:.1f}%)")
    print(f"CPU cores: {os.cpu_count()}")
    print(f"OpenVINO threads: {openvino_threads}")
    if r3d_wrapper:
        device_name = torch.cuda.get_device_name(0) if CUDA_AVAILABLE else "CPU"
        print(f"R3D device: {device_name}")
    print(f"Actions detected: {detection_count}")
    print("=" * 60)

    return all_actions


# =============================
# Debug / Analysis Functions
# =============================
def print_top_actions(all_actions, top_n=20):
    sorted_actions = sorted(all_actions, key=lambda x: x[3], reverse=True)
    print(f"\nTop {min(top_n, len(sorted_actions))} actions (by confidence):")
    for i, item in enumerate(sorted_actions[:top_n]):
        if len(item) == 6:
            timestamp, frame_id, action_id, score, action_name, model_type = item
            mins, secs = divmod(int(timestamp), 60)
            print(f"{i + 1:2d}. {mins:02d}:{secs:02d} -> {action_name} "
                  f"[{model_type}] (score:{score:.3f})")
        else:
            timestamp, frame_id, action_id, score, action_name = item
            mins, secs = divmod(int(timestamp), 60)
            print(f"{i + 1:2d}. {mins:02d}:{secs:02d} -> {action_name} (score:{score:.3f})")


def print_most_common_actions(all_actions, top_n=20):
    action_names = [item[4] for item in all_actions]
    counter = Counter(action_names)
    print(f"\nTop {min(top_n, len(counter))} most common actions:")
    for i, (action_name, count) in enumerate(counter.most_common(top_n)):
        print(f"{i + 1:2d}. {action_name} ({count} occurrences)")


def detect_action_sequences(all_actions, score_threshold=0.01, min_duration=1.0):
    sequences = []
    current_seq = None
    for item in all_actions:
        if len(item) == 6:
            timestamp, frame_id, action_id, score, action_name, model_type = item
        else:
            timestamp, frame_id, action_id, score, action_name = item
            model_type = 'unknown'
        if score < score_threshold:
            if current_seq:
                current_seq['end_time'] = timestamp
                sequences.append(current_seq)
                current_seq = None
            continue
        if current_seq and current_seq['action_name'] == action_name:
            current_seq['max_score'] = max(current_seq['max_score'], score)
            current_seq['end_time'] = timestamp
        else:
            if current_seq:
                sequences.append(current_seq)
            current_seq = {'action_name': action_name, 'start_time': timestamp,
                           'end_time': timestamp, 'max_score': score, 'model_type': model_type}
    if current_seq:
        sequences.append(current_seq)
    sequences = [seq for seq in sequences
                 if (seq['end_time'] - seq['start_time']) >= min_duration]
    return sequences


def print_action_sequences(all_actions):
    sequences = detect_action_sequences(all_actions)
    print(f"\nAction sequences detected ({len(sequences)}):")
    for i, seq in enumerate(sequences):
        duration = seq['end_time'] - seq['start_time']
        start_mins, start_secs = divmod(int(seq['start_time']), 60)
        end_mins, end_secs = divmod(int(seq['end_time']), 60)
        model_info = f" [{seq.get('model_type', 'unknown')}]" if 'model_type' in seq else ""
        print(f"{i + 1:2d}. {seq['action_name']}{model_info} Duration: {duration:.1f}s "
              f"({start_mins:02d}:{start_secs:02d} - {end_mins:02d}:{end_secs:02d}) "
              f"Max score: {seq['max_score']:.3f}")


# =============================
# CLI
# =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Action Recognition — OPTIMIZED with R3D/CUDA support")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--device", type=str, default="AUTO", help="OpenVINO device (AUTO, CPU, GPU)")
    parser.add_argument("--sample-rate", type=int, default=5, help="Frame sampling rate")
    parser.add_argument("--log-file", type=str, default="action_log.csv", help="CSV log output")
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument("--show-video", action="store_true", help="Show video preview")
    parser.add_argument("--top-k", type=int, default=10, help="Top K actions to consider")
    parser.add_argument("--confidence", type=float, default=0.01, help="Confidence threshold")
    parser.add_argument("--draw-bboxes", action="store_true",
                        help="Draw bounding boxes on frames")
    parser.add_argument("--annotated-output", type=str,
                        help="Output path for annotated video")
    parser.add_argument("--use-person-detection", action="store_true",
                        help="Enable person detection")
    parser.add_argument("--max-people", type=int, default=2,
                        help="Maximum number of people to track")
    parser.add_argument("--interesting-actions", type=str, nargs="+",
                        help="Specific actions to detect")
    parser.add_argument("--yolo-workers", type=int, default=2,
                        help="Number of parallel YOLO workers")
    parser.add_argument("--yolo-skip", type=int, default=4,
                        help="Skip YOLO detection every N frames")
    parser.add_argument("--downscale-factor", type=float, default=0.5,
                        help="Downscale factor for high-res videos (0.1-1.0)")
    parser.add_argument("--openvino-threads", type=int, default=None,
                        help="OpenVINO inference threads (default: half of CPU cores)")
    parser.add_argument("--preprocess-workers", type=int, default=2,
                        help="Preprocessing thread pool size")
    # ---- R3D / CUDA options ----
    parser.add_argument("--enable-r3d", action="store_true",
                        help="Enable R3D model on CUDA (or CPU fallback)")
    parser.add_argument("--r3d-model", type=str, default="r3d_18",
                        choices=["r3d_18", "mc3_18", "r2plus1d_18"],
                        help="R3D model variant (default: r3d_18)")
    parser.add_argument("--r3d-no-half", action="store_true",
                        help="Disable FP16 for R3D on CUDA (use FP32)")

    args = parser.parse_args()

    print("=" * 60)
    print("🎯 ACTION RECOGNITION — OPTIMIZED + R3D/CUDA")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"OpenVINO device: {args.device}")
    print(f"Person detection: {'ENABLED' if args.use_person_detection else 'DISABLED'}")
    if args.use_person_detection:
        print(f"YOLO workers: {args.yolo_workers}, Skip frames: {args.yolo_skip}")
        print(f"Downscale factor: {args.downscale_factor}")
    print(f"Bounding boxes: {'ENABLED' if args.draw_bboxes else 'DISABLED'}")
    if args.annotated_output:
        print(f"Annotated output: {args.annotated_output}")
    if args.openvino_threads:
        print(f"OpenVINO threads: {args.openvino_threads}")
    print(f"Preprocess workers: {args.preprocess_workers}")
    if args.enable_r3d:
        print(f"R3D model: {args.r3d_model} | FP16: {not args.r3d_no_half}")
    else:
        print("R3D/CUDA: DISABLED")
    print("=" * 60)

    try:
        results = run_action_detection(
            video_path=args.input,
            device=args.device,
            sample_rate=args.sample_rate,
            log_file=args.log_file,
            debug=args.debug,
            top_k=args.top_k,
            confidence_threshold=args.confidence,
            show_video=args.show_video,
            draw_bboxes=args.draw_bboxes,
            annotated_output=args.annotated_output,
            use_person_detection=args.use_person_detection,
            max_people=args.max_people,
            interesting_actions=args.interesting_actions,
            yolo_workers=args.yolo_workers,
            yolo_skip_frames=args.yolo_skip,
            downscale_factor=args.downscale_factor,
            openvino_threads=args.openvino_threads,
            preprocess_workers=args.preprocess_workers,
            enable_r3d=args.enable_r3d,
            r3d_model_name=args.r3d_model,
            r3d_half=not args.r3d_no_half,
        )

        print_top_actions(results)
        print_most_common_actions(results)
        print_action_sequences(results)
        print(f"\n✅ Processing complete. Found {len(results)} action detections.")
    finally:
        gc.collect()
        if CUDA_AVAILABLE:
            torch.cuda.empty_cache()
        print("🧹 Final cleanup complete")