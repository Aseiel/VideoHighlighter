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
        top_tracks = sorted_tracks[:2]
        
        return [(track_id, data['box']) for track_id, data in top_tracks]
    
    def reset(self):
        self.tracks = {}
        self.next_id = 0

def detect_top2_people(frame, detector, tracker=None):
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
        return detected_boxes[:2]
    else:
        tracked = tracker.update(detected_boxes)
        return tracked

def merge_boxes(boxes):
    if len(boxes) == 0:
        return None
    if len(boxes) == 1:
        return boxes[0]

    (x1a, y1a, x2a, y2a) = boxes[0]
    (x1b, y1b, x2b, y2b) = boxes[1]
    return (min(x1a, x1b), min(y1a, y1b), max(x2a, x2b), max(y2a, y2b))

def crop_roi(frame, roi, output_size):
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

    # Avoid empty crop
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
# Configuration (merged)
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
    # Two-phase training: base and finetune settings
    "base_epochs": 25,                  # initial training epochs
    "base_learning_rate": 1e-4,         # initial LR
    "finetune_learning_rate": 1e-5,     # LR when resuming (10x smaller)
    "max_finetune_epochs": 15,          # safety ceiling for fine-tuning additional epochs
    "early_stopping_patience": 3,       # patience for early stopping (during finetune or both)
    "min_delta": 0.001,                 # minimal improvement in val loss to reset patience

    # previously used keys kept for compatibility
    "sequence_length": 16,
    "crop_size": (224, 224),
    "model_save_path": "intel_finetuned_classifier_3d.pth",
    "checkpoint_path": r"D:\movie_highlighter\checkpoints\checkpoint_latest.pth",  # for resume
    "save_checkpoint_every": 5,  # Save checkpoint every N epochs
    "checkpoint_dir": "checkpoints",  # Directory for checkpoint saves
    "min_train_per_action": 5,
    "min_val_per_action": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "create_visualizations": False,  # Set to True to create sample videos
    "num_visualization_samples": 2,
    "visualization_sample_rate": 5,
    "use_roi_smoothing": True,
}

BASE_DIR = os.getcwd()
ENCODER_XML = os.path.join(BASE_DIR, "models/intel_action/encoder/FP32/action-recognition-0001-encoder.xml")
ENCODER_BIN = os.path.join(BASE_DIR, "models/intel_action/encoder/FP32/action-recognition-0001-encoder.bin")

# =============================
# Video Loading
# =============================
def load_video_normalized(path, is_training=True):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        return []

    indices = np.linspace(0, total_frames - 1, CONFIG["sequence_length"]).astype(int)
    
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
        
        if person_tracker:
            tracked = detect_top2_people(frame, yolo_people, person_tracker)
            # tracked returns [(id, box), ...] when using tracker
            if tracked and isinstance(tracked[0], tuple) and len(tracked[0]) == 2 and isinstance(tracked[0][1], tuple):
                people = [box for _, box in tracked]
            else:
                people = tracked
        else:
            people = detect_top2_people(frame, yolo_people)
        
        merged_roi = merge_boxes(people)
        
        if roi_smoother:
            roi = roi_smoother.update(merged_roi)
        else:
            roi = merged_roi
        
        frame = crop_roi(frame, roi, crop_size)
        frame = frame.astype(np.float32) / 255.0
        mean = np.array(CONFIG["mean"], dtype=np.float32)
        std = np.array(CONFIG["std"], dtype=np.float32)
        frame = (frame - mean) / std
        
        output_frames.append(frame)

    cap.release()
    
    if len(output_frames) == 0:
        return []
    
    # If fewer than sequence_length frames were read, pad by repeating last frame
    if len(output_frames) < CONFIG["sequence_length"]:
        last = output_frames[-1]
        while len(output_frames) < CONFIG["sequence_length"]:
            output_frames.append(last.copy())

    return np.stack(output_frames, axis=0)

# =============================
# Dataset
# =============================
class VideoDataset(Dataset):
    def __init__(self, root, sequence_length=16, is_training=True):
        self.video_samples = []
        self.sequence_length = sequence_length
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
        frames = load_video_normalized(video_path, self.is_training)
        
        if len(frames) == 0:
            frames = np.zeros((self.sequence_length, CONFIG["crop_size"][0], CONFIG["crop_size"][1], 3), dtype=np.float32)
        
        frames = np.transpose(frames, (0, 3, 1, 2))
        return torch.tensor(frames, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def get_label_mapping(self):
        return self.label_to_idx, self.idx_to_label

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
        Returns: torch.FloatTensor of encoded features shape (B, T * feat_dim_per_frame) or (B, T, feat_dim)
        """
        if isinstance(frames_batch, torch.Tensor):
            frames_batch = frames_batch.cpu().numpy()

        B, T, C, H, W = frames_batch.shape
        feats = []
        
        for batch_idx in range(B):
            batch_feats = []
            for time_idx in range(T):
                frame = frames_batch[batch_idx, time_idx]
                # frame shape (C,H,W) with channels normalized already
                frame_batch = np.expand_dims(frame, axis=0)  # (1, C, H, W)
                frame_batch = self._preprocess_batch(frame_batch)  # preprocess to expected input
                # OpenVINO compiled model inference expects a dict or sequence; older API accepted list
                out = self.encoder([frame_batch])
                # out is a dictionary-like mapping; fetch first output
                # Safely get output tensor
                try:
                    output_node = self.encoder.output(0)
                    feat = out[output_node]
                except Exception:
                    # fallback: take first value
                    feat = list(out.values())[0]
                
                if feat.ndim > 1:
                    feat = feat.reshape(feat.shape[0], -1)  # (1, feat_dim)
                batch_feats.append(feat)
            
            batch_feats = np.concatenate(batch_feats, axis=0)  # (T, feat_dim)
            feats.append(batch_feats)
        
        feats = np.stack(feats, axis=0)  # (B, T, feat_dim)
        # Convert to torch tensor
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
    def __init__(self, feature_dim=1024, hidden_dim=512, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
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
    """Validate model on validation set and return loss and accuracy"""
    model.eval()
    total_correct = 0
    total_samples = 0
    running_loss = 0.0
    
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
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    avg_loss = running_loss / total_samples if total_samples > 0 else float('inf')
    return avg_loss, accuracy

def train_classifier(encoder, train_loader, val_loader, num_classes, label_to_idx, idx_to_label):
    device = torch.device(CONFIG["device"])
    
    # Determine feature dimension using a sample
    with torch.no_grad():
        sample_frames, _ = next(iter(train_loader))
        dummy_feats = encoder.encode(sample_frames[0:1].cpu())
        feature_dim = dummy_feats.shape[-1]
    
    print(f"Feature dimension: {feature_dim}")
    
    # Initialize model & criterion
    model = EncoderLSTM(feature_dim=feature_dim, hidden_dim=512, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Decide if resuming
    is_resuming = CONFIG.get('checkpoint_path') and os.path.exists(CONFIG['checkpoint_path'])
    
    # Choose LR and max epochs based on resume status
    if is_resuming:
        lr = CONFIG['finetune_learning_rate']
        print(f"🔄 Resume detected: using finetune LR {lr}")
    else:
        lr = CONFIG['base_learning_rate']
        print(f"🆕 Training from scratch: using base LR {lr}")
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Try to load checkpoint if resuming
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
    
    # Determine max_epochs
    if is_resuming:
        # When resuming: run at most start_epoch + max_finetune_epochs (so we run additional fine-tune epochs)
        max_epochs = start_epoch + CONFIG.get('max_finetune_epochs', 15)
        print(f"   Fine-tuning mode: starting at epoch {start_epoch}, will run up to epoch {max_epochs}")
    else:
        max_epochs = CONFIG.get('base_epochs', 25)
        print(f"   Fresh training mode: will run up to epoch {max_epochs}")
    
    patience_counter = 0

    # Training loop
    for epoch in range(start_epoch, max_epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")
        for frames, labels in pbar:
            frames, labels = frames.to(device), labels.to(device)
            
            # Encode features (encoder works on numpy/CPU so send frames.cpu())
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

        print(f"\n📊 Epoch {epoch+1}/{max_epochs}")
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validation step
        val_loss = float('inf')
        val_acc = 0.0
        if len(val_loader) > 0:
            val_loss, val_acc = validate_classifier(encoder, model, val_loader, device, criterion)
            print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Early stopping logic (based on validation loss)
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

        # Periodic checkpoint save
        if CONFIG.get('save_checkpoint_every') and (epoch + 1) % CONFIG['save_checkpoint_every'] == 0:
            checkpoint_name = f"checkpoint_epoch_{epoch+1}.pth"
            checkpoint_path = os.path.join(CONFIG.get('checkpoint_dir', '.'), checkpoint_name)
            save_checkpoint(model, optimizer, epoch, best_val_acc, 
                          label_to_idx, idx_to_label, feature_dim, checkpoint_path, best_val_loss)

        # Save latest checkpoint
        if CONFIG.get('checkpoint_dir'):
            latest_checkpoint = os.path.join(CONFIG['checkpoint_dir'], 'checkpoint_latest.pth')
            save_checkpoint(model, optimizer, epoch, best_val_acc, 
                          label_to_idx, idx_to_label, feature_dim, latest_checkpoint, best_val_loss)
    
    # Load best model if available
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"\n✅ Loaded best model (val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})")

    # Save final model and mapping
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
    print("🎯 ACTION RECOGNITION TRAINING")
    print("=" * 60)
    
    # Display training mode intention
    if CONFIG.get('checkpoint_path'):
        print(f"\n📂 Config checkpoint path: {CONFIG['checkpoint_path']}")
    print(f"   Device: {CONFIG['device']}")
    print(f"   Batch size: {CONFIG['batch_size']}")
    print("=" * 60)
    
    # Load datasets
    print("\n📁 Loading datasets...")
    train_dataset = VideoDataset(os.path.join(CONFIG['data_path'], "train"), is_training=True)
    val_dataset = VideoDataset(os.path.join(CONFIG['data_path'], "val"), is_training=False)
    
    print(f"\n📊 Dataset statistics:")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Number of classes: {len(train_dataset.labels)}")
    
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
