import os
import glob
import cv2
import json
import random
import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from openvino.runtime import Core
from tqdm import tqdm

# =============================
# How to
# =============================
# Put short 5-10s videos to dataset/train/{action_name} and dataset/val/{action_name}
# Usually You should put minimum 100+ videos for train and 20+ for val

# =============================
# Set Random Seed for Reproducibility
# =============================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================
# Configuration
# =============================
CONFIG = {
    "data_path": "dataset",
    "batch_size": 2,
    "learning_rate": 1e-4,
    "num_epochs": 25,
    "sequence_length": 16,
    "frame_size": (256, 256),
    "crop_size": (224, 224),
    "fps_target": 30,
    "center_jitter": 0.1,
    "target_fps": 15,
    "clip_stride": 8,
    "model_save_path": "intel_finetuned_classifier_3d.pth",
    "label_mapping_save_path": "label_mapping.json",
    "early_stopping_patience": 5,
    "min_train_per_action": 5,
    "min_val_per_action": 0,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# =============================
# Intel OpenVINO model paths
# =============================
BASE_DIR = os.getcwd()
ENCODER_XML = os.path.join(BASE_DIR, "models/intel_action/encoder/FP32/action-recognition-0001-encoder.xml")
ENCODER_BIN = os.path.join(BASE_DIR, "models/intel_action/encoder/FP32/action-recognition-0001-encoder.bin")

# =============================
# VIDEO PREPROCESSING HELPERS
# =============================
def center_crop(frame, crop_size=(224, 224), jitter=0.1):
    """Crop around center with slight random jitter."""
    h, w, _ = frame.shape
    ch, cw = crop_size
    jh = int(jitter * (h - ch))
    jw = int(jitter * (w - cw))
    y = max(0, h // 2 - ch // 2 + random.randint(-jh, jh))
    x = max(0, w // 2 - cw // 2 + random.randint(-jw, jw))
    return frame[y:y+ch, x:x+cw]

def load_video_normalized(path, is_training=True):
    """Load video, normalize FPS, resize, and crop center with jitter."""
    cap = cv2.VideoCapture(path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or CONFIG["fps_target"]
    frame_skip = max(1, int(round(orig_fps / CONFIG["fps_target"])))

    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_skip == 0:
            frame = cv2.resize(frame, CONFIG["frame_size"])
            if is_training:
                frame = center_crop(frame, CONFIG["crop_size"], CONFIG["center_jitter"])
            else:
                # Center crop without jitter for validation
                h, w, _ = frame.shape
                ch, cw = CONFIG["crop_size"]
                y = max(0, h // 2 - ch // 2)
                x = max(0, w // 2 - cw // 2)
                frame = frame[y:y+ch, x:x+cw]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        idx += 1

    cap.release()
    return np.array(frames)

# =============================
# VideoDataset
# =============================
class VideoDataset(Dataset):
    def __init__(self, root, sequence_length=16, target_fps=15, stride=8, is_training=True):
        self.video_samples = []  # list of (video_path, label)
        self.sequence_length = sequence_length
        self.target_fps = target_fps
        self.stride = stride
        self.is_training = is_training

        if not os.path.exists(root):
            print(f"Warning: Dataset path {root} does not exist")
            self.labels = []
            self.label_to_idx = {}
            self.idx_to_label = {}
            self.samples = []  # legacy support
            return

        # First pass: collect all class folders
        class_folders = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        
        # Create label mapping from folder names
        self.label_to_idx = {label: idx for idx, label in enumerate(class_folders)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.labels = class_folders
        
        print(f"📊 Detected {len(self.labels)} action classes:")
        for label, idx in self.label_to_idx.items():
            print(f"  {idx}: {label}")

        # Second pass: collect videos with correct labels
        video_count = 0
        for label in self.labels:
            label_path = os.path.join(root, label)
            video_files = glob.glob(os.path.join(label_path, "*.mp4")) + \
                          glob.glob(os.path.join(label_path, "*.avi")) + \
                          glob.glob(os.path.join(label_path, "*.mov"))
            for video_path in video_files:
                self.video_samples.append((video_path, self.label_to_idx[label]))
                video_count += 1

        print(f"✅ Found {video_count} videos in {root} across {len(self.labels)} actions")
        print(f"✅ Label index range: 0 to {len(self.labels) - 1}")

        # ✅ Legacy support for old code
        self.samples = self.video_samples

    def __len__(self):
        return len(self.video_samples)

    def __getitem__(self, idx):
        video_path, label = self.video_samples[idx]
        
        # Load video with our improved preprocessing
        frames = load_video_normalized(video_path, self.is_training)
        
        # Generate multiple clips from the video
        clips = self._generate_clips(frames)
        
        if self.is_training:
            # During training: randomly select one clip for data augmentation
            selected_clip = random.choice(clips) if clips else None
        else:
            # During validation: use the center clip for consistency
            selected_clip = clips[len(clips) // 2] if len(clips) > 0 else clips[0] if clips else None
        
        if selected_clip is None:
            # Create a dummy clip if no clips were generated
            selected_clip = np.zeros((self.sequence_length, CONFIG["crop_size"][0], CONFIG["crop_size"][1], 3), dtype=np.float32)
        
        # Convert to tensor format
        selected_clip = np.transpose(selected_clip, (0, 3, 1, 2))  # TCHW format
        return torch.tensor(selected_clip, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def _generate_clips(self, frames):
        """Split frames into overlapping clips of sequence_length"""
        clips = []
        total_frames = len(frames)
        
        if total_frames < self.sequence_length:
            # If video is too short, pad with last frame
            padded_frames = np.zeros((self.sequence_length, *frames.shape[1:]), dtype=frames.dtype)
            padded_frames[:total_frames] = frames
            padded_frames[total_frames:] = frames[-1]
            clips.append(padded_frames)
        else:
            # Generate overlapping clips
            for start_idx in range(0, total_frames - self.sequence_length + 1, self.stride):
                end_idx = start_idx + self.sequence_length
                clip = frames[start_idx:end_idx]
                clips.append(clip)
            
            # Ensure we include the last frames
            if (total_frames - self.sequence_length) % self.stride != 0:
                last_clip = frames[-self.sequence_length:]
                clips.append(last_clip)
        
        return clips

    def get_label_mapping(self):
        return self.label_to_idx, self.idx_to_label

    def analyze_class_distribution(self):
        """Analyze the distribution of videos per class"""
        print(f"\n📊 Class Distribution Analysis:")
        class_counts = {label: 0 for label in self.labels}
        
        for _, label in self.video_samples:
            class_name = self.idx_to_label[label]
            class_counts[class_name] += 1
        
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {count} videos")
        
        return class_counts

# =============================
# Dataset validation function
# =============================
def validate_dataset_size(train_dataset, val_dataset):
    """Check if dataset has minimum required videos per action"""
    min_train = CONFIG['min_train_per_action']
    min_val = CONFIG['min_val_per_action']
    
    print(f"\n📊 Validating dataset size...")
    print(f"  Minimum training videos per action: {min_train}")
    print(f"  Minimum validation videos per action: {min_val}")
    
    # Analyze class distributions
    train_counts = train_dataset.analyze_class_distribution()
    val_counts = val_dataset.analyze_class_distribution()
    
    valid_actions = []
    for action in train_dataset.labels:
        train_count = train_counts.get(action, 0)
        val_count = val_counts.get(action, 0)
        
        if train_count >= min_train and val_count >= min_val:
            print(f"  ✅ Action '{action}': {train_count} train, {val_count} val")
            valid_actions.append(action)
        else:
            if train_count < min_train:
                print(f"  ⚠️  Action '{action}': {train_count} train videos (need {min_train}) - SKIPPED")
            if val_count < min_val:
                print(f"  ⚠️  Action '{action}': {val_count} val videos (need {min_val}) - SKIPPED")
    
    if len(valid_actions) == 0:
        print("\n❌ No actions meet minimum requirements. Please collect more videos.")
        return False, []
    
    print(f"\n✅ Dataset validation passed! Training with {len(valid_actions)} actions:")
    for action in valid_actions:
        train_count = train_counts.get(action, 0)
        val_count = val_counts.get(action, 0)
        print(f"   - {action}: {train_count} train, {val_count} val")
    
    return True, valid_actions

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

        print(f"Encoder input name: {self.input_name}")
        print(f"Encoder input shape: {self.input_shape}")

    def encode(self, frames_batch):
        """frames_batch: [B, T, C, H, W] float32 tensor in range 0-1"""
        B, T, C, H, W = frames_batch.shape
        feats = []
        
        # Process each batch item and time step separately
        for batch_idx in range(B):
            batch_feats = []
            for time_idx in range(T):
                # Get single frame: [C, H, W]
                frame = frames_batch[batch_idx, time_idx].numpy()  # [C, H, W]
                
                # Add batch dimension: [1, C, H, W]
                frame_batch = np.expand_dims(frame, axis=0)
                
                # Preprocess for Intel model
                frame_batch = self._preprocess_batch(frame_batch)
                
                # Run inference - model expects [1, 3, 224, 224]
                out = self.encoder([frame_batch])
                feat = out[self.encoder.output(0)]
                
                # Flatten the feature if it's not 1D
                if feat.ndim > 1:
                    feat = feat.reshape(feat.shape[0], -1)  # [1, feature_dim]
                batch_feats.append(feat)
            
            # Stack time steps for this batch item: [T, feature_dim]
            batch_feats = np.concatenate(batch_feats, axis=0)  # [T, feature_dim]
            feats.append(batch_feats)
        
        # Stack batch items: [B, T, feature_dim]
        feats = np.stack(feats, axis=0)
        return torch.tensor(feats, dtype=torch.float32)

    def _preprocess_batch(self, batch_frames):
        """Preprocess batch of frames for Intel model"""
        # batch_frames: [1, C, H, W] in range 0-1
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
        
        # Denormalize from [0,1] to [0,255] then apply model normalization
        batch_frames = batch_frames * 255.0
        batch_frames = (batch_frames / 255.0 - mean) / std
        return batch_frames.astype(np.float32)

# =============================
# SIMPLIFIED LSTM MODEL
# =============================
class EncoderLSTM(nn.Module):
    def __init__(self, feature_dim=1024, hidden_dim=512, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x should be [batch_size, sequence_length, feature_dim]
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take last frame output
        return self.fc(out)

# =============================
# Model Manager
# =============================
class ActionRecognitionModel:
    def __init__(self, model, label_to_idx, idx_to_label, feature_dim, sequence_length):
        self.model = model
        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.num_classes = len(label_to_idx)
        
        print(f"✅ Model initialized with {self.num_classes} classes")
        print(f"✅ Label range: 0 to {self.num_classes - 1}")
    
    def save(self, path):
        """Save both model and label mapping"""
        torch.save(self.model.state_dict(), path)
        
        mapping_path = path.replace('.pth', '_mapping.json')
        mapping_data = {
            'label_to_idx': self.label_to_idx,
            'idx_to_label': self.idx_to_label,
            'feature_dim': self.feature_dim,
            'sequence_length': self.sequence_length,
            'num_classes': self.num_classes
        }
        with open(mapping_path, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        print(f"✓ Model saved to {path}")
        print(f"✓ Label mapping saved to {mapping_path}")
        print(f"✓ Classes: {list(self.label_to_idx.keys())}")
    
    @classmethod
    def load(cls, model_path, device='cpu'):
        """Load both model and label mapping"""
        mapping_path = model_path.replace('.pth', '_mapping.json')
        
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Label mapping file not found: {mapping_path}")
        
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)
        
        model = EncoderLSTM(
            feature_dim=mapping_data['feature_dim'],
            hidden_dim=512,
            num_classes=mapping_data['num_classes']
        )
        
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        idx_to_label = {int(k): v for k, v in mapping_data['idx_to_label'].items()}
        
        print(f"✅ Model loaded with {mapping_data['num_classes']} classes")
        print(f"✅ Classes: {list(mapping_data['label_to_idx'].keys())}")
        
        return cls(model, mapping_data['label_to_idx'], idx_to_label, 
                  mapping_data['feature_dim'], mapping_data['sequence_length'])

# =============================
# TRAINING FUNCTION
# =============================
def train_classifier(encoder, train_loader, val_loader, num_classes, label_to_idx, idx_to_label):
    device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")

    # Determine feature dimension by processing a single sample
    with torch.no_grad():
        sample_frames, sample_labels = next(iter(train_loader))
        single_frames = sample_frames[0:1]  # Take first sample only [1, T, C, H, W]
        dummy_feats = encoder.encode(single_frames.cpu())
        feature_dim = dummy_feats.shape[-1]
        print(f"Determined feature_dim: {feature_dim}")

    # Calculate class weights for imbalanced datasets
    class_counts = [0] * num_classes
    for _, label in train_loader.dataset.samples:
        if label < num_classes:  # Only count valid labels
            class_counts[label] += 1
    
    print(f"\n📊 Class distribution: {class_counts}")
    
    # Check for severely imbalanced classes
    min_samples = min(class_counts)
    max_samples = max(class_counts)
    if min_samples < 5:
        print(f"⚠️  WARNING: Some classes have very few samples (min={min_samples})!")
        print(f"⚠️  Consider collecting more data for better results.")
    if max_samples / (min_samples + 1e-6) > 5:
        print(f"⚠️  WARNING: Severe class imbalance detected (ratio: {max_samples}/{min_samples})!")
    
    # Calculate weights (inverse frequency)
    class_counts = np.array(class_counts, dtype=np.float32)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.max()  # normalize scale only

    print(f"📊 Class weights (for balancing): {[f'{w:.4f}' for w in class_weights.tolist()]}\n")

    model = EncoderLSTM(
        feature_dim=feature_dim,
        hidden_dim=512,
        num_classes=num_classes
    ).to(device)
    
    # Use weighted loss for imbalanced classes
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        
        for frames, labels in pbar:
            frames, labels = frames.to(device), labels.to(device)
            
            # Extract features using batch method
            with torch.no_grad():
                feats = encoder.encode(frames.cpu())
            
            # Ensure features are on the right device
            feats = feats.to(device)
            
            # Forward pass
            outputs = model(feats)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            preds = outputs.argmax(1)
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_samples += labels.size(0)
            
            running_loss += loss.item() * frames.size(0)
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct/labels.size(0):.4f}'
            })

        # Epoch statistics
        if total_samples > 0:
            avg_loss = running_loss / total_samples
            epoch_acc = total_correct / total_samples
            print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, Acc: {epoch_acc:.4f}")

            # Validation
            if len(val_loader) > 0:
                val_acc = validate_classifier(encoder, model, val_loader, device, idx_to_label)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                    print(f"  ✓ New best validation accuracy: {val_acc:.4f}")
                else:
                    patience_counter += 1
                    print(f"  No improvement ({patience_counter}/{CONFIG['early_stopping_patience']})")
                    
                    if patience_counter >= CONFIG['early_stopping_patience']:
                        print(f"\n⏹️  Early stopping triggered at epoch {epoch+1}")
                        print(f"  Best validation accuracy: {best_val_acc:.4f}")
                        break

    # Restore best model
    if best_model_state is not None:
        print(f"✓ Restoring best model (val_acc={best_val_acc:.4f})")
        model.load_state_dict(best_model_state)

    model.eval()

    # Wrap & save
    action_model = ActionRecognitionModel(
        model=model,
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label,
        feature_dim=feature_dim,
        sequence_length=CONFIG['sequence_length']
    )
    action_model.save(CONFIG['model_save_path'])
    return action_model

# =============================
# VALIDATION FUNCTION
# =============================
def validate_classifier(encoder, model, val_loader, device, idx_to_label):
    """Simplified validation function"""
    model.eval()
    torch.set_grad_enabled(False)
    
    total_correct = 0
    total_samples = 0
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for frames, labels in val_loader:
            frames, labels = frames.to(device), labels.to(device)
            
            # Extract features
            feats = encoder.encode(frames.cpu())
            
            # Predict
            outputs = model(feats.to(device))
            preds = outputs.argmax(1)
            
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            # Per-class accuracy
            for label, pred in zip(labels, preds):
                label_item = label.item()
                class_total[label_item] = class_total.get(label_item, 0) + 1
                if label == pred:
                    class_correct[label_item] = class_correct.get(label_item, 0) + 1
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"  Validation Accuracy: {accuracy:.4f} ({total_correct}/{total_samples})")
    
    # Print per-class accuracy
    if class_total:
        print(f"  Per-class accuracy:")
        for idx in sorted(class_total.keys()):
            correct = class_correct.get(idx, 0)
            total = class_total[idx]
            acc = correct / total if total > 0 else 0
            label_name = idx_to_label[idx]
            print(f"    {label_name} (class {idx}): {acc:.4f} ({correct}/{total})")
    
    torch.set_grad_enabled(True)
    return accuracy

# =============================
# Main with AUTO CLASS DETECTION
# =============================
if __name__ == "__main__":
    set_seed(42)
    print("✓ Random seed set to 42 for reproducibility\n")
    
    if not os.path.exists(ENCODER_XML) or not os.path.exists(ENCODER_BIN):
        print(f"Error: Intel model files not found at:")
        print(f"  XML: {ENCODER_XML}")
        print(f"  BIN: {ENCODER_BIN}")
        exit(1)

    # Initialize datasets - they will auto-detect the correct number of classes
    train_dataset = VideoDataset(os.path.join(CONFIG['data_path'], "train"), is_training=True)
    val_dataset = VideoDataset(os.path.join(CONFIG['data_path'], "val"), is_training=False)
    
    if len(train_dataset.samples) == 0:
        print("No training samples found! Please check your dataset structure.")
        exit(1)
        
    print(f"\n📁 Dataset Information:")
    print(f"  Training samples:   {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Total classes detected: {len(train_dataset.labels)}")
    print()
    
    is_valid, valid_actions = validate_dataset_size(train_dataset, val_dataset)
    if not is_valid:
        print("\n❌ Training aborted due to insufficient data.")
        exit(1)
    
    # Use ALL detected classes (no filtering needed since we auto-detect correctly)
    label_to_idx, idx_to_label = train_dataset.get_label_mapping()
    
    print(f"\n🎯 Final training configuration:")
    print(f"  Number of classes: {len(train_dataset.labels)}")
    print(f"  Training samples: {len(train_dataset.samples)}")
    print(f"  Validation samples: {len(val_dataset.samples)}")
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    encoder = IntelFeatureExtractor(ENCODER_XML, ENCODER_BIN)
    
    print(f"\n🚀 Starting training...\n")
    action_model = train_classifier(encoder, train_loader, val_loader, 
                                  num_classes=len(train_dataset.labels),
                                  label_to_idx=label_to_idx,
                                  idx_to_label=idx_to_label)
    
    if len(val_loader) > 0:
        print(f"\n📊 Final Validation:")
        device = torch.device(CONFIG["device"])
        validate_classifier(encoder, action_model.model, val_loader, device, idx_to_label)
    
    print(f"\n✅ Training completed! Model and labels saved.")
    print(f"✓ Model path: {CONFIG['model_save_path']}")