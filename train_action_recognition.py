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
# How to
# =============================
# put short 5-10s videos to dataset/train/action_name and dataset/val/action_name
# usually You should put minimum 100+ videos for train and 20+ for val


# =============================
# Configuration
# =============================
CONFIG = {
    "data_path": "dataset",
    "batch_size": 2,           # small batch for CPU
    "learning_rate": 1e-3,
    "num_epochs": 20,          # Increased from 5 to 20
    "sequence_length": 16,
    "frame_size": (224, 224),  # Intel encoder expects 224x224
    "model_save_path": "intel_finetuned_classifier_3d.pth",
    "label_mapping_save_path": "label_mapping.json",
    "early_stopping_patience": 5,  # Early stopping
    "min_train_per_action": 5,     # Minimum videos per action in training
    "min_val_per_action": 0,       # Minimum videos per action in validation
}

# =============================
# Intel OpenVINO model paths
# =============================
BASE_DIR = os.getcwd()
ENCODER_XML = os.path.join(BASE_DIR, "models/intel_action/encoder/FP32/action-recognition-0001-encoder.xml")
ENCODER_BIN = os.path.join(BASE_DIR, "models/intel_action/encoder/FP32/action-recognition-0001-encoder.bin")
SEQUENCE_LENGTH = CONFIG['sequence_length']

# =============================
# Dataset 
# =============================
class VideoDataset(Dataset):
    def __init__(self, root, sequence_length=16):
        self.samples = []
        self.sequence_length = sequence_length
        if not os.path.exists(root):
            print(f"Warning: Dataset path {root} does not exist")
            self.labels = []
            self.label_to_idx = {}
            return
            
        self.labels = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        for label in self.labels:
            label_path = os.path.join(root, label)
            video_files = glob.glob(os.path.join(label_path, "*.mp4")) + \
                          glob.glob(os.path.join(label_path, "*.avi")) + \
                          glob.glob(os.path.join(label_path, "*.mov"))
            for video_path in video_files:
                self.samples.append((video_path, self.label_to_idx[label]))
        
        print(f"Found {len(self.samples)} samples in {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._load_video(video_path)
        frames = self._sample_frames(frames)
        frames = np.array(frames, dtype=np.float32)
        return frames, label

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, CONFIG['frame_size'])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        return frames

    def _sample_frames(self, frames):
        total = len(frames)
        if total >= self.sequence_length:
            indices = np.linspace(0, total-1, self.sequence_length, dtype=int)
            return [frames[i] for i in indices]
        else:
            last = frames[-1]
            return frames + [last] * (self.sequence_length - total)

    def get_label_mapping(self):
        """Return label to index and index to label mappings"""
        return self.label_to_idx, self.idx_to_label

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
    
    # Count videos per action in training set
    train_counts = {}
    for _, label in train_dataset.samples:
        action = train_dataset.idx_to_label[label]
        train_counts[action] = train_counts.get(action, 0) + 1
    
    # Count videos per action in validation set
    val_counts = {}
    for _, label in val_dataset.samples:
        action = val_dataset.idx_to_label[label]
        val_counts[action] = val_counts.get(action, 0) + 1
    
    # Check thresholds and filter valid actions
    valid_actions = []
    for action in train_dataset.labels:
        train_count = train_counts.get(action, 0)
        val_count = val_counts.get(action, 0)  # Use get() to avoid KeyError
        
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
        # Use get() to safely access counts
        train_count = train_counts.get(action, 0)
        val_count = val_counts.get(action, 0)  # This will now work even if action is missing from val_counts
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
        
        # ✅ Fix: extract proper input name and shape
        input_tensor = self.encoder.inputs[0]
        self.input_name = input_tensor.get_any_name()
        self.input_shape = list(input_tensor.get_shape())

        print(f"Encoder input name: {self.input_name}")
        print(f"Encoder input shape: {self.input_shape}")


    def preprocess_frame(self, frame):
        """
        frame: HxWxC, dtype=float32, 0..255
        returns: (3, H, W) float32 aligned to model input
        """
        # Ensure float32 and shape HWC
        frame = frame.astype(np.float32)

        # Normalize — model expects mean/std in pixel scale
        mean = np.array([0.485, 0.456, 0.406]) * 255.0
        std = np.array([0.229, 0.224, 0.225]) * 255.0
        # broadcasting over H,W,C
        frame = (frame - mean) / std

        # HWC -> CHW
        frame = frame.transpose(2, 0, 1)
        return frame

    def extract_sequence(self, frames):
        """
        frames: np.ndarray of shape (T, H, W, C) or list of frames
        returns: torch.Tensor of shape (T, feature_dim)
        """
        features = []
        seen_shape = None

        for i, frame in enumerate(frames):
            inp = self.preprocess_frame(frame)  # (3,H,W)
            inp = np.expand_dims(inp, axis=0).astype(np.float32)  # (1,3,H,W)

            # Run inference
            result = self.encoder({self.input_name: inp})

            # Get the output
            out_tensor = list(result.values())[0]

            # Convert to NumPy if it is a ConstOutput
            if hasattr(out_tensor, "numpy"):
                out_np = out_tensor.numpy()
            else:
                out_np = np.array(out_tensor)

            out_np = np.squeeze(out_np)  # remove batch dimensions

            # Flatten to 1D vector
            if out_np.ndim == 0:
                out_np = out_np.reshape(1)
            elif out_np.ndim > 1:
                out_np = out_np.ravel()

            # Check consistency
            if seen_shape is None:
                seen_shape = out_np.shape
                print(f"[extract_sequence] per-frame feature shape determined: {seen_shape}")
            else:
                if out_np.shape != seen_shape:
                    raise RuntimeError(f"Inconsistent feature shape at frame {i}: {out_np.shape} != {seen_shape}")

            features.append(torch.from_numpy(out_np.astype(np.float32)))

        return torch.stack(features, dim=0)  # (T, feature_dim)



# =============================
# Sequence-aware Classifier
# =============================
class SequenceActionClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes, sequence_length=16):
        super().__init__()
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        
        # Use LSTM to capture temporal patterns
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=256,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, feature_dim)
        
        # LSTM processing
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Use the last hidden state for classification
        out = self.classifier(hidden[-1])
        return out

# =============================
# Model Manager to handle labels and model together
# =============================
class ActionRecognitionModel:
    def __init__(self, model, label_to_idx, idx_to_label, feature_dim, sequence_length):
        self.model = model
        self.label_to_idx = label_to_idx
        self.idx_to_label = idx_to_label
        self.feature_dim = feature_dim
        self.sequence_length = sequence_length
        self.num_classes = len(label_to_idx)
    
    def save(self, path):
        """Save both model and label mapping"""
        # Save model state
        torch.save(self.model.state_dict(), path)
        
        # Save label mapping as JSON
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
    
    @classmethod
    def load(cls, model_path, device='cpu'):
        """Load both model and label mapping"""
        mapping_path = model_path.replace('.pth', '_mapping.json')
        
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Label mapping file not found: {mapping_path}")
        
        # Load label mapping
        with open(mapping_path, 'r') as f:
            mapping_data = json.load(f)
        
        # Recreate model architecture
        model = SequenceActionClassifier(
            feature_dim=mapping_data['feature_dim'],
            num_classes=mapping_data['num_classes'],
            sequence_length=mapping_data['sequence_length']
        )
        
        # Load model weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        # Convert string keys back to integers for idx_to_label
        idx_to_label = {int(k): v for k, v in mapping_data['idx_to_label'].items()}
        
        return cls(model, mapping_data['label_to_idx'], idx_to_label, 
                  mapping_data['feature_dim'], mapping_data['sequence_length'])
    
    def predict(self, encoder, frames_sequence):
        """Predict action for a sequence of frames"""
        self.model.eval()
        with torch.no_grad():
            # Extract features
            features = encoder.extract_sequence(frames_sequence)  # (T, feature_dim)
            features = features.unsqueeze(0)  # Add batch dimension: (1, T, feature_dim)
            
            # Predict
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_idx = outputs.argmax(1).item()
            confidence = probabilities[0][predicted_idx].item()
            
            predicted_label = self.idx_to_label[predicted_idx]
            
            return {
                'action': predicted_label,
                'action_id': predicted_idx,
                'confidence': confidence,
                'probabilities': {
                    self.idx_to_label[i]: prob.item() 
                    for i, prob in enumerate(probabilities[0])
                }
            }

# =============================
# Training loop (IMPROVED)
# =============================
def train_classifier(encoder, train_loader, val_loader, num_classes, label_to_idx, idx_to_label):
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Get feature_dim and sequence_length from a single dummy clip
    with torch.no_grad():
        dummy_frames_batch, _ = next(iter(train_loader))
        print(f"Dummy frames batch shape (raw from loader): {dummy_frames_batch.shape}")

        clip0 = dummy_frames_batch[0].numpy() if isinstance(dummy_frames_batch, torch.Tensor) else np.array(dummy_frames_batch[0])
        feat_seq0 = encoder.extract_sequence(clip0)
        print(f"Feature sequence shape from encoder: {feat_seq0.shape}")

        feature_dim = feat_seq0.shape[1]
        sequence_length = feat_seq0.shape[0]
        print(f"Determined feature_dim={feature_dim}, sequence_length={sequence_length}")

    model = SequenceActionClassifier(feature_dim, num_classes, sequence_length).to(device)
    
    # Calculate class weights to handle imbalance
    class_counts = [0] * num_classes
    for _, label in train_loader.dataset:
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
    class_weights = torch.tensor([1.0 / (count + 1e-6) for count in class_counts], dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * num_classes
    print(f"📊 Class weights (for balancing): {class_weights.tolist()}\n")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # Early stopping variables
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        for m in model.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        class_counts_epoch = {}

        for batch_idx, (clips, labels) in enumerate(train_loader):
            batch_sequences = []
            for clip in clips:
                clip_np = clip.numpy() if isinstance(clip, torch.Tensor) else np.array(clip)
                feat_seq = encoder.extract_sequence(clip_np)
                batch_sequences.append(feat_seq)

            batch_sequences = torch.stack(batch_sequences, dim=0).to(device)
            labels = labels.to(device, dtype=torch.long)

            if batch_idx == 0 and epoch == 0:
                print(f"[DEBUG] BATCH 0 shapes: batch_sequences={batch_sequences.shape}, labels={labels.shape}")

            optimizer.zero_grad()
            outputs = model(batch_sequences)

            if outputs.ndim != 2 or outputs.shape[1] != num_classes:
                raise RuntimeError(f"Model output shape {outputs.shape} incompatible with num_classes={num_classes}")

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            batch_correct = (preds == labels).sum().item()
            batch_size = labels.size(0)

            total_loss += loss.item() * batch_size
            total_correct += batch_correct
            total_samples += batch_size

            for l in labels.cpu().numpy():
                class_counts_epoch[int(l)] = class_counts_epoch.get(int(l), 0) + 1

            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}/{CONFIG['num_epochs']}, Batch {batch_idx}: Loss={loss.item():.4f}, BatchAcc={batch_correct/batch_size:.4f}")

        # Epoch metrics
        if total_samples > 0:
            epoch_loss = total_loss / total_samples
            epoch_acc = total_correct / total_samples
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} Summary:")
            print(f"  Training Loss: {epoch_loss:.4f}")
            print(f"  Training Acc:  {epoch_acc:.4f}")
            print(f"  Samples:       {total_samples}")
            print(f"  Label distribution: {class_counts_epoch}")
            
            # Validation
            if len(val_loader) > 0:
                temp_model = ActionRecognitionModel(model, label_to_idx, idx_to_label, 
                                                   feature_dim, sequence_length)
                val_acc = validate_classifier(encoder, temp_model, val_loader, device)
                
                # Early stopping check
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
            print(f"{'='*60}\n")
        else:
            print(f"Epoch {epoch+1}: No samples processed!")

    # Restore best model if we have validation data
    if best_model_state is not None:
        print(f"✓ Restoring best model (val_acc={best_val_acc:.4f})")
        model.load_state_dict(best_model_state)

    # Wrap & save
    action_model = ActionRecognitionModel(
        model=model,
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label,
        feature_dim=feature_dim,
        sequence_length=sequence_length
    )
    action_model.save(CONFIG['model_save_path'])
    return action_model

# =============================
# Validation function (IMPROVED)
# =============================
def validate_classifier(encoder, action_model, val_loader, device):
    action_model.model.eval()
    total_correct = 0
    total_samples = 0
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for clips, labels in val_loader:
            batch_sequences = []
            for clip in clips:
                clip_np = clip.numpy()
                feat_seq = encoder.extract_sequence(clip_np)
                batch_sequences.append(feat_seq)
            
            batch_sequences = torch.stack(batch_sequences, dim=0).to(device)
            labels = labels.to(device)
            
            outputs = action_model.model(batch_sequences)
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
            label_name = action_model.idx_to_label[idx]
            print(f"    {label_name} (class {idx}): {acc:.4f} ({correct}/{total})")
    
    return accuracy

# =============================
# Example of how to load and use the model later
# =============================
def load_and_test_model(encoder, test_video_path):
    """Example function showing how to load and use the saved model"""
    try:
        action_model = ActionRecognitionModel.load(CONFIG['model_save_path'])
        
        dataset = VideoDataset("dummy")
        frames = dataset._load_video(test_video_path)
        frames = dataset._sample_frames(frames)
        frames = np.array(frames, dtype=np.float32)
        
        result = action_model.predict(encoder, frames)
        
        print(f"\nPrediction for {test_video_path}:")
        print(f"Action: {result['action']} (ID: {result['action_id']})")
        print(f"Confidence: {result['confidence']:.4f}")
        print("All probabilities:")
        for action, prob in result['probabilities'].items():
            print(f"  {action}: {prob:.4f}")
            
        return result
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# =============================
# Main (IMPROVED)
# =============================
if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)
    print("✓ Random seed set to 42 for reproducibility\n")
    
    # Check if model files exist
    if not os.path.exists(ENCODER_XML) or not os.path.exists(ENCODER_BIN):
        print(f"Error: Intel model files not found at:")
        print(f"  XML: {ENCODER_XML}")
        print(f"  BIN: {ENCODER_BIN}")
        print("Please download the model using the OpenVINO Model Downloader")
        exit(1)

    train_dataset = VideoDataset(os.path.join(CONFIG['data_path'], "train"))
    val_dataset = VideoDataset(os.path.join(CONFIG['data_path'], "val"))
    
    if len(train_dataset) == 0:
        print("No training samples found! Please check your dataset structure.")
        exit(1)
        
    print(f"\n📁 Dataset Information:")
    print(f"  Training samples:   {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Classes: {train_dataset.labels}")
    print()
    
    # Validate dataset size and filter actions
    is_valid, valid_actions = validate_dataset_size(train_dataset, val_dataset)
    if not is_valid:
        print("\n❌ Training aborted due to insufficient data.")
        exit(1)
    
    # Filter datasets to only include valid actions
    if len(valid_actions) < len(train_dataset.labels):
        print(f"\n🔄 Filtering datasets to include only valid actions...")
        
        # Create old to new index mapping
        old_to_new_idx = {train_dataset.label_to_idx[action]: new_idx for new_idx, action in enumerate(valid_actions)}
        
        # Filter training samples
        train_dataset.samples = [
            (path, old_to_new_idx[label]) 
            for path, label in train_dataset.samples 
            if label in old_to_new_idx
        ]
        
        # Filter validation samples
        val_dataset.samples = [
            (path, old_to_new_idx[label]) 
            for path, label in val_dataset.samples 
            if label in old_to_new_idx
        ]
        
        # Update label mappings
        train_dataset.labels = valid_actions
        train_dataset.label_to_idx = {label: idx for idx, label in enumerate(valid_actions)}
        train_dataset.idx_to_label = {idx: label for label, idx in train_dataset.label_to_idx.items()}
        
        val_dataset.labels = valid_actions
        val_dataset.label_to_idx = train_dataset.label_to_idx
        val_dataset.idx_to_label = train_dataset.idx_to_label
        
        print(f"✅ Filtered dataset:")
        print(f"  Training samples: {len(train_dataset.samples)}")
        print(f"  Validation samples: {len(val_dataset.samples)}")
        print(f"  Final classes: {valid_actions}")
    
    # Get label mappings from dataset
    label_to_idx, idx_to_label = train_dataset.get_label_mapping()
    print(f"\n📝 Final label mapping:")
    for label, idx in sorted(label_to_idx.items(), key=lambda x: x[1]):
        print(f"  {idx}: {label}")
    print()
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

    encoder = IntelFeatureExtractor(ENCODER_XML, ENCODER_BIN)
    
    # Train and get the action model with label information
    print(f"\n🚀 Starting training...\n")
    action_model = train_classifier(encoder, train_loader, val_loader, 
                                  num_classes=len(train_dataset.labels),
                                  label_to_idx=label_to_idx,
                                  idx_to_label=idx_to_label)
    
    # Final validation
    if len(val_loader) > 0:
        print(f"\n📊 Final Validation:")
        device = torch.device("cpu")
        validate_classifier(encoder, action_model, val_loader, device)
    
    print(f"\n✅ Training completed! Model and labels saved.")
    print(f"✓ Model path: {CONFIG['model_save_path']}")
    print(f"✓ You can now load the model using: ActionRecognitionModel.load('{CONFIG['model_save_path']}')")
    print(f"\n💡 Tips for better results:")
    print(f"  - Ensure each class has at least 20-30 videos")
    print(f"  - Balance your dataset (similar number of videos per class)")
    print(f"  - Use validation data to monitor training progress")