import os
import glob
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from openvino.runtime import Core

# =============================
# How to
# =============================
# put short 5-10s videos to dataset/train/ and dataset/val/, usually You can put 100+ videos for train and 20+ for val


# =============================
# Configuration
# =============================
CONFIG = {
    "data_path": "dataset",
    "batch_size": 2,           # small batch for CPU
    "learning_rate": 1e-3,
    "num_epochs": 5,
    "sequence_length": 16,
    "frame_size": (224, 224),  # Intel encoder expects 224x224
    "model_save_path": "intel_finetuned_classifier_3d.pth",
    "label_mapping_save_path": "label_mapping.json",  # NEW: Save label mapping
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
# Intel Feature Extractor
# =============================
class IntelFeatureExtractor:
    def __init__(self, encoder_xml, encoder_bin):
        self.ie = Core()
        self.encoder_model = self.ie.read_model(model=encoder_xml, weights=encoder_bin)
        self.encoder = self.ie.compile_model(self.encoder_model, device_name="CPU")
        self.input_name = next(iter(self.encoder.inputs))
        
        # Get input shape information
        self.input_shape = self.encoder.inputs[0].shape
        print(f"Encoder input shape: {self.input_shape}")

    def preprocess_frame(self, frame):
        """Preprocess frame for Intel encoder"""
        # Convert to float32
        frame = frame.astype(np.float32)
        
        # Normalize: subtract mean and divide by std as expected by the model
        # These values are typical for Intel models
        mean = np.array([0.485, 0.456, 0.406]) * 255.0
        std = np.array([0.229, 0.224, 0.225]) * 255.0
        frame = (frame - mean) / std
        
        # Change from HWC to CHW format
        frame = frame.transpose(2, 0, 1)
        
        return frame

    def extract_sequence(self, frames):
        """
        frames: (T,H,W,C) numpy array
        returns: (T, feature_dim) torch tensor
        """
        features = []
        for frame in frames:
            # Preprocess the frame
            inp = self.preprocess_frame(frame)
            # Add batch dimension
            inp = inp[np.newaxis, ...]  # (1, 3, H, W)
            
            # Run inference
            out = self.encoder([inp])
            feat = list(out.values())[0]
            
            # Ensure we get the right shape
            feat = feat.squeeze()  # Remove batch dimension if present
            features.append(torch.from_numpy(feat))
        
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
# NEW: Model Manager to handle labels and model together
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
# Training loop 
# =============================
def train_classifier(encoder, train_loader, val_loader, num_classes, label_to_idx, idx_to_label):
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Determine feature dimension from dummy batch
    with torch.no_grad():
        dummy_frames, _ = next(iter(train_loader))
        print(f"Dummy frames shape: {dummy_frames.shape}")
        
        # Process first clip to get feature dimension
        feat_seq = encoder.extract_sequence(dummy_frames[0].numpy())
        print(f"Feature sequence shape: {feat_seq.shape}")
        
        feature_dim = feat_seq.shape[1]
        sequence_length = feat_seq.shape[0]
        print(f"Feature dimension: {feature_dim}, Sequence length: {sequence_length}")

    # Create model
    model = SequenceActionClassifier(feature_dim, num_classes, sequence_length).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    for epoch in range(CONFIG['num_epochs']):
        model.train()
        total_loss, total_correct = 0, 0
        batch_count = 0

        for batch_idx, (clips, labels) in enumerate(train_loader):
            batch_sequences = []
            for clip in clips:
                # Convert to numpy for OpenVINO processing
                clip_np = clip.numpy()
                feat_seq = encoder.extract_sequence(clip_np)   # (T, feature_dim)
                batch_sequences.append(feat_seq)
            
            # Stack sequences: (batch_size, sequence_length, feature_dim)
            batch_sequences = torch.stack(batch_sequences, dim=0).to(device)
            labels = labels.to(device)
            
            print(f"Batch {batch_idx}: sequences shape={batch_sequences.shape}, labels shape={labels.shape}")

            optimizer.zero_grad()
            outputs = model(batch_sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(1) == labels).sum().item()
            batch_count += 1

            if batch_idx % 5 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss={loss.item():.4f}")

        if batch_count > 0:
            acc = total_correct / len(train_loader.dataset)
            print(f"Epoch {epoch+1}: Loss={total_loss/batch_count:.4f}, Acc={acc:.4f}")
        else:
            print(f"Epoch {epoch+1}: No batches processed")

    # Wrap model with label information
    action_model = ActionRecognitionModel(
        model=model,
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label,
        feature_dim=feature_dim,
        sequence_length=sequence_length
    )
    
    # Save using the new method that preserves labels
    action_model.save(CONFIG['model_save_path'])
    
    return action_model

# =============================
# Validation function
# =============================
def validate_classifier(encoder, action_model, val_loader, device):
    action_model.model.eval()
    total_correct = 0
    total_samples = 0
    
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
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # Print class-wise accuracy
    print("\nClass mappings:")
    for idx, label in action_model.idx_to_label.items():
        print(f"  {idx}: {label}")
    
    return accuracy

# =============================
# NEW: Example of how to load and use the model later
# =============================
def load_and_test_model(encoder, test_video_path):
    """Example function showing how to load and use the saved model"""
    try:
        # Load the saved model with labels
        action_model = ActionRecognitionModel.load(CONFIG['model_save_path'])
        
        # Load and preprocess test video
        dataset = VideoDataset("dummy")  # Create dummy dataset to use its methods
        frames = dataset._load_video(test_video_path)
        frames = dataset._sample_frames(frames)
        frames = np.array(frames, dtype=np.float32)
        
        # Predict
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
# Main (MODIFIED to use new structure)
# =============================
if __name__ == "__main__":
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
        
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.labels}")
    
    # Get label mappings from dataset
    label_to_idx, idx_to_label = train_dataset.get_label_mapping()
    print(f"Label mapping: {label_to_idx}")
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, drop_last=True)

    encoder = IntelFeatureExtractor(ENCODER_XML, ENCODER_BIN)
    
    # Train and get the action model with label information
    action_model = train_classifier(encoder, train_loader, val_loader, 
                                  num_classes=len(train_dataset.labels),
                                  label_to_idx=label_to_idx,
                                  idx_to_label=idx_to_label)
    
    # Validate the model
    device = torch.device("cpu")
    validate_classifier(encoder, action_model, val_loader, device)
    
    print(f"\n✓ Training completed! Model and labels saved.")
    print(f"✓ You can now load the model using: ActionRecognitionModel.load('{CONFIG['model_save_path']}')")