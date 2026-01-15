# convert_pth_to_openvino.py
import torch
import torch.nn as nn
import openvino as ov
import numpy as np
import os
import json

class EncoderLSTM(nn.Module):
    """Enhanced classifier matching your checkpoint structure"""
    def __init__(self, feature_dim=512, hidden_dim=256, num_classes=31, 
                 num_layers=2, dropout=0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 2-layer bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(hidden_dim * 2)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Second normalization after attention
        self.ln2 = nn.LayerNorm(hidden_dim * 2)
        
        # Dropout before classification
        self.dropout = nn.Dropout(dropout)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, feature_dim)
        Returns: (batch_size, num_classes)
        """
        # LSTM with 2 layers
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Layer normalization
        lstm_out = self.ln1(lstm_out)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.nn.functional.softmax(attention_weights, dim=1)
        
        # Context vector: weighted sum of LSTM outputs
        context = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Second normalization
        context = self.ln2(context)
        
        # Dropout
        context = self.dropout(context)
        
        # Classification
        logits = self.classifier(context)
        
        return logits  # Only return logits for inference

def inspect_checkpoint():
    """Inspect the checkpoint to understand its structure"""
    print("🔍 Inspecting checkpoint...")
    checkpoint = torch.load("intel_finetuned_classifier_3d.pth", map_location='cpu', weights_only=False)
    
    print("\nCheckpoint keys:")
    for key in checkpoint.keys():
        print(f"  - {key}: {checkpoint[key].shape}")
    
    return checkpoint

def convert_current_model():
    # First, inspect the checkpoint
    checkpoint = inspect_checkpoint()
    
    # Load the mapping file to get model parameters
    mapping_path = "intel_finetuned_classifier_3d_mapping.json"
    
    if not os.path.exists(mapping_path):
        print(f"\n❌ Error: Mapping file {mapping_path} not found!")
        return
    
    with open(mapping_path, 'r') as f:
        mapping_data = json.load(f)
    
    # Get parameters from mapping file
    feature_dim = mapping_data['feature_dim']
    sequence_length = mapping_data['sequence_length'] 
    num_classes = mapping_data['num_classes']
    hidden_dim = mapping_data.get('hidden_dim', 256)
    num_layers = mapping_data.get('num_layers', 2)
    
    print(f"\n📋 Model parameters from mapping file:")
    print(f"  - Feature dimension: {feature_dim}")
    print(f"  - Hidden dimension: {hidden_dim}")
    print(f"  - Sequence length: {sequence_length}")
    print(f"  - Number of layers: {num_layers}")
    print(f"  - Number of classes: {num_classes}")
    
    # Verify dimensions from checkpoint
    lstm_weight_shape = checkpoint['lstm.weight_ih_l0'].shape
    feature_dim_from_checkpoint = lstm_weight_shape[1]
    hidden_dim_from_checkpoint = lstm_weight_shape[0] // 4  # Divided by 4 (LSTM gates)
    
    classifier_weight_shape = checkpoint['classifier.3.weight'].shape
    output_dim = classifier_weight_shape[0]
    
    print(f"\n📊 Checkpoint structure:")
    print(f"  - Feature dimension: {feature_dim_from_checkpoint}")
    print(f"  - Hidden dimension: {hidden_dim_from_checkpoint}")
    print(f"  - Output classes: {output_dim}")
    
    # Verify dimensions match
    if feature_dim_from_checkpoint != feature_dim:
        print(f"\n⚠️  Warning: Checkpoint feature_dim ({feature_dim_from_checkpoint}) != mapping file ({feature_dim})")
        print(f"   Using checkpoint feature_dim: {feature_dim_from_checkpoint}")
        feature_dim = feature_dim_from_checkpoint
    
    if hidden_dim_from_checkpoint != hidden_dim:
        print(f"\n⚠️  Warning: Checkpoint hidden_dim ({hidden_dim_from_checkpoint}) != mapping file ({hidden_dim})")
        print(f"   Using checkpoint hidden_dim: {hidden_dim_from_checkpoint}")
        hidden_dim = hidden_dim_from_checkpoint
    
    if output_dim != num_classes:
        print(f"\n⚠️  Warning: Checkpoint output dim ({output_dim}) != num_classes ({num_classes})")
        print(f"   Using checkpoint output dim: {output_dim}")
        num_classes = output_dim
    
    # Create the correct model architecture
    print("\n🔨 Creating EncoderLSTM model...")
    model = EncoderLSTM(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=0.3
    )
    
    # Load the checkpoint
    model.load_state_dict(checkpoint)
    model.eval()
    
    print("✓ Model loaded successfully!")
    print("\nModel architecture:")
    print(model)
    
    # Create dummy input matching the checkpoint's expected input
    # Shape: [batch_size, sequence_length, feature_dim]
    dummy_input = torch.randn(1, sequence_length, feature_dim)
    
    print(f"\n🔄 Converting to ONNX...")
    print(f"  - Input shape: [batch_size, {sequence_length}, {feature_dim}]")
    
    # Convert to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        "temp_model_3d.onnx",
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},  
            'output': {0: 'batch_size'}
        },
        opset_version=13,
        verbose=False
    )
    
    # Convert ONNX to OpenVINO
    print("🔄 Converting to OpenVINO format...")
    ov_model = ov.convert_model("temp_model_3d.onnx")
    
    # Save the model
    output_path = "action_classifier_3d.xml"
    ov.save_model(ov_model, output_path)
    
    # Clean up
    if os.path.exists("temp_model_3d.onnx"):
        os.remove("temp_model_3d.onnx")
    
    print("\n✅ Conversion successful!")
    print(f"✓ Architecture: 2-layer BiLSTM with Attention")
    print(f"✓ Input shape: [batch_size, {sequence_length}, {feature_dim}]")
    print(f"✓ Hidden dimension: {hidden_dim}")
    print(f"✓ Number of classes: {num_classes}")
    print(f"✓ Model saved as: {output_path}")
    
    # Update mapping file with correct dimensions
    mapping_data['model_feature_dim'] = feature_dim
    mapping_data['model_hidden_dim'] = hidden_dim
    mapping_data['model_sequence_length'] = sequence_length
    mapping_data['model_num_layers'] = num_layers
    with open(mapping_path, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    print(f"✓ Updated mapping file with model dimensions")
    
    # Print class labels for reference
    print(f"\n📝 Class labels ({num_classes} classes):")
    idx_to_label = mapping_data['idx_to_label']
    for idx in sorted([int(k) for k in idx_to_label.keys()]):
        label = idx_to_label[str(idx)]
        print(f"  {idx}: {label}")
    
    return feature_dim, sequence_length, num_classes

def test_converted_model(feature_dim, sequence_length, num_classes):
    """Test the converted OpenVINO model"""
    print("\n🧪 Testing converted model...")
    
    # Load the mapping file
    mapping_path = "intel_finetuned_classifier_3d_mapping.json"
    with open(mapping_path, 'r') as f:
        mapping_data = json.load(f)
    
    # Load OpenVINO model
    core = ov.Core()
    compiled_model = core.compile_model("action_classifier_3d.xml", "CPU")
    
    # Create test input
    test_input = np.random.randn(1, sequence_length, feature_dim).astype(np.float32)
    
    # Run inference
    result = compiled_model([test_input])
    output = result[0]
    
    print(f"✓ Model test successful!")
    print(f"✓ Input shape: {test_input.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Apply softmax to get probabilities
    exp_output = np.exp(output - np.max(output))
    probs = exp_output / exp_output.sum()
    
    # Show predicted class
    predicted_class = np.argmax(output, axis=1)[0]
    class_name = mapping_data['idx_to_label'][str(predicted_class)]
    confidence = probs[0][predicted_class]
    
    print(f"✓ Test prediction: {class_name} (class {predicted_class})")
    print(f"✓ Confidence: {confidence:.4f}")
    
    # Show top 3 predictions
    top3_indices = np.argsort(output[0])[-3:][::-1]
    print(f"\n🏆 Top 3 predictions:")
    for i, idx in enumerate(top3_indices, 1):
        label = mapping_data['idx_to_label'][str(idx)]
        score = probs[0][idx]
        print(f"  {i}. {label}: {score:.4f}")

if __name__ == "__main__":
    try:
        feature_dim, sequence_length, num_classes = convert_current_model()
        test_converted_model(feature_dim, sequence_length, num_classes)
        print("\n✅ All done! Your model is ready to use.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()