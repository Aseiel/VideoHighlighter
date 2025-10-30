# convert_pth_to_openvino_3d.py
import torch
import openvino as ov
import numpy as np
import os
import json
from train_action_recognition import EncoderLSTM  # Import the correct class

def convert_current_model():
    # Load the mapping file to get model parameters
    mapping_path = "intel_finetuned_classifier_3d_mapping.json"
    
    if not os.path.exists(mapping_path):
        print(f"Error: Mapping file {mapping_path} not found!")
        return
    
    with open(mapping_path, 'r') as f:
        mapping_data = json.load(f)
    
    # Get parameters from mapping file
    feature_dim = mapping_data['feature_dim']
    sequence_length = mapping_data['sequence_length'] 
    num_classes = mapping_data['num_classes']
    
    print(f"Model parameters from mapping file:")
    print(f"  - Feature dimension: {feature_dim}")
    print(f"  - Sequence length: {sequence_length}")
    print(f"  - Number of classes: {num_classes}")
    
    # Load your trained 3D model
    print("Loading your trained 3D model...")
    model = EncoderLSTM(
        feature_dim=feature_dim,
        hidden_dim=512,  # Default hidden dimension
        num_classes=num_classes
    )
    
    # Load the checkpoint
    checkpoint = torch.load("intel_finetuned_classifier_3d.pth", map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    
    print("Model architecture:")
    print(model)
    
    # Create dummy input with batch_size=1
    dummy_input = torch.randn(1, sequence_length, feature_dim)
    
    # Convert to ONNX
    print("Converting to ONNX...")
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
        verbose=True
    )
    
    # Convert ONNX to OpenVINO
    print("Converting to OpenVINO format...")
    ov_model = ov.convert_model("temp_model_3d.onnx")
    
    # Save the model
    ov.save_model(ov_model, "action_classifier_3d.xml")
    
    # Clean up
    if os.path.exists("temp_model_3d.onnx"):
        os.remove("temp_model_3d.onnx")
    
    print("✓ Conversion successful!")
    print(f"✓ Input shape: [batch_size, {sequence_length}, {feature_dim}]")
    print(f"✓ Number of classes: {num_classes}")
    print(f"✓ Model saved as: action_classifier_3d.xml")
    
    # Print class labels for reference
    print(f"✓ Class labels: {list(mapping_data['label_to_idx'].keys())}")

def test_converted_model():
    """Test the converted OpenVINO model"""
    print("\n🧪 Testing converted model...")
    
    # Load the mapping file
    mapping_path = "intel_finetuned_classifier_3d_mapping.json"
    with open(mapping_path, 'r') as f:
        mapping_data = json.load(f)
    
    feature_dim = mapping_data['feature_dim']
    sequence_length = mapping_data['sequence_length']
    
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
    
    # Show predicted class
    predicted_class = np.argmax(output, axis=1)[0]
    class_name = mapping_data['idx_to_label'][str(predicted_class)]
    print(f"✓ Test prediction: {class_name} (class {predicted_class})")

if __name__ == "__main__":
    convert_current_model()
    test_converted_model()