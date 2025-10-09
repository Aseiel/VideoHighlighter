# convert_pth_to_openvino_3d.py
import torch
import openvino as ov
import numpy as np
import os
from train_action_recognition import SequenceActionClassifier  # Import the new class

def convert_current_model():
    # Your model parameters
    feature_dim = 512
    num_classes = 1  # Adjust based on your actual classes
    sequence_length = 16
    
    # Load your trained 3D model
    print("Loading your trained 3D model...")
    model = SequenceActionClassifier(feature_dim, num_classes, sequence_length)
    model.load_state_dict(torch.load("intel_finetuned_classifier_3d.pth", map_location='cpu'))
    model.eval()
    
    print("Model architecture:")
    print(model)
    
    # Create dummy input with 3D shape
    dummy_input = torch.randn(1, sequence_length, feature_dim)  # (1, 16, 512)
    
    # Convert to ONNX first for better compatibility
    torch.onnx.export(
        model,
        dummy_input,
        "temp_model_3d.onnx",
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},  # Only batch dimension is dynamic
            'output': {0: 'batch_size'}
        },
        opset_version=13
    )
    
    # Convert ONNX to OpenVINO
    print("Converting to OpenVINO format...")
    ov_model = ov.convert_model("temp_model_3d.onnx")
    
    # Save the model
    ov.save_model(ov_model, "action_classifier_3d.xml")
    
    # Clean up temporary file
    if os.path.exists("temp_model_3d.onnx"):
        os.remove("temp_model_3d.onnx")
    
    print("✓ Conversion successful!")
    print(f"✓ Input shape: [batch_size, 16, 512]")
    print(f"✓ Model saved as: action_classifier_3d.xml")

if __name__ == "__main__":
    convert_current_model()
