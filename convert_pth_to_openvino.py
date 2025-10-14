# convert_pth_to_openvino_3d.py
import torch
import openvino as ov
import numpy as np
import os
from train_action_recognition import SequenceActionClassifier  # Import the new class

def convert_current_model():
    # Your model parameters
    feature_dim = 512
    sequence_length = 16
    
    # Load checkpoint to inspect it
    checkpoint = torch.load("intel_finetuned_classifier_3d.pth", map_location='cpu')
    
    # Detect num_classes from the checkpoint
    num_classes = checkpoint['classifier.3.weight'].shape[0]
    print(f"Detected {num_classes} classes from checkpoint")
    
    # Load your trained 3D model
    print("Loading your trained 3D model...")
    model = SequenceActionClassifier(feature_dim, num_classes, sequence_length)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print("Model architecture:")
    print(model)
    
    # Create dummy input with batch_size=1 to avoid LSTM warning
    dummy_input = torch.randn(1, sequence_length, feature_dim)
    
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
        verbose=False  # Reduce warnings
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
    
if __name__ == "__main__":
    convert_current_model()