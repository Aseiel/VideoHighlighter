"""
Intel Training Configuration
==============================

Frozen OpenVINO encoder → trainable decoder.
Input: 224×224, ImageNet normalization.

decoder_type options:
    "mlp"  → EncoderMLP   (default) — static shapes, compiles on OpenVINO GPU ✓
    "lstm" → EncoderLSTM  (legacy)  — dynamic shapes, CPU-only at inference  ✗
"""

import os
import torch

# Resolved against the repository, not the working directory: a bare relative
# name dropped the trained model wherever the run happened to start, which for
# the GUI's trainer subprocess is the install root. Everything a run produces
# now lands in models/actions/, beside models/custom/ where the object
# detectors go, and that is the first place the app looks for it.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG = {
    # --- paths ---
    "data_path": "dataset",
    "model_save_path": os.path.join(
        _REPO_ROOT, "models", "actions", "intel_finetuned_classifier_3d.pth"),
    "checkpoint_dir": os.path.join(_REPO_ROOT, "checkpoints_intel"),
    "checkpoint_path": None,           # set to resume, e.g. "checkpoints_intel/checkpoint_latest.pth"

    # --- encoder ---
    "encoder_xml": os.path.join(os.getcwd(), "models/intel_action/encoder/FP32/action-recognition-0001-encoder.xml"),
    "encoder_bin": os.path.join(os.getcwd(), "models/intel_action/encoder/FP32/action-recognition-0001-encoder.bin"),

    # --- input ---
    "sequence_length": 16,
    "crop_size": (224, 224),
    "mean": [0.485, 0.456, 0.406],      # ImageNet
    "std": [0.229, 0.224, 0.225],

    # --- decoder architecture ---
    # "mlp"  → GPU-compatible at inference (recommended)
    # "lstm" → CPU-only at inference (legacy)
    "decoder_type": "mlp",
    "hidden_dim": 256,
    "num_layers": 2,       # only used by lstm
    "dropout": 0.3,

    # --- training ---
    "batch_size": 2,
    "base_epochs": 25,
    "base_learning_rate": 1e-4,
    "finetune_learning_rate": 1e-5,
    "max_finetune_epochs": 15,
    "early_stopping_patience": 5,
    "min_delta": 0.001,
    "use_class_weights": True,
    "augmentation_prob": 0.3,
    "device": "xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu",

    # --- checkpointing ---
    "save_checkpoint_every": 5,

    # --- data loading ---
    "num_workers": 4,
    "prefetch_factor": 2,
    "persistent_workers": True,

    # --- dataset validation ---
    "min_train_per_action": 5,
    "min_val_per_action": 2,

    # --- frame sampling ---
    "sampling_strategy": "temporal_stride",
    "default_stride": 4,
    "min_stride": 3,
    "max_stride": 8,

    # --- detection / ROI ---
    "use_adaptive_cropping": True,
    "use_roi_smoothing": True,
    "adaptive_smoothing": True,
    "smoothing_base_alpha": 0.5,
    "smoothing_window_size": 5,
    "motion_threshold": 2.0,
    "use_pose_guided_crop": True,
    "pose_model": None,  # no permissive pose model yet; ROI uses person boxes
    "pose_conf_threshold": 0.3,
    "max_action_people": 2,
    "allow_dynamic_group": True,
    "sticky_frames": 10,

    # --- visualization ---
    "create_visualizations": True,
    "num_visualization_samples": 4,
    "visualization_sample_rate": 5,
    "visualize_skeletons": False,

    # --- production model ---
    "min_production_accuracy": 0.3,

    # --- debug ---
    "debug_mode": False,
    "debug_motion_analysis": False,
    "debug_smoothing": False,
}