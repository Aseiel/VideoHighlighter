"""
R3D Training Configuration
============================

End-to-end 3D CNN fine-tuning on CUDA (or CPU fallback).
Input: 112×112, Kinetics-400 normalization.
"""

import torch

CONFIG = {
    # --- paths ---
    "data_path": "dataset",
    "model_save_path": "r3d_finetuned.pth",
    "checkpoint_dir": "checkpoints_r3d",
    "checkpoint_path": None,

    # --- model ---
    "model_variant": "r3d_18",          # r3d_18 | mc3_18 | r2plus1d_18
    "pretrained": True,

    # --- input ---
    "sequence_length": 16,
    "crop_size": (112, 112),             # R3D native resolution
    "mean": [0.43216, 0.394666, 0.37645],  # Kinetics-400
    "std": [0.22803, 0.22145, 0.216989],

    # --- training ---
    "batch_size": 4,
    "base_epochs": 30,
    "base_learning_rate": 1e-4,
    "finetune_learning_rate": 1e-5,
    "max_finetune_epochs": 15,
    "early_stopping_patience": 7,
    "min_delta": 0.001,
    "use_class_weights": True,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # --- backbone control ---
    "freeze_backbone": False,            # True = only train classifier head
    "freeze_bn": True,                   # freeze BatchNorm (recommended for fine-tuning)
    "differential_lr": True,             # backbone_lr = base_lr * backbone_lr_factor
    "backbone_lr_factor": 0.1,

    # --- regularisation ---
    "dropout": 0.4,
    "weight_decay": 1e-4,
    "gradient_clip_norm": 1.0,
    "label_smoothing": 0.1,

    # --- mixed precision ---
    "use_amp": True,                     # automatic mixed precision (FP16 on CUDA)

    # --- augmentation (stronger than Intel pipeline) ---
    "augmentation_prob": 0.5,
    "temporal_jitter": True,             # random frame offset within stride
    "random_crop_scale": (0.8, 1.0),
    "color_jitter": True,
    "color_jitter_brightness": 0.2,
    "color_jitter_contrast": 0.2,
    "color_jitter_saturation": 0.2,
    "horizontal_flip_prob": 0.5,

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
    "motion_threshold": 2.0,
    "use_pose_guided_crop": True,
    "pose_model": "yolo11n-pose.pt",
    "pose_conf_threshold": 0.3,
    "max_action_people": 2,

    # --- checkpointing ---
    "save_checkpoint_every": 5,

    # --- dataset ---
    "min_train_per_action": 5,
    "min_val_per_action": 2,

    # --- data loading ---
    "num_workers": 4,                    # parallel data loading (safe with ROI cache)
    "prefetch_factor": 2,               # batches to prefetch per worker
    "persistent_workers": True,          # keep workers alive between epochs

    # --- export ---
    "export_onnx": True,
    "export_openvino": False,            # requires openvino dev tools

    # --- production ---
    "min_production_accuracy": 0.3,

    # --- visualization ---
    "create_visualizations": False,
    "num_visualization_samples": 2,
    "visualization_sample_rate": 5,

    # --- debug ---
    "debug_mode": False,
    "debug_smoothing": False,
}