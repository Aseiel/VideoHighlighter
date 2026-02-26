"""
Model Training Package for Action Recognition
===============================================

Supports two training pipelines:
  - Intel OpenVINO: Frozen encoder → LSTM decoder (CPU-friendly)
  - R3D/CUDA: End-to-end 3D CNN fine-tuning (GPU-accelerated)

Both pipelines share person detection, dataset loading, and evaluation code.

Usage (always run from project root, e.g. D:\\movie_highlighter):

    python -m model_training.intel.train
    python -m model_training.r3d.train
    python -m model_training.r3d.train --model r2plus1d_18 --epochs 40
"""