"""npu_obs — run object detection on an Intel NPU next to OBS Studio.

A small standalone tool built on ``modules.vision.npu_detector``: it scans
recordings, or watches a running OBS, with the detector on the NPU so the
game, OBS and its encoder keep the CPU and GPU. See README.md.

    python -m npu_obs probe
    python -m npu_obs video recording.mp4
    python -m npu_obs live
"""
