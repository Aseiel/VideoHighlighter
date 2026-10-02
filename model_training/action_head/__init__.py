"""Taught actions: a small head on the frozen frame encoder.

    python -m model_training.action_head.train --data-path <dataset> [--out <folder>]

The encoder (``modules/vision/frame_encoder.py``) is never trained; it turns
each frame into a vector. What a user teaches is the head, a few hundred
thousand numbers that read a clip's frame vectors and name the action, so it
trains in a minute on a processor and is small enough to share.

- ``features``: clip -> frames evenly across it -> encoder vectors, cached.
- ``head``: the head, its training recipe, and its ONNX export.
- ``trust``: per-class trust thresholds and held-out scores (numpy only).
- ``train``: the command line. Scores the head on whole source videos it never
  saw, sets each class's trust threshold from those scores, then trains the
  model it saves on everything.

Kept identical in both editions.
"""
