"""Taught actions: a small head on the frozen frame encoder.

    python -m model_training.action_head.train --data-path <dataset> [--out <folder>]

By default the encoder (``modules/vision/frame_encoder.py``) is not trained; it turns
each frame into a vector. What a user teaches is the head, a few hundred
thousand numbers that read a clip's frame vectors and name the action, so it
trains in a minute on a processor and is small enough to share.

- ``features``: clip -> frames evenly across it -> encoder vectors, cached.
- ``head``: the head, its training recipe, and its ONNX export.
- ``trust``: per-class trust thresholds and held-out scores (numpy only).
- ``train``: the command line. Scores the head on whole source videos it never
  saw, sets each class's trust threshold from those scores, then trains the
  model it saves on everything.

With ``--finetune-blocks N`` (a graphics card, about an hour) the top N blocks
of the image tower are trained too, starting from the frozen head, and the
model carries its own tower (~186 MB). It is saved only when it beats the
frozen head on the same unseen videos:

- ``frames``: every clip decoded once into a frame cache, the app's pixels.
- ``finetune``: the tower wrapper, augmentation and the LP-FT loop.
- ``tower``: exporting the tower and head and installing them as the folder
  the app loads.

Kept identical in both editions.
"""
