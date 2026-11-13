"""The learned shot-boundary scorer: TransNetV2 through OpenVINO.

TransNetV2 (Soucek and Lokoc, 2020; code and weights MIT, github.com/soCzech/
TransNetV2) is a small 3D-convolutional network that reads 100 frames at 48x27
and says, for each, how likely it is that a transition is happening there. It
was trained on hand-marked shot changes, which is why it needs no sensitivity:
the question it answers is "is this an edit", not "did the pixels change".

It runs here from an OpenVINO export of the authors' weights
(``tools/export_transnetv2.py`` makes one; OpenVINO reads their saved model
directly, so TensorFlow is not involved at any point). The export is
``transnetv2.xml`` + ``.bin`` in ``models/transnetv2/``. When it is not there,
``load`` returns None and ``shot_boundaries`` falls back to frame differences.

The windowing is the authors': the video is padded with 25 copies of its first
frame, read in windows of 100 that advance by 50, and only the middle 50
answers of each window are kept - every frame is judged with at least 25
frames of context on each side.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

METHOD = "transnetv2"
MODEL_SUBDIR = "transnetv2"
MODEL_FILE = "transnetv2.xml"

WINDOW, STRIDE, CONTEXT = 100, 50, 25
HEIGHT, WIDTH = 27, 48

# The authors' own decision level for "this frame is a transition". It is the
# network's 50% and not a tuning knob: the output is a probability.
TRANSITION_PROBABILITY = 0.5

# A transition lasting more frames than this is a dissolve, fade or wipe, not
# a hard cut seen over two frames.
GRADUAL_MIN_FRAMES = 3


def model_path() -> Optional[str]:
    """The installed export's .xml, or None."""
    roots = []
    try:
        from modules.system import app_paths
        roots.append(app_paths.user_data_dir())
        roots.append(os.path.dirname(app_paths.resource_path("models")))
    except Exception:  # noqa: BLE001 - fall through to the working directory
        pass
    roots.append(os.getcwd())
    for root in roots:
        path = os.path.join(root, "models", MODEL_SUBDIR, MODEL_FILE)
        if os.path.isfile(path) and os.path.isfile(path[:-4] + ".bin"):
            return path
    return None


def _device() -> str:
    """The Intel GPU when there is one (the same choice the frame encoder
    makes), otherwise the processor."""
    try:
        from modules.vision.frame_encoder import _openvino_gpu
        return _openvino_gpu()[0] or "CPU"
    except Exception:  # noqa: BLE001
        return "CPU"


class ShotModel:
    """Frames in, in order, a block at a time; one probability per frame out."""

    def __init__(self, xml_path: str, device: str = "CPU"):
        import openvino as ov
        core = ov.Core()
        model = core.read_model(xml_path)
        model.reshape([1, WINDOW, HEIGHT, WIDTH, 3])
        self.device = device
        self._request = core.compile_model(model, device).create_infer_request()
        self._buffer = np.zeros((0, HEIGHT, WIDTH, 3), dtype=np.uint8)
        self._scores: List[np.ndarray] = []
        self._count = 0
        self._last = None

    def _window(self, frames: np.ndarray) -> np.ndarray:
        out = self._request.infer({0: frames[None].astype(np.float32)})
        logits = np.asarray(out[self._request.model_outputs[0]]).reshape(-1)
        return 1.0 / (1.0 + np.exp(-logits[CONTEXT:CONTEXT + STRIDE]))

    def _drain(self) -> None:
        while self._buffer.shape[0] >= WINDOW:
            self._scores.append(self._window(self._buffer[:WINDOW]))
            self._buffer = self._buffer[STRIDE:]

    def feed(self, block: np.ndarray) -> None:
        if block.shape[0] == 0:
            return
        if self._count == 0:
            pad = np.repeat(block[:1], CONTEXT, axis=0)
            self._buffer = np.concatenate([pad, block])
        else:
            self._buffer = np.concatenate([self._buffer, block])
        self._count += block.shape[0]
        self._last = block[-1:]
        self._drain()

    def result(self) -> np.ndarray:
        """The probability of a transition at each frame fed so far."""
        if self._count == 0:
            return np.zeros(0)
        scored = sum(s.size for s in self._scores)
        if scored < self._count:
            # Pad with the last frame until the frames still waiting have
            # their 25 of context behind them and fill a whole window.
            missing = self._count - scored
            windows = -(-missing // STRIDE)
            need = (windows - 1) * STRIDE + WINDOW - self._buffer.shape[0]
            if need > 0:
                self._buffer = np.concatenate(
                    [self._buffer, np.repeat(self._last, need, axis=0)])
            self._drain()
        return np.concatenate(self._scores)[:self._count]


def cuts_from_scores(scores, level: float = TRANSITION_PROBABILITY
                     ) -> Tuple[np.ndarray, list]:
    """``(cut_frames, gradual)`` from per-frame transition probabilities.

    A run of frames at or above ``level`` is one transition. A hard cut is a
    run of one or two frames - the network marks the last frame of the old
    shot - and the new shot starts on the frame after it. A longer run is a
    dissolve, fade or wipe: there is no frame where the old shot ends and the
    new one begins, so its cut is the middle of the run, and the run itself
    is returned in ``gradual`` as ``(first, last)``.

    A run that reaches the first or last frame is the video fading in or out,
    not a change of shot, and is not listed.
    """
    scores = np.asarray(scores, dtype=np.float64)
    on = scores >= level
    n = on.size
    cuts, gradual = [], []
    i = 0
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and on[j + 1]:
            j += 1
        if i > 0 and j < n - 1:
            if j - i + 1 >= GRADUAL_MIN_FRAMES:
                cuts.append((i + j + 1) // 2)
                gradual.append((i, j))
            else:
                cuts.append(j + 1)
        i = j + 1
    return np.array(cuts, dtype=int), gradual


def load(log_fn=print) -> Optional[ShotModel]:
    """The installed model, compiled, or None (frame differences are used)."""
    path = model_path()
    if not path:
        return None
    device = _device()
    for dev in dict.fromkeys((device, "CPU")):
        try:
            return ShotModel(path, dev)
        except Exception as e:  # noqa: BLE001 - try the processor, then give up
            print(f"shot_model: could not start on {dev}: {type(e).__name__}: {e}")
    log_fn("⚠️ The shot-boundary model could not start; using frame differences")
    return None
