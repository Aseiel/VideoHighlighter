"""Built-in expression classes for a detected face.

The companion to ``face_examples``: that module matches whatever the user points
at, this one names five expressions with no setup at all. They are the same two
tiers the rest of the app already has — a fixed vocabulary that works out of the
box, and taught categories for everything the vocabulary has no word for.

Both consume the *same crop*. ``face_examples.scan_frames`` finds and crops the
faces once; the crop then goes to CLIP for taught categories and here for the
built-in classes. No second decode, no second detection pass.

Model: ``emotions-recognition-retail-0003`` from Intel's Open Model Zoo — 5
classes, ~9.5 MB in FP32, Apache 2.0, and OpenVINO IR, which is the runtime
and the packaging path this project already uses for its action classifier. The licence
matters as much as the accuracy here: most off-the-shelf face models are
derivatives that would not survive the commercial build.

What it cannot do is worth stating plainly, because the failure is quiet. Five
coarse classes trained on posed, frontal, well-lit faces: it degrades on profile
and occlusion, has no notion of intensity or blended expressions, and it will
return a label for *every* face it is given, confident or not. Anything outside
those five words is not a low score here — it is absent, and it is what
``face_examples`` exists for.
"""
from __future__ import annotations

import os
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

# The model's output order. Fixed by the network, not by us.
EMOTION_LABELS = ("neutral", "happy", "sad", "surprise", "anger")

# Where the model is expected to sit, beside the face detector and recogniser.
DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "video_ai_editor", "models", "emotions-recognition-retail-0003.xml",
)

# The network's input geometry.
INPUT_SIZE = (64, 64)

# Below this the classifier is picking between near-ties. A face it cannot read
# should produce nothing rather than the most likely of five guesses.
DEFAULT_CONFIDENCE = 0.5


def top_emotion(probabilities: Sequence[float]) -> tuple:
    """``(label, confidence)`` for one face."""
    values = np.asarray(probabilities, dtype=float).ravel()
    if values.size == 0:
        return ("", 0.0)
    index = int(np.argmax(values))
    label = EMOTION_LABELS[index] if index < len(EMOTION_LABELS) else ""
    return (label, float(values[index]))


def emotions_by_second(crops: Sequence,
                       probabilities: Sequence,
                       *,
                       min_confidence: float = DEFAULT_CONFIDENCE) -> dict:
    """``{second: (label, confidence)}``, keeping the most confident face.

    A second can hold several faces. Reporting the clearest one is the honest
    reduction: averaging labels across people would invent an expression nobody
    wore.
    """
    best: dict = {}
    for crop, probs in zip(crops, probabilities):
        label, confidence = top_emotion(probs)
        if not label or confidence < min_confidence:
            continue
        sec = int(getattr(crop, "timestamp", 0))
        if confidence > best.get(sec, ("", 0.0))[1]:
            best[sec] = (label, confidence)
    return best


def to_signal(best: Mapping[int, tuple],
              duration: float,
              *,
              labels: Sequence[str],
              points: float) -> np.ndarray:
    """Per-second points for the expressions the user asked to score.

    Flat points for a match, like every other signal in the weight table, and
    only for the labels selected — scoring all five would mean scoring every
    second a face is visible, which distinguishes nothing.
    """
    wanted = {str(label).lower() for label in labels or ()}
    signal = np.zeros(int(duration) + 1, dtype=float)
    if not wanted:
        return signal
    for sec, (label, _confidence) in best.items():
        if 0 <= sec < len(signal) and label in wanted:
            signal[sec] = points
    return signal


def preprocess(crop: np.ndarray) -> np.ndarray:
    """One BGR crop to the network's ``(3, 64, 64)`` input."""
    import cv2

    resized = cv2.resize(crop, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    return np.transpose(resized.astype(np.float32), (2, 0, 1))


def classify_crops(crops: Sequence[np.ndarray],
                   infer_fn: Callable,
                   *,
                   preprocess_fn: Optional[Callable] = None,
                   batch: int = 16) -> np.ndarray:
    """Probabilities for each crop, batched.

    ``infer_fn`` takes an ``(n, 3, 64, 64)`` array and returns ``(n, 5)``. It is
    injected so this is testable without OpenVINO or a model file, and so the
    caller owns the device.

    ``preprocess_fn`` defaults to :func:`preprocess`, the module's one call into
    an image library. Injectable for the same reason: the test suite stubs cv2,
    and a resize that quietly returns a mock is worse than one that is absent.
    """
    if not len(crops):
        return np.zeros((0, len(EMOTION_LABELS)), dtype=np.float32)
    prepare = preprocess_fn or preprocess
    out = []
    for start in range(0, len(crops), batch):
        chunk = crops[start:start + batch]
        stacked = np.stack([prepare(c) for c in chunk])
        out.append(np.asarray(infer_fn(stacked), dtype=np.float32).reshape(
            len(chunk), -1))
    return np.concatenate(out, axis=0)


class EmotionClassifier:
    """The Open Model Zoo classifier, loaded lazily.

    Optional by construction: the model is not bundled, and a build without it
    must lose the built-in classes rather than fail to analyse a video. Callers
    check :meth:`available` and skip.
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH, device: str = "AUTO"):
        self.model_path = model_path
        self.device = device
        self._compiled = None
        self._output = None
        # None once the network accepts a dynamic batch; 1 when it insists
        # on the shape it was published with.
        self._batch = None

    def available(self) -> bool:
        """Whether the model files are actually present."""
        return (os.path.exists(self.model_path)
                and os.path.exists(os.path.splitext(self.model_path)[0] + ".bin"))

    def load(self) -> bool:
        """Compile the network. False — with a reason logged — if it cannot be."""
        if self._compiled is not None:
            return True
        if not self.available():
            print(f"ℹ Expression classifier not installed ({self.model_path}); "
                  "built-in expression classes are unavailable. Taught face "
                  "categories are unaffected.")
            return False
        try:
            from openvino import Core, PartialShape

            core = Core()
            model = core.read_model(self.model_path)
            # The published model has a static batch of 1, so feeding it a
            # batch raises rather than looping internally. Reshape to a dynamic
            # batch where the runtime allows it, and remember when it does not
            # so the caller feeds one crop at a time instead of failing.
            try:
                model.reshape({model.input(0): PartialShape([-1, 3, *INPUT_SIZE])})
                self._batch = None
            except Exception:
                self._batch = 1
            self._compiled = core.compile_model(model, self.device)
            self._output = self._compiled.output(0)
            print(f"✅ Expression classifier loaded ({self.device}"
                  f"{', batch of 1' if self._batch == 1 else ''})")
            return True
        except Exception as exc:
            print(f"⚠️ Expression classifier failed to load: {exc}")
            self._compiled = None
            return False

    def infer(self, batch: np.ndarray) -> np.ndarray:
        """Raw network call on an ``(n, 3, 64, 64)`` batch."""
        if self._compiled is None:
            raise RuntimeError("call load() before infer()")
        return self._compiled(batch)[self._output]

    def classify(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Probabilities per crop, or an empty array if the model is not loaded."""
        if self._compiled is None and not self.load():
            return np.zeros((0, len(EMOTION_LABELS)), dtype=np.float32)
        return classify_crops(crops, self.infer, batch=self._batch or 16)
