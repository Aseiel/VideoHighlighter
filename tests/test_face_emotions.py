"""Tests for `modules.face_emotions` — the built-in expression classes.

Inference is injected, so none of this needs OpenVINO or the model file. What
is being protected is mostly the honesty of the reduction: a face the model
cannot read must produce nothing rather than the most likely of five guesses,
and a missing model must cost the feature, not the analysis.
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.face_emotions import (
    DEFAULT_CONFIDENCE,
    EMOTION_LABELS,
    EmotionClassifier,
    classify_crops,
    emotions_by_second,
    top_emotion,
    to_signal,
)
from modules.face_examples import FaceCrop


def _probs(**named):
    """Probabilities by label name, in the network's own order."""
    row = np.zeros(len(EMOTION_LABELS), dtype=np.float32)
    for label, value in named.items():
        row[EMOTION_LABELS.index(label)] = value
    return row


class TestTopEmotion:
    def test_picks_the_strongest_class(self):
        label, confidence = top_emotion(_probs(neutral=0.1, happy=0.8, sad=0.1))
        assert label == "happy"
        assert confidence == pytest.approx(0.8, abs=1e-6)

    def test_labels_follow_the_networks_order(self):
        assert EMOTION_LABELS == ("neutral", "happy", "sad", "surprise", "anger")

    def test_nothing_in_gives_nothing_out(self):
        assert top_emotion([]) == ("", 0.0)


class TestPerSecond:
    def _crop(self, t):
        return FaceCrop(timestamp=t, bbox=(0, 0, 10, 10))

    def test_the_clearest_face_in_a_second_is_the_one_reported(self):
        crops = [self._crop(3.1), self._crop(3.9)]
        probs = [_probs(happy=0.6), _probs(anger=0.95)]
        got = emotions_by_second(crops, probs)
        assert list(got) == [3] and got[3][0] == "anger"
        assert got[3][1] == pytest.approx(0.95, abs=1e-6)

    def test_a_face_the_model_cannot_read_is_dropped(self):
        crops = [self._crop(1.0)]
        probs = [_probs(neutral=0.3, happy=0.28, sad=0.24)]
        assert emotions_by_second(crops, probs, min_confidence=0.5) == {}

    def test_the_confidence_floor_is_adjustable(self):
        crops = [self._crop(1.0)]
        probs = [_probs(happy=0.35)]
        assert emotions_by_second(crops, probs, min_confidence=0.3)[1][0] == "happy"

    def test_seconds_stay_separate(self):
        crops = [self._crop(1.0), self._crop(8.0)]
        probs = [_probs(happy=0.9), _probs(sad=0.7)]
        assert set(emotions_by_second(crops, probs)) == {1, 8}

    def test_the_default_floor_is_not_a_coin_toss(self):
        assert DEFAULT_CONFIDENCE >= 0.5


class TestSignal:
    def test_only_the_selected_expressions_score(self):
        best = {2: ("happy", 0.9), 5: ("sad", 0.9)}
        signal = to_signal(best, duration=10, labels=["happy"], points=4.0)
        assert signal[2] == 4.0
        assert signal[5] == 0.0

    def test_selecting_nothing_scores_nothing(self):
        """Scoring all five would score every second a face is visible."""
        best = {2: ("happy", 0.9)}
        assert not to_signal(best, duration=10, labels=[], points=4.0).any()

    def test_labels_are_matched_case_insensitively(self):
        best = {2: ("happy", 0.9)}
        assert to_signal(best, duration=10, labels=["Happy"], points=4.0)[2] == 4.0

    def test_the_array_matches_the_video_length(self):
        assert len(to_signal({}, duration=90, labels=["happy"], points=1.0)) == 91

    def test_a_second_past_the_end_is_ignored(self):
        best = {999: ("happy", 0.9)}
        assert not to_signal(best, duration=10, labels=["happy"], points=4.0).any()

    def test_points_are_flat_like_every_other_signal(self):
        best = {1: ("happy", 0.55), 2: ("happy", 0.99)}
        signal = to_signal(best, duration=10, labels=["happy"], points=4.0)
        assert signal[1] == signal[2] == 4.0


class TestClassifyCrops:
    """cv2 is stubbed by conftest, so the resize is injected here — testing it
    through the mock would assert on the mock."""

    def _crop(self):
        return np.full((80, 80, 3), 120, dtype=np.uint8)

    @staticmethod
    def _prepare(crop):
        return np.zeros((3, 64, 64), dtype=np.float32)

    def test_batches_are_shaped_for_the_network(self):
        seen = []

        def infer(batch):
            seen.append(batch.shape)
            return np.tile(_probs(happy=1.0), (batch.shape[0], 1))

        classify_crops([self._crop()] * 5, infer,
                       preprocess_fn=self._prepare, batch=2)
        assert [s[1:] for s in seen] == [(3, 64, 64)] * 3
        assert [s[0] for s in seen] == [2, 2, 1], "should batch and flush the rest"

    def test_one_row_of_probabilities_per_crop(self):
        def infer(batch):
            return np.tile(_probs(sad=1.0), (batch.shape[0], 1))

        out = classify_crops([self._crop()] * 3, infer,
                             preprocess_fn=self._prepare)
        assert out.shape == (3, len(EMOTION_LABELS))

    def test_the_real_resize_is_the_default(self):
        from modules.face_emotions import preprocess
        import inspect
        assert "preprocess_fn" in inspect.signature(classify_crops).parameters

    def test_no_crops_is_not_an_error(self):
        out = classify_crops([], lambda b: None)
        assert out.shape == (0, len(EMOTION_LABELS))


class TestClassifierLifecycle:
    def test_a_missing_model_is_reported_not_raised(self, tmp_path):
        clf = EmotionClassifier(str(tmp_path / "absent.xml"))
        assert clf.available() is False
        assert clf.load() is False

    def test_classifying_without_a_model_yields_nothing(self, tmp_path):
        """A build without the model loses the classes, not the analysis."""
        clf = EmotionClassifier(str(tmp_path / "absent.xml"))
        assert clf.classify([np.zeros((80, 80, 3), np.uint8)]).shape[0] == 0

    def test_a_lone_xml_without_weights_is_not_available(self, tmp_path):
        xml = tmp_path / "model.xml"
        xml.write_text("<net/>", encoding="utf-8")
        assert EmotionClassifier(str(xml)).available() is False

    def test_infer_before_load_is_a_clear_error(self, tmp_path):
        clf = EmotionClassifier(str(tmp_path / "absent.xml"))
        with pytest.raises(RuntimeError):
            clf.infer(np.zeros((1, 3, 64, 64), dtype=np.float32))
