"""Peak detection over int16 audio, where the loudest sample is the hard case.

Reported from a run: "RuntimeWarning: overflow encountered in scalar absolute"
on every chunk. The warning is the visible half. The invisible half is that a
sample at full negative scale, -32768, has no positive counterpart in int16, so
``abs()`` wrapped it back to -32768, it failed a comparison against a positive
threshold, and the single loudest moment in clipped audio was dropped — by the
function whose whole job is to find loud moments.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from modules.audio_peaks import peaks_in_chunk

FULL_SCALE_NEGATIVE = -32768        # no positive twin exists in int16
FULL_SCALE_POSITIVE = 32767


def reference(chunk, threshold):
    """The original per-sample logic, with the overflow removed.

    Kept as the definition of correct so the vectorised version is checked
    against behaviour rather than against itself.
    """
    out = []
    values = [int(v) for v in chunk]
    for i in range(1, len(values) - 1):
        here = abs(values[i])
        if here > threshold and here >= abs(values[i - 1]) and here >= abs(values[i + 1]):
            out.append((i, here))
    return out


class TestTheOverflow:
    def test_the_loudest_possible_sample_is_found(self):
        """This is the bug. -32768 was read as a negative magnitude, so the
        peak that matters most never appeared."""
        chunk = np.array([0, 100, FULL_SCALE_NEGATIVE, 100, 0], dtype=np.int16)

        offsets, magnitudes = peaks_in_chunk(chunk, threshold_linear=1000.0)

        assert list(offsets) == [2]
        assert list(magnitudes) == [32768]

    def test_it_warns_about_nothing(self):
        """The warning was the user-visible half, one line per chunk, tens of
        thousands of them in a long run."""
        chunk = np.array([FULL_SCALE_NEGATIVE] * 8, dtype=np.int16)

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            peaks_in_chunk(chunk, threshold_linear=10.0)

    def test_both_extremes_measure_the_same_way(self):
        """A clipped waveform hits both rails; neither should be special."""
        loud_negative = np.array([0, FULL_SCALE_NEGATIVE, 0], dtype=np.int16)
        loud_positive = np.array([0, FULL_SCALE_POSITIVE, 0], dtype=np.int16)

        _, negative = peaks_in_chunk(loud_negative, 1000.0)
        _, positive = peaks_in_chunk(loud_positive, 1000.0)

        assert int(negative[0]) == 32768
        assert int(positive[0]) == 32767


class TestItMatchesTheOriginal:
    """Vectorising must not change which samples are peaks."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_random_audio_agrees_with_the_reference(self, seed):
        rng = np.random.default_rng(seed)
        chunk = rng.integers(FULL_SCALE_NEGATIVE, FULL_SCALE_POSITIVE + 1,
                             size=441, dtype=np.int16)
        threshold = 3276.8         # -20 dBFS, the default

        offsets, magnitudes = peaks_in_chunk(chunk, threshold)

        assert list(zip([int(o) for o in offsets],
                        [int(m) for m in magnitudes])) == reference(chunk, threshold)

    def test_clipped_audio_agrees_too(self):
        """Random samples rarely hit the rails; real loud footage does."""
        rng = np.random.default_rng(7)
        chunk = rng.choice([FULL_SCALE_NEGATIVE, FULL_SCALE_POSITIVE, 0, 500],
                           size=200).astype(np.int16)
        threshold = 100.0

        offsets, magnitudes = peaks_in_chunk(chunk, threshold)

        assert list(zip([int(o) for o in offsets],
                        [int(m) for m in magnitudes])) == reference(chunk, threshold)


class TestEdges:
    def test_the_first_and_last_sample_are_never_peaks(self):
        """They have only one neighbour, and the original skipped them."""
        chunk = np.array([FULL_SCALE_POSITIVE, 0, FULL_SCALE_POSITIVE],
                         dtype=np.int16)

        offsets, _ = peaks_in_chunk(chunk, 100.0)

        assert list(offsets) == []

    @pytest.mark.parametrize("size", [0, 1, 2])
    def test_a_chunk_too_short_to_have_an_inside(self, size):
        offsets, magnitudes = peaks_in_chunk(
            np.zeros(size, dtype=np.int16), 100.0)

        assert len(offsets) == 0 and len(magnitudes) == 0

    def test_a_plateau_counts_once_per_sample_as_before(self):
        """`>=` on both sides means a flat top reports every sample in it. That
        is what the original did, and the merge step downstream collapses them
        into one event."""
        chunk = np.array([0, 9000, 9000, 9000, 0], dtype=np.int16)

        offsets, _ = peaks_in_chunk(chunk, 1000.0)

        assert list(offsets) == [1, 2, 3]

    def test_silence_finds_nothing(self):
        offsets, _ = peaks_in_chunk(np.zeros(441, dtype=np.int16), 1000.0)

        assert list(offsets) == []
