"""Timebase for the edit-timeline export.

No media file and no Qt. The frame count is the r_frame_rate fraction.
"""

from __future__ import annotations

import pytest

from video_ai_editor.timeline_export import (
    RECORD_START_HOUR,
    RECORD_START_ZERO,
    ExportError,
    MediaSource,
    Sequence,
    Span,
    Timebase,
    edl_frame_index,
    edl_fps_for_nominal,
    record_start_seconds,
    to_frames,
)


def _source(num: int, den: int) -> MediaSource:
    return MediaSource(
        path="clip.mp4", duration=10.0, fps_num=num, fps_den=den,
        width=1920, height=1080, has_audio=True, audio_rate=48000,
        audio_channels=2)


# ---------------------------------------------------------------------------
# Frame counts
# ---------------------------------------------------------------------------

def test_five_seconds_at_29_97_is_150_frames():
    """The rational path. 5 * 29.97 is 149.85, which is not a frame count."""
    assert to_frames(5, 30000, 1001) == 150
    assert 5 * 29.97 != 150


def test_the_float_shortcut_drifts_on_a_long_clip():
    """round(seconds * 29.97) is already a frame out well before a feature."""
    seconds = 33334
    assert to_frames(seconds, 30000, 1001) == 999021
    assert int(round(seconds * 29.97)) == 999020


def test_half_a_second_at_100_fps_is_frame_50():
    assert to_frames(0.5, 100, 1) == 50


def test_five_seconds_at_120_fps_is_600_source_frames_on_a_60_fps_edl_clock():
    rate = Timebase.from_fraction(120, 1)

    assert to_frames(5, 120, 1) == 600
    assert rate.edl_fps == 60
    assert rate.frame_duration == "100/12000s"
    assert edl_frame_index(5, rate) == 300


def test_five_seconds_at_240_fps_uses_the_same_60_fps_edl_clock():
    rate = Timebase.from_fraction(240, 1)

    assert to_frames(5, 240, 1) == 1200
    assert rate.edl_fps == 60
    assert rate.frame_duration == "100/24000s"
    assert rate.nominal == 240
    assert edl_frame_index(5, rate) == 300


def test_29_97_timecode_counts_the_real_frames_at_30():
    rate = Timebase.from_fraction(30000, 1001)

    assert rate.frame_duration == "1001/30000s"
    assert rate.edl_fps == 30
    assert edl_frame_index(5, rate) == 150
    assert not rate.coarsened


# ---------------------------------------------------------------------------
# Which rates are named
# ---------------------------------------------------------------------------

def test_a_whole_rate_outside_the_table_is_named():
    rate = Timebase.from_fraction(90, 1)

    assert rate.recognised
    assert "unrecognised" not in rate.describe()
    assert rate.describe() == "90 fps, 90/1"
    assert rate.edl_fps == 90
    assert rate.frame_duration == "1/90s"


def test_a_non_whole_rate_outside_the_table_is_unrecognised():
    """48000/1001 does not reduce, and it is not 30 fps in disguise."""
    rate = Timebase.from_fraction(48000, 1001)

    assert not rate.recognised
    assert rate.describe() == "unrecognised, 48000/1001"
    assert rate.frame_duration == "1001/48000s"
    assert rate.nominal == 48
    assert rate.edl_fps == 48


def test_a_reducible_whole_rate_is_the_reduced_rate():
    rate = Timebase.from_fraction(180, 2)

    assert rate.fps_num == 90
    assert rate.recognised
    assert rate.describe() == "90 fps, 90/1"


def test_known_ntsc_rate_is_named_with_its_fraction():
    rate = Timebase.from_fraction(30000, 1001)

    assert rate.recognised
    assert rate.describe() == "29.97 fps, 30000/1001"


@pytest.mark.parametrize("nominal,edl", [
    (24, 24),
    (48, 48),
    (100, 100),
    (120, 60),
    (240, 60),
    (144, 48),
])
def test_the_edl_clock_fits_in_two_frame_digits(nominal, edl):
    assert edl_fps_for_nominal(nominal) == edl
    assert edl <= 100


# ---------------------------------------------------------------------------
# Record start and the sequence
# ---------------------------------------------------------------------------

def test_record_start_is_only_zero_or_one_hour():
    assert record_start_seconds(RECORD_START_ZERO) == 0
    assert record_start_seconds(RECORD_START_HOUR) == 3600


def test_any_other_record_start_is_refused():
    with pytest.raises(ExportError, match="01:00:00:00"):
        record_start_seconds("10:00:00:00")


def test_a_sequence_rejects_an_unknown_record_start():
    with pytest.raises(ExportError):
        Sequence(title="Morning", source=_source(30, 1), clips=((0.0, 5.0),),
                 record_start="10:00:00:00")


def test_a_sequence_defaults_to_a_zero_record_start_and_no_spans():
    sequence = Sequence(title="Morning", source=_source(120, 1),
                        clips=((0.0, 5.0),))

    assert sequence.record_start == RECORD_START_ZERO
    assert sequence.spans == ()
    assert sequence.timebase().edl_fps == 60


def test_a_span_kind_outside_speech_detection_face_is_refused():
    with pytest.raises(ExportError, match="span kind"):
        Span(kind="scene", start=0.0, end=1.0, label="cut")


def test_a_span_that_ends_before_it_starts_is_refused():
    with pytest.raises(ExportError, match="ends before"):
        Span(kind="speech", start=4.0, end=1.0, label="hello")


def test_a_non_positive_frame_rate_is_refused():
    with pytest.raises(ExportError):
        Timebase.from_fraction(0, 1)
    with pytest.raises(ExportError):
        to_frames(1.0, 30, 0)
