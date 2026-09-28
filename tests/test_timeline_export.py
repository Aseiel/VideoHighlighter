"""Timebase for the edit-timeline export.

No media file and no Qt. The frame count is the r_frame_rate fraction.
"""

from __future__ import annotations

import os

import pytest

from video_ai_editor.timeline_export import (
    RECORD_START_HOUR,
    RECORD_START_ZERO,
    ExportError,
    MediaSource,
    Sequence,
    Span,
    Timebase,
    cmx_text,
    edl_frame_index,
    edl_fps_for_nominal,
    record_start_seconds,
    reel_name,
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


# ---------------------------------------------------------------------------
# CMX 3600
# ---------------------------------------------------------------------------

def _media(path, num, den, *, audio=True, duration=120.0):
    return MediaSource(
        path=str(path), duration=duration, fps_num=num, fps_den=den,
        width=1920, height=1080, has_audio=audio,
        audio_rate=48000 if audio else 0,
        audio_channels=2 if audio else 0)


def test_thirty_fps_clips_abut_on_the_record_side(tmp_path):
    source = tmp_path / "morning.mp4"
    source.write_bytes(b"")
    out = tmp_path / "cut.edl"

    from video_ai_editor.timeline_export import TimelineExporter
    result = TimelineExporter.to_edl(
        [(10.0, 15.0), (0.0, 4.0)], str(source), str(out),
        source=_media(source, 30, 1))

    text = out.read_text(encoding="utf-8")
    assert "\r" not in text
    assert text.startswith(
        "TITLE: morning\n"
        "FCM: NON-DROP FRAME\n"
        "* SOURCE TIMES ARE FROM THE START OF THE FILE, NOT CAMERA TIMECODE\n")
    assert "001  MORNING  V     C        00:00:10:00 00:00:15:00 00:00:00:00 00:00:05:00" in text
    assert "002  MORNING  V     C        00:00:00:00 00:00:04:00 00:00:05:00 00:00:09:00" in text
    assert result.skipped == 0
    assert os.path.basename(result) == "cut.edl"


def test_one_hour_record_start_moves_only_the_record_columns(tmp_path):
    source = tmp_path / "morning.mp4"
    source.write_bytes(b"")
    out = tmp_path / "cut.edl"

    from video_ai_editor.timeline_export import TimelineExporter
    TimelineExporter.to_edl(
        [(10.0, 15.0)], str(source), str(out),
        source=_media(source, 30, 1), record_start="01:00:00:00")

    text = out.read_text(encoding="utf-8")
    assert "00:00:10:00 00:00:15:00 01:00:00:00 01:00:05:00" in text
    assert "00:00:00:00 00:00:05:00" not in text


def test_audio_events_share_the_picture_event_number(tmp_path):
    source = tmp_path / "morning.mp4"
    source.write_bytes(b"")
    out = tmp_path / "cut.edl"

    from video_ai_editor.timeline_export import TimelineExporter
    TimelineExporter.to_edl(
        [(10.0, 15.0)], str(source), str(out),
        source=_media(source, 30, 1, audio=True))

    lines = [line for line in out.read_text(encoding="utf-8").splitlines()
             if line[:3].isdigit()]
    assert len(lines) == 2
    assert lines[0].startswith("001  MORNING  V     C")
    assert lines[1].startswith("001  MORNING  A     C")
    assert lines[0].split()[-4:] == lines[1].split()[-4:]


def test_a_silent_file_has_no_audio_event(tmp_path):
    source = tmp_path / "morning.mp4"
    source.write_bytes(b"")
    out = tmp_path / "cut.edl"

    from video_ai_editor.timeline_export import TimelineExporter
    TimelineExporter.to_edl(
        [(0.0, 2.0)], str(source), str(out),
        source=_media(source, 30, 1, audio=False))

    text = out.read_text(encoding="utf-8")
    assert " A     C" not in text
    assert " V     C" in text


def test_a_long_reel_name_is_truncated_and_the_filename_is_kept(tmp_path):
    source = tmp_path / "longmorningname.mp4"
    source.write_bytes(b"")
    out = tmp_path / "cut.edl"

    from video_ai_editor.timeline_export import TimelineExporter
    TimelineExporter.to_edl(
        [(0.0, 1.0)], str(source), str(out),
        source=_media(source, 30, 1, audio=False))

    text = out.read_text(encoding="utf-8")
    assert "LONGMORN " in text
    assert "LONGMORNING" not in text.split("FROM CLIP NAME")[0]
    assert "* FROM CLIP NAME: longmorningname.mp4" in text
    assert f"* SOURCE FILE: {os.path.abspath(source)}" in text


def test_120_fps_is_five_seconds_of_60_fps_timecode_and_a_single_frame_is_skipped(tmp_path):
    source = tmp_path / "x6.mp4"
    source.write_bytes(b"")
    out = tmp_path / "cut.edl"

    from video_ai_editor.timeline_export import TimelineExporter
    result = TimelineExporter.to_edl(
        [(0.0, 5.0), (0.0, 1 / 120)], str(source), str(out),
        source=_media(source, 120, 1, audio=False))

    text = out.read_text(encoding="utf-8")
    assert "00:00:00:00 00:00:05:00 00:00:00:00 00:00:05:00" in text
    assert text.count(" V     C") == 1
    assert result.skipped == 1
    assert ":100" not in text


def test_an_empty_clip_list_writes_nothing(tmp_path):
    source = tmp_path / "morning.mp4"
    source.write_bytes(b"")
    out = tmp_path / "cut.edl"

    from video_ai_editor.timeline_export import TimelineExporter
    with pytest.raises(ExportError, match="no clips"):
        TimelineExporter.to_edl([], str(source), str(out),
                                source=_media(source, 30, 1))

    assert not out.exists()
    assert not (tmp_path / "cut.edl.part").exists()


def test_omitting_the_output_path_writes_beside_the_source(tmp_path):
    source = tmp_path / "morning.mp4"
    source.write_bytes(b"")

    from video_ai_editor.timeline_export import TimelineExporter
    result = TimelineExporter.to_edl(
        [(0.0, 1.0)], str(source), source=_media(source, 30, 1, audio=False))

    assert result.path == str(tmp_path / "morning_edit.edl")
    assert os.path.isfile(result.path)


def test_a_failed_replace_leaves_no_partial_file(tmp_path, monkeypatch):
    source = tmp_path / "morning.mp4"
    source.write_bytes(b"")
    out = tmp_path / "cut.edl"

    def fail_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail_replace)
    from video_ai_editor.timeline_export import TimelineExporter
    with pytest.raises(OSError, match="disk full"):
        TimelineExporter.to_edl(
            [(0.0, 1.0)], str(source), str(out),
            source=_media(source, 30, 1, audio=False))

    assert not out.exists()
    assert not (tmp_path / "cut.edl.part").exists()


def test_the_cmx_writer_does_not_use_the_yaml_cut_list():
    import inspect
    import video_ai_editor.timeline_export as exporter

    assert "modules.media.edl" not in inspect.getsource(exporter)


def test_event_numbers_wrap_to_001_after_999():
    source = _media("morning.mp4", 30, 1, audio=False)
    clips = tuple((i / 30, (i + 1) / 30) for i in range(1000))
    text, skipped = cmx_text(Sequence(title="morning", source=source, clips=clips))

    numbers = [line[:3] for line in text.splitlines() if line[:3].isdigit()]
    assert numbers[0] == "001"
    assert numbers[998] == "999"
    assert numbers[999] == "001"
    assert skipped == 0
    assert " D " not in text


def test_a_reel_name_keeps_only_ascii_letters_and_digits():
    assert reel_name("my-clip") == "MYCLIP  "
    assert reel_name("—") == "REEL    "
    assert reel_name("longmorningname") == "LONGMORN"
