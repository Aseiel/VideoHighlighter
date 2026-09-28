"""Timebase for the edit-timeline export.

No media file and no Qt. The frame count is the r_frame_rate fraction.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

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


# ---------------------------------------------------------------------------
# FCPXML 1.9
# ---------------------------------------------------------------------------

def _ratio(token: str) -> tuple[int, int]:
    assert token.endswith("s")
    body = token[:-1]
    if "/" not in body:
        return int(body), 1
    num, den = body.split("/", 1)
    return int(num), int(den)


def _frames(time_value: str, frame_duration: str) -> int:
    """How many real frames a rational time is, at ``frame_duration``."""
    num, den = _ratio(time_value)
    fd_num, fd_den = _ratio(frame_duration)
    numerator = num * fd_den
    denominator = den * fd_num
    assert denominator and numerator % denominator == 0
    return numerator // denominator


def _read_fcpxml(path):
    text = path.read_text(encoding="utf-8")
    assert "\r" not in text
    assert text.startswith(
        '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n')
    return text, ET.fromstring(text)


def test_two_ntsc_clips_round_trip_to_frame_counts(tmp_path):
    source = tmp_path / "morning.mp4"
    source.write_bytes(b"")
    out = tmp_path / "cut.fcpxml"

    from video_ai_editor.timeline_export import TimelineExporter
    result = TimelineExporter.to_fcp_xml(
        [(10.0, 15.0), (0.0, 4.0)], str(source), str(out),
        source=_media(source, 30000, 1001, duration=60.0))

    text, root = _read_fcpxml(out)
    assert root.tag == "fcpxml" and root.get("version") == "1.9"
    assert "colorSpace" not in text
    formats = root.findall("resources/format")
    assets = root.findall("resources/asset")
    clips = root.findall("library/event/project/sequence/spine/asset-clip")
    assert len(formats) == 1 and len(assets) == 1 and len(clips) == 2
    assert [clip.get("name") for clip in clips] == ["Clip 1", "Clip 2"]
    assert all(clip.get("ref") == "r2" for clip in clips)

    frame = formats[0].get("frameDuration")
    assert frame == "1001/30000s"
    assert formats[0].get("name") == "FFVideoFormat1080p2997"
    sequence = root.find("library/event/project/sequence")
    assert _frames(clips[0].get("start"), frame) == to_frames(10.0, 30000, 1001)
    assert _frames(clips[0].get("duration"), frame) == 150
    assert _frames(clips[1].get("start"), frame) == 0
    assert _frames(clips[1].get("duration"), frame) == to_frames(4.0, 30000, 1001)
    assert _frames(clips[0].get("offset"), frame) == 0
    assert _frames(clips[1].get("offset"), frame) == _frames(clips[0].get("duration"), frame)
    clip_frames = sum(_frames(clip.get("duration"), frame) for clip in clips)
    assert _frames(sequence.get("duration"), frame) == clip_frames
    assert _frames(assets[0].get("duration"), frame) == to_frames(60.0, 30000, 1001)
    assert _frames(assets[0].get("duration"), frame) != clip_frames
    assert result.skipped == 0


def test_120_and_240_fps_keep_the_real_frame(tmp_path):
    source = tmp_path / "x6.mp4"
    source.write_bytes(b"")
    out = tmp_path / "cut.fcpxml"

    from video_ai_editor.timeline_export import TimelineExporter
    result = TimelineExporter.to_fcp_xml(
        [(0.0, 5.0), (0.0, 1 / 120)], str(source), str(out),
        source=_media(source, 120, 1, audio=False))

    text, root = _read_fcpxml(out)
    frame = root.find("resources/format").get("frameDuration")
    assert frame == "100/12000s"
    assert "p120" in root.find("resources/format").get("name")
    clips = root.findall("library/event/project/sequence/spine/asset-clip")
    assert _frames(clips[0].get("duration"), frame) == 600
    assert _frames(clips[1].get("duration"), frame) == 1
    assert result.skipped == 0
    assert "p60" not in text

    out240 = tmp_path / "fast.fcpxml"
    TimelineExporter.to_fcp_xml(
        [(0.0, 1.0)], str(source), str(out240),
        source=_media(source, 240, 1, audio=False))
    _, root240 = _read_fcpxml(out240)
    format240 = root240.find("resources/format")
    assert format240.get("frameDuration") == "100/24000s"
    assert "p240" in format240.get("name")
    clip = root240.find("library/event/project/sequence/spine/asset-clip")
    assert _frames(clip.get("duration"), format240.get("frameDuration")) == 240


def test_a_path_with_a_space_is_percent_encoded(tmp_path):
    source = tmp_path / "my movie.mp4"
    source.write_bytes(b"")
    out = tmp_path / "cut.fcpxml"

    from video_ai_editor.timeline_export import TimelineExporter
    TimelineExporter.to_fcp_xml(
        [(0.0, 1.0)], str(source), str(out),
        source=_media(source, 30, 1, audio=False))

    src = _read_fcpxml(out)[1].find("resources/asset/media-rep").get("src")
    assert src.startswith("file://")
    assert "my%20movie.mp4" in src
    assert " " not in src


def test_display_size_is_what_format_uses_after_rotation(tmp_path):
    """A 90° turn is already applied: stored 1920×1080 becomes 1080×1920."""
    source = tmp_path / "phone.mp4"
    source.write_bytes(b"")
    out = tmp_path / "cut.fcpxml"
    media = _media(source, 30, 1, audio=False)
    media = MediaSource(
        path=media.path, duration=media.duration, fps_num=30, fps_den=1,
        width=1080, height=1920, has_audio=False, audio_rate=0, audio_channels=0,
    )

    from video_ai_editor.timeline_export import TimelineExporter
    TimelineExporter.to_fcp_xml([(0.0, 1.0)], str(source), str(out), source=media)

    fmt = _read_fcpxml(out)[1].find("resources/format")
    assert fmt.get("width") == "1080"
    assert fmt.get("height") == "1920"


def test_a_silent_file_declares_no_audio(tmp_path):
    source = tmp_path / "morning.mp4"
    source.write_bytes(b"")
    out = tmp_path / "cut.fcpxml"

    from video_ai_editor.timeline_export import TimelineExporter
    TimelineExporter.to_fcp_xml(
        [(0.0, 1.0)], str(source), str(out),
        source=_media(source, 30, 1, audio=False))

    text, root = _read_fcpxml(out)
    asset = root.find("resources/asset")
    assert asset.get("hasAudio") == "0"
    for name in ("audioSources", "audioChannels", "audioRate", "audioLayout"):
        assert asset.get(name) is None
        assert name not in text


def test_stereo_is_declared_only_for_two_channels(tmp_path):
    source = tmp_path / "morning.mp4"
    source.write_bytes(b"")
    stereo = tmp_path / "stereo.fcpxml"
    mono = tmp_path / "mono.fcpxml"

    from video_ai_editor.timeline_export import TimelineExporter
    TimelineExporter.to_fcp_xml(
        [(0.0, 1.0)], str(source), str(stereo),
        source=_media(source, 30, 1, audio=True))
    mono_media = MediaSource(
        path=str(source), duration=120.0, fps_num=30, fps_den=1,
        width=1920, height=1080, has_audio=True,
        audio_rate=44100, audio_channels=1,
    )
    TimelineExporter.to_fcp_xml(
        [(0.0, 1.0)], str(source), str(mono), source=mono_media)

    stereo_root = _read_fcpxml(stereo)[1]
    asset = stereo_root.find("resources/asset")
    sequence = stereo_root.find("library/event/project/sequence")
    assert asset.get("audioChannels") == "2"
    assert asset.get("audioRate") == "48000"
    assert sequence.get("audioLayout") == "stereo"
    assert sequence.get("audioRate") == "48k"
    assert asset.get("audioLayout") is None

    mono_text, mono_root = _read_fcpxml(mono)
    assert mono_root.find("resources/asset").get("audioChannels") == "1"
    assert mono_root.find("resources/asset").get("audioRate") == "44100"
    assert "audioLayout" not in mono_text
    assert mono_root.find("library/event/project/sequence").get("audioRate") == "44.1k"


def test_one_hour_record_start_is_the_sequence_origin(tmp_path):
    source = tmp_path / "morning.mp4"
    source.write_bytes(b"")
    out = tmp_path / "cut.fcpxml"

    from video_ai_editor.timeline_export import TimelineExporter
    TimelineExporter.to_fcp_xml(
        [(10.0, 15.0), (0.0, 4.0)], str(source), str(out),
        source=_media(source, 30, 1, audio=False),
        record_start="01:00:00:00")

    root = _read_fcpxml(out)[1]
    sequence = root.find("library/event/project/sequence")
    clips = root.findall("library/event/project/sequence/spine/asset-clip")
    frame = root.find("resources/format").get("frameDuration")
    assert sequence.get("tcStart") == "3600s"
    assert clips[0].get("offset") == sequence.get("tcStart")
    assert _frames(clips[0].get("offset"), frame) == 3600 * 30
    assert _frames(clips[1].get("offset"), frame) == (
        _frames(clips[0].get("offset"), frame)
        + _frames(clips[0].get("duration"), frame))


def test_a_zero_frame_clip_is_skipped_and_an_empty_list_writes_nothing(tmp_path):
    source = tmp_path / "morning.mp4"
    source.write_bytes(b"")
    out = tmp_path / "cut.fcpxml"

    from video_ai_editor.timeline_export import TimelineExporter
    result = TimelineExporter.to_fcp_xml(
        [(0.0, 0.001), (0.0, 1.0)], str(source), str(out),
        source=_media(source, 30, 1, audio=False))
    clips = _read_fcpxml(out)[1].findall(
        "library/event/project/sequence/spine/asset-clip")
    assert len(clips) == 1
    assert result.skipped == 1

    empty = tmp_path / "empty.fcpxml"
    with pytest.raises(ExportError, match="no clips"):
        TimelineExporter.to_fcp_xml([], str(source), str(empty),
                                    source=_media(source, 30, 1))
    assert not empty.exists()
    assert not (tmp_path / "empty.fcpxml.part").exists()

    gone = tmp_path / "gone.fcpxml"
    with pytest.raises(ExportError, match="shorter than one frame"):
        TimelineExporter.to_fcp_xml(
            [(0.0, 0.001)], str(source), str(gone),
            source=_media(source, 30, 1))
    assert not gone.exists()


def test_omitting_the_fcpxml_path_writes_beside_the_source(tmp_path):
    source = tmp_path / "morning.mp4"
    source.write_bytes(b"")

    from video_ai_editor.timeline_export import TimelineExporter
    result = TimelineExporter.to_fcp_xml(
        [(0.0, 1.0)], str(source), source=_media(source, 30, 1, audio=False))

    assert result.path == str(tmp_path / "morning_edit.fcpxml")
    assert os.path.isfile(result.path)


def test_a_failed_fcpxml_replace_leaves_no_partial_file(tmp_path, monkeypatch):
    source = tmp_path / "morning.mp4"
    source.write_bytes(b"")
    out = tmp_path / "cut.fcpxml"

    def fail_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail_replace)
    from video_ai_editor.timeline_export import TimelineExporter
    with pytest.raises(OSError, match="disk full"):
        TimelineExporter.to_fcp_xml(
            [(0.0, 1.0)], str(source), str(out),
            source=_media(source, 30, 1, audio=False))

    assert not out.exists()
    assert not (tmp_path / "cut.fcpxml.part").exists()
