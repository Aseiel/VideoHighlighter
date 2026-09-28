"""Edit-timeline export: a sequence model and the files an NLE can open.

The timebase and the sequence types are pure data. Writers below them turn a
:class:`Sequence` into CMX 3600 and FCPXML. Frame counts always come from the
``r_frame_rate`` fraction. A float such as 29.97 is not a frame boundary.
"""

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from xml.dom import minidom

RECORD_START_ZERO = "00:00:00:00"
RECORD_START_HOUR = "01:00:00:00"
RECORD_STARTS = (RECORD_START_ZERO, RECORD_START_HOUR)

_SPAN_KINDS = ("speech", "detection", "face")

# Real r_frame_rate, the integer CMX counts at, the FCPXML frameDuration
# (the real frame, not the CMX clock), and the token after "p" in an
# FFVideoFormat name. 120 and 240 are real rates in the XML and a 60-frame
# clock in the EDL: a CMX frame field only holds 00–99.
_RATE_TABLE = {
    (24000, 1001): (24, 24, "1001/24000s", "2398"),
    (24, 1): (24, 24, "100/2400s", "24"),
    (25, 1): (25, 25, "100/2500s", "25"),
    (30000, 1001): (30, 30, "1001/30000s", "2997"),
    (30, 1): (30, 30, "100/3000s", "30"),
    (48, 1): (48, 48, "100/4800s", "48"),
    (50, 1): (50, 50, "100/5000s", "50"),
    (60000, 1001): (60, 60, "1001/60000s", "5994"),
    (60, 1): (60, 60, "100/6000s", "60"),
    (100, 1): (100, 100, "100/10000s", "100"),
    (120, 1): (120, 60, "100/12000s", "120"),
    (240, 1): (240, 60, "100/24000s", "240"),
}

_NTSC_LABELS = {
    (24000, 1001): "23.976",
    (30000, 1001): "29.97",
    (60000, 1001): "59.94",
}


class ExportError(ValueError):
    """A sequence or a rate the exporter cannot describe."""


def to_frames(seconds: float, num: int, den: int) -> int:
    """Frame index at ``seconds`` on the real rate ``num/den``.

    ``round(seconds * 29.97)`` is not this. Five seconds at 30000/1001 is
    exactly 150 frames; the float product ``5 * 29.97`` is 149.85.
    """
    if num <= 0 or den <= 0:
        raise ExportError(f"frame rate {num}/{den} is not a positive fraction")
    return int(round(float(seconds) * num / den))


def _reduce(num: int, den: int) -> tuple[int, int]:
    if num <= 0 or den <= 0:
        raise ExportError(f"frame rate {num}/{den} is not a positive fraction")
    factor = math.gcd(int(num), int(den))
    return int(num) // factor, int(den) // factor


def edl_fps_for_nominal(nominal: int) -> int:
    """CMX counter for a whole-number picture rate.

    Two frame digits hold 00–99, so a rate of 100 still fits and a rate above
    it does not. The coarsened clock is the largest whole divisor at or under
    60, which is 60 for both 120 and 240.
    """
    if nominal < 1:
        raise ExportError(f"nominal frame rate {nominal} is not positive")
    if nominal <= 100:
        return nominal
    return max(divisor for divisor in range(1, 61) if nominal % divisor == 0)


@dataclass(frozen=True)
class Timebase:
    """How one source file is counted in an EDL and in FCPXML.

    ``fps_num/fps_den`` is the reduced ``r_frame_rate``. ``nominal`` is the
    whole number of frames the picture steps at (30 for 29.97 non-drop, 120
    for a 120 fps file). ``edl_fps`` is the clock the CMX timecode actually
    uses, equal to ``nominal`` until ``nominal`` no longer fits in two digits.
    ``frame_duration`` is the FCPXML value and always describes the real frame.
    """

    fps_num: int
    fps_den: int
    nominal: int
    edl_fps: int
    frame_duration: str
    recognised: bool
    rate_token: str

    @classmethod
    def from_fraction(cls, num: int, den: int) -> "Timebase":
        num, den = _reduce(num, den)
        known = _RATE_TABLE.get((num, den))
        if known is not None:
            nominal, edl_fps, duration, token = known
            return cls(num, den, nominal, edl_fps, duration, True, token)
        nominal = int(round(num / den))
        if nominal < 1:
            nominal = 1
        # A whole number of frames per second is a rate we can name. Anything
        # else that is not in the table is unrecognised: the dialog says so,
        # and the file still uses the reduced fraction rather than 30.
        whole = den == 1
        return cls(
            num, den, nominal, edl_fps_for_nominal(nominal),
            f"{den}/{num}s", whole, str(nominal),
        )

    @property
    def coarsened(self) -> bool:
        """True when the EDL clock is coarser than the picture rate."""
        return self.edl_fps != self.nominal

    def describe(self) -> str:
        """The rate line the export dialog shows."""
        if not self.recognised:
            return f"unrecognised, {self.fps_num}/{self.fps_den}"
        pretty = _NTSC_LABELS.get((self.fps_num, self.fps_den))
        if pretty is None:
            pretty = str(self.nominal)
        return f"{pretty} fps, {self.fps_num}/{self.fps_den}"


def edl_frame_index(seconds: float, timebase: Timebase) -> int:
    """Timecode frame index for ``seconds``, on ``timebase.edl_fps``.

    Non-drop 29.97 numbers the real frame index at 30. A 120 fps file cannot:
    the index is scaled onto the 60 fps clock, so five seconds is 300 EDL
    frames rather than 600 source frames.
    """
    source = to_frames(seconds, timebase.fps_num, timebase.fps_den)
    if not timebase.coarsened:
        return source
    return int(round(source * timebase.edl_fps / timebase.nominal))


def record_start_seconds(value: str) -> int:
    """Seconds of sequence clock for a dialog choice. Only two values exist."""
    if value == RECORD_START_ZERO:
        return 0
    if value == RECORD_START_HOUR:
        return 3600
    raise ExportError(
        f"record start must be {RECORD_START_ZERO} or {RECORD_START_HOUR}, "
        f"got {value!r}")


@dataclass(frozen=True)
class MediaSource:
    """One file the sequence cuts. Width and height are the display size."""

    path: str
    duration: float
    fps_num: int
    fps_den: int
    width: int
    height: int
    has_audio: bool
    audio_rate: int
    audio_channels: int

    def timebase(self) -> Timebase:
        return Timebase.from_fraction(self.fps_num, self.fps_den)


@dataclass(frozen=True)
class Span:
    """One stretch of speech, one detection class, or one face, in source seconds."""

    kind: str
    start: float
    end: float
    label: str

    def __post_init__(self) -> None:
        if self.kind not in _SPAN_KINDS:
            raise ExportError(
                f"span kind must be one of {', '.join(_SPAN_KINDS)}, "
                f"got {self.kind!r}")
        if self.end < self.start:
            raise ExportError(
                f"span {self.kind!r} ends before it starts "
                f"({self.start}..{self.end})")


@dataclass(frozen=True)
class Sequence:
    """The edit timeline, in order, plus the spans that fall in those clips."""

    title: str
    source: MediaSource
    clips: tuple[tuple[float, float], ...]
    record_start: str = RECORD_START_ZERO
    spans: tuple[Span, ...] = ()

    def __post_init__(self) -> None:
        record_start_seconds(self.record_start)
        self.source.timebase()

    def timebase(self) -> Timebase:
        return self.source.timebase()


@dataclass(frozen=True)
class WrittenExport:
    """A file that was written, and how many clips were shorter than a frame.

    ``str`` and ``os.path`` both see :attr:`path`, so a caller that still
    treats the return value as a path keeps working.
    """

    path: str
    skipped: int

    def __fspath__(self) -> str:
        return self.path

    def __str__(self) -> str:
        return self.path


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0] or "untitled"


def reel_name(stem: str) -> str:
    """Eight-character CMX reel. The filename itself goes in a comment."""
    cleaned = "".join(
        ch for ch in stem.upper() if ch.isascii() and ch.isalnum())
    if not cleaned:
        cleaned = "REEL"
    return f"{cleaned[:8]:<8}"


def frames_to_timecode(frames: int, fps: int) -> str:
    """``HH:MM:SS:FF`` at ``fps`` frames per timecode second. Non-drop."""
    if fps < 1:
        raise ExportError(f"timecode rate {fps} is not positive")
    frames = max(0, int(frames))
    frames_per_minute = 60 * fps
    frames_per_hour = 3600 * fps
    hours, frames = divmod(frames, frames_per_hour)
    minutes, frames = divmod(frames, frames_per_minute)
    seconds, frame = divmod(frames, fps)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frame:02d}"


def _event_number(index: int) -> int:
    """1..999, then 1 again. CMX event numbers are three digits."""
    return (index % 999) + 1


def _quantise_clip(start: float, end: float, timebase: Timebase) -> tuple[int, int] | None:
    """Source in/out on the EDL clock, or None when the clip has no frames."""
    if end <= start:
        return None
    src_in = edl_frame_index(start, timebase)
    src_out = edl_frame_index(end, timebase)
    if src_out <= src_in:
        return None
    return src_in, src_out


def _event_line(number: int, reel: str, track: str,
                src_in: str, src_out: str, rec_in: str, rec_out: str) -> str:
    # Track is a 5-character field ("V    ", "A    "). Eight spaces follow C.
    return (f"{number:03d}  {reel:8} {track:<5} C        "
            f"{src_in} {src_out} {rec_in} {rec_out}")


def cmx_text(sequence: Sequence) -> tuple[str, int]:
    """CMX 3600 text for ``sequence``, and how many clips were skipped.

    Raises :class:`ExportError` when there is nothing to write. Does not
    touch the disk.
    """
    if not sequence.clips:
        raise ExportError("nothing to export — the edit timeline has no clips")

    timebase = sequence.timebase()
    kept: list[tuple[int, int]] = []
    skipped = 0
    for start, end in sequence.clips:
        quantised = _quantise_clip(start, end, timebase)
        if quantised is None:
            skipped += 1
            continue
        kept.append(quantised)
    if not kept:
        raise ExportError(
            f"every clip is shorter than one frame at {timebase.edl_fps} fps")

    stem = _stem(sequence.source.path)
    reel = reel_name(stem)
    filename = os.path.basename(sequence.source.path)
    source_file = os.path.abspath(sequence.source.path)
    fps = timebase.edl_fps
    record = record_start_seconds(sequence.record_start) * fps

    lines = [
        f"TITLE: {stem}",
        "FCM: NON-DROP FRAME",
        "* SOURCE TIMES ARE FROM THE START OF THE FILE, NOT CAMERA TIMECODE",
        "",
    ]
    for index, (src_in, src_out) in enumerate(kept):
        number = _event_number(index)
        duration = src_out - src_in
        rec_in = record
        rec_out = record + duration
        record = rec_out
        src_in_tc = frames_to_timecode(src_in, fps)
        src_out_tc = frames_to_timecode(src_out, fps)
        rec_in_tc = frames_to_timecode(rec_in, fps)
        rec_out_tc = frames_to_timecode(rec_out, fps)

        lines.append(_event_line(number, reel, "V",
                                 src_in_tc, src_out_tc, rec_in_tc, rec_out_tc))
        lines.append(f"* FROM CLIP NAME: {filename}")
        lines.append(f"* SOURCE FILE: {source_file}")
        if sequence.source.has_audio:
            lines.append(_event_line(number, reel, "A",
                                     src_in_tc, src_out_tc, rec_in_tc, rec_out_tc))
            lines.append(f"* FROM CLIP NAME: {filename}")
        if index != len(kept) - 1:
            lines.append("")

    return "\n".join(lines) + "\n", skipped


def _write_text(path: str, text: str) -> None:
    """Write ``text`` to ``path`` via a sibling ``.part`` file.

    A failure removes the partial file. The destination is replaced only
    after the partial file is complete.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    partial = path + ".part"
    try:
        with open(partial, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(partial, path)
    except Exception:
        try:
            os.remove(partial)
        except OSError:
            pass
        raise


def write_cmx(sequence: Sequence, output_path: str | None = None) -> WrittenExport:
    """Write a CMX 3600 EDL. ``output_path`` defaults to ``{stem}_edit.edl``."""
    text, skipped = cmx_text(sequence)
    if output_path is None:
        directory = os.path.dirname(os.path.abspath(sequence.source.path))
        output_path = os.path.join(directory, f"{_stem(sequence.source.path)}_edit.edl")
    _write_text(output_path, text)
    return WrittenExport(path=output_path, skipped=skipped)


class TimelineExporter:
    """Export edit timeline to various formats"""

    @staticmethod
    def to_edl(clips, video_path, output_path=None, fps=30, *,
               source: MediaSource | None = None,
               record_start: str = RECORD_START_ZERO,
               spans: tuple = ()):
        """Write a CMX 3600 EDL for ``clips``.

        Pass ``source`` to use the probed frame rate. Without it, ``fps`` is
        a whole-number stand-in so existing callers still get a file; the
        export dialog will probe instead of relying on that.
        """
        if source is None:
            source = MediaSource(
                path=str(video_path), duration=0.0,
                fps_num=int(fps), fps_den=1,
                width=0, height=0, has_audio=False,
                audio_rate=0, audio_channels=0,
            )
        sequence = Sequence(
            title=_stem(source.path or str(video_path)),
            source=source,
            clips=tuple((float(start), float(end)) for start, end in clips),
            record_start=record_start,
            spans=tuple(spans),
        )
        return write_cmx(sequence, output_path)
    
    @staticmethod
    def to_fcp_xml(clips, video_path, output_path=None, fps=30):
        """
        Export to Final Cut Pro XML format (DaVinci Resolve compatible)
        
        Args:
            clips: List of (start_time, end_time) tuples in seconds
            video_path: Path to source video
            output_path: Output file path (None = auto-generate)
            fps: Frames per second
        """
        if not clips:
            return None
            
        if output_path is None:
            base = os.path.splitext(video_path)[0]
            output_path = f"{base}_edit.xml"
        
        video_name = os.path.basename(video_path)
        
        # Create XML structure
        fcpxml = ET.Element("fcpxml", version="1.9")
        resources = ET.SubElement(fcpxml, "resources")
        library = ET.SubElement(fcpxml, "library")
        event = ET.SubElement(library, "event", name="AI Video Edit")
        project = ET.SubElement(event, "project", name="Edited Timeline")
        sequence = ET.SubElement(project, "sequence", format="r1")
        
        # Add format
        format_elem = ET.SubElement(resources, "format", 
                                   id="r1",
                                   name="FFVideoFormat1080p2997",
                                   frameDuration="1001/30000",
                                   width="1920",
                                   height="1080")
        
        # Add asset
        asset_id = f"asset-{hash(video_path) % 10000}"
        asset = ET.SubElement(resources, "asset",
                            id=asset_id,
                            name=video_name,
                            src=f"file://{video_path}")
        
        # Media duration
        duration_sec = clips[-1][1] - clips[0][0] if clips else 60
        duration_frames = int(duration_sec * fps)
        
        # Add sequence
        spine = ET.SubElement(sequence, "spine")
        
        # Total duration in frames
        total_duration = sum(end - start for start, end in clips)
        sequence.set("duration", f"{int(total_duration * fps * 100)}s")
        
        # Add each clip
        for i, (start, end) in enumerate(clips, 1):
            duration = end - start
            duration_frames = int(duration * fps * 100)
            start_frames = int(start * fps * 100)
            
            clip = ET.SubElement(spine, "clip",
                               name=f"Clip {i}",
                               duration=f"{duration_frames}s",
                               start=f"{start_frames}s")
            
            # Add video
            video = ET.SubElement(clip, "video")
            ET.SubElement(video, "offset", relative="start", value=f"{start_frames}s")
            
            # Add audio
            audio = ET.SubElement(clip, "audio")
            ET.SubElement(audio, "offset", relative="start", value=f"{start_frames}s")
            
            ET.SubElement(clip, "asset-ref", id=asset_id)
        
        # Pretty print
        xml_str = ET.tostring(fcpxml, encoding='utf-8')
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="  ")
        
        # Remove XML declaration if minidom adds it weird
        lines = pretty_xml.split('\n')
        if lines[0].startswith('<?xml'):
            lines[0] = '<?xml version="1.0" encoding="utf-8"?>'
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return output_path
    
    @staticmethod
    def get_export_formats():
        """Return list of available export formats"""
        return [
            ("EDL (CMX3600)", "*.edl"),
            ("FCPXML (DaVinci Resolve)", "*.xml"),
            ("CSV", "*.csv"),
            ("JSON", "*.json")
        ]
    
    @staticmethod
    def export_auto(clips, video_path, format='edl'):
        """Auto-export based on format name"""
        format = format.lower()
        if format == 'edl':
            return TimelineExporter.to_edl(clips, video_path)
        elif format in ('fcpxml', 'xml', 'fcp'):
            return TimelineExporter.to_fcp_xml(clips, video_path)
        else:
            return None