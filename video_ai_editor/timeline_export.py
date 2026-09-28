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


class TimelineExporter:
    """Export edit timeline to various formats"""
    
    @staticmethod
    def to_edl(clips, video_path, output_path=None, fps=30):
        """Export to CMX3600 EDL - DaVinci Resolve compatible"""
        
        def seconds_to_timecode(seconds, fps=30, drop_frame=False):
            """Convert seconds to SMPTE timecode"""
            total_frames = int(round(seconds * fps))
            hours = total_frames // (3600 * fps)
            minutes = (total_frames // (60 * fps)) % 60
            secs = (total_frames // fps) % 60
            frames = total_frames % fps
            return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"
        
        lines = []
        
        # Header
        lines.append("TITLE: AI Video Editor Edit")
        lines.append("FCM: NON-DROP FRAME")
        lines.append("")
        
        # Reel name from filename (without extension)
        reel_name = os.path.splitext(os.path.basename(video_path))[0]
        # Limit reel name to 8 chars for compatibility
        reel_name = reel_name[:8].upper()
        
        # Add source file reference
        lines.append(f"* SOURCE FILE: {video_path}")
        lines.append("")
        
        # Each clip
        for i, (start, end) in enumerate(clips, 1):
            duration = end - start
            
            # Calculate cumulative time for record track
            record_start = sum(clips[j][1] - clips[j][0] for j in range(i-1))
            record_end = record_start + duration
            
            # Convert to timecode
            source_in = seconds_to_timecode(start, fps)
            source_out = seconds_to_timecode(end, fps)
            record_in = seconds_to_timecode(record_start, fps)
            record_out = seconds_to_timecode(record_end, fps)
            
            # EDL entry - proper format for DaVinci Resolve
            lines.append(f"{i:03d}  {reel_name:8} V     C        {source_in} {source_out} {record_in} {record_out}")
            lines.append(f"* FROM CLIP NAME: {os.path.basename(video_path)}")
            lines.append(f"* COMMENT: Clip {i} - {duration:.1f}s")
            lines.append("")
        
        # Write file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        return output_path
    
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