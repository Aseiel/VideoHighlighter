"""Edit-timeline export: a sequence model and the files an NLE can open.

The timebase and the sequence types are pure data. Writers below them turn a
:class:`Sequence` into CMX 3600 and FCPXML. Frame counts always come from the
``r_frame_rate`` fraction. A float such as 29.97 is not a frame boundary.
"""

from __future__ import annotations

import csv
import io
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.request import pathname2url
from xml.dom import minidom

from modules.media.video_probe import _rotation_from_stream

RECORD_START_ZERO = "00:00:00:00"
RECORD_START_HOUR = "01:00:00:00"
RECORD_STARTS = (RECORD_START_ZERO, RECORD_START_HOUR)

_SPAN_KINDS = ("speech", "detection", "face")

# Same gap SignalTimelineScene uses to join sampled hits into one run.
# Copied so this module does not import the timeline scene.
EVENT_RUN_GAP = 2.0

_KIND_ORDER = {"speech": 0, "face": 1, "detection": 2}
_KIND_COLOR = {"speech": "cyan", "detection": "yellow", "face": "green"}

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


def _as_float(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_marker_text(label: str) -> str:
    """One line. A leading ``*`` would look like a second EDL comment."""
    text = " ".join(str(label).split())
    if text.startswith("*"):
        text = text[1:].lstrip()
    return text


def _display_line(kind: str, label: str) -> tuple[str, str | None]:
    """The shared marker line, and the speech note (the transcript in full)."""
    text = _clean_marker_text(label)
    if kind == "speech":
        if not text:
            return "", None
        return f"Speech: {text}", text
    if kind == "detection":
        if not text:
            return "", None
        return f"Detection: {text}", None
    if not text or text == "Face":
        return "Face", None
    return f"Face: {text}", None


def _truncate_edl(line: str, limit: int = 80) -> str:
    """At most ``limit`` characters, still one line, ellipsis inside the limit."""
    if len(line) <= limit:
        return line
    if limit <= 3:
        return "." * limit
    return line[:limit - 3] + "..."


def _runs(times: list[float]) -> list[tuple[float, float]]:
    """Hits more than ``EVENT_RUN_GAP`` apart start a new span.

    A single hit has equal start and end. The writer treats that as one frame.
    """
    if not times:
        return []
    ordered = sorted(times)
    start = previous = ordered[0]
    runs: list[tuple[float, float]] = []
    for time in ordered[1:]:
        if time - previous > EVENT_RUN_GAP:
            runs.append((start, previous))
            start = time
        previous = time
    runs.append((start, previous))
    return runs


def _speech_spans(cache: dict) -> list[Span]:
    transcript = cache.get("transcript") or {}
    if not isinstance(transcript, dict):
        return []
    spans: list[Span] = []
    for segment in transcript.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        label = str(segment.get("text") or "")
        if not label.strip():
            continue
        start = _as_float(segment.get("start"))
        if start is None:
            continue
        end = _as_float(segment.get("end"))
        if end is None:
            end = start
        if end < start:
            continue
        spans.append(Span(kind="speech", start=start, end=end, label=label))
    return spans


def _detection_spans(cache: dict) -> list[Span]:
    """Actions first, then objects. The same class name is one series of hits."""
    hits: dict[str, list[float]] = {}

    def add(name, timestamp) -> None:
        if not isinstance(name, str):
            return
        cleaned = name.strip()
        when = _as_float(timestamp)
        if not cleaned or when is None:
            return
        hits.setdefault(cleaned, []).append(when)

    for action in cache.get("actions") or []:
        if not isinstance(action, dict):
            continue
        add(action.get("action_name") or action.get("action") or action.get("class"),
            action.get("timestamp", action.get("start_time", action.get("time"))))
    for item in cache.get("objects") or []:
        if not isinstance(item, dict):
            continue
        when = item.get("timestamp", item.get("time"))
        for name in item.get("objects") or []:
            add(name, when)

    spans: list[Span] = []
    for name, times in hits.items():
        for start, end in _runs(times):
            spans.append(Span(kind="detection", start=start, end=end, label=name))
    return spans


def _face_spans(cache: dict) -> list[Span]:
    """One identity per name, or per track when the face has no name."""
    groups: dict[tuple, list[float]] = {}
    labels: dict[tuple, str] = {}
    for entry in cache.get("object_bboxes") or []:
        if not isinstance(entry, dict):
            continue
        if "identity_names" not in entry or "track_ids" not in entry:
            continue
        when = _as_float(entry.get("timestamp"))
        if when is None:
            continue
        names = entry.get("identity_names") or []
        tracks = entry.get("track_ids") or []
        for index, track in enumerate(tracks):
            raw = names[index] if index < len(names) else None
            name = _clean_marker_text(raw) if isinstance(raw, str) else ""
            if name and name != "Face":
                key = ("name", name)
                label = name
            else:
                key = ("track", track)
                label = "Face"
            groups.setdefault(key, []).append(when)
            labels[key] = label
    spans: list[Span] = []
    for key, times in groups.items():
        for start, end in _runs(times):
            spans.append(Span(kind="face", start=start, end=end, label=labels[key]))
    return spans


def spans_from_analysis(cache) -> tuple[Span, ...]:
    """Speech, detections, and faces in ``cache``.

    Visibility, confidence, and the merge slider are display preferences.
    They are not read. An empty cache produces an empty tuple.
    """
    if not isinstance(cache, dict):
        return ()
    return tuple(_speech_spans(cache) + _detection_spans(cache) + _face_spans(cache))


@dataclass(frozen=True)
class _Mark:
    """One overlap of a span and a kept clip, in one clock's frame index."""

    kind: str
    label: str
    line: str
    note: str | None
    src_in: int
    src_out: int


def _overlap_bounds(span: Span, clip_start: float, clip_end: float) -> tuple[float, float] | None:
    """The seconds of ``span`` that fall inside the clip, or a point for one hit."""
    if span.start == span.end:
        if span.start < clip_start or span.start >= clip_end:
            return None
        return span.start, span.start
    if span.end <= clip_start or span.start >= clip_end:
        return None
    return max(span.start, clip_start), min(span.end, clip_end)


def _overlaps(spans, clip_start: float, clip_end: float,
              clip_in: int, clip_out: int, index_of) -> list[_Mark]:
    """Overlaps on ``index_of``'s clock. A zero-frame overlap is dropped.

    A span whose start and end are the same time is one frame, clipped to
    the clip.
    """
    marks: list[_Mark] = []
    for span in spans:
        bounds = _overlap_bounds(span, clip_start, clip_end)
        if bounds is None:
            continue
        ov_start, ov_end = bounds
        src_in = index_of(ov_start)
        src_out = src_in + 1 if ov_end == ov_start else index_of(ov_end)
        src_in = max(src_in, clip_in)
        src_out = min(src_out, clip_out)
        if src_out <= src_in:
            continue
        line, note = _display_line(span.kind, span.label)
        if not line:
            continue
        marks.append(_Mark(span.kind, _clean_marker_text(span.label) or span.label,
                           line, note, src_in, src_out))
    marks.sort(key=lambda mark: (mark.src_in, _KIND_ORDER[mark.kind], mark.label))
    return marks


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
    kept: list[tuple[float, float, int, int, int]] = []
    skipped = 0
    fps = timebase.edl_fps
    record = record_start_seconds(sequence.record_start) * fps
    for start, end in sequence.clips:
        quantised = _quantise_clip(start, end, timebase)
        if quantised is None:
            skipped += 1
            continue
        src_in, src_out = quantised
        kept.append((float(start), float(end), src_in, src_out, record))
        record += src_out - src_in
    if not kept:
        raise ExportError(
            f"every clip is shorter than one frame at {timebase.edl_fps} fps")

    stem = _stem(sequence.source.path)
    reel = reel_name(stem)
    filename = os.path.basename(sequence.source.path)
    source_file = os.path.abspath(sequence.source.path)

    lines = [
        f"TITLE: {stem}",
        "FCM: NON-DROP FRAME",
        "* SOURCE TIMES ARE FROM THE START OF THE FILE, NOT CAMERA TIMECODE",
        "",
    ]
    locators: list[tuple[int, _Mark]] = []
    for index, (start, end, src_in, src_out, rec_in) in enumerate(kept):
        number = _event_number(index)
        duration = src_out - src_in
        rec_out = rec_in + duration
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
        for mark in _overlaps(
                sequence.spans, start, end, src_in, src_out,
                lambda seconds: edl_frame_index(seconds, timebase)):
            locators.append((rec_in + (mark.src_in - src_in), mark))

    if locators:
        locators.sort(key=lambda item: (
            item[0], _KIND_ORDER[item[1].kind], item[1].label))
        lines.append("")
        for record_frame, mark in locators:
            timecode = frames_to_timecode(record_frame, fps)
            text = _truncate_edl(mark.line)
            lines.append(f"* LOC: {timecode} {_KIND_COLOR[mark.kind]} {text}")

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


def _source_clip(start: float, end: float, timebase: Timebase) -> tuple[int, int] | None:
    """Source in/out in real frames, or None when the clip has no frames.

    FCPXML keeps every real frame. The EDL clock is coarser above 100 fps,
    so a one-frame clip at 120 fps is kept here and dropped by the EDL.
    """
    if end <= start:
        return None
    src_in = to_frames(start, timebase.fps_num, timebase.fps_den)
    src_out = to_frames(end, timebase.fps_num, timebase.fps_den)
    if src_out <= src_in:
        return None
    return src_in, src_out


def _format_rational(num: int, den: int) -> str:
    """Reduced rational seconds, ``5s`` or ``1001/200s``."""
    if den <= 0:
        raise ExportError(f"time denominator {den} is not positive")
    if num <= 0:
        return "0s"
    factor = math.gcd(int(num), int(den))
    num //= factor
    den //= factor
    if den == 1:
        return f"{num}s"
    return f"{num}/{den}s"


def _parse_rational(token: str) -> tuple[int, int]:
    body = token[:-1] if token.endswith("s") else token
    if "/" not in body:
        return int(body), 1
    num, den = body.split("/", 1)
    return int(num), int(den)


def _add_rational(left: str, right: str) -> str:
    left_num, left_den = _parse_rational(left)
    right_num, right_den = _parse_rational(right)
    return _format_rational(
        left_num * right_den + right_num * left_den,
        left_den * right_den,
    )


def frames_to_time(frames: int, timebase: Timebase) -> str:
    """``frames`` of the real frame duration, reduced.

    A 30 fps file writes ``5s`` for 150 frames, not ``15000/3000s``.
    """
    if frames <= 0:
        return "0s"
    return _format_rational(int(frames) * timebase.fps_den, timebase.fps_num)


def file_url(path: str) -> str:
    """``file://`` URL. Spaces and non-ASCII are percent-encoded.

    ``pathname2url`` leaves the leading slash, so a POSIX path becomes
    ``file:///...`` and a Windows path becomes ``file:///C:/...``.
    """
    return "file://" + pathname2url(os.path.abspath(path))


def _sequence_audio_rate(rate: int) -> str:
    """FCPXML sequence rates are named (``48k``); the asset uses hertz."""
    named = {44100: "44.1k", 88200: "88.2k", 176400: "176.4k"}
    if rate in named:
        return named[rate]
    if rate > 0 and rate % 1000 == 0:
        return f"{rate // 1000}k"
    return str(rate)


def _record_origin(record_start: str) -> str:
    """Sequence ``tcStart``. One hour is ``3600s`` on every rate."""
    seconds = record_start_seconds(record_start)
    if seconds == 0:
        return "0s"
    return _format_rational(seconds, 1)


def _serialize_fcpxml(root: ET.Element) -> str:
    pretty = minidom.parseString(
        ET.tostring(root, encoding="unicode")).toprettyxml(indent="  ")
    lines = [line for line in pretty.splitlines() if line.strip()]
    if lines and lines[0].startswith("<?xml"):
        lines[0] = '<?xml version="1.0" encoding="UTF-8"?>'
    else:
        lines.insert(0, '<?xml version="1.0" encoding="UTF-8"?>')
    lines.insert(1, "<!DOCTYPE fcpxml>")
    return "\n".join(lines) + "\n"


def fcpxml_text(sequence: Sequence) -> tuple[str, int]:
    """FCPXML 1.9 for ``sequence``, and how many clips were skipped.

    Raises :class:`ExportError` when there is nothing to write. Does not
    touch the disk. A span is a marker on each clip it overlaps.
    """
    if not sequence.clips:
        raise ExportError("nothing to export — the edit timeline has no clips")

    timebase = sequence.timebase()
    kept: list[tuple[float, float, int, int]] = []
    skipped = 0
    for start, end in sequence.clips:
        quantised = _source_clip(start, end, timebase)
        if quantised is None:
            skipped += 1
            continue
        kept.append((float(start), float(end), quantised[0], quantised[1]))
    if not kept:
        raise ExportError(
            "every clip is shorter than one frame at "
            f"{timebase.fps_num}/{timebase.fps_den}")

    source = sequence.source
    filename = os.path.basename(source.path) or "untitled"
    fmt = ET.Element("format", {
        "id": "r1",
        "name": f"FFVideoFormat{source.height}p{timebase.rate_token}",
        "frameDuration": timebase.frame_duration,
        "width": str(source.width),
        "height": str(source.height),
    })
    asset_attrib = {
        "id": "r2",
        "name": filename,
        "start": "0s",
        "duration": frames_to_time(
            to_frames(source.duration, timebase.fps_num, timebase.fps_den),
            timebase,
        ),
        "hasVideo": "1",
        "hasAudio": "1" if source.has_audio else "0",
        "format": "r1",
    }
    if source.has_audio:
        asset_attrib["audioSources"] = "1"
        asset_attrib["audioChannels"] = str(source.audio_channels)
        asset_attrib["audioRate"] = str(source.audio_rate)
    asset = ET.Element("asset", asset_attrib)
    ET.SubElement(asset, "media-rep", {
        "kind": "original-media",
        "src": file_url(source.path),
    })
    resources = ET.Element("resources")
    resources.append(fmt)
    resources.append(asset)

    origin = _record_origin(sequence.record_start)
    total_frames = sum(src_out - src_in for _, _, src_in, src_out in kept)
    sequence_attrib = {
        "format": "r1",
        "duration": frames_to_time(total_frames, timebase),
        "tcStart": origin,
        "tcFormat": "NDF",
    }
    if source.has_audio and source.audio_channels == 2:
        sequence_attrib["audioLayout"] = "stereo"
    if source.has_audio and source.audio_rate:
        sequence_attrib["audioRate"] = _sequence_audio_rate(source.audio_rate)
    sequence_el = ET.Element("sequence", sequence_attrib)
    spine = ET.SubElement(sequence_el, "spine")
    offset = origin
    index_of = lambda seconds: to_frames(seconds, timebase.fps_num, timebase.fps_den)
    for index, (start, end, src_in, src_out) in enumerate(kept, start=1):
        duration = frames_to_time(src_out - src_in, timebase)
        clip = ET.SubElement(spine, "asset-clip", {
            "ref": "r2",
            "offset": offset,
            "name": f"Clip {index}",
            "start": frames_to_time(src_in, timebase),
            "duration": duration,
            "tcFormat": "NDF",
        })
        for mark in _overlaps(sequence.spans, start, end, src_in, src_out, index_of):
            attrib = {
                "start": frames_to_time(mark.src_in - src_in, timebase),
                "duration": frames_to_time(mark.src_out - mark.src_in, timebase),
                "value": mark.line,
            }
            if mark.note is not None:
                attrib["note"] = mark.note
            ET.SubElement(clip, "marker", attrib)
        offset = _add_rational(offset, duration)

    project = ET.Element("project", {"name": sequence.title or _stem(source.path)})
    project.append(sequence_el)
    event = ET.Element("event", {"name": "VideoHighlighter"})
    event.append(project)
    library = ET.Element("library")
    library.append(event)
    root = ET.Element("fcpxml", {"version": "1.9"})
    root.append(resources)
    root.append(library)
    return _serialize_fcpxml(root), skipped


_CSV_COLUMNS = (
    "Clip", "Start (s)", "End (s)", "Duration (s)", "Frame in", "Frame out",
)


def _csv_seconds(frames: int, timebase: Timebase) -> str:
    """Quantised seconds at a real frame boundary.

    Six decimal places, then trailing zeros drop back to two, so a 30 fps
    cut reads ``10.00`` and a 29.97 cut keeps the fraction the frame is.
    """
    if frames < 0:
        return "-" + _csv_seconds(-frames, timebase)
    numerator = int(frames) * timebase.fps_den
    denominator = timebase.fps_num
    scaled, remainder = divmod(numerator * 1_000_000, denominator)
    # Half to even, matching the frame counter's rounding.
    if remainder * 2 > denominator or (remainder * 2 == denominator and scaled % 2):
        scaled += 1
    whole, frac = divmod(int(scaled), 1_000_000)
    digits = f"{frac:06d}".rstrip("0")
    if len(digits) < 2:
        digits = digits.ljust(2, "0")
    return f"{whole}.{digits}"


def csv_text(sequence: Sequence) -> tuple[str, int]:
    """Spreadsheet rows for ``sequence``, and how many clips were skipped.

    Columns are the quantised source range. There is no sequence clock and
    no marker row. Raises :class:`ExportError` when there is nothing to write.
    """
    if not sequence.clips:
        raise ExportError("nothing to export — the edit timeline has no clips")

    timebase = sequence.timebase()
    kept: list[tuple[int, int]] = []
    skipped = 0
    for start, end in sequence.clips:
        quantised = _source_clip(start, end, timebase)
        if quantised is None:
            skipped += 1
            continue
        kept.append(quantised)
    if not kept:
        raise ExportError(
            "every clip is shorter than one frame at "
            f"{timebase.fps_num}/{timebase.fps_den}")

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_CSV_COLUMNS)
    for number, (src_in, src_out) in enumerate(kept, start=1):
        writer.writerow([
            number,
            _csv_seconds(src_in, timebase),
            _csv_seconds(src_out, timebase),
            _csv_seconds(src_out - src_in, timebase),
            src_in,
            src_out,
        ])
    return buffer.getvalue(), skipped


def write_csv(sequence: Sequence, output_path: str | None = None) -> WrittenExport:
    """Write the clip spreadsheet. ``output_path`` defaults to ``{stem}_edit.csv``."""
    text, skipped = csv_text(sequence)
    if output_path is None:
        directory = os.path.dirname(os.path.abspath(sequence.source.path))
        output_path = os.path.join(
            directory, f"{_stem(sequence.source.path)}_edit.csv")
    _write_text(output_path, text)
    return WrittenExport(path=output_path, skipped=skipped)


def write_fcpxml(sequence: Sequence, output_path: str | None = None) -> WrittenExport:
    """Write FCPXML 1.9. ``output_path`` defaults to ``{stem}_edit.fcpxml``."""
    text, skipped = fcpxml_text(sequence)
    if output_path is None:
        directory = os.path.dirname(os.path.abspath(sequence.source.path))
        output_path = os.path.join(
            directory, f"{_stem(sequence.source.path)}_edit.fcpxml")
    _write_text(output_path, text)
    return WrittenExport(path=output_path, skipped=skipped)


def _parse_rate(text: str) -> tuple[int, int] | None:
    """``num/den`` from an ffprobe rate, or None when it is not a fraction."""
    num_text, slash, den_text = str(text or "").partition("/")
    if not slash:
        return None
    try:
        return int(num_text), int(den_text)
    except ValueError:
        return None


def _frame_rate(stream: dict) -> tuple[int, int]:
    """``r_frame_rate``, or ``avg_frame_rate`` when that rate is ``0/0``."""
    raw = stream.get("r_frame_rate")
    parsed = _parse_rate(raw)
    if parsed == (0, 0) or not raw:
        parsed = _parse_rate(stream.get("avg_frame_rate"))
    if parsed is None or parsed[0] <= 0 or parsed[1] <= 0:
        raise ExportError("the probe did not report a frame rate")
    return parsed


def _as_int(value) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _positive_float(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number > 0 else 0.0


def media_source_from_probe(path: str, data: dict) -> MediaSource:
    """One ``MediaSource`` from an ffprobe-shaped document.

    Width and height are the display size: a 90° or 270° rotation swaps
    the stored dimensions. The file's own rotation metadata is not rewritten.
    """
    if not isinstance(data, dict):
        raise ExportError(f"the probe of {path} did not return a media description")
    streams = data.get("streams") or []
    video = next((stream for stream in streams
                  if isinstance(stream, dict) and stream.get("codec_type") == "video"),
                 None)
    if video is None:
        raise ExportError(f"no video stream in {path}")
    fps_num, fps_den = _frame_rate(video)
    container = data.get("format") if isinstance(data.get("format"), dict) else {}
    duration = _positive_float(container.get("duration")) or _positive_float(
        video.get("duration"))
    width = _as_int(video.get("width"))
    height = _as_int(video.get("height"))
    if _rotation_from_stream(video) in (90, 270):
        width, height = height, width
    audio = next((stream for stream in streams
                  if isinstance(stream, dict) and stream.get("codec_type") == "audio"),
                 None)
    return MediaSource(
        path=str(path),
        duration=duration,
        fps_num=fps_num,
        fps_den=fps_den,
        width=width,
        height=height,
        has_audio=audio is not None,
        audio_rate=_as_int(audio.get("sample_rate")) if audio else 0,
        audio_channels=_as_int(audio.get("channels")) if audio else 0,
    )


def probe_media_source(path: str) -> MediaSource:
    """Probe ``path`` once. A failure propagates and nothing is written."""
    from modules.media.ffmpeg_tools import probe
    return media_source_from_probe(path, probe(path))


def _export_sequence(clips, video_path, source: MediaSource | None,
                     record_start: str, spans) -> Sequence:
    """Build the sequence. A missing ``source`` probes ``video_path`` once."""
    if source is None:
        source = probe_media_source(str(video_path))
    return Sequence(
        title=_stem(source.path or str(video_path)),
        source=source,
        clips=tuple((float(start), float(end)) for start, end in clips),
        record_start=record_start,
        spans=tuple(spans),
    )


def count_markers(sequence: Sequence) -> int:
    """Marker overlaps FCPXML would write, counted on the real frame clock.

    A span that crosses two clips counts twice. A span that misses every
    clip, and an overlap that quantises to zero frames, count as nothing.
    """
    timebase = sequence.timebase()

    def index_of(seconds: float) -> int:
        return to_frames(seconds, timebase.fps_num, timebase.fps_den)

    total = 0
    for start, end in sequence.clips:
        quantised = _source_clip(start, end, timebase)
        if quantised is None:
            continue
        src_in, src_out = quantised
        total += len(_overlaps(
            sequence.spans, start, end, src_in, src_out, index_of))
    return total


def export_summary(clip_count: int, duration: float, timebase: Timebase,
                   markers: int) -> str:
    """The info line under the format combo."""
    mark = "marker" if markers == 1 else "markers"
    text = (
        f"Exporting {clip_count} clips, total duration: {duration:.1f}s. "
        f"{timebase.describe()}. {markers} {mark}."
    )
    if timebase.coarsened:
        text += (
            f" FCPXML keeps every frame. "
            f"The EDL is counted at {timebase.edl_fps} fps."
        )
    return text


def record_start_for_format(format_name: str, chosen: str) -> str:
    """Sequence start for this format. CSV has no sequence clock."""
    if str(format_name).startswith("CSV"):
        return RECORD_START_ZERO
    if chosen not in RECORD_STARTS:
        raise ExportError(
            f"record start must be {RECORD_START_ZERO} or {RECORD_START_HOUR}, "
            f"got {chosen!r}")
    return chosen


def skipped_note(count: int) -> str:
    """Success-message suffix. Zero skips add nothing."""
    if count <= 0:
        return ""
    if count == 1:
        return "\n1 clip was shorter than one frame and was skipped."
    return f"\n{count} clips were shorter than one frame and were skipped."


def default_export_path(video_path: str, pattern: str) -> tuple[str, str]:
    """Save path beside the source, and the file-dialog filter for ``pattern``."""
    directory = os.path.dirname(os.path.abspath(str(video_path)))
    stem = _stem(video_path)
    filters = {
        "*.edl": ("edl", "EDL files (*.edl)"),
        "*.fcpxml": ("fcpxml", "FCPXML files (*.fcpxml)"),
        "*.csv": ("csv", "CSV files (*.csv)"),
    }
    chosen = filters.get(pattern)
    if chosen is None:
        raise ExportError(f"unknown export pattern {pattern!r}")
    extension, file_filter = chosen
    return os.path.join(directory, f"{stem}_edit.{extension}"), file_filter


def prepare_export(video_path, clips, cache, duration):
    """Probe once and describe the export. Does not write a file.

    A probe failure propagates. The summary counts markers on the real
    frame clock, which is what FCPXML writes.
    """
    source = probe_media_source(str(video_path))
    spans = spans_from_analysis(cache)
    sequence = _export_sequence(
        clips, video_path, source, RECORD_START_ZERO, spans)
    summary = export_summary(
        len(clips), float(duration), sequence.timebase(), count_markers(sequence))
    return source, spans, summary


class TimelineExporter:
    """Export edit timeline to various formats"""

    @staticmethod
    def to_edl(clips, video_path, output_path=None, fps=30, *,
               source: MediaSource | None = None,
               record_start: str = RECORD_START_ZERO,
               spans: tuple = ()):
        """Write a CMX 3600 EDL for ``clips``.

        Pass ``source`` to skip probing. Without it, ``video_path`` is probed
        once and ``fps`` is not used as a frame rate.
        """
        del fps
        sequence = _export_sequence(
            clips, video_path, source, record_start, spans)
        return write_cmx(sequence, output_path)
    
    @staticmethod
    def to_fcp_xml(clips, video_path, output_path=None, fps=30, *,
                   source: MediaSource | None = None,
                   record_start: str = RECORD_START_ZERO,
                   spans: tuple = ()):
        """Write an FCPXML 1.9 sequence for ``clips``.

        Pass ``source`` to skip probing. Without it, ``video_path`` is probed
        once and ``fps`` is not used as a frame rate.
        """
        del fps
        sequence = _export_sequence(
            clips, video_path, source, record_start, spans)
        return write_fcpxml(sequence, output_path)

    @staticmethod
    def to_csv(clips, video_path, output_path=None, fps=30, *,
               source: MediaSource | None = None,
               record_start: str = RECORD_START_ZERO,
               spans: tuple = ()):
        """Write a spreadsheet of ``clips``.

        Times are quantised to the real frame rate. ``record_start`` is
        accepted and ignored: a CSV has no sequence clock. Spans are not rows.
        Pass ``source`` to skip probing. Without it, ``video_path`` is probed
        once and ``fps`` is not used as a frame rate.
        """
        del fps
        sequence = _export_sequence(
            clips, video_path, source, record_start, spans)
        return write_csv(sequence, output_path)

    @staticmethod
    def get_export_formats():
        """The three formats the export dialog offers."""
        return [
            ("EDL (CMX 3600)", "*.edl"),
            ("FCPXML", "*.fcpxml"),
            ("CSV", "*.csv"),
        ]

    @staticmethod
    def export_auto(clips, video_path, format='edl'):
        """Dispatch ``format`` to its own writer. Unknown names write nothing."""
        key = str(format).lower()
        if key == "edl":
            return TimelineExporter.to_edl(clips, video_path)
        if key in ("fcpxml", "xml", "fcp"):
            return TimelineExporter.to_fcp_xml(clips, video_path)
        if key == "csv":
            return TimelineExporter.to_csv(clips, video_path)
        raise ExportError(
            f"unknown export format {format!r}; choose edl, fcpxml, or csv")