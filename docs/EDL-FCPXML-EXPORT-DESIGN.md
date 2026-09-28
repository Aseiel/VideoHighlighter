# Design: EDL and FCPXML export

How the Timeline Viewer's edit timeline becomes a CMX 3600 EDL and an FCPXML
1.9 sequence. What the feature is for, and what it deliberately leaves alone,
is in [EDL-FCPXML-EXPORT.md](EDL-FCPXML-EXPORT.md).

The writers live in `video_ai_editor/timeline_export.py`, which is already
imported by the viewer and, through it, by the chat command. That module has
no Qt in it. It stays there. `modules/media/edl.py` is a different format and
is not involved.

## Sequence model

Both writers take one value, built by the caller from the clips it already
holds. Building it is separate from formatting it, so the tests can hand in a
known frame rate and never call ffprobe.

```python
@dataclass(frozen=True)
class MediaSource:
    path: str
    duration: float          # seconds, from the container
    fps_num: int             # r_frame_rate, e.g. 30000
    fps_den: int             # e.g. 1001
    width: int               # display width (rotation already applied)
    height: int              # display height
    has_audio: bool
    audio_rate: int          # 0 when there is no audio stream
    audio_channels: int

@dataclass(frozen=True)
class Span:
    kind: str                # "speech", "detection", or "face"
    start: float             # source seconds
    end: float
    label: str               # the line, the class, or the face's name

@dataclass(frozen=True)
class Sequence:
    title: str               # source filename without extension
    source: MediaSource
    clips: tuple[tuple[float, float], ...]   # (in, out) seconds, edit order
    record_start: str = "00:00:00:00"        # or "01:00:00:00"
    spans: tuple[Span, ...] = ()
```

`spans_from_analysis(cache)` builds the tuple from the viewer's analysis
cache. The writer does not import the timeline scene. Tests pass spans in
directly. An empty cache produces an empty tuple, and the cut is written
with no markers.

`clips` is `EditTimelineScene.clips`: source seconds, edit order, one file.
A later multi-file export would add a source index on each clip. v1 has one
source, so the index would only be noise.

`TimelineExporter.to_edl` and `to_fcp_xml` keep those names. They gain a
`source: MediaSource | None = None` argument. When it is omitted they probe
`video_path` themselves, which is what both callers want. Tests pass a
`MediaSource` and touch no media.

## Timebase

Times in the app are seconds. Times in both files are frame counts. The
conversion happens once:

```python
def to_frames(seconds: float, num: int, den: int) -> int:
    return int(round(seconds * num / den))
```

A clip from frame `in` inclusive to frame `out` exclusive lasts
`out - in` frames. CMX source-out and FCPXML `start + duration` both use that
exclusive end, so the two files describe the same frames.

A clip whose `in` and `out` quantise to the same frame is omitted. It is
shorter than a frame; writing it as one frame would add picture the user did
not pick. The writer returns how many clips it skipped, and the button puts
that count in the success message. If every clip is skipped, the writer
raises and no file is written.

`r_frame_rate` is the fraction, never a float. `29.97 * t` is not a frame
boundary.

CMX timecode counts at the nominal rate, not the real one. 29.97 non-drop
media is numbered 30 frames to the second. The same is true of 23.976 (count
at 24) and 59.94 (count at 60). FCPXML does not have this split: its
`frameDuration` is the real frame length.

| `r_frame_rate` | CMX counts at | FCPXML `frameDuration` | `tcFormat` |
|---|---|---|---|
| 24000/1001 | 24 | `1001/24000s` | NDF |
| 24/1 | 24 | `100/2400s` | NDF |
| 25/1 | 25 | `100/2500s` | NDF |
| 30000/1001 | 30 | `1001/30000s` | NDF |
| 30/1 | 30 | `100/3000s` | NDF |
| 48/1 | 48 | `100/4800s` | NDF |
| 50/1 | 50 | `100/5000s` | NDF |
| 60000/1001 | 60 | `1001/60000s` | NDF |
| 60/1 | 60 | `100/6000s` | NDF |
| 100/1 | 100 | `100/10000s` | NDF |
| 120/1 | 60 | `100/12000s` | NDF |
| 240/1 | 60 | `100/24000s` | NDF |

An Insta360 X6 file uses these whole rates: 24, 25, 30, 48, 50, 60, 100,
120, and 240. FCPXML is frame-accurate at every one of them.
`frameDuration` is the real frame length, so a 120 fps file is `100/12000s`
and a 240 fps file is `100/24000s`. Neither is rewritten as 30.

CMX timecode has two digits for the frame, so the frame number in one second
has to be 0–99. 48 and 100 fit: a 100 fps second is frames `00` through `99`,
and the EDL counts at 100. 120 and 240 do not fit. A frame field of `120`
is not a CMX timecode.

For those two, the EDL counts at 60, the largest whole rate at or under 60
that divides them. One EDL frame is two source frames at 120 fps and four at
240 fps. Each cut is rounded to the nearest 60 fps boundary, so it can move
by at most one source frame at 120 fps and two at 240 fps. The FCPXML of the
same export is not rounded. The dialog says this when the file is over 100
fps, including which format is the frame-accurate one, so a 120 fps file
cannot pass for 30.

Any other fraction is reduced and counted at `round(num/den)`, with
`frameDuration="{den}/{num}s"`. When that nominal rate is over 100, the EDL
uses the same divisor rule. The dialog names the rate. It says "unrecognised"
only when the fraction is neither one of the rows above nor a whole number
of frames per second.

Header for both formats is non-drop (`FCM: NON-DROP FRAME`, `tcFormat="NDF"`).
Drop-frame is a numbering scheme. v1 does not write it.

Clips abut. The edit timeline has no gaps. The assembled timeline is numbered
from a start the user picks in the export dialog. There are two legal values:

| Dialog | Default | EDL record in of the first clip | FCPXML |
|---|---|---|---|
| `00:00:00:00` | yes | `00:00:00:00` | `tcStart="0s"`, first `offset="0s"` |
| `01:00:00:00` | | `01:00:00:00` | `tcStart="3600s"`, first `offset="3600s"` |

A five-second first clip then occupies the next five seconds of that clock,
and the clip after it starts where this one ends. Both files get the same
choice, so an EDL and an FCPXML written from one dialog agree.

`01:00:00:00` is the videotape house origin, not a CMX requirement. Masters
were striped so the programme began one hour in: the hour before it held
bars, tone, slate, and black, and a deck could park before the programme
without wrapping through zero. Resolve's new timeline still starts there. It
is the choice for someone importing into a stock Resolve timeline. It is not
the default here, because the edit timeline the user just built starts at
zero, and so does a new Final Cut project.

FCPXML `offset` is absolute on that clock. With the one-hour start, the first
clip's `offset` equals `tcStart`, not `0s`. An offset of zero against a
`tcStart` of one hour places the clip an hour before the sequence.

Source in/out are the other clock. In both files they count from the first
frame of the media (`00:00:00:00`, asset `start="0s"`), at either record
start. The dialog does not move them. CSV lists those source ranges and has
no sequence clock, so the control is disabled when CSV is selected.

## Probing

One ffprobe JSON call, through `modules.media.ffmpeg_tools.probe`, at export
time. `probe_video` is not enough: it collapses the frame rate to a float and
says nothing about audio.

From that JSON:

- video `r_frame_rate` parsed as `num/den` (fall back to `avg_frame_rate` only
  when `r_frame_rate` is `0/0`),
- container duration,
- stored width and height, then swapped when rotation is 90 or 270 — the same
  rotation rules as `video_probe._rotation_from_stream`, so a portrait phone
  file becomes a portrait sequence,
- first audio stream: present or not, sample rate, channel count.

The sequence `format` in FCPXML uses the display size. The file's own rotation
metadata is left in the file; the NLE applies it when it opens the media.
Putting the stored landscape size on a rotated portrait clip gives a landscape
sequence with a sideways picture.

If the probe fails, export stops with the probe error. It does not assume
30 fps. That assumption is the bug being removed.

No timecode is read. Source `00:00:00:00` is the first frame of the file. The
EDL says this in its header comment, in one line, so a user whose camera
stamps time-of-day knows why the NLE numbers do not match the camera clock.

## CMX 3600

UTF-8, LF line endings. Resolve accepts LF. A TITLE longer than the source
stem is unnecessary; the stem is the title. This sample uses the default
record start, `00:00:00:00`. Choosing `01:00:00:00` adds one hour to every
record column and leaves the source columns alone.

```
TITLE: morning
FCM: NON-DROP FRAME
* SOURCE TIMES ARE FROM THE START OF THE FILE, NOT CAMERA TIMECODE

001  MORNING  V     C        00:00:10:00 00:00:15:00 00:00:00:00 00:00:05:00
* FROM CLIP NAME: morning.mp4
* SOURCE FILE: /Users/eric/Movies/morning.mp4
001  MORNING  A     C        00:00:10:00 00:00:15:00 00:00:00:00 00:00:05:00
* FROM CLIP NAME: morning.mp4

002  MORNING  V     C        00:01:02:10 00:01:08:10 00:00:05:00 00:00:11:00
* FROM CLIP NAME: morning.mp4
* SOURCE FILE: /Users/eric/Movies/morning.mp4
002  MORNING  A     C        00:01:02:10 00:01:08:10 00:00:05:00 00:00:11:00
* FROM CLIP NAME: morning.mp4
```

Event line, matching the spacing Resolve already accepts from the current
writer:

```text
{n:03d}  {reel:8} {track:5} C        {src_in} {src_out} {rec_in} {rec_out}
```

- Event numbers start at 001. Picture and sound of one cut share a number.
  The next cut takes the next number. Past 999 the number wraps to 001; a
  highlight reel does not get there, and CMX has three digits.
- Reel is the stem, uppercased, restricted to `A–Z` and `0–9`, truncated to
  8 characters, space-padded. `morning` becomes `MORNING `. The real name is
  not the reel. CMX only has eight characters, and Resolve relinks on the
  comment below.
- Track is `V` or `A`, in a 5-character field (`V    `, `A    `).
- Transition is `C`. The timeline has no dissolves.
- Timecode is `HH:MM:SS:FF` at the nominal rate in the table above.
- `* FROM CLIP NAME:` is the filename, not the path. This is the line Resolve
  uses to find the clip.
- `* SOURCE FILE:` is the absolute path. It is a comment, which strict CMX
  ignores, and it is how Resolve relinks when the clip is not already in a bin.

The audio event is omitted when `has_audio` is false. One `A` event is
enough when audio exists: the NLE opens the file and uses its real channel
layout. The EDL does not try to describe a 4-channel or 8-channel split.
`audio_channels` is still probed, because FCPXML wants the number.

## FCPXML 1.9

Version 1.9 is what current Final Cut Pro and Resolve both import. 1.10 and
1.11 add nothing this sequence uses.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE fcpxml>
<fcpxml version="1.9">
  <resources>
    <format id="r1" name="FFVideoFormat1080p2997" frameDuration="1001/30000s"
            width="1920" height="1080"/>
    <asset id="r2" name="morning.mp4" start="0s" duration="1801800/30000s"
           hasVideo="1" hasAudio="1" format="r1"
           audioSources="1" audioChannels="2" audioRate="48000">
      <media-rep kind="original-media" src="file:///Users/eric/Movies/morning.mp4"/>
    </asset>
  </resources>
  <library>
    <event name="VideoHighlighter">
      <project name="morning">
        <sequence format="r1" duration="180180/30000s" tcStart="0s"
                  tcFormat="NDF" audioLayout="stereo" audioRate="48k">
          <spine>
            <asset-clip ref="r2" offset="0s" name="Clip 1"
                        start="300300/30000s" duration="150150/30000s"
                        tcFormat="NDF"/>
            <asset-clip ref="r2" offset="150150/30000s" name="Clip 2"
                        start="1876870/30000s" duration="180180/30000s"
                        tcFormat="NDF"/>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
```

Times are rational seconds in the real frame duration, reduced by their
greatest common divisor so a 30 fps file writes `150/30s` rather than a pile
of unreduced thousands. Frame alignment is what matters; the reduced form is
what a person can check.

- One `format`, one `asset`, one `asset-clip` per clip. Every clip references
  `r2`. The current writer invents a `clip` / `video` / `asset-ref` tree that
  is not in the format.
- `asset` `duration` is the whole file, quantised with `to_frames`. It is not
  the span from the first used frame to the last.
- `asset` `start` is `0s`. Same decision as the EDL: file start, not camera
  timecode.
- `asset-clip` `start` is the source in-point. `duration` is the clip length.
  `offset` is where that clip sits on the sequence clock: the chosen record
  start, plus the sum of the durations before it. At the default that is
  `0s` for the first clip. At `01:00:00:00` the first offset equals `tcStart`
  (`3600s` at a whole-second rate, or the same hour in the sequence's
  `frameDuration`).
- `sequence` `duration` is the sum of the clip durations.
- `media-rep` `src` is a file URL from `urllib.request.pathname2url`, with
  `file://` in front. Spaces and non-ASCII are percent-encoded. A Windows
  path comes out as `file:///C:/...`. A raw `file://` plus an unescaped path
  breaks on the first space.
- `format` `name` is the conventional `FFVideoFormat{height}p{rate}` token
  (`1080p2997`, `1080p25`, `1080p30`). Resolve keys off `frameDuration`,
  `width`, and `height`. The name is there because Final Cut expects one.
  Rate tokens: `2398`, `24`, `25`, `2997`, `30`, `48`, `50`, `5994`, `60`,
  `100`, `120`, `240`. An unrecognised rate uses `p{nominal}`. The token for
  120 and 240 is the real rate, matching `frameDuration`, not the 60 fps
  clock the EDL had to use.
- No `colorSpace`. Guessing Rec. 709 on HDR footage is worse than leaving it
  to the file.
- `hasAudio="0"` and no `audioChannels` / `audioRate` / `audioLayout` when the
  file has no audio stream. When it has one, `audioChannels` is the probed
  count and `audioRate` is the probed rate. `audioLayout` is `stereo` for two
  channels and omitted otherwise, so a mono file is not declared stereo.
- Clip `name` is `Clip 1`, `Clip 2`, in edit order.

The doctype is `<!DOCTYPE fcpxml>`. Final Cut wants it. Resolve ignores it.

## Markers

One span, one marker, in both files. The span is clipped to each exported
clip it overlaps. A span that misses every clip is omitted. A span that
crosses two clips is written twice, once for each overlap.

### Where a span comes from

`spans_from_analysis` reads three lists in the analysis cache.

| Kind | Cache | One span is |
|---|---|---|
| `speech` | `transcript.segments` | One segment that has text. `start` and `end` are the segment's own. The label is the text. |
| `detection` | `actions`, then `objects` | One class name. Hits of that name are sorted by time. A hit more than 2 seconds after the previous one starts a new span. The span runs from the first hit to the last. |
| `face` | `object_bboxes` entries that carry `identity_names` and `track_ids` | One identity. The group key is the name the user gave that face, or the track id when the face has no name. The same 2 second gap applies. The label is the name, or `Face` when there is none. |

The 2 seconds are `SignalTimelineScene.EVENT_RUN_GAP`, the gap already used
to join sampled events into one run. It is copied as a constant in the
export module. The viewer's merge slider defaults to off and is a display
preference, so the export does not read it. Hidden rows and the confidence
sliders do not change the file either. The cache is the record.

A span whose first and last hit are the same time is one frame long. A
detection and a face that occupy the same seconds are both written. They
name different things.

### Clip-relative time

The overlap is `[max(span.start, clip.start), min(span.end, clip.end)]`.
An overlap shorter than a frame after quantising is omitted.

FCPXML measures the marker from the first frame of that `asset-clip`. A clip
of source `10s`–`20s` and a span of `12s`–`16s` produce a marker at `2s`
lasting `4s`. The sequence-start choice does not move it. Clip-relative time
is already inside the clip, and `tcStart` only numbers the sequence.

The EDL locator uses the record clock. The same overlap begins two seconds
into the clip, so with the default record start the line reads
`00:00:02:00`, and with `01:00:00:00` it reads `01:00:02:00`.

### The text

One display line, shared by both files:

| Kind | Line |
|---|---|
| `speech` | `Speech: ` plus the transcript text |
| `detection` | `Detection: ` plus the class name |
| `face` | `Face: ` plus the name, or `Face` when there is no name |

Whitespace, including newlines, collapses to single spaces. A leading `*`
is stripped so a transcript line cannot look like a second EDL comment.

The EDL keeps at most 80 characters of that line, on one line. A longer
transcript is cut and ends with `...` inside the 80. The length of the span
is not written. CMX has nowhere to put it.

FCPXML uses the same line as the marker's `value`, without the 80-character
cut. For speech, `note` is the transcript text in full. For a detection or
a face, `note` is omitted.

### EDL

Locator lines come after the last cut, so they never sit between a clip's
picture event and its audio event. Record-time order, then speech, face,
detection, then the label.

```text
* LOC: 00:00:02:00 cyan Speech: hello there
* LOC: 00:00:04:12 yellow Detection: Person
* LOC: 00:00:04:12 green Face: Ada
```

The color is a Resolve marker color: `cyan` for speech, `yellow` for a
detection, `green` for a face. Resolve shows the locator as a marker.
Premiere does not. Final Cut never reads the file. The events above the
locators are unchanged.

### FCPXML

The marker is a child of the `asset-clip` it falls in. `start` and
`duration` use the sequence frame duration, the same rationals as the clip.

```xml
<asset-clip ref="r2" offset="0s" name="Clip 1" start="10s" duration="10s" tcFormat="NDF">
  <marker start="2s" duration="4s" value="Speech: hello there" note="hello there"/>
  <marker start="4s" duration="1/30s" value="Detection: Person"/>
</asset-clip>
```

`value` and `note` are XML-escaped. Final Cut shows a marker that lasts
`duration` and opens `note` as its body. Resolve imports the marker as a
point at `start` and keeps `value`. Premiere's FCPXML import drops the
element. A `keyword` was not used: that files a range in Final Cut's
browser, and this marker is a mark on the timeline.

## Call sites

### Export button

`on_export_clicked` keeps its dialog and its save dialog.

`get_export_formats` returns three entries:

| Label | Extension |
|---|---|
| EDL (CMX 3600) | `.edl` |
| FCPXML | `.fcpxml` |
| CSV | `.csv` |

JSON comes off the list. It was never implemented; choosing it wrote a CSV.

The info line under the combo gains the frame rate (`29.97 fps, 30000/1001`)
and the marker count once the probe and the span build have run, which is
before the dialog opens. A failed probe is a message box and no dialog.

Under the format combo, a second combo chooses the sequence start:

| Label | Value |
|---|---|
| Start at 00:00:00:00 | `00:00:00:00` (default, preselected) |
| Start at 01:00:00:00 | `01:00:00:00` |

The second label's tooltip says it matches a new Resolve timeline. Choosing
CSV disables the combo and leaves it on the default, because a CSV has no
sequence clock. The value is passed through as `Sequence.record_start`.

Default save path stays next to the source: `{stem}_edit.edl` or
`{stem}_edit.fcpxml`. The extension on FCPXML changes from `.xml` to
`.fcpxml`, which is what Final Cut and Resolve both associate with this
format.

CSV moves into `TimelineExporter.to_csv` so the dialog and the writer agree.
Columns stay `Clip, Start (s), End (s), Duration (s)`, now in the quantised
times rather than the raw floats, plus a `Frame in` and `Frame out` so the
spreadsheet can be checked against the EDL. Seconds stay in the file because
that is the unit the rest of the app shows.

### Chat command

`[CMD:export format=edl]` and `format=xml` / `fcpxml` call the same methods.
`xml` remains accepted as an alias. The output path is
`{directory of video}/{stem}_edit.edl` or `.fcpxml`. The reply includes that
path. A probe failure or an empty timeline is the reply text, not a traceback
and not a file.

The chat command has no dialog, so the sequence start stays `00:00:00:00`
unless the command says otherwise: `start=01:00:00:00`. Any other value is
an error, not a silent default.

## Failures

| Situation | Result |
|---|---|
| Edit timeline empty | Button: the existing "add some clips" warning. Writer: raises, so the chat command cannot write an empty sequence either. |
| Probe fails | No file. The message is the probe error. |
| A clip quantises to zero frames | Skipped, counted, export continues. |
| Every clip quantises to zero | No file. Error names the frame rate. |
| `out` before `in` before quantising | That clip is skipped. The timeline refuses to create these; a hand-built list should not abort the rest. |
| Disk write fails | The exception reaches the existing error message box. No half-written file: write to `{path}.part` and replace. |

A missing source file is the probe's failure. Export does not copy media and
does not check that an NLE on another machine will see the same path.

## Tests

`tests/test_timeline_export.py`. No ffmpeg, no Qt. A `MediaSource` is built
in the test.

- At 30/1, a clip from 10.0 to 15.0 is source `00:00:10:00`–`00:00:15:00` and,
  at the default record start, record `00:00:00:00`–`00:00:05:00`. The second
  clip starts on the record side where the first ended. The FCPXML `offset`
  of that same clip is `0s`.
- The same clip with `record_start="01:00:00:00"` is record
  `01:00:00:00`–`01:00:05:00`, and the FCPXML sequence has `tcStart` of one
  hour with the first clip's `offset` equal to that `tcStart`.
- At 30000/1001, five real seconds is 150 frames of 1001/30000, and the CMX
  timecode counts those 150 frames at 30 nominal, so it reads five seconds.
  The test that would have caught the float path: `5 * 29.97` is not 150.
- At 100/1, half a second is source frame `50` (`00:00:00:50`), and the
  FCPXML `frameDuration` is `100/10000s`. The EDL counts at 100.
- At 120/1, five seconds is 600 source frames. The FCPXML `frameDuration` is
  `100/12000s` and the clip duration is those 600 frames. The EDL counts the
  same clip at 60, so its timecode reads five seconds, and a one-source-frame
  clip is omitted from the EDL because it is shorter than one 60 fps frame.
- At 240/1, the FCPXML `frameDuration` is `100/24000s`. The EDL counts at 60.
- Two clips at 30000/1001 produce FCPXML `asset-clip` elements whose `start`,
  `duration`, and `offset` parse back to those frame counts. `sequence`
  duration equals the sum. `asset` duration equals the file, not the used span.
- A path with a space is percent-encoded in `media-rep src`.
- A 90° rotation swaps width and height on the `format` element.
- No audio stream means no `A` event and `hasAudio="0"`.
- A zero-frame clip is absent from both outputs and reported as skipped.
- An empty clip list raises and leaves no file.
- `to_edl` without an output path writes `{stem}_edit.edl` beside the source
  (the chat-command case).
- A transcript segment from 12s to 16s overlapping a clip from 10s to 20s
  writes one FCPXML `marker` at `2s` with duration `4s` and `note` equal to
  the line, and one EDL `* LOC:` at the record time of that same start.
  Choosing `01:00:00:00` moves the locator by one hour and leaves the
  marker's `start` where it was.
- A segment that ends before the first clip writes nothing.
- Action hits of `Person` at 1.0s and 2.5s are one detection span. Hits at
  1.0s and 4.0s are two.
- An EDL speech locator longer than 80 characters is one line, ends with
  `...`, and contains no raw newline.
- A face with no identity name is labeled `Face`. Two unnamed tracks stay
  two spans.
- Reel names longer than 8 characters are truncated, and the full filename
  survives in `FROM CLIP NAME`.

The FCPXML assertions parse with `xml.etree`. They do not compare a whole
golden document, so a whitespace change is not a failure.

## Unchanged

- `modules/media/edl.py`, `tests/test_edl.py`, the sidecar `/edl` routes, and
  the auto pipeline.
- **Render Highlight Video** and the encoder combo beside it.
- The clip model in `EditTimelineScene`. Export reads `clips`; it does not
  store a second copy.
- Detection markers, transitions, music, and captions. They are not on this
  timeline.
