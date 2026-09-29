# EDL and FCPXML export

Export the edit timeline as a cut list another editor can open. The user keeps
working in DaVinci Resolve or Final Cut Pro instead of being handed only a
flattened mp4.

This document says what the feature is, what it is not, and where it sits in
the app. The format and module design is in
[EDL-FCPXML-EXPORT-DESIGN.md](EDL-FCPXML-EXPORT-DESIGN.md). The implementation
plan is [plans/2026-09-28-edl-fcpxml-export.md](plans/2026-09-28-edl-fcpxml-export.md).

## What the user gets

From the Timeline Viewer, with clips already on the edit timeline:

1. **Export** opens a format choice and a save dialog, as it does today.
2. **EDL** writes a CMX 3600 cut list (`.edl`). Resolve imports this. Premiere
   does too. Final Cut does not.
3. **FCPXML** writes Final Cut Pro XML 1.9 (`.fcpxml`). Final Cut Pro and
   Resolve both import this.

The imported sequence is the edit timeline in order: each clip is a range of
the source file, laid back to back, with hard cuts. Picture and sound come
from that same file. Nothing is re-encoded. The source file stays where it
is; the NLE relinks to it.

Each transcript line, detection, and face that falls inside an exported clip
is also written as a marker. A detection or a face that stays on screen is
one marker for that stretch, not one per frame. What each app does with that
marker is below.

**Render Highlight Video** stays the way to bake an mp4 inside this app. Export
is the way to leave.

## What "the selected footage" is

The edit timeline in the Timeline Viewer: the ordered list of `(start, end)`
ranges, in seconds, into the one video that viewer has open
(`EditTimelineScene.clips` in `video_ai_editor/edit_timeline.py`).

That list is the user's cut. It arrives from one of three places:

- this run's `final_segments` (what the pipeline actually kept),
- the newest highlight version in the cache, when the viewer is opened on its own,
- or the user, by dragging a region down, **Add Clip**, a waveform click, or a
  bar's "add to edit" action.

Order is edit order. The user can reorder, trim, split, and delete. Export
writes that order, not source order, and not "highest score first".

Two different "selections" sit on that timeline:

| | Every clip | The yellow highlight |
|---|---|---|
| What it is | The cut. What **Play Edit** plays, in order. | A temporary UI selection. Click, Shift-click, or Ctrl+A. |
| What it is for | The sequence handed to Resolve or Final Cut. | **Delete** (and the context menu on those clips). |
| If you highlighted one clip to look at it | The whole reel is still exported. | Only that one clip would be exported. |
| If nothing is highlighted | Export still writes the reel. | There is nothing to export. |

Export writes every clip. The yellow set stays what Delete uses. Someone who
clicks a clip to inspect it should not lose the rest of the reel on the way
out.

Every range belongs to that one open file. The viewer has a single
`video_path`. A multi-file reel is a different document (below) and is not
this feature.

## Two things already called "EDL"

They must not be merged.

| | Internal cut list | This feature |
|---|---|---|
| File | `film.edl.yaml` | `name_edit.edl`, `name_edit.fcpxml` |
| Module | `modules/media/edl.py` | `video_ai_editor/timeline_export.py` |
| Read by | this app (`load_edl`, `render_edl`, the sidecar `/edl` routes, the auto pipeline) | Resolve, Final Cut, Premiere |
| Sources | many files, plus transitions, music, captions | one file, hard cuts |
| Purpose | edit the numbers and render again inside VideoHighlighter | hand the cut to another program |

`modules/media/edl.py` is a YAML document this app invented so a reel can be
revised without re-rolling the pipeline. An NLE will not open it. This
feature does not change that format, its tests, or the auto pipeline.

## What is already wired, and why it does not work

`TimelineExporter` in `video_ai_editor/timeline_export.py` is called from two
places:

- the **Export** button on the edit toolbar (`SignalTimelineWindow.on_export_clicked` in `signal_timeline_viewer.py`),
- the timeline chat command `[CMD:export format=edl]` (or `xml`) in `llm/llm_timeline_bridge.py`.

The button's dialog offers four formats. Two of them do not do what the label
says.

**CMX EDL** is the right shape of file and the wrong times. The event line,
the 8-character reel, and `* FROM CLIP NAME:` are what Resolve looks for. The
frame rate is hardcoded to 30, so a 23.976, 25, 29.97, or 60 fps source gets
timecodes that do not land on its frames. There is a video event and no audio
event. The chat command calls `to_edl` without a path, and the writer then
opens `None`.

**FCPXML is not FCPXML.** The writer emits elements the format does not have
(`clip`, `video`, `offset`, `asset-ref`), a hardcoded 1080p29.97 format, and
times computed as `seconds * fps * 100` with the letter `s` stuck on the end.
FCPXML time is a rational (`1001/30000s`). The asset's duration is the span
from the first clip's start to the last clip's end, which is neither the
file length nor the sequence length. The path is not a file URL. Final Cut
and Resolve reject this, or open a sequence that does not match the timeline.

**JSON** is in the format list and has no writer. Choosing it falls through
to the CSV branch. **CSV** is written inline in the button handler, not in
`TimelineExporter`, as a spreadsheet of start, end, and duration. It is useful
and it is not an NLE format.

There are no tests for `timeline_export.py`.

`probe_video` (`modules/media/video_probe.py`) returns duration, stored
width and height, a float fps, and rotation. It does not return the frame-rate
fraction (`30000/1001`), whether an audio stream exists, or a timecode. Float
fps is the wrong input for a timecode: `29.97 * seconds` drifts.

## Where a correct export hooks in

The clip list and the button stay. The writers are replaced. Callers pass the
source path and the clip list they already have; the writer probes the file
once and quantises.

```
edit timeline clips  ──►  Export button  ──►  TimelineExporter  ──►  .edl / .fcpxml
        │                                              ▲
        └──►  [CMD:export]  ──────────────────────────┘
```

Same writer for both callers, so the chat command cannot drift from the
button. The button keeps its dialog. The dialog tells the user the frame rate
it is about to write, because that is the fact the current exporter gets
wrong in silence.

**Render Highlight Video** (`on_render_highlight_clicked`) is untouched. It
reads the same clip list and ffmpeg-concats an mp4. Export does not go through
ffmpeg.

The chat command grows a real output path: next to the source,
`{stem}_edit.edl` or `{stem}_edit.fcpxml`, which is where the button already
suggests saving. Today that command cannot succeed for EDL, because no path
is passed.

An Insta360 X6 clip at 24, 25, 30, 48, 50, 60, or 100 fps is frame-accurate
in both files. At 120 and 240 fps the FCPXML stays frame-accurate. A CMX
timecode has only two digits for the frame, so it cannot count a 120 fps or
240 fps second, and the EDL for those two rates is numbered at 60 fps. A cut
can move by one source frame at 120 fps, or two at 240 fps. The dialog says
so. The details are in the design doc.

Nothing in `main.py`, the auto pipeline, or the web UI calls
`TimelineExporter`. The web app renders through the sidecar and the YAML cut
list. Putting an NLE download on the web app would be a separate feature; the
writer should stay free of Qt so that can call it later without dragging the
viewer along.

## Markers

A span is one stretch of one thing: a transcript line, one detection class
held across neighbouring samples, or one face held across neighbouring
samples. Samples of the same thing more than two seconds apart are two spans.
The two seconds are the gap the timeline already uses to treat a run of
sampled events as one bar.

Only the part of a span that lies inside an exported clip is written. A span
in a part of the file the edit does not use is left out. A span that crosses
two clips is marked once in each, for the piece inside that clip.

The two files cannot store that the same way.

| | EDL | FCPXML |
|---|---|---|
| What is written | A `* LOC:` comment at the start of the piece | A `marker` with a start and a duration |
| Who shows it as a marker | Resolve | Final Cut Pro shows the range. Resolve shows a point at the start and keeps the name. |
| Who does not | Premiere keeps the line as a comment, if it keeps it at all. Final Cut does not open the EDL. | Premiere's FCPXML import drops markers. |
| The words | One line, cut at 80 characters. A transcript line loses its tail. | The marker name is the same short line. The note holds the full transcript line. |
| The length | Not stored. The time is only the start. | Stored on the marker, clipped to the clip. |

The EDL line does not change the cut. An editor that ignores `* LOC:` still
imports the same clips.

## What this feature does not do

- It does not export `film.edl.yaml`, and it does not import one.
- It does not export several source files, transitions, a music bed, burnt-in
  captions, or the graphics overlays. The edit timeline has none of those.
- It does not read camera timecode. Ranges are seconds from the start of the
  file, which is what the timeline stores. A camera whose timecode starts at
  the time of day will not match those numbers inside the NLE until a later
  feature reads the timecode track. v1 says so in a comment at the top of the
  EDL.
- It does not write drop-frame timecode. Phone and action-camera files are
  non-drop. Drop-frame is a numbering scheme, not a frame rate, and getting
  it wrong moves every cut.
- It does not replace **Render Highlight Video**.

## Decisions

1. **Every clip on the edit timeline.** The yellow highlight is only the
   Delete selection. See the table above.
2. **Resolve opens both files. Final Cut Pro opens the FCPXML.** Premiere
   imports a CMX EDL on its own and is not a target for the FCPXML. A
   Premiere-native XML (`xmeml`) is out of scope.
3. **The assembled timeline's start is a choice in the export dialog.**
   Default `00:00:00:00`. The other choice is `01:00:00:00`, the videotape
   origin Resolve's new timelines still use. The same choice is written into
   the EDL record columns and the FCPXML `tcStart`. Source times are not
   shifted. CSV has no sequence clock, so the control is disabled for CSV.
4. **One audio event per clip** in the EDL, with the same in and out as the
   picture, when the file has an audio stream. No audio stream means no
   audio event.
5. **CSV stays in the dialog. JSON comes off it.** JSON was listed and then
   written as CSV. CSV remains a spreadsheet of the same clips, not an NLE
   format. Markers are not rows in the CSV.
6. **One marker per span, in both interchange files.** FCPXML stores the
   length. The EDL stores the start, as a Resolve locator comment. Premiere
   does not gain markers from either file.
