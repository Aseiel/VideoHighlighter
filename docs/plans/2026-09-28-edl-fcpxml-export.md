# EDL and FCPXML export

Implementation plan for the Timeline Viewer export. The behaviour is fixed in
[EDL-FCPXML-EXPORT.md](../EDL-FCPXML-EXPORT.md) and
[EDL-FCPXML-EXPORT-DESIGN.md](../EDL-FCPXML-EXPORT-DESIGN.md). This file is the
work, in the order to do it. Check a box when that item is done.

Do not change `modules/media/edl.py`, `tests/test_edl.py`, the sidecar `/edl`
routes, the auto pipeline, or **Render Highlight Video**. `film.edl.yaml`
stays this app's own cut list.

## Task 1 — Sequence model and timebase

Pure data, no file writing and no Qt. Lives in
`video_ai_editor/timeline_export.py`.

### Steps

- [x] Add `MediaSource`, `Span`, and `Sequence` as specified in the design.
- [x] Add `to_frames(seconds, num, den)` using `round(seconds * num / den)`.
- [x] Add the rate table: NTSC fractions count at their nominal rate, and the
      whole rates 24, 25, 30, 48, 50, 60, 100, 120, and 240 are recognised.
- [x] FCPXML `frameDuration` follows that table, including `100/12000s` at
      120 fps and `100/24000s` at 240 fps.
- [x] EDL counts at the real rate when the frame number fits in two digits
      (nominal fps ≤ 100). Above that, count at the largest whole divisor
      that is ≤ 60. For 120 and 240 that divisor is 60.
- [x] Any other fraction is reduced and counted at `round(num/den)`. It is
      "unrecognised" only when it is neither a table row nor a whole number
      of frames per second.
- [x] `record_start` accepts only `00:00:00:00` and `01:00:00:00`.

### Acceptance criteria

- [x] Five seconds at `30000/1001` is 150 frames, not `round(5 * 29.97)`.
- [x] Five seconds at `120/1` is 600 source frames. The EDL clock for that
      clip is 60 fps. Five seconds at `240/1` is on the same 60 fps clock.
- [x] Half a second at `100/1` is frame 50.
- [x] A rate such as `90/1` is named, not reported as unrecognised. A
      non-whole fraction outside the table is reported as unrecognised.
- [x] No test in this task reads a media file or imports Qt.

## Task 2 — CMX 3600 writer

`TimelineExporter.to_edl`. One source, hard cuts, every clip on the sequence.

### Steps

- [ ] Write UTF-8 with LF line endings. Title is the source stem.
- [ ] Header is `TITLE`, `FCM: NON-DROP FRAME`, and one comment that source
      times count from the first frame of the file, not from a camera clock.
- [ ] Each clip is a `V` event and, when `has_audio` is true, an `A` event
      with the same in and out. They share an event number. The next clip
      takes the next number. Past 999 the number wraps to 001.
- [ ] Event line spacing matches the design. Reel is the stem, uppercased,
      `A–Z` and `0–9` only, truncated to 8 characters, space-padded.
- [ ] `* FROM CLIP NAME:` is the filename. `* SOURCE FILE:` is the absolute
      path.
- [ ] Source columns count from the first frame of the file. Record columns
      start at `Sequence.record_start` and abut.
- [ ] Apply the Task 1 EDL clock, so 120 and 240 fps cuts are rounded to the
      60 fps grid. A clip that becomes zero frames on that grid is skipped
      and counted.
- [ ] Omit a final transition field. Every join is `C`.
- [ ] `output_path is None` writes `{stem}_edit.edl` beside the source.
- [ ] Write to `{path}.part` and replace the destination. A failed write
      leaves no half file.

### Acceptance criteria

- [ ] At 30 fps, a clip from 10.0s to 15.0s is source `00:00:10:00`–`00:00:15:00`
      and, at the default record start, record `00:00:00:00`–`00:00:05:00`.
      The next clip's record in is that record out.
- [ ] `record_start="01:00:00:00"` moves only the record columns, by one hour.
- [ ] A file with audio has an `A` line per clip. A file without audio has none.
- [ ] A reel name longer than 8 characters is truncated, and the full filename
      survives in `FROM CLIP NAME`.
- [ ] At 120 fps, a five-second clip reads as five seconds of 60 fps timecode,
      and a one-source-frame clip is absent and counted as skipped.
- [ ] An empty clip list raises and leaves no file.
- [ ] `modules/media/edl.py` is not imported.

## Task 3 — FCPXML 1.9 writer

`TimelineExporter.to_fcp_xml`. Same `Sequence` as the EDL.

### Steps

- [ ] Write the document in the design: doctype, `fcpxml version="1.9"`, one
      `format`, one `asset`, one `library` / `event` / `project` / `sequence`
      / `spine`, one `asset-clip` per kept clip.
- [ ] Times are reduced rational seconds in the real frame duration. The
      asset duration is the whole file. The sequence duration is the sum of
      the clips.
- [ ] `asset-clip` `start` is the source in-point. `offset` is the record
      start plus the durations before it. At `01:00:00:00` the first offset
      equals `tcStart`.
- [ ] `frameDuration`, width, and height come from `MediaSource`. Width and
      height are the display size. The format name uses the design's rate
      token, including `120` and `240` for those real rates.
- [ ] `media-rep src` is a file URL from `urllib.request.pathname2url`.
- [ ] Audio attributes follow the probe: absent when there is no audio
      stream, otherwise the probed channel count and sample rate.
      `audioLayout="stereo"` only for two channels.
- [ ] No `colorSpace`. Clip names are `Clip 1`, `Clip 2`, in edit order.
- [ ] Same skip rule, same `.part` replace, and `output_path is None` writes
      `{stem}_edit.fcpxml` beside the source.

### Acceptance criteria

- [ ] A two-clip sequence at `30000/1001` parses with `xml.etree`. Clip
      `start`, `duration`, and `offset` convert back to the frame counts.
      Sequence duration is the sum. Asset duration is the file length.
- [ ] At 120 fps, `frameDuration` is `100/12000s` and a five-second clip is
      600 frames. At 240 fps, `frameDuration` is `100/24000s`.
- [ ] A path containing a space is percent-encoded.
- [ ] A 90° rotation swaps width and height on `format`.
- [ ] No audio stream produces `hasAudio="0"` and no audio attributes.
- [ ] `record_start="01:00:00:00"` sets `tcStart` to one hour and the first
      clip's `offset` equal to that `tcStart`.

## Task 4 — CSV, and dropping JSON

### Steps

- [ ] Move the spreadsheet into `TimelineExporter.to_csv`.
- [ ] Columns are clip number, quantised start, end, and duration in seconds,
      plus frame in and frame out on the real frame rate (not the coarsened
      EDL clock).
- [ ] `get_export_formats` returns EDL (`.edl`), FCPXML (`.fcpxml`), and CSV
      (`.csv`). Remove JSON.

### Acceptance criteria

- [ ] Choosing a format cannot fall through into another format's writer.
- [ ] CSV has no marker rows and no sequence-start column.
- [ ] Frame in and frame out at 120 fps use 120, not 60.

## Task 5 — Spans and markers

`spans_from_analysis(cache)` in `timeline_export.py`. The constant for the
gap is 2.0 seconds, the same value as `SignalTimelineScene.EVENT_RUN_GAP`,
copied here so this module does not import the timeline scene.

### Steps

- [ ] Speech spans are `transcript.segments` entries that have text. Start,
      end, and label come from the segment. Do not re-merge them.
- [ ] Detection spans come from `actions`, then `objects`, grouped by class
      name. Hits more than 2.0 seconds apart start a new span. The span runs
      from the first hit to the last. A single hit is one frame long.
- [ ] Face spans come from `object_bboxes` entries that carry `identity_names`
      and `track_ids`. Group by the name when there is one, otherwise by
      track id. The same 2.0 second gap. The label is the name, or `Face`.
- [ ] Ignore the viewer's hidden rows, confidence sliders, and merge slider.
- [ ] Clip each span to every exported clip it overlaps. Drop an overlap that
      quantises to zero frames. A span that misses every clip is omitted.
- [ ] Display line is `Speech: …`, `Detection: …`, or `Face: …` / `Face`.
      Collapse whitespace. Strip a leading `*`.
- [ ] EDL: after the last cut, one `* LOC:` per overlap, at the record time
      of the overlap start. Color is `cyan`, `yellow`, or `green`. Text is
      one line, at most 80 characters, ending in `...` when cut. Order is
      record time, then speech, face, detection, then the label.
- [ ] FCPXML: one `marker` child of the `asset-clip`, `start` measured from
      the first frame of that clip, `duration` the overlap. `value` is the
      display line without the 80-character cut. Speech also sets `note` to
      the full line. Escape both for XML.
- [ ] The record-start choice moves the EDL locator and does not move the
      marker's clip-relative `start`.

### Acceptance criteria

- [ ] A transcript segment 12s–16s overlapping a clip 10s–20s writes one
      marker at 2s for 4s with `note` equal to the line, and one locator at
      the matching record time.
- [ ] `01:00:00:00` adds one hour to that locator and leaves the marker
      `start` at 2s.
- [ ] A segment that ends before the first clip writes nothing in either file.
- [ ] `Person` hits at 1.0s and 2.5s are one detection span. Hits at 1.0s and
      4.0s are two.
- [ ] Two unnamed face tracks stay two spans, each labeled `Face`.
- [ ] A speech locator longer than 80 characters is one line, ends with
      `...`, and contains no newline.
- [ ] An empty analysis cache writes the cut and no markers.

## Task 6 — Probe once, then export

### Steps

- [ ] When `source` is omitted, probe `video_path` once through
      `modules.media.ffmpeg_tools.probe`.
- [ ] Read `r_frame_rate` as `num/den`, falling back to `avg_frame_rate` only
      when `r_frame_rate` is `0/0`. Read container duration, stored width and
      height, and the first audio stream's presence, sample rate, and channel
      count.
- [ ] Apply rotation with `video_probe._rotation_from_stream` on that same
      video stream. Swap width and height when the result is 90 or 270.
- [ ] A failed probe raises. Do not assume 30 fps.

### Acceptance criteria

- [ ] Tests that pass a `MediaSource` do not call ffprobe.
- [ ] A probe failure produces no file.
- [ ] `probe_video`'s public return value is unchanged. Callers outside
      export still receive a float fps.

## Task 7 — Export dialog

`SignalTimelineWindow.on_export_clicked` in `signal_timeline_viewer.py`.

### Steps

- [ ] Probe and build spans before the dialog. On probe failure, show the
      error and do not open the dialog.
- [ ] Keep the format combo, now the three formats from Task 4. Default save
      names stay `{stem}_edit` with `.edl`, `.fcpxml`, or `.csv`.
- [ ] Add a sequence-start combo: `00:00:00:00` (preselected) and
      `01:00:00:00`. The second entry's tooltip says it matches a new Resolve
      timeline. Disable the combo while CSV is selected, and leave it on the
      default.
- [ ] The info line shows clip count, duration, the frame rate as a fraction,
      and the marker count. When the rate is over 100 fps, the line says the
      FCPXML keeps every frame and the EDL is counted at the coarsened rate.
- [ ] Pass `record_start` and the spans into the writer. An empty edit
      timeline still shows the existing "add some clips" warning and writes
      nothing.
- [ ] Show the skipped-clip count in the success message when it is not zero.

### Acceptance criteria

- [ ] Export still writes every clip on the edit timeline. The yellow
      highlight is not consulted.
- [ ] CSV cannot be given a sequence start other than the default.
- [ ] **Render Highlight Video** and the encoder combo are unchanged.

## Task 8 — Chat command

`llm/llm_timeline_bridge.py`, `[CMD:export]`.

### Steps

- [ ] Call the same `TimelineExporter` methods. `xml` remains an alias for
      FCPXML.
- [ ] Write beside the source: `{stem}_edit.edl` or `{stem}_edit.fcpxml`.
      The reply includes that path.
- [ ] Default `record_start` is `00:00:00:00`. `start=01:00:00:00` selects
      the other. Any other `start` value is an error reply and writes nothing.
- [ ] Include spans from the viewer's analysis cache. An empty timeline or a
      probe failure is a reply, not a traceback and not a file.
- [ ] Update the command help string so it mentions `fcpxml` and `start`.

### Acceptance criteria

- [ ] `format=edl` with no path argument no longer opens `None`.
- [ ] `start=01:00:00:00` matches a dialog export with that start. Omitting
      `start` matches the dialog default.
- [ ] `start=00:00:00:00` is accepted. `start=10:00:00:00` is rejected.

## Task 9 — Tests

`tests/test_timeline_export.py`. No ffmpeg, no Qt, except the probe-failure
case which can pass a fake probe. Everything else is given a `MediaSource`.

### Steps

- [ ] Cover the acceptance criteria of Tasks 1 through 5 with the cases named
      in the design's Tests section.
- [ ] Run `pytest tests/test_timeline_export.py tests/test_edl.py` and keep
      both green.

### Acceptance criteria

- [ ] `tests/test_edl.py` passes without modification.
- [ ] The new tests fail if 120 fps FCPXML is written as 30 fps, if a 120 fps
      EDL frame field contains a number above 99, or if a locator is inserted
      between a clip's `V` and `A` lines.

## Task 10 — Import check on a real editor

Automated tests cannot open an NLE. After Task 9 is green, import one
exported edit into each editor available on the machine and record the
result here. Skip a row only by checking "not installed".

Use one file with two clips, a known frame rate, one speech span inside a
clip, and the default record start. Repeat the EDL once with
`01:00:00:00` if Resolve is installed.

### Steps

- [ ] Export a `.edl` and a `.fcpxml` from the Timeline Viewer.
- [ ] Import the `.edl` into DaVinci Resolve.
- [ ] Import the `.fcpxml` into DaVinci Resolve.
- [ ] Import the `.edl` into Premiere.
- [ ] Import the `.fcpxml` into Final Cut Pro.

### Acceptance criteria

- [ ] Resolve, `.edl`: both clips, in order, relinked to the source, audio
      present when the file has audio. Locator visible. Record start matches
      the dialog.
- [ ] Resolve, `.fcpxml`: same clips and relink. Marker name visible.
- [ ] Premiere, `.edl`: both clips, in order, relinked. Markers are not
      required.
- [ ] Final Cut Pro, `.fcpxml`: both clips, in order, relinked. Marker shows
      the span.
- [ ] Not installed, checked only for an editor this machine does not have:
      - [ ] Resolve
      - [ ] Premiere
      - [ ] Final Cut Pro
