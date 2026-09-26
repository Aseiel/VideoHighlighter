# Composition rules on outlines instead of boxes

**Question (Szymek):** can the composition engine use object contours instead of
bounding boxes, and how hard is it?

**Answer:** yes, and most of it is now built. The engine change was small and
is done. Producing the outlines was the real question. There is a
working no-model path (GrabCut) and a better model path (SAM) wired behind the
same interface. The model path still needs one session on real footage with the
weights downloaded, which this environment could not do.

## Why boxes are wrong for composition

The engine had one spatial test: *is the centre of the source box inside the
region box?* That fails in exactly the cases composition cares about:

| situation | box says | truth |
|---|---|---|
| person reaching sideways; object in the empty part of their box | inside | not inside |
| two diagonal things (a bat, a board) whose boxes overlap | touching | apart |
| two things side by side, boxes sharing an edge | nothing to say | may or may not touch |

Measured on synthetic shapes (below), a bounding box covers the real shape
with IoU 0.20 for a diagonal object and 0.47 for an L-shaped one. More than
half of what the engine was reasoning about was empty space.

## What was built

### 1. Relations on shapes (done): `modules/rules/shapes.py`

A rule can now say how the two things relate:

```yaml
outliner: grabcut            # or sam
events:
  - name: held
    rules:
      - {source: <thing>, region: <holder>, relation: overlaps, min_overlap: 0.5, outline: true}
      - {source: <a>, region: <b>, relation: touches, max_gap: 0.01, outline: true}
      - {source: <c>, region: <d>}          # unchanged: centre inside, on boxes
```

- `inside` (default): the source's centre lies in the region. On two boxes this
  is the original test, bit for bit. Every existing rule gives the same answer,
  and the 136 existing composition tests pass unchanged.
- `overlaps`: at least `min_overlap` of the source's area is inside the region.
- `touches`: the shapes meet or come within `max_gap` (fraction of frame width).

A detection without an outline is treated as its box, so every relation
works on boxes too. Outlines only make the answer more exact. Areas are
measured by rasterising both shapes on a 128-cell grid fitted to them (no
geometry dependency, accurate to about 1% of the smaller shape at any scale).

### 2. Outlines on demand (done): `modules/vision/outlines.py`

A rule opts in with `outline: true`. Then:

- **Only where it can matter.** An outline lies inside its box, so shapes
  can only meet where their boxes meet. Only detections whose box meets a box
  of their rule's partner class are traced; everywhere else the boxes already
  say "no". On typical footage this skips most frames.
- **Once.** Outlines are stored in the analysis cache as `contours`, aligned
  with `bboxes`. A detection that was tried and gave nothing is stored as
  `[]`, so it isn't retried. Re-running with a new threshold costs nothing.
- **At the detector's rate.** Detection keeps one frame per second, so an hour
  of video is at most 3,600 frames, before the filter.
- Both places rules run use it: *Re-apply to cache* (`run_composition`) and
  the full pipeline (`compose_events.apply_rules`).
- Outlines are simplified to a few dozen points (Douglas-Peucker, ~2 px at
  1080p) before storage.

**GrabCut** (default; OpenCV only, no model, nothing to download), measured here:

| shape | box IoU with truth | GrabCut IoU | time per object |
|---|---|---|---|
| L-shape, clean background | 0.47 | 0.97-0.99 | 66-144 ms |
| diagonal bar, clean | 0.20 | 0.95-0.97 | 61-85 ms |
| disc, clean | 0.79 | 0.99 | 53-71 ms |
| L-shape, 60 same-colour distractors | 0.47 | 0.88 | ~100 ms |
| diagonal, low contrast + distractors | 0.20 | 0.93-0.95 | ~80 ms |

These are synthetic scenes (a textured background with a distinctly
coloured object), so they are an upper bound. GrabCut separates by colour
statistics. It will do worse on real footage where the object shares colours
with its surroundings, and it knows nothing about what an object is.

**SAM** (`outliner: sam`): Segment Anything prompted with the detector's boxes,
through the `transformers` the app already bundles. Default checkpoint
`Zigeng/SlimSAM-uniform-77` (a distilled SAM, tens of MB). One image encoding
per frame serves every box in it. The glue is tested against a stand-in that
follows the `transformers` SAM API. **Not yet measured:** Hugging Face is
blocked in the environment this was built in, so the weights were never
loaded. Before shipping, one session is needed to:

1. confirm the checkpoint's licence (the SlimSAM repo states Apache-2.0; SAM
   itself is Apache-2.0) and that `SamModel.from_pretrained` loads it;
2. measure quality against GrabCut on a few real videos (the IoU script used
   for the table above is easy to point at hand-traced masks);
3. measure speed: expected well under a second per frame on a CPU and tens of
   ms on a GPU. If CPU is too slow, export it through OpenVINO, as CLIP is.
4. decide packaging: download on first use into the models folder, like the
   CLIP pack (`modules/packs`), rather than growing the installer.

### 3. Saving from the UI no longer drops fields (done)

Both rule editors rebuilt the file from their table rows, so anything
without a column was deleted on Save.
- **Advanced tab (Qt):** this would have dropped `relation`, `outline` and the
  top-level `outliner`. It now carries every field it has no column for from
  the rule's original (`modules/rules/rules_file.py`).
- **Web UI (sidecar):** this was worse and predates this change. Every save
  deleted all signal conditions, signal-only events and event fields such as
  `min_duration_secs`. It now merges onto the file the same way.

## What is left, and how hard

| piece | effort | notes |
|---|---|---|
| SAM measured on real footage, licence confirmed, packaged | 1 day | steps above |
| Draw outlines in the timeline overlay | 0.5-1 day | composed entries already carry `event_contours`; the overlay draws `bboxes` with `drawRect`, and needs a `drawPolygon` branch |
| `Relation` and `Outline` columns in the Advanced tab | 0.5 day | YAML works today; the table now preserves the fields, it just doesn't show them |
| Body-part sources from pose (`person.left_wrist inside cup`) | 1-2 days | RTMPose (Apache-2.0) already runs in the cropper; a keypoint is a precise point source for `inside` and `touches`. Probably the biggest precision gain for rules about people, at little cost |
| Outlines in the *live* overlay (every frame, while playing) | hard; not recommended | needs real-time segmentation. Composition is an offline reading of cached detections, and that's where outlines pay |

Things that are *not* needed:

- **A segmentation model per class.** SAM and GrabCut are class-agnostic: they
  outline whatever box they're given. So outlines work for every stock COCO
  class *and* for every class taught with `modules/teach` without retraining.
- **New dependencies.** GrabCut is OpenCV, and SAM goes through `transformers`.
  Both are permissive and both are already in the build.

## Files

- `modules/rules/shapes.py`: polygons, relations, simplification
- `modules/vision/outlines.py`: GrabCut / SAM outliners, the on-demand pass
- `video_ai_editor/composition_engine.py`: `relation`, `min_overlap`,
  `max_gap`, `outline` on rules; `outliner` at the top; `contours` read from the cache
- `modules/rules/compose_events.py`, `modules/report/analysis_ondemand.py`:
  the pass wired into both places rules run
- `modules/rules/rules_file.py`, `main.py`, `sidecar/server.py`: saving keeps fields
- Tests: `tests/test_composition_shapes.py`, `tests/test_outlines.py`,
  `tests/test_sidecar_rules_save.py`
