# Card to film

Six pieces that turn a camera card into a finished video, and one runner that
drives them in order. Each is usable on its own; the Auto and Timeline tabs are
the assembled version.

| Piece | Module | What it owns |
|---|---|---|
| Ingest | `modules/gopro_ingest.py` | Finding a card, copying it off safely |
| Script | `modules/script_plan.py` | What the film should contain |
| Music | `modules/music_analysis.py` | Where the beats are |
| Cut list | `modules/edl.py` | Which piece of which file, when |
| Transitions | `modules/transitions.py` | How one clip becomes the next |
| Runner | `modules/auto_pipeline.py` | Running the stages, and resuming them |

---

## Ingest

A card is identified by its layout — `DCIM/1xxGOPRO/` must exist — not by drive
letter or volume label. A reused card can still be labelled from whatever it
held last, and drive letters move between sessions; the folder structure is the
only thing that reliably means "camera card". `MISC/version.txt` is JSON the
camera writes (model, firmware, serial); when present it names the destination
folder and is recorded as provenance.

### The filename trap

GoPro names a file `GXccnnnn.MP4`: codec letters, a two-digit **chapter**, then
a four-digit **file number**. The chapter comes first, so sorting by name
interleaves separate recordings:

```
GH010527  GH010528  GH020527  GH020528     <- sorted by name
GH010527  GH020527  GH010528  GH020528     <- actual recording order
```

Ordering therefore keys on `(file_number, chapter)`. Files sharing a file
number are chapters of one continuous recording — the camera splits on a size
limit — and are grouped into a single take.

### Copy safety

Copies land in a `.part` file and are renamed into place only once the byte
count matches, so an interrupted run cannot leave a short file that looks
finished. A file already at the destination with the right size is skipped,
which makes re-running cheap and idempotent. `verify="hash"` adds a BLAKE2b
comparison; the default size check is what catches the realistic failure, a
truncated transfer.

Nothing is ever deleted from the card.

```python
from modules.gopro_ingest import find_gopro_cards, ingest, write_manifest

card = find_gopro_cards()[0]
result = ingest(card, r"D:\movies\GoPro")
write_manifest(result)          # ingest.json, next to the footage
result.paths                    # clips in recording order
```

---

## Script

A script says what the film should contain, beat by beat. It exists because
scoring alone cannot express intent: the selector will happily return the six
highest-scoring moments in the video, all from the same thirty seconds. A
script is also a file you can edit and re-run, rather than a slider you nudge.

```yaml
title: Morning ride
music: D:\music\track.mp3
total_duration: 120

beats:
  - name: Establishing
    duration: 12
    order: chronological
    match:
      objects: [boat, water]      # your terms; nothing is built in

  - name: Action
    repeat: 3                     # three clips from this beat
    duration: [6, 10]             # a range, not a fixed length

  - name: Calm
    duration: [8, 14]
```

Every match term is supplied by you. The module ships no category lists,
presets, or vocabulary of its own — see the content-neutrality rule in
`CLAUDE.md`.

Unknown keys are refused rather than ignored, with the line number and a
suggestion:

```
ScriptError: unknown beat key 'durations' — did you mean 'duration'? (line 3)
```

That is deliberate. A silently dropped key is the failure mode that costs an
hour: the run completes, the output ignores half of what you wrote, and nothing
says why.

```python
from modules.script_plan import load_script, compile_directives

script = load_script("script.yaml")
script.clip_count            # 5  (Action counts three times)
script.target_duration       # 120
compile_directives(script)   # repeats flattened, in script order
```

### What reaches the engine today

`auto_pipeline.apply_script_to_config` folds a script into the engine config:
the union of every beat's match terms becomes the interest lists, and the
script's total becomes the duration budget.

Per-beat ordering and per-beat duration do **not** reach the engine yet. The
current selector ranks the whole video globally and has no notion of "this clip
belongs to beat 3", so translating those would produce a config that looks like
it honours the script and does not. `compile_directives()` carries them for a
caller that can use them, and the selector is where that work belongs.

---

## Music

Beats, downbeats, tempo and energy sections, so cuts can land on the music
rather than near it.

The core runs on numpy and ffmpeg alone — no new dependency. `librosa` is used
only when it is already installed and only as a refinement; the numpy path is
the one that must work. (`madmom`, the usual recommendation, does not import on
Python 3.10+: it still reads `MutableSequence` from `collections`.)

How it works: decode to mono 22.05 kHz → STFT → spectral flux onset envelope →
autocorrelation for tempo, weighted by a log-normal prior centred on 120 BPM
(this is what prevents the classic half/double-tempo error) → Ellis
dynamic-programming beat tracking → downbeat phase chosen by summed onset
strength.

Measured against synthetic click tracks: within 0.11 BPM at 100, 128 and
140 BPM, with beats landing a mean 14 ms from truth — under one 23 ms STFT hop.

Degenerate input (silence, a single sample, a file shorter than one FFT frame)
returns `bpm=0.0` and no beats rather than raising, and snapping against an
empty grid is the identity. A music file that cannot be analysed must not cost
you the film.

```python
from modules.music_analysis import analyze_music, snap_segments

a = analyze_music("track.mp3")
a.bpm, len(a.beats), len(a.downbeats)
snap_segments([(1.0, 3.0), (4.0, 6.0)], a, min_duration=0.5)
```

---

## The cut list

`film.edl.yaml`, written next to the film. Explicit sources and explicit
timestamps — what the automatic pass decided, as a document you can edit:

```yaml
cuts:
  - source: GX013762_highlight.mp4
    in: 0:00.0
    out: 0:03.6
    transition: crossfade
    transition_duration: 0.6
  - source: GX013763_highlight.mp4
    in: 0:00.0
    out: 0:07.3
```

The note you have after watching a first pass is never "score action higher",
it is "that clip should start two seconds later, and lose the one after it".
A script cannot express that and a cut list can. Edit it, render again, and
nothing else moves — detection is not re-run.

Timestamps are written the way people write them (`8`, `0:08`, `1:23.5`,
`1:02:03.5`). Unknown keys are refused with the line and a suggestion.

## Transitions

`cut`, `crossfade`, `dissolve`, `dip_to_black`, `dip_to_white`, four `wipe_*`,
four `slide_*`, `smooth_left`/`smooth_right`, `circle_open`/`circle_close`,
`radial`.

A transition needs both clips on screen at once, which means decoding both and
re-encoding — it cannot be the stream copy `combine_videos` uses. So it is a
separate path, and an all-cuts reel still takes the fast one.

Runs of hard cuts are joined *before* the blend, outside any filtergraph, so
`xfade` only ever sees whole runs. Two things forced that, both of which cost a
render to find:

- The concat **filter** cannot feed `xfade`. ffmpeg reports "Could not open
  encoder before EOF" and writes nothing, so a graph that expressed cuts inline
  worked only while no reel happened to begin with one.
- Every `xfade` input must share a timebase. A run that was copied through keeps
  a different one from a run that was re-encoded, and the filter refuses:
  *"First input link main timebase (1/15360) do not match ... (1/90000)"*. Every
  input is now pinned to `1/90000` before it reaches a filter that cares.

Two more consequences worth knowing:

- **The reel gets shorter.** Every transition overlaps two clips, so a 90
  second target with fourteen 0.6 s crossfades delivers about 82. `Edl.duration`
  reports the real number rather than the sum of the cuts.
- **A transition cannot outrun its clips.** ffmpeg fails rather than clamping,
  and it fails *after* the normalise pass has already spent minutes, so every
  duration is clamped to a third of the shorter neighbour up front.

## Cutting to the music

`quantise_to_music` rounds every clip to a whole number of bars. It works on
*durations*, not positions, and that is the whole trick: a reel plays back to
back, so the time a cut happens is the sum of every clip before it. Snapping
each clip's start to a beat would snap nothing. Make every clip a whole number
of bars and the arithmetic does the rest — the first cut lands on a downbeat
and so does every one after it.

When the nearest whole number of bars would run past the end of the source, the
*next lower* one is used rather than the leftover footage. A 6.03 s clip against
a 3.64 s bar becomes one bar, not 6.03 s. Clamping instead is what the first
real render did, and it put every following cut off the grid.

## The runner

Every stage records what it produced, and a re-run skips whatever is still on
disk. That is the whole point: the full sequence over a 4 GB card is tens of
minutes, and the expensive middle is exactly what gets interrupted. Resume is
the default, not a mode.

```
ingest -> music -> highlight -> combine -> music_mix
```

State lives in `job.json` beside the output and is written after *every*
transition — the failure being protected against is the one where the process
never reaches its cleanup.

A stage recorded as done whose output has since been deleted is **not**
satisfied, and is redone. Marching past it would fail later with a confusing
missing-file error from a stage that is not the problem.

The music stages are optional in the strict sense: if analysis or the final mux
fails, the run is marked not-ok and the errors are recorded, but the silent reel
is still returned as the output. The expensive work is detection; losing it to
an audio filter would be absurd.

```python
from modules.auto_pipeline import run_auto_pipeline

result = run_auto_pipeline(
    dest_root=r"D:\movies\GoPro",
    card=card,                       # or source_paths=[...]
    script_path="script.yaml",       # optional
    music_path="track.mp3",          # optional
    progress_fn=lambda f, d: ...,    # 0..1
    stage_fn=lambda name, status, detail: ...,
)
result.output, result.ok, result.state.errors
```

`highlight_runner` overrides the engine call, which is how the tests exercise
the orchestration without torch.

---

## From the UI

The **Auto** tab. Cards are detected on open (a lone card is selected for you),
the script has a template and a Check button that parses without running
anything, music shows its tempo and a beat strip, and the pipeline is drawn as
stages rather than one bar — a percentage says nothing useful about a run that
spends forty minutes inside a single stage.

Scoring settings come from the other tabs, so what you already configured is
what the automatic run uses.

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/gopro/cards` | Mounted cards and what is on them |
| GET | `/script/example` | Starter template |
| POST | `/script/validate` | Parse without running |
| GET | `/music/analysis?path=` | Beat grid |
| GET | `/transitions` | Names the renderer accepts |
| GET | `/edl?path=` | Read a cut list |
| POST | `/edl` | Write one back |
| POST | `/edl/render` | Render a timeline (job kind `edl`) |
| POST | `/auto/run` | Start the pipeline (job kind `auto`) |
| GET | `/auto/job?root=` | Saved state — what a resume would skip |

The pipeline runs on the same `RunManager` as every other long operation, so
pause, cancel and the event socket work unchanged. Stage transitions arrive as
`{"type": "stage", ...}` events.
