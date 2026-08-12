<p align="center">
  <img src="assets/icon.png" alt="VideoHighlighter" width="160">
</p>

VideoHighlighter (Freeware)

<!-- hy-mt2-i18n:start -->
**English** | [中文](./README_zh-CN.md) | [日本語](./README_ja.md) | [Español](./README_es.md)
<!-- hy-mt2-i18n:end -->

A Python tool to automatically generate highlight clips from videos using scene detection, motion detection, audio peaks, object detection, action recognition, and transcript analysis.

> **It's free.** To make sure you see new releases in future, please click the
> motivation button: the ⭐ at the top of the page. It's the cheapest payment we
> accept.


Features

Detects:
- Scenes using OpenCV.
- Motion peaks and scene changes.
- Objects
- Actions
- Audio peaks.

Generates transcript subtitles via OpenAI Whisper.
Cuts and merges top scoring segments into a highlight video.
Combines many clips into a single reel, with an optional music bed
(replace / mix / duck) and correct handling of rotated (phone / GoPro) footage.
Optionally penalizes blurry clips so sharp moments win.
Fully configurable: frame skip, highlight duration, keywords.

Two front ends over one engine:
- **Qt desktop GUI** (`main.py`) — the original, with the full Timeline Viewer.
- **Web app** (`frontend/` + `sidecar/`) — a Tauri v2 shell around a React UI,
  with the Python engine running behind it as a FastAPI sidecar. Adds
  folder-at-once input, the reel + music controls, and the blur gate. See
  [`frontend/README.md`](frontend/README.md). It still launches the Qt window
  for the Timeline Viewer.

## Card to film

The **Auto** tab runs the whole thing as one resumable job: find the camera
card, copy it off, find the highlights, build the reel, lay the music.

- **Ingest** — cards are found by layout, not drive letter, and GoPro's
  chapter-before-file-number naming is sorted back into recording order (the
  reason a plain listing interleaves separate takes). Copies verify before they
  land, so an interrupted transfer can't leave a short file that looks whole.
  Re-running copies nothing. Nothing is deleted from the card.
- **Script** — a YAML file saying what the film should contain, beat by beat,
  so a run can express intent instead of just "the highest-scoring seconds".
  Unknown keys are refused with a line number and a suggestion rather than
  silently ignored.
- **Music** — beats, downbeats, tempo and energy sections, on numpy and ffmpeg
  alone. Cuts can then land on the beat rather than near it.
- **Resume** — every stage records what it produced and a re-run skips whatever
  is still on disk, because the expensive middle is exactly what gets
  interrupted.

Full detail: [docs/AUTO-PIPELINE.md](docs/AUTO-PIPELINE.md).

Not sure which detector to reach for? See
[docs/DETECTION-GUIDE.md](docs/DETECTION-GUIDE.md) — what object recognition,
action recognition, CLIP search and the composition engine are each good at,
and where each one falls down.

> **Want real-time detection?** Everything above runs offline, after the fact.
> [VideoHighlighter Pro](#pro-edition) adds live object and action overlays
> during playback, teach-by-example categories, open-vocabulary detection and
> counter detection. [See what's different →](#pro-edition)


## Preview

![VideoHighlighter](assets/Highlighter.png)

## Timeline Viewer
![Timeline Viewer](assets/TimelineViewer.png)

## Action Recognition
![Action Recognition](assets/power_rangers_actions_annotated.gif)

## Workflow Stages
![Workflow Stages](assets/workflow_stages.png)

## Pro edition

This edition already includes live face detection, VR side-by-side playback and
rendering, offline analysis, CLIP search, the composition engine, and the
training scripts.

[VideoHighlighter Pro](https://aseiel.github.io/VideoHighlighter-site/) adds:

- **Live object and action overlays** — real-time detection during playback,
  including on side-by-side VR footage.
- **Teach a category by pointing** — draw a box around anything, name it, and
  it is scored live from then on. No dataset, no training run.
- **Find more like this** — pick a region in one frame and search the whole
  video for it.
- **Open-vocabulary detection** — type a plain word and find it, with no
  trained model for it.
- **Counter / scoreboard detection** — if the footage has an on-screen counter,
  every tick proves an event, so Pro can show which real moments the detector
  missed.

This edition remains free and AGPL-3.0 licensed.

## Installation

### Windows (recommended)
Download the latest `.exe` from [Releases](https://github.com/Aseiel/VideoHighlighter/releases) — no Python or dependencies required.

### Linux / Building from Source
1. **Python & FFmpeg**
   FFmpeg must be installed and available in your system PATH.

## Usage
Linux: python main.py 
Windows: run Videohighlighter.exe
Mac: I think not working, will fix it one day. DMG file is still generated

## Discord
VideoHighlighter occasionally has feelings about your footage. When it does:
[Join the Discord](https://discord.gg/cUPJqPAMmm) and yell in #support, I'm usually around.


## Notes

OpenAI Whisper is MIT licensed — freely usable.

Google Translate API is optional. If using unofficial libraries (googletrans), no API key is needed, but results may break if Google changes endpoints.

This project does not include any paid API keys. Users must provide their own if using official services.


## License

This repository is released under the GNU Affero General Public License v3.0 (AGPLv3). You are free to use, modify, and distribute the code, provided that any modified versions, including those offered over a network, make their complete source code available under the same license.


## Project Background

This project started as a personal tool to automatically generate subtitles for videos, for my young 7 years old son. Over time, it evolved into a highlights generator for movies, sports, and personal videos.

The primary goal remains practical: speed up video analysis, generate highlights, and create accessible subtitles automatically.

![Stars History](assets/star-history-2026630.png)
