<p align="center">
  <img src="assets/icon.png" alt="VideoHighlighter" width="160">
</p>

# VideoHighlighter

<!-- hy-mt2-i18n:start -->
**English** | [中文](./README_zh-CN.md) | [日本語](./README_ja.md) | [Español](./README_es.md)
<!-- hy-mt2-i18n:end -->

**Find and explain the moments that matter in footage you won't upload — then export a cut, on your machine.**

A local desktop app. Drop in raw video; it scores strong moments from scene,
motion, audio, objects, actions and the transcript, shows you *why* each one
scored on a signal timeline and in a report, then exports a highlight reel and
the individual clips. Nothing is uploaded for analysis, and the basic pipeline
needs no API key.

Free and open source (AGPL-3.0). Windows, macOS, Linux. Intel, AMD and NVIDIA
GPUs.

> **It's free.** To make sure you see new releases in future, please click the
> motivation button: the ⭐ at the top of the page. It's the cheapest payment we
> accept.

## What people use it for

- **A 4-hour Twitch or YouTube VOD → a few minutes of highlights.** Crowd noise,
  your own reactions and scene changes all score, so the loud parts surface
  without scrubbing.
- **A night of CCTV or trail-camera footage → the minutes something moved.**
  Object detection plus motion, on hours of nothing, locally — which matters
  when the footage is of your own property.
- **A GoPro or drone card → a finished film.** The **Auto** tab finds the card,
  copies it off, cuts the highlights, builds the reel and lays music on the
  beat, as one resumable job.
- **A match → the goals.** Write a rule for "ball inside net" and it scores that
  event by name, instead of hoping a 400-class action model has a word for it.
- **A long interview, lecture or podcast → chapters and subtitles.** Local
  Whisper transcript, chaptered video, optional local translation.
- **Dashcam or bodycam review → an account you can hand to someone.** The report
  states what was measured, what was only said, and what it could not determine.

Not sure which detector fits your footage?
[docs/DETECTION-GUIDE.md](docs/DETECTION-GUIDE.md) covers what each one is good
at and where it falls down.

## What makes it different

**[Every run explains itself.](docs/REPORTS.md)** The report is the arithmetic
behind each kept moment — the per-signal point breakdown, what fired, what
scored well and still missed the cut. It names the claims that came from the
transcript and were never measured, instead of scoring them anyway.
**[Open a real one →](https://aseiel.github.io/VideoHighlighter-site/example-report.html)**

**[Composition rules.](docs/DETECTION-GUIDE.md#4-the-composition-engine)** You
say what a *combination* of detections means for your footage — one class inside
another, counted, held over a window — and that becomes an event under a name
you choose. Because a rule re-reads detections that already exist, editing one
and re-running costs milliseconds.

## Preview

![VideoHighlighter](assets/Highlighter.png)

**Timeline Viewer**

![Timeline Viewer](assets/TimelineViewer.png)

**Demo**

https://github.com/user-attachments/assets/5c85af94-9228-4537-926a-1ed7a91fa5ee

## Install

Grab a build from [Releases](https://github.com/Aseiel/VideoHighlighter/releases):

- **Windows** — run
  [`00-VideoHighlighter-Windows-Setup.exe`](https://github.com/Aseiel/VideoHighlighter/releases/latest/download/00-VideoHighlighter-Windows-Setup.exe).
  Per-user, no admin. It is not code-signed yet, so click through *More info →
  Run anyway*.
- **macOS** — drag the `.dmg` into Applications, then clear the quarantine flag
  once: `xattr -dr com.apple.quarantine /Applications/VideoHighlighter.app`
  ([why](docs/INSTALL.md#macos)).
- **Linux / from source** — `pip install -r requirements.txt && python main.py`.
  FFmpeg comes with it.

Portable builds, GPU setup, where it writes, and fixing an oversized UI:
**[docs/INSTALL.md](docs/INSTALL.md)**.

## Documentation

| | |
| --- | --- |
| [Choosing a detector](docs/DETECTION-GUIDE.md) | Objects, actions, CLIP search, composition rules — what each is for |
| [Why these moments](docs/REPORTS.md) | What the report contains and why it is built that way |
| [The Auto pipeline](docs/AUTO-PIPELINE.md) | Card → ingest → script → music → reel, resumable |
| [Installing](docs/INSTALL.md) | Every platform, GPU backends, settings |
| [Training a model](docs/CUSTOM-MODEL-TRAINING.md) | Label your own class and train it |
| [Community models](docs/COMMUNITY-MODELS.md) | Install models other people trained, publish your own |
| [Intel GPU](docs/INTEL-GPU.md) · [AMD GPU](docs/AMD-GPU.md) | Vendor-specific acceleration |
| [Remote ollama](docs/OLLAMA-REMOTE.md) | Run the local LLM on another box on your LAN |

## Pro edition

**VideoHighlighter — this repository — is free software under AGPL-3.0, and
stays that way.** It includes offline analysis, live face detection, VR
side-by-side playback, CLIP search, the composition engine, model training and
the model hub.

**[VideoHighlighter Pro](https://aseiel.github.io/VideoHighlighter-site/)** is a
separate paid edition that adds real-time work on top: live object and action
overlays during playback, teaching a category by drawing a box around it,
find-more-like-this search, open-vocabulary detection and counter/scoreboard
detection.

Explanation is not among them. The report, the findings and the advisor are
identical in both editions — a cloud tool gives you a button and a result you
cannot interrogate; answering "why", locally, is what this is instead.

## Community

VideoHighlighter occasionally has feelings about your footage. When it does:
[join the Discord](https://discord.gg/cUPJqPAMmm) and yell in #support, I'm
usually around. Bugs and ideas are welcome in
[Issues](https://github.com/Aseiel/VideoHighlighter/issues).

## License

Copyright (C) 2026 Przemysław Kreft and Meric Donmezer.

Released under the GNU Affero General Public License v3.0 — use, modify and
distribute it freely, provided modified versions, including ones offered over a
network, make their complete source available under the same license. Full text
in [LICENSE](LICENSE); notice in [COPYRIGHT](COPYRIGHT).

Contributors keep copyright in their own work — see
[CONTRIBUTING.md](CONTRIBUTING.md) and [CLA.md](CLA.md). VideoHighlighter is
also offered under a separate commercial license by the copyright holders.

## Background

This started as a personal tool to generate subtitles for videos, for my
7-year-old son. Over time it turned into a highlights generator for movies,
sports and personal footage. The goal is unchanged: speed up video analysis,
generate highlights you can explain, and create subtitles automatically —
without uploading footage you would rather keep local.

![Stars History](assets/star-history-2026630.png)
