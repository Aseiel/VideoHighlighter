# Installing VideoHighlighter

Downloads are on the [Releases page](https://github.com/Aseiel/VideoHighlighter/releases).

## GPU support

VideoHighlighter runs on **Intel, AMD and NVIDIA** GPUs, and on CPU alone if
there is nothing else. Tested on:

| GPU | Backend |
| --- | ------- |
| Intel Arc A750 | OpenVINO |
| NVIDIA GeForce GTX 1060 | CUDA |
| AMD Radeon RX 570 | DirectML |

Per-vendor setup and troubleshooting: [Intel](INTEL-GPU.md) · [AMD](AMD-GPU.md).

## Windows

**Installer (recommended).** Download
[`00-VideoHighlighter-Windows-Setup.exe`](https://github.com/Aseiel/VideoHighlighter/releases/latest/download/00-VideoHighlighter-Windows-Setup.exe)
and run it. It fetches both archive parts and unpacks them — roughly a 3 GB
download, installed per-user, no administrator rights needed.

The build is **not code-signed** yet, so Windows SmartScreen or Chrome may block
it. Use *Keep* in the browser, and *More info → Run anyway* in the SmartScreen
dialog.

**Portable.** Download **both** `VideoHighlighter-Windows-*.7z.001` and
`.7z.002` into the same folder, extract the `.001` file with
[7-Zip](https://www.7-zip.org/) (it pulls in the second part itself), and run
`VideoHighlighter.exe`.

Then: run `VideoHighlighter.exe` from the extracted build.

## macOS

Download the `.dmg` from
[Releases](https://github.com/Aseiel/VideoHighlighter/releases) and drag
**VideoHighlighter** into Applications.

The app is ad-hoc signed and **not notarised**, so macOS quarantines it and the
first launch fails — usually as *"VideoHighlighter is damaged and can't be
opened"*. Nothing is damaged: that is Gatekeeper reacting to the missing
notarisation, and it says the same thing about a perfectly good download. Clear
the quarantine flag once, in Terminal:

```bash
xattr -dr com.apple.quarantine /Applications/VideoHighlighter.app
```

Then open it normally from Applications. Repeat this after each update, since
the flag comes back with the new download.

Mac builds get far less testing than Windows — please
[open an issue](https://github.com/Aseiel/VideoHighlighter/issues) when
something breaks.

## Linux / from source

```bash
pip install -r requirements.txt
python main.py
```

FFmpeg comes with it via `imageio-ffmpeg` — nothing to install separately. An
FFmpeg already on your `PATH` is used instead when there is one.

## Two front ends

Both drive the same engine:

- **Qt desktop GUI** (`main.py`) — the original, with the full Timeline Viewer.
- **Web app** (`frontend/` + `sidecar/`) — a Tauri v2 shell around a React UI,
  with the Python engine running behind it as a FastAPI sidecar. Adds
  folder-at-once input, the reel + music controls, and the blur gate. See
  [`frontend/README.md`](../frontend/README.md). It still launches the Qt window
  for the Timeline Viewer.

## Where it writes

Beside the executable, so a portable install stays self-contained: copy the
folder and your caches come with it. When that folder refuses writes, it uses
your local app data folder instead, rather than requiring you to start the app
as an administrator.

macOS always uses `~/Library/Application Support/VideoHighlighter`, because a
bundle is not a place to write.

Footage, transcripts and local models stay on disk. The basic pipeline needs no
API key.

## Interface too large?

A 55" 4K panel is scaled by the system for a television, not for an app at desk
distance. Set `ui_scale` in `config.yaml` (or the `VH_UI_SCALE` environment
variable) to a multiplier applied on top of the system's own: `0.75` on a 200%
display gives you 150%. It is read once at startup, so restart the app after
changing it.

## Optional pieces

- **Transcripts** run on OpenAI Whisper, locally. Whisper is MIT licensed.
- **Subtitle translation** runs on a local LLM through
  [ollama](https://ollama.com) — no translation service, API key or account is
  involved, and the text never leaves the machine. Without ollama, subtitles are
  written in the language that was spoken. Running ollama on another box on your
  LAN: [OLLAMA-REMOTE.md](OLLAMA-REMOTE.md).

This project ships no paid API keys. Bring your own if you want to use a hosted
service.
