# Video Highlighter — web UI

A Tauri v2 desktop app: React + shadcn/ui front end, with the existing Python
engine running behind it as a FastAPI sidecar.

```
frontend/           Vite + React + Tailwind v4 + shadcn/ui   (this directory)
frontend/src-tauri/ Tauri shell (WebView2); spawns and reaps the sidecar
sidecar/server.py   FastAPI over the engine; streams log/progress on a WebSocket
sidecar/worker.py   Runs each job in a child process (see "Why a child process")
```

The engine itself is untouched: the sidecar calls the same
`pipeline.run_highlighter` the Qt app calls, with `log_fn` / `progress_fn` /
`cancel_flag` routed to a socket instead of Qt signals. Heavy compute stays in
Python, so nothing performance-sensitive crosses the webview boundary. The
native Qt timeline viewer is still used for the video-heavy surfaces — the
"Timeline Viewer" button launches it.

## What the web app adds over the Qt GUI

- **Folder-at-once input** — "Add Folder" scans a directory (recursively) for
  videos, so you can point it at a Downloads/GoPro folder instead of picking
  files one by one (`GET /scan-folder`).
- **Reel + music** — with more than one video, "Combine into one reel" stitches
  the produced highlights into a single video after the run. An optional music
  bed is applied once to the reel, in one of three modes: **replace** clip
  audio, **mix** under it, or **duck** it (music dips when the clip is loud).
  Runs as a `combine` job (`POST /combine`).
- **Rotation-correct output** — the engine reads each clip's orientation
  metadata (`GET /video-info` → `rotation`) and bakes rotated phone/GoPro
  footage upright when cutting and combining, so the reel isn't sideways.
- **Blur gate** — "Penalize blurry clips" scores candidates by
  variance-of-Laplacian sharpness and demotes soft ones (off by default).

These live in the engine (`modules/{video_probe,combine_videos,music_track,clip_quality}.py`),
so the Qt app can call them too; the web UI is just the first to surface them.

## Develop

```powershell
.\dev.ps1
```

Rust on Windows needs the MSVC toolchain on PATH and `LIB` set. The default
shell doesn't have it, so `dev.ps1` sources `vcvars64.bat` before running
`pnpm tauri dev` — without it you get `LNK1104: cannot open file 'msvcrt.lib'`.

In dev the shell runs the sidecar straight from the project venv
(`.venv/Scripts/python.exe -m sidecar.server`), so Python changes need only a
restart, not a rebuild. Release builds use the bundled PyInstaller output.

To run the sidecar on its own (useful when debugging the API):

```powershell
..\.venv\Scripts\python.exe -m sidecar.server --port 8756
# then http://127.0.0.1:8756/docs
```

## Build

The sidecar has to be built first — `tauri.conf.json` bundles
`dist-sidecar/vh-sidecar` as a resource.

```powershell
# from the project root
.\.venv\Scripts\python.exe -m PyInstaller packaging/vh-sidecar.spec --noconfirm `
  --distpath dist-sidecar --workpath build-sidecar

cd frontend
pnpm tauri build
```

CI does the same in the `build-web-windows` job of
`.github/workflows/build-release.yaml`.

### Why a resource, not `externalBin`

Tauri's `externalBin` takes a single file, which would mean a PyInstaller
*onefile* build — and that unpacks ~2GB of torch/openvino into a temp directory
on **every** launch. The spec builds *onedir* instead; the folder ships as a
bundled resource and the Rust shell resolves `vh-sidecar/vh-sidecar.exe` inside
it via the path resolver.

### Why a child process

torch's native runtime can hard-crash the interpreter (access violation
`0xc0000005` in `c10.dll`) when a run is cancelled mid-inference. In-process
that would kill the HTTP server and leave the UI staring at a dead socket, so
each job runs in a spawned child; if it dies, the parent reports
"The processing engine stopped unexpectedly" and stays up for the next run.
`main.py` works around the same class of problem by hard-exiting on close.
