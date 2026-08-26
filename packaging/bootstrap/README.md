# Bootstrap installer (isolated)

**Does not touch** the app, `build-release.yaml`, or Free/Pro packaging
pipelines. Everything here is opt-in: you run it locally, test it, and only
later (deliberately) attach a built installer to a GitHub Release.

## Why

GitHub Release assets are capped at **2 GB**. The Windows build is larger, so
it ships as split 7z (`.001` + `.002`). ChatGPT-driven users often download
only the first part → ~100 failed installs on Free 0.9.0 (269 vs 156 downloads).

This folder is a **thin downloader + extractor**: one small zip / script that
fetches both parts and unpacks them.

## Layout

| Path | Role |
|------|------|
| `config.json` | Which release assets to pull (edit for Free vs Pro / version) |
| `Install-VideoHighlighter.ps1` | The bootstrap (download both volumes + extract) |
| `Install-VideoHighlighter.bat` | Double-click entry for Windows users |
| `out/` | Local downloads / extract target (gitignored) |

## Which release it installs

With `"use_latest": true` the script asks the GitHub API for the current
release and takes every asset matching `asset_pattern`, so a new Free release
needs no edit here. `tag` / `assets` / `base_url` are the fallback used when
the API cannot be reached.

Pro assets live on a private repo that an anonymous API call cannot see, so
`config.pro.example.json` keeps `use_latest` off and pins the tag.

## Try it (nothing pushed to customers)

```powershell
cd packaging\bootstrap
.\Install-VideoHighlighter.bat
```

Or:

```powershell
cd packaging\bootstrap
powershell -ExecutionPolicy Bypass -File .\Install-VideoHighlighter.ps1
```

Default `config.json` points at the **public Free** Windows split for the
version in `version.py`. Do not edit the tag by hand — it is generated, and a
test fails when it drifts from `version.py`. After a version bump:

```powershell
python tools/build_bootstrap_zip.py --edition free --write-config
```

## Requirements on the machine

- Windows + PowerShell 5+
- Network access to `github.com`
- **7-Zip** on PATH (`7z`), *or* the script will offer to download the
  official `7zr.exe` into `out\tools\` (no system install required)

## What this does *not* do yet

- No change to Free or Pro release CI
- No CUDA / pip-in-app component downloads (Flowframes-style — later)
- No code signing of the bootstrap itself

## Shipping later (separate decision)

When happy:

1. CI builds `VideoHighlighter-Windows-Setup.zip` automatically (`tools/build_bootstrap_zip.py`).
2. Attach that zip as the **first** asset on the Free release page (filename must stay
   constant so `/releases/latest/download/VideoHighlighter-Windows-Setup.zip` works).
3. Keep the `.7z.001` / `.7z.002` assets as today — bootstrap just downloads them.

Pro customers use Lemon Squeezy (single `.7z`); see `docs/LS-PRODUCT-SETUP.md` in the Pro repo.

## Safety

- Lives under `packaging/bootstrap/` — outside `tools/check_pro_boundary.py`
  default targets.
- `out/` is gitignored; do not commit downloaded archives.
- Pushing this folder alone does not change how the app is built.
