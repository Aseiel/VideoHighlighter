# In-place updates

A per-user Windows install updates itself: the app reads a small channel file,
downloads only the files that changed, verifies each one, and swaps them in.
This page is for whoever cuts releases.

## How it fits together

```
build-release.yaml ──► R2 bucket vh-updates (public, HTTPS)
  builds core + packs      files/<sha256>                              blobs, deduplicated forever
  stages blobs and the     releases/<edition>/windows/<v>/manifest.json   unsigned until step 3
  UNSIGNED manifest        releases/<edition>/windows/<v>/manifest.json.sig
                           channels/<edition>.json                     "latest is <v>"  ◄── publish-update.yaml
```

1. **Build & Release** (`build-release.yaml`) builds the app, smoke-tests it,
   and stages the release on the bucket: every file under `files/` by its
   SHA-256 (only new ones are uploaded) and the manifest listing them.
2. You **publish the GitHub release**, then sign the manifest on the machine
   that holds the release key:

   ```
   python tools/publish_release.py sign --version 0.13.0 --base-url https://<your update host>
   ```

   It fetches the staged manifest, shows what it describes, signs it, checks
   the signature the way the app will, and prints the next command.
3. **Publish update** (`publish-update.yaml`, run with that signature) checks
   the signature against the key the release was built with, checks every blob
   is on the bucket and publicly reachable, uploads the `.sig`, and only then
   writes `channels/<edition>.json`. From that moment installed copies see it.

The private key never goes to CI. It decides what runs on every user's machine,
and an Actions secret can be read by anyone who can edit a workflow.

## What the app refuses

A signed manifest is still refused (and the user offered the download page) if
it is for the other edition, another platform, not newer than the running
build, or declares a `min_version` above it. The channel file is unsigned, so
these are what stop it from rolling installs back or cross-grading them. Only
Windows installs in a writable folder ("for me only") update in place; macOS
and all-users installs get the download page.

## One-time setup

1. **Release key.** On your own machine, from the repo root:

   ```
   pip install cryptography
   python tools/build_manifest.py keygen --update-module
   ```

   This writes `.secrets/release_signing_key.pem` (gitignored; back it up
   offline, since losing it means installed copies can never be updated in
   place again) and embeds the public half in
   `modules/update/update_manifest.py`. Commit that one-line change. Builds
   made before this commit cannot update in place; they get the download page.

2. **Bucket.** In Cloudflare R2, bucket `vh-updates` → Settings → **Public
   access**: connect a custom domain (for example `updates.<your domain>`), or
   enable the `r2.dev` URL. Cloudflare rate-limits `r2.dev` and recommends a
   custom domain for real traffic.

3. **API token.** R2 → Manage R2 API tokens → create one with *Object Read &
   Write* on `vh-updates` only.

4. **GitHub** → Settings → Secrets and variables → Actions:

   | Kind | Name | Value |
   |---|---|---|
   | Secret | `R2_ACCESS_KEY_ID` | the token's access key ID |
   | Secret | `R2_SECRET_ACCESS_KEY` | the token's secret |
   | Variable | `R2_ACCOUNT_ID` | the Cloudflare account ID (in the dashboard URL) |
   | Variable | `UPDATE_BASE_URL` | the bucket's public URL, `https://…`, no trailing slash |

`UPDATE_BASE_URL` is stamped into each build's `manifest.json`, and that is how
an installed copy finds its channel file, so a build made without it cannot
find the update host. Builds without it still read the marketing site's
channel file (`update_check._MANIFEST_BASE`), as every build did before.

## Build & Release options

- **dry_run**: build and smoke-test everything, publish nothing (no draft
  release, nothing on the bucket). Use it to try a workflow change.
- **web_ui**: also build the Tauri web UI. Off by default: its sidecar still
  bundles CUDA PyTorch itself, which costs about 40 Windows minutes.
- **update_min_version**: set it when a release changes the install's layout
  so that older installs cannot be updated in place safely. They get the
  download page instead.

## PyTorch is built once per torch version, not per release

The desktop app ships as a core without PyTorch plus packs (`torch-cpu`,
`torch-cu128`, `models-clip`) published under a `packs-torch-<version>` release.
A normal release downloads those packs and never installs or compresses CUDA.
The packs are rebuilt only when torch or CLIP changes:

1. **Build packs** with *rebuild_packs* ticked;
2. **Publish packs** with that run's id (a new tag; published packs are never
   replaced);
3. move `PACKS_BASE` in `build-release.yaml` and `build-packs.yaml` to the new tag.

Packs live in `packs/` beside the app and are not in the update manifest, so
updates never touch them.
