"""Put a built release on the update host, and sign it.

The update host is an S3-compatible bucket (Cloudflare R2, ``vh-updates``)
served over public HTTPS. Its layout::

    files/<sha256>                                   one blob per distinct file
    releases/<edition>/<platform>/<version>/manifest.json
    releases/<edition>/<platform>/<version>/manifest.json.sig
    channels/<edition>.json                          "the latest is X" pointer

A release goes out in three steps, and only the middle one needs the key:

1. **CI** (build-release.yaml) builds the app, then ``prepare`` lays it out as
   above and uploads the blobs and the *unsigned* manifest. Harmless on their
   own: blobs are only trusted when a signed manifest names them, and nothing
   points at the manifest yet.
2. **You**, on the machine that holds the release key::

       python tools/publish_release.py sign --version 0.12.2

   fetches that manifest from the host, shows what it describes, signs it, and
   prints the signature.
3. **CI** (publish-update.yaml, run with that signature) checks it against the
   public key the release was built with, checks every blob is on the host,
   uploads the ``.sig``, and only then writes ``channels/<edition>.json`` —
   the moment installed copies can see the release.

Why the key never enters CI: it decides what executes on every user's machine,
and a secret in Actions is readable by anyone who can edit a workflow. The
signature is 86 characters; carrying it by hand costs nothing.

Why content-addressed
---------------------
Blobs are named by their hash, not their path, so uploading a release is a
*sync*: anything already on the host is skipped, and each release pushes only
genuinely new bytes. An install can also jump several versions at once,
because every blob any manifest ever referenced is still there.

Hardlinks are used where the filesystem allows it, so preparing a 2 GB bundle
does not cost another 2 GB of disk.
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.update.update_manifest import (  # noqa: E402
    MANIFEST_FILENAME,
    SIGNATURE_FILENAME,
    local_path,
    verify_manifest,
)

DEFAULT_KEY_PATH = os.path.join(".secrets", "release_signing_key.pem")
BASE_URL_ENV = "VH_UPDATE_BASE_URL"


def channel_name(edition: str) -> str:
    """``"pro"`` or ``"free"``, as ``update_check._channel`` names it."""
    return "pro" if (edition or "").strip().lower() == "pro" else "free"


def release_prefix(manifest: dict) -> str:
    """Where a release's manifest lives on the host, without a leading slash."""
    return "releases/{}/{}/{}".format(
        channel_name(manifest.get("edition", "")),
        str(manifest.get("platform") or "windows").lower(),
        manifest["version"])


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def prepare(args) -> int:
    root = os.path.abspath(args.root)
    out = os.path.abspath(args.out)
    manifest_path = os.path.join(root, MANIFEST_FILENAME)

    if not os.path.exists(manifest_path):
        print(f"FAIL:no manifest in {root}")
        print("  run: python tools/build_manifest.py generate --root <bundle> ...")
        return 1

    with open(manifest_path, "rb") as handle:
        raw = handle.read()
    manifest = json.loads(raw.decode("utf-8"))

    if not manifest.get("base_url"):
        print("FAIL:manifest has no base_url — the updater would not know where")
        print("  to fetch files. Re-generate with --base-url <public URL>.")
        return 1

    signature_path = os.path.join(root, SIGNATURE_FILENAME)
    if not os.path.exists(signature_path) and not args.allow_unsigned:
        print(f"FAIL:{SIGNATURE_FILENAME} missing — an unsigned release is")
        print("  rejected by every installed copy. Sign it, or pass")
        print("  --allow-unsigned to stage it for signing later.")
        return 1

    blobs = os.path.join(out, "files")
    os.makedirs(blobs, exist_ok=True)

    linked = copied = skipped = 0
    new_bytes = stored_bytes = 0
    seen = set()
    gz = manifest.get("compression") == "gzip"

    for entry in manifest["files"]:
        digest = entry["sha256"]
        if digest in seen:
            continue          # same content twice in the bundle: one blob
        seen.add(digest)

        destination = os.path.join(blobs, digest + (".gz" if gz else ""))
        if os.path.exists(destination):
            skipped += 1
            stored_bytes += os.path.getsize(destination)
            continue

        source = local_path(root, entry["path"])
        if gz:
            _gzip_file(source, destination)
            copied += 1
        else:
            try:
                os.link(source, destination)
                linked += 1
            except OSError:
                # Different volume, or a filesystem without hardlinks.
                shutil.copy2(source, destination)
                copied += 1
        new_bytes += int(entry.get("size", 0))
        stored_bytes += os.path.getsize(destination)

    prefix = release_prefix(manifest)
    release_dir = os.path.join(out, *prefix.split("/"))
    os.makedirs(release_dir, exist_ok=True)
    # Byte for byte: the signature covers these exact bytes.
    with open(os.path.join(release_dir, MANIFEST_FILENAME), "wb") as handle:
        handle.write(raw)
    if os.path.exists(signature_path):
        shutil.copy2(signature_path, os.path.join(release_dir, SIGNATURE_FILENAME))

    total = sum(int(e.get("size", 0)) for e in manifest["files"])
    print(f"OK:{out}")
    print(f"  version      {manifest.get('version')} {manifest.get('edition', '')} "
          f"{manifest.get('platform', 'windows')}")
    print(f"  base_url     {manifest['base_url']}")
    print(f"  manifest     {prefix}/{MANIFEST_FILENAME}")
    print(f"  blobs        {len(seen)} distinct ({linked} linked, {copied} copied, "
          f"{skipped} already prepared)")
    print(f"  bytes        {total / (1024 ** 2):.1f} MB in the release, "
          f"{stored_bytes / (1024 ** 2):.1f} MB as stored"
          + (" (gzip)" if gz else ""))
    if args.github_output:
        with open(args.github_output, "a", encoding="utf-8") as handle:
            handle.write(f"prefix={prefix}\n")
    return 0


def _gzip_file(source: str, destination: str) -> None:
    """gzip ``source`` to ``destination`` with no timestamp or name in the
    header, so the same file always compresses to the same bytes: a re-staged
    release then matches what is on the host, and ``sync --size-only`` is
    right to skip it."""
    import gzip

    tmp = destination + ".tmp"
    with open(source, "rb") as src, open(tmp, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6,
                           mtime=0) as out:
            shutil.copyfileobj(src, out, 1024 * 1024)
    os.replace(tmp, destination)


# ---------------------------------------------------------------------------
# sign
# ---------------------------------------------------------------------------

def _fetch(url: str) -> bytes:
    import urllib.request

    from modules.system import https_certs

    request = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
    with urllib.request.urlopen(request, timeout=60,
                                **https_certs.opener_kwargs()) as response:
        return response.read()


def sign(args) -> int:
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    if args.manifest:
        with open(args.manifest, "rb") as handle:
            raw = handle.read()
        source = args.manifest
    else:
        base = (args.base_url or os.environ.get(BASE_URL_ENV, "")).rstrip("/")
        if not base or not args.version:
            print("FAIL:say which manifest: --manifest <file>, or --version with")
            print(f"  --base-url (or {BASE_URL_ENV}) naming the update host.")
            return 1
        from version import __edition__
        stub = {"edition": args.edition or __edition__,
                "platform": args.platform, "version": args.version}
        source = f"{base}/{release_prefix(stub)}/{MANIFEST_FILENAME}"
        try:
            raw = _fetch(source)
        except Exception as exc:
            print(f"FAIL:could not fetch {source}: {exc}")
            return 1

    try:
        manifest = json.loads(raw.decode("utf-8"))
        files = manifest["files"]
    except (ValueError, KeyError, TypeError, UnicodeDecodeError) as exc:
        print(f"FAIL:{source} is not a release manifest ({exc})")
        return 1

    if args.version and str(manifest.get("version")) != args.version:
        print(f"FAIL:{source} describes {manifest.get('version')}, not {args.version}")
        return 1

    total = sum(int(e.get("size", 0)) for e in files)
    print(f"Signing {source}")
    print(f"  version   {manifest.get('version')}")
    print(f"  edition   {manifest.get('edition')}")
    print(f"  platform  {manifest.get('platform', 'windows')}")
    print(f"  base_url  {manifest.get('base_url')}")
    print(f"  files     {len(files)} ({total / (1024 ** 2):.1f} MB)")
    if manifest.get("min_version"):
        print(f"  min       {manifest['min_version']} (older installs get the download)")

    with open(args.key, "rb") as handle:
        private = load_pem_private_key(handle.read(), password=None)
    encoded = base64.urlsafe_b64encode(private.sign(raw)).decode("ascii").rstrip("=")

    # The same check every installed copy will make. A key that does not match
    # the embedded public one signs something nobody can install.
    if verify_manifest(raw, encoded) is None:
        print("FAIL:the app would reject this signature — is this the key whose")
        print("  public half is RELEASE_PUBLIC_KEY_HEX in modules/update/update_manifest.py?")
        return 1

    out = args.out or f"manifest-{manifest.get('version')}.json.sig"
    with open(out, "w", encoding="ascii") as handle:
        handle.write(encoded + "\n")

    print()
    print(f"OK:signature (also written to {out}):")
    print(f"  {encoded}")
    print()
    print("Publish it once the GitHub release is out:")
    print(f"  gh workflow run publish-update.yaml -f version={manifest.get('version')} "
          f"-f signature={encoded}")
    return 0


# ---------------------------------------------------------------------------
# channel
# ---------------------------------------------------------------------------

def build_channel(manifest: dict, manifest_url: str, *, previous: dict = None,
                  notes: str = "", notes_url: str = "",
                  download_url: str = "") -> dict:
    """The channel file that announces ``manifest`` to installed copies.

    Other platforms' entries in ``previous`` are kept: publishing the Windows
    build must not withdraw a macOS one. A platform whose entry is older than
    the version announced is dropped rather than left pointing at a release
    that is no longer the latest.
    """
    version = str(manifest["version"])
    platform = str(manifest.get("platform") or "windows").lower()

    manifests = {}
    previous = previous if isinstance(previous, dict) else {}
    if str(previous.get("version")) == version and isinstance(previous.get("manifests"), dict):
        manifests.update(previous["manifests"])
    manifests[platform] = manifest_url

    return {
        "version": version,
        "date": str(manifest.get("date") or _dt.date.today().isoformat()),
        "notes": notes or str(manifest.get("notes") or ""),
        "notes_url": notes_url,
        "download_url": download_url,
        "manifests": manifests,
    }


def channel(args) -> int:
    from modules.update.update_check import is_newer

    with open(args.manifest, "rb") as handle:
        manifest = json.loads(handle.read().decode("utf-8"))
    previous = {}
    if args.previous and os.path.exists(args.previous):
        try:
            with open(args.previous, "r", encoding="utf-8") as handle:
                previous = json.load(handle)
        except (OSError, ValueError):
            previous = {}

    # Moving the channel backwards would tell everyone on the newer release
    # that nothing is newer, and everyone else to fetch an older one.
    if isinstance(previous, dict) and is_newer(str(previous.get("version", "")),
                                               str(manifest["version"])):
        print(f"FAIL:the channel already announces {previous.get('version')}, "
              f"newer than {manifest['version']}.")
        return 1

    base = str(manifest.get("base_url") or "").rstrip("/")
    url = f"{base}/{release_prefix(manifest)}/{MANIFEST_FILENAME}"
    payload = build_channel(manifest, url, previous=previous, notes=args.notes or "",
                            notes_url=args.notes_url or "",
                            download_url=args.download_url or "")
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps(payload, indent=2))
    return 0


# ---------------------------------------------------------------------------
# check (CI, before anything is made visible)
# ---------------------------------------------------------------------------

def check(args) -> int:
    """Everything that must hold before the channel may point at a release.

    Run by publish-update.yaml on the manifest as stored on the host, with the
    signature the maintainer supplied and the bucket's listing of ``files/``.
    """
    with open(args.manifest, "rb") as handle:
        raw = handle.read()
    with open(args.sig, "r", encoding="ascii") as handle:
        signature = handle.read().strip()

    manifest = verify_manifest(raw, signature)
    if manifest is None:
        print("FAIL:the signature does not verify against the public key this")
        print("  release was built with. Nothing was published.")
        return 1

    problems = []
    if str(manifest.get("version")) != args.version:
        problems.append(f"manifest is {manifest.get('version')}, not {args.version}")
    if args.edition and channel_name(manifest.get("edition", "")) != channel_name(args.edition):
        problems.append(f"manifest is the {manifest.get('edition')} edition, not {args.edition}")
    base = str(manifest.get("base_url") or "").rstrip("/")
    if base != args.base_url.rstrip("/"):
        problems.append(f"manifest fetches from {base or 'nowhere'}, "
                        f"not the update host {args.base_url}")

    with open(args.listing, "r", encoding="utf-8") as handle:
        listing = json.load(handle) or []
    # `aws s3api list-objects-v2 --query 'Contents[].[Key,Size]'` output.
    on_host = {str(key).rsplit("/", 1)[-1]: int(size) for key, size in listing}
    gz = manifest.get("compression") == "gzip"
    missing = wrong = 0
    for entry in manifest["files"]:
        size = on_host.get(entry["sha256"] + (".gz" if gz else ""))
        if size is None:
            missing += 1
        elif not gz and size != int(entry.get("size", -1)):
            # A compressed blob's size is not in the manifest; its content is
            # still checked by every client, against the file's own hash.
            wrong += 1
    if missing or wrong:
        problems.append(f"{missing} blob(s) missing and {wrong} the wrong size "
                        "on the host; re-run the build's staging step")

    if problems:
        for problem in problems:
            print(f"FAIL:{problem}")
        return 1
    print(f"OK:{manifest['version']} {manifest.get('edition')} "
          f"{manifest.get('platform', 'windows')}: signature valid, "
          f"{len(manifest['files'])} files all on the host")
    return 0


# ---------------------------------------------------------------------------
# gc (prune-updates.yaml)
# ---------------------------------------------------------------------------

# Blobs younger than this are never deleted: a build uploads its blobs minutes
# before its manifest, and a listing taken in between would see them as
# unreferenced.
GC_GRACE_HOURS = 48


def unreferenced(listing, manifests: dict, channels: dict, keep: int = 3,
                 now=None, grace_hours: float = GC_GRACE_HOURS) -> dict:
    """Which ``files/`` blobs no release worth keeping refers to.

    ``listing``: ``[[key, size, last_modified_iso], ...]`` for the bucket.
    ``manifests``: ``{"releases/<ed>/<plat>/<ver>": manifest dict}``.
    ``channels``: ``{"channels/<ed>.json": channel dict}``.

    Kept: every blob of the newest ``keep`` releases of each edition and
    platform, and of every release a channel names. An install always updates
    *to* one of those, so nothing older is ever downloaded again. Manifests and
    signatures are never touched, only blobs, so an old release stays
    described even when its files are gone.
    """
    import datetime as dt

    from modules.update.update_check import parse_version

    now = now or dt.datetime.now(dt.timezone.utc)
    groups: dict = {}
    for prefix in manifests:
        parts = prefix.split("/")
        if len(parts) == 4:
            groups.setdefault(tuple(parts[1:3]), []).append(prefix)
    kept_releases = set()
    for prefixes in groups.values():
        prefixes.sort(key=lambda p: parse_version(p.rsplit("/", 1)[-1]), reverse=True)
        kept_releases.update(prefixes[:max(1, int(keep))])
    for channel in channels.values():
        for url in (channel.get("manifests") or {}).values():
            marker = "/releases/"
            if marker in str(url):
                kept_releases.add("releases/" + str(url).split(marker, 1)[1]
                                  .rsplit("/manifest.json", 1)[0])

    wanted = set()
    for prefix in kept_releases:
        manifest = manifests.get(prefix)
        if manifest is None:
            continue
        gz = ".gz" if manifest.get("compression") == "gzip" else ""
        wanted.update(f"files/{e['sha256']}{gz}" for e in manifest.get("files", []))

    delete, freed, young = [], 0, 0
    for row in listing or []:
        key, size = str(row[0]), int(row[1])
        if not key.startswith("files/") or key in wanted:
            continue
        stamp = row[2] if len(row) > 2 else None
        if stamp:
            when = dt.datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
            if (now - when).total_seconds() < grace_hours * 3600:
                young += 1
                continue
        else:
            young += 1           # no timestamp: never guess
            continue
        delete.append(key)
        freed += size
    return {"delete": sorted(delete), "bytes": freed, "kept_releases": sorted(kept_releases),
            "kept_blobs": len(wanted), "too_new": young}


def gc(args) -> int:
    with open(args.listing, "r", encoding="utf-8") as handle:
        listing = json.load(handle) or []
    manifests, channels = {}, {}
    for dirpath, _, names in os.walk(args.root):
        for name in names:
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, args.root).replace(os.sep, "/")
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, ValueError):
                continue
            if rel.endswith("/manifest.json") and rel.startswith("releases/"):
                manifests[rel.rsplit("/manifest.json", 1)[0]] = data
            elif rel.startswith("channels/"):
                channels[rel] = data
    result = unreferenced(listing, manifests, channels, keep=args.keep)
    # One delete-objects request per thousand keys, which is S3's (and R2's) cap.
    os.makedirs(args.out, exist_ok=True)
    keys = result["delete"]
    for n, start in enumerate(range(0, len(keys), 1000)):
        with open(os.path.join(args.out, f"delete-{n:03d}.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({"Objects": [{"Key": k} for k in keys[start:start + 1000]],
                       "Quiet": True}, handle)
    print(f"OK:{len(result['delete'])} blob(s), {result['bytes'] / 2**20:.1f} MB, "
          f"not referenced by the {len(result['kept_releases'])} kept release(s); "
          f"{result['too_new']} too new to judge")
    for prefix in result["kept_releases"]:
        print(f"  keeping {prefix}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Lay a built bundle out for the host.")
    p.add_argument("--root", required=True, help="Built bundle (with manifest.json).")
    p.add_argument("--out", default="publish", help="Folder to create.")
    p.add_argument("--allow-unsigned", action="store_true",
                   help="Stage without a signature, to be signed later.")
    p.add_argument("--github-output", dest="github_output",
                   help="Append prefix=<release prefix> here (CI).")
    p.set_defaults(func=prepare)

    p = sub.add_parser("sign", help="Sign a staged release manifest (offline key).")
    p.add_argument("--version", help="Release to sign, fetched from the host.")
    p.add_argument("--edition", help="Default: this checkout's version.__edition__.")
    p.add_argument("--platform", default="windows")
    p.add_argument("--base-url", dest="base_url",
                   help=f"Update host (default: ${BASE_URL_ENV}).")
    p.add_argument("--manifest", help="Sign this local file instead of fetching.")
    p.add_argument("--key", default=DEFAULT_KEY_PATH)
    p.add_argument("--out")
    p.set_defaults(func=sign)

    p = sub.add_parser("check", help="Verify a staged release before publishing it.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--sig", required=True)
    p.add_argument("--version", required=True)
    p.add_argument("--edition")
    p.add_argument("--base-url", dest="base_url", required=True)
    p.add_argument("--listing", required=True,
                   help="JSON [[key, size], ...] of the bucket's files/ prefix.")
    p.set_defaults(func=check)

    p = sub.add_parser("gc", help="Blobs no kept release refers to (prune-updates.yaml).")
    p.add_argument("--listing", required=True,
                   help="JSON [[key, size, last_modified], ...] of files/")
    p.add_argument("--root", required=True,
                   help="Local copy of releases/**/manifest.json and channels/*.json")
    p.add_argument("--keep", type=int, default=3,
                   help="Newest releases kept per edition and platform")
    p.add_argument("--out", required=True,
                   help="Folder for the delete-objects requests (1000 keys each)")
    p.set_defaults(func=gc)

    p = sub.add_parser("prefix", help="Print where a release's manifest lives.")
    p.add_argument("--version", required=True)
    p.add_argument("--edition", required=True)
    p.add_argument("--platform", default="windows")
    p.set_defaults(func=lambda a: print(release_prefix(
        {"version": a.version, "edition": a.edition, "platform": a.platform})) or 0)

    p = sub.add_parser("channel", help="Write the channel file for a release.")
    p.add_argument("--manifest", required=True)
    p.add_argument("--previous", help="The channel file currently published.")
    p.add_argument("--notes")
    p.add_argument("--notes-url", dest="notes_url")
    p.add_argument("--download-url", dest="download_url")
    p.add_argument("--out", required=True)
    p.set_defaults(func=channel)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    from modules.system.debug_console import force_utf8_stdio
    force_utf8_stdio()
    raise SystemExit(main())
