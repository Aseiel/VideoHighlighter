"""Turn a built bundle into the folder you upload to the update host.

    python tools/publish_release.py prepare --root dist/VideoHighlighter --out publish

Produces a content-addressed tree::

    publish/manifest.json          <- signed release description
    publish/manifest.json.sig
    publish/files/<sha256>         <- one blob per distinct file

Why content-addressed
---------------------
Blobs are named by their hash, not their path, so uploading a release is a
*sync*: anything already on the host is skipped. The multi-GB models are
uploaded once and never again; each later release pushes only genuinely new
bytes. It also means an install can jump several versions at once, because
every blob any manifest ever referenced is still there under the same name.

Hardlinks are used where the filesystem allows it, so preparing a 4 GB bundle
does not cost another 4 GB of disk.

Uploading
---------
Any S3-compatible tool works; nothing here needs credentials, so none live in
this repo. With rclone configured for the bucket::

    rclone sync publish/files r2:vh-updates/files --checksum
    rclone copy publish/manifest.json publish/manifest.json.sig r2:vh-updates

Upload the blobs FIRST and the manifest LAST. The manifest is what tells
customers a release exists; publishing it before its files are in place points
everyone at downloads that 404.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.update_manifest import (  # noqa: E402
    MANIFEST_FILENAME,
    SIGNATURE_FILENAME,
    local_path,
)


def prepare(args) -> int:
    root = os.path.abspath(args.root)
    out = os.path.abspath(args.out)
    manifest_path = os.path.join(root, MANIFEST_FILENAME)

    if not os.path.exists(manifest_path):
        print(f"FAIL:no manifest in {root}")
        print("  run: python tools/build_manifest.py generate --root <bundle> ...")
        return 1

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if not manifest.get("base_url"):
        print("FAIL:manifest has no base_url — the updater would not know where")
        print("  to fetch files. Re-generate with --base-url <public URL>, then")
        print("  sign it (the signature covers base_url).")
        return 1

    signature_path = os.path.join(root, SIGNATURE_FILENAME)
    if not os.path.exists(signature_path) and not args.allow_unsigned:
        print(f"FAIL:{SIGNATURE_FILENAME} missing — an unsigned release is")
        print("  rejected by every installed copy. Sign it first:")
        print("    python tools/build_manifest.py sign --root " + args.root)
        return 1

    blobs = os.path.join(out, "files")
    os.makedirs(blobs, exist_ok=True)

    linked = copied = skipped = 0
    new_bytes = 0
    seen = set()

    for entry in manifest["files"]:
        digest = entry["sha256"]
        if digest in seen:
            continue          # same content twice in the bundle: one blob
        seen.add(digest)

        destination = os.path.join(blobs, digest)
        if os.path.exists(destination):
            skipped += 1
            continue

        source = local_path(root, entry["path"])
        try:
            os.link(source, destination)
            linked += 1
        except OSError:
            # Different volume, or a filesystem without hardlinks.
            shutil.copy2(source, destination)
            copied += 1
        new_bytes += int(entry.get("size", 0))

    shutil.copy2(manifest_path, os.path.join(out, MANIFEST_FILENAME))
    if os.path.exists(signature_path):
        shutil.copy2(signature_path, os.path.join(out, SIGNATURE_FILENAME))

    total = sum(int(e.get("size", 0)) for e in manifest["files"])
    print(f"OK:{out}")
    print(f"  version      {manifest.get('version')} {manifest.get('edition', '')}")
    print(f"  base_url     {manifest['base_url']}")
    print(f"  blobs        {len(seen)} distinct ({linked} linked, {copied} copied, "
          f"{skipped} already prepared)")
    print(f"  new bytes    {new_bytes / (1024 ** 2):.1f} MB of "
          f"{total / (1024 ** 2):.1f} MB total")
    print()
    print("  upload blobs first, manifest last:")
    print(f"    rclone sync {args.out}/files r2:<bucket>/files --checksum")
    print(f"    rclone copy {args.out}/{MANIFEST_FILENAME} r2:<bucket>")
    print(f"    rclone copy {args.out}/{SIGNATURE_FILENAME} r2:<bucket>")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Build the upload folder.")
    p.add_argument("--root", required=True, help="Built bundle (with manifest.json).")
    p.add_argument("--out", default="publish", help="Folder to create.")
    p.add_argument("--allow-unsigned", action="store_true",
                   help="Prepare without a signature (testing only).")
    p.set_defaults(func=prepare)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
