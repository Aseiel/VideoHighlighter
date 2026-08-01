"""Generate, sign and verify the per-file release manifest.

Why per-file
------------
The build is several GB, but a normal release changes a few MB of it — the app
code moves, the models and the PyTorch/Qt runtime do not. A manifest listing
every shipped file with its SHA-256 lets the updater download only the files
that actually differ, so a bug-fix update is megabytes instead of gigabytes.
No separate "delta build" step is needed: the hashes are the delta.

Why the signature is a separate file
------------------------------------
``manifest.json`` is signed as raw bytes and the signature ships beside it as
``manifest.json.sig``. Signing the file rather than a canonicalised re-encoding
of its contents means there is no ambiguity about *what* was signed — no key
ordering, no whitespace, no float formatting. The updater verifies the exact
bytes it downloaded.

**The signature is the whole security model.** The updater replaces executable
files on a customer's machine; whoever can change the manifest decides what
runs. A compromised host, a hijacked account or a MITM all stop at this check,
and nothing else in the chain would catch them.

Use a release key that is NOT the license signing key: they have different
blast radii and different rotation needs, and a leaked release key must not
also mint licenses.

Usage
-----
    python tools/build_manifest.py keygen --update-module
    python tools/build_manifest.py generate --root dist/VideoHighlighter \\
        --version 0.9.1 --edition Pro
    python tools/build_manifest.py sign --root dist/VideoHighlighter
    python tools/build_manifest.py verify --root dist/VideoHighlighter
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.update_manifest import (  # noqa: E402
    MANIFEST_FILENAME,
    MANIFEST_FORMAT,
    SIGNATURE_FILENAME,
    hash_file,
    verify_manifest,
)

DEFAULT_KEY_PATH = os.path.join(".secrets", "release_signing_key.pem")
_MODULE_PATH = os.path.join("modules", "update_manifest.py")

# Never listed in the manifest: the manifest cannot contain its own hash, and
# the signature covers the manifest rather than being covered by it.
_SELF = {MANIFEST_FILENAME, SIGNATURE_FILENAME}


def _walk(root: str):
    """Every shipped file, as ``(relative_posix_path, absolute_path)``.

    Paths are stored posix-style so a manifest generated on any platform reads
    identically — the updater joins them back with the local separator.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(filenames):
            absolute = os.path.join(dirpath, name)
            relative = os.path.relpath(absolute, root).replace(os.sep, "/")
            if relative in _SELF:
                continue
            yield relative, absolute


def generate(args) -> int:
    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"FAIL:not a directory: {root}")
        return 1

    files, total = [], 0
    for relative, absolute in _walk(root):
        size = os.path.getsize(absolute)
        files.append({
            "path": relative,
            "size": size,
            "sha256": hash_file(absolute),
        })
        total += size

    manifest = {
        "format": MANIFEST_FORMAT,
        "version": args.version,
        "edition": args.edition,
        "date": args.date or _dt.date.today().isoformat(),
        "notes": args.notes or "",
        # Filled in at publish time — where the individual files can be fetched.
        # Left empty here so the same build can be published anywhere.
        "base_url": args.base_url or "",
        "files": files,
    }

    out = args.out or os.path.join(root, MANIFEST_FILENAME)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=1, sort_keys=False)
        handle.write("\n")

    print(f"OK:{out}")
    print(f"  {len(files)} files, {total / (1024 ** 3):.2f} GB")
    return 0


def _load_private_key(path: str):
    from cryptography.hazmat.primitives.serialization import load_pem_private_key

    with open(path, "rb") as handle:
        return load_pem_private_key(handle.read(), password=None)


def keygen(args) -> int:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding, NoEncryption, PrivateFormat, PublicFormat,
    )

    if os.path.exists(args.out) and not args.force:
        print(f"FAIL:{args.out} exists. --force to overwrite.")
        print("  Overwriting invalidates every manifest already published:")
        print("  installed copies verify against the OLD public key.")
        return 1

    private = Ed25519PrivateKey.generate()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as handle:
        handle.write(private.private_bytes(
            Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()))

    public_hex = private.public_key().public_bytes(
        Encoding.Raw, PublicFormat.Raw).hex()

    print(f"OK:private key: {args.out}  (NEVER commit; back it up offline)")
    print(f"  public key:  {public_hex}")

    if args.update_module:
        with open(_MODULE_PATH, "r", encoding="utf-8") as handle:
            source = handle.read()
        patched, count = re.subn(
            r'RELEASE_PUBLIC_KEY_HEX\s*=\s*"[0-9a-fA-F]*"',
            f'RELEASE_PUBLIC_KEY_HEX = "{public_hex}"',
            source, count=1)
        if count != 1:
            print(f"FAIL:could not patch {_MODULE_PATH}; paste the key by hand.")
            return 1
        with open(_MODULE_PATH, "w", encoding="utf-8") as handle:
            handle.write(patched)
        print(f"OK:embedded in {_MODULE_PATH}")
    return 0


def sign(args) -> int:
    manifest_path = args.manifest or os.path.join(args.root, MANIFEST_FILENAME)
    with open(manifest_path, "rb") as handle:
        raw = handle.read()

    private = _load_private_key(args.key)
    signature = private.sign(raw)
    encoded = base64.urlsafe_b64encode(signature).decode("ascii").rstrip("=")

    out = args.out or manifest_path + ".sig"
    with open(out, "w", encoding="ascii") as handle:
        handle.write(encoded + "\n")

    print(f"OK:signed {os.path.basename(manifest_path)} ({len(raw)} bytes)")
    print(f"  {out}")
    return 0


def verify(args) -> int:
    manifest_path = args.manifest or os.path.join(args.root, MANIFEST_FILENAME)
    signature_path = args.sig or manifest_path + ".sig"
    with open(manifest_path, "rb") as handle:
        raw = handle.read()
    with open(signature_path, "r", encoding="ascii") as handle:
        signature = handle.read().strip()

    manifest = verify_manifest(raw, signature)
    if manifest is None:
        print("FAIL:SIGNATURE INVALID — this manifest would be rejected by the app.")
        return 1
    print(f"OK:signature valid: {manifest['version']} {manifest.get('edition', '')}, "
          f"{len(manifest['files'])} files")

    if args.check_files:
        root = os.path.abspath(args.root)
        missing = wrong = 0
        for entry in manifest["files"]:
            absolute = os.path.join(root, entry["path"].replace("/", os.sep))
            if not os.path.exists(absolute):
                print(f"  FAIL:missing: {entry['path']}")
                missing += 1
            elif hash_file(absolute) != entry["sha256"]:
                print(f"  FAIL:changed: {entry['path']}")
                wrong += 1
        if missing or wrong:
            print(f"FAIL:{missing} missing, {wrong} modified")
            return 1
        print("OK:every file on disk matches the manifest")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="Hash a built bundle into a manifest.")
    p_gen.add_argument("--root", required=True, help="Built bundle directory.")
    p_gen.add_argument("--version", required=True)
    p_gen.add_argument("--edition", default="Pro")
    p_gen.add_argument("--date", help="Release date (default: today).")
    p_gen.add_argument("--notes", help="One line shown in the update banner.")
    p_gen.add_argument("--base-url", dest="base_url",
                       help="Where the individual files will be served from.")
    p_gen.add_argument("--out")
    p_gen.set_defaults(func=generate)

    p_key = sub.add_parser("keygen", help="Create the release signing keypair.")
    p_key.add_argument("--out", default=DEFAULT_KEY_PATH)
    p_key.add_argument("--update-module", action="store_true",
                       help=f"Embed the public key in {_MODULE_PATH}.")
    p_key.add_argument("--force", action="store_true")
    p_key.set_defaults(func=keygen)

    p_sign = sub.add_parser("sign", help="Sign a manifest.")
    p_sign.add_argument("--root", default=".")
    p_sign.add_argument("--manifest")
    p_sign.add_argument("--key", default=DEFAULT_KEY_PATH)
    p_sign.add_argument("--out")
    p_sign.set_defaults(func=sign)

    p_ver = sub.add_parser("verify", help="Verify a manifest against the embedded key.")
    p_ver.add_argument("--root", default=".")
    p_ver.add_argument("--manifest")
    p_ver.add_argument("--sig")
    p_ver.add_argument("--check-files", action="store_true",
                       help="Also re-hash every file on disk.")
    p_ver.set_defaults(func=verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
