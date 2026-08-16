"""Build VideoHighlighter-Windows-Setup.zip for GitHub Release attachment.

The zip is tiny (~10 KB): double-click Install-VideoHighlighter.bat on Windows,
and the script downloads both split 7z volumes plus extracts them.

The tag defaults to version.py, so bumping the app bumps what the installer
asks for. It used to be typed in by hand in two places -- the CI argument and
the committed config.json -- and the committed copy simply went stale: it still
named 0.9.0 several releases later, which is the version the installer falls
back to whenever the GitHub API cannot be reached.

Usage::

    python tools/build_bootstrap_zip.py --edition free
    python tools/build_bootstrap_zip.py --edition pro --tag 0.9.0-Pro
    python tools/build_bootstrap_zip.py --edition free --write-config
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP = ROOT / "packaging" / "bootstrap"
ZIP_MEMBERS = (
    "Install-VideoHighlighter.bat",
    "Install-VideoHighlighter.ps1",
    "config.json",
)

FREE_REPO = "Aseiel/VideoHighlighter"
PRO_REPO = "Aseiel/VideoHighlighter-pro"

# The committed configs, one per edition. These are what someone gets when they
# run the installer straight from a checkout, and what the offline fallback in a
# shipped zip is copied from -- so they have to track version.py rather than be
# remembered.
CONFIG_PATHS = {
    "free": BOOTSTRAP / "config.json",
    "pro": BOOTSTRAP / "config.pro.example.json",
}


def default_tag(edition: str) -> str:
    """The tag this checkout would release under, per version.py.

    Mirrors the slug the release workflow computes: the Free tag is the bare
    version, Pro appends its edition. Deriving it here means the installer and
    the app can no longer disagree about which release is current.
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import version

    return (f"{version.__version__}-Pro" if edition.lower() == "pro"
            else version.__version__)


def _windows_assets(tag: str, *, pro: bool) -> tuple[str, ...]:
    if pro:
        return (
            f"VideoHighlighter-Windows-{tag}.7z.001",
            f"VideoHighlighter-Windows-{tag}.7z.002",
        )
    return (
        f"VideoHighlighter-Windows-{tag}.7z.001",
        f"VideoHighlighter-Windows-{tag}.7z.002",
    )


def make_config(*, edition: str, tag: str) -> dict:
    pro = edition.lower() == "pro"
    repo = PRO_REPO if pro else FREE_REPO
    assets = list(_windows_assets(tag, pro=pro))
    return {
        "product_name": "VideoHighlighter",
        "edition": "Pro" if pro else "Free",
        "repo": repo,
        "use_latest": not pro,
        "asset_pattern": r"^VideoHighlighter-Windows-.*\.7z\.\d{3}$",
        "tag": tag,
        "assets": assets,
        "base_url": f"https://github.com/{repo}/releases/download/{tag}",
        "notes": (
            "Pro: private repo — anonymous download URLs need auth; "
            "customers install from Lemon Squeezy my-orders (single .7z). "
            "This zip is for maintainers with local volumes or GitHub access."
            if pro
            else "use_latest asks the GitHub API for the current release."
        ),
    }


def write_config(*, edition: str, tag: str) -> Path:
    """Rewrite the committed config for `edition` at the given tag.

    Only the version-bearing fields are regenerated. The hand-written `notes`
    is kept as it is -- the Pro one carries the command for running the
    installer against that config, which no generator knows about.
    """
    path = CONFIG_PATHS[edition.lower()]
    if not path.exists():
        raise SystemExit(f"no committed config for {edition}: {path}")

    config = make_config(edition=edition, tag=tag)
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        existing = {}
    if existing.get("notes"):
        config["notes"] = existing["notes"]

    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"OK {path} (tag={tag})")
    return path


def build_zip(*, edition: str, tag: str, out: Path) -> Path:
    if not BOOTSTRAP.is_dir():
        raise SystemExit(f"bootstrap folder missing: {BOOTSTRAP}")

    missing = [name for name in ZIP_MEMBERS[:2] if not (BOOTSTRAP / name).exists()]
    if missing:
        raise SystemExit(f"missing bootstrap files: {', '.join(missing)}")

    config = make_config(edition=edition, tag=tag)
    staging = out.parent / f".bootstrap-staging-{out.stem}"
    staging.mkdir(parents=True, exist_ok=True)
    try:
        for name in ZIP_MEMBERS[:2]:
            (staging / name).write_bytes((BOOTSTRAP / name).read_bytes())
        (staging / "config.json").write_text(
            json.dumps(config, indent=2) + "\n",
            encoding="utf-8",
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.unlink()
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(staging.iterdir()):
                zf.write(path, arcname=path.name)
    finally:
        for child in staging.iterdir():
            child.unlink(missing_ok=True)
        staging.rmdir()

    size_kb = out.stat().st_size / 1024
    print(f"OK {out} ({size_kb:.1f} KB)")
    print(f"   edition={edition} tag={tag} repo={config['repo']}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--edition",
        choices=("free", "pro"),
        required=True,
        help="Free (public GitHub) or Pro (pinned tag; LS for customers).",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Release tag, e.g. 0.9.0 or 0.9.0-Pro. Defaults to version.py.",
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Rewrite the committed bootstrap config for this edition instead "
             "of building the zip. Run this after bumping version.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "packaging" / "bootstrap" / "VideoHighlighter-Windows-Setup.zip",
        help="Output zip path.",
    )
    args = parser.parse_args(argv)
    tag = args.tag or default_tag(args.edition)
    if args.write_config:
        write_config(edition=args.edition, tag=tag)
    else:
        build_zip(edition=args.edition, tag=tag, out=args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
