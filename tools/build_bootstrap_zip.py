"""Build VideoHighlighter-Windows-Setup.zip for GitHub Release attachment.

The zip is tiny (~10 KB): double-click Install-VideoHighlighter.bat on Windows,
and the script downloads both split 7z volumes plus extracts them.

Usage::

    python tools/build_bootstrap_zip.py --edition free --tag 0.9.0
    python tools/build_bootstrap_zip.py --edition pro --tag 0.9.0-Pro
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
        required=True,
        help="Release tag, e.g. 0.9.0 or 0.9.0-Pro.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "packaging" / "bootstrap" / "VideoHighlighter-Windows-Setup.zip",
        help="Output zip path.",
    )
    args = parser.parse_args(argv)
    build_zip(edition=args.edition, tag=args.tag, out=args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
