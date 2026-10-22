"""Copy published GitHub releases onto the update host, as a download mirror.

GitHub's release downloads are throttled or cut off in some places, mainland
China above all, where a Cloudflare R2 bucket on a custom domain is usually
reachable. So each published release of the free edition, the app's and the
packs', is copied to the bucket the in-place updates already use
(docs/UPDATES.md), under::

    downloads/<tag>/<asset>          every asset of the release, as GitHub has it
    downloads/<tag>/index.json       {"tag", "assets": [{"name", "bytes", "sha256"}]}
    downloads/latest/<edition>.json  that index, for the newest app release
    downloads/latest/index.html      a plain page linking the newest release's files
    downloads/latest/00-*            the fixed-name installers, so a link never moves
    channels/<edition>.json          the release, announced for download, unless the
                                     channel already names it or something newer

Anything that downloads ``github.com/<repo>/releases/download/<tag>/<asset>``
finds the same bytes at ``<host>/downloads/<tag>/<asset>``, and tries that
first: the packs lock (tools/write_packs_lock.py), Setup's component downloads
(packaging/installer/videohighlighter-packs.iss) and the bootstrap zip's config
(tools/build_bootstrap_zip.py). GitHub stays the fallback. Nothing is trusted
for having come from the mirror: packs and Setup's components are pinned by
SHA-256 at build time, whichever host served them.

Only the free edition is mirrored. Its releases are public already; a Pro
release copied here would be public too.

mirror-release.yaml does the copying, and prune-updates.yaml the clearing out;
what to write and what to delete is decided here, where it is tested.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOWNLOADS = "downloads"
LATEST = "latest"
# Fixed names, no version in them: copied to downloads/latest/ as well.
FIXED_PREFIX = "00-"


def mirror_base(host: str, tag: str) -> str:
    """Where ``tag``'s assets are on the mirror, or ``""`` with no host."""
    host = (host or "").strip().rstrip("/")
    return f"{host}/{DOWNLOADS}/{tag}" if host else ""


def is_app_tag(tag: str) -> bool:
    """An app release, rather than a packs one (packs-torch-...)."""
    return not str(tag).startswith("packs-")


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------

def build_index(tag: str, rows) -> dict:
    """The index of a mirrored release, from ``(name, bytes, sha256)`` rows."""
    assets = []
    for name, size, sha in rows:
        sha = str(sha).strip().lower()
        if len(sha) != 64:
            raise ValueError(f"{name}: {sha!r} is not a SHA-256")
        assets.append({"name": str(name), "bytes": int(size), "sha256": sha})
    if not assets:
        raise ValueError(f"{tag} has no assets")
    return {"tag": str(tag), "assets": sorted(assets, key=lambda a: a["name"])}


def fixed_names(index: dict) -> list:
    return [a["name"] for a in index["assets"] if a["name"].startswith(FIXED_PREFIX)]


# ---------------------------------------------------------------------------
# the page
# ---------------------------------------------------------------------------

def latest_page(index: dict, *, notes_url: str = "") -> str:
    """A plain page for downloads/latest/: every file of the newest release.

    Links are relative (``../<tag>/<asset>``), so the page works on whatever
    domain the bucket is served from. The fixed-name installers link to their
    copy beside the page, which is the address worth putting on a website.
    """
    tag = html.escape(index["tag"])
    rows = []
    for asset in index["assets"]:
        name = asset["name"]
        href = name if name.startswith(FIXED_PREFIX) else f"../{index['tag']}/{name}"
        mb = asset["bytes"] / 2**20
        rows.append(
            f'<tr><td><a href="{html.escape(href, quote=True)}">{html.escape(name)}</a></td>'
            f'<td class="n">{mb:,.0f} MB</td><td><code>{asset["sha256"]}</code></td></tr>')
    notes = (f'<p><a href="{html.escape(notes_url, quote=True)}">What is new in {tag}</a></p>'
             if notes_url else "")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VideoHighlighter {tag}</title>
<style>
body {{ font: 15px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 60rem; padding: 0 1rem; }}
table {{ border-collapse: collapse; width: 100%; }}
td {{ border-top: 1px solid #ccc; padding: .4rem .6rem .4rem 0; vertical-align: top; }}
td.n {{ text-align: right; white-space: nowrap; }}
code {{ font-size: 12px; word-break: break-all; }}
</style>
</head>
<body>
<h1>VideoHighlighter {tag}</h1>
<p>A mirror of the release on GitHub. Windows: run
<strong>00-VideoHighlighter-Windows-Setup.exe</strong>. It downloads GPU
acceleration and the search models from this mirror too.</p>
{notes}<table>
{chr(10).join(rows)}
</table>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# the channel
# ---------------------------------------------------------------------------

def download_channel(index: dict, previous: dict, *, notes_url: str = "",
                     download_url: str = "", today: str = "") -> dict | None:
    """The channel file announcing ``index``'s release for download, or None
    when the channel should be left as it is.

    Left alone when it already names this release or a newer one. For this
    release that is publish-update.yaml's signed entry, which must not lose
    its manifests to a download-only one; for a newer one, moving the channel
    back would tell everyone on it that nothing is newer.
    """
    from modules.update.update_check import is_newer

    version = str(index["tag"])
    previous = previous if isinstance(previous, dict) else {}
    if previous.get("version") and not is_newer(version, str(previous["version"])):
        return None
    return {
        "version": version,
        "date": today or _dt.date.today().isoformat(),
        "notes": "",
        "notes_url": notes_url,
        "download_url": download_url,
        # No manifests: a release is offered in place only once it is signed
        # (publish-update.yaml), and an older one's would install the old build.
        "manifests": {},
    }


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------

def tags_to_delete(tags, keep: int, latest_tags=()) -> list:
    """Which ``downloads/<tag>/`` folders can go.

    Kept: the newest ``keep`` app releases, whatever ``downloads/latest/``
    names, and every packs release. A packs tag is fetched by every build whose
    lock names it, for as long as that build is installed anywhere; they are a
    few a year.
    """
    from modules.update.update_check import parse_version

    app = sorted({t for t in tags if t and t != LATEST and is_app_tag(t)},
                 key=parse_version, reverse=True)
    kept = set(app[:max(1, int(keep))]) | set(latest_tags)
    return sorted((t for t in app if t not in kept), key=parse_version)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read_json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8-sig") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _index(args) -> int:
    rows = []
    with open(args.hashed, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(line.rstrip("\n").split("\t"))
    index = build_index(args.tag, rows)
    _write(args.out, json.dumps(index, indent=2) + "\n")
    print(f"OK:{args.tag}: {len(index['assets'])} asset(s), "
          f"{sum(a['bytes'] for a in index['assets']) / 2**30:.2f} GB")
    for name in fixed_names(index):
        print(f"fixed:{name}")
    return 0


def _page(args) -> int:
    _write(args.out, latest_page(_read_json(args.index), notes_url=args.notes_url or ""))
    return 0


def _channel(args) -> int:
    payload = download_channel(_read_json(args.index), _read_json(args.previous),
                               notes_url=args.notes_url or "",
                               download_url=args.download_url or "")
    if payload is None:
        print(f"SKIP:the channel already announces {_read_json(args.previous).get('version')}")
        if os.path.exists(args.out):
            os.remove(args.out)
        return 0
    _write(args.out, json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


def _prune(args) -> int:
    # `aws s3api list-objects-v2 --delimiter / --query 'CommonPrefixes[].Prefix'`
    # output: ["downloads/<tag>/", ...], or null when there is nothing.
    with open(args.tags, encoding="utf-8") as handle:
        prefixes = json.load(handle) or []
    tags = [str(p).strip("/").rsplit("/", 1)[-1] for p in prefixes if p]
    latest = []
    for name in os.listdir(args.latest) if args.latest and os.path.isdir(args.latest) else []:
        if name.endswith(".json"):
            tag = _read_json(os.path.join(args.latest, name)).get("tag")
            if tag:
                latest.append(str(tag))
    doomed = tags_to_delete(tags, args.keep, latest)
    for tag in doomed:
        print(tag)
    print(f"OK:{len(doomed)} release folder(s) to delete, keeping the newest "
          f"{args.keep} app release(s), {', '.join(latest) or 'nothing'} as latest, "
          f"and every packs release", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("index", help="Write downloads/<tag>/index.json.")
    p.add_argument("--tag", required=True)
    p.add_argument("--hashed", required=True, help="TSV: name, bytes, sha256 per asset.")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_index)

    p = sub.add_parser("page", help="Write downloads/latest/index.html.")
    p.add_argument("--index", required=True)
    p.add_argument("--notes-url", dest="notes_url")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_page)

    p = sub.add_parser("channel", help="Announce a mirrored release for download.")
    p.add_argument("--index", required=True)
    p.add_argument("--previous", required=True, help="The channel file published now.")
    p.add_argument("--notes-url", dest="notes_url")
    p.add_argument("--download-url", dest="download_url")
    p.add_argument("--out", required=True, help="Written only when the channel should change.")
    p.set_defaults(func=_channel)

    p = sub.add_parser("prune", help="Print the downloads/<tag>/ folders that can go.")
    p.add_argument("--tags", required=True, help='JSON ["downloads/<tag>/", ...] (or null).')
    p.add_argument("--latest", help="Local copy of downloads/latest/*.json.")
    p.add_argument("--keep", type=int, default=3)
    p.set_defaults(func=_prune)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    from modules.system.debug_console import force_utf8_stdio
    force_utf8_stdio()
    raise SystemExit(main())
