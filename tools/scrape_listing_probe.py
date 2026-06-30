"""
Downloader Step 1 probe — validate listing scrape (url / title / thumbnail) before
building the picker UI.

Runs downloader.extract_video_entries() on a real listing page and prints what it
found for each video, so you can eyeball whether titles + thumbnails are correct.

    python tools/scrape_listing_probe.py --url "https://example.com/videos" --pattern "/video/"
    python tools/scrape_listing_probe.py --url "..." --browser always   # force Selenium
    python tools/scrape_listing_probe.py --url "..." --browser never     # static only

Be considerate of the target site: respect its Terms of Service and robots.txt,
and only download content you're permitted to.
"""
from __future__ import annotations

import argparse
import os
import sys

# Runnable from the repo root as `python tools/scrape_listing_probe.py`: ensure the
# root (where downloader.py lives) is importable regardless of the cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from downloader import extract_video_entries, dump_listing_cards  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Probe a listing page for video entries")
    ap.add_argument("--url", required=True, help="listing/index page URL")
    ap.add_argument("--pattern", default="/video/", help="substring that marks a video link")
    ap.add_argument("--browser", default="auto", choices=["auto", "never", "always"],
                    help="auto=static then Selenium fallback; never=static only; always=Selenium")
    ap.add_argument("--limit", type=int, default=40, help="max entries to print")
    ap.add_argument("--dump", type=int, default=0,
                    help="print the raw HTML of the first N matching cards (for tuning), then exit")
    args = ap.parse_args()

    if args.dump > 0:
        dump_listing_cards(args.url, args.pattern, count=args.dump,
                           browser=(args.browser != "never"))
        return 0

    entries = extract_video_entries(args.url, pattern=args.pattern, use_browser=args.browser)

    print(f"\n=== {len(entries)} video(s) found ===")
    with_thumb = sum(1 for e in entries if e["thumbnail_url"])
    print(f"with thumbnail: {with_thumb}/{len(entries)}\n")

    for i, e in enumerate(entries[:args.limit], 1):
        thumb = e["thumbnail_url"] or "—(no thumb)—"
        print(f"[{i:>3}] {e['title'][:70]}")
        print(f"      url:   {e['url']}")
        print(f"      thumb: {thumb}")
    if len(entries) > args.limit:
        print(f"\n... and {len(entries) - args.limit} more (raise --limit to see all)")

    if not entries:
        print("No entries. Try --browser always (JS-rendered), or check --pattern "
              "matches the video links' href.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
