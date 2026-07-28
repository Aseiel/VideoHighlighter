#!/usr/bin/env python3
"""
tools/probe_player.py — dev-only player diagnostic + in-session downloader.

Two failure modes the normal downloader cannot get past, both seen in the wild:

1. A pre-roll holds the <video> element, and the real stream is never requested
   while a headless sniffer is watching. Seeking the ad to its end does not
   always help — some players validate it and restart.

2. The CDN refuses any request that did not come from the live browser session.
   Extracting the URL and handing it to yt-dlp then 403s even with Referer,
   User-Agent and Cookie replayed, because what is being checked is the session
   itself (TLS/connection/token), not the headers.

So this script does two things the app cannot:

  * REPORT — logs every media response with its timing and status, and polls
    each <video>.currentSrc so a post-advert source switch is visible even when
    the network log is ambiguous. Facts first, before changing any app code.

  * FETCH THROUGH THE SESSION — downloads via Playwright's APIRequestContext,
    which reuses the page's cookies and connection. That is the only mechanism
    that works against a session-locked CDN.

Dev-only: not imported by the app and not part of the shipped bundle, so its
dependency on Playwright never reaches a customer build.

    python tools/probe_player.py URL                  # observe only
    python tools/probe_player.py URL --headed         # watch it happen
    python tools/probe_player.py URL --wait 120       # let a long ad finish
    python tools/probe_player.py URL --download 1 -o D:/movies
"""

import argparse
import os
import re
import sys
import time
import urllib.parse
from typing import Dict, List, Optional

MEDIA_EXT_RE = re.compile(r'\.(m3u8|mpd|mp4|m4v|webm|mov|mkv|flv|ts|m4s)(?:$|\?)',
                          re.IGNORECASE)
MEDIA_CT_RE = re.compile(r'(video/|audio/|mpegurl|dash\+xml|octet-stream)',
                         re.IGNORECASE)
# Segments are derivable from their manifest; listing them all is just noise.
SEGMENT_RE = re.compile(r'\.(ts|m4s)(?:$|\?)', re.IGNORECASE)


def human(n: Optional[int]) -> str:
    if not n:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def observe(url: str, wait_s: int, headed: bool, download_idx: Optional[int],
            out_dir: str):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright is not installed:  pip install playwright"
              "  &&  playwright install chromium", file=sys.stderr)
        return 2

    hits: Dict[str, Dict] = {}       # url -> {t, status, ctype, size, kind}
    t0 = time.time()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=not headed,
            args=["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage",
                  "--autoplay-policy=no-user-gesture-required",
                  "--disable-features=IsolateOrigins,site-per-process"],
        )
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1366, "height": 768},
            ignore_https_errors=True,
        )
        page = context.new_page()

        # Every click on an ad-backed player opens a popup tab. Close them as
        # they appear, or a long watch buries the desktop in advert windows.
        def on_popup(p):
            try:
                p.close()
            except Exception:
                pass

        context.on("page", on_popup)

        def on_response(resp):
            try:
                u = resp.url
                ct = (resp.headers or {}).get("content-type", "")
                if not (MEDIA_EXT_RE.search(u) or MEDIA_CT_RE.search(ct)):
                    return
                if SEGMENT_RE.search(u):
                    # Count segments, don't list them: their presence proves a
                    # stream is actually playing, which is the useful signal.
                    key = "segments::" + urllib.parse.urlparse(u).netloc
                    e = hits.setdefault(key, {"t": time.time() - t0, "status": 0,
                                              "ctype": "segments", "size": 0,
                                              "kind": "segment", "count": 0})
                    e["count"] += 1
                    e["size"] += int((resp.headers or {}).get("content-length") or 0)
                    return
                if u in hits:
                    return
                size = int((resp.headers or {}).get("content-length") or 0)
                hits[u] = {"t": time.time() - t0, "status": resp.status,
                           "ctype": ct.split(";")[0], "size": size,
                           "kind": "manifest" if ".m3u8" in u.lower()
                                   or ".mpd" in u.lower() else "media"}
                print(f"  [{hits[u]['t']:5.1f}s] {resp.status} "
                      f"{human(size):>7} {ct.split(';')[0]:<28} {u[:96]}")
            except Exception:
                pass

        page.on("response", on_response)

        print(f"\n▶ Loading {url}")
        try:
            page.goto(url, timeout=60000, wait_until="domcontentloaded")
        except Exception as e:
            print(f"  page.goto: {str(e)[:120]}")

        print(f"\n▶ Watching for {wait_s}s "
              f"(letting any pre-roll play out — no seeking)\n")

        last_state = {}
        clicked = False
        deadline = time.time() + wait_s
        while time.time() < deadline:
            try:
                # Nudge playback without seeking: seeking is what some ad players
                # detect and punish by restarting the advert.
                for fr in page.frames:
                    try:
                        fr.evaluate(
                            "() => { document.querySelectorAll('video').forEach(v => {"
                            " try { v.muted = true; const p = v.play();"
                            " if (p && p.catch) p.catch(() => {}); } catch(e){} }); }")
                    except Exception:
                        pass
                if not clicked:
                    # ONCE only — each further click opens another advert tab.
                    clicked = True
                    try:
                        page.mouse.click(683, 384)
                    except Exception:
                        pass

                # Poll each <video> so a source switch after the advert is visible
                # even when the network log is ambiguous.
                for fr in page.frames:
                    try:
                        states = fr.evaluate("""
                            () => Array.from(document.querySelectorAll('video')).map(v => ({
                                src: v.currentSrc || v.src || '',
                                dur: isFinite(v.duration) ? Math.round(v.duration) : -1,
                                now: Math.round(v.currentTime),
                                paused: v.paused
                            }))
                        """) or []
                    except Exception:
                        continue
                    for i, s in enumerate(states):
                        key = f"{fr.url[:40]}#{i}"
                        sig = (s["src"], s["dur"])
                        if s["src"] and last_state.get(key) != sig:
                            last_state[key] = sig
                            el = time.time() - t0
                            print(f"  [{el:5.1f}s] <video> src -> dur={s['dur']}s "
                                  f"{s['src'][:92]}")
                page.wait_for_timeout(2000)
            except Exception as e:
                # A popup closing or an ad frame detaching must not end the run.
                print(f"  (watch loop: {type(e).__name__}: {str(e)[:80]})")
                time.sleep(1)

        # ---- report -----------------------------------------------------
        print("\n" + "=" * 78)
        segs = {k: v for k, v in hits.items() if v["kind"] == "segment"}
        cands = [(u, v) for u, v in hits.items() if v["kind"] != "segment"]
        cands.sort(key=lambda kv: (kv[1]["kind"] != "manifest", kv[1]["t"]))

        if segs:
            print("Stream segments actually fetched (proves playback happened):")
            for k, v in segs.items():
                print(f"   {v['count']:4d} segments, {human(v['size'])} from "
                      f"{k.split('::')[1]}")
            print()

        if not cands:
            print("No media URLs seen at all. The player never loaded a stream —\n"
                  "try --headed to watch, or a longer --wait.")
        else:
            print(f"{len(cands)} media candidate(s):\n")
            for i, (u, v) in enumerate(cands, 1):
                print(f"  [{i}] {v['kind']:<8} at {v['t']:5.1f}s  status={v['status']}"
                      f"  {human(v['size']):>7}  {v['ctype']}")
                print(f"      {u[:110]}")

            # The decisive test: can the session fetch it when yt-dlp cannot?
            print("\nIn-session reachability (Playwright APIRequestContext):")
            for i, (u, v) in enumerate(cands, 1):
                try:
                    r = context.request.get(u, headers={"Range": "bytes=0-0"},
                                            timeout=20000)
                    note = "OK" if r.status < 400 else "BLOCKED"
                    print(f"  [{i}] {r.status} {note}"
                          f"  ({r.headers.get('content-range') or r.headers.get('content-length') or '?'})")
                except Exception as e:
                    print(f"  [{i}] error: {str(e)[:90]}")

        # ---- optional download through the session ----------------------
        if download_idx is not None:
            if not cands or download_idx < 1 or download_idx > len(cands):
                print(f"\n--download {download_idx}: no such candidate.")
            else:
                target = cands[download_idx - 1][0]
                os.makedirs(out_dir, exist_ok=True)
                print(f"\n▶ Downloading candidate {download_idx} through the "
                      f"browser session...")
                ok = session_download(context, target, out_dir)
                print("✅ done" if ok else "❌ failed")

        context.close()
        browser.close()
    return 0


def session_download(context, url: str, out_dir: str) -> bool:
    """Fetch through the browser's own session (cookies + connection reused)."""
    base = re.sub(r'[\\/:*?"<>|]', "_", urllib.parse.urlparse(url).path.split("/")[-1]
                  or "video")[:80]
    low = url.lower()

    if ".m3u8" in low:
        try:
            r = context.request.get(url, timeout=30000)
            if r.status >= 400:
                print(f"   playlist HTTP {r.status}")
                return False
            text = r.text()
        except Exception as e:
            print(f"   playlist error: {str(e)[:110]}")
            return False

        if "#EXT-X-STREAM-INF" in text:
            best, best_bw = None, -1
            lines = text.splitlines()
            for i, line in enumerate(lines):
                if line.startswith("#EXT-X-STREAM-INF"):
                    m = re.search(r"BANDWIDTH=(\d+)", line)
                    bw = int(m.group(1)) if m else 0
                    for nxt in lines[i + 1:]:
                        if nxt.strip() and not nxt.startswith("#"):
                            if bw > best_bw:
                                best, best_bw = urllib.parse.urljoin(url, nxt.strip()), bw
                            break
            if not best:
                print("   master playlist had no variants")
                return False
            print(f"   master playlist -> variant at {best_bw / 1000:.0f} kbps")
            return session_download(context, best, out_dir)

        if "#EXT-X-ENDLIST" not in text:
            print("   this is a LIVE playlist (no #EXT-X-ENDLIST) — refusing, "
                  "it would never finish")
            return False
        if "#EXT-X-KEY" in text and "METHOD=NONE" not in text:
            print("   playlist is AES-encrypted; decryption is not implemented "
                   "here — use the cookies with yt-dlp instead")
            return False

        segs = [urllib.parse.urljoin(url, l.strip())
                for l in text.splitlines() if l.strip() and not l.startswith("#")]
        if not segs:
            print("   playlist listed no segments")
            return False
        out = os.path.join(out_dir, base.replace(".m3u8", "") + ".ts")
        total = 0
        print(f"   {len(segs)} segments -> {out}")
        with open(out, "wb") as f:
            for i, s in enumerate(segs, 1):
                try:
                    rr = context.request.get(s, timeout=30000)
                    if rr.status >= 400:
                        print(f"\n   segment {i} HTTP {rr.status} — stopping")
                        break
                    body = rr.body()
                    f.write(body)
                    total += len(body)
                except Exception as e:
                    print(f"\n   segment {i} error: {str(e)[:80]}")
                    break
                if i % 20 == 0 or i == len(segs):
                    print(f"\r   {i}/{len(segs)} segments, {human(total)}",
                          end="", flush=True)
        print()
        if total == 0:
            return False
        print(f"   wrote {human(total)}. Remux to mp4 with:\n"
              f'   ffmpeg -i "{out}" -c copy "{out[:-3]}.mp4"')
        return True

    # Progressive file
    out = os.path.join(out_dir, base if "." in base else base + ".mp4")
    try:
        r = context.request.get(url, timeout=120000)
        if r.status >= 400:
            print(f"   HTTP {r.status}")
            return False
        body = r.body()
        with open(out, "wb") as f:
            f.write(body)
        print(f"   wrote {human(len(body))} -> {out}")
        return len(body) > 0
    except Exception as e:
        print(f"   error: {str(e)[:110]}")
        return False


def main():
    ap = argparse.ArgumentParser(
        description="Diagnose a video page's player and optionally download "
                    "through the browser session.")
    ap.add_argument("url", help="Watch-page URL")
    ap.add_argument("--wait", type=int, default=75,
                    help="Seconds to observe (default 75; raise for long ads)")
    ap.add_argument("--headed", action="store_true",
                    help="Show the browser window instead of running headless")
    ap.add_argument("--download", type=int, metavar="N",
                    help="Download candidate N through the session")
    ap.add_argument("-o", "--out", default="downloads",
                    help="Output directory for --download")
    args = ap.parse_args()
    sys.exit(observe(args.url, args.wait, args.headed, args.download, args.out))


if __name__ == "__main__":
    main()
