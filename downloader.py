"""
Video Downloader Module for fetching videos from websites.
- JS duration extraction
- Made Selenium optional and added error handling; suggest alternatives if it fails
- Added fallback to search for duration in URL query params (rare but sometimes present)
- Improved candidate filtering: use mode or cluster to pick likely real duration if many
- Removed redundant Firefox fallback (focus on Chrome)
- Added simple clustering to group similar durations and pick from the largest group
Note: If Selenium still doesn't work, consider using Playwright as alternative (not implemented here).
"""

import os
import re
import subprocess
import json
import tempfile
import requests
from bs4 import BeautifulSoup
from typing import List, Optional, Callable, Tuple, Any, Dict
from pathlib import Path
import urllib.parse
import time
from collections import Counter
import math
import hashlib


from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service

class DownloadError(Exception):
    """Custom exception for download errors"""
    pass

# -----------------------------
# Helper function: extract domain
# -----------------------------
def extract_domain(u: str) -> str:
    """Extract domain from URL for caching purposes"""
    try:
        p = urllib.parse.urlparse(u)
        return f"{p.scheme}://{p.netloc}"
    except Exception:
        return "https://example.com"

# -----------------------------
# Duration parsing helpers
# -----------------------------
def get_duration_from_ffprobe(filepath: str, log_fn: Callable = print) -> Optional[float]:
    """
    Get real duration in seconds by probing the downloaded file.
    Requires ffprobe (part of ffmpeg) installed and available on PATH.
    Returns:
        duration seconds, or None if unavailable/fails.
    """
    try:
        if not filepath or not os.path.exists(filepath):
            return None
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            filepath
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=False)
        out = (p.stdout or "").strip()
        if not out:
            # Sometimes ffprobe writes useful hints to stderr
            return None
        d = float(out)
        if d > 0:
            return d
    except FileNotFoundError:
        log_fn("⚠️ ffprobe not found. Install ffmpeg (ffprobe) and ensure it's on PATH.")
    except Exception as e:
        log_fn(f"⚠️ ffprobe failed: {str(e)[:120]}...")
    return None

def parse_iso8601_duration_enhanced(duration_str: str) -> Optional[float]:
    """
    Enhanced ISO 8601 duration parser.
    Handles:
      - PT5M30S
      - PT5M30.5S
      - P1DT5H30M
      - P0DT0H0M0.5S
    """
    if not duration_str or not isinstance(duration_str, str):
        return None
    s = duration_str.strip().upper()
    # ISO8601 duration (days optional, time section optional)
    # PnDTnHnMnS or PTnHnMnS
    pattern = r'^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?$'
    m = re.match(pattern, s)
    if not m:
        # Also accept "PT..." without leading "P" mistakes? (rare)
        if s.startswith("PT"):
            m = re.match(r'^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?$', s)
            if m:
                h = int(m.group(1) or 0)
                mi = int(m.group(2) or 0)
                sec = float(m.group(3) or 0)
                total = h * 3600 + mi * 60 + sec
                return total if total > 0 else None
        return None
    days = int(m.group(1) or 0)
    hours = int(m.group(2) or 0)
    minutes = int(m.group(3) or 0)
    seconds = float(m.group(4) or 0)
    total_seconds = days * 86400 + hours * 3600 + minutes * 60 + seconds
    return total_seconds if total_seconds > 0 else None

def parse_duration_text(value: str) -> Optional[float]:
    """
    Parse duration from common representations:
    - ISO8601: PT5M30S
    - "HH:MM:SS"
    - "MM:SS"
    - "1h 2m 3s", "2m 10s"
    - plain number (seconds)
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # ISO first
    iso = parse_iso8601_duration_enhanced(s)
    if iso:
        return iso
    # plain seconds float
    try:
        num = float(s)
        # sanity bound: < 24h
        if 0 < num <= 86400:
            return num
        # sometimes ms (common in some metadata)
        if num > 86400:
            ms = num / 1000.0
            if 0 < ms <= 86400:
                return ms
    except ValueError:
        pass
    # HH:MM:SS
    m = re.search(r'(\d+):(\d+):(\d+)', s)
    if m:
        try:
            h, mi, sec = map(int, m.groups())
            total = h * 3600 + mi * 60 + sec
            return float(total) if total > 0 else None
        except Exception:
            pass
    # MM:SS
    m = re.search(r'(\d+):(\d+)', s)
    if m:
        try:
            mi, sec = map(int, m.groups())
            total = mi * 60 + sec
            return float(total) if total > 0 else None
        except Exception:
            pass
    # "Xh Ym Zs" / "Xm Ys" (require units present)
    if re.search(r'[hms]', s, re.IGNORECASE):
        m = re.search(
            r'(?:(\d+)\s*h(?:our(?:s)?)?\s*)?'
            r'(?:(\d+)\s*m(?:in(?:ute(?:s)?)?)?\s*)?'
            r'(?:(\d+(?:\.\d+)?)\s*s(?:ec(?:ond(?:s)?)?)?)?',
            s,
            re.IGNORECASE
        )
        if m:
            try:
                h = int(m.group(1) or 0)
                mi = int(m.group(2) or 0)
                sec = float(m.group(3) or 0)
                total = h * 3600 + mi * 60 + sec
                return float(total) if total > 0 else None
            except Exception:
                pass
    return None

def _get_tag_value(tag, key: str) -> Optional[str]:
    """
    Safe attribute/text extraction:
    - If key exists as attribute -> use it
    - Otherwise fallback to tag text (useful for <span itemprop="duration">PT5M</span>)
    """
    if tag is None:
        return None
    if key and key in tag.attrs:
        v = tag.attrs.get(key)
        return str(v).strip() if v is not None else None
    txt = tag.get_text(" ", strip=True)
    return txt if txt else None

def parse_duration_from_json_ld(html: str) -> Optional[float]:
    """
    Extract duration from JSON-LD structured data.
    Supports:
    - single object or list
    - nested @graph
    - nested objects
    Looks for:
    - duration
    - contentDuration
    - VideoObject / types containing "Video"
    """
    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
    def walk(obj: Any) -> Optional[float]:
        if isinstance(obj, dict):
            # if this dict claims a type with Video
            t = obj.get("@type") or obj.get("type")
            is_videoish = False
            if isinstance(t, str) and "VIDEO" in t.upper():
                is_videoish = True
            elif isinstance(t, list) and any(isinstance(x, str) and "VIDEO" in x.upper() for x in t):
                is_videoish = True
            # duration fields can appear even without @type
            for k in ("duration", "contentDuration"):
                if k in obj and obj.get(k) is not None:
                    d = parse_duration_text(obj.get(k))
                    if d:
                        return d
            # if videoish, sometimes nested duration under other keys; still walk
            for v in obj.values():
                r = walk(v)
                if r:
                    return r
        elif isinstance(obj, list):
            for it in obj:
                r = walk(it)
                if r:
                    return r
        return None
    for script in scripts:
        raw = script.string or script.get_text(strip=True)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        # handle @graph
        if isinstance(data, dict) and "@graph" in data:
            r = walk(data.get("@graph"))
            if r:
                return r
        r = walk(data)
        if r:
            return r
    return None

def parse_duration_from_javascript(html: str, log_fn: Callable = print) -> Optional[float]:
    """
    Improved inline JS duration scan.
    - More patterns
    - Log candidates
    - Cluster similar durations and pick from largest cluster
    - Ignore common ad lengths if better exist
    - Prefer longest in reasonable range
    """
    js_patterns = [
        # seconds
        r'["\']?duration["\']?\s*[:=]\s*["\']?(\d+(?:\.\d+)?)',
        r'["\']?videoDuration["\']?\s*[:=]\s*["\']?(\d+(?:\.\d+)?)',
        r'["\']?contentDuration["\']?\s*[:=]\s*["\']?(\d+(?:\.\d+)?)',
        r'"duration"\s*:\s*(\d+(?:\.\d+)?)',
        r'["\']?videoLength["\']?\s*[:=]\s*["\']?(\d+(?:\.\d+)?)',
        r'["\']?media_duration["\']?\s*[:=]\s*["\']?(\d+(?:\.\d+)?)',
        r'["\']?durationSeconds["\']?\s*[:=]\s*["\']?(\d+(?:\.\d+)?)',
        r'["\']?lengthSeconds["\']?\s*[:=]\s*["\']?(\d+(?:\.\d+)?)',
        # NEW patterns
        r'["\']?length["\']?\s*[:=]\s*["\']?(\d+(?:\.\d+)?)',
        r'["\']?mediaLength["\']?\s*[:=]\s*["\']?(\d+(?:\.\d+)?)',
        r'["\']?totalDuration["\']?\s*[:=]\s*["\']?(\d+(?:\.\d+)?)',
        r'["\']?clipDuration["\']?\s*[:=]\s*["\']?(\d+(?:\.\d+)?)',
        r'["\']?runtime["\']?\s*[:=]\s*["\']?(\d+(?:\.\d+)?)',
        r'["\']?video_time["\']?\s*[:=]\s*["\']?(\d+(?:\.\d+)?)',
        r'["\']?time_length["\']?\s*[:=]\s*["\']?(\d+(?:\.\d+)?)',
        # milliseconds
        r'["\']?durationMs["\']?\s*[:=]\s*["\']?(\d+)',
        r'["\']?approxDurationMs["\']?\s*[:=]\s*["\']?(\d+)',
        r'["\']?lengthMs["\']?\s*[:=]\s*["\']?(\d+)',
    ]
    candidates: List[float] = []
    for pat in js_patterns:
        for m in re.findall(pat, html, re.IGNORECASE):
            try:
                v = float(m)
            except ValueError:
                continue
            if "MS" in pat.upper():
                v = v / 1000.0
            if 1 <= v <= 86400:
                candidates.append(v)
    # ISO durations in JS
    for m in re.findall(r'["\']duration["\']\s*:\s*["\'](P[^"\']+)["\']', html, re.IGNORECASE):
        d = parse_iso8601_duration_enhanced(m)
        if d and 1 <= d <= 86400:
            candidates.append(d)
    if not candidates:
        return None
    # Dedup and sort
    candidates = sorted(set(candidates))
    log_fn(f" • JS candidates: {', '.join(f'{x:.1f}' for x in candidates)}")
    # Common ad durations to deprioritize
    ad_common = {5.0, 6.0, 10.0, 15.0, 30.0}
    # Filter reasonable (>=30s) and tiny
    reasonable = [x for x in candidates if x >= 30 and x not in ad_common]
    tiny = [x for x in candidates if x < 30]
    if reasonable:
        # Pick the longest reasonable (often the main video is the longest mentioned)
        return max(reasonable)
    elif tiny:
        # If only tiny, pick the longest (better than 5s if there's 15s)
        return max(tiny)
    return None

def extract_duration_from_player_config(html: str) -> Optional[float]:
    """
    Extract duration from some common player setups (best-effort heuristic).
    """
    # JW Player setup patterns (very heuristic)
    jw_patterns = [
        r'jwplayer\([^)]*\)\.setup\(\s*\{(.+?)\}\s*\)',
        r'playerInstance\.setup\(\s*\{(.+?)\}\s*\)',
    ]
    for pat in jw_patterns:
        for blob in re.findall(pat, html, re.DOTALL | re.IGNORECASE):
            m = re.search(r'["\']?duration["\']?\s*[:=]\s*["\']?(\d+(?:\.\d+)?)', blob, re.IGNORECASE)
            if m:
                try:
                    d = float(m.group(1))
                    if 1 <= d <= 86400:
                        return d
                except ValueError:
                    pass
    # Video.js / others sometimes use data-duration
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(["video", "div"], attrs={"data-duration": True}):
        try:
            d = float(tag.get("data-duration"))
            if 1 <= d <= 86400:
                return d
        except Exception:
            pass
    return None

def parse_duration_from_html_meta(html: str) -> Optional[float]:
    """
    Parse duration from meta/microdata-ish tags.
    Fixes your earlier bug: <span itemprop="duration"> usually has TEXT, not content attr.
    """
    soup = BeautifulSoup(html, "html.parser")
    patterns = [
        {"name": "meta", "attrs": {"property": "og:video:duration"}, "key": "content"},
        {"name": "meta", "attrs": {"property": "video:duration"}, "key": "content"},
        {"name": "meta", "attrs": {"name": "twitter:player:stream:duration"}, "key": "content"},
        {"name": "meta", "attrs": {"name": "duration"}, "key": "content"},
        {"name": "meta", "attrs": {"itemprop": "duration"}, "key": "content"},
        {"name": "time", "attrs": {"itemprop": "duration"}, "key": "datetime"},
        {"name": "span", "attrs": {"itemprop": "duration"}, "key": ""}, # text fallback
    ]
    for pat in patterns:
        tag = soup.find(pat["name"], attrs=pat["attrs"])
        value = _get_tag_value(tag, pat.get("key", "content"))
        if value:
            d = parse_duration_text(value)
            if d:
                return d
    return None

def parse_duration_from_url(url: str) -> Optional[float]:
    """Rare: some URLs have ?t=300 or duration=5m in query"""
    try:
        parsed = urllib.parse.urlparse(url)
        params = urllib.parse.parse_qs(parsed.query)
        for k in ["duration", "t", "time", "length"]:
            if k in params:
                v = params[k][0]
                d = parse_duration_text(v)
                if d and d > 0:
                    return d
    except Exception:
        pass
    return None

def parse_duration_comprehensive(html: str, url: str = None, log_fn: Callable = print) -> Optional[float]:
    log_fn(" • Checking HTML meta/microdata...")
    d = parse_duration_from_html_meta(html)
    if d:
        log_fn(f" ✓ Found in meta/microdata: {d:.1f}s")
        return d
    log_fn(" • Checking JSON-LD...")
    d = parse_duration_from_json_ld(html)
    if d:
        log_fn(f" ✓ Found in JSON-LD: {d:.1f}s")
        return d
    log_fn(" • Checking inline JavaScript...")
    d = parse_duration_from_javascript(html, log_fn)
    if d:
        # JS durations are noisy. Ignore tiny ones unless nothing else works.
        if d < 30:
            log_fn(f" ⚠ JS duration {d:.1f}s looks suspicious (ad/preview?). Continuing...")
        else:
            log_fn(f" ✓ Found in JavaScript: {d:.1f}s")
            return d
    log_fn(" • Checking player configurations...")
    d2 = extract_duration_from_player_config(html)
    if d2:
        log_fn(f" ✓ Found in player config: {d2:.1f}s")
        return d2
    if url:
        log_fn(" • Checking URL params...")
        d3 = parse_duration_from_url(url)
        if d3:
            log_fn(f" ✓ Found in URL: {d3:.1f}s")
            return d3
    # NEW: if we had a small JS duration and found nothing better, return it as last resort
    if d and d > 0:
        log_fn(f" ⚠ Returning low-confidence JS duration: {d:.1f}s")
        return d
    log_fn(" ✗ No duration found in HTML/JS")
    return None

# -----------------------------
# Manifest-based duration
# -----------------------------
def try_duration_from_manifest(url: str, log_fn: Callable = print) -> Optional[float]:
    """
    If URL points to a media manifest:
      - HLS (.m3u8): sum EXTINF durations (works for VOD playlists)
      - DASH (.mpd): parse mediaPresentationDuration ISO string
    """
    if not url:
        return None
    base = url.lower().split("?")[0]
    if not (base.endswith(".m3u8") or base.endswith(".mpd")):
        return None
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200 or not r.text:
            return None
        text = r.text
        if base.endswith(".m3u8"):
            total = 0.0
            found = False
            for m in re.finditer(r"#EXTINF:([\d\.]+)", text):
                try:
                    total += float(m.group(1))
                    found = True
                except Exception:
                    pass
            if found and total > 0:
                log_fn(f" ✓ Found via HLS EXTINF sum: {total:.1f}s")
                return float(total)
        if base.endswith(".mpd"):
            m = re.search(r'mediaPresentationDuration="([^"]+)"', text)
            if m:
                d = parse_iso8601_duration_enhanced(m.group(1))
                if d:
                    log_fn(f" ✓ Found via DASH MPD duration: {d:.1f}s")
                    return float(d)
    except Exception as e:
        log_fn(f" ⚠ Manifest duration check failed: {str(e)[:80]}...")
    return None

# -----------------------------
# Optional browser automation fallback (selenium)
# -----------------------------
def get_duration_with_browser_automation(url: str, log_fn: Callable = print) -> Optional[float]:
    """
    Last resort: Use selenium + iframe switching to find and read <video>.duration.
    Improvements in this version:
    - Tries main page first
    - Then scans for promising iframes and switches into them
    - Longer wait + more aggressive metadata loading
    - Better shadow DOM / nested video detection
    - More logging to understand what fails
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.by import By
        from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException

        log_fn(" • Starting browser automation (with iframe support)...")
        chrome_options = Options()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--log-level=3")
        chrome_options.add_argument("--autoplay-policy=no-user-gesture-required")
        chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")
        driver = None
        try:
            # Prefer webdriver-manager if available
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.chrome.service import Service
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            log_fn(" ✓ Using webdriver-manager ChromeDriver")
        except ImportError:
            driver = webdriver.Chrome(options=chrome_options)
            log_fn(" ✓ Using system ChromeDriver")
        driver.set_page_load_timeout(40)
        # ────────────────────────────────────────────────
        # Step 1: Load the page
        # ────────────────────────────────────────────────
        log_fn(f" • Loading URL: {url[:90]}...")
        try:
            driver.get(url)
        except WebDriverException as e:
            log_fn(f" ✗ Page load failed: {str(e)[:120]}")
            driver.quit()
            return None
        # Give page some initial breathing room
        time.sleep(3)
        # ────────────────────────────────────────────────
        # Step 2: Try to find and switch into promising iframes
        # ────────────────────────────────────────────────
        current_context = "main"
        best_duration = None
        # List of keywords that suggest an iframe contains a video player
        video_keywords = ["player", "video", "embed", "stream", "cdn", "media", "watch", "content", "jwplayer", "videojs", "plyr"]
        def try_get_duration_in_current_context() -> Optional[float]:
            nonlocal best_duration
            try:
                WebDriverWait(driver, 12).until(
                    lambda d: d.execute_script("return document.querySelectorAll('video').length > 0")
                )
                log_fn(f" ✓ Found <video> tag(s) in {current_context}")
            except TimeoutException:
                log_fn(f" • No <video> found in {current_context} after wait")
                return None
            # Poll for duration
            deadline = time.time() + 35
            local_best = None
            while time.time() < deadline:
                duration = driver.execute_script("""
                    function findBestVideoDuration() {
                        const candidates = [];
                        // Direct video elements
                        document.querySelectorAll('video').forEach(v => {
                            try {
                                if (v.preload !== 'auto') v.preload = 'auto';
                                if (v.readyState < 2 && typeof v.load === 'function') v.load();
                                if (v.duration && v.duration > 0 && v.duration !== Infinity && !isNaN(v.duration)) {
                                    candidates.push(v.duration);
                                }
                            } catch(e) {}
                        });
                        // Try shadow DOM / nested
                        document.querySelectorAll('*').forEach(el => {
                            if (el.shadowRoot) {
                                const shadowVids = el.shadowRoot.querySelectorAll('video');
                                shadowVids.forEach(v => {
                                    try {
                                        if (v.duration && v.duration > 0 && v.duration !== Infinity && !isNaN(v.duration)) {
                                            candidates.push(v.duration);
                                        }
                                    } catch(e) {}
                                });
                            }
                        });
                        if (candidates.length === 0) return null;
                        const big = candidates.filter(d => d >= 30);
                        return big.length ? Math.max(...big) : Math.max(...candidates);
                    }
                    return findBestVideoDuration();
                """)
                if duration:
                    log_fn(f" → Got duration in {current_context}: {duration:.1f}s")
                    if duration > (local_best or 0):
                        local_best = duration
                    if duration >= 30:
                        return duration # early exit on good candidate
                time.sleep(1.2)
            return local_best
        # ────────────────────────────────────────────────
        # First: try main page
        # ────────────────────────────────────────────────
        log_fn(" • Trying main document...")
        best_duration = try_get_duration_in_current_context()
        # ────────────────────────────────────────────────
        # Then: try switching into iframes
        # ────────────────────────────────────────────────
        if best_duration is None or best_duration < 30:
            log_fn(" • Main page had no good duration → checking iframes...")
            driver.switch_to.default_content()
            iframes = driver.find_elements(By.TAG_NAME, "iframe")
            log_fn(f" • Found {len(iframes)} iframe(s)")
            for i, iframe in enumerate(iframes, 1):
                try:
                    src = iframe.get_attribute("src") or ""
                    short_src = src[:80] + "..." if len(src) > 80 else src
                    # Skip clearly non-video iframes (ads, comments, etc.)
                    if not any(kw in src.lower() for kw in video_keywords) and not any(kw in (iframe.get_attribute("id") or "").lower() for kw in video_keywords):
                        continue
                    log_fn(f" • Trying iframe {i}/{len(iframes)}: {short_src}")
                    driver.switch_to.frame(iframe)
                    current_context = f"iframe {i} ({short_src})"
                    duration = try_get_duration_in_current_context()
                    if duration and (best_duration is None or duration > best_duration):
                        best_duration = duration
                        log_fn(f" → Better duration found in iframe: {duration:.1f}s")
                    driver.switch_to.default_content()
                    if best_duration and best_duration >= 30:
                        break # no need to check more iframes
                except Exception as e:
                    log_fn(f" ⚠ Iframe {i} failed: {str(e)[:80]}")
                    driver.switch_to.default_content()
        driver.quit()
        if best_duration and best_duration > 0:
            log_fn(f" ✓ Final best duration from browser: {best_duration:.1f}s")
            return float(best_duration)
        else:
            log_fn(" ✗ No usable duration found even after checking iframes")
    except ImportError:
        log_fn(" ⚠ Selenium not installed → run: pip install selenium")
    except Exception as e:
        log_fn(f" ✗ Browser automation crashed: {type(e).__name__}: {str(e)[:120]}")
    log_fn(" 💡 Tip: If this keeps failing, the site may require Playwright or yt-dlp is more reliable here.")
    return None

def get_duration_by_downloading_segment(url: str, log_fn: Callable = print) -> Optional[float]:
    """
    Download a small segment to force extractor to resolve info; parse duration from JSON.
    """
    try:
        log_fn("📥 Downloading small segment to get duration...")
        with tempfile.TemporaryDirectory() as tmpdir:
            output_template = os.path.join(tmpdir, "temp_segment.%(ext)s")
            cmd = [
                "yt-dlp",
                "-o", output_template,
                "--download-sections", "*0-2",
                "--dump-single-json",
                "--no-warnings",
                "--no-playlist",
                "--force-ipv4",
                "--socket-timeout", "60",
                "--retries", "3",
                "--fragment-retries", "5",
                url
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=90,
                check=False
            )
            if result.returncode == 0 and (result.stdout or "").strip():
                try:
                    info = json.loads(result.stdout.strip())
                except Exception:
                    info = None
                d = _parse_yt_dlp_json_duration(info)
                if d and d > 0:
                    log_fn(f"✅ Got duration from segment JSON: {d:.1f}s")
                    return d
            log_fn("⚠️ Segment download didn't provide duration")
    except subprocess.TimeoutExpired:
        log_fn("⏰ Segment download timed out")
    except Exception as e:
        log_fn(f"⚠️ Segment download failed: {str(e)[:100]}...")
    return None

# -----------------------------
# yt-dlp duration detection
# -----------------------------
def _parse_yt_dlp_json_duration(obj: Any) -> Optional[float]:
    """
    Extract duration from a yt-dlp JSON object (single video entry).
    """
    if not isinstance(obj, dict):
        return None
    for field in ("duration", "approx_duration", "length", "length_seconds"):
        v = obj.get(field)
        if v is None:
            continue
        try:
            d = float(v)
            if d > 0:
                return d
        except Exception:
            pass
    # sometimes ms-like fields appear in custom extractors
    for field in ("durationMs", "approxDurationMs"):
        v = obj.get(field)
        if v is None:
            continue
        try:
            d = float(v) / 1000.0
            if d > 0:
                return d
        except Exception:
            pass
    # sometimes per-format duration
    fmts = obj.get("formats")
    if isinstance(fmts, list):
        for f in fmts:
            if isinstance(f, dict) and f.get("duration") is not None:
                try:
                    d = float(f.get("duration"))
                    if d > 0:
                        return d
                except Exception:
                    pass
    return None

# Global cache for successful duration extraction methods
_duration_method_cache = {"last_successful": None, "domain": None}

def reset_duration_method_cache():
    """Reset the cached duration extraction method. Useful when switching domains or if cached method stops working."""
    global _duration_method_cache
    _duration_method_cache = {"last_successful": None, "domain": None}

def get_video_duration_advanced(url: str, log_fn: Callable = print, skip_cache: bool = False) -> Optional[float]:
    """
    Get duration using:
    1) Manifest parse (if URL is .m3u8 / .mpd)
    2) HTML/JS comprehensive parser (fast, no yt-dlp)
    3) yt-dlp --dump-single-json (best if supported)
    4) yt-dlp alternative flags
    5) Browser automation if all else fails
   
    Caches the last successful method to speed up subsequent downloads from the same domain.
    """
    current_domain = extract_domain(url)
   
    # Try cached method first if it's from the same domain
    if not skip_cache and _duration_method_cache["last_successful"] and _duration_method_cache["domain"] == current_domain:
        cached_method = _duration_method_cache["last_successful"]
        log_fn(f"🚀 Trying cached method first: {cached_method['name']}")
       
        try:
            if cached_method["type"] == "manifest":
                d = try_duration_from_manifest(url, log_fn)
                if d:
                    log_fn(f"✅ Cached method worked: {d:.1f}s")
                    return d
            elif cached_method["type"] == "html_js":
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                resp = requests.get(url, headers=headers, timeout=12)
                if resp.status_code == 200 and resp.text:
                    d = parse_duration_comprehensive(resp.text, url, log_fn)
                    if d and d >= 30:
                        log_fn(f"✅ Cached method worked: {d:.1f}s")
                        return d
            elif cached_method["type"] == "yt_dlp":
                cmd = cached_method["cmd"] + [url]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=45, check=False)
               
                if result.returncode == 0 and (result.stdout or "").strip():
                    out = result.stdout.strip()
                   
                    if cached_method["name"] == "print duration":
                        if out not in ("NA", "None", ""):
                            try:
                                d = float(out)
                                if 0 < d <= 86400:
                                    log_fn(f"✅ Cached method worked: {d:.1f}s")
                                    return d
                            except Exception:
                                pass
                    else:
                        try:
                            data = json.loads(out)
                        except Exception:
                            first_line = out.splitlines()[0].strip()
                            try:
                                data = json.loads(first_line)
                            except Exception:
                                data = None
                       
                        if data:
                            d = _parse_yt_dlp_json_duration(data)
                            if d and 0 < d <= 86400:
                                log_fn(f"✅ Cached method worked: {d:.1f}s")
                                return d
                           
                            entries = data.get("entries") if isinstance(data, dict) else None
                            if isinstance(entries, list):
                                for e in entries:
                                    d = _parse_yt_dlp_json_duration(e)
                                    if d and 0 < d <= 86400:
                                        log_fn(f"✅ Cached method worked: {d:.1f}s")
                                        return d
        except Exception as e:
            log_fn(f"⚠️ Cached method failed: {str(e)[:80]}... Trying all methods")
       
        log_fn("⚠️ Cached method didn't work, trying all methods...")
    
    # Continue with normal duration extraction if cached method fails or not available
    log_fn("🔍 Trying manifest duration...")
    d = try_duration_from_manifest(url, log_fn)
    if d and d > 0:
        _duration_method_cache["last_successful"] = {"type": "manifest", "name": "manifest", "domain": current_domain}
        return d
   
    log_fn("🔍 Trying HTML/JS extraction...")
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        resp = requests.get(url, headers=headers, timeout=12)
        if resp.status_code == 200 and resp.text:
            d = parse_duration_comprehensive(resp.text, url, log_fn)
            if d and d >= 30:
                _duration_method_cache["last_successful"] = {"type": "html_js", "name": "html_js", "domain": current_domain}
                return d
    except Exception as e:
        log_fn(f"⚠️ HTML fetch failed: {e}")
   
    log_fn("🔍 Trying yt-dlp info...")
    try:
        # Try --print duration (fast)
        cmd_print = ["yt-dlp", "--print", "%(duration)s", "--no-warnings", "--no-playlist", "--force-ipv4", "--socket-timeout", "15", url]
        result = subprocess.run(cmd_print, capture_output=True, text=True, timeout=20, check=False)
        if result.returncode == 0 and result.stdout.strip() and result.stdout.strip() not in ("NA", "None", ""):
            try:
                d = float(result.stdout.strip())
                if 0 < d <= 86400:
                    log_fn(f"✅ Got duration from yt-dlp print: {d:.1f}s")
                    _duration_method_cache["last_successful"] = {"type": "yt_dlp", "name": "print duration", "cmd": cmd_print[:-1], "domain": current_domain}
                    return d
            except Exception:
                pass
       
        # Try --dump-json
        cmd_json = ["yt-dlp", "--dump-single-json", "--no-warnings", "--no-playlist", "--force-ipv4", "--socket-timeout", "20", url]
        result = subprocess.run(cmd_json, capture_output=True, text=True, timeout=30, check=False)
        if result.returncode == 0 and result.stdout.strip():
            data = None
            try:
                data = json.loads(result.stdout.strip())
            except Exception:
                first_line = result.stdout.splitlines()[0].strip()
                try:
                    data = json.loads(first_line)
                except Exception:
                    pass
            if data:
                d = _parse_yt_dlp_json_duration(data)
                if d and 0 < d <= 86400:
                    log_fn(f"✅ Got duration from yt-dlp JSON: {d:.1f}s")
                    _duration_method_cache["last_successful"] = {"type": "yt_dlp", "name": "dump json", "cmd": cmd_json[:-1], "domain": current_domain}
                    return d
    except Exception as e:
        log_fn(f"⚠️ yt-dlp duration extraction failed: {e}")
   
    log_fn("🔍 Trying segment download...")
    d = get_duration_by_downloading_segment(url, log_fn)
    if d and d > 0:
        _duration_method_cache["last_successful"] = {"type": "segment", "name": "segment download", "domain": current_domain}
        return d
   
    log_fn("🔍 Trying browser automation...")
    d = get_duration_with_browser_automation(url, log_fn)
    if d and d > 0:
        _duration_method_cache["last_successful"] = {"type": "browser", "name": "browser automation", "domain": current_domain}
        return d
   
    log_fn("❌ All duration extraction methods failed")
    return None

# -----------------------------
# URL filename extraction
# -----------------------------
def extract_filename_from_url(url: str) -> Optional[str]:
    """
    Extract a clean filename from a URL.
    Returns None if no good filename can be extracted.
    """
    try:
        parsed = urllib.parse.urlparse(url)
       
        # Get the path component
        path = parsed.path.strip('/')
        if not path:
            return None
           
        # Extract the last segment
        filename = path.split('/')[-1]
       
        # Remove query string if present
        filename = filename.split('?')[0]
        filename = filename.split('#')[0]
       
        # Clean up the filename
        filename = urllib.parse.unquote(filename) # Decode URL-encoded characters
       
        # Remove problematic characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
       
        # Remove common tracking parameters that might be in the filename
        filename = re.sub(r'[_-]*(?:utm_|source|medium|campaign|term|content)[_-].*$', '', filename, flags=re.IGNORECASE)
       
        # Remove unwanted extensions
        unwanted_extensions = ['.html', '.htm', '.php', '.aspx', '.jsp', '.asp']
        for ext in unwanted_extensions:
            if filename.lower().endswith(ext):
                filename = filename[:-len(ext)]
                break
       
        # Also remove .html if it's followed by query-like pattern
        filename = re.sub(r'\.html(?:_\d+)?$', '', filename, flags=re.IGNORECASE)
       
        # Remove trailing special characters
        filename = filename.strip('.-_')
       
        # Check if it looks like a valid filename (not just a page identifier)
        invalid_patterns = [
            r'^index$',
            r'^default$',
            r'^video$',
            r'^watch$',
            r'^play$',
            r'^\d+$', # Just numbers
            r'^[a-f0-9]{8,}$', # Hex/hash-like
        ]
       
        for pattern in invalid_patterns:
            if re.match(pattern, filename, re.IGNORECASE):
                return None
       
        # Ensure it's not too short
        if len(filename) < 3:
            return None
           
        return filename
       
    except Exception:
        return None

def get_safe_filename(url: str, index: int, log_fn: Callable = print) -> str:
    """
    Get a safe filename for downloading, preferring URL-based filename.
    Falls back to yt-dlp title extraction.
    """
    # Try to get filename from URL first
    url_filename = extract_filename_from_url(url)
    if url_filename:
        log_fn(f"📝 Extracted filename from URL: {url_filename}")
       
        # Check if it already has a video extension
        video_extensions = ['.mp4', '.mkv', '.webm', '.avi', '.mov', '.flv', '.m4v', '.wmv']
        has_video_ext = any(url_filename.lower().endswith(ext) for ext in video_extensions)
       
        if has_video_ext:
            # Use the filename as-is with index prefix
            clean_name = re.sub(r'[<>:"/\\|?*]', '_', url_filename)
            return f"{index:03d} - {clean_name}"
        else:
            # Add index prefix but let yt-dlp add extension
            clean_name = re.sub(r'[<>:"/\\|?*]', '_', url_filename)
            return f"{index:03d} - {clean_name}.%(ext)s"
   
    # Fallback to yt-dlp's title extraction
    log_fn("📝 Getting title from yt-dlp...")
    try:
        cmd = [
            "yt-dlp",
            "--print", "%(title)s",
            "--no-warnings",
            "--no-playlist",
            "--force-ipv4",
            "--socket-timeout", "30",
            url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
        if result.returncode == 0 and result.stdout.strip():
            title = result.stdout.strip()
            title = re.sub(r'[\\/*?:"<>|]', "_", title)
            return f"{index:03d} - {title}.%(ext)s"
    except Exception as e:
        log_fn(f"⚠️ Failed to get title: {e}")
   
    # Last resort: use URL hash
    url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{index:03d} - video_{url_hash}.%(ext)s"

# -----------------------------
# URL helpers + link extraction
# -----------------------------
def make_absolute_url(url: str, base_url: str) -> Optional[str]:
    """Convert relative URL to absolute URL"""
    if not url:
        return None
    if url.startswith(("http://", "https://")):
        return url
    try:
        parsed_base = urllib.parse.urlparse(base_url)
        if url.startswith("//"):
            return f"{parsed_base.scheme}:{url}"
        if url.startswith("/"):
            return f"{parsed_base.scheme}://{parsed_base.netloc}{url}"
        path = parsed_base.path.rsplit("/", 1)[0] if "/" in parsed_base.path else ""
        return f"{parsed_base.scheme}://{parsed_base.netloc}{path}/{url}"
    except Exception:
        return None

def extract_video_links(url: str, pattern: str = "/video/", log_fn: Callable = print) -> List[str]:
    """
    Extract video links from a webpage.
    NOTE: Duration extraction on listing pages is often meaningless; we no longer do it here.
    """
    log_fn(f"🌐 Fetching page: {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise DownloadError(f"Failed to fetch page: {e}")
    soup = BeautifulSoup(response.text, "html.parser")
    video_links: List[str] = []
    # Strategy 1: <a> tags with pattern
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if pattern in href:
            full_url = make_absolute_url(href, url)
            if full_url and full_url not in video_links:
                video_links.append(full_url)
    # Strategy 2: video tags
    if not video_links:
        for video in soup.find_all("video"):
            if video.get("src"):
                full_url = make_absolute_url(video["src"], url)
                if full_url and full_url not in video_links:
                    video_links.append(full_url)
            for source in video.find_all("source"):
                if source.get("src"):
                    full_url = make_absolute_url(source["src"], url)
                    if full_url and full_url not in video_links:
                        video_links.append(full_url)
    # Strategy 3: iframes
    if not video_links:
        for iframe in soup.find_all("iframe"):
            if iframe.get("src"):
                src = iframe["src"]
                if any(domain in src for domain in ["youtube.com", "youtu.be", "vimeo.com", "dailymotion.com"]):
                    if src not in video_links:
                        video_links.append(src)
    log_fn(f"🎬 Found {len(video_links)} video links")
    if not video_links:
        log_fn("⚠️ No video links found. Page structure may have changed.")
        log_fn("💡 Try a different pattern or check the page manually")
    return video_links

# -----------------------------
# Download logic
# -----------------------------
# ────────────────────────────────────────────────
# NEW: Playwright-based duration extraction fallback
# ────────────────────────────────────────────────

def get_duration_with_playwright_automation(
    url: str,
    log_fn: Callable = print,
    timeout: int = 90,
    use_network_capture: bool = True
) -> Optional[float]:
    """
    Improved Playwright duration extraction with better error handling and network capture.
    Fixed: Properly filters out .mp4.jpg and other non-video URLs
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout, Error as PWError

        log_fn("  • Starting Playwright automation...")

        browser = None
        context = None
        page = None

        with sync_playwright() as p:
            # Launch with more lenient settings
            browser = p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--autoplay-policy=no-user-gesture-required",
                    "--disable-background-timer-throttling",
                    "--disable-renderer-backgrounding",
                    "--disable-web-security",
                    "--allow-running-insecure-content"
                ]
            )

            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
                viewport={"width": 1920, "height": 1080},
                ignore_https_errors=True,
                java_script_enabled=True,
                extra_http_headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1"
                }
            )

            # Enable request interception for network capture
            if use_network_capture:
                page = context.new_page()
                
                # Track video-related requests with proper filtering
                video_urls = set()
                
                def is_video_url(url: str) -> bool:
                    """Check if URL is actually a video and not an image thumbnail"""
                    url_lower = url.lower()
                    
                    # First, filter out obvious non-videos
                    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg', '.ico']
                    if any(url_lower.endswith(ext) for ext in image_extensions):
                        return False
                    
                    # Filter out URLs that look like thumbnails even if they have video extensions
                    thumbnail_patterns = ['thumb', 'thumbnail', 'preview', 'poster', 'cover', 'snapshot']
                    if any(pattern in url_lower for pattern in thumbnail_patterns):
                        return False
                    
                    # Now check for actual video indicators
                    video_patterns = [
                        # Video extensions
                        '.mp4', '.mkv', '.webm', '.avi', '.mov', '.flv', '.m4v', '.wmv',
                        '.3gp', '.ogv', '.mpeg', '.mpg', '.ts', '.m3u8', '.mpd',
                        # Video parameters
                        'videoplayback', 'master.m3u8', 'playlist.m3u8', '.m3u8?',
                        '/video/', '/videos/', '/media/', '/stream/',
                        # Video CDNs
                        'mycdn.me', 'cloudfront.net', 'akamaihd.net',
                        # Video ID patterns
                        'videoid=', 'video_id=', 'video-id='
                    ]
                    
                    return any(pattern in url_lower for pattern in video_patterns)
                
                def handle_request(request):
                    req_url = request.url
                    if is_video_url(req_url):
                        # Double-check it's not an image with video extension in path
                        parsed = urllib.parse.urlparse(req_url)
                        path = parsed.path.lower()
                        
                        # Final check: if it has image extension in path, filter it out
                        if any(path.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                            return
                        
                        video_urls.add(req_url)
                        log_fn(f"      📡 Captured video request: {req_url[:100]}...")
                
                page.on("request", handle_request)
            else:
                page = context.new_page()

            log_fn(f"    • Loading: {url[:90]}...")
            
            # Use domcontentloaded first, then wait for networkidle
            response = page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
            
            if not response:
                log_fn("    ✗ No response")
                return None
            
            if response.status >= 400:
                log_fn(f"    ✗ Bad response: {response.status}")
                # Continue anyway, some sites work despite 4xx
            
            # Wait a bit for initial load
            page.wait_for_timeout(5000)
            
            # Scroll to trigger lazy loading
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(2000)
            page.evaluate("window.scrollTo(0, 0)")
            page.wait_for_timeout(1000)

            # Try multiple strategies to find the video
            best_dur = None
            
            # Strategy 1: Look for video elements directly
            log_fn("    • Strategy 1: Looking for video elements...")
            
            # Check for video elements without waiting
            has_video = page.evaluate("document.querySelectorAll('video').length > 0")
            if has_video:
                log_fn("    ✓ Found video element(s)")
                
                # Try to trigger video loading
                page.evaluate("""
                    document.querySelectorAll('video').forEach(v => {
                        v.preload = 'auto';
                        v.load();
                        v.muted = true;
                        v.play().catch(e => console.log('Play failed:', e));
                    });
                """)
                
                # Poll for duration
                for attempt in range(15):  # ~30 seconds
                    duration = page.evaluate("""
                        () => {
                            const videos = document.querySelectorAll('video');
                            let best = null;
                            for (const v of videos) {
                                if (v.duration && v.duration > 0 && v.duration < 86400 && !isNaN(v.duration)) {
                                    if (!best || v.duration > best) {
                                        best = v.duration;
                                    }
                                }
                            }
                            return best;
                        }
                    """)
                    
                    if duration and duration > 0:
                        log_fn(f"      → Video duration: {duration:.1f}s")
                        if duration >= 30:
                            best_dur = duration
                            break
                        elif duration > (best_dur or 0):
                            best_dur = duration
                    
                    page.wait_for_timeout(2000)
            
            # Strategy 2: Check iframes (if main strategy failed)
            if not best_dur or best_dur < 30:
                log_fn("    • Strategy 2: Checking iframes...")
                iframes = page.query_selector_all("iframe")
                log_fn(f"      Found {len(iframes)} iframes")
                
                for i, iframe in enumerate(iframes, 1):
                    try:
                        # Try to get iframe src for logging
                        src = iframe.get_attribute("src") or ""
                        if not src or "ad" in src.lower() or "facebook" in src.lower():
                            continue
                        
                        log_fn(f"      • Trying iframe {i}: {src[:80]}...")
                        
                        frame = iframe.content_frame()
                        if not frame:
                            continue
                        
                        # Wait a bit for iframe content
                        frame.wait_for_timeout(3000)
                        
                        # Check for video in iframe
                        dur = frame.evaluate("""
                            () => {
                                const v = document.querySelector('video');
                                return v && v.duration > 0 && v.duration < 86400 && !isNaN(v.duration) ? v.duration : null;
                            }
                        """)
                        
                        if dur and dur > (best_dur or 0):
                            log_fn(f"        → Duration in iframe {i}: {dur:.1f}s")
                            best_dur = dur
                        
                        if best_dur and best_dur >= 30:
                            break
                            
                    except Exception as iframe_err:
                        log_fn(f"      ⚠ Iframe {i} failed: {str(iframe_err)[:80]}")
            
            # Strategy 3: Network capture (if enabled)
            if use_network_capture and (not best_dur or best_dur < 30):
                log_fn("    • Strategy 3: Analyzing captured network requests...")
                
                if video_urls:
                    log_fn(f"      Found {len(video_urls)} potential video URLs")
                    
                    # Filter out any remaining non-video URLs
                    filtered_urls = []
                    for vu in video_urls:
                        vu_lower = vu.lower()
                        
                        # Skip image files masquerading as videos
                        if any(vu_lower.endswith(ext) for ext in ['.mp4.jpg', '.mp4.jpeg', '.mp4.png', '.mp4.gif']):
                            log_fn(f"      ⚠ Skipping image file masquerading as video: {vu[:80]}...")
                            continue
                        
                        # Skip URLs that clearly indicate thumbnails
                        if any(pattern in vu_lower for pattern in ['thumb', 'poster', 'cover', 'snapshot']):
                            log_fn(f"      ⚠ Skipping thumbnail URL: {vu[:80]}...")
                            continue
                        
                        filtered_urls.append(vu)
                    
                    log_fn(f"      After filtering: {len(filtered_urls)} actual video URLs")
                    
                    # Try to get duration from first video URL
                    for video_url in filtered_urls[:3]:  # Try first 3
                        log_fn(f"      • Testing video URL: {video_url[:100]}...")
                        
                        # Try to get duration from manifest
                        dur = try_duration_from_manifest(video_url, log_fn)
                        if dur and dur > (best_dur or 0):
                            log_fn(f"        → Got duration from manifest: {dur:.1f}s")
                            best_dur = dur
                            break
                        
                        # If it's a direct MP4, we might need to probe
                        if ".mp4" in video_url.lower() and not best_dur:
                            # We could potentially probe with a small download, but that's heavy
                            pass
            
            # ========== ENHANCED PLAY BUTTON DEBUGGING ==========
            # Strategy 4: Try to click common play buttons (if no video found)
            if not has_video:
                log_fn("    • 🔘 Strategy 4: Looking for play buttons...")
                log_fn("    • 🔍 DEBUG: Starting play button detection...")
                
                # Comprehensive list of play button selectors organized by category
                play_selectors = {
                    "generic": [
                        "button[aria-label*='play' i]",
                        "button[aria-label*='Play' i]",
                        "button[title*='play' i]",
                        "button[title*='Play' i]",
                        "[role='button'][aria-label*='play' i]",
                        "[role='button'][aria-label*='Play' i]",
                        "[role='button'][title*='play' i]",
                        "[role='button'][title*='Play' i]",
                    ],
                    "class_names": [
                        ".play-button",
                        ".play-btn",
                        ".btn-play",
                        ".vjs-big-play-button",
                        ".vjs-play-control",
                        ".ytp-large-play-button",
                        ".mejs__button--playpause",
                        ".plyr__control--play",
                        ".jw-icon-play",
                        ".jw-button-play",
                        ".jwplayer__playbutton",
                        ".video-js .vjs-big-play-button",
                        ".big-play-button",
                        ".play-icon",
                        ".icon-play",
                        ".fa-play",
                        ".glyphicon-play",
                    ],
                    "data_attributes": [
                        "[data-play]",
                        "[data-role='play']",
                        "[data-action='play']",
                        "[data-testid='play']",
                        "[data-testid='play-button']",
                        "[data-qa='play-button']",
                    ],
                    "id_based": [
                        "[id*='play' i]",
                        "[id*='Play' i]",
                        "#play-button",
                        "#playBtn",
                        "#btnPlay",
                        "#player-play",
                    ],
                    "text_content": [
                        "button:has-text('Play')",
                        "button:has-text('play')",
                        "button:has-text('▶')",
                        "button:has-text('►')",
                        "button:has-text('播放')",
                        "button:has-text('再生')",
                        "button:has-text('Watch')",
                        "button:has-text('watch')",
                        "button:has-text('Start')",
                        "button:has-text('start')",
                    ],
                    "svg_icons": [
                        "svg[aria-label*='play' i]",
                        "svg[title*='play' i]",
                        "button svg",
                        "div[role='button'] svg",
                    ],
                    "video_player_specific": [
                        ".jw-icon.jw-icon-inline.jw-button-color.jw-reset.jw-play-btn",
                        ".videoPlayer__play",
                        ".player-play-btn",
                        ".vjs-play-button",
                        ".vjs-default-skin .vjs-big-play-button",
                        ".html5-video-player .ytp-play-button",
                        ".player .play-button",
                        ".mejs-play",
                        ".mejs-playbutton",
                        ".plyr__controls .plyr__control--play",
                    ],
                    "css_selectors": [
                        "[class*='play']",
                        "[class*='Play']",
                        "[id*='play']",
                        "[id*='Play']",
                        "[class*='btnPlay']",
                        "[class*='btn-play']",
                    ],
                    "wildcard_fallback": [
                        ":is(button, div, span)[class*='play']",
                        ":is(button, div, span)[class*='Play']",
                        ":is(button, div, span)[id*='play']",
                        ":is(button, div, span)[id*='Play']",
                    ]
                }
                
                # First, let's debug what elements exist on the page
                log_fn("    • 🔍 DEBUG: Analyzing page structure for clickable elements...")
                
                # Get all buttons and their attributes for debugging
                all_buttons = page.evaluate("""
                    () => {
                        const buttons = [];
                        document.querySelectorAll('button, [role="button"], a[href], [onclick], .clickable, [class*="btn"], [class*="button"]').forEach(el => {
                            buttons.push({
                                tag: el.tagName,
                                id: el.id,
                                class: el.className,
                                text: el.innerText?.slice(0, 30),
                                type: el.type,
                                'aria-label': el.getAttribute('aria-label'),
                                title: el.title,
                                href: el.href,
                                onclick: !!el.onclick,
                                visible: el.offsetParent !== null,
                                rect: el.getBoundingClientRect() ? {
                                    width: el.offsetWidth,
                                    height: el.offsetHeight,
                                    top: el.offsetTop,
                                    left: el.offsetLeft
                                } : null
                            });
                        });
                        return buttons;
                    }
                """)
                
                if all_buttons and len(all_buttons) > 0:
                    log_fn(f"    • 🔍 DEBUG: Found {len(all_buttons)} potentially clickable elements")
                    # Log first 5 buttons for debugging
                    for i, btn in enumerate(all_buttons[:5]):
                        log_fn(f"      • Button {i+1}: {btn}")
                else:
                    log_fn("    • 🔍 DEBUG: No clickable elements found on page")
                
                # Also check for video containers that might need interaction
                video_containers = page.evaluate("""
                    () => {
                        const containers = [];
                        const selectors = ['.player', '.video-player', '.video-container', '.video-wrapper', 
                                          '.media-player', '.plyr', '.video-js', '.jwplayer', '.mejs-container'];
                        selectors.forEach(sel => {
                            document.querySelectorAll(sel).forEach(el => {
                                containers.push({
                                    selector: sel,
                                    id: el.id,
                                    class: el.className,
                                    hasVideo: !!el.querySelector('video')
                                });
                            });
                        });
                        return containers;
                    }
                """)
                
                if video_containers and len(video_containers) > 0:
                    log_fn(f"    • 🔍 DEBUG: Found {len(video_containers)} video player containers")
                    for container in video_containers:
                        log_fn(f"      • Container: {container}")
                
                # Now try each category of selectors with detailed logging
                for category, selectors in play_selectors.items():
                    log_fn(f"    • 🔍 Trying {category} selectors...")
                    
                    for selector in selectors:
                        try:
                            # Check if element exists without waiting
                            elements = page.query_selector_all(selector)
                            
                            if elements and len(elements) > 0:
                                log_fn(f"      ✓ Found {len(elements)} element(s) with selector: {selector}")
                                
                                for idx, element in enumerate(elements):
                                    try:
                                        # Check if element is visible and clickable
                                        is_visible = element.is_visible()
                                        is_enabled = element.is_enabled()
                                        
                                        if is_visible and is_enabled:
                                            log_fn(f"        • Element {idx+1}: Visible ✓, Enabled ✓")
                                            
                                            # Get element details for debugging
                                            element_info = element.evaluate("""
                                                (el) => ({
                                                    tag: el.tagName,
                                                    id: el.id,
                                                    class: el.className,
                                                    text: el.innerText?.slice(0, 50),
                                                    'aria-label': el.getAttribute('aria-label'),
                                                    title: el.title,
                                                    type: el.type,
                                                    role: el.getAttribute('role'),
                                                    onclick: !!el.onclick,
                                                    rect: {
                                                        width: el.offsetWidth,
                                                        height: el.offsetHeight,
                                                        top: el.offsetTop,
                                                        left: el.offsetLeft
                                                    }
                                                })
                                            """)
                                            log_fn(f"        • Element details: {element_info}")
                                            
                                            # Try to click
                                            log_fn(f"        • Attempting to click...")
                                            # Scroll element into view
                                            element.scroll_into_view_if_needed()
                                            page.wait_for_timeout(500)
                                            
                                            # Try different click strategies
                                            click_success = False
                                            
                                            # Strategy A: Regular click
                                            try:
                                                element.click(timeout=5000)
                                                log_fn(f"        • ✓ Regular click succeeded")
                                                click_success = True
                                            except Exception as click_err:
                                                log_fn(f"        • ⚠ Regular click failed: {str(click_err)[:80]}")
                                                
                                                # Strategy B: Force click via JavaScript
                                                try:
                                                    element.evaluate("el => el.click()")
                                                    log_fn(f"        • ✓ JavaScript click succeeded")
                                                    click_success = True
                                                except Exception as js_click_err:
                                                    log_fn(f"        • ⚠ JavaScript click failed: {str(js_click_err)[:80]}")
                                                    
                                                    # Strategy C: Dispatch click event
                                                    try:
                                                        element.evaluate("""
                                                            el => {
                                                                const event = new MouseEvent('click', {
                                                                    view: window,
                                                                    bubbles: true,
                                                                    cancelable: true
                                                                });
                                                                el.dispatchEvent(event);
                                                            }
                                                        """)
                                                        log_fn(f"        • ✓ DispatchEvent succeeded")
                                                        click_success = True
                                                    except Exception as dispatch_err:
                                                        log_fn(f"        • ⚠ DispatchEvent failed: {str(dispatch_err)[:80]}")
                                                
                                                if click_success:
                                                    log_fn(f"      • ✅ Successfully clicked play button with selector: {selector}")
                                                    page.wait_for_timeout(5000)
                                                    
                                                    # Check again for video after click
                                                    post_click_video = page.evaluate("document.querySelectorAll('video').length > 0")
                                                    if post_click_video:
                                                        log_fn(f"      • ✓ Video element appeared after clicking play button!")
                                                        
                                                        # Try to get duration now
                                                        dur = page.evaluate("""
                                                            () => {
                                                                const v = document.querySelector('video');
                                                                return v && v.duration > 0 ? v.duration : null;
                                                            }
                                                        """)
                                                        if dur and dur > (best_dur or 0):
                                                            log_fn(f"        → Got duration after play click: {dur:.1f}s")
                                                            best_dur = dur
                                                            break
                                                    else:
                                                        log_fn(f"      • ⚠ No video element appeared after click")
                                                    
                                                    # If we clicked successfully, break out of element loop
                                                    if best_dur:
                                                        break
                                                else:
                                                    log_fn(f"        • ✗ All click strategies failed")
                                                    
                                        else:
                                            log_fn(f"        • Element {idx+1}: Visible: {is_visible}, Enabled: {is_enabled} - SKIPPING")
                                            
                                    except Exception as e:
                                        log_fn(f"        • ⚠ Error checking element: {str(e)[:80]}")
                                        continue
                                
                                # If we found and clicked a play button, break out of selector loop
                                if best_dur:
                                    break
                                    
                        except Exception as selector_err:
                            log_fn(f"      ⚠ Error with selector {selector}: {str(selector_err)[:80]}")
                            continue
                    
                    # If we found duration, break out of category loop
                    if best_dur:
                        log_fn(f"    • ✅ Found duration from {category} selectors: {best_dur:.1f}s")
                        break
                
                if not best_dur:
                    log_fn("    • ✗ No play button could be clicked successfully")
                    
                    # Additional debug: Try to find any clickable area in video players
                    log_fn("    • 🔍 DEBUG: Looking for any clickable area in video players...")
                    
                    # Try clicking on video player containers
                    player_containers = [
                        ".player", ".video-player", ".video-container", ".media-player",
                        ".plyr", ".video-js", ".jwplayer", ".mejs-container",
                        "[class*='player']", "[id*='player']"
                    ]
                    
                    for container_selector in player_containers:
                        containers = page.query_selector_all(container_selector)
                        if containers:
                            log_fn(f"      • Found {len(containers)} container(s) with selector: {container_selector}")
                            for idx, container in enumerate(containers):
                                if container.is_visible():
                                    log_fn(f"        • Clicking container {idx+1}")
                                    try:
                                        container.click(timeout=5000)
                                        log_fn(f"        • ✓ Container clicked")
                                        page.wait_for_timeout(3000)
                                        
                                        # Check for video after container click
                                        post_click_video = page.evaluate("document.querySelectorAll('video').length > 0")
                                        if post_click_video:
                                            log_fn(f"        • ✓ Video appeared after container click!")
                                            break
                                    except Exception as container_err:
                                        log_fn(f"        • ⚠ Container click failed: {str(container_err)[:80]}")
            
            # Final result
            if best_dur and best_dur > 0:
                log_fn(f"    ✓ Final Playwright duration: {best_dur:.1f}s")
                return float(best_dur)
            elif video_urls and not best_dur:
                # If we captured video URLs but no duration, return a placeholder
                log_fn("    ⚠ Found video URLs but couldn't get duration")
                return None
            else:
                log_fn("    ✗ No reliable duration found with Playwright")
                return None

    except ImportError:
        log_fn("    ⚠ Playwright not installed → pip install playwright && playwright install")
        return None
    except Exception as e:
        log_fn(f"    ✗ Playwright crashed: {type(e).__name__}: {str(e)[:120]}")
        return None
    finally:
        # Safe cleanup (this will run even if an exception occurred)
        try:
            if page:
                page.close()
            if context:
                context.close()
            if browser:
                browser.close()
        except:
            pass

def get_downloaded_videos(directory: str) -> List[str]:
    """Get list of video files in a directory."""
    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".m4v", ".wmv"}
    if not os.path.exists(directory):
        return []
    files = []
    for fn in os.listdir(directory):
        if Path(fn).suffix.lower() in video_extensions:
            files.append(os.path.join(directory, fn))
    return sorted(files)

def download_video(
    url: str,
    save_dir: str,
    log_fn: Callable = print,
    time_range: Optional[Tuple[float, float]] = None,
    download_full: bool = True,
    use_percentages: bool = False,
    process_callback: Optional[Callable] = None,
    video_index: int = 0,
    total_videos: int = 1,
    skip_existing: bool = True,
    use_url_filename: bool = True
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    def is_suspicious_duration(d: Optional[float]) -> bool:
        return d is None or d <= 0 or d < 5
    
    def apply_ffprobe_duration(path: str) -> None:
        if not path or not os.path.exists(path):
            return
        real = get_duration_from_ffprobe(path, log_fn)
        if real and real > 0:
            metadata["duration_real"] = real
            if is_suspicious_duration(metadata.get("duration")):
                metadata["duration"] = real
                log_fn(f"✅ Duration corrected via ffprobe: {real:.1f}s")
    
    metadata: Dict[str, Any] = {
        "url": url,
        "index": video_index,
        "total": total_videos,
        "download_time": None,
        "file_size": 0,
        "duration": None,
        "skipped": False
    }
    
    try:
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. FIRST: Check if video already exists (BEFORE any extraction logic)
        if skip_existing:
            log_fn("🔍 Checking if video already exists...")
           
            # Strategy 1: Try to get title from yt-dlp (fast)
            title_cmd = [
                "yt-dlp",
                "--print", "%(title)s",
                "--no-warnings",
                "--no-playlist",
                "--force-ipv4",
                "--socket-timeout", "10", # Shorter timeout for quick check
                url
            ]
           
            try:
                result = subprocess.run(title_cmd, capture_output=True, text=True, timeout=15, check=False)
                if result.returncode == 0 and (result.stdout or "").strip():
                    title = re.sub(r'[\\/*?:"<>|]', "_", result.stdout.strip())
                   
                    # Check for existing files with this title
                    video_extensions = [".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".m4v", ".wmv"]
                    for ext in video_extensions:
                        candidate = os.path.join(save_dir, f"{video_index:03d} - {title}{ext}")
                        if os.path.exists(candidate):
                            size_mb = os.path.getsize(candidate) / (1024 * 1024)
                            log_fn(f"⏭️ Video already exists: {os.path.basename(candidate)}")
                            log_fn(f"   File size: {size_mb:.2f} MB")
                            log_fn("   Skipping download...")
                            metadata.update({
                                "skipped": True,
                                "filepath": candidate,
                                "file_size": size_mb,
                                "existing": True
                            })
                            # Correct duration for existing file
                            apply_ffprobe_duration(candidate)
                            if process_callback:
                                log_fn("🔄 Running processing callback for existing file...")
                                try:
                                    process_result = process_callback(candidate, metadata)
                                    if process_result:
                                        metadata["processed"] = True
                                        metadata["process_result"] = process_result
                                except Exception as e:
                                    log_fn(f"⚠️ Processing failed for existing file: {e}")
                                    metadata["processed"] = False
                            return True, candidate, metadata
            except Exception as e:
                log_fn(f"⚠️ Quick title check failed: {e}")
                # Continue with other checks
           
            # Strategy 2: Check for URL-based filename
            if use_url_filename:
                url_filename = extract_filename_from_url(url)
                if url_filename:
                    clean_name = re.sub(r'[<>:"/\\|?*]', '_', url_filename)
                   
                    # Check without and with video extensions
                    possible_filenames = [f"{video_index:03d} - {clean_name}"]
                    video_extensions = [".mp4", ".mkv", ".webm", ".avi", ".mov", ".flv", ".m4v", ".wmv"]
                   
                    # Add extensions if not already present
                    for ext in video_extensions:
                        if not clean_name.lower().endswith(ext):
                            possible_filenames.append(f"{video_index:03d} - {clean_name}{ext}")
                   
                    for candidate_name in possible_filenames:
                        candidate_path = os.path.join(save_dir, candidate_name)
                        if os.path.exists(candidate_path):
                            size_mb = os.path.getsize(candidate_path) / (1024 * 1024)
                            log_fn(f"⏭️ Video already exists (URL-based): {os.path.basename(candidate_path)}")
                            log_fn(f"   File size: {size_mb:.2f} MB")
                            log_fn("   Skipping download...")
                            metadata.update({
                                "skipped": True,
                                "filepath": candidate_path,
                                "file_size": size_mb,
                                "existing": True
                            })
                            apply_ffprobe_duration(candidate_path)
                            if process_callback:
                                try:
                                    process_result = process_callback(candidate_path, metadata)
                                    if process_result:
                                        metadata["processed"] = True
                                        metadata["process_result"] = process_result
                                except Exception as e:
                                    log_fn(f"⚠️ Processing failed for existing file: {e}")
                                    metadata["processed"] = False
                            return True, candidate_path, metadata
        
        # 2. Build output template (only if we haven't found existing file)
        if use_url_filename:
            base_filename = get_safe_filename(url, video_index, log_fn)
            output_template = os.path.join(save_dir, base_filename)
            if '.%(ext)s' not in base_filename and '%(ext)s' not in base_filename:
                output_template += '.%(ext)s'
        else:
            output_template = os.path.join(save_dir, f"{video_index:03d} - %(title)s.%(ext)s")
        
        # 3. Determine if we need duration for percentage-based slicing
        needs_duration = bool(time_range and not download_full and use_percentages)
       
        if needs_duration:
            start_pct, end_pct = time_range
            if not (0 <= start_pct <= 100 and 0 <= end_pct <= 100):
                log_fn(f"❌ Invalid percentage range: {start_pct}%-{end_pct}%")
                return False, None, metadata
            if end_pct <= start_pct:
                log_fn("❌ Invalid percentage range: end must be > start")
                return False, None, metadata
            log_fn("📊 Getting video duration to calculate time range...")
            log_fn(f"   Target: {start_pct:.1f}% to {end_pct:.1f}%")
            
            # Get duration
            duration = get_video_duration_advanced(url, log_fn)
            
            if is_suspicious_duration(duration):
                log_fn("⚠️ Duration looks wrong (too small). Trying segment download...")
                duration = get_duration_by_downloading_segment(url, log_fn)
            
            if is_suspicious_duration(duration):
                log_fn("🌐 Trying browser automation...")
                duration = get_duration_with_browser_automation(url, log_fn)
            
            if is_suspicious_duration(duration):
                log_fn("📡 Trying Playwright automation...")
                duration = get_duration_with_playwright_automation(url, log_fn)
                if duration and duration > 0:
                    _duration_method_cache["last_successful"] = {
                        "type": "playwright",
                        "name": "Playwright browser duration",
                        "domain": extract_domain(url)
                    }
                    log_fn(f"  ✓ Playwright gave us duration: {duration:.1f}s")
            
            metadata["duration"] = duration
            
            if is_suspicious_duration(duration):
                log_fn("❌ Could not determine reliable duration for % slicing")
                log_fn("💡 Falling back to full video download")
                download_full = True
                use_percentages = False
            else:
                start_seconds = max(0.0, min((start_pct / 100.0) * duration, max(duration - 1.0, 0.0)))
                end_seconds = max(start_seconds + 1.0, min((end_pct / 100.0) * duration, duration))
                if end_seconds <= start_seconds:
                    log_fn(f"⚠️ Calculated invalid range: {start_seconds:.1f}s to {end_seconds:.1f}s")
                    log_fn("💡 Falling back to full video download")
                    download_full = True
                else:
                    log_fn(f"⏱️ Calculated time range: {start_seconds:.1f}s to {end_seconds:.1f}s")
                    log_fn(f"   ({int(start_pct)}% to {int(end_pct)}% = {end_seconds - start_seconds:.1f}s)")
                    time_range = (start_seconds, end_seconds)
                    use_percentages = False
              
        # 5. Build yt-dlp download command
        cmd = [
            "yt-dlp",
            "-o", output_template,
            "--no-playlist",
            "--no-warnings",
            "--force-ipv4",
            "--socket-timeout", "120",
            "--retries", "10",
            "--fragment-retries", "10",
            "--concurrent-fragments", "4",
        ]
        
        if skip_existing:
            cmd.extend(["--no-overwrites", "--ignore-errors"])
        
        # Add time range
        if time_range and not download_full and not use_percentages:
            start_time, end_time = time_range
            if end_time <= start_time:
                log_fn(f"❌ Invalid time range: {start_time:.1f}s to {end_time:.1f}s")
                return False, None, metadata
            section = f"*{start_time:.1f}-{end_time:.1f}"
            cmd.extend(["--download-sections", section])
            log_fn(f"⏱️ Downloading section: {start_time:.1f}s to {end_time:.1f}s")
            log_fn(f"   Duration: {end_time - start_time:.1f}s")
        elif download_full:
            log_fn("📥 Downloading full video")
        else:
            log_fn("📥 Downloading video (no time range specified)")
        
        cmd.append(url)
        log_fn(f"⬇️ Downloading: {url}")
        log_fn(f"📁 Saving to: {save_dir}")
        
        # 6. Execute download
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
        metadata["download_time"] = time.time() - t0
        out_text = (result.stdout or "") + "\n" + (result.stderr or "")
        
        # Check if yt-dlp reported "already exists"
        if skip_existing and ("already been downloaded" in out_text or "already exists" in out_text):
            log_fn("⏭️ Video already exists (reported by yt-dlp)")
            metadata["skipped"] = True
            # Find the newest video file
            files = get_downloaded_videos(save_dir)
            if files:
                files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                existing = files[0]
                metadata["filepath"] = existing
                metadata["file_size"] = os.path.getsize(existing) / (1024 * 1024)
                apply_ffprobe_duration(existing)
                if process_callback:
                    try:
                        process_result = process_callback(existing, metadata)
                        metadata["processed"] = True
                        metadata["process_result"] = process_result
                    except Exception as e:
                        log_fn(f"⚠️ Processing failed: {e}")
                        metadata["processed"] = False
                return True, existing, metadata
        
        # Handle errors
        if result.returncode != 0:
            log_fn(f"❌ Download failed with exit code: {result.returncode}")
            err = (result.stderr or "").lower()
            if "http error 403" in err:
                log_fn("🔒 HTTP 403: Access forbidden")
            elif "http error 404" in err:
                log_fn("🔍 HTTP 404: Video not found")
            elif "unable to extract" in err:
                log_fn("🔧 Extraction failed. Try: pip install --upgrade yt-dlp")
            if result.stderr:
                for line in result.stderr.strip().split("\n")[:3]:
                    if line.strip():
                        log_fn(f"   {line.strip()}")
            
            # Fallback download
            log_fn("🔄 Trying fallback download method...")
            fallback_cmd = [
                "yt-dlp",
                "-o", output_template,
                "--no-playlist",
                "--format", "best[height<=720]/best",
                "--force-ipv4",
                url
            ]
            if skip_existing:
                fallback_cmd.extend(["--no-overwrites", "--ignore-errors"])
            fr = subprocess.run(fallback_cmd, capture_output=True, text=True, timeout=300, check=False)
            if fr.returncode != 0:
                log_fn("❌ Fallback also failed")
                return False, None, metadata
            log_fn("✅ Fallback download succeeded!")
            result = fr
        
        # Find output file
        filename = None
        for line in (result.stdout or "").split("\n"):
            if "Destination:" in line:
                m = re.search(r"Destination:\s+(.+)", line)
                if m:
                    filename = m.group(1).strip()
                    break
        if not filename:
            log_fn("⚠️ Could not parse filename; finding newest file...")
            files = get_downloaded_videos(save_dir)
            if files:
                files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                filename = files[0]
                log_fn(f"📄 Found newest file: {os.path.basename(filename)}")
        
        if filename and os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            metadata["file_size"] = size_mb
            apply_ffprobe_duration(filename)
            log_fn(f"✅ Downloaded: {os.path.basename(filename)}")
            log_fn(f"📊 File size: {size_mb:.2f} MB")
            log_fn(f"⏱️ Download time: {metadata['download_time']:.1f} seconds")
            if size_mb < 0.1:
                log_fn("⚠️ Warning: File is very small, might be corrupted")
            if process_callback:
                log_fn("🔄 Processing video immediately...")
                try:
                    process_result = process_callback(filename, metadata)
                    if process_result:
                        log_fn("✅ Video processed successfully")
                        metadata["processed"] = True
                        metadata["process_result"] = process_result
                    else:
                        log_fn("⚠️ Processing returned no result")
                        metadata["processed"] = False
                except Exception as e:
                    log_fn(f"❌ Processing failed: {e}")
                    metadata["processed"] = False
                    metadata["process_error"] = str(e)
            return True, filename, metadata
        
        log_fn("❌ Download completed but file not found")
        return False, None, metadata
        
    except subprocess.TimeoutExpired:
        log_fn("⏰ Download timed out")
        metadata["error"] = "timeout"
        return False, None, metadata
    except Exception as e:
        log_fn(f"❌ Unexpected error: {e}")
        metadata["error"] = str(e)
        return False, None, metadata

# -----------------------------
# Batch downloading + processing
# -----------------------------
def download_videos_with_immediate_processing(
    search_url: str,
    save_dir: str,
    pattern: str = "/video/",
    log_fn: Callable = print,
    progress_fn: Optional[Callable] = None,
    process_callback: Optional[Callable] = None,
    cancel_flag=None,
    time_range: Optional[Tuple[float, float]] = None,
    download_full: bool = True,
    use_percentages: bool = False,
    max_workers: int = 1,
    use_url_filenames: bool = True
) -> List[Dict[str, Any]]:
    """
    Sequential downloader with single URL detection.
    """
    os.makedirs(save_dir, exist_ok=True)
    log_fn(f"📁 Save directory: {save_dir}")
    
    # Check if this is a direct video URL (contains /preview/ or common video patterns)
    if "/preview/" in search_url or any(x in search_url for x in ['.mp4', '.m3u8', '/video/']):
        log_fn("🔍 Detected single video URL - using direct download...")
        success, filepath, metadata = download_video(
            search_url,
            save_dir,
            log_fn,
            time_range=time_range,
            download_full=download_full,
            use_percentages=use_percentages,
            process_callback=process_callback,
            video_index=1,
            total_videos=1,
            use_url_filename=use_url_filenames
        )
        metadata["success"] = success
        metadata["filepath"] = filepath
        return [metadata] if success else []
    
    # Rest of the existing code for multiple links...
    try:
        video_links = extract_video_links(search_url, pattern, log_fn)
    except DownloadError as e:
        log_fn(f"❌ {e}")
        return []
    
    total = len(video_links)
    results: List[Dict[str, Any]] = []
    
    for idx, link in enumerate(video_links, start=1):
        if cancel_flag:
            if hasattr(cancel_flag, "is_cancelled") and cancel_flag.is_cancelled():
                log_fn("⏹️ Download cancelled by user")
                break
            if hasattr(cancel_flag, "is_set") and cancel_flag.is_set():
                log_fn("⏹️ Download cancelled by user")
                break
        
        if progress_fn:
            progress_fn(idx - 1, total, "Downloading Videos", f"Video {idx}/{total}")
        
        log_fn(f"\n{'='*60}")
        log_fn(f"[{idx}/{total}] Processing: {link}")
        
        success, filepath, metadata = download_video(
            link,
            save_dir,
            log_fn,
            time_range=time_range,
            download_full=download_full,
            use_percentages=use_percentages,
            process_callback=process_callback,
            video_index=idx,
            total_videos=total,
            use_url_filename=use_url_filenames
        )
        
        metadata["success"] = success
        metadata["filepath"] = filepath
        results.append(metadata)
        
        if success and filepath:
            log_fn(f"✅ Video {idx}/{total} completed")
            if metadata.get("processed"):
                log_fn("✅ Video processed immediately")
        else:
            log_fn(f"❌ Video {idx}/{total} failed")
        
        if progress_fn:
            status = "Processed" if metadata.get("processed") else "Downloaded"
            progress_fn(idx, total, "Downloading Videos", f"{status} {idx}/{total} videos")
    
    log_fn(f"\n{'='*60}")
    successful = sum(1 for r in results if r.get("success"))
    processed = sum(1 for r in results if r.get("processed", False))
    log_fn("📊 Download Summary:")
    log_fn(f"   Total videos: {total}")
    log_fn(f"   Successful downloads: {successful}")
    log_fn(f"   Processed: {processed}")
    log_fn(f"   Save location: {save_dir}")
    
    if progress_fn:
        progress_fn(total, total, "Download Complete", f"Downloaded {successful}/{total} videos")
    
    return results

# -----------------------------
# Example processing callback + test
# -----------------------------
def example_process_callback(filepath: str, metadata: Dict) -> Dict:
    print(f"🔧 Processing video: {os.path.basename(filepath)}")
    time.sleep(1)
    return {
        "processed_at": time.time(),
        "original_size": metadata.get("file_size", 0),
        "status": "success",
    }

def test_downloader():
    import sys
    def test_log(text):
        print(text)
    def test_progress(current, total, status, message):
        print(f"[Progress {current}/{total}] {status}: {message}")
    def mock_process_callback(filepath, metadata):
        print(f"🎬 MOCK PROCESSING: {os.path.basename(filepath)}")
        print(f"   Size: {metadata.get('file_size', 0):.2f} MB")
        print(f"   Download time: {metadata.get('download_time', 0):.1f}s")
        return {"status": "mock_processed"}
    
    if len(sys.argv) > 1:
        url = sys.argv[1]
        save_dir = "test_downloads"
        os.makedirs(save_dir, exist_ok=True)
        print(f"Testing enhanced downloader with URL: {url}")
        print(f"{'='*60}\n")
        success, filepath, metadata = download_video(
            url,
            save_dir,
            test_log,
            time_range=(10, 30),
            download_full=False,
            use_percentages=True,
            process_callback=mock_process_callback,
        )
        print(f"\n{'='*60}")
        print("RESULT:")
        print(f"Success: {success}")
        print(f"File: {filepath}")
        print(f"Metadata: {metadata}")
    else:
        print("Usage: python video_downloader.py <video_url>")

if __name__ == "__main__":
    test_downloader()