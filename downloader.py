"""
Video downloader module for fetching videos from websites.

This module provides functions to:
- Extract video links from web pages
- Download videos using yt-dlp
- Progress tracking for GUI integration
- Time range downloads (download only specific portions)
- Percentage-based downloads with fallback methods
- Small-chunk download for duration detection
- HTML metadata parsing for duration
- Immediate processing after each download
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
import threading
import queue
import time


class DownloadError(Exception):
    """Custom exception for download errors"""
    pass


def parse_duration_from_html(html: str) -> Optional[float]:
    """
    Parse duration from common meta tags.
    
    Args:
        html: HTML content as string
        
    Returns:
        Duration in seconds, or None if not found
    """
    soup = BeautifulSoup(html, 'html.parser')
    
    # Common patterns for duration in meta tags
    patterns = [
        {'name': 'meta', 'attrs': {'property': 'og:video:duration'}, 'key': 'content'},
        {'name': 'meta', 'attrs': {'itemprop': 'duration'}, 'key': 'content'},
        {'name': 'meta', 'attrs': {'name': 'duration'}, 'key': 'content'},
        {'name': 'meta', 'attrs': {'property': 'video:duration'}, 'key': 'content'},
        {'name': 'meta', 'attrs': {'name': 'twitter:player:stream:duration'}, 'key': 'content'},
        # Additional patterns for schema.org
        {'name': 'span', 'attrs': {'itemprop': 'duration'}, 'key': 'content'},
        {'name': 'time', 'attrs': {'itemprop': 'duration'}, 'key': 'datetime'},
        {'name': 'meta', 'attrs': {'itemprop': 'duration'}, 'key': 'content', 'iso': True},
    ]
    
    for pat in patterns:
        tag = soup.find(pat['name'], attrs=pat['attrs'])
        if tag and pat['key'] in tag.attrs:
            value = tag[pat['key']]
            if pat.get('iso'):  # Parse ISO 8601 format (e.g., PT5M30S → 330s)
                try:
                    match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', value.upper())
                    if match:
                        h = int(match.group(1) or 0)
                        m = int(match.group(2) or 0)
                        s = int(match.group(3) or 0)
                        duration = h * 3600 + m * 60 + s
                        if duration > 0:
                            return float(duration)
                except (ValueError, AttributeError):
                    pass
            else:
                try:
                    # Try to parse as float
                    duration = float(value)
                    if duration > 0:
                        return duration
                except ValueError:
                    # Try to parse common time formats
                    time_patterns = [
                        r'(\d+):(\d+):(\d+)',  # HH:MM:SS
                        r'(\d+):(\d+)',        # MM:SS
                        r'(\d+)\s*h(?:our(?:s)?)?\s*(\d+)\s*m(?:in(?:ute(?:s)?)?)?\s*(\d+)\s*s(?:ec(?:ond(?:s)?)?)?',  # Xh Ym Zs
                        r'(\d+)\s*m(?:in(?:ute(?:s)?)?)?\s*(\d+)\s*s(?:ec(?:ond(?:s)?)?)?',  # Xm Ys
                    ]
                    
                    for pattern in time_patterns:
                        match = re.search(pattern, value, re.IGNORECASE)
                        if match:
                            groups = match.groups()
                            if len(groups) == 3:  # HH:MM:SS or Xh Ym Zs
                                try:
                                    h, m, s = map(int, groups)
                                    duration = h * 3600 + m * 60 + s
                                    if duration > 0:
                                        return float(duration)
                                except (ValueError, TypeError):
                                    continue
                            elif len(groups) == 2:  # MM:SS or Xm Ys
                                try:
                                    m, s = map(int, groups)
                                    duration = m * 60 + s
                                    if duration > 0:
                                        return float(duration)
                                except (ValueError, TypeError):
                                    continue
    return None


def extract_video_links(url: str, pattern: str = "/video/", log_fn: Callable = print) -> List[str]:
    """
    Extract video links from a webpage.
    
    Args:
        url: The webpage URL to scrape
        pattern: Pattern to match in href attributes (default: "/video/")
        log_fn: Logging function (default: print)
        
    Returns:
        List of full video URLs
        
    Raises:
        DownloadError: If page fetch fails
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
    
    # Try to parse duration from HTML (we have the response already)
    html_duration = parse_duration_from_html(response.text)
    if html_duration:
        log_fn(f"⏱️ Found duration in HTML metadata: {html_duration:.1f}s")
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract video links
    video_links = []
    
    # Try multiple strategies to find video links
    # Strategy 1: Look for <a> tags with pattern
    for a in soup.find_all("a", href=True):
        href = a["href"]
        
        # Check if link matches pattern
        if pattern in href:
            full_url = make_absolute_url(href, url)
            if full_url and full_url not in video_links:
                video_links.append(full_url)
    
    # Strategy 2: Look for video tags
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
    
    # Strategy 3: Look for iframes
    if not video_links:
        for iframe in soup.find_all("iframe"):
            if iframe.get("src"):
                src = iframe["src"]
                # Check if it's a video iframe (YouTube, Vimeo, etc.)
                if any(domain in src for domain in ["youtube.com", "youtu.be", "vimeo.com", "dailymotion.com"]):
                    if src not in video_links:
                        video_links.append(src)
    
    log_fn(f"🎬 Found {len(video_links)} video links")
    
    if not video_links:
        log_fn("⚠️ No video links found. Page structure may have changed.")
        log_fn("💡 Try a different pattern or check the page manually")
    
    return video_links


def make_absolute_url(url: str, base_url: str) -> Optional[str]:
    """Convert relative URL to absolute URL"""
    if not url:
        return None
    
    # Already absolute
    if url.startswith(("http://", "https://")):
        return url
    
    # Parse base URL
    try:
        parsed_base = urllib.parse.urlparse(base_url)
        
        # Handle relative URLs
        if url.startswith("//"):
            # Protocol-relative URL
            return f"{parsed_base.scheme}:{url}"
        elif url.startswith("/"):
            # Root-relative URL
            return f"{parsed_base.scheme}://{parsed_base.netloc}{url}"
        else:
            # Path-relative URL
            path = parsed_base.path.rsplit("/", 1)[0] if "/" in parsed_base.path else ""
            return f"{parsed_base.scheme}://{parsed_base.netloc}{path}/{url}"
    except Exception:
        return None


def get_video_duration_advanced(url: str, log_fn: Callable = print) -> Optional[float]:
    """
    Advanced method to get video duration using multiple strategies.
    
    Args:
        url: Video URL
        log_fn: Logging function
        
    Returns:
        Duration in seconds, or None if cannot determine
    """
    def extract_domain(url: str) -> str:
        """Extract domain from URL for referer"""
        try:
            parsed = urllib.parse.urlparse(url)
            return f"{parsed.scheme}://{parsed.netloc}"
        except:
            return "https://example.com"
    
    def parse_duration_from_json(json_output: str) -> Optional[float]:
        """Parse duration from JSON output"""
        try:
            data = json.loads(json_output.strip())
            duration_fields = ['duration', 'approx_duration', 'length', 'length_seconds']
            for field in duration_fields:
                if field in data and data[field] is not None:
                    try:
                        duration = float(data[field])
                        if duration > 0:
                            return duration
                    except (ValueError, TypeError):
                        continue
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return None
    
    # New: Try to get duration from HTML first (if URL is a webpage)
    try:
        log_fn("🔍 Checking HTML metadata for duration...")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            html_duration = parse_duration_from_html(response.text)
            if html_duration:
                log_fn(f"✅ Got duration from HTML metadata: {html_duration:.1f}s")
                return html_duration
    except Exception as e:
        log_fn(f"⚠️ HTML metadata check failed: {str(e)[:50]}...")
    
    duration_methods = [
        # Method 1: Standard JSON with playlist handling
        {
            "name": "Standard JSON",
            "cmd": ["yt-dlp", "--dump-json", "--no-warnings", "--playlist-end", "1"],
            "parser": lambda output: parse_duration_from_json(output)
        },
        
        # Method 2: With network optimizations
        {
            "name": "Network optimized",
            "cmd": ["yt-dlp", "--dump-json", "--no-warnings", "--force-ipv4", 
                   "--socket-timeout", "30", "--retries", "3"],
            "parser": lambda output: parse_duration_from_json(output)
        },
        
        # Method 3: Simple print method
        {
            "name": "Simple print",
            "cmd": ["yt-dlp", "--print", "%(duration)s", "--no-warnings"],
            "parser": lambda output: float(output.strip()) if output.strip() 
                       and output.strip() not in ["NA", "None", ""] else None
        },
        
        # Method 4: With referer and cookies
        {
            "name": "With referer",
            "cmd": ["yt-dlp", "--dump-json", "--no-warnings", 
                   "--referer", extract_domain(url),
                   "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"],
            "parser": lambda output: parse_duration_from_json(output)
        },
        
        # Method 5: No certificate check
        {
            "name": "No cert check",
            "cmd": ["yt-dlp", "--dump-json", "--no-warnings", "--no-check-certificate"],
            "parser": lambda output: parse_duration_from_json(output)
        },
        
        # Method 6: Try to follow redirects
        {
            "name": "Follow redirects",
            "cmd": ["yt-dlp", "--dump-json", "--no-warnings", 
                   "--playlist-end", "1", "--socket-timeout", "45"],
            "parser": lambda output: parse_duration_from_json(output)
        }
    ]
    
    for method in duration_methods:
        try:
            log_fn(f"📊 Trying duration method: {method['name']}...")
            
            cmd = method["cmd"] + [url]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=45,
                check=False  # Don't raise on non-zero exit
            )
            
            if result.returncode == 0 and result.stdout.strip():
                duration = method["parser"](result.stdout)
                if duration and duration > 0:
                    log_fn(f"✅ Got duration via {method['name']}: {duration:.1f}s")
                    return duration
                    
        except subprocess.TimeoutExpired:
            log_fn(f"⏰ {method['name']} timed out")
            continue
        except Exception as e:
            log_fn(f"⚠️ {method['name']} failed: {str(e)[:80]}...")
            continue
    
    log_fn("🔍 All direct methods failed, trying fallback methods...")
    return None


def get_duration_by_downloading_segment(url: str, log_fn: Callable = print) -> Optional[float]:
    """
    Download a small segment to determine video duration.
    This is a fallback method when direct metadata extraction fails.
    
    Args:
        url: Video URL
        log_fn: Logging function
        
    Returns:
        Duration in seconds, or None if cannot determine
    """
    try:
        log_fn("📥 Downloading small segment to get duration...")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create output template
            output_template = os.path.join(tmpdir, "temp_segment.%(ext)s")
            
            # Try to download first 2 seconds (very small segment)
            cmd = [
                "yt-dlp",
                "-o", output_template,
                "--download-sections", "*0-2",  # Download first 2 seconds
                "--dump-json",
                "--no-warnings",
                "--no-playlist",
                "--force-ipv4",
                "--socket-timeout", "60",
                "--retries", "3",
                "--fragment-retries", "5",
                url
            ]
            
            # Run with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=90,  # Longer timeout for download
                check=False
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # Parse JSON output
                video_info = json.loads(result.stdout.strip())
                
                # Look for duration in multiple possible fields
                duration_fields = ['duration', 'approx_duration', 'length', 'length_seconds']
                for field in duration_fields:
                    if field in video_info and video_info[field]:
                        try:
                            duration = float(video_info[field])
                            if duration > 0:
                                log_fn(f"✅ Got duration from segment download: {duration:.1f}s")
                                return duration
                        except (ValueError, TypeError):
                            continue
                
                # Alternative: try to get duration from format info
                if 'formats' in video_info and video_info['formats']:
                    for fmt in video_info['formats']:
                        if 'duration' in fmt and fmt['duration']:
                            try:
                                duration = float(fmt['duration'])
                                if duration > 0:
                                    log_fn(f"✅ Got duration from format info: {duration:.1f}s")
                                    return duration
                            except (ValueError, TypeError):
                                continue
            
            # If JSON parsing failed, try to get duration another way
            log_fn("⚠️ Could not parse duration from segment download, trying alternative...")
            
            # Alternative: Use yt-dlp's info extractor directly
            info_cmd = [
                "yt-dlp",
                "--print", "%(duration)s",
                "--no-warnings",
                "--force-ipv4",
                "--socket-timeout", "60",
                url
            ]
            
            result = subprocess.run(
                info_cmd,
                capture_output=True,
                text=True,
                timeout=60,
                check=False
            )
            
            if result.returncode == 0 and result.stdout.strip():
                output = result.stdout.strip()
                if output not in ["NA", "None", ""]:
                    try:
                        duration = float(output)
                        if duration > 0:
                            log_fn(f"✅ Got duration from info extractor: {duration:.1f}s")
                            return duration
                    except ValueError:
                        pass
                        
    except subprocess.TimeoutExpired:
        log_fn("⏰ Segment download timed out")
    except json.JSONDecodeError:
        log_fn("⚠️ Failed to parse JSON from segment download")
    except Exception as e:
        log_fn(f"⚠️ Segment download failed: {str(e)[:100]}...")
    
    return None


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
    skip_existing: bool = True
) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Download a single video using yt-dlp with optional time range.
    Improved version with better duration detection and fallbacks.
    
    Args:
        url: Video URL to download
        save_dir: Directory to save video
        log_fn: Logging function
        time_range: Tuple of (start, end) in seconds OR percentages (0-100)
        download_full: If True, download full video ignoring time_range
        use_percentages: If True, time_range is in percentages (0-100)
        process_callback: Callback function to process video immediately after download
        video_index: Index of current video (for progress reporting)
        total_videos: Total number of videos to download
        skip_existing: If True, skip download if file already exists (default: True)
        
    Returns:
        Tuple of (success: bool, filepath: Optional[str], metadata: Dict)
    """
    metadata = {
        'url': url,
        'index': video_index,
        'total': total_videos,
        'download_time': None,
        'file_size': 0,
        'duration': None,
        'skipped': False
    }
    
    try:
        # NEW: Check if file already exists before downloading
        if skip_existing:
            log_fn(f"🔍 Checking if video already exists...")
            
            # First try to get video title/info to predict filename
            info_cmd = [
                "yt-dlp",
                "--print", "%(title)s",
                "--no-warnings",
                "--force-ipv4",
                "--socket-timeout", "30",
                url
            ]
            
            result = subprocess.run(
                info_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                check=False
            )
            
            video_title = None
            if result.returncode == 0 and result.stdout.strip():
                video_title = result.stdout.strip()
                # Sanitize filename
                video_title = re.sub(r'[\\/*?:"<>|]', "_", video_title)
                
                # Check for common video extensions
                video_extensions = ['.mp4', '.mkv', '.webm', '.avi', '.mov', '.flv', '.m4v', '.wmv']
                
                for ext in video_extensions:
                    potential_file = os.path.join(save_dir, f"{video_title}{ext}")
                    if os.path.exists(potential_file):
                        file_size = os.path.getsize(potential_file) / (1024 * 1024)  # MB
                        log_fn(f"⏭️ Video already exists: {os.path.basename(potential_file)}")
                        log_fn(f"   File size: {file_size:.2f} MB")
                        log_fn(f"   Skipping download...")
                        
                        # Update metadata
                        metadata['skipped'] = True
                        metadata['filepath'] = potential_file
                        metadata['file_size'] = file_size
                        metadata['existing'] = True
                        
                        # Check if processing is still needed
                        if process_callback:
                            log_fn("🔄 Running processing callback for existing file...")
                            try:
                                process_result = process_callback(potential_file, metadata)
                                if process_result:
                                    log_fn("✅ Existing file processed")
                                    metadata['processed'] = True
                                    metadata['process_result'] = process_result
                            except Exception as e:
                                log_fn(f"⚠️ Processing failed for existing file: {e}")
                                metadata['processed'] = False
                        
                        return True, potential_file, metadata
        
        # Create output template
        output_template = os.path.join(save_dir, "%(title)s.%(ext)s")
        
        # Track if we need to convert percentages to time range
        needs_duration = (time_range and not download_full and use_percentages)
        
        # Step 1: Get video duration if using percentages
        if needs_duration:
            start_pct, end_pct = time_range
            
            # Validate percentages
            if not (0 <= start_pct <= 100 and 0 <= end_pct <= 100):
                log_fn(f"❌ Invalid percentage range: {start_pct}%-{end_pct}%")
                return False, None, metadata
            
            if end_pct <= start_pct:
                log_fn(f"❌ Invalid percentage range: end must be > start")
                return False, None, metadata
            
            log_fn(f"📊 Getting video duration to calculate time range...")
            log_fn(f"   Target: {start_pct:.1f}% to {end_pct:.1f}%")
            
            # Try to get duration using multiple methods
            duration = None
            
            # Method 1: Check if URL is a webpage with HTML metadata
            try:
                # Only check if it looks like a webpage URL (not direct video)
                if not any(url.lower().endswith(ext) for ext in ['.mp4', '.webm', '.avi', '.mov', '.mkv', '.flv']):
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    }
                    response = requests.get(url, headers=headers, timeout=10)
                    if response.status_code == 200:
                        duration = parse_duration_from_html(response.text)
                        if duration:
                            log_fn(f"✅ Got duration from HTML metadata: {duration:.1f}s")
            except Exception:
                pass  # Fall through to other methods
            
            # Method 2: Advanced duration detection
            if duration is None or duration <= 0:
                duration = get_video_duration_advanced(url, log_fn)
            
            # Method 3: Fallback to downloading small segment
            if duration is None or duration <= 0:
                log_fn("⚠️ Direct methods failed, trying segment download fallback...")
                duration = get_duration_by_downloading_segment(url, log_fn)
            
            # Method 4: Last resort - try with verbose output
            if duration is None or duration <= 0:
                log_fn("🔍 Trying verbose info extraction...")
                try:
                    cmd = [
                        "yt-dlp",
                        "--print", "%(duration)s",
                        "--verbose",
                        "--no-warnings",
                        url
                    ]
                    
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=45,
                        check=False
                    )
                    
                    # Look for duration in stderr (verbose output)
                    if result.stderr:
                        for line in result.stderr.split('\n'):
                            if 'duration' in line.lower():
                                match = re.search(r'(\d+\.?\d*)\s*(?:seconds?|sec|s)', line.lower())
                                if match:
                                    duration = float(match.group(1))
                                    log_fn(f"📏 Found duration in verbose output: {duration:.1f}s")
                                    break
                    
                    # Check stdout too
                    if (duration is None or duration <= 0) and result.stdout.strip():
                        output = result.stdout.strip()
                        if output not in ["NA", "None", ""]:
                            try:
                                duration = float(output)
                                if duration > 0:
                                    log_fn(f"📏 Found duration in stdout: {duration:.1f}s")
                            except ValueError:
                                pass
                                
                except Exception as e:
                    log_fn(f"⚠️ Verbose method failed: {e}")
            
            metadata['duration'] = duration
            
            # If we still don't have duration, fall back to full download
            if duration is None or duration <= 0:
                log_fn("❌ Could not determine video duration")
                log_fn("💡 Falling back to full video download")
                download_full = True
                use_percentages = False
            else:
                # Calculate actual time range from percentages
                # Ensure percentages don't exceed video length
                start_seconds = max(0, min((start_pct / 100.0) * duration, duration - 1))
                end_seconds = max(start_seconds + 1, min((end_pct / 100.0) * duration, duration))
                
                # Validate calculated range
                if end_seconds <= start_seconds:
                    log_fn(f"⚠️ Calculated invalid range: {start_seconds:.1f}s to {end_seconds:.1f}s")
                    log_fn("💡 Falling back to full video download")
                    download_full = True
                else:
                    log_fn(f"⏱️ Calculated time range: {start_seconds:.1f}s to {end_seconds:.1f}s")
                    log_fn(f"   ({int(start_pct)}% to {int(end_pct)}% = {end_seconds - start_seconds:.1f}s)")
                    
                    # Convert to time range in seconds for download
                    time_range = (start_seconds, end_seconds)
                    use_percentages = False  # Now we have actual seconds
        
        # Step 2: Build download command
        cmd = [
            "yt-dlp",
            "-o", output_template,
            "--no-playlist",
            "--no-warnings",
            "--force-ipv4",  # Add network optimization
            "--socket-timeout", "120",  # Longer timeout for downloads
            "--retries", "10",
            "--fragment-retries", "10",
            "--concurrent-fragments", "4",  # Parallel download for faster speed
        ]
        
        # Add skip existing check for yt-dlp as well
        if skip_existing:
            cmd.extend(["--no-overwrites", "--ignore-errors"])
        
        # Add time range if specified (and we're not falling back to full)
        if time_range and not download_full and not use_percentages:
            start_time, end_time = time_range
            
            # Validate time range
            if end_time <= start_time:
                log_fn(f"❌ Invalid time range: {start_time:.1f}s to {end_time:.1f}s")
                return False, None, metadata
            
            # yt-dlp --download-sections format: "*start_seconds-end_seconds"
            section = f"*{start_time:.1f}-{end_time:.1f}"
            cmd.extend(["--download-sections", section])
            
            duration_dl = end_time - start_time
            log_fn(f"⏱️ Downloading section: {start_time:.1f}s to {end_time:.1f}s")
            log_fn(f"   Duration: {duration_dl:.1f}s")
            log_fn(f"🔧 yt-dlp format: {section}")
        elif download_full:
            log_fn("📥 Downloading full video")
        else:
            log_fn("📥 Downloading video (no time range specified)")
        
        # Add the URL
        cmd.append(url)
        
        log_fn(f"⬇️ Downloading: {url}")
        log_fn(f"📁 Saving to: {save_dir}")
        
        # Run yt-dlp
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for download
            check=False  # Don't raise on error, we'll handle it
        )
        download_time = time.time() - start_time
        metadata['download_time'] = download_time
        
        # Check if download was skipped (yt-dlp might report it)
        output_text = result.stdout + result.stderr
        if skip_existing and ("already been downloaded" in output_text or "already exists" in output_text):
            log_fn("⏭️ Video already exists (reported by yt-dlp)")
            metadata['skipped'] = True
            
            # Try to find the existing file
            if os.path.exists(save_dir):
                # Get list of video files
                video_files = get_downloaded_videos(save_dir)
                if video_files:
                    # Get the most recently modified video file
                    video_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    existing_file = video_files[0]
                    metadata['filepath'] = existing_file
                    metadata['file_size'] = os.path.getsize(existing_file) / (1024 * 1024)
                    
                    # Process if needed
                    if process_callback:
                        try:
                            process_result = process_callback(existing_file, metadata)
                            metadata['processed'] = True
                            metadata['process_result'] = process_result
                        except Exception as e:
                            log_fn(f"⚠️ Processing failed: {e}")
                            metadata['processed'] = False
                    
                    return True, existing_file, metadata
        
        # Check for errors
        if result.returncode != 0:
            log_fn(f"❌ Download failed with exit code: {result.returncode}")
            
            # Try to provide helpful error messages
            error_msg = result.stderr.lower()
            if "http error 403" in error_msg:
                log_fn("🔒 HTTP 403: Access forbidden. The website might be blocking downloads.")
            elif "http error 404" in error_msg:
                log_fn("🔍 HTTP 404: Video not found. The URL might be invalid.")
            elif "unable to extract" in error_msg:
                log_fn("🔧 Extraction failed. yt-dlp might need an update for this website.")
                log_fn("💡 Try: pip install --upgrade yt-dlp")
            elif "unsupported url" in error_msg:
                log_fn("🌐 Unsupported URL. This website might not be supported by yt-dlp.")
            elif "this video is only available" in error_msg:
                log_fn("🔞 Video might require age verification or login.")
            elif "sign in" in error_msg or "login" in error_msg:
                log_fn("🔐 Authentication required. This video might need cookies or login.")
            elif "private video" in error_msg:
                log_fn("🔒 Private video. This video is not publicly accessible.")
            elif "georestricted" in error_msg or "country" in error_msg:
                log_fn("🌍 Geo-restricted. This video is not available in your country.")
            elif "copyright" in error_msg:
                log_fn("©️ Copyright issue. This video might be blocked due to copyright.")
            elif "format not available" in error_msg:
                log_fn("🎬 Format not available. Try a different format with --format option.")
            
            if result.stderr:
                # Show first 3 lines of error
                error_lines = result.stderr.strip().split('\n')
                for line in error_lines[:3]:
                    if line.strip():
                        log_fn(f"   {line.strip()}")
            
            # Try fallback with simpler command
            log_fn("🔄 Trying fallback download method...")
            fallback_cmd = [
                "yt-dlp",
                "-o", output_template,
                "--no-playlist",
                "--format", "best[height<=720]/best",  # Try for 720p or best
                "--force-ipv4",
                url
            ]
            
            if skip_existing:
                fallback_cmd.extend(["--no-overwrites", "--ignore-errors"])
            
            try:
                fallback_result = subprocess.run(
                    fallback_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300,
                    check=False
                )
                
                if fallback_result.returncode == 0:
                    result = fallback_result  # Use fallback result
                    log_fn("✅ Fallback download succeeded!")
                else:
                    log_fn(f"❌ Fallback also failed: {fallback_result.stderr[:200]}...")
                    return False, None, metadata
                    
            except Exception as fallback_error:
                log_fn(f"❌ Fallback also failed: {fallback_error}")
                return False, None, metadata
        
        # Try to extract filename from output
        filename = None
        output_lines = result.stdout.split("\n")
        
        for line in output_lines:
            if "Destination:" in line:
                match = re.search(r'Destination:\s+(.+)', line)
                if match:
                    filename = match.group(1).strip()
                    break
            elif "has already been downloaded" in line:
                match = re.search(r'(.+\.(?:mp4|mkv|webm|avi|flv|mov|m4v|wmv))', line)
                if match:
                    filename = match.group(1).strip()
                    break
        
        # If we couldn't parse the filename, try to find the newest file
        if not filename:
            log_fn("⚠️ Could not parse filename from output, finding newest file...")
            try:
                video_files = get_downloaded_videos(save_dir)
                
                if video_files:
                    # Sort by modification time, newest first
                    video_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                    filename = video_files[0]
                    log_fn(f"📄 Found newest file: {os.path.basename(filename)}")
                else:
                    log_fn("⚠️ No video files found in directory")
                    
            except Exception as e:
                log_fn(f"⚠️ Could not find downloaded file: {e}")
        
        if filename and os.path.exists(filename):
            file_size = os.path.getsize(filename) / (1024 * 1024)  # Convert to MB
            metadata['file_size'] = file_size
            log_fn(f"✅ Downloaded successfully: {os.path.basename(filename)}")
            log_fn(f"📊 File size: {file_size:.2f} MB")
            log_fn(f"⏱️ Download time: {download_time:.1f} seconds")
            
            # Check if file has content (not empty)
            if file_size < 0.1:  # Less than 100KB
                log_fn("⚠️ Warning: Downloaded file is very small, might be corrupted")
            
            # Call process callback if provided
            if process_callback:
                log_fn("🔄 Processing video immediately...")
                try:
                    process_result = process_callback(filename, metadata)
                    if process_result:
                        log_fn("✅ Video processed successfully")
                        metadata['processed'] = True
                        metadata['process_result'] = process_result
                    else:
                        log_fn("⚠️ Video processing returned no result")
                        metadata['processed'] = False
                except Exception as e:
                    log_fn(f"❌ Video processing failed: {e}")
                    import traceback
                    log_fn(f"Traceback: {traceback.format_exc()[:500]}...")
                    metadata['processed'] = False
                    metadata['process_error'] = str(e)
            
            return True, filename, metadata
        else:
            log_fn("❌ Download completed but file not found")
            return False, None, metadata
        
    except subprocess.TimeoutExpired:
        log_fn("⏰ Download timed out")
        metadata['error'] = 'timeout'
        return False, None, metadata
    except Exception as e:
        log_fn(f"❌ Unexpected error: {e}")
        import traceback
        log_fn(f"Traceback: {traceback.format_exc()[:200]}...")
        metadata['error'] = str(e)
        return False, None, metadata


def download_videos_with_immediate_processing(
    search_url: str,
    save_dir: str,
    pattern: str = "/video/",
    log_fn: Callable = print,
    progress_fn: Optional[Callable] = None,
    process_callback: Optional[Callable] = None,
    cancel_flag = None,
    time_range: Optional[Tuple[float, float]] = None,
    download_full: bool = True,
    use_percentages: bool = False,
    max_workers: int = 1
) -> List[Dict[str, Any]]:
    """
    Download all videos from a search page with immediate processing after each download.
    
    Args:
        search_url: URL of the page containing video links
        save_dir: Directory to save downloaded videos
        pattern: Pattern to match in video URLs
        log_fn: Logging function
        progress_fn: Function to update progress (receives current, total, status, message)
        process_callback: Function to process video immediately after download
                         Called with (filepath, metadata)
        cancel_flag: Flag to check for cancellation (QThread or threading.Event)
        time_range: Time range in seconds or percentages
        download_full: Whether to download full videos
        use_percentages: Whether time_range is in percentages
        max_workers: Maximum number of concurrent downloads (default: 1 for sequential)
        
    Returns:
        List of dictionaries with download results for each video
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    log_fn(f"📁 Save directory: {save_dir}")
    
    # Extract video links
    try:
        video_links = extract_video_links(search_url, pattern, log_fn)
    except DownloadError as e:
        log_fn(f"❌ {e}")
        return []
    
    if not video_links:
        log_fn("⚠️ No video links found")
        return []
    
    total = len(video_links)
    results = []
    
    if max_workers > 1:
        # Concurrent downloads with processing
        return download_videos_concurrent(
            video_links, save_dir, log_fn, progress_fn, 
            process_callback, cancel_flag, time_range, 
            download_full, use_percentages, max_workers
        )
    
    # Sequential downloads with immediate processing
    for idx, link in enumerate(video_links, start=1):
        # Check for cancellation
        if cancel_flag:
            if hasattr(cancel_flag, 'is_cancelled') and cancel_flag.is_cancelled():
                log_fn("⏹️ Download cancelled by user")
                break
            elif hasattr(cancel_flag, 'is_set') and cancel_flag.is_set():
                log_fn("⏹️ Download cancelled by user")
                break
        
        # Update progress
        if progress_fn:
            progress_fn(idx - 1, total, "Downloading Videos", 
                       f"Video {idx}/{total}: {os.path.basename(link[:50])}...")
        
        log_fn(f"\n{'='*60}")
        log_fn(f"[{idx}/{total}] Processing: {link}")
        
        # Download with immediate processing
        success, filepath, metadata = download_video(
            link, 
            save_dir, 
            log_fn,
            time_range=time_range,
            download_full=download_full,
            use_percentages=use_percentages,
            process_callback=process_callback,
            video_index=idx,
            total_videos=total
        )
        
        metadata['success'] = success
        metadata['filepath'] = filepath
        results.append(metadata)
        
        if success and filepath:
            log_fn(f"✅ Video {idx}/{total} completed successfully")
            if metadata.get('processed'):
                log_fn("✅ Video processed immediately after download")
        else:
            log_fn(f"❌ Video {idx}/{total} failed")
        
        # Update progress after processing
        if progress_fn:
            status = "Processed" if metadata.get('processed') else "Downloaded"
            progress_fn(idx, total, "Downloading Videos", 
                       f"{status} {idx}/{total} videos")
    
    # Final summary
    log_fn(f"\n{'='*60}")
    successful = sum(1 for r in results if r.get('success'))
    processed = sum(1 for r in results if r.get('processed', False))
    
    log_fn(f"📊 Download Summary:")
    log_fn(f"   Total videos: {total}")
    log_fn(f"   Successful downloads: {successful}")
    log_fn(f"   Immediate processing: {processed}")
    log_fn(f"   Save location: {save_dir}")
    
    if progress_fn:
        progress_fn(total, total, "Download Complete", 
                   f"Downloaded {successful}/{total} videos")
    
    return results


def download_videos_concurrent(
    video_links: List[str],
    save_dir: str,
    log_fn: Callable,
    progress_fn: Optional[Callable],
    process_callback: Optional[Callable],
    cancel_flag,
    time_range: Optional[Tuple[float, float]],
    download_full: bool,
    use_percentages: bool,
    max_workers: int
) -> List[Dict[str, Any]]:
    """
    Download videos concurrently with immediate processing.
    """
    import concurrent.futures
    from concurrent.futures import ThreadPoolExecutor
    
    total = len(video_links)
    results = []
    completed = 0
    
    def download_single_video(link_idx: Tuple[int, str]) -> Dict[str, Any]:
        """Download a single video with its index"""
        idx, link = link_idx
        
        # Check for cancellation
        if cancel_flag:
            if hasattr(cancel_flag, 'is_cancelled') and cancel_flag.is_cancelled():
                return {'index': idx, 'url': link, 'cancelled': True}
            elif hasattr(cancel_flag, 'is_set') and cancel_flag.is_set():
                return {'index': idx, 'url': link, 'cancelled': True}
        
        log_fn(f"\n[Thread {idx}/{total}] Downloading: {link}")
        
        success, filepath, metadata = download_video(
            link, 
            save_dir, 
            lambda msg: log_fn(f"[Thread {idx}] {msg}"),
            time_range=time_range,
            download_full=download_full,
            use_percentages=use_percentages,
            process_callback=process_callback,
            video_index=idx,
            total_videos=total
        )
        
        metadata['success'] = success
        metadata['filepath'] = filepath
        
        # Update progress
        nonlocal completed
        completed += 1
        if progress_fn:
            status = "Processed" if metadata.get('processed') else "Downloaded"
            progress_fn(completed, total, "Downloading Videos", 
                       f"{status} {completed}/{total} videos")
        
        return metadata
    
    log_fn(f"🚀 Starting concurrent downloads with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Create list of (index, url) pairs
        tasks = list(enumerate(video_links, start=1))
        
        # Submit all tasks
        future_to_idx = {executor.submit(download_single_video, task): task[0] 
                        for task in tasks}
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result(timeout=1)
                results.append(result)
            except Exception as e:
                log_fn(f"❌ Error in video {idx}: {e}")
                results.append({
                    'index': idx,
                    'url': video_links[idx-1],
                    'success': False,
                    'error': str(e)
                })
    
    return sorted(results, key=lambda x: x.get('index', 0))


def get_downloaded_videos(directory: str) -> List[str]:
    """
    Get list of video files in a directory.
    
    Args:
        directory: Directory to scan
        
    Returns:
        List of video file paths
    """
    video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.m4v', '.wmv'}
    video_files = []
    
    if not os.path.exists(directory):
        return video_files
    
    for file in os.listdir(directory):
        if Path(file).suffix.lower() in video_extensions:
            video_files.append(os.path.join(directory, file))
    
    return sorted(video_files)


# Example processing callback function
def example_process_callback(filepath: str, metadata: Dict) -> Dict:
    """
    Example callback function for processing videos immediately after download.
    
    Args:
        filepath: Path to the downloaded video file
        metadata: Dictionary with video metadata
        
    Returns:
        Dictionary with processing results
    """
    import time
    
    print(f"🔧 Processing video: {os.path.basename(filepath)}")
    
    # Simulate some processing (e.g., transcoding, analysis, etc.)
    time.sleep(2)  # Simulate processing time
    
    # Example processing results
    return {
        'processed_at': time.time(),
        'original_size': metadata.get('file_size', 0),
        'processed_file': filepath.replace('.mp4', '_processed.mp4'),
        'status': 'success'
    }


# Test function for development
def test_immediate_processing():
    """Test function for immediate processing after download"""
    import sys
    
    def test_log(text):
        print(text)
    
    def test_progress(current, total, status, message):
        print(f"[Progress {current}/{total}] {status}: {message}")
    
    def mock_process_callback(filepath, metadata):
        print(f"🎬 MOCK PROCESSING: {os.path.basename(filepath)}")
        print(f"   Size: {metadata.get('file_size', 0):.2f} MB")
        print(f"   Download time: {metadata.get('download_time', 0):.1f}s")
        return {'status': 'mock_processed'}
    
    if len(sys.argv) > 1:
        url = sys.argv[1]
        save_dir = "test_downloads"
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Testing immediate processing with URL: {url}")
        
        # Test with immediate processing
        results = download_videos_with_immediate_processing(
            url,
            save_dir,
            log_fn=test_log,
            progress_fn=test_progress,
            process_callback=mock_process_callback,
            time_range=(10, 30),  # 10% to 30%
            download_full=False,
            use_percentages=True,
            max_workers=2  # Try concurrent downloads
        )
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS:")
        for result in results:
            idx = result.get('index', '?')
            url = result.get('url', 'unknown')
            success = result.get('success', False)
            processed = result.get('processed', False)
            print(f"Video {idx}: {'✅' if success else '❌'} "
                  f"{'🔧' if processed else '📥'} "
                  f"{url[:50]}...")
            
    else:
        print("Usage: python downloader.py <url>")


if __name__ == "__main__":
    test_immediate_processing()