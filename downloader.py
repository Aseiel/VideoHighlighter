"""
Video downloader module for fetching videos from websites.

This module provides functions to:
- Extract video links from web pages
- Download videos using yt-dlp
- Progress tracking for GUI integration
"""

import os
import re
import subprocess
import requests
from bs4 import BeautifulSoup
from typing import List, Optional, Callable, Tuple
from pathlib import Path


class DownloadError(Exception):
    """Custom exception for download errors"""
    pass


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
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        raise DownloadError(f"Failed to fetch page: {e}")
    
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract video links
    video_links = []
    base_domain = "/".join(url.split("/")[:3])  # Extract https://domain.com
    
    for a in soup.find_all("a", href=True):
        href = a["href"]
        
        # Check if link matches pattern
        if pattern in href:
            # Make absolute URL if relative
            if href.startswith("/"):
                full_url = f"{base_domain}{href}"
            elif href.startswith("http"):
                full_url = href
            else:
                continue
            
            # Avoid duplicates
            if full_url not in video_links:
                video_links.append(full_url)
    
    log_fn(f"🎬 Found {len(video_links)} video links")
    
    if not video_links:
        log_fn("⚠️ No video links found. Page structure may have changed.")
    
    return video_links


def download_video(url: str, save_dir: str, log_fn: Callable = print) -> Tuple[bool, Optional[str]]:
    """
    Download a single video using yt-dlp.
    
    Args:
        url: Video URL to download
        save_dir: Directory to save video
        log_fn: Logging function
        
    Returns:
        Tuple of (success: bool, filepath: Optional[str])
    """
    try:
        # Prepare yt-dlp command
        output_template = os.path.join(save_dir, "%(title)s.%(ext)s")
        
        cmd = [
            "yt-dlp",
            "-o", output_template,
            "--no-playlist",  # Don't download playlists
            "--no-warnings",  # Suppress warnings
            url
        ]
        
        log_fn(f"⬇️ Downloading: {url}")
        
        # Run yt-dlp
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Try to extract filename from output
        output_lines = result.stdout.split("\n")
        filename = None
        for line in output_lines:
            if "Destination:" in line or "has already been downloaded" in line:
                # Extract filename from yt-dlp output
                match = re.search(r'(?:Destination:|downloaded) (.+\.(?:mp4|mkv|webm|avi))', line)
                if match:
                    filename = match.group(1)
                    break
        
        log_fn(f"✅ Downloaded successfully")
        return True, filename
        
    except subprocess.CalledProcessError as e:
        log_fn(f"❌ Download failed: {e}")
        return False, None
    except Exception as e:
        log_fn(f"❌ Unexpected error: {e}")
        return False, None


def download_videos(
    search_url: str,
    save_dir: str,
    pattern: str = "/video/",
    log_fn: Callable = print,
    progress_fn: Optional[Callable] = None,
    cancel_flag = None
) -> List[str]:
    """
    Download all videos from a search page.
    
    Args:
        search_url: URL of the page containing video links
        save_dir: Directory to save videos
        pattern: Pattern to match video links (default: "/video/")
        log_fn: Logging function
        progress_fn: Progress callback function(current, total, task, details)
        cancel_flag: Threading event for cancellation support
        
    Returns:
        List of successfully downloaded file paths
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
        return []
    
    # Download each video
    downloaded_files = []
    total = len(video_links)
    
    for idx, link in enumerate(video_links, start=1):
        # Check for cancellation
        if cancel_flag and cancel_flag.is_set():
            log_fn("⏹️ Download cancelled by user")
            break
        
        # Update progress
        if progress_fn:
            progress_fn(idx - 1, total, "Downloading Videos", 
                       f"Video {idx}/{total}")
        
        log_fn(f"\n[{idx}/{total}] Processing: {link}")
        
        # Download
        success, filepath = download_video(link, save_dir, log_fn)
        
        if success and filepath:
            downloaded_files.append(filepath)
    
    # Final progress update
    if progress_fn:
        progress_fn(total, total, "Download Complete", 
                   f"Downloaded {len(downloaded_files)}/{total} videos")
    
    log_fn(f"\n{'='*60}")
    log_fn(f"✅ Download complete!")
    log_fn(f"📊 Successfully downloaded: {len(downloaded_files)}/{total}")
    log_fn(f"📁 Location: {save_dir}")
    log_fn(f"{'='*60}")
    
    return downloaded_files


def get_downloaded_videos(directory: str) -> List[str]:
    """
    Get list of video files in a directory.
    
    Args:
        directory: Directory to scan
        
    Returns:
        List of video file paths
    """
    video_extensions = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv'}
    video_files = []
    
    if not os.path.exists(directory):
        return video_files
    
    for file in os.listdir(directory):
        if Path(file).suffix.lower() in video_extensions:
            video_files.append(os.path.join(directory, file))
    
    return sorted(video_files)


# =============================
# CLI Interface
# =============================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download videos from web pages using yt-dlp"
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="URL of the page containing video links"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="D:\\movies",
        help="Directory to save videos (default: D:\\movies)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="/video/",
        help="Pattern to match in video links (default: /video/)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎬 VIDEO DOWNLOADER")
    print("=" * 60)
    print(f"Search URL: {args.url}")
    print(f"Save directory: {args.save_dir}")
    print(f"Link pattern: {args.pattern}")
    print("=" * 60)
    print()
    
    # Run download
    downloaded = download_videos(
        search_url=args.url,
        save_dir=args.save_dir,
        pattern=args.pattern
    )
    
    if downloaded:
        print("\n📋 Downloaded files:")
        for file in downloaded:
            print(f"  • {os.path.basename(file)}")
    else:
        print("\n⚠️ No videos were downloaded")