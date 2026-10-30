"""The browser-session download: what it accepts as a result, and which
browser it runs. Needs the downloader's own imports (selenium, bs4)."""

from __future__ import annotations

import os
import subprocess

import pytest

pytest.importorskip("selenium")
pytest.importorskip("bs4")

import downloader  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_a_playlist_saved_as_mp4_is_not_a_download_that_worked(tmp_path):
    """What happened: the in-session step fetched the .m3u8 itself, saved those
    few hundred bytes as .mp4 and reported success with a 0.0 MB file."""
    fake = tmp_path / "001 - clip.mp4"
    fake.write_text("#EXTM3U\n#EXT-X-STREAM-INF:BANDWIDTH=1\nindex.m3u8\n")
    said = []

    assert not downloader._check_video_file(str(fake), said.append)
    assert not fake.exists()
    assert "Not a playable video" in said[0]


def test_a_real_video_passes(tmp_path):
    clip = tmp_path / "clip.mp4"
    made = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
         "testsrc2=size=320x240:rate=25:duration=4", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", str(clip)], capture_output=True)
    if made.returncode != 0:
        pytest.skip("ffmpeg cannot make a test clip here")
    assert downloader._check_video_file(str(clip), lambda _t: None)
    assert clip.exists()


class _Chromium:
    def __init__(self, missing=(None,), broken=False):
        self.missing = set(missing)
        self.broken = broken
        self.tried = []

    def launch(self, channel=None, **kwargs):
        self.tried.append(channel)
        if self.broken:
            raise RuntimeError("crashed on start")
        if channel in self.missing:
            raise RuntimeError("Executable doesn't exist at C:\\ms-playwright\\chrome.exe")
        return f"browser:{channel}"


class _Playwright:
    def __init__(self, chromium):
        self.chromium = chromium


def test_without_a_bundled_browser_the_installed_edge_is_used():
    chromium = _Chromium(missing=(None,))
    assert downloader._launch_chromium(_Playwright(chromium), headless=True) == "browser:msedge"
    assert chromium.tried == [None, "msedge"]


def test_chrome_when_there_is_no_edge():
    chromium = _Chromium(missing=(None, "msedge"))
    assert downloader._launch_chromium(_Playwright(chromium)) == "browser:chrome"


def test_no_browser_at_all_says_what_to_install():
    chromium = _Chromium(missing=(None, "msedge", "chrome"))
    said = []
    with pytest.raises(RuntimeError, match="Executable doesn't exist"):
        downloader._launch_chromium(_Playwright(chromium), said.append)
    assert "Edge or Google Chrome" in said[0]


def test_a_crash_is_not_mistaken_for_a_missing_browser():
    chromium = _Chromium(broken=True)
    with pytest.raises(RuntimeError, match="crashed"):
        downloader._launch_chromium(_Playwright(chromium))
    assert chromium.tried == [None]


def test_the_popup_closer_spares_the_session_page():
    """The "page" event fires for new_page() too; closing every new page shut
    the session's own page before it had loaded."""
    with open(os.path.join(ROOT, "downloader.py"), encoding="utf-8") as fh:
        source = fh.read()
    body = source[source.index("def download_via_browser_session"):
                  source.index("def download_from_extracted_source")]
    assert 'context.on("page", lambda pg: pg.close() if pg else None)' not in body
    assert "pg is not page" in body
    assert body.index("page = context.new_page()") < body.index('context.on("page"')


def test_the_release_build_bundles_playwright():
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as fh:
        assert any(l.split("#")[0].strip() == "playwright" for l in fh)
    with open(os.path.join(ROOT, ".github", "workflows", "build-release.yaml"),
              encoding="utf-8") as fh:
        assert fh.read().count("--collect-all playwright") == 2
