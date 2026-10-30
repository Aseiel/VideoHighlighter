"""Download runs: live progress, and stopped by a stall or a cancel, not length.

The child processes here are tiny Python scripts that print what yt-dlp or
ffmpeg would, so the real tools and the network are never involved.
"""

from __future__ import annotations

import os
import re
import sys
import threading
import time

from modules.media import process_progress as pp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _child(script: str) -> list:
    return [sys.executable, "-u", "-c", script]


def test_lines_the_parser_uses_are_logged_and_the_rest_kept():
    logged = []
    script = (
        "print('[download] Destination: out.mp4')\n"
        "print('DLP| 42.0%|1.20MiB/s|00:30|50.00MiB')\n"
        "print('[download] 100% done')\n"
    )
    result = pp._run_ytdlp_with_progress(_child(script), log_fn=logged.append)

    assert result.returncode == 0
    assert logged == ["⬇️ 42.0%  •  1.20MiB/s  •  ETA 00:30  •  50.00MiB"]
    assert "Destination: out.mp4" in result.stdout
    assert "DLP|" not in result.stdout
    assert not result.stalled and not result.cancelled


def test_a_long_download_is_not_cut_short_while_it_moves():
    """Progress keeps arriving for longer than the stall limit in all: still
    allowed to finish. A fixed cap is what killed hour-long streams."""
    script = (
        "import time\n"
        "for i in range(8):\n"
        "    print(f'DLP| {i}%|1MiB/s|1:00|NA', flush=True)\n"
        "    time.sleep(0.25)\n"
    )
    result = pp._run_ytdlp_with_progress(_child(script), log_fn=lambda _t: None,
                                         idle_timeout=1.0)

    assert result.returncode == 0
    assert not result.stalled


def test_a_silent_download_is_stopped_as_stalled():
    script = "import time\nprint('started', flush=True)\ntime.sleep(30)\n"
    t0 = time.time()
    result = pp._run_ytdlp_with_progress(_child(script), log_fn=lambda _t: None,
                                         idle_timeout=1.0)

    assert result.stalled and not result.cancelled
    assert time.time() - t0 < 10


def test_a_cancel_stops_the_run_and_says_so():
    flag = threading.Event()
    threading.Timer(0.5, flag.set).start()
    script = "import time\nwhile True:\n    print('DLP| 1%|x|y|z', flush=True)\n    time.sleep(0.1)\n"
    result = pp._run_ytdlp_with_progress(_child(script), log_fn=lambda _t: None,
                                         cancel_flag=flag)

    assert result.cancelled and not result.stalled
    said = []
    pp._report_stop(result, said.append, "yt-dlp")
    assert said == ["⏹️ Download cancelled by user"]


def test_ffmpeg_progress_reads_against_the_known_length():
    logged = []
    on_line = pp._ffmpeg_progress(logged.append, total_seconds=3268)
    for line in ("out_time_us=1634000000", "total_size=104857600",
                 "speed=12.3x", "progress=continue"):
        assert on_line(line)
    assert not on_line("[https @ 000001] Opening 'seg-12.ts' for reading")

    assert logged == ["⬇️ 27:14 of 54:28 (50%)  •  100.0 MB  •  12.3x"]


def test_ffmpeg_progress_without_a_length_counts_up():
    logged = []
    on_line = pp._ffmpeg_progress(logged.append)
    on_line("out_time_us=4000000")
    on_line("progress=continue")
    assert logged == ["⬇️ 0:04 so far  •  0.0 MB  •  ?"]


def test_ffmpeg_gets_progress_on_stdout_ahead_of_its_inputs(monkeypatch):
    seen = {}
    monkeypatch.setattr(pp, "_stream_process",
                        lambda cmd, *a, **k: seen.setdefault("cmd", cmd))
    pp._run_ffmpeg_with_progress(["ffmpeg", "-y", "-i", "in.m3u8", "out.mp4"])

    cmd = seen["cmd"]
    assert cmd[0] == "ffmpeg"
    assert cmd.index("-progress") < cmd.index("-i")
    assert cmd[cmd.index("-progress") + 1] == "pipe:1"


def test_no_download_in_the_downloader_has_a_fixed_time_cap():
    with open(os.path.join(ROOT, "downloader.py"), encoding="utf-8") as fh:
        source = fh.read()
    # The three download runs: main, Fallback 1, the extracted source (yt-dlp
    # and ffmpeg). Probes and title lookups keep their short timeouts.
    assert "_run_ytdlp_with_progress(cmd, timeout=" not in source
    assert not re.search(r"subprocess\.run\(fallback_cmd", source)
    assert not re.search(r"subprocess\.run\(cmd, capture_output=True, text=True, timeout=400", source)
    assert not re.search(r"subprocess\.run\(cmd, check=True, timeout=400\)", source)
    assert source.count("*YTDLP_PROGRESS_ARGS") == 3
