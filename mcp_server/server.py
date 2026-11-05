"""The FastMCP app: tool names, descriptions and arguments, and the stdio entry.

The descriptions are what the client's model reads to decide which tool to call
and how, so they say what each tool returns and how long it takes. The work
itself is in ``tools.py``.
"""
from __future__ import annotations

import os
from typing import Optional

INSTRUCTIONS = """\
VideoHighlighter finds and cuts highlights from video files on this computer.
Paths are local file paths. Analysis and searches run as background jobs, one
at a time: a start_* tool returns a job_id at once; call get_job with it until
status is done, failed or cancelled. A first search on a video builds a frame
index and can take minutes; later searches on the same video reuse it.
Results are paths, timestamps (seconds) and scores, never images."""


def build_server(jobs=None):
    """A FastMCP instance with every tool registered. `jobs` is injectable so
    tests get a fresh queue."""
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    from mcp_server import tools
    from mcp_server.jobs import JobManager

    jobs = jobs if jobs is not None else JobManager()
    mcp = FastMCP("videohighlighter", instructions=INSTRUCTIONS)
    read_only = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
    starts = ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                             openWorldHint=False)

    @mcp.tool(annotations=read_only)
    def get_info() -> dict:
        """App version, edition, data folder, config path, and which job is
        running or queued."""
        return tools.get_info(jobs)

    @mcp.tool(annotations=read_only)
    def probe_video(path: str) -> dict:
        """Duration (s), width, height, fps and rotation of one video file."""
        return tools.probe_video(path)

    @mcp.tool(annotations=read_only)
    def scan_folder(path: str, recursive: bool = False, limit: int = 1000) -> dict:
        """List the video files in a folder (sorted; subfolders too when
        recursive)."""
        return tools.scan_folder(path, recursive=recursive, limit=limit)

    @mcp.tool(annotations=read_only)
    def get_config() -> dict:
        """The saved settings (config.yaml) the app and every run start from."""
        return tools.get_config()

    @mcp.tool(annotations=starts)
    def start_highlight_run(video_paths: list[str], preset: str = "simple",
                            length: str = "medium",
                            output_name: Optional[str] = None,
                            overrides: Optional[dict] = None) -> dict:
        """Analyse videos and cut a highlight reel from each. Returns a job_id;
        poll get_job. Takes minutes per video (longer with transcription).

        preset: "simple" = the app's one-button scoring (scene changes, motion
        peaks, loudness), "saved" = the user's saved settings from config.yaml.
        length: "short" (~90 s), "medium" (~4 min) or "long" (~7 min) for
        "simple". output_name: file name for the reel, written beside each
        source (default <video>_highlight.mp4). overrides: pipeline settings
        for this run only, e.g. {"max_duration": 120, "clip_time": 6}.
        The finished job's result lists each input, the reel it wrote, and the
        run's report (report_json: why each moment was kept or left out)."""
        return tools.start_highlight_run(jobs, video_paths, preset=preset,
                                         length=length, output_name=output_name,
                                         overrides=overrides)

    @mcp.tool(annotations=starts)
    def start_visual_search(video_path: str, query: str, top_k: int = 20) -> dict:
        """Find the moments in a video that look like a text description
        (e.g. "a dog jumping into water"). Returns a job_id; poll get_job.
        The result lists the best `top_k` timestamps (s), best first, each with
        a score in 0..1 and the margin it won by."""
        return tools.start_visual_search(jobs, video_path, query, top_k=top_k)

    def start_example_search(video_path: str, roi: list[float], name: str,
                             example_timestamps: Optional[list[float]] = None,
                             interval: float = 1.5,
                             threshold: Optional[float] = None,
                             top_k: int = 100) -> dict:
        """Find more like this: score every moment of a video against an example
        region. roi is [x, y, width, height] as fractions of the frame (0..1);
        example_timestamps are the seconds where that region shows the example
        (default: the middle of the video). name labels the search and keys its
        cache. Returns a job_id; poll get_job. The result lists timestamps (s)
        whose region scores at or above threshold (default 0.5), best first."""
        return tools.start_example_search(
            jobs, video_path, roi, name,
            example_timestamps=example_timestamps or (), interval=interval,
            threshold=threshold, top_k=top_k)

    # Only where this edition has search by example; listing a tool that can
    # only fail would just send the client down a dead end.
    if tools.has_example_search():
        mcp.tool(annotations=starts)(start_example_search)

    def start_counter_check(video_path: str, roi: list[float],
                            clips: Optional[list[list[float]]] = None,
                            report_json: Optional[str] = None,
                            sensitivity: str = "medium") -> dict:
        """Read an on-screen counter (a tally the video shows, e.g. "1", "2",
        "3" in a corner): every time it goes up, an event provably happened.
        roi is [x, y, width, height] around the counter as fractions of the
        frame (0..1). Pass clips as [[start, end], ...] seconds, or report_json
        (a run's _why.json) to check which ticks those clips cover. sensitivity:
        "low", "medium" or "high". Returns a job_id; poll get_job. The result
        lists each tick's time (s), the value read when the digits are legible,
        whether a clip covers it, and suggested spans for the missed ones."""
        return tools.start_counter_check(jobs, video_path, roi, clips=clips,
                                         report_json=report_json,
                                         sensitivity=sensitivity)

    if tools.has_counter_check():
        mcp.tool(annotations=starts)(start_counter_check)

    def start_counter_guided_run(video_path: str, roi: list[float],
                                 classes: list[str],
                                 output_name: Optional[str] = None,
                                 before_peak: float = 15.0,
                                 after_peak: float = 5.0,
                                 search_before: float = 10.0,
                                 search_after: float = 60.0,
                                 sensitivity: str = "medium") -> dict:
        """For footage that counts its own events on screen: one clip per
        counter value. roi is [x, y, width, height] around the counter as
        fractions of the frame (0..1); classes are action-model classes to look
        for. For each value, the window where those classes score highest
        between search_before seconds before it appears and search_after
        seconds after (never past the next value) is the peak; the clip runs
        from before_peak seconds before it to after_peak after (a model scores
        an event highest once its result is visible, after it began), never
        starting before the span searched. Writes
        <output_name>.mp4 beside the video, its clips folder and an
        _events.json, and saves the clips as the video's newest highlight
        version, so the app's timeline viewer opens on them for adjusting and
        exporting. Returns a job_id; poll get_job. The result lists each value,
        the span searched, the clip taken, its peak and score."""
        return tools.start_counter_guided_run(
            jobs, video_path, roi, classes, output_name=output_name,
            before_peak=before_peak, after_peak=after_peak,
            search_before=search_before,
            search_after=search_after, sensitivity=sensitivity)

    if tools.has_counter_guided():
        mcp.tool(annotations=starts)(start_counter_guided_run)

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False,
                                          idempotentHint=True, openWorldHint=False))
    def set_timeline_clips(video_path: str, clips: list[list[float]],
                           note: Optional[str] = None) -> dict:
        """Put clips on the app's edit timeline for a video: clips are
        [[start, end], ...] in seconds. They are saved as the video's newest
        highlight version, so opening the video in the timeline viewer shows
        them, ready to adjust and export there. Use it to hand the user a
        selection, or to correct one a run made. A later run replaces them."""
        return tools.set_timeline_clips(video_path, clips, note=note)

    @mcp.tool(annotations=read_only)
    def get_job(job_id: str, log_lines: int = 20) -> dict:
        """Status of a job (queued, running, done, failed, cancelled), its
        progress, the last log lines, and its result once done."""
        return tools.get_job(jobs, job_id, log_lines=log_lines)

    @mcp.tool(annotations=read_only)
    def list_jobs() -> dict:
        """Every job this server has run or queued (most recent last)."""
        return tools.list_jobs(jobs)

    @mcp.tool(annotations=ToolAnnotations(readOnlyHint=False,
                                          destructiveHint=False,
                                          idempotentHint=True,
                                          openWorldHint=False))
    def cancel_job(job_id: str) -> dict:
        """Stop a job. A queued job is dropped; a running one stops at its next
        checkpoint, which can take a little while."""
        return tools.cancel_job(jobs, job_id)

    return mcp


async def _serve(mcp, transport_out) -> None:
    import io

    import anyio
    from mcp.server.stdio import stdio_server

    out = anyio.wrap_file(io.TextIOWrapper(transport_out, encoding="utf-8",
                                           write_through=True))
    async with stdio_server(stdout=out) as (read_stream, write_stream):
        # FastMCP.run_stdio_async does exactly this but always on sys.stdout,
        # which stdio_guard has pointed at stderr.
        server = mcp._mcp_server
        await server.run(read_stream, write_stream,
                         server.create_initialization_options())


def main() -> None:
    from mcp_server import stdio_guard

    transport_out = stdio_guard.claim_stdout()
    stdio_guard.install_debug_log()

    # The engine's relative paths (./cache above all) mean "the app's data
    # folder", which is where the GUI runs from. A client starts us anywhere.
    from modules.system.app_paths import user_data_dir
    try:
        os.chdir(user_data_dir())
    except OSError as e:
        print(f"⚠️ mcp_server: could not work from {user_data_dir()}: {e}")

    import anyio

    print(f"🔌 mcp_server: serving on stdio from {os.getcwd()}")
    anyio.run(_serve, build_server(), transport_out)
