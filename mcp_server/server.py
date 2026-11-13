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
Results are paths, timestamps (seconds) and scores, never images.

It also keeps a hand-sorted training dataset healthy (folders
<dataset>/train/<class>/*.mp4). The loop: dataset_health says what is thin;
plan_reviewed_video then add_reviewed_video add a reviewed video's crops,
thin classes only and capped; start_dataset_comparison measures, on the
dataset's other videos, whether that helped and which part did not;
drop_reviewed_folders or undo_reviewed_video take out what measured worse.
Decide by the comparison's numbers: an addition is kept only when it did
not measure worse. Classes are shown as codes (C01, ...) unless the user has
turned real names on; the user has the table."""


def build_server(jobs=None):
    """A FastMCP instance with every tool registered. `jobs` is injectable so
    tests get a fresh queue."""
    from mcp.server.fastmcp import FastMCP
    from mcp.types import ToolAnnotations

    from mcp_server import dataset_tools, tools
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
        The result lists the best `top_k` moments, best first: timestamp (s) is
        the best frame, start and end the stretch it belongs to, with a score
        in 0..1 and the margin it won by. A vague or unusual word ranks poorly:
        scores are then all alike and right answers sit far down. Once the
        user confirms any match as right, pass it to more_like_these."""
        return tools.start_visual_search(jobs, video_path, query, top_k=top_k)

    @mcp.tool(annotations=read_only)
    def more_like_these(video_path: str, timestamps: list[float],
                        top_k: int = 20) -> dict:
        """Find the moments of a video that look like moments already known to
        be right. timestamps: seconds of confirmed matches (one is enough;
        more sharpen it), e.g. hits from start_visual_search the user agreed
        with. Answers at once from the index that search saved, so run
        start_visual_search on the video first. Returns up to `top_k` moments,
        best first: timestamp (s) is the best frame, start and end the
        stretch, likeness in 0..1 (the examples themselves score near 1;
        comparable within one answer only), frames how many sampled frames
        it holds. Usually finds what the word missed."""
        return tools.more_like_these(video_path, timestamps, top_k=top_k)

    @mcp.tool(annotations=starts)
    def start_expression_scan(video_path: str,
                              labels: Optional[list[str]] = None,
                              min_confidence: float = 0.0,
                              rescan: bool = False) -> dict:
        """Where faces in a video show an expression: neutral, happy, sad,
        surprise or anger, read on the clearest face once a second. Returns a
        job_id; poll get_job. The first scan of a video takes minutes; it is
        saved and later calls return at once (rescan=true to redo it). The
        result has how many seconds each expression accounts for and, per
        expression, the time ranges (s) it holds. labels: which expressions to
        list ranges for (default: all found except neutral). min_confidence
        (0..1) drops weak readings. A second with no readable face is absent,
        not neutral; profile and covered faces read poorly. The app's timeline
        viewer shows the same scan as its EXPRESSIONS row. Pass ranges to
        set_timeline_clips to put them on the edit timeline."""
        return tools.start_expression_scan(
            jobs, video_path, labels=labels, min_confidence=min_confidence,
            rescan=rescan)

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
    def dataset_health(dataset: str, aliases: Optional[str] = None) -> dict:
        """Is a hand-sorted training dataset healthy? dataset is the folder
        holding train/, val/ and test/. Reads file names only (seconds).
        Returns findings (each with a level, what is wrong and the fix):
        classes from fewer than 3 source videos, one video flooding a class or
        the dataset, clips whose name gives no source video, val/ or test/
        sharing videos with train/, the same file twice. `wanted` lists the
        classes more videos would help most, thinnest first: what to look for
        next. aliases: optional JSON file renaming or leaving out folders."""
        return dataset_tools.dataset_health(dataset, aliases=aliases)

    @mcp.tool(annotations=read_only)
    def plan_reviewed_video(dataset: str, review: str, video: str,
                            thin_under: int = 200, cap: int = 50,
                            exclude: Optional[list[str]] = None,
                            aliases: Optional[str] = None) -> dict:
        """What adding a reviewed video to a dataset would do; changes nothing.
        review is a folder with one sub-folder per class holding that video's
        checked crops; video is the name it gets in the dataset (one source
        video, so held-out tests keep it on one side). A folder is taken only
        when one of its classes has fewer than thin_under clips in the dataset,
        and at most cap crops of it, spread over the video. exclude: folders to
        leave out. Returns per folder what was found, what would be taken and
        why not, the share of the training clips this video would become, and
        warnings."""
        return dataset_tools.plan_reviewed_video(
            dataset, review, video, thin_under=thin_under, cap=cap,
            exclude=exclude or (), aliases=aliases)

    @mcp.tool(annotations=starts)
    def add_reviewed_video(dataset: str, review: str, video: str,
                           thin_under: int = 200, cap: int = 50,
                           exclude: Optional[list[str]] = None,
                           aliases: Optional[str] = None) -> dict:
        """Add a reviewed video's crops to the dataset's train/ folders, as
        plan_reviewed_video described (same arguments). The crops are linked,
        not moved, and exactly what was added is recorded in the dataset, so it
        can be taken out again. Safe to repeat. Follow it with
        start_dataset_comparison: adding is not known to help until measured."""
        return dataset_tools.add_reviewed_video(
            dataset, review, video, thin_under=thin_under, cap=cap,
            exclude=exclude or (), aliases=aliases)

    @mcp.tool(annotations=starts)
    def start_dataset_comparison(dataset: str, added: Optional[list[str]] = None,
                                 seeds: int = 5, parts: Optional[str] = "kinds",
                                 aliases: Optional[str] = None) -> dict:
        """Did the added videos make the model better on the dataset's OTHER
        videos? Trains small heads with and without the added clips on the same
        folds and scores both on the same held-out clips; the added videos are
        never scored. Returns a job_id; poll get_job. Takes about 5-20 minutes.
        added: video names (default: those add_reviewed_video recorded).
        parts: "kinds" also scores, each on its own, the clips of classes the
        dataset already has and the clips of classes only the added video has;
        "folders" scores every class folder on its own (slow); null skips it.
        The result has verdict (better, worse, no clear change), keep (false
        when it measured worse), top-1 with and without, per-class recall
        changes, and drop_folders: the folders that measured worse on their
        own and should be taken out with drop_reviewed_folders. A change counts
        only when every seed agrees and it is clear of the seeds' spread, so
        use 5 seeds or more for a decision."""
        return dataset_tools.start_dataset_comparison(
            jobs, dataset, added=added, seeds=seeds, parts=parts, aliases=aliases)

    removes = ToolAnnotations(readOnlyHint=False, destructiveHint=True,
                              idempotentHint=True, openWorldHint=False)

    @mcp.tool(annotations=removes)
    def drop_reviewed_folders(dataset: str, video: str, folders: list[str],
                              aliases: Optional[str] = None) -> dict:
        """Take out of the dataset the crops add_reviewed_video put into these
        class folders for this video, and keep the rest. Only files that tool
        added are removed; the review folder and the dataset's own clips are
        never touched. Use it with a comparison's drop_folders."""
        return dataset_tools.drop_reviewed_folders(dataset, video, folders,
                                                   aliases=aliases)

    @mcp.tool(annotations=removes)
    def undo_reviewed_video(dataset: str, video: str) -> dict:
        """Take out everything add_reviewed_video added for this video: those
        files and only those. The review folder is never touched."""
        return dataset_tools.undo_reviewed_video(dataset, video)

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
