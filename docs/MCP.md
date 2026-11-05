# Using VideoHighlighter from Claude (MCP)

`mcp_server/` is an [MCP](https://modelcontextprotocol.io) server. Claude
Desktop and Claude Code use it to drive VideoHighlighter as a set of tools:
probe a video, list a folder, cut a highlight reel, search a video by text or
by an example region, and follow the jobs. It runs on this machine, over stdio.
The engine is the same one the app uses, and it reads the same `config.yaml`
and `./cache`.

Tool results are paths, timestamps, scores and log text. No frames, crops or
audio go back to the client.

## Requirements

- The app's dependencies, which include `mcp` (`requirements.txt` in the free
  edition, `requirements-pro.txt` in Pro), in the Python you point the client
  at. Use that interpreter's full path below, and the folder you cloned the app
  into in place of `D:\path\to\VideoHighlighter`.
- Running from source. The packaged exe does not expose the server yet.

Check that it starts. It should sit waiting for input; stop it with Ctrl+C:

```bash
python -m mcp_server
```

## Claude Desktop

Edit `claude_desktop_config.json`: `%APPDATA%\Claude\` on Windows,
`~/Library/Application Support/Claude/` on macOS. You can also open it from
Settings → Developer → Edit Config. Then restart Claude Desktop.

```json
{
  "mcpServers": {
    "videohighlighter": {
      "command": "C:\\Users\\you\\AppData\\Local\\Programs\\Python\\Python312\\python.exe",
      "args": ["D:\\path\\to\\VideoHighlighter\\mcp_server\\__main__.py"]
    }
  }
}
```

The script path works from any working directory, which matters because
Claude Desktop does not let you set one. The server moves into the app's data
folder on its own.

## Claude Code

```bash
claude mcp add videohighlighter --scope user -- python D:\path\to\VideoHighlighter\mcp_server\__main__.py
```

Leave out `--scope user` to register it for the current project only. Run
`claude mcp list` to check it connected.

## Tools

| Tool | What it does |
|---|---|
| `get_info` | Version, data folder, config path, running and queued jobs |
| `probe_video` | Duration, size, fps and rotation of one file |
| `scan_folder` | Video files in a folder, optionally recursive |
| `get_config` | The saved settings every run starts from |
| `start_highlight_run` | Analyse videos and cut a reel from each (job) |
| `start_visual_search` | Moments that match a text description (job) |
| `start_example_search` | Find more like this: an example region at given timestamps (job). Listed only in editions that have search by example |
| `set_timeline_clips` | Put a list of clips on the app's edit timeline for a video, to adjust and export there |
| `start_counter_check` | Read an on-screen counter: when each value appears, and which of a run's clips cover them (job). Pro |
| `start_counter_guided_run` | One clip per counter value, placed where the action model scores the chosen classes highest near it (job). Pro |
| `get_job` / `list_jobs` / `cancel_job` | Follow, list or stop jobs |

Analysis and searches are jobs. A `start_*` tool returns a `job_id` at once,
and `get_job` reports progress, the last log lines and, once the job is done,
its result. Jobs run one at a time, in the order they were started. A finished
highlight run lists each reel and its `_why.json` report. Cancelling never
kills the job's thread. A running job stops at the engine's next check.

The first search on a video builds its frame index. That takes about as long
as a Visual Search scan in the app, and later searches on the same video reuse
it.

## Seeing the result in the app

Clips the tools choose (`set_timeline_clips`, a counter-guided run) are saved
as the video's newest highlight version. When the timeline viewer opens that
video, its edit timeline holds them, ready to adjust and export.

In Pro, when the app is already open, they also appear in it straight away.
The app listens on a local channel that only your user account on this
computer can reach. The timeline viewer opens on the video, or updates if it
is already open, and Ctrl+Z there brings back what was on the edit timeline
before. A tool's result says `"live": true` when that happened, or why not,
for example when the app isn't running.

## Logs

Diagnostics go to `debug-mcp.log` in the app's data folder, beside the GUI's
`debug.log`, and to stderr, which both clients keep in their MCP logs. Nothing
but the protocol is ever written to stdout.
