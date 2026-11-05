"""MCP server: lets Claude Desktop / Claude Code drive VideoHighlighter as tools.

The hosted LLM backends are VideoHighlighter calling a model. This is the other
direction: an MCP client (Claude Desktop has no local model server, so MCP is
the only way it can reach the app) calling VideoHighlighter. It runs over
stdio: the client launches ``python -m mcp_server`` and talks JSON-RPC on its
stdin/stdout. Registration snippets are in ``docs/MCP.md``.

Shared by both editions and kept byte-identical in both. The one edition-specific
tool, search by example, is registered only where ``llm/example_search.py``
exists (see ``tools.has_example_search``), so the files never need to diverge.

Why in-process engine calls and not a proxy to the HTTP sidecar
--------------------------------------------------------------
The FastAPI sidecar (``sidecar/server.py``) exists only in the free edition, and
only while its web shell has started it. The Qt app has none, and neither does
the Pro edition. Calling the engine directly works the same everywhere. It also
means one process for the client to start and stop, no port to pick or collide
on, and nothing listening on the network. The tools call the same entry points
the Qt GUI does (``pipeline.run_highlighter``, ``video_probe.probe_video``,
``clip_index.load_or_build`` and, where present,
``example_search.search_region_in_video``), so the MCP layer adds no engine
logic of its own.

Jobs run on one background thread, one at a time, the way the GUI's ``Worker``
runs a pipeline (it is a QThread, not a child process). A tool that starts
analysis or a search returns a job id at once; ``get_job`` reports progress, a
log tail and, when done, the result. Cancel trips the job's flag and returns:
it never kills the thread, for the reason ``main.Worker.cancel`` gives.

What crosses the wire
---------------------
Paths, timestamps, scores and log text, never frames, crops or audio. Search
results are ``{timestamp, score}`` lists; a run returns the paths it wrote.

stdout is the transport. Before anything else runs, ``stdio_guard`` moves the
real stdout aside for the protocol and points file descriptor 1 at stderr, so a
stray ``print()``, a native library or a child process (ffmpeg) cannot corrupt
the JSON-RPC stream. Diagnostics go through ``print()`` into ``debug-mcp.log``
beside the GUI's ``debug.log``.
"""
