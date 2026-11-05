"""Keep the MCP transport's stdout away from everything else in the process.

Over stdio the client reads JSON-RPC from our stdout, one message per line, and
anything else on it breaks the session. The engine prints freely (and is meant
to: ``print()`` is the diagnostic channel), OpenVINO and torch write from native
code, and ffmpeg children inherit the handle. Swapping ``sys.stdout`` would
only catch the first of those, so this works one level down, on the file
descriptor: duplicate fd 1 privately for the protocol, then point fd 1 at
stderr. Children spawned later inherit stderr in its place, and the private
copy is not inheritable (``os.dup`` makes it so), so they never see it.

Call :func:`claim_stdout` before importing anything that might print.
"""
from __future__ import annotations

import io
import os
import sys


def claim_stdout() -> io.BufferedWriter:
    """Take fd 1 for the protocol; return a binary writer on the private copy.

    After this, fd 1 and ``sys.stdout`` both lead to stderr. On Windows the CRT's
    ``_dup2`` onto fd 1 also updates the process's standard output handle, which
    is what ``subprocess`` hands a child that does not redirect its stdout.
    """
    try:
        sys.stdout.flush()
    except Exception:
        pass
    transport_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return open(transport_fd, "wb", buffering=0)


def install_debug_log() -> None:
    """Tee ``print()`` into ``debug-mcp.log`` (and stderr, which MCP clients log).

    The GUI's ``debug.log`` is not shared: ``debug_console.install`` rotates the
    file it opens, and an MCP server started while the GUI is open would rotate
    the GUI's live log away from under it. So the same tee is pointed at a
    sibling file of its own.
    """
    try:
        from modules.system import debug_console
        from modules.system.app_paths import user_data_dir

        debug_console.log_file_path = lambda: os.path.join(
            user_data_dir(), "debug-mcp.log")
        debug_console.install()
    except Exception as e:  # a log is an aid; never let it stop the server
        print(f"⚠️ mcp_server: debug log unavailable ({e})")
