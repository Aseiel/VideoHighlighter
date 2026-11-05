"""``python -m mcp_server`` from the repo root, or ``python <repo>/mcp_server/__main__.py``
from anywhere (which is what an MCP client's config can do without a cwd)."""
import os
import sys

if __package__ in (None, ""):
    # Run as a script, sys.path[0] is this folder, where tools.py would shadow
    # the repo's tools/ package. Put the repo root there instead.
    sys.path[0] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from mcp_server.server import main  # noqa: E402

main()
