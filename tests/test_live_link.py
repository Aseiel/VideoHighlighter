"""The live link (`modules.system.live_link`): the app's local listener and the
plain-Python client the MCP server uses, talking to each other for real."""
from __future__ import annotations

import os
import threading
import time
import uuid

import pytest

pytest.importorskip("PySide6.QtNetwork")

from modules.system import live_link  # noqa: E402


@pytest.fixture(scope="module")
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtWidgets import QApplication
    return QCoreApplication.instance() or QApplication([])


@pytest.fixture
def channel():
    return f"vh-test-{uuid.uuid4().hex[:10]}"


def ask(app, *a, **kw):
    """Run the blocking client on a thread while the Qt loop serves it."""
    from PySide6.QtCore import QCoreApplication

    box = {}

    def work():
        try:
            box["result"] = live_link.send(*a, **kw)
        except Exception as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=work)
    t.start()
    deadline = time.monotonic() + 10
    while t.is_alive() and time.monotonic() < deadline:
        QCoreApplication.processEvents()
        time.sleep(0.005)
    t.join(1)
    if "error" in box:
        raise box["error"]
    return box.get("result")


def test_a_command_runs_on_the_app_side_and_its_result_comes_back(app, channel):
    seen = []
    server = live_link.LiveLinkServer(
        {"show": lambda args: seen.append(args) or {"shown": len(args["clips"])}},
        name=channel)
    try:
        assert server.listening
        result = ask(app, "show", {"clips": [[1, 2], [3, 4]]}, name=channel, timeout=5)
        assert result == {"shown": 2}
        assert seen == [{"clips": [[1, 2], [3, 4]]}]
    finally:
        server.close()


def test_unknown_commands_and_handler_errors_are_refused_not_crashed(app, channel):
    def boom(args):
        raise ValueError("bad clip")

    server = live_link.LiveLinkServer({"boom": boom}, name=channel)
    try:
        with pytest.raises(live_link.LiveLinkError, match="Unknown command"):
            ask(app, "format_disk", name=channel, timeout=5)
        with pytest.raises(live_link.LiveLinkError, match="bad clip"):
            ask(app, "boom", name=channel, timeout=5)
        # Still serving after both.
        assert server.handle(b'{"cmd": "boom"}')["ok"] is False
    finally:
        server.close()


def test_no_app_listening_is_none_not_an_error(channel):
    t0 = time.monotonic()
    assert live_link.send("show", {}, name=channel, timeout=5) is None
    assert time.monotonic() - t0 < 2


def test_channels_are_per_edition_and_user():
    assert live_link.channel_name("Pro") != live_link.channel_name("Free")
    assert live_link.channel_name("Pro").startswith("VideoHighlighter-Pro-live-")
