"""A live link into the running app, for the MCP server (Pro).

The MCP server is its own process: an MCP client (Claude Desktop, Claude Code)
starts it, and it runs the engine beside the app, not inside it. Its results
reached the app only through the data folder, so a change showed up the next
time the app read it: the user had to reopen the timeline viewer, or load a
highlight version by hand, to see clips the server had just chosen.

This is the other half. The app listens on a local channel; the MCP tools try
it first and fall back to the data folder when no app is running:

* **Local and per user.** Qt's ``QLocalServer`` (a named pipe on Windows, a
  Unix socket elsewhere) with ``UserAccessOption``, so only the same user on the
  same machine can connect. Nothing listens on the network.
* **A short, fixed list of commands.** The app registers a handler per
  command; anything else is refused. A command changes what the user sees, the
  same way their own click would, and is undoable the same way.
* **One request per connection,** a JSON line each way:
  ``{"cmd": ..., "args": {...}}`` then ``{"ok": true, "result": ...}`` or
  ``{"ok": false, "error": ...}``.

The server side needs Qt and runs on the GUI thread, so handlers may touch
widgets. The client (:func:`send`) is plain Python with no Qt and no event
loop, so the MCP server can call it from a worker thread.
"""
from __future__ import annotations

import getpass
import json
import os
import re
import socket
import sys
import tempfile
import threading
import time
from typing import Callable, Mapping, Optional

MAX_MESSAGE = 4 * 1024 * 1024        # a clip list is a few KB; this is a ceiling


def channel_name(edition: Optional[str] = None) -> str:
    """Per edition and per user, so a free and a Pro install never answer
    each other's tools and two accounts on one machine stay apart."""
    if edition is None:
        try:
            from version import __edition__ as edition
        except Exception:  # noqa: BLE001 - outside the app tree
            edition = "app"
    user = re.sub(r"[^A-Za-z0-9_.-]", "_", getpass.getuser() or "user")
    return f"VideoHighlighter-{edition}-live-{user}"


# ── the app's side ───────────────────────────────────────────────────────────


class LiveLinkServer:
    """Listens for commands while the app runs. ``handlers`` maps a command
    name to ``fn(args: dict) -> JSON-able``; it is called on the GUI thread."""

    def __init__(self, handlers: Mapping[str, Callable[[dict], object]], *,
                 name: Optional[str] = None, parent=None,
                 log: Callable[[str], None] = print):
        from PySide6.QtNetwork import QLocalServer

        self.handlers = dict(handlers)
        self.name = name or channel_name()
        self.log = log
        from PySide6.QtCore import QCoreApplication

        self._buffers: dict = {}
        self._sockets: set = set()
        # Always a Qt parent, so Python's garbage collector never frees the
        # server (and its sockets) on the spot: on Windows a socket freed with
        # pipe I/O still pending crashes when that I/O completes. close() hands
        # it to deleteLater instead.
        self._server = QLocalServer(parent or QCoreApplication.instance())
        self._server.setSocketOptions(QLocalServer.UserAccessOption)
        self._server.newConnection.connect(self._accept)
        if not self._server.listen(self.name):
            # A socket file left by a crashed run (not on Windows, where pipes
            # vanish with their process). The app is single-instance, so it is
            # never a live one's.
            QLocalServer.removeServer(self.name)
            self._server.listen(self.name)
        self.listening = self._server.isListening()
        if self.listening:
            print(f"🔗 Live link listening on {self.name}")
        else:
            print(f"⚠️ Live link unavailable ({self._server.errorString()}); "
                  "MCP changes will show the next time a video is opened")

    def close(self) -> None:
        # Sockets are the server's children and die with it. Closed while one
        # still had a disconnect or a deleteLater queued, the queued work later
        # ran against the destroyed socket and crashed natively. So every
        # socket is cut loose first: no connections left, nothing pending.
        for sock in list(self._sockets):
            try:
                sock.disconnect()
                sock.abort()
            except RuntimeError:
                pass                             # already gone
        self._sockets.clear()
        self._buffers.clear()
        self._server.close()
        self._server.deleteLater()

    def _accept(self) -> None:
        while self._server.hasPendingConnections():
            sock = self._server.nextPendingConnection()
            self._sockets.add(sock)
            self._buffers[sock] = b""
            sock.readyRead.connect(lambda s=sock: self._read(s))
            sock.disconnected.connect(lambda s=sock: self._forget(s))

    def _forget(self, sock) -> None:
        self._sockets.discard(sock)
        self._buffers.pop(sock, None)
        sock.deleteLater()

    def _read(self, sock) -> None:
        if sock not in self._buffers:
            return                               # already answered
        data = self._buffers[sock] + bytes(sock.readAll())
        if b"\n" not in data and len(data) < MAX_MESSAGE:
            self._buffers[sock] = data
            return
        line = data.split(b"\n", 1)[0]
        del self._buffers[sock]
        reply = self.handle(line)
        sock.write((json.dumps(reply) + "\n").encode("utf-8"))
        sock.flush()
        sock.disconnectFromServer()

    def handle(self, line: bytes) -> dict:
        """One request line -> one reply. Never raises."""
        try:
            request = json.loads(line.decode("utf-8"))
            cmd = str(request.get("cmd") or "")
            handler = self.handlers.get(cmd)
            if handler is None:
                return {"ok": False, "error": f"Unknown command: {cmd or '(none)'}"}
            args = request.get("args") or {}
            if not isinstance(args, dict):
                return {"ok": False, "error": "args must be an object"}
            print(f"🔗 Live link: {cmd}")
            return {"ok": True, "result": handler(args)}
        except Exception as e:  # noqa: BLE001 - the reply is the report
            import traceback
            traceback.print_exc()
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ── the MCP server's side ───────────────────────────────────────────────────


class LiveLinkError(RuntimeError):
    """The app answered and refused, or stopped answering mid-request."""


def _exchange_windows(name: str, payload: bytes, timeout: float) -> Optional[bytes]:
    path = r"\\.\pipe" + "\\" + name
    deadline = time.monotonic() + min(timeout, 3.0)
    while True:
        try:
            pipe = open(path, "r+b", buffering=0)
            break
        except FileNotFoundError:
            return None                          # no app listening
        except OSError:
            # Busy: another client has the instance. Brief, then give up.
            if time.monotonic() > deadline:
                return None
            time.sleep(0.05)
    with pipe:
        pipe.write(payload)
        reply = b""
        while b"\n" not in reply and len(reply) < MAX_MESSAGE:
            chunk = pipe.read(65536)
            if not chunk:
                break
            reply += chunk
    return reply


def _exchange_unix(name: str, payload: bytes, timeout: float) -> Optional[bytes]:
    path = name if os.path.isabs(name) else os.path.join(tempfile.gettempdir(), name)
    if not os.path.exists(path):
        return None
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect(path)
        except (ConnectionRefusedError, FileNotFoundError):
            return None
        s.sendall(payload)
        reply = b""
        while b"\n" not in reply and len(reply) < MAX_MESSAGE:
            chunk = s.recv(65536)
            if not chunk:
                break
            reply += chunk
    return reply


def send(cmd: str, args: Optional[dict] = None, *, timeout: float = 120.0,
         name: Optional[str] = None) -> Optional[object]:
    """Ask the running app to do ``cmd``. Returns its result, or ``None`` when
    no app is listening. Raises :class:`LiveLinkError` when the app refused or
    did not answer within ``timeout`` (opening a long video can take a while,
    so the default is generous)."""
    name = name or channel_name()
    payload = (json.dumps({"cmd": cmd, "args": args or {}}) + "\n").encode("utf-8")
    exchange = _exchange_windows if sys.platform == "win32" else _exchange_unix
    box: dict = {}

    def work():
        try:
            box["reply"] = exchange(name, payload, timeout)
        except Exception as e:  # noqa: BLE001 - reported below
            box["error"] = e

    # A blocking pipe read cannot time out on its own, so it runs on a daemon
    # thread and is abandoned if the app stops answering.
    worker = threading.Thread(target=work, daemon=True, name="live-link")
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise LiveLinkError(f"The app did not answer '{cmd}' within {timeout:.0f} s")
    if "error" in box:
        raise LiveLinkError(f"Live link failed: {box['error']}")
    raw = box.get("reply")
    if raw is None:
        return None
    try:
        reply = json.loads(raw.split(b"\n", 1)[0].decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        raise LiveLinkError(f"Unreadable reply from the app: {e}")
    if not reply.get("ok"):
        raise LiveLinkError(reply.get("error") or "The app refused the request")
    return reply.get("result")
