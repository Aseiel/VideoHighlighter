"""A minimal obs-websocket v5 client: just the requests the live loop needs.

OBS Studio 28+ ships obs-websocket built in (Tools -> WebSocket Server
Settings). It is the supported way for another process to talk to a running
OBS, and the one used here, because OBS has no Python API outside its own
in-process scripting host.

Four requests don't justify another dependency (and the best-known client,
obsws-python, is GPL-3.0), so this speaks the protocol directly over
``websocket-client`` (Apache-2.0). The protocol is small:
https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md

    Hello (op 0) -> Identify (op 1) -> Identified (op 2)
    Request (op 6) -> RequestResponse (op 7)

No event subscriptions are requested, so OBS sends nothing unasked.
"""
from __future__ import annotations

import base64
import hashlib
import itertools
import json

OP_HELLO, OP_IDENTIFY, OP_IDENTIFIED = 0, 1, 2
OP_REQUEST, OP_REQUEST_RESPONSE = 6, 7
RPC_VERSION = 1
SUBPROTOCOL = "obswebsocket.json"


class ObsError(RuntimeError):
    """OBS refused a request, or the connection failed."""

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


def auth_string(password: str, salt: str, challenge: str) -> str:
    """The Identify ``authentication`` value, per the protocol:
    base64(sha256(base64(sha256(password + salt)) + challenge))."""
    secret = base64.b64encode(hashlib.sha256((password + salt).encode()).digest()).decode()
    return base64.b64encode(hashlib.sha256((secret + challenge).encode()).digest()).decode()


class ObsClient:
    """Blocking client. ``with ObsClient(...) as obs: obs.request("GetVersion")``."""

    def __init__(self, host: str = "localhost", port: int = 4455,
                 password: str = "", timeout: float = 5.0):
        self.url = f"ws://{host}:{port}"
        self.password = password
        self.timeout = timeout
        self._ws = None
        self._ids = itertools.count(1)

    def connect(self) -> "ObsClient":
        try:
            import websocket  # websocket-client, lazy: only live mode needs it
        except ImportError:
            raise ObsError("OBS mode needs the websocket-client package "
                           "(pip install websocket-client)") from None
        try:
            self._ws = websocket.create_connection(
                self.url, timeout=self.timeout, subprotocols=[SUBPROTOCOL])
        except Exception as e:  # noqa: BLE001 - socket/handshake errors vary
            raise ObsError(
                f"Could not reach OBS at {self.url} ({e}). Is OBS running, and is "
                "Tools -> WebSocket Server Settings -> Enable WebSocket server on?"
            ) from None

        hello = self._recv_op(OP_HELLO)
        identify = {"rpcVersion": RPC_VERSION, "eventSubscriptions": 0}
        auth = hello.get("authentication")
        if auth:
            if not self.password:
                self.close()
                raise ObsError("OBS asks for a WebSocket password — pass --obs-password "
                               "or set OBS_WEBSOCKET_PASSWORD")
            identify["authentication"] = auth_string(
                self.password, auth["salt"], auth["challenge"])
        self._send(OP_IDENTIFY, identify)
        try:
            self._recv_op(OP_IDENTIFIED)
        except ObsError:
            self.close()
            raise ObsError("OBS rejected the connection — wrong WebSocket password?") from None
        return self

    def request(self, request_type: str, data: dict | None = None) -> dict:
        """Send one request and return its ``responseData`` ({} when none).
        Raises ObsError when OBS reports failure."""
        if self._ws is None:
            raise ObsError("Not connected")
        rid = str(next(self._ids))
        payload = {"requestType": request_type, "requestId": rid}
        if data:
            payload["requestData"] = data
        self._send(OP_REQUEST, payload)
        while True:
            d = self._recv_op(OP_REQUEST_RESPONSE)
            if d.get("requestId") == rid:
                break
        status = d.get("requestStatus") or {}
        if not status.get("result"):
            raise ObsError(f"{request_type} failed: {status.get('comment') or 'code ' + str(status.get('code'))}",
                           status.get("code"))
        return d.get("responseData") or {}

    def close(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def __enter__(self) -> "ObsClient":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- convenience wrappers -------------------------------------------------

    def program_scene(self) -> str:
        d = self.request("GetCurrentProgramScene")
        return d.get("sceneName") or d.get("currentProgramSceneName") or ""

    def screenshot_jpeg(self, source: str, width: int, quality: int = 80) -> bytes:
        """The source as OBS renders it, scaled to ``width`` (aspect kept) by OBS
        on the GPU, as JPEG bytes."""
        d = self.request("GetSourceScreenshot", {
            "sourceName": source, "imageFormat": "jpg",
            "imageWidth": int(width), "imageCompressionQuality": int(quality)})
        data = d.get("imageData", "")
        return base64.b64decode(data.split(",", 1)[1] if "," in data else data)

    def record_status(self) -> dict:
        """{"outputActive": bool, "outputPaused": bool, "outputDuration": ms, ...}"""
        return self.request("GetRecordStatus")

    # -- wire -----------------------------------------------------------------

    def _send(self, op: int, d: dict) -> None:
        self._ws.send(json.dumps({"op": op, "d": d}))

    def _recv_op(self, op: int) -> dict:
        try:
            while True:
                msg = json.loads(self._ws.recv())
                if msg.get("op") == op:
                    return msg.get("d") or {}
        except Exception as e:  # noqa: BLE001 - closed socket, timeout, bad frame
            raise ObsError(f"Connection to OBS lost while waiting for op {op}: {e}") from None
