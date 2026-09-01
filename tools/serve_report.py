"""A static file server that supports HTTP Range, for reading a report on a phone.

`python -m http.server` is not enough here. It has no Range support, so a
browser asked to start a video at 79 minutes has no way to ask for the bytes at
79 minutes -- it can only fetch from the beginning, which for this footage is a
400 MB download before the first frame appears. That reads as "the player does
not work" and is really "the server cannot be seeked".

So this adds the one missing piece: honour `Range: bytes=...` with a 206 and
`Accept-Ranges: bytes`, and let everything else fall through to the stock
handler. Nothing else about it is special.

    python serve_report.py "D:\\movies" 8000

Then browse to http://<this-machine>:8000/ from the phone and tap the report.
Serving is read-only, but it is the whole directory to anyone on the network --
stop it when you are done.
"""
from __future__ import annotations

import os
import re
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

_RANGE = re.compile(r"bytes=(\d*)-(\d*)$")


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        header = self.headers.get("Range")
        if not header:
            # No range asked for: the stock path already does the right thing,
            # except that a client cannot know seeking is available unless the
            # first response says so.
            self._plain = True
            return super().send_head()
        self._plain = False

        match = _RANGE.match(header.strip())
        path = self.translate_path(self.path)
        if not match or os.path.isdir(path):
            return super().send_head()
        try:
            fh = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        size = os.fstat(fh.fileno()).st_size
        first, last = match.group(1), match.group(2)
        if first == "":                      # a suffix range: the last N bytes
            length = int(last or 0)
            start, end = max(0, size - length), size - 1
        else:
            start = int(first)
            end = int(last) if last else size - 1
        end = min(end, size - 1)
        if start > end or start >= size:
            fh.close()
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None

        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        fh.seek(start)
        self._remaining = end - start + 1
        return fh

    def copyfile(self, source, outputfile):
        if getattr(self, "_plain", True):
            return super().copyfile(source, outputfile)
        remaining = self._remaining
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            try:
                outputfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                # The phone seeked again, or closed the tab. Normal, not an error.
                return
            remaining -= len(chunk)

    def end_headers(self):
        if getattr(self, "_plain", True):
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    directory = argv[0] if argv else os.getcwd()
    port = int(argv[1]) if len(argv) > 1 else 8000
    if not os.path.isdir(directory):
        print(f"not a directory: {directory}")
        return 2
    handler = partial(RangeHandler, directory=directory)
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)
    print(f"serving {directory} on port {port} — Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
