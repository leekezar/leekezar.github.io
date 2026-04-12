#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import errno
import os
import re
import shutil
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
FEEDBACK_PATH = ROOT / "asl-dashboard" / "artifacts" / "calibration_feedback.json"
STEM_ANNOTATION_FEEDBACK_PATH = ROOT / "asl-dashboard" / "stem-annotation" / "artifacts" / "stem_annotation_feedback.json"
FEEDBACK_ENDPOINTS = {
    "/__asl_dashboard_feedback__": FEEDBACK_PATH,
    "/__stem_annotation_feedback__": STEM_ANNOTATION_FEEDBACK_PATH,
}


class DashboardHandler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, **kwargs):
        self._range = None
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _send_json(self, payload: object, status: int = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self._send_no_cache_headers()
        self.end_headers()
        self.wfile.write(encoded)

    def _send_no_cache_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")

    def end_headers(self) -> None:
        self._send_no_cache_headers()
        super().end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in FEEDBACK_ENDPOINTS:
            feedback_path = FEEDBACK_ENDPOINTS[parsed.path]
            if feedback_path.exists():
                try:
                    payload = json.loads(feedback_path.read_text(encoding="utf-8"))
                except Exception:
                    payload = {}
            else:
                payload = {}
            self._send_json(payload)
            return
        try:
            super().do_GET()
        except BrokenPipeError:
            return
        except ConnectionResetError:
            return

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in FEEDBACK_ENDPOINTS:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self._send_no_cache_headers()
            self.end_headers()
            return
        try:
            super().do_HEAD()
        except BrokenPipeError:
            return
        except ConnectionResetError:
            return

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in FEEDBACK_ENDPOINTS:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        feedback_path = FEEDBACK_ENDPOINTS[parsed.path]
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(max(0, length))
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        feedback_path.parent.mkdir(parents=True, exist_ok=True)
        feedback_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        self._send_json({"ok": True, "path": str(feedback_path.relative_to(ROOT))})

    def copyfile(self, source, outputfile):
        try:
            if self._range is None:
                super().copyfile(source, outputfile)
                return
            start, end = self._range
            source.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = source.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        except BrokenPipeError:
            return
        except OSError as exc:
            if exc.errno in {errno.EPIPE, errno.ECONNRESET}:
                return
            raise

    def send_head(self):
        self._range = None
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        ctype = self.guess_type(path)
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None
        try:
            fs = os.fstat(f.fileno())
            size = fs.st_size
            range_header = self.headers.get("Range")
            if range_header:
                match = re.match(r"bytes=(\d*)-(\d*)$", range_header.strip())
                if match:
                    start_str, end_str = match.groups()
                    if start_str == "" and end_str == "":
                        start = 0
                        end = size - 1
                    elif start_str == "":
                        suffix = int(end_str)
                        start = max(0, size - suffix)
                        end = size - 1
                    else:
                        start = int(start_str)
                        end = int(end_str) if end_str else size - 1
                    if start < size:
                        end = min(end, size - 1)
                        if start <= end:
                            self._range = (start, end)
                            self.send_response(HTTPStatus.PARTIAL_CONTENT)
                            self.send_header("Content-Type", ctype)
                            self.send_header("Accept-Ranges", "bytes")
                            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                            self.send_header("Content-Length", str(end - start + 1))
                            self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
                            self._send_no_cache_headers()
                            self.end_headers()
                            return f
                self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                f.close()
                return None
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
            self.send_header("Accept-Ranges", "bytes")
            self._send_no_cache_headers()
            self.end_headers()
            return f
        except Exception:
            f.close()
            raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler)
    print(f"Serving on http://127.0.0.1:{args.port}")
    print(f"Feedback files: {FEEDBACK_PATH} | {STEM_ANNOTATION_FEEDBACK_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
