#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


class RequestStore:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.all_path = self.output_dir / "requests.jsonl"
        self.wheel_path = self.output_dir / "wheel_photo_requests.jsonl"
        self.wash_path = self.output_dir / "wash_event_requests.jsonl"
        self._lock = threading.Lock()
        self.total = 0
        self.wheel_total = 0
        self.wash_total = 0

    def append(self, path, headers, body_bytes):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else None
        except Exception:
            payload = body_bytes.decode("utf-8", errors="replace")
        record = {
            "receivedAt": now,
            "path": path,
            "headers": headers,
            "payload": payload,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self.total += 1
            with self.all_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            if path.endswith("/api/vehicle/wheel-photo") or path.endswith("/wheel-photo"):
                self.wheel_total += 1
                with self.wheel_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            if path.endswith("/api/vehicle/wash-event") or path.endswith("/wash-event"):
                self.wash_total += 1
                with self.wash_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        return record

    def snapshot(self):
        with self._lock:
            return {
                "total": self.total,
                "wheelPhoto": self.wheel_total,
                "washEvent": self.wash_total,
                "outputDir": str(self.output_dir),
            }


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        server_version = "WheelPhotoMock/1.0"

        def log_message(self, fmt, *args):
            return

        def _send_json(self, status, payload):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/health", "/stats"):
                self._send_json(200, {"ok": True, "stats": store.snapshot()})
                return
            self._send_json(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length", "0") or 0)
            body = self.rfile.read(length) if length > 0 else b""
            headers = {key: self.headers.get(key) for key in self.headers.keys()}
            record = store.append(parsed.path, headers, body)
            payload = record.get("payload")
            if isinstance(payload, dict):
                print(
                    "[POST] path={path} photoUrl={photoUrl} type={type} cleanValue={cleanValue}".format(
                        path=parsed.path,
                        photoUrl=payload.get("photoUrl", ""),
                        type=payload.get("type", ""),
                        cleanValue=payload.get("cleanValue", ""),
                    ),
                    flush=True,
                )
            else:
                print("[POST] path={} non-json payload".format(parsed.path), flush=True)
            self._send_json(200, {"ok": True, "receivedAt": record["receivedAt"], "stats": store.snapshot()})

    return Handler


def parse_args():
    parser = argparse.ArgumentParser(description="Mock receiver for CleaningCar wheel-photo POST requests.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=28014)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir or Path("mock_api_output") / timestamp)
    store = RequestStore(output_dir)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
    print("mock API listening on http://{}:{}".format(args.host, args.port), flush=True)
    print("wheel-photo URL: http://<PC_IP>:{}/api/vehicle/wheel-photo".format(args.port), flush=True)
    print("output dir: {}".format(output_dir.resolve()), flush=True)
    try:
        while True:
            server.handle_request()
            if int(time.time()) % 30 == 0:
                print("[stats] {}".format(json.dumps(store.snapshot(), ensure_ascii=False)), flush=True)
    except KeyboardInterrupt:
        print("stopping mock API", flush=True)
    finally:
        server.server_close()
        print("final stats: {}".format(json.dumps(store.snapshot(), ensure_ascii=False)), flush=True)


if __name__ == "__main__":
    main()
