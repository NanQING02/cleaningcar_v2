#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def add_project_to_path(project_root):
    project_root = Path(project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def resolve_path(base, value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(base) / path).resolve()


def _entry_to_json(item):
    if not isinstance(item, dict):
        return item
    result = dict(item)
    image = result.pop("imageJpegBytes", b"") or b""
    if image:
        result["imageJpegBase64"] = base64.b64encode(image).decode("ascii")
    return result


def _float_arg(query, name, default):
    try:
        return float((query.get(name) or [default])[0])
    except (TypeError, ValueError):
        return float(default)


def _int_arg(query, name, default=0):
    try:
        return int((query.get(name) or [default])[0])
    except (TypeError, ValueError):
        return int(default)


def parse_args():
    parser = argparse.ArgumentParser(description="Run CleaningCar wheel sidechain as a standalone local service.")
    parser.add_argument("--project-root", default=".", help="CleaningCar project root.")
    parser.add_argument("--config", default="configs/config.json", help="Config path.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=28015)
    parser.add_argument("--hw-decode", action="store_true")
    parser.add_argument("--left-source", default="")
    parser.add_argument("--right-source", default="")
    parser.add_argument("--target-fps", type=float, default=0.0)
    parser.add_argument("--active-target-fps", type=float, default=-1.0)
    parser.add_argument("--reader-idle-fps", type=float, default=-1.0)
    parser.add_argument("--image-quality", type=int, default=85)
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = add_project_to_path(args.project_root)
    config_path = resolve_path(project_root, args.config)

    from cleaningcar.runtime_config import load_config
    from cleaningcar.wheel import WheelDetectionService

    config = load_config(str(config_path))
    wheel_cfg = config.setdefault("wheel", {})
    if args.left_source:
        wheel_cfg["left_source"] = args.left_source
    if args.right_source:
        wheel_cfg["right_source"] = args.right_source
    if args.target_fps > 0:
        wheel_cfg["target_fps"] = args.target_fps
    if args.active_target_fps >= 0:
        wheel_cfg["active_target_fps"] = args.active_target_fps
    if args.reader_idle_fps >= 0:
        wheel_cfg["reader_idle_fps"] = args.reader_idle_fps
    if wheel_cfg.get("left_source") or wheel_cfg.get("right_source"):
        wheel_cfg["enabled"] = True

    service = WheelDetectionService(
        config=config,
        base_dir=config_path.parent,
        hw_decode=bool(args.hw_decode),
        image_quality=int(args.image_quality),
    )
    if not service.start():
        print("failed to start wheel sidechain service", file=sys.stderr)
        return 3

    class Handler(BaseHTTPRequestHandler):
        server_version = "CleaningCarWheelSidechain/1.0"

        def log_message(self, fmt, *values):
            print("[wheel-server] " + (fmt % values), flush=True)

        def _send_json(self, payload, status=200):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
            except ValueError:
                length = 0
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))

        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            now_ts = _float_arg(query, "now_ts", time.time())
            ref_ts = _float_arg(query, "reference_ts", now_ts)
            track_id = _int_arg(query, "track_id", 0)
            try:
                if parsed.path == "/health":
                    self._send_json({"ok": True, "active_sides": service.active_sides})
                elif parsed.path == "/stats":
                    self._send_json({"ok": True, "active_sides": service.active_sides, "stats": service.snapshot_stats()})
                elif parsed.path == "/recent":
                    items = service.get_recent_result_entries(now_ts=now_ts, reference_ts=ref_ts, track_id=track_id)
                    self._send_json({"ok": True, "items": [_entry_to_json(item) for item in items]})
                elif parsed.path == "/photo-candidates":
                    items = service.get_photo_candidate_entries(track_id, now_ts=now_ts, reference_ts=ref_ts)
                    self._send_json({"ok": True, "items": [_entry_to_json(item) for item in items]})
                elif parsed.path == "/claimed":
                    items = service.get_claimed_result_entries(track_id, now_ts=now_ts, reference_ts=ref_ts)
                    self._send_json({"ok": True, "items": [_entry_to_json(item) for item in items]})
                else:
                    self._send_json({"ok": False, "error": "not_found"}, status=404)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)

        def do_POST(self):
            try:
                payload = self._read_json()
                if self.path == "/activity":
                    service.update_track_activity(
                        int(payload.get("track_id", 0) or 0),
                        frame_ts=float(payload.get("frame_ts", time.time()) or time.time()),
                        active=bool(payload.get("active", False)),
                    )
                    self._send_json({"ok": True})
                elif self.path == "/claim":
                    ok = service.claim_result_entry(
                        int(payload.get("track_id", 0) or 0),
                        int(payload.get("entry_id", 0) or 0),
                    )
                    self._send_json({"ok": bool(ok)})
                else:
                    self._send_json({"ok": False, "error": "not_found"}, status=404)
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, status=500)

    httpd = ThreadingHTTPServer((args.host, int(args.port)), Handler)
    print(
        f"[wheel-server] listening on {args.host}:{args.port} "
        f"config={config_path} sides={','.join(service.active_sides)}",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("[wheel-server] stopping", flush=True)
    finally:
        httpd.server_close()
        service.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
