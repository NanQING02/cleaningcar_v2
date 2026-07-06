#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path


class DummyZoneManager:
    @staticmethod
    def update_track(track_id, anchor_point, frame_idx):
        return None, {"enter_a": False, "exit_a": False, "enter_b": False, "exit_b": False}

    @staticmethod
    def drop_track(track_id):
        return None

    @staticmethod
    def resolve_direction(state):
        return 0, ""


class LocalWheelPhotoCollector:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "local_wheel_photo_payloads.jsonl"
        self.items = []

    def enqueue(self, payload):
        item = dict(payload or {})
        self.items.append(item)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(
            "[local-wheel-photo] photoUrl={photoUrl} type={type} cleanValue={cleanValue}".format(
                **{
                    "photoUrl": item.get("photoUrl", ""),
                    "type": item.get("type", ""),
                    "cleanValue": item.get("cleanValue", ""),
                }
            ),
            flush=True,
        )

    def close(self):
        return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run only the CleaningCar wheel sidechain and emit wheel-photo payloads."
    )
    parser.add_argument("--project-root", default=".", help="CleaningCar project root on the board.")
    parser.add_argument("--config", default="configs/config.json", help="Config path, relative to project root if needed.")
    parser.add_argument("--left-source", default="", help="Override wheel.left_source.")
    parser.add_argument("--right-source", default="", help="Override wheel.right_source.")
    parser.add_argument("--wheel-photo-url", default="", help="Override POST /api/vehicle/wheel-photo URL.")
    parser.add_argument("--api-token", default="", help="Bearer token for the mock or real API.")
    parser.add_argument("--photo-base-dir", default="", help="Override system.wheel_photo_base_dir.")
    parser.add_argument("--output-dir", default="", help="Probe output dir. Default: wheel_probe_output/<timestamp>.")
    parser.add_argument("--duration", type=float, default=120.0, help="Probe duration in seconds.")
    parser.add_argument("--poll-interval", type=float, default=0.2, help="How often to bind cache entries to the fake track.")
    parser.add_argument("--status-interval", type=float, default=5.0, help="How often to print stats.")
    parser.add_argument("--flush-timeout", type=float, default=10.0, help="Wait for upload queue to drain on exit.")
    parser.add_argument("--track-id", type=int, default=1, help="Fake main-camera track id used for binding.")
    parser.add_argument("--image-quality", type=int, default=85, help="JPEG quality for cached wheel frames.")
    parser.add_argument("--hw-decode", action="store_true", help="Prefer hardware decode for wheel RTSP readers.")
    parser.add_argument("--force-event-driven", choices=["keep", "true", "false"], default="keep")
    parser.add_argument("--target-fps", type=float, default=0.0, help="Override wheel.target_fps when > 0.")
    parser.add_argument("--photo-bucket-seconds", type=float, default=0.0, help="Override wheel.photo_bucket_seconds when > 0.")
    parser.add_argument("--photo-min-score", type=float, default=-1.0, help="Override wheel.photo_min_score when >= 0.")
    parser.add_argument("--dry-run", action="store_true", help="Load config and print resolved settings without starting RTSP/RKNN.")
    return parser.parse_args()


def resolve_path(base, value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (Path(base) / path).resolve()


def add_project_to_path(project_root):
    project_root = Path(project_root).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def count_history(track_state):
    history = track_state.get("wheel_photo_history") or {}
    result = {}
    for side in ("left", "right"):
        side_history = history.get(side) or {}
        candidates = 0
        representatives = 0
        for bucket in side_history.values():
            if not isinstance(bucket, dict):
                continue
            candidates += len(bucket.get("candidates") or [])
            if isinstance(bucket.get("representative"), dict):
                representatives += 1
        result[side] = {
            "buckets": len(side_history),
            "candidates": candidates,
            "representatives": representatives,
        }
    return result


def flatten_photo_history(track_state):
    photos = []
    history = track_state.get("wheel_photo_history") or {}
    for side in ("left", "right"):
        for bucket_id, bucket in (history.get(side) or {}).items():
            if not isinstance(bucket, dict):
                continue
            rep = bucket.get("representative")
            if not isinstance(rep, dict):
                continue
            photos.append(
                {
                    "side": side,
                    "bucket": str(bucket_id),
                    "photoUrl": rep.get("photoUrl", ""),
                    "type": rep.get("type", ""),
                    "cleanValue": rep.get("cleanValue", 0),
                    "score": rep.get("score", 0.0),
                    "capture_ts": rep.get("capture_ts", 0.0),
                    "entryId": rep.get("entryId", 0),
                    "seq": rep.get("seq", 0),
                }
            )
    photos.sort(key=lambda item: (float(item.get("capture_ts") or 0.0), item.get("side", "")))
    return photos


def wait_uploader_drain(uploader, timeout):
    db = getattr(uploader, "db", None)
    if db is None:
        return None
    deadline = time.time() + max(0.0, float(timeout))
    pending = db.pending()
    while pending and time.time() < deadline:
        time.sleep(0.2)
        pending = db.pending()
    return pending


def main():
    args = parse_args()
    project_root = add_project_to_path(args.project_root)
    config_path = resolve_path(project_root, args.config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else project_root / "wheel_probe_output" / timestamp
    if not output_dir.is_absolute():
        output_dir = (project_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from cleaningcar.events import EventManager, WheelPhotoUploader
    from cleaningcar.runtime_config import load_config
    from cleaningcar.wheel import WheelDetectionService, resolve_wheel_settings

    config = load_config(str(config_path))
    wheel_cfg = config.setdefault("wheel", {})
    if args.left_source:
        wheel_cfg["left_source"] = args.left_source
    if args.right_source:
        wheel_cfg["right_source"] = args.right_source
    if args.target_fps > 0:
        wheel_cfg["target_fps"] = args.target_fps
    if args.photo_bucket_seconds > 0:
        wheel_cfg["photo_bucket_seconds"] = args.photo_bucket_seconds
    if args.photo_min_score >= 0:
        wheel_cfg["photo_min_score"] = args.photo_min_score
    if args.force_event_driven == "true":
        wheel_cfg["event_driven"] = True
    elif args.force_event_driven == "false":
        wheel_cfg["event_driven"] = False
    if wheel_cfg.get("left_source") or wheel_cfg.get("right_source"):
        wheel_cfg["enabled"] = True

    wheel_photo_url = args.wheel_photo_url or config.get("wheel_photo_url") or ""
    photo_base_dir = args.photo_base_dir or config.get("wheel_photo_base_dir") or "/data/ftp"
    settings = resolve_wheel_settings(config, base_dir=config_path.parent)
    resolved = {
        "config": str(config_path),
        "output_dir": str(output_dir),
        "wheel_enabled": settings.get("enabled"),
        "left_source": settings.get("left_source"),
        "right_source": settings.get("right_source"),
        "model": settings.get("model"),
        "classes": settings.get("classes"),
        "target_fps": settings.get("target_fps"),
        "event_driven": settings.get("event_driven"),
        "bind_window_seconds": settings.get("bind_window_seconds"),
        "wheel_photo_url": wheel_photo_url,
        "photo_base_dir": photo_base_dir,
        "photo_bucket_seconds": wheel_cfg.get("photo_bucket_seconds", 1.0),
        "photo_min_score": wheel_cfg.get("photo_min_score", 0.3),
    }
    (output_dir / "resolved_settings.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(resolved, ensure_ascii=False, indent=2), flush=True)

    if args.dry_run:
        return 0
    if not settings.get("enabled"):
        print("wheel sidechain is disabled or no source is configured", file=sys.stderr)
        return 2

    service = WheelDetectionService(
        config=config,
        base_dir=config_path.parent,
        hw_decode=bool(args.hw_decode),
        image_quality=args.image_quality,
    )
    uploader = None
    if wheel_photo_url:
        uploader = WheelPhotoUploader(
            wheel_photo_url,
            args.api_token or config.get("api_token", ""),
            queue_path=output_dir / "wheel_photo_queue.db",
        )
    else:
        uploader = LocalWheelPhotoCollector(output_dir)

    event_manager = EventManager(
        config,
        fps=25.0,
        frame_size=(128, 128),
        zone_manager=DummyZoneManager(),
        uploader=None,
        wheel_result_provider=service,
        wheel_photo_uploader=uploader,
        wheel_photo_base_dir=photo_base_dir,
        session_id="probe_" + datetime.now().strftime("%H%M%S"),
        wheel_photo_bucket_seconds=float(wheel_cfg.get("photo_bucket_seconds", 1.0)),
        wheel_photo_min_score=float(wheel_cfg.get("photo_min_score", 0.3)),
    )
    track_state = {
        "wheel_results_locked": {},
        "wheel_photo_history": {"left": {}, "right": {}},
        "wheel_photo_seq": {"left": 0, "right": 0},
    }

    started = False
    try:
        started = service.start()
        if not started:
            print("failed to start wheel sidechain; check sources/model/runtime", file=sys.stderr)
            return 3
        service.update_track_activity(args.track_id, track_state, frame_ts=time.time(), active=True)
        deadline = time.time() + max(1.0, float(args.duration))
        next_status = 0.0
        print("wheel probe started; press Ctrl+C to stop early", flush=True)
        while time.time() < deadline:
            now = time.time()
            event_manager._update_track_wheel_results(args.track_id, track_state, frame_ts=now)
            if now >= next_status:
                stats = service.snapshot_stats()
                print(
                    "[status] stats={stats} history={history} locked={locked}".format(
                        stats=json.dumps(stats, ensure_ascii=False),
                        history=json.dumps(count_history(track_state), ensure_ascii=False),
                        locked=list((track_state.get("wheel_results_locked") or {}).keys()),
                    ),
                    flush=True,
                )
                next_status = now + max(1.0, float(args.status_interval))
            time.sleep(max(0.05, float(args.poll_interval)))
    except KeyboardInterrupt:
        print("wheel probe interrupted; emitting collected photos", flush=True)
    finally:
        try:
            service.update_track_activity(args.track_id, track_state, frame_ts=time.time(), active=False)
        except Exception:
            pass
        event_manager._enqueue_wheel_photos(track_state)
        pending = wait_uploader_drain(uploader, args.flush_timeout)
        if pending is not None:
            print("upload queue pending={}".format(pending), flush=True)
        if started:
            service.stop()
        try:
            uploader.close()
        except Exception:
            pass
        summary = {
            "finishedAt": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "history": count_history(track_state),
            "lockedSides": sorted((track_state.get("wheel_results_locked") or {}).keys()),
            "photos": flatten_photo_history(track_state),
        }
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print("summary saved: {}".format(summary_path), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
