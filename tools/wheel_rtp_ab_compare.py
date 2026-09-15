#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from queue import Empty


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_config(config_path):
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    wheel = config.get("wheel", {}) or {}
    video = config.get("video", {}) or {}
    sources = {side: str(wheel.get(f"{side}_source") or "").strip() for side in ("left", "right")}
    if not all(sources.values()):
        raise RuntimeError("left/right wheel source is required")
    settings = {
        "latency_ms": max(0, int(video.get("rtsp_latency_ms", 200))),
        "max_buffers": max(1, int(video.get("rtsp_appsink_max_buffers", 1))),
        "timeout_seconds": max(0.1, float(video.get("reader_frame_timeout_seconds", 5.0))),
    }
    return sources, settings


def _capture_worker(side, source, settings, compatibility_enabled, ready, start, output):
    from cleaningcar.wheel_gstreamer import create_wheel_gstreamer_capture

    ready.set()
    start.wait(timeout=5.0)
    began = time.monotonic()
    capture = create_wheel_gstreamer_capture(
        source=source,
        side=side,
        latency_ms=settings["latency_ms"],
        max_buffers=settings["max_buffers"],
        open_timeout_seconds=settings["timeout_seconds"],
        read_timeout_seconds=settings["timeout_seconds"],
        ignore_broken_rtp_info=compatibility_enabled,
    )
    try:
        diag = capture.diagnostics()
        output.put(
            {
                "side": side,
                "mode": "fixed" if compatibility_enabled else "unfixed",
                "elapsed_seconds": time.monotonic() - began,
                "state": str(diag.get("state") or ""),
                "first_rtp_seconds": float(diag.get("first_rtp_seconds", -1.0)),
                "first_bgr_seconds": float(diag.get("first_bgr_seconds", -1.0)),
                "first_rtp_seqnum": (
                    int(diag["first_rtp_seqnum"]) if diag.get("first_rtp_seqnum") is not None else None
                ),
                "advertised_seqnum_base": (
                    int(diag["advertised_seqnum_base"])
                    if diag.get("advertised_seqnum_base") is not None
                    else None
                ),
                "advertised_clock_base": (
                    int(diag["advertised_clock_base"])
                    if diag.get("advertised_clock_base") is not None
                    else None
                ),
                "caps_sanitized": bool(diag.get("caps_sanitized")),
                "width": int(diag.get("width") or 0),
                "height": int(diag.get("height") or 0),
                "fps": float(diag.get("fps") or 0.0),
                "last_error": str(diag.get("last_read_error") or ""),
            }
        )
    finally:
        capture.release()


def _run_pair(context, side, source, settings, round_index):
    output = context.Queue()
    start = context.Event()
    processes = []
    ready_events = []
    for enabled in (False, True):
        ready = context.Event()
        process = context.Process(
            target=_capture_worker,
            args=(side, source, settings, enabled, ready, start, output),
            name=f"wheel-ab-{side}-{'fixed' if enabled else 'unfixed'}",
        )
        process.start()
        processes.append(process)
        ready_events.append(ready)
    for ready in ready_events:
        ready.wait(timeout=5.0)
    released_at = time.time()
    start.set()
    deadline = time.monotonic() + settings["timeout_seconds"] + 6.0
    for process in processes:
        process.join(timeout=max(0.0, deadline - time.monotonic()))
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=1.0)
    results = []
    for _ in range(2):
        try:
            results.append(output.get(timeout=1.0))
        except Empty:
            break
    by_mode = {item["mode"]: item for item in results}
    record = {
        "round": round_index,
        "side": side,
        "start_skew_control": "shared multiprocessing event",
        "start_epoch": released_at,
        "unfixed": by_mode.get("unfixed"),
        "fixed": by_mode.get("fixed"),
    }
    unfixed = record["unfixed"] or {}
    fixed = record["fixed"] or {}
    high_half = int(unfixed.get("first_rtp_seqnum") or 0) > 32768 and int(fixed.get("first_rtp_seqnum") or 0) > 32768
    record["high_half_comparison"] = high_half
    record["demonstrates_fix"] = bool(
        high_half
        and unfixed.get("state") != "bgr_ready"
        and fixed.get("state") == "bgr_ready"
        and float(fixed.get("first_bgr_seconds", -1.0)) <= settings["timeout_seconds"]
        and unfixed.get("advertised_seqnum_base") == 1
        and fixed.get("advertised_seqnum_base") == 1
        and not unfixed.get("caps_sanitized")
        and fixed.get("caps_sanitized")
    )
    return record


def main():
    parser = argparse.ArgumentParser(description="同步对比车轮 RTP-Info 兼容修复开启和关闭时的真实首帧。")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--side", choices=("left", "right", "both"), default="both")
    parser.add_argument("--pause-seconds", type=float, default=0.5)
    args = parser.parse_args()
    sources, settings = _load_config(args.config)
    context = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    sides = ("left", "right") if args.side == "both" else (args.side,)
    records = []
    for index in range(1, max(1, args.rounds) + 1):
        for side in sides:
            record = _run_pair(context, side, sources[side], settings, index)
            records.append(record)
            print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)
            time.sleep(max(0.0, args.pause_seconds))
    summary = {
        "kind": "ab_summary",
        "comparisons": len(records),
        "high_half_comparisons": sum(int(item["high_half_comparison"]) for item in records),
        "demonstrates_fix": sum(int(item["demonstrates_fix"]) for item in records),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    raise SystemExit(0 if summary["demonstrates_fix"] > 0 else 2)


if __name__ == "__main__":
    main()
