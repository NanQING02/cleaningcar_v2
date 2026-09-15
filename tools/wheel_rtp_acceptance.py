#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import threading
import time
from pathlib import Path

import psutil


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cleaningcar.wheel_gstreamer import create_wheel_gstreamer_capture


def _percentile(values, percentile):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return -1.0
    rank = (len(ordered) - 1) * float(percentile) / 100.0
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _load_sources(config_path):
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    wheel = config.get("wheel", {}) or {}
    video = config.get("video", {}) or {}
    sources = {side: str(wheel.get(f"{side}_source") or "").strip() for side in ("left", "right")}
    if not all(sources.values()):
        raise RuntimeError("left/right wheel source is required")
    return sources, {
        "latency_ms": max(0, int(video.get("rtsp_latency_ms", 200))),
        "max_buffers": max(1, int(video.get("rtsp_appsink_max_buffers", 1))),
        "timeout_seconds": max(0.1, float(video.get("reader_frame_timeout_seconds", 5.0))),
    }


def _open_pair(sources, settings):
    captures = {}
    errors = {}
    threads = []

    def open_one(side):
        try:
            captures[side] = create_wheel_gstreamer_capture(
                source=sources[side],
                side=side,
                latency_ms=settings["latency_ms"],
                max_buffers=settings["max_buffers"],
                open_timeout_seconds=settings["timeout_seconds"],
                read_timeout_seconds=settings["timeout_seconds"],
                ignore_broken_rtp_info=True,
            )
        except Exception as exc:
            errors[side] = f"{type(exc).__name__}: {exc}"

    for side in ("left", "right"):
        thread = threading.Thread(target=open_one, args=(side,), name=f"accept-open-{side}")
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join(timeout=settings["timeout_seconds"] + 4.0)
    for side in ("left", "right"):
        capture = captures.get(side)
        if capture is None:
            errors.setdefault(side, "open thread did not return")
        elif not capture.isOpened():
            errors.setdefault(side, str(capture.diagnostics().get("last_read_error") or "capture not opened"))
    return captures, errors


def _safe_diag(capture):
    diag = dict(capture.diagnostics() or {})
    return {
        "state": diag.get("state"),
        "rtsp_negotiation_seconds": diag.get("rtsp_negotiation_seconds"),
        "first_rtp_seconds": diag.get("first_rtp_seconds"),
        "first_bgr_seconds": diag.get("first_bgr_seconds"),
        "first_rtp_seqnum": diag.get("first_rtp_seqnum"),
        "advertised_seqnum_base": diag.get("advertised_seqnum_base"),
        "advertised_clock_base": diag.get("advertised_clock_base"),
        "caps_sanitized": bool(diag.get("caps_sanitized")),
        "width": int(diag.get("width") or 0),
        "height": int(diag.get("height") or 0),
        "fps": float(diag.get("fps") or 0.0),
        "sample_count": int(diag.get("sample_count") or 0),
        "last_error": str(diag.get("last_read_error") or ""),
    }


def _release_pair(captures):
    for capture in captures.values():
        try:
            capture.release()
            capture.release()
        except Exception:
            pass


def run_rounds(sources, settings, rounds, pause_seconds, first_frame_limit):
    process = psutil.Process()
    baseline_threads = process.num_threads()
    baseline_connections = len(process.net_connections(kind="inet"))
    records = []
    for index in range(1, rounds + 1):
        captures, errors = _open_pair(sources, settings)
        record = {"round": index, "errors": errors, "sides": {}}
        for side in ("left", "right"):
            capture = captures.get(side)
            if capture is not None:
                record["sides"][side] = _safe_diag(capture)
        _release_pair(captures)
        time.sleep(max(0.0, pause_seconds))
        record["threads_after_release"] = process.num_threads()
        record["connections_after_release"] = len(process.net_connections(kind="inet"))
        records.append(record)
        print(json.dumps(record, ensure_ascii=False, sort_keys=True), flush=True)

    summary = {
        "kind": "rounds_summary",
        "rounds": rounds,
        "baseline_threads": baseline_threads,
        "final_threads": process.num_threads(),
        "baseline_connections": baseline_connections,
        "final_connections": len(process.net_connections(kind="inet")),
        "sides": {},
    }
    passed = True
    for side in ("left", "right"):
        diagnostics = [record["sides"].get(side, {}) for record in records]
        times = [float(item.get("first_bgr_seconds", -1.0)) for item in diagnostics if item]
        success = sum(
            1
            for item in diagnostics
            if item.get("state") == "bgr_ready"
            and item.get("caps_sanitized")
            and 0.0 <= float(item.get("first_bgr_seconds", -1.0)) <= first_frame_limit
        )
        high_half = [item.get("first_rtp_seqnum") for item in diagnostics if int(item.get("first_rtp_seqnum") or 0) > 32768]
        summary["sides"][side] = {
            "success": success,
            "max_first_bgr_seconds": max(times) if times else -1.0,
            "p95_first_bgr_seconds": _percentile(times, 95),
            "high_half_first_seqnums": high_half,
        }
        passed = passed and success == rounds
    warmed_threads = int(records[0]["threads_after_release"]) if records else baseline_threads
    max_threads_after_warmup = max(
        (int(record["threads_after_release"]) for record in records[1:]),
        default=warmed_threads,
    )
    summary["warmed_threads"] = warmed_threads
    summary["max_threads_after_warmup"] = max_threads_after_warmup
    passed = passed and summary["final_threads"] <= warmed_threads + 1
    passed = passed and max_threads_after_warmup <= warmed_threads + 1
    passed = passed and summary["final_connections"] <= baseline_connections
    summary["passed"] = passed
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return passed


def run_steady(sources, settings, duration_seconds, sample_interval, first_frame_limit):
    process = psutil.Process()
    captures, errors = _open_pair(sources, settings)
    started = time.monotonic()
    start_diag = {side: _safe_diag(cap) for side, cap in captures.items()}
    max_rss = process.memory_info().rss
    cpu_samples = []
    passed = not errors and len(captures) == 2
    try:
        while passed and time.monotonic() - started < duration_seconds:
            interval = min(sample_interval, max(0.1, duration_seconds - (time.monotonic() - started)))
            cpu_samples.append(process.cpu_percent(interval=interval))
            max_rss = max(max_rss, process.memory_info().rss)
            for capture in captures.values():
                if not capture.isOpened():
                    passed = False
                    break
    finally:
        end_diag = {side: _safe_diag(cap) for side, cap in captures.items()}
        _release_pair(captures)
    elapsed = max(1e-6, time.monotonic() - started)
    sides = {}
    for side in ("left", "right"):
        first = start_diag.get(side, {})
        final = end_diag.get(side, {})
        sample_fps = (int(final.get("sample_count") or 0) - int(first.get("sample_count") or 0)) / elapsed
        source_fps = float(final.get("fps") or 0.0)
        sides[side] = {**final, "measured_sample_fps": sample_fps}
        passed = passed and final.get("state") == "bgr_ready"
        passed = passed and 0.0 <= float(final.get("first_bgr_seconds", -1.0)) <= first_frame_limit
        passed = passed and (source_fps <= 0.0 or sample_fps >= source_fps * 0.97)
    summary = {
        "kind": "steady_summary",
        "duration_seconds": elapsed,
        "errors": errors,
        "cpu_core_equivalent_mean": statistics.mean(cpu_samples) / 100.0 if cpu_samples else 0.0,
        "cpu_core_equivalent_max": max(cpu_samples) / 100.0 if cpu_samples else 0.0,
        "peak_rss_mib": max_rss / (1024.0 * 1024.0),
        "sides": sides,
        "passed": passed,
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return passed


def main():
    parser = argparse.ArgumentParser(description="车轮 RTP-Info 兼容修复实流验收，不加载模型或上传数据。")
    parser.add_argument("--config", required=True)
    parser.add_argument("--rounds", type=int, default=0)
    parser.add_argument("--steady-seconds", type=float, default=0.0)
    parser.add_argument("--pause-seconds", type=float, default=0.2)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--first-frame-limit", type=float, default=5.0)
    args = parser.parse_args()
    sources, settings = _load_sources(args.config)
    ok = True
    if args.rounds > 0:
        ok = run_rounds(sources, settings, args.rounds, args.pause_seconds, args.first_frame_limit) and ok
    if args.steady_seconds > 0.0:
        ok = run_steady(sources, settings, args.steady_seconds, args.sample_interval, args.first_frame_limit) and ok
    if args.rounds <= 0 and args.steady_seconds <= 0.0:
        parser.error("provide --rounds or --steady-seconds")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
