#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cleaningcar.video_io import create_video_reader  # noqa: E402


RGA_KERNEL_ERROR_RE = re.compile(
    r"RGA_MMU unsupported memory larger than 4G|"
    r"job buffer map failed|rga_job_commit.*failed|"
    r"request commit failed|submit failed",
    re.IGNORECASE,
)
DMESG_TIMESTAMP_RE = re.compile(r"^\[\s*(\d+(?:\.\d+)?)\]")


def _read_dmesg_lines():
    try:
        proc = subprocess.run(
            ["dmesg", "--color=never"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except Exception:
        return None
    return (proc.stdout or "").splitlines()


def _kernel_cursor():
    lines = _read_dmesg_lines()
    if lines is None:
        return -1.0
    latest = 0.0
    for line in lines:
        match = DMESG_TIMESTAMP_RE.match(line)
        if match:
            latest = max(latest, float(match.group(1)))
    return latest


def _kernel_rga_error_count_since(cursor):
    lines = _read_dmesg_lines()
    if lines is None or cursor < 0.0:
        return -1
    count = 0
    for line in lines:
        match = DMESG_TIMESTAMP_RE.match(line)
        if not match or float(match.group(1)) <= cursor:
            continue
        if RGA_KERNEL_ERROR_RE.search(line):
            count += 1
    return count


def _redact(text):
    value = str(text or "")
    value = re.sub(r"(?:rtsp|rtsps)://[^\s]+", "<stream-redacted>", value, flags=re.IGNORECASE)
    return value


def main():
    parser = argparse.ArgumentParser(description="Single-stream FFmpeg RKMPP+RGA safety probe")
    parser.add_argument("--config", required=True)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--core", choices=("auto", "rga3_core0", "rga3_core1"), default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    video = config.setdefault("video", {})
    source = video.get("source")
    if not isinstance(source, str) or not source.strip():
        raise SystemExit("video.source missing")
    video["decode_backend"] = "ffmpeg_rga"
    rga = video.setdefault("ffmpeg_rga", {})
    if args.core is not None:
        rga["core"] = args.core
    if args.width is not None or args.height is not None:
        if args.width is None or args.height is None:
            raise SystemExit("--width and --height must be provided together")
        rga["width"] = args.width
        rga["height"] = args.height

    reader_args = SimpleNamespace(hw_decode=True, _config=config)
    kernel_cursor = _kernel_cursor()
    cap, meta = create_video_reader(source, reader_args)
    if cap is None or not cap.isOpened():
        print(f"probe_open_failed meta={meta}")
        return 2

    started = time.monotonic()
    deadline = started + max(1.0, float(args.duration))
    frames = 0
    read_failed = False
    try:
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if not ok or frame is None:
                read_failed = True
                break
            frames += 1
    finally:
        diag = cap.diagnostics() if hasattr(cap, "diagnostics") else {}
        cap.release()

    elapsed = max(0.001, time.monotonic() - started)
    delta = _kernel_rga_error_count_since(kernel_cursor)
    print(
        "probe_result "
        f"frames={frames} elapsed={elapsed:.2f}s fps={frames / elapsed:.2f} "
        f"read_failed={int(read_failed)} kernel_rga_error_delta={delta} "
        f"breaker={int(bool(diag.get('circuit_breaker_tripped')))}"
    )
    if diag.get("last_read_error"):
        print(f"last_read_error={_redact(diag['last_read_error'])}")
    for line in (diag.get("recent_error_lines") or [])[-6:]:
        print(f"stderr={_redact(line)}")

    if read_failed or diag.get("circuit_breaker_tripped") or delta > 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
