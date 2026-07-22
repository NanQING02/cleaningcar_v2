#!/usr/bin/env python3
import argparse
import copy
import json
import os
import re
import signal
import subprocess
import sys
import threading
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
STREAM_RE = re.compile(r"(?:rtsp|rtsps)://[^\s]+", re.IGNORECASE)


def _redact(value):
    return STREAM_RE.sub("<stream-redacted>", str(value or ""))


def _dmesg_lines():
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


def _dmesg_cursor():
    latest = 0.0
    lines = _dmesg_lines()
    if lines is None:
        return -1.0
    for line in lines:
        match = DMESG_TIMESTAMP_RE.match(line)
        if match:
            latest = max(latest, float(match.group(1)))
    return latest


def _new_kernel_rga_errors(cursor):
    newest = cursor
    matches = []
    lines = _dmesg_lines()
    if lines is None:
        return cursor, matches
    for line in lines:
        match = DMESG_TIMESTAMP_RE.match(line)
        if not match:
            continue
        timestamp = float(match.group(1))
        newest = max(newest, timestamp)
        if timestamp > cursor and RGA_KERNEL_ERROR_RE.search(line):
            matches.append(_redact(line))
    return newest, matches


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _nested(config, *keys):
    value = config
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _load_sources(config_path, bypass_config_path):
    config = _read_json(config_path)
    sources = [
        ("main", _nested(config, "video", "source")),
        ("wheel_left", _nested(config, "wheel", "left_source")),
        ("wheel_right", _nested(config, "wheel", "right_source")),
    ]
    if bypass_config_path:
        bypass = _read_json(bypass_config_path)
        sources.append(("bypass", _nested(bypass, "video", "source")))
    missing = [label for label, source in sources if not source]
    if missing:
        raise ValueError("missing stream fields: " + ",".join(missing))
    return config, sources


def _cpu_ticks():
    try:
        fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
        values = [int(value) for value in fields]
    except Exception:
        return None
    total = sum(values)
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return idle, total


def _system_snapshot(previous_cpu):
    current_cpu = _cpu_ticks()
    cpu_percent = None
    if previous_cpu and current_cpu:
        idle_delta = current_cpu[0] - previous_cpu[0]
        total_delta = current_cpu[1] - previous_cpu[1]
        if total_delta > 0:
            cpu_percent = round(100.0 * (1.0 - idle_delta / total_delta), 1)

    memory = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable"}:
                memory[key] = int(value.strip().split()[0])
    except Exception:
        pass

    temperatures = []
    for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            raw = float(path.read_text(encoding="utf-8").strip())
            temperatures.append(raw / 1000.0 if raw > 1000 else raw)
        except Exception:
            continue

    rga_load = {}
    try:
        text = Path("/sys/kernel/debug/rkrga/load").read_text(encoding="utf-8")
        for match in re.finditer(
            r"scheduler\[(\d+)\]:\s+(\S+).*?load\s*=\s*(\d+)%",
            text,
            flags=re.DOTALL,
        ):
            rga_load[f"{match.group(2)}_{match.group(1)}"] = int(match.group(3))
    except Exception:
        pass

    try:
        load1 = round(os.getloadavg()[0], 2)
    except Exception:
        load1 = None
    return current_cpu, {
        "cpu_percent": cpu_percent,
        "load1": load1,
        "mem_available_mb": round(memory.get("MemAvailable", 0) / 1024.0, 1),
        "mem_total_mb": round(memory.get("MemTotal", 0) / 1024.0, 1),
        "max_temp_c": round(max(temperatures), 1) if temperatures else None,
        "rga_load_percent": rga_load,
    }


class JsonlRecorder:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event, **fields):
        record = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "event": event,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        print(line, flush=True)


class ReaderWorker(threading.Thread):
    def __init__(self, label, source, config, core, stop_event, recorder):
        super().__init__(name=f"rga-soak-{label}", daemon=True)
        self.label = label
        self.source = source
        self.config = copy.deepcopy(config)
        self.core = core
        self.stop_event = stop_event
        self.recorder = recorder
        self.capture = None
        self.frames = 0
        self.started_mono = 0.0
        self.failure = ""
        self.diagnostics = {}

    def run(self):
        video = self.config.setdefault("video", {})
        video["source"] = self.source
        video["hw_decode"] = True
        video["decode_backend"] = "ffmpeg_rga"
        rga = video.setdefault("ffmpeg_rga", {})
        rga.update(
            {
                "core": self.core,
                "width": 0,
                "height": 0,
                "async_depth": 1,
                "afbc": False,
                "breaker_enabled": True,
                "breaker_error_threshold": 1,
                "breaker_window_seconds": 60.0,
                "breaker_cooldown_seconds": 300.0,
            }
        )
        args = SimpleNamespace(hw_decode=True, _config=self.config)
        self.started_mono = time.monotonic()
        try:
            cap, meta = create_video_reader(self.source, args)
            self.capture = cap
            if cap is None or not cap.isOpened():
                reason = (meta or {}).get("fallback_reason", "open_failed")
                self.failure = f"open_failed:{reason}"
                self.stop_event.set()
                return
            self.recorder.write("reader_opened", label=self.label, core=self.core)
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.failure = "read_failed"
                    self.stop_event.set()
                    break
                self.frames += 1
        except Exception as exc:
            self.failure = "reader_exception:" + _redact(exc)
            self.stop_event.set()
        finally:
            cap = self.capture
            if cap is not None:
                try:
                    if hasattr(cap, "diagnostics"):
                        self.diagnostics = cap.diagnostics() or {}
                finally:
                    cap.release()

    def snapshot(self):
        elapsed = max(0.001, time.monotonic() - self.started_mono) if self.started_mono else 0.001
        diag = dict(self.diagnostics)
        cap = self.capture
        if cap is not None and hasattr(cap, "diagnostics"):
            try:
                diag = cap.diagnostics() or diag
            except Exception:
                pass
        return {
            "label": self.label,
            "core": self.core,
            "frames": self.frames,
            "fps": round(self.frames / elapsed, 2),
            "failure": _redact(self.failure),
            "breaker": bool(diag.get("circuit_breaker_tripped")),
            "last_read_error": _redact(diag.get("last_read_error", "")),
        }


def main():
    parser = argparse.ArgumentParser(description="Four-stream FFmpeg RKMPP+RGA soak test")
    parser.add_argument("--config", required=True)
    parser.add_argument("--bypass-config", required=True)
    parser.add_argument("--duration", type=float, default=7200.0)
    parser.add_argument("--sample-interval", type=float, default=60.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config, sources = _load_sources(args.config, args.bypass_config)
    stop_event = threading.Event()
    interrupted = {"signal": 0}

    def stop_handler(signum, _frame):
        interrupted["signal"] = int(signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)

    recorder = JsonlRecorder(args.output)
    workers = []
    for index, (label, source) in enumerate(sources):
        core = "rga3_core0" if index % 2 == 0 else "rga3_core1"
        workers.append(ReaderWorker(label, source, config, core, stop_event, recorder))

    cursor = _dmesg_cursor()
    deadline = time.monotonic() + max(1.0, args.duration)
    recorder.write(
        "soak_started",
        duration_seconds=max(1.0, args.duration),
        stream_count=len(workers),
        core_assignment={worker.label: worker.core for worker in workers},
        dmesg_available=cursor >= 0.0,
    )
    for worker in workers:
        worker.start()

    kernel_errors = []
    next_sample = time.monotonic()
    previous_cpu = _cpu_ticks()
    while not stop_event.is_set() and time.monotonic() < deadline:
        now = time.monotonic()
        cursor, new_errors = _new_kernel_rga_errors(cursor)
        if new_errors:
            kernel_errors.extend(new_errors)
            recorder.write("kernel_rga_error", lines=new_errors[-8:])
            stop_event.set()
            break
        if now >= next_sample:
            previous_cpu, system = _system_snapshot(previous_cpu)
            recorder.write(
                "sample",
                readers=[worker.snapshot() for worker in workers],
                system=system,
            )
            next_sample = now + max(5.0, args.sample_interval)
        stop_event.wait(1.0)

    stop_event.set()
    for worker in workers:
        worker.join(timeout=10.0)
    for worker in workers:
        cap = worker.capture
        if worker.is_alive() and cap is not None:
            try:
                cap.release()
            except Exception:
                pass
    for worker in workers:
        worker.join(timeout=3.0)

    snapshots = [worker.snapshot() for worker in workers]
    failed = bool(
        kernel_errors
        or interrupted["signal"]
        or any(worker.is_alive() or item["failure"] or item["breaker"] for worker, item in zip(workers, snapshots))
    )
    completed = not failed and time.monotonic() >= deadline
    recorder.write(
        "soak_finished",
        completed=completed,
        failed=failed,
        signal=interrupted["signal"],
        kernel_rga_error_count=len(kernel_errors),
        readers=snapshots,
    )
    return 0 if completed else 3


if __name__ == "__main__":
    raise SystemExit(main())
