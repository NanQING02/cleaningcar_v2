#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass, field
from pathlib import Path


OPEN_RE = re.compile(
    r"\[wheel:(left|right)\] reader opened .*?open_count=(\d+) .*?reopen_count=(\d+) "
    r".*?reason=([a-z_]+) .*?open_delay=([0-9.]+)s .*?last_frame_gap=([-0-9.]+)s"
)
LEGACY_OPEN_RE = re.compile(r"\[wheel:(left|right)\] reader opened mode=([a-z]+) backend=([a-z0-9_]+)")
STALL_RE = re.compile(
    r"\[wheel:(left|right)\] reader stalled, reconnect #(\d+) "
    r"reason=([a-z_]+) consecutive_fails=(\d+) threshold=(\d+) "
    r"frames_since_open=(\d+) last_frame_gap=([-0-9.]+)s open_age=([-0-9.]+)s"
)
LOST_RE = re.compile(
    r"\[wheel:(left|right)\] reader lost opened-state, reconnect #(\d+) "
    r"reason=([a-z_]+) frames_since_open=(\d+) last_frame_gap=([-0-9.]+)s open_age=([-0-9.]+)s"
)
PERF_RE = re.compile(
    r"车轮\[l:([0-9.]+)/([0-9.]+)fps/([0-9.]+)ms/c(\d+), "
    r"r:([0-9.]+)/([0-9.]+)fps/([0-9.]+)ms/c(\d+)\]"
)
MONITOR_RE = re.compile(r"\[monitor\] risk=([^ ]+)")
DIAG_RE = re.compile(r"\[diag\] flags=([^ ]+)")


@dataclass
class SideState:
    opened: int = 0
    stalled: int = 0
    lost_opened_state: int = 0
    last_open_reason: str = ""
    last_stall_reason: str = ""
    max_last_frame_gap: float = 0.0
    max_open_age: float = 0.0
    max_open_delay: float = 0.0
    max_frames_since_open_before_reconnect: int = 0


@dataclass
class MonitorState:
    sides: dict = field(default_factory=lambda: {"left": SideState(), "right": SideState()})
    perf_windows: int = 0
    asymmetric_windows: int = 0
    left_only_hits: int = 0
    right_only_hits: int = 0
    no_hits_windows: int = 0
    monitor_windows: int = 0
    diag_windows: int = 0
    monitor_risks: dict = field(default_factory=dict)
    diag_flags: dict = field(default_factory=dict)


def _inc(counter: dict, key: str):
    key = str(key or "").strip() or "-"
    counter[key] = int(counter.get(key, 0) or 0) + 1


def _record_csv_flags(counter: dict, raw: str):
    raw = str(raw or "").strip()
    if not raw:
        _inc(counter, "-")
        return
    for item in raw.split(","):
        _inc(counter, item.strip() or "-")


def parse_args():
    parser = argparse.ArgumentParser(description="监控或离线分析 wheel reader 重连与左右命中不对称。")
    parser.add_argument("logfile", help="推理日志路径。")
    parser.add_argument("--follow", action="store_true", help="持续跟随日志。")
    parser.add_argument("--summary-interval", type=float, default=10.0, help="follow 模式下汇总间隔秒数。")
    return parser.parse_args()


def _update_side_max(state: SideState, gap: float, open_age: float, open_delay: float | None, frames_since_open: int | None):
    state.max_last_frame_gap = max(state.max_last_frame_gap, max(0.0, float(gap)))
    state.max_open_age = max(state.max_open_age, max(0.0, float(open_age)))
    if open_delay is not None:
        state.max_open_delay = max(state.max_open_delay, max(0.0, float(open_delay)))
    if frames_since_open is not None:
        state.max_frames_since_open_before_reconnect = max(
            state.max_frames_since_open_before_reconnect,
            max(0, int(frames_since_open)),
        )


def process_line(line: str, state: MonitorState):
    match = OPEN_RE.search(line)
    if match:
        side, _open_count, _reopen_count, reason, open_delay, last_gap = match.groups()
        side_state = state.sides[side]
        side_state.opened += 1
        side_state.last_open_reason = reason
        _update_side_max(side_state, float(last_gap), 0.0, float(open_delay), None)
        return

    match = LEGACY_OPEN_RE.search(line)
    if match:
        side, _mode, _backend = match.groups()
        side_state = state.sides[side]
        side_state.opened += 1
        if not side_state.last_open_reason:
            side_state.last_open_reason = "legacy_open_log"
        return

    match = STALL_RE.search(line)
    if match:
        side, _reconnect_count, reason, _fails, _threshold, frames_since_open, last_gap, open_age = match.groups()
        side_state = state.sides[side]
        side_state.stalled += 1
        side_state.last_stall_reason = reason
        _update_side_max(side_state, float(last_gap), float(open_age), None, int(frames_since_open))
        return

    match = LOST_RE.search(line)
    if match:
        side, _reconnect_count, reason, frames_since_open, last_gap, open_age = match.groups()
        side_state = state.sides[side]
        side_state.lost_opened_state += 1
        side_state.last_stall_reason = reason
        _update_side_max(side_state, float(last_gap), float(open_age), None, int(frames_since_open))
        return

    match = PERF_RE.search(line)
    if match:
        _l_dec, _l_inf, _l_ms, l_hits, _r_dec, _r_inf, _r_ms, r_hits = match.groups()
        l_hits = int(l_hits)
        r_hits = int(r_hits)
        state.perf_windows += 1
        if l_hits == 0 and r_hits == 0:
            state.no_hits_windows += 1
        elif l_hits > 0 and r_hits == 0:
            state.asymmetric_windows += 1
            state.left_only_hits += 1
        elif r_hits > 0 and l_hits == 0:
            state.asymmetric_windows += 1
            state.right_only_hits += 1
        return

    match = MONITOR_RE.search(line)
    if match:
        state.monitor_windows += 1
        _record_csv_flags(state.monitor_risks, match.group(1))
        return

    match = DIAG_RE.search(line)
    if match:
        state.diag_windows += 1
        _record_csv_flags(state.diag_flags, match.group(1))
        return


def print_summary(state: MonitorState, label: str):
    print(f"=== {label} ===", flush=True)
    for side in ("left", "right"):
        s = state.sides[side]
        print(
            f"[{side}] opened={s.opened} stalled={s.stalled} lost_opened_state={s.lost_opened_state} "
            f"last_open_reason={s.last_open_reason or '-'} last_reconnect_reason={s.last_stall_reason or '-'} "
            f"max_open_delay={s.max_open_delay:.2f}s max_last_frame_gap={s.max_last_frame_gap:.2f}s "
            f"max_open_age={s.max_open_age:.2f}s max_frames_before_reconnect={s.max_frames_since_open_before_reconnect}",
            flush=True,
        )
    print(
        f"[perf] windows={state.perf_windows} asymmetric={state.asymmetric_windows} "
        f"left_only={state.left_only_hits} right_only={state.right_only_hits} no_hits={state.no_hits_windows}",
        flush=True,
    )
    print(
        f"[monitor] windows={state.monitor_windows} risks={state.monitor_risks or {}}",
        flush=True,
    )
    print(
        f"[diag] windows={state.diag_windows} flags={state.diag_flags or {}}",
        flush=True,
    )


def read_existing(path: Path, state: MonitorState):
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            process_line(line, state)


def follow_file(path: Path, state: MonitorState, summary_interval: float):
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        fh.seek(0, 2)
        last_summary = time.time()
        while True:
            line = fh.readline()
            if not line:
                now = time.time()
                if now - last_summary >= max(1.0, float(summary_interval)):
                    print_summary(state, time.strftime("%Y-%m-%d %H:%M:%S"))
                    last_summary = now
                time.sleep(0.2)
                continue
            process_line(line, state)


def main():
    args = parse_args()
    path = Path(args.logfile).expanduser()
    if not path.exists():
        raise SystemExit(f"log file not found: {path}")

    state = MonitorState()
    read_existing(path, state)
    print_summary(state, "initial scan")
    if args.follow:
        follow_file(path, state, args.summary_interval)


if __name__ == "__main__":
    main()
