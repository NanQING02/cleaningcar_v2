#!/usr/bin/env python3
import argparse
import json
import math
import re
import statistics
from pathlib import Path


MONITOR_RE = re.compile(
    r'proc_cpu=(?P<cpu>[0-9.]+)%.*?rss=(?P<rss>[0-9.]+)MB.*?threads=(?P<threads>[0-9]+).*?uptime=(?P<uptime>[0-9]+)s'
)
PERF_RE = re.compile(r'解码FPS=(?P<decode>[0-9.]+) 管线FPS=(?P<pipeline>[0-9.]+)')
DIAG_RE = re.compile(
    r'q=task:(?P<task>[0-9]+)/(?P<capacity>[0-9]+),result:(?P<result>[0-9]+),pending:(?P<pending>[0-9]+) '
    r'drop=win:[0-9]+,total:(?P<drop>[0-9]+).*?main_reader_reconnect=win:[0-9]+,total:(?P<reconnect>[0-9]+)'
)


def percentile(values, ratio):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.ceil(len(ordered) * ratio)) - 1))
    return float(ordered[index])


def linear_slope_per_hour(points):
    if len(points) < 2:
        return 0.0
    xs = [float(point[0]) / 3600.0 for point in points]
    ys = [float(point[1]) for point in points]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((value - x_mean) ** 2 for value in xs)
    if denominator <= 0.0:
        return 0.0
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


def analyze(path, warmup_seconds):
    monitor = []
    perf = []
    diag = []
    with Path(path).open('r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            match = MONITOR_RE.search(line)
            if match:
                item = {key: float(value) for key, value in match.groupdict().items()}
                if item['uptime'] >= warmup_seconds:
                    monitor.append(item)
            match = PERF_RE.search(line)
            if match:
                perf.append({key: float(value) for key, value in match.groupdict().items()})
            match = DIAG_RE.search(line)
            if match:
                diag.append({key: int(value) for key, value in match.groupdict().items()})
    rss = [item['rss'] for item in monitor]
    cpu = [item['cpu'] for item in monitor]
    return {
        'path': str(path),
        'warmup_seconds': warmup_seconds,
        'monitor_samples': len(monitor),
        'observed_hours': round((monitor[-1]['uptime'] - monitor[0]['uptime']) / 3600.0, 3) if len(monitor) >= 2 else 0.0,
        'rss_mb': {
            'first': rss[0] if rss else 0.0,
            'last': rss[-1] if rss else 0.0,
            'mean': round(statistics.fmean(rss), 3) if rss else 0.0,
            'median': round(statistics.median(rss), 3) if rss else 0.0,
            'p95': round(percentile(rss, 0.95), 3),
            'max': max(rss) if rss else 0.0,
            'linear_slope_mb_per_hour': round(
                linear_slope_per_hour([(item['uptime'], item['rss']) for item in monitor]),
                3,
            ),
        },
        'cpu_percent': {
            'mean': round(statistics.fmean(cpu), 3) if cpu else 0.0,
            'median': round(statistics.median(cpu), 3) if cpu else 0.0,
            'p95': round(percentile(cpu, 0.95), 3),
            'max': max(cpu) if cpu else 0.0,
        },
        'pipeline_fps': round(statistics.fmean(item['pipeline'] for item in perf), 3) if perf else 0.0,
        'decode_fps': round(statistics.fmean(item['decode'] for item in perf), 3) if perf else 0.0,
        'max_task_queue': max((item['task'] for item in diag), default=0),
        'max_result_queue': max((item['result'] for item in diag), default=0),
        'dropped_frames_total': max((item['drop'] for item in diag), default=0),
        'reader_reconnect_total': max((item['reconnect'] for item in diag), default=0),
    }


def main():
    parser = argparse.ArgumentParser(description='汇总CleaningCar monitor/perf/diag日志。')
    parser.add_argument('logs', nargs='+')
    parser.add_argument('--warmup-seconds', type=float, default=300.0)
    args = parser.parse_args()
    print(json.dumps(
        [analyze(path, max(0.0, args.warmup_seconds)) for path in args.logs],
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == '__main__':
    main()
