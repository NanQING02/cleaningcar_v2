#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='以隔离配置运行推理并采样进程资源。')
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--python', default='')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--duration', type=float, default=0.0)
    parser.add_argument('--warmup-seconds', type=float, default=20.0)
    parser.add_argument('--sample-interval', type=float, default=0.5)
    parser.add_argument('--label', default='benchmark')
    return parser.parse_args()


def read_kb_fields(path, names):
    values = {name: 0 for name in names}
    try:
        for line in Path(path).read_text(encoding='utf-8').splitlines():
            key, _sep, rest = line.partition(':')
            if key not in values:
                continue
            match = re.search(r'([0-9]+)', rest)
            if match:
                values[key] = int(match.group(1))
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        pass
    return values


def process_ticks(pid):
    try:
        text = Path(f'/proc/{pid}/stat').read_text(encoding='utf-8')
        remainder = text[text.rfind(')') + 2:].split()
        return int(remainder[11]) + int(remainder[12])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
        return 0


def child_pids(pid):
    try:
        text = Path(f'/proc/{pid}/task/{pid}/children').read_text(encoding='utf-8').strip()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return []
    return [int(value) for value in text.split() if value.isdigit()]


def process_tree(root_pid):
    pending = [int(root_pid)]
    seen = set()
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        pending.extend(child_pids(pid))
    return sorted(seen)


def sample_process_tree(root_pid):
    rss_kb = 0
    hwm_kb = 0
    threads = 0
    ticks = 0
    pids = process_tree(root_pid)
    for pid in pids:
        fields = read_kb_fields(f'/proc/{pid}/status', {'VmRSS', 'VmHWM', 'Threads'})
        rss_kb += fields['VmRSS']
        hwm_kb += fields['VmHWM']
        threads += fields['Threads']
        ticks += process_ticks(pid)
    return {
        'pids': pids,
        'rss_kb': rss_kb,
        'hwm_kb': hwm_kb,
        'threads': threads,
        'ticks': ticks,
    }


def system_memory_kb():
    fields = read_kb_fields('/proc/meminfo', {'MemTotal', 'MemAvailable'})
    return fields['MemTotal'], fields['MemAvailable']


def sanitized_config(source_path, output_dir):
    data = json.loads(Path(source_path).read_text(encoding='utf-8'))
    system = data.setdefault('system', {})
    api = system.setdefault('api', {})
    api['url'] = ''
    api['token'] = ''
    api['wheel_photo_url'] = ''
    system['performance_lock_enabled'] = False
    system['metrics_path'] = str(output_dir / 'metrics.json')
    system['heartbeat_path'] = str(output_dir / 'heartbeat.json')
    system['startup_flag_path'] = str(output_dir / 'started.flag')
    system['command_dir'] = str(output_dir / 'cmd')
    video = data.setdefault('video', {})
    video['csv'] = ''
    video['debug_frame_path'] = str(output_dir / 'debug.jpg')
    wheel = data.setdefault('wheel', {})
    wheel['enabled'] = False
    logic = data.setdefault('logic', {})
    logic['enable_event_disk'] = False
    logic['enable_per_id_video'] = False
    logic['event_trace_enabled'] = False
    data['event_output_dir'] = str(output_dir / 'events')
    data['event_capture_dir'] = str(output_dir / 'captures')
    return data


def main():
    args = parse_args()
    root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(args.config).resolve()
    benchmark_config = output_dir / 'benchmark_config.json'
    benchmark_config.write_text(
        json.dumps(sanitized_config(config_path, output_dir), ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    python_path = Path(args.python).resolve() if args.python else (root / 'venv-gst' / 'bin' / 'python')
    command = [str(python_path), str(root / 'run_zone_detect.py'), '--config', str(benchmark_config)]
    if args.limit > 0:
        command.extend(['--limit', str(args.limit)])
    log_path = output_dir / 'inference.log'
    samples = []
    peak_rss_kb = 0
    peak_hwm_kb = 0
    peak_threads = 0
    min_available_kb = None
    started = time.time()
    clock_ticks = max(1, int(os.sysconf(os.sysconf_names['SC_CLK_TCK'])))
    with log_path.open('w', encoding='utf-8') as log_handle:
        proc = subprocess.Popen(
            command,
            cwd=str(root),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        last_ticks = 0
        while proc.poll() is None:
            snapshot = sample_process_tree(proc.pid)
            _mem_total, mem_available = system_memory_kb()
            elapsed = max(0.0, time.time() - started)
            last_ticks = max(last_ticks, snapshot['ticks'])
            peak_rss_kb = max(peak_rss_kb, snapshot['rss_kb'])
            peak_hwm_kb = max(peak_hwm_kb, snapshot['hwm_kb'])
            peak_threads = max(peak_threads, snapshot['threads'])
            min_available_kb = mem_available if min_available_kb is None else min(min_available_kb, mem_available)
            samples.append({
                'elapsed_seconds': round(elapsed, 3),
                'rss_kb': snapshot['rss_kb'],
                'threads': snapshot['threads'],
                'mem_available_kb': mem_available,
            })
            if args.duration > 0.0 and elapsed >= float(args.duration):
                proc.terminate()
                try:
                    proc.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5.0)
                break
            time.sleep(max(0.1, float(args.sample_interval)))
        return_code = proc.wait()
    elapsed = max(1e-6, time.time() - started)
    log_text = log_path.read_text(encoding='utf-8', errors='replace')
    frame_match = re.findall(r'Video .* frames=([0-9]+) elapsed=([0-9.]+)s \(([0-9.]+) FPS\)', log_text)
    dropped_matches = re.findall(r'drop=win:[0-9]+,total:([0-9]+)', log_text)
    steady_samples = [
        item for item in samples
        if float(item.get('elapsed_seconds', 0.0)) >= max(0.0, float(args.warmup_seconds))
    ]
    if not steady_samples:
        steady_samples = samples
    steady_average_rss_mb = (
        sum(float(item.get('rss_kb', 0)) for item in steady_samples) / len(steady_samples) / 1024.0
        if steady_samples else 0.0
    )
    result = {
        'label': args.label,
        'return_code': return_code,
        'command': command,
        'elapsed_seconds': round(elapsed, 3),
        'average_process_cpu_percent': round((last_ticks / clock_ticks) / elapsed * 100.0, 2),
        'peak_rss_mb': round(peak_rss_kb / 1024.0, 2),
        'steady_average_rss_mb': round(steady_average_rss_mb, 2),
        'peak_hwm_mb': round(peak_hwm_kb / 1024.0, 2),
        'peak_threads': peak_threads,
        'min_system_available_mb': round((min_available_kb or 0) / 1024.0, 2),
        'video_summary': frame_match[-1] if frame_match else None,
        'dropped_frames_total': int(dropped_matches[-1]) if dropped_matches else 0,
        'samples': samples,
        'log_path': str(log_path),
    }
    result_path = output_dir / 'result.json'
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'samples'}, ensure_ascii=False, indent=2))
    raise SystemExit(return_code)


if __name__ == '__main__':
    main()
