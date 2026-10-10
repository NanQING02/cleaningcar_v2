#!/usr/bin/env python3
"""只用于非生产设备的离线四路A/B；不会启动业务链路。"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from cleaningcar.rga3_guard import prepare_environment  # noqa: E402


def available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    return 0


def kernel_counts(since):
    text = subprocess.check_output(['journalctl', '-k', '-b', '--since', '@' + str(since), '--no-pager'], text=True)
    return {k: text.count(k) for k in ('job buffer map failed', 'RGA_MMU unsupported memory larger than 4G', 'scheduler core[4]', 'hardware timeout', 'Out of memory')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--main-a', required=True)
    p.add_argument('--main-b', required=True)
    p.add_argument('--left', required=True)
    p.add_argument('--right', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--memory-mib', type=int, default=0)
    p.add_argument('--loops', type=int, default=1)
    p.add_argument('--unpaced', action='store_true')
    p.add_argument('--drop', action='store_true')
    args = p.parse_args()
    if not 0 <= args.memory_mib <= 2048:
        raise SystemExit('仅允许0到2048MiB受控占用')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    sources = [('main_a', args.main_a, 'h264'), ('main_b', args.main_b, 'h264'),
               ('left', args.left, 'h265'), ('right', args.right, 'h265')]
    for mode in ('default', 'rga3'):
        mem = None
        processes, logs = [], []
        if available() < (args.memory_mib + 1536) * 1024 * 1024:
            raise SystemExit('可用内存不足，停止压力测试')
        try:
            if args.memory_mib:
                mem = bytearray(args.memory_mib * 1024 * 1024)
                mem[::4096] = b'x' * len(mem[::4096])
            env = dict(os.environ) if mode == 'default' else prepare_environment(ROOT)
            since = time.time()
            started = time.monotonic()
            minimum = available()
            for name, path, codec in sources:
                log = (output / f'{mode}_{name}.log').open('w')
                logs.append(log)
                cmd = [sys.executable, str(ROOT / 'tools/rga3_guard_probe.py'), '--source', path,
                       '--codec', codec, '--loops', str(args.loops)]
                if args.unpaced:
                    cmd.append('--unpaced')
                if args.drop:
                    cmd.append('--drop')
                processes.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            while any(proc.poll() is None for proc in processes):
                minimum = min(minimum, available())
                if minimum < 768 * 1024 * 1024 or time.monotonic() - started > 120:
                    raise RuntimeError('内存/运行超时门禁触发')
                time.sleep(0.5)
            results[mode] = {'elapsed_s': time.monotonic() - started, 'memory_available_min_bytes': minimum,
                             'pids': [proc.pid for proc in processes], 'codes': [proc.returncode for proc in processes],
                             'kernel': kernel_counts(since)}
            print(json.dumps({mode: results[mode]}, ensure_ascii=False), flush=True)
        finally:
            for proc in processes:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=5)
            for log in logs:
                log.close()
            mem = None
        (output / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
