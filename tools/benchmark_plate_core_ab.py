#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_case(args, label, core_mask):
    output_dir = Path(args.output_dir).resolve() / label
    command = [
        str(Path(args.python).resolve() if args.python else sys.executable),
        str(Path(args.project_root).resolve() / 'tools' / 'benchmark_runtime_resources.py'),
        '--project-root', str(Path(args.project_root).resolve()),
        '--config', str(Path(args.config).resolve()),
        '--output-dir', str(output_dir),
        '--duration', str(args.duration),
        '--warmup-seconds', str(args.warmup_seconds),
        '--sample-interval', str(args.sample_interval),
        '--label', label,
        '--plate-core-mask', str(core_mask),
    ]
    if args.python:
        command.extend(['--python', str(Path(args.python).resolve())])
    if args.keep_wheel:
        command.append('--keep-wheel')
    completed = subprocess.run(command, check=False)
    result_path = output_dir / 'result.json'
    if not result_path.exists():
        raise RuntimeError(f'{label} did not produce {result_path}; exit={completed.returncode}')
    return json.loads(result_path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description='板端车牌NPU core绑定A/B资源基准。')
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--python', default='')
    parser.add_argument('--baseline-core', default='0')
    parser.add_argument('--candidate-core', default='2')
    parser.add_argument('--duration', type=float, default=180.0)
    parser.add_argument('--warmup-seconds', type=float, default=30.0)
    parser.add_argument('--sample-interval', type=float, default=0.5)
    parser.add_argument('--keep-wheel', action='store_true')
    args = parser.parse_args()

    root = Path(args.output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    baseline = run_case(args, f'plate-core-{args.baseline_core}', args.baseline_core)
    candidate = run_case(args, f'plate-core-{args.candidate_core}', args.candidate_core)

    def delta(key):
        return round(float(candidate.get(key, 0.0)) - float(baseline.get(key, 0.0)), 3)

    summary = {
        'baseline': baseline,
        'candidate': candidate,
        'delta_candidate_minus_baseline': {
            'average_process_cpu_percent': delta('average_process_cpu_percent'),
            'steady_average_rss_mb': delta('steady_average_rss_mb'),
            'peak_rss_mb': delta('peak_rss_mb'),
            'dropped_frames_total': (
                int(candidate.get('dropped_frames_total', 0))
                - int(baseline.get('dropped_frames_total', 0))
            ),
        },
        'acceptance': {
            'candidate_return_code_zero': int(candidate.get('return_code', 1)) == 0,
            'candidate_no_extra_drops': (
                int(candidate.get('dropped_frames_total', 0))
                <= int(baseline.get('dropped_frames_total', 0))
            ),
            'manual_checks': [
                '对比两轮inference.log中的主检测infer_ms与车牌触发帧耗时',
                'keep-wheel时确认core2占用和左右车轮照片完整率没有恶化',
                '两轮必须使用同一模型、RTSP、频率策略和时长',
            ],
        },
    }
    output_path = root / 'plate_core_ab_summary.json'
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
