#!/usr/bin/env python3
"""离线MPP到BGR契约探针：按源时基读每帧，不运行模型或上报业务事件。"""
import argparse
import json
import os
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--codec', choices=('h264', 'h265'), required=True)
    p.add_argument('--loops', type=int, default=1)
    p.add_argument('--max-frames', type=int, default=0)
    p.add_argument('--unpaced', action='store_true', help='离线快速取帧负载，不用于评估实时业务FPS')
    p.add_argument('--drop', action='store_true', help='模拟项目的appsink丢弃旧帧，只用于错误路径测试')
    p.add_argument('--samples', help='保存前10帧供两个核心策略的像素比较，仅用于离线探针')
    args = p.parse_args()
    path = Path(args.source).resolve()
    if not path.is_file() or '"' in str(path):
        raise SystemExit('只支持已存在的离线MP4文件')
    import cv2
    import numpy as np
    pipeline = (f'filesrc location="{path}" ! qtdemux ! {args.codec}parse ! '
                'mppvideodec format=BGR ! video/x-raw,format=BGR ! '
                f'appsink drop={"true" if args.drop else "false"} max-buffers=1 sync={"false" if args.unpaced else "true"}')
    counts, opens, reads, samples = [], [], [], []
    started = time.monotonic()
    for cycle in range(args.loops):
        t = time.monotonic()
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise SystemExit('MPP打开失败')
        opens.append(time.monotonic() - t)
        frames = 0
        try:
            while not args.max_frames or frames < args.max_frames:
                t = time.monotonic()
                ok, frame = cap.read()
                elapsed = time.monotonic() - t
                if not ok:
                    break
                if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                    raise SystemExit('BGR帧无效')
                reads.append(elapsed)
                frames += 1
                if args.samples and cycle == 0 and frames <= 10:
                    samples.append(frame.copy())
        finally:
            cap.release()
        if not frames:
            raise SystemExit('未取得任何BGR帧')
        counts.append(frames)
    duration = time.monotonic() - started
    if args.samples:
        np.savez_compressed(args.samples, frames=np.stack(samples))
    print(json.dumps({'pid': os.getpid(), 'loops': args.loops, 'frames_per_loop': counts,
                      'frames': sum(counts), 'elapsed_s': duration, 'wall_fps': sum(counts) / duration,
                      'open_ms_max': max(opens) * 1000, 'read_ms_p50': float(np.percentile(reads, 50)) * 1000,
                      'read_ms_p95': float(np.percentile(reads, 95)) * 1000}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
