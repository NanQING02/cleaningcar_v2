import argparse
import json
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cleaningcar.plate_lpr import DualPlateRecognizer


def parse_args():
    parser = argparse.ArgumentParser(description='核验双模型车牌检测、识别与颜色输出。')
    parser.add_argument('--images-dir', required=True, help='待识别图片目录。')
    parser.add_argument('--detect-model', default='models/plate/plate_detect.rknn')
    parser.add_argument('--rec-model', default='models/plate/plate_rec_color.rknn')
    parser.add_argument('--conf', type=float, default=0.3)
    parser.add_argument('--iou', type=float, default=0.5)
    parser.add_argument('--limit', type=int, default=0)
    return parser.parse_args()


def image_paths(images_dir, limit):
    suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    paths = sorted(path for path in Path(images_dir).rglob('*') if path.suffix.lower() in suffixes)
    return paths[:limit] if limit > 0 else paths


def main():
    args = parse_args()
    paths = image_paths(args.images_dir, args.limit)
    if not paths:
        raise SystemExit(f'未找到可识别图片：{args.images_dir}')

    recognizer = DualPlateRecognizer(args.detect_model, args.rec_model)
    totals = {'images': 0, 'detections': 0, 'with_text': 0, 'elapsedMs': 0.0}
    try:
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None:
                print(json.dumps({'image': str(path), 'error': 'imread_failed'}, ensure_ascii=False))
                continue
            started = time.perf_counter()
            results = recognizer.infer_frame(frame, conf_thresh=args.conf, iou_thresh=args.iou)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            totals['images'] += 1
            totals['detections'] += len(results)
            totals['with_text'] += sum(1 for result in results if result.get('text'))
            totals['elapsedMs'] += elapsed_ms
            print(json.dumps({
                'image': path.name,
                'elapsedMs': round(elapsed_ms, 2),
                'results': results,
            }, ensure_ascii=False))
    finally:
        recognizer.release()

    totals['avgMs'] = round(totals['elapsedMs'] / max(totals['images'], 1), 2)
    print(json.dumps({'summary': totals}, ensure_ascii=False))


if __name__ == '__main__':
    main()
