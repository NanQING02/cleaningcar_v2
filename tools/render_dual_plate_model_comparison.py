import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cleaningcar.plate_lpr import DualPlateRecognizer


def parse_args():
    parser = argparse.ArgumentParser(description='渲染FP与INT8双模型车牌输出对照图。')
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--fp-detect-model', required=True)
    parser.add_argument('--int8-detect-model', default='models/plate/plate_detect.rknn')
    parser.add_argument('--rec-model', default='models/plate/plate_rec_color.rknn')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--conf', type=float, default=0.3)
    parser.add_argument('--iou', type=float, default=0.5)
    return parser.parse_args()


def image_paths(images_dir):
    suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    return sorted(path for path in Path(images_dir).rglob('*') if path.suffix.lower() in suffixes)


def render(image, results, title):
    canvas = image.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(canvas, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    for index, result in enumerate(results, start=1):
        x1, y1, x2, y2 = [int(round(value)) for value in result['box']]
        color = (0, 190, 0) if result.get('text') else (0, 150, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f'#{index} {result.get("score", 0.0):.2f} {result.get("plate_color", "")}'
        cv2.putText(canvas, label, (x1, max(48, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return canvas


def main():
    args = parse_args()
    paths = image_paths(args.images_dir)
    if not paths:
        raise SystemExit(f'未找到可识别图片：{args.images_dir}')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fp_recognizer = DualPlateRecognizer(args.fp_detect_model, args.rec_model)
    int8_recognizer = DualPlateRecognizer(args.int8_detect_model, args.rec_model)
    summary = []
    try:
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                continue
            fp_results = fp_recognizer.infer_frame(image, args.conf, args.iou)
            int8_results = int8_recognizer.infer_frame(image, args.conf, args.iou)
            fp_view = render(image, fp_results, f'FP16 detect | {len(fp_results)} detections')
            int8_view = render(image, int8_results, f'INT8 detect | {len(int8_results)} detections')
            comparison = cv2.hconcat((fp_view, int8_view))
            output_path = output_dir / f'{path.stem}_compare.jpg'
            cv2.imwrite(str(output_path), comparison)
            summary.append({
                'image': path.name,
                'fp': fp_results,
                'int8': int8_results,
                'output': output_path.name,
            })
    finally:
        fp_recognizer.release()
        int8_recognizer.release()
    (output_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'images': len(summary), 'outputDir': str(output_dir)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
