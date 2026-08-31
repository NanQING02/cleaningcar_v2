import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='汇总双模型车牌 trace 中的文本和颜色候选。')
    parser.add_argument('--trace-file', required=True)
    parser.add_argument('--track-id', type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    text_counts = Counter()
    color_counts = Counter()
    pair_counts = Counter()
    frames = []
    for line in Path(args.trace_file).read_text(encoding='utf-8').splitlines():
        record = json.loads(line)
        if record.get('kind') != 'track_input':
            continue
        data = record.get('data') or {}
        if args.track_id is not None and int(data.get('trackId', -1)) != args.track_id:
            continue
        if not data.get('isPlateUpdate'):
            continue
        text = str(data.get('plateText') or '')
        color = str(data.get('plateColor') or '')
        if not text and not color:
            continue
        text_counts[text] += 1
        color_counts[color] += 1
        pair_counts[(text, color)] += 1
        frames.append({
            'frameIdx': data.get('frameIdx'),
            'text': text,
            'color': color,
            'colorConfidence': data.get('plateColorConfidence'),
            'plateConfidence': data.get('plateConfidence'),
            'box': data.get('plateBox'),
        })
    print(json.dumps({
        'textCounts': text_counts.most_common(),
        'colorCounts': color_counts.most_common(),
        'textColorCounts': [(text, color, count) for (text, color), count in pair_counts.most_common()],
        'frames': frames,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
