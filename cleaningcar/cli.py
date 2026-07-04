import argparse
import os
from pathlib import Path

from .runtime_config import (
    DEFAULT_CONFIG_PATH,
    LEGACY_CONFIG_PATH,
    apply_class_thresholds_from_config,
    apply_cli_overrides,
    load_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_bool_like(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _configure_rga_environment(config):
    os.environ["CLEANINGCAR_RGA_DISABLE"] = "1"

def parse_args():
    ap = argparse.ArgumentParser(description='Multithread RKNN detector demo.')
    ap.add_argument('--model', default='models/detection/best.rknn')
    ap.add_argument('--video', help='Single video file to process.')
    ap.add_argument('--video_dir', help='Directory of videos to process sequentially.')
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.30)
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--max_det', type=int, default=300)
    ap.add_argument('--fp_output_mode', choices=['6', '9'], default='6',
                    help='FP 检测模型输出解析模式：6=只用 box+class，9=使用 box+class+score。')
    ap.add_argument('--workers', type=int, default=2, help='Number of inference workers.')
    ap.add_argument('--queue_size', type=int, default=32)
    ap.add_argument('--core_mask', default='all', help="Which NPU cores to use: e.g. '0-2', '0,2', '1', 'all', 'auto'.")
    ap.add_argument('--hw_decode', action='store_true', help='Prefer FFmpeg hardware decode, then GStreamer+mpp, then software decode.')
    ap.add_argument('--csv', help='CSV path, append per detection.')
    ap.add_argument('--output_dir', help='When batch processing, auto-save mp4/csv into this directory using video stem names.')
    ap.add_argument('--no_draw', action='store_true', help='Do not draw boxes on frames.')
    ap.add_argument('--draw_plate_boxes', action='store_true', help='Draw license plate boxes/text when drawing is enabled.')
    ap.add_argument('--monitor_interval', type=float, default=0.0, help='Seconds between resource logs (0 disables).')
    ap.add_argument('--limit', type=int, default=0, help='Optional frame limit for quick tests.')
    ap.add_argument('--plate_detect_model', default='models/plate/plate_detect.rknn',
                    help='Path to plate detection RKNN used by the dual-model LPR pipeline.')
    ap.add_argument('--plate_rec_model', default='models/plate/plate_rec_color.rknn',
                    help='Path to plate text/color RKNN used by the dual-model LPR pipeline.')
    ap.add_argument('--plate_lock_frames', type=int, default=5, help='Frames required before plate text is locked.')
    ap.add_argument('--plate_infer_stride', type=int, default=1,
                    help='Run dual-plate inference every N frames (>=1) to reduce CPU load.')
    ap.add_argument('--plate_core_mask', default='auto',
                    help="NPU cores for plate detection/recognition RKNNs; empty/auto uses RKNN default.")
    ap.add_argument('--config', default=str(DEFAULT_CONFIG_PATH), help='JSON config describing ROI/event logic.')
    ap.add_argument('--camera', help='当配置包含多个 camera 条目时，指定要运行的 key。')
    ap.add_argument('--debug_rois', action='store_true', help='Visualize stage lines on output frames.')
    ap.add_argument('--debug_tracks', action='store_true', help='Overlay per-track state info on frames.')
    ap.add_argument('--source_mode', choices=['auto', 'camera', 'file'], default='auto',
                    help='数据源类型：camera 为实时流（可自动重连），file 为本地视频（读到末尾即停止）。')
    ap.add_argument('--event_log', nargs='?', const='auto',
                    help='Optional path for事件日志; 不加参数时使用默认 events/event_log.csv。')
    ap.add_argument('--roi_setup', action='store_true', help='Launch ROI editor before running detection.')
    ap.add_argument('--api_url', help='远端车辆冲洗事件上报接口 URL (POST)。')
    ap.add_argument('--api_token', help='用于 HTTP Authorization: Bearer 的 Token。')
    ap.add_argument('--capture_mode', choices=['path', 'base64'], default='path',
                    help='事件截图在上报时的字段格式：文件路径或Base64。')
    ap.add_argument('--lane', help='覆盖事件上报中的 lane 字段。')
    ap.add_argument('--detect_roi_only', action='store_true', help='仅在配置的 detect_roi 多边形内进行检测。')
    defaults = ap.parse_args(args=[])
    args = ap.parse_args()
    setattr(args, '_defaults', defaults)
    return args

def iter_videos(args):
    if args.video:
        yield args.video
    if args.video_dir:
        for name in sorted(os.listdir(args.video_dir)):
            if name.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
                yield os.path.join(args.video_dir, name)


def main():
    args = parse_args()

    config_path = getattr(args, 'config', None)
    if config_path:
        resolved_config_path = Path(config_path).expanduser()
        if not resolved_config_path.is_absolute():
            resolved_config_path = (Path.cwd() / resolved_config_path).resolve()
    else:
        resolved_config_path = DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else LEGACY_CONFIG_PATH
    try:
        config = load_config(str(resolved_config_path))
    except ValueError as exc:
        print(exc)
        return
    _configure_rga_environment(config)
    from .pipeline import process_video
    apply_cli_overrides(args, config)
    apply_class_thresholds_from_config(config)
    setattr(args, 'config', str(resolved_config_path))
    setattr(args, '_config', config)
    setattr(args, '_config_path', resolved_config_path)
    setattr(args, '_config_dir', resolved_config_path.parent)
    videos = list(iter_videos(args))
    if not videos:
        print('No videos specified.')
        return
    for path in videos:
        process_video(path, args)
