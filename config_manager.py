import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse, urlunparse

DEFAULT_PER_ID_VIDEO_DIR = 'video_result/per_id'
DEFAULT_WHEEL_MODEL = 'models/wheel/2026.4.28CRwheel.rknn'


def _derive_wheel_photo_url(api_url: str) -> str:
    api_url = str(api_url or '').strip()
    if not api_url:
        return ''
    parsed = urlparse(api_url)
    path = (parsed.path or '').rstrip('/')
    if '/' in path:
        parent = path.rsplit('/', 1)[0]
    else:
        parent = ''
    new_path = f"{parent}/wheel-photo" if parent else "/wheel-photo"
    return urlunparse((parsed.scheme, parsed.netloc, new_path, '', '', ''))


class ConfigError(Exception):
    """Raised when a config file is missing or invalid."""


def _ensure_polygon(points: List[List[float]]) -> List[Tuple[float, float]]:
    if not isinstance(points, list) or len(points) < 3:
        raise ConfigError('Polygon requires at least 3 points')
    poly = []
    for item in points:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ConfigError(f'Invalid point format: {item}')
        poly.append((float(item[0]), float(item[1])))
    return poly


def _ensure_point(point: List[float]) -> Tuple[float, float]:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ConfigError(f'Invalid point: {point}')
    return float(point[0]), float(point[1])


class ConfigManager:
    def __init__(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            raise ConfigError(f'config file not found: {self.path}')
        with self.path.open('r', encoding='utf-8') as f:
            self.data: Dict[str, Any] = json.load(f)
        self._validate()

    def _validate(self):
        system = self.data.setdefault('system', {})
        system.setdefault('device_id', 'RK3588')
        system.setdefault('monitor_interval', 2.0)
        system.setdefault('api', {})
        system['api'].setdefault('url', '')
        system['api'].setdefault('token', '')
        system['api'].setdefault('capture_mode', 'path')
        system['api'].setdefault('wheel_photo_url', '')
        derived_wheel_photo_url = _derive_wheel_photo_url(system['api'].get('url', ''))
        current_wheel_photo_url = str(system['api'].get('wheel_photo_url') or '').strip()
        previous_auto = getattr(self, '_wheel_photo_url_auto', None)
        previous_auto_value = getattr(self, '_wheel_photo_url_auto_value', '')
        if previous_auto is True:
            auto_wheel_photo_url = (
                not current_wheel_photo_url
                or current_wheel_photo_url == previous_auto_value
            )
        else:
            auto_wheel_photo_url = (
                not current_wheel_photo_url
                or current_wheel_photo_url == derived_wheel_photo_url
            )
        self._wheel_photo_url_auto = bool(auto_wheel_photo_url)
        self._wheel_photo_url_auto_value = derived_wheel_photo_url if auto_wheel_photo_url else ''
        if auto_wheel_photo_url:
            system['api']['wheel_photo_url'] = derived_wheel_photo_url
        system.setdefault('wheel_photo_base_dir', '/data/ftp')
        system.setdefault('metrics_path', '/dev/shm/cleaningcar_metrics.json')
        system.setdefault('command_dir', '/dev/shm/cleaningcar_cmd')
        system.setdefault('startup_flag_path', '/dev/shm/cleaningcar_started.flag')
        system.setdefault('heartbeat_path', '/dev/shm/cleaningcar_heartbeat.json')
        system.setdefault('heartbeat_interval_seconds', 1.0)
        system.setdefault('heartbeat_timeout_seconds', 30.0)
        system.setdefault('progress_timeout_seconds', 90.0)
        system.setdefault('heartbeat_startup_grace_seconds', 90.0)
        system.setdefault('startup_capture_dir', 'captures/startup')
        system.setdefault('manual_capture_dir', 'captures/manual')
        system.setdefault('cpu_mask', '')
        system.setdefault('performance_lock_enabled', True)
        system.setdefault('performance_lock_restore_on_stop', False)
        system.setdefault('performance_lock_script', '')

        video = self.data.setdefault('video', {})
        if 'source' not in video:
            raise ConfigError('video.source missing')
        video.pop('save_video', None)
        video.pop('rga_enable', None)
        video.setdefault('source_mode', 'auto')
        video.setdefault('hw_decode', False)
        video.setdefault('workers', 2)
        video.setdefault('core_mask', '0-2')
        video.setdefault('fp_output_mode', '6')
        video.setdefault('csv', '')
        video.setdefault('debug_frame_path', 'off')
        video.setdefault('debug_frame_interval', 30)
        video.setdefault('debug_frame_max_width', 960)
        video.setdefault('debug_frame_quality', 80)
        try:
            reader_frame_timeout = float(video.get('reader_frame_timeout_seconds', 5.0))
        except (TypeError, ValueError):
            reader_frame_timeout = 5.0
        video['reader_frame_timeout_seconds'] = max(0.0, reader_frame_timeout)
        try:
            segment_minutes = int(video.get('segment_minutes', 60))
        except (TypeError, ValueError):
            segment_minutes = 0
        video['segment_minutes'] = max(0, segment_minutes)

        wheel = self.data.setdefault('wheel', {})
        enabled = wheel.get('enabled', False)
        if isinstance(enabled, str):
            wheel['enabled'] = enabled.strip().lower() in {'1', 'true', 'yes', 'on'}
        else:
            wheel['enabled'] = bool(enabled)
        event_driven = wheel.get('event_driven', True)
        if isinstance(event_driven, str):
            wheel['event_driven'] = event_driven.strip().lower() in {'1', 'true', 'yes', 'on'}
        else:
            wheel['event_driven'] = bool(event_driven)
        reader_event_driven = wheel.get('reader_event_driven', False)
        if isinstance(reader_event_driven, str):
            wheel['reader_event_driven'] = reader_event_driven.strip().lower() in {'1', 'true', 'yes', 'on'}
        else:
            wheel['reader_event_driven'] = bool(reader_event_driven)
        run_mode = str(wheel.get('run_mode', 'embedded') or 'embedded').strip().lower()
        if run_mode not in {'embedded', 'remote'}:
            run_mode = 'embedded'
        wheel['run_mode'] = run_mode
        wheel['service_url'] = str(wheel.get('service_url', '') or '').strip()
        try:
            service_timeout = float(wheel.get('service_timeout_seconds', 0.5))
        except (TypeError, ValueError):
            service_timeout = 0.5
        wheel['service_timeout_seconds'] = max(0.1, service_timeout)
        try:
            reader_idle_fps = float(wheel.get('reader_idle_fps', 0.0))
        except (TypeError, ValueError):
            reader_idle_fps = 0.0
        wheel['reader_idle_fps'] = max(0.0, reader_idle_fps)
        wheel['left_source'] = str(wheel.get('left_source', '') or '').strip()
        wheel['right_source'] = str(wheel.get('right_source', '') or '').strip()
        wheel['model'] = str(wheel.get('model', DEFAULT_WHEEL_MODEL) or DEFAULT_WHEEL_MODEL).strip()
        classes = wheel.get('classes')
        if not isinstance(classes, list) or not classes:
            wheel['classes'] = ['0-25', '25-50', '50-75', '75-100']
        else:
            wheel['classes'] = [str(item).strip() for item in classes if str(item).strip()] or ['0-25', '25-50', '50-75', '75-100']
        try:
            target_fps = float(wheel.get('target_fps', 5))
        except (TypeError, ValueError):
            target_fps = 5.0
        wheel['target_fps'] = max(0.1, target_fps)
        try:
            active_target_fps = float(wheel.get('active_target_fps', wheel['target_fps']))
        except (TypeError, ValueError):
            active_target_fps = wheel['target_fps']
        wheel['active_target_fps'] = max(0.0, active_target_fps)
        wheel['core_mask'] = str(wheel.get('core_mask', '') or '').strip()
        try:
            wheel_imgsz = int(wheel.get('imgsz', 640))
        except (TypeError, ValueError):
            wheel_imgsz = 640
        wheel['imgsz'] = max(64, wheel_imgsz)
        wheel.pop('center_min_margin_ratio', None)
        try:
            bind_window = float(wheel.get('bind_window_seconds', 30))
        except (TypeError, ValueError):
            bind_window = 30.0
        wheel['bind_window_seconds'] = max(1.0, bind_window)
        try:
            bind_pre_start = float(wheel.get('bind_pre_start_seconds', 3.0))
        except (TypeError, ValueError):
            bind_pre_start = 3.0
        wheel['bind_pre_start_seconds'] = max(0.0, bind_pre_start)
        try:
            bind_after_end = float(wheel.get('bind_after_end_seconds', 3.0))
        except (TypeError, ValueError):
            bind_after_end = 3.0
        wheel['bind_after_end_seconds'] = max(0.0, bind_after_end)
        bind_require_active = wheel.get('bind_require_active', True)
        if isinstance(bind_require_active, str):
            wheel['bind_require_active'] = bind_require_active.strip().lower() in {'1', 'true', 'yes', 'on'}
        else:
            wheel['bind_require_active'] = bool(bind_require_active)
        try:
            bind_wait = float(wheel.get('bind_wait_seconds', 2.0))
        except (TypeError, ValueError):
            bind_wait = 2.0
        wheel['bind_wait_seconds'] = max(0.0, bind_wait)
        try:
            bind_wait_poll = float(wheel.get('bind_wait_poll_seconds', 0.08))
        except (TypeError, ValueError):
            bind_wait_poll = 0.08
        wheel['bind_wait_poll_seconds'] = max(0.02, bind_wait_poll)
        try:
            photo_bucket = float(wheel.get('photo_bucket_seconds', 1.0))
        except (TypeError, ValueError):
            photo_bucket = 0.25
        wheel['photo_bucket_seconds'] = max(0.05, photo_bucket)
        try:
            photo_min_score = float(wheel.get('photo_min_score', 0.3))
        except (TypeError, ValueError):
            photo_min_score = 0.3
        wheel['photo_min_score'] = max(0.0, min(photo_min_score, 1.0))
        try:
            reader_stale_seconds = float(wheel.get('reader_stale_seconds', 5.0))
        except (TypeError, ValueError):
            reader_stale_seconds = 5.0
        wheel['reader_stale_seconds'] = max(0.0, reader_stale_seconds)
        try:
            reader_stale_check_interval = int(wheel.get('reader_stale_check_interval_frames', 15))
        except (TypeError, ValueError):
            reader_stale_check_interval = 15
        wheel['reader_stale_check_interval_frames'] = max(1, reader_stale_check_interval)
        try:
            reader_stale_hash_size = int(wheel.get('reader_stale_hash_size', 16))
        except (TypeError, ValueError):
            reader_stale_hash_size = 16
        wheel['reader_stale_hash_size'] = max(4, min(reader_stale_hash_size, 64))
        wheel.pop('photo_per_bucket', None)

        zones = self.data.setdefault('zones', {})
        zones['zone_a_detection'] = _ensure_polygon(zones.get('zone_a_detection', []))
        zones['zone_b_wash'] = _ensure_polygon(zones.get('zone_b_wash', []))
        flow_vec = zones.get('flow_vector', {})
        zones['flow_vector'] = {
            'start': _ensure_point(flow_vec.get('start', (0.0, 0.0))),
            'end': _ensure_point(flow_vec.get('end', (0.0, 1.0)))
        }

        logic = self.data.setdefault('logic', {})
        logic.pop('enable_global_video', None)
        for obsolete_key in (
            'per_id_downscale_ratio',
            'per_id_frame_stride',
            'per_id_target_width',
            'per_id_target_height',
            'per_id_auto_adapt',
            'per_id_auto_cpu_high',
            'per_id_auto_cpu_low',
            'per_id_max_frame_stride',
        ):
            logic.pop(obsolete_key, None)
        logic.setdefault('detection_anchor', 'bottom_center')
        logic.setdefault('zone_a_mask_enable', True)
        logic.setdefault('zone_b_entry_hysteresis', 3)
        logic.setdefault('zone_b_exit_hysteresis', 3)
        logic.setdefault('stationary_min_frames', 0)
        logic.setdefault('stationary_speed_thresh', 8.0)
        logic.setdefault('type34_min_interval_frames', 5)
        logic.setdefault('track_timeout_frames', 90)
        logic.setdefault('track_max_age', 120)
        logic.setdefault('vehicle_iou_threshold', 0.3)
        logic.setdefault('vehicle_center_gate_ratio', 0.0)
        logic.setdefault('vehicle_tracker_impl', 'bytetrack')
        logic['disable_plate_only_events'] = True
        logic['single_lifecycle_events'] = True
        logic.setdefault('min_zone_a_dwell_frames_for_type5', 25)
        logic.setdefault('min_track_frames_for_type1', 5)
        logic.setdefault('no_draw', False)
        logic.setdefault('draw_plate_boxes', False)
        logic.setdefault('require_vehicle_type_for_events', False)
        logic.setdefault('lane_name', '冲洗')
        logic.setdefault('vehicle_shrink_ratio', 0.35)
        logic.setdefault('vehicle_lock_min_votes', 40)
        logic.setdefault('vehicle_lock_on_confirm', True)
        logic.setdefault('plate_lock_frames', 6)
        logic.setdefault('default_plate_color', '')
        logic.setdefault('default_plate_color_conf', 0.0)
        logic.setdefault('default_cleanliness', 0)
        logic.setdefault('car_plate_cache_ttl', 60)
        logic.setdefault('allowed_event_types', [1, 2, 3, 4, 5, 6])
        logic.setdefault('anchor_offset_ratio', 0.0)
        logic.setdefault('zone_b_anchor_min_frames', 0)
        logic.setdefault('debug_overlay', False)
        logic.setdefault('debug_track_state', False)
        logic.setdefault('debug_anchor_points', False)
        logic.setdefault('debug_water_boxes', False)
        logic.setdefault('wash_duration_offset_seconds', 0.0)
        logic.setdefault('min_zone_b_dwell_frames_for_type4', 60)
        logic.setdefault('enable_per_id_video', True)
        logic.setdefault('per_id_video_dir', DEFAULT_PER_ID_VIDEO_DIR)
        logic.setdefault('per_id_video_queue_size', 8)
        per_id_video_source = str(logic.get('per_id_video_source', 'auto') or 'auto').strip().lower()
        if per_id_video_source not in {'auto', 'raw', 'source', 'original', 'origin', 'annotated', 'draw', 'debug'}:
            per_id_video_source = 'auto'
        logic['per_id_video_source'] = per_id_video_source
        try:
            per_id_raw_prebuffer = float(logic.get('per_id_raw_prebuffer_seconds', 3.0))
        except (TypeError, ValueError):
            per_id_raw_prebuffer = 3.0
        logic['per_id_raw_prebuffer_seconds'] = max(0.0, per_id_raw_prebuffer)
        logic.setdefault('copy_track_last_frame', False)
        logic.setdefault('copy_raw_frame_cache', False)
        logic.setdefault('plate_core_mask', '')
        per_id_video_dir = str(logic.get('per_id_video_dir', '') or '').strip()
        logic['per_id_video_dir'] = per_id_video_dir or DEFAULT_PER_ID_VIDEO_DIR
        logic.setdefault('enable_event_disk', False)
        shadow = logic.setdefault('shadow_plate_pool', {})
        shadow.setdefault('max_candidates', 50)
        shadow.setdefault('max_age_frames', 120)

        self.data.pop('storage', None)

        self.data.setdefault('event_capture_quality', 70)

        cfg_name = self.path.stem or 'default'
        default_events = f'events/{cfg_name}'
        default_captures = f'captures/{cfg_name}'
        self.data.setdefault('event_output_dir', default_events)
        self.data.setdefault('event_capture_dir', default_captures)

    def save(self):
        self._validate()
        with self.path.open('w', encoding='utf-8') as f:
            safe_data = self._sanitize_json_output(self.data)
            if getattr(self, '_wheel_photo_url_auto', False):
                safe_data.setdefault('system', {}).setdefault('api', {})['wheel_photo_url'] = ''
            json.dump(safe_data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _sanitize_json_output(data: Any) -> Any:
        """Replace lone surrogates so JSON serialisation never crashes with
        ``UnicodeEncodeError: surrogates not allowed``."""
        if isinstance(data, str):
            return data.encode("utf-8", errors="surrogateescape").decode("utf-8", errors="replace")
        if isinstance(data, dict):
            return {ConfigManager._sanitize_json_output(k): ConfigManager._sanitize_json_output(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return [ConfigManager._sanitize_json_output(item) for item in data]
        return data

    @property
    def zones(self):
        return self.data['zones']

    @property
    def logic(self):
        return self.data['logic']

    @property
    def video(self):
        return self.data['video']

    @property
    def system(self):
        return self.data['system']

    @property
    def storage(self):
        return self.data.get('storage', {})
