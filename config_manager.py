import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse, urlunparse

DEFAULT_PER_ID_VIDEO_DIR = 'video_result/per_id'
DEFAULT_WHEEL_MODEL = 'models/wheel/2026.4.28CRwheel.rknn'
DEFAULT_EVENT_CAPTURE_BASE_DIR = '/data/ftp/event_captures'
NORMALIZED_COORD_TOLERANCE = 1e-3


def _clamp_near_normalized_points(points):
    parsed = [(float(item[0]), float(item[1])) for item in points]
    tolerance = NORMALIZED_COORD_TOLERANCE
    if all(
        -tolerance <= x <= 1.0 + tolerance
        and -tolerance <= y <= 1.0 + tolerance
        for x, y in parsed
    ):
        return [(max(0.0, min(1.0, x)), max(0.0, min(1.0, y))) for x, y in parsed]
    return parsed


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
    for item in points:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ConfigError(f'Invalid point format: {item}')
    return _clamp_near_normalized_points(points)


def _ensure_point(point: List[float]) -> Tuple[float, float]:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ConfigError(f'Invalid point: {point}')
    return _clamp_near_normalized_points([point])[0]


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
        capture_mode = str(system['api'].get('capture_mode', 'path') or 'path').strip().lower()
        if capture_mode not in {'path', 'base64'}:
            capture_mode = 'path'
        system['api']['capture_mode'] = capture_mode
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
        system.setdefault('auto_restart_max_attempts', 3)
        system.setdefault('auto_restart_window_seconds', 600.0)
        system.setdefault('auto_restart_backoff_seconds', 5.0)
        system.setdefault('auto_restart_backoff_max_seconds', 60.0)
        system.setdefault('startup_capture_dir', 'captures/startup')
        system.setdefault('manual_capture_dir', 'captures/manual')
        system.setdefault('cpu_mask', '')
        system.setdefault('performance_lock_enabled', True)
        system.setdefault('performance_lock_restore_on_stop', False)
        system.setdefault('performance_lock_script', '')

        self.data.setdefault('reader_fail_threshold', 5)
        self.data.setdefault('reader_reconnect_delay', 2.0)
        self.data.setdefault('reader_max_reconnect', 20)
        self.data.setdefault('reader_fps_log_interval', 10.0)

        video = self.data.setdefault('video', {})
        if 'source' not in video:
            raise ConfigError('video.source missing')
        video.pop('save_video', None)
        video.pop('rga_enable', None)
        video.setdefault('source_mode', 'auto')
        video.setdefault('hw_decode', True)
        decode_backend = str(video.get('decode_backend', 'auto') or 'auto').strip().lower()
        source_mode = str(video.get('source_mode', 'auto') or 'auto').strip().lower()
        source_value = str(video.get('source', '') or '').strip().lower()
        is_rtsp_source = source_value.startswith(('rtsp://', 'rtsps://'))
        source_path = Path(str(video.get('source', '') or '')).expanduser()
        if not source_path.is_absolute():
            source_path = self.path.parent / source_path
        is_local_file_source = bool(source_value and source_path.is_file())
        is_camera_source = bool(is_rtsp_source or (source_mode == 'camera' and not is_local_file_source))
        if is_camera_source:
            decode_backend = 'gstreamer'
            video['gstreamer_bgr_mode'] = 'direct'
        elif decode_backend not in {'auto', 'gstreamer', 'ffmpeg'}:
            decode_backend = 'auto'
        video['decode_backend'] = decode_backend
        # RGA 管控（2026-08-26）：ffmpeg_rga/scale_rkrga 解码后端已按死机排查结论移除，
        # 残留在配置里的 ffmpeg_rga 段直接丢弃；真实RTSP/camera强制归一到
        # GStreamer direct-BGR，实际存在的离线文件保留明确指定的安全硬解后端。
        video.pop('ffmpeg_rga', None)
        video.setdefault('workers', 2)
        video.setdefault('core_mask', '0-2')
        video['fp_output_mode'] = '6'
        if is_camera_source:
            video['hw_decode'] = True
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
        pause_bypass_enabled = wheel.get('pause_bypass_during_wash_enabled', False)
        if isinstance(pause_bypass_enabled, str):
            wheel['pause_bypass_during_wash_enabled'] = pause_bypass_enabled.strip().lower() in {'1', 'true', 'yes', 'on'}
        else:
            wheel['pause_bypass_during_wash_enabled'] = bool(pause_bypass_enabled)
        wheel['pause_bypass_config_key'] = str(
            wheel.get('pause_bypass_config_key', 'config_绕行.json') or 'config_绕行.json'
        ).strip()
        try:
            pause_resume_delay = float(wheel.get('pause_bypass_resume_delay_seconds', 0.5))
        except (TypeError, ValueError):
            pause_resume_delay = 0.5
        wheel['pause_bypass_resume_delay_seconds'] = max(0.0, pause_resume_delay)
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
        ignore_broken_rtp_info = wheel.get('ignore_broken_rtp_info', True)
        if isinstance(ignore_broken_rtp_info, str):
            wheel['ignore_broken_rtp_info'] = ignore_broken_rtp_info.strip().lower() in {'1', 'true', 'yes', 'on'}
        else:
            wheel['ignore_broken_rtp_info'] = bool(ignore_broken_rtp_info)
        wheel.pop('run_mode', None)
        wheel.pop('service_url', None)
        wheel.pop('service_timeout_seconds', None)
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
            photo_bucket = float(wheel.get('photo_bucket_seconds', 0.5))
        except (TypeError, ValueError):
            photo_bucket = 0.5
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
        reference_edge = zones.get('direction_reference_edge', 'auto')
        if isinstance(reference_edge, str) and reference_edge.strip().lower() == 'auto':
            zones['direction_reference_edge'] = 'auto'
        else:
            try:
                reference_edge = int(reference_edge)
            except (TypeError, ValueError):
                reference_edge = -1
            edge_count = len(zones['zone_a_detection'])
            zones['direction_reference_edge'] = (
                reference_edge if 0 <= reference_edge < edge_count else 'auto'
            )

        logic = self.data.setdefault('logic', {})
        logic.pop('enable_global_video', None)
        logic.pop('zone_a_mask_enable', None)
        for removed_key in (
            'no_draw',
            'draw_plate_boxes',
            'debug_overlay',
            'debug_track_state',
            'debug_anchor_points',
            'debug_water_boxes',
            'plate_draw_stable_only',
            'per_id_type6_require_plate_candidate',
            'track_timeout_frames',
            'track_max_age',
            'type34_min_interval_frames',
            'water_window_min_hits',
            'allowed_event_types',
            'stationary_min_frames',
            'stationary_speed_thresh',
            'min_water_hit_frames_for_wash',
            'water_window_size',
            'min_zone_a_dwell_frames_for_type5',
            'event_track_quality',
            'min_zone_b_dwell_frames_for_type4',
            'require_vehicle_type_for_events',
            'vehicle_shrink_ratio',
            'vehicle_lock_min_votes',
            'vehicle_lock_on_confirm',
        ):
            logic.pop(removed_key, None)
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
        logic.setdefault('zone_b_entry_hysteresis', 3)
        logic.setdefault('zone_b_exit_hysteresis', 5)
        try:
            water_confirm_frames = int(logic.get('water_confirm_frames', 3) or 3)
        except (TypeError, ValueError):
            water_confirm_frames = 3
        logic['water_confirm_frames'] = max(1, min(water_confirm_frames, 30))
        try:
            track_lost_grace_seconds = float(logic.get('track_lost_grace_seconds', 4.0) or 0.0)
        except (TypeError, ValueError):
            track_lost_grace_seconds = 4.0
        logic['track_lost_grace_seconds'] = max(0.0, min(track_lost_grace_seconds, 60.0))
        logic['event_trace_enabled'] = bool(logic.get('event_trace_enabled', False))
        logic['event_trace_dir'] = str(logic.get('event_trace_dir', 'event_traces') or 'event_traces').strip()
        try:
            event_trace_queue_size = int(logic.get('event_trace_queue_size', 4096) or 4096)
        except (TypeError, ValueError):
            event_trace_queue_size = 4096
        logic['event_trace_queue_size'] = max(128, min(event_trace_queue_size, 100000))
        for key, default, minimum, maximum in (
            ('zone_a_margin_ratio', 0.10, 0.0, 1.0),
            ('zone_a_margin_min_px', 4.0, 0.0, 256.0),
            ('zone_a_margin_max_px', 24.0, 0.0, 512.0),
        ):
            try:
                value = float(logic.get(key, default))
            except (TypeError, ValueError):
                value = default
            logic[key] = max(minimum, min(maximum, value))
        logic['zone_a_margin_max_px'] = max(
            logic['zone_a_margin_min_px'],
            logic['zone_a_margin_max_px'],
        )
        for key, default in (
            ('zone_a_observed_outside_hits', 3),
            ('zone_a_enter_core_hits', 3),
            ('zone_a_exit_outside_hits', 5),
        ):
            try:
                value = int(logic.get(key, default))
            except (TypeError, ValueError):
                value = default
            logic[key] = max(1, min(100, value))
        logic.setdefault('vehicle_iou_threshold', 0.3)
        logic.setdefault('vehicle_center_gate_ratio', 0.0)
        logic.setdefault('vehicle_tracker_impl', 'bytetrack')
        logic.setdefault('plate_requires_vehicle', None)
        logic.setdefault('pending_plate_cache_ttl_frames', 40)
        logic.setdefault('pending_plate_cache_max_entries', 30)
        logic['disable_plate_only_events'] = True
        logic['single_lifecycle_events'] = True
        logic.setdefault('min_track_frames_for_type1', 5)
        logic.setdefault('lane_name', '冲洗')
        legacy_plate_lock_frames = logic.pop('plate_lock_frames', None)
        if legacy_plate_lock_frames is not None:
            logic.setdefault('plate_track_lock_frames', legacy_plate_lock_frames)
        logic.setdefault('plate_track_lock_frames', 6)
        logic.setdefault('event_plate_lock_frames', 6)
        logic.setdefault('plate_text_min_detection_confidence', 0.65)
        logic.setdefault('plate_text_min_recognition_confidence', 0.75)
        logic.setdefault('plate_text_max_streak_gap_frames', 2)
        logic.setdefault('plate_correction_confirm_hits', 12)
        logic.setdefault('plate_color_correction_hits', 5)
        logic.setdefault('allowed_plate_colors', ['蓝色', '黄色', '绿色'])
        logic.setdefault('plate_output_shape_log_once', True)
        logic.setdefault('default_plate_color', '')
        logic.setdefault('default_plate_color_conf', 0.0)
        logic.setdefault('default_cleanliness', 0)
        logic.setdefault('car_plate_cache_ttl', 60)
        logic.setdefault('anchor_offset_ratio', 0.0)
        logic['anchor_mode'] = 'neutral'
        logic['anchor_shadow_compare'] = bool(logic.get('anchor_shadow_compare', False))
        for key, default, minimum, maximum in (
            ('anchor_legacy_flow_shift_ratio', 0.3, 0.0, 1.0),
            ('anchor_adaptive_vertical_ratio', 0.08, 0.0, 0.5),
            ('anchor_adaptive_vertical_cap_ratio', 0.02, 0.0, 0.2),
            ('anchor_edge_margin_ratio', 0.01, 0.0, 0.1),
            ('anchor_direction_consistency', 0.7, 0.5, 1.0),
            ('anchor_direction_min_displacement_ratio', 0.03, 0.0, 0.5),
        ):
            try:
                value = float(logic.get(key, default))
            except (TypeError, ValueError):
                value = default
            logic[key] = max(minimum, min(maximum, value))
        for key, default, minimum, maximum in (
            ('anchor_history_size', 20, 2, 200),
            ('anchor_reuse_max_frames', 12, 0, 300),
            ('anchor_direction_window', 8, 3, 60),
            ('anchor_direction_min_points', 5, 3, 60),
            ('anchor_direction_blend_frames', 5, 1, 30),
        ):
            try:
                value = int(logic.get(key, default))
            except (TypeError, ValueError):
                value = default
            logic[key] = max(minimum, min(maximum, value))
        logic['anchor_direction_min_points'] = min(
            logic['anchor_direction_min_points'],
            logic['anchor_direction_window'],
        )
        logic.setdefault('zone_b_anchor_min_frames', 0)
        logic.setdefault('wash_duration_offset_seconds', 0.0)
        try:
            type4_dwell_seconds = float(logic.get('min_zone_b_dwell_seconds_for_type4', 0.5) or 0.0)
        except (TypeError, ValueError):
            type4_dwell_seconds = 0.5
        logic['min_zone_b_dwell_seconds_for_type4'] = max(0.0, min(type4_dwell_seconds, 60.0))
        for key, default in (
            ('pre_type2_video_segment_seconds', 600.0),
            ('post_type2_force_finalize_seconds', 900.0),
        ):
            try:
                value = float(logic.get(key, default) or 0.0)
            except (TypeError, ValueError):
                value = default
            logic[key] = max(0.0, min(value, 86400.0))
        logic.setdefault('enable_per_id_video', True)
        logic.setdefault('per_id_video_dir', DEFAULT_PER_ID_VIDEO_DIR)
        logic.setdefault('per_id_video_queue_size', 8)
        per_id_video_source = str(logic.get('per_id_video_source', 'auto') or 'auto').strip().lower()
        if per_id_video_source not in {'auto', 'raw', 'source', 'original', 'origin', 'annotated', 'draw', 'debug'}:
            per_id_video_source = 'auto'
        logic['per_id_video_source'] = per_id_video_source
        logic.setdefault('copy_track_last_frame', False)
        logic.setdefault('copy_raw_frame_cache', False)
        logic.setdefault('plate_core_mask', '')
        per_id_video_dir = str(logic.get('per_id_video_dir', '') or '').strip()
        logic['per_id_video_dir'] = per_id_video_dir or DEFAULT_PER_ID_VIDEO_DIR
        logic.setdefault('enable_event_disk', False)
        shadow = logic.get('shadow_plate_pool')
        if not isinstance(shadow, dict):
            shadow = {}
            logic['shadow_plate_pool'] = shadow
        shadow.setdefault('max_candidates', 50)
        shadow.setdefault('max_age_frames', 120)
        shadow.setdefault('text_window_frames', 50)
        shadow.setdefault('text_margin_ratio', 0.12)
        shadow.setdefault('text_switch_min_consecutive', 6)
        shadow.setdefault('text_switch_gain_ratio', 1.2)
        shadow.setdefault('text_switch_margin_ratio', 0.18)
        shadow.setdefault('color_min_confidence', 0.70)
        shadow.setdefault('color_lock_frames', 3)
        shadow.setdefault('color_window_frames', 50)
        shadow.setdefault('color_switch_min_consecutive', 3)
        shadow.setdefault('color_switch_gain_ratio', 1.2)
        shadow.setdefault('color_switch_margin', 0.5)
        self.data.pop('storage', None)

        self.data.setdefault('event_capture_quality', 70)

        cfg_name = self.path.stem or 'default'
        default_events = f'events/{cfg_name}'
        default_captures = f'{DEFAULT_EVENT_CAPTURE_BASE_DIR}/{cfg_name}'
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
