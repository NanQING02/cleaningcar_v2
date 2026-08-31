from pathlib import Path

from config_manager import ConfigError, ConfigManager

from .constants import CLASS_ALIAS_TO_ID, CLASS_NAMES, CLASS_THRESH

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / 'configs' / 'config.json'
LEGACY_CONFIG_PATH = PROJECT_ROOT / 'config.json'


def _default_config_path():
    if DEFAULT_CONFIG_PATH.exists():
        return DEFAULT_CONFIG_PATH
    return LEGACY_CONFIG_PATH

def load_config(path):
    if path:
        cfg_path = Path(path).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = (Path.cwd() / cfg_path).resolve()
    else:
        cfg_path = _default_config_path()
    try:
        mgr = ConfigManager(cfg_path)
    except ConfigError as exc:
        raise ValueError(str(exc)) from exc
    system = mgr.system
    video = mgr.video
    logic = mgr.logic
    wheel = mgr.data.get('wheel', {})
    zones = mgr.zones
    shadow_cfg = logic.get('shadow_plate_pool', {})
    event_capture_dir = mgr.data.get('event_capture_dir', './captures')
    event_output_dir = mgr.data.get('event_output_dir', './events')
    merged = {
        'config_path': str(cfg_path),
        'config_name': cfg_path.name,
        'camera_id': system.get('device_id', 'RK3588'),
        'event_capture_dir': str(Path(event_capture_dir)),
        'event_output_dir': str(Path(event_output_dir)),
        'api_url': system.get('api', {}).get('url', ''),
        'api_token': system.get('api', {}).get('token', ''),
        'wheel_photo_url': system.get('api', {}).get('wheel_photo_url', ''),
        'wheel_photo_base_dir': system.get('wheel_photo_base_dir', '/data/ftp'),
        'capture_mode': system.get('api', {}).get('capture_mode', 'path'),
        'monitor_interval': float(system.get('monitor_interval', 2.0)),
        'system': system,
        'video': video,
        'wheel': wheel,
        'logic': logic,
        'zones': zones,
        'shadow_pool': shadow_cfg,
        'allowed_event_types': logic.get('allowed_event_types', [1, 2, 3, 4, 5]),
        'track_timeout_frames': int(logic.get('track_timeout_frames', 90)),
        'track_max_age': int(logic.get('track_max_age', 60)),
        'lane_name': logic.get('lane_name', '冲洗'),
        'stationary_speed_thresh': float(logic.get('stationary_speed_thresh', 8.0)),
        'vehicle_shrink_ratio': float(logic.get('vehicle_shrink_ratio', 0.35)),
        'vehicle_lock_min_votes': int(logic.get('vehicle_lock_min_votes', 80)),
        'vehicle_lock_on_confirm': bool(logic.get('vehicle_lock_on_confirm', True)),
        'default_plate_color': logic.get('default_plate_color', ''),
        'default_plate_color_conf': float(logic.get('default_plate_color_conf', 0.0)),
        'default_cleanliness': int(logic.get('default_cleanliness', 0)),
        'stationary_min_frames': int(logic.get('stationary_min_frames', 0)),
        'type34_min_interval_frames': int(logic.get('type34_min_interval_frames', 5)),
        'car_plate_cache_ttl': int(logic.get('car_plate_cache_ttl', 60)),
    }
    return merged


def apply_cli_overrides(args, config):
    defaults = getattr(args, '_defaults', None)

    def maybe_set(attr, value):
        if value is None:
            return
        if defaults is None:
            setattr(args, attr, value)
            return
        if getattr(args, attr) == getattr(defaults, attr):
            setattr(args, attr, value)

    video_cfg = (config or {}).get('video', {})
    maybe_set('video', video_cfg.get('source'))
    maybe_set('source_mode', video_cfg.get('source_mode'))
    maybe_set('hw_decode', video_cfg.get('hw_decode'))
    maybe_set('workers', video_cfg.get('workers'))
    maybe_set('core_mask', video_cfg.get('core_mask'))
    maybe_set('fp_output_mode', video_cfg.get('fp_output_mode'))
    maybe_set('csv', video_cfg.get('csv'))
    cfg = config or {}
    sys_cfg = cfg.get('system', {})
    monitor_interval = cfg.get('monitor_interval', sys_cfg.get('monitor_interval'))
    maybe_set('monitor_interval', monitor_interval)
    maybe_set('api_url', cfg.get('api_url'))
    maybe_set('api_token', cfg.get('api_token'))
    maybe_set('capture_mode', cfg.get('capture_mode'))
    logic = (config or {}).get('logic', {})
    maybe_set('plate_track_lock_frames', logic.get('plate_track_lock_frames'))
    maybe_set('plate_infer_stride', logic.get('plate_infer_stride'))
    maybe_set('plate_core_mask', logic.get('plate_core_mask'))
    maybe_set('no_draw', logic.get('no_draw'))
    maybe_set('draw_plate_boxes', logic.get('draw_plate_boxes'))


def apply_class_thresholds_from_config(config):
    custom = (config or {}).get('class_thresholds')
    if not custom:
        custom = (config or {}).get('logic', {}).get('class_thresholds')
    if not custom:
        return
    for key, value in custom.items():
        try:
            thresh = float(value)
        except (TypeError, ValueError):
            continue
        idx = None
        if isinstance(key, int):
            idx = key
        else:
            key_str = str(key).strip().lower()
            if key_str in CLASS_ALIAS_TO_ID:
                idx = CLASS_ALIAS_TO_ID[key_str]
            else:
                try:
                    idx = int(key_str)
                except ValueError:
                    idx = None
        if idx is None:
            continue
        if idx < 0 or idx >= len(CLASS_NAMES):
            continue
        CLASS_THRESH[idx] = thresh
