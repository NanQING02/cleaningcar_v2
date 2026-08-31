import atexit
import csv
from math import hypot
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue

import cv2
import numpy as np

from zone_manager import ZoneManager

from .anchor import AnchorEstimator
from .constants import (
    CAR_PLATE_CACHE_TTL,
    CLASS_COLORS,
    CLASS_NAMES,
    LICENSE_CLASS,
    PLATE_CAR_LINK_IOU,
    REPORT_MIN_FRAMES,
    VEHICLE_CLASS_IDS,
    WATER_CLASS_IDS,
    localize_cleaning,
    localize_vehicle,
    select_box_color,
)
from .events import EventManager, EventUploader, WheelPhotoUploader
from .log_throttle import WindowedLogThrottle
from .monitoring import monitor_loop
from .npu_monitor import format_npu_status, npu_status_flags, snapshot_npu_status
from .plate import PlateTextTracker, is_valid_plate, normalize_plate_candidate_text
from .resize_accel import resize_backend_name, resize_bgr
from .runtime_config import load_config
from .runtime_signals import (
    load_json_file,
    resolve_runtime_settings,
    save_snapshot_images,
    write_json_atomic,
)
from .storage_cleanup import RetentionPolicy, RuntimeStorageCleaner
from .text_render import draw_text
from .tracking import VehicleTracker, resolve_track_retention_frames
from .video_io import (
    AsyncPerIdVideoWriter,
    _resolve_runtime_path,
    core_mask_to_indices,
    create_h264_video_writer,
    create_video_reader,
    detect_source_mode,
    emit_per_id_video_type6,
    finalize_per_id_recording,
    parse_core_mask,
    resolve_auto_plate_core_mask,
    resolve_worker_core_masks,
)
from .vision import box_iou, point_in_box, scale_point, scale_polygon
from .wheel import WheelDetectionService
from .worker import DetectWorker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PER_ID_VIDEO_DIR = (PROJECT_ROOT / 'video_result' / 'per_id').resolve()


def _process_rss_kb():
    try:
        with open('/proc/self/status', 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1])
    except Exception:
        return None
    return None


def _normalize_track_id(value):
    try:
        track_id = int(value)
    except (TypeError, ValueError):
        return None
    return track_id if track_id > 0 else None


def _should_drop_stale_frames(source_mode):
    return str(source_mode or '').strip().lower() != 'file'


def _resolve_per_id_recording_params(width, height, source_fps, logic_cfg=None):
    del logic_cfg
    try:
        resolved_fps = float(source_fps or 0.0)
    except (TypeError, ValueError):
        resolved_fps = 0.0
    if resolved_fps <= 0.0:
        resolved_fps = 20.0
    return {
        'width': max(1, int(width)),
        'height': max(1, int(height)),
        'fps': resolved_fps,
        'frame_stride': 1,
    }


def _resolve_per_id_video_source(logic_cfg=None, no_draw=False, draw_enabled=False):
    logic_cfg = logic_cfg or {}
    raw_value = str(logic_cfg.get('per_id_video_source', 'auto') or 'auto').strip().lower()
    if raw_value in {'raw', 'source', 'original', 'origin'}:
        return 'raw'
    if raw_value in {'annotated', 'draw', 'debug'}:
        return 'annotated'
    return 'annotated' if (not no_draw and draw_enabled) else 'raw'


def _ensure_plate_binding_state(frame_idx, state=None):
    state = state if isinstance(state, dict) else {}
    state.setdefault('locked_car_id', None)
    state.setdefault('candidate_car_id', None)
    state.setdefault('candidate_hits', 0)
    state.setdefault('best_score', 0.0)
    state.setdefault('last_seen', int(frame_idx))
    state.setdefault('locked_missing_hits', 0)
    return state


def _plate_car_match_score(plate_box, car_box, frame_size=None):
    if plate_box is None or car_box is None:
        return 0.0
    px1, py1, px2, py2 = plate_box
    cx1, cy1, cx2, cy2 = car_box
    pcenter = ((px1 + px2) * 0.5, (py1 + py2) * 0.5)
    score = float(box_iou(plate_box, car_box))
    if point_in_box(pcenter, car_box):
        score = max(score, 1.0)
    if frame_size is not None and score > 0.0:
        frame_width, frame_height = frame_size
        car_width = max(float(cx2) - float(cx1), 1.0)
        car_height = max(float(cy2) - float(cy1), 1.0)
        truncated = (
            float(cx1) <= 2.0
            or float(cy1) <= 2.0
            or float(cx2) >= float(frame_width) - 2.0
            or float(cy2) >= float(frame_height) - 2.0
        )
        oversized = car_width >= float(frame_width) * 0.85 or car_height >= float(frame_height) * 0.85
        vertical_ratio = (pcenter[1] - float(cy1)) / car_height
        if truncated and oversized and vertical_ratio < 0.35:
            return 0.0
    return score


def _plate_box_mutual_score(dual_plate_box, primary_plate_box):
    if dual_plate_box is None or primary_plate_box is None:
        return 0.0
    iou = float(box_iou(dual_plate_box, primary_plate_box))
    dual_center = (
        0.5 * (float(dual_plate_box[0]) + float(dual_plate_box[2])),
        0.5 * (float(dual_plate_box[1]) + float(dual_plate_box[3])),
    )
    primary_center = (
        0.5 * (float(primary_plate_box[0]) + float(primary_plate_box[2])),
        0.5 * (float(primary_plate_box[1]) + float(primary_plate_box[3])),
    )
    dual_width = max(float(dual_plate_box[2]) - float(dual_plate_box[0]), 1.0)
    dual_height = max(float(dual_plate_box[3]) - float(dual_plate_box[1]), 1.0)
    primary_width = max(float(primary_plate_box[2]) - float(primary_plate_box[0]), 1.0)
    primary_height = max(float(primary_plate_box[3]) - float(primary_plate_box[1]), 1.0)
    center_distance = hypot(dual_center[0] - primary_center[0], dual_center[1] - primary_center[1])
    scale = max(hypot(dual_width, dual_height), hypot(primary_width, primary_height), 1.0)
    center_score = max(0.0, 1.0 - center_distance / (scale * 1.5))
    return max(iou, center_score)


def _stabilize_plate_binding(
    state,
    frame_idx,
    candidate_car_id,
    candidate_score,
    locked_score,
    lock_hits=2,
    switch_hits=3,
):
    state = _ensure_plate_binding_state(frame_idx, state)
    state['last_seen'] = int(frame_idx)
    lock_hits = max(1, int(lock_hits))
    switch_hits = max(1, int(switch_hits))
    candidate_car_id = _normalize_track_id(candidate_car_id)
    candidate_score = float(candidate_score or 0.0)
    locked_car_id = _normalize_track_id(state.get('locked_car_id'))
    state['locked_car_id'] = locked_car_id

    if candidate_car_id is None:
        state['candidate_car_id'] = None
        state['candidate_hits'] = 0
        state['best_score'] = 0.0
        return state

    if locked_car_id is None:
        if state.get('candidate_car_id') == candidate_car_id:
            state['candidate_hits'] = int(state.get('candidate_hits', 0)) + 1
            state['best_score'] = max(float(state.get('best_score', 0.0)), candidate_score)
        else:
            state['candidate_car_id'] = candidate_car_id
            state['candidate_hits'] = 1
            state['best_score'] = candidate_score
        if state['candidate_hits'] >= lock_hits:
            state['locked_car_id'] = candidate_car_id
            state['candidate_car_id'] = None
            state['candidate_hits'] = 0
            state['best_score'] = 0.0
            state['locked_missing_hits'] = 0
        return state

    if candidate_car_id == locked_car_id:
        state['candidate_car_id'] = None
        state['candidate_hits'] = 0
        state['best_score'] = 0.0
        state['locked_missing_hits'] = 0
        return state

    is_better = locked_score is not None and candidate_score > float(locked_score)
    if not is_better:
        state['candidate_car_id'] = None
        state['candidate_hits'] = 0
        state['best_score'] = 0.0
        return state

    if state.get('candidate_car_id') == candidate_car_id:
        state['candidate_hits'] = int(state.get('candidate_hits', 0)) + 1
        state['best_score'] = max(float(state.get('best_score', 0.0)), candidate_score)
    else:
        state['candidate_car_id'] = candidate_car_id
        state['candidate_hits'] = 1
        state['best_score'] = candidate_score

    if state['candidate_hits'] >= switch_hits:
        state['locked_car_id'] = candidate_car_id
        state['candidate_car_id'] = None
        state['candidate_hits'] = 0
        state['best_score'] = 0.0
        state['locked_missing_hits'] = 0
    return state


def _cleanup_plate_binding_states(
    plate_binding_states,
    frame_idx,
    active_car_ids,
    plate_timeout_frames,
    vehicle_missing_frames,
):
    frame_idx = int(frame_idx)
    plate_timeout_frames = max(0, int(plate_timeout_frames))
    vehicle_missing_frames = max(0, int(vehicle_missing_frames))
    active_car_ids = set(active_car_ids or ())
    removed_locked_ids = set()

    for plate_id in list(plate_binding_states.keys()):
        state = _ensure_plate_binding_state(frame_idx, plate_binding_states.get(plate_id))
        plate_binding_states[plate_id] = state
        last_seen = int(state.get('last_seen', frame_idx))
        locked_car_id = _normalize_track_id(state.get('locked_car_id'))
        state['locked_car_id'] = locked_car_id

        if frame_idx - last_seen > plate_timeout_frames:
            if locked_car_id is not None:
                removed_locked_ids.add(locked_car_id)
            plate_binding_states.pop(plate_id, None)
            continue

        if locked_car_id is None:
            state['locked_missing_hits'] = 0
            continue
        if locked_car_id in active_car_ids:
            state['locked_missing_hits'] = 0
            continue
        state['locked_missing_hits'] = int(state.get('locked_missing_hits', 0)) + 1
        if state['locked_missing_hits'] > vehicle_missing_frames:
            removed_locked_ids.add(locked_car_id)
            plate_binding_states.pop(plate_id, None)

    return removed_locked_ids


def _refresh_car_plate_cache_from_locked(
    plate_binding_states,
    car_plate_cache,
    active_car_ids,
    car_plate_cache_ttl,
):
    active_car_ids = set(active_car_ids or ())
    car_plate_cache_ttl = max(0, int(car_plate_cache_ttl))
    locked_car_to_plate = {}

    for plate_id, state in plate_binding_states.items():
        locked_car_id = _normalize_track_id(state.get('locked_car_id'))
        if locked_car_id is None:
            continue
        if locked_car_id in locked_car_to_plate:
            old_plate_id = locked_car_to_plate[locked_car_id]
            old_state = plate_binding_states.get(old_plate_id) or {}
            old_seen = int(old_state.get('last_seen', -1))
            new_seen = int(state.get('last_seen', -1))
            if old_seen >= new_seen:
                continue
        locked_car_to_plate[locked_car_id] = int(plate_id)

    for car_id in list(car_plate_cache.keys()):
        entry = car_plate_cache.get(car_id) or {}
        locked_plate_id = locked_car_to_plate.get(car_id)
        if locked_plate_id is None or int(entry.get('plate_id', -1)) != int(locked_plate_id):
            car_plate_cache.pop(car_id, None)

    for car_id, plate_id in locked_car_to_plate.items():
        if car_id not in active_car_ids:
            continue
        car_plate_cache[car_id] = {'plate_id': plate_id, 'age': 0}

    for car_id in list(car_plate_cache.keys()):
        if car_id in active_car_ids:
            continue
        car_plate_cache[car_id]['age'] = int(car_plate_cache[car_id].get('age', 0)) + 1
        if car_plate_cache[car_id]['age'] > car_plate_cache_ttl:
            car_plate_cache.pop(car_id, None)

    return locked_car_to_plate


def _append_pending_plate_candidate(
    pending_plate_cache,
    plate_id,
    text,
    frame_idx,
    conf=None,
    box=None,
    plate_color='',
    plate_color_conf=None,
    plate_type='',
    text_conf=None,
    mutual_verified=False,
    motion_consistent=False,
    trusted=False,
    max_entries=30,
):
    plate_id = _normalize_track_id(plate_id)
    if plate_id is None:
        return False
    text = normalize_plate_candidate_text(text)
    if not text or not is_valid_plate(text):
        return False
    try:
        frame_idx = int(frame_idx)
    except (TypeError, ValueError):
        frame_idx = 0
    history = pending_plate_cache.setdefault(plate_id, deque(maxlen=max(1, int(max_entries))))
    history.append({
        'text': text,
        'conf': conf,
        'frame': frame_idx,
        'box': list(box) if box is not None else None,
        'plate_color': str(plate_color or ''),
        'plate_color_conf': plate_color_conf,
        'plate_type': str(plate_type or ''),
        'text_conf': text_conf,
        'mutual_verified': bool(mutual_verified),
        'motion_consistent': bool(motion_consistent),
        'trusted': bool(trusted),
    })
    return True


def _valid_pending_plate_candidates(pending_plate_cache, plate_id, frame_idx, ttl_frames):
    plate_id = _normalize_track_id(plate_id)
    if plate_id is None:
        return []
    try:
        frame_idx = int(frame_idx)
    except (TypeError, ValueError):
        frame_idx = 0
    ttl_frames = max(0, int(ttl_frames))
    history = pending_plate_cache.get(plate_id) or ()
    entries = []
    for entry in history:
        if not isinstance(entry, dict):
            continue
        try:
            entry_frame = int(entry.get('frame', frame_idx))
        except (TypeError, ValueError):
            continue
        age = frame_idx - entry_frame
        if 0 <= age <= ttl_frames:
            entries.append(dict(entry))
    return entries


def _consume_pending_plate_candidates(pending_plate_cache, plate_id, frame_idx, ttl_frames):
    entries = _valid_pending_plate_candidates(pending_plate_cache, plate_id, frame_idx, ttl_frames)
    if entries:
        pending_plate_cache.pop(int(plate_id), None)
    return entries


def _cleanup_pending_plate_cache(pending_plate_cache, frame_idx, ttl_frames):
    try:
        frame_idx = int(frame_idx)
    except (TypeError, ValueError):
        frame_idx = 0
    ttl_frames = max(0, int(ttl_frames))
    for plate_id in list(pending_plate_cache.keys()):
        history = pending_plate_cache.get(plate_id)
        if not history:
            pending_plate_cache.pop(plate_id, None)
            continue
        maxlen = getattr(history, 'maxlen', None)
        kept = deque(maxlen=maxlen)
        for entry in history:
            if not isinstance(entry, dict):
                continue
            try:
                entry_frame = int(entry.get('frame', frame_idx))
            except (TypeError, ValueError):
                continue
            age = frame_idx - entry_frame
            if 0 <= age <= ttl_frames:
                kept.append(entry)
        if kept:
            pending_plate_cache[plate_id] = kept
        else:
            pending_plate_cache.pop(plate_id, None)


def process_video(path, args):
    cap, decode_meta = create_video_reader(path, args)

    def _log_decode_open_result(stage, meta, success):
        mode = str((meta or {}).get('decode_mode') or 'none')
        fallback_used = bool((meta or {}).get('fallback_used'))
        fallback_reason = str((meta or {}).get('fallback_reason') or '')
        source_kind = str((meta or {}).get('source_kind') or 'other')
        if success:
            if mode == 'hw':
                if fallback_used:
                    print(
                        f'[reader] {stage}成功：硬解备用链路成功 '
                        f'source_kind={source_kind} fallback_reason={fallback_reason or "unknown"}'
                    )
                else:
                    print(f'[reader] {stage}成功：硬解成功 source_kind={source_kind}')
                return
            if fallback_used:
                print(
                    f'[reader] {stage}成功：解码备用链路成功 '
                    f'source_kind={source_kind} fallback_reason={fallback_reason or "unknown"}'
                )
                return
            print(f'[reader] {stage}成功：解码成功 source_kind={source_kind}')
            return
        if fallback_used:
            print(
                f'[reader] {stage}失败：硬解链路不可用 '
                f'source_kind={source_kind} fallback_reason={fallback_reason or "unknown"}'
            )
            return
        print(
            f'[reader] {stage}失败：没有可用硬解链路 '
            f'source_kind={source_kind} fallback_reason={fallback_reason or "unknown"}'
        )

    if cap is None or not hasattr(cap, 'isOpened') or not cap.isOpened():
        _log_decode_open_result('首次打开', decode_meta, False)
        print(f'failed to open {path}')
        return
    _log_decode_open_result('首次打开', decode_meta, True)
    fps = None
    if hasattr(cap, 'get'):
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
        except AttributeError:
            fps = None
    if not fps:
        fps = getattr(cap, 'fps', None)
    if not fps:
        fps = 25.0
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    config = getattr(args, '_config', load_config(None))
    base_dir = getattr(args, '_config_dir', Path.cwd())
    video_cfg = config.get('video', {})
    logic_cfg = config.get('logic', {})
    track_retention = resolve_track_retention_frames(fps, logic_cfg=logic_cfg, config=config)
    config['track_max_age'] = track_retention['tracker_max_age_frames']
    config['track_timeout_frames'] = track_retention['event_timeout_frames']
    print(
        f'[tracker] source_fps={track_retention["source_fps"]:.2f} '
        f'lost_grace={track_retention["grace_seconds"]:.2f}s '
        f'max_age={track_retention["tracker_max_age_frames"]}frames '
        f'event_timeout={track_retention["event_timeout_frames"]}frames'
    )
    system_cfg = config.get('system', {})
    reader_fail_threshold = max(1, int(config.get('reader_fail_threshold', 5)))
    reader_reconnect_delay = max(0.0, float(config.get('reader_reconnect_delay', 2.0)))
    reader_max_reconnect = max(0, int(config.get('reader_max_reconnect', 0)))
    source_mode = detect_source_mode(path, getattr(args, 'source_mode', 'auto'), base_dir=base_dir)
    is_file_input = (source_mode == 'file')
    if is_file_input:
        reader_max_reconnect = 0
    else:
        source_mode = 'camera'
    print(f'[reader] source_mode={source_mode}')
    drop_stale_frames = _should_drop_stale_frames(source_mode)
    consecutive_fails = 0
    reconnect_count = 0
    log_throttle = WindowedLogThrottle()

    def _throttled_log(key, message, window_seconds=10.0):
        log_throttle.log(key=key, message=message, window_seconds=window_seconds, emit=print)

    def _capture_failure_extra(cap_obj):
        if cap_obj is None or not hasattr(cap_obj, 'diagnostics'):
            return {}
        try:
            diag = cap_obj.diagnostics() or {}
        except Exception:
            return {}
        recent_errors = diag.get('recent_error_lines') or []
        extra = {
            'reader_last_error': str(diag.get('last_read_error') or ''),
            'reader_recent_error_count': int(diag.get('recent_error_match_count') or 0),
        }
        if recent_errors:
            extra['reader_recent_error_tail'] = recent_errors[-4:]
        return extra

    def _capture_failure_summary(cap_obj):
        extra = _capture_failure_extra(cap_obj)
        last_error = str(extra.get('reader_last_error') or '')
        recent_error_count = int(extra.get('reader_recent_error_count') or 0)
        error_tail = extra.get('reader_recent_error_tail') or []
        parts = []
        if last_error:
            parts.append(f'last_error={last_error}')
        if recent_error_count:
            parts.append(f'recent_decode_errors={recent_error_count}')
        if error_tail:
            parts.append(f'error_tail={error_tail}')
        return ' '.join(parts), extra

    config_path = str(getattr(args, '_config_path', config.get('config_path', '')) or '')
    config_name = str(config.get('config_name') or (Path(config_path).name if config_path else ''))
    config_namespace_hint = Path(config_path).stem if config_path else ''
    launch_id = str(os.environ.get('CLEANINGCAR_LAUNCH_ID', '') or '').strip()
    runtime_namespace_key_env = str(os.environ.get('CLEANINGCAR_RUNTIME_NAMESPACE_KEY', '') or '').strip()
    runtime_device_id_env = str(os.environ.get('CLEANINGCAR_DEVICE_ID', '') or '').strip()

    metrics_path_conf = system_cfg.get('metrics_path', '/dev/shm/cleaningcar_metrics.json')
    metrics_path = None
    if metrics_path_conf:
        metrics_path = _resolve_runtime_path(metrics_path_conf, base_dir)
    runtime_settings = resolve_runtime_settings(config, base_dir, namespace_hint=config_namespace_hint or None)
    command_dir = runtime_settings['command_dir']
    heartbeat_path = runtime_settings['heartbeat_path']
    startup_flag_path = runtime_settings['startup_flag_path']
    startup_capture_dir = runtime_settings['startup_capture_dir']
    manual_capture_dir = runtime_settings['manual_capture_dir']
    runtime_namespace_key = runtime_namespace_key_env or str(runtime_settings.get('runtime_namespace_key') or '')
    runtime_device_id = runtime_device_id_env or str(runtime_settings.get('device_id') or system_cfg.get('device_id') or '')
    debug_frame_file = runtime_settings.get('debug_frame_path')
    heartbeat_interval_seconds = float(runtime_settings['heartbeat_interval_seconds'])
    command_poll_interval = min(heartbeat_interval_seconds, 0.5)
    storage_cfg = config.get('storage', {}) or {}
    zones_cfg = config.get('zones', {})
    logic_cfg = config.get('logic', {})
    zone_b_anchor_min_frames = int(logic_cfg.get('zone_b_anchor_min_frames', 0))
    if zone_b_anchor_min_frames < 0:
        zone_b_anchor_min_frames = 0
    zone_a_pts = scale_polygon(zones_cfg.get('zone_a_detection', []), width, height)
    zone_b_pts = scale_polygon(zones_cfg.get('zone_b_wash', []), width, height)
    flow_vec = zones_cfg.get('flow_vector', {})
    flow_start = scale_point(flow_vec.get('start', (0.0, 0.0)), width, height)
    flow_end = scale_point(flow_vec.get('end', (0.0, 1.0)), width, height)
    zone_mgr = ZoneManager(
        zone_a_pts,
        zone_b_pts,
        (flow_start, flow_end),
        entry_hysteresis=int(logic_cfg.get('zone_b_entry_hysteresis', 3)),
        exit_hysteresis=int(logic_cfg.get('zone_b_exit_hysteresis', 3)),
        zone_a_margin_ratio=float(logic_cfg.get('zone_a_margin_ratio', 0.10)),
        zone_a_margin_min_px=float(logic_cfg.get('zone_a_margin_min_px', 4.0)),
        zone_a_margin_max_px=float(logic_cfg.get('zone_a_margin_max_px', 24.0)),
        zone_a_observed_outside_hits=int(logic_cfg.get('zone_a_observed_outside_hits', 3)),
        zone_a_enter_core_hits=int(logic_cfg.get('zone_a_enter_core_hits', 3)),
        zone_a_exit_outside_hits=int(logic_cfg.get('zone_a_exit_outside_hits', 5)),
    )
    anchor_estimator = AnchorEstimator(
        frame_size=(width, height),
        flow_vector=(flow_start, flow_end),
        logic_cfg=logic_cfg,
    )
    anchor_trace_recorded = set()

    def anchor_result_for(track_id, box, frame_idx):
        result = anchor_estimator.estimate(track_id, box, frame_idx)
        if result is None:
            return None
        trace_key = (int(result.track_id), int(result.frame_idx))
        if trace_key not in anchor_trace_recorded:
            anchor_trace_recorded.add(trace_key)
            try:
                event_manager.trace_record('anchor_update', result.to_dict())
            except Exception:
                pass
        return result

    def anchor_point_for(track_id, box, frame_idx):
        result = anchor_result_for(track_id, box, frame_idx)
        return result.selected_point if result is not None else None

    def anchor_evidence_for(track_id, box, frame_idx):
        result = anchor_result_for(track_id, box, frame_idx)
        if result is None:
            return None, None
        return result.selected_point, {
            'motion_direction': result.motion_direction,
            'direction_confidence': result.direction_confidence,
            'direction_locked': result.direction_locked,
        }

    output_dir = getattr(args, 'output_dir', None)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    csv_writer = None
    csv_f = None
    plate_track_lock_frames = int(getattr(args, 'plate_track_lock_frames', 6))
    plate_tracker = PlateTextTracker(
        lock_frames=plate_track_lock_frames,
        max_age=max(int(config.get('track_timeout_frames', 60)) * 2, plate_track_lock_frames * 6),
        min_detection_confidence=float(logic_cfg.get('plate_text_min_detection_confidence', 0.65)),
        min_recognition_confidence=float(logic_cfg.get('plate_text_min_recognition_confidence', 0.75)),
        max_streak_gap_frames=max(1, int(logic_cfg.get('plate_text_max_streak_gap_frames', 2))),
    )
    vehicle_iou_thresh = float(config.get('vehicle_iou_threshold', 0.3))
    if vehicle_iou_thresh < 0.0:
        vehicle_iou_thresh = 0.0
    elif vehicle_iou_thresh > 1.0:
        vehicle_iou_thresh = 1.0
    vehicle_center_gate_ratio = float(config.get('vehicle_center_gate_ratio', 0.0) or 0.0)
    if vehicle_center_gate_ratio < 0.0:
        vehicle_center_gate_ratio = 0.0
    vehicle_tracker_impl = str(logic_cfg.get('vehicle_tracker_impl', 'bytetrack') or 'bytetrack').strip().lower()
    vehicle_tracker = VehicleTracker(
        iou_thresh=vehicle_iou_thresh,
        max_age=int(config.get('track_max_age', 60)),
        center_gate_ratio=vehicle_center_gate_ratio,
        tracker_impl=vehicle_tracker_impl,
    )
    print(f'[tracker] impl={vehicle_tracker_impl}')
    default_event_log = os.path.join(config.get('event_output_dir', './events'), 'event_log.csv')
    event_log_arg = getattr(args, 'event_log', None)
    event_log_path = default_event_log if (not event_log_arg or event_log_arg == 'auto') else event_log_arg
    uploader = None
    if getattr(args, 'api_url', None):
        queue_db = Path(config.get('event_output_dir', './events')) / 'upload_queue.db'
        uploader = EventUploader(args.api_url, getattr(args, 'api_token', None), queue_path=queue_db)
    wheel_photo_uploader = None
    wheel_photo_url = config.get('wheel_photo_url') or getattr(args, 'wheel_photo_url', '')
    if wheel_photo_url:
        wheel_photo_queue_db = Path(config.get('event_output_dir', './events')) / 'wheel_photo_queue.db'
        wheel_photo_uploader = WheelPhotoUploader(
            wheel_photo_url,
            getattr(args, 'api_token', None),
            queue_path=wheel_photo_queue_db,
        )
    wheel_cfg = config.get('wheel', {}) or {}
    wheel_photo_base_dir = config.get('wheel_photo_base_dir') or '/data/ftp'
    session_id = datetime.now().strftime('%H%M%S')
    capture_mode = getattr(args, 'capture_mode', 'path')
    event_manager = EventManager(config, fps, (width, height), zone_mgr, event_log_path, uploader=uploader,
                                 capture_mode=capture_mode,
                                 wheel_photo_uploader=wheel_photo_uploader,
                                 wheel_photo_base_dir=wheel_photo_base_dir,
                                 session_id=session_id,
                                 wheel_photo_bucket_seconds=float(wheel_cfg.get('photo_bucket_seconds', 0.5)),
                                 wheel_photo_min_score=float(wheel_cfg.get('photo_min_score', 0.3)))
    wheel_service = None
    candidate_wheel_service = None
    try:
        candidate_wheel_service = WheelDetectionService(
            config=config,
            base_dir=base_dir,
            hw_decode=bool(getattr(args, 'hw_decode', False)),
            imgsz=int(getattr(args, 'imgsz', 640) or 640),
            image_quality=int(config.get('event_capture_quality', 85) or 85),
        )
        if candidate_wheel_service.start():
            wheel_service = candidate_wheel_service
            event_manager.wheel_result_provider = wheel_service
    except Exception as exc:
        try:
            if candidate_wheel_service is not None:
                candidate_wheel_service.stop()
        except Exception:
            pass
        print(f'[wheel] sidechain init failed, continue without wheel binding: {exc}')
    if args.csv or output_dir:
        csv_path = args.csv
        if csv_path and os.path.isdir(csv_path):
            csv_path = os.path.join(csv_path, Path(path).stem + '.csv')
        if not csv_path and output_dir:
            csv_path = os.path.join(output_dir, Path(path).stem + '.csv')
        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        csv_f = open(csv_path, 'w', newline='', encoding='utf-8', buffering=1024 * 1024)
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(['frame', 'class', 'score', 'x1', 'y1', 'x2', 'y2', 'track_id', 'text', 'raw_text'])

    debug_overlay_flag = bool(logic_cfg.get('debug_overlay', False))
    debug_tracks_cfg = bool(logic_cfg.get('debug_track_state', False))
    debug_anchor_points = bool(logic_cfg.get('debug_anchor_points', False) or debug_overlay_flag)
    debug_water_boxes = bool(logic_cfg.get('debug_water_boxes', False) or debug_overlay_flag)
    debug_rois = getattr(args, 'debug_rois', False) or debug_overlay_flag
    debug_tracks = getattr(args, 'debug_tracks', False) or debug_tracks_cfg or debug_overlay_flag
    draw_plate_boxes = bool(getattr(args, 'draw_plate_boxes', False) or logic_cfg.get('draw_plate_boxes', False))
    setattr(args, 'draw_plate_boxes', draw_plate_boxes)
    plate_draw_stable_only = bool(logic_cfg.get('plate_draw_stable_only', True))
    event_use_annotated_frame = bool(not args.no_draw and (draw_plate_boxes or debug_water_boxes))
    per_id_draw_enabled = bool(not args.no_draw)

    if debug_frame_file:
        debug_frame_file = Path(debug_frame_file)
        if debug_frame_file.is_dir():
            debug_frame_file = debug_frame_file / 'latest.jpg'
        try:
            debug_frame_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    debug_frame_interval = max(1, int(video_cfg.get('debug_frame_interval', 30)))
    copy_raw_frame_cache = bool(logic_cfg.get('copy_raw_frame_cache', False))
    per_id_video_source = _resolve_per_id_video_source(
        logic_cfg,
        no_draw=args.no_draw,
        draw_enabled=per_id_draw_enabled,
    )
    if (
        args.no_draw and (debug_frame_file or debug_tracks or debug_rois or debug_water_boxes or debug_anchor_points)
    ) or (
        bool(logic_cfg.get('enable_per_id_video', False)) and per_id_video_source == 'raw'
    ):
        copy_raw_frame_cache = True
    debug_frame_max_width = max(0, int(video_cfg.get('debug_frame_max_width', 960) or 0))
    debug_frame_quality = min(max(int(video_cfg.get('debug_frame_quality', 80) or 80), 1), 100)

    start = time.time()
    total_frames = 0
    next_frame_to_write = 0
    pending = {}
    raw_frame_cache = {}
    reader_log_interval = float(config.get('reader_fps_log_interval', 10.0))
    reader_log_last_time = start
    reader_log_frames = 0
    latest_raw_frame = None
    latest_annotated_frame = None
    latest_frame_idx = -1
    latest_capture_ts = None
    last_progress_ts = start
    last_frame_read_ts = None
    last_result_ts = None
    last_heartbeat_write = 0.0
    startup_emitted = False
    last_command_poll = 0.0
    runtime_paused = False
    runtime_pause_reason = ''
    runtime_pause_changed_ts = None
    wash_priority_active = False
    wash_priority_active_tracks = []
    wash_priority_last_change_ts = None
    task_q = Queue(maxsize=args.queue_size)
    result_q = Queue()
    dropped_frame_count = 0
    dropped_frame_ids = set()

    car_plate_cache = {}
    car_plate_cache_ttl = int(config.get('car_plate_cache_ttl', CAR_PLATE_CACHE_TTL))
    plate_binding_states = {}
    plate_binding_timeout_frames = max(1, int(config.get('track_timeout_frames', 60)))
    plate_binding_vehicle_missing_frames = max(1, int(config.get('track_timeout_frames', 60)))
    pending_plate_cache = {}
    pending_plate_cache_ttl_frames = max(1, int(logic_cfg.get('pending_plate_cache_ttl_frames', 40) or 40))
    pending_plate_cache_max_entries = max(1, int(logic_cfg.get('pending_plate_cache_max_entries', 30) or 30))
    alias_confirm = {}
    alias_timeout = int(config.get('track_timeout_frames', 60))
    enable_per_id_video = bool(logic_cfg.get('enable_per_id_video', False))
    event_manager.per_id_video_enabled = enable_per_id_video
    benchmark_force_recording_track_id = int(logic_cfg.get('benchmark_force_recording_track_id', 0) or 0)
    benchmark_force_capture_stride = max(0, int(logic_cfg.get('benchmark_force_capture_stride', 0) or 0))
    per_id_video_dir = None
    per_id_writers = {}
    per_id_params = _resolve_per_id_recording_params(width, height, fps, logic_cfg=logic_cfg)
    per_id_target_width = per_id_params['width']
    per_id_target_height = per_id_params['height']
    per_id_output_fps = per_id_params['fps']
    per_id_record_stride = per_id_params['frame_stride']
    per_id_video_queue_size = max(1, int(logic_cfg.get('per_id_video_queue_size', 8) or 8))
    resize_backend_label = (
        'passthrough'
        if per_id_target_width == width and per_id_target_height == height
        else resize_backend_name()
    )

    def resize_per_id_frame(frame_to_write):
        if frame_to_write is None:
            return None
        if (
            frame_to_write.shape[1] == per_id_target_width
            and frame_to_write.shape[0] == per_id_target_height
        ):
            return frame_to_write
        return resize_bgr(frame_to_write, (per_id_target_width, per_id_target_height))

    def probe_writable_directory(directory: Path):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return False, str(exc)
        probe_path = directory / f'.write_probe_{os.getpid()}_{time.time_ns()}'
        try:
            with probe_path.open('w', encoding='utf-8') as f:
                f.write('ok')
            probe_path.unlink()
            return True, ''
        except Exception as exc:
            try:
                if probe_path.exists():
                    probe_path.unlink()
            except Exception:
                pass
            return False, str(exc)

    def cleanup_empty_per_id_dirs(directory: Path, stop_at: Path):
        try:
            current = Path(directory).resolve()
            stop_dir = Path(stop_at).resolve()
        except Exception:
            return
        while current != stop_dir and current != current.parent:
            try:
                current.rmdir()
            except OSError:
                break
            current = current.parent

    def resolve_per_id_video_root():
        raw_path = str(logic_cfg.get('per_id_video_dir', '') or '').strip()
        if raw_path:
            configured_dir = _resolve_runtime_path(raw_path, PROJECT_ROOT)
            ok, reason = probe_writable_directory(configured_dir)
            if ok:
                return configured_dir
            print(
                f'[per-id-video] configured directory not writable, fallback to local default: '
                f'configured={configured_dir} reason={reason}'
            )
        else:
            print(f'[per-id-video] per_id_video_dir is empty, fallback to local default: {DEFAULT_PER_ID_VIDEO_DIR}')

        ok, reason = probe_writable_directory(DEFAULT_PER_ID_VIDEO_DIR)
        if ok:
            return DEFAULT_PER_ID_VIDEO_DIR

        print(
            f'[per-id-video] local fallback directory unavailable, disable per-id video: '
            f'dir={DEFAULT_PER_ID_VIDEO_DIR} reason={reason}'
        )
        return None

    def build_per_id_writer(track_id, track_state, capture_dt, session_id):
        nonlocal enable_per_id_video, per_id_video_dir
        if per_id_video_dir is None:
            return None

        fname = f'{session_id}.mp4'
        date_dir = capture_dt.strftime('%Y%m%d')
        hour_dir = capture_dt.strftime('%H')
        roots = [per_id_video_dir]
        if per_id_video_dir != DEFAULT_PER_ID_VIDEO_DIR:
            roots.append(DEFAULT_PER_ID_VIDEO_DIR)

        for root in roots:
            is_fallback_root = (root == DEFAULT_PER_ID_VIDEO_DIR and root != per_id_video_dir)
            base_dir = root / date_dir / hour_dir
            try:
                base_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                label = 'local fallback' if is_fallback_root else 'configured'
                print(
                    f'[per-id-video] failed to create output directory ({label}), '
                    f'track={track_id} dir={base_dir} reason={exc}'
                )
                cleanup_empty_per_id_dirs(base_dir, root)
                continue

            target_path = base_dir / fname
            try:
                writer_obj, writer_meta = create_h264_video_writer(
                    str(target_path),
                    per_id_target_width,
                    per_id_target_height,
                    per_id_output_fps,
                )
            except Exception as exc:
                label = 'local fallback' if is_fallback_root else 'configured'
                print(
                    f'[per-id-video] writer init raised ({label}), '
                    f'track={track_id} path={target_path} reason={exc}'
                )
                cleanup_empty_per_id_dirs(base_dir, root)
                continue

            if writer_obj is not None and writer_obj.is_opened():
                if writer_meta:
                    print(
                        f'[per-id-video] writer selected backend={writer_meta.get("writer_backend")} '
                        f'mode={writer_meta.get("writer_mode")} path={target_path}'
                    )
                if is_fallback_root:
                    print(f'[per-id-video] switched to local fallback directory: {root}')
                    per_id_video_dir = root
                return AsyncPerIdVideoWriter(
                    writer_obj,
                    resize_fn=resize_per_id_frame,
                    queue_size=per_id_video_queue_size,
                    log_interval=reader_log_interval,
                )

            label = 'local fallback' if is_fallback_root else 'configured'
            print(f'[per-id-video] H.264 writer init failed ({label}), track={track_id} path={target_path}')
            cleanup_empty_per_id_dirs(base_dir, root)

        print(f'[per-id-video] no usable writer output path, disable per-id video for this run: track={track_id}')
        enable_per_id_video = False
        event_manager.per_id_video_enabled = False
        return None

    if enable_per_id_video:
        per_id_video_dir = resolve_per_id_video_root()
        if per_id_video_dir is None:
            enable_per_id_video = False
            event_manager.per_id_video_enabled = False
        else:
            event_manager.per_id_video_enabled = True
            print(
                f'[per-id-video] target_size={per_id_target_width}x{per_id_target_height} '
                f'fps={per_id_output_fps:.2f} stride={per_id_record_stride} '
                f'source={per_id_video_source} queue={per_id_video_queue_size} '
                f'resize_backend={resize_backend_label}'
            )

    def collect_per_id_cleanup_roots():
        roots = [DEFAULT_PER_ID_VIDEO_DIR]
        raw_path = str(logic_cfg.get('per_id_video_dir', '') or '').strip()
        if raw_path:
            configured_dir = _resolve_runtime_path(raw_path, PROJECT_ROOT)
            if configured_dir:
                roots.insert(0, configured_dir)
        unique = []
        seen = set()
        for root in roots:
            resolved = Path(root).resolve()
            key = str(resolved)
            if key in seen:
                continue
            seen.add(key)
            unique.append(resolved)
        return unique

    cleanup_policies = []
    capture_keep_days = int(storage_cfg.get('capture_keep_days', 30) or 0)
    capture_keep_count = int(storage_cfg.get('capture_keep_count', 3000) or 0)
    per_id_video_keep_days = int(storage_cfg.get('per_id_video_keep_days', 15) or 0)
    per_id_video_keep_count = int(storage_cfg.get('per_id_video_keep_count', 500) or 0)
    event_capture_dir = _resolve_runtime_path(config.get('event_capture_dir'), base_dir)
    if event_capture_dir:
        cleanup_policies.append(
            RetentionPolicy(
                root=event_capture_dir,
                keep_days=capture_keep_days,
                keep_count=capture_keep_count,
                label='event-captures',
            )
        )
    if startup_capture_dir:
        cleanup_policies.append(
            RetentionPolicy(
                root=startup_capture_dir,
                keep_days=capture_keep_days,
                keep_count=capture_keep_count,
                label='startup-captures',
                preserve_latest_groups=1,
                snapshot_grouping=True,
            )
        )
    if manual_capture_dir:
        cleanup_policies.append(
            RetentionPolicy(
                root=manual_capture_dir,
                keep_days=capture_keep_days,
                keep_count=capture_keep_count,
                label='manual-captures',
                preserve_latest_groups=1,
                snapshot_grouping=True,
            )
        )
    for cleanup_root in collect_per_id_cleanup_roots():
        cleanup_policies.append(
            RetentionPolicy(
                root=cleanup_root,
                keep_days=per_id_video_keep_days,
                keep_count=per_id_video_keep_count,
                label='per-id-video',
            )
        )
    storage_cleaner = RuntimeStorageCleaner(
        cleanup_policies,
        int(storage_cfg.get('clean_interval_seconds', 600) or 600),
    )

    for runtime_dir in (command_dir, heartbeat_path.parent, startup_flag_path.parent):
        try:
            Path(runtime_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    try:
        if startup_flag_path.exists():
            startup_flag_path.unlink()
    except Exception:
        pass

    def _safe_qsize(queue_obj):
        try:
            return int(queue_obj.qsize())
        except Exception:
            return -1

    def _mark_dropped_frame(frame_idx):
        nonlocal dropped_frame_count, next_frame_to_write
        if frame_idx is None:
            return
        try:
            frame_idx = int(frame_idx)
        except (TypeError, ValueError):
            return
        if frame_idx < next_frame_to_write:
            return
        dropped_frame_ids.add(frame_idx)
        raw_frame_cache.pop(frame_idx, None)
        dropped_frame_count += 1
        try:
            event_manager.trace_record('dropped_frame', {
                'frameIdx': frame_idx,
                'droppedFramesTotal': dropped_frame_count,
            })
        except Exception:
            pass
        _throttled_log(
            'realtime.dropped_stale_frame',
            f'[realtime] dropped stale frame idx={frame_idx}, total={dropped_frame_count}',
            window_seconds=10.0,
        )

    def _advance_dropped_frames():
        nonlocal next_frame_to_write
        moved = 0
        while next_frame_to_write in dropped_frame_ids and next_frame_to_write not in pending:
            dropped_frame_ids.discard(next_frame_to_write)
            raw_frame_cache.pop(next_frame_to_write, None)
            next_frame_to_write += 1
            moved += 1
        return moved

    def _drop_stale_task_for_realtime():
        try:
            stale_item = task_q.get_nowait()
        except Empty:
            return False
        task_q.task_done()
        if stale_item is None:
            return False
        stale_idx = stale_item[0] if isinstance(stale_item, tuple) and stale_item else None
        _mark_dropped_frame(stale_idx)
        return True

    def write_startup_heartbeat(stage='booting'):
        _refresh_wash_priority_state()
        now = time.time()
        payload = {
            'timestamp': now,
            'pid': os.getpid(),
            'launch_id': launch_id,
            'runtime_namespace_key': runtime_namespace_key,
            'device_id': runtime_device_id,
            'status': 'starting',
            'startup_emitted': startup_emitted,
            'total_frames': total_frames,
            'next_frame_to_write': next_frame_to_write,
            'pending_results': len(pending),
            'task_queue_size': _safe_qsize(task_q),
            'result_queue_size': _safe_qsize(result_q),
            'dropped_frames': dropped_frame_count,
            'dropped_pending': len(dropped_frame_ids),
            'last_progress_ts': now,
            'last_frame_read_ts': last_frame_read_ts,
            'last_result_ts': last_result_ts,
            'latest_frame_idx': latest_frame_idx,
            'latest_capture_ts': latest_capture_ts,
            'reconnect_count': reconnect_count,
            'config_path': config_path,
            'config_name': config_name,
            'source': str(path),
            'startup_stage': stage,
            'runtime_paused': runtime_paused,
            'runtime_pause_reason': runtime_pause_reason,
            'runtime_pause_changed_ts': runtime_pause_changed_ts,
            'wash_priority_active': wash_priority_active,
            'wash_priority_active_tracks': wash_priority_active_tracks,
            'wash_priority_last_change_ts': wash_priority_last_change_ts,
        }
        try:
            write_json_atomic(heartbeat_path, payload)
        except Exception:
            pass

    def write_heartbeat(status='running', force=False, extra=None):
        nonlocal last_heartbeat_write
        now = time.time()
        if not force and (now - last_heartbeat_write) < heartbeat_interval_seconds:
            return
        _refresh_wash_priority_state()
        payload = {
            'timestamp': now,
            'pid': os.getpid(),
            'launch_id': launch_id,
            'runtime_namespace_key': runtime_namespace_key,
            'device_id': runtime_device_id,
            'status': status,
            'startup_emitted': startup_emitted,
            'total_frames': total_frames,
            'next_frame_to_write': next_frame_to_write,
            'pending_results': len(pending),
            'task_queue_size': _safe_qsize(task_q),
            'result_queue_size': _safe_qsize(result_q),
            'dropped_frames': dropped_frame_count,
            'dropped_pending': len(dropped_frame_ids),
            'last_progress_ts': last_progress_ts,
            'last_frame_read_ts': last_frame_read_ts,
            'last_result_ts': last_result_ts,
            'latest_frame_idx': latest_frame_idx,
            'latest_capture_ts': latest_capture_ts,
            'reconnect_count': reconnect_count,
            'config_path': config_path,
            'config_name': config_name,
            'source': str(path),
            'runtime_paused': runtime_paused,
            'runtime_pause_reason': runtime_pause_reason,
            'runtime_pause_changed_ts': runtime_pause_changed_ts,
            'wash_priority_active': wash_priority_active,
            'wash_priority_active_tracks': wash_priority_active_tracks,
            'wash_priority_last_change_ts': wash_priority_last_change_ts,
        }
        if extra:
            payload.update(extra)
        try:
            write_json_atomic(heartbeat_path, payload)
            last_heartbeat_write = now
        except Exception:
            pass

    def _refresh_wash_priority_state():
        nonlocal wash_priority_active, wash_priority_active_tracks, wash_priority_last_change_ts
        active_tracks = []
        for tid, st in list(getattr(event_manager, 'tracks', {}).items()):
            if not isinstance(st, dict):
                continue
            events = st.get('events') or set()
            if st.get('closed') or 5 in events:
                continue
            zone_active = bool(st.get('zone_a_enter_frame', -1) is not None and int(st.get('zone_a_enter_frame', -1) or -1) >= 0)
            can_type1_fn = getattr(event_manager, '_can_emit_type1', None)
            stable_zone_active = bool(zone_active and callable(can_type1_fn) and can_type1_fn(st))
            if 1 in events or stable_zone_active:
                active_tracks.append(int(tid))
        active_tracks.sort()
        active_now = bool(active_tracks)
        if active_now != wash_priority_active or active_tracks != wash_priority_active_tracks:
            wash_priority_active = active_now
            wash_priority_active_tracks = active_tracks
            wash_priority_last_change_ts = time.time()

    def emit_startup_signal_if_needed():
        nonlocal startup_emitted
        if startup_emitted or latest_frame_idx < 0:
            return
        if latest_raw_frame is None and latest_annotated_frame is None:
            return
        saved = save_snapshot_images(
            startup_capture_dir,
            latest_frame_idx,
            latest_raw_frame,
            latest_annotated_frame,
            capture_ts=latest_capture_ts,
            tag='startup',
            save_raw=True,
            save_annotated=True,
            signal_text='SYSTEM STARTED',
        )
        payload = {
            'timestamp': time.time(),
            'pid': os.getpid(),
            'launch_id': launch_id,
            'runtime_namespace_key': runtime_namespace_key,
            'device_id': runtime_device_id,
            'config_path': config_path,
            'config_name': config_name,
            'frame_idx': latest_frame_idx,
            'source': str(path),
            'captures': saved,
        }
        write_json_atomic(startup_flag_path, payload)
        startup_emitted = True
        print(
            f'[startup-signal] ready frame={latest_frame_idx} '
            f'flag={startup_flag_path} captures={saved}'
        )

    def handle_keep_snapshot(command_payload):
        tag = command_payload.get('tag') or 'manual'
        save_raw = bool(command_payload.get('raw', True))
        save_annotated = bool(command_payload.get('annotated', True))
        if not save_raw and not save_annotated:
            raise ValueError('raw/annotated cannot both be false')
        if latest_frame_idx < 0:
            raise RuntimeError('no frame cached yet')
        saved = save_snapshot_images(
            manual_capture_dir,
            latest_frame_idx,
            latest_raw_frame,
            latest_annotated_frame,
            capture_ts=latest_capture_ts,
            tag=tag,
            save_raw=save_raw,
            save_annotated=save_annotated,
        )
        if not saved:
            raise RuntimeError('snapshot save failed')
        print(f'[snapshot] kept frame={latest_frame_idx} tag={tag} files={saved}')
        return saved

    def handle_set_runtime_pause(command_payload):
        nonlocal cap, runtime_paused, runtime_pause_reason, runtime_pause_changed_ts, last_progress_ts
        paused = command_payload.get('paused', False)
        if isinstance(paused, str):
            paused = paused.strip().lower() in {'1', 'true', 'yes', 'on'}
        else:
            paused = bool(paused)
        reason = str(command_payload.get('reason') or 'runtime_command').strip() or 'runtime_command'
        if runtime_paused == paused and runtime_pause_reason == reason:
            return {'paused': runtime_paused, 'reason': runtime_pause_reason}
        runtime_paused = paused
        runtime_pause_reason = reason if paused else ''
        runtime_pause_changed_ts = time.time()
        last_progress_ts = runtime_pause_changed_ts
        if paused and cap is not None:
            try:
                cap.release()
            except Exception:
                pass
            cap = None
        print(f'[runtime-pause] paused={int(runtime_paused)} reason={reason}')
        return {'paused': runtime_paused, 'reason': reason}

    def _command_result_path(command_path, suffix):
        name = command_path.name
        if name.endswith('.cmd.json'):
            name = f'{name[:-9]}.{suffix}.json'
        else:
            name = f'{name}.{suffix}.json'
        return command_path.with_name(name)

    def poll_runtime_commands(force=False):
        nonlocal last_command_poll
        now = time.time()
        if not force and (now - last_command_poll) < command_poll_interval:
            return
        last_command_poll = now
        try:
            command_files = sorted(Path(command_dir).glob('*.cmd.json'))
        except Exception:
            return
        for command_path in command_files[:20]:
            payload = load_json_file(command_path)
            meta = {
                'command_file': str(command_path),
                'handled_at': time.time(),
                'pid': os.getpid(),
            }
            try:
                if not payload:
                    raise ValueError('invalid command payload')
                command_name = str(payload.get('cmd', '')).strip().lower()
                if command_name == 'keep_snapshot':
                    saved = handle_keep_snapshot(payload)
                    result_payload = {
                        **meta,
                        'status': 'done',
                        'cmd': command_name,
                        'frame_idx': latest_frame_idx,
                        'captures': saved,
                    }
                elif command_name == 'set_runtime_pause':
                    pause_result = handle_set_runtime_pause(payload)
                    result_payload = {
                        **meta,
                        'status': 'done',
                        'cmd': command_name,
                        **pause_result,
                    }
                else:
                    raise ValueError(f'unsupported command: {command_name or "empty"}')
                write_json_atomic(_command_result_path(command_path, 'done'), result_payload)
            except Exception as exc:
                write_json_atomic(
                    _command_result_path(command_path, 'failed'),
                    {
                        **meta,
                        'status': 'failed',
                        'reason': str(exc),
                    },
                )
            finally:
                try:
                    command_path.unlink()
                except Exception:
                    pass

    storage_cleaner.run_once(reason='startup')
    write_heartbeat(status='starting', force=True)

    startup_heartbeat_stop = threading.Event()

    def startup_heartbeat_loop():
        while not startup_heartbeat_stop.wait(heartbeat_interval_seconds):
            write_startup_heartbeat(stage='booting')

    core_mask = parse_core_mask(args.core_mask)
    plate_core_mask = parse_core_mask(getattr(args, 'plate_core_mask', None))
    worker_core_strategy = str(video_cfg.get('worker_core_strategy', 'auto') or 'auto').strip().lower()
    worker_core_masks = resolve_worker_core_masks(core_mask, args.workers, strategy=worker_core_strategy)
    plate_core_mask = resolve_auto_plate_core_mask(plate_core_mask, core_mask, worker_core_masks)
    print(
        f'[npu] main_core_mask={core_mask} '
        f'worker_core_masks={worker_core_masks} '
        f'plate_core_mask={plate_core_mask} '
        f'worker_core_strategy={worker_core_strategy} '
        f'main_core_indices={core_mask_to_indices(core_mask)}'
    )
    print(f'[npu-status] {format_npu_status(snapshot_npu_status())}')
    write_startup_heartbeat(stage='before_worker_init')
    startup_heartbeat_thread = threading.Thread(target=startup_heartbeat_loop, daemon=True)
    startup_heartbeat_thread.start()
    workers = []
    try:
        for i in range(args.workers):
            worker = DetectWorker(
                i,
                args,
                worker_core_masks[i] if i < len(worker_core_masks) else core_mask,
                task_q,
                result_q,
                plate_core_mask=plate_core_mask,
            )
            workers.append(worker)
            write_startup_heartbeat(stage=f'worker_{i}_ready')
        for w in workers:
            w.start()
        write_startup_heartbeat(stage='workers_started')
    finally:
        startup_heartbeat_stop.set()
        startup_heartbeat_thread.join(timeout=0.2)

    finished_workers = 0
    worker_last_frames = [0 for _ in workers]
    worker_last_infer = [0.0 for _ in workers]
    wheel_last_stats = {}
    reader_log_last_dropped = 0
    reader_log_last_reconnect_count = 0
    last_perf_snapshot = {}
    recent_drain_times = deque(maxlen=25)
    last_event_saved_count = 0

    def _snapshot_per_id_video_metrics():
        stats = {
            'enabled': bool(enable_per_id_video),
            'active_writers': len(per_id_writers),
            'queued': 0,
            'queue_size_total': 0,
            'enqueued': 0,
            'written': 0,
            'dropped': 0,
            'write_errors': 0,
            'writers': [],
        }
        for tid, writer in list(per_id_writers.items()):
            snap_fn = getattr(writer, 'snapshot_stats', None)
            if callable(snap_fn):
                try:
                    item = snap_fn()
                except Exception:
                    item = {}
            else:
                item = {
                    'backend': getattr(writer, 'backend', ''),
                    'encoder': getattr(writer, 'encoder', ''),
                    'path': getattr(writer, 'path', ''),
                }
            item['track_id'] = tid
            stats['writers'].append(item)
            for key in ('queued', 'queue_size', 'enqueued', 'written', 'dropped', 'write_errors'):
                try:
                    value = int(item.get(key, 0) or 0)
                except Exception:
                    value = 0
                if key == 'queue_size':
                    stats['queue_size_total'] += value
                else:
                    stats[key] += value
        return stats

    def _upload_pending_count(uploader_like):
        db = getattr(uploader_like, 'db', None)
        if db is None:
            return 0
        pending_fn = getattr(db, 'pending', None)
        if not callable(pending_fn):
            return 0
        try:
            return int(pending_fn())
        except Exception:
            return -1

    def _write_metrics(status='running', perf_snapshot=None):
        if not metrics_path:
            return
        payload = {
            'timestamp': time.time(),
            'pid': os.getpid(),
            'status': status,
            'launch_id': launch_id,
            'runtime_namespace_key': runtime_namespace_key,
            'device_id': runtime_device_id,
            'config_path': config_path,
            'config_name': config_name,
            'source': str(path),
            'source_mode': source_mode,
            'decode': {
                'mode': str((decode_meta or {}).get('decode_mode') or ''),
                'backend': str((decode_meta or {}).get('decode_backend') or ''),
                'fallback_used': bool((decode_meta or {}).get('fallback_used')),
                'fallback_reason': str((decode_meta or {}).get('fallback_reason') or ''),
            },
            'frames': {
                'total_read': total_frames,
                'next_result': next_frame_to_write,
                'latest_frame_idx': latest_frame_idx,
                'dropped': dropped_frame_count,
                'dropped_pending': len(dropped_frame_ids),
            },
            'queues': {
                'task': _safe_qsize(task_q),
                'result': _safe_qsize(result_q),
                'pending_results': len(pending),
            },
            'perf': perf_snapshot or last_perf_snapshot,
            'per_id_video': _snapshot_per_id_video_metrics(),
            'events': event_manager.snapshot_metrics() if hasattr(event_manager, 'snapshot_metrics') else {},
            'wheel': wheel_service.snapshot_stats() if wheel_service is not None else {},
            'process': {
                'rss_kb': _process_rss_kb(),
            },
        }
        try:
            write_json_atomic(metrics_path, payload)
        except Exception:
            pass

    monitor_stop = threading.Event()
    monitor_thread = None
    if args.monitor_interval > 0:
        monitor_thread = threading.Thread(target=monitor_loop, args=(args.monitor_interval, monitor_stop), daemon=True)
        monitor_thread.start()

    def close_per_id_writer(track_id, track_state):
        lifecycle = event_manager.lifecycle_manager.get(track_id)
        writer_key = lifecycle.event_id if lifecycle is not None else track_state.get('session_id', track_id)
        writer = per_id_writers.pop(writer_key, None)
        if writer is None:
            return False
        return finalize_per_id_recording(
            writer,
            track_id,
            track_state,
            event_manager,
            per_id_video_enabled=True,
        )

    def finalize_per_id_for_track(track_id, track_state):
        emitted = close_per_id_writer(track_id, track_state)
        if emitted:
            return
        if not enable_per_id_video:
            emit_per_id_video_type6(
                track_id,
                track_state,
                event_manager,
                per_id_video_enabled=False,
            )

    cleanup_done = False

    def cleanup_runtime():
        nonlocal cleanup_done
        if cleanup_done:
            return
        cleanup_done = True

        if csv_f:
            try:
                csv_f.close()
            except Exception:
                pass
        if wheel_service is not None:
            try:
                wheel_service.stop()
            except Exception:
                pass
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        monitor_stop.set()
        if monitor_thread:
            try:
                monitor_thread.join(timeout=0.5)
            except Exception:
                pass
        try:
            event_manager.flush_inactive(
                set(),
                total_frames + int(config.get('track_timeout_frames', 60)) + 1,
                finalize_per_id_for_track,
                timeout_reason='file_eof' if is_file_input else 'runtime_stop',
            )
        except Exception:
            pass
        if enable_per_id_video and per_id_writers:
            for writer_key in list(per_id_writers.keys()):
                track_item = next(
                    (
                        (tid, state) for tid, state in event_manager.tracks.items()
                        if state.get('session_id') == writer_key
                    ),
                    None,
                )
                if track_item is None:
                    writer = per_id_writers.pop(writer_key, None)
                    if writer is not None:
                        writer.release()
                    continue
                close_per_id_writer(*track_item)
        elif not enable_per_id_video:
            for tid, track_state in list(event_manager.tracks.items()):
                emit_per_id_video_type6(
                    tid,
                    track_state,
                    event_manager,
                    per_id_video_enabled=False,
                )
        try:
            cleanup_alias_confirm(total_frames + alias_timeout + 1)
        except Exception:
            pass
        try:
            event_manager.close()
        except Exception:
            pass
        if uploader:
            try:
                uploader.close()
            except Exception:
                pass
        if wheel_photo_uploader:
            try:
                wheel_photo_uploader.close()
            except Exception:
                pass

    atexit.register(cleanup_runtime)

    def mark_alias_confirm(alias_id, has_plate_text, frame_idx, require_text):
        if alias_id <= 0:
            return False
        state = alias_confirm.setdefault(alias_id, {'frames': 0, 'confirmed': False, 'last_seen': frame_idx})
        state['frames'] += 1
        state['last_seen'] = frame_idx
        if not state['confirmed']:
            if has_plate_text:
                state['confirmed'] = True
            elif not require_text and state['frames'] >= REPORT_MIN_FRAMES:
                state['confirmed'] = True
        return state['confirmed']

    def cleanup_alias_confirm(frame_idx):
        for aid in list(alias_confirm.keys()):
            state = alias_confirm[aid]
            if (frame_idx - state.get('last_seen', frame_idx)) > alias_timeout or aid not in event_manager.tracks:
                alias_confirm.pop(aid, None)

    def annotate_locked_label(track_id, det_ref, rows_ref, frame_img):
        if det_ref is None or det_ref.get('cls') not in VEHICLE_CLASS_IDS:
            return
        locked = event_manager.get_locked_vehicle(track_id)
        row_idx = det_ref.get('row_idx', -1)
        if row_idx is not None and 0 <= row_idx < len(rows_ref):
            if locked:
                rows_ref[row_idx][1] = locked
        label_now = det_ref.get('label')
        mismatch = bool(locked and label_now and locked != label_now)
        if mismatch:
            return
        if not args.no_draw and frame_img is not None:
            x1, y1, x2, y2 = det_ref['box']
            color = select_box_color(label_now or locked or '')
            cv2.rectangle(frame_img, (x1, y1), (x2, y2), color, 2)
            disp_label = locked or label_now or ''
            if disp_label:
                text_cn = localize_vehicle(disp_label)
                draw_text(
                    frame_img,
                    text_cn,
                    (x1, max(0, y1 - 18)),
                    font_scale=0.75,
                    color=(255, 255, 255),
                    thickness=2,
                    anchor='lb',
                )

    def apply_stable_plate_fields(track_id, det_ref, rows_ref):
        if det_ref is None or det_ref.get('cls') != LICENSE_CLASS:
            return
        raw_text = str(det_ref.get('raw_text', det_ref.get('text', '')) or '').strip()
        det_ref['raw_text'] = raw_text
        locked_info = event_manager.get_locked_plate(track_id) if track_id and track_id > 0 else {}
        locked_text = str((locked_info or {}).get('text') or '').strip()
        locked_color = str((locked_info or {}).get('plate_color') or '').strip()
        locked_color_conf = (locked_info or {}).get('plate_color_conf')
        if plate_draw_stable_only or locked_text:
            det_ref['text'] = locked_text
            det_ref['plate_color'] = locked_color
            det_ref['plate_color_conf'] = locked_color_conf if locked_color else None
        row_idx = det_ref.get('row_idx', -1)
        if row_idx is not None and 0 <= row_idx < len(rows_ref):
            rows_ref[row_idx][-2] = str(det_ref.get('text') or '')
            rows_ref[row_idx][-1] = raw_text

    def draw_stable_plate_overlay(frame_img, det_items):
        if args.no_draw or frame_img is None or not draw_plate_boxes:
            return
        for det in det_items or []:
            if det.get('cls') != LICENSE_CLASS:
                continue
            box = det.get('box')
            if not box or len(box) != 4:
                continue
            try:
                x1, y1, x2, y2 = [int(round(v)) for v in box]
            except Exception:
                continue
            label_name = CLASS_NAMES[LICENSE_CLASS]
            score = det.get('score')
            plate_text = str(det.get('text') or '').strip()
            plate_color = str(det.get('plate_color') or '').strip()
            label = plate_text or label_name
            if plate_color and plate_text:
                label = f'{label} {plate_color}'
            try:
                if score is not None:
                    label = f'{label} {float(score):.2f}'
            except Exception:
                pass
            color = select_box_color(label_name)
            cv2.rectangle(frame_img, (x1, y1), (x2, y2), color, 2)
            draw_text(
                frame_img,
                label,
                (x1, max(0, y1 - 12)),
                font_scale=0.65,
                color=(255, 255, 255),
                thickness=2,
                anchor='lb',
            )
            for pt in det.get('landmarks', []) or []:
                if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                    continue
                cv2.circle(
                    frame_img,
                    (int(round(pt[0])), int(round(pt[1]))),
                    3,
                    (0, 255, 255),
                    -1,
                )

    def draw_debug_detections(frame_img, det_items):
        if frame_img is None:
            return
        for det in det_items or []:
            box = det.get('box')
            if not box or len(box) != 4:
                continue
            try:
                cls_id = int(det.get('cls', -1))
                x1, y1, x2, y2 = [int(round(v)) for v in box]
            except Exception:
                continue
            if cls_id < 0 or cls_id >= len(CLASS_NAMES):
                continue
            label_name = str(det.get('label') or CLASS_NAMES[cls_id])
            color = select_box_color(label_name)
            label = label_name
            track_id = _normalize_track_id(det.get('track_id'))
            if cls_id in VEHICLE_CLASS_IDS:
                locked = event_manager.get_locked_vehicle(track_id) if track_id else ''
                label = localize_vehicle(locked or label_name)
                if track_id:
                    label = f"ID:{track_id} {label}"
            elif cls_id == LICENSE_CLASS:
                if not draw_plate_boxes:
                    continue
                plate_text = str(det.get('text') or '').strip()
                label = plate_text or 'license'
            elif cls_id in WATER_CLASS_IDS:
                label = localize_cleaning(label_name)
            try:
                score = det.get('score')
                if score is not None:
                    label = f"{label} {float(score):.2f}"
            except Exception:
                pass
            cv2.rectangle(frame_img, (x1, y1), (x2, y2), color, 2)
            draw_text(
                frame_img,
                label,
                (x1, max(0, y1 - 12)),
                font_scale=0.65,
                color=(255, 255, 255),
                thickness=2,
                anchor='lb',
            )
            if cls_id == LICENSE_CLASS:
                for pt in det.get('landmarks', []) or []:
                    if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                        continue
                    cv2.circle(
                        frame_img,
                        (int(round(pt[0])), int(round(pt[1]))),
                        3,
                        (0, 255, 255),
                        -1,
                    )

    def draw_anchor_comparison_overlay(frame_img, vehicle_refs, frame_idx):
        if frame_img is None or not (debug_anchor_points or anchor_estimator.shadow_compare):
            return
        for det_ref in vehicle_refs or []:
            track_id = _normalize_track_id(det_ref.get('track_id'))
            box = det_ref.get('box')
            if track_id is None or not box or len(box) != 4:
                continue
            result = anchor_result_for(track_id, box, frame_idx)
            if result is None:
                continue
            history = anchor_estimator.get_history(track_id)
            legacy_history = np.array(
                [[int(round(item[1][0])), int(round(item[1][1]))] for item in history],
                dtype=np.int32,
            )
            neutral_history = np.array(
                [[int(round(item[2][0])), int(round(item[2][1]))] for item in history],
                dtype=np.int32,
            )
            directional_history = np.array(
                [[int(round(item[3][0])), int(round(item[3][1]))] for item in history],
                dtype=np.int32,
            )
            if len(legacy_history) >= 2:
                cv2.polylines(frame_img, [legacy_history], False, (0, 140, 255), 2, cv2.LINE_AA)
            if len(neutral_history) >= 2:
                cv2.polylines(frame_img, [neutral_history], False, (0, 255, 255), 2, cv2.LINE_AA)
            if len(directional_history) >= 2:
                cv2.polylines(frame_img, [directional_history], False, (255, 255, 0), 2, cv2.LINE_AA)
            lx, ly = [int(round(value)) for value in result.legacy_point]
            hx, hy = [int(round(value)) for value in result.neutral_point]
            dx, dy = [int(round(value)) for value in result.directional_point]
            sx, sy = [int(round(value)) for value in result.selected_point]
            cv2.line(frame_img, (lx, ly), (hx, hy), (180, 180, 180), 1, cv2.LINE_AA)
            cv2.line(frame_img, (hx, hy), (dx, dy), (180, 180, 180), 1, cv2.LINE_AA)
            cv2.circle(frame_img, (lx, ly), 6, (0, 140, 255), -1)
            cv2.circle(frame_img, (hx, hy), 6, (0, 255, 255), -1)
            cv2.circle(frame_img, (dx, dy), 6, (255, 255, 0), -1)
            cv2.circle(frame_img, (sx, sy), 9, (255, 255, 255), 2)
            cv2.putText(frame_img, f'L{track_id}', (lx + 7, ly - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 140, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_img, f'H{track_id}', (hx + 7, hy - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_img, f'D{track_id}', (dx + 7, dy + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 2, cv2.LINE_AA)
            edge_text = '/'.join(result.truncated_edges) if result.truncated_edges else '-'
            status_text = (
                f'anchor={result.mode} dir={result.motion_direction} '
                f'conf={result.direction_confidence:.2f} blend={result.direction_blend:.2f} '
                f'edge={edge_text}'
            )
            x1, y1, _, _ = [int(round(value)) for value in box]
            cv2.putText(frame_img, status_text, (x1, max(18, y1 - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

    def draw_debug_track_state(frame_img, vehicle_refs):
        if not debug_tracks or frame_img is None:
            return
        for det_ref in vehicle_refs or []:
            car_id = det_ref.get('track_id', -1)
            if car_id <= 0:
                continue
            info = event_manager.get_track_debug(car_id)
            if not info:
                continue
            x1, y1, _, _ = det_ref['box']
            text = (f"ID:{car_id} {info['state']} sf:{info['stationary']} "
                    f"spd:{info['speed']:.1f} water:{'Y' if info['water'] else 'N'} "
                    f"dur:{info['wash_duration']:.1f} zb:{info.get('zone_b_elapsed',0)}")
            cv2.putText(frame_img, text, (x1, max(0, y1 - 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

    def draw_debug_rois(frame_img):
        if not debug_rois or frame_img is None:
            return
        if len(zone_a_pts) >= 3:
            roi_np = np.array(zone_a_pts, dtype=np.int32)
            cv2.polylines(frame_img, [roi_np], True, (0, 200, 0), 2, cv2.LINE_AA)
            anchor = tuple(map(int, roi_np[0]))
            cv2.putText(frame_img, 'Zone A', (anchor[0], max(0, anchor[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2, cv2.LINE_AA)
        if len(zone_b_pts) >= 3:
            roi_np = np.array(zone_b_pts, dtype=np.int32)
            cv2.polylines(frame_img, [roi_np], True, (0, 0, 255), 2, cv2.LINE_AA)
            anchor = tuple(map(int, roi_np[0]))
            cv2.putText(frame_img, 'Zone B', (anchor[0], max(0, anchor[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        flow_start_int = tuple(map(int, flow_start))
        flow_end_int = tuple(map(int, flow_end))
        cv2.arrowedLine(frame_img, flow_start_int, flow_end_int, (255, 0, 0), 2, tipLength=0.08)

    def save_debug_frame(frame_img, det_items, vehicle_refs, frame_capture_ts, frame_idx):
        if not debug_frame_file or frame_img is None:
            return
        try:
            debug_frame = frame_img.copy()
            draw_debug_detections(debug_frame, det_items)
            draw_anchor_comparison_overlay(debug_frame, vehicle_refs, frame_idx)
            draw_debug_track_state(debug_frame, vehicle_refs)
            draw_debug_rois(debug_frame)
            ts_now = datetime.now()
            ts_str = ts_now.strftime("%Y-%m-%d %H:%M:%S")
            latency_ms = None
            if frame_capture_ts is not None:
                try:
                    latency_ms = int((ts_now.timestamp() - float(frame_capture_ts)) * 1000.0)
                except Exception:
                    latency_ms = None
            text = ts_str
            if latency_ms is not None and latency_ms >= 0:
                text = f"{ts_str} Δ{latency_ms}ms"
            h_dbg, w_dbg = debug_frame.shape[:2]
            margin = 10
            base = min(w_dbg, h_dbg)
            scale = max(0.5, base / 960.0 * 0.7)
            thick_outline = max(2, int(scale * 3))
            thick_text = max(1, int(scale * 1.5))
            cv2.putText(
                debug_frame,
                text,
                (margin, h_dbg - margin),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (0, 0, 0),
                thick_outline,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug_frame,
                text,
                (margin, h_dbg - margin),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (255, 255, 255),
                thick_text,
                cv2.LINE_AA,
            )
            if debug_frame_max_width > 0 and debug_frame.shape[1] > debug_frame_max_width:
                ratio = debug_frame_max_width / float(debug_frame.shape[1])
                target_h = max(1, int(round(debug_frame.shape[0] * ratio)))
                debug_frame = resize_bgr(debug_frame, (debug_frame_max_width, target_h))
            cv2.imwrite(str(debug_frame_file), debug_frame, [int(cv2.IMWRITE_JPEG_QUALITY), debug_frame_quality])
        except Exception:
            pass

    def ensure_benchmark_recording_track(frame_idx, frame_capture_ts):
        if benchmark_force_recording_track_id <= 0:
            return
        track_id = benchmark_force_recording_track_id
        track_state = event_manager.tracks.setdefault(track_id, {})
        track_state.setdefault('events', set())
        track_state.setdefault('record_start_frame', 0)
        track_state.setdefault('record_stop_frame', None)
        track_state.setdefault('type2_qualified', True)
        track_state.setdefault('type2_qualified_frame', 0)
        track_state.setdefault('last_frame_idx', frame_idx)
        if frame_capture_ts is not None and not track_state.get('type1_capture_time'):
            try:
                track_state['type1_capture_time'] = datetime.fromtimestamp(float(frame_capture_ts)).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                track_state['type1_capture_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not track_state.get('session_id'):
            base_time = track_state.get('type1_capture_time') or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            try:
                dt = datetime.strptime(base_time, "%Y-%m-%d %H:%M:%S")
            except Exception:
                dt = datetime.now()
            ts_str = dt.strftime("%Y%m%d%H%M")
            device_name = config.get('system', {}).get('device_id') or event_manager.camera_id
            track_state['session_id'] = f"{device_name}-{ts_str}-{track_id}"

    def maybe_force_benchmark_capture(frame_idx, frame_img):
        if benchmark_force_capture_stride <= 0 or frame_img is None:
            return
        if frame_idx % benchmark_force_capture_stride != 0:
            return
        try:
            event_manager._save_event_capture(98, benchmark_force_recording_track_id or 999999, frame_idx, frame_img)
        except Exception:
            pass

    def drain_results(block=True):
        nonlocal next_frame_to_write, finished_workers, car_plate_cache, plate_binding_states, per_id_writers
        nonlocal enable_per_id_video, latest_raw_frame, latest_annotated_frame, dropped_frame_ids
        nonlocal latest_frame_idx, latest_capture_ts, last_progress_ts, last_result_ts
        nonlocal last_event_saved_count, recent_drain_times
        try:
            item = result_q.get(block=block, timeout=1 if block else 0)
        except Empty:
            return False
        if item is None:
            finished_workers += 1
        else:
            if len(item) == 4:
                idx, frame_out, rows, det_payload = item
                capture_ts = None
            else:
                idx, capture_ts, frame_out, rows, det_payload = item
            pending[idx] = (frame_out, rows, det_payload, capture_ts)
            _advance_dropped_frames()
            while next_frame_to_write in pending:
                drain_t0 = time.perf_counter()
                frame_out, rows, det_payload, capture_ts = pending.pop(next_frame_to_write)
                raw_frame_for_idx = raw_frame_cache.pop(next_frame_to_write, None)
                if capture_ts is not None:
                    try:
                        event_manager.record_frame_timing(next_frame_to_write, capture_ts, time.time())
                    except Exception:
                        pass
                vehicle_dets = []
                vehicle_payload_refs = []
                car_boxes = {}
                car_labels = {}
                water_boxes = []
                cleaning_label = ''
                if det_payload:
                    for det in det_payload:
                        if det.get('cls') in VEHICLE_CLASS_IDS:
                            vehicle_dets.append({'box': det['box'], 'score': det['score'], 'cls': det['cls'], 'row_idx': det.get('row_idx')})
                            vehicle_payload_refs.append(det)
                        elif det.get('cls') in WATER_CLASS_IDS:
                            water_boxes.append(det['box'])
                            name = CLASS_NAMES[det['cls']]
                            if name == 'manual':
                                cleaning_label = 'manual'
                            elif not cleaning_label:
                                cleaning_label = name
                if debug_water_boxes and not args.no_draw and water_boxes and frame_out is not None:
                    for wb in water_boxes:
                        wx1, wy1, wx2, wy2 = wb
                        cv2.rectangle(frame_out, (wx1, wy1), (wx2, wy2), CLASS_COLORS.get('water', (0, 160, 255)), 2)
                        cv2.putText(frame_out, 'Water', (wx1, max(0, wy1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 160, 255), 1, cv2.LINE_AA)
                event_frame_for_idx = (
                    frame_out
                    if event_use_annotated_frame and frame_out is not None
                    else (raw_frame_for_idx if raw_frame_for_idx is not None else frame_out)
                )
                assignments = vehicle_tracker.update(next_frame_to_write, vehicle_dets)
                retained_track_ids = set(vehicle_tracker.get_retained_track_ids())
                event_manager.trace_record('frame_tracks', {
                    'frameIdx': int(next_frame_to_write),
                    'captureTs': capture_ts,
                    'vehicleDetections': vehicle_dets,
                    'assignments': assignments,
                    'retainedTrackIds': sorted(retained_track_ids),
                    'waterBoxes': water_boxes,
                })
                for det_ref, track_id in zip(vehicle_payload_refs, assignments):
                    det_ref['track_id'] = track_id
                    car_boxes[track_id] = det_ref['box']
                    car_labels[track_id] = det_ref.get('label', '')
                    row_idx = det_ref.get('row_idx', -1)
                    if row_idx is not None and 0 <= row_idx < len(rows):
                        rows[row_idx][7] = track_id
                active_car_ids = {det_ref.get('track_id', -1) for det_ref in vehicle_payload_refs if det_ref.get('track_id', -1) > 0}
                _cleanup_pending_plate_cache(
                    pending_plate_cache,
                    next_frame_to_write,
                    pending_plate_cache_ttl_frames,
                )
                license_dets = [
                    detection for detection in det_payload
                    if detection.get('cls') == LICENSE_CLASS and detection.get('source') == 'dual_plate'
                ] if det_payload else []
                primary_plate_dets = [
                    detection for detection in det_payload
                    if detection.get('cls') == LICENSE_CLASS and detection.get('source') == 'primary_plate_aux'
                ] if det_payload else []
                if license_dets and car_boxes:
                    for det in license_dets:
                        best_id = None
                        best_score = 0.0
                        plate_box = det['box']
                        for car_id, cbox in car_boxes.items():
                            score = _plate_car_match_score(plate_box, cbox, frame_size=(width, height))
                            if score > best_score:
                                best_score = score
                                best_id = car_id
                        if best_id is not None and best_score >= PLATE_CAR_LINK_IOU:
                            det['candidate_car_track'] = best_id
                            det['candidate_score'] = float(best_score)
                            det['vehicle_box_candidate'] = car_boxes[best_id]
                        mutual_score = max(
                            (_plate_box_mutual_score(det['box'], primary_det['box']) for primary_det in primary_plate_dets),
                            default=0.0,
                        )
                        det['primary_plate_mutual_score'] = float(mutual_score)
                        det['primary_plate_verified'] = bool(mutual_score >= 0.20)
                plate_track_info = {}
                updates = plate_tracker.update(next_frame_to_write, license_dets)
                for det, upd in zip(license_dets, updates):
                    plate_id = upd.get('track_id', -1)
                    text_val = upd.get('text', '')
                    text_conf_val = upd.get('plate_text_conf')
                    is_guess = bool(upd.get('is_guess', False))
                    det['raw_text'] = str(det.get('raw_text', det.get('text', '')) or '')
                    row_idx = det.get('row_idx', -1)
                    if row_idx is not None and 0 <= row_idx < len(rows):
                        rows[row_idx][-1] = det.get('raw_text', '')
                        if text_val and not plate_draw_stable_only:
                            rows[row_idx][-2] = text_val
                    if plate_id > 0:
                        state = _ensure_plate_binding_state(
                            next_frame_to_write,
                            plate_binding_states.get(plate_id),
                        )
                        plate_binding_states[plate_id] = state
                        locked_car_id = _normalize_track_id(state.get('locked_car_id'))
                        locked_score = None
                        if locked_car_id is not None and locked_car_id in car_boxes:
                            locked_score = _plate_car_match_score(
                                det['box'], car_boxes[locked_car_id], frame_size=(width, height)
                            )
                        _stabilize_plate_binding(
                            state,
                            frame_idx=next_frame_to_write,
                            candidate_car_id=det.get('candidate_car_track', -1),
                            candidate_score=float(det.get('candidate_score', 0.0)),
                            locked_score=locked_score,
                        )
                        event_manager.trace_record('plate_binding', {
                            'frameIdx': int(next_frame_to_write),
                            'plateTrackId': int(plate_id),
                            'candidateCarTrackId': det.get('candidate_car_track', -1),
                            'candidateScore': float(det.get('candidate_score', 0.0) or 0.0),
                            'lockedCarTrackId': locked_car_id,
                            'lockedScore': locked_score,
                            'plateBox': det.get('box'),
                            'plateText': text_val,
                        })
                        locked_car_id = _normalize_track_id(state.get('locked_car_id'))
                        resolved_track = locked_car_id if locked_car_id is not None else plate_id
                        if row_idx is not None and 0 <= row_idx < len(rows):
                            rows[row_idx][7] = resolved_track
                        vehicle_box = None
                        if locked_car_id is not None and locked_car_id in car_boxes:
                            vehicle_box = car_boxes[locked_car_id]
                        elif det.get('vehicle_box_candidate') is not None:
                            vehicle_box = det.get('vehicle_box_candidate')
                        if locked_car_id is not None and vehicle_box is not None:
                            car_boxes.setdefault(locked_car_id, vehicle_box)
                        plate_candidate_history = []
                        if locked_car_id is not None:
                            plate_candidate_history = _consume_pending_plate_candidates(
                                pending_plate_cache,
                                plate_id,
                                next_frame_to_write,
                                pending_plate_cache_ttl_frames,
                            )
                        else:
                            raw_text = str(det.get('raw_text', det.get('text', '')) or '')
                            candidate_text = raw_text
                            if not is_valid_plate(normalize_plate_candidate_text(candidate_text)):
                                candidate_text = text_val if text_val and not is_guess else ''
                            _append_pending_plate_candidate(
                                pending_plate_cache,
                                plate_id,
                                candidate_text,
                                next_frame_to_write,
                                conf=det.get('score'),
                                box=det.get('box'),
                                plate_color=det.get('plate_color', ''),
                                plate_color_conf=det.get('plate_color_conf'),
                                plate_type=det.get('plate_type', ''),
                                text_conf=det.get('plate_text_conf'),
                                mutual_verified=bool(det.get('primary_plate_verified', False)),
                                trusted=False,
                                max_entries=pending_plate_cache_max_entries,
                            )
                        plate_track_info[plate_id] = {
                            'box': det['box'],
                            'text': text_val,
                            'text_conf': text_conf_val,
                            'primary_plate_verified': bool(det.get('primary_plate_verified', False)),
                            'primary_plate_mutual_score': float(det.get('primary_plate_mutual_score', 0.0) or 0.0),
                            'is_guess': is_guess,
                            'vehicle_box': vehicle_box,
                            'score': float(det.get('score', 0.0)),
                            'plate_color': det.get('plate_color', ''),
                            'plate_color_conf': det.get('plate_color_conf'),
                            'plate_type': det.get('plate_type', ''),
                            'car_id': locked_car_id if locked_car_id is not None else -1,
                            'plate_candidate_history': plate_candidate_history,
                            'det_ref': det,
                        }
                removed_locked_ids = _cleanup_plate_binding_states(
                    plate_binding_states=plate_binding_states,
                    frame_idx=next_frame_to_write,
                    active_car_ids=active_car_ids,
                    plate_timeout_frames=plate_binding_timeout_frames,
                    vehicle_missing_frames=plate_binding_vehicle_missing_frames,
                )
                for removed_car_id in removed_locked_ids:
                    car_plate_cache.pop(removed_car_id, None)
                car_to_plate = _refresh_car_plate_cache_from_locked(
                    plate_binding_states=plate_binding_states,
                    car_plate_cache=car_plate_cache,
                    active_car_ids=active_car_ids,
                    car_plate_cache_ttl=car_plate_cache_ttl,
                )
                plate_to_car = {plate_id: car_id for car_id, plate_id in car_to_plate.items()}
                alias_seen = set()
                plates_with_updates = set()
                t_before_updates = time.perf_counter()
                for plate_id, info in plate_track_info.items():
                    car_id = plate_to_car.get(plate_id, info.get('car_id', -1))
                    if car_id <= 0:
                        det_ref = info.get('det_ref')
                        if det_ref is not None:
                            det_ref['track_id'] = -1
                        continue
                    track_key = car_id if car_id > 0 else plate_id
                    vehicle_box = info.get('vehicle_box')
                    if vehicle_box is None:
                        if car_id > 0 and car_id in car_boxes:
                            vehicle_box = car_boxes[car_id]
                    anchor_pt, anchor_direction = anchor_evidence_for(
                        track_key,
                        vehicle_box or info['box'],
                        next_frame_to_write,
                    )
                    confirmed_alias = mark_alias_confirm(track_key, bool(info.get('text')), next_frame_to_write, True)
                    event_manager.update_track(
                        track_key,
                        info['box'],
                        vehicle_box,
                        info.get('text', ''),
                        next_frame_to_write,
                        event_frame_for_idx,
                        water_boxes,
                        bool(water_boxes),
                        is_plate=True,
                        vehicle_label=car_labels.get(track_key, ''),
                        vehicle_conf=None,
                        plate_conf=info.get('score'),
                        plate_text_conf=info.get('text_conf'),
                        plate_mutual_verified=bool(info.get('primary_plate_verified', False)),
                        confirmed=confirmed_alias,
                        cleaning_label=cleaning_label,
                        anchor_point=anchor_pt,
                        plate_is_guess=bool(info.get('is_guess', False)),
                        plate_color=info.get('plate_color', ''),
                        plate_color_conf=info.get('plate_color_conf'),
                        plate_type=info.get('plate_type', ''),
                        plate_candidate_history=info.get('plate_candidate_history'),
                        anchor_direction=anchor_direction,
                    )
                    det_ref = info.get('det_ref')
                    if det_ref is not None:
                        det_ref['track_id'] = track_key
                        apply_stable_plate_fields(track_key, det_ref, rows)
                    alias_seen.add(track_key)
                    plates_with_updates.add(track_key)
                for det_ref in vehicle_payload_refs:
                    car_id = det_ref.get('track_id', -1)
                    if car_id <= 0:
                        continue
                    alias_plate_id = car_to_plate.get(car_id)
                    if alias_plate_id:
                        row_idx = det_ref.get('row_idx', -1)
                        if row_idx is not None and 0 <= row_idx < len(rows):
                            rows[row_idx][7] = car_id
                        if car_id not in plates_with_updates:
                            known_text = ''
                            track_state = event_manager.tracks.get(car_id)
                            if track_state:
                                known_text = track_state.get('plate_text', '')
                            confirmed_alias = mark_alias_confirm(car_id, bool(known_text), next_frame_to_write, True)
                            anchor_pt, anchor_direction = anchor_evidence_for(
                                car_id,
                                det_ref['box'],
                                next_frame_to_write,
                            )
                            plate_candidate_history = _consume_pending_plate_candidates(
                                pending_plate_cache,
                                alias_plate_id,
                                next_frame_to_write,
                                pending_plate_cache_ttl_frames,
                            )
                            event_manager.update_track(
                                car_id,
                                None,
                                det_ref['box'],
                                '',
                                next_frame_to_write,
                                event_frame_for_idx,
                                water_boxes,
                                bool(water_boxes),
                                is_plate=True,
                                vehicle_label=det_ref.get('label', ''),
                                vehicle_conf=det_ref.get('score'),
                                plate_conf=None,
                                confirmed=confirmed_alias,
                                cleaning_label=cleaning_label,
                                anchor_point=anchor_pt,
                                plate_candidate_history=plate_candidate_history,
                                anchor_direction=anchor_direction,
                            )
                        annotate_locked_label(car_id, det_ref, rows, frame_out)
                        alias_seen.add(car_id)
                        continue
                    cache_entry = car_plate_cache.get(car_id)
                    if cache_entry and cache_entry.get('age', 0) <= car_plate_cache_ttl:
                        cache_entry['age'] = cache_entry.get('age', 0) + 1
                        row_idx = det_ref.get('row_idx', -1)
                        if row_idx is not None and 0 <= row_idx < len(rows):
                            rows[row_idx][7] = car_id
                        known_text = ''
                        track_state = event_manager.tracks.get(car_id)
                        if track_state:
                            known_text = track_state.get('plate_text', '')
                        confirmed_alias = mark_alias_confirm(car_id, bool(known_text), next_frame_to_write, True)
                        anchor_pt, anchor_direction = anchor_evidence_for(
                            car_id,
                            det_ref['box'],
                            next_frame_to_write,
                        )
                        plate_candidate_history = _consume_pending_plate_candidates(
                            pending_plate_cache,
                            cache_entry.get('plate_id'),
                            next_frame_to_write,
                            pending_plate_cache_ttl_frames,
                        )
                        event_manager.update_track(
                            car_id,
                            None,
                            det_ref['box'],
                            '',
                            next_frame_to_write,
                            event_frame_for_idx,
                            water_boxes,
                            bool(water_boxes),
                            is_plate=True,
                            vehicle_label=det_ref.get('label', ''),
                            vehicle_conf=det_ref.get('score'),
                            plate_conf=None,
                            confirmed=confirmed_alias,
                            cleaning_label=cleaning_label,
                            anchor_point=anchor_pt,
                            plate_candidate_history=plate_candidate_history,
                            anchor_direction=anchor_direction,
                        )
                        alias_seen.add(car_id)
                        continue
                    fallback_id = car_id
                    row_idx = det_ref.get('row_idx', -1)
                    if row_idx is not None and 0 <= row_idx < len(rows):
                        rows[row_idx][7] = fallback_id
                    confirmed_alias = mark_alias_confirm(fallback_id, False, next_frame_to_write, True)
                    anchor_pt, anchor_direction = anchor_evidence_for(
                        fallback_id,
                        det_ref['box'],
                        next_frame_to_write,
                    )
                    event_manager.update_track(
                        fallback_id,
                        None,
                        det_ref['box'],
                        '',
                        next_frame_to_write,
                        event_frame_for_idx,
                        water_boxes,
                        bool(water_boxes),
                        is_plate=False,
                        vehicle_label=det_ref.get('label', ''),
                        vehicle_conf=det_ref.get('score'),
                        plate_conf=None,
                        confirmed=confirmed_alias,
                        cleaning_label=cleaning_label,
                        anchor_point=anchor_pt,
                        anchor_direction=anchor_direction,
                    )
                    annotate_locked_label(fallback_id, det_ref, rows, frame_out)
                    alias_seen.add(fallback_id)
                draw_stable_plate_overlay(frame_out, license_dets)
                if frame_out is not None and not args.no_draw:
                    draw_anchor_comparison_overlay(frame_out, vehicle_payload_refs, next_frame_to_write)
                t_after_updates = time.perf_counter()
                if debug_tracks:
                    for det_ref in vehicle_payload_refs:
                        car_id = det_ref.get('track_id', -1)
                        if car_id <= 0:
                            continue
                        info = event_manager.get_track_debug(car_id)
                        if not info:
                            continue
                        x1, y1, _, _ = det_ref['box']
                        text = (f"ID:{car_id} {info['state']} sf:{info['stationary']} "
                                f"spd:{info['speed']:.1f} water:{'Y' if info['water'] else 'N'} "
                                f"dur:{info['wash_duration']:.1f} zb:{info.get('zone_b_elapsed',0)}")
                        cv2.putText(frame_out, text, (x1, max(0, y1 - 25)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
                if debug_rois:
                    if len(zone_a_pts) >= 3:
                        roi_np = np.array(zone_a_pts, dtype=np.int32)
                        cv2.polylines(frame_out, [roi_np], True, (0, 200, 0), 2, cv2.LINE_AA)
                        anchor = tuple(map(int, roi_np[0]))
                        cv2.putText(frame_out, 'Zone A', (anchor[0], max(0, anchor[1] - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2, cv2.LINE_AA)
                    if len(zone_b_pts) >= 3:
                        roi_np = np.array(zone_b_pts, dtype=np.int32)
                        cv2.polylines(frame_out, [roi_np], True, (0, 0, 255), 2, cv2.LINE_AA)
                        anchor = tuple(map(int, roi_np[0]))
                        cv2.putText(frame_out, 'Zone B', (anchor[0], max(0, anchor[1] - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    flow_start_int = tuple(map(int, flow_start))
                    flow_end_int = tuple(map(int, flow_end))
                    cv2.arrowedLine(frame_out, flow_start_int, flow_end_int, (255, 0, 0), 2, tipLength=0.08)
                debug_frame_source = raw_frame_for_idx if raw_frame_for_idx is not None else frame_out
                if debug_frame_source is not None and (next_frame_to_write % debug_frame_interval == 0):
                    save_debug_frame(
                        debug_frame_source,
                        det_payload,
                        vehicle_payload_refs,
                        capture_ts,
                        next_frame_to_write,
                    )
                ensure_benchmark_recording_track(next_frame_to_write, capture_ts)
                maybe_force_benchmark_capture(next_frame_to_write, event_frame_for_idx)
                if enable_per_id_video and frame_out is not None:
                    written_lifecycle_ids = set()
                    for tid, st in event_manager.tracks.items():
                        lifecycle = event_manager.lifecycle_manager.get(tid)
                        writer_key = lifecycle.event_id if lifecycle is not None else st.get('session_id', tid)
                        if writer_key in written_lifecycle_ids:
                            continue
                        written_lifecycle_ids.add(writer_key)
                        start_f = st.get('record_start_frame')
                        stop_f = st.get('record_stop_frame')
                        if start_f is None:
                            continue
                        if stop_f is not None and next_frame_to_write > stop_f:
                            close_per_id_writer(tid, st)
                            continue
                        if next_frame_to_write < start_f:
                            continue
                        writer = per_id_writers.get(writer_key)
                        if writer is None:
                            st_capture_time = st.get('type1_capture_time')
                            if not st_capture_time:
                                for ev_type in (5, 4, 3, 2, 1):
                                    ev_key = f'last_event_t{ev_type}_capture_time'
                                    val = st.get(ev_key)
                                    if val:
                                        st_capture_time = val
                                        break
                            if not st_capture_time:
                                st_capture_time = event_manager.frame_timestamp(start_f)
                            try:
                                dt = datetime.strptime(st_capture_time, "%Y-%m-%d %H:%M:%S")
                            except Exception:
                                dt = datetime.now()
                            session_id = st.get('session_id')
                            if not session_id:
                                ts_str = dt.strftime("%Y%m%d%H%M")
                                device_name = config.get('system', {}).get('device_id') or event_manager.camera_id
                                session_id = f"{device_name}-{ts_str}-{tid}"
                            writer_obj = build_per_id_writer(tid, st, dt, session_id)
                            if writer_obj is not None:
                                per_id_writers[writer_key] = writer_obj
                                writer = writer_obj
                            else:
                                writer = None
                                break
                        if writer is not None:
                            if per_id_video_source == 'raw' and raw_frame_for_idx is not None:
                                frame_to_write = raw_frame_for_idx
                            else:
                                frame_to_write = frame_out
                            if frame_to_write is not None:
                                writer.write(frame_to_write)
                t_after_perid = time.perf_counter()
                if raw_frame_for_idx is not None:
                    latest_raw_frame = raw_frame_for_idx
                elif frame_out is not None and latest_raw_frame is None:
                    latest_raw_frame = frame_out.copy() if copy_raw_frame_cache else frame_out
                if frame_out is not None:
                    latest_annotated_frame = frame_out
                latest_frame_idx = next_frame_to_write
                latest_capture_ts = capture_ts
                last_result_ts = time.time()
                last_progress_ts = last_result_ts
                emit_startup_signal_if_needed()
                poll_runtime_commands(force=True)
                write_heartbeat(status='running')
                if csv_writer and rows:
                    csv_writer.writerows(rows)
                next_frame_to_write += 1
                _advance_dropped_frames()
                t_before_flush = time.perf_counter()
                event_manager.flush_inactive(
                    alias_seen,
                    next_frame_to_write,
                    finalize_per_id_for_track,
                    capture_ts=capture_ts,
                )
                anchor_estimator.cleanup(next_frame_to_write, config.get('track_timeout_frames', 60))
                t_after_flush = time.perf_counter()
                cleanup_alias_confirm(next_frame_to_write)
                t_after_cleanup = time.perf_counter()
                drain_elapsed = t_after_cleanup - drain_t0
                recent_drain_times.append(drain_elapsed)
                current_event_count = event_manager._capture_metrics.get('saved', 0)
                if current_event_count > last_event_saved_count:
                    new_events = current_event_count - last_event_saved_count
                    last_event_saved_count = current_event_count
                    if len(recent_drain_times) >= 10:
                        total_time = sum(recent_drain_times)
                        short_fps = len(recent_drain_times) / max(total_time, 1e-6)
                        avg_drain_ms = total_time / len(recent_drain_times) * 1000.0
                        enc_ms = event_manager._capture_metrics.get('encode_seconds', 0.0) * 1000.0
                        wr_ms = event_manager._capture_metrics.get('write_seconds', 0.0) * 1000.0
                        t_pre_updates = t_before_updates - drain_t0
                        t_updates = t_after_updates - t_before_updates
                        t_perid = t_after_perid - t_after_updates
                        t_post_perid = t_before_flush - t_after_perid
                        t_flush = t_after_flush - t_before_flush
                        t_cleanup = t_after_cleanup - t_after_flush
                        print(
                            f'[perf:event] 事件触发! 新事件={new_events} '
                            f'短窗FPS={short_fps:.2f} 本帧={drain_elapsed*1000:.0f}ms | '
                            f'前置={t_pre_updates*1000:.0f}ms '
                            f'update_track={t_updates*1000:.0f}ms '
                            f'per-id视频={t_perid*1000:.0f}ms '
                            f'心跳等={t_post_perid*1000:.0f}ms '
                            f'flush={t_flush*1000:.0f}ms '
                            f'cleanup={t_cleanup*1000:.0f}ms | '
                            f'累计编码={enc_ms:.0f}ms 写盘={wr_ms:.0f}ms'
                        )
            return True

    frame_limit = args.limit if args.limit and args.limit > 0 else None

    while True:
        _advance_dropped_frames()
        poll_runtime_commands()
        if runtime_paused:
            last_progress_ts = time.time()
            drain_results(block=False)
            event_manager.flush_inactive(set(), next_frame_to_write, finalize_per_id_for_track)
            write_heartbeat(
                status='paused',
                force=True,
                extra={'runtime_pause_poll_interval_seconds': 0.1},
            )
            poll_runtime_commands(force=True)
            if runtime_paused:
                time.sleep(0.1)
                continue
        if cap is None:
            write_heartbeat(
                status='reader_reopen',
                force=True,
                extra={'reopen_after_runtime_pause': True},
            )
            cap, decode_meta = create_video_reader(path, args)
            if cap and hasattr(cap, 'isOpened') and cap.isOpened():
                _log_decode_open_result('恢复打开', decode_meta, True)
                consecutive_fails = 0
                write_heartbeat(status='running', force=True)
            else:
                _log_decode_open_result('恢复打开', decode_meta, False)
                reconnect_count += 1
                if reader_max_reconnect and reconnect_count >= reader_max_reconnect:
                    print('[reader] max reconnect attempts reached during runtime resume, aborting stream.')
                    break
                time.sleep(reader_reconnect_delay)
                continue
        write_heartbeat(status='running')
        storage_cleaner.run_due(reason='periodic')
        if frame_limit is not None and total_frames >= frame_limit:
            break
        ret, frame = cap.read()
        if not ret or frame is None:
            failure_summary, failure_extra = _capture_failure_summary(cap)
            consecutive_fails += 1
            if is_file_input:
                print('[reader] local file reached EOF or failed, stopping.')
                break
            if consecutive_fails < reader_fail_threshold:
                write_heartbeat(
                    status='waiting_reader',
                    force=True,
                    extra={
                        'consecutive_reader_failures': consecutive_fails,
                        **failure_extra,
                    },
                )
                time.sleep(0.05)
                continue
            reconnect_count += 1
            consecutive_fails = 0
            _throttled_log(
                'reader.reconnect_attempt',
                f'[reader] capture stalled, reconnect attempt #{reconnect_count}'
                + (f' {failure_summary}' if failure_summary else ''),
                window_seconds=15.0,
            )
            write_heartbeat(
                status='reader_reconnect',
                force=True,
                extra={
                    'reconnect_attempt': reconnect_count,
                    **failure_extra,
                },
            )
            try:
                cap.release()
            except Exception:
                pass
            time.sleep(reader_reconnect_delay)
            cap, decode_meta = create_video_reader(path, args)
            if cap and hasattr(cap, 'isOpened') and cap.isOpened():
                _log_decode_open_result('重连', decode_meta, True)
                write_heartbeat(status='running', force=True)
                continue
            _log_decode_open_result('重连', decode_meta, False)
            if reader_max_reconnect and reconnect_count >= reader_max_reconnect:
                print('[reader] max reconnect attempts reached, aborting stream.')
                break
            time.sleep(reader_reconnect_delay)
            continue
        consecutive_fails = 0
        capture_ts = time.time()
        latest_raw_frame = frame.copy() if copy_raw_frame_cache else frame
        latest_frame_idx = total_frames
        latest_capture_ts = capture_ts
        last_frame_read_ts = capture_ts
        last_progress_ts = capture_ts
        frame_idx = total_frames
        raw_frame_cache[frame_idx] = latest_raw_frame
        while True:
            try:
                task_q.put_nowait((frame_idx, frame, capture_ts))
                break
            except Full:
                dropped = False
                if drop_stale_frames:
                    dropped = _drop_stale_task_for_realtime()
                    if not dropped:
                        drain_results(block=False)
                        time.sleep(0.001)
                else:
                    drain_results(block=True)
                _advance_dropped_frames()
                poll_runtime_commands(force=True)
                write_heartbeat(
                    status='backpressure',
                    force=True,
                    extra={
                        'dropped_frames': dropped_frame_count,
                        'dropped_pending': len(dropped_frame_ids),
                    },
                )
        total_frames += 1
        reader_log_frames += 1
        now = time.time()
        if reader_log_interval > 0 and now - reader_log_last_time >= reader_log_interval:
            elapsed_window = now - reader_log_last_time
            decode_fps_window = reader_log_frames / max(elapsed_window, 1e-6)
            pipeline_frames_window = 0
            worker_msgs = []
            worker_perf = []
            for i, w in enumerate(workers):
                frames_delta = max(0, w.frames - worker_last_frames[i])
                infer_delta = max(0.0, w.infer_time - worker_last_infer[i])
                worker_last_frames[i] = w.frames
                worker_last_infer[i] = w.infer_time
                if frames_delta > 0 and elapsed_window > 0:
                    worker_fps = frames_delta / elapsed_window
                else:
                    worker_fps = 0.0
                if frames_delta > 0 and infer_delta > 0.0:
                    infer_ms = infer_delta * 1000.0 / frames_delta
                else:
                    infer_ms = 0.0
                pipeline_frames_window += frames_delta
                worker_msgs.append(f'w{i}:{worker_fps:.2f}fps/{infer_ms:.1f}ms')
                worker_perf.append({
                    'idx': i,
                    'fps': worker_fps,
                    'infer_ms': infer_ms,
                    'frames_delta': frames_delta,
                })
            wheel_msgs = []
            wheel_perf = {}
            wheel_stats_now = {}
            if wheel_service is not None:
                stats_now = wheel_service.snapshot_stats()
                wheel_stats_now = stats_now
                for side in ('left', 'right'):
                    side_stats = stats_now.get(side)
                    if not side_stats:
                        continue
                    prev_stats = wheel_last_stats.get(side, {})
                    decode_delta = max(0, int(side_stats.get('decode_frames', 0)) - int(prev_stats.get('decode_frames', 0)))
                    infer_delta = max(0, int(side_stats.get('infer_frames', 0)) - int(prev_stats.get('infer_frames', 0)))
                    infer_time_delta = max(
                        0.0,
                        float(side_stats.get('infer_time', 0.0)) - float(prev_stats.get('infer_time', 0.0)),
                    )
                    center_delta = max(0, int(side_stats.get('center_hits', 0)) - int(prev_stats.get('center_hits', 0)))
                    decode_fps = decode_delta / max(elapsed_window, 1e-6)
                    infer_fps = infer_delta / max(elapsed_window, 1e-6)
                    infer_ms = (infer_time_delta * 1000.0 / infer_delta) if infer_delta > 0 else 0.0
                    reader_open_count = int(side_stats.get('reader_open_count', 0) or 0)
                    reader_reconnect_count = int(side_stats.get('reader_reconnect_count', 0) or 0)
                    reader_open_delta = max(0, reader_open_count - int(prev_stats.get('reader_open_count', 0) or 0))
                    reader_reconnect_delta = max(
                        0,
                        reader_reconnect_count - int(prev_stats.get('reader_reconnect_count', 0) or 0),
                    )
                    wheel_msgs.append(
                        f'{side[0]}:{decode_fps:.2f}/{infer_fps:.2f}fps/{infer_ms:.1f}ms/c{center_delta}'
                    )
                    wheel_perf[side] = {
                        'decode_fps': decode_fps,
                        'infer_fps': infer_fps,
                        'infer_ms': infer_ms,
                        'center_hits_delta': center_delta,
                        'reader_open_count': reader_open_count,
                        'reader_open_delta': reader_open_delta,
                        'reader_reconnect_count': reader_reconnect_count,
                        'reader_reconnect_delta': reader_reconnect_delta,
                        'reader_last_open_reason': str(side_stats.get('reader_last_open_reason') or ''),
                        'reader_last_reconnect_reason': str(side_stats.get('reader_last_reconnect_reason') or ''),
                        'reader_last_frame_gap': float(side_stats.get('reader_last_frame_gap', -1.0) or -1.0),
                        'reader_last_open_age': float(side_stats.get('reader_last_open_age', -1.0) or -1.0),
                        'reader_alive': bool(side_stats.get('reader_alive', False)),
                        'processor_alive': bool(side_stats.get('processor_alive', False)),
                    }
                wheel_last_stats = stats_now
            pipeline_fps_window = pipeline_frames_window / max(elapsed_window, 1e-6) if pipeline_frames_window > 0 else 0.0
            task_q_size = _safe_qsize(task_q)
            result_q_size = _safe_qsize(result_q)
            dropped_delta = max(0, dropped_frame_count - reader_log_last_dropped)
            main_reconnect_delta = max(0, reconnect_count - reader_log_last_reconnect_count)
            per_id_diag = _snapshot_per_id_video_metrics()
            event_upload_pending = _upload_pending_count(uploader)
            wheel_upload_pending = _upload_pending_count(wheel_photo_uploader)
            npu_status = snapshot_npu_status()
            npu_flags = npu_status_flags(npu_status)
            diag_flags = []
            if dropped_delta > 0 or task_q_size > max(1, args.queue_size // 2) or result_q_size > max(1, args.queue_size // 2):
                diag_flags.append('pipeline_backpressure')
            if main_reconnect_delta > 0:
                diag_flags.append('main_reader_reconnect')
            if any(int(item.get('reader_reconnect_delta', 0) or 0) > 0 for item in wheel_perf.values()):
                diag_flags.append('wheel_reader_reconnect')
            if event_upload_pending > 0 or wheel_upload_pending > 0:
                diag_flags.append('upload_backlog')
            if int(per_id_diag.get('dropped', 0) or 0) > 0 or int(per_id_diag.get('write_errors', 0) or 0) > 0:
                diag_flags.append('per_id_video_issue')
            if pipeline_fps_window < decode_fps_window * 0.8 and decode_fps_window > 1.0:
                diag_flags.append('algorithm_or_queue_slow')
            diag_flags.extend(flag for flag in npu_flags if flag not in diag_flags)
            last_perf_snapshot = {
                'window_seconds': elapsed_window,
                'decode_fps': decode_fps_window,
                'pipeline_fps': pipeline_fps_window,
                'workers': worker_perf,
                'wheel': wheel_perf,
                'npu': npu_status,
                'task_queue_size': task_q_size,
                'result_queue_size': result_q_size,
                'pending_results': len(pending),
                'dropped_frames_total': dropped_frame_count,
                'dropped_frames_delta': dropped_delta,
                'main_reader_reconnect_count': reconnect_count,
                'main_reader_reconnect_delta': main_reconnect_delta,
                'event_upload_pending': event_upload_pending,
                'wheel_upload_pending': wheel_upload_pending,
                'diag_flags': diag_flags,
            }
            workers_str = ', '.join(worker_msgs)
            wheel_str = f' 车轮[{", ".join(wheel_msgs)}]' if wheel_msgs else ''
            print(
                f'[perf] 解码FPS={decode_fps_window:.2f} 管线FPS={pipeline_fps_window:.2f} '
                f'窗口={elapsed_window:.1f}s 总帧数={total_frames} 工人[{workers_str}]{wheel_str}'
            )
            wheel_reader_parts = []
            for side in ('left', 'right'):
                item = wheel_perf.get(side)
                if not item:
                    continue
                wheel_reader_parts.append(
                    f'{side[0]}:open={int(item.get("reader_open_count", 0) or 0)}'
                    f'/reconn={int(item.get("reader_reconnect_count", 0) or 0)}'
                    f'/d_reconn={int(item.get("reader_reconnect_delta", 0) or 0)}'
                    f'/age={float(item.get("reader_last_open_age", -1.0) or -1.0):.1f}s'
                    f'/gap={float(item.get("reader_last_frame_gap", -1.0) or -1.0):.1f}s'
                    f'/reason={item.get("reader_last_reconnect_reason") or item.get("reader_last_open_reason") or "-"}'
                    f'/alive={int(bool(item.get("reader_alive", False)))}'
                )
            service_stats = wheel_stats_now.get('_service', {}) if isinstance(wheel_stats_now, dict) else {}
            print(
                f'[diag] flags={",".join(diag_flags) if diag_flags else "ok"} '
                f'q=task:{task_q_size}/{args.queue_size},result:{result_q_size},pending:{len(pending)} '
                f'drop=win:{dropped_delta},total:{dropped_frame_count},pending:{len(dropped_frame_ids)} '
                f'main_reader_reconnect=win:{main_reconnect_delta},total:{reconnect_count} '
                f'uploads=event:{event_upload_pending},wheel:{wheel_upload_pending} '
                f'per_id=active:{per_id_diag.get("active_writers", 0)},queued:{per_id_diag.get("queue_size_total", 0)},'
                f'dropped:{per_id_diag.get("dropped", 0)},errors:{per_id_diag.get("write_errors", 0)} '
                f'wheel_service=active:{int(bool(service_stats.get("inference_active", False)))},'
                f'boost:{int(bool(service_stats.get("boost_active", False)))},'
                f'tracks:{service_stats.get("active_track_count", 0)} '
                f'wheel_reader=[{"; ".join(wheel_reader_parts) if wheel_reader_parts else "-"}] '
                f'npu=[{format_npu_status(npu_status)}]'
            )
            _write_metrics(status='running', perf_snapshot=last_perf_snapshot)
            reader_log_last_dropped = dropped_frame_count
            reader_log_last_reconnect_count = reconnect_count
            reader_log_frames = 0
            reader_log_last_time = now
        while result_q.qsize() > args.queue_size // 2:
            drain_results(block=False)

    write_heartbeat(status='stopping', force=True)
    for _ in workers:
        while True:
            try:
                task_q.put(None, timeout=0.5)
                break
            except Full:
                drain_results(block=False)
                poll_runtime_commands(force=True)
                write_heartbeat(status='stopping', force=True)
    task_q.join()
    while finished_workers < len(workers):
        if not drain_results(block=True):
            poll_runtime_commands(force=True)
            write_heartbeat(status='draining', force=True)

    cleanup_runtime()
    try:
        atexit.unregister(cleanup_runtime)
    except Exception:
        pass

    elapsed = time.time() - start
    write_heartbeat(status='stopped', force=True, extra={'elapsed_seconds': elapsed})
    _write_metrics(status='stopped', perf_snapshot=last_perf_snapshot)
    if total_frames:
        print(f'Video {path}: frames={total_frames} elapsed={elapsed:.2f}s ({total_frames/elapsed:.2f} FPS)')
    agg_frames = sum(w.frames for w in workers)
    agg_infer = sum(w.infer_time for w in workers)
    if agg_frames:
        print(f'Average inference {agg_infer/agg_frames*1000:.2f} ms over {agg_frames} frames')
    for w in workers:
        if w.frames:
            rate = w.frames / elapsed
            print(f'Worker {w.idx}: frames={w.frames} infer_ms={w.infer_time*1000/w.frames:.2f} throughput={rate:.2f} FPS')
